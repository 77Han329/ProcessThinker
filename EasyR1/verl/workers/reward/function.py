# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


class FunctionRewardManager(ABC):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer

    @abstractmethod
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute reward for a batch of data."""
        ...


class SequentialFunctionRewardManager(FunctionRewardManager):
    reward_fn: SequentialRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            score = self.reward_fn(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                }
            )
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs = []
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            
            # DEBUG: print the raw token ids of the first sample
            if i == 0:
                first_10_ids = valid_response_ids[:10].tolist()
                first_10_decoded = self.tokenizer.decode(valid_response_ids[:10], skip_special_tokens=False)
                first_10_decoded_skip = self.tokenizer.decode(valid_response_ids[:10], skip_special_tokens=True)
                print(f"\n[DEBUG function.py] sample 0:")
                print(f"  - first 10 response token ids: {first_10_ids}")
                print(f"  - decode (skip=False): {repr(first_10_decoded)}")
                print(f"  - decode (skip=True): {repr(first_10_decoded_skip)}")
                # check the token id of <think>
                think_token_id = self.tokenizer.convert_tokens_to_ids("<think>")
                print(f"  - token id of <think>: {think_token_id}")
                print(f"  - is the first token id {first_10_ids[0]} == <think>: {first_10_ids[0] == think_token_id}")
            
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )

            multi_modal_data = None
            if "multi_modal_data" in data.non_tensor_batch:
                multi_modal_data = data.non_tensor_batch["multi_modal_data"][i]
            
            reward_inputs.append(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                    "data_type": data.non_tensor_batch["data_type"][i],
                    "problem_type": data.non_tensor_batch["problem_type"][i],
                    "problem": data.non_tensor_batch["problem_reserved_text"][i],
                    "problem_id": data.non_tensor_batch["problem_id"][i],
                    "multi_modal_data": data.non_tensor_batch["multi_modal_data"][i],
                }
            )


        # print(data)
        # print("\n\n\n\n\n\n\n\n\n")
        # print(reward_inputs)

        scores = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        # GDPO-only: 4 independent score tensors (normalized separately)
        gdpo_r1_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        gdpo_beta_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        gdpo_cot_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        gdpo_step_bonus_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            
            # GDPO: store the 4 independent scores (if present)
            if "gdpo_r1" in score:
                gdpo_r1_tensor[i, cur_response_length - 1] = score["gdpo_r1"]
            if "gdpo_beta" in score:
                gdpo_beta_tensor[i, cur_response_length - 1] = score["gdpo_beta"]
            if "gdpo_cot" in score:
                gdpo_cot_tensor[i, cur_response_length - 1] = score["gdpo_cot"]
            if "gdpo_step_bonus" in score:
                gdpo_step_bonus_tensor[i, cur_response_length - 1] = score["gdpo_step_bonus"]
            
            for key, value in score.items():
                reward_metrics[key].append(value)
        
        # Store the 4 GDPO tensors in metrics (passed via a special key);
        # they will be picked up by code in ray_trainer.py
        reward_metrics["__gdpo_r1_tensor__"] = gdpo_r1_tensor
        reward_metrics["__gdpo_beta_tensor__"] = gdpo_beta_tensor
        reward_metrics["__gdpo_cot_tensor__"] = gdpo_cot_tensor
        reward_metrics["__gdpo_step_bonus_tensor__"] = gdpo_step_bonus_tensor

        return reward_tensor, reward_metrics
