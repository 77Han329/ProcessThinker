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
"""
A unified tracking interface that supports logging data to different backend
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch

from ..py_functional import convert_dict_to_str, flatten_dict, is_package_available, unflatten_dict
from .gen_logger import AggregateGenerationsLogger


if is_package_available("mlflow"):
    import mlflow  # type: ignore


if is_package_available("tensorboard"):
    from torch.utils.tensorboard import SummaryWriter


if is_package_available("wandb"):
    import wandb  # type: ignore


if is_package_available("swanlab"):
    import swanlab  # type: ignore


class Logger(ABC):
    @abstractmethod
    def __init__(self, config: dict[str, Any]) -> None: ...

    @abstractmethod
    def log(self, data: dict[str, Any], step: int) -> None: ...

    def finish(self) -> None:
        pass


class ConsoleLogger(Logger):
    def __init__(self, config: dict[str, Any]) -> None:
        print("Config\n" + convert_dict_to_str(config))

    def log(self, data: dict[str, Any], step: int) -> None:
        print(f"Step {step}\n" + convert_dict_to_str(unflatten_dict(data)))


class FileLogger(Logger):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        checkpoint_path = config["trainer"]["save_checkpoint_path"]
        print(f"Initializing logging file to {checkpoint_path}.")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # 保存配置文件（总是覆盖）
        with open(os.path.join(checkpoint_path, "experiment_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # 检查日志文件是否存在，存在则不覆盖（支持 resume）
        log_file = os.path.join(checkpoint_path, "experiment_log.jsonl")
        if os.path.exists(log_file):
            print(f"[Resume] Found existing experiment_log.jsonl, will append to it.")
        else:
            with open(log_file, "w") as f:
                pass

        gen_file = os.path.join(checkpoint_path, "generations.log")
        if os.path.exists(gen_file):
            print(f"[Resume] Found existing generations.log, will append to it.")
        else:
            with open(gen_file, "w") as f:
                pass

    def log(self, data: dict[str, Any], step: int) -> None:
        with open(os.path.join(self.config["trainer"]["save_checkpoint_path"], "experiment_log.jsonl"), "a") as f:
            f.write(json.dumps({"step": step, **unflatten_dict(data)}) + "\n")


class MlflowLogger(Logger):
    def __init__(self, config: dict[str, Any]) -> None:
        mlflow.start_run(run_name=config["trainer"]["experiment_name"])
        mlflow.log_params(flatten_dict(config))

    def log(self, data: dict[str, Any], step: int) -> None:
        mlflow.log_metrics(metrics=data, step=step)


class SwanlabLogger(Logger):
    def __init__(self, config: dict[str, Any]) -> None:
        swanlab_key = os.getenv("SWANLAB_API_KEY")
        swanlab_dir = os.getenv("SWANLAB_DIR", "swanlab_log")
        swanlab_mode = os.getenv("SWANLAB_MODE", "cloud")
        if swanlab_key:
            swanlab.login(swanlab_key)

        swanlab.init(
            project=config["trainer"]["project_name"],
            experiment_name=config["trainer"]["experiment_name"],
            config={"UPPERFRAMEWORK": "EasyR1", "FRAMEWORK": "veRL", **config},
            logdir=swanlab_dir,
            mode=swanlab_mode,
        )

    def log(self, data: dict[str, Any], step: int) -> None:
        swanlab.log(data=data, step=step)

    def finish(self) -> None:
        swanlab.finish()


class TensorBoardLogger(Logger):
    def __init__(self, config: dict[str, Any]) -> None:
        tensorboard_dir = os.getenv("TENSORBOARD_DIR", "tensorboard_log")
        tensorboard_dir = os.path.join(
            tensorboard_dir, config["trainer"]["project_name"], config["trainer"]["experiment_name"]
        )
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)
        config_dict = {}
        for key, value in flatten_dict(config).items():
            if isinstance(value, (int, float, str, bool, torch.Tensor)):
                config_dict[key] = value
            else:
                config_dict[key] = str(value)

        self.writer.add_hparams(hparam_dict=config_dict, metric_dict={"placeholder": 0})

    def log(self, data: dict[str, Any], step: int) -> None:
        for key, value in data.items():
            self.writer.add_scalar(key, value, step)

    def finish(self):
        self.writer.close()


class WandbLogger(Logger):
    def __init__(self, config: dict[str, Any]) -> None:
        import os
        entity = os.environ.get("WANDB_ENTITY", None)  # 从环境变量获取
        mode = os.environ.get("WANDB_MODE", "online")  # 默认 online，可设置为 offline
        
        # 增加 wandb 初始化超时时间，防止网络慢时超时
        # 特别是通过代理访问时需要更长时间
        wandb_settings = wandb.Settings(
            init_timeout=300,  # 增加到 5 分钟 (默认 90 秒)
            _disable_stats=True,  # 禁用系统统计，减少网络请求
        )
        
        wandb.init(
            entity=entity,
            project=config["trainer"]["project_name"],
            name=config["trainer"]["experiment_name"],
            config=config,
            mode=mode,
            settings=wandb_settings,
        )

    def log(self, data: dict[str, Any], step: int) -> None:
        wandb.log(data=data, step=step)

    def finish(self) -> None:
        wandb.finish()


LOGGERS = {
    "console": ConsoleLogger,
    "file": FileLogger,
    "mlflow": MlflowLogger,
    "swanlab": SwanlabLogger,
    "tensorboard": TensorBoardLogger,
    "wandb": WandbLogger,
}


class Tracker:
    def __init__(self, loggers: Union[str, list[str]] = "console", config: Optional[dict[str, Any]] = None):
        if isinstance(loggers, str):
            loggers = [loggers]

        self.loggers: list[Logger] = []
        for logger in loggers:
            if logger not in LOGGERS:
                raise ValueError(f"{logger} is not supported.")

            self.loggers.append(LOGGERS[logger](config))

        self.gen_logger = AggregateGenerationsLogger(loggers, config)

    def log(self, data: dict[str, Any], step: int) -> None:
        for logger in self.loggers:
            logger.log(data=data, step=step)

    def log_generation(self, samples: list[tuple[str, str, str, float]], step: int) -> None:
        self.gen_logger.log(samples, step)

    def __del__(self):
        for logger in self.loggers:
            logger.finish()
