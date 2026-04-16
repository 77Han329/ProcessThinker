#!/usr/bin/env python3
"""
单次 Rollout 测试脚本
测试模型是否能自主生成 <think><step>...</step></think><answer>...</answer> 格式
"""

import json
import random
import re
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils.vision_process import fetch_video

# ==================== 配置 ====================
# 模型路径 (请修改为你的模型路径)
MODEL_PATH = "/hnvme/workspace/v100dd13-reasoning_model/Processthinker/LLaMA-Factory/new_ckpts/processthinker_sft_70k/checkpoint-200"
# MODEL_PATH = "/hnvme/workspace/v100dd13-reasoning_model/Processthinker/LLaMA-Factory/models/PG-8B-SFT-v2"

# 数据集路径
DATASET_PATH = "/hnvme/workspace/v100dd13-reasoning_model/Processthinker/LLaMA-Factory/data/train_data/processthinker_rl_10k.json"

# 生成配置
MAX_NEW_TOKENS = 1024
TEMPERATURE = 1.0
DO_SAMPLE = False

# ==================== Prompt 模式选择 ====================
# "simple"  - 简单模式，接近 SFT 训练时的格式（只有问题+选项）
# "easyr1"  - EasyR1 模式，带格式说明（放在 user content 里）
# "system"  - System 模式，格式说明放在 system prompt 里（和 LMF 训练一致）
PROMPT_MODE = "system"

# 简单模式 prompt（接近 SFT 训练格式）
SIMPLE_TEMPLATE = "{Question}"

# EasyR1 模式 prompt（格式说明在 user content 里）
EASYR1_TEMPLATE = (
    "{Question}\n"
    "Please answer this question based on the visual content."
    "Provide your thinking process between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
    "At the end, you must output the final answer in the format:\n"
    "<answer><your_answer_here></answer>\n"
    "you must provide your answer in chinese"
)

# System 模式 prompt（格式说明在 system message 里，和 LMF 训练一致）
SYSTEM_PROMPT = (
    "Please answer this question based on the visual content. "
    "Provide your step-by-step thinking process between the <think> and </think> tags, "
    "using <step>...</step> for each step as much as needed. "
    "Then give your final answer between the <answer> and </answer> tags. "
    "At the end, you must output the final answer in the format: <answer><your_answer_here></answer>, "
    "Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer>...</answer> tags.\n"
    "Example:\n<answer>A</answer>"
)

TYPE_TEMPLATE = {
    "multiple choice": (
        "\nPlease provide only the single option letter (e.g., A, B, C, D, etc.) "
        "within the <answer>...</answer> tags."
    )
}

# ==================== 格式检查正则 ====================
THINK_ANSWER_PATTERN = re.compile(
    r"\A\s*<think>.*?</think>\s*<answer>.*?</answer>\s*\Z",
    re.DOTALL,
)
STEP_RE = re.compile(r"<step>(.*?)</step>", re.DOTALL)


def check_format(response: str) -> dict:
    """检查输出格式是否正确"""
    result = {
        "has_think_tag": "<think>" in response and "</think>" in response,
        "has_answer_tag": "<answer>" in response and "</answer>" in response,
        "has_step_tag": "<step>" in response and "</step>" in response,
        "full_format_valid": bool(THINK_ANSWER_PATTERN.fullmatch(response or "")),
        "step_count": len(STEP_RE.findall(response or "")),
        "starts_with_think": response.strip().startswith("<think>"),
    }
    return result


def process_video(video_path: str, min_pixels: int = 4*32*32, max_pixels: int = 64*32*32, 
                  max_frames: int = 128, video_fps: float = 2.0):
    """处理视频"""
    vision_info = {
        "video": video_path, 
        "min_pixels": min_pixels, 
        "max_pixels": max_pixels, 
        "max_frames": max_frames, 
        "fps": video_fps
    }
    return fetch_video(vision_info, image_patch_size=16, return_video_sample_fps=True, return_video_metadata=True)


def build_prompt(example: dict, mode: str = "simple") -> str:
    """构建 prompt"""
    question = example.get("prompt", "")
    problem_type = example.get("problem_type", "")
    options = example.get("options", [])
    
    # 添加选项
    if problem_type == "multiple choice" and options:
        opts = "\n".join(options)
        question = f"{question}\nOptions:\n{opts}"
    
    # 根据模式选择模板
    if mode == "simple":
        prompt_str = SIMPLE_TEMPLATE.format(Question=question)
    elif mode == "system":
        # system 模式：只返回问题，格式说明放在 system message 里
        prompt_str = question
    else:  # easyr1
        prompt_str = EASYR1_TEMPLATE.format(Question=question)
        tail = TYPE_TEMPLATE.get(problem_type, "")
        prompt_str = prompt_str + tail
    
    return prompt_str


def load_sample(dataset_path: str, sample_idx: int = None):
    """从 RL 数据集加载一个样本"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if sample_idx is None:
        sample_idx = random.randint(0, len(data) - 1)
    
    sample = data[sample_idx]
    
    # 提取视频路径 (RL 数据集已经是完整路径)
    videos = sample.get('videos', [])
    video_path = videos[0] if videos else None
    
    # 从 solution 中提取答案
    solution = sample.get("solution", "")
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", solution)
    ground_truth = answer_match.group(1).strip() if answer_match else solution
    
    return {
        "sample_idx": sample_idx,
        "prompt": sample.get("problem", ""),
        "problem_type": sample.get("problem_type", ""),
        "options": sample.get("options", []),
        "video_path": video_path,
        "ground_truth": ground_truth,
    }


def main():
    print("=" * 60)
    print(f"单次 Rollout 测试 (Prompt 模式: {PROMPT_MODE})")
    print("=" * 60)
    
    # 1. 加载样本
    print("\n[1] 加载样本...")
    sample = load_sample(DATASET_PATH)
    print(f"    样本索引: {sample['sample_idx']}")
    print(f"    视频路径: {sample['video_path']}")
    print(f"    问题类型: {sample['problem_type']}")
    print(f"    原始问题: {sample['prompt'][:150]}...")
    print(f"    选项: {sample['options']}")
    print(f"    Ground Truth: {sample['ground_truth']}")
    
    # 2. 构建 prompt
    prompt_str = build_prompt(sample, mode=PROMPT_MODE)
    print(f"\n[2] 构建的 Prompt ({PROMPT_MODE} 模式):")
    print("-" * 40)
    print(prompt_str[:600] + "..." if len(prompt_str) > 600 else prompt_str)
    print("-" * 40)
    
    # 3. 加载模型
    print(f"\n[3] 加载模型: {MODEL_PATH}")
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        local_files_only=True,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    print("    模型加载完成!")
    
    # 4. 构建消息
    print("\n[4] 构建输入...")
    
    if sample['video_path']:
        print(f"    处理视频: {sample['video_path']}")
        processed_video, video_fps = process_video(sample['video_path'])
        processed_video, video_metadata = processed_video
        
        content_list = [
            {"type": "video"},
            {"type": "text", "text": prompt_str},
        ]
        
        # 根据模式构建消息
        if PROMPT_MODE == "system":
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content_list}
            ]
        else:
            messages = [{"role": "user", "content": content_list}]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = processor(
            text=[text],
            videos=[processed_video],
            video_metadata=[video_metadata],
            add_special_tokens=False,
            return_tensors="pt",
            do_resize=False,
            do_sample_frames=False,
        )
    else:
        # 根据模式构建消息
        if PROMPT_MODE == "system":
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_str}
            ]
        else:
            messages = [{"role": "user", "content": prompt_str}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            add_special_tokens=False,
            return_tensors="pt",
        )
    
    inputs = inputs.to(model.device)
    print(f"    Input IDs shape: {inputs['input_ids'].shape}")
    
    # 5. 生成（让模型自主生成，不强制添加任何前缀）
    print(f"\n[5] 开始生成 (max_new_tokens={MAX_NEW_TOKENS}, do_sample={DO_SAMPLE})...")
    print("    模型将自主生成输出...")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE if DO_SAMPLE else None,
            do_sample=DO_SAMPLE,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    
    # 解码输出
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]
    
    # 6. 显示结果
    print("\n" + "=" * 60)
    print("生成结果 (模型自主生成)")
    print("=" * 60)
    print(output_text)
    
    # 7. 格式检查
    print("\n" + "=" * 60)
    print("格式检查")
    print("=" * 60)
    format_result = check_format(output_text)
    for key, value in format_result.items():
        status = "✅" if value else "❌"
        if key == "step_count":
            print(f"    {key}: {value}")
        else:
            print(f"    {status} {key}: {value}")
    
    # 8. 提取答案并对比
    print("\n" + "=" * 60)
    print("答案对比")
    print("=" * 60)
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", output_text, re.DOTALL)
    predicted_answer = answer_match.group(1).strip() if answer_match else "N/A"
    print(f"    预测答案: {predicted_answer}")
    print(f"    真实答案: {sample['ground_truth']}")
    print(f"    是否正确: {'✅' if predicted_answer.upper() == sample['ground_truth'].upper() else '❌'}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
