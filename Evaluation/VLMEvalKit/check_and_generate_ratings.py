#!/usr/bin/env python3
"""
检查并生成缺失的rating.json文件
用于MLVU、Video_MMLU_QA、TempCompass等数据集
"""
import os
import sys
import json
from pathlib import Path
from vlmeval.smp import load, dump, get_intermediate_file_path
from vlmeval.dataset import build_dataset

# 数据集配置
DATASET_CONFIGS = {
    'MLVU_128frame': {
        'dataset_name': 'MLVU_128frame',
        'eval_file_patterns': ['*.xlsx', '*.pkl'],
        'rating_file': '*_rating.json',
        'judge_model': 'chatgpt-0125'  # MLVU_MCQ使用chatgpt-0125
    },
    'Video_MMLU_QA_128frame': {
        'dataset_name': 'Video_MMLU_QA_128frame',
        'eval_file_patterns': ['*.xlsx', '*.pkl', '*.json'],
        'rating_file': '*_rating.json',
        'judge_model': 'qwen-72b'  # 根据run.py中的配置
    },
    'TempCompass_128frame': {
        'dataset_name': 'TempCompass_128frame',
        'eval_file_patterns': ['*.xlsx', '*.pkl'],
        'rating_file': '*_rating.json',
        'judge_model': 'chatgpt-1106'  # TempCompass使用chatgpt-1106
    }
}


def find_eval_files(base_dir, patterns):
    """查找评估文件"""
    eval_files = []
    base_path = Path(base_dir)
    
    for pattern in patterns:
        for file_path in base_path.rglob(pattern):
            # 排除中间文件和rating文件
            if '_rating' not in file_path.name and '_score' not in file_path.name and '_tmp' not in file_path.name:
                eval_files.append(str(file_path))
    
    return eval_files


def check_rating_exists(base_dir, rating_pattern):
    """检查rating.json是否存在"""
    base_path = Path(base_dir)
    rating_files = list(base_path.rglob(rating_pattern))
    return len(rating_files) > 0, rating_files


def generate_rating_for_dataset(dataset_name, eval_file, work_dir, judge_model=None):
    """为数据集生成rating.json"""
    print(f"\n{'='*60}")
    print(f"处理数据集: {dataset_name}")
    print(f"评估文件: {eval_file}")
    print(f"工作目录: {work_dir}")
    print(f"{'='*60}")
    
    try:
        # 构建数据集
        print(f"正在构建数据集 {dataset_name}...")
        dataset = build_dataset(dataset_name)
        
        if dataset is None:
            print(f"❌ 无法构建数据集 {dataset_name}")
            return False
        
        # 准备judge参数
        judge_kwargs = {
            'model': judge_model,
            'nproc': 4,
            'verbose': True,
            'retry': 3
        }
        
        # 运行评估
        print(f"正在运行评估，使用judge模型: {judge_model}...")
        rating = dataset.evaluate(eval_file, **judge_kwargs)
        
        if rating is None:
            print(f"❌ 评估返回None")
            return False
        
        # 保存rating.json
        # 根据数据集类型确定rating文件路径
        eval_file_path = Path(eval_file)
        
        # 对于Video_MMLU_QA，rating文件已经在evaluate方法中保存
        if 'Video_MMLU' in dataset_name:
            rating_file = get_intermediate_file_path(eval_file, f'_{judge_model}_rating', 'json')
            if os.path.exists(rating_file):
                print(f"✅ Rating文件已生成: {rating_file}")
                return True
            else:
                # 手动保存
                dump(rating, rating_file)
                print(f"✅ Rating文件已保存: {rating_file}")
                return True
        else:
            # 对于其他数据集，需要手动保存
            rating_file = eval_file_path.parent / f"{eval_file_path.stem}_rating.json"
            dump(rating, str(rating_file))
            print(f"✅ Rating文件已保存: {rating_file}")
            return True
            
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    base_dir = "/hnvme/workspace/v100dd13-reasoning_model/Processthinker/Evaluation/VLMEvalKit/Grpo_prompt_outputs"
    
    if not os.path.exists(base_dir):
        print(f"❌ 基础目录不存在: {base_dir}")
        return
    
    print("="*60)
    print("检查并生成缺失的rating.json文件")
    print("="*60)
    
    # 检查每个数据集
    for dataset_key, config in DATASET_CONFIGS.items():
        dataset_dir = os.path.join(base_dir, f"single_{dataset_key}")
        
        if not os.path.exists(dataset_dir):
            print(f"\n⚠️  数据集目录不存在: {dataset_dir}")
            continue
        
        print(f"\n{'='*60}")
        print(f"检查数据集: {dataset_key}")
        print(f"{'='*60}")
        
        # 查找所有模型目录
        for model_dir in Path(dataset_dir).iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            print(f"\n检查模型: {model_name}")
            
            # 查找评估文件
            eval_files = find_eval_files(model_dir, config['eval_file_patterns'])
            
            if not eval_files:
                print(f"  ⚠️  未找到评估文件")
                continue
            
            print(f"  找到 {len(eval_files)} 个评估文件")
            
            # 检查是否已有rating文件
            has_rating, rating_files = check_rating_exists(model_dir, config['rating_file'])
            
            if has_rating:
                print(f"  ✅ 已存在rating文件: {[str(f) for f in rating_files]}")
                continue
            
            # 为每个评估文件生成rating
            for eval_file in eval_files:
                print(f"\n  处理文件: {eval_file}")
                
                # 检查文件是否有效
                try:
                    data = load(eval_file)
                    if data is None or len(data) == 0:
                        print(f"    ⚠️  文件为空或无效")
                        continue
                    print(f"    ✓ 文件有效，包含 {len(data)} 条记录")
                except Exception as e:
                    print(f"    ❌ 无法加载文件: {str(e)}")
                    continue
                
                # 生成rating
                work_dir = str(Path(eval_file).parent)
                success = generate_rating_for_dataset(
                    config['dataset_name'],
                    eval_file,
                    work_dir,
                    judge_model=config.get('judge_model')
                )
                
                if success:
                    print(f"    ✅ 成功生成rating文件")
                else:
                    print(f"    ❌ 生成rating文件失败")


if __name__ == '__main__':
    # 切换到VLMEvalKit目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    main()
