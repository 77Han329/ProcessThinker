#!/usr/bin/env python3
"""
训练指标可视化脚本
用法: python plot_training_metrics.py <experiment_log.jsonl> [output_dir]
"""

import json
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 无头模式，适用于服务器
import numpy as np
from scipy.ndimage import gaussian_filter1d

def load_jsonl(filepath):
    """加载 jsonl 文件（跳过不完整或损坏的行）"""
    data = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"⚠️ 跳过第 {i+1} 行（JSON 解析失败）: {str(e)[:50]}...")
    return data

def extract_metrics(data):
    """提取所有指标"""
    metrics = {
        'step': [],
        # Response Length
        'response_length_mean': [],
        'response_length_max': [],
        'response_length_min': [],
        # Prompt Length
        'prompt_length_mean': [],
        # GPU/Memory
        'max_memory_allocated_gb': [],
        'max_memory_reserved_gb': [],
        'cpu_memory_used_gb': [],
        # Performance
        'throughput': [],
        'time_per_step': [],
        # Reward - Overall
        'reward_overall': [],
        'format_reward': [],
        # Reward - Format
        'r1': [],
        'beta': [],
        'format_valid': [],
        'length_valid': [],
        'step_count': [],
        # Reward - Accuracy & COT
        'acc_reward': [],
        'cot_reward_raw': [],  # 原始 COT reward（penalty 处理前）
        'cot_reward': [],      # 最终 COT reward（penalty 处理后）
        'step_bonus': [],
        'r2': [],
        # Critic scores
        'critic_score_mean': [],
        'critic_score_max': [],
        'critic_score_min': [],
        # Actor
        'pg_loss': [],
        'entropy_loss': [],
        'grad_norm': [],
    }
    
    for entry in data:
        metrics['step'].append(entry['step'])
        
        # Response/Prompt Length
        metrics['response_length_mean'].append(entry['response_length']['mean'])
        metrics['response_length_max'].append(entry['response_length']['max'])
        metrics['response_length_min'].append(entry['response_length']['min'])
        metrics['prompt_length_mean'].append(entry['prompt_length']['mean'])
        
        # GPU/Memory
        metrics['max_memory_allocated_gb'].append(entry['perf']['max_memory_allocated_gb'])
        metrics['max_memory_reserved_gb'].append(entry['perf']['max_memory_reserved_gb'])
        metrics['cpu_memory_used_gb'].append(entry['perf']['cpu_memory_used_gb'])
        
        # Performance
        metrics['throughput'].append(entry['perf']['throughput'])
        metrics['time_per_step'].append(entry['perf']['time_per_step'])
        
        # Reward
        reward = entry['reward']
        metrics['reward_overall'].append(reward['overall'])
        metrics['format_reward'].append(reward['format_reward'])
        metrics['r1'].append(reward['r1'])
        metrics['beta'].append(reward['beta'])
        metrics['format_valid'].append(reward['format_valid'])
        metrics['length_valid'].append(reward['length_valid'])
        metrics['step_count'].append(reward['step_count'])
        metrics['acc_reward'].append(reward['acc_reward'])
        # cot_reward_raw 可能不存在（旧日志），使用 cot_reward 作为默认值
        metrics['cot_reward_raw'].append(reward.get('cot_reward_raw', reward['cot_reward']))
        metrics['cot_reward'].append(reward['cot_reward'])
        metrics['step_bonus'].append(reward['step_bonus'])
        metrics['r2'].append(reward['r2'])
        
        # Critic
        metrics['critic_score_mean'].append(entry['critic']['score']['mean'])
        metrics['critic_score_max'].append(entry['critic']['score']['max'])
        metrics['critic_score_min'].append(entry['critic']['score']['min'])
        
        # Actor
        metrics['pg_loss'].append(entry['actor']['pg_loss'])
        metrics['entropy_loss'].append(entry['actor']['entropy_loss'])
        metrics['grad_norm'].append(entry['actor']['grad_norm'])
    
    return metrics

def ema_smooth(y, alpha=0.97):
    """指数移动平均平滑 (EMA)"""
    y = np.array(y, dtype=float)
    if len(y) < 2:
        return y
    smoothed = np.zeros_like(y)
    smoothed[0] = y[0]
    for i in range(1, len(y)):
        smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * y[i]
    return smoothed

# 定义统一的颜色
STYLE = {
    'raw_color': '#B0B0B0',      # 浅灰色
    'smooth_color': '#2E7D32',   # 深绿色
    'alpha': 0.97,                # EMA 平滑系数
}

def plot_with_smooth(ax, x, y, color=None, label=None, marker='o', show_legend=True):
    """绘制原始数据（浅灰）+ 平滑曲线（深绿）- 学术风格"""
    x = np.array(x)
    y = np.array(y)
    y_smooth = ema_smooth(y, alpha=STYLE['alpha'])
    
    # 原始数据 - 浅灰色
    raw_label = 'Raw Data' if show_legend else None
    ax.plot(x, y, '-', color=STYLE['raw_color'], linewidth=0.8, alpha=0.6, label=raw_label)
    # 平滑曲线 - 深绿色
    smooth_label = f"Smoothed Data (α={STYLE['alpha']})" if show_legend else None
    ax.plot(x, y_smooth, '-', color=STYLE['smooth_color'], linewidth=2, 
            alpha=1.0, label=smooth_label)

def plot_multi_smooth(ax, x, y_dict, colors_dict):
    """绘制多条曲线，每条都有 raw + smooth"""
    x = np.array(x)
    for name, y in y_dict.items():
        y = np.array(y)
        y_smooth = ema_smooth(y, alpha=STYLE['alpha'])
        color = colors_dict.get(name, STYLE['smooth_color'])
        # 原始数据 - 透明
        ax.plot(x, y, '-', color=color, linewidth=0.8, alpha=0.3)
        # 平滑曲线
        ax.plot(x, y_smooth, '-', color=color, linewidth=2, alpha=1.0, label=name)

def plot_metrics(metrics, output_dir, exp_name):
    """绑制所有指标图表"""
    steps = metrics['step']
    
    # 设置全局样式 - 简洁学术风格
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'axes.titleweight': 'bold',
        'axes.grid': True,
        'grid.color': '#E0E0E0',
        'grid.linewidth': 0.5,
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'font.size': 10,
        'axes.titlesize': 11,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#CCCCCC',
    })
    colors = plt.cm.tab10.colors
    
    # ==================== 图1: Reward 相关 (4x3) ====================
    fig = plt.figure(figsize=(15, 16))
    fig.suptitle(f'{exp_name} - Reward Metrics', fontsize=14, fontweight='bold', y=0.98)
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1.1 Overall Reward
    ax = fig.add_subplot(gs[0, 0])
    plot_with_smooth(ax, steps, metrics['reward_overall'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.set_title('Overall Reward', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 1.2 Accuracy Reward
    ax = fig.add_subplot(gs[0, 1])
    plot_with_smooth(ax, steps, metrics['acc_reward'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.set_title('Accuracy Reward', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 1.3 COT Reward (Raw vs Final)
    ax = fig.add_subplot(gs[0, 2])
    plot_multi_smooth(ax, steps, 
                      {'Raw': metrics['cot_reward_raw'], 'Final': metrics['cot_reward']},
                      {'Raw': '#2E7D32', 'Final': '#C62828'})
    ax.axhline(y=0.5, color='#FF6B6B', linestyle='--', alpha=0.7, linewidth=1.5, label='Threshold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.set_title('COT Reward (Raw vs Final)', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 1.4 Format Reward (R1)
    ax = fig.add_subplot(gs[1, 0])
    plot_with_smooth(ax, steps, metrics['format_reward'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.set_title('Format Reward (R1)', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 1.5 R2 (Acc + COT)
    ax = fig.add_subplot(gs[1, 1])
    plot_with_smooth(ax, steps, metrics['r2'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.set_title('R2 (Acc + COT)', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 1.6 Step Bonus
    ax = fig.add_subplot(gs[1, 2])
    plot_with_smooth(ax, steps, metrics['step_bonus'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Bonus')
    ax.set_title('Step Bonus', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 1.7 Format Valid & Length Valid
    ax = fig.add_subplot(gs[2, 0])
    plot_multi_smooth(ax, steps,
                      {'Format': metrics['format_valid'], 'Length': metrics['length_valid']},
                      {'Format': '#1976D2', 'Length': '#FF9800'})
    ax.set_xlabel('Step')
    ax.set_ylabel('Ratio')
    ax.set_title('Valid Ratio', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 1.8 Step Count
    ax = fig.add_subplot(gs[2, 1])
    plot_with_smooth(ax, steps, metrics['step_count'], show_legend=False)
    ax.axhline(y=2, color='#FF6B6B', linestyle='--', alpha=0.7, linewidth=1.5, label='Min=2')
    ax.axhline(y=6, color='#FF6B6B', linestyle='--', alpha=0.7, linewidth=1.5, label='Max=6')
    ax.set_xlabel('Step')
    ax.set_ylabel('Count')
    ax.set_title('Step Count', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 1.9 Response Length
    ax = fig.add_subplot(gs[2, 2])
    plot_with_smooth(ax, steps, metrics['response_length_mean'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Tokens')
    ax.set_title('Response Length', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 1.10 COT Difference (Raw - Final) - 显示 penalty 的影响
    ax = fig.add_subplot(gs[3, 0])
    cot_diff = [raw - final for raw, final in zip(metrics['cot_reward_raw'], metrics['cot_reward'])]
    plot_with_smooth(ax, steps, cot_diff)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Difference')
    ax.set_title('COT Difference (Raw - Final)', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 1.11 Penalty Rate (COT < 0.5 被惩罚的比例)
    ax = fig.add_subplot(gs[3, 1])
    penalty_applied = [1 if diff > 0 else 0 for diff in cot_diff]
    plot_with_smooth(ax, steps, penalty_applied)
    ax.set_xlabel('Step')
    ax.set_ylabel('Rate')
    ax.set_title('Penalty Applied Rate', fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 1.12 COT Raw Distribution (< 0.5 vs >= 0.5)
    ax = fig.add_subplot(gs[3, 2])
    cot_below_threshold = [1 if raw < 0.5 else 0 for raw in metrics['cot_reward_raw']]
    plot_with_smooth(ax, steps, cot_below_threshold, show_legend=False)
    ax.axhline(y=0.5, color='#FF6B6B', linestyle='--', alpha=0.7, linewidth=1.5, label='50%')
    ax.set_xlabel('Step')
    ax.set_ylabel('Rate')
    ax.set_title('COT Raw < 0.5 Rate', fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'reward_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ==================== 图2: Training/Gradient 相关 (2x2) ====================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{exp_name} - Training Metrics', fontsize=14, fontweight='bold')
    
    # 2.1 PG Loss
    ax = axes[0, 0]
    plot_with_smooth(ax, steps, metrics['pg_loss'])
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Policy Gradient Loss', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 2.2 Entropy Loss
    ax = axes[0, 1]
    plot_with_smooth(ax, steps, metrics['entropy_loss'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Entropy')
    ax.set_title('Entropy Loss', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 2.3 Critic Score
    ax = axes[1, 0]
    plot_with_smooth(ax, steps, metrics['critic_score_mean'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Score')
    ax.set_title('Critic Score (Mean)', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 2.4 Gradient Norm
    ax = axes[1, 1]
    plot_with_smooth(ax, steps, metrics['grad_norm'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Norm')
    ax.set_title('Gradient Norm', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ==================== 图3: Performance 相关 (2x2) ====================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{exp_name} - Performance Metrics', fontsize=14, fontweight='bold')
    
    # 3.1 Throughput
    ax = axes[0, 0]
    plot_with_smooth(ax, steps, metrics['throughput'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Tokens/s')
    ax.set_title('Throughput', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 3.2 Time per Step
    ax = axes[0, 1]
    plot_with_smooth(ax, steps, metrics['time_per_step'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Seconds')
    ax.set_title('Time per Step', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 3.3 GPU Memory
    ax = axes[1, 0]
    plot_multi_smooth(ax, steps,
                      {'Allocated': metrics['max_memory_allocated_gb'], 
                       'Reserved': metrics['max_memory_reserved_gb']},
                      {'Allocated': '#2E7D32', 'Reserved': '#1976D2'})
    ax.set_xlabel('Step')
    ax.set_ylabel('Memory (GB)')
    ax.set_title('GPU Memory', fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 3.4 CPU Memory
    ax = axes[1, 1]
    plot_with_smooth(ax, steps, metrics['cpu_memory_used_gb'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Memory (GB)')
    ax.set_title('CPU Memory', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 图表已保存到: {output_dir}")
    print(f"   - reward_metrics.png (Reward 相关)")
    print(f"   - training_metrics.png (梯度/训练相关)")
    print(f"   - performance_metrics.png (性能相关)")

def main():
    if len(sys.argv) < 2:
        print("用法: python plot_training_metrics.py <experiment_log.jsonl> [output_dir]")
        print("示例: python plot_training_metrics.py experiment_log.jsonl ./plots")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # 默认输出目录为 jsonl 文件所在目录
    if len(sys.argv) >= 3:
        output_dir = sys.argv[2]
    else:
        output_dir = os.path.dirname(input_file) or '.'
    
    # 实验名称
    exp_name = os.path.basename(os.path.dirname(input_file))
    if not exp_name:
        exp_name = 'Experiment'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📊 加载数据: {input_file}")
    data = load_jsonl(input_file)
    print(f"   共 {len(data)} 个 step")
    
    print(f"📈 绑制图表...")
    metrics = extract_metrics(data)
    plot_metrics(metrics, output_dir, exp_name)

if __name__ == '__main__':
    main()
