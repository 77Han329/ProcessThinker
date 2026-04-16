#!/usr/bin/env python3
"""
实验日志可视化脚本
用于绘制 experiment_log.jsonl 中的训练指标

使用方式:
    python plot_experiment_log.py <experiment_log.jsonl> [--output_dir OUTPUT_DIR]
    
示例:
    python plot_experiment_log.py ../gdpo/gdpo-xxx/experiment_log.jsonl
    python plot_experiment_log.py ../gdpo/gdpo-xxx/experiment_log.jsonl --output_dir ./plots
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置全局样式
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c', '#e91e63', '#00bcd4']


def load_jsonl(filepath: str) -> list[dict]:
    """加载 jsonl 文件"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_metrics(data: list[dict]) -> dict:
    """从数据中提取各类指标"""
    metrics = defaultdict(list)
    
    for entry in data:
        step = entry.get('step', len(metrics['step']))
        metrics['step'].append(step)
        
        # Reward 指标
        reward = entry.get('reward', {})
        for key in ['overall', 'format_reward', 'r1', 'beta', 'format_valid', 'length_valid',
                    'step_count', 'acc_reward_raw', 'acc_reward', 'cot_reward_raw', 'cot_reward',
                    'step_bonus', 'r2', 'gdpo_r1', 'gdpo_beta', 'gdpo_cot', 'gdpo_step_bonus']:
            if key in reward:
                metrics[f'reward_{key}'].append(reward[key])
        
        # Actor 指标
        actor = entry.get('actor', {})
        for key in ['pg_loss', 'entropy_loss', 'ppo_kl', 'grad_norm', 'lr']:
            if key in actor:
                metrics[f'actor_{key}'].append(actor[key])
        
        # Critic 指标
        critic = entry.get('critic', {})
        for key in ['score', 'rewards', 'advantages', 'returns']:
            if key in critic and isinstance(critic[key], dict):
                for stat in ['mean', 'max', 'min']:
                    if stat in critic[key]:
                        metrics[f'critic_{key}_{stat}'].append(critic[key][stat])
        
        # 性能指标
        perf = entry.get('perf', {})
        for key in ['mfu_actor', 'throughput', 'max_memory_allocated_gb']:
            if key in perf:
                metrics[f'perf_{key}'].append(perf[key])
        
        # 长度指标
        for length_type in ['response_length', 'prompt_length']:
            length_data = entry.get(length_type, {})
            for stat in ['mean', 'max', 'min']:
                if stat in length_data:
                    metrics[f'{length_type}_{stat}'].append(length_data[stat])
        
        # 时间指标
        timing = entry.get('timing_s', {})
        for key in ['gen', 'reward', 'update_actor', 'step']:
            if key in timing:
                metrics[f'timing_{key}'].append(timing[key])
    
    return dict(metrics)


def plot_reward_metrics(metrics: dict, output_dir: Path, exp_name: str):
    """绘制 Reward 相关指标"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    steps = metrics['step']
    
    # 1. Overall Reward
    ax = axes[0, 0]
    if 'reward_overall' in metrics:
        ax.plot(steps, metrics['reward_overall'], color=COLORS[0], linewidth=2, label='Overall')
    if 'reward_r2' in metrics:
        ax.plot(steps, metrics['reward_r2'], color=COLORS[1], linewidth=2, label='R2 (Correctness)')
    if 'reward_format_reward' in metrics:
        ax.plot(steps, metrics['reward_format_reward'], color=COLORS[2], linewidth=2, label='Format Reward')
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.set_title('Overall Reward Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Format 细分
    ax = axes[0, 1]
    if 'reward_r1' in metrics:
        ax.plot(steps, metrics['reward_r1'], color=COLORS[0], linewidth=2, label='R1 (Format Valid)')
    if 'reward_beta' in metrics:
        ax.plot(steps, metrics['reward_beta'], color=COLORS[1], linewidth=2, label='Beta (Length)')
    if 'reward_format_valid' in metrics:
        ax.plot(steps, metrics['reward_format_valid'], color=COLORS[2], linewidth=2, linestyle='--', label='Format Valid Ratio')
    if 'reward_length_valid' in metrics:
        ax.plot(steps, metrics['reward_length_valid'], color=COLORS[3], linewidth=2, linestyle='--', label='Length Valid Ratio')
    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    ax.set_title('Format Reward Breakdown')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. COT & Accuracy
    ax = axes[1, 0]
    if 'reward_cot_reward_raw' in metrics:
        ax.plot(steps, metrics['reward_cot_reward_raw'], color=COLORS[0], linewidth=2, linestyle='--', label='COT Raw')
    if 'reward_cot_reward' in metrics:
        ax.plot(steps, metrics['reward_cot_reward'], color=COLORS[1], linewidth=2, label='COT (After Penalty)')
    if 'reward_acc_reward' in metrics:
        ax.plot(steps, metrics['reward_acc_reward'], color=COLORS[2], linewidth=2, label='Accuracy')
    if 'reward_step_bonus' in metrics:
        ax.plot(steps, metrics['reward_step_bonus'], color=COLORS[3], linewidth=2, label='Step Bonus')
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.set_title('COT & Accuracy Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. GDPO Scores (4 维度)
    ax = axes[1, 1]
    if 'reward_gdpo_r1' in metrics:
        ax.plot(steps, metrics['reward_gdpo_r1'], color=COLORS[0], linewidth=2, label='r1')
    if 'reward_gdpo_beta' in metrics:
        ax.plot(steps, metrics['reward_gdpo_beta'], color=COLORS[1], linewidth=2, label='beta')
    if 'reward_gdpo_cot' in metrics:
        ax.plot(steps, metrics['reward_gdpo_cot'], color=COLORS[2], linewidth=2, label='cot')
    if 'reward_gdpo_step_bonus' in metrics:
        ax.plot(steps, metrics['reward_gdpo_step_bonus'], color=COLORS[3], linewidth=2, label='step_bonus')
    ax.set_xlabel('Step')
    ax.set_ylabel('Score')
    ax.set_title('GDPO: 4 Reward Dimensions')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Reward Metrics - {exp_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'reward_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: reward_metrics.png")


def plot_actor_metrics(metrics: dict, output_dir: Path, exp_name: str):
    """绘制 Actor 训练指标"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    steps = metrics['step']
    
    # 1. PG Loss
    ax = axes[0, 0]
    if 'actor_pg_loss' in metrics:
        ax.plot(steps, metrics['actor_pg_loss'], color=COLORS[0], linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Policy Gradient Loss')
    ax.grid(True, alpha=0.3)
    
    # 2. Entropy Loss
    ax = axes[0, 1]
    if 'actor_entropy_loss' in metrics:
        ax.plot(steps, metrics['actor_entropy_loss'], color=COLORS[1], linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Entropy')
    ax.set_title('Entropy Loss')
    ax.grid(True, alpha=0.3)
    
    # 3. Gradient Norm
    ax = axes[1, 0]
    if 'actor_grad_norm' in metrics:
        ax.plot(steps, metrics['actor_grad_norm'], color=COLORS[2], linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Grad Norm')
    ax.set_title('Gradient Norm')
    ax.grid(True, alpha=0.3)
    
    # 4. PPO KL
    ax = axes[1, 1]
    if 'actor_ppo_kl' in metrics:
        ax.plot(steps, metrics['actor_ppo_kl'], color=COLORS[3], linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('KL')
    ax.set_title('PPO KL Divergence')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Actor Training Metrics - {exp_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'actor_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: actor_metrics.png")


def plot_critic_metrics(metrics: dict, output_dir: Path, exp_name: str):
    """绘制 Critic 指标（Advantages/Returns）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    steps = metrics['step']
    
    # 1. Advantages
    ax = axes[0]
    if 'critic_advantages_mean' in metrics:
        ax.plot(steps, metrics['critic_advantages_mean'], color=COLORS[0], linewidth=2, label='Mean')
    if 'critic_advantages_max' in metrics:
        ax.fill_between(steps, 
                        metrics.get('critic_advantages_min', [0]*len(steps)), 
                        metrics['critic_advantages_max'],
                        alpha=0.2, color=COLORS[0], label='Min-Max Range')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Advantage')
    ax.set_title('Advantages Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Returns
    ax = axes[1]
    if 'critic_returns_mean' in metrics:
        ax.plot(steps, metrics['critic_returns_mean'], color=COLORS[1], linewidth=2, label='Mean')
    if 'critic_returns_max' in metrics:
        ax.fill_between(steps, 
                        metrics.get('critic_returns_min', [0]*len(steps)), 
                        metrics['critic_returns_max'],
                        alpha=0.2, color=COLORS[1], label='Min-Max Range')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Return')
    ax.set_title('Returns Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Critic Metrics - {exp_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'critic_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: critic_metrics.png")


def plot_length_metrics(metrics: dict, output_dir: Path, exp_name: str):
    """绘制序列长度指标"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    steps = metrics['step']
    
    # 1. Response Length
    ax = axes[0]
    if 'response_length_mean' in metrics:
        ax.plot(steps, metrics['response_length_mean'], color=COLORS[0], linewidth=2, label='Mean')
    if 'response_length_max' in metrics and 'response_length_min' in metrics:
        ax.fill_between(steps, metrics['response_length_min'], metrics['response_length_max'],
                        alpha=0.2, color=COLORS[0], label='Min-Max')
    ax.set_xlabel('Step')
    ax.set_ylabel('Length (tokens)')
    ax.set_title('Response Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Prompt Length
    ax = axes[1]
    if 'prompt_length_mean' in metrics:
        ax.plot(steps, metrics['prompt_length_mean'], color=COLORS[1], linewidth=2, label='Mean')
    if 'prompt_length_max' in metrics and 'prompt_length_min' in metrics:
        ax.fill_between(steps, metrics['prompt_length_min'], metrics['prompt_length_max'],
                        alpha=0.2, color=COLORS[1], label='Min-Max')
    ax.set_xlabel('Step')
    ax.set_ylabel('Length (tokens)')
    ax.set_title('Prompt Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Sequence Length Metrics - {exp_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'length_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: length_metrics.png")


def plot_performance_metrics(metrics: dict, output_dir: Path, exp_name: str):
    """绘制性能指标"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    steps = metrics['step']
    
    # 1. Throughput
    ax = axes[0, 0]
    if 'perf_throughput' in metrics:
        ax.plot(steps, metrics['perf_throughput'], color=COLORS[0], linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Tokens/sec')
    ax.set_title('Throughput')
    ax.grid(True, alpha=0.3)
    
    # 2. MFU
    ax = axes[0, 1]
    if 'perf_mfu_actor' in metrics:
        ax.plot(steps, metrics['perf_mfu_actor'], color=COLORS[1], linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('MFU')
    ax.set_title('Model FLOPs Utilization (Actor)')
    ax.grid(True, alpha=0.3)
    
    # 3. Memory
    ax = axes[1, 0]
    if 'perf_max_memory_allocated_gb' in metrics:
        ax.plot(steps, metrics['perf_max_memory_allocated_gb'], color=COLORS[2], linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Memory (GB)')
    ax.set_title('Max GPU Memory Allocated')
    ax.grid(True, alpha=0.3)
    
    # 4. Time per Step
    ax = axes[1, 1]
    if 'timing_step' in metrics:
        ax.plot(steps, metrics['timing_step'], color=COLORS[3], linewidth=2, label='Total')
    if 'timing_gen' in metrics:
        ax.plot(steps, metrics['timing_gen'], color=COLORS[4], linewidth=2, linestyle='--', label='Generation')
    if 'timing_update_actor' in metrics:
        ax.plot(steps, metrics['timing_update_actor'], color=COLORS[5], linewidth=2, linestyle='--', label='Update')
    ax.set_xlabel('Step')
    ax.set_ylabel('Time (s)')
    ax.set_title('Time per Step')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Performance Metrics - {exp_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: performance_metrics.png")


def plot_summary_dashboard(metrics: dict, output_dir: Path, exp_name: str):
    """绘制综合仪表盘"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    steps = metrics['step']
    
    # 1. Overall Reward
    ax = axes[0, 0]
    if 'reward_overall' in metrics:
        ax.plot(steps, metrics['reward_overall'], color=COLORS[0], linewidth=2.5)
        ax.fill_between(steps, 0, metrics['reward_overall'], alpha=0.2, color=COLORS[0])
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.set_title('Overall Reward', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 2. GDPO 4 Dimensions
    ax = axes[0, 1]
    if 'reward_gdpo_r1' in metrics:
        ax.plot(steps, metrics['reward_gdpo_r1'], color=COLORS[0], linewidth=2, label='r1')
    if 'reward_gdpo_beta' in metrics:
        ax.plot(steps, metrics['reward_gdpo_beta'], color=COLORS[1], linewidth=2, label='beta')
    if 'reward_gdpo_cot' in metrics:
        ax.plot(steps, metrics['reward_gdpo_cot'], color=COLORS[2], linewidth=2, label='cot')
    if 'reward_gdpo_step_bonus' in metrics:
        ax.plot(steps, metrics['reward_gdpo_step_bonus'], color=COLORS[3], linewidth=2, label='step_bonus')
    ax.set_xlabel('Step')
    ax.set_ylabel('Score')
    ax.set_title('GDPO: 4 Reward Dimensions', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. COT Reward
    ax = axes[0, 2]
    if 'reward_cot_reward' in metrics:
        ax.plot(steps, metrics['reward_cot_reward'], color=COLORS[3], linewidth=2)
        ax.fill_between(steps, 0, metrics['reward_cot_reward'], alpha=0.2, color=COLORS[3])
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.set_title('COT Reward', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. PG Loss
    ax = axes[1, 0]
    if 'actor_pg_loss' in metrics:
        ax.plot(steps, metrics['actor_pg_loss'], color=COLORS[4], linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Policy Gradient Loss', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 5. Response Length
    ax = axes[1, 1]
    if 'response_length_mean' in metrics:
        ax.plot(steps, metrics['response_length_mean'], color=COLORS[5], linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Length')
    ax.set_title('Response Length (Mean)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 6. Format Valid Ratio
    ax = axes[1, 2]
    if 'reward_format_valid' in metrics:
        ax.plot(steps, metrics['reward_format_valid'], color=COLORS[6], linewidth=2, label='Format')
    if 'reward_length_valid' in metrics:
        ax.plot(steps, metrics['reward_length_valid'], color=COLORS[7], linewidth=2, label='Length')
    ax.set_xlabel('Step')
    ax.set_ylabel('Ratio')
    ax.set_title('Valid Ratios', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Training Dashboard - {exp_name}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: dashboard.png")


def main():
    parser = argparse.ArgumentParser(description='Plot experiment_log.jsonl metrics')
    parser.add_argument('log_file', type=str, help='Path to experiment_log.jsonl')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Output directory for plots (default: same as log file)')
    args = parser.parse_args()
    
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: File not found: {log_path}")
        return 1
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = log_path.parent / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 提取实验名称
    exp_name = log_path.parent.name
    if len(exp_name) > 60:
        exp_name = exp_name[:60] + "..."
    
    print(f"📊 Plotting experiment: {exp_name}")
    print(f"📁 Input: {log_path}")
    print(f"📁 Output: {output_dir}")
    print()
    
    # 加载数据
    print("Loading data...")
    data = load_jsonl(log_path)
    print(f"  Loaded {len(data)} steps")
    
    # 提取指标
    metrics = extract_metrics(data)
    
    # 绘制各类图表
    print("\nGenerating plots...")
    plot_summary_dashboard(metrics, output_dir, exp_name)
    plot_reward_metrics(metrics, output_dir, exp_name)
    plot_actor_metrics(metrics, output_dir, exp_name)
    plot_critic_metrics(metrics, output_dir, exp_name)
    plot_length_metrics(metrics, output_dir, exp_name)
    plot_performance_metrics(metrics, output_dir, exp_name)
    
    print(f"\n✅ All plots saved to: {output_dir}")
    return 0


if __name__ == '__main__':
    exit(main())
