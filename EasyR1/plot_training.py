#!/usr/bin/env python3
"""
训练日志可视化脚本
用法: python plot_training.py [--log_file PATH] [--output OUTPUT]
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 无头模式，不需要显示器

# 设置中文字体（如果有的话）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

def load_jsonl(filepath):
    """加载 JSONL 文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def extract_metrics(data):
    """提取关键指标"""
    metrics = {
        'step': [],
        # Reward 相关
        'overall': [],
        'acc_reward': [],
        'cot_reward': [],
        'format_valid': [],
        'format_reward': [],
        'r2': [],
        'step_count': [],
        'step_bonus': [],
        # Length Reward 相关
        'length_valid': [],
        # Actor 相关
        'pg_loss': [],
        'entropy_loss': [],
        'grad_norm': [],
        # Response 相关
        'response_length_mean': [],
        'response_length_max': [],
        'response_length_min': [],
        'response_clip_ratio': [],
        # 性能相关
        'throughput': [],
        'time_per_step': [],
        # Advantages
        'advantages_mean': [],
    }
    
    for entry in data:
        metrics['step'].append(entry['step'])
        
        # Reward
        reward = entry.get('reward', {})
        metrics['overall'].append(reward.get('overall', 0))
        metrics['acc_reward'].append(reward.get('acc_reward', 0))
        metrics['cot_reward'].append(reward.get('cot_reward', 0))
        metrics['format_valid'].append(reward.get('format_valid', 0))
        metrics['format_reward'].append(reward.get('format_reward', 0))
        metrics['r2'].append(reward.get('r2', 0))
        metrics['step_count'].append(reward.get('step_count', 0))
        metrics['step_bonus'].append(reward.get('step_bonus', 0))
        metrics['length_valid'].append(reward.get('length_valid', 0))
        
        # Actor
        actor = entry.get('actor', {})
        metrics['pg_loss'].append(actor.get('pg_loss', 0))
        metrics['entropy_loss'].append(actor.get('entropy_loss', 0))
        metrics['grad_norm'].append(actor.get('grad_norm', 0))
        
        # Response length
        resp_len = entry.get('response_length', {})
        metrics['response_length_mean'].append(resp_len.get('mean', 0))
        metrics['response_length_max'].append(resp_len.get('max', 0))
        metrics['response_length_min'].append(resp_len.get('min', 0))
        metrics['response_clip_ratio'].append(resp_len.get('clip_ratio', 0))
        
        # Performance
        perf = entry.get('perf', {})
        metrics['throughput'].append(perf.get('throughput', 0))
        metrics['time_per_step'].append(perf.get('time_per_step', 0))
        
        # Advantages
        critic = entry.get('critic', {})
        adv = critic.get('advantages', {})
        metrics['advantages_mean'].append(adv.get('mean', 0))
    
    return metrics

def plot_metrics(metrics, output_path):
    """绑定多个子图"""
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    fig.suptitle('RL Training Metrics', fontsize=16, fontweight='bold')
    
    steps = metrics['step']
    
    # ========== Row 1: Reward 相关 ==========
    # 1. Overall Reward
    ax = axes[0, 0]
    ax.plot(steps, metrics['overall'], 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('Step')
    ax.set_ylabel('Overall Reward')
    ax.set_title('Overall Reward')
    ax.grid(True, alpha=0.3)
    
    # 2. Accuracy vs COT Reward
    ax = axes[0, 1]
    ax.plot(steps, metrics['acc_reward'], 'g-o', label='Accuracy', linewidth=2, markersize=6)
    ax.plot(steps, metrics['cot_reward'], 'r-s', label='COT (Process)', linewidth=2, markersize=6)
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.set_title('Accuracy vs COT Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # 3. Format Valid Rate
    ax = axes[0, 2]
    ax.plot(steps, metrics['format_valid'], 'm-o', linewidth=2, markersize=6)
    ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Target (100%)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Format Valid Rate')
    ax.set_title('Format Valid Rate')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    # ========== Row 2: Format Reward & R2 ==========
    # 4. Format Reward (r1 + beta)
    ax = axes[1, 0]
    ax.plot(steps, metrics['format_reward'], 'purple', marker='o', linewidth=2, markersize=6)
    ax.set_xlabel('Step')
    ax.set_ylabel('Format Reward')
    ax.set_title('Format Reward (r1 + beta)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(metrics['format_reward']) * 1.2 if metrics['format_reward'] and max(metrics['format_reward']) > 0 else 1)
    
    # 5. R2 (acc_weight * acc + cot_weight * cot + step_bonus)
    ax = axes[1, 1]
    ax.plot(steps, metrics['r2'], 'teal', marker='o', linewidth=2, markersize=6)
    ax.set_xlabel('Step')
    ax.set_ylabel('R2')
    ax.set_title('R2 (Accuracy + COT + Step Bonus)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(metrics['r2']) * 1.2 if metrics['r2'] and max(metrics['r2']) > 0 else 1)
    
    # 6. Step Bonus
    ax = axes[1, 2]
    ax.plot(steps, metrics['step_bonus'], 'olive', marker='o', linewidth=2, markersize=6)
    ax.set_xlabel('Step')
    ax.set_ylabel('Step Bonus')
    ax.set_title('Step Bonus')
    ax.grid(True, alpha=0.3)
    
    # ========== Row 3: Length Reward 相关 ==========
    # 7. Length Valid Rate
    ax = axes[2, 0]
    ax.plot(steps, metrics['length_valid'], 'darkorange', marker='o', linewidth=2, markersize=6)
    ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Target (100%)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Length Valid Rate')
    ax.set_title('Length Valid Rate (in [l_min, l_max])')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    # 8. Response Length Mean (no range)
    ax = axes[2, 1]
    ax.plot(steps, metrics['response_length_mean'], 'b-o', label='Mean', linewidth=2, markersize=6)
    ax.set_xlabel('Step')
    ax.set_ylabel('Response Length (tokens)')
    ax.set_title('Response Length (Mean)')
    ax.grid(True, alpha=0.3)
    
    # 9. Response Clip Ratio
    ax = axes[2, 2]
    ax.plot(steps, metrics['response_clip_ratio'], 'red', marker='o', linewidth=2, markersize=6)
    ax.set_xlabel('Step')
    ax.set_ylabel('Clip Ratio')
    ax.set_title('Response Clip Ratio (truncated)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(0.1, max(metrics['response_clip_ratio']) * 1.2) if metrics['response_clip_ratio'] else 0.1)
    
    # ========== Row 4: Actor 相关 ==========
    # 10. PG Loss
    ax = axes[3, 0]
    ax.plot(steps, metrics['pg_loss'], 'c-o', linewidth=2, markersize=6)
    ax.set_xlabel('Step')
    ax.set_ylabel('PG Loss')
    ax.set_title('Policy Gradient Loss')
    ax.grid(True, alpha=0.3)
    
    # 11. Entropy Loss
    ax = axes[3, 1]
    ax.plot(steps, metrics['entropy_loss'], 'orange', marker='o', linewidth=2, markersize=6)
    ax.set_xlabel('Step')
    ax.set_ylabel('Entropy')
    ax.set_title('Entropy Loss')
    ax.grid(True, alpha=0.3)
    
    # 12. Gradient Norm
    ax = axes[3, 2]
    ax.plot(steps, metrics['grad_norm'], 'purple', marker='o', linewidth=2, markersize=6)
    ax.set_xlabel('Step')
    ax.set_ylabel('Grad Norm')
    ax.set_title('Gradient Norm')
    ax.grid(True, alpha=0.3)
    
    # ========== Row 5: Step Count & Performance ==========
    # 13. Step Count
    ax = axes[4, 0]
    ax.plot(steps, metrics['step_count'], 'green', marker='o', linewidth=2, markersize=6)
    ax.axhline(y=2, color='r', linestyle='--', alpha=0.5, label='Min (2)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Step Count')
    ax.set_title('Average Step Count in <think>')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 14. Throughput
    ax = axes[4, 1]
    ax.plot(steps, metrics['throughput'], 'red', marker='o', linewidth=2, markersize=6)
    ax.set_xlabel('Step')
    ax.set_ylabel('Throughput (tokens/s)')
    ax.set_title('Training Throughput')
    ax.grid(True, alpha=0.3)
    
    # 15. Time per Step
    ax = axes[4, 2]
    ax.plot(steps, metrics['time_per_step'], 'navy', marker='o', linewidth=2, markersize=6)
    ax.set_xlabel('Step')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Time per Step')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    
    # 同时打印统计摘要
    print("\n" + "="*60)
    print("训练统计摘要")
    print("="*60)
    print(f"总步数: {len(steps)}")
    print(f"\n[Reward]")
    print(f"  Overall:     {metrics['overall'][0]:.4f} → {metrics['overall'][-1]:.4f} (变化: {metrics['overall'][-1] - metrics['overall'][0]:+.4f})")
    print(f"  Accuracy:    {metrics['acc_reward'][0]:.4f} → {metrics['acc_reward'][-1]:.4f} (变化: {metrics['acc_reward'][-1] - metrics['acc_reward'][0]:+.4f})")
    print(f"  COT:         {metrics['cot_reward'][0]:.4f} → {metrics['cot_reward'][-1]:.4f} (变化: {metrics['cot_reward'][-1] - metrics['cot_reward'][0]:+.4f})")
    print(f"  Format Valid: {metrics['format_valid'][0]*100:.1f}% → {metrics['format_valid'][-1]*100:.1f}%")
    print(f"  Length Valid: {metrics['length_valid'][0]*100:.1f}% → {metrics['length_valid'][-1]*100:.1f}%")
    print(f"\n[Response Length]")
    print(f"  Mean:        {metrics['response_length_mean'][0]:.1f} → {metrics['response_length_mean'][-1]:.1f} tokens")
    print(f"  Min:         {metrics['response_length_min'][0]:.1f} → {metrics['response_length_min'][-1]:.1f} tokens")
    print(f"  Max:         {metrics['response_length_max'][0]:.1f} → {metrics['response_length_max'][-1]:.1f} tokens")
    print(f"  Clip Ratio:  {metrics['response_clip_ratio'][0]*100:.1f}% → {metrics['response_clip_ratio'][-1]*100:.1f}%")
    print(f"  Step Count:  {metrics['step_count'][0]:.2f} → {metrics['step_count'][-1]:.2f}")
    print(f"\n[Actor]")
    print(f"  PG Loss:     {metrics['pg_loss'][0]:.4f} → {metrics['pg_loss'][-1]:.4f}")
    print(f"  Entropy:     {metrics['entropy_loss'][0]:.4f} → {metrics['entropy_loss'][-1]:.4f}")
    print(f"  Grad Norm:   {metrics['grad_norm'][0]:.4f} → {metrics['grad_norm'][-1]:.4f}")
    print(f"\n[Performance]")
    print(f"  Throughput:  {sum(metrics['throughput'])/len(metrics['throughput']):.1f} tokens/s (avg)")
    print(f"  Time/Step:   {sum(metrics['time_per_step'])/len(metrics['time_per_step']):.1f}s (avg)")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='可视化 RL 训练日志')
    parser.add_argument('--log_file', type=str, 
                        default='/hnvme/workspace/v100dd13-reasoning_model/Processthinker/EasyR1/checkpoints/experiment_log.jsonl',
                        help='experiment_log.jsonl 文件路径')
    parser.add_argument('--output', type=str,
                        default='/hnvme/workspace/v100dd13-reasoning_model/Processthinker/EasyR1/training_metrics.png',
                        help='输出图片路径')
    args = parser.parse_args()
    
    print(f"读取日志文件: {args.log_file}")
    data = load_jsonl(args.log_file)
    print(f"加载了 {len(data)} 个步骤的数据")
    
    metrics = extract_metrics(data)
    plot_metrics(metrics, args.output)

if __name__ == '__main__':
    main()
