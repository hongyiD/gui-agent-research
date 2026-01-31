#!/usr/bin/env python3
"""
训练日志 Loss 曲线可视化脚本

从 v1.md, v2.md, v3.md 等日志文件中提取 loss 数据，
并绘制到同一张图中进行对比。
"""

import re
import os
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def extract_loss_from_log(log_content: str) -> list[tuple[int, float]]:
    """
    从日志内容中提取 step-loss 数据
    
    支持两种格式：
    1. 每步的 loss: {'loss': 1.8505, ...}
    2. 最终的 train_loss: {'train_loss': 2.6357, ...}
    
    Returns:
        list of (step, loss) tuples
    """
    losses = []
    
    # 模式1: 每步的 loss 记录 - {'loss': 1.8505, 'grad_norm': ...}
    step_loss_pattern = r"\{'loss':\s*([\d.]+)"
    step_losses = re.findall(step_loss_pattern, log_content)
    
    if step_losses:
        for i, loss in enumerate(step_losses, 1):
            losses.append((i, float(loss)))
    else:
        # 模式2: 只有最终的 train_loss
        train_loss_pattern = r"'train_loss':\s*([\d.]+)"
        match = re.search(train_loss_pattern, log_content)
        if match:
            losses.append((1, float(match.group(1))))
    
    return losses


def extract_config_info(log_content: str) -> dict:
    """提取训练配置信息"""
    config = {}
    
    # 提取 LoRA 配置
    lora_r_match = re.search(r'LoRA r:\s*(\d+)', log_content)
    if lora_r_match:
        config['lora_r'] = int(lora_r_match.group(1))
    
    lora_alpha_match = re.search(r'LoRA alpha:\s*(\d+)', log_content)
    if lora_alpha_match:
        config['lora_alpha'] = int(lora_alpha_match.group(1))
    
    # 提取量化配置
    if 'Use 4-bit quantization: True' in log_content:
        config['quantization'] = '4-bit'
    else:
        config['quantization'] = 'None'
    
    # 提取 max_length
    max_len_match = re.search(r'Max length:\s*(\d+)', log_content)
    if max_len_match:
        config['max_length'] = int(max_len_match.group(1))
    
    # 判断是否是纯文本训练
    if '[IMAGE]' in log_content or '图像被替换为文本占位符' in log_content:
        config['mode'] = '纯文本(忽略图像)'
    else:
        config['mode'] = '多模态'
    
    return config


def create_version_label(version: str, config: dict) -> str:
    """根据配置信息创建版本标签"""
    parts = [version]
    
    if 'mode' in config:
        parts.append(config['mode'])
    
    if 'lora_r' in config:
        parts.append(f"r={config['lora_r']}")
    
    if config.get('quantization') == '4-bit':
        parts.append('4-bit')
    
    if 'max_length' in config:
        parts.append(f"len={config['max_length']}")
    
    return ' | '.join(parts)


def plot_loss_curves(log_dir: str, output_path: str = None):
    """
    绘制所有版本的 loss 曲线
    
    Args:
        log_dir: 日志文件目录
        output_path: 输出图片路径，如果为 None 则显示图片
    """
    log_dir = Path(log_dir)
    
    # 查找所有 v*.md 文件
    log_files = sorted(log_dir.glob('v*.md'))
    
    if not log_files:
        print(f"未找到日志文件在 {log_dir}")
        return
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 颜色和样式
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    all_losses = {}
    
    for i, log_file in enumerate(log_files):
        version = log_file.stem  # e.g., "v1", "v2", "v3"
        
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        losses = extract_loss_from_log(content)
        config = extract_config_info(content)
        
        if not losses:
            print(f"警告: {log_file.name} 中未找到 loss 数据")
            continue
        
        all_losses[version] = {
            'losses': losses,
            'config': config
        }
        
        steps = [x[0] for x in losses]
        loss_values = [x[1] for x in losses]
        
        label = create_version_label(version.upper(), config)
        
        # 绘制曲线
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        ax.plot(steps, loss_values, 
                label=label,
                color=color, 
                marker=marker,
                markersize=6,
                linewidth=2,
                alpha=0.8)
        
        # 标注起始和结束 loss
        ax.annotate(f'{loss_values[0]:.3f}', 
                    xy=(steps[0], loss_values[0]),
                    xytext=(5, 10), textcoords='offset points',
                    fontsize=8, color=color)
        ax.annotate(f'{loss_values[-1]:.3f}', 
                    xy=(steps[-1], loss_values[-1]),
                    xytext=(5, -15), textcoords='offset points',
                    fontsize=8, color=color)
    
    # 设置图表样式
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('MAI-UI Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 设置 y 轴范围，留出一些空间
    all_loss_values = []
    for data in all_losses.values():
        all_loss_values.extend([x[1] for x in data['losses']])
    
    if all_loss_values:
        y_min = max(0, min(all_loss_values) - 0.2)
        y_max = max(all_loss_values) + 0.3
        ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {output_path}")
    else:
        plt.show()
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("训练版本对比统计")
    print("=" * 60)
    
    for version, data in sorted(all_losses.items()):
        losses = data['losses']
        config = data['config']
        loss_values = [x[1] for x in losses]
        
        print(f"\n{version.upper()}:")
        print(f"  配置: {config}")
        print(f"  总步数: {len(losses)}")
        print(f"  初始 Loss: {loss_values[0]:.4f}")
        print(f"  最终 Loss: {loss_values[-1]:.4f}")
        print(f"  Loss 下降: {loss_values[0] - loss_values[-1]:.4f} "
              f"({(loss_values[0] - loss_values[-1]) / loss_values[0] * 100:.1f}%)")
        print(f"  最低 Loss: {min(loss_values):.4f} (Step {loss_values.index(min(loss_values)) + 1})")
    
    return fig


if __name__ == '__main__':
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    
    # 输出路径
    output_file = script_dir / 'loss_curves_comparison.png'
    
    # 绘制图表
    plot_loss_curves(script_dir, output_file)
