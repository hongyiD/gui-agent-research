#!/usr/bin/env python3
"""
计算 sft_train.jsonl 中样本的 token 数统计

用于确认 max_length 配置是否足够，避免训练时样本被截断。

Usage:
    python calc_token_stats.py --data_file ../dataset/20260119_201327/sft_train.jsonl
    python calc_token_stats.py --data_file ../dataset/20260119_201327/sft_train.jsonl --model_path /path/to/model
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from PIL import Image


# ============== 图像 token 计算（参考 qwen_vl_utils/vision_process.py）==============

MAX_RATIO = 200
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer >= 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer <= 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, 
    width: int, 
    factor: int, 
    min_pixels: Optional[int] = None, 
    max_pixels: Optional[int] = None
) -> Tuple[int, int]:
    """
    Rescales the image so that:
    1. Both dimensions are divisible by 'factor'.
    2. Total pixels is within [min_pixels, max_pixels].
    3. Aspect ratio is maintained as closely as possible.
    """
    max_pixels = max_pixels if max_pixels is not None else (IMAGE_MAX_TOKEN_NUM * factor ** 2)
    min_pixels = min_pixels if min_pixels is not None else (IMAGE_MIN_TOKEN_NUM * factor ** 2)
    
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"Aspect ratio too extreme: {max(height, width) / min(height, width)}")
    
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(int(height / beta), factor)
        w_bar = floor_by_factor(int(width / beta), factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(int(height * beta), factor)
        w_bar = ceil_by_factor(int(width * beta), factor)
    
    return h_bar, w_bar


def calc_image_tokens(image_path: str, patch_size: int = 14, merge_size: int = 2) -> int:
    """
    计算单张图像的 token 数
    
    Args:
        image_path: 图像文件路径
        patch_size: ViT patch 大小，Qwen3-VL 默认 14
        merge_size: 空间合并大小，Qwen3-VL 默认 2
    
    Returns:
        图像对应的 token 数
    """
    patch_factor = patch_size * merge_size  # 28
    
    try:
        img = Image.open(image_path)
        width, height = img.size
        resized_h, resized_w = smart_resize(height, width, factor=patch_factor)
        return (resized_h * resized_w) // (patch_factor ** 2)
    except Exception as e:
        print(f"Warning: 无法加载图像 {image_path}: {e}")
        return 0


# ============== 文本 token 计算 ==============

def extract_text_from_messages(messages: List[Dict[str, Any]]) -> str:
    """从 messages 中提取所有文本内容"""
    texts = []
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
    return "\n".join(texts)


def extract_images_from_messages(messages: List[Dict[str, Any]], data_dir: str) -> List[str]:
    """从 messages 中提取所有图像路径"""
    images = []
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    image_url = item.get("image_url", "")
                    if image_url and not image_url.startswith(("http://", "https://", "data:")):
                        # 相对路径，拼接 data_dir
                        full_path = os.path.join(data_dir, image_url)
                        images.append(full_path)
    return images


def calc_text_tokens_simple(text: str) -> int:
    """
    简单估算文本 token 数（不加载模型）
    
    规则：
    - 英文单词约 1 token
    - 中文字符约 1-2 tokens（按 1.5 估算）
    - 特殊字符和标点约 1 token
    """
    import re
    
    # 分离中文和非中文
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    non_chinese = re.sub(r'[\u4e00-\u9fff]', '', text)
    
    # 英文按空格分词
    english_tokens = len(non_chinese.split())
    
    # 估算总 token 数
    return int(chinese_chars * 1.5) + english_tokens + 50  # 加 50 作为特殊 token 余量


def calc_text_tokens_with_tokenizer(text: str, tokenizer) -> int:
    """使用 tokenizer 精确计算 token 数"""
    return len(tokenizer.encode(text))


# ============== 主逻辑 ==============

def analyze_sample(
    sample: Dict[str, Any], 
    data_dir: str, 
    tokenizer=None
) -> Dict[str, Any]:
    """分析单个样本的 token 数"""
    messages = sample.get("messages", [])
    
    # 文本 token
    text = extract_text_from_messages(messages)
    if tokenizer:
        text_tokens = calc_text_tokens_with_tokenizer(text, tokenizer)
    else:
        text_tokens = calc_text_tokens_simple(text)
    
    # 图像 token
    images = extract_images_from_messages(messages, data_dir)
    image_tokens = sum(calc_image_tokens(img) for img in images)
    
    return {
        "text_tokens": text_tokens,
        "image_tokens": image_tokens,
        "total_tokens": text_tokens + image_tokens,
        "num_images": len(images),
        "images": images,
    }


def main():
    parser = argparse.ArgumentParser(description="计算 sft_train.jsonl 样本的 token 数统计")
    parser.add_argument("--data_file", type=str, required=True, help="sft_train.jsonl 文件路径")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径（可选，用于精确计算文本 token）")
    parser.add_argument("--max_length", type=int, default=4096, help="当前配置的 max_length")
    args = parser.parse_args()
    
    data_file = Path(args.data_file)
    if not data_file.exists():
        print(f"错误：文件不存在 {data_file}")
        return
    
    data_dir = str(data_file.parent)
    
    # 尝试加载 tokenizer
    tokenizer = None
    if args.model_path:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            print(f"已加载 tokenizer: {args.model_path}")
        except Exception as e:
            print(f"Warning: 无法加载 tokenizer ({e})，将使用估算方法")
    
    if not tokenizer:
        print("使用简单估算方法计算文本 token（结果可能有 10-20% 误差）")
        print("如需精确计算，请指定 --model_path 参数\n")
    
    # 读取数据
    samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"数据文件: {data_file}")
    print(f"数据目录: {data_dir}")
    print(f"总样本数: {len(samples)}")
    print(f"当前 max_length: {args.max_length}")
    print("-" * 60)
    
    # 分析每个样本
    results = []
    for i, sample in enumerate(samples):
        result = analyze_sample(sample, data_dir, tokenizer)
        result["index"] = i + 1
        results.append(result)
    
    # 统计
    total_tokens_list = [r["total_tokens"] for r in results]
    text_tokens_list = [r["text_tokens"] for r in results]
    image_tokens_list = [r["image_tokens"] for r in results]
    
    max_total = max(total_tokens_list)
    min_total = min(total_tokens_list)
    avg_total = sum(total_tokens_list) / len(total_tokens_list)
    
    max_idx = total_tokens_list.index(max_total)
    min_idx = total_tokens_list.index(min_total)
    
    # 超过 max_length 的样本
    over_limit = [r for r in results if r["total_tokens"] > args.max_length]
    
    print("\n" + "=" * 60)
    print("样本 Token 统计")
    print("=" * 60)
    print(f"  总样本数: {len(samples)}")
    print(f"  最大 token 数: {max_total} (样本 #{max_idx + 1})")
    print(f"  最小 token 数: {min_total} (样本 #{min_idx + 1})")
    print(f"  平均 token 数: {avg_total:.0f}")
    print()
    print(f"  文本 token 平均: {sum(text_tokens_list) / len(text_tokens_list):.0f}")
    print(f"  图像 token 平均: {sum(image_tokens_list) / len(image_tokens_list):.0f}")
    print()
    
    if over_limit:
        print(f"  ⚠️  超过 {args.max_length} 的样本: {len(over_limit)} 个 ({len(over_limit) / len(samples) * 100:.1f}%)")
        print(f"  超限样本详情:")
        for r in over_limit:
            print(f"    样本 #{r['index']}: {r['total_tokens']} tokens (文本:{r['text_tokens']}, 图像:{r['image_tokens']}, {r['num_images']}张图)")
    else:
        print(f"  ✅ 所有样本都在 {args.max_length} 以内")
    
    # 建议
    print()
    print("-" * 60)
    print("建议:")
    if max_total > args.max_length:
        # 建议值：向上取整到 512 的倍数
        suggested = math.ceil(max_total / 512) * 512
        print(f"  建议将 max_length 调整为: {suggested}")
        print(f"  或考虑减少每个样本的图像数量")
    else:
        buffer = args.max_length - max_total
        print(f"  当前 max_length={args.max_length} 足够")
        print(f"  最大样本还有 {buffer} tokens 余量 ({buffer / args.max_length * 100:.1f}%)")
    
    # 详细输出
    print()
    print("-" * 60)
    print("各样本详情 (前 10 个):")
    print("-" * 60)
    for r in results[:10]:
        status = "⚠️" if r["total_tokens"] > args.max_length else "✓"
        print(f"  {status} 样本 #{r['index']:2d}: 总计 {r['total_tokens']:5d} tokens "
              f"(文本: {r['text_tokens']:4d}, 图像: {r['image_tokens']:4d}, {r['num_images']}张图)")


if __name__ == "__main__":
    main()
