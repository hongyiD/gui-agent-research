#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 labels 创建逻辑

验证 _create_labels 方法是否正确地为 assistant response 创建 labels。
"""

import sys
import torch
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from transformers import AutoProcessor
    from sft_trainer import MultiModalDataCollator
except ImportError as e:
    print(f"Error importing: {e}")
    print("Make sure you're running from the trainer directory")
    sys.exit(1)


def test_labels_creation():
    """测试 labels 创建逻辑"""
    print("=" * 60)
    print("Testing Labels Creation")
    print("=" * 60)
    
    # 加载处理器
    model_path = "/workspace/MAI-UI-2B"
    print(f"\nLoading processor from: {model_path}")
    
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading processor: {e}")
        print("Note: This test requires the model to be available.")
        return False
    
    # 创建 collator
    collator = MultiModalDataCollator(processor=processor, max_length=2048)
    
    # 创建测试消息
    messages = [
        {"role": "user", "content": "What is in this image?"},
        {"role": "assistant", "content": "This is a test response."}
    ]
    
    print("\nTest messages:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")
    
    # 使用 apply_chat_template 格式化
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    
    print(f"\nFormatted text:\n{text}")
    
    # Tokenize
    inputs = processor(text=[text], padding=False, return_tensors="pt")
    input_ids = inputs["input_ids"].squeeze(0)
    
    print(f"\nInput IDs shape: {input_ids.shape}")
    print(f"Input IDs (first 30): {input_ids[:30].tolist()}")
    
    # 详细分析 token 序列
    print("\n" + "=" * 60)
    print("Token Sequence Analysis")
    print("=" * 60)
    tokenizer = processor.tokenizer
    
    # 检查关键 token IDs
    print("\nKey token IDs:")
    try:
        im_start_user_tokens = tokenizer.encode("<|im_start|>user", add_special_tokens=False)
        print(f"  <|im_start|>user: {im_start_user_tokens}")
    except:
        print("  <|im_start|>user: (error)")
    
    try:
        im_start_assistant_tokens = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
        print(f"  <|im_start|>assistant: {im_start_assistant_tokens}")
    except:
        print("  <|im_start|>assistant: (error)")
    
    try:
        assistant_only_tokens = tokenizer.encode("assistant", add_special_tokens=False)
        print(f"  assistant (only): {assistant_only_tokens}")
    except:
        print("  assistant (only): (error)")
    
    try:
        im_end_tokens = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        print(f"  <|im_end|>: {im_end_tokens}")
    except:
        print("  <|im_end|>: (error)")
    
    eos_token_id = getattr(tokenizer, 'eos_token_id', None)
    print(f"  eos_token_id: {eos_token_id}")
    
    # 详细解码每个 token
    print("\nDetailed token breakdown:")
    for i, tid in enumerate(input_ids):
        decoded = tokenizer.decode([tid], skip_special_tokens=False)
        # 标记关键位置
        markers = []
        if tid == 151644:  # <|im_start|>
            markers.append("IM_START")
        if tid == 77091:  # assistant
            markers.append("ASSISTANT")
        if tid == 151645:  # <|im_end|>
            markers.append("IM_END")
        if tid == 198:  # \n
            markers.append("NEWLINE")
        
        marker_str = " [" + ", ".join(markers) + "]" if markers else ""
        print(f"  [{i:2d}] {tid:6d} = {repr(decoded):30s}{marker_str}")
    
    # 模拟官方实现的正确逻辑（用于对比）
    print("\n" + "=" * 60)
    print("Simulating Official Implementation Logic")
    print("=" * 60)
    
    # 官方实现查找 77091 (assistant token)
    OFFICIAL_ASSISTANT_TOKEN = 77091
    OFFICIAL_IM_END_TOKEN = 151645
    
    input_ids_list = input_ids.tolist()
    L = len(input_ids_list)
    
    print(f"\nOfficial implementation logic:")
    print(f"  Looking for assistant token: {OFFICIAL_ASSISTANT_TOKEN}")
    print(f"  Looking for <|im_end|> token: {OFFICIAL_IM_END_TOKEN}")
    
    correct_labels = torch.full_like(input_ids, -100)
    pos = 0
    found_assistant = False
    
    while pos < L:
        if input_ids_list[pos] == OFFICIAL_ASSISTANT_TOKEN:
            found_assistant = True
            print(f"\n  Found assistant token at position {pos}")
            ans_start = pos + 2  # 官方实现跳过 assistant token 和换行符
            print(f"  ans_start = {pos} + 2 = {ans_start}")
            
            ans_end = ans_start
            while ans_end < L and input_ids_list[ans_end] != OFFICIAL_IM_END_TOKEN:
                ans_end += 1
            
            if ans_end < L:
                print(f"  Found <|im_end|> at position {ans_end}")
                end_inclusive = ans_end + 2  # 官方实现包含 <|im_end|> 和换行符
                print(f"  end_inclusive = {ans_end} + 2 = {end_inclusive}")
                print(f"  Marking positions {ans_start} to {end_inclusive-1}")
                
                correct_labels[ans_start : end_inclusive] = input_ids[ans_start : end_inclusive]
                pos = ans_end
        pos += 1
    
    if not found_assistant:
        print(f"\n  ⚠️  Could not find assistant token ({OFFICIAL_ASSISTANT_TOKEN})")
    
    # 分析正确的结果
    correct_labeled_indices = (correct_labels != -100).nonzero(as_tuple=True)[0]
    if len(correct_labeled_indices) > 0:
        correct_labeled_input_ids = input_ids[correct_labeled_indices]
        correct_decoded = tokenizer.decode(correct_labeled_input_ids, skip_special_tokens=False)
        print(f"\n  Correct labeled portion:\n{correct_decoded}")
    
    # 创建 labels（使用当前实现）
    print("\n" + "=" * 60)
    print("Creating Labels (Current Implementation)")
    print("=" * 60)
    labels = collator._create_labels(input_ids, messages)
    
    print(f"\nLabels shape: {labels.shape}")
    print(f"Labels (first 30): {labels[:30].tolist()}")
    
    # 分析结果
    total_tokens = len(input_ids)
    masked_tokens = (labels == -100).sum().item()
    labeled_tokens = total_tokens - masked_tokens
    
    print(f"\n" + "=" * 60)
    print("Analysis")
    print("=" * 60)
    print(f"Total tokens: {total_tokens}")
    print(f"Masked tokens (prompt): {masked_tokens}")
    print(f"Labeled tokens (response): {labeled_tokens}")
    print(f"Labeled percentage: {labeled_tokens/total_tokens*100:.1f}%")
    
    # 验证：检查是否有 labeled tokens
    if labeled_tokens == 0:
        print("\n❌ ERROR: No tokens are labeled! All tokens are masked.")
        return False
    
    # 验证：检查 labeled tokens 是否在合理范围内
    # Assistant response 应该只占一小部分
    if labeled_tokens / total_tokens > 0.5:
        print(f"\n⚠️  WARNING: Labeled tokens ({labeled_tokens}) seem too high.")
        print("   This might indicate that prompt tokens are not being masked correctly.")
    
    # 找到 assistant 开始位置
    # 尝试多种方法找到 assistant 位置
    assistant_start_pos = None
    assistant_token_id = None
    
    # 方法1：查找 "assistant" token (77091)
    ASSISTANT_TOKEN_77091 = 77091
    for i, tid in enumerate(input_ids):
        if tid == ASSISTANT_TOKEN_77091:
            assistant_start_pos = i
            assistant_token_id = ASSISTANT_TOKEN_77091
            print(f"\nFound 'assistant' token (77091) at position {i}")
            break
    
    # 方法2：如果没找到，尝试查找 "<|im_start|>assistant" 的第一个 token
    if assistant_start_pos is None:
        try:
            im_start_assistant_tokens = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
            if im_start_assistant_tokens:
                first_token = im_start_assistant_tokens[0]
                for i, tid in enumerate(input_ids):
                    if tid == first_token:
                        # 检查后面是否跟着 assistant token
                        if i + 1 < len(input_ids) and input_ids[i + 1] == ASSISTANT_TOKEN_77091:
                            assistant_start_pos = i
                            assistant_token_id = first_token
                            print(f"\nFound '<|im_start|>assistant' first token ({first_token}) at position {i}")
                            break
        except:
            pass
    
    # 验证：检查 user 部分是否被正确屏蔽
    if assistant_start_pos is not None:
        user_labels = labels[:assistant_start_pos]
        user_labeled_count = (user_labels != -100).sum().item()
        
        print(f"\nAssistant starts at position: {assistant_start_pos}")
        print(f"User portion (positions 0-{assistant_start_pos-1}):")
        print(f"  Masked tokens: {(user_labels == -100).sum().item()}")
        print(f"  Labeled tokens: {user_labeled_count}")
        
        if user_labeled_count > 0:
            print(f"\n❌ ERROR: User portion contains {user_labeled_count} labeled tokens!")
            print("   User portion should be completely masked (-100).")
            user_labeled_indices = (user_labels != -100).nonzero(as_tuple=True)[0]
            print(f"   Labeled positions in user portion: {user_labeled_indices.tolist()}")
            return False
        else:
            print("   ✅ User portion is correctly masked.")
    else:
        print("\n⚠️  WARNING: Could not find assistant start token!")
        print("   Tried: 77091 (assistant token) and <|im_start|>assistant first token")
        print("   This might indicate a problem with token matching.")
    
    # 解码 labeled 部分
    labeled_indices = (labels != -100).nonzero(as_tuple=True)[0]
    if len(labeled_indices) > 0:
        labeled_input_ids = input_ids[labeled_indices]
        decoded_response = processor.tokenizer.decode(labeled_input_ids, skip_special_tokens=False)
        print(f"\nDecoded labeled portion:\n{decoded_response}")
        
        # 检查是否包含 assistant response
        contains_assistant = "test response" in decoded_response.lower() or "assistant" in decoded_response.lower()
        contains_user = "what is in this image" in decoded_response.lower() or "<|im_start|>user" in decoded_response.lower()
        
        if contains_assistant and not contains_user:
            print("\n✅ SUCCESS: Labeled portion contains only assistant response!")
        elif contains_assistant and contains_user:
            print("\n❌ ERROR: Labeled portion contains both user and assistant content!")
            print("   Only assistant response should be labeled.")
            return False
        elif not contains_assistant:
            print("\n⚠️  WARNING: Labeled portion might not contain the expected response.")
            return False
    else:
        print("\n❌ ERROR: No tokens are labeled!")
        return False
    
    # 对比当前实现和正确实现
    print("\n" + "=" * 60)
    print("Comparison: Current vs Correct")
    print("=" * 60)
    
    current_labeled = (labels != -100).sum().item()
    correct_labeled = (correct_labels != -100).sum().item()
    
    print(f"\nCurrent implementation:")
    print(f"  Labeled tokens: {current_labeled}")
    print(f"  Labeled positions: {(labels != -100).nonzero(as_tuple=True)[0].tolist()}")
    
    print(f"\nCorrect implementation (official):")
    print(f"  Labeled tokens: {correct_labeled}")
    print(f"  Labeled positions: {(correct_labels != -100).nonzero(as_tuple=True)[0].tolist()}")
    
    # 检查是否匹配
    labels_match = torch.equal(labels, correct_labels)
    if labels_match:
        print("\n✅ Labels match the official implementation!")
    else:
        print("\n❌ Labels do NOT match the official implementation!")
        diff_positions = (labels != correct_labels).nonzero(as_tuple=True)[0]
        print(f"  Different positions: {diff_positions.tolist()}")
        for pos in diff_positions[:10]:  # 只显示前10个差异
            print(f"    Position {pos}: current={labels[pos].item()}, correct={correct_labels[pos].item()}")
    
    # 最终验证总结
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    if assistant_start_pos is not None and user_labeled_count == 0 and contains_assistant and not contains_user and labels_match:
        print("✅ ALL CHECKS PASSED: Labels are correctly created!")
        print("   - User portion is properly masked")
        print("   - Only assistant response is labeled")
        print("   - Matches official implementation")
        return True
    else:
        print("❌ SOME CHECKS FAILED: Please review the output above.")
        if not labels_match:
            print("   - Labels do not match official implementation")
        return False


if __name__ == "__main__":
    success = test_labels_creation()
    sys.exit(0 if success else 1)
