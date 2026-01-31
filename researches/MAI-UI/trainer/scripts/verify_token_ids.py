#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证 token ID 获取脚本

用于验证 sft_trainer.py 中动态获取 token ID 的逻辑是否正确。
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from transformers import AutoProcessor
except ImportError:
    print("Error: transformers not installed. Install with: pip install transformers")
    sys.exit(1)


def verify_token_ids(model_path: str):
    """验证 token ID 的获取是否正确。
    
    Args:
        model_path: 模型路径或模型名称
    """
    print(f"Loading processor from: {model_path}")
    
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = processor.tokenizer
    except Exception as e:
        print(f"Error loading processor: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("Token ID Verification")
    print("=" * 60)
    
    # 初始化变量
    IM_START_TOKEN = 151644  # 默认值
    ASSISTANT_START_TOKEN = None
    
    # 1. 验证 assistant token ID（与 sft_trainer.py 保持一致）
    # 注意：sft_trainer.py 编码 "assistant" 获取 token ID，而不是 "<|im_start|>assistant" 的第一个 token
    # 因为 "<|im_start|>assistant" 会被编码为 [<|im_start|>, assistant] = [151644, 77091]
    # 我们需要查找的是 "assistant" token (77091)，并验证它前面是 <|im_start|> token (151644)
    print("\n1. Verifying assistant token ID:")
    try:
        # 首先验证 <|im_start|> token
        im_start_tokens = tokenizer.encode("<|im_start|>", add_special_tokens=False)
        IM_START_TOKEN = im_start_tokens[0] if im_start_tokens else 151644
        print(f"   <|im_start|> token ID: {IM_START_TOKEN} (expected: 151644)")
        
        # 验证完整的 <|im_start|>assistant 编码
        assistant_start_tokens = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
        print(f"   <|im_start|>assistant encoded tokens: {assistant_start_tokens} (expected: [151644, 77091])")
        
        # 编码 "assistant" 获取 token ID（与 sft_trainer.py 一致）
        assistant_tokens = tokenizer.encode("assistant", add_special_tokens=False)
        print(f"   'assistant' encoded tokens: {assistant_tokens}")
        if assistant_tokens:
            ASSISTANT_START_TOKEN = assistant_tokens[0]
            print(f"   ✅ Assistant token ID: {ASSISTANT_START_TOKEN}")
            print(f"   Expected (from qwenvl): 77091")
            if ASSISTANT_START_TOKEN == 77091:
                print(f"   ✅ Matches expected value!")
            else:
                print(f"   ⚠️  Does not match expected value (77091)")
        else:
            print("   ❌ Failed to encode 'assistant'")
            ASSISTANT_START_TOKEN = None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        ASSISTANT_START_TOKEN = None
    
    # 2. 验证 <|im_end|> token ID (eos_token_id)
    print("\n2. Verifying <|im_end|> token ID (eos_token_id):")
    try:
        eos_token_id = getattr(tokenizer, 'eos_token_id', None)
        print(f"   eos_token_id from tokenizer: {eos_token_id}")
        print(f"   Expected (from config.json): 151645")
        if eos_token_id == 151645:
            print(f"   ✅ Matches expected value!")
        elif eos_token_id is not None:
            print(f"   ⚠️  Different value, but will use fallback (151645)")
        else:
            print(f"   ⚠️  eos_token_id not found, will use fallback (151645)")
        
        # 也尝试直接编码
        im_end_tokens = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        print(f"   Encoded <|im_end|> tokens: {im_end_tokens}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 3. 验证完整的对话格式
    print("\n3. Verifying complete conversation format:")
    try:
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        # 使用 apply_chat_template
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        print(f"   Formatted text (first 200 chars):\n   {text[:200]}...")
        
        # Tokenize
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        print(f"   Total tokens: {len(input_ids)}")
        print(f"   First 20 token IDs: {input_ids[:20]}")
        
        # 查找 assistant token 的位置（与 sft_trainer.py 一致）
        # 查找所有有效的 assistant token 位置（前面必须是 <|im_start|>）
        if ASSISTANT_START_TOKEN:
            assistant_positions = []
            for pos in range(len(input_ids)):
                if input_ids[pos] == ASSISTANT_START_TOKEN:
                    # 验证前面是 <|im_start|> token（与 sft_trainer.py 一致）
                    if pos > 0 and input_ids[pos - 1] == IM_START_TOKEN:
                        assistant_positions.append(pos)
            
            print(f"   Assistant token (with <|im_start|> prefix) found at positions: {assistant_positions}")
            
            if assistant_positions:
                pos = assistant_positions[-1]  # 最后一个（assistant 回复）
                print(f"   Using last position: {pos}")
                print(f"   Token IDs around position {pos}: {input_ids[max(0, pos-2):pos+5]}")
                
                # 查找对应的 <|im_end|>
                IM_END_TOKEN = eos_token_id or 151645
                ans_start = pos + 2  # 跳过 "assistant" token 和换行符（与 sft_trainer.py 一致）
                ans_end = ans_start
                while ans_end < len(input_ids) and input_ids[ans_end] != IM_END_TOKEN:
                    ans_end += 1
                
                if ans_end < len(input_ids):
                    print(f"   ✅ Found assistant response range: [{ans_start}, {ans_end+2}]")
                    print(f"   Response tokens: {input_ids[ans_start:ans_end+2]}")
                    print(f"   Decoded response: {tokenizer.decode(input_ids[ans_start:ans_end+2])}")
                else:
                    print(f"   ❌ Could not find <|im_end|> token")
            else:
                print(f"   ⚠️  Could not find valid assistant token position (with <|im_start|> prefix)")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Verification Complete")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify token IDs for Qwen models")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/workspace/MAI-UI-2B",
        help="Model path or name (default: Tongyi-MAI/MAI-UI-2B)"
    )
    
    args = parser.parse_args()
    
    success = verify_token_ids(args.model_path)
    sys.exit(0 if success else 1)
