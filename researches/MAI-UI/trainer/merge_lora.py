#!/usr/bin/env python3
"""Merge LoRA weights into base model for MAI-UI.

This script merges LoRA adapter weights into the base model, which is necessary
for vLLM deployment since vLLM only supports LoRA for language model parts
and ignores visual encoder LoRA weights in multimodal models.

Usage:
    python merge_lora.py \
        --base-model /workspace/MAI-UI-2B \
        --lora-path /workspace/mai-ui-trainer/trainer/models/sft_model/20260202_150238 \
        --output-path /workspace/MAI-UI-2B-merged \
        --dtype float16

    # Or with auto dtype detection:
    python merge_lora.py \
        --base-model Tongyi-MAI/MAI-UI-2B \
        --lora-path ./models/sft_model/latest \
        --output-path ./models/merged_model

Author: Damon Li
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch # type: ignore
from tqdm import tqdm # type: ignore


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights into base model for MAI-UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge with float16 precision
  python merge_lora.py \\
      --base-model /workspace/MAI-UI-2B \\
      --lora-path /workspace/models/sft_model/20260202_150238 \\
      --output-path /workspace/MAI-UI-2B-merged \\
      --dtype float16

  # Merge with auto dtype (uses model's original dtype)
  python merge_lora.py \\
      --base-model Tongyi-MAI/MAI-UI-2B \\
      --lora-path ./models/sft_model/latest \\
      --output-path ./models/merged_model
        """,
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to the base model (local path or HuggingFace model ID)",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        required=True,
        help="Path to the LoRA adapter weights directory",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the merged model",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype for loading and saving (default: auto)",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map for model loading (default: auto)",
    )
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="2GB",
        help="Maximum size of each model shard when saving (default: 2GB)",
    )
    parser.add_argument(
        "--safe-serialization",
        action="store_true",
        default=True,
        help="Use safetensors format for saving (default: True)",
    )
    parser.add_argument(
        "--no-safe-serialization",
        action="store_false",
        dest="safe_serialization",
        help="Use pytorch bin format instead of safetensors",
    )
    return parser.parse_args()


def get_torch_dtype(dtype_str: str) -> torch.dtype | str:
    """Convert dtype string to torch dtype."""
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, "auto")


def detect_model_type(model_path: str) -> str:
    """Detect model type from config.

    Returns:
        One of: 'qwen3vl', 'qwen2vl', 'causal_lm'
    """
    from transformers import AutoConfig # type: ignore

    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        arch = config.architectures[0] if config.architectures else ""

        if "Qwen3VL" in arch:
            return "qwen3vl"
        elif "Qwen2VL" in arch:
            return "qwen2vl"
        else:
            return "causal_lm"
    except Exception as e:
        print(f"Warning: Could not detect model type: {e}")
        return "causal_lm"


def load_base_model(
    model_path: str,
    dtype: torch.dtype | str,
    device_map: str,
    model_type: str,
):
    """Load base model based on detected type.

    Args:
        model_path: Path to the base model
        dtype: Model dtype
        device_map: Device map strategy
        model_type: Detected model type

    Returns:
        Tuple of (model, tokenizer, processor)
    """
    from transformers import AutoProcessor, AutoTokenizer # type: ignore

    print(f"Loading base model from: {model_path}")
    print(f"  Model type: {model_type}")
    print(f"  Dtype: {dtype}")
    print(f"  Device map: {device_map}")

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
    }
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype

    processor = None
    tokenizer = None
    model = None

    if model_type == "qwen3vl":
        try:
            from transformers import Qwen3VLForConditionalGeneration # type: ignore

            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, **model_kwargs
            )
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = processor.tokenizer
            print("  Loaded as Qwen3VL model")
        except ImportError:
            print("  Warning: Qwen3VL not available, falling back to AutoModel")
            model_type = "causal_lm"

    elif model_type == "qwen2vl":
        try:
            from transformers import Qwen2VLForConditionalGeneration # type: ignore

            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, **model_kwargs
            )
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = processor.tokenizer
            print("  Loaded as Qwen2VL model")
        except ImportError:
            print("  Warning: Qwen2VL not available, falling back to AutoModel")
            model_type = "causal_lm"

    if model is None:
        from transformers import AutoModelForCausalLM # type: ignore

        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("  Loaded as CausalLM model")

    return model, tokenizer, processor


def merge_lora_weights(model, lora_path: str):
    """Merge LoRA weights into the base model.

    Args:
        model: Base model
        lora_path: Path to LoRA adapter

    Returns:
        Merged model
    """
    from peft import PeftModel # type: ignore

    print(f"\nLoading LoRA adapter from: {lora_path}")

    # Verify LoRA path exists and has required files
    lora_path_obj = Path(lora_path)
    if not lora_path_obj.exists():
        raise FileNotFoundError(f"LoRA path does not exist: {lora_path}")

    adapter_config = lora_path_obj / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(
            f"adapter_config.json not found in {lora_path}. "
            "Make sure this is a valid LoRA adapter directory."
        )

    # Check for adapter weights
    has_safetensors = (lora_path_obj / "adapter_model.safetensors").exists()
    has_bin = (lora_path_obj / "adapter_model.bin").exists()
    if not has_safetensors and not has_bin:
        raise FileNotFoundError(
            f"No adapter weights found in {lora_path}. "
            "Expected adapter_model.safetensors or adapter_model.bin"
        )

    print(f"  Adapter config: {adapter_config}")
    print(f"  Weights format: {'safetensors' if has_safetensors else 'bin'}")

    # Load LoRA adapter
    peft_model = PeftModel.from_pretrained(model, lora_path)
    print("  LoRA adapter loaded successfully")

    # Merge and unload
    print("  Merging weights...")
    merged_model = peft_model.merge_and_unload()
    print("  Merge completed")

    return merged_model


def save_merged_model(
    model,
    tokenizer,
    processor,
    output_path: str,
    max_shard_size: str,
    safe_serialization: bool,
):
    """Save the merged model, tokenizer, and processor.

    Args:
        model: Merged model
        tokenizer: Tokenizer
        processor: Processor (for VL models, may be None)
        output_path: Output directory
        max_shard_size: Maximum shard size
        safe_serialization: Whether to use safetensors format
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving merged model to: {output_path}")
    print(f"  Max shard size: {max_shard_size}")
    print(f"  Format: {'safetensors' if safe_serialization else 'pytorch bin'}")

    # Save model
    print("  Saving model weights...")
    model.save_pretrained(
        output_path,
        max_shard_size=max_shard_size,
        safe_serialization=safe_serialization,
    )

    # Save tokenizer
    print("  Saving tokenizer...")
    tokenizer.save_pretrained(output_path)

    # Save processor for VL models
    if processor is not None:
        print("  Saving processor (VL model)...")
        processor.save_pretrained(output_path)

    # List saved files
    print("\nSaved files:")
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.2f} MB")


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("MAI-UI LoRA Merge Tool")
    print("=" * 60)

    # Validate paths
    if not os.path.exists(args.base_model) and not args.base_model.startswith(
        ("Tongyi-MAI/", "Qwen/")
    ):
        # Check if it's a HuggingFace model ID
        print(f"Note: Base model path '{args.base_model}' not found locally.")
        print("      Will attempt to download from HuggingFace Hub.")

    if not os.path.exists(args.lora_path):
        print(f"Error: LoRA path does not exist: {args.lora_path}")
        sys.exit(1)

    # Get dtype
    dtype = get_torch_dtype(args.dtype)

    # Detect model type
    print("\nDetecting model type...")
    model_type = detect_model_type(args.base_model)

    # Load base model
    model, tokenizer, processor = load_base_model(
        args.base_model,
        dtype,
        args.device_map,
        model_type,
    )

    # Merge LoRA weights
    merged_model = merge_lora_weights(model, args.lora_path)

    # Save merged model
    save_merged_model(
        merged_model,
        tokenizer,
        processor,
        args.output_path,
        args.max_shard_size,
        args.safe_serialization,
    )

    print("\n" + "=" * 60)
    print("Merge completed successfully!")
    print("=" * 60)
    print(f"\nMerged model saved to: {args.output_path}")
    print("\nTo deploy with vLLM:")
    print(f"  python -m vllm.entrypoints.openai.api_server \\")
    print(f"      --model {args.output_path} \\")
    print(f"      --served-model-name MAI-UI-merged \\")
    print(f"      --trust-remote-code \\")
    print(f"      --max-model-len 16384")


if __name__ == "__main__":
    main()
