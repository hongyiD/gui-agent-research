#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SFT Trainer for MAI-UI Models.

This script fine-tunes a pre-trained MAI-UI model using trajectory-based SFT data,
with configuration-driven training and proper prompt/response formatting.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

# Try to import Qwen3VL model class if available
try:
    from transformers import Qwen3VLForConditionalGeneration
    HAS_QWEN3VL = True
except ImportError:
    HAS_QWEN3VL = False

# Try to import AutoModelForVision2Seq if available
try:
    from transformers import AutoModelForVision2Seq
    HAS_VISION2SEQ = True
except ImportError:
    HAS_VISION2SEQ = False

# Try to import PEFT for LoRA
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType,
    )
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("Warning: peft not installed. LoRA will not be available. Install with: pip install peft")


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_config(config: dict) -> dict:
    """Normalize config to support both old format and unified format.
    
    Old format:
        data.path
        training.output_dir
        training.max_length
    
    Unified format:
        data.processed_data_file
        model.sft_output_dir
        training.sft.max_length
    """
    normalized = config.copy()
    
    # Check if using unified format (has training.sft or data.processed_data_file)
    is_unified_format = (
        "training" in config and "sft" in config.get("training", {})
    ) or (
        "data" in config and "processed_data_file" in config.get("data", {})
    )
    
    if is_unified_format:
        # Convert unified format to old format
        if "data" not in normalized:
            normalized["data"] = {}
        if "path" not in normalized["data"]:
            # Try processed_data_file first, then fallback to other fields
            normalized["data"]["path"] = (
                config.get("data", {}).get("processed_data_file")
                or config.get("data", {}).get("path")
            )
        
        if "model" not in normalized:
            normalized["model"] = {}
        if "path" not in normalized["model"]:
            normalized["model"]["path"] = config.get("model", {}).get("path")
        
        if "training" not in normalized:
            normalized["training"] = {}
        
        # Extract SFT training config (prefer training.sft over top-level training)
        sft_config = config.get("training", {}).get("sft", {})
        top_level_training = config.get("training", {})
        
        # Map unified format to old format
        if "output_dir" not in normalized["training"]:
            normalized["training"]["output_dir"] = (
                config.get("model", {}).get("sft_output_dir")
                or "./models/sft_model"
            )
        
        # Use sft_config if available, otherwise fallback to top-level training
        source_config = sft_config if sft_config else top_level_training
        
        if "max_length" not in normalized["training"]:
            normalized["training"]["max_length"] = source_config.get("max_length", 2048)
        
        if "num_train_epochs" not in normalized["training"]:
            normalized["training"]["num_train_epochs"] = source_config.get(
                "num_train_epochs", 3
            )
        
        if "per_device_train_batch_size" not in normalized["training"]:
            normalized["training"]["per_device_train_batch_size"] = source_config.get(
                "per_device_train_batch_size", 1
            )
        
        if "gradient_accumulation_steps" not in normalized["training"]:
            normalized["training"]["gradient_accumulation_steps"] = source_config.get(
                "gradient_accumulation_steps", 1
            )
        
        if "learning_rate" not in normalized["training"]:
            normalized["training"]["learning_rate"] = source_config.get(
                "learning_rate", 2.0e-5
            )
        
        if "logging_steps" not in normalized["training"]:
            normalized["training"]["logging_steps"] = source_config.get(
                "logging_steps", 10
            )
        
        if "save_steps" not in normalized["training"]:
            normalized["training"]["save_steps"] = source_config.get("save_steps", 500)
        
        if "warmup_steps" not in normalized["training"]:
            normalized["training"]["warmup_steps"] = source_config.get(
                "warmup_steps", 0
            )
        
        # Copy over LoRA config if present
        if "lora" in top_level_training:
            normalized["training"]["lora"] = top_level_training["lora"]
        
        # Copy over memory optimization settings
        if "gradient_checkpointing" in top_level_training:
            normalized["training"]["gradient_checkpointing"] = top_level_training["gradient_checkpointing"]
        if "use_4bit" in top_level_training:
            normalized["training"]["use_4bit"] = top_level_training["use_4bit"]
    
    return normalized


def merge_args_with_config(args: argparse.Namespace, config: dict) -> dict:
    """Merge CLI arguments with config, CLI takes precedence."""
    merged = config.copy()
    
    # Ensure required sections exist
    if "model" not in merged:
        merged["model"] = {}
    if "data" not in merged:
        merged["data"] = {}
    if "training" not in merged:
        merged["training"] = {}
    
    # Model config
    if args.model_name_or_path:
        merged["model"]["path"] = args.model_name_or_path
    if args.output_dir:
        merged["training"]["output_dir"] = args.output_dir
    
    # Data config
    if args.data_path:
        merged["data"]["path"] = args.data_path
    
    # Training config
    if args.max_length:
        merged["training"]["max_length"] = args.max_length
    if args.num_train_epochs:
        merged["training"]["num_train_epochs"] = args.num_train_epochs
    if args.per_device_train_batch_size:
        merged["training"]["per_device_train_batch_size"] = args.per_device_train_batch_size
    if args.learning_rate:
        merged["training"]["learning_rate"] = args.learning_rate
    if args.save_steps:
        merged["training"]["save_steps"] = args.save_steps
    
    return merged


def convert_messages_to_prompt_response(messages: list[dict]) -> tuple[str, str]:
    """Convert OpenAI messages format to prompt/response format.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        
    Returns:
        Tuple of (prompt, response) strings
    """
    prompt_parts = []
    response = ""
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        # Handle different content formats
        if isinstance(content, list):
            # Multi-modal content (text + images)
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        # For vision models, image URLs are handled separately
                        # Here we just note that an image is present
                        text_parts.append("[IMAGE]")
                elif isinstance(item, str):
                    text_parts.append(item)
            content = " ".join(text_parts)
        elif not isinstance(content, str):
            content = str(content)
        
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            if response:
                # Multiple assistant messages, concatenate
                response += "\n" + content
            else:
                response = content
    
    prompt = "\n".join(prompt_parts)
    return prompt, response


def detect_and_convert_data_format(examples: dict) -> dict:
    """Detect data format and convert to prompt/response format if needed.
    
    Supports:
    - prompt_response format: {"prompt": str, "response": str}
    - openai_messages format: {"messages": [{"role": str, "content": str}]}
    
    Args:
        examples: Dict with batched data (each key maps to a list)
    
    Returns:
        Dict with "prompt" and "response" keys (both are lists in batched mode)
    """
    # Check if already in prompt_response format
    if "prompt" in examples and "response" in examples:
        # Ensure both are lists (for batched processing)
        if isinstance(examples["prompt"], list) and isinstance(examples["response"], list):
            return examples
        else:
            # Convert single values to lists
            return {
                "prompt": [examples["prompt"]] if not isinstance(examples["prompt"], list) else examples["prompt"],
                "response": [examples["response"]] if not isinstance(examples["response"], list) else examples["response"],
            }
    
    # Check if in openai_messages format
    if "messages" in examples:
        prompts = []
        responses = []
        
        # Handle batched examples (messages is a list of message lists)
        messages_list = examples["messages"]
        
        # Ensure messages_list is a list
        if not isinstance(messages_list, list):
            messages_list = [messages_list]
        
        # Process each message list in the batch
        for messages in messages_list:
            # Ensure messages is a list
            if not isinstance(messages, list):
                messages = [messages]
            
            prompt, response = convert_messages_to_prompt_response(messages)
            prompts.append(prompt)
            responses.append(response)
        
        return {"prompt": prompts, "response": responses}
    
    # Unknown format
    available_keys = list(examples.keys())
    raise ValueError(
        f"Unknown data format. Expected 'prompt'/'response' or 'messages' fields. "
        f"Found keys: {available_keys}. "
        f"Please ensure your data is in prompt_response or openai_messages format."
    )


def tokenize_function(examples, tokenizer, max_length: int):
    """Tokenize examples with proper prompt/response masking.
    
    This function correctly handles tokenization by:
    1. Tokenizing prompt and response separately to get accurate lengths
    2. Properly masking prompt tokens with -100 (not used for loss)
    3. Keeping response tokens for loss computation
    4. Handling padding and truncation correctly
    """
    # Detect and convert data format if needed
    examples = detect_and_convert_data_format(examples)
    
    prompts = examples["prompt"]
    responses = examples["response"]
    
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for prompt, response in zip(prompts, responses):
        # Tokenize prompt and response separately
        prompt_tokens = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_tensors=None,
        )
        
        response_tokens = tokenizer(
            response,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_tensors=None,
        )
        
        prompt_input_ids = prompt_tokens["input_ids"]
        response_input_ids = response_tokens["input_ids"]
        
        # Combine prompt and response
        full_input_ids = prompt_input_ids + response_input_ids
        
        # Truncate if too long
        if len(full_input_ids) > max_length:
            # Keep full prompt, truncate response
            available_response_len = max_length - len(prompt_input_ids)
            if available_response_len > 0:
                response_input_ids = response_input_ids[:available_response_len]
                full_input_ids = prompt_input_ids + response_input_ids
            else:
                # Prompt itself is too long, truncate prompt
                full_input_ids = prompt_input_ids[:max_length]
                response_input_ids = []
        
        # Create labels: mask prompt with -100, keep response
        labels = [-100] * len(prompt_input_ids) + response_input_ids
        
        # Create attention mask (all 1s for non-padded tokens)
        attention_mask = [1] * len(full_input_ids)
        
        # Pad to max_length
        padding_length = max_length - len(full_input_ids)
        if padding_length > 0:
            full_input_ids = full_input_ids + [tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            labels = labels + [-100] * padding_length
        
        input_ids_list.append(full_input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)
    
    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to sft_config.yaml")
    parser.add_argument("--model_name_or_path", type=str, help="Pre-trained model name or path")
    parser.add_argument("--data_path", type=str, help="Path to SFT training data (JSONL file)")
    parser.add_argument("--output_dir", type=str, help="Directory to save the fine-tuned model")
    parser.add_argument("--max_length", type=int, help="Maximum sequence length")
    parser.add_argument("--num_train_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--save_steps", type=int, help="Save checkpoint every X steps")
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        # Default config if no file provided
        config = {
            "model": {"path": "Tongyi-MAI/MAI-UI-2B"},
            "data": {"path": None},
            "training": {
                "output_dir": "./mai-ui-2b-sft",
                "max_length": 2048,
                "num_train_epochs": 3,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "learning_rate": 2.0e-5,
                "logging_steps": 10,
                "save_steps": 500,
                "warmup_steps": 100,
            },
        }
    
    # Normalize config to support both old and unified formats
    config = normalize_config(config)
    
    # Merge CLI args
    config = merge_args_with_config(args, config)
    
    # Validate required fields
    model_path = config.get("model", {}).get("path")
    if not model_path:
        raise ValueError(
            "Model path must be provided via --model_name_or_path or config file "
            "(config.model.path or config.model.sft_model_path)"
        )
    
    data_path = config.get("data", {}).get("path")
    if not data_path:
        raise ValueError(
            "Data path must be provided via --data_path or config file "
            "(config.data.path or config.data.processed_data_file)"
        )
    
    # Convert relative path to absolute path if needed
    original_data_path = data_path
    tried_paths = []
    
    if not os.path.isabs(data_path):
        # Try resolving relative to config file directory first
        if args.config:
            config_dir = os.path.dirname(os.path.abspath(args.config))
            config_relative_path = os.path.normpath(os.path.join(config_dir, data_path))
            tried_paths.append(config_relative_path)
            
            if os.path.exists(config_relative_path):
                data_path = config_relative_path
            else:
                # If not found relative to config, try relative to current working directory
                cwd_relative_path = os.path.abspath(data_path)
                tried_paths.append(cwd_relative_path)
                
                if os.path.exists(cwd_relative_path):
                    data_path = cwd_relative_path
                else:
                    # Neither location found, use config-relative path for error message
                    data_path = config_relative_path
        else:
            # No config file, resolve relative to current working directory
            data_path = os.path.abspath(data_path)
            tried_paths.append(data_path)
    
    # Validate data file exists
    if not os.path.exists(data_path):
        error_msg = f"Data file not found: {data_path}\n"
        if tried_paths:
            error_msg += "Tried the following paths:\n"
            for tried_path in tried_paths:
                error_msg += f"  - {tried_path}\n"
        error_msg += "\nPlease ensure the file exists or update the path in your config file."
        error_msg += f"\nOriginal path in config: {original_data_path}"
        raise FileNotFoundError(error_msg)
    
    output_dir = config.get("training", {}).get("output_dir")
    if not output_dir:
        raise ValueError(
            "Output directory must be provided via --output_dir or config file "
            "(config.training.output_dir or config.model.sft_output_dir)"
        )
    
    # Convert relative output directory to absolute path if needed
    if not os.path.isabs(output_dir):
        if args.config:
            config_dir = os.path.dirname(os.path.abspath(args.config))
            output_dir = os.path.normpath(os.path.join(config_dir, output_dir))
        else:
            output_dir = os.path.abspath(output_dir)
    
    max_length = config.get("training", {}).get("max_length", 2048)
    
    # LoRA configuration (enabled by default for memory efficiency)
    lora_config_dict = config.get("training", {}).get("lora", {})
    use_lora = lora_config_dict.get("enabled", True)  # Default to True for memory efficiency
    lora_r = lora_config_dict.get("r", 16)
    lora_alpha = lora_config_dict.get("alpha", 32)
    lora_dropout = lora_config_dict.get("dropout", 0.05)
    lora_target_modules = lora_config_dict.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Quantization config for 4-bit loading (optional, saves more memory)
    use_4bit = config.get("training", {}).get("use_4bit", False)
    
    # Gradient checkpointing (enabled by default for memory efficiency)
    use_gradient_checkpointing = config.get("training", {}).get(
        "gradient_checkpointing", True
    )
    
    # Print configuration summary
    print("=" * 60)
    print("Training Configuration Summary")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Max length: {max_length}")
    print(f"Use LoRA: {use_lora}")
    if use_lora:
        print(f"  LoRA r: {lora_r}")
        print(f"  LoRA alpha: {lora_alpha}")
        print(f"  LoRA dropout: {lora_dropout}")
        print(f"  LoRA target modules: {lora_target_modules}")
    print(f"Use 4-bit quantization: {use_4bit}")
    print(f"Gradient checkpointing: {use_gradient_checkpointing}")
    print("=" * 60)
    print()
    
    # Check if LoRA is requested but peft is not installed
    if use_lora and not HAS_PEFT:
        raise ImportError(
            "LoRA is enabled but peft is not installed. "
            "Install with: pip install peft\n"
            "Or disable LoRA in config: training.lora.enabled: false"
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare quantization config if using 4-bit
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("Using 4-bit quantization for memory efficiency")
    
    # Common model loading kwargs
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
    
    # Try to load as AutoModelForCausalLM first, fallback to vision-language model classes
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    except ValueError as e:
        # If AutoModelForCausalLM fails (e.g., for Qwen3VL), try vision-language model classes
        if "Unrecognized configuration class" in str(e) or "Qwen3VL" in str(e):
            print(
                f"Info: {model_path} appears to be a vision-language model. "
                "Trying Qwen3VLForConditionalGeneration or AutoModel..."
            )
            
            # Try Qwen3VLForConditionalGeneration first (most specific)
            if HAS_QWEN3VL and model is None:
                try:
                    model = Qwen3VLForConditionalGeneration.from_pretrained(
                        model_path, **model_kwargs
                    )
                except Exception as e2:
                    print(f"Qwen3VLForConditionalGeneration failed: {e2}")
            
            # Try AutoModelForVision2Seq if available
            if HAS_VISION2SEQ and model is None:
                try:
                    model = AutoModelForVision2Seq.from_pretrained(
                        model_path, **model_kwargs
                    )
                except Exception as e2:
                    print(f"AutoModelForVision2Seq failed: {e2}")
            
            # Final fallback to AutoModel
            if model is None:
                print("Trying AutoModel as final fallback...")
                model = AutoModel.from_pretrained(model_path, **model_kwargs)
                # Verify model supports generation
                if not hasattr(model, "generate"):
                    raise ValueError(
                        f"Model {model_path} does not support text generation. "
                        "Please use a model that supports causal language modeling."
                    )
        else:
            raise
    
    # Enable gradient checkpointing for memory efficiency
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        print("Gradient checkpointing enabled")
    
    # Apply LoRA if enabled
    if use_lora:
        if use_4bit:
            model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("LoRA applied successfully")
    
    # Load dataset
    dataset = load_dataset("json", data_files=data_path, split="train")
    
    # Print dataset info for debugging
    print(f"Dataset columns: {dataset.column_names}")
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        print(f"Sample keys: {list(dataset[0].keys())}")
        # Show a sample (first 200 chars of each field)
        sample = dataset[0]
        for key, value in sample.items():
            if isinstance(value, str):
                preview = value[:200] + "..." if len(value) > 200 else value
                print(f"  {key}: {preview}")
            elif isinstance(value, list):
                print(f"  {key}: list with {len(value)} items")
                if len(value) > 0:
                    print(f"    First item type: {type(value[0])}")
            else:
                print(f"  {key}: {type(value)}")
    print()
    
    # Tokenize
    def tokenize_fn(examples):
        return tokenize_function(examples, tokenizer, max_length)
    
    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    
    # Set up training arguments
    training_config = config.get("training", {})
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=training_config.get("num_train_epochs", 3),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 16),
        learning_rate=training_config.get("learning_rate", 2.0e-4),
        logging_steps=training_config.get("logging_steps", 10),
        save_steps=training_config.get("save_steps", 500),
        save_total_limit=training_config.get("save_total_limit", 2),
        logging_dir=os.path.join(output_dir, "logs"),
        bf16=training_config.get("bf16", True),
        warmup_steps=training_config.get("warmup_steps", 0),
        gradient_checkpointing=use_gradient_checkpointing,
        optim="adamw_torch",  # Use standard AdamW for better compatibility
        report_to="none",
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    # Start training
    print("Starting SFT training...")
    trainer.train()
    
    # Save the final model
    if use_lora:
        # Save LoRA weights only
        model.save_pretrained(output_dir)
        print(f"LoRA weights saved to {output_dir}")
    else:
        # Save full model
        trainer.save_model(output_dir)
        print(f"Full model saved to {output_dir}")
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f"SFT training completed. Output saved to {output_dir}")


if __name__ == "__main__":
    main()
