#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SFT Trainer for MAI-UI Models.

This script fine-tunes a pre-trained MAI-UI model using trajectory-based SFT data,
with configuration-driven training and proper prompt/response formatting.

Supports multi-modal training with base64-encoded images in JSONL format.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import inspect
import io
import json
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from PIL import Image
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
import time


class DetailedLoggingCallback(TrainerCallback):
    """Callback for printing detailed training progress and GPU stats."""
    
    def __init__(self):
        self.start_time = None
        self.step_start_time = None
        self.total_steps = 0
        self.current_epoch = 0
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.total_steps = state.max_steps
        print("\n" + "=" * 70)
        print("TRAINING STARTED")
        print("=" * 70)
        print(f"  Total steps: {self.total_steps}")
        print(f"  Epochs: {args.num_train_epochs}")
        print(f"  Batch size: {args.per_device_train_batch_size}")
        print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"  Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  FP16: {args.fp16}, BF16: {args.bf16}")
        print(f"  Gradient checkpointing: {args.gradient_checkpointing}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  GPU Memory: {mem_total:.1f} GB total")
        print("=" * 70 + "\n")
    
    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        step_time = time.time() - self.step_start_time if self.step_start_time else 0
        
        # Only log at logging_steps interval or first step
        if state.global_step % args.logging_steps == 0 or state.global_step == 1:
            # Get GPU memory stats
            gpu_info = ""
            if torch.cuda.is_available():
                mem_used = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_util = (mem_used / mem_total) * 100
                gpu_info = f" | VRAM: {mem_used:.1f}/{mem_total:.1f}GB ({gpu_util:.0f}%)"
            
            # Calculate ETA
            elapsed = time.time() - self.start_time if self.start_time else 0
            steps_done = state.global_step
            steps_remaining = self.total_steps - steps_done
            if steps_done > 0:
                avg_step_time = elapsed / steps_done
                eta_seconds = avg_step_time * steps_remaining
                eta_str = self._format_time(eta_seconds)
            else:
                eta_str = "calculating..."
            
            # Get current loss and learning rate
            loss_str = ""
            if state.log_history:
                for log in reversed(state.log_history):
                    if "loss" in log:
                        loss_str = f" | loss: {log['loss']:.4f}"
                        break
            
            lr_str = ""
            if state.log_history:
                for log in reversed(state.log_history):
                    if "learning_rate" in log:
                        lr_str = f" | lr: {log['learning_rate']:.2e}"
                        break
            
            print(
                f"[Step {state.global_step}/{self.total_steps}] "
                f"time: {step_time:.2f}s{loss_str}{lr_str}{gpu_info} | ETA: {eta_str}"
            )
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.current_epoch += 1
        print(f"\n--- Epoch {self.current_epoch}/{int(args.num_train_epochs)} ---")
    
    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time if self.start_time else 0
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED")
        print("=" * 70)
        print(f"  Total time: {self._format_time(total_time)}")
        print(f"  Total steps: {state.global_step}")
        if state.log_history:
            final_loss = None
            for log in reversed(state.log_history):
                if "loss" in log:
                    final_loss = log["loss"]
                    break
            if final_loss:
                print(f"  Final loss: {final_loss:.4f}")
        if torch.cuda.is_available():
            max_mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  Peak GPU memory: {max_mem:.2f} GB")
        print("=" * 70 + "\n")
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = seconds / 60
            return f"{mins:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

# Try to import Qwen2VL model and processor
try:
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
    HAS_QWEN2VL = True
except ImportError:
    HAS_QWEN2VL = False

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


def decode_base64_image(data_url: str) -> Image.Image:
    """Decode base64 image from data URL.
    
    Args:
        data_url: Data URL like "data:image/png;base64,iVBORw0KG..."
        
    Returns:
        PIL Image object
    """
    # Extract base64 data from data URL
    if data_url.startswith("data:"):
        # Format: data:image/png;base64,xxxxx
        match = re.match(r"data:[^;]+;base64,(.+)", data_url)
        if match:
            base64_data = match.group(1)
        else:
            raise ValueError(f"Invalid data URL format: {data_url[:50]}...")
    else:
        # Assume it's raw base64
        base64_data = data_url
    
    # Decode base64 to bytes
    image_bytes = base64.b64decode(base64_data)
    
    # Open as PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary (some PNGs have alpha channel)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    return image


def extract_images_from_messages(
    messages: list[dict],
    data_dir: str | None = None,
) -> list[Image.Image]:
    """Extract all images from OpenAI messages format.
    
    Supports:
    - Base64 encoded images (data:image/... URLs)
    - File paths (relative or absolute)
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        data_dir: Base directory for resolving relative image paths
        
    Returns:
        List of PIL Image objects
    """
    images = []
    
    for msg in messages:
        content = msg.get("content", "")
        
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    image_url = item.get("image_url", {})
                    if isinstance(image_url, dict):
                        url = image_url.get("url", "")
                    else:
                        url = image_url
                    
                    if url:
                        try:
                            # Check if it's a base64 data URL
                            if url.startswith("data:"):
                                image = decode_base64_image(url)
                                images.append(image)
                            else:
                                # Treat as file path
                                image_path = url
                                if not os.path.isabs(image_path) and data_dir:
                                    image_path = os.path.join(data_dir, image_path)
                                
                                if os.path.exists(image_path):
                                    image = Image.open(image_path)
                                    if image.mode != "RGB":
                                        image = image.convert("RGB")
                                    images.append(image)
                                else:
                                    print(f"Warning: Image file not found: {image_path}")
                        except Exception as e:
                            print(f"Warning: Failed to load image: {e}")
    
    return images


def extract_images_from_prompt_response(prompt: str, data_dir: str | None = None) -> list[Image.Image]:
    """Extract images from prompt_response format prompt string.
    
    Supports:
    - <image_base64>...</image_base64>: Base64 encoded image data
    - <image_path>...</image_path>: Path to image file (relative or absolute)
    
    Args:
        prompt: Prompt string that may contain image tags
        data_dir: Base directory for resolving relative image paths
        
    Returns:
        List of PIL Image objects
    """
    images = []
    
    # Extract base64 images
    base64_pattern = r"<image_base64>(.*?)</image_base64>"
    for match in re.finditer(base64_pattern, prompt, re.DOTALL):
        base64_data = match.group(1).strip()
        try:
            image = decode_base64_image(base64_data)
            images.append(image)
        except Exception as e:
            print(f"Warning: Failed to decode base64 image: {e}")
    
    # Extract path images
    path_pattern = r"<image_path>(.*?)</image_path>"
    for match in re.finditer(path_pattern, prompt, re.DOTALL):
        image_path = match.group(1).strip()
        
        # Resolve path
        if not os.path.isabs(image_path):
            if data_dir:
                # Try relative to data directory
                resolved_path = os.path.join(data_dir, image_path)
                if not os.path.exists(resolved_path):
                    # Try just the filename in data directory
                    resolved_path = os.path.join(data_dir, os.path.basename(image_path))
            else:
                resolved_path = image_path
        else:
            resolved_path = image_path
        
        # Load image from file
        if os.path.exists(resolved_path):
            try:
                image = Image.open(resolved_path)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                images.append(image)
            except Exception as e:
                print(f"Warning: Failed to load image from {resolved_path}: {e}")
        else:
            print(f"Warning: Image path not found: {resolved_path} (original: {image_path})")
    
    return images


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
                        # Use <image> placeholder for VL models
                        text_parts.append("<image>")
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


def convert_full_trajectory_to_prompt_response(trajectory: dict) -> list[tuple[str, str]]:
    """Convert full_trajectory format to list of (prompt, response) pairs.
    
    Full trajectory format contains multiple steps, each step becomes a training sample.
    
    Args:
        trajectory: Full trajectory dict with 'task_goal', 'steps', 'metadata'
    
    Returns:
        List of (prompt, response) tuples, one per step
    """
    import sys
    from pathlib import Path
    
    # Try to import official prompt
    system_prompt = "You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task."
    
    # Try importing from prompts directory
    prompts_file = Path(__file__).parent / "prompts" / "maiui_official_prompts.py"
    if prompts_file.exists():
        try:
            sys.path.insert(0, str(prompts_file.parent))
            from maiui_official_prompts import MAI_MOBILE_SYS_PROMPT_ASK_USER_MCP
            system_prompt = MAI_MOBILE_SYS_PROMPT_ASK_USER_MCP.render(tools="")
        except ImportError:
            pass
    
    task_goal = trajectory.get("task_goal", "")
    steps = trajectory.get("steps", [])
    metadata = trajectory.get("metadata", {})
    
    samples = []
    
    for i, step in enumerate(steps):
        # Build prompt with task goal and history
        prompt_parts = [system_prompt]
        prompt_parts.append(f"\n## Task Goal\n{task_goal}\n")
        
        # Add history (previous steps)
        if i > 0:
            prompt_parts.append("## Action History\n")
            for hist_step in steps[max(0, i-3):i]:  # Last 3 steps
                hist_thinking = hist_step.get("thinking", "")
                hist_action = hist_step.get("action", {})
                if hist_thinking:
                    prompt_parts.append(f"Step {hist_step.get('step_index', '?')}: {hist_thinking}")
                if hist_action:
                    prompt_parts.append(f"Action: {json.dumps(hist_action, ensure_ascii=False)}")
                prompt_parts.append("")
        
        # Add current observation (screenshot)
        screenshot_path = step.get("screenshot_path", "")
        if screenshot_path:
            prompt_parts.append("## Current Observation\n")
            prompt_parts.append(f"<image_path>{screenshot_path}</image_path>")
        
        prompt = "\n".join(prompt_parts)
        
        # Build response (thinking + action)
        thinking = step.get("thinking", "")
        action = step.get("action", {})
        
        response_parts = []
        if thinking:
            response_parts.append(f"<thinking>\n{thinking}\n</thinking>")
        if action:
            action_json = json.dumps(action, ensure_ascii=False)
            response_parts.append(f"<tool_call>\n{{\"name\": \"mobile_use\", \"arguments\": {action_json}}}\n</tool_call>")
        
        response = "\n".join(response_parts)
        
        samples.append((prompt, response))
    
    return samples


def detect_and_convert_data_format(examples: dict) -> dict:
    """Detect data format and convert to prompt/response format if needed.
    
    Supports:
    - prompt_response format: {"prompt": str, "response": str}
    - openai_messages format: {"messages": [{"role": str, "content": str}]}
    - full_trajectory format: {"task_goal": str, "steps": [...], "metadata": {...}}
    
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
    
    # Check if in full_trajectory format
    if "task_goal" in examples and "steps" in examples:
        prompts = []
        responses = []
        
        # Handle batched examples (trajectories is a list of trajectory dicts)
        trajectories_list = []
        
        # Check if examples contains a single trajectory or multiple
        if isinstance(examples["task_goal"], list):
            # Multiple trajectories
            for i in range(len(examples["task_goal"])):
                trajectory = {
                    "task_goal": examples["task_goal"][i],
                    "steps": examples["steps"][i] if i < len(examples["steps"]) else [],
                    "metadata": examples.get("metadata", [{}])[i] if isinstance(examples.get("metadata"), list) else examples.get("metadata", {}),
                }
                trajectories_list.append(trajectory)
        else:
            # Single trajectory
            trajectory = {
                "task_goal": examples["task_goal"],
                "steps": examples["steps"],
                "metadata": examples.get("metadata", {}),
            }
            trajectories_list.append(trajectory)
        
        # Convert each trajectory to multiple (prompt, response) pairs
        for trajectory in trajectories_list:
            step_samples = convert_full_trajectory_to_prompt_response(trajectory)
            for prompt, response in step_samples:
                prompts.append(prompt)
                responses.append(response)
        
        return {"prompt": prompts, "response": responses}
    
    # Unknown format
    available_keys = list(examples.keys())
    raise ValueError(
        f"Unknown data format. Expected one of:\n"
        f"  - prompt_response: 'prompt'/'response' fields\n"
        f"  - openai_messages: 'messages' field\n"
        f"  - full_trajectory: 'task_goal'/'steps' fields\n"
        f"Found keys: {available_keys}."
    )


def tokenize_function(examples, tokenizer, max_length: int):
    """Tokenize examples with proper prompt/response masking (text-only fallback).
    
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


class ListDataset(torch.utils.data.Dataset):
    """Simple Dataset wrapper for list data."""
    
    def __init__(self, data: list[dict]):
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]


class PreprocessedMultiModalDataset(torch.utils.data.Dataset):
    """Dataset that preprocesses images ONCE during initialization.
    
    This avoids the massive overhead of decoding base64 images on every batch.
    Images are decoded once and cached in memory.
    
    Supports both openai_messages and prompt_response formats.
    """
    
    def __init__(
        self, 
        raw_data: list[dict], 
        processor: Any,
        max_length: int = 2048,
        max_images_per_sample: int = 3,  # Limit images to avoid OOM/slow training
        show_progress: bool = True,
        data_dir: str | None = None,  # Base directory for resolving image paths
    ):
        self.processor = processor
        self.max_length = max_length
        self.max_images_per_sample = max_images_per_sample
        self.data_dir = data_dir
        self.preprocessed_data = []
        
        print(f"\n{'='*60}")
        print(f"DATASET PREPROCESSING")
        print(f"{'='*60}")
        print(f"  Max images per sample: {max_images_per_sample}")
        print(f"  Total samples: {len(raw_data)}")
        if data_dir:
            print(f"  Data directory: {data_dir}")
        
        iterator = raw_data
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(raw_data, desc="Decoding images")
        
        total_images = 0
        total_original_images = 0
        truncated_samples = 0
        image_sizes = []
        
        for idx, sample in enumerate(iterator):
            images = []
            
            # Check format and extract images accordingly
            if "messages" in sample:
                # openai_messages format
                messages = sample.get("messages", [])
                images = extract_images_from_messages(messages, data_dir=self.data_dir)
            elif "prompt" in sample:
                # prompt_response format - extract from prompt string
                prompt = sample.get("prompt", "")
                images = extract_images_from_prompt_response(prompt, data_dir=self.data_dir)
            else:
                # Unknown format, skip
                pass
            
            original_count = len(images)
            total_original_images += original_count
            
            # Limit number of images per sample to avoid OOM and slow training
            # Keep only the LAST N images (most recent/relevant for GUI tasks)
            if len(images) > max_images_per_sample:
                truncated_samples += 1
                images = images[-max_images_per_sample:]
            
            # Collect image size stats
            for img in images:
                if hasattr(img, 'size'):
                    image_sizes.append(img.size)
            
            total_images += len(images)
            
            # Store preprocessed data
            self.preprocessed_data.append({
                "sample": sample,  # Store original sample for processing
                "images": images,  # Already decoded PIL Images (limited)
            })
        
        # Print summary statistics
        print(f"\n{'='*60}")
        print(f"PREPROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"  Samples processed: {len(self.preprocessed_data)}")
        print(f"  Original images: {total_original_images} ({total_original_images/len(raw_data):.1f} avg/sample)")
        print(f"  After limiting: {total_images} ({total_images/len(raw_data):.1f} avg/sample)")
        if truncated_samples > 0:
            print(f"  Truncated samples: {truncated_samples} ({truncated_samples*100/len(raw_data):.0f}%)")
        if image_sizes:
            avg_w = sum(s[0] for s in image_sizes) / len(image_sizes)
            avg_h = sum(s[1] for s in image_sizes) / len(image_sizes)
            print(f"  Avg image size: {avg_w:.0f}x{avg_h:.0f} pixels")
        print(f"{'='*60}\n")
    
    def __len__(self) -> int:
        return len(self.preprocessed_data)
    
    def __getitem__(self, idx: int) -> dict:
        return self.preprocessed_data[idx]


@dataclass
class MultiModalDataCollator:
    """Data collator for multi-modal (image + text) training.
    
    This collator processes each sample individually using the VL processor,
    handling base64 images from the JSONL data.
    
    For Qwen2VL/Qwen3VL, we use the processor's apply_chat_template method
    to properly format messages with images.
    """
    
    processor: Any
    max_length: int = 2048
    
    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        """Collate batch of features into model inputs."""
        import time
        batch_start = time.time()
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_pixel_values = []
        batch_image_grid_thw = []
        
        for i, feature in enumerate(features):
            sample_start = time.time()
            
            # Use preprocessed images if available (from PreprocessedMultiModalDataset)
            # This avoids decoding base64 on every batch - HUGE speedup!
            if "images" in feature:
                images = feature["images"]
            else:
                images = []
            
            # Handle different data formats
            sample = feature.get("sample", feature)  # Get original sample if available
            
            if "messages" in sample:
                # openai_messages format
                messages = sample.get("messages", [])
                if not images:
                    images = extract_images_from_messages(messages)
                # Convert messages to Qwen VL format
                qwen_messages = self._convert_to_qwen_format(messages, images)
            elif "prompt" in sample:
                # prompt_response format - convert to messages format
                prompt = sample.get("prompt", "")
                response = sample.get("response", "")
                if not images:
                    # Try to get data_dir from metadata or sample
                    data_dir = None
                    if "metadata" in sample:
                        source = sample.get("metadata", {}).get("source", "")
                        if source:
                            data_dir = os.path.dirname(source)
                    images = extract_images_from_prompt_response(prompt, data_dir=data_dir)
                # Convert prompt_response to messages format for Qwen VL
                qwen_messages = self._convert_prompt_response_to_qwen_format(prompt, response, images)
            else:
                # Fallback: try to extract from messages
                messages = sample.get("messages", [])
                if not images:
                    images = extract_images_from_messages(messages)
                qwen_messages = self._convert_to_qwen_format(messages, images)
            
            try:
                # Use apply_chat_template if available (preferred method)
                if hasattr(self.processor, 'apply_chat_template'):
                    t0 = time.time()
                    text = self.processor.apply_chat_template(
                        qwen_messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    t1 = time.time()
                    
                    if images:
                        inputs = self.processor(
                            text=[text],
                            images=images,
                            padding=False,
                            return_tensors="pt",
                        )
                    else:
                        inputs = self.processor(
                            text=[text],
                            padding=False,
                            return_tensors="pt",
                        )
                    t2 = time.time()
                    
                    # Debug: print timing for first batch only
                    if i == 0 and len(batch_input_ids) == 0:
                        print(f"[DEBUG] apply_chat_template: {t1-t0:.2f}s, processor: {t2-t1:.2f}s")
                else:
                    # Fallback: manual formatting
                    text = self._format_messages_manually(qwen_messages)
                    if images:
                        inputs = self.processor(
                            text=[text],
                            images=images,
                            padding=False,
                            return_tensors="pt",
                        )
                    else:
                        inputs = self.processor(
                            text=[text],
                            padding=False,
                            return_tensors="pt",
                        )
                
                input_ids = inputs["input_ids"].squeeze(0)
                attention_mask = inputs["attention_mask"].squeeze(0)
                
                # Create labels: mask all tokens except the assistant response
                labels = self._create_labels(input_ids, qwen_messages)
                
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_labels.append(labels)
                
                if "pixel_values" in inputs:
                    pv = inputs["pixel_values"]
                    if pv.dim() == 5:  # [batch, num_images, C, H, W]
                        pv = pv.squeeze(0)
                    batch_pixel_values.append(pv)
                if "image_grid_thw" in inputs:
                    igt = inputs["image_grid_thw"]
                    if igt.dim() == 3:
                        igt = igt.squeeze(0)
                    batch_image_grid_thw.append(igt)
                    
            except Exception as e:
                print(f"Warning: Failed to process sample: {e}")
                # Create a minimal valid sample
                dummy_text = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi<|im_end|>"
                inputs = self.processor.tokenizer(
                    dummy_text,
                    return_tensors="pt",
                    padding=False,
                )
                batch_input_ids.append(inputs["input_ids"].squeeze(0))
                batch_attention_mask.append(inputs["attention_mask"].squeeze(0))
                batch_labels.append(torch.full_like(inputs["input_ids"].squeeze(0), -100))
        
        # Pad sequences
        result = self._pad_batch(
            batch_input_ids, batch_attention_mask, batch_labels
        )
        
        # Handle pixel values
        if batch_pixel_values:
            try:
                # Try to stack (same shape)
                result["pixel_values"] = torch.cat(batch_pixel_values, dim=0)
            except Exception:
                # Different shapes - this is expected for variable image sizes
                result["pixel_values"] = torch.cat([
                    pv.reshape(-1, *pv.shape[-3:]) for pv in batch_pixel_values
                ], dim=0)
        
        if batch_image_grid_thw:
            try:
                result["image_grid_thw"] = torch.cat(batch_image_grid_thw, dim=0)
            except Exception:
                result["image_grid_thw"] = torch.cat([
                    igt.reshape(-1, 3) for igt in batch_image_grid_thw
                ], dim=0)
        
        # Debug: print batch collation time
        batch_time = time.time() - batch_start
        if batch_time > 1.0:  # Only print if took more than 1 second
            print(f"[DEBUG] Batch collation took {batch_time:.2f}s for {len(features)} samples")
        
        return result
    
    def _convert_to_qwen_format(
        self, messages: list[dict], images: list[Image.Image]
    ) -> list[dict]:
        """Convert OpenAI format messages to Qwen VL format."""
        qwen_messages = []
        image_idx = 0
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if isinstance(content, str):
                qwen_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                # Multi-modal content
                qwen_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            qwen_content.append({
                                "type": "text",
                                "text": item.get("text", "")
                            })
                        elif item.get("type") == "image_url":
                            if image_idx < len(images):
                                qwen_content.append({
                                    "type": "image",
                                    "image": images[image_idx]
                                })
                                image_idx += 1
                qwen_messages.append({"role": role, "content": qwen_content})
        
        return qwen_messages
    
    def _convert_prompt_response_to_qwen_format(
        self, prompt: str, response: str, images: list[Image.Image]
    ) -> list[dict]:
        """Convert prompt_response format to Qwen VL messages format.
        
        Args:
            prompt: Prompt string (may contain <image_base64> or <image_path> tags)
            response: Response string
            images: List of PIL Image objects
            
        Returns:
            List of message dicts in Qwen VL format
        """
        # Remove image tags from prompt (images are passed separately)
        prompt_clean = re.sub(r"<image_base64>.*?</image_base64>", "<image>", prompt, flags=re.DOTALL)
        prompt_clean = re.sub(r"<image_path>.*?</image_path>", "<image>", prompt_clean, flags=re.DOTALL)
        
        # Build messages
        messages = []
        
        # Extract system prompt if present
        if "You are a GUI agent" in prompt_clean or "System:" in prompt_clean:
            # Try to extract system prompt
            system_match = re.search(r"(You are a GUI agent[^\n]*(?:\n[^\n]*)*)", prompt_clean)
            if system_match:
                system_prompt = system_match.group(1).strip()
                messages.append({"role": "system", "content": system_prompt})
                prompt_clean = prompt_clean.replace(system_prompt, "").strip()
        
        # Build user content with images
        user_content = [{"type": "text", "text": prompt_clean}]
        for img in images:
            user_content.append({"type": "image", "image": img})
        
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": response})
        
        return messages
    
    def _format_messages_manually(self, messages: list[dict]) -> str:
        """Fallback: format messages manually."""
        text_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if isinstance(content, list):
                content_text = " ".join(
                    item.get("text", "") 
                    for item in content 
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            else:
                content_text = str(content)
            
            text_parts.append(f"<|im_start|>{role}\n{content_text}<|im_end|>")
        
        return "\n".join(text_parts)
    
    def _create_labels(
        self, input_ids: torch.Tensor, messages: list[dict]
    ) -> torch.Tensor:
        """Create labels with prompt tokens masked (-100)."""
        labels = input_ids.clone()
        
        # Find assistant response tokens
        # Simple heuristic: mask everything before the last assistant turn
        # For proper masking, we would need to track token positions
        
        # Get the assistant response text
        assistant_response = ""
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    assistant_response = content
                elif isinstance(content, list):
                    assistant_response = " ".join(
                        item.get("text", "") 
                        for item in content 
                        if isinstance(item, dict) and item.get("type") == "text"
                    )
        
        if assistant_response:
            # Tokenize just the response to find its length
            response_tokens = self.processor.tokenizer(
                assistant_response,
                add_special_tokens=False,
                return_tensors="pt",
            )
            response_len = response_tokens["input_ids"].shape[1]
            
            # Mask everything except the last response_len tokens
            # (plus a few tokens for <|im_end|> etc.)
            mask_len = max(0, len(labels) - response_len - 5)
            labels[:mask_len] = -100
        else:
            # No assistant response found, mask all
            labels[:] = -100
        
        return labels
    
    def _pad_batch(
        self,
        batch_input_ids: list[torch.Tensor],
        batch_attention_mask: list[torch.Tensor],
        batch_labels: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Pad batch to same length."""
        max_len = max(ids.shape[0] for ids in batch_input_ids)
        pad_token_id = self.processor.tokenizer.pad_token_id or 0
        
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for input_ids, attn_mask, labels in zip(
            batch_input_ids, batch_attention_mask, batch_labels
        ):
            pad_len = max_len - input_ids.shape[0]
            if pad_len > 0:
                input_ids = torch.cat([
                    input_ids,
                    torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype)
                ])
                attn_mask = torch.cat([
                    attn_mask,
                    torch.zeros(pad_len, dtype=attn_mask.dtype)
                ])
                labels = torch.cat([
                    labels,
                    torch.full((pad_len,), -100, dtype=labels.dtype)
                ])
            
            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attn_mask)
            padded_labels.append(labels)
        
        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
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
    # Note: output_dir is resolved relative to current working directory (not config file)
    # This is more intuitive for users who run the script from the trainer directory
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)
    
    # Add timestamp subdirectory to avoid overwriting previous runs
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, timestamp)
    
    max_length = config.get("training", {}).get("max_length", 2048)
    
    # LoRA configuration (enabled by default for memory efficiency)
    # Default values aligned with Qwen3-VL official training code
    lora_config_dict = config.get("training", {}).get("lora", {})
    use_lora = lora_config_dict.get("enabled", True)  # Default to True for memory efficiency
    lora_r = lora_config_dict.get("r", 64)  # Qwen3-VL default: 64
    lora_alpha = lora_config_dict.get("alpha", 128)  # Qwen3-VL default: 128
    lora_dropout = lora_config_dict.get("dropout", 0.05)
    lora_target_modules = lora_config_dict.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj"  # Qwen attention layers only
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
    
    # Prepare quantization config if using 4-bit
    quantization_config = None
    if use_4bit:
        # Detect compute capability for dtype selection
        # V100 (compute capability 7.0) does not support bfloat16
        if torch.cuda.is_available():
            compute_cap = torch.cuda.get_device_capability()
            if compute_cap[0] >= 8:
                bnb_compute_dtype = torch.bfloat16
            else:
                bnb_compute_dtype = torch.float16
        else:
            bnb_compute_dtype = torch.float32
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=bnb_compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        print(f"Using 4-bit quantization with compute dtype: {bnb_compute_dtype}")
    
    # Detect compute capability for dtype selection
    # V100 (compute capability 7.0) does not support bfloat16, use float16 instead
    if torch.cuda.is_available():
        compute_capability = torch.cuda.get_device_capability()
        # bfloat16 requires compute capability >= 8.0 (A100, etc.)
        if compute_capability[0] >= 8:
            default_dtype = torch.bfloat16
            print(f"GPU compute capability {compute_capability}, using bfloat16")
        else:
            default_dtype = torch.float16
            print(f"GPU compute capability {compute_capability}, using float16 (V100 or older)")
    else:
        default_dtype = torch.float32
        print("No CUDA available, using float32")
    
    # Common model loading kwargs
    # Note: Use 'dtype' instead of deprecated 'torch_dtype' for newer transformers
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    else:
        # Try 'dtype' first (newer API), fall back to 'torch_dtype' for compatibility
        model_kwargs["torch_dtype"] = default_dtype
    
    # Detect if this is a VL model and load appropriate processor/model
    is_vl_model = False
    processor = None
    tokenizer = None
    model = None
    
    # Try to load as Qwen3VL first (MAI-UI-2B is based on Qwen3-VL architecture)
    if HAS_QWEN3VL:
        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
            is_vl_model = True
            tokenizer = processor.tokenizer
            print(f"Loaded as Qwen3VL model with multi-modal support")
        except Exception as e:
            print(f"Qwen3VL loading failed: {e}, trying other methods...")
    
    # Try Qwen2VL if Qwen3VL failed (for older models)
    if model is None and HAS_QWEN2VL:
        try:
            processor = Qwen2VLProcessor.from_pretrained(model_path, trust_remote_code=True)
            model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
            is_vl_model = True
            tokenizer = processor.tokenizer
            print(f"Loaded as Qwen2VL model with multi-modal support")
        except Exception as e:
            print(f"Qwen2VL loading failed: {e}, trying other methods...")
    
    # Try AutoModelForVision2Seq
    if model is None and HAS_VISION2SEQ:
        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForVision2Seq.from_pretrained(model_path, **model_kwargs)
            is_vl_model = True
            tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
            print(f"Loaded as Vision2Seq model with multi-modal support")
        except Exception as e:
            print(f"AutoModelForVision2Seq loading failed: {e}, trying other methods...")
    
    # Fallback to text-only model
    if model is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            print(f"Loaded as text-only CausalLM model")
        except Exception as e:
            # Final fallback to AutoModel
            print(f"AutoModelForCausalLM failed: {e}, trying AutoModel...")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_path, **model_kwargs)
            if not hasattr(model, "generate"):
                raise ValueError(
                    f"Model {model_path} does not support text generation. "
                    "Please use a model that supports causal language modeling."
                )
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"VL Model: {is_vl_model}")
    
    # Disable KV cache for training (following Qwen3-VL official training)
    model.config.use_cache = False
    
    # Enable gradient checkpointing for memory efficiency
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        print("Gradient checkpointing enabled")
    
    # Apply LoRA if enabled
    if use_lora:
        # Freeze all parameters first (following Qwen3-VL official training)
        for p in model.parameters():
            p.requires_grad = False
        print("All model parameters frozen for LoRA training")
        
        if use_4bit:
            model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA with task_type (aligned with Qwen3-VL official)
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
    
    # Load dataset manually to avoid PyArrow type inconsistency issues
    # (messages[].content can be either string or array in OpenAI format)
    print(f"Loading dataset from: {data_path}")
    raw_data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    raw_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
    
    # Print dataset info for debugging
    print(f"Dataset size: {len(raw_data)}")
    if len(raw_data) > 0:
        sample = raw_data[0]
        print(f"Sample keys: {list(sample.keys())}")
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
    
    # Check if data has images (multi-modal)
    has_images = False
    if len(raw_data) > 0:
        sample = raw_data[0]
        # Check openai_messages format
        if "messages" in sample:
            messages = sample.get("messages", [])
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            has_images = True
                            break
                if has_images:
                    break
        # Check full_trajectory format
        elif "steps" in sample:
            steps = sample.get("steps", [])
            for step in steps[:1]:  # Check first step only
                if step.get("screenshot_path"):
                    has_images = True
                    break
        # Check prompt_response format (may contain image paths in prompt)
        elif "prompt" in sample:
            prompt = sample.get("prompt", "")
            if "<image" in prompt.lower() or "screenshot" in prompt.lower():
                has_images = True
    
    print(f"Data contains images: {has_images}")
    
    # Set up training arguments
    training_config = config.get("training", {})
    
    # Determine mixed precision settings based on GPU capability
    # V100 (compute capability 7.0) only supports fp16, not bf16
    use_bf16 = False
    use_fp16 = False
    if torch.cuda.is_available():
        compute_cap = torch.cuda.get_device_capability()
        if compute_cap[0] >= 8:
            # A100, H100, etc. - use bf16
            use_bf16 = training_config.get("bf16", True)
            print(f"Training with bf16={use_bf16} (GPU supports bfloat16)")
        else:
            # V100 or older - use fp16 instead
            use_fp16 = training_config.get("bf16", True)  # Use the bf16 config to decide if mixed precision is wanted
            print(f"Training with fp16={use_fp16} (GPU does not support bfloat16, using float16)")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=training_config.get("num_train_epochs", 3),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 16),
        learning_rate=training_config.get("learning_rate", 2.0e-5),
        logging_steps=training_config.get("logging_steps", 10),
        save_steps=training_config.get("save_steps", 500),
        save_total_limit=training_config.get("save_total_limit", 2),
        logging_dir=os.path.join(output_dir, "logs"),
        bf16=use_bf16,
        fp16=use_fp16,
        warmup_steps=training_config.get("warmup_steps", 0),
        gradient_checkpointing=use_gradient_checkpointing,
        optim="adamw_torch",  # Use standard AdamW for better compatibility
        report_to="none",
        remove_unused_columns=False,  # Important for multi-modal data
        dataloader_pin_memory=False,  # Helps with multi-modal data loading
    )
    
    # Choose data collator based on model type and data
    if is_vl_model and has_images and processor is not None:
        # Get max images per sample from config (default 3 to avoid OOM/slow training)
        max_images_per_sample = training_config.get("max_images_per_sample", 3)
        print(f"Using MultiModalDataCollator for vision-language training (max {max_images_per_sample} images/sample)")
        
        data_collator = MultiModalDataCollator(
            processor=processor,
            max_length=max_length,
        )
        # Use PreprocessedMultiModalDataset to decode images ONCE (not every batch)
        # This provides a HUGE speedup compared to decoding base64 on every iteration
        # Determine data directory for resolving image paths
        # Use the directory containing the data file (sft_train.jsonl)
        data_dir = os.path.dirname(os.path.abspath(data_path))
        print(f"Image base directory: {data_dir}")
        
        train_dataset = PreprocessedMultiModalDataset(
            raw_data=raw_data,
            processor=processor,
            max_length=max_length,
            max_images_per_sample=max_images_per_sample,
            show_progress=True,
            data_dir=data_dir,
        )
    else:
        print("Using text-only tokenization")
        # Convert to HuggingFace Dataset for tokenization
        from datasets import Dataset
        dataset = Dataset.from_list(raw_data)
        
        # Tokenize for text-only training
        def tokenize_fn(examples):
            return tokenize_function(examples, tokenizer, max_length)
        
        train_dataset = dataset.map(
            tokenize_fn, 
            batched=True, 
            remove_columns=dataset.column_names
        )
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Initialize Trainer with detailed logging callback
    # Use processing_class instead of deprecated tokenizer parameter (transformers >= 4.46)
    logging_callback = DetailedLoggingCallback()
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            callbacks=[logging_callback],
        )
    else:
        # Fallback for older transformers versions
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[logging_callback],
        )
    
    # Start training
    print("\nNOTE: First step may be slow due to JIT compilation and image processing.")
    print("      Watch for detailed step-by-step logs below.\n")
    
    # Filter expected warnings to keep logs clean
    # 1. Gradient checkpointing warning when visual encoder inputs don't require grad
    #    (expected because visual encoder is frozen, only LLM has LoRA)
    # 2. use_cache warning (automatically handled by transformers)
    warnings.filterwarnings(
        "ignore",
        message="None of the inputs have requires_grad=True",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*use_cache=True.*incompatible with gradient checkpointing.*",
        category=UserWarning,
    )
    
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
    
    # Save processor for VL models (important for inference)
    if is_vl_model and processor is not None:
        processor.save_pretrained(output_dir)
        print(f"Processor saved to {output_dir}")
    
    print(f"SFT training completed. Output saved to {output_dir}")


if __name__ == "__main__":
    main()
