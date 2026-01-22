#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified Data Processor for MAI-UI Training.

This module consolidates functionality from process_trajectory_jsonl.py and build_data.py,
providing a single, consistent interface for data processing with multiple output formats.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from collections.abc import Generator
from io import BytesIO
from pathlib import Path
from typing import Any

import sys
import yaml
from PIL import Image
from tqdm import tqdm

# Try to import official MAI-UI prompt from multiple sources (priority order)
USE_OFFICIAL_PROMPT = False
MAI_MOBILE_SYS_PROMPT_ASK_USER_MCP = None

# Priority 1: Import from trainer/prompts/maiui_official_prompts.py (local copy)
prompts_file = Path(__file__).parent.parent / "prompts" / "maiui_official_prompts.py"
if prompts_file.exists():
    try:
        # Add prompts directory to path
        prompts_dir = prompts_file.parent
        if str(prompts_dir) not in sys.path:
            sys.path.insert(0, str(prompts_dir))
        from maiui_official_prompts import MAI_MOBILE_SYS_PROMPT_ASK_USER_MCP
        USE_OFFICIAL_PROMPT = True
    except ImportError as e:
        # Silently continue to next import source
        pass

# Priority 2: Import from MAI-UI-WebUI (official source)
if not USE_OFFICIAL_PROMPT:
    webui_path = Path(__file__).parent.parent.parent.parent / "MAI-UI-WebUI" / "src"
    if webui_path.exists():
        sys.path.insert(0, str(webui_path))
        try:
            from prompt import MAI_MOBILE_SYS_PROMPT_ASK_USER_MCP
            USE_OFFICIAL_PROMPT = True
        except ImportError:
            pass

# Priority 3: Import from mobile_world (fallback)
if not USE_OFFICIAL_PROMPT:
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from mobile_world.agents.utils.prompts import MAI_MOBILE_SYS_PROMPT_ASK_USER_MCP
        USE_OFFICIAL_PROMPT = True
    except ImportError:
        pass

from data_formats import (
    OutputFormat,
    ImageFormat,
    ProcessingConfig,
    ProcessingStats,
    TrajectoryStep,
    TrajectoryMetadata,
    OpenAIMessage,
    OpenAIFormatSample,
    PromptResponseSample,
    FullTrajectorySample,
)

# Use official MAI-UI prompt if available, otherwise fallback to simplified version
if USE_OFFICIAL_PROMPT:
    # Official prompt is a Jinja2 Template, render it without tools for data processing
    GUI_AGENT_SYSTEM_PROMPT = MAI_MOBILE_SYS_PROMPT_ASK_USER_MCP.render(tools="")
else:
    # Fallback: simplified prompt (should not be used in production)
    # WARNING: This fallback prompt may not match the official MAI-UI format.
    # Please ensure MAI-UI-WebUI/src/prompt.py is accessible or mobile_world is installed.
    GUI_AGENT_SYSTEM_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
For each function call, return the thinking process in <thinking> </thinking> tags, and a json object with function name and arguments within <tool_call></tool_call> XML tags:
```
<thinking>
...
</thinking>
<tool_call>
{"name": "mobile_use", "arguments": <args-json-object>}
</tool_call>
```

## Action Space

{"action": "click", "coordinate": [x, y]}
{"action": "long_press", "coordinate": [x, y]}
{"action": "type", "text": ""}
{"action": "swipe", "direction": "up or down or left or right", "coordinate": [x, y]}
{"action": "open", "text": "app_name"}
{"action": "drag", "start_coordinate": [x1, y1], "end_coordinate": [x2, y2]}
{"action": "system_button", "button": "button_name"}
{"action": "wait"}
{"action": "terminate", "status": "success or fail"}
{"action": "answer", "text": "xxx"}
{"action": "ask_user", "text": "xxx"}
{"action": "double_click", "coordinate": [x, y]}

## Behavior Rules
1. Track your previous action. If it was a swipe, do NOT swipe in the same direction more than 5 times consecutively.
2. If you swipe in the same direction 2-3 times but the screen content doesn't change significantly (likely reached the boundary), try swiping in the **opposite direction**.
3. Different apps may require different swipe directions to view history or more content. Adjust flexibly based on actual screen feedback.
4. Strictly follow the user's instructions. If you have had a conversation with the user, prioritize the **latest instruction**.

## Note
- Write a small plan and finally summarize your next action (with its target element) in one sentence in <thinking></thinking> part.
- **IMPORTANT: When the user asks to open an app (by name like "微信", "高德地图", "淘宝", etc.), ALWAYS use the `open` action with the app name first!** This is the fastest way to launch apps.
- Example: If user says "打开微信", use `{"action": "open", "text": "微信"}` instead of navigating manually.
- You must follow the Action Space strictly, and return the correct json object within <thinking> </thinking> and <tool_call></tool_call> XML tags.
"""


class UnifiedDataProcessor:
    """Unified processor for trajectory data with multiple output formats."""
    
    def __init__(self, config: ProcessingConfig):
        """Initialize processor.
        
        Args:
            config: Processing configuration.
        """
        self.config = config
        self.stats = ProcessingStats()
    
    def load_image_as_base64(
        self,
        image_path: str,
        max_size: tuple[int, int] | None = None,
        quality: int = 85,
    ) -> str:
        """Load image and encode as base64 with optional compression.
        
        Args:
            image_path: Path to the image file.
            max_size: Maximum (width, height) for resizing.
            quality: JPEG quality (0-100) for compression.
            
        Returns:
            Base64 encoded image string.
        """
        try:
            img = Image.open(image_path)
            
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            if max_size:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            buffer = BytesIO()
            
            if self.config.compress_images:
                img.save(buffer, format="JPEG", quality=quality, optimize=True)
            else:
                img.save(buffer, format="PNG")
            
            img_bytes = buffer.getvalue()
            return base64.b64encode(img_bytes).decode("utf-8")
        
        except Exception as e:
            error_msg = f"Failed to load image {image_path}: {e}"
            self.stats.processing_errors.append(error_msg)
            raise
    
    def resolve_screenshot_path(
        self,
        screenshot_path: str,
        trajectory_dir: str,
    ) -> str | None:
        """Resolve screenshot path using multiple strategies.
        
        Args:
            screenshot_path: Original screenshot path from trajectory.
            trajectory_dir: Directory containing the trajectory file.
            
        Returns:
            Resolved absolute path or None if not found.
        """
        if not screenshot_path:
            return None
        
        if os.path.isabs(screenshot_path) and os.path.exists(screenshot_path):
            return screenshot_path
        
        strategies = [
            lambda: os.path.join(trajectory_dir, screenshot_path),
            lambda: os.path.join(trajectory_dir, os.path.basename(screenshot_path)),
            lambda: os.path.join(os.path.dirname(trajectory_dir), screenshot_path),
            lambda: os.path.join(
                os.path.dirname(trajectory_dir),
                os.path.basename(screenshot_path)
            ),
        ]
        
        for strategy in strategies:
            try:
                candidate_path = strategy()
                if os.path.exists(candidate_path):
                    return candidate_path
            except Exception:
                continue
        
        return None
    
    def build_user_content(
        self,
        instruction: str,
        screenshot_path: str | None,
        history: list[dict],
    ) -> list[dict]:
        """Build user message content.
        
        Args:
            instruction: The task instruction.
            screenshot_path: Path to the current screenshot.
            history: List of previous steps for context.
            
        Returns:
            List of content items.
        """
        content = []
        
        if screenshot_path and os.path.exists(screenshot_path):
            if self.config.image_format == ImageFormat.BASE64:
                img_b64 = self.load_image_as_base64(
                    screenshot_path,
                    max_size=self.config.image_max_size,
                    quality=self.config.image_quality,
                )
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
            elif self.config.image_format == ImageFormat.PATH:
                content.append({
                    "type": "text",
                    "text": f"<image_path>{screenshot_path}</image_path>"
                })
        
        instruction_text = f"Task: {instruction}\n"
        
        if self.config.include_history and history:
            instruction_text += "\nPrevious actions:\n"
            for step in history[-self.config.history_window:]:
                action = step.get("action", {})
                action_type = step.get("action_type", "unknown")
                thinking = step.get("thinking", "")
                
                if thinking:
                    instruction_text += f"- {thinking}\n"
                if action:
                    instruction_text += f"  Action: {json.dumps(action, ensure_ascii=False)}\n"
        
        content.append({"type": "text", "text": instruction_text})
        return content
    
    def build_assistant_content(self, thinking: str, action: dict) -> list[dict]:
        """Build assistant message content.
        
        Args:
            thinking: The thinking process.
            action: The action taken.
            
        Returns:
            List with text content.
        """
        response = thinking
        if action and isinstance(action, dict):
            response += f"\nAction: {json.dumps(action, ensure_ascii=False)}"
        
        return [{"type": "text", "text": response}]
    
    def trajectory_to_openai_format(
        self,
        trajectory_path: str,
    ) -> Generator[OpenAIFormatSample, None, None]:
        """Convert trajectory to OpenAI format samples.
        
        Args:
            trajectory_path: Path to trajectory.jsonl file.
            
        Yields:
            OpenAI format samples.
        """
        trajectory_dir = os.path.dirname(trajectory_path)
        
        steps = []
        with open(trajectory_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    steps.append(json.loads(line))
        
        if not steps:
            return
        
        instruction = steps[0].get("instruction", "")
        
        max_samples = self.config.max_samples_per_trajectory
        for i, step in enumerate(steps):
            if max_samples > 0 and i >= max_samples:
                break
            
            thinking = step.get("thinking", "")
            action = step.get("action", {})
            screenshot_path = step.get("screenshot_path", "")
            
            full_screenshot_path = self.resolve_screenshot_path(
                screenshot_path,
                trajectory_dir
            )
            
            if screenshot_path and not full_screenshot_path:
                if self.config.skip_failed_steps:
                    self.stats.skipped_steps += 1
                    continue
            
            history = steps[:i]
            user_content = self.build_user_content(
                instruction=instruction,
                screenshot_path=full_screenshot_path,
                history=history,
            )
            
            assistant_content = self.build_assistant_content(thinking, action)
            
            messages = [
                OpenAIMessage(role="system", content=[{"type": "text", "text": GUI_AGENT_SYSTEM_PROMPT}]),
                OpenAIMessage(role="user", content=user_content),
                OpenAIMessage(role="assistant", content=assistant_content),
            ]
            
            metadata = {
                "source": trajectory_path,
                "step_index": step.get("step_index", i),
                "action_type": step.get("action_type"),
                "success": step.get("success", True),
                "timestamp": step.get("timestamp"),
                "screenshot_path": full_screenshot_path if full_screenshot_path else screenshot_path,  # Preserve screenshot path
            }
            
            sample = OpenAIFormatSample(messages=messages, metadata=metadata)
            
            self.stats.total_steps += 1
            if metadata["success"]:
                self.stats.successful_steps += 1
            else:
                self.stats.failed_steps += 1
            
            action_type = metadata.get("action_type", "unknown")
            self.stats.action_type_counts[action_type] = (
                self.stats.action_type_counts.get(action_type, 0) + 1
            )
            
            yield sample
    
    def trajectory_to_prompt_response_format(
        self,
        trajectory_path: str,
    ) -> Generator[PromptResponseSample, None, None]:
        """Convert trajectory to prompt-response format samples.
        
        Args:
            trajectory_path: Path to trajectory.jsonl file.
            
        Yields:
            Prompt-response format samples.
        """
        for openai_sample in self.trajectory_to_openai_format(trajectory_path):
            system_msg = next((m for m in openai_sample.messages if m.role == "system"), None)
            user_msg = next((m for m in openai_sample.messages if m.role == "user"), None)
            assistant_msg = next((m for m in openai_sample.messages if m.role == "assistant"), None)
            
            if not (system_msg and user_msg and assistant_msg):
                continue
            
            system_text = system_msg.content[0]["text"] if system_msg.content else ""
            
            user_text_parts = []
            for content_item in user_msg.content:
                if content_item["type"] == "text":
                    user_text_parts.append(content_item["text"])
                elif content_item["type"] == "image_url":
                    # Preserve image information based on image_format setting
                    image_url = content_item.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:image"):
                        # Extract base64 data from data URL
                        if "," in image_url:
                            base64_data = image_url.split(",", 1)[1]
                            if self.config.image_format == ImageFormat.BASE64:
                                user_text_parts.append(f"<image_base64>{base64_data}</image_base64>")
                            elif self.config.image_format == ImageFormat.PATH:
                                # Try to get path from metadata
                                screenshot_path = openai_sample.metadata.get("screenshot_path", "")
                                if screenshot_path:
                                    user_text_parts.append(f"<image_path>{screenshot_path}</image_path>")
                                else:
                                    user_text_parts.append("[Image]")
                            else:  # SKIP
                                user_text_parts.append("[Image]")
                        else:
                            user_text_parts.append("[Image]")
                    else:
                        user_text_parts.append(f"<image_path>{image_url}</image_path>")
            user_text = "\n".join(user_text_parts)
            
            assistant_text = assistant_msg.content[0]["text"] if assistant_msg.content else ""
            
            prompt = f"{system_text}\n\n{user_text}"
            response = assistant_text
            
            yield PromptResponseSample(
                prompt=prompt,
                response=response,
                metadata=openai_sample.metadata
            )
    
    def trajectory_to_full_trajectory_format(
        self,
        trajectory_path: str,
    ) -> Generator[FullTrajectorySample, None, None]:
        """Convert trajectory to full trajectory format.
        
        Args:
            trajectory_path: Path to trajectory.jsonl file.
            
        Yields:
            Full trajectory format samples.
        """
        trajectory_dir = os.path.dirname(trajectory_path)
        
        with open(trajectory_path, "r", encoding="utf-8") as f:
            steps_data = [json.loads(line.strip()) for line in f if line.strip()]
        
        if not steps_data:
            return
        
        instruction = steps_data[0].get("instruction", "")
        task_name = os.path.basename(trajectory_dir)
        
        steps = []
        for step_data in steps_data:
            screenshot_path = step_data.get("screenshot_path", "")
            full_screenshot_path = self.resolve_screenshot_path(
                screenshot_path,
                trajectory_dir
            )
            
            step = TrajectoryStep(
                step_index=step_data.get("step_index", 0),
                thinking=step_data.get("thinking", ""),
                action=step_data.get("action", {}),
                action_type=step_data.get("action_type", "unknown"),
                screenshot_path=full_screenshot_path,
                success=step_data.get("success", True),
                message=step_data.get("message"),
                timestamp=step_data.get("timestamp"),
                instruction=step_data.get("instruction"),
            )
            steps.append(step)
        
        metadata = TrajectoryMetadata(
            source_file=trajectory_path,
            task_name=task_name,
            total_steps=len(steps),
            task_goal=instruction,
        )
        
        yield FullTrajectorySample(
            task_goal=instruction,
            steps=steps,
            metadata=metadata,
        )
    
    def process_trajectory(
        self,
        trajectory_path: str,
    ) -> Generator[Any, None, None]:
        """Process a single trajectory file.
        
        Args:
            trajectory_path: Path to trajectory.jsonl file.
            
        Yields:
            Samples in the configured output format.
        """
        self.stats.total_trajectories += 1
        
        try:
            if self.config.output_format == OutputFormat.OPENAI_MESSAGES:
                yield from self.trajectory_to_openai_format(trajectory_path)
            elif self.config.output_format == OutputFormat.PROMPT_RESPONSE:
                yield from self.trajectory_to_prompt_response_format(trajectory_path)
            elif self.config.output_format == OutputFormat.FULL_TRAJECTORY:
                yield from self.trajectory_to_full_trajectory_format(trajectory_path)
        except Exception as e:
            error_msg = f"Error processing {trajectory_path}: {e}"
            self.stats.processing_errors.append(error_msg)
    
    def process_directory(
        self,
        input_dir: str,
        output_file: str,
        pattern: str = "**/trajectory.jsonl",
    ) -> ProcessingStats:
        """Process all trajectory files in a directory.
        
        Args:
            input_dir: Root directory to search for trajectory files.
            output_file: Output JSONL file path.
            pattern: Glob pattern to find trajectory files.
            
        Returns:
            Processing statistics.
        """
        input_path = Path(input_dir)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        trajectory_files = list(input_path.glob(pattern))
        
        with open(output_path, "w", encoding="utf-8") as out_f:
            for traj_file in tqdm(trajectory_files, desc="Processing trajectories"):
                for sample in self.process_trajectory(str(traj_file)):
                    out_f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
                    self.stats.total_output_samples += 1
        
        self.stats.compute_averages()
        return self.stats
    
    def process_single_file(
        self,
        input_file: str,
        output_file: str,
    ) -> ProcessingStats:
        """Process a single trajectory file.
        
        Args:
            input_file: Input trajectory.jsonl file.
            output_file: Output JSONL file.
            
        Returns:
            Processing statistics.
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as out_f:
            for sample in tqdm(
                self.process_trajectory(input_file),
                desc=f"Processing {Path(input_file).name}"
            ):
                out_f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
                self.stats.total_output_samples += 1
        
        self.stats.compute_averages()
        return self.stats


def load_config_from_yaml(config_path: str) -> ProcessingConfig:
    """Load processing config from YAML file.
    
    Args:
        config_path: Path to YAML config file.
        
    Returns:
        ProcessingConfig instance.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    
    return ProcessingConfig(
        output_format=OutputFormat(config_dict.get("output_format", "prompt_response")),
        image_format=ImageFormat(config_dict.get("image_format", "base64")),
        max_samples_per_trajectory=config_dict.get("max_samples_per_trajectory", -1),
        include_history=config_dict.get("include_history", True),
        history_window=config_dict.get("history_window", 5),
        image_max_size=tuple(config_dict["image_max_size"]) if config_dict.get("image_max_size") else None,
        image_quality=config_dict.get("image_quality", 85),
        skip_failed_steps=config_dict.get("skip_failed_steps", True),
        compress_images=config_dict.get("compress_images", True),
    )


def print_stats_report(stats: ProcessingStats) -> None:
    """Print processing statistics report.
    
    Args:
        stats: Processing statistics.
    """
    print("\n" + "="*60)
    print("Data Processing Statistics")
    print("="*60)
    print(f"Total trajectories: {stats.total_trajectories}")
    print(f"Total steps: {stats.total_steps}")
    print(f"Successful steps: {stats.successful_steps}")
    print(f"Failed steps: {stats.failed_steps}")
    print(f"Skipped steps: {stats.skipped_steps}")
    print(f"Output samples: {stats.total_output_samples}")
    print(f"Avg steps per trajectory: {stats.avg_steps_per_trajectory:.2f}")
    
    if stats.action_type_counts:
        print("\nAction Type Distribution:")
        for action_type, count in sorted(
            stats.action_type_counts.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            percentage = (count / stats.total_steps) * 100 if stats.total_steps > 0 else 0
            print(f"  {action_type}: {count} ({percentage:.1f}%)")
    
    if stats.processing_errors:
        print(f"\nProcessing Errors: {len(stats.processing_errors)}")
        for error in stats.processing_errors[:5]:
            print(f"  - {error}")
        if len(stats.processing_errors) > 5:
            print(f"  ... and {len(stats.processing_errors) - 5} more errors")
    
    print("="*60 + "\n")


def main() -> None:
    # Print prompt source information
    if USE_OFFICIAL_PROMPT:
        print("✓ Using official MAI-UI prompt from MAI_MOBILE_SYS_PROMPT_ASK_USER_MCP")
    else:
        print("⚠ WARNING: Using fallback prompt. Official prompt not found.")
        print("  Please ensure one of the following is available:")
        print("    1. trainer/prompts/maiui_official_prompts.py (recommended)")
        print("    2. MAI-UI-WebUI/src/prompt.py")
        print("    3. mobile_world/agents/utils/prompts.py")
        print("  Fallback prompt may not match the official MAI-UI format.\n")
    
    parser = argparse.ArgumentParser(
        description="Unified data processor for MAI-UI training"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input trajectory.jsonl file or directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output JSONL file",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["openai_messages", "prompt_response", "full_trajectory"],
        default="prompt_response",
        help="Output format",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        choices=["base64", "path", "skip"],
        default="base64",
        help="Image format",
    )
    parser.add_argument(
        "--compress-images",
        action="store_true",
        default=True,
        help="Compress images to reduce file size",
    )
    parser.add_argument(
        "--image-quality",
        type=int,
        default=85,
        help="JPEG quality for image compression (0-100)",
    )
    
    args = parser.parse_args()
    
    if args.config:
        config = load_config_from_yaml(args.config)
    else:
        config = ProcessingConfig(
            output_format=OutputFormat(args.output_format),
            image_format=ImageFormat(args.image_format),
            compress_images=args.compress_images,
            image_quality=args.image_quality,
        )
    
    processor = UnifiedDataProcessor(config)
    
    input_path = Path(args.input)
    if input_path.is_file():
        stats = processor.process_single_file(args.input, args.output)
    elif input_path.is_dir():
        stats = processor.process_directory(args.input, args.output)
    else:
        print(f"Error: Input path does not exist: {args.input}")
        return
    
    print_stats_report(stats)
    
    with open(Path(args.output).parent / "processing_stats.json", "w") as f:
        json.dump(stats.to_dict(), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
