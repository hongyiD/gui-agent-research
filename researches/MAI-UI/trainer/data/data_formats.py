#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standard Data Format Definitions for MAI-UI Training.

This module defines the canonical data formats used throughout the training pipeline,
ensuring consistency across data preprocessing, training, and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OutputFormat(Enum):
    """Supported output formats for training data."""
    
    OPENAI_MESSAGES = "openai_messages"
    PROMPT_RESPONSE = "prompt_response"
    FULL_TRAJECTORY = "full_trajectory"


class ImageFormat(Enum):
    """Image encoding formats."""
    
    BASE64 = "base64"
    PATH = "path"
    SKIP = "skip"


@dataclass
class TrajectoryStep:
    """Single step in a trajectory."""
    
    step_index: int
    thinking: str
    action: dict[str, Any]
    action_type: str
    screenshot_path: str | None = None
    success: bool = True
    message: str | None = None
    timestamp: str | None = None
    instruction: str | None = None


@dataclass
class TrajectoryMetadata:
    """Metadata for a trajectory."""
    
    source_file: str
    task_name: str
    total_steps: int
    success_rate: float = 1.0
    task_goal: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class OpenAIMessage:
    """OpenAI-compatible message format."""
    
    role: str
    content: list[dict[str, Any]]


@dataclass
class OpenAIFormatSample:
    """OpenAI-compatible training sample."""
    
    messages: list[OpenAIMessage]
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content
                }
                for msg in self.messages
            ],
            "metadata": self.metadata
        }


@dataclass
class PromptResponseSample:
    """Prompt-response format for SFT training."""
    
    prompt: str
    response: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt,
            "response": self.response,
            "metadata": self.metadata
        }


@dataclass
class FullTrajectorySample:
    """Full trajectory format for RL training."""
    
    task_goal: str
    steps: list[TrajectoryStep]
    metadata: TrajectoryMetadata
    tools: list[dict] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_goal": self.task_goal,
            "steps": [
                {
                    "step_index": step.step_index,
                    "thinking": step.thinking,
                    "action": step.action,
                    "action_type": step.action_type,
                    "screenshot_path": step.screenshot_path,
                    "success": step.success,
                    "message": step.message,
                    "timestamp": step.timestamp,
                    "instruction": step.instruction,
                }
                for step in self.steps
            ],
            "metadata": {
                "source_file": self.metadata.source_file,
                "task_name": self.metadata.task_name,
                "total_steps": self.metadata.total_steps,
                "success_rate": self.metadata.success_rate,
                "task_goal": self.metadata.task_goal,
                "tags": self.metadata.tags,
            },
            "tools": self.tools,
        }


@dataclass
class ProcessingConfig:
    """Configuration for data processing."""
    
    output_format: OutputFormat = OutputFormat.PROMPT_RESPONSE
    image_format: ImageFormat = ImageFormat.BASE64
    max_samples_per_trajectory: int = -1
    include_history: bool = True
    history_window: int = 5
    image_max_size: tuple[int, int] | None = None
    image_quality: int = 85
    skip_failed_steps: bool = True
    compress_images: bool = True


@dataclass
class ProcessingStats:
    """Statistics from data processing."""
    
    total_trajectories: int = 0
    total_steps: int = 0
    successful_steps: int = 0
    failed_steps: int = 0
    skipped_steps: int = 0
    action_type_counts: dict[str, int] = field(default_factory=dict)
    avg_steps_per_trajectory: float = 0.0
    avg_trajectory_length: float = 0.0
    total_output_samples: int = 0
    processing_errors: list[str] = field(default_factory=list)
    
    def compute_averages(self) -> None:
        """Compute average statistics."""
        if self.total_trajectories > 0:
            self.avg_steps_per_trajectory = self.total_steps / self.total_trajectories
        if self.total_steps > 0:
            self.avg_trajectory_length = self.total_steps / self.total_trajectories
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_trajectories": self.total_trajectories,
            "total_steps": self.total_steps,
            "successful_steps": self.successful_steps,
            "failed_steps": self.failed_steps,
            "skipped_steps": self.skipped_steps,
            "action_type_counts": self.action_type_counts,
            "avg_steps_per_trajectory": self.avg_steps_per_trajectory,
            "avg_trajectory_length": self.avg_trajectory_length,
            "total_output_samples": self.total_output_samples,
            "processing_errors_count": len(self.processing_errors),
        }
