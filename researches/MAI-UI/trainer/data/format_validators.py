#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Format Validators for MAI-UI Training.

This module provides validation functions to ensure data quality and format consistency.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from data_formats import (
    OpenAIFormatSample,
    PromptResponseSample,
    FullTrajectorySample,
    TrajectoryStep,
)


class ValidationError(Exception):
    """Raised when data validation fails."""
    pass


class DataValidator:
    """Validator for training data formats."""
    
    def __init__(self, strict: bool = False):
        """Initialize validator.
        
        Args:
            strict: If True, raise exceptions on validation errors.
                   If False, collect errors and warnings.
        """
        self.strict = strict
        self.errors: list[str] = []
        self.warnings: list[str] = []
    
    def validate_trajectory_step(self, step: dict[str, Any]) -> bool:
        """Validate a single trajectory step.
        
        Args:
            step: Dictionary representing a trajectory step.
            
        Returns:
            True if valid, False otherwise.
        """
        required_fields = ["step_index", "thinking", "action", "action_type"]
        
        for field in required_fields:
            if field not in step:
                error_msg = f"Missing required field: {field}"
                self.errors.append(error_msg)
                if self.strict:
                    raise ValidationError(error_msg)
                return False
        
        if not isinstance(step["action"], dict):
            error_msg = f"'action' must be a dict, got {type(step['action'])}"
            self.errors.append(error_msg)
            if self.strict:
                raise ValidationError(error_msg)
            return False
        
        if "screenshot_path" in step and step["screenshot_path"]:
            screenshot_path = step["screenshot_path"]
            if not os.path.exists(screenshot_path):
                warning_msg = f"Screenshot not found: {screenshot_path}"
                self.warnings.append(warning_msg)
        
        return True
    
    def validate_openai_sample(self, sample: dict[str, Any]) -> bool:
        """Validate OpenAI format sample.
        
        Args:
            sample: Dictionary representing an OpenAI format sample.
            
        Returns:
            True if valid, False otherwise.
        """
        if "messages" not in sample:
            error_msg = "Missing 'messages' field"
            self.errors.append(error_msg)
            if self.strict:
                raise ValidationError(error_msg)
            return False
        
        messages = sample["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            error_msg = "messages must be a list with at least 2 messages"
            self.errors.append(error_msg)
            if self.strict:
                raise ValidationError(error_msg)
            return False
        
        for i, msg in enumerate(messages):
            if "role" not in msg or "content" not in msg:
                error_msg = f"Message {i} missing 'role' or 'content'"
                self.errors.append(error_msg)
                if self.strict:
                    raise ValidationError(error_msg)
                return False
            
            if msg["role"] not in ["system", "user", "assistant"]:
                error_msg = f"Invalid role: {msg['role']}"
                self.errors.append(error_msg)
                if self.strict:
                    raise ValidationError(error_msg)
                return False
        
        return True
    
    def validate_prompt_response_sample(self, sample: dict[str, Any]) -> bool:
        """Validate prompt-response format sample.
        
        Args:
            sample: Dictionary representing a prompt-response sample.
            
        Returns:
            True if valid, False otherwise.
        """
        if "prompt" not in sample or "response" not in sample:
            error_msg = "Missing 'prompt' or 'response' field"
            self.errors.append(error_msg)
            if self.strict:
                raise ValidationError(error_msg)
            return False
        
        if not isinstance(sample["prompt"], str) or not isinstance(sample["response"], str):
            error_msg = "'prompt' and 'response' must be strings"
            self.errors.append(error_msg)
            if self.strict:
                raise ValidationError(error_msg)
            return False
        
        if len(sample["prompt"]) == 0 or len(sample["response"]) == 0:
            error_msg = "Empty prompt or response"
            self.errors.append(error_msg)
            if self.strict:
                raise ValidationError(error_msg)
            return False
        
        return True
    
    def validate_full_trajectory_sample(self, sample: dict[str, Any]) -> bool:
        """Validate full trajectory format sample.
        
        Args:
            sample: Dictionary representing a full trajectory sample.
            
        Returns:
            True if valid, False otherwise.
        """
        required_fields = ["task_goal", "steps", "metadata"]
        
        for field in required_fields:
            if field not in sample:
                error_msg = f"Missing required field: {field}"
                self.errors.append(error_msg)
                if self.strict:
                    raise ValidationError(error_msg)
                return False
        
        if not isinstance(sample["steps"], list):
            error_msg = "'steps' must be a list"
            self.errors.append(error_msg)
            if self.strict:
                raise ValidationError(error_msg)
            return False
        
        for i, step in enumerate(sample["steps"]):
            if not self.validate_trajectory_step(step):
                error_msg = f"Invalid step at index {i}"
                self.errors.append(error_msg)
                if self.strict:
                    raise ValidationError(error_msg)
                return False
        
        return True
    
    def validate_jsonl_file(self, file_path: str, format_type: str) -> dict[str, Any]:
        """Validate a JSONL file.
        
        Args:
            file_path: Path to JSONL file.
            format_type: Expected format type ('openai', 'prompt_response', 'full_trajectory').
            
        Returns:
            Validation report dictionary.
        """
        self.errors.clear()
        self.warnings.clear()
        
        if not os.path.exists(file_path):
            raise ValidationError(f"File not found: {file_path}")
        
        total_lines = 0
        valid_lines = 0
        invalid_lines = 0
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                total_lines += 1
                line = line.strip()
                
                if not line:
                    continue
                
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as e:
                    error_msg = f"Line {line_num}: Invalid JSON - {e}"
                    self.errors.append(error_msg)
                    invalid_lines += 1
                    continue
                
                is_valid = False
                if format_type == "openai":
                    is_valid = self.validate_openai_sample(sample)
                elif format_type == "prompt_response":
                    is_valid = self.validate_prompt_response_sample(sample)
                elif format_type == "full_trajectory":
                    is_valid = self.validate_full_trajectory_sample(sample)
                else:
                    raise ValueError(f"Unknown format_type: {format_type}")
                
                if is_valid:
                    valid_lines += 1
                else:
                    invalid_lines += 1
        
        return {
            "file_path": file_path,
            "format_type": format_type,
            "total_lines": total_lines,
            "valid_lines": valid_lines,
            "invalid_lines": invalid_lines,
            "errors": self.errors.copy(),
            "warnings": self.warnings.copy(),
            "is_valid": invalid_lines == 0,
        }
    
    def get_report(self) -> dict[str, Any]:
        """Get validation report.
        
        Returns:
            Dictionary with errors and warnings.
        """
        return {
            "errors": self.errors.copy(),
            "warnings": self.warnings.copy(),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }


def validate_trajectory_jsonl(file_path: str, strict: bool = False) -> dict[str, Any]:
    """Convenience function to validate a trajectory JSONL file.
    
    Args:
        file_path: Path to the JSONL file.
        strict: If True, raise exceptions on validation errors.
        
    Returns:
        Validation report dictionary.
    """
    validator = DataValidator(strict=strict)
    
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if not first_line:
            raise ValidationError("Empty file")
        
        sample = json.loads(first_line)
        
        if "messages" in sample:
            format_type = "openai"
        elif "prompt" in sample and "response" in sample:
            format_type = "prompt_response"
        elif "task_goal" in sample and "steps" in sample:
            format_type = "full_trajectory"
        else:
            raise ValidationError("Unable to determine format type")
    
    return validator.validate_jsonl_file(file_path, format_type)


def print_validation_report(report: dict[str, Any]) -> None:
    """Print a formatted validation report.
    
    Args:
        report: Validation report dictionary.
    """
    print(f"\n{'='*60}")
    print(f"Validation Report: {report['file_path']}")
    print(f"{'='*60}")
    print(f"Format: {report['format_type']}")
    print(f"Total lines: {report['total_lines']}")
    print(f"Valid lines: {report['valid_lines']}")
    print(f"Invalid lines: {report['invalid_lines']}")
    print(f"Status: {'✓ VALID' if report['is_valid'] else '✗ INVALID'}")
    
    if report['errors']:
        print(f"\n{len(report['errors'])} Error(s):")
        for error in report['errors'][:10]:
            print(f"  - {error}")
        if len(report['errors']) > 10:
            print(f"  ... and {len(report['errors']) - 10} more errors")
    
    if report['warnings']:
        print(f"\n{len(report['warnings'])} Warning(s):")
        for warning in report['warnings'][:10]:
            print(f"  - {warning}")
        if len(report['warnings']) > 10:
            print(f"  ... and {len(report['warnings']) - 10} more warnings")
    
    print(f"{'='*60}\n")
