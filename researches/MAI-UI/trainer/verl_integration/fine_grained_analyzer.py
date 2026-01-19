#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-grained Trajectory Analyzer for MAI-UI.

Uses MLLM-as-Judge to perform fine-grained correctness analysis,
identifying the first error step and extracting correct prefixes
from failed trajectories.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger


class FineGrainedTrajectoryAnalyzer:
    """Fine-grained trajectory analyzer using MLLM-as-Judge.
    
    Analyzes trajectories step-by-step to identify the first error,
    enabling extraction of correct prefixes from failed trajectories.
    """
    
    def __init__(
        self,
        judge_model: str = "gpt-4",
        judge_api_key: str | None = None,
        extract_correct_prefix: bool = True,
    ):
        """Initialize analyzer.
        
        Args:
            judge_model: Name of the judge model (e.g., "gpt-4", "claude-3-opus")
            judge_api_key: API key for judge model (uses main API key if None)
            extract_correct_prefix: Whether to extract correct prefixes
        """
        self.judge_model = judge_model
        self.judge_api_key = judge_api_key
        self.extract_correct_prefix = extract_correct_prefix
        
        # Lazy import of judge client
        self._judge_client = None
    
    def _get_judge_client(self):
        """Lazy load judge client."""
        if self._judge_client is None:
            # Import here to avoid dependency if not used
            try:
                from openai import OpenAI
                
                self._judge_client = OpenAI(
                    api_key=self.judge_api_key or "dummy"
                )
            except ImportError:
                logger.warning(
                    "OpenAI client not available. "
                    "Install with: pip install openai"
                )
                self._judge_client = None
        
        return self._judge_client
    
    def analyze_trajectory(
        self,
        task_goal: str,
        trajectory: list[dict[str, Any]],
        screenshots: list[Any] | None = None,
    ) -> int:
        """Analyze trajectory to find first error step.
        
        Args:
            task_goal: Task goal description
            trajectory: List of trajectory steps
            screenshots: Optional list of screenshots for each step
        
        Returns:
            Index of first error step, or -1 if all steps are correct
        """
        if not trajectory:
            return -1
        
        # For now, use a simple heuristic: check if task was successful
        # Full MLLM-as-Judge implementation would analyze each step
        
        # Check if trajectory ended successfully
        last_step = trajectory[-1]
        action = last_step.get("action", {})
        action_type = action.get("action_type", "")
        
        # If task completed successfully, all steps are correct
        if action_type in ["finished", "answer"]:
            task_status = action.get("goal_status") or action.get("text", "")
            if "complete" in str(task_status).lower() or action_type == "answer":
                return -1  # All steps correct
        
        # Use MLLM-as-Judge if available
        if self._get_judge_client():
            return self._analyze_with_judge(task_goal, trajectory, screenshots)
        
        # Fallback: assume first step is correct if we can't analyze
        logger.debug("Judge model not available, using fallback analysis")
        return len(trajectory) - 1  # Assume error at last step
    
    def _analyze_with_judge(
        self,
        task_goal: str,
        trajectory: list[dict[str, Any]],
        screenshots: list[Any] | None = None,
    ) -> int:
        """Analyze trajectory using MLLM-as-Judge.
        
        This is a placeholder implementation. Full implementation would:
        1. For each step, construct prompt with task goal, history, current step
        2. Ask judge model if the step is correct
        3. Return index of first incorrect step
        """
        # Placeholder: would implement full MLLM-as-Judge logic here
        logger.debug("MLLM-as-Judge analysis not fully implemented")
        return len(trajectory) - 1
    
    def extract_correct_prefix(
        self,
        trajectory: list[dict[str, Any]],
        error_step: int,
    ) -> list[dict[str, Any]]:
        """Extract correct prefix from trajectory.
        
        Args:
            trajectory: Full trajectory
            error_step: Index of first error step (-1 if all correct)
        
        Returns:
            List of correct steps (prefix before error)
        """
        if error_step < 0:
            # All steps are correct
            return trajectory
        
        if error_step == 0:
            # First step is wrong, return empty
            return []
        
        # Return all steps before the error
        return trajectory[:error_step]
    
    def analyze_and_extract(
        self,
        task_goal: str,
        trajectory: list[dict[str, Any]],
        screenshots: list[Any] | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Analyze trajectory and extract correct prefix.
        
        Args:
            task_goal: Task goal description
            trajectory: Full trajectory
            screenshots: Optional screenshots
        
        Returns:
            Tuple of (correct_prefix, error_step_index)
        """
        error_step = self.analyze_trajectory(task_goal, trajectory, screenshots)
        
        if self.extract_correct_prefix:
            correct_prefix = self.extract_correct_prefix(trajectory, error_step)
            return correct_prefix, error_step
        
        return trajectory, error_step
    
    def batch_analyze(
        self,
        task_goals: list[str],
        trajectories: list[list[dict[str, Any]]],
        screenshots_list: list[list[Any]] | None = None,
    ) -> list[tuple[list[dict[str, Any]], int]]:
        """Batch analyze multiple trajectories.
        
        Args:
            task_goals: List of task goals
            trajectories: List of trajectories
            screenshots_list: Optional list of screenshot lists
        
        Returns:
            List of (correct_prefix, error_step) tuples
        """
        if screenshots_list is None:
            screenshots_list = [None] * len(trajectories)
        
        results = []
        for goal, traj, screenshots in zip(
            task_goals, trajectories, screenshots_list
        ):
            result = self.analyze_and_extract(goal, traj, screenshots)
            results.append(result)
        
        return results
