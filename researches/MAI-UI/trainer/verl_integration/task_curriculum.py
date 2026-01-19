#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Task Curriculum Learning for MAI-UI.

Implements dynamic task difficulty stratification based on pass@K success rate,
as described in the MAI-UI paper.
"""

from __future__ import annotations

import collections
import random
from typing import Any

from loguru import logger


class TaskCurriculum:
    """Task curriculum learning manager.
    
    Dynamically stratifies tasks into four difficulty levels based on
    current policy's pass@K success rate:
    - Frontier (0-25%): Push capability boundaries
    - Exploration (25-50%): Drive skill development
    - Near-mastery (50-75%): Approach proficiency
    - Exploitation (75-100%): Reinforce learned behaviors
    """
    
    def __init__(
        self,
        difficulty_levels: dict[str, tuple[float, float]] | None = None,
        min_attempts_per_task: int = 5,
        initial_distribution: dict[str, float] | None = None,
    ):
        """Initialize task curriculum.
        
        Args:
            difficulty_levels: Mapping from level name to (min, max) success rate range
            min_attempts_per_task: Minimum attempts before using success rate for stratification
            initial_distribution: Initial task distribution across difficulty levels
        """
        self.task_stats: dict[str, dict[str, Any]] = collections.defaultdict(
            lambda: {
                "successes": 0,
                "attempts": 0,
                "success_rate": 0.0,
                "last_update": 0,
            }
        )
        
        self.difficulty_levels = difficulty_levels or {
            "frontier": (0.0, 0.25),
            "exploration": (0.25, 0.50),
            "near_mastery": (0.50, 0.75),
            "exploitation": (0.75, 1.0),
        }
        
        self.min_attempts_per_task = min_attempts_per_task
        
        # Initial distribution (will adapt over time)
        self.current_distribution = initial_distribution or {
            "frontier": 0.1,
            "exploration": 0.3,
            "near_mastery": 0.4,
            "exploitation": 0.2,
        }
        
        # Track all available tasks
        self.all_tasks: list[str] = []
        self.tasks_by_level: dict[str, list[str]] = {
            level: [] for level in self.difficulty_levels.keys()
        }
    
    def register_tasks(self, task_names: list[str]) -> None:
        """Register available tasks."""
        self.all_tasks = task_names
        # Initially assign all tasks to exploration level
        self.tasks_by_level["exploration"] = task_names.copy()
        logger.info(f"Registered {len(task_names)} tasks")
    
    def update_success_rate(
        self,
        task_name: str,
        success: bool,
        step: int = 0,
    ) -> None:
        """Update task success rate.
        
        Args:
            task_name: Name of the task
            success: Whether the task was completed successfully
            step: Current training step (for tracking)
        """
        stats = self.task_stats[task_name]
        stats["attempts"] += 1
        if success:
            stats["successes"] += 1
        
        # Update success rate
        if stats["attempts"] > 0:
            stats["success_rate"] = stats["successes"] / stats["attempts"]
        stats["last_update"] = step
        
        # Re-stratify if we have enough attempts
        if stats["attempts"] >= self.min_attempts_per_task:
            self._re_stratify_task(task_name)
    
    def _re_stratify_task(self, task_name: str) -> None:
        """Re-stratify a task based on its current success rate."""
        stats = self.task_stats[task_name]
        success_rate = stats["success_rate"]
        
        # Remove from current level
        for level_tasks in self.tasks_by_level.values():
            if task_name in level_tasks:
                level_tasks.remove(task_name)
                break
        
        # Assign to appropriate level
        target_level = self.get_task_difficulty(task_name)
        if target_level:
            self.tasks_by_level[target_level].append(task_name)
            logger.debug(
                f"Re-stratified {task_name} to {target_level} "
                f"(success_rate={success_rate:.2f})"
            )
    
    def get_task_difficulty(self, task_name: str) -> str | None:
        """Get difficulty level for a task.
        
        Returns:
            Difficulty level name, or None if not enough attempts
        """
        stats = self.task_stats.get(task_name)
        if not stats or stats["attempts"] < self.min_attempts_per_task:
            return None
        
        success_rate = stats["success_rate"]
        
        for level_name, (min_rate, max_rate) in self.difficulty_levels.items():
            if min_rate <= success_rate < max_rate:
                return level_name
        
        # Handle edge case: success_rate == 1.0
        if success_rate >= 1.0:
            return "exploitation"
        
        return None
    
    def sample_task(
        self,
        difficulty_level: str | None = None,
        step: int = 0,
    ) -> str:
        """Sample a task based on current distribution.
        
        Args:
            difficulty_level: Specific difficulty level to sample from (optional)
            step: Current training step (for adaptive distribution)
        
        Returns:
            Task name
        """
        if difficulty_level:
            # Sample from specific level
            tasks = self.tasks_by_level.get(difficulty_level, [])
            if not tasks:
                logger.warning(
                    f"No tasks available in level {difficulty_level}, "
                    "falling back to all tasks"
                )
                tasks = self.all_tasks
        else:
            # Sample according to current distribution
            level = self._sample_level_by_distribution()
            tasks = self.tasks_by_level.get(level, [])
            
            # Fallback if level is empty
            if not tasks:
                tasks = self.all_tasks
        
        if not tasks:
            raise ValueError("No tasks available for sampling")
        
        return random.choice(tasks)
    
    def _sample_level_by_distribution(self) -> str:
        """Sample a difficulty level according to current distribution."""
        levels = list(self.current_distribution.keys())
        weights = [self.current_distribution[level] for level in levels]
        return random.choices(levels, weights=weights, k=1)[0]
    
    def get_current_distribution(self) -> dict[str, float]:
        """Get current task distribution across difficulty levels."""
        return self.current_distribution.copy()
    
    def update_distribution(
        self,
        new_distribution: dict[str, float],
    ) -> None:
        """Update task distribution (e.g., based on training progress).
        
        Args:
            new_distribution: New distribution mapping level -> weight
        """
        # Normalize to ensure sum = 1.0
        total = sum(new_distribution.values())
        if total > 0:
            self.current_distribution = {
                level: weight / total
                for level, weight in new_distribution.items()
            }
        else:
            logger.warning("Invalid distribution (sum=0), keeping current")
    
    def adapt_distribution(self, step: int, total_steps: int) -> None:
        """Adapt distribution based on training progress.
        
        Early stages emphasize simpler tasks, gradually shifting to
        challenging tasks as training progresses.
        
        Args:
            step: Current training step
            total_steps: Total training steps
        """
        progress = min(step / total_steps, 1.0)
        
        # Early training: more exploitation and near_mastery
        # Late training: more frontier and exploration
        early_dist = {
            "frontier": 0.05,
            "exploration": 0.20,
            "near_mastery": 0.45,
            "exploitation": 0.30,
        }
        
        late_dist = {
            "frontier": 0.25,
            "exploration": 0.40,
            "near_mastery": 0.25,
            "exploitation": 0.10,
        }
        
        # Interpolate
        new_dist = {}
        for level in self.difficulty_levels.keys():
            new_dist[level] = (
                early_dist.get(level, 0.25) * (1 - progress)
                + late_dist.get(level, 0.25) * progress
            )
        
        self.update_distribution(new_dist)
        logger.info(
            f"Adapted distribution at step {step}: {self.current_distribution}"
        )
    
    def get_statistics(self) -> dict[str, Any]:
        """Get curriculum statistics."""
        stats_by_level = {}
        for level in self.difficulty_levels.keys():
            tasks = self.tasks_by_level[level]
            if tasks:
                avg_success_rate = sum(
                    self.task_stats[task]["success_rate"] for task in tasks
                ) / len(tasks)
                stats_by_level[level] = {
                    "num_tasks": len(tasks),
                    "avg_success_rate": avg_success_rate,
                }
            else:
                stats_by_level[level] = {
                    "num_tasks": 0,
                    "avg_success_rate": 0.0,
                }
        
        return {
            "distribution": self.current_distribution,
            "stats_by_level": stats_by_level,
            "total_tasks": len(self.all_tasks),
            "total_attempts": sum(
                s["attempts"] for s in self.task_stats.values()
            ),
        }
