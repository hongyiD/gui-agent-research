#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Iterative Rejection Sampling for MAI-UI.

Implements the iterative self-evolution data pipeline described in the MAI-UI paper,
where trajectories are generated, filtered by correctness, and used to train
the next iteration of the model.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from .fine_grained_analyzer import FineGrainedTrajectoryAnalyzer


class IterativeRejectionSampling:
    """Iterative rejection sampling manager.
    
    Manages the iterative self-evolution loop:
    1. Rollout trajectories with current model
    2. Filter by fine-grained correctness
    3. Mix with synthesis trajectories
    4. Train next model iteration
    """
    
    def __init__(
        self,
        num_iterations: int = 5,
        min_correctness_ratio: float = 0.5,
        judge_model: str = "gpt-4",
        judge_api_key: str | None = None,
    ):
        """Initialize iterative rejection sampling.
        
        Args:
            num_iterations: Number of iterations in the self-evolution loop
            min_correctness_ratio: Minimum ratio of correct steps to keep trajectory
            judge_model: Judge model for fine-grained analysis
            judge_api_key: API key for judge model
        """
        self.num_iterations = num_iterations
        self.current_iteration = 0
        self.min_correctness_ratio = min_correctness_ratio
        
        self.trajectory_buffer: list[dict[str, Any]] = []
        self.analyzer = FineGrainedTrajectoryAnalyzer(
            judge_model=judge_model,
            judge_api_key=judge_api_key,
            extract_correct_prefix=True,
        )
        
        # Statistics
        self.stats = {
            "total_rollouts": 0,
            "filtered_rollouts": 0,
            "extracted_prefixes": 0,
        }
    
    def collect_rollouts(
        self,
        rollouts: list[dict[str, Any]],
        task_goals: list[str] | None = None,
    ) -> None:
        """Collect rollouts from current iteration.
        
        Args:
            rollouts: List of rollout trajectories
            task_goals: Optional list of task goals for each rollout
        """
        self.stats["total_rollouts"] += len(rollouts)
        self.trajectory_buffer.extend(rollouts)
        logger.info(
            f"Iteration {self.current_iteration}: "
            f"Collected {len(rollouts)} rollouts"
        )
    
    def filter_by_correctness(
        self,
        rollouts: list[dict[str, Any]],
        task_goals: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Filter rollouts by fine-grained correctness.
        
        Args:
            rollouts: List of rollout trajectories
            task_goals: Optional list of task goals
        
        Returns:
            Filtered list of trajectories (including extracted prefixes)
        """
        if task_goals is None:
            task_goals = [r.get("task_goal", "") for r in rollouts]
        
        filtered = []
        
        for rollout, goal in zip(rollouts, task_goals):
            trajectory = rollout.get("trajectory", [])
            if not trajectory:
                continue
            
            # Analyze trajectory
            correct_prefix, error_step = self.analyzer.analyze_and_extract(
                goal, trajectory
            )
            
            if error_step < 0:
                # All steps correct, keep full trajectory
                filtered.append(rollout)
            elif len(correct_prefix) > 0:
                # Extract correct prefix
                correctness_ratio = len(correct_prefix) / len(trajectory)
                
                if correctness_ratio >= self.min_correctness_ratio:
                    # Create new rollout with correct prefix
                    prefix_rollout = rollout.copy()
                    prefix_rollout["trajectory"] = correct_prefix
                    prefix_rollout["original_length"] = len(trajectory)
                    prefix_rollout["prefix_length"] = len(correct_prefix)
                    prefix_rollout["correctness_ratio"] = correctness_ratio
                    filtered.append(prefix_rollout)
                    self.stats["extracted_prefixes"] += 1
                else:
                    # Correctness ratio too low, discard
                    logger.debug(
                        f"Discarding trajectory: "
                        f"correctness_ratio={correctness_ratio:.2f} < "
                        f"{self.min_correctness_ratio}"
                    )
            # else: error at first step, discard
        
        self.stats["filtered_rollouts"] += len(filtered)
        logger.info(
            f"Filtered {len(rollouts)} -> {len(filtered)} trajectories "
            f"({len(filtered)/len(rollouts)*100:.1f}% kept)"
        )
        
        return filtered
    
    def get_training_data(
        self,
        synthesis_trajectories: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Get training data for current iteration.
        
        Combines filtered rejection-sampled trajectories with
        synthesis trajectories (manual annotation + automatic rollout).
        
        Args:
            synthesis_trajectories: Optional synthesis trajectories to mix in
        
        Returns:
            Combined training data
        """
        # Filter current buffer
        filtered = self.filter_by_correctness(self.trajectory_buffer)
        
        # Mix with synthesis trajectories
        if synthesis_trajectories:
            training_data = filtered + synthesis_trajectories
            logger.info(
                f"Mixing {len(filtered)} filtered + "
                f"{len(synthesis_trajectories)} synthesis = "
                f"{len(training_data)} total trajectories"
            )
        else:
            training_data = filtered
        
        return training_data
    
    def next_iteration(self) -> bool:
        """Move to next iteration.
        
        Returns:
            True if more iterations remaining, False otherwise
        """
        if self.current_iteration >= self.num_iterations:
            return False
        
        self.current_iteration += 1
        # Clear buffer for next iteration (keep stats)
        self.trajectory_buffer = []
        logger.info(f"Starting iteration {self.current_iteration}/{self.num_iterations}")
        return True
    
    def get_statistics(self) -> dict[str, Any]:
        """Get rejection sampling statistics."""
        return {
            "current_iteration": self.current_iteration,
            "total_iterations": self.num_iterations,
            "total_rollouts": self.stats["total_rollouts"],
            "filtered_rollouts": self.stats["filtered_rollouts"],
            "extracted_prefixes": self.stats["extracted_prefixes"],
            "filter_rate": (
                self.stats["filtered_rollouts"] / self.stats["total_rollouts"]
                if self.stats["total_rollouts"] > 0
                else 0.0
            ),
        }
