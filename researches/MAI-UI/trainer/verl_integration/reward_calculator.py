#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reward Calculator for MAI-UI.

Implements reward computation with task completion reward and
repetition penalty as described in the MAI-UI paper.
"""

from __future__ import annotations

from typing import Any


class MAIUIRewardCalculator:
    """MAI-UI reward calculator.
    
    Computes rewards with:
    - Task completion reward (binary: 1.0 for success, 0.0 for failure)
    - Repetition penalty (for immediate repetition and 3-5 step cycles)
    """
    
    def __init__(
        self,
        repetition_penalty: float = 0.1,
        cycle_penalty_multiplier: float = 2.0,
        min_reward: float = 0.0,
    ):
        """Initialize reward calculator.
        
        Args:
            repetition_penalty: Penalty for immediate action repetition
            cycle_penalty_multiplier: Multiplier for cycle penalties
            min_reward: Minimum reward value (clipping lower bound)
        """
        self.repetition_penalty = repetition_penalty
        self.cycle_penalty_multiplier = cycle_penalty_multiplier
        self.min_reward = min_reward
    
    def compute_reward(
        self,
        task_score: float,
        trajectory: list[dict[str, Any]],
        repetition_penalty: float | None = None,
    ) -> float:
        """Compute reward for a trajectory.
        
        Args:
            task_score: Task completion score (0.0-1.0)
            trajectory: List of trajectory steps, each containing 'action' dict
            repetition_penalty: Override default repetition penalty
        
        Returns:
            Computed reward value
        """
        if repetition_penalty is None:
            repetition_penalty = self.repetition_penalty
        
        reward = task_score
        
        # Extract action sequence
        action_sequence = []
        for step in trajectory:
            action = step.get("action", {})
            action_type = action.get("action_type")
            if action_type:
                action_sequence.append(action_type)
        
        if len(action_sequence) < 2:
            return max(reward, self.min_reward)
        
        # Check for immediate repetition
        for i in range(len(action_sequence) - 1):
            if action_sequence[i] == action_sequence[i + 1]:
                reward -= repetition_penalty
        
        # Check for cycles of 3-5 actions
        for cycle_len in [3, 4, 5]:
            if len(action_sequence) >= cycle_len * 2:
                for i in range(len(action_sequence) - cycle_len * 2 + 1):
                    cycle1 = action_sequence[i : i + cycle_len]
                    cycle2 = action_sequence[i + cycle_len : i + cycle_len * 2]
                    if cycle1 == cycle2:
                        reward -= repetition_penalty * self.cycle_penalty_multiplier
                        break  # Only penalize once per trajectory
        
        return max(reward, self.min_reward)
    
    def compute_reward_batch(
        self,
        task_scores: list[float],
        trajectories: list[list[dict[str, Any]]],
        repetition_penalty: float | None = None,
    ) -> list[float]:
        """Compute rewards for a batch of trajectories.
        
        Args:
            task_scores: List of task completion scores
            trajectories: List of trajectories
            repetition_penalty: Override default repetition penalty
        
        Returns:
            List of computed rewards
        """
        return [
            self.compute_reward(score, traj, repetition_penalty)
            for score, traj in zip(task_scores, trajectories)
        ]
