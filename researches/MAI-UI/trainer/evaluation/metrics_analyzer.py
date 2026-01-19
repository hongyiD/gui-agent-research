#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metrics Analyzer for MAI-UI Model Evaluation.

This module analyzes evaluation results and computes various metrics.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TaskMetrics:
    """Metrics for a single task."""
    
    task_name: str
    success: bool
    score: float
    total_steps: int
    action_types: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class ModelMetrics:
    """Comprehensive metrics for a model checkpoint."""
    
    checkpoint_name: str
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    success_rate: float = 0.0
    avg_steps: float = 0.0
    avg_score: float = 0.0
    action_distribution: dict[str, int] = field(default_factory=dict)
    action_repetition_rate: float = 0.0
    task_metrics: list[TaskMetrics] = field(default_factory=list)
    
    def compute_statistics(self) -> None:
        """Compute aggregate statistics from task metrics."""
        if not self.task_metrics:
            return
        
        self.total_tasks = len(self.task_metrics)
        self.successful_tasks = sum(1 for tm in self.task_metrics if tm.success)
        self.failed_tasks = self.total_tasks - self.successful_tasks
        
        if self.total_tasks > 0:
            self.success_rate = self.successful_tasks / self.total_tasks
            self.avg_steps = sum(tm.total_steps for tm in self.task_metrics) / self.total_tasks
            self.avg_score = sum(tm.score for tm in self.task_metrics) / self.total_tasks
        
        all_actions = []
        for tm in self.task_metrics:
            all_actions.extend(tm.action_types)
        
        self.action_distribution = dict(Counter(all_actions))
        
        repetitions = 0
        total_actions = 0
        for tm in self.task_metrics:
            if len(tm.action_types) > 1:
                for i in range(len(tm.action_types) - 1):
                    if tm.action_types[i] == tm.action_types[i + 1]:
                        repetitions += 1
                total_actions += len(tm.action_types) - 1
        
        if total_actions > 0:
            self.action_repetition_rate = repetitions / total_actions


class MetricsAnalyzer:
    """Analyzer for evaluation metrics."""
    
    def __init__(self):
        """Initialize analyzer."""
        self.checkpoints: dict[str, ModelMetrics] = {}
    
    def load_evaluation_results(self, results_path: str) -> ModelMetrics:
        """Load evaluation results from a JSON file.
        
        Args:
            results_path: Path to evaluation results JSON.
            
        Returns:
            ModelMetrics object.
        """
        checkpoint_name = Path(results_path).parent.name
        
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        
        metrics = ModelMetrics(checkpoint_name=checkpoint_name)
        
        for result in results:
            task_metric = TaskMetrics(
                task_name=result.get("task_name", "unknown"),
                success=result.get("success", False),
                score=result.get("score", 0.0),
                total_steps=result.get("total_steps", 0),
                action_types=result.get("action_types", []),
                error=result.get("error"),
            )
            metrics.task_metrics.append(task_metric)
        
        metrics.compute_statistics()
        self.checkpoints[checkpoint_name] = metrics
        
        return metrics
    
    def load_multiple_checkpoints(self, results_dir: str) -> dict[str, ModelMetrics]:
        """Load results from multiple checkpoints.
        
        Args:
            results_dir: Directory containing checkpoint evaluation results.
            
        Returns:
            Dictionary mapping checkpoint names to ModelMetrics.
        """
        results_path = Path(results_dir)
        
        for checkpoint_dir in results_path.iterdir():
            if not checkpoint_dir.is_dir():
                continue
            
            results_file = checkpoint_dir / "results.json"
            if results_file.exists():
                self.load_evaluation_results(str(results_file))
        
        return self.checkpoints
    
    def compare_checkpoints(
        self,
        checkpoint_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare metrics across multiple checkpoints.
        
        Args:
            checkpoint_names: List of checkpoint names to compare.
                             If None, compare all loaded checkpoints.
            
        Returns:
            Comparison dictionary with statistics.
        """
        if checkpoint_names is None:
            checkpoint_names = list(self.checkpoints.keys())
        
        comparison = {
            "checkpoints": [],
            "best_checkpoint": None,
            "improvement": {},
        }
        
        for cp_name in checkpoint_names:
            if cp_name not in self.checkpoints:
                continue
            
            metrics = self.checkpoints[cp_name]
            comparison["checkpoints"].append({
                "name": cp_name,
                "success_rate": metrics.success_rate,
                "avg_steps": metrics.avg_steps,
                "avg_score": metrics.avg_score,
                "action_repetition_rate": metrics.action_repetition_rate,
            })
        
        if comparison["checkpoints"]:
            best = max(
                comparison["checkpoints"],
                key=lambda x: (x["success_rate"], -x["avg_steps"])
            )
            comparison["best_checkpoint"] = best["name"]
            
            first = comparison["checkpoints"][0]
            last = comparison["checkpoints"][-1]
            
            comparison["improvement"] = {
                "success_rate": last["success_rate"] - first["success_rate"],
                "avg_steps": last["avg_steps"] - first["avg_steps"],
                "avg_score": last["avg_score"] - first["avg_score"],
                "action_repetition_rate": (
                    last["action_repetition_rate"] - first["action_repetition_rate"]
                ),
            }
        
        return comparison
    
    def get_checkpoint_metrics(self, checkpoint_name: str) -> ModelMetrics | None:
        """Get metrics for a specific checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint.
            
        Returns:
            ModelMetrics or None if not found.
        """
        return self.checkpoints.get(checkpoint_name)
    
    def get_task_breakdown(self, checkpoint_name: str) -> dict[str, Any]:
        """Get detailed task breakdown for a checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint.
            
        Returns:
            Task breakdown dictionary.
        """
        if checkpoint_name not in self.checkpoints:
            return {}
        
        metrics = self.checkpoints[checkpoint_name]
        
        breakdown = {
            "successful_tasks": [],
            "failed_tasks": [],
            "tasks_by_steps": defaultdict(list),
        }
        
        for tm in metrics.task_metrics:
            task_info = {
                "name": tm.task_name,
                "score": tm.score,
                "steps": tm.total_steps,
            }
            
            if tm.success:
                breakdown["successful_tasks"].append(task_info)
            else:
                task_info["error"] = tm.error
                breakdown["failed_tasks"].append(task_info)
            
            step_range = f"{(tm.total_steps // 10) * 10}-{(tm.total_steps // 10) * 10 + 9}"
            breakdown["tasks_by_steps"][step_range].append(task_info)
        
        breakdown["tasks_by_steps"] = dict(breakdown["tasks_by_steps"])
        
        return breakdown
