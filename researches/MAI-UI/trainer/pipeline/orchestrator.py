#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline Orchestrator for MAI-UI Training.

This module provides end-to-end orchestration of the training pipeline,
from data preprocessing through training to evaluation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from checkpoint_manager import CheckpointManager


@dataclass
class PipelineStage:
    """Represents a single stage in the pipeline."""
    
    name: str
    description: str
    script: str
    args: dict[str, Any]
    depends_on: list[str]
    enabled: bool = True
    allow_failure: bool = False


class PipelineOrchestrator:
    """Orchestrates the complete training pipeline."""
    
    def __init__(self, config_path: str):
        """Initialize orchestrator.
        
        Args:
            config_path: Path to pipeline configuration YAML.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.checkpoint_manager = CheckpointManager(
            self.config.get("checkpoint_dir", "./pipeline_checkpoints")
        )
        
        self.stages = self._build_stages()
        self.execution_log: list[dict[str, Any]] = []
    
    def _build_stages(self) -> list[PipelineStage]:
        """Build pipeline stages from config.
        
        Returns:
            List of PipelineStage objects.
        """
        stages = []
        
        for stage_config in self.config.get("stages", []):
            stage = PipelineStage(
                name=stage_config["name"],
                description=stage_config.get("description", ""),
                script=stage_config["script"],
                args=stage_config.get("args", {}),
                depends_on=stage_config.get("depends_on", []),
                enabled=stage_config.get("enabled", True),
                allow_failure=stage_config.get("allow_failure", False),
            )
            stages.append(stage)
        
        return stages
    
    def _resolve_args(self, args: dict[str, Any]) -> list[str]:
        """Resolve stage arguments to command-line format.
        
        Args:
            args: Dictionary of arguments.
            
        Returns:
            List of command-line argument strings.
        """
        cmd_args = []
        
        for key, value in args.items():
            if value is None:
                continue
            
            if isinstance(value, bool):
                if value:
                    cmd_args.append(f"--{key}")
            elif isinstance(value, list):
                for item in value:
                    cmd_args.extend([f"--{key}", str(item)])
            else:
                cmd_args.extend([f"--{key}", str(value)])
        
        return cmd_args
    
    def _run_stage(self, stage: PipelineStage) -> dict[str, Any]:
        """Run a single pipeline stage.
        
        Args:
            stage: Pipeline stage to run.
            
        Returns:
            Execution result dictionary.
        """
        print(f"\n{'='*60}")
        print(f"Running stage: {stage.name}")
        print(f"Description: {stage.description}")
        print(f"{'='*60}\n")
        
        checkpoint_id = self.checkpoint_manager.create_checkpoint(
            stage=stage.name,
            config=stage.args,
            status="running",
        )
        
        cmd_args = self._resolve_args(stage.args)
        cmd = [sys.executable, stage.script] + cmd_args
        
        print(f"Command: {' '.join(cmd)}\n")
        
        result = {
            "stage": stage.name,
            "checkpoint_id": checkpoint_id,
            "success": False,
            "return_code": None,
            "error": None,
        }
        
        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=not stage.allow_failure,
            )
            
            result["success"] = process.returncode == 0
            result["return_code"] = process.returncode
            
            if process.returncode != 0:
                result["error"] = process.stderr
                print(f"Stage {stage.name} failed with return code {process.returncode}")
                print(f"Error: {process.stderr}")
            else:
                print(f"Stage {stage.name} completed successfully")
            
            print(f"\nStdout:\n{process.stdout}")
            
            if not result["success"]:
                self.checkpoint_manager.update_checkpoint(
                    checkpoint_id,
                    status="failed",
                    error=result["error"],
                )
            else:
                self.checkpoint_manager.update_checkpoint(
                    checkpoint_id,
                    status="completed",
                )
        
        except Exception as e:
            result["error"] = str(e)
            print(f"Exception during stage {stage.name}: {e}")
            self.checkpoint_manager.update_checkpoint(
                checkpoint_id,
                status="failed",
                error=str(e),
            )
        
        self.execution_log.append(result)
        return result
    
    def _check_dependencies(self, stage: PipelineStage) -> bool:
        """Check if all dependencies for a stage are satisfied.
        
        Args:
            stage: Pipeline stage to check.
            
        Returns:
            True if all dependencies are satisfied.
        """
        if not stage.depends_on:
            return True
        
        for dep_stage_name in stage.depends_on:
            dep_result = next(
                (r for r in self.execution_log if r["stage"] == dep_stage_name),
                None
            )
            
            if dep_result is None:
                print(f"Dependency {dep_stage_name} has not been executed")
                return False
            
            if not dep_result["success"]:
                print(f"Dependency {dep_stage_name} failed")
                return False
        
        return True
    
    def run(self, start_from: str | None = None, stop_at: str | None = None) -> bool:
        """Run the complete pipeline.
        
        Args:
            start_from: Stage name to start from (skip earlier stages).
            stop_at: Stage name to stop at (skip later stages).
            
        Returns:
            True if all stages succeeded.
        """
        print(f"\n{'*'*60}")
        print(f"Starting MAI-UI Training Pipeline")
        print(f"{'*'*60}\n")
        
        should_run = start_from is None
        all_success = True
        
        for stage in self.stages:
            if start_from and stage.name == start_from:
                should_run = True
            
            if not should_run:
                print(f"Skipping stage: {stage.name} (before start_from)")
                continue
            
            if not stage.enabled:
                print(f"Skipping stage: {stage.name} (disabled)")
                continue
            
            if not self._check_dependencies(stage):
                print(f"Skipping stage: {stage.name} (dependencies not satisfied)")
                all_success = False
                continue
            
            result = self._run_stage(stage)
            
            if not result["success"] and not stage.allow_failure:
                print(f"\nPipeline stopped due to failure in stage: {stage.name}")
                all_success = False
                break
            
            if stop_at and stage.name == stop_at:
                print(f"\nPipeline stopped at stage: {stage.name}")
                break
        
        print(f"\n{'*'*60}")
        print(f"Pipeline Execution Summary")
        print(f"{'*'*60}\n")
        
        self._print_summary()
        
        log_file = Path(self.config.get("log_dir", ".")) / "pipeline_execution.json"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(self.execution_log, f, indent=2, ensure_ascii=False)
        
        print(f"\nExecution log saved to: {log_file}")
        
        return all_success
    
    def _print_summary(self) -> None:
        """Print execution summary."""
        total = len(self.execution_log)
        successful = sum(1 for r in self.execution_log if r["success"])
        failed = total - successful
        
        print(f"Total stages: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print()
        
        for result in self.execution_log:
            status = "✓" if result["success"] else "✗"
            print(f"{status} {result['stage']}")
            if result.get("error"):
                print(f"  Error: {result['error'][:100]}...")
    
    def resume(self) -> bool:
        """Resume pipeline from the last failed stage.
        
        Returns:
            True if resumed successfully.
        """
        if not self.execution_log:
            print("No execution log found. Starting from beginning.")
            return self.run()
        
        last_failed = None
        for result in reversed(self.execution_log):
            if not result["success"]:
                last_failed = result["stage"]
                break
        
        if last_failed is None:
            print("No failed stages found. Pipeline already completed successfully.")
            return True
        
        print(f"Resuming from failed stage: {last_failed}")
        return self.run(start_from=last_failed)


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(description="MAI-UI Training Pipeline Orchestrator")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to pipeline configuration YAML",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        help="Stage name to start from (skip earlier stages)",
    )
    parser.add_argument(
        "--stop-at",
        type=str,
        help="Stage name to stop at (skip later stages)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last failed stage",
    )
    
    args = parser.parse_args()
    
    orchestrator = PipelineOrchestrator(args.config)
    
    if args.resume:
        success = orchestrator.resume()
    else:
        success = orchestrator.run(
            start_from=args.start_from,
            stop_at=args.stop_at,
        )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
