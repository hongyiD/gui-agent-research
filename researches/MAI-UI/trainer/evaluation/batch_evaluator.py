#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batch Evaluator for MAI-UI Models.

This module enables batch evaluation of multiple model checkpoints,
automatically generating comprehensive performance reports.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from metrics_analyzer import MetricsAnalyzer
from report_generator import ReportGenerator


class BatchEvaluator:
    """Batch evaluator for multiple model checkpoints."""
    
    def __init__(self, config: dict[str, Any]):
        """Initialize batch evaluator.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.evaluate_script = config.get("evaluate_script", "../evaluate.py")
        self.results_dir = Path(config.get("results_dir", "./batch_eval_results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def find_checkpoints(self, model_dir: str) -> list[str]:
        """Find all checkpoints in a model directory.
        
        Args:
            model_dir: Directory containing model checkpoints.
            
        Returns:
            List of checkpoint paths.
        """
        model_path = Path(model_dir)
        checkpoints = []
        
        if (model_path / "pytorch_model.bin").exists() or (model_path / "model.safetensors").exists():
            checkpoints.append(str(model_path))
        
        for checkpoint_dir in sorted(model_path.glob("checkpoint-*")):
            if checkpoint_dir.is_dir():
                if (checkpoint_dir / "pytorch_model.bin").exists() or (checkpoint_dir / "model.safetensors").exists():
                    checkpoints.append(str(checkpoint_dir))
        
        return checkpoints
    
    def evaluate_checkpoint(
        self,
        checkpoint_path: str,
        checkpoint_name: str,
        eval_args: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate a single checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint.
            checkpoint_name: Name for the checkpoint.
            eval_args: Evaluation arguments.
            
        Returns:
            Evaluation result dictionary.
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {checkpoint_name}")
        print(f"Path: {checkpoint_path}")
        print(f"{'='*60}\n")
        
        output_dir = self.results_dir / checkpoint_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "results.json"
        
        cmd = [
            sys.executable,
            self.evaluate_script,
            "--model_name", checkpoint_path,
            "--log_root", str(output_dir / "logs"),
            "--output_path", str(output_file),
        ]
        
        for key, value in eval_args.items():
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            elif isinstance(value, list):
                for item in value:
                    cmd.extend([f"--{key}", str(item)])
            else:
                cmd.extend([f"--{key}", str(value)])
        
        result = {
            "checkpoint_name": checkpoint_name,
            "checkpoint_path": checkpoint_path,
            "success": False,
            "output_file": str(output_file),
        }
        
        try:
            print(f"Running: {' '.join(cmd)}\n")
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            
            result["success"] = process.returncode == 0
            result["return_code"] = process.returncode
            
            if process.returncode != 0:
                result["error"] = process.stderr
                print(f"Evaluation failed: {process.stderr}")
            else:
                print(f"Evaluation completed successfully")
            
            if process.stdout:
                print(f"Output:\n{process.stdout}\n")
        
        except Exception as e:
            result["error"] = str(e)
            print(f"Exception during evaluation: {e}")
        
        with open(output_dir / "evaluation_info.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result
    
    def run_batch_evaluation(
        self,
        model_dir: str,
        eval_args: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Run batch evaluation on all checkpoints.
        
        Args:
            model_dir: Directory containing model checkpoints.
            eval_args: Evaluation arguments.
            
        Returns:
            List of evaluation results.
        """
        checkpoints = self.find_checkpoints(model_dir)
        
        if not checkpoints:
            print(f"No checkpoints found in {model_dir}")
            return []
        
        print(f"Found {len(checkpoints)} checkpoint(s):")
        for cp in checkpoints:
            print(f"  - {cp}")
        print()
        
        results = []
        
        for checkpoint_path in tqdm(checkpoints, desc="Evaluating checkpoints"):
            checkpoint_name = Path(checkpoint_path).name
            
            result = self.evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                checkpoint_name=checkpoint_name,
                eval_args=eval_args,
            )
            results.append(result)
        
        return results
    
    def generate_reports(self) -> None:
        """Generate evaluation reports from batch results."""
        print(f"\n{'='*60}")
        print("Generating Reports")
        print(f"{'='*60}\n")
        
        analyzer = MetricsAnalyzer()
        
        for checkpoint_dir in self.results_dir.iterdir():
            if not checkpoint_dir.is_dir():
                continue
            
            results_file = checkpoint_dir / "results.json"
            if results_file.exists():
                try:
                    analyzer.load_evaluation_results(str(results_file))
                except Exception as e:
                    print(f"Warning: Failed to load {results_file}: {e}")
        
        if not analyzer.checkpoints:
            print("No valid evaluation results found")
            return
        
        generator = ReportGenerator(analyzer)
        
        markdown_report = self.results_dir / "evaluation_report.md"
        generator.generate_markdown_report(str(markdown_report))
        print(f"Generated Markdown report: {markdown_report}")
        
        json_report = self.results_dir / "evaluation_report.json"
        generator.generate_json_report(str(json_report))
        print(f"Generated JSON report: {json_report}")
        
        html_report = self.results_dir / "evaluation_report.html"
        generator.generate_html_report(str(html_report))
        print(f"Generated HTML report: {html_report}")


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluation for MAI-UI models")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to batch evaluation config YAML",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./batch_eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        default="mai_ui_agent",
        help="Agent type",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        required=True,
        help="LLM base URL",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        help="Comma-separated list of tasks or path to task file",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=50,
        help="Maximum steps per task",
    )
    parser.add_argument(
        "--enable-mcp",
        action="store_true",
        help="Enable MCP tools",
    )
    
    args = parser.parse_args()
    
    if args.config:
        config = load_config(args.config)
    else:
        config = {
            "results_dir": args.results_dir,
        }
    
    eval_args = {
        "agent_type": args.agent_type,
        "llm_base_url": args.llm_base_url,
        "max_step": args.max_step,
    }
    
    if args.tasks:
        eval_args["tasks"] = args.tasks
    
    if args.enable_mcp:
        eval_args["enable_mcp"] = True
    
    evaluator = BatchEvaluator(config)
    
    results = evaluator.run_batch_evaluation(
        model_dir=args.model_dir,
        eval_args=eval_args,
    )
    
    print(f"\n{'='*60}")
    print("Batch Evaluation Summary")
    print(f"{'='*60}\n")
    
    total = len(results)
    successful = sum(1 for r in results if r["success"])
    failed = total - successful
    
    print(f"Total checkpoints: {total}")
    print(f"Successful evaluations: {successful}")
    print(f"Failed evaluations: {failed}")
    print()
    
    evaluator.generate_reports()
    
    print(f"\nAll results saved to: {evaluator.results_dir}")


if __name__ == "__main__":
    main()
