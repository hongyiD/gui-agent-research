#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Report Generator for MAI-UI Model Evaluation.

This module generates comprehensive evaluation reports in various formats.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from metrics_analyzer import MetricsAnalyzer, ModelMetrics


class ReportGenerator:
    """Generator for evaluation reports."""
    
    def __init__(self, analyzer: MetricsAnalyzer):
        """Initialize report generator.
        
        Args:
            analyzer: MetricsAnalyzer instance with loaded results.
        """
        self.analyzer = analyzer
    
    def generate_markdown_report(
        self,
        output_path: str,
        checkpoint_names: list[str] | None = None,
    ) -> None:
        """Generate a Markdown evaluation report.
        
        Args:
            output_path: Path to save the Markdown report.
            checkpoint_names: List of checkpoint names to include.
        """
        if checkpoint_names is None:
            checkpoint_names = list(self.analyzer.checkpoints.keys())
        
        lines = []
        
        lines.append("# MAI-UI Model Evaluation Report")
        lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        comparison = self.analyzer.compare_checkpoints(checkpoint_names)
        
        lines.append("## Summary")
        lines.append(f"\n- **Total Checkpoints Evaluated:** {len(checkpoint_names)}")
        if comparison["best_checkpoint"]:
            lines.append(f"- **Best Checkpoint:** {comparison['best_checkpoint']}")
        lines.append("")
        
        lines.append("## Performance Comparison")
        lines.append("")
        lines.append("| Checkpoint | Success Rate | Avg Steps | Avg Score | Repetition Rate |")
        lines.append("|------------|--------------|-----------|-----------|-----------------|")
        
        for cp_data in comparison["checkpoints"]:
            lines.append(
                f"| {cp_data['name']} "
                f"| {cp_data['success_rate']:.1%} "
                f"| {cp_data['avg_steps']:.1f} "
                f"| {cp_data['avg_score']:.3f} "
                f"| {cp_data['action_repetition_rate']:.1%} |"
            )
        
        lines.append("")
        
        if comparison["improvement"]:
            lines.append("## Improvement Over Training")
            lines.append("")
            imp = comparison["improvement"]
            lines.append(f"- **Success Rate:** {imp['success_rate']:+.1%}")
            lines.append(f"- **Average Steps:** {imp['avg_steps']:+.1f}")
            lines.append(f"- **Average Score:** {imp['avg_score']:+.3f}")
            lines.append(f"- **Repetition Rate:** {imp['action_repetition_rate']:+.1%}")
            lines.append("")
        
        for cp_name in checkpoint_names:
            if cp_name not in self.analyzer.checkpoints:
                continue
            
            metrics = self.analyzer.checkpoints[cp_name]
            
            lines.append(f"## Checkpoint: {cp_name}")
            lines.append("")
            lines.append(f"- **Total Tasks:** {metrics.total_tasks}")
            lines.append(f"- **Successful:** {metrics.successful_tasks}")
            lines.append(f"- **Failed:** {metrics.failed_tasks}")
            lines.append(f"- **Success Rate:** {metrics.success_rate:.1%}")
            lines.append(f"- **Average Steps:** {metrics.avg_steps:.1f}")
            lines.append(f"- **Average Score:** {metrics.avg_score:.3f}")
            lines.append("")
            
            if metrics.action_distribution:
                lines.append("### Action Type Distribution")
                lines.append("")
                
                total_actions = sum(metrics.action_distribution.values())
                sorted_actions = sorted(
                    metrics.action_distribution.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                for action_type, count in sorted_actions[:10]:
                    percentage = (count / total_actions) * 100
                    lines.append(f"- **{action_type}:** {count} ({percentage:.1f}%)")
                
                lines.append("")
            
            breakdown = self.analyzer.get_task_breakdown(cp_name)
            
            if breakdown.get("failed_tasks"):
                lines.append("### Failed Tasks")
                lines.append("")
                for task in breakdown["failed_tasks"][:10]:
                    lines.append(f"- **{task['name']}** (Steps: {task['steps']}, Score: {task['score']:.3f})")
                    if task.get("error"):
                        lines.append(f"  - Error: {task['error'][:100]}...")
                
                if len(breakdown["failed_tasks"]) > 10:
                    lines.append(f"\n... and {len(breakdown['failed_tasks']) - 10} more failed tasks")
                
                lines.append("")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    
    def generate_json_report(
        self,
        output_path: str,
        checkpoint_names: list[str] | None = None,
    ) -> None:
        """Generate a JSON evaluation report.
        
        Args:
            output_path: Path to save the JSON report.
            checkpoint_names: List of checkpoint names to include.
        """
        if checkpoint_names is None:
            checkpoint_names = list(self.analyzer.checkpoints.keys())
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "checkpoints": {},
            "comparison": self.analyzer.compare_checkpoints(checkpoint_names),
        }
        
        for cp_name in checkpoint_names:
            if cp_name not in self.analyzer.checkpoints:
                continue
            
            metrics = self.analyzer.checkpoints[cp_name]
            breakdown = self.analyzer.get_task_breakdown(cp_name)
            
            report["checkpoints"][cp_name] = {
                "total_tasks": metrics.total_tasks,
                "successful_tasks": metrics.successful_tasks,
                "failed_tasks": metrics.failed_tasks,
                "success_rate": metrics.success_rate,
                "avg_steps": metrics.avg_steps,
                "avg_score": metrics.avg_score,
                "action_distribution": metrics.action_distribution,
                "action_repetition_rate": metrics.action_repetition_rate,
                "task_breakdown": breakdown,
            }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    def generate_html_report(
        self,
        output_path: str,
        checkpoint_names: list[str] | None = None,
    ) -> None:
        """Generate an HTML evaluation report with charts.
        
        Args:
            output_path: Path to save the HTML report.
            checkpoint_names: List of checkpoint names to include.
        """
        if checkpoint_names is None:
            checkpoint_names = list(self.analyzer.checkpoints.keys())
        
        comparison = self.analyzer.compare_checkpoints(checkpoint_names)
        
        html_lines = []
        html_lines.append("<!DOCTYPE html>")
        html_lines.append("<html>")
        html_lines.append("<head>")
        html_lines.append("  <meta charset='utf-8'>")
        html_lines.append("  <title>MAI-UI Evaluation Report</title>")
        html_lines.append("  <style>")
        html_lines.append("    body { font-family: Arial, sans-serif; margin: 20px; }")
        html_lines.append("    h1, h2, h3 { color: #333; }")
        html_lines.append("    table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
        html_lines.append("    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html_lines.append("    th { background-color: #4CAF50; color: white; }")
        html_lines.append("    tr:nth-child(even) { background-color: #f2f2f2; }")
        html_lines.append("    .metric { font-size: 1.2em; margin: 10px 0; }")
        html_lines.append("    .success { color: green; }")
        html_lines.append("    .failure { color: red; }")
        html_lines.append("  </style>")
        html_lines.append("</head>")
        html_lines.append("<body>")
        
        html_lines.append("  <h1>MAI-UI Model Evaluation Report</h1>")
        html_lines.append(f"  <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        html_lines.append("  <h2>Summary</h2>")
        html_lines.append(f"  <p>Total Checkpoints Evaluated: {len(checkpoint_names)}</p>")
        if comparison["best_checkpoint"]:
            html_lines.append(f"  <p>Best Checkpoint: <strong>{comparison['best_checkpoint']}</strong></p>")
        
        html_lines.append("  <h2>Performance Comparison</h2>")
        html_lines.append("  <table>")
        html_lines.append("    <tr>")
        html_lines.append("      <th>Checkpoint</th>")
        html_lines.append("      <th>Success Rate</th>")
        html_lines.append("      <th>Avg Steps</th>")
        html_lines.append("      <th>Avg Score</th>")
        html_lines.append("      <th>Repetition Rate</th>")
        html_lines.append("    </tr>")
        
        for cp_data in comparison["checkpoints"]:
            html_lines.append("    <tr>")
            html_lines.append(f"      <td>{cp_data['name']}</td>")
            html_lines.append(f"      <td>{cp_data['success_rate']:.1%}</td>")
            html_lines.append(f"      <td>{cp_data['avg_steps']:.1f}</td>")
            html_lines.append(f"      <td>{cp_data['avg_score']:.3f}</td>")
            html_lines.append(f"      <td>{cp_data['action_repetition_rate']:.1%}</td>")
            html_lines.append("    </tr>")
        
        html_lines.append("  </table>")
        
        for cp_name in checkpoint_names:
            if cp_name not in self.analyzer.checkpoints:
                continue
            
            metrics = self.analyzer.checkpoints[cp_name]
            
            html_lines.append(f"  <h2>Checkpoint: {cp_name}</h2>")
            html_lines.append(f"  <p class='metric'>Total Tasks: {metrics.total_tasks}</p>")
            html_lines.append(f"  <p class='metric success'>Successful: {metrics.successful_tasks}</p>")
            html_lines.append(f"  <p class='metric failure'>Failed: {metrics.failed_tasks}</p>")
            html_lines.append(f"  <p class='metric'>Success Rate: {metrics.success_rate:.1%}</p>")
        
        html_lines.append("</body>")
        html_lines.append("</html>")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(html_lines))


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate evaluation reports")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output file path (format determined by extension)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json", "html"],
        help="Output format (overrides file extension)",
    )
    
    args = parser.parse_args()
    
    analyzer = MetricsAnalyzer()
    analyzer.load_multiple_checkpoints(args.results_dir)
    
    generator = ReportGenerator(analyzer)
    
    output_format = args.format
    if output_format is None:
        ext = Path(args.output).suffix.lower()
        if ext == ".md":
            output_format = "markdown"
        elif ext == ".json":
            output_format = "json"
        elif ext == ".html":
            output_format = "html"
        else:
            output_format = "markdown"
    
    if output_format == "markdown":
        generator.generate_markdown_report(args.output)
    elif output_format == "json":
        generator.generate_json_report(args.output)
    elif output_format == "html":
        generator.generate_html_report(args.output)
    
    print(f"Report generated: {args.output}")


if __name__ == "__main__":
    main()
