#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standalone Data Validation Tool for MAI-UI Training Data.

This script validates training data files and reports issues.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

from format_validators import (
    DataValidator,
    validate_trajectory_jsonl,
    print_validation_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate MAI-UI training data files"
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        required=True,
        help="Path to JSONL file to validate",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["openai", "prompt_response", "full_trajectory", "auto"],
        default="auto",
        help="Expected format type (auto-detect if not specified)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise exceptions on validation errors",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Save validation report to file",
    )
    
    args = parser.parse_args()
    
    try:
        if args.format == "auto":
            report = validate_trajectory_jsonl(args.file, strict=args.strict)
        else:
            validator = DataValidator(strict=args.strict)
            report = validator.validate_jsonl_file(args.file, args.format)
        
        print_validation_report(report)
        
        if args.output:
            import json
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"Validation report saved to: {output_path}")
        
        sys.exit(0 if report["is_valid"] else 1)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
