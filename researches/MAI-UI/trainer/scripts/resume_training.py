#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Resume Training Tool for MAI-UI Pipeline.

This script helps resume interrupted training sessions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from checkpoint_manager import CheckpointManager


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume interrupted training session"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./pipeline_checkpoints",
        help="Directory containing pipeline checkpoints",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all checkpoints",
    )
    parser.add_argument(
        "--stage",
        type=str,
        help="Filter checkpoints by stage",
    )
    parser.add_argument(
        "--status",
        type=str,
        choices=["pending", "running", "completed", "failed"],
        help="Filter checkpoints by status",
    )
    parser.add_argument(
        "--delete",
        type=str,
        help="Delete a specific checkpoint by ID",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up old checkpoints (keep last 5)",
    )
    
    args = parser.parse_args()
    
    manager = CheckpointManager(args.checkpoint_dir)
    
    if args.list:
        checkpoints = manager.list_checkpoints(
            stage=args.stage,
            status=args.status,
        )
        
        if not checkpoints:
            print("No checkpoints found")
            return
        
        print(f"\n{'='*80}")
        print(f"Checkpoints in {args.checkpoint_dir}")
        print(f"{'='*80}\n")
        
        for cp in checkpoints:
            print(f"ID: {cp.checkpoint_id}")
            print(f"  Stage: {cp.stage}")
            print(f"  Status: {cp.status}")
            print(f"  Timestamp: {cp.timestamp}")
            if cp.error:
                print(f"  Error: {cp.error[:100]}...")
            if cp.metrics:
                print(f"  Metrics: {cp.metrics}")
            print()
        
        return
    
    if args.delete:
        manager.delete_checkpoint(args.delete, remove_files=True)
        print(f"Deleted checkpoint: {args.delete}")
        return
    
    if args.cleanup:
        manager.cleanup_old_checkpoints(keep_last=5, stage=args.stage)
        print("Cleaned up old checkpoints")
        return
    
    latest = manager.get_latest_checkpoint(stage=args.stage)
    
    if latest is None:
        print("No checkpoints found")
        return
    
    print(f"\nLatest checkpoint:")
    print(f"  ID: {latest.checkpoint_id}")
    print(f"  Stage: {latest.stage}")
    print(f"  Status: {latest.status}")
    print(f"  Timestamp: {latest.timestamp}")
    
    if latest.status == "failed":
        print(f"\nTo resume from this checkpoint, run:")
        print(f"  python pipeline/orchestrator.py --config pipeline/pipeline_config.yaml --start-from {latest.stage}")
    elif latest.status == "completed":
        print(f"\nThis checkpoint completed successfully")
    else:
        print(f"\nThis checkpoint is in '{latest.status}' state")


if __name__ == "__main__":
    main()
