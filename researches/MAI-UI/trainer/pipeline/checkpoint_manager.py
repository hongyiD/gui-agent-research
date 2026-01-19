#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Checkpoint Manager for MAI-UI Training Pipeline.

This module manages training checkpoints, enabling resumption and recovery.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    
    checkpoint_id: str
    stage: str
    timestamp: str
    status: str
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class CheckpointManager:
    """Manages checkpoints for the training pipeline."""
    
    def __init__(self, checkpoint_dir: str):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "checkpoints.json"
        self.checkpoints: dict[str, CheckpointMetadata] = {}
        self._load_checkpoints()
    
    def _load_checkpoints(self) -> None:
        """Load existing checkpoints from metadata file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for cp_id, cp_data in data.items():
                    self.checkpoints[cp_id] = CheckpointMetadata(**cp_data)
    
    def _save_checkpoints(self) -> None:
        """Save checkpoints metadata to file."""
        data = {
            cp_id: {
                "checkpoint_id": cp.checkpoint_id,
                "stage": cp.stage,
                "timestamp": cp.timestamp,
                "status": cp.status,
                "config": cp.config,
                "metrics": cp.metrics,
                "error": cp.error,
            }
            for cp_id, cp in self.checkpoints.items()
        }
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def create_checkpoint(
        self,
        stage: str,
        config: dict[str, Any],
        status: str = "pending",
    ) -> str:
        """Create a new checkpoint.
        
        Args:
            stage: Pipeline stage name.
            config: Configuration dict for this stage.
            status: Initial status.
            
        Returns:
            Checkpoint ID.
        """
        checkpoint_id = f"{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        checkpoint = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            stage=stage,
            timestamp=datetime.now().isoformat(),
            status=status,
            config=config,
        )
        
        self.checkpoints[checkpoint_id] = checkpoint
        self._save_checkpoints()
        
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        return checkpoint_id
    
    def update_checkpoint(
        self,
        checkpoint_id: str,
        status: str | None = None,
        metrics: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Update checkpoint status and metrics.
        
        Args:
            checkpoint_id: Checkpoint ID.
            status: New status.
            metrics: Metrics to update.
            error: Error message if any.
        """
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        checkpoint = self.checkpoints[checkpoint_id]
        
        if status is not None:
            checkpoint.status = status
        
        if metrics is not None:
            checkpoint.metrics.update(metrics)
        
        if error is not None:
            checkpoint.error = error
        
        self._save_checkpoints()
    
    def get_checkpoint(self, checkpoint_id: str) -> CheckpointMetadata | None:
        """Get checkpoint metadata.
        
        Args:
            checkpoint_id: Checkpoint ID.
            
        Returns:
            CheckpointMetadata or None if not found.
        """
        return self.checkpoints.get(checkpoint_id)
    
    def get_latest_checkpoint(self, stage: str | None = None) -> CheckpointMetadata | None:
        """Get the latest checkpoint, optionally filtered by stage.
        
        Args:
            stage: Pipeline stage to filter by.
            
        Returns:
            Latest CheckpointMetadata or None.
        """
        filtered = [
            cp for cp in self.checkpoints.values()
            if stage is None or cp.stage == stage
        ]
        
        if not filtered:
            return None
        
        return max(filtered, key=lambda cp: cp.timestamp)
    
    def get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get filesystem path for a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID.
            
        Returns:
            Path to checkpoint directory.
        """
        return self.checkpoint_dir / checkpoint_id
    
    def list_checkpoints(
        self,
        stage: str | None = None,
        status: str | None = None,
    ) -> list[CheckpointMetadata]:
        """List checkpoints with optional filters.
        
        Args:
            stage: Filter by stage.
            status: Filter by status.
            
        Returns:
            List of matching checkpoints.
        """
        result = []
        for cp in self.checkpoints.values():
            if stage is not None and cp.stage != stage:
                continue
            if status is not None and cp.status != status:
                continue
            result.append(cp)
        
        return sorted(result, key=lambda cp: cp.timestamp, reverse=True)
    
    def delete_checkpoint(self, checkpoint_id: str, remove_files: bool = True) -> None:
        """Delete a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID to delete.
            remove_files: If True, also remove checkpoint files from disk.
        """
        if checkpoint_id in self.checkpoints:
            del self.checkpoints[checkpoint_id]
            self._save_checkpoints()
        
        if remove_files:
            checkpoint_path = self.get_checkpoint_path(checkpoint_id)
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
    
    def cleanup_old_checkpoints(
        self,
        keep_last: int = 5,
        stage: str | None = None,
    ) -> None:
        """Clean up old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_last: Number of checkpoints to keep.
            stage: Only clean up checkpoints for this stage.
        """
        checkpoints = self.list_checkpoints(stage=stage)
        
        if len(checkpoints) <= keep_last:
            return
        
        to_delete = checkpoints[keep_last:]
        for cp in to_delete:
            if cp.status != "completed":
                continue
            self.delete_checkpoint(cp.checkpoint_id, remove_files=True)
