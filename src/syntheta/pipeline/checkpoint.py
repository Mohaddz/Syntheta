"""Checkpoint management for pipeline fault tolerance."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from syntheta.exceptions import CheckpointError

logger = logging.getLogger("syntheta.checkpoint")


class CheckpointState:
    """Represents the state of a pipeline at a checkpoint."""

    def __init__(
        self,
        batch_num: int = 0,
        total_generated: int = 0,
        total_passed: int = 0,
        config_hash: str = "",
        filter_stats: dict[str, int] | None = None,
        cost_usage: dict[str, Any] | None = None,
    ) -> None:
        self.batch_num = batch_num
        self.total_generated = total_generated
        self.total_passed = total_passed
        self.config_hash = config_hash
        self.filter_stats = filter_stats or {}
        self.cost_usage = cost_usage or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_num": self.batch_num,
            "total_generated": self.total_generated,
            "total_passed": self.total_passed,
            "config_hash": self.config_hash,
            "filter_stats": self.filter_stats,
            "cost_usage": self.cost_usage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointState:
        return cls(**data)


class CheckpointManager:
    """Manages pipeline checkpoints for fault tolerance and resume."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self._state_file = self.checkpoint_path / "state.json"

    def save(self, state: CheckpointState) -> None:
        """Save current state to state.json."""
        try:
            with self._state_file.open("w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f, indent=2)
            logger.debug("Checkpoint saved: batch %d", state.batch_num)
        except OSError as e:
            raise CheckpointError(f"Failed to save checkpoint: {e}") from e

    def load(self) -> CheckpointState | None:
        """Load state from state.json. Returns None if no checkpoint exists."""
        if not self._state_file.exists():
            return None
        try:
            with self._state_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return CheckpointState.from_dict(data)
        except (json.JSONDecodeError, OSError, KeyError) as e:
            raise CheckpointError(f"Failed to load checkpoint: {e}") from e

    def verify_config_hash(self, current_hash: str) -> bool:
        """Check if current config matches the checkpointed config."""
        state = self.load()
        if state is None:
            return True
        return state.config_hash == current_hash

    @staticmethod
    def compute_config_hash(config_dict: dict[str, Any]) -> str:
        """Compute a deterministic hash of a config dict."""
        serialized = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
