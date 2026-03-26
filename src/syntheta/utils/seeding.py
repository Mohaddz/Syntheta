"""Seed management for reproducibility."""

from __future__ import annotations

import random


def create_rng(seed: int | None = None) -> random.Random:
    """Create a seeded Random instance for deterministic operations."""
    return random.Random(seed)


def deterministic_shuffle(items: list, rng: random.Random) -> list:
    """Shuffle a list in-place using a seeded RNG. Returns the list."""
    rng.shuffle(items)
    return items
