"""Shared test fixtures for Syntheta."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from syntheta.schema.sample import Sample, Turn


@pytest.fixture
def sample_sft() -> Sample:
    """A minimal SFT sample."""
    return Sample(
        instruction="What is Python?",
        response="Python is a high-level programming language.",
        domain="programming",
        topic="Python",
        task_type="qa",
        language="en",
        difficulty=1,
    )


@pytest.fixture
def sample_chat() -> Sample:
    """A multi-turn chat sample."""
    return Sample(
        turns=[
            Turn(role="user", content="Hello!"),
            Turn(role="assistant", content="Hi! How can I help you?"),
            Turn(role="user", content="Tell me about Python."),
            Turn(role="assistant", content="Python is a programming language."),
        ],
    )


@pytest.fixture
def sample_dpo() -> Sample:
    """A DPO preference sample."""
    return Sample(
        instruction="Explain gravity.",
        chosen="Gravity is the fundamental force of attraction between objects with mass.",
        rejected="Gravity is what makes things fall.",
    )


@pytest.fixture
def sample_rlvr() -> Sample:
    """An RLVR judge sample."""
    return Sample(
        instruction="What is 2 + 2?",
        info={"answer": 4, "type": "arithmetic"},
    )


@pytest.fixture
def sample_pretrain() -> Sample:
    """A pretraining text sample."""
    return Sample(
        text=(
            "Python is a high-level, general-purpose programming language. "
            "Its design philosophy emphasizes code readability."
        ),
    )


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory."""
    return tmp_path


@pytest.fixture
def sft_jsonl(tmp_path: Path) -> Path:
    """Create a temporary JSONL file with SFT-format data."""
    path = tmp_path / "sft_data.jsonl"
    records = [
        {
            "instruction": f"Question {i}",
            "response": f"Answer {i} with enough detail to pass validation checks.",
            "domain": "test",
            "language": "en",
        }
        for i in range(5)
    ]
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path
