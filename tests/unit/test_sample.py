"""Tests for Sample and Turn models."""

import pytest
from pydantic import ValidationError

from syntheta.schema.sample import Sample, Turn


class TestTurn:
    def test_valid_turn(self):
        t = Turn(role="user", content="Hello")
        assert t.role == "user"
        assert t.content == "Hello"

    def test_invalid_role(self):
        with pytest.raises(ValidationError):
            Turn(role="invalid", content="Hello")


class TestSample:
    def test_minimal_instruction(self):
        s = Sample(instruction="What is Python?")
        assert s.instruction == "What is Python?"
        assert s.id  # auto-generated UUID
        assert s.response is None
        assert s.extra == {}
        assert s.evolution_history == []

    def test_minimal_turns(self):
        s = Sample(turns=[Turn(role="user", content="Hi")])
        assert len(s.turns) == 1

    def test_minimal_text(self):
        s = Sample(text="Some pretraining text here.")
        assert s.text == "Some pretraining text here."

    def test_no_content_field_raises(self):
        with pytest.raises(ValidationError, match="At least one content field"):
            Sample()

    def test_all_fields(self, sample_sft):
        assert sample_sft.domain == "programming"
        assert sample_sft.difficulty == 1

    def test_extra_preserved(self):
        s = Sample(instruction="test", extra={"custom_field": 42})
        assert s.extra["custom_field"] == 42

    def test_unique_ids(self):
        s1 = Sample(instruction="a")
        s2 = Sample(instruction="b")
        assert s1.id != s2.id

    def test_serialization_round_trip(self, sample_sft):
        data = sample_sft.model_dump()
        restored = Sample(**data)
        assert restored.instruction == sample_sft.instruction
        assert restored.response == sample_sft.response
        assert restored.domain == sample_sft.domain

    def test_dpo_fields(self, sample_dpo):
        assert sample_dpo.chosen is not None
        assert sample_dpo.rejected is not None

    def test_rlvr_fields(self, sample_rlvr):
        assert sample_rlvr.info == {"answer": 4, "type": "arithmetic"}

    def test_evolution_history_default(self):
        s = Sample(instruction="test")
        assert s.evolution_history == []

    def test_evolution_history_set(self):
        s = Sample(instruction="test", evolution_history=["deepen", "concretize"])
        assert len(s.evolution_history) == 2
