"""Tests for EvolInstruct transformer."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from syntheta.schema.sample import Sample
from syntheta.transformers.evol_instruct import EvolInstruct


class TestEvolInstruct:
    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.complete = AsyncMock(
            return_value={"content": "Evolved instruction with more depth and complexity"}
        )
        return llm

    @pytest.mark.asyncio
    async def test_single_round(self, mock_llm):
        evol = EvolInstruct(rounds=1, strategies=["deepen"], seed=42, llm=mock_llm)
        samples = [Sample(instruction="What is Python?")]
        result = await evol.transform(samples)
        assert result[0].instruction == "Evolved instruction with more depth and complexity"
        assert "deepen" in result[0].evolution_history

    @pytest.mark.asyncio
    async def test_multi_round(self, mock_llm):
        call_count = 0

        async def mock_complete(**kwargs):
            nonlocal call_count
            call_count += 1
            return {"content": f"Evolution round {call_count}"}

        mock_llm.complete = mock_complete
        evol = EvolInstruct(rounds=3, strategies=["deepen"], seed=42, llm=mock_llm)
        samples = [Sample(instruction="Original")]
        result = await evol.transform(samples)
        assert len(result[0].evolution_history) == 3

    @pytest.mark.asyncio
    async def test_strategy_rotation(self, mock_llm):
        evol = EvolInstruct(
            rounds=1,
            strategies=["deepen", "broaden", "rephrase"],
            seed=42,
            llm=mock_llm,
        )
        samples = [
            Sample(instruction="Q1"),
            Sample(instruction="Q2"),
            Sample(instruction="Q3"),
        ]
        result = await evol.transform(samples)
        # Each sample gets a strategy from the seeded RNG
        strategies_used = [s.evolution_history[0] for s in result]
        assert all(s in ["deepen", "broaden", "rephrase"] for s in strategies_used)

    @pytest.mark.asyncio
    async def test_skips_empty_instruction(self, mock_llm):
        evol = EvolInstruct(rounds=1, llm=mock_llm)
        samples = [Sample(text="Some pretraining text")]
        result = await evol.transform(samples)
        assert result[0].text == "Some pretraining text"
        assert len(result[0].evolution_history) == 0

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            EvolInstruct(strategies=["nonexistent"])

    @pytest.mark.asyncio
    async def test_handles_llm_failure(self, mock_llm):
        mock_llm.complete = AsyncMock(side_effect=Exception("LLM error"))
        evol = EvolInstruct(rounds=1, strategies=["deepen"], llm=mock_llm)
        samples = [Sample(instruction="Original instruction")]
        result = await evol.transform(samples)
        # Should keep original on failure
        assert result[0].instruction == "Original instruction"
