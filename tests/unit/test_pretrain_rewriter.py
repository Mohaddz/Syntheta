"""Tests for PretrainRewriter."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from syntheta.pretrain.rewriter import PretrainRewriter
from syntheta.schema.sample import Sample


class TestPretrainRewriter:
    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.complete = AsyncMock(
            return_value={"content": "Q: What is X?\nA: X is a concept that..."}
        )
        return llm

    @pytest.mark.asyncio
    async def test_rewrites_text(self, mock_llm):
        rewriter = PretrainRewriter(strategies=["faq"], llm=mock_llm)
        samples = [Sample(text="Python is a programming language used for many purposes.")]
        result = await rewriter.transform(samples)
        assert result[0].text.startswith("Q:")
        assert (
            result[0].extra["original_text"]
            == "Python is a programming language used for many purposes."
        )
        assert result[0].extra["pretrain_strategy"] == "faq"

    @pytest.mark.asyncio
    async def test_round_robin_strategies(self, mock_llm):
        call_count = 0

        async def mock_complete(**kwargs):
            nonlocal call_count
            call_count += 1
            return {"content": f"Rewritten {call_count}"}

        mock_llm.complete = mock_complete
        rewriter = PretrainRewriter(strategies=["faq", "math", "table"], llm=mock_llm)
        samples = [
            Sample(text="Text A - long enough for processing"),
            Sample(text="Text B - long enough for processing"),
            Sample(text="Text C - long enough for processing"),
        ]
        result = await rewriter.transform(samples)
        assert result[0].extra["pretrain_strategy"] == "faq"
        assert result[1].extra["pretrain_strategy"] == "math"
        assert result[2].extra["pretrain_strategy"] == "table"

    @pytest.mark.asyncio
    async def test_skips_non_text_samples(self, mock_llm):
        rewriter = PretrainRewriter(llm=mock_llm)
        samples = [Sample(instruction="This has no text field")]
        result = await rewriter.transform(samples)
        assert result[0].instruction == "This has no text field"
        assert "original_text" not in result[0].extra

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown pretrain strategy"):
            PretrainRewriter(strategies=["invalid"])

    @pytest.mark.asyncio
    async def test_handles_llm_failure(self, mock_llm):
        mock_llm.complete = AsyncMock(side_effect=Exception("LLM error"))
        rewriter = PretrainRewriter(strategies=["faq"], llm=mock_llm)
        samples = [Sample(text="Original text that should be preserved.")]
        result = await rewriter.transform(samples)
        # On failure, original text is preserved
        assert result[0].text == "Original text that should be preserved."
