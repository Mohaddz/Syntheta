"""Tests for ResponseGenerator with mocked LLM."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from syntheta.schema.sample import Sample
from syntheta.transformers.response import ResponseGenerator


class TestResponseGenerator:
    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.complete_batch = AsyncMock(
            return_value=[
                {"content": "Python is a programming language.", "model": "test"},
                {"content": "Java is also a programming language.", "model": "test"},
            ]
        )
        return llm

    @pytest.mark.asyncio
    async def test_generates_responses(self, mock_llm):
        gen = ResponseGenerator(llm=mock_llm)
        samples = [
            Sample(instruction="What is Python?"),
            Sample(instruction="What is Java?"),
        ]

        result = await gen.transform(samples)
        assert len(result) == 2
        assert result[0].response == "Python is a programming language."
        assert result[1].response == "Java is also a programming language."
        assert result[0].generation_model == "test"

    @pytest.mark.asyncio
    async def test_skips_samples_with_response(self, mock_llm):
        gen = ResponseGenerator(llm=mock_llm)
        samples = [
            Sample(instruction="Already answered", response="Existing response"),
            Sample(instruction="Needs answer"),
        ]

        mock_llm.complete_batch = AsyncMock(
            return_value=[{"content": "New response", "model": "test"}]
        )

        result = await gen.transform(samples)
        assert result[0].response == "Existing response"  # Unchanged
        assert result[1].response == "New response"

    @pytest.mark.asyncio
    async def test_cot_mode(self, mock_llm):
        gen = ResponseGenerator(use_cot=True, llm=mock_llm)
        samples = [Sample(instruction="Solve this.")]

        mock_llm.complete_batch = AsyncMock(
            return_value=[{"content": "Step 1: ... Answer: 42", "model": "test"}]
        )

        result = await gen.transform(samples)
        assert "42" in result[0].response

    @pytest.mark.asyncio
    async def test_empty_batch(self, mock_llm):
        gen = ResponseGenerator(llm=mock_llm)
        result = await gen.transform([])
        assert result == []
