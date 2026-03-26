"""Tests for QualityFilter."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from syntheta.filters.quality import QualityFilter, _parse_quality_response
from syntheta.schema.sample import Sample


class TestParseQualityResponse:
    def test_valid_json(self):
        score, reason = _parse_quality_response('{"score": 0.85, "reason": "good quality"}')
        assert score == 0.85
        assert reason == "good quality"

    def test_json_with_markdown(self):
        score, reason = _parse_quality_response('```json\n{"score": 0.9, "reason": "ok"}\n```')
        assert score == 0.9

    def test_score_clamped(self):
        score, _ = _parse_quality_response('{"score": 1.5}')
        assert score == 1.0

    def test_fallback_number_extraction(self):
        score, _ = _parse_quality_response("The quality score is 4 out of 5")
        assert score == 0.8  # 4/5

    def test_fallback_decimal(self):
        score, _ = _parse_quality_response("Score: 0.72")
        assert score == 0.72

    def test_unparseable_defaults(self):
        score, _ = _parse_quality_response("This is not a score at all with no numbers")
        assert score == 0.5


class TestQualityFilter:
    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.complete = AsyncMock()
        return llm

    @pytest.mark.asyncio
    async def test_passes_high_quality(self, mock_llm):
        mock_llm.complete = AsyncMock(
            return_value={"content": '{"score": 0.9, "reason": "excellent"}'}
        )
        f = QualityFilter(min_score=0.7, llm=mock_llm)
        samples = [Sample(instruction="What is X?", response="X is a well-explained concept.")]
        result = await f.filter(samples)
        assert len(result) == 1
        assert result[0].quality_score == 0.9

    @pytest.mark.asyncio
    async def test_rejects_low_quality(self, mock_llm):
        mock_llm.complete = AsyncMock(
            return_value={"content": '{"score": 0.3, "reason": "poor quality"}'}
        )
        f = QualityFilter(min_score=0.7, llm=mock_llm)
        samples = [Sample(instruction="What?", response="Dunno.")]
        result = await f.filter(samples)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_passes_through_without_llm(self):
        f = QualityFilter(min_score=0.7)
        samples = [Sample(instruction="Test", response="Answer.")]
        result = await f.filter(samples)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_non_scoreable_passes_through(self, mock_llm):
        f = QualityFilter(min_score=0.7, llm=mock_llm)
        samples = [Sample(text="Just text, no instruction/response pair")]
        result = await f.filter(samples)
        assert len(result) == 1
