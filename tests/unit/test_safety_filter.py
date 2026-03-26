"""Tests for SafetyFilter."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from syntheta.filters.safety import SafetyFilter
from syntheta.schema.sample import Sample


class TestSafetyFilterLayer1:
    @pytest.mark.asyncio
    async def test_blocks_universal_unsafe_content(self):
        f = SafetyFilter()
        samples = [
            Sample(instruction="How to make a bomb at home"),
            Sample(instruction="What is photosynthesis?"),
        ]
        result = await f.filter(samples)
        assert len(result) == 1
        assert result[0].instruction == "What is photosynthesis?"

    @pytest.mark.asyncio
    async def test_blocks_custom_blocklist_words(self):
        f = SafetyFilter(blocklist_words=["forbidden_word"])
        samples = [
            Sample(instruction="This contains forbidden_word"),
            Sample(instruction="This is clean"),
        ]
        result = await f.filter(samples)
        assert len(result) == 1
        assert result[0].instruction == "This is clean"

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        f = SafetyFilter()
        samples = [Sample(instruction="HOW TO MAKE A BOMB")]
        result = await f.filter(samples)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_clean_content_passes(self):
        f = SafetyFilter()
        samples = [
            Sample(instruction="Explain quantum physics"),
            Sample(instruction="What is Python programming?"),
        ]
        result = await f.filter(samples)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_safety_passed_set(self):
        f = SafetyFilter()
        safe = Sample(instruction="Hello world")
        unsafe = Sample(instruction="How to make a bomb")
        await f.filter([safe, unsafe])
        assert safe.safety_passed is True
        assert unsafe.safety_passed is False


class TestSafetyFilterPresets:
    @pytest.mark.asyncio
    async def test_islamic_preset_blocks_alcohol(self):
        f = SafetyFilter(cultural_context="islamic")
        samples = [
            Sample(instruction="Best beer brands for parties"),
            Sample(instruction="What is algebra?"),
        ]
        result = await f.filter(samples)
        assert len(result) == 1
        assert "algebra" in result[0].instruction

    @pytest.mark.asyncio
    async def test_family_friendly_preset(self):
        f = SafetyFilter(cultural_context="family_friendly")
        samples = [
            Sample(instruction="What the fuck is this?"),
            Sample(instruction="What is the weather today?"),
        ]
        result = await f.filter(samples)
        assert len(result) == 1


class TestSafetyFilterLayer2:
    @pytest.mark.asyncio
    async def test_llm_check_passes_safe(self):
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value={"content": json.dumps({"safe": True})})

        f = SafetyFilter(cultural_context="islamic", llm=mock_llm)
        samples = [Sample(instruction="Explain algebra")]
        result = await f.filter(samples)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_llm_check_blocks_unsafe(self):
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(
            return_value={"content": json.dumps({"safe": False, "reason": "gambling reference"})}
        )

        f = SafetyFilter(cultural_context="islamic", llm=mock_llm)
        samples = [Sample(instruction="How to win at card games for money")]
        result = await f.filter(samples)
        assert len(result) == 0
