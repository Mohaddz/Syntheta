"""Tests for PersonaGenerator."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from syntheta.generators.persona import PersonaGenerator


class TestPersonaGenerator:
    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        personas = [
            {"name": "Dr. Smith", "background": "professor of biology", "perspective": "academic"},
            {"name": "Jane", "background": "high school student", "perspective": "beginner"},
        ]
        questions = ["What is DNA?", "How do cells divide?"]

        llm.complete = AsyncMock(
            side_effect=[
                {"content": json.dumps(personas)},  # persona generation
                {"content": json.dumps(questions)},  # questions for persona 1
                {"content": json.dumps(questions)},  # questions for persona 2
            ]
        )
        return llm

    @pytest.mark.asyncio
    async def test_generates_samples(self, mock_llm):
        gen = PersonaGenerator(
            domain="Biology",
            n_personas=2,
            questions_per_persona=2,
            llm=mock_llm,
        )
        batches = []
        async for batch in gen.generate(n=4):
            batches.append(batch)

        all_samples = [s for b in batches for s in b]
        assert len(all_samples) == 4
        assert all(s.domain == "Biology" for s in all_samples)
        assert all(s.persona for s in all_samples)

    @pytest.mark.asyncio
    async def test_respects_n_limit(self, mock_llm):
        gen = PersonaGenerator(domain="Test", n_personas=2, questions_per_persona=5, llm=mock_llm)
        all_samples = []
        async for batch in gen.generate(n=2):
            all_samples.extend(batch)
        assert len(all_samples) == 2
