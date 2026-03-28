"""Tests for SeedDatasetGenerator."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from syntheta.generators.seed_dataset import SeedDatasetGenerator


class TestSeedDatasetGenerator:
    @pytest.fixture
    def seed_file(self, tmp_path):
        path = tmp_path / "seed.jsonl"
        records = [
            {"instruction": "What is Python?", "response": "A programming language"},
            {"instruction": "Explain OOP", "response": "Object-oriented programming..."},
            {"instruction": "What is a function?", "response": "A reusable block of code"},
        ]
        with path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return str(path)

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.complete = AsyncMock(
            return_value={"content": '["What is a class?", "Explain inheritance"]'}
        )
        return llm

    @pytest.mark.asyncio
    async def test_generates_from_seed(self, mock_llm, seed_file):
        gen = SeedDatasetGenerator(source=seed_file, n_few_shot=2, seed=42, llm=mock_llm)
        all_samples = []
        async for batch in gen.generate(n=4):
            all_samples.extend(batch)
        assert len(all_samples) == 4
        assert all(s.instruction for s in all_samples)

    @pytest.mark.asyncio
    async def test_source_id_set(self, mock_llm, seed_file):
        gen = SeedDatasetGenerator(source=seed_file, llm=mock_llm)
        all_samples = []
        async for batch in gen.generate(n=2):
            all_samples.extend(batch)
        assert all(s.source_id == "seed_dataset" for s in all_samples)
