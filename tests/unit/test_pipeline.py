"""Tests for the Pipeline engine with dummy components."""

from __future__ import annotations

from collections.abc import AsyncIterator

from syntheta.pipeline.base import BaseFilter, BaseGenerator, BaseTransformer
from syntheta.pipeline.pipeline import Pipeline
from syntheta.schema.sample import Sample


class DummyGenerator(BaseGenerator):
    """Generates samples with sequential instructions."""

    async def generate(self, n: int, batch_size: int = 100) -> AsyncIterator[list[Sample]]:
        generated = 0
        batch_num = 0
        while generated < n:
            batch = []
            for i in range(min(batch_size, n - generated)):
                batch.append(
                    Sample(
                        instruction=f"Question {generated + i}",
                        response=f"Answer {generated + i} with enough detail to be useful.",
                        domain="test",
                    )
                )
            generated += len(batch)
            batch_num += 1
            yield batch


class UpperTransformer(BaseTransformer):
    """Uppercases all instruction text."""

    async def transform(self, samples: list[Sample]) -> list[Sample]:
        for s in samples:
            if s.instruction:
                s.instruction = s.instruction.upper()
        return samples


class EvenFilter(BaseFilter):
    """Keeps only samples with even index in instruction."""

    rejection_reason = "odd_index"

    async def filter(self, samples: list[Sample]) -> list[Sample]:
        result = []
        for s in samples:
            if s.instruction:
                # Extract number from "Question N" or "QUESTION N"
                num_str = s.instruction.split()[-1]
                try:
                    if int(num_str) % 2 == 0:
                        result.append(s)
                except ValueError:
                    result.append(s)
        return result


class TestPipeline:
    def test_basic_pipeline(self, tmp_path):
        pipe = Pipeline(
            generator=DummyGenerator(),
            batch_size=5,
            over_generate_factor=1.0,
        )
        ds = pipe.run(
            n=10,
            output=tmp_path / "out.jsonl",
            checkpoint_path=tmp_path / "ckpt",
        )
        assert len(ds) == 10
        assert (tmp_path / "out.jsonl").exists()

    def test_pipeline_with_transformer(self, tmp_path):
        pipe = Pipeline(
            generator=DummyGenerator(),
            transformers=[UpperTransformer()],
            batch_size=5,
            over_generate_factor=1.0,
        )
        ds = pipe.run(
            n=5,
            output=tmp_path / "out.jsonl",
            checkpoint_path=tmp_path / "ckpt",
        )
        assert ds[0].instruction.startswith("QUESTION")

    def test_pipeline_with_filter(self, tmp_path):
        pipe = Pipeline(
            generator=DummyGenerator(),
            filters=[EvenFilter()],
            batch_size=10,
            over_generate_factor=2.0,  # Over-generate to get enough even samples
        )
        ds = pipe.run(
            n=5,
            output=tmp_path / "out.jsonl",
            checkpoint_path=tmp_path / "ckpt",
        )
        assert len(ds) == 5
        # Filter summary should show rejections
        assert pipe.filter_summary.total_rejected > 0
        assert "odd_index" in pipe.filter_summary.rejections

    def test_pipeline_creates_checkpoint(self, tmp_path):
        pipe = Pipeline(
            generator=DummyGenerator(),
            batch_size=5,
            over_generate_factor=1.0,
        )
        pipe.run(
            n=10,
            output=tmp_path / "out.jsonl",
            checkpoint_path=tmp_path / "ckpt",
        )
        assert (tmp_path / "ckpt" / "state.json").exists()

    def test_pipeline_over_generation_with_filter(self, tmp_path):
        """Over-generation compensates for filter losses."""
        pipe = Pipeline(
            generator=DummyGenerator(),
            filters=[EvenFilter()],
            batch_size=10,
            over_generate_factor=2.0,
        )
        ds = pipe.run(
            n=5,
            output=tmp_path / "out.jsonl",
            checkpoint_path=tmp_path / "ckpt",
        )
        assert len(ds) == 5
        # Total generated should be more than 5 due to over-generation
        assert pipe.filter_summary.total_generated >= 10
        assert pipe.filter_summary.total_rejected > 0
