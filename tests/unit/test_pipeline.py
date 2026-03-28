"""Tests for the Pipeline engine with dummy components."""

from __future__ import annotations

from collections.abc import AsyncIterator

from syntheta.pipeline.base import BaseFilter, BaseGenerator, BaseTransformer
from syntheta.pipeline.pipeline import Pipeline
from syntheta.schema.sample import Sample


class DummyGenerator(BaseGenerator):
    """Generates samples with sequential instructions."""

    async def generate(self, n: int) -> AsyncIterator[list[Sample]]:
        for i in range(n):
            yield [
                Sample(
                    instruction=f"Question {i}",
                    response=f"Answer {i} with enough detail to be useful.",
                    domain="test",
                )
            ]


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

            over_generate_factor=1.0,
            show_progress=False,
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

            over_generate_factor=1.0,
            show_progress=False,
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

            over_generate_factor=2.0,
            show_progress=False,
        )
        ds = pipe.run(
            n=5,
            output=tmp_path / "out.jsonl",
            checkpoint_path=tmp_path / "ckpt",
        )
        assert len(ds) == 5
        assert pipe.filter_summary.total_rejected > 0
        assert "odd_index" in pipe.filter_summary.rejections

    def test_pipeline_creates_checkpoint(self, tmp_path):
        pipe = Pipeline(
            generator=DummyGenerator(),

            over_generate_factor=1.0,
            show_progress=False,
        )
        pipe.run(
            n=10,
            output=tmp_path / "out.jsonl",
            checkpoint_path=tmp_path / "ckpt",
        )
        assert (tmp_path / "ckpt" / "state.json").exists()

    def test_pipeline_exact_count(self, tmp_path):
        """Pipeline should produce exactly n samples, not more."""
        pipe = Pipeline(
            generator=DummyGenerator(),

            over_generate_factor=1.5,
            show_progress=False,
        )
        ds = pipe.run(
            n=10,
            output=tmp_path / "out.jsonl",
            checkpoint_path=tmp_path / "ckpt",
        )
        assert len(ds) == 10

    def test_pipeline_over_generation_with_filter(self, tmp_path):
        """Over-generation compensates for filter losses."""
        pipe = Pipeline(
            generator=DummyGenerator(),
            filters=[EvenFilter()],

            over_generate_factor=2.0,
            show_progress=False,
        )
        ds = pipe.run(
            n=5,
            output=tmp_path / "out.jsonl",
            checkpoint_path=tmp_path / "ckpt",
        )
        assert len(ds) == 5
        assert pipe.filter_summary.total_rejected > 0
