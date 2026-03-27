"""Tests for JSONLWriter."""

import asyncio
import json

from syntheta.pipeline.writer import JSONLWriter
from syntheta.schema.sample import Sample


class TestJSONLWriter:
    def test_write_sample(self, tmp_path):
        writer = JSONLWriter(tmp_path / "out.jsonl")

        async def _write():
            await writer.write_sample(Sample(instruction="Q1"))
            await writer.write_sample(Sample(instruction="Q2"))

        asyncio.run(_write())

        assert writer.count == 2
        with (tmp_path / "out.jsonl").open() as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2
        assert json.loads(lines[0])["instruction"] == "Q1"

    def test_concurrent_writes(self, tmp_path):
        writer = JSONLWriter(tmp_path / "out.jsonl")

        async def _write():
            tasks = [writer.write_sample(Sample(instruction=f"Q{i}")) for i in range(20)]
            await asyncio.gather(*tasks)

        asyncio.run(_write())

        assert writer.count == 20
        with (tmp_path / "out.jsonl").open() as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 20

    def test_preserves_content(self, tmp_path):
        writer = JSONLWriter(tmp_path / "out.jsonl")
        sample = Sample(
            instruction="Test question",
            response="Test answer",
            domain="test",
            quality_score=0.9,
        )

        asyncio.run(writer.write_sample(sample))

        with (tmp_path / "out.jsonl").open() as f:
            record = json.loads(f.readline())
        assert record["instruction"] == "Test question"
        assert record["response"] == "Test answer"
        assert record["quality_score"] == 0.9

    def test_resume_appends(self, tmp_path):
        path = tmp_path / "out.jsonl"

        # Write 3 samples initially
        writer1 = JSONLWriter(path)
        asyncio.run(writer1.write_sample(Sample(instruction="Q1")))
        asyncio.run(writer1.write_sample(Sample(instruction="Q2")))
        asyncio.run(writer1.write_sample(Sample(instruction="Q3")))
        assert writer1.count == 3

        # Resume -- should count existing and append
        writer2 = JSONLWriter(path, resume=True)
        assert writer2.count == 3
        asyncio.run(writer2.write_sample(Sample(instruction="Q4")))
        assert writer2.count == 4

        with path.open() as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 4

    def test_fresh_start_truncates(self, tmp_path):
        path = tmp_path / "out.jsonl"

        # Write some data
        writer1 = JSONLWriter(path)
        asyncio.run(writer1.write_sample(Sample(instruction="old")))

        # Fresh start -- should truncate
        writer2 = JSONLWriter(path, resume=False)
        assert writer2.count == 0
        asyncio.run(writer2.write_sample(Sample(instruction="new")))

        with path.open() as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0])["instruction"] == "new"
