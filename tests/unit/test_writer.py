"""Tests for JSONLWriter."""

import json

from syntheta.pipeline.writer import JSONLWriter
from syntheta.schema.sample import Sample


class TestJSONLWriter:
    def test_write_batch(self, tmp_path):
        writer = JSONLWriter(tmp_path / "ckpt")
        samples = [Sample(instruction=f"Q{i}") for i in range(3)]
        path = writer.write_batch(samples, batch_num=1)
        assert path.exists()
        assert path.name == "batch_0001.jsonl"
        with path.open() as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 3

    def test_consolidate(self, tmp_path):
        writer = JSONLWriter(tmp_path / "ckpt")
        samples1 = [Sample(instruction="Q1"), Sample(instruction="Q2")]
        samples2 = [Sample(instruction="Q3")]
        writer.write_batch(samples1, 1)
        writer.write_batch(samples2, 2)

        output = tmp_path / "final.jsonl"
        total = writer.consolidate(output)
        assert total == 3
        with output.open() as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 3

    def test_consolidate_preserves_order(self, tmp_path):
        writer = JSONLWriter(tmp_path / "ckpt")
        writer.write_batch([Sample(instruction="first")], 1)
        writer.write_batch([Sample(instruction="second")], 2)

        output = tmp_path / "out.jsonl"
        writer.consolidate(output)
        with output.open() as f:
            records = [json.loads(l) for l in f if l.strip()]
        assert records[0]["instruction"] == "first"
        assert records[1]["instruction"] == "second"
