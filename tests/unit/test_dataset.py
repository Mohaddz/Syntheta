"""Tests for SynthDataset wrapper."""

import json

import pytest

from syntheta.schema.dataset import SynthDataset
from syntheta.schema.sample import Sample


class TestSynthDataset:
    def test_from_jsonl(self, sft_jsonl):
        ds = SynthDataset.from_jsonl(sft_jsonl)
        assert len(ds) == 5
        assert ds[0].instruction == "Question 0"

    def test_save_and_reload(self, sample_sft, tmp_dir):
        ds = SynthDataset([sample_sft])
        path = tmp_dir / "out.jsonl"
        count = ds.save(path, format="sft")
        assert count == 1
        # Verify file content
        with path.open() as f:
            record = json.loads(f.readline())
        assert record["instruction"] == sample_sft.instruction

    def test_filter(self):
        samples = [
            Sample(instruction="easy", response="A sufficiently long answer.", difficulty=1),
            Sample(instruction="hard", response="Another sufficiently long answer.", difficulty=5),
        ]
        ds = SynthDataset(samples)
        hard_only = ds.filter(lambda s: s.difficulty == 5)
        assert len(hard_only) == 1
        assert hard_only[0].instruction == "hard"

    def test_len_and_iter(self, sft_jsonl):
        ds = SynthDataset.from_jsonl(sft_jsonl)
        assert len(ds) == 5
        count = sum(1 for _ in ds)
        assert count == 5

    def test_getitem(self, sft_jsonl):
        ds = SynthDataset.from_jsonl(sft_jsonl)
        assert ds[2].instruction == "Question 2"

    def test_repr(self):
        ds = SynthDataset([Sample(instruction="x")])
        assert "SynthDataset(n=1)" in repr(ds)

    def test_unknown_format_raises(self, sample_sft, tmp_dir):
        ds = SynthDataset([sample_sft])
        with pytest.raises(ValueError, match="Unknown export format"):
            ds.save(tmp_dir / "out.jsonl", format="unknown")

    def test_empty_dataset(self):
        ds = SynthDataset()
        assert len(ds) == 0
