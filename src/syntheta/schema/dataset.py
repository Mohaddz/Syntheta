"""SynthDataset: in-memory wrapper for analysis, stats, and export."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

from syntheta.schema.exporters import (
    export_chat,
    export_dpo,
    export_pretrain,
    export_rlvr,
    export_sft,
)
from syntheta.schema.sample import Sample
from syntheta.utils.jsonl import read_jsonl, write_jsonl

_EXPORTERS = {
    "sft": export_sft,
    "chat": export_chat,
    "dpo": export_dpo,
    "rlvr": export_rlvr,
    "pretrain": export_pretrain,
}


class SynthDataset:
    """In-memory convenience wrapper over a list of Samples.

    Not the pipeline's internal representation — this is for post-pipeline
    analysis, filtering, and export.
    """

    def __init__(self, samples: list[Sample] | None = None) -> None:
        self._samples: list[Sample] = samples or []

    @classmethod
    def from_jsonl(cls, path: str | Path) -> SynthDataset:
        """Load a SynthDataset from a JSONL file."""
        samples = [Sample(**record) for record in read_jsonl(path)]
        return cls(samples)

    def save(
        self,
        path: str | Path,
        format: str = "sft",
        rename: dict[str, str] | None = None,
    ) -> int:
        """Export and save to JSONL. Returns number of records written.

        Args:
            path: Output file path.
            format: Export format (sft, chat, dpo, rlvr, pretrain).
            rename: Optional column renaming dict.
        """
        exporter = _EXPORTERS.get(format)
        if exporter is None:
            raise ValueError(f"Unknown export format '{format}'. Choose from: {list(_EXPORTERS)}")
        records = exporter(self._samples, rename=rename)
        return write_jsonl(path, records, mode="w")

    def filter(self, predicate: Callable[[Sample], bool]) -> SynthDataset:
        """Return a new SynthDataset containing only samples matching the predicate."""
        return SynthDataset([s for s in self._samples if predicate(s)])

    @property
    def samples(self) -> list[Sample]:
        """Access the underlying sample list."""
        return self._samples

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    def __getitem__(self, index: int) -> Sample:
        return self._samples[index]

    def __repr__(self) -> str:
        return f"SynthDataset(n={len(self._samples)})"
