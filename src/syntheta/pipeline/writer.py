"""JSONL writer for streaming pipeline output."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from syntheta.schema.sample import Sample

logger = logging.getLogger("syntheta.pipeline")


class JSONLWriter:
    """Append-mode writer that writes individual samples to a JSONL file.

    Thread-safe via asyncio.Lock -- multiple concurrent tasks can write
    without corrupting the file.

    Supports resume: if resume=True, counts existing lines and appends
    instead of truncating.
    """

    def __init__(self, output_path: Path, resume: bool = False) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

        if resume and self.output_path.exists():
            # Count existing lines for resume
            self._count = sum(1 for line in self.output_path.open() if line.strip())
        else:
            # Fresh start -- truncate
            self.output_path.write_text("", encoding="utf-8")
            self._count = 0

    async def write_sample(self, sample: Sample) -> None:
        """Write a single sample to the output file. Safe for concurrent calls."""
        line = json.dumps(sample.model_dump(mode="json"), ensure_ascii=False)
        async with self._lock:
            with self.output_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._count += 1

    @property
    def count(self) -> int:
        return self._count
