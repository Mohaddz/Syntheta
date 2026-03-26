"""JSONL writer for streaming pipeline output."""

from __future__ import annotations

import logging
from pathlib import Path

from syntheta.schema.sample import Sample
from syntheta.utils.jsonl import write_jsonl

logger = logging.getLogger("syntheta.pipeline")


class JSONLWriter:
    """Writes samples to JSONL files in batches, then consolidates."""

    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

    def write_batch(self, samples: list[Sample], batch_num: int) -> Path:
        """Write a batch of samples to a numbered JSONL file.

        Returns the path to the batch file.
        """
        batch_file = self.checkpoint_path / f"batch_{batch_num:04d}.jsonl"
        records = [s.model_dump(mode="json") for s in samples]
        write_jsonl(batch_file, records, mode="w")
        logger.debug("Wrote %d samples to %s", len(samples), batch_file)
        return batch_file

    def consolidate(self, output_path: str | Path) -> int:
        """Concatenate all batch files into the final output file.

        Returns the total number of records written.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        batch_files = sorted(self.checkpoint_path.glob("batch_*.jsonl"))
        total = 0
        with output_path.open("w", encoding="utf-8") as out:
            for batch_file in batch_files:
                with batch_file.open("r", encoding="utf-8") as bf:
                    for line in bf:
                        line = line.strip()
                        if line:
                            out.write(line + "\n")
                            total += 1

        logger.info(
            "Consolidated %d samples from %d batches to %s", total, len(batch_files), output_path
        )
        return total
