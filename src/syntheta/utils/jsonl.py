"""JSONL read/write helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Lazily read a JSONL file, yielding one dict per line.

    Skips blank lines. Raises on malformed JSON with the line number.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON at {path}:{line_num}: {e}") from e


def write_jsonl(
    path: str | Path,
    records: Iterable[dict[str, Any]],
    mode: str = "a",
) -> int:
    """Write records to a JSONL file. Returns number of records written.

    Args:
        path: Output file path.
        records: Iterable of dicts to serialize.
        mode: File open mode ('a' for append, 'w' for overwrite).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
