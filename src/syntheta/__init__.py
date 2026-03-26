"""Syntheta — open-source synthetic LLM training data generation."""

from syntheta._version import __version__
from syntheta.schema.column_mapper import ColumnMapper
from syntheta.schema.dataset import SynthDataset
from syntheta.schema.sample import Sample, Turn
from syntheta.utils.jsonl import read_jsonl

__all__ = [
    "ColumnMapper",
    "Sample",
    "SynthDataset",
    "Turn",
    "__version__",
    "inspect",
    "load",
    "read_jsonl",
    "validate",
]


def load(
    path: str,
    column_map: dict[str, str] | None = None,
) -> SynthDataset:
    """Load a dataset from a JSONL file into a SynthDataset.

    Args:
        path: Path to a JSONL file.
        column_map: Optional column mapping overrides.
    """
    records = list(read_jsonl(path))
    mapper = ColumnMapper(column_map=column_map)
    samples = mapper.map(records)
    return SynthDataset(samples)


def inspect(path: str) -> dict:
    """Inspect a dataset and return statistics.

    Args:
        path: Path to a JSONL file.

    Returns:
        Dict with sample count, field population, and distributions.
    """
    from collections import Counter

    ds = SynthDataset.from_jsonl(path)
    fields = {}
    for field in ["instruction", "response", "quality_score", "domain", "language"]:
        fields[field] = sum(1 for s in ds if getattr(s, field) is not None)

    distributions = {}
    for field in ["language", "task_type", "difficulty"]:
        values = [getattr(s, field) for s in ds if getattr(s, field) is not None]
        if values:
            distributions[field] = dict(Counter(values))

    return {
        "samples": len(ds),
        "fields": fields,
        "distributions": distributions,
    }


def validate(path: str, format: str) -> dict:
    """Validate a dataset against format requirements.

    Args:
        path: Path to a JSONL file.
        format: Expected format (sft, chat, dpo, rlvr, pretrain).

    Returns:
        Dict with 'passed', 'failed', and 'errors' keys.
    """
    from syntheta.cli.validate import _FORMAT_CHECKS

    ds = SynthDataset.from_jsonl(path)
    check_fn = _FORMAT_CHECKS.get(format)
    if check_fn is None:
        raise ValueError(f"Unknown format '{format}'. Choose from: {list(_FORMAT_CHECKS)}")

    passed = 0
    errors = []
    for i, sample in enumerate(ds):
        ok, reason = check_fn(sample)
        if ok:
            passed += 1
        else:
            errors.append({"row": i, "id": sample.id, "reason": reason})

    return {
        "total": len(ds),
        "passed": passed,
        "failed": len(errors),
        "errors": errors[:20],
    }
