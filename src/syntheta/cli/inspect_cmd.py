"""CLI inspect command."""

from __future__ import annotations

from collections import Counter

import click

from syntheta.schema.dataset import SynthDataset


@click.command("inspect")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--json", "json_output", is_flag=True, help="Machine-readable JSON output")
@click.option("--samples", type=int, default=3, help="Number of preview samples")
def inspect_cmd(file_path: str, json_output: bool, samples: int) -> None:
    """Inspect a dataset file and show statistics."""
    from pathlib import Path

    ds = SynthDataset.from_jsonl(file_path)
    path = Path(file_path)

    if json_output:
        _json_output(ds, path)
        return

    click.echo(f"\nDataset: {path.name}")
    click.echo("=" * 50)
    click.echo("\nBasics:")
    click.echo(f"  Samples:    {len(ds):,}")
    click.echo(f"  File size:  {path.stat().st_size / 1024 / 1024:.1f} MB")

    # Field population
    fields = [
        "instruction",
        "response",
        "system_prompt",
        "turns",
        "text",
        "quality_score",
        "domain",
        "topic",
        "language",
        "difficulty",
    ]
    click.echo("\nField population:")
    for field in fields:
        count = sum(1 for s in ds if getattr(s, field) is not None)
        pct = count / len(ds) * 100 if ds else 0
        click.echo(f"  {field + ':':<18} {count:>6} / {len(ds):>6}  ({pct:.1f}%)")

    # Distributions
    click.echo("\nDistributions:")
    for field in ["language", "task_type", "difficulty"]:
        values = [getattr(s, field) for s in ds if getattr(s, field) is not None]
        if values:
            counter = Counter(values)
            dist_str = " | ".join(f"{k}: {v}" for k, v in counter.most_common(5))
            click.echo(f"  {field + ':':<14} {dist_str}")

    # Quality stats
    scores = [s.quality_score for s in ds if s.quality_score is not None]
    if scores:
        click.echo("\nQuality:")
        sorted_scores = sorted(scores)
        p10_idx = max(0, int(len(sorted_scores) * 0.1) - 1)
        p90_idx = min(len(sorted_scores) - 1, int(len(sorted_scores) * 0.9))
        click.echo(
            f"  Mean: {sum(scores) / len(scores):.2f} | "
            f"Median: {sorted_scores[len(sorted_scores) // 2]:.2f} | "
            f"p10: {sorted_scores[p10_idx]:.2f} | "
            f"p90: {sorted_scores[p90_idx]:.2f}"
        )

    # Preview
    click.echo(f"\nSample preview (first {min(samples, len(ds))}):")
    for i, s in enumerate(ds.samples[:samples]):
        text = s.instruction or s.text or "(no text)"
        click.echo(f"  [{i + 1}] {text[:80]}{'...' if len(text) > 80 else ''}")


def _json_output(ds: SynthDataset, path) -> None:
    """Output machine-readable JSON stats."""
    import json as json_mod
    from collections import Counter

    stats = {
        "samples": len(ds),
        "file_size_bytes": path.stat().st_size,
        "fields": {},
        "distributions": {},
    }

    fields = ["instruction", "response", "quality_score", "domain", "language"]
    for field in fields:
        count = sum(1 for s in ds if getattr(s, field) is not None)
        stats["fields"][field] = count

    for field in ["language", "task_type", "difficulty"]:
        values = [getattr(s, field) for s in ds if getattr(s, field) is not None]
        if values:
            stats["distributions"][field] = dict(Counter(values))

    click.echo(json_mod.dumps(stats, indent=2))
