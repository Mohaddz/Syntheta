"""Dataset health report generation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from syntheta.schema.dataset import SynthDataset


class DatasetHealthReport:
    """Generates a health report summarizing dataset quality and coverage."""

    def __init__(self, dataset: SynthDataset, target_topics: list[str] | None = None) -> None:
        self._samples = dataset.samples
        self._target_topics = target_topics or []

    @classmethod
    def from_dataset(cls, dataset: SynthDataset, **kwargs) -> DatasetHealthReport:
        return cls(dataset, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Compute all metrics and return as dict."""
        samples = self._samples
        n = len(samples)

        # Coverage
        topics = Counter(s.topic for s in samples if s.topic)
        task_types = Counter(s.task_type for s in samples if s.task_type)
        languages = Counter(s.language for s in samples if s.language)
        difficulties = Counter(s.difficulty for s in samples if s.difficulty is not None)

        # Quality
        scores = [s.quality_score for s in samples if s.quality_score is not None]
        sorted_scores = sorted(scores) if scores else []
        quality = {}
        if sorted_scores:
            quality = {
                "mean": round(sum(sorted_scores) / len(sorted_scores), 3),
                "median": round(sorted_scores[len(sorted_scores) // 2], 3),
                "p10": round(sorted_scores[max(0, int(len(sorted_scores) * 0.1) - 1)], 3),
                "p90": round(
                    sorted_scores[min(len(sorted_scores) - 1, int(len(sorted_scores) * 0.9))], 3
                ),
            }

        # Safety
        safety_blocked = sum(1 for s in samples if s.safety_passed is False)

        # Template collapse
        collapse_flagged = sum(1 for s in samples if s.extra.get("template_collapse"))

        # Warnings
        warnings = []
        if self._target_topics:
            found = set(topics.keys())
            missing = set(self._target_topics) - found
            if missing:
                warnings.append(f"{len(missing)} target topics have zero coverage")
        if difficulties:
            total_d = sum(difficulties.values())
            easy_pct = sum(difficulties.get(d, 0) for d in [1, 2]) / total_d if total_d else 0
            if easy_pct > 0.7:
                warnings.append(
                    f"Difficulty distribution heavily skewed toward easy "
                    f"({easy_pct * 100:.0f}% at levels 1-2)"
                )

        return {
            "total_samples": n,
            "coverage": {
                "topics": {"found": len(topics), "distribution": dict(topics.most_common(10))},
                "task_types": dict(task_types),
                "languages": dict(languages),
                "difficulty_distribution": {str(k): v for k, v in sorted(difficulties.items())},
            },
            "quality": quality,
            "safety": {"blocked": safety_blocked},
            "diversity": {"template_collapse_flagged": collapse_flagged},
            "warnings": warnings,
        }

    def display(self) -> str:
        """Format a human-readable health report."""
        data = self.to_dict()
        lines = [
            "Dataset Health Report",
            "=" * 50,
            f"\nSamples: {data['total_samples']:,}",
        ]

        # Coverage
        lines.append("\nCoverage:")
        cov = data["coverage"]
        lines.append(f"  Topics:      {cov['topics']['found']} unique")
        if cov["task_types"]:
            tt_str = ", ".join(f"{k}: {v}" for k, v in cov["task_types"].items())
            lines.append(f"  Task types:  {tt_str}")
        if cov["languages"]:
            lang_str = ", ".join(f"{k}: {v}" for k, v in cov["languages"].items())
            lines.append(f"  Languages:   {lang_str}")
        if cov["difficulty_distribution"]:
            diff_str = ", ".join(f"{k}: {v}" for k, v in cov["difficulty_distribution"].items())
            lines.append(f"  Difficulty:  {diff_str}")

        # Quality
        if data["quality"]:
            q = data["quality"]
            lines.append(
                f"\nQuality:\n  Mean: {q['mean']:.2f} | Median: {q['median']:.2f} | "
                f"p10: {q['p10']:.2f} | p90: {q['p90']:.2f}"
            )

        # Warnings
        if data["warnings"]:
            lines.append("\nWarnings:")
            for w in data["warnings"]:
                lines.append(f"  ⚠ {w}")

        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Save the report as JSON."""
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
