"""Filter summary: tracks rejection counts by reason."""

from __future__ import annotations

from collections import defaultdict


class FilterSummary:
    """Accumulates filter rejection stats and formats a summary report."""

    def __init__(self) -> None:
        self.total_generated = 0
        self.total_passed = 0
        self._rejections: dict[str, int] = defaultdict(int)
        self._warnings: list[str] = []
        self.rate_limit_count = 0
        self.retry_count = 0

    def record_generated(self, n: int) -> None:
        """Record n samples generated (pre-filter)."""
        self.total_generated += n

    def record_passed(self, n: int) -> None:
        """Record n samples that passed all filters."""
        self.total_passed += n

    def record_rejection(self, reason: str, n: int = 1) -> None:
        """Record n samples rejected for a given reason."""
        self._rejections[reason] += n

    def record_warning(self, msg: str) -> None:
        """Record a warning message."""
        self._warnings.append(msg)

    @property
    def total_rejected(self) -> int:
        return sum(self._rejections.values())

    @property
    def rejections(self) -> dict[str, int]:
        return dict(self._rejections)

    def display(self) -> str:
        """Format a human-readable filter summary."""
        lines = ["Filter summary:"]
        lines.append(f"  Input:     {self.total_generated:,} samples generated")
        pct = (self.total_passed / self.total_generated * 100) if self.total_generated else 0
        lines.append(f"  Output:    {self.total_passed:,} samples passed ({pct:.1f}%)")
        lines.append("")

        if self._rejections:
            rej_pct = (
                (self.total_rejected / self.total_generated * 100) if self.total_generated else 0
            )
            lines.append(f"  Rejected:  {self.total_rejected:,} samples ({rej_pct:.1f}%)")
            for reason, count in sorted(self._rejections.items(), key=lambda x: -x[1]):
                r_pct = (count / self.total_generated * 100) if self.total_generated else 0
                lines.append(
                    f"    {reason}:{' ' * max(1, 30 - len(reason))}{count:>5}  ({r_pct:.1f}%)"
                )

        if self._warnings:
            lines.append("")
            lines.append("  Warnings:")
            for w in self._warnings:
                lines.append(f"    {w}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for JSON output."""
        return {
            "total_generated": self.total_generated,
            "total_passed": self.total_passed,
            "total_rejected": self.total_rejected,
            "rejections": dict(self._rejections),
            "warnings": self._warnings,
        }
