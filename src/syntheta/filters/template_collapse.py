"""TemplateCollapseDetector: heuristic detection of repetitive output prefixes."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from syntheta.pipeline.base import BaseFilter
from syntheta.pipeline.registry import register_filter
from syntheta.schema.sample import Sample

logger = logging.getLogger("syntheta.filters.template_collapse")


@register_filter("template_collapse")
class TemplateCollapseDetector(BaseFilter):
    """Detects repetitive output prefixes indicating the LLM is stuck in a template.

    Pure heuristic — no LLM call needed.

    Inspired by FinePhrase / Synthetic Data Playbook (2026).
    """

    rejection_reason = "template_collapse"

    def __init__(
        self,
        prefix_length: int = 50,
        max_repeat: int = 3,
        action: str = "report",
        **kwargs: Any,
    ) -> None:
        """
        Args:
            prefix_length: Number of characters to compare as prefix.
            max_repeat: Maximum allowed repetitions of any prefix.
            action: "flag" (set metadata), "remove" (drop samples), "report" (log only).
        """
        super().__init__(**kwargs)
        self.prefix_length = prefix_length
        self.max_repeat = max_repeat
        self.action = action

    async def filter(self, samples: list[Sample]) -> list[Sample]:
        """Detect and handle template collapse in sample responses/text."""
        if not samples:
            return samples

        # Extract prefixes
        prefixes = []
        for sample in samples:
            text = sample.response or sample.text or ""
            prefix = text[: self.prefix_length].strip().lower()
            prefixes.append(prefix)

        # Count prefix frequencies
        prefix_counts = Counter(prefixes)

        # Find collapsing prefixes
        collapsing = {p for p, count in prefix_counts.items() if count > self.max_repeat and p}

        if not collapsing:
            return samples

        collapse_count = sum(1 for p in prefixes if p in collapsing)
        logger.info(
            "Template collapse detected: %d samples share %d repeated prefixes",
            collapse_count,
            len(collapsing),
        )

        if self.action == "remove":
            # Keep only samples whose prefix is not collapsing
            result = []
            seen_counts: dict[str, int] = {}
            for sample, prefix in zip(samples, prefixes):
                if prefix in collapsing:
                    seen_counts[prefix] = seen_counts.get(prefix, 0) + 1
                    if seen_counts[prefix] <= self.max_repeat:
                        result.append(sample)
                else:
                    result.append(sample)
            return result

        elif self.action == "flag":
            # Set metadata flag but keep all samples
            for sample, prefix in zip(samples, prefixes):
                if prefix in collapsing:
                    sample.extra["template_collapse"] = True
            return samples

        else:  # "report" — log only, keep all
            return samples
