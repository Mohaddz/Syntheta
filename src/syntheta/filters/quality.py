"""QualityFilter: LLM-as-judge multi-dimensional scoring."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from syntheta.pipeline.base import BaseFilter
from syntheta.pipeline.registry import register_filter
from syntheta.prompts import load_prompt, render_template
from syntheta.schema.sample import Sample

logger = logging.getLogger("syntheta.filters.quality")


@register_filter("quality")
class QualityFilter(BaseFilter):
    """Scores samples using LLM-as-judge and filters below a threshold."""

    rejection_reason = "quality_below_threshold"

    def __init__(
        self,
        min_score: float = 0.7,
        prompt_overrides: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.min_score = min_score
        self.prompt_overrides = prompt_overrides

    async def filter(self, samples: list[Sample]) -> list[Sample]:
        """Score all samples in parallel and keep those above min_score."""
        if not samples or not self.llm:
            return samples

        # Only score samples that have instruction + response
        scoreable = [s for s in samples if s.instruction and s.response]
        non_scoreable = [s for s in samples if not (s.instruction and s.response)]

        if not scoreable:
            return samples

        # Score all in parallel (semaphore controls concurrency)
        tasks = [self._score_one(sample) for sample in scoreable]
        await asyncio.gather(*tasks)

        # Apply threshold
        passed = list(non_scoreable)
        for sample in scoreable:
            if sample.quality_score is not None and sample.quality_score >= self.min_score:
                passed.append(sample)
            elif sample.quality_score is None:
                # Scoring failed, pass through
                passed.append(sample)
            else:
                logger.debug(
                    "Sample %s rejected: score=%.2f (min=%.2f), reason=%s",
                    sample.id,
                    sample.quality_score,
                    self.min_score,
                    sample.quality_reason,
                )

        return passed

    async def _score_one(self, sample: Sample) -> None:
        """Score a single sample. Sets quality_score and quality_reason on the sample."""
        template = load_prompt("filters.quality_judge", self.prompt_overrides)
        prompt = render_template(
            template,
            instruction=sample.instruction,
            response=sample.response,
        )

        try:
            result = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                stage="quality_filter",
            )
            score, reason = _parse_quality_response(result["content"])
            sample.quality_score = score
            sample.quality_reason = reason
        except Exception as e:
            logger.warning("Quality scoring failed for sample %s: %s", sample.id, e)
            # On error, leave quality_score as None (will pass through)


def _parse_quality_response(response_text: str) -> tuple[float, str]:
    """Parse the LLM quality judge response, with fallbacks for malformed output.

    Returns (score, reason) tuple.
    """
    text = response_text.strip()

    # Try JSON parsing first
    try:
        # Remove markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        data = json.loads(text)
        score = float(data.get("score", 0.5))
        reason = str(data.get("reason", ""))
        return (max(0.0, min(1.0, score)), reason)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Fallback: try to extract a number from the response
    numbers = re.findall(r"(\d+\.?\d*)", text)
    if numbers:
        score = float(numbers[0])
        # Normalize if on 1-5 or 1-10 scale
        if score > 1.0:
            if score <= 5.0:
                score = score / 5.0
            elif score <= 10.0:
                score = score / 10.0
            else:
                score = 0.5  # Can't interpret
        return (max(0.0, min(1.0, score)), "parsed from raw response")

    # Last resort: default to 0.5
    return (0.5, "could not parse quality score")
