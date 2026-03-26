"""PretrainRewriter: rewrites text into structured formats for pretraining data."""

from __future__ import annotations

import logging
from typing import Any

from syntheta.pipeline.base import BaseTransformer
from syntheta.pipeline.registry import register_transformer
from syntheta.prompts import load_prompt, render_template
from syntheta.schema.sample import Sample
from syntheta.utils.seeding import create_rng

logger = logging.getLogger("syntheta.pretrain.rewriter")

STRATEGIES = ["faq", "math", "table", "tutorial"]


@register_transformer("pretrain_rewriter")
class PretrainRewriter(BaseTransformer):
    """Rewrites text into structured formats for pretraining data.

    Papers: WRAP (2024), Phi series (2023-2024), FinePhrase / Synthetic Data Playbook (2026)

    Four strategies empirically validated by FinePhrase (90 experiments, 1T+ tokens):
    - faq: Strong on reading comprehension
    - math: Only strategy that moves GSM8K
    - table: Strongest ARC boost
    - tutorial: Only strategy that improves DROP
    """

    def __init__(
        self,
        strategies: list[str] | None = None,
        seed: int | None = None,
        prompt_overrides: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.strategies = strategies or STRATEGIES
        self.seed = seed
        self.prompt_overrides = prompt_overrides
        self._rng = create_rng(seed)
        self._strategy_idx = 0

        for s in self.strategies:
            if s not in STRATEGIES:
                raise ValueError(f"Unknown pretrain strategy '{s}'. Choose from: {STRATEGIES}")

    async def transform(self, samples: list[Sample]) -> list[Sample]:
        """Rewrite each sample's text using the configured strategies."""
        for sample in samples:
            text = sample.text
            if not text:
                continue

            # Select strategy (round-robin)
            strategy = self.strategies[self._strategy_idx % len(self.strategies)]
            self._strategy_idx += 1

            template = load_prompt(f"pretrain.{strategy}", self.prompt_overrides)
            prompt = render_template(template, text=text)

            try:
                result = await self.llm.complete(
                    messages=[{"role": "user", "content": prompt}],
                    stage="pretrain_rewriter",
                )
                rewritten = result["content"].strip()
                if rewritten:
                    # Preserve original in extra
                    sample.extra["original_text"] = text
                    sample.extra["pretrain_strategy"] = strategy
                    sample.text = rewritten
            except Exception as e:
                logger.warning(
                    "Pretrain rewrite failed for sample %s (strategy=%s): %s",
                    sample.id,
                    strategy,
                    e,
                )

        return samples
