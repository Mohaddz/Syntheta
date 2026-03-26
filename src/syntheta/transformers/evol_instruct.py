"""EvolInstruct: evolve instructions using 6 mutation strategies."""

from __future__ import annotations

import logging
from typing import Any

from syntheta.pipeline.base import BaseTransformer
from syntheta.pipeline.registry import register_transformer
from syntheta.prompts import load_prompt, render_template
from syntheta.schema.sample import Sample
from syntheta.utils.seeding import create_rng

logger = logging.getLogger("syntheta.transformers.evol_instruct")

STRATEGIES = ["deepen", "broaden", "concretize", "rephrase", "complicate", "switch_task"]


@register_transformer("evol_instruct")
class EvolInstruct(BaseTransformer):
    """Evolve instructions using mutation strategies from WizardLM/Evol-Instruct.

    Paper: WizardLM / Evol-Instruct (2023)

    Six strategies: deepen, broaden, concretize, rephrase, complicate, switch_task.
    Each has its own external prompt template.
    """

    def __init__(
        self,
        rounds: int = 1,
        strategies: list[str] | None = None,
        seed: int | None = None,
        prompt_overrides: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.rounds = rounds
        self.strategies = strategies or STRATEGIES
        self.seed = seed
        self.prompt_overrides = prompt_overrides
        self._rng = create_rng(seed)

        # Validate strategies
        for s in self.strategies:
            if s not in STRATEGIES:
                raise ValueError(f"Unknown strategy '{s}'. Choose from: {STRATEGIES}")

    async def transform(self, samples: list[Sample]) -> list[Sample]:
        """Apply evolution rounds to each sample."""
        for _round in range(self.rounds):
            for sample in samples:
                if not sample.instruction:
                    continue

                strategy = self._rng.choice(self.strategies)
                template = load_prompt(f"transformers.evol_{strategy}", self.prompt_overrides)
                prompt = render_template(template, instruction=sample.instruction)

                try:
                    result = await self.llm.complete(
                        messages=[{"role": "user", "content": prompt}],
                        stage="evol_instruct",
                    )
                    evolved = result["content"].strip()
                    if evolved:
                        sample.instruction = evolved
                        sample.evolution_history.append(strategy)
                except Exception as e:
                    logger.warning(
                        "Evolution failed for sample %s (strategy=%s): %s",
                        sample.id,
                        strategy,
                        e,
                    )
                    # Keep original instruction on failure

        return samples
