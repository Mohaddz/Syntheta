"""EvolInstruct: evolve instructions using 6 mutation strategies."""

from __future__ import annotations

import asyncio
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
        """Apply evolution rounds. Parallel within each round, sequential across rounds."""
        for _round in range(self.rounds):
            # Pre-assign strategies for all samples (preserves deterministic RNG order)
            evolvable = [(i, s) for i, s in enumerate(samples) if s.instruction]
            assignments = [
                (idx, sample, self._rng.choice(self.strategies))
                for idx, sample in evolvable
            ]

            # Fire all evolutions in parallel within this round
            tasks = [
                self._evolve_one(sample, strategy)
                for _idx, sample, strategy in assignments
            ]
            results = await asyncio.gather(*tasks)

            # Apply results
            for (_idx, sample, strategy), evolved in zip(assignments, results):
                if evolved is not None:
                    sample.instruction = evolved
                    sample.evolution_history.append(strategy)

        return samples

    async def _evolve_one(self, sample: Sample, strategy: str) -> str | None:
        """Evolve a single sample's instruction. Returns new instruction or None on failure."""
        template = load_prompt(f"transformers.evol_{strategy}", self.prompt_overrides)
        prompt = render_template(template, instruction=sample.instruction)

        try:
            result = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                stage="evol_instruct",
            )
            evolved = result["content"].strip()
            return evolved if evolved else None
        except Exception as e:
            logger.warning(
                "Evolution failed for sample %s (strategy=%s): %s",
                sample.id,
                strategy,
                e,
            )
            return None
