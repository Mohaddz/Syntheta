"""SeedDatasetGenerator: Self-Instruct from existing data."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from syntheta.generators.topic_tree import _parse_json
from syntheta.pipeline.base import BaseGenerator
from syntheta.pipeline.registry import register_generator
from syntheta.prompts import load_prompt, render_template
from syntheta.schema.column_mapper import ColumnMapper
from syntheta.schema.sample import Sample
from syntheta.utils.jsonl import read_jsonl
from syntheta.utils.seeding import create_rng

logger = logging.getLogger("syntheta.generators.seed_dataset")


@register_generator("seed_dataset")
class SeedDatasetGenerator(BaseGenerator):
    """Generates new instructions by sampling from existing data as few-shot examples.

    Paper: Self-Instruct (ACL 2023)
    """

    def __init__(
        self,
        source: str = "",
        n_few_shot: int = 3,
        column_map: dict[str, str] | None = None,
        seed: int | None = None,
        prompt_overrides: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.source = source
        self.n_few_shot = n_few_shot
        self.column_map = column_map
        self.seed = seed
        self.prompt_overrides = prompt_overrides
        self._rng = create_rng(seed)

    async def generate(self, n: int, batch_size: int = 100) -> AsyncIterator[list[Sample]]:
        """Load seed data, sample few-shot examples, generate new instructions iteratively."""
        # Load seed data
        records = list(read_jsonl(self.source))
        mapper = ColumnMapper(column_map=self.column_map)
        seed_samples = mapper.map(records)

        if not seed_samples:
            logger.warning("No seed data loaded from %s", self.source)
            return

        template = load_prompt("generators.seed_instruct", self.prompt_overrides)
        batch: list[Sample] = []
        total = 0

        while total < n:
            # Sample few-shot examples
            few_shot = self._rng.sample(seed_samples, min(self.n_few_shot, len(seed_samples)))
            examples = "\n".join(f"- {s.instruction}" for s in few_shot if s.instruction)

            remaining = min(5, n - total)  # Generate up to 5 per call
            prompt = render_template(template, examples=examples, n=remaining)

            result = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                stage="seed_dataset",
            )

            try:
                new_instructions = _parse_json(result["content"])
                if not isinstance(new_instructions, list):
                    new_instructions = [new_instructions]
            except (json.JSONDecodeError, TypeError):
                new_instructions = [result["content"].strip()]

            for inst in new_instructions:
                if total >= n:
                    break
                inst_str = str(inst).strip()
                if not inst_str:
                    continue
                sample = Sample(
                    instruction=inst_str,
                    source_id="seed_dataset",
                )
                batch.append(sample)
                total += 1

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

        if batch:
            yield batch
