"""PersonaGenerator: generates diverse instructions from synthetic personas."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from syntheta.generators.topic_tree import _parse_json
from syntheta.pipeline.base import BaseGenerator
from syntheta.pipeline.registry import register_generator
from syntheta.prompts import load_prompt, render_template
from syntheta.schema.sample import Sample
from syntheta.utils.seeding import create_rng

logger = logging.getLogger("syntheta.generators.persona")


@register_generator("persona")
class PersonaGenerator(BaseGenerator):
    """Generates instructions by creating diverse personas and asking them domain questions.

    Papers: Scaling Synthetic Data with 1B Personas (2024), MATRIX-Gen (ACL 2025)
    """

    def __init__(
        self,
        domain: str = "general",
        n_personas: int = 10,
        questions_per_persona: int = 5,
        persona_seed_traits: list[str] | None = None,
        languages: list[str] | None = None,
        generate_responses: bool = False,
        seed: int | None = None,
        prompt_overrides: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.domain = domain
        self.n_personas = n_personas
        self.questions_per_persona = questions_per_persona
        self.persona_seed_traits = persona_seed_traits or []
        self.languages = languages or ["en"]
        self.generate_responses = generate_responses
        self.seed = seed
        self.prompt_overrides = prompt_overrides
        self._rng = create_rng(seed)

    async def generate(self, n: int, batch_size: int = 100) -> AsyncIterator[list[Sample]]:
        """Generate personas, then generate questions from each persona's perspective."""
        # Step 1: Generate personas
        personas = await self._generate_personas()

        # Step 2: Generate questions from each persona
        batch: list[Sample] = []
        total = 0

        for persona in personas:
            if total >= n:
                break

            for lang in self.languages:
                if total >= n:
                    break

                questions = await self._generate_questions(persona, lang)
                for q in questions:
                    if total >= n:
                        break
                    sample = Sample(
                        instruction=q,
                        domain=self.domain,
                        persona=(
                            f"{persona.get('name', 'unknown')}: "
                            f"{persona.get('background', '')}"
                        ),
                        language=lang,
                    )
                    batch.append(sample)
                    total += 1

                    if len(batch) >= batch_size:
                        yield batch
                        batch = []

        if batch:
            yield batch

    async def _generate_personas(self) -> list[dict]:
        """Generate diverse synthetic personas via LLM."""
        template = load_prompt("generators.persona", self.prompt_overrides)
        prompt = render_template(template, domain=self.domain, n=self.n_personas)

        result = await self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            stage="persona_generation",
        )

        try:
            personas = _parse_json(result["content"])
            if isinstance(personas, list):
                return personas
            return [personas]
        except (json.JSONDecodeError, TypeError):
            logger.warning("Could not parse personas, using default")
            return [
                {"name": "General User", "background": "curious person", "perspective": "general"}
            ]

    async def _generate_questions(self, persona: dict, language: str) -> list[str]:
        """Generate questions from a persona's perspective."""
        template = load_prompt("generators.persona_questions", self.prompt_overrides)
        prompt = render_template(
            template,
            persona_name=persona.get("name", "User"),
            persona_background=persona.get("background", "a curious person"),
            domain=self.domain,
            n=self.questions_per_persona,
            language=language,
        )

        result = await self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            stage="persona_generation",
        )

        try:
            questions = _parse_json(result["content"])
            if isinstance(questions, list):
                return [str(q) for q in questions if q]
            return [str(questions)]
        except (json.JSONDecodeError, TypeError):
            content = result["content"].strip()
            return [content] if content else []
