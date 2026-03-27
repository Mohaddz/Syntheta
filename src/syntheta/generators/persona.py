"""PersonaGenerator: generates diverse instructions from synthetic personas."""

from __future__ import annotations

import asyncio
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

        # Step 2: Build all (persona, language) pairs
        pairs = []
        for persona in personas:
            for lang in self.languages:
                pairs.append((persona, lang))

        # Step 3: Fire all question tasks, yield each as it completes
        async def generate_questions(persona: dict, lang: str) -> list[Sample]:
            questions = await self._generate_questions_safe(persona, lang)
            return [
                Sample(
                    instruction=q,
                    domain=self.domain,
                    persona=(
                        f"{persona.get('name', 'unknown')}: "
                        f"{persona.get('background', '')}"
                    ),
                    language=lang,
                )
                for q in questions
            ]

        tasks = [asyncio.create_task(generate_questions(p, lang)) for p, lang in pairs]
        total = 0

        for coro in asyncio.as_completed(tasks):
            samples = await coro
            if samples and total < n:
                # Trim if we'd exceed n
                remaining = n - total
                if len(samples) > remaining:
                    samples = samples[:remaining]
                total += len(samples)
                yield samples
            if total >= n:
                for t in tasks:
                    if not t.done():
                        t.cancel()
                break

    async def _generate_questions_safe(self, persona: dict, language: str) -> list[str]:
        """Wrapper with error handling for parallel execution."""
        try:
            return await self._generate_questions(persona, language)
        except Exception as e:
            logger.warning("Question generation failed for persona=%s: %s", persona.get("name"), e)
            return []

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
