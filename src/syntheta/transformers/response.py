"""ResponseGenerator: generates responses for instructions that don't have them."""

from __future__ import annotations

import logging
from typing import Any

from syntheta.pipeline.base import BaseTransformer
from syntheta.pipeline.registry import register_transformer
from syntheta.prompts import load_prompt, render_template
from syntheta.schema.sample import Sample

logger = logging.getLogger("syntheta.transformers.response")


@register_transformer("response_generator")
class ResponseGenerator(BaseTransformer):
    """Generates responses for instructions using the response_model role.

    Supports optional Chain-of-Thought (CoT) prompting and custom system prompts.
    """

    def __init__(
        self,
        use_cot: bool = False,
        system_prompt: str | None = None,
        prompt_overrides: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.use_cot = use_cot
        self.system_prompt = system_prompt or "You are a helpful, accurate assistant."
        self.prompt_overrides = prompt_overrides

    async def transform(self, samples: list[Sample]) -> list[Sample]:
        """Generate responses for samples that don't already have one."""
        needs_response = [s for s in samples if s.instruction and not s.response]
        if not needs_response:
            return samples

        # Build message batches for concurrent completion
        message_batches = []
        for sample in needs_response:
            cot_instruction = ""
            if self.use_cot:
                cot_instruction = "Think through this step-by-step before giving your final answer."

            template = load_prompt("transformers.response_generator", self.prompt_overrides)
            prompt = render_template(
                template,
                system_prompt=self.system_prompt,
                cot_instruction=cot_instruction,
                instruction=sample.instruction,
            )

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            message_batches.append(messages)

        # Concurrent batch completion using response_model role
        results = await self.llm.complete_batch(
            message_batches,
            model_role="response_model",
            stage="response_generator",
        )

        # Attach responses to samples
        for sample, result in zip(needs_response, results):
            sample.response = result["content"]
            sample.generation_model = result.get("model")

        return samples
