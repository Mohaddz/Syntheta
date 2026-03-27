"""SafetyFilter: two-layer content filtering for safety and cultural appropriateness."""

from __future__ import annotations

import asyncio
import json
import logging
from importlib import resources
from typing import Any

import yaml

from syntheta.pipeline.base import BaseFilter
from syntheta.pipeline.registry import register_filter
from syntheta.prompts import load_prompt, render_template
from syntheta.schema.sample import Sample

logger = logging.getLogger("syntheta.filters.safety")

# Layer 1: Universal safety — always on, non-configurable
# These keywords trigger immediate rejection without LLM check
_UNIVERSAL_BLOCKLIST = [
    "how to make a bomb",
    "how to make explosives",
    "how to hack into",
    "child abuse",
    "child exploitation",
    "suicide instructions",
    "self-harm instructions",
    "how to poison",
    "biological weapon",
    "chemical weapon",
]


@register_filter("safety")
class SafetyFilter(BaseFilter):
    """Two-layer safety filter: universal keyword check + optional LLM cultural check."""

    rejection_reason = "safety_blocked"

    def __init__(
        self,
        cultural_context: str | None = None,
        blocklist_topics: list[str] | None = None,
        blocklist_words: list[str] | None = None,
        custom_instruction: str | None = None,
        prompt_overrides: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cultural_context = cultural_context
        self.blocklist_topics = blocklist_topics or []
        self.blocklist_words = blocklist_words or []
        self.custom_instruction = custom_instruction
        self.prompt_overrides = prompt_overrides

        # Load and merge cultural preset
        if cultural_context:
            preset = _load_preset(cultural_context)
            self.blocklist_topics = list(
                set(self.blocklist_topics + preset.get("blocklist_topics", []))
            )
            self.blocklist_words = list(
                set(self.blocklist_words + preset.get("blocklist_words", []))
            )
            if not self.custom_instruction:
                self.custom_instruction = preset.get("custom_instruction", "")

    async def filter(self, samples: list[Sample]) -> list[Sample]:
        """Filter samples through Layer 1 (keywords) and Layer 2 (LLM cultural check)."""
        # Layer 1: Universal keyword check (fast, no LLM)
        passed_layer1 = []
        for sample in samples:
            text = _extract_text(sample).lower()
            if _keyword_check(text, _UNIVERSAL_BLOCKLIST):
                logger.info("Sample %s blocked by universal safety check", sample.id)
                sample.safety_passed = False
                continue
            # Also check user-provided blocklist words
            if self.blocklist_words and _keyword_check(text, self.blocklist_words):
                logger.info("Sample %s blocked by custom blocklist", sample.id)
                sample.safety_passed = False
                continue
            passed_layer1.append(sample)

        # Layer 2: LLM cultural check (only if cultural_context is set and LLM available)
        if self.cultural_context and self.llm:
            return await self._llm_check(passed_layer1)

        # Mark all survivors as passed
        for s in passed_layer1:
            s.safety_passed = True
        return passed_layer1

    async def _llm_check(self, samples: list[Sample]) -> list[Sample]:
        """Run LLM-based cultural safety check in parallel."""
        if not samples:
            return samples

        # Check all samples in parallel (semaphore controls concurrency)
        tasks = [self._check_one(sample) for sample in samples]
        await asyncio.gather(*tasks)

        # Collect results
        return [s for s in samples if s.safety_passed]

    async def _check_one(self, sample: Sample) -> None:
        """Check a single sample for cultural safety. Sets safety_passed on the sample."""
        template = load_prompt("filters.safety_check", self.prompt_overrides)
        cultural_instruction = self.custom_instruction or ""
        content = _extract_text(sample)
        prompt = render_template(
            template,
            cultural_instruction=cultural_instruction,
            content=content,
        )

        try:
            result = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                stage="safety_filter",
            )
            response = result["content"].strip()
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                if '"safe": true' in response.lower() or '"safe":true' in response.lower():
                    data = {"safe": True}
                elif '"safe": false' in response.lower() or '"safe":false' in response.lower():
                    data = {"safe": False, "reason": "flagged by LLM"}
                else:
                    data = {"safe": True}

            if data.get("safe", True):
                sample.safety_passed = True
            else:
                sample.safety_passed = False
                logger.info(
                    "Sample %s blocked by LLM safety check: %s",
                    sample.id,
                    data.get("reason", "no reason"),
                )
        except Exception as e:
            # On LLM error, pass the sample through (fail-open)
            logger.warning("Safety check LLM error for sample %s: %s", sample.id, e)
            sample.safety_passed = True


def _extract_text(sample: Sample) -> str:
    """Extract all text content from a sample for safety checking."""
    parts = []
    if sample.instruction:
        parts.append(sample.instruction)
    if sample.response:
        parts.append(sample.response)
    if sample.text:
        parts.append(sample.text)
    if sample.turns:
        for t in sample.turns:
            parts.append(t.content)
    return " ".join(parts)


def _keyword_check(text: str, blocklist: list[str]) -> bool:
    """Check if any blocklist keyword appears in the text."""
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in blocklist)


def _load_preset(name: str) -> dict:
    """Load a cultural safety preset by name."""
    try:
        preset_file = resources.files("syntheta.presets").joinpath(f"{name}.yaml")
        return yaml.safe_load(preset_file.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not load preset '%s': %s", name, e)
        return {}
