"""TopicTreeGenerator: generates instructions from scratch using a hierarchical topic tree."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from syntheta.exceptions import MalformedResponseError
from syntheta.pipeline.base import BaseGenerator
from syntheta.pipeline.registry import register_generator
from syntheta.prompts import load_prompt, render_template
from syntheta.schema.sample import Sample
from syntheta.utils.seeding import create_rng

logger = logging.getLogger("syntheta.generators.topic_tree")


@dataclass
class TopicNode:
    """A node in the topic tree."""

    name: str
    description: str = ""
    subtopics: list[TopicNode] = field(default_factory=list)

    def flatten(self) -> list[str]:
        """Return all leaf topic names."""
        if not self.subtopics:
            return [self.name]
        leaves = []
        for sub in self.subtopics:
            leaves.extend(sub.flatten())
        return leaves


@dataclass
class PlanEntry:
    """One planned generation: topic × task_type × language × difficulty."""

    topic: str
    task_type: str
    language: str
    difficulty: int


@register_generator("topic_tree")
class TopicTreeGenerator(BaseGenerator):
    """Generates diverse instructions by building a topic taxonomy and sampling across it.

    Papers: TreeSynth (2025), Seed-Free SDG (2024)
    """

    def __init__(
        self,
        domain: str = "general",
        task_types: list[str] | None = None,
        languages: list[str] | None = None,
        difficulty_range: tuple[int, int] = (1, 5),
        topic_depth: int = 2,
        topic_breadth: int = 5,
        generate_responses: bool = False,
        seed: int | None = None,
        prompt_overrides: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.domain = domain
        self.task_types = task_types or ["qa", "explain", "compare", "creative"]
        self.languages = languages or ["en"]
        self.difficulty_range = difficulty_range
        self.topic_depth = topic_depth
        self.topic_breadth = topic_breadth
        self.generate_responses = generate_responses
        self.seed = seed
        self.prompt_overrides = prompt_overrides
        self._rng = create_rng(seed)

    async def generate(self, n: int, batch_size: int = 100) -> AsyncIterator[list[Sample]]:
        """Build topic tree, plan distribution, generate instructions concurrently.

        Yields individual samples as they complete -- no chunking, no waiting
        for all instructions to finish. The LLM semaphore controls concurrency.
        """
        # Step 1: Build topic tree
        tree = await self._build_tree()
        topics = tree.flatten()
        if not topics:
            logger.warning("Topic tree produced no topics, using domain as single topic")
            topics = [self.domain]

        # Step 2: Plan distribution
        plan = self._plan_distribution(topics, n)

        # Step 3: Fire all instruction tasks, yield each sample as it completes
        async def generate_sample(entry: PlanEntry) -> list[Sample]:
            instructions = await self._generate_one(entry)
            return [
                Sample(
                    instruction=inst,
                    domain=self.domain,
                    topic=entry.topic,
                    task_type=entry.task_type,
                    language=entry.language,
                    difficulty=entry.difficulty,
                )
                for inst in instructions
            ]

        tasks = [asyncio.create_task(generate_sample(entry)) for entry in plan]

        for coro in asyncio.as_completed(tasks):
            samples = await coro
            if samples:
                yield samples

    async def _generate_one(self, entry: PlanEntry) -> list[str]:
        """Generate instructions for a single plan entry. Safe for parallel execution."""
        try:
            return await self._generate_instructions(entry)
        except Exception as e:
            logger.warning("Instruction generation failed for topic=%s: %s", entry.topic, e)
            return []

    async def _build_tree(self) -> TopicNode:
        """Build a hierarchical topic tree via LLM. Retries on malformed JSON."""
        template = load_prompt("generators.topic_tree", self.prompt_overrides)
        prompt = render_template(
            template,
            domain=self.domain,
            depth=self.topic_depth,
            breadth=self.topic_breadth,
        )

        max_attempts = 3
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            result = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                stage="topic_tree",
            )

            try:
                data = _parse_json(result["content"])
                return _parse_tree(data)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                last_error = e
                logger.warning(
                    "Topic tree parse failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_attempts,
                    e,
                )

        raise MalformedResponseError(
            f"Failed to parse topic tree after {max_attempts} attempts: {last_error}"
        )

    def _plan_distribution(self, topics: list[str], n: int) -> list[PlanEntry]:
        """Allocate n samples across topic × task_type × language × difficulty."""
        difficulties = list(range(self.difficulty_range[0], self.difficulty_range[1] + 1))

        # Build all combinations
        combos = []
        for topic in topics:
            for task_type in self.task_types:
                for lang in self.languages:
                    for diff in difficulties:
                        combos.append(PlanEntry(topic, task_type, lang, diff))

        # Shuffle deterministically
        self._rng.shuffle(combos)

        # Allocate samples to combos (round-robin)
        plan = []
        idx = 0
        remaining = n
        while remaining > 0:
            plan.append(combos[idx % len(combos)])
            idx += 1
            remaining -= 1

        return plan

    async def _generate_instructions(self, entry: PlanEntry) -> list[str]:
        """Generate instructions for a single plan entry."""
        template = load_prompt("generators.topic_tree_instructions", self.prompt_overrides)
        prompt = render_template(
            template,
            domain=self.domain,
            topic=entry.topic,
            task_type=entry.task_type,
            difficulty=entry.difficulty,
            language=entry.language,
            n=1,
        )

        result = await self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            stage="topic_tree",
        )

        try:
            instructions = _parse_json(result["content"])
            if isinstance(instructions, list):
                return [str(i) for i in instructions if i]
            return [str(instructions)]
        except (json.JSONDecodeError, TypeError):
            # Fall back to treating the whole response as a single instruction
            content = result["content"].strip()
            if content:
                return [content]
            return []


def _parse_json(text: str) -> Any:
    """Extract and parse JSON from LLM output.

    Handles: markdown code blocks, truncated JSON, extra text around JSON.
    """
    text = text.strip()

    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object or array from the text
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        # Find the last matching bracket
        end = text.rfind(end_char)
        if end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

    # Try to repair truncated JSON by closing open brackets
    for start_char, _end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start != -1:
            fragment = text[start:]
            repaired = _repair_json(fragment)
            if repaired is not None:
                return repaired

    # Give up -- raise the original error
    return json.loads(text)


def _repair_json(text: str) -> Any | None:
    """Try to repair truncated JSON by closing open brackets/braces."""
    # Count open vs close brackets
    opens = {"[": 0, "{": 0}
    closes = {"]": "[", "}": "{"}

    for char in text:
        if char in opens:
            opens[char] += 1
        elif char in closes:
            opens[closes[char]] = max(0, opens[closes[char]] - 1)

    # Truncate at the last valid comma or complete value, then close brackets
    # Try progressively shorter substrings
    for trim in range(0, min(200, len(text)), 10):
        candidate = text[: len(text) - trim] if trim else text
        # Remove trailing partial content after last comma
        last_comma = candidate.rfind(",")
        if last_comma > 0 and trim > 0:
            candidate = candidate[:last_comma]
        # Close any open brackets
        suffix = ""
        for char in reversed(candidate):
            if char == "{":
                suffix += "}"
            elif char == "[":
                suffix += "]"
            elif char == "}":
                suffix = suffix[:-1] if suffix.endswith("}") else suffix
            elif char == "]":
                suffix = suffix[:-1] if suffix.endswith("]") else suffix

        # Recount what's needed
        o = {"[": 0, "{": 0}
        for c in candidate:
            if c in o:
                o[c] += 1
            elif c in closes:
                o[closes[c]] = max(0, o[closes[c]] - 1)

        suffix = "}" * o["{"] + "]" * o["["]

        try:
            return json.loads(candidate + suffix)
        except json.JSONDecodeError:
            continue

    return None


def _parse_tree(data: dict | list) -> TopicNode:
    """Parse a JSON dict or list into a TopicNode tree.

    Handles both {"topics": [...]} and bare [...] (from truncated JSON repair).
    """
    topics_data = data if isinstance(data, list) else data.get("topics", [])
    root = TopicNode(name="root", subtopics=[])
    for t in topics_data:
        if isinstance(t, dict):
            root.subtopics.append(_parse_node(t))
    return root


def _parse_node(data: dict) -> TopicNode:
    """Recursively parse a topic node."""
    node = TopicNode(
        name=data.get("name", "unknown"),
        description=data.get("description", ""),
    )
    for sub in data.get("subtopics", []):
        node.subtopics.append(_parse_node(sub))
    return node
