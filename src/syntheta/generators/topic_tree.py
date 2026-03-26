"""TopicTreeGenerator: generates instructions from scratch using a hierarchical topic tree."""

from __future__ import annotations

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
        """Build topic tree, plan distribution, generate instructions in batches."""
        # Step 1: Build topic tree
        tree = await self._build_tree()
        topics = tree.flatten()
        if not topics:
            logger.warning("Topic tree produced no topics, using domain as single topic")
            topics = [self.domain]

        # Step 2: Plan distribution
        plan = self._plan_distribution(topics, n)

        # Step 3: Generate instructions in batches
        batch: list[Sample] = []
        for entry in plan:
            instructions = await self._generate_instructions(entry)
            for instruction in instructions:
                sample = Sample(
                    instruction=instruction,
                    domain=self.domain,
                    topic=entry.topic,
                    task_type=entry.task_type,
                    language=entry.language,
                    difficulty=entry.difficulty,
                )
                batch.append(sample)

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

        if batch:
            yield batch

    async def _build_tree(self) -> TopicNode:
        """Build a hierarchical topic tree via LLM."""
        template = load_prompt("generators.topic_tree", self.prompt_overrides)
        prompt = render_template(
            template,
            domain=self.domain,
            depth=self.topic_depth,
            breadth=self.topic_breadth,
        )

        result = await self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            stage="topic_tree",
        )

        try:
            data = _parse_json(result["content"])
            return _parse_tree(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise MalformedResponseError(
                f"Failed to parse topic tree from LLM response: {e}"
            ) from e

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
    """Extract and parse JSON from LLM output, handling markdown code blocks."""
    text = text.strip()
    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (```json and ```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


def _parse_tree(data: dict) -> TopicNode:
    """Parse a JSON dict into a TopicNode tree."""
    topics_data = data.get("topics", [])
    root = TopicNode(name="root", subtopics=[])
    for t in topics_data:
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
