"""TopicTreeGenerator: spatial-partitioning tree for diverse instruction generation.

Implements the TreeSynth algorithm (Wang et al., 2025 — arxiv.org/abs/2503.17195).
Builds a BFS tree where each node discovers a differentiating dimension and splits
the data space along it, then generates instructions from leaf subspaces.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from syntheta.pipeline.base import BaseGenerator
from syntheta.pipeline.registry import register_generator
from syntheta.prompts import load_prompt, render_template
from syntheta.schema.sample import Sample
from syntheta.utils.seeding import create_rng

logger = logging.getLogger("syntheta.generators.topic_tree")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TreeNode:
    """A node in the spatial-partitioning tree.

    Each internal node represents a split along a discovered *dimension*
    (e.g. "reasoning_type").  Its children represent mutually exclusive
    *attribute values* for that dimension (e.g. "deductive", "inductive").
    """

    depth: int
    dimension: str | None = None
    attribute_value: str | None = None
    parent: TreeNode | None = field(default=None, repr=False)
    children: list[TreeNode] = field(default_factory=list)
    samples: list[str] = field(default_factory=list, repr=False)
    excluded_dimensions: set[str] = field(default_factory=set)
    is_infinite: bool = False
    all_attributes: list[str] = field(default_factory=list)

    @property
    def path_constraints(self) -> list[dict[str, str]]:
        """Collect dimension/attribute pairs from root to this node."""
        constraints: list[dict[str, str]] = []
        node: TreeNode | None = self
        while node is not None:
            if node.attribute_value and node.parent and node.parent.dimension:
                constraints.append(
                    {
                        "dimension": node.parent.dimension,
                        "attribute_value": node.attribute_value,
                    }
                )
            node = node.parent
        constraints.reverse()
        return constraints

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


@dataclass
class LeafTask:
    """One generation unit: a leaf node + output parameters."""

    leaf: TreeNode
    num_samples: int
    task_type: str
    language: str
    difficulty: int


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


@register_generator("topic_tree")
class TopicTreeGenerator(BaseGenerator):
    """Generates diverse instructions via TreeSynth spatial-partitioning tree.

    Paper: TreeSynth (Wang et al., 2025) — arxiv.org/abs/2503.17195

    Algorithm:
    1. BFS tree construction — at each node, generate pivot samples, discover a
       splitting dimension, expand its attribute values, create children.
    2. Leaf generation — for each leaf, serialize the root-to-leaf path as
       dimensional constraints and generate instructions that satisfy them all.
    """

    def __init__(
        self,
        domain: str = "general",
        description: str | None = None,
        task_types: list[str] | None = None,
        languages: list[str] | None = None,
        difficulty_range: tuple[int, int] = (1, 5),
        max_depth: int = 4,
        num_samples_per_node: int = 10,
        max_attribute_count: int = 50,
        seed: int | None = None,
        prompt_overrides: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.domain = domain
        self.description = description or ""
        self.task_types = task_types or ["qa", "explain", "compare", "creative"]
        self.languages = languages or ["en"]
        self.difficulty_range = difficulty_range
        self.max_depth = max_depth
        self.num_samples_per_node = num_samples_per_node
        self.max_attribute_count = max_attribute_count
        self.seed = seed
        self.prompt_overrides = prompt_overrides
        self._rng = create_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(self, n: int) -> AsyncIterator[list[Sample]]:
        """Build spatial-partitioning tree, then generate from leaves.

        Phase 1 — tree construction (no samples yielded).
        Phase 2 — sliding-window generation over leaf tasks.
        """
        # Phase 1: build tree
        root = TreeNode(depth=0)
        await self._build_tree(root)
        leaves = _collect_leaves(root)
        if not leaves:
            logger.warning("Tree produced no leaves, using root as single leaf")
            leaves = [root]

        logger.info(
            "Tree built: %d leaves across %d depth levels",
            len(leaves),
            self.max_depth,
        )

        # Phase 2: assign tasks and generate with sliding window
        plan = self._assign_leaf_tasks(leaves, n)

        async def run_task(task: LeafTask) -> list[Sample]:
            try:
                return await self._generate_leaf_samples(task)
            except Exception as e:
                logger.warning("Leaf generation failed: %s", e)
                return []

        window = min(self.max_concurrent, len(plan))
        pending: set[asyncio.Task] = set()
        idx = 0

        while idx < window:
            pending.add(asyncio.create_task(run_task(plan[idx])))
            idx += 1

        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for _ in done:
                if idx < len(plan):
                    pending.add(asyncio.create_task(run_task(plan[idx])))
                    idx += 1
            for task in done:
                samples = task.result()
                if samples:
                    yield samples

    # ------------------------------------------------------------------
    # Tree construction (BFS)
    # ------------------------------------------------------------------

    async def _build_tree(self, root: TreeNode) -> None:
        """Expand the tree level-by-level (BFS) up to max_depth."""
        current_level = [root]
        for depth in range(self.max_depth):
            if not current_level:
                break
            logger.info(
                "Expanding tree level %d (%d nodes)", depth, len(current_level)
            )
            results = await self._expand_level(current_level)
            next_level: list[TreeNode] = []
            for node, children in zip(current_level, results):
                node.children = children
                for child in children:
                    child.parent = node
                next_level.extend(children)
            current_level = next_level

    async def _expand_level(
        self, nodes: list[TreeNode]
    ) -> list[list[TreeNode]]:
        """Expand all nodes at one BFS level using a sliding window."""
        results: list[list[TreeNode]] = [[] for _ in nodes]

        async def expand_one(index: int) -> tuple[int, list[TreeNode]]:
            children = await self._expand_node(nodes[index])
            return (index, children)

        window = min(self.max_concurrent, len(nodes))
        pending: set[asyncio.Task] = set()
        idx = 0

        while idx < window:
            pending.add(asyncio.create_task(expand_one(idx)))
            idx += 1

        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for _ in done:
                if idx < len(nodes):
                    pending.add(asyncio.create_task(expand_one(idx)))
                    idx += 1
            for task in done:
                i, children = task.result()
                results[i] = children

        return results

    async def _expand_node(self, node: TreeNode) -> list[TreeNode]:
        """Run the TreeSynth per-node expansion: pivot → dimension → expand."""
        # Step 1: generate pivot samples
        pivot_samples = await self._generate_pivot_samples(node)
        if not pivot_samples:
            return []
        node.samples = pivot_samples

        # Step 2: discover splitting dimension
        dim_result = await self._discover_dimension(node, pivot_samples)
        if dim_result is None:
            return []  # graceful degradation — node becomes leaf
        dimension, attribute_map = dim_result
        node.dimension = dimension

        # Step 3: expand attribute values
        initial_attrs = list(attribute_map.keys())
        attributes = await self._expand_attributes(node, dimension, initial_attrs)

        # Build children
        new_excluded = node.excluded_dimensions | {dimension}
        if len(attributes) > self.max_attribute_count:
            # Infinite node: single child holding all attributes
            child = TreeNode(
                depth=node.depth + 1,
                excluded_dimensions=new_excluded,
                is_infinite=True,
                all_attributes=attributes[: self.max_attribute_count],
            )
            return [child]

        children: list[TreeNode] = []
        for attr in attributes:
            child = TreeNode(
                depth=node.depth + 1,
                attribute_value=attr,
                excluded_dimensions=new_excluded,
            )
            children.append(child)
        return children

    # ------------------------------------------------------------------
    # LLM interaction helpers
    # ------------------------------------------------------------------

    async def _generate_pivot_samples(self, node: TreeNode) -> list[str]:
        """Generate diverse pivot samples for a node's subspace."""
        template = load_prompt("generators.topic_tree", self.prompt_overrides)
        prompt = render_template(
            template,
            domain=self.domain,
            description=self.description,
            subspace_description=self._subspace_description(node),
            num_samples=self.num_samples_per_node,
        )
        result = await self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            stage="topic_tree",
        )
        try:
            parsed = _parse_json(result["content"])
            if isinstance(parsed, list):
                return [str(s) for s in parsed if s]
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse pivot samples at depth %d", node.depth)
        return []

    async def _discover_dimension(
        self, node: TreeNode, pivot_samples: list[str]
    ) -> tuple[str, dict[str, list[int]]] | None:
        """Discover a splitting dimension from pivot samples. Up to 5 retries."""
        template = load_prompt(
            "generators.topic_tree_dimension", self.prompt_overrides
        )
        indexed_samples = "\n".join(
            f"[{i}] {s}" for i, s in enumerate(pivot_samples)
        )
        excluded = ", ".join(sorted(node.excluded_dimensions)) or "none"

        for attempt in range(5):
            prompt = render_template(
                template,
                domain=self.domain,
                description=self.description,
                subspace_description=self._subspace_description(node),
                indexed_samples=indexed_samples,
                excluded_dimensions=excluded,
            )
            result = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                stage="topic_tree",
            )
            try:
                data = _parse_json(result["content"])
                dimension = str(data["dimension"])
                attributes: dict[str, list[int]] = data["attributes"]

                # Validation
                if dimension in node.excluded_dimensions:
                    logger.warning(
                        "Dimension '%s' is excluded (attempt %d/5)", dimension, attempt + 1
                    )
                    continue
                if not isinstance(attributes, dict) or len(attributes) < 2:
                    logger.warning(
                        "Need >=2 attributes, got %d (attempt %d/5)",
                        len(attributes) if isinstance(attributes, dict) else 0,
                        attempt + 1,
                    )
                    continue
                # Mutual exclusivity check
                all_indices: list[int] = []
                for indices in attributes.values():
                    if isinstance(indices, list):
                        all_indices.extend(indices)
                if len(all_indices) != len(set(all_indices)):
                    logger.warning(
                        "Non-exclusive attribute assignment (attempt %d/5)", attempt + 1
                    )
                    continue

                return (dimension, attributes)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(
                    "Dimension discovery parse failed (attempt %d/5): %s",
                    attempt + 1,
                    e,
                )

        logger.info(
            "Dimension discovery failed after 5 attempts at depth %d — node becomes leaf",
            node.depth,
        )
        return None

    async def _expand_attributes(
        self,
        node: TreeNode,
        dimension: str,
        initial_attrs: list[str],
    ) -> list[str]:
        """Iteratively expand attribute values until complete or max reached."""
        attributes = list(initial_attrs)
        template = load_prompt(
            "generators.topic_tree_expand", self.prompt_overrides
        )

        while len(attributes) < self.max_attribute_count:
            prompt = render_template(
                template,
                domain=self.domain,
                description=self.description,
                subspace_description=self._subspace_description(node),
                dimension=dimension,
                existing_attributes=json.dumps(attributes),
            )
            result = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                stage="topic_tree",
            )
            content = result["content"].strip()

            # "null" means the list is already complete
            if content.lower().strip('"') == "null":
                break

            try:
                data = _parse_json(content)
                if isinstance(data, dict):
                    new_attrs = data.get("attributes", [])
                    completeness = data.get("completeness", "complete")
                elif isinstance(data, list):
                    new_attrs = data
                    completeness = "complete"
                else:
                    break

                for attr in new_attrs:
                    a = str(attr)
                    if a not in attributes:
                        attributes.append(a)

                if completeness != "infinite":
                    break
            except (json.JSONDecodeError, TypeError):
                break

        return attributes

    # ------------------------------------------------------------------
    # Leaf generation
    # ------------------------------------------------------------------

    def _assign_leaf_tasks(
        self, leaves: list[TreeNode], n: int
    ) -> list[LeafTask]:
        """Distribute n samples across leaves × task_type × language × difficulty."""
        difficulties = list(
            range(self.difficulty_range[0], self.difficulty_range[1] + 1)
        )
        combos: list[LeafTask] = []
        for leaf in leaves:
            for task_type in self.task_types:
                for lang in self.languages:
                    for diff in difficulties:
                        combos.append(
                            LeafTask(
                                leaf=leaf,
                                num_samples=1,
                                task_type=task_type,
                                language=lang,
                                difficulty=diff,
                            )
                        )

        self._rng.shuffle(combos)

        # Round-robin allocate n tasks
        plan: list[LeafTask] = []
        for i in range(n):
            plan.append(combos[i % len(combos)])
        return plan

    async def _generate_leaf_samples(self, task: LeafTask) -> list[Sample]:
        """Generate instructions from a leaf node using path constraints."""
        leaf = task.leaf
        constraints = list(leaf.path_constraints)

        # For infinite nodes, randomly pick one attribute
        if leaf.is_infinite and leaf.all_attributes:
            parent_dim = leaf.parent.dimension if leaf.parent else "variant"
            chosen_attr = self._rng.choice(leaf.all_attributes)
            constraints.append(
                {"dimension": parent_dim, "attribute_value": chosen_attr}
            )

        # If no constraints at all (root is leaf), use domain as context
        constraints_str = (
            json.dumps(constraints, indent=2)
            if constraints
            else json.dumps(
                [{"dimension": "domain", "attribute_value": self.domain}]
            )
        )

        template = load_prompt(
            "generators.topic_tree_instructions", self.prompt_overrides
        )
        prompt = render_template(
            template,
            domain=self.domain,
            description=self.description,
            constraints_json=constraints_str,
            task_type=task.task_type,
            language=task.language,
            difficulty=task.difficulty,
            num_samples=task.num_samples,
        )

        result = await self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            stage="topic_tree",
        )

        try:
            instructions = _parse_json(result["content"])
            if not isinstance(instructions, list):
                instructions = [instructions]
        except (json.JSONDecodeError, TypeError):
            content = result["content"].strip()
            instructions = [content] if content else []

        topic_str = (
            " > ".join(
                c["attribute_value"]
                for c in constraints
                if c.get("attribute_value")
            )
            or self.domain
        )

        return [
            Sample(
                instruction=str(inst),
                domain=self.domain,
                topic=topic_str,
                task_type=task.task_type,
                language=task.language,
                difficulty=task.difficulty,
            )
            for inst in instructions
            if inst
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _subspace_description(self, node: TreeNode) -> str:
        """Human-readable description of a node's subspace."""
        constraints = node.path_constraints
        if not constraints:
            return f"The entire domain of '{self.domain}'"
        parts = [
            f"{c['dimension']}={c['attribute_value']}" for c in constraints
        ]
        return f"Domain '{self.domain}' where {', '.join(parts)}"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _collect_leaves(root: TreeNode) -> list[TreeNode]:
    """BFS collection of all leaf nodes."""
    leaves: list[TreeNode] = []
    queue = [root]
    while queue:
        node = queue.pop(0)
        if node.is_leaf:
            leaves.append(node)
        else:
            queue.extend(node.children)
    return leaves


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

    # Give up — raise the original error
    return json.loads(text)


def _repair_json(text: str) -> Any | None:
    """Try to repair truncated JSON by closing open brackets/braces."""
    closes = {"]": "[", "}": "{"}

    for trim in range(0, min(200, len(text)), 10):
        candidate = text[: len(text) - trim] if trim else text
        last_comma = candidate.rfind(",")
        if last_comma > 0 and trim > 0:
            candidate = candidate[:last_comma]

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
