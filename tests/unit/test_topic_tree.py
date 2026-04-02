"""Tests for TopicTreeGenerator (TreeSynth algorithm) with mocked LLM."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from syntheta.generators.topic_tree import (
    LeafTask,
    TopicTreeGenerator,
    TreeNode,
    _collect_leaves,
    _parse_json,
)


# ---------------------------------------------------------------------------
# TreeNode tests
# ---------------------------------------------------------------------------


class TestTreeNode:
    def test_path_constraints_root(self):
        root = TreeNode(depth=0)
        assert root.path_constraints == []

    def test_path_constraints_depth_3(self):
        root = TreeNode(depth=0, dimension="topic_area")
        child = TreeNode(
            depth=1,
            attribute_value="science",
            parent=root,
            dimension="complexity",
        )
        grandchild = TreeNode(
            depth=2,
            attribute_value="advanced",
            parent=child,
        )
        constraints = grandchild.path_constraints
        assert len(constraints) == 2
        assert constraints[0] == {
            "dimension": "topic_area",
            "attribute_value": "science",
        }
        assert constraints[1] == {
            "dimension": "complexity",
            "attribute_value": "advanced",
        }

    def test_is_leaf_no_children(self):
        node = TreeNode(depth=0)
        assert node.is_leaf is True

    def test_is_leaf_with_children(self):
        child = TreeNode(depth=1)
        node = TreeNode(depth=0, children=[child])
        assert node.is_leaf is False

    def test_collect_leaves(self):
        leaf1 = TreeNode(depth=2, attribute_value="a")
        leaf2 = TreeNode(depth=2, attribute_value="b")
        leaf3 = TreeNode(depth=1, attribute_value="c")  # leaf at different depth
        internal = TreeNode(depth=1, children=[leaf1, leaf2])
        root = TreeNode(depth=0, children=[internal, leaf3])
        leaves = _collect_leaves(root)
        assert len(leaves) == 3
        assert set(l.attribute_value for l in leaves) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# Dimension discovery validation tests
# ---------------------------------------------------------------------------


class TestDimensionDiscovery:
    @pytest.fixture
    def generator(self):
        llm = MagicMock()
        llm.complete = AsyncMock()
        return TopicTreeGenerator(domain="test", llm=llm)

    @pytest.mark.asyncio
    async def test_rejects_excluded_dimension(self, generator):
        """Dimension in excluded set should trigger retry."""
        node = TreeNode(depth=1, excluded_dimensions={"topic_area"})
        samples = ["sample 1", "sample 2", "sample 3"]

        # All 5 attempts return the excluded dimension
        excluded_response = {
            "content": json.dumps({
                "dimension": "topic_area",
                "attributes": {"a": [0], "b": [1, 2]},
            }),
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            "model": "test",
        }
        generator.llm.complete = AsyncMock(return_value=excluded_response)

        result = await generator._discover_dimension(node, samples)
        assert result is None
        assert generator.llm.complete.call_count == 5

    @pytest.mark.asyncio
    async def test_rejects_non_exclusive_indices(self, generator):
        """Overlapping sample indices should trigger retry."""
        node = TreeNode(depth=0)
        samples = ["s1", "s2", "s3"]

        overlap_response = {
            "content": json.dumps({
                "dimension": "type",
                "attributes": {"a": [0, 1], "b": [1, 2]},  # index 1 in both
            }),
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            "model": "test",
        }
        generator.llm.complete = AsyncMock(return_value=overlap_response)

        result = await generator._discover_dimension(node, samples)
        assert result is None

    @pytest.mark.asyncio
    async def test_rejects_single_attribute(self, generator):
        """Need at least 2 attribute values."""
        node = TreeNode(depth=0)
        samples = ["s1", "s2"]

        single_response = {
            "content": json.dumps({
                "dimension": "type",
                "attributes": {"only_one": [0, 1]},
            }),
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            "model": "test",
        }
        generator.llm.complete = AsyncMock(return_value=single_response)

        result = await generator._discover_dimension(node, samples)
        assert result is None

    @pytest.mark.asyncio
    async def test_accepts_valid_response(self, generator):
        """Valid dimension discovery should succeed."""
        node = TreeNode(depth=0)
        samples = ["explain sorting", "write a poem", "solve integral"]

        valid_response = {
            "content": json.dumps({
                "dimension": "task_domain",
                "attributes": {
                    "computer_science": [0],
                    "creative_writing": [1],
                    "mathematics": [2],
                },
            }),
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            "model": "test",
        }
        generator.llm.complete = AsyncMock(return_value=valid_response)

        result = await generator._discover_dimension(node, samples)
        assert result is not None
        dim, attrs = result
        assert dim == "task_domain"
        assert len(attrs) == 3
        assert generator.llm.complete.call_count == 1


# ---------------------------------------------------------------------------
# Attribute expansion tests
# ---------------------------------------------------------------------------


class TestAttributeExpansion:
    @pytest.fixture
    def generator(self):
        llm = MagicMock()
        llm.complete = AsyncMock()
        return TopicTreeGenerator(domain="test", max_attribute_count=10, llm=llm)

    @pytest.mark.asyncio
    async def test_null_stops(self, generator):
        """'null' response means list is complete."""
        node = TreeNode(depth=0)
        generator.llm.complete = AsyncMock(
            return_value={
                "content": "null",
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
                "model": "test",
            }
        )

        result = await generator._expand_attributes(node, "type", ["a", "b"])
        assert result == ["a", "b"]
        assert generator.llm.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_complete_stops(self, generator):
        """completeness=complete adds attrs and stops."""
        node = TreeNode(depth=0)
        generator.llm.complete = AsyncMock(
            return_value={
                "content": json.dumps({
                    "attributes": ["c", "d"],
                    "completeness": "complete",
                }),
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
                "model": "test",
            }
        )

        result = await generator._expand_attributes(node, "type", ["a", "b"])
        assert result == ["a", "b", "c", "d"]
        assert generator.llm.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_infinite_loops_until_max(self, generator):
        """completeness=infinite keeps looping until max_attribute_count."""
        node = TreeNode(depth=0)

        call_count = 0

        async def infinite_response(**kwargs):
            nonlocal call_count
            call_count += 1
            return {
                "content": json.dumps({
                    "attributes": [f"attr_{call_count}_{i}" for i in range(3)],
                    "completeness": "infinite",
                }),
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
                "model": "test",
            }

        generator.llm.complete = AsyncMock(side_effect=infinite_response)

        result = await generator._expand_attributes(node, "type", ["a", "b"])
        # max_attribute_count=10, starts with 2, adds 3 per call
        # After call 1: 5, call 2: 8, call 3: 11 >= 10 → stop
        assert len(result) >= generator.max_attribute_count

    @pytest.mark.asyncio
    async def test_dedup(self, generator):
        """Duplicate attributes should not be added."""
        node = TreeNode(depth=0)
        generator.llm.complete = AsyncMock(
            return_value={
                "content": json.dumps({
                    "attributes": ["a", "b", "c"],  # a and b are dupes
                    "completeness": "complete",
                }),
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
                "model": "test",
            }
        )

        result = await generator._expand_attributes(node, "type", ["a", "b"])
        assert result == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Full generator tests
# ---------------------------------------------------------------------------


class TestTopicTreeGenerator:
    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.complete = AsyncMock()
        llm.complete_batch = AsyncMock()
        return llm

    def _pivot_response(self, samples: list[str]) -> dict:
        return {
            "content": json.dumps(samples),
            "usage": {"prompt_tokens": 50, "completion_tokens": 50},
            "model": "test",
        }

    def _dimension_response(
        self, dimension: str, attributes: dict[str, list[int]]
    ) -> dict:
        return {
            "content": json.dumps(
                {"dimension": dimension, "attributes": attributes}
            ),
            "usage": {"prompt_tokens": 50, "completion_tokens": 50},
            "model": "test",
        }

    def _expand_null_response(self) -> dict:
        return {
            "content": "null",
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            "model": "test",
        }

    def _instruction_response(self, instructions: list[str]) -> dict:
        return {
            "content": json.dumps(instructions),
            "usage": {"prompt_tokens": 50, "completion_tokens": 30},
            "model": "test",
        }

    @pytest.mark.asyncio
    async def test_generate_produces_samples(self, mock_llm):
        """Full flow: tree construction → leaf generation → samples."""
        # With max_depth=1, we get: root pivot → dimension → expand → leaf gen
        pivot = self._pivot_response(["s1", "s2", "s3"])
        dim = self._dimension_response("area", {"math": [0], "code": [1, 2]})
        expand_null = self._expand_null_response()
        # Depth 1 children: each gets pivot → dimension fails (returns bad json)
        child_pivot = self._pivot_response(["c1", "c2"])
        dim_fail = {
            "content": "invalid",
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            "model": "test",
        }
        inst = self._instruction_response(["What is 2+2?"])

        # Call sequence for max_depth=1:
        # 1. root pivot
        # 2. root dimension discovery
        # 3. root attribute expansion
        # Then the 2 children at depth 1 are NOT expanded further (max_depth=1)
        # so they become leaves. Then leaf generation for each task.
        mock_llm.complete = AsyncMock(
            side_effect=[pivot, dim, expand_null] + [inst] * 20
        )

        gen = TopicTreeGenerator(
            domain="test",
            task_types=["qa"],
            languages=["en"],
            difficulty_range=(1, 1),
            max_depth=1,
            num_samples_per_node=3,
            seed=42,
            llm=mock_llm,
        )

        batches = []
        async for batch in gen.generate(n=3):
            batches.append(batch)

        all_samples = [s for b in batches for s in b]
        assert len(all_samples) == 3
        assert all(s.domain == "test" for s in all_samples)
        assert all(s.instruction for s in all_samples)

    @pytest.mark.asyncio
    async def test_failed_root_still_generates(self, mock_llm):
        """If pivot generation fails at root, root becomes leaf."""
        # Pivot returns invalid JSON
        bad_pivot = {
            "content": "not json",
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            "model": "test",
        }
        inst = self._instruction_response(["Explain gravity"])

        mock_llm.complete = AsyncMock(
            side_effect=[bad_pivot] + [inst] * 10
        )

        gen = TopicTreeGenerator(
            domain="science",
            task_types=["explain"],
            languages=["en"],
            difficulty_range=(1, 1),
            max_depth=2,
            num_samples_per_node=3,
            seed=42,
            llm=mock_llm,
        )

        batches = []
        async for batch in gen.generate(n=2):
            batches.append(batch)

        all_samples = [s for b in batches for s in b]
        assert len(all_samples) == 2

    @pytest.mark.asyncio
    async def test_seed_determinism(self, mock_llm):
        """Same seed should produce same leaf task assignment."""
        gen1 = TopicTreeGenerator(domain="test", seed=42, llm=mock_llm)
        gen2 = TopicTreeGenerator(domain="test", seed=42, llm=mock_llm)

        leaves = [
            TreeNode(depth=1, attribute_value="a"),
            TreeNode(depth=1, attribute_value="b"),
            TreeNode(depth=1, attribute_value="c"),
        ]

        plan1 = gen1._assign_leaf_tasks(leaves, 10)
        plan2 = gen2._assign_leaf_tasks(leaves, 10)

        assert [
            (t.leaf.attribute_value, t.task_type, t.difficulty) for t in plan1
        ] == [
            (t.leaf.attribute_value, t.task_type, t.difficulty) for t in plan2
        ]


# ---------------------------------------------------------------------------
# JSON parsing tests (kept from original)
# ---------------------------------------------------------------------------


class TestParseJson:
    def test_parse_direct(self):
        assert _parse_json('["a", "b"]') == ["a", "b"]

    def test_parse_markdown_fence(self):
        text = '```json\n["a", "b"]\n```'
        assert _parse_json(text) == ["a", "b"]

    def test_parse_extra_text(self):
        text = 'Here is the result: {"key": "val"} hope this helps'
        assert _parse_json(text) == {"key": "val"}
