"""Tests for TopicTreeGenerator with mocked LLM."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from syntheta.generators.topic_tree import TopicNode, TopicTreeGenerator, _parse_tree


class TestTopicNode:
    def test_flatten_leaf(self):
        node = TopicNode(name="leaf")
        assert node.flatten() == ["leaf"]

    def test_flatten_tree(self):
        tree = TopicNode(
            name="root",
            subtopics=[
                TopicNode(name="a", subtopics=[TopicNode(name="a1"), TopicNode(name="a2")]),
                TopicNode(name="b"),
            ],
        )
        leaves = tree.flatten()
        assert set(leaves) == {"a1", "a2", "b"}


class TestParseTree:
    def test_parse_simple_tree(self):
        data = {
            "topics": [
                {
                    "name": "History",
                    "description": "Historical events",
                    "subtopics": [
                        {"name": "Ancient", "description": "Ancient history", "subtopics": []},
                    ],
                },
                {
                    "name": "Science",
                    "description": "Scientific topics",
                    "subtopics": [],
                },
            ]
        }
        tree = _parse_tree(data)
        assert len(tree.subtopics) == 2
        leaves = tree.flatten()
        assert "Ancient" in leaves
        assert "Science" in leaves


class TestTopicTreeGenerator:
    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.complete = AsyncMock()
        llm.complete_batch = AsyncMock()
        return llm

    @pytest.fixture
    def tree_response(self):
        tree = {
            "topics": [
                {
                    "name": "Algorithms",
                    "description": "Computer algorithms",
                    "subtopics": [
                        {"name": "Sorting", "description": "Sorting algorithms", "subtopics": []},
                        {"name": "Search", "description": "Search algorithms", "subtopics": []},
                    ],
                },
                {
                    "name": "Data Structures",
                    "description": "Data organization",
                    "subtopics": [
                        {"name": "Trees", "description": "Tree structures", "subtopics": []},
                    ],
                },
            ]
        }
        return {
            "content": json.dumps(tree),
            "usage": {"prompt_tokens": 100, "completion_tokens": 200},
            "model": "test",
        }

    @pytest.fixture
    def instruction_response(self):
        return {
            "content": '["What is bubble sort?"]',
            "usage": {"prompt_tokens": 50, "completion_tokens": 30},
            "model": "test",
        }

    @pytest.mark.asyncio
    async def test_generate_produces_samples(self, mock_llm, tree_response, instruction_response):
        mock_llm.complete = AsyncMock(side_effect=[tree_response] + [instruction_response] * 100)

        gen = TopicTreeGenerator(
            domain="Computer Science",
            task_types=["qa"],
            languages=["en"],
            difficulty_range=(1, 2),
            topic_depth=2,
            topic_breadth=2,
            seed=42,
            llm=mock_llm,
        )

        batches = []
        async for batch in gen.generate(n=5):
            batches.append(batch)

        all_samples = [s for b in batches for s in b]
        assert len(all_samples) == 5
        assert all(s.domain == "Computer Science" for s in all_samples)
        assert all(s.instruction for s in all_samples)
        assert all(s.task_type == "qa" for s in all_samples)

    @pytest.mark.asyncio
    async def test_distribution_is_seeded(self, mock_llm, tree_response, instruction_response):
        mock_llm.complete = AsyncMock(side_effect=[tree_response] + [instruction_response] * 100)

        gen1 = TopicTreeGenerator(domain="Test", seed=42, llm=mock_llm)
        plan1 = gen1._plan_distribution(["a", "b", "c"], 10)

        gen2 = TopicTreeGenerator(domain="Test", seed=42, llm=mock_llm)
        plan2 = gen2._plan_distribution(["a", "b", "c"], 10)

        # Same seed → same plan
        assert [(p.topic, p.task_type, p.difficulty) for p in plan1] == [
            (p.topic, p.task_type, p.difficulty) for p in plan2
        ]
