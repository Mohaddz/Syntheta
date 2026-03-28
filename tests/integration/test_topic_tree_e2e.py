"""End-to-end integration test for TopicTreeGenerator with OpenRouter."""

import pytest

from syntheta.generators.topic_tree import TopicTreeGenerator


@pytest.mark.integration
class TestTopicTreeE2E:
    @pytest.mark.asyncio
    async def test_generate_small_dataset(self, openrouter_llm):
        """Generate 3 samples end-to-end with a real LLM."""
        gen = TopicTreeGenerator(
            domain="Python programming",
            task_types=["qa"],
            languages=["en"],
            difficulty_range=(1, 2),
            topic_depth=1,
            topic_breadth=2,
            seed=42,
            llm=openrouter_llm,
        )

        all_samples = []
        async for batch in gen.generate(n=3):
            all_samples.extend(batch)

        assert len(all_samples) >= 3
        for s in all_samples:
            assert s.instruction
            assert s.domain == "Python programming"
            assert s.task_type == "qa"
            assert s.language == "en"
