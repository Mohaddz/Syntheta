"""Integration tests for LLM backend against OpenRouter."""

import pytest


@pytest.mark.integration
class TestLLMBackendIntegration:
    @pytest.mark.asyncio
    async def test_simple_completion(self, openrouter_llm):
        result = await openrouter_llm.complete(
            messages=[{"role": "user", "content": "Say hello in one word."}],
            stage="test",
        )
        assert result["content"]
        assert len(result["content"]) > 0
        assert result["usage"]["prompt_tokens"] > 0
        assert result["usage"]["completion_tokens"] > 0

    @pytest.mark.asyncio
    async def test_batch_completion(self, openrouter_llm):
        results = await openrouter_llm.complete_batch(
            [
                [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
                [{"role": "user", "content": "What is 3+3? Answer with just the number."}],
            ],
            stage="test",
        )
        assert len(results) == 2
        assert all(r["content"] for r in results)

    @pytest.mark.asyncio
    async def test_cost_tracking(self, openrouter_llm):
        await openrouter_llm.complete(
            messages=[{"role": "user", "content": "Hi"}],
            stage="test_stage",
        )
        assert openrouter_llm.cost_tracker.total_tokens > 0
