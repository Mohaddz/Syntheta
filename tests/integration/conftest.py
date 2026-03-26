"""Integration test fixtures using OpenRouter."""

from __future__ import annotations

import os

import pytest


@pytest.fixture
def openrouter_llm():
    """Create an OpenAICompatibleLLM pointed at OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set")

    from syntheta.llm.backend import OpenAICompatibleLLM

    return OpenAICompatibleLLM(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model="google/gemma-3-4b-it:free",
        max_concurrent=2,
        max_retries=3,
        timeout=60.0,
    )
