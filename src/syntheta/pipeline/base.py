"""Abstract base classes for pipeline components: Generator, Transformer, Filter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from syntheta.llm.backend import OpenAICompatibleLLM
    from syntheta.schema.sample import Sample


class BaseGenerator(ABC):
    """Generates samples from nothing or from a source. Yields batches (streaming)."""

    def __init__(self, llm: OpenAICompatibleLLM | None = None, **kwargs) -> None:
        self.llm = llm

    @property
    def max_concurrent(self) -> int:
        """Max concurrent LLM calls from the rate limiter. Used as sliding window size."""
        try:
            val = self.llm.rate_limiter.max_concurrent
            if isinstance(val, int):
                return val
        except (AttributeError, TypeError):
            pass
        return 10

    @abstractmethod
    async def generate(self, n: int) -> AsyncIterator[list[Sample]]:
        """Yield batches of samples.

        Args:
            n: Total number of samples to generate.
        """
        yield []  # type: ignore[misc]


class BaseTransformer(ABC):
    """Takes a batch of samples and returns transformed samples."""

    def __init__(self, llm: OpenAICompatibleLLM | None = None, **kwargs) -> None:
        self.llm = llm

    @abstractmethod
    async def transform(self, samples: list[Sample]) -> list[Sample]:
        """Transform a batch of samples. Returns transformed samples."""
        ...


class BaseFilter(ABC):
    """Scores and removes samples from a batch. Tracks rejection reasons."""

    rejection_reason: str = "filtered"

    def __init__(self, llm: OpenAICompatibleLLM | None = None, **kwargs) -> None:
        self.llm = llm

    @abstractmethod
    async def filter(self, samples: list[Sample]) -> list[Sample]:
        """Filter a batch, returning only passing samples."""
        ...
