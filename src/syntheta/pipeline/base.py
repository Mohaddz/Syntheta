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

    @abstractmethod
    async def generate(self, n: int, batch_size: int = 100) -> AsyncIterator[list[Sample]]:
        """Yield batches of samples.

        Args:
            n: Total number of samples to generate.
            batch_size: Number of samples per batch.
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
