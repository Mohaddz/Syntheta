"""OpenAI-compatible LLM backend wrapping AsyncOpenAI with max_retries=0."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import openai
from openai import AsyncOpenAI

from syntheta.exceptions import (
    AuthenticationError,
    LLMError,
    ModelNotFoundError,
)
from syntheta.llm.cost import CostTracker
from syntheta.llm.rate_limiter import AdaptiveRateLimiter

logger = logging.getLogger("syntheta.llm")


class OpenAICompatibleLLM:
    """Unified LLM backend for all Syntheta operations.

    Wraps AsyncOpenAI with max_retries=0 so we control retry/concurrency ourselves.
    Supports three model roles: model, response_model, embedding_model.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        response_model: str | None = None,
        embedding_model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        max_concurrent: int = 10,
        max_retries: int = 3,
        timeout: float = 60.0,
        max_backoff: float = 300.0,
        pricing: dict[str, float] | None = None,
    ) -> None:
        # Resolve API key
        resolved_key = api_key or os.environ.get(api_key_env)
        if not resolved_key:
            raise AuthenticationError(
                f"No API key provided. Set the {api_key_env} environment variable "
                "or pass api_key directly."
            )

        self.model = model
        self.response_model = response_model
        self.embedding_model = embedding_model
        self.max_retries = max_retries

        self._client = AsyncOpenAI(
            api_key=resolved_key,
            base_url=base_url,
            max_retries=0,  # We handle retries ourselves
            timeout=timeout,
        )

        self.rate_limiter = AdaptiveRateLimiter(
            max_concurrent=max_concurrent,
            max_backoff=max_backoff,
        )
        self.cost_tracker = CostTracker(pricing_override=pricing)

    def _resolve_model(self, model_role: str) -> str:
        """Resolve which model to use based on role. response_model falls back to model."""
        if model_role == "response_model":
            return self.response_model or self.model
        if model_role == "embedding_model":
            if self.embedding_model:
                return self.embedding_model
            raise LLMError("No embedding_model configured. Set llm.embedding_model in config.")
        return self.model

    async def complete(
        self,
        messages: list[dict[str, str]],
        model_role: str = "model",
        stage: str = "unknown",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a chat completion request with retry and rate limiting.

        Args:
            messages: List of message dicts (role, content).
            model_role: Which model role to use (model, response_model).
            stage: Pipeline stage name for cost tracking.
            **kwargs: Extra args passed to chat.completions.create.

        Returns:
            Dict with 'content', 'usage', 'model' keys.
        """
        model = self._resolve_model(model_role)
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            await self.rate_limiter.acquire()
            try:
                response = await self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs,
                )
                self.rate_limiter.release()
                self.rate_limiter.on_success()

                # Track usage
                if response.usage:
                    self.cost_tracker.add_usage(
                        stage,
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens,
                    )

                content = response.choices[0].message.content if response.choices else None
                return {
                    "content": content or "",
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": (
                            response.usage.completion_tokens if response.usage else 0
                        ),
                    },
                    "model": response.model,
                }

            except openai.AuthenticationError as e:
                self.rate_limiter.release()
                raise AuthenticationError(f"Authentication failed: {e}") from e

            except openai.NotFoundError as e:
                self.rate_limiter.release()
                raise ModelNotFoundError(
                    f"Model '{model}' not found. Check llm.model in config: {e}"
                ) from e

            except openai.RateLimitError as e:
                self.rate_limiter.release()
                retry_after = _extract_retry_after(e)
                logger.warning(
                    "Rate limited (attempt %d/%d). Retry-after: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    retry_after,
                )
                await self.rate_limiter.on_rate_limit(retry_after)
                last_error = e

            except (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
            ) as e:
                self.rate_limiter.release()
                last_error = e
                if attempt < self.max_retries:
                    backoff = min(2**attempt, 30.0)
                    logger.warning(
                        "Transient error (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt + 1,
                        self.max_retries + 1,
                        type(e).__name__,
                        backoff,
                    )
                    await asyncio.sleep(backoff)

        raise LLMError(f"All {self.max_retries + 1} attempts failed: {last_error}") from last_error

    async def complete_batch(
        self,
        message_batches: list[list[dict[str, str]]],
        model_role: str = "model",
        stage: str = "unknown",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Send multiple chat completions concurrently with rate limiting.

        Args:
            message_batches: List of message lists, one per call.
            model_role: Which model role to use.
            stage: Pipeline stage name for cost tracking.

        Returns:
            List of result dicts (same order as input).
        """
        tasks = [
            self.complete(messages, model_role=model_role, stage=stage, **kwargs)
            for messages in message_batches
        ]
        return await asyncio.gather(*tasks)

    async def embed(
        self,
        texts: list[str],
        stage: str = "unknown",
    ) -> list[list[float]]:
        """Get embeddings for a list of texts.

        Args:
            texts: List of strings to embed.
            stage: Pipeline stage name for cost tracking.

        Returns:
            List of embedding vectors.
        """
        model = self._resolve_model("embedding_model")
        await self.rate_limiter.acquire()
        try:
            response = await self._client.embeddings.create(
                model=model,
                input=texts,
            )
            self.rate_limiter.release()
            self.rate_limiter.on_success()

            if response.usage:
                self.cost_tracker.add_usage(stage, response.usage.prompt_tokens, 0)

            return [item.embedding for item in response.data]

        except openai.AuthenticationError as e:
            self.rate_limiter.release()
            raise AuthenticationError(f"Authentication failed: {e}") from e
        except openai.NotFoundError as e:
            self.rate_limiter.release()
            raise ModelNotFoundError(f"Embedding model '{model}' not found: {e}") from e
        except Exception as e:
            self.rate_limiter.release()
            raise LLMError(f"Embedding request failed: {e}") from e

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()


def _extract_retry_after(error: openai.RateLimitError) -> float | None:
    """Try to extract retry-after seconds from a rate limit error."""
    if hasattr(error, "response") and error.response is not None:
        header = error.response.headers.get("retry-after")
        if header:
            try:
                return float(header)
            except (ValueError, TypeError):
                pass
    return None
