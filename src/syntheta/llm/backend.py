"""OpenAI-compatible LLM backend wrapping AsyncOpenAI with max_retries=0."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from typing import Any

import httpx
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

# Callback: (call_id, stage, token_text) -> None
# call_id is unique per LLM call so the display can track multiple concurrent streams
OnTokenCallback = Callable[[str, str, str], None]


class OpenAICompatibleLLM:
    """Unified LLM backend for all Syntheta operations.

    Wraps AsyncOpenAI with max_retries=0 so we control retry/concurrency ourselves.
    Supports three model roles: model, response_model, embedding_model.

    When on_token is set, completions use streaming mode and fire the callback
    for each token as it arrives, enabling real-time display of generation.
    """

    _call_counter: int = 0

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
        disable_thinking: bool = False,
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
        self.disable_thinking = disable_thinking

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

        # Optional callback for real-time token streaming to display
        self.on_token: OnTokenCallback | None = None

        # Live stats for verbose display
        self.active_calls = 0
        self.total_calls = 0
        self.total_retries = 0
        self.total_errors = 0

    def _resolve_model(self, model_role: str) -> str:
        """Resolve which model to use based on role. response_model falls back to model."""
        if model_role == "response_model":
            return self.response_model or self.model
        if model_role == "embedding_model":
            if self.embedding_model:
                return self.embedding_model
            raise LLMError("No embedding_model configured. Set llm.embedding_model in config.")
        return self.model

    def _next_call_id(self) -> str:
        OpenAICompatibleLLM._call_counter += 1
        return f"call_{OpenAICompatibleLLM._call_counter}"

    async def complete(
        self,
        messages: list[dict[str, str]],
        model_role: str = "model",
        stage: str = "unknown",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a chat completion request with retry and rate limiting.

        When on_token is set, uses streaming mode and fires the callback
        for each token chunk as it arrives from the API.

        Returns:
            Dict with 'content', 'usage', 'model' keys.
        """
        model = self._resolve_model(model_role)
        use_streaming = self.on_token is not None
        last_error: Exception | None = None

        # Inject disable_thinking via extra_body if configured
        if self.disable_thinking:
            extra = kwargs.pop("extra_body", {}) or {}
            extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
            kwargs["extra_body"] = extra

        self.total_calls += 1

        for attempt in range(self.max_retries + 1):
            await self.rate_limiter.acquire()
            self.active_calls += 1
            try:
                if use_streaming:
                    result = await self._complete_streaming(
                        model, messages, stage, **kwargs
                    )
                else:
                    result = await self._complete_standard(
                        model, messages, stage, **kwargs
                    )

                self.rate_limiter.release()
                self.active_calls -= 1
                self.rate_limiter.on_success()
                return result

            except openai.AuthenticationError as e:
                self.rate_limiter.release()
                self.active_calls -= 1
                self.total_errors += 1
                raise AuthenticationError(f"Authentication failed: {e}") from e

            except openai.NotFoundError as e:
                self.rate_limiter.release()
                self.active_calls -= 1
                self.total_errors += 1
                raise ModelNotFoundError(
                    f"Model '{model}' not found. Check llm.model in config: {e}"
                ) from e

            except openai.RateLimitError as e:
                self.rate_limiter.release()
                self.active_calls -= 1
                self.total_retries += 1
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
                openai.APIError,
            ) as e:
                self.rate_limiter.release()
                self.active_calls -= 1
                self.total_retries += 1
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

        self.total_errors += 1
        raise LLMError(
            f"All {self.max_retries + 1} attempts failed: {last_error}"
        ) from last_error

    async def _complete_standard(
        self,
        model: str,
        messages: list[dict[str, str]],
        stage: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Non-streaming completion."""
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )

        if response.usage:
            self.cost_tracker.add_usage(
                stage,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )

        content = _extract_content(response)
        return {
            "content": content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": (
                    response.usage.completion_tokens if response.usage else 0
                ),
            },
            "model": response.model,
        }

    async def _complete_streaming(
        self,
        model: str,
        messages: list[dict[str, str]],
        stage: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Streaming completion -- fires on_token callback for each chunk."""
        call_id = self._next_call_id()
        accumulated = []
        response_model_name = model
        prompt_tokens = 0
        completion_tokens = 0

        stream = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **kwargs,
        )

        try:
            async for chunk in stream:
                # Extract token content (check content first, fallback to reasoning)
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta:
                        token = delta.content or getattr(delta, "reasoning", None) or ""
                        if token:
                            accumulated.append(token)
                            if self.on_token:
                                self.on_token(call_id, stage, "".join(accumulated))

                # Extract model name from first chunk
                if chunk.model:
                    response_model_name = chunk.model

                # Extract usage from final chunk (stream_options=include_usage)
                if chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
        except (httpx.ReadError, httpx.RemoteProtocolError, httpx.ReadTimeout) as e:
            # Network died mid-stream. Re-raise as APIConnectionError so the
            # retry loop in complete() catches it and retries the full call.
            raise openai.APIConnectionError(request=None) from e

        content = "".join(accumulated)

        # Track usage
        if prompt_tokens or completion_tokens:
            self.cost_tracker.add_usage(stage, prompt_tokens, completion_tokens)

        return {
            "content": content,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            "model": response_model_name,
        }

    async def complete_batch(
        self,
        message_batches: list[list[dict[str, str]]],
        model_role: str = "model",
        stage: str = "unknown",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Send multiple chat completions concurrently with rate limiting."""
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
        """Get embeddings for a list of texts."""
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


def _extract_content(response: Any) -> str:
    """Extract text content from a completion response.

    Checks content first, falls back to reasoning field (for thinking models
    served by vLLM where content is None and output is in reasoning).
    """
    if not response.choices:
        return ""

    msg = response.choices[0].message
    content = msg.content
    if content:
        return content

    # Fallback: check reasoning field (vLLM thinking models)
    reasoning = getattr(msg, "reasoning", None)
    if reasoning is None:
        # Also check model_dump for fields not in the SDK type
        try:
            msg_dict = msg.model_dump()
            reasoning = msg_dict.get("reasoning")
        except Exception:
            pass

    if reasoning:
        logger.warning(
            "Model returned thinking output only (content=None, reasoning present). "
            "Consider setting llm.disable_thinking=true or disabling thinking on your server."
        )
        return str(reasoning)

    return ""


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
