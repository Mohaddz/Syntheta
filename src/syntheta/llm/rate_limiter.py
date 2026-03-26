"""Adaptive rate limiter with dynamic concurrency control."""

from __future__ import annotations

import asyncio
import logging
import time

from syntheta.exceptions import PersistentRateLimitError

logger = logging.getLogger("syntheta.rate_limit")


class AdaptiveRateLimiter:
    """Manages concurrency via asyncio.Semaphore with adaptive backoff on rate limits.

    On 429/RateLimitError:
      - Reduces concurrency by half (minimum 1)
      - Backs off for retry_after seconds (or exponential backoff)
      - Logs event with timestamp and duration

    On success after rate limit:
      - Gradually ramps concurrency back up toward max_concurrent

    If cumulative backoff exceeds max_backoff, raises PersistentRateLimitError.
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        max_backoff: float = 300.0,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.max_backoff = max_backoff
        self._current_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._cumulative_backoff = 0.0
        self._last_rate_limit_time: float | None = None
        self._consecutive_successes = 0

    @property
    def current_concurrent(self) -> int:
        return self._current_concurrent

    async def acquire(self) -> None:
        """Acquire a slot before making an LLM call."""
        await self._semaphore.acquire()

    def release(self) -> None:
        """Release a slot after an LLM call completes."""
        self._semaphore.release()

    async def on_rate_limit(self, retry_after: float | None = None) -> None:
        """Called when a rate limit (429) is hit. Reduces concurrency and backs off.

        Args:
            retry_after: Seconds to wait (from Retry-After header), or None for exponential.
        """
        if retry_after is None:
            # Exponential backoff: 1, 2, 4, 8, ... seconds
            backoff_count = sum(1 for _ in range(10) if self._cumulative_backoff > 2**_)
            retry_after = min(2**backoff_count, 60.0)

        self._cumulative_backoff += retry_after

        if self._cumulative_backoff > self.max_backoff:
            raise PersistentRateLimitError(
                f"Rate limits have persisted for {self._cumulative_backoff:.0f}s "
                f"(exceeds max_backoff={self.max_backoff}s). "
                "Consider reducing max_concurrent or waiting before retrying."
            )

        # Reduce concurrency by half (minimum 1)
        new_concurrent = max(1, self._current_concurrent // 2)
        if new_concurrent < self._current_concurrent:
            logger.warning(
                "Rate limit hit. Reducing concurrency %d -> %d. "
                "Backing off %.1fs (cumulative: %.1fs)",
                self._current_concurrent,
                new_concurrent,
                retry_after,
                self._cumulative_backoff,
            )
            self._current_concurrent = new_concurrent
            self._rebuild_semaphore()

        self._last_rate_limit_time = time.monotonic()
        self._consecutive_successes = 0

        await asyncio.sleep(retry_after)

    def on_success(self) -> None:
        """Called after a successful LLM call. Gradually ramps concurrency back up."""
        self._consecutive_successes += 1

        # Reset cumulative backoff on sustained success
        if self._consecutive_successes >= 10:
            self._cumulative_backoff = 0.0

        # Ramp up after every 5 consecutive successes
        if self._current_concurrent < self.max_concurrent and self._consecutive_successes % 5 == 0:
            self._current_concurrent = min(self._current_concurrent + 1, self.max_concurrent)
            self._rebuild_semaphore()
            logger.info(
                "Ramping concurrency back up to %d (max: %d)",
                self._current_concurrent,
                self.max_concurrent,
            )

    def _rebuild_semaphore(self) -> None:
        """Rebuild the semaphore with the new concurrency limit."""
        self._semaphore = asyncio.Semaphore(self._current_concurrent)
