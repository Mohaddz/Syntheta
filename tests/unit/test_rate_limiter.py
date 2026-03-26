"""Tests for AdaptiveRateLimiter."""

import pytest

from syntheta.exceptions import PersistentRateLimitError
from syntheta.llm.rate_limiter import AdaptiveRateLimiter


class TestAdaptiveRateLimiter:
    @pytest.fixture
    def limiter(self):
        return AdaptiveRateLimiter(max_concurrent=10, max_backoff=10.0)

    @pytest.mark.asyncio
    async def test_initial_concurrency(self, limiter):
        assert limiter.current_concurrent == 10

    @pytest.mark.asyncio
    async def test_acquire_release(self, limiter):
        await limiter.acquire()
        limiter.release()

    @pytest.mark.asyncio
    async def test_rate_limit_reduces_concurrency(self, limiter):
        await limiter.on_rate_limit(retry_after=0.01)  # tiny sleep
        assert limiter.current_concurrent == 5

    @pytest.mark.asyncio
    async def test_double_rate_limit(self, limiter):
        await limiter.on_rate_limit(retry_after=0.01)
        assert limiter.current_concurrent == 5
        await limiter.on_rate_limit(retry_after=0.01)
        assert limiter.current_concurrent == 2

    @pytest.mark.asyncio
    async def test_minimum_concurrency_is_one(self):
        limiter = AdaptiveRateLimiter(max_concurrent=2, max_backoff=100.0)
        await limiter.on_rate_limit(retry_after=0.01)
        assert limiter.current_concurrent == 1
        await limiter.on_rate_limit(retry_after=0.01)
        assert limiter.current_concurrent == 1

    @pytest.mark.asyncio
    async def test_ramp_up_on_success(self, limiter):
        await limiter.on_rate_limit(retry_after=0.01)
        assert limiter.current_concurrent == 5
        # 5 successes triggers one ramp-up
        for _ in range(5):
            limiter.on_success()
        assert limiter.current_concurrent == 6

    @pytest.mark.asyncio
    async def test_persistent_rate_limit_raises(self):
        limiter = AdaptiveRateLimiter(max_concurrent=10, max_backoff=0.05)
        with pytest.raises(PersistentRateLimitError, match="persisted"):
            await limiter.on_rate_limit(retry_after=0.1)

    @pytest.mark.asyncio
    async def test_cumulative_backoff_reset_on_sustained_success(self, limiter):
        await limiter.on_rate_limit(retry_after=0.01)
        assert limiter._cumulative_backoff > 0
        for _ in range(10):
            limiter.on_success()
        assert limiter._cumulative_backoff == 0.0
