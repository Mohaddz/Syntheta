"""Cost tracking: per-stage token accumulation and pricing lookup."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from importlib import resources
from typing import Any

logger = logging.getLogger("syntheta.cost")


class CostTracker:
    """Tracks token usage per pipeline stage and computes costs from pricing data."""

    def __init__(self, pricing_override: dict[str, float] | None = None) -> None:
        """Initialize cost tracker.

        Args:
            pricing_override: User-provided pricing dict with keys
                'prompt_per_million' and 'completion_per_million'.
                These are converted to per-token internally.
        """
        # Convert user override from per-million to per-token for consistency
        self._pricing_override: dict[str, float] | None = None
        if pricing_override:
            self._pricing_override = {
                "input_cost_per_token": pricing_override["prompt_per_million"] / 1_000_000,
                "output_cost_per_token": pricing_override["completion_per_million"] / 1_000_000,
            }
        self._model_prices = _load_model_prices()
        self._usage: dict[str, dict[str, int]] = defaultdict(
            lambda: {"prompt_tokens": 0, "completion_tokens": 0}
        )

    def add_usage(self, stage: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Record token usage for a pipeline stage."""
        self._usage[stage]["prompt_tokens"] += prompt_tokens
        self._usage[stage]["completion_tokens"] += completion_tokens

    @property
    def total_prompt_tokens(self) -> int:
        return sum(u["prompt_tokens"] for u in self._usage.values())

    @property
    def total_completion_tokens(self) -> int:
        return sum(u["completion_tokens"] for u in self._usage.values())

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def get_cost(self, model: str) -> dict[str, Any]:
        """Compute cost breakdown by stage.

        Returns a dict with total cost and per-stage breakdown.
        """
        pricing = self._resolve_pricing(model)
        if pricing is None:
            logger.warning(
                "Model '%s' not found in pricing database. "
                "Set llm.pricing in config for cost estimates.",
                model,
            )
            return {
                "model": model,
                "total_tokens": self.total_tokens,
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "total_cost": None,
                "stages": {
                    stage: {
                        "prompt_tokens": u["prompt_tokens"],
                        "completion_tokens": u["completion_tokens"],
                        "cost": None,
                    }
                    for stage, u in self._usage.items()
                },
            }

        prompt_rate = pricing["input_cost_per_token"]
        completion_rate = pricing["output_cost_per_token"]

        stages = {}
        total_cost = 0.0
        for stage, u in self._usage.items():
            cost = u["prompt_tokens"] * prompt_rate + u["completion_tokens"] * completion_rate
            total_cost += cost
            stages[stage] = {
                "prompt_tokens": u["prompt_tokens"],
                "completion_tokens": u["completion_tokens"],
                "cost": round(cost, 4),
            }

        return {
            "model": model,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_cost": round(total_cost, 4),
            "stages": stages,
        }

    def _resolve_pricing(self, model: str) -> dict[str, float] | None:
        """Resolve pricing: user override > bundled prices (fuzzy) > None.

        Tries multiple lookup strategies since providers use different naming:
        - Exact match: "openai/gpt-4o-mini"
        - Strip provider prefix: "gpt-4o-mini" (from "openai/gpt-4o-mini")
        - Strip `:free` suffix: "google/gemma-3-4b-it" (from "google/gemma-3-4b-it:free")
        - Any key containing the base model name as substring
        """
        if self._pricing_override:
            return self._pricing_override

        # 1. Exact match
        if model in self._model_prices:
            return self._model_prices[model]

        # 2. Strip :free or :extended suffix
        base_model = model.split(":")[0] if ":" in model else model
        if base_model != model and base_model in self._model_prices:
            return self._model_prices[base_model]

        # 3. Strip provider prefix (openai/gpt-4o -> gpt-4o)
        if "/" in base_model:
            short_name = base_model.split("/", 1)[1]
            if short_name in self._model_prices:
                return self._model_prices[short_name]

            # 4. Find any key ending with the short model name
            for key, pricing in self._model_prices.items():
                if key.endswith("/" + short_name) or key == short_name:
                    return pricing

        return None

    def reset(self) -> None:
        """Clear all accumulated usage."""
        self._usage.clear()


def _load_model_prices() -> dict[str, dict[str, float]]:
    """Load bundled model_prices.json."""
    try:
        price_file = resources.files("syntheta.llm").joinpath("model_prices.json")
        return json.loads(price_file.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Could not load bundled model_prices.json, using empty pricing.")
        return {}
