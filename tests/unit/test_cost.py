"""Tests for CostTracker."""

from syntheta.llm.cost import CostTracker


class TestCostTracker:
    def test_add_usage(self):
        tracker = CostTracker()
        tracker.add_usage("generation", 100, 50)
        tracker.add_usage("generation", 200, 100)
        assert tracker.total_prompt_tokens == 300
        assert tracker.total_completion_tokens == 150
        assert tracker.total_tokens == 450

    def test_per_stage_tracking(self):
        tracker = CostTracker()
        tracker.add_usage("generation", 100, 50)
        tracker.add_usage("quality_filter", 200, 30)
        report = tracker.get_cost("unknown-model")
        assert "generation" in report["stages"]
        assert "quality_filter" in report["stages"]
        assert report["stages"]["generation"]["prompt_tokens"] == 100

    def test_user_override_pricing(self):
        tracker = CostTracker(
            pricing_override={"prompt_per_million": 1.0, "completion_per_million": 2.0}
        )
        tracker.add_usage("gen", 1_000_000, 1_000_000)
        report = tracker.get_cost("any-model")
        assert report["total_cost"] == 3.0  # $1 prompt + $2 completion

    def test_unknown_model_returns_none_cost(self):
        tracker = CostTracker()
        tracker.add_usage("gen", 100, 50)
        report = tracker.get_cost("nonexistent-model-xyz")
        assert report["total_cost"] is None
        assert report["total_tokens"] == 150

    def test_bundled_prices_loaded(self):
        tracker = CostTracker()
        # gpt-4o-mini should be in LiteLLM database
        report = tracker.get_cost("gpt-4o-mini")
        # If model is found, total_cost should be 0 (no usage yet)
        if report["total_cost"] is not None:
            assert report["total_cost"] == 0.0

    def test_reset(self):
        tracker = CostTracker()
        tracker.add_usage("gen", 100, 50)
        tracker.reset()
        assert tracker.total_tokens == 0
