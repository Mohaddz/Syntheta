"""Tests for TemplateCollapseDetector."""

import pytest

from syntheta.filters.template_collapse import TemplateCollapseDetector
from syntheta.schema.sample import Sample


class TestTemplateCollapseDetector:
    def _make_samples(self, responses: list[str]) -> list[Sample]:
        return [Sample(instruction=f"Q{i}", response=r) for i, r in enumerate(responses)]

    @pytest.mark.asyncio
    async def test_no_collapse(self):
        detector = TemplateCollapseDetector(prefix_length=20, max_repeat=2)
        samples = self._make_samples(
            [
                "The answer to this question is unique.",
                "Here's a different response entirely.",
                "This one is also distinct from others.",
            ]
        )
        result = await detector.filter(samples)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_detect_collapse_report_mode(self):
        detector = TemplateCollapseDetector(prefix_length=20, max_repeat=2, action="report")
        responses = [
            "Here is the answer to your question about X",
            "Here is the answer to your question about Y",
            "Here is the answer to your question about Z",
            "Something completely different",
        ]
        samples = self._make_samples(responses)
        result = await detector.filter(samples)
        # Report mode keeps all samples
        assert len(result) == 4

    @pytest.mark.asyncio
    async def test_detect_collapse_remove_mode(self):
        detector = TemplateCollapseDetector(prefix_length=20, max_repeat=1, action="remove")
        responses = [
            "Here is the answer to your question about X",
            "Here is the answer to your question about Y",
            "Here is the answer to your question about Z",
            "Something completely different",
        ]
        samples = self._make_samples(responses)
        result = await detector.filter(samples)
        # max_repeat=1, so only 1 sample with the repeated prefix + the unique one
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_detect_collapse_flag_mode(self):
        detector = TemplateCollapseDetector(prefix_length=20, max_repeat=2, action="flag")
        responses = [
            "Repeated prefix response one.",
            "Repeated prefix response two.",
            "Repeated prefix response three.",
            "Unique response here.",
        ]
        samples = self._make_samples(responses)
        result = await detector.filter(samples)
        # Flag mode keeps all samples but sets metadata
        assert len(result) == 4
        flagged = [s for s in result if s.extra.get("template_collapse")]
        assert len(flagged) == 3

    @pytest.mark.asyncio
    async def test_empty_samples(self):
        detector = TemplateCollapseDetector()
        assert await detector.filter([]) == []

    @pytest.mark.asyncio
    async def test_uses_text_field_fallback(self):
        detector = TemplateCollapseDetector(prefix_length=10, max_repeat=1, action="remove")
        samples = [
            Sample(text="Same start here with different endings A"),
            Sample(text="Same start here with different endings B"),
            Sample(text="Same start here with different endings C"),
            Sample(text="Unique text"),
        ]
        result = await detector.filter(samples)
        assert len(result) == 2  # 1 from repeated prefix + 1 unique
