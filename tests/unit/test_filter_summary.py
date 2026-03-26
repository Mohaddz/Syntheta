"""Tests for FilterSummary."""

from syntheta.observability.filter_summary import FilterSummary


class TestFilterSummary:
    def test_basic_tracking(self):
        fs = FilterSummary()
        fs.record_generated(100)
        fs.record_passed(80)
        fs.record_rejection("quality", 15)
        fs.record_rejection("safety", 5)
        assert fs.total_generated == 100
        assert fs.total_passed == 80
        assert fs.total_rejected == 20
        assert fs.rejections["quality"] == 15
        assert fs.rejections["safety"] == 5

    def test_display_format(self):
        fs = FilterSummary()
        fs.record_generated(1000)
        fs.record_passed(800)
        fs.record_rejection("quality_below_threshold", 150)
        fs.record_rejection("safety_blocked", 50)

        output = fs.display()
        assert "1,000 samples generated" in output
        assert "800 samples passed" in output
        assert "quality_below_threshold" in output
        assert "safety_blocked" in output

    def test_to_dict(self):
        fs = FilterSummary()
        fs.record_generated(100)
        fs.record_passed(90)
        fs.record_rejection("test", 10)
        d = fs.to_dict()
        assert d["total_generated"] == 100
        assert d["total_passed"] == 90
        assert d["rejections"]["test"] == 10

    def test_empty_summary(self):
        fs = FilterSummary()
        output = fs.display()
        assert "0 samples generated" in output

    def test_warnings(self):
        fs = FilterSummary()
        fs.record_generated(10)
        fs.record_passed(10)
        fs.record_warning("Test warning message")
        output = fs.display()
        assert "Test warning message" in output
