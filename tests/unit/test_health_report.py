"""Tests for DatasetHealthReport."""

from syntheta.observability.health_report import DatasetHealthReport
from syntheta.schema.dataset import SynthDataset
from syntheta.schema.sample import Sample


class TestHealthReport:
    def _make_dataset(self) -> SynthDataset:
        samples = []
        for i in range(20):
            samples.append(
                Sample(
                    instruction=f"Question {i}",
                    response=f"Answer {i} with enough detail.",
                    domain="test",
                    topic=f"topic_{i % 5}",
                    task_type=["qa", "explain", "compare", "creative"][i % 4],
                    language=["en", "ar"][i % 2],
                    difficulty=(i % 5) + 1,
                    quality_score=0.5 + (i % 10) * 0.05,
                )
            )
        return SynthDataset(samples)

    def test_basic_report(self):
        ds = self._make_dataset()
        report = DatasetHealthReport.from_dataset(ds)
        data = report.to_dict()
        assert data["total_samples"] == 20
        assert data["coverage"]["topics"]["found"] == 5
        assert len(data["coverage"]["task_types"]) == 4
        assert len(data["coverage"]["languages"]) == 2

    def test_quality_stats(self):
        ds = self._make_dataset()
        report = DatasetHealthReport.from_dataset(ds)
        data = report.to_dict()
        assert "mean" in data["quality"]
        assert 0.0 <= data["quality"]["mean"] <= 1.0

    def test_display_format(self):
        ds = self._make_dataset()
        report = DatasetHealthReport.from_dataset(ds)
        output = report.display()
        assert "Dataset Health Report" in output
        assert "20" in output
        assert "Topics:" in output

    def test_warnings_skewed_difficulty(self):
        # Create heavily skewed dataset
        samples = [Sample(instruction=f"Q{i}", difficulty=1) for i in range(80)] + [
            Sample(instruction=f"Q{i}", difficulty=5) for i in range(20)
        ]
        ds = SynthDataset(samples)
        report = DatasetHealthReport.from_dataset(ds)
        data = report.to_dict()
        assert any("skewed" in w for w in data["warnings"])

    def test_save(self, tmp_path):
        ds = self._make_dataset()
        report = DatasetHealthReport.from_dataset(ds)
        path = tmp_path / "health.json"
        report.save(path)
        assert path.exists()

    def test_empty_dataset(self):
        ds = SynthDataset()
        report = DatasetHealthReport.from_dataset(ds)
        data = report.to_dict()
        assert data["total_samples"] == 0
