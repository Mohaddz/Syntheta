"""Tests for seeding utilities."""

from syntheta.utils.seeding import create_rng, deterministic_shuffle


class TestSeeding:
    def test_same_seed_same_output(self):
        rng1 = create_rng(42)
        rng2 = create_rng(42)
        assert [rng1.randint(0, 100) for _ in range(10)] == [
            rng2.randint(0, 100) for _ in range(10)
        ]

    def test_different_seed_different_output(self):
        rng1 = create_rng(42)
        rng2 = create_rng(99)
        seq1 = [rng1.randint(0, 1000) for _ in range(10)]
        seq2 = [rng2.randint(0, 1000) for _ in range(10)]
        assert seq1 != seq2

    def test_deterministic_shuffle(self):
        items1 = list(range(20))
        items2 = list(range(20))
        deterministic_shuffle(items1, create_rng(42))
        deterministic_shuffle(items2, create_rng(42))
        assert items1 == items2

    def test_no_seed_is_random(self):
        rng = create_rng(None)
        # Should work without error
        assert isinstance(rng.randint(0, 100), int)
