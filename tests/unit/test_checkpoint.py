"""Tests for CheckpointManager."""

from syntheta.pipeline.checkpoint import CheckpointManager, CheckpointState


class TestCheckpointState:
    def test_round_trip(self):
        state = CheckpointState(
            total_generated=500,
            total_passed=400,
            config_hash="abc123",
            filter_stats={"quality": 80, "safety": 20},
        )
        data = state.to_dict()
        restored = CheckpointState.from_dict(data)
        assert restored.total_generated == 500
        assert restored.total_passed == 400
        assert restored.filter_stats["quality"] == 80

    def test_from_dict_ignores_unknown_fields(self):
        """Old checkpoints with batch_num/cost_usage should load without error."""
        data = {
            "batch_num": 5,
            "total_generated": 100,
            "total_passed": 90,
            "config_hash": "abc",
            "filter_stats": {},
            "cost_usage": {"tokens": 1000},
        }
        state = CheckpointState.from_dict(data)
        assert state.total_generated == 100
        assert state.total_passed == 90


class TestCheckpointManager:
    def test_save_and_load(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt")
        state = CheckpointState(total_passed=3, config_hash="hash1")
        mgr.save(state)
        loaded = mgr.load()
        assert loaded is not None
        assert loaded.total_passed == 3
        assert loaded.config_hash == "hash1"

    def test_load_no_checkpoint(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt")
        assert mgr.load() is None

    def test_verify_config_hash_match(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt")
        state = CheckpointState(config_hash="hash1")
        mgr.save(state)
        assert mgr.verify_config_hash("hash1") is True

    def test_verify_config_hash_mismatch(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt")
        state = CheckpointState(config_hash="hash1")
        mgr.save(state)
        assert mgr.verify_config_hash("hash2") is False

    def test_compute_config_hash_deterministic(self):
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 2, "a": 1}
        h1 = CheckpointManager.compute_config_hash(d1)
        h2 = CheckpointManager.compute_config_hash(d2)
        assert h1 == h2

    def test_compute_config_hash_different(self):
        h1 = CheckpointManager.compute_config_hash({"a": 1})
        h2 = CheckpointManager.compute_config_hash({"a": 2})
        assert h1 != h2

    def test_atomic_write_no_temp_file_remains(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt")
        state = CheckpointState(total_passed=10, config_hash="h")
        mgr.save(state)
        tmp_file = (tmp_path / "ckpt" / "state.json.tmp")
        assert not tmp_file.exists()
        assert (tmp_path / "ckpt" / "state.json").exists()
