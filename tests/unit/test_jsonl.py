"""Tests for JSONL read/write utilities."""

import pytest

from syntheta.utils.jsonl import read_jsonl, write_jsonl


class TestReadJsonl:
    def test_read_valid_file(self, tmp_path):
        path = tmp_path / "test.jsonl"
        path.write_text('{"a": 1}\n{"a": 2}\n')
        records = list(read_jsonl(path))
        assert len(records) == 2
        assert records[0]["a"] == 1

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "test.jsonl"
        path.write_text('{"a": 1}\n\n\n{"a": 2}\n')
        records = list(read_jsonl(path))
        assert len(records) == 2

    def test_malformed_json_raises(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"a": 1}\nnot json\n')
        with pytest.raises(ValueError, match=r"Malformed JSON.*:2:"):
            list(read_jsonl(path))

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert list(read_jsonl(path)) == []


class TestWriteJsonl:
    def test_write_and_read_back(self, tmp_path):
        path = tmp_path / "out.jsonl"
        records = [{"x": 1}, {"x": 2}]
        count = write_jsonl(path, records, mode="w")
        assert count == 2
        loaded = list(read_jsonl(path))
        assert loaded == records

    def test_append_mode(self, tmp_path):
        path = tmp_path / "out.jsonl"
        write_jsonl(path, [{"a": 1}], mode="w")
        write_jsonl(path, [{"a": 2}], mode="a")
        loaded = list(read_jsonl(path))
        assert len(loaded) == 2

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "out.jsonl"
        write_jsonl(path, [{"a": 1}], mode="w")
        assert path.exists()

    def test_unicode(self, tmp_path):
        path = tmp_path / "unicode.jsonl"
        records = [{"text": "مرحبا بالعالم"}]
        write_jsonl(path, records, mode="w")
        loaded = list(read_jsonl(path))
        assert loaded[0]["text"] == "مرحبا بالعالم"
