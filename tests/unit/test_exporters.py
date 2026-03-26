"""Tests for typed exporters."""

import pytest

from syntheta.exceptions import ExportValidationError
from syntheta.schema.exporters import (
    export_chat,
    export_dpo,
    export_pretrain,
    export_rlvr,
    export_sft,
)
from syntheta.schema.sample import Sample, Turn


class TestExportSFT:
    def test_valid_export(self, sample_sft):
        result = export_sft([sample_sft])
        assert len(result) == 1
        assert result[0]["instruction"] == sample_sft.instruction
        assert result[0]["response"] == sample_sft.response

    def test_missing_response_raises(self):
        s = Sample(instruction="test")
        with pytest.raises(ExportValidationError, match="missing instruction or response"):
            export_sft([s])

    def test_short_response_raises(self):
        s = Sample(instruction="test", response="short")
        with pytest.raises(ExportValidationError, match="shorter than 10 chars"):
            export_sft([s])

    def test_rename(self, sample_sft):
        result = export_sft([sample_sft], rename={"instruction": "prompt"})
        assert "prompt" in result[0]
        assert "instruction" not in result[0]

    def test_includes_system_prompt(self):
        s = Sample(
            instruction="test",
            response="A sufficiently long response for the test.",
            system_prompt="You are helpful.",
        )
        result = export_sft([s])
        assert result[0]["system_prompt"] == "You are helpful."


class TestExportChat:
    def test_valid_export(self, sample_chat):
        result = export_chat([sample_chat])
        assert len(result) == 1
        assert len(result[0]["conversations"]) == 4

    def test_insufficient_turns_raises(self):
        s = Sample(turns=[Turn(role="user", content="hi")])
        with pytest.raises(ExportValidationError, match="insufficient turns"):
            export_chat([s])


class TestExportDPO:
    def test_valid_export(self, sample_dpo):
        result = export_dpo([sample_dpo])
        assert result[0]["prompt"] == sample_dpo.instruction
        assert result[0]["chosen"] == sample_dpo.chosen

    def test_identical_chosen_rejected_raises(self):
        s = Sample(instruction="test", chosen="same", rejected="same")
        with pytest.raises(ExportValidationError, match="identical"):
            export_dpo([s])

    def test_missing_fields_raises(self):
        s = Sample(instruction="test", chosen="good")
        with pytest.raises(ExportValidationError):
            export_dpo([s])


class TestExportRLVR:
    def test_valid_export(self, sample_rlvr):
        result = export_rlvr([sample_rlvr])
        assert result[0]["question"] == sample_rlvr.instruction
        assert result[0]["info"] == sample_rlvr.info

    def test_missing_info_raises(self):
        s = Sample(instruction="test")
        with pytest.raises(ExportValidationError):
            export_rlvr([s])


class TestExportPretrain:
    def test_valid_export(self, sample_pretrain):
        result = export_pretrain([sample_pretrain])
        assert result[0]["text"] == sample_pretrain.text

    def test_short_text_raises(self):
        s = Sample(text="too short")
        with pytest.raises(ExportValidationError, match="too short text"):
            export_pretrain([s])
