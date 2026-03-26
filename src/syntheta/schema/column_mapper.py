"""ColumnMapper: auto-detect input format and map columns to Sample fields."""

from __future__ import annotations

from typing import Any

from syntheta.schema.sample import Sample, Turn

# Known format signatures (exact column set match)
_FORMAT_SIGNATURES: dict[frozenset[str], str] = {
    frozenset({"instruction", "input", "output"}): "alpaca",
    frozenset({"conversations"}): "sharegpt",
    frozenset({"messages"}): "openai_messages",
    frozenset({"prompt", "chosen", "rejected"}): "dpo",
    frozenset({"question", "answer"}): "qa",
    frozenset({"question", "info"}): "rlvr",
    frozenset({"text"}): "pretrain",
}

# Fuzzy synonym mapping: internal field -> recognized synonyms
_SYNONYMS: dict[str, list[str]] = {
    "instruction": ["instruction", "prompt", "query", "question", "input", "user"],
    "response": ["response", "output", "answer", "completion", "assistant", "reply"],
    "chosen": ["chosen", "preferred", "accepted", "positive"],
    "rejected": ["rejected", "dispreferred", "negative", "worse"],
}

# Reverse lookup: synonym -> internal field
_SYNONYM_LOOKUP: dict[str, str] = {}
for _field, _synonyms in _SYNONYMS.items():
    for _syn in _synonyms:
        _SYNONYM_LOOKUP[_syn] = _field

# Fields that map directly to Sample (no conversion needed)
_DIRECT_FIELDS = {
    "id",
    "instruction",
    "response",
    "system_prompt",
    "text",
    "chosen",
    "rejected",
    "chosen_score",
    "rejected_score",
    "info",
    "domain",
    "topic",
    "persona",
    "task_type",
    "language",
    "variant",
    "formality",
    "difficulty",
    "quality_score",
    "quality_reason",
    "safety_passed",
    "generation_model",
    "source_id",
    "evolution_history",
    "extra",
}


class ColumnMapper:
    """Maps input records to Sample objects via auto-detection or explicit column mapping."""

    def __init__(self, column_map: dict[str, str] | None = None) -> None:
        """Initialize with optional user-provided column mapping.

        Args:
            column_map: Dict mapping source column names to Sample field names.
                        E.g. {"my_col": "instruction"}. Takes precedence over auto-detect.
        """
        self.column_map = column_map or {}

    def map(self, records: list[dict[str, Any]]) -> list[Sample]:
        """Map a list of raw dicts to Sample objects."""
        if not records:
            return []

        first = records[0]
        detected_format = self._detect_format(first)
        return [self._map_record(record, detected_format) for record in records]

    def _detect_format(self, record: dict[str, Any]) -> str | None:
        """Detect the input format by examining column names."""
        columns = frozenset(record.keys())
        for sig, fmt in _FORMAT_SIGNATURES.items():
            if sig.issubset(columns):
                return fmt
        return None

    def _map_record(self, record: dict[str, Any], detected_format: str | None) -> Sample:
        """Map a single raw dict to a Sample, applying user overrides, format-specific
        logic, and fuzzy synonym matching in order."""
        fields: dict[str, Any] = {}
        used_keys: set[str] = set()

        # Step 1: Apply user-provided column_map first (highest priority)
        for src_col, target_field in self.column_map.items():
            if src_col in record:
                fields[target_field] = record[src_col]
                used_keys.add(src_col)

        # Step 2: Apply format-specific mapping
        if detected_format and not self.column_map:
            fmt_fields, fmt_used = self._apply_format(record, detected_format)
            fields.update(fmt_fields)
            used_keys.update(fmt_used)

        # Step 3: Direct field matching + fuzzy synonym matching for remaining keys
        for key, value in record.items():
            if key in used_keys:
                continue

            if key in _DIRECT_FIELDS:
                if key not in fields:
                    fields[key] = value
                used_keys.add(key)
            elif key in _SYNONYM_LOOKUP:
                target = _SYNONYM_LOOKUP[key]
                if target not in fields:
                    fields[target] = value
                used_keys.add(key)

        # Step 4: Remaining unmapped columns go into extra
        extra = fields.get("extra", {})
        for key, value in record.items():
            if key not in used_keys:
                extra[key] = value
        if extra:
            fields["extra"] = extra

        return Sample(**fields)

    def _apply_format(self, record: dict[str, Any], fmt: str) -> tuple[dict[str, Any], set[str]]:
        """Apply format-specific column mapping. Returns (fields_dict, used_keys)."""
        fields: dict[str, Any] = {}
        used: set[str] = set()

        if fmt == "alpaca":
            fields["instruction"] = record.get("instruction", "")
            if record.get("input"):
                fields["instruction"] = f"{fields['instruction']}\n{record['input']}"
            fields["response"] = record.get("output")
            used = {"instruction", "input", "output"}

        elif fmt == "sharegpt":
            conversations = record.get("conversations", [])
            turns = []
            for turn in conversations:
                role_map = {"human": "user", "gpt": "assistant", "system": "system"}
                role = role_map.get(turn.get("from", ""), turn.get("from", "user"))
                turns.append(Turn(role=role, content=turn.get("value", "")))
            fields["turns"] = turns
            used = {"conversations"}

        elif fmt == "openai_messages":
            messages = record.get("messages", [])
            turns = [
                Turn(role=m.get("role", "user"), content=m.get("content", "")) for m in messages
            ]
            fields["turns"] = turns
            used = {"messages"}

        elif fmt == "dpo":
            fields["instruction"] = record.get("prompt")
            fields["chosen"] = record.get("chosen")
            fields["rejected"] = record.get("rejected")
            used = {"prompt", "chosen", "rejected"}

        elif fmt == "qa":
            fields["instruction"] = record.get("question")
            fields["response"] = record.get("answer")
            used = {"question", "answer"}

        elif fmt == "rlvr":
            fields["instruction"] = record.get("question")
            fields["info"] = record.get("info")
            used = {"question", "info"}

        elif fmt == "pretrain":
            fields["text"] = record.get("text")
            used = {"text"}

        return fields, used
