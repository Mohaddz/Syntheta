"""Typed exporters: validate and convert Sample lists to output format dicts."""

from __future__ import annotations

from syntheta.exceptions import ExportValidationError
from syntheta.schema.sample import Sample


def export_sft(
    samples: list[Sample],
    rename: dict[str, str] | None = None,
) -> list[dict]:
    """Export as SFT format. Requires instruction + response."""
    rename = rename or {}
    results = []
    errors = []

    for i, s in enumerate(samples):
        if not s.instruction or not s.response:
            errors.append(f"Sample {i} (id={s.id}): missing instruction or response")
            continue
        if len(s.response.strip()) < 10:
            errors.append(f"Sample {i} (id={s.id}): response shorter than 10 chars")
            continue
        row = {"instruction": s.instruction, "response": s.response}
        if s.system_prompt:
            row["system_prompt"] = s.system_prompt
        results.append(_apply_rename(row, rename))

    if errors:
        raise ExportValidationError(
            f"{len(errors)} samples failed SFT validation:\n" + "\n".join(errors[:10])
        )
    return results


def export_chat(
    samples: list[Sample],
    rename: dict[str, str] | None = None,
) -> list[dict]:
    """Export as ShareGPT chat format. Requires turns with >=2 entries."""
    rename = rename or {}
    results = []
    errors = []

    for i, s in enumerate(samples):
        if not s.turns or len(s.turns) < 2:
            errors.append(f"Sample {i} (id={s.id}): missing or insufficient turns (need >=2)")
            continue
        conversations = [{"from": t.role, "value": t.content} for t in s.turns]
        results.append(_apply_rename({"conversations": conversations}, rename))

    if errors:
        raise ExportValidationError(
            f"{len(errors)} samples failed Chat validation:\n" + "\n".join(errors[:10])
        )
    return results


def export_dpo(
    samples: list[Sample],
    rename: dict[str, str] | None = None,
) -> list[dict]:
    """Export as DPO format. Requires instruction + chosen + rejected."""
    rename = rename or {}
    results = []
    errors = []

    for i, s in enumerate(samples):
        if not s.instruction or not s.chosen or not s.rejected:
            errors.append(f"Sample {i} (id={s.id}): missing instruction, chosen, or rejected")
            continue
        if s.chosen == s.rejected:
            errors.append(f"Sample {i} (id={s.id}): chosen and rejected are identical")
            continue
        row = {"prompt": s.instruction, "chosen": s.chosen, "rejected": s.rejected}
        results.append(_apply_rename(row, rename))

    if errors:
        raise ExportValidationError(
            f"{len(errors)} samples failed DPO validation:\n" + "\n".join(errors[:10])
        )
    return results


def export_rlvr(
    samples: list[Sample],
    rename: dict[str, str] | None = None,
) -> list[dict]:
    """Export as RLVR judge format. Requires instruction + info dict."""
    rename = rename or {}
    results = []
    errors = []

    for i, s in enumerate(samples):
        if not s.instruction or s.info is None:
            errors.append(f"Sample {i} (id={s.id}): missing instruction or info")
            continue
        row = {"question": s.instruction, "info": s.info}
        results.append(_apply_rename(row, rename))

    if errors:
        raise ExportValidationError(
            f"{len(errors)} samples failed RLVR validation:\n" + "\n".join(errors[:10])
        )
    return results


def export_pretrain(
    samples: list[Sample],
    rename: dict[str, str] | None = None,
) -> list[dict]:
    """Export as pretraining format. Requires text field with >50 chars."""
    rename = rename or {}
    results = []
    errors = []

    for i, s in enumerate(samples):
        if not s.text or len(s.text.strip()) < 50:
            errors.append(f"Sample {i} (id={s.id}): missing or too short text (<50 chars)")
            continue
        results.append(_apply_rename({"text": s.text}, rename))

    if errors:
        raise ExportValidationError(
            f"{len(errors)} samples failed Pretrain validation:\n" + "\n".join(errors[:10])
        )
    return results


def _apply_rename(row: dict, rename: dict[str, str]) -> dict:
    """Apply column renaming to an output row."""
    return {rename.get(k, k): v for k, v in row.items()}
