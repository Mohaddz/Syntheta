"""Unified Sample and Turn models — the core data schema for Syntheta."""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Turn(BaseModel):
    """A single turn in a multi-turn conversation."""

    role: Literal["user", "assistant", "system"]
    content: str


class Sample(BaseModel):
    """One data point with all fields. Single unified class with optional fields per paradigm.

    At least one content group must be populated: instruction, turns, or text.
    Unrecognized fields from input are preserved in `extra`.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=lambda: str(uuid4()))

    # ── Core content (at least one group must be populated) ──
    instruction: str | None = None
    response: str | None = None
    system_prompt: str | None = None
    turns: list[Turn] | None = None
    text: str | None = None

    # ── Preference fields (DPO) ──
    chosen: str | None = None
    rejected: str | None = None
    chosen_score: float | None = None
    rejected_score: float | None = None

    # ── RLVR fields ──
    info: dict[str, Any] | None = None

    # ── Metadata ──
    domain: str | None = None
    topic: str | None = None
    persona: str | None = None
    task_type: str | None = None
    language: str | None = None
    variant: str | None = None
    formality: str | None = None
    difficulty: int | None = None

    # ── Quality & provenance ──
    quality_score: float | None = None
    quality_reason: str | None = None
    safety_passed: bool | None = None
    generation_model: str | None = None
    source_id: str | None = None
    evolution_history: list[str] = Field(default_factory=list)

    # ── Escape hatch ──
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_at_least_one_content_field(self) -> Sample:
        if self.instruction is None and self.turns is None and self.text is None:
            raise ValueError(
                "At least one content field must be populated: instruction, turns, or text"
            )
        return self
