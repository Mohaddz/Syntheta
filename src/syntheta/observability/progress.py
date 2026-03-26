"""Rich-based live progress display for pipeline execution."""

from __future__ import annotations

from collections import OrderedDict, deque
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

# Friendly stage names for display
_STAGE_LABELS = {
    "topic_tree": "generating",
    "persona_generation": "generating",
    "seed_dataset": "generating",
    "response_generator": "responding",
    "evol_instruct": "evolving",
    "quality_filter": "scoring",
    "safety_filter": "safety check",
    "pretrain_rewriter": "rewriting",
}


class ProgressDisplay:
    """Live CLI display showing a progress bar and per-sample streaming text.

    Layout:
        Generating ==================== 7/10 samples  00:32
        [streaming] generating    What is the role of cata...
        [streaming] responding    Catalysts are substances...
        [done]      sample #7    What is the role of catalysts in...
        [done]      sample #6    Explain the difference between...
    """

    MAX_VISIBLE_ROWS = 15
    MAX_TEXT_WIDTH = 90

    def __init__(self, target_n: int, console: Console | None = None) -> None:
        self._console = console or Console()
        self._target_n = target_n
        self._completed_rows: deque[dict[str, Any]] = deque(maxlen=self.MAX_VISIBLE_ROWS)
        self._live: Live | None = None

        # Streaming rows: call_id -> {stage, text} (ordered so newest shows last)
        self._streams: OrderedDict[str, dict[str, str]] = OrderedDict()

        # Main progress bar
        self._progress = Progress(
            SpinnerColumn("dots"),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[cyan]{task.completed}[/cyan]/[cyan]{task.total}[/cyan] samples"),
            TimeElapsedColumn(),
            console=self._console,
        )
        self._task_id = self._progress.add_task("Generating", total=target_n)

    def start(self) -> None:
        """Start the live display."""
        self._live = Live(
            self._build_layout(),
            console=self._console,
            refresh_per_second=12,
            transient=False,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if self._live:
            self._live.stop()
            self._live = None

    def update_progress(self, completed: int) -> None:
        """Update the main progress bar."""
        self._progress.update(self._task_id, completed=min(completed, self._target_n))
        self._refresh()

    def on_token(self, call_id: str, stage: str, accumulated_text: str) -> None:
        """Called for each token chunk from the LLM. Updates the streaming row
        for this call_id with the text accumulated so far."""
        label = _STAGE_LABELS.get(stage, stage)
        self._streams[call_id] = {"stage": label, "text": accumulated_text}
        # Keep only the most recent streams visible
        while len(self._streams) > 6:
            self._streams.popitem(last=False)
        self._refresh()

    def finish_stream(self, call_id: str) -> None:
        """Remove a streaming row when that LLM call is done."""
        self._streams.pop(call_id, None)
        self._refresh()

    def add_completed_sample(self, sample_num: int, text: str) -> None:
        """Add a completed sample row to the scrolling log."""
        self._completed_rows.append({"num": sample_num, "text": text})
        self._refresh()

    def add_event(self, stage: str, text: str) -> None:
        """Add a transient event row (filter rejection, etc.)."""
        label = _STAGE_LABELS.get(stage, stage)
        self._completed_rows.append({"num": None, "stage": label, "text": text})
        self._refresh()

    def _build_layout(self) -> Group:
        """Build the full layout: progress bar + streaming rows + completed rows."""
        table = Table(
            show_header=False,
            show_edge=False,
            show_lines=False,
            pad_edge=False,
            expand=True,
            padding=(0, 1),
        )
        table.add_column("status", width=3, no_wrap=True)
        table.add_column("label", width=16, no_wrap=True)
        table.add_column("text", ratio=1, no_wrap=True, overflow="ellipsis")

        # Streaming rows (active LLM calls with text appearing in real-time)
        for _call_id, stream in self._streams.items():
            truncated = _truncate(stream["text"], self.MAX_TEXT_WIDTH)
            table.add_row(
                Text(" ~ ", style="yellow"),
                Text(stream["stage"], style="yellow"),
                Text(truncated, style="white"),
            )

        # Completed sample rows
        for row in self._completed_rows:
            if row.get("num") is not None:
                truncated = _truncate(row["text"], self.MAX_TEXT_WIDTH)
                table.add_row(
                    Text(" + ", style="green"),
                    Text(f"sample #{row['num']}", style="green"),
                    Text(truncated),
                )
            else:
                truncated = _truncate(row["text"], self.MAX_TEXT_WIDTH)
                table.add_row(
                    Text(" . ", style="dim"),
                    Text(row.get("stage", ""), style="dim"),
                    Text(truncated, style="dim"),
                )

        return Group(self._progress, table)

    def _refresh(self) -> None:
        """Refresh the live display."""
        if self._live:
            self._live.update(self._build_layout())


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len, replacing newlines with spaces."""
    text = text.replace("\n", " ").replace("\r", "").strip()
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text
