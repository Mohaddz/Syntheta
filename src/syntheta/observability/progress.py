"""Rich-based live progress display for pipeline execution."""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from collections.abc import Callable
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

# Callback type for pulling live stats from the LLM backend
StatsProvider = Callable[[], dict[str, Any]]


class ProgressDisplay:
    """Live CLI display showing a progress bar, optional stats bar,
    and per-sample streaming text.

    Normal mode:
        Generating ==================== 7/10 samples  00:32
         ~  responding    The role of catalysts in...
         +  sample #7     What is the role of catalysts in...

    Verbose mode:
        Generating ==================== 7/10 samples  00:32
        [active: 8/10] [calls: 42] [retries: 2] [rejected: 3] [tokens: 12,340]
         ~  responding    The role of catalysts in...
         +  sample #7     What is the role of catalysts in...
    """

    MAX_VISIBLE_ROWS = 15
    MAX_TEXT_WIDTH = 90

    def __init__(
        self,
        target_n: int,
        max_streams: int = 10,
        verbose: bool = True,
        console: Console | None = None,
    ) -> None:
        self._console = console or Console()
        self._target_n = target_n
        self._max_streams = max_streams
        self._verbose = verbose
        self._completed_rows: deque[dict[str, Any]] = deque(maxlen=self.MAX_VISIBLE_ROWS)
        self._live: Live | None = None
        self._stats_provider: StatsProvider | None = None
        self._rejected_count = 0
        self._last_refresh: float = 0.0

        # Streaming rows: call_id -> {stage, text}
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

    def set_stats_provider(self, provider: StatsProvider) -> None:
        """Set a callable that returns live stats dict from the LLM backend."""
        self._stats_provider = provider

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

    # Stages worth showing streaming output for (user-facing content)
    _VISIBLE_STAGES: frozenset = frozenset({
        "topic_tree", "persona_generation", "seed_dataset",
        "response_generator", "evol_instruct", "pretrain_rewriter",
    })

    def on_token(self, call_id: str, stage: str, accumulated_text: str) -> None:
        """Called for each token chunk from the LLM.
        Only shows streaming rows for content-producing stages (generating,
        responding, evolving). Internal stages like scoring/safety are hidden.

        Refreshes are throttled to avoid choking the event loop at high
        concurrency (250 streams × many tokens/sec = thousands of calls).
        """
        if stage not in self._VISIBLE_STAGES:
            return
        label = _STAGE_LABELS.get(stage, stage)
        self._streams[call_id] = {"stage": label, "text": accumulated_text}
        while len(self._streams) > self._max_streams:
            self._streams.popitem(last=False)

        # Throttle: refresh at most 12 times/sec (matches Rich Live refresh rate)
        now = time.monotonic()
        if now - self._last_refresh >= 0.083:
            self._last_refresh = now
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
        """Record an event. Rejections only update the stats counter,
        not the visible row list (stats bar already shows rejected count)."""
        if "rejected" in text:
            self._rejected_count += 1
            self._refresh()
            return
        # Non-rejection events still show as rows
        label = _STAGE_LABELS.get(stage, stage)
        self._completed_rows.append({"num": None, "stage": label, "text": text})
        self._refresh()

    def _build_layout(self) -> Group:
        """Build the full layout."""
        parts: list[Any] = [self._progress]

        # Verbose stats bar
        if self._verbose and self._stats_provider:
            stats = self._stats_provider()
            stats_text = Text.assemble(
                ("  [", "dim"),
                ("active: ", "dim"),
                (f"{stats.get('active', 0)}/{stats.get('max_concurrent', '?')}", "cyan"),
                ("]  [", "dim"),
                ("calls: ", "dim"),
                (f"{stats.get('total_calls', 0)}", "cyan"),
                ("]  [", "dim"),
                ("retries: ", "dim"),
                (
                    f"{stats.get('retries', 0)}",
                    "red" if stats.get("retries", 0) > 0 else "cyan",
                ),
                ("]  [", "dim"),
                ("rejected: ", "dim"),
                (f"{self._rejected_count}", "yellow" if self._rejected_count > 0 else "cyan"),
                ("]  [", "dim"),
                ("tokens: ", "dim"),
                (f"{stats.get('tokens', 0):,}", "cyan"),
                ("]  [", "dim"),
                ("tok/s: ", "dim"),
                (f"{stats.get('tok_per_sec', 0):,.0f}", "cyan"),
                ("]  [", "dim"),
                ("tok/s/sample: ", "dim"),
                (
                    f"{stats.get('avg_sample_tok_rate', 0):,.0f}"
                    if stats.get("avg_sample_tok_rate", 0) > 0
                    else "—",
                    "cyan",
                ),
                ("]  [", "dim"),
                ("concurrency: ", "dim"),
                (
                    f"{stats.get('current_concurrent', '?')}/{stats.get('max_concurrent', '?')}",
                    (
                        "red"
                        if stats.get("current_concurrent", 0) < stats.get("max_concurrent", 0)
                        else "cyan"
                    ),
                ),
                ("]", "dim"),
            )
            parts.append(stats_text)

        # Sample table
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

        # Streaming rows (show tail of text -- news ticker effect)
        for _call_id, stream in self._streams.items():
            tail = _truncate_tail(stream["text"], self.MAX_TEXT_WIDTH)
            table.add_row(
                Text(" ~ ", style="yellow"),
                Text(stream["stage"], style="yellow"),
                Text(tail, style="white"),
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

        parts.append(table)
        return Group(*parts)

    def _refresh(self) -> None:
        """Refresh the live display."""
        if self._live:
            self._live.update(self._build_layout())


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len, taking only the first meaningful line."""
    for line in text.split("\n"):
        line = line.strip()
        if line:
            text = line
            break
    else:
        text = text.replace("\n", " ").strip()

    text = text.lstrip("#").lstrip("*").lstrip("-").strip()

    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _truncate_tail(text: str, max_len: int) -> str:
    """Show the tail of the text (last max_len chars), like a news ticker.

    The display follows the latest generated token -- the user sees the
    text being written in real-time at the end.
    """
    text = text.replace("\n", " ").replace("\r", "").strip()
    text = text.lstrip("#").lstrip("*").lstrip("-").strip()

    if len(text) > max_len:
        return "..." + text[-(max_len - 3) :]
    return text
