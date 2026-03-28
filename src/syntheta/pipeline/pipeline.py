"""Pipeline: sample-level concurrent orchestrator.

Each sample flows through the full pipeline (transform -> filter -> write)
as an independent async task. The LLM semaphore controls concurrency.
No batching -- samples complete and show progress individually.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from syntheta.exceptions import ConfigHashMismatchError
from syntheta.llm.backend import OpenAICompatibleLLM
from syntheta.observability.filter_summary import FilterSummary
from syntheta.observability.progress import ProgressDisplay
from syntheta.pipeline.base import BaseFilter, BaseGenerator, BaseTransformer
from syntheta.pipeline.checkpoint import CheckpointManager, CheckpointState
from syntheta.pipeline.writer import JSONLWriter
from syntheta.schema.dataset import SynthDataset
from syntheta.schema.sample import Sample

logger = logging.getLogger("syntheta.pipeline")


class Pipeline:
    """Sample-level concurrent pipeline.

    Each sample independently flows through: transformers -> filters -> write.
    Samples that finish first show progress immediately -- one slow sample
    doesn't block the rest. Concurrency is controlled by the LLM semaphore.

    No pre-calculated over-generation. Generates exactly n samples first,
    processes them, then generates more only if filters rejected some.
    """

    def __init__(
        self,
        generator: BaseGenerator,
        transformers: list[BaseTransformer] | None = None,
        filters: list[BaseFilter] | None = None,
        llm: OpenAICompatibleLLM | None = None,
        seed: int | None = None,
        over_generate_factor: float = 1.0,
        show_progress: bool = True,
    ) -> None:
        self.generator = generator
        self.transformers = transformers or []
        self.filters = filters or []
        self.llm = llm
        self.seed = seed
        self.over_generate_factor = over_generate_factor
        self.show_progress = show_progress
        self.filter_summary = FilterSummary()

        # Inject LLM into components that need it
        if llm:
            if generator.llm is None:
                generator.llm = llm
            for t in self.transformers:
                if t.llm is None:
                    t.llm = llm
            for f in self.filters:
                if f.llm is None:
                    f.llm = llm

    def run(
        self,
        n: int,
        output: str | Path = "output.jsonl",
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
        config_dict: dict[str, Any] | None = None,
        checkpoint_interval: int = 100,
    ) -> SynthDataset:
        """Run the pipeline synchronously."""
        return asyncio.run(
            self._run(n, output, checkpoint_path, resume, config_dict, checkpoint_interval)
        )

    async def _run(
        self,
        n: int,
        output: str | Path,
        checkpoint_path: str | Path | None,
        resume: bool,
        config_dict: dict[str, Any] | None,
        checkpoint_interval: int,
    ) -> SynthDataset:
        """Sample-level concurrent pipeline execution."""
        output = Path(output)
        if checkpoint_path is None:
            checkpoint_path = output.parent / f"{output.stem}_checkpoints"
        checkpoint_path = Path(checkpoint_path)

        writer = JSONLWriter(output, resume=resume)
        ckpt_mgr = CheckpointManager(checkpoint_path)
        config_hash = ckpt_mgr.compute_config_hash(config_dict or {})

        # Handle resume
        if resume:
            state = ckpt_mgr.load()
            if state is not None:
                if state.config_hash != config_hash:
                    raise ConfigHashMismatchError(
                        "Config has changed since the last run. "
                        "Use a new checkpoint_path or remove the existing checkpoint."
                    )
                self.filter_summary.total_generated = state.total_generated
                self.filter_summary.total_passed = state.total_passed
                for reason, count in state.filter_stats.items():
                    self.filter_summary.record_rejection(reason, count)
                logger.info("Resuming: %d already passed", state.total_passed)

        # Set up progress display
        display: ProgressDisplay | None = None
        if self.show_progress:
            # Suppress console log output while rich display is active
            # (log messages overlap with the live display)
            logging.getLogger("syntheta").setLevel(logging.CRITICAL)

            max_streams = self.llm.rate_limiter.max_concurrent if self.llm else 10
            display = ProgressDisplay(target_n=n, max_streams=max_streams)

        sample_tok_rates: list[float] = []  # Track tok/s per completed sample

        if display:
            # Wire stats provider for live stats bar
            if self.llm:
                llm_ref = self.llm
                stats_start_time = time.time()

                def get_stats() -> dict:
                    elapsed = time.time() - stats_start_time
                    total_tokens = llm_ref.cost_tracker.total_tokens
                    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                    avg_tok_rate = sum(sample_tok_rates) / len(sample_tok_rates) if sample_tok_rates else 0
                    return {
                        "active": llm_ref.active_calls,
                        "total_calls": llm_ref.total_calls,
                        "retries": llm_ref.total_retries,
                        "tokens": total_tokens,
                        "tok_per_sec": tok_per_sec,
                        "avg_sample_tok_rate": avg_tok_rate,
                        "current_concurrent": llm_ref.rate_limiter.current_concurrent,
                        "max_concurrent": llm_ref.rate_limiter.max_concurrent,
                    }

                display.set_stats_provider(get_stats)

            display.start()
            if self.llm:
                self.llm.on_token = display.on_token

        start_time = time.time()
        passed_count = writer.count  # Non-zero if resuming
        passed_lock = asyncio.Lock()
        done_event = asyncio.Event()
        if passed_count >= n:
            done_event.set()

        async def process_sample(sample: Sample) -> Sample | None:
            """Run one sample through transformers -> filters."""
            nonlocal passed_count
            sample_start = time.time()
            tokens_before = self.llm.cost_tracker.total_tokens if self.llm else 0

            if done_event.is_set():
                return None

            # Transformers
            for transformer in self.transformers:
                try:
                    result = await transformer.transform([sample])
                    if not result:
                        return None
                    sample = result[0]
                except Exception as e:
                    logger.warning(
                        "Transformer %s failed for sample %s: %s",
                        type(transformer).__name__,
                        sample.id,
                        e,
                    )
                    return None

            # Validate: if sample has instruction but no response after transformers,
            # the ResponseGenerator failed -- reject it
            if sample.instruction and not sample.response:
                self.filter_summary.record_rejection("empty_response")
                if display:
                    display.add_event("empty_response", "1 sample rejected")
                return None

            # Filters
            for filt in self.filters:
                try:
                    result = await filt.filter([sample])
                    if not result:
                        self.filter_summary.record_rejection(filt.rejection_reason)
                        if display:
                            display.add_event(filt.rejection_reason, "1 sample rejected")
                        return None
                except Exception as e:
                    logger.warning(
                        "Filter %s failed for sample %s: %s",
                        type(filt).__name__,
                        sample.id,
                        e,
                    )

            # Passed -- write and update
            async with passed_lock:
                if passed_count >= n:
                    return None
                passed_count += 1
                current = passed_count

            await writer.write_sample(sample)
            self.filter_summary.record_passed(1)
            sample_dur = time.time() - sample_start
            tokens_used = (self.llm.cost_tracker.total_tokens - tokens_before) if self.llm else 0
            if sample_dur > 0 and tokens_used > 0:
                sample_tok_rates.append(tokens_used / sample_dur)

            if display:
                text = sample.instruction or sample.text or ""
                display.add_completed_sample(current, text)
                display.update_progress(current)

            # Periodic checkpoint
            if checkpoint_interval > 0 and current % checkpoint_interval == 0:
                state = CheckpointState(
                    total_generated=self.filter_summary.total_generated,
                    total_passed=current,
                    config_hash=config_hash,
                    filter_stats=self.filter_summary.rejections,
                )
                ckpt_mgr.save(state)

            if current >= n:
                done_event.set()

            return sample

        try:
            # Generate samples, process concurrently, generate more if needed
            remaining = n - passed_count
            max_rounds = 5

            for _round in range(max_rounds):
                if remaining <= 0 or done_event.is_set():
                    break

                # Stream from generator -- create tasks as samples arrive,
                # don't collect all into memory first
                tasks: list[asyncio.Task] = []
                async for batch in self.generator.generate(remaining):
                    self.filter_summary.record_generated(len(batch))
                    for sample in batch:
                        if done_event.is_set():
                            break
                        task = asyncio.create_task(process_sample(sample))
                        tasks.append(task)
                    if done_event.is_set():
                        break

                if not tasks:
                    logger.warning("Generator produced no samples")
                    break

                # Process results as they complete
                for coro in asyncio.as_completed(tasks):
                    await coro
                    if done_event.is_set():
                        for t in tasks:
                            if not t.done():
                                t.cancel()
                        break

                await asyncio.gather(*tasks, return_exceptions=True)

                # Check how many more we need
                async with passed_lock:
                    remaining = n - passed_count

                if remaining > 0 and not done_event.is_set() and display:
                    display.add_event(
                        "pipeline",
                        f"Generating {remaining} more (filters rejected some)...",
                    )

        finally:
            if self.llm:
                self.llm.on_token = None
            if display:
                display.stop()
                logging.getLogger("syntheta").setLevel(logging.WARNING)

        # Save checkpoint
        state = CheckpointState(
            total_generated=self.filter_summary.total_generated,
            total_passed=self.filter_summary.total_passed,
            config_hash=config_hash,
            filter_stats=self.filter_summary.rejections,
        )
        ckpt_mgr.save(state)

        elapsed = time.time() - start_time
        logger.info("Pipeline complete: %d samples in %.1fs", writer.count, elapsed)

        print(self.filter_summary.display())

        return SynthDataset.from_jsonl(output)
