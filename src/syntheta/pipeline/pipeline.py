"""Pipeline: streaming orchestrator for Generator -> Transformers -> Filters -> Writer."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from pathlib import Path
from typing import Any

from syntheta.exceptions import ConfigHashMismatchError, DiskFullError
from syntheta.llm.backend import OpenAICompatibleLLM
from syntheta.observability.filter_summary import FilterSummary
from syntheta.observability.progress import ProgressDisplay
from syntheta.pipeline.base import BaseFilter, BaseGenerator, BaseTransformer
from syntheta.pipeline.checkpoint import CheckpointManager, CheckpointState
from syntheta.pipeline.writer import JSONLWriter
from syntheta.schema.dataset import SynthDataset

logger = logging.getLogger("syntheta.pipeline")


class Pipeline:
    """Streaming pipeline: Generator -> Transformers -> Filters -> Writer.

    Processes samples in batches. Writes to disk after each batch.
    Supports checkpointing and resume.
    """

    def __init__(
        self,
        generator: BaseGenerator,
        transformers: list[BaseTransformer] | None = None,
        filters: list[BaseFilter] | None = None,
        llm: OpenAICompatibleLLM | None = None,
        seed: int | None = None,
        over_generate_factor: float = 1.2,
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> None:
        self.generator = generator
        self.transformers = transformers or []
        self.filters = filters or []
        self.llm = llm
        self.seed = seed
        self.over_generate_factor = over_generate_factor
        self.batch_size = batch_size
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
        checkpoint_path: str | Path = "./checkpoints",
        resume: bool = False,
        config_dict: dict[str, Any] | None = None,
    ) -> SynthDataset:
        """Run the pipeline synchronously. Wraps the async implementation."""
        return asyncio.run(self._run(n, output, checkpoint_path, resume, config_dict))

    async def _run(
        self,
        n: int,
        output: str | Path,
        checkpoint_path: str | Path,
        resume: bool,
        config_dict: dict[str, Any] | None,
    ) -> SynthDataset:
        """Async pipeline execution."""
        output = Path(output)
        checkpoint_path = Path(checkpoint_path)

        writer = JSONLWriter(checkpoint_path)
        ckpt_mgr = CheckpointManager(checkpoint_path)
        config_hash = ckpt_mgr.compute_config_hash(config_dict or {})

        # Handle resume
        start_batch = 0
        if resume:
            state = ckpt_mgr.load()
            if state is not None:
                if state.config_hash and state.config_hash != config_hash:
                    raise ConfigHashMismatchError(
                        "Config has changed since the last run. "
                        "Use a new checkpoint_path or remove the existing checkpoint."
                    )
                start_batch = state.batch_num + 1
                self.filter_summary.total_generated = state.total_generated
                self.filter_summary.total_passed = state.total_passed
                for reason, count in state.filter_stats.items():
                    self.filter_summary.record_rejection(reason, count)
                logger.info("Resuming from batch %d", start_batch)

        # Generate exactly n (not n * factor). Over-generation only kicks in
        # if filters reject samples and we need to generate more.
        n_generate = math.ceil(n * self.over_generate_factor)

        # Set up progress display
        display: ProgressDisplay | None = None
        if self.show_progress:
            display = ProgressDisplay(target_n=n)
            display.start()

            # Wire LLM streaming callback -> display for real-time token display
            if self.llm:
                self.llm.on_token = display.on_token

        start_time = time.time()
        batch_num = start_batch
        sample_counter = self.filter_summary.total_passed

        try:
            # Streaming pipeline loop
            async for batch in self.generator.generate(n_generate, self.batch_size):
                if batch_num < start_batch:
                    batch_num += 1
                    continue

                batch_generated = len(batch)
                self.filter_summary.record_generated(batch_generated)

                # Run transformers sequentially
                for transformer in self.transformers:
                    batch = await transformer.transform(batch)

                # Run filters sequentially
                for filt in self.filters:
                    before = len(batch)
                    batch = await filt.filter(batch)
                    rejected = before - len(batch)
                    if rejected > 0:
                        self.filter_summary.record_rejection(filt.rejection_reason, rejected)
                        if display:
                            display.add_event(
                                filt.rejection_reason, f"{rejected} samples rejected"
                            )

                # Trim batch if we have more than needed
                remaining = n - self.filter_summary.total_passed
                if len(batch) > remaining:
                    batch = batch[:remaining]

                self.filter_summary.record_passed(len(batch))

                # Show each completed sample and update progress per-sample
                if display:
                    for s in batch:
                        sample_counter += 1
                        text = s.instruction or s.text or ""
                        display.add_completed_sample(sample_counter, text)
                        display.update_progress(sample_counter)

                # Write batch to disk
                try:
                    writer.write_batch(batch, batch_num)
                except OSError as e:
                    raise DiskFullError(
                        f"Failed to write batch {batch_num}: {e}. "
                        f"Free disk space and resume with --resume."
                    ) from e

                # Save checkpoint
                state = CheckpointState(
                    batch_num=batch_num,
                    total_generated=self.filter_summary.total_generated,
                    total_passed=self.filter_summary.total_passed,
                    config_hash=config_hash,
                    filter_stats=self.filter_summary.rejections,
                )
                ckpt_mgr.save(state)

                logger.info(
                    "Batch %d: generated=%d, passed=%d, total_passed=%d/%d",
                    batch_num,
                    batch_generated,
                    len(batch),
                    self.filter_summary.total_passed,
                    n,
                )

                batch_num += 1

                # Stop if we have enough samples
                if self.filter_summary.total_passed >= n:
                    break

        finally:
            # Clean up display and LLM callback
            if self.llm:
                self.llm.on_token = None
            if display:
                display.stop()

        # Consolidate batch files into final output
        total_written = writer.consolidate(output)

        elapsed = time.time() - start_time
        logger.info(
            "Pipeline complete: %d samples in %.1fs",
            total_written,
            elapsed,
        )

        # Print filter summary
        print(self.filter_summary.display())

        # Load from disk instead of keeping in memory
        return SynthDataset.from_jsonl(output)
