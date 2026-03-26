"""Pipeline: streaming orchestrator for Generator → Transformers → Filters → Writer."""

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
from syntheta.pipeline.base import BaseFilter, BaseGenerator, BaseTransformer
from syntheta.pipeline.checkpoint import CheckpointManager, CheckpointState
from syntheta.pipeline.writer import JSONLWriter
from syntheta.schema.dataset import SynthDataset
from syntheta.schema.sample import Sample

logger = logging.getLogger("syntheta.pipeline")


class Pipeline:
    """Streaming pipeline: Generator → Transformers → Filters → Writer.

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
    ) -> None:
        self.generator = generator
        self.transformers = transformers or []
        self.filters = filters or []
        self.llm = llm
        self.seed = seed
        self.over_generate_factor = over_generate_factor
        self.batch_size = batch_size
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
        """Run the pipeline synchronously. Wraps the async implementation.

        Args:
            n: Target number of output samples.
            output: Path for final output JSONL.
            checkpoint_path: Directory for checkpoints.
            resume: If True, resume from last checkpoint.
            config_dict: Config dict for hash-based change detection on resume.

        Returns:
            SynthDataset of all output samples.
        """
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

        # Calculate how many samples to generate (over-generate to account for filtering)
        n_generate = math.ceil(n * self.over_generate_factor)
        logger.info(
            "Pipeline starting: target=%d, generating=%d (%.0f%% over-generation)",
            n,
            n_generate,
            (self.over_generate_factor - 1) * 100,
        )

        start_time = time.time()
        batch_num = start_batch

        # Streaming pipeline loop — samples flow through in batches,
        # are written to disk immediately, and are NOT accumulated in memory.
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

            self.filter_summary.record_passed(len(batch))

            # Write batch to disk immediately — no in-memory accumulation
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

        # Load from disk instead of keeping in memory — truly streaming
        return SynthDataset.from_jsonl(output)
