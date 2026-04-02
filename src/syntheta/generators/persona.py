"""PersonaGenerator: extract personas from a text corpus for diverse instruction synthesis.

Implements the Persona Hub methodology (Chan et al., 2024 — arxiv.org/abs/2406.20094).
Pipeline: text-to-persona extraction → persona-to-persona expansion → dedup → synthesis.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import struct
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from syntheta.pipeline.base import BaseGenerator
from syntheta.pipeline.registry import register_generator
from syntheta.prompts import load_prompt, render_template
from syntheta.schema.sample import Sample
from syntheta.utils.jsonl import read_jsonl
from syntheta.utils.seeding import create_rng

logger = logging.getLogger("syntheta.generators.persona")

# ---------------------------------------------------------------------------
# MinHash helpers (pure Python, no external deps)
# ---------------------------------------------------------------------------

_NUM_HASHES = 128
_SHINGLE_K = 3
_SEEDS = [i.to_bytes(4, "big") for i in range(_NUM_HASHES)]


def _shingles(text: str) -> set[str]:
    """Character 3-gram shingles."""
    t = text.lower().strip()
    if len(t) < _SHINGLE_K:
        return {t}
    return {t[i : i + _SHINGLE_K] for i in range(len(t) - _SHINGLE_K + 1)}


def _minhash_signature(shingle_set: set[str]) -> list[int]:
    """Compute MinHash signature of length _NUM_HASHES."""
    sig = [float("inf")] * _NUM_HASHES
    for shingle in shingle_set:
        shingle_bytes = shingle.encode("utf-8")
        for i, seed in enumerate(_SEEDS):
            h = struct.unpack("<Q", hashlib.sha256(seed + shingle_bytes).digest()[:8])[0]
            if h < sig[i]:
                sig[i] = h
    return sig


def _jaccard_estimate(sig_a: list[int], sig_b: list[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    return sum(a == b for a, b in zip(sig_a, sig_b)) / len(sig_a)


def _minhash_dedup(personas: list[str], threshold: float = 0.8) -> list[str]:
    """Remove near-duplicate personas via MinHash."""
    kept_sigs: list[list[int]] = []
    kept: list[str] = []
    for persona in personas:
        sig = _minhash_signature(_shingles(persona))
        if any(_jaccard_estimate(sig, ks) > threshold for ks in kept_sigs):
            continue
        kept_sigs.append(sig)
        kept.append(persona)
    return kept


# ---------------------------------------------------------------------------
# Embedding dedup helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


@register_generator("persona")
class PersonaGenerator(BaseGenerator):
    """Generates instructions by extracting personas from a text corpus.

    Paper: Persona Hub (Chan et al., 2024) — arxiv.org/abs/2406.20094

    Pipeline:
    1. Text-to-Persona — extract persona descriptions from corpus documents
    2. Persona-to-Persona — expand via relationship hops
    3. Deduplicate — MinHash + embedding similarity
    4. Synthesize — generate instructions from each persona's perspective
    """

    def __init__(
        self,
        source: str,
        text_field: str = "text",
        max_personas: int = 500,
        expansion_rounds: int = 6,
        dedup_threshold: float = 0.9,
        seed: int | None = None,
        prompt_overrides: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.source = source
        self.text_field = text_field
        self.max_personas = max_personas
        self.expansion_rounds = expansion_rounds
        self.dedup_threshold = dedup_threshold
        self.seed = seed
        self.prompt_overrides = prompt_overrides
        self._rng = create_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(self, n: int) -> AsyncIterator[list[Sample]]:
        """Extract personas from corpus, expand, dedup, then synthesize instructions."""
        # Phase 1: Extract
        raw_personas = await self._extract_personas()
        if not raw_personas:
            logger.error("No personas extracted from corpus — cannot generate")
            return

        # Phase 2: Expand
        expanded = await self._expand_personas(raw_personas)

        # Phase 3: Deduplicate
        deduped = await self._deduplicate(expanded)
        logger.info(
            "Persona pool: %d extracted, %d after expansion, %d after dedup",
            len(raw_personas),
            len(expanded),
            len(deduped),
        )

        if not deduped:
            logger.error("All personas removed during dedup — cannot generate")
            return

        # Phase 4: Synthesize
        async for batch in self._synthesize(deduped, n):
            yield batch

    # ------------------------------------------------------------------
    # Phase 1: Corpus loading + persona extraction
    # ------------------------------------------------------------------

    def _load_corpus(self) -> list[str]:
        """Load text documents from corpus file."""
        path = Path(self.source)
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.source}")

        documents: list[str] = []
        suffix = path.suffix.lower()

        if suffix == ".jsonl":
            for record in read_jsonl(str(path)):
                text = record.get(self.text_field)
                if text and isinstance(text, str) and text.strip():
                    documents.append(text.strip())
                else:
                    logger.warning("Skipping record missing '%s' field", self.text_field)
        elif suffix == ".txt":
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        documents.append(line)
        else:
            raise ValueError(f"Unsupported corpus format '{suffix}'. Use .jsonl or .txt")

        if not documents:
            raise ValueError(f"Corpus file '{self.source}' contains no valid documents")

        # Sample down to max_personas if corpus is larger
        if len(documents) > self.max_personas:
            documents = self._rng.sample(documents, self.max_personas)
            logger.info("Sampled %d documents from corpus", self.max_personas)
        else:
            logger.info("Loaded %d documents from corpus", len(documents))

        return documents

    async def _extract_personas(self) -> list[str]:
        """Extract one persona per document via LLM."""
        documents = self._load_corpus()
        template = load_prompt("generators.persona", self.prompt_overrides)

        personas: list[str] = []

        async def extract_one(doc: str) -> str | None:
            prompt = render_template(template, text=doc)
            try:
                result = await self.llm.complete(
                    messages=[{"role": "user", "content": prompt}],
                    stage="persona_extraction",
                )
                text = result["content"].strip()
                return text if text else None
            except Exception as e:
                logger.warning("Persona extraction failed: %s", e)
                return None

        # Sliding window
        window = min(self.max_concurrent, len(documents))
        pending: set[asyncio.Task] = set()
        idx = 0

        while idx < window:
            pending.add(asyncio.create_task(extract_one(documents[idx])))
            idx += 1

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for _ in done:
                if idx < len(documents):
                    pending.add(asyncio.create_task(extract_one(documents[idx])))
                    idx += 1
            for task in done:
                persona = task.result()
                if persona:
                    personas.append(persona)

        return personas

    # ------------------------------------------------------------------
    # Phase 2: Persona-to-Persona expansion
    # ------------------------------------------------------------------

    async def _expand_personas(self, seed_personas: list[str]) -> list[str]:
        """Expand persona pool via relationship hops."""
        all_personas = list(seed_personas)
        current_round = list(seed_personas)
        template = load_prompt("generators.persona_expand", self.prompt_overrides)
        pool_cap = self.max_personas * 10

        async def expand_one(persona: str) -> str | None:
            prompt = render_template(template, persona=persona)
            try:
                result = await self.llm.complete(
                    messages=[{"role": "user", "content": prompt}],
                    stage="persona_expansion",
                )
                text = result["content"].strip()
                return text if text else None
            except Exception as e:
                logger.warning("Persona expansion failed: %s", e)
                return None

        for round_num in range(1, self.expansion_rounds + 1):
            if len(all_personas) >= pool_cap:
                logger.info("Persona pool cap reached (%d), stopping expansion", pool_cap)
                break

            new_personas: list[str] = []

            # Sliding window over current round's personas
            window = min(self.max_concurrent, len(current_round))
            pending: set[asyncio.Task] = set()
            idx = 0

            while idx < window:
                pending.add(asyncio.create_task(expand_one(current_round[idx])))
                idx += 1

            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for _ in done:
                    if idx < len(current_round):
                        pending.add(asyncio.create_task(expand_one(current_round[idx])))
                        idx += 1
                for task in done:
                    persona = task.result()
                    if persona:
                        new_personas.append(persona)

            all_personas.extend(new_personas)
            current_round = new_personas
            logger.info(
                "Expansion round %d: +%d personas (total: %d)",
                round_num,
                len(new_personas),
                len(all_personas),
            )

            if not new_personas:
                logger.info("No new personas in round %d, stopping expansion", round_num)
                break

        return all_personas

    # ------------------------------------------------------------------
    # Phase 3: Deduplication
    # ------------------------------------------------------------------

    async def _deduplicate(self, personas: list[str]) -> list[str]:
        """Two-stage dedup: MinHash (near-exact) then embedding (semantic)."""
        # Stage A: MinHash
        after_minhash = _minhash_dedup(personas, threshold=0.8)
        logger.info("MinHash dedup: %d → %d personas", len(personas), len(after_minhash))

        # Stage B: Embedding dedup
        after_embed = await self._embedding_dedup(after_minhash)
        return after_embed

    async def _embedding_dedup(self, personas: list[str]) -> list[str]:
        """Remove semantically similar personas via embedding cosine similarity."""
        if not hasattr(self.llm, "embed"):
            logger.warning("LLM client has no embed() method, skipping embedding dedup")
            return personas

        try:
            # Batch embeddings in chunks of 100
            embeddings: list[list[float]] = []
            batch_size = 100
            for i in range(0, len(personas), batch_size):
                chunk = personas[i : i + batch_size]
                chunk_embeddings = await self.llm.embed(chunk, stage="persona_dedup")
                embeddings.extend(chunk_embeddings)
        except Exception as e:
            logger.warning("Embedding dedup failed (%s), skipping", e)
            return personas

        # Greedy dedup: keep persona if not too similar to any already kept
        kept_indices: list[int] = []
        kept_embeddings: list[list[float]] = []

        for i, emb in enumerate(embeddings):
            is_dup = any(
                _cosine_similarity(emb, ke) > self.dedup_threshold for ke in kept_embeddings
            )
            if not is_dup:
                kept_indices.append(i)
                kept_embeddings.append(emb)

        result = [personas[i] for i in kept_indices]
        logger.info("Embedding dedup: %d → %d personas", len(personas), len(result))
        return result

    # ------------------------------------------------------------------
    # Phase 4: Instruction synthesis
    # ------------------------------------------------------------------

    async def _synthesize(self, personas: list[str], n: int) -> AsyncIterator[list[Sample]]:
        """Generate instructions from personas via round-robin assignment."""
        template = load_prompt("generators.persona_questions", self.prompt_overrides)

        async def synthesize_one(persona: str) -> Sample | None:
            prompt = render_template(template, persona=persona)
            try:
                result = await self.llm.complete(
                    messages=[{"role": "user", "content": prompt}],
                    stage="persona_synthesis",
                )
                instruction = result["content"].strip()
                if not instruction:
                    return None
                return Sample(instruction=instruction, persona=persona)
            except Exception as e:
                logger.warning("Instruction synthesis failed: %s", e)
                return None

        # Build task list: round-robin over personas
        assignments = [personas[i % len(personas)] for i in range(n)]

        # Sliding window
        window = min(self.max_concurrent, len(assignments))
        pending: set[asyncio.Task] = set()
        idx = 0
        total = 0

        while idx < window:
            pending.add(asyncio.create_task(synthesize_one(assignments[idx])))
            idx += 1

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            # Refill
            for _ in done:
                if idx < len(assignments) and total < n:
                    pending.add(asyncio.create_task(synthesize_one(assignments[idx])))
                    idx += 1

            batch: list[Sample] = []
            for task in done:
                sample = task.result()
                if sample and total < n:
                    batch.append(sample)
                    total += 1

            if batch:
                yield batch

            if total >= n:
                for t in pending:
                    t.cancel()
                pending.clear()
                return
