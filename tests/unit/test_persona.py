"""Tests for PersonaGenerator (Persona Hub methodology)."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from syntheta.generators.persona import (
    PersonaGenerator,
    _cosine_similarity,
    _jaccard_estimate,
    _minhash_dedup,
    _minhash_signature,
    _shingles,
)

# ---------------------------------------------------------------------------
# MinHash unit tests
# ---------------------------------------------------------------------------


class TestMinHash:
    def test_shingles_basic(self):
        result = _shingles("hello")
        assert "hel" in result
        assert "ell" in result
        assert "llo" in result

    def test_shingles_short_text(self):
        result = _shingles("ab")
        assert result == {"ab"}

    def test_minhash_dedup_removes_duplicates(self):
        personas = [
            "A software engineer specializing in backend systems",
            "A software engineer specializing in backend systems",
            "A pediatric nurse working in a children's hospital",
        ]
        result = _minhash_dedup(personas, threshold=0.8)
        assert len(result) == 2

    def test_minhash_dedup_keeps_distinct(self):
        personas = [
            "A marine biologist studying coral reef ecosystems in tropical waters",
            "A financial analyst working on algorithmic trading strategies",
            "A kindergarten teacher developing new play-based learning curricula",
        ]
        result = _minhash_dedup(personas, threshold=0.8)
        assert len(result) == 3

    def test_jaccard_identical_signatures(self):
        sig = _minhash_signature(_shingles("test persona description"))
        assert _jaccard_estimate(sig, sig) == 1.0

    def test_jaccard_different_signatures(self):
        sig_a = _minhash_signature(_shingles("a marine biologist studying coral reefs"))
        sig_b = _minhash_signature(_shingles("a financial analyst on wall street"))
        sim = _jaccard_estimate(sig_a, sig_b)
        assert sim < 0.5


# ---------------------------------------------------------------------------
# Cosine similarity tests
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == 1.0

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1, 0], [0, 1]) == 0.0

    def test_zero_vector(self):
        assert _cosine_similarity([0, 0], [1, 1]) == 0.0


# ---------------------------------------------------------------------------
# Corpus loading tests
# ---------------------------------------------------------------------------


class TestCorpusLoading:
    def test_load_corpus_jsonl(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        docs = [
            {"text": "Document about machine learning algorithms."},
            {"text": "Article on ancient Roman architecture."},
            {"text": ""},  # empty — should be skipped
        ]
        corpus.write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")

        gen = PersonaGenerator(source=str(corpus), llm=MagicMock())
        result = gen._load_corpus()
        assert len(result) == 2

    def test_load_corpus_txt(self, tmp_path):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text(
            "First document about biology.\nSecond document about physics.\n\n",
            encoding="utf-8",
        )

        gen = PersonaGenerator(source=str(corpus), llm=MagicMock())
        result = gen._load_corpus()
        assert len(result) == 2

    def test_load_corpus_custom_text_field(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        docs = [{"content": "Some text here."}]
        corpus.write_text(json.dumps(docs[0]), encoding="utf-8")

        gen = PersonaGenerator(source=str(corpus), text_field="content", llm=MagicMock())
        result = gen._load_corpus()
        assert len(result) == 1
        assert result[0] == "Some text here."

    def test_load_corpus_file_not_found(self):
        gen = PersonaGenerator(source="/nonexistent/path.jsonl", llm=MagicMock())
        with pytest.raises(FileNotFoundError):
            gen._load_corpus()

    def test_load_corpus_unsupported_format(self, tmp_path):
        corpus = tmp_path / "corpus.csv"
        corpus.write_text("a,b\n1,2", encoding="utf-8")

        gen = PersonaGenerator(source=str(corpus), llm=MagicMock())
        with pytest.raises(ValueError, match="Unsupported corpus format"):
            gen._load_corpus()

    def test_load_corpus_samples_down(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        docs = [{"text": f"Document {i}"} for i in range(100)]
        corpus.write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")

        gen = PersonaGenerator(source=str(corpus), max_personas=10, seed=42, llm=MagicMock())
        result = gen._load_corpus()
        assert len(result) == 10


# ---------------------------------------------------------------------------
# Extraction tests
# ---------------------------------------------------------------------------


class TestExtractPersonas:
    @pytest.mark.asyncio
    async def test_extract_personas(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        docs = [{"text": "A paper on neural networks."}, {"text": "A cooking recipe for pasta."}]
        corpus.write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")

        llm = MagicMock()
        llm.complete = AsyncMock(
            side_effect=[
                {"content": "A machine learning researcher focused on deep learning"},
                {"content": "A home cook interested in Italian cuisine"},
            ]
        )
        llm.rate_limiter = MagicMock()
        llm.rate_limiter.max_concurrent = 10

        gen = PersonaGenerator(source=str(corpus), llm=llm)
        result = await gen._extract_personas()
        assert len(result) == 2
        combined = " ".join(r.lower() for r in result)
        assert "machine learning" in combined

    @pytest.mark.asyncio
    async def test_bad_document_skipped(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        docs = [{"text": "Good document."}, {"text": "Another doc."}]
        corpus.write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")

        llm = MagicMock()
        llm.complete = AsyncMock(
            side_effect=[
                Exception("LLM error"),
                {"content": "A curious reader"},
            ]
        )
        llm.rate_limiter = MagicMock()
        llm.rate_limiter.max_concurrent = 10

        gen = PersonaGenerator(source=str(corpus), llm=llm)
        result = await gen._extract_personas()
        assert len(result) == 1
        assert result[0] == "A curious reader"


# ---------------------------------------------------------------------------
# Expansion tests
# ---------------------------------------------------------------------------


class TestExpandPersonas:
    @pytest.mark.asyncio
    async def test_expand_personas(self):
        llm = MagicMock()
        llm.complete = AsyncMock(
            side_effect=[
                {"content": "A data engineer working alongside the researcher"},
                {"content": "A product manager collaborating with the engineer"},
            ]
        )
        llm.rate_limiter = MagicMock()
        llm.rate_limiter.max_concurrent = 10

        gen = PersonaGenerator(source="dummy.jsonl", expansion_rounds=2, llm=llm)
        seeds = ["A machine learning researcher"]
        result = await gen._expand_personas(seeds)
        # 1 seed + 1 from round 1 + 1 from round 2
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Embedding dedup tests
# ---------------------------------------------------------------------------


class TestEmbeddingDedup:
    @pytest.mark.asyncio
    async def test_embedding_dedup(self):
        llm = MagicMock()
        # Two similar vectors and one different
        llm.embed = AsyncMock(
            return_value=[
                [1.0, 0.0, 0.0],
                [0.99, 0.01, 0.0],  # very similar to first
                [0.0, 0.0, 1.0],  # completely different
            ]
        )

        gen = PersonaGenerator(source="dummy.jsonl", dedup_threshold=0.9, llm=llm)
        personas = ["persona A", "persona B (similar to A)", "persona C (different)"]
        result = await gen._embedding_dedup(personas)
        assert len(result) == 2
        assert result[0] == "persona A"
        assert result[1] == "persona C (different)"

    @pytest.mark.asyncio
    async def test_embedding_dedup_skipped_without_embed(self):
        llm = MagicMock(spec=[])  # no embed method

        gen = PersonaGenerator(source="dummy.jsonl", llm=llm)
        personas = ["persona A", "persona A duplicate"]
        result = await gen._embedding_dedup(personas)
        assert len(result) == 2  # no dedup happened


# ---------------------------------------------------------------------------
# End-to-end generation tests
# ---------------------------------------------------------------------------


class TestGenerate:
    @pytest.mark.asyncio
    async def test_generate_end_to_end(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        docs = [{"text": "A research paper on quantum computing."}]
        corpus.write_text(json.dumps(docs[0]), encoding="utf-8")

        llm = MagicMock()
        llm.rate_limiter = MagicMock()
        llm.rate_limiter.max_concurrent = 10

        # Phase 1: extraction (1 doc)
        # Phase 2: expansion (1 round, 1 persona)
        # Phase 4: synthesis (4 instructions)
        llm.complete = AsyncMock(
            side_effect=[
                # Extraction
                {"content": "A quantum computing researcher"},
                # Expansion round 1
                {"content": "A physics graduate student"},
                # Expansion round 2
                {"content": "A university professor supervising the student"},
                # Expansion round 3
                {"content": "A lab technician maintaining quantum hardware"},
                # Expansion round 4
                {"content": "An industry engineer exploring quantum applications"},
                # Expansion round 5
                {"content": "A science journalist covering quantum breakthroughs"},
                # Expansion round 6
                {"content": "A general reader curious about quantum technology"},
                # Synthesis (4 instructions)
                {"content": "Explain quantum entanglement in simple terms"},
                {"content": "What are the best resources for learning quantum computing?"},
                {"content": "Help me write a grant proposal for quantum research"},
                {"content": "Compare superconducting vs trapped-ion qubits"},
            ]
        )
        # Skip embedding dedup
        llm.embed = AsyncMock(side_effect=Exception("no embedding model"))

        gen = PersonaGenerator(
            source=str(corpus),
            max_personas=10,
            expansion_rounds=6,
            llm=llm,
        )

        all_samples = []
        async for batch in gen.generate(n=4):
            all_samples.extend(batch)

        assert len(all_samples) == 4
        assert all(s.instruction for s in all_samples)
        assert all(s.persona for s in all_samples)
        # Persona should be free-text, not JSON
        assert all("{" not in s.persona for s in all_samples)

    @pytest.mark.asyncio
    async def test_respects_n_limit(self, tmp_path):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("A document about software engineering.", encoding="utf-8")

        llm = MagicMock()
        llm.rate_limiter = MagicMock()
        llm.rate_limiter.max_concurrent = 10

        llm.complete = AsyncMock(
            side_effect=[
                # Extraction
                {"content": "A senior software engineer"},
                # Expansion (6 rounds, 1 persona each)
                {"content": "A junior developer mentored by the engineer"},
                {"content": "A QA tester collaborating with the developer"},
                {"content": "A project manager overseeing the QA team"},
                {"content": "A product owner defining requirements"},
                {"content": "A UX designer working with the product owner"},
                {"content": "An end user providing feedback to the designer"},
                # Synthesis — more than needed, but n=2 should stop early
                {"content": "How to implement a rate limiter in Python?"},
                {"content": "Best practices for code review?"},
                {"content": "Should not reach this instruction"},
            ]
        )
        llm.embed = AsyncMock(side_effect=Exception("skip"))

        gen = PersonaGenerator(source=str(corpus), expansion_rounds=6, llm=llm)
        all_samples = []
        async for batch in gen.generate(n=2):
            all_samples.extend(batch)

        assert len(all_samples) == 2
