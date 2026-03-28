# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Syntheta is an open-source Python SDK+CLI for generating synthetic LLM training datasets.
Streaming pipeline architecture with an OpenAI-compatible backend, external prompt templates,
and configurable safety/cultural filters.

## Quick Reference
- **Language:** Python 3.10+
- **Build backend:** setuptools (pyproject.toml)
- **Source layout:** `src/syntheta/` (src-layout)
- **Tests:** pytest with pytest-asyncio (`asyncio_mode = "auto"`)
- **CLI framework:** Click
- **Entry point:** `syntheta` command

## Build and Install
```bash
uv pip install -e ".[dev]"
```

## Running Tests
```bash
uv run pytest                           # All tests
uv run pytest tests/unit/               # Unit tests only
uv run pytest tests/cli/                # CLI tests only
uv run pytest tests/unit/test_pipeline.py::test_basic_pipeline  # Single test
uv run pytest -m integration            # Integration tests (requires OPENROUTER_API_KEY)
uv run pytest -m "not integration"      # Skip integration tests
uv run pytest --cov=syntheta            # With coverage
```

## Linting
```bash
uv run ruff check src/ tests/           # Lint (line-length=100)
uv run ruff format src/ tests/          # Format
```

## Architecture

### Pipeline data flow
```
CLI / SDK API  (sync)
    │
    ▼
Pipeline.run() → asyncio.run(_run())
    │
    ▼
Generator  ──yields──▶  [Sample]  ──▶  Transformers  ──▶  Filters  ──▶  Writer
(async)                                 (sequential)       (sequential)   (JSONL)
```

**Sample-level concurrency, not batch-level.** Each sample is an independent async task.
The LLM's `AdaptiveRateLimiter` semaphore is the single concurrency bottleneck — all generators,
transformers, and filters compete for it.

**Sliding window pattern:** Generators and pipeline use `asyncio.wait(FIRST_COMPLETED)` to
process results as they finish, refilling the window up to `max_concurrent`.

**Over-generation:** If filters reject samples, the pipeline re-calls the generator for exactly
the missing count (up to `max_rounds=5`).

### Config layering
```
defaults.py → recipe YAML → user YAML → CLI dot-notation overrides
```
Each layer deep-merges into the previous. Config is a plain dict, not a Pydantic model.
CLI overrides use dot-notation expansion: `--llm.model=gpt-4o` → `{"llm": {"model": "gpt-4o"}}`.

### Key design decisions
1. **OpenAI SDK with `max_retries=0`** — we control retries (exponential backoff) and concurrency ourselves via `AdaptiveRateLimiter`.
2. **async internally, sync API externally** — `Pipeline.run()` is sync, calls `asyncio.run()`. All LLM-touching code is `async def`.
3. **Simple `{{variable}}` template substitution** — no Jinja2. Prompts loaded from `.txt` files via `load_prompt()` / `render_template()`.
4. **Checkpoint every 100 samples** — config hash validated on resume to prevent misconfigurations.

### Module organization
- `schema/` — Sample (unified model for SFT/chat/DPO/RLVR/pretrain), SynthDataset, exporters
- `llm/` — OpenAICompatibleLLM, AdaptiveRateLimiter (semaphore + adaptive backoff), CostTracker
- `pipeline/` — BaseGenerator/Transformer/Filter ABCs, Pipeline orchestrator, Checkpoint, Writer
- `generators/` — TopicTreeGenerator, PersonaGenerator, SeedDatasetGenerator
- `transformers/` — EvolInstruct (6 strategies, multi-round), ResponseGenerator (optional CoT)
- `filters/` — SafetyFilter (blocklist + LLM cultural check), QualityFilter, TemplateCollapseDetector
- `prompts/` — External `.txt` template files + loader/renderer
- `config/` — defaults dict, YAML loader with deep-merge and dot-notation expansion
- `cli/` — Click commands (inspect command is in `inspect_cmd.py` to avoid shadowing builtin)
- `observability/` — ProgressDisplay (Rich live), health report, filter summary, cost report

## Code Conventions
- Pydantic `BaseModel` for all data classes and config schemas
- `async def` for anything that calls the LLM; sync wrapper for public API
- All LLM prompts in external `.txt` files under `src/syntheta/prompts/`; use `{{variable}}` syntax (not Jinja2, not f-strings)
- Filters track rejection reasons via `rejection_reason` class attribute for observability
- Unit tests mock `AsyncOpenAI` with `unittest.mock` / `pytest-mock`; use dummy generator/transformer/filter subclasses
- Integration tests use `@pytest.mark.integration` marker and require `OPENROUTER_API_KEY`
- Logging via module-level `logger = logging.getLogger(__name__)`
- Error handling in async loops: log warnings, reject samples gracefully — don't stop the pipeline
- Use `create_rng(seed)` for reproducibility; seed passed through to all generators/transformers
