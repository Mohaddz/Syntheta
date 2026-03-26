# CLAUDE.md - Syntheta Development Guide

## Project Overview
Syntheta is an open-source Python SDK+CLI for generating synthetic LLM training datasets.
Streaming pipeline architecture with an OpenAI-compatible backend, external prompt templates,
and configurable safety/cultural filters.

## Quick Reference
- **Language:** Python 3.10+
- **Package manager:** pip with hatchling build backend
- **Source layout:** `src/syntheta/` (src-layout)
- **Tests:** `tests/` using pytest
- **CLI framework:** Click
- **Entry point:** `syntheta` command

## Build and Install
```bash
pip install -e ".[dev]"
```

## Running Tests
```bash
pytest                           # All unit tests
pytest tests/unit/               # Unit tests only
pytest tests/cli/                # CLI tests only
pytest -m integration            # Integration tests (requires OPENROUTER_API_KEY)
pytest -m "not integration"      # Skip integration tests
pytest --cov=syntheta            # With coverage
```

## Linting
```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Architecture
```
User API / CLI
    |
Pipeline (generator -> transformers -> filters -> writer)
    |
OpenAICompatibleLLM (AsyncOpenAI with max_retries=0, adaptive rate limiting)
    |
External prompt templates ({{variable}} substitution)
```

### Key decisions:
1. **OpenAI SDK with max_retries=0** for LLM calls — typed responses, error classes. We build our own retry/concurrency on top.
2. **asyncio internally, sync API externally** — Pipeline.run() is sync, calls asyncio.run() internally.
3. **Simple {{variable}} template substitution** — no Jinja2.

### Module organization:
- `schema/` — Sample, Turn, ColumnMapper, exporters, SynthDataset
- `llm/` — OpenAICompatibleLLM, AdaptiveRateLimiter, CostTracker
- `pipeline/` — BaseGenerator/Transformer/Filter ABCs, Pipeline, Checkpoint, Writer
- `generators/` — TopicTreeGenerator, PersonaGenerator, SeedDatasetGenerator
- `transformers/` — EvolInstruct, ResponseGenerator
- `filters/` — SafetyFilter, QualityFilter, TemplateCollapseDetector
- `pretrain/` — PretrainRewriter
- `prompts/` — External .txt template files + loader/renderer
- `config/` — SynthetaConfig pydantic model, YAML loader, layered overrides
- `cli/` — Click commands
- `recipes/` — Built-in YAML recipe files
- `presets/` — Cultural safety preset YAML files
- `observability/` — Logging, health report, filter summary, cost report

## Code Conventions
- pydantic BaseModel for all data classes and config schemas
- async def for anything that calls the LLM; sync wrapper for public API
- All LLM prompts in external .txt files under src/syntheta/prompts/
- {{variable}} syntax in prompts (not Jinja2)
- Filters track rejection reasons for the filter summary
- Unit tests mock AsyncOpenAI with unittest.mock / pytest-mock
- Integration tests use @pytest.mark.integration marker
- CLI inspect command is in inspect_cmd.py (avoids shadowing builtin)
