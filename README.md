# Syntheta

**Open-source Python SDK and CLI for generating synthetic LLM training datasets.**

Generate high-quality SFT, chat, DPO, RLVR, and pretrain datasets using any OpenAI-compatible API. Syntheta gives you a streaming pipeline with built-in safety filters, quality scoring, and checkpoint/resume support.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [CLI Usage](#cli-usage)
- [SDK Usage](#sdk-usage)
- [Configuration](#configuration)
- [Generators](#generators)
- [Transformers](#transformers)
- [Filters](#filters)
- [Recipes](#recipes)
- [Safety Presets](#safety-presets)
- [Data Formats](#data-formats)
- [Architecture](#architecture)
- [Development](#development)
- [License](#license)

---

## Quick Start

```bash
# Install
pip install -e ".[all]"

# Set your API key
export OPENAI_API_KEY="sk-..."

# Generate 100 SFT samples about Python programming
syntheta generate --recipe sft --domain "Python programming" --n 100 --output python_sft.jsonl

# Inspect what you got
syntheta inspect python_sft.jsonl

# Validate the output format
syntheta validate python_sft.jsonl --format sft
```

Or use any OpenAI-compatible provider (OpenRouter, Together, Ollama, etc.):

```bash
export OPENROUTER_API_KEY="sk-or-..."

syntheta generate --recipe sft \
  --domain "Machine Learning" \
  --n 500 \
  --output ml_sft.jsonl \
  llm.base_url=https://openrouter.ai/api/v1 \
  llm.api_key_env=OPENROUTER_API_KEY \
  llm.model=deepseek/deepseek-chat-v3-0324
```

---

## Installation

**Requirements:** Python 3.10+

```bash
# Basic install
pip install -e .

# With Hugging Face datasets support
pip install -e ".[hub]"

# With all extras
pip install -e ".[all]"

# Development (includes testing and linting tools)
pip install -e ".[dev]"
```

Using [uv](https://github.com/astral-sh/uv):

```bash
uv pip install -e ".[dev]"
```

---

## CLI Usage

### `syntheta generate` -- Generate synthetic data

```bash
# From a built-in recipe
syntheta generate --recipe sft --domain "Biology" --n 1000 --output bio.jsonl

# From a custom config file
syntheta generate --config myconfig.yaml

# With CLI overrides (dot-notation)
syntheta generate --recipe sft --n 500 \
  llm.model=gpt-4o \
  llm.max_concurrent=20 \
  filters.quality.min_score=0.8 \
  responses.use_cot=true

# Dry run (estimate cost without generating)
syntheta generate --recipe sft --n 1000 --dry-run

# Resume from checkpoint
syntheta generate --config myconfig.yaml --resume

# Persona-based generation
syntheta generate --method persona --domain "Data Science" --n 500

# Augment existing data (Self-Instruct)
syntheta generate --recipe augment --n 200 \
  seed_dataset.source=existing_data.jsonl
```

**Options:**

| Flag | Description |
|------|-------------|
| `--config` | Path to a YAML config file |
| `--recipe` | Built-in recipe: `sft`, `sft_persona`, `pretrain`, `augment` |
| `--domain` | Domain/topic for generation |
| `--n` | Number of samples to generate |
| `--method` | Generator: `topic_tree`, `persona`, `seed_dataset` |
| `--output` | Output JSONL file path |
| `--seed` | Random seed for reproducibility |
| `--dry-run` | Estimate cost without generating |
| `--resume` | Resume from last checkpoint |

Any additional arguments are parsed as dot-notation overrides (e.g., `llm.model=gpt-4o`).

### `syntheta inspect` -- Dataset statistics

```bash
syntheta inspect dataset.jsonl
syntheta inspect dataset.jsonl --json         # Machine-readable output
syntheta inspect dataset.jsonl --samples 5    # Show 5 preview samples
```

Shows sample count, field population percentages, distributions (language, task type, difficulty), and quality score statistics.

### `syntheta validate` -- Format validation

```bash
syntheta validate dataset.jsonl --format sft
syntheta validate dataset.jsonl --format chat --strict
```

Supported formats: `sft`, `chat`, `dpo`, `rlvr`, `pretrain`.

### `syntheta convert` -- Format conversion

```bash
syntheta convert dataset.jsonl --to chat --output dataset_chat.jsonl
syntheta convert dataset.jsonl --to dpo --output dataset_dpo.jsonl
```

### `syntheta recipes` -- Recipe management

```bash
syntheta recipes list                          # List available recipes
syntheta recipes show sft                      # Display recipe contents
syntheta recipes export sft --output my_sft.yaml  # Export for customization
```

---

## SDK Usage

### Loading and inspecting data

```python
import syntheta

# Load a JSONL dataset
ds = syntheta.load("data.jsonl")

# Auto-detect column format (Alpaca, ShareGPT, OpenAI, etc.)
ds = syntheta.load("alpaca_data.jsonl")

# Inspect statistics
stats = syntheta.inspect("data.jsonl")
print(stats["samples"], stats["distributions"])

# Validate format
result = syntheta.validate("data.jsonl", format="sft")
print(f"{result['passed']}/{result['total']} passed")
```

### Working with datasets

```python
from syntheta import Sample, SynthDataset

# Create samples programmatically
samples = [
    Sample(instruction="Explain recursion", response="Recursion is..."),
    Sample(instruction="What is a hash map?", response="A hash map is..."),
]
ds = SynthDataset(samples)

# Filter
high_quality = ds.filter(lambda s: (s.quality_score or 0) >= 0.8)

# Export to different formats
ds.save("output_sft.jsonl", format="sft")
ds.save("output_chat.jsonl", format="chat")
ds.save("output_dpo.jsonl", format="dpo")
```

### Running the pipeline programmatically

```python
from syntheta.llm.backend import OpenAICompatibleLLM
from syntheta.generators import TopicTreeGenerator
from syntheta.transformers import ResponseGenerator
from syntheta.filters import SafetyFilter, QualityFilter
from syntheta.pipeline.pipeline import Pipeline

llm = OpenAICompatibleLLM(model="gpt-4o-mini")

pipeline = Pipeline(
    generator=TopicTreeGenerator(
        domain="Machine Learning",
        topic_depth=2,
        topic_breadth=5,
        llm=llm,
    ),
    transformers=[ResponseGenerator(llm=llm, use_cot=True)],
    filters=[
        SafetyFilter(llm=llm),
        QualityFilter(min_score=0.7, llm=llm),
    ],
    llm=llm,
)

dataset = pipeline.run(n=500, output="ml_dataset.jsonl")
```

---

## Configuration

Syntheta uses a layered configuration system. Each layer deep-merges into the previous:

```
defaults --> recipe YAML --> user YAML --> CLI dot-notation overrides
```

### Full config reference

```yaml
config_version: "0.1"
seed: null                       # Random seed for reproducibility
domain: "general"                # Target domain
n: 100                           # Number of samples
method: "topic_tree"             # Generator: topic_tree | persona | seed_dataset
languages: ["en"]                # Target languages
task_types: ["qa", "explain", "compare", "creative"]
difficulty: [1, 5]               # Range 1-5
over_generate_factor: 1.0        # Generate extra, filter down

# Generator: Topic Tree
topic_tree:
  depth: 2                       # Topic hierarchy depth
  breadth: 5                     # Subtopics per topic

# Transformer: Evol-Instruct
evolve:
  enabled: false
  rounds: 1                      # Evolution rounds
  strategies: ["deepen", "concretize", "complicate"]

# Transformer: Response Generation
responses:
  enabled: true
  use_cot: false                 # Chain-of-thought prompting
  max_tokens: null               # Max response length

# Filters
filters:
  safety:
    enabled: true
    cultural_context: null       # academic | family_friendly | islamic
    blocklist_topics: []
    blocklist_words: []
    custom_instruction: null
  quality:
    enabled: false
    min_score: 0.7               # 0.0-1.0, LLM-as-judge threshold
  template_collapse:
    enabled: false
    prefix_length: 50
    max_repeat: 3
    action: "report"             # report | flag | remove

# LLM Backend
llm:
  base_url: null                 # Custom API endpoint
  api_key_env: "OPENAI_API_KEY"  # Environment variable name
  model: "gpt-4o-mini"
  response_model: null           # Separate model for responses
  max_concurrent: 10             # Parallel requests
  max_retries: 3
  timeout: 300
  pricing: null                  # Custom pricing override

# Output
output: "output.jsonl"
checkpoint_interval: 100         # Save progress every N samples
log_prompts: false               # Log raw LLM prompts (debugging)
prompts: {}                      # Template overrides
```

### CLI dot-notation overrides

Any config key can be overridden from the command line:

```bash
syntheta generate --recipe sft \
  llm.model=gpt-4o \
  llm.max_concurrent=20 \
  filters.quality.enabled=true \
  filters.quality.min_score=0.8
```

### Custom prompt templates

Override any prompt template via the `prompts` config key:

```yaml
prompts:
  generators.topic_tree: |
    Your custom topic generation prompt with {{domain}} and {{breadth}}...
  filters.quality_judge: /path/to/your/quality_prompt.txt
```

Templates use simple `{{variable}}` substitution (not Jinja2).

---

## Generators

### Topic Tree (`topic_tree`)

Builds a hierarchical topic taxonomy for the domain, then generates diverse instructions across all topics.

```yaml
method: topic_tree
domain: "Computer Science"
topic_tree:
  depth: 3       # 3 levels of subtopics
  breadth: 5     # 5 subtopics per node
```

### Persona (`persona`)

Creates synthetic personas and generates questions from their unique perspectives, producing naturally diverse instructions.

```yaml
method: persona
domain: "Healthcare"
persona:
  n_personas: 20
  questions_per_persona: 10
```

### Seed Dataset (`seed_dataset`)

Self-Instruct style: uses examples from an existing dataset as few-shot prompts to generate new, similar instructions.

```yaml
method: seed_dataset
seed_dataset:
  source: "existing_data.jsonl"
  n_few_shot: 3
```

---

## Transformers

### Evol-Instruct

Evolves instructions through mutation strategies inspired by the [WizardLM paper](https://arxiv.org/abs/2304.12244):

| Strategy | Effect |
|----------|--------|
| `deepen` | Add complexity and depth |
| `broaden` | Cover more aspects |
| `concretize` | Add specific examples |
| `rephrase` | Reword while preserving intent |
| `complicate` | Add constraints and requirements |
| `switch_task` | Change the task type |

```yaml
evolve:
  enabled: true
  rounds: 2                    # Apply mutations twice
  strategies: ["deepen", "concretize", "complicate"]
```

### Response Generator

Generates high-quality responses for instructions. Supports optional chain-of-thought reasoning.

```yaml
responses:
  enabled: true
  use_cot: true     # Enable chain-of-thought
  max_tokens: 2048
```

---

## Filters

### Safety Filter

Two-layer filtering:
1. **Fast blocklist check** (no LLM call) -- catches obvious unsafe content via keyword matching
2. **LLM cultural/context check** (optional) -- nuanced safety evaluation using cultural presets

```yaml
filters:
  safety:
    enabled: true
    cultural_context: "family_friendly"
    blocklist_topics: ["gambling", "drugs"]
    blocklist_words: ["inappropriate_word"]
    custom_instruction: "Reject any content promoting misinformation"
```

### Quality Filter

LLM-as-judge scoring across multiple dimensions (clarity, helpfulness, accuracy). Samples below the threshold are rejected.

```yaml
filters:
  quality:
    enabled: true
    min_score: 0.7    # Reject samples scoring below 0.7
```

### Template Collapse Detector

Heuristic detection of repetitive LLM outputs (when the model gets stuck producing the same prefix). No LLM call required.

```yaml
filters:
  template_collapse:
    enabled: true
    prefix_length: 50
    max_repeat: 3
    action: "remove"    # report | flag | remove
```

---

## Recipes

Built-in recipes provide pre-configured pipelines for common use cases:

| Recipe | Description |
|--------|-------------|
| `sft` | Topic-tree generation + responses + safety + quality filtering |
| `sft_persona` | Persona-based SFT generation |
| `pretrain` | Pretrain corpus text generation |
| `augment` | Augment existing data via Self-Instruct |

```bash
# Use a recipe as a starting point
syntheta recipes show sft

# Export and customize
syntheta recipes export sft --output my_recipe.yaml
# Edit my_recipe.yaml...
syntheta generate --config my_recipe.yaml --domain "Finance" --n 2000
```

---

## Safety Presets

Cultural context presets for the safety filter:

| Preset | Use Case |
|--------|----------|
| `academic` | Academic and educational content |
| `family_friendly` | Content safe for all ages |
| `islamic` | Islamic cultural context and values |

```yaml
filters:
  safety:
    enabled: true
    cultural_context: "islamic"
```

---

## Data Formats

Syntheta uses a unified `Sample` model internally and can export to multiple formats:

### Supported export formats

| Format | Fields | Use Case |
|--------|--------|----------|
| `sft` | instruction, response | Supervised fine-tuning |
| `chat` | conversations (ShareGPT) | Multi-turn chat training |
| `dpo` | prompt, chosen, rejected | Direct Preference Optimization |
| `rlvr` | question, info | Reinforcement Learning with Verifiable Rewards |
| `pretrain` | text | Continued pretraining |

### Auto-detected input formats

The `ColumnMapper` automatically detects and converts from:
- **Alpaca**: instruction, input, output
- **ShareGPT**: conversations
- **OpenAI**: messages
- **DPO**: prompt, chosen, rejected
- **Q&A**: question, answer
- **RLVR**: question, info
- **Pretrain**: text

```bash
# Convert between formats
syntheta convert alpaca_data.jsonl --to chat --output chat_data.jsonl
```

---

## Architecture

```
CLI / SDK API  (sync)
    |
    v
Pipeline.run() --> asyncio.run(_run_async())
    |
    v
Generator  --yields-->  [Sample]  -->  Transformers  -->  Filters  -->  Writer
(async)                                (sequential)       (sequential)   (JSONL)
```

**Key design principles:**

- **Sample-level concurrency** -- each sample is an independent async task. The `AdaptiveRateLimiter` semaphore is the single concurrency bottleneck.
- **Sliding window** -- uses `asyncio.wait(FIRST_COMPLETED)` to process results as they finish, keeping the pipeline saturated.
- **Over-generation** -- if filters reject samples, the pipeline automatically re-generates the missing count (up to 5 rounds).
- **Checkpoint/resume** -- saves progress every N samples with config hash validation to prevent misconfiguration on resume.
- **Async internally, sync externally** -- `Pipeline.run()` is a sync call that wraps `asyncio.run()` internally. All LLM-touching code is `async def`.
- **OpenAI-compatible** -- works with any API that speaks the OpenAI chat completions protocol. Retries and rate limiting are handled internally.

### Module organization

```
src/syntheta/
  schema/         Sample model, SynthDataset, exporters, column mapper
  llm/            OpenAI-compatible backend, adaptive rate limiter, cost tracker
  pipeline/       Base ABCs, Pipeline orchestrator, checkpoint, writer
  generators/     TopicTree, Persona, SeedDataset
  transformers/   EvolInstruct (6 strategies), ResponseGenerator
  filters/        Safety (blocklist + LLM), Quality (LLM-as-judge), TemplateCollapse
  prompts/        External .txt templates + loader/renderer
  config/         Defaults, YAML loader with deep-merge
  cli/            Click commands (generate, inspect, validate, convert, recipes)
  observability/  Rich progress display, cost/health/filter reports
```

---

## Development

### Setup

```bash
git clone https://github.com/syntheta/syntheta.git
cd syntheta
pip install -e ".[dev]"
```

### Running tests

```bash
pytest                                    # All tests
pytest tests/unit/                        # Unit tests only
pytest tests/cli/                         # CLI tests only
pytest -m "not integration"               # Skip integration tests
pytest -m integration                     # Integration tests (requires OPENROUTER_API_KEY)
pytest --cov=syntheta                     # With coverage
```

### Linting

```bash
ruff check src/ tests/                    # Lint
ruff format src/ tests/                   # Format
```

### Code conventions

- Pydantic `BaseModel` for data classes
- `async def` for anything that calls the LLM
- All LLM prompts in external `.txt` files under `src/syntheta/prompts/` using `{{variable}}` syntax
- Unit tests mock `AsyncOpenAI`; integration tests require `OPENROUTER_API_KEY`
- Module-level `logger = logging.getLogger(__name__)` for logging
- `create_rng(seed)` for reproducible random number generation

---

## License

[Apache License 2.0](LICENSE)
