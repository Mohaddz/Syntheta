# Syntheta — Product Requirements Document

**Version:** 3.0
**Date:** March 25, 2026
**Status:** Planning

---

## 1. Executive summary

Syntheta is an open-source Python SDK and CLI for generating, filtering, and analyzing synthetic training datasets for large language models. It supports generating data from scratch (no seed data required), from existing datasets, and from documents/books — producing output suitable for SFT, chat fine-tuning, continued pretraining, DPO/preference optimization, and RLVR (judge-based) training paradigms.

Syntheta is research-backed: every generation technique implements a peer-reviewed method, and the SDK makes these techniques accessible through a simple Python API and YAML-configurable CLI.

**What makes Syntheta different from distilabel and other tools:**

1. **Generate from nothing.** Most tools require input data to transform. Syntheta builds datasets from scratch given only a domain description using topic-tree generation, persona-driven synthesis, and other zero-seed methods.

2. **Full training paradigm coverage.** Not just SFT/instruct — also pretraining data (with empirically-validated FinePhrase-style rephrasing), DPO preference pairs, multi-turn chat, and RLVR judge environments.

3. **Translation as a first-class feature.** Culturally-adapted translation with dialect/variant support, reference localization, back-translation verification, and naturalness scoring.

4. **Safety and cultural awareness built in.** Configurable safety filtering with cultural presets (Islamic, family-friendly, academic) ensures generated content respects cultural norms and audience requirements.

5. **Research-backed techniques, production-ready code.** Each generation method maps to a published paper. The implementation is engineered for reliability with streaming pipelines, checkpointing, adaptive rate limiting, and cost tracking.

---

## 2. Target audience

Syntheta serves four user segments, in order of priority:

1. **Open-source model builders** — Fine-tuning Llama, Qwen, Mistral, etc. Need to create SFT, DPO, and RLVR datasets quickly. Primary users.
2. **ML engineers at companies** — Building internal models or RAG systems. Need domain-specific training data.
3. **Individual researchers and hobbyists** — Experimenting with training recipes, need fast dataset prototyping.
4. **Data teams building training pipelines** — Need programmatic, reproducible dataset generation at scale.

---

## 3. Licensing and distribution

- **License:** Apache 2.0 (fully open source)
- **Package name:** TBD (candidate: `syntheta`, pending PyPI availability check)
- **Distribution:** PyPI, GitHub
- **CLI entry point:** `syntheta` command

---

## 4. Architecture overview

### 4.1 High-level design

```
┌─────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE                         │
│             Python API          YAML Config + CLI               │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                      STREAMING PIPELINE                         │
│  Generator → Transformers → Filters → Writer (chunked I/O)     │
│  Checkpointing, logging, batching, cost tracking, health report │
└──────┬──────────────────┬───────────────────┬───────────────────┘
       │                  │                   │
┌──────▼──────┐  ┌────────▼────────┐  ┌───────▼───────┐
│   SOURCES   │  │  TRANSFORMERS   │  │    FILTERS    │
│             │  │                 │  │               │
│ Generators  │  │ EvolInstruct    │  │ QualityFilter │
│ Loaders     │  │ ResponseGen     │  │ DiversityFilt │
│ Translators │  │ PreferencePairs │  │ SafetyFilter  │
│             │  │                 │  │ CollapseDetect│
└──────┬──────┘  └────────┬────────┘  └───────┬───────┘
       │                  │                   │
┌──────▼──────────────────▼───────────────────▼───────────────────┐
│              UNIFIED OPENAI-COMPATIBLE BACKEND                  │
│    /v1/chat/completions  +  /v1/embeddings                     │
│    Adaptive rate limiting, retries, concurrency, cost tracking  │
└─────────────────────────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────────┐
│                       I/O LAYER                                 │
│  ColumnMapper (auto-detect input) → Unified Sample → Exporters │
│                  Prompt templates (external, overridable)       │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Core abstractions

| Concept | Description |
|---------|-------------|
| **Sample** | One data point with all fields (instruction, response, chosen, rejected, turns, text, etc.). Single unified class with optional fields per paradigm. |
| **SynthDataset** | In-memory convenience wrapper for analysis, stats, and export. Loads from JSONL files produced by the pipeline. Not the pipeline's internal representation. |
| **Generator** | Creates samples from nothing or from a source. Yields batches (streaming). |
| **Transformer** | Takes batches in, yields batches out. Middle nodes (evolve, augment, generate responses). |
| **Filter** | Scores and removes low-quality, unsafe, or duplicate samples. Yields passing samples. |
| **Pipeline** | Chains Source → Transformers → Filters → Writer. Streams end-to-end with chunked checkpointing. |

### 4.3 LLM backend — unified OpenAI-compatible

**Everything goes through the OpenAI-compatible API.** One backend, two endpoint types:

- `/v1/chat/completions` — all generation, judging, translation, rephrasing
- `/v1/embeddings` — DiversityFilter, BackTranslationFilter

This covers: OpenAI, Together AI, Fireworks AI, Groq, DeepInfra, vLLM serve, Ollama, and any provider exposing these standard endpoints.

**Three model roles** (user picks which model to assign to each):

| Role | Used by | Recommendation |
|------|---------|----------------|
| `model` | Generation, topic trees, EvolInstruct, info building, quality scoring | Fast, cheap model |
| `response_model` | Response generation, translation, preference judging | Stronger model (optional — falls back to `model`) |
| `embedding_model` | DiversityFilter, BackTranslationFilter | Any embedding model via `/v1/embeddings` (optional — only needed if using these filters) |

**Config:**

```yaml
llm:
  base_url: null                         # null = OpenAI default; set for other providers
  api_key_env: OPENAI_API_KEY
  model: gpt-4o-mini
  response_model: null                   # optional, falls back to model
  embedding_model: null                  # optional, for filters that need embeddings
  max_concurrent: 10
  max_retries: 3
  timeout: 60
```

**Adaptive rate limiting:**

- On HTTP 429, automatically back off using the `retry-after` header (or exponential backoff if no header)
- Reduce concurrency temporarily when hitting rate limits, then gradually ramp back up
- Log every rate limit event with timestamp and backoff duration
- Users can set `max_tokens_per_minute` as an optional ceiling to stay under provider limits proactively

### 4.4 Streaming pipeline architecture

The pipeline streams end-to-end. Samples are never held entirely in memory — they flow through the pipeline in configurable batches and are written to disk incrementally.

**How it works:**

1. **Generator** yields batches of N samples (default batch_size: 100)
2. Each batch flows through **Transformers** sequentially (EvolInstruct, ResponseGenerator, etc.)
3. Each batch flows through **Filters** sequentially (SafetyFilter, QualityFilter, DiversityFilter, etc.)
4. Surviving samples are **written to JSONL** immediately (append mode)
5. After each batch, a **checkpoint** is saved (batch number + running stats)
6. If the pipeline crashes, resume from the last completed batch

**Two-pass operations:**

Some filters (DiversityFilter) need global context that a single streaming pass can't provide. These run as a second pass after the streaming pipeline completes:

- Pass 1 (streaming): Generate → Transform → Quality/Safety Filter → Write to temp JSONL
- Pass 2 (batch): Load temp JSONL → DiversityFilter (needs all embeddings) → Write final JSONL

This keeps the common case (no dedup) fully streaming while supporting global operations when needed.

**Scale characteristics:**

| Dataset size | Behavior |
|---|---|
| < 50K samples | Streams but effectively fits in memory. `SynthDataset` can load the whole thing for analysis. |
| 50K - 1M samples | Streaming is essential. Checkpointing prevents lost work. |
| > 1M samples (pretrain) | Streaming + chunked checkpointing + incremental Hub uploads. Matches FinePhrase's DataTrove pattern. |

**Future optimization — pipeline parallelism:**

v0.1 processes pipeline stages sequentially within each batch (generate batch → transform batch → filter batch → write). A future optimization is pipelined parallelism: transformer 2 starts processing batch 1 while transformer 1 works on batch 2. This would improve throughput for multi-stage pipelines but adds complexity to error handling and checkpointing. Acknowledged as a post-v1.0 optimization — the current sequential model is simpler, correct, and fast enough for most workloads since the bottleneck is LLM API latency, not local processing.

### 4.5 Prompt management

Every technique's prompts are stored as **external template files**, not hardcoded in Python.

**Directory structure:**

```
syntheta/prompts/
├── generators/
│   ├── topic_tree.txt
│   ├── persona.txt
│   └── seed_instruct.txt
├── transformers/
│   ├── evol_deepen.txt
│   ├── evol_broaden.txt
│   ├── evol_concretize.txt
│   ├── evol_rephrase.txt
│   ├── evol_complicate.txt
│   ├── evol_switch_task.txt
│   └── response_generator.txt
├── filters/
│   ├── quality_judge.txt
│   └── safety_check.txt
├── pretrain/
│   ├── faq.txt
│   ├── math.txt
│   ├── table.txt
│   └── tutorial.txt
└── translate/
    ├── culturally_adapted.txt
    ├── refinement.txt
    └── naturalness_judge.txt
```

**User overrides:** Any prompt can be overridden in config:

```yaml
prompts:
  evolve.deepen: ./my_prompts/custom_deepen.txt    # file path
  quality_judge: |                                   # inline
    Rate this sample on a scale of 1-5 for: ...
```

**Template variables:** Prompts use `{{variable}}` placeholders filled at runtime:

```
Given the domain "{{domain}}", generate {{n}} diverse subtopics covering:
- Different aspects of the subject
- Various difficulty levels from {{difficulty_min}} to {{difficulty_max}}
...
```

**Versioning:** Built-in prompts are versioned with the package. When we update a prompt in a new release, the changelog documents what changed. Users who override prompts are unaffected.

### 4.6 Reproducibility and seeding

**What Syntheta guarantees reproducible (given same seed):**

- Topic tree structure (same seed → same topics, same hierarchy)
- Sample ordering (deterministic iteration order)
- Distribution planning (same seed → same allocation of samples per topic/difficulty/language)
- Filter thresholds (deterministic given same scores)

**What Syntheta cannot guarantee reproducible:**

- LLM outputs (temperature > 0 produces different text across calls; even temperature = 0 is not guaranteed deterministic across API providers)
- Token counts and costs (vary with LLM non-determinism)

**Implementation:**

```yaml
seed: 42  # optional; if set, all deterministic operations use this seed
```

**Full prompt logging (opt-in):** When `log_prompts: true` is set in config (default: false), every LLM call is logged with:

- Timestamp
- Prompt sent (full text)
- Response received (full text)
- Token usage (prompt + completion)
- Model used
- Latency

Logs are saved to `{checkpoint_path}/prompts.jsonl` in append mode. This provides full auditability — a researcher can inspect exactly what the LLM was asked and what it returned for every sample in the dataset.

---

## 5. Data schema

### 5.1 Unified Sample

One class with optional fields per training paradigm. The design principle: a pipeline stage should work regardless of which paradigm the data targets.

```python
class Sample(BaseModel):
    id: str                                  # auto-generated UUID

    # ─── Core content (at least one group must be populated) ───
    instruction: str | None                  # single-turn user input
    response: str | None                     # single-turn model output
    system_prompt: str | None                # optional system context
    turns: list[Turn] | None                 # multi-turn conversation
    text: str | None                         # raw text (pretraining)

    # ─── Preference fields (DPO) ───
    chosen: str | None                       # preferred response
    rejected: str | None                     # dispreferred response
    chosen_score: float | None
    rejected_score: float | None

    # ─── RLVR fields ───
    info: dict | None                        # metadata passed to judge/verifier

    # ─── Metadata ───
    domain: str | None
    topic: str | None
    persona: str | None
    task_type: str | None
    language: str | None
    variant: str | None                      # dialect / regional variety
    formality: str | None                    # casual / neutral / formal / honorific
    difficulty: int | None                   # 1-5 scale

    # ─── Quality & provenance ───
    quality_score: float | None              # 0.0-1.0
    quality_reason: str | None
    safety_passed: bool | None               # True if passed safety filter
    generation_model: str | None
    source_id: str | None                    # link to original if augmented
    evolution_history: list[str]             # chain of transforms applied

    # ─── Escape hatch ───
    extra: dict[str, Any]                    # all unrecognized fields preserved here
```

**Key design rules:**

1. `extra` is the escape hatch — any input column that doesn't map to a known field goes into `extra`, survives the entire pipeline, and appears in exports. On export: flattened by default (each key becomes a column), with option to drop or nest as JSON.
2. The `Turn` type: `{role: "user"|"assistant"|"system", content: str}`.
3. At least one content group must be populated: `instruction`, `turns`, or `text`.

### 5.2 Input: ColumnMapper

Auto-detects input format by examining column names. Supports explicit user overrides.

**Known format signatures (exact column match):**

| Columns present | Detected format |
|----------------|-----------------|
| `instruction`, `input`, `output` | Alpaca |
| `conversations` (list of dicts) | ShareGPT |
| `messages` (list of dicts) | OpenAI messages |
| `prompt`, `chosen`, `rejected` | DPO |
| `question`, `answer` | QA / verifiers |
| `question`, `info` | RLVR judge |
| `text` | Pretraining |

**Fuzzy synonym mapping (when auto-detect fails):**

| Internal field | Recognized synonyms |
|---------------|---------------------|
| `instruction` | instruction, prompt, query, question, input, user |
| `response` | response, output, answer, completion, assistant, reply |
| `chosen` | chosen, preferred, accepted, positive |
| `rejected` | rejected, dispreferred, negative, worse |

**User override:** `column_map={"my_col": "instruction"}` always takes precedence.

**Supported input file formats:** JSONL, JSON, CSV, TSV, Parquet, HuggingFace Dataset (local or Hub).

### 5.3 Output: Typed exporters

Strict validation on export. Each export format checks that the required fields are populated and raises a clear error if not.

| Export format | Required fields | Column names in output |
|---------------|----------------|----------------------|
| SFT | instruction + response | instruction, response |
| Chat / ShareGPT | turns | conversations: [{from, value}] |
| OpenAI messages | turns | messages: [{role, content}] |
| DPO | instruction + chosen + rejected | prompt, chosen, rejected |
| RLVR (judge) | instruction + info | question, info |
| Pretraining | text | text |
| Scored | instruction + response + quality_score | prompt, response, score |

**Output file formats:** JSONL, JSON, CSV, Parquet, HuggingFace Dataset (push to Hub).

**Column renaming:** Users can rename any column at export time: `rename={"instruction": "prompt"}`.

**HuggingFace Hub integration:** `dataset.push_to_hub("org/name")` — direct upload with auto-generated dataset card including generation metadata, pipeline config, and stats.

---

## 6. Generation techniques

Every technique maps to at least one peer-reviewed paper. Organized by data source.

All techniques use the unified LLM backend. The user configures which model to use in the `llm` config block — Syntheta doesn't mandate specific models.

### 6.1 From scratch (no seed data)

#### 6.1.1 TopicTreeGenerator

**Paper:** TreeSynth (2025), Seed-Free SDG (2024)

**How it works:**

1. Takes a domain description
2. Builds a hierarchical topic taxonomy by prompting the LLM recursively
3. Plans a distribution: how many samples per (topic × task_type × language × difficulty)
4. Generates diverse instructions by sampling across the tree
5. Optionally generates responses in the same pass

**Parameters:**

- `domain`: str
- `task_types`: list — qa, explain, compare, creative, instruct, debate, classify, summarize (+ custom)
- `languages`: list — ISO-639 codes
- `difficulty_range`: (int, int) — 1-5 scale
- `topic_depth`: int (2-4 recommended)
- `topic_breadth`: int (3-8 recommended)
- `generate_responses`: bool

#### 6.1.2 PersonaGenerator

**Paper:** Scaling Synthetic Data with 1B Personas (2024), MATRIX-Gen (ACL 2025)

**How it works:**

1. Generates N diverse synthetic personas
2. Each persona generates questions from their perspective about the domain
3. Produces more naturalistic, varied instructions than topic-only approaches

**Parameters:**

- `domain`: str
- `n_personas`: int
- `questions_per_persona`: int
- `persona_seed_traits`: list
- `languages`: list
- `generate_responses`: bool

#### 6.1.3 MagpieGenerator [P2]

**Paper:** Magpie (ICLR 2025)

Feeds an aligned LLM only its pre-query chat template. Requires access to model weights or a provider that supports empty-template inference. This is the one exception to the "everything via standard API" rule.

### 6.2 From existing datasets

#### 6.2.1 SeedDatasetGenerator (Self-Instruct)

**Paper:** Self-Instruct (ACL 2023)

Samples a few examples from an existing dataset as few-shot prompts, asks the LLM to generate new, different instructions. Iterative.

#### 6.2.2 InstructionBacktranslation [P2]

**Paper:** Instruction Backtranslation (2023), Apple AFM (2024)

Starts with high-quality responses and reverse-engineers the instruction.

#### 6.2.3 Genstruct [P2]

**Paper:** Genstruct (2024), Source2Synth (2025)

Takes raw text corpus and generates instruction-response pairs grounded in the text content.

### 6.3 From documents and books

#### 6.3.1 DocumentQAGenerator

**Input:** PDF, EPUB, or plain text files.

**How it works:**

1. Parse document into chunks (respecting section boundaries)
2. For each chunk, generate diverse QA pairs, explanation requests, summarization tasks
3. Metadata tracks which chunk/page sourced each sample

**Additional dependency:** PyMuPDF or pdfplumber for PDF parsing (optional install extra).

#### 6.3.2 PretrainRewriter

**Papers:** WRAP (2024), Phi series (2023-2024), FinePhrase / Synthetic Data Playbook (HuggingFace, 2026)

Rewrites existing text into structured formats for pretraining data.

**Four built-in rephrasing strategies** (empirically validated by FinePhrase, 90 experiments, 1T+ tokens):

| Strategy | Description | Benchmark signature |
|----------|-------------|--------------------|
| `faq` | Rewrite as comprehensive FAQ | Strong on reading comprehension |
| `math` | Rewrite as math word problem with solution | Only strategy that moves GSM8K |
| `table` | Rewrite as structured table with QA pair | Strongest ARC boost |
| `tutorial` | Rewrite as step-by-step tutorial | Only strategy that improves DROP |

**Key research findings integrated:**

1. Prompt design is the single biggest lever — these four formats are the defaults.
2. 1B-class models are sufficient for rephrasing. Scaling up provides no improvement.
3. Synthetic-only pretraining is never enough — always mix ~30% synthetic with original data.
4. Source data quality is secondary when paired with a strong mix-in dataset.
5. Template diversity beats template polish — the TemplateCollapseDetector catches this.
6. Quality scores don't predict downstream performance for synthetic pretraining data.

### 6.4 Transformers (post-generation)

#### 6.4.1 EvolInstruct

**Paper:** WizardLM / Evol-Instruct (2023)

Six mutation strategies: deepen, broaden, concretize, rephrase, complicate, switch_task. Each strategy has its own external prompt template in `syntheta/prompts/transformers/`.

#### 6.4.2 ResponseGenerator

Generates responses for instructions that don't have them. Uses `response_model` from config (falls back to `model`). Supports custom system prompts and CoT for complex questions.

#### 6.4.3 PreferencePairGenerator [P2]

**Paper:** UltraFeedback (2024), RLAIF (2024)

Generates K candidate responses, scores with LLM-as-judge, picks best/worst as chosen/rejected.

#### 6.4.4 ArenaGenerator [P2]

**Paper:** Arena Learning (2024)

Multiple different LLMs answer the same prompt. Requires configuring multiple model endpoints.

#### 6.4.5 RLVRInfoBuilder [P2]

Enriches samples with structured `info` metadata for RLVR judge-based training.

### 6.5 Filters

#### 6.5.1 QualityFilter (LLM-as-judge)

Multi-dimensional scoring: relevance, clarity, complexity, naturalness. Prompt template in `syntheta/prompts/filters/quality_judge.txt` — user-overridable.

**Caveat for pretraining data:** Per FinePhrase research, LLM-based quality scores do not reliably predict downstream performance for synthetic pretraining data.

#### 6.5.2 SafetyFilter

Two-layer content filtering for safety and cultural appropriateness.

**Layer 1: Universal safety (always on, non-configurable)**

Blocks content referencing: CSAM, detailed violence/self-harm instructions, and other universally harmful material. Uses keyword matching + optional LLM check. Prompt template: `syntheta/prompts/filters/safety_check.txt`.

**Layer 2: Cultural/domain filter (configurable)**

```yaml
safety:
  enabled: true
  cultural_context: "islamic"            # built-in preset (or null for none)
  blocklist_topics: []                   # additional topic blocklist
  blocklist_words: []                    # additional word blocklist
  custom_instruction: null               # appended to LLM safety check prompt
```

**Built-in cultural presets:**

| Preset | Blocks |
|--------|--------|
| `islamic` | Alcohol, gambling, adult entertainment, pork-related content, content contradicting Islamic values |
| `family-friendly` | Adult content, violence, profanity, substance use |
| `academic` | Informal language, slang, opinioned content, marketing/promotional tone |

Users can extend presets with `blocklist_topics` and `blocklist_words` for domain-specific needs, or create fully custom profiles via `custom_instruction`.

**Interaction with translation module:** Syntheta does NOT auto-apply cultural filters based on target language — that would make assumptions about user intent. Instead, when a translation pipeline runs without a `cultural_context` set, the health report will surface a warning: "Translating to Arabic without cultural context — consider setting `safety.cultural_context: islamic` if your audience requires it." The user decides. Cultural filters are always opt-in, never silently applied.

#### 6.5.3 DiversityFilter (embedding-based)

Uses the `/v1/embeddings` endpoint via the configured `embedding_model`. Two modes:

- **Dedup:** Greedy removal of near-duplicates above cosine similarity threshold
- **Diverse:** Dedup + MMR-style selection to maximize coverage of the embedding space

**Note:** This is a two-pass filter. The streaming pipeline writes samples to a temp file first, then DiversityFilter runs as a batch operation on the full output.

#### 6.5.4 TemplateCollapseDetector

**Inspired by:** FinePhrase / Synthetic Data Playbook (2026)

Heuristic analysis — no model needed. Detects repetitive output prefixes that indicate the generation model is stuck in a template.

**Parameters:**

- `prefix_length`: int — tokens to compare (default 50)
- `max_repeat`: int — maximum allowed repetitions of any prefix (default 3)
- `action`: "flag" | "remove" | "report" (default "report")

#### 6.5.5 BackTranslationFilter [P2]

For translation quality: translate back, compare semantic similarity via embeddings.

#### 6.5.6 NaturalnessJudge [P2]

LLM judges whether text sounds native or translated.

---

## 7. Translation module (v1)

### 7.1 Overview

Not just "translate text" — a multi-stage pipeline handling cultural adaptation, dialect/variant support, reference localization, and quality verification.

### 7.2 Translation strategies

#### Strategy 1: Culturally-adapted translation

For universal content. Translates the text, replaces culturally specific references, validates with back-translation.

#### Strategy 2: Cross-lingual structure transfer

Takes the structure of a source dataset but regenerates content natively in the target language.

#### Strategy 3: Variant adaptation

Converts between language variants: dialect, regional variety, formality register.

### 7.3 Translation pipeline stages

1. **Source quality filter** — remove bad rows before translating
2. **Safety filter** — apply cultural context for target language
3. **Field policy assignment** — which fields to translate, preserve, or adapt
4. **Chunked thematic batching** — group related rows for terminology consistency
5. **Structure-aware translation** — translate instruction + response together
6. **Refinement pass** — second LLM call to polish naturalness (uses `response_model`)
7. **Glossary enforcement** — ensure domain terms are translated consistently
8. **Reference localization** — swap names, places, currencies, units, cultural references
9. **Back-translation consistency check** — translate back, compare semantic similarity
10. **Naturalness + difficulty scoring** — LLM judge: does it sound native?
11. **Format preservation check** — lists still lists, code intact, markdown preserved

### 7.4 Parameters

- `target_lang`: str — ISO-639 code
- `variant`: str — dialect or regional variety (optional)
- `formality`: str — casual / neutral / formal / honorific (optional)
- `strategy`: str — "culturally_adapted", "cross_lingual_transfer", "variant_adapt"
- `localize_references`: bool
- `glossary`: dict — term → translation mapping
- `refine`: bool — two-pass translation
- `back_translate_filter`: bool
- `min_consistency`: float
- `field_policies`: dict — per-field translation behavior

---

## 8. Multi-turn / chat data generation (v1)

### 8.1 Approach

Three methods:

#### Method 1: Simulated conversation

Two LLM calls role-play as user and assistant. Uses `model` for user simulation and `response_model` for assistant responses.

#### Method 2: Single-turn expansion

Expand existing single-turn data with follow-ups, clarifications, topic pivots, and corrections.

#### Method 3: Tree-branching

Generate a first turn, then branch into multiple possible continuations.

### 8.2 Output format

Multi-turn data uses the `turns` field. Exports to ShareGPT, OpenAI messages, or Alpaca formats.

---

## 9. Error handling and observability

### 9.1 Structured logging

Every pipeline run produces structured logs with categorized events:

| Category | Examples | Level |
|----------|----------|-------|
| `generation` | Sample generated, prompt sent, response received | DEBUG |
| `filter_reject` | Quality below threshold, safety blocked, duplicate detected, template collapse | INFO |
| `rate_limit` | 429 received, backoff applied, concurrency reduced | WARNING |
| `malformed_response` | JSON parse failed, empty response, unexpected format | WARNING |
| `api_error` | Timeout, connection error, 5xx response | ERROR |
| `checkpoint` | Batch completed, checkpoint saved, pipeline resumed | INFO |

### 9.2 Post-run filter summary

After every pipeline run, a filter summary is printed and saved:

```
Filter summary:
  Input:     6,500 samples generated
  Output:    5,253 samples passed (80.8%)

  Rejected:  1,247 samples (19.2%)
    safety_blocked:           142  (2.2%)
    quality_below_threshold:  758  (11.7%)
    duplicate:                234  (3.6%)
    template_collapse:         87  (1.3%)
    malformed_llm_response:    26  (0.4%)

  Warnings:
    Rate limits hit: 14 times (avg backoff: 2.3s)
    Retries needed:  89 / 6,500 calls (1.4%)
```

This is saved to `{checkpoint_path}/run_summary.json` for programmatic access.

### 9.3 Prompt logging

When `log_prompts: true` (default: false, for cost/storage reasons), every LLM call is logged to `{checkpoint_path}/prompts.jsonl`:

```json
{"timestamp": "...", "stage": "topic_tree", "model": "gpt-4o-mini", "prompt": "...", "response": "...", "tokens": {"prompt": 234, "completion": 89}, "latency_ms": 1203}
```

### 9.4 Fail-fast conditions

Some errors should abort the pipeline immediately without retry:

| Error | Detection | Message |
|-------|-----------|---------|
| API key invalid | HTTP 401 on first call | "Invalid API key — check `api_key_env` in config." |
| API key env var not set | Config load time | "Environment variable `OPENAI_API_KEY` not set." |
| Model not found | HTTP 404 on first call | "Model 'xyz' not found. Check `llm.model` in config." |
| Disk full | OSError on write | "Disk full — pipeline paused at batch N. Free space and use `--resume`." |
| Persistent rate limit | Backoff exceeds `max_backoff` (default 300s) | "Rate limited for >5 minutes. Pipeline paused. Check provider quotas or reduce `max_concurrent`." |
| Invalid config | Pydantic validation at load time | Standard Pydantic error with field path and expected type. |

These are checked at pipeline start (preflight) or on first occurrence. No retries on 401 or 404. Disk full and persistent rate limits pause the pipeline (resumable) rather than crashing.

---

## 10. Dataset health report

After every pipeline run, Syntheta automatically generates a health report summarizing the dataset quality and coverage. This is the foundation for the Phase 3 gap analysis features.

### 10.1 Report contents

```
Dataset Health Report
═══════════════════════════════════════════════════

Samples: 5,000 (6,500 generated, 1,247 filtered, 253 safety-blocked)

Coverage:
  Topics:      47 unique (target: 50 — missing: music, cuisine, architecture)
  Task types:  4/4 ✓ (qa: 28%, explain: 25%, compare: 24%, creative: 23%)
  Difficulty:  ⚠ skewed (1: 30%, 2: 40%, 3: 20%, 4: 7%, 5: 3%)
  Languages:   ar: 2,480 (49.6%) / en: 2,520 (50.4%) ✓

Quality:
  Mean score:  0.78  (min: 0.60, p10: 0.65, median: 0.79, p90: 0.91)
  Bottom 10%:  concentrated in topics "greetings" and "weather"

Diversity:
  Template collapse ratio: 0.4% ✓
  Top redundant clusters:  "greetings" (89 samples), "weather" (67)

Safety:
  Blocked:     142 samples (2.2%)
  Top reasons: alcohol_reference (63), gambling_reference (41), other (38)

Warnings:
  ⚠ 3 target topics have zero coverage
  ⚠ Difficulty distribution heavily skewed toward easy (70% at levels 1-2)
  ⚠ Bottom 10% quality concentrated in 2 topics — consider regenerating
```

### 10.2 Machine-readable output

The report is also saved as `{output_path}.health.json` for programmatic consumption:

```json
{
  "total_samples": 5000,
  "generated": 6500,
  "filtered": 1247,
  "coverage": {
    "topics": {"found": 47, "target": 50, "missing": ["music", "cuisine", "architecture"]},
    "difficulty_distribution": {"1": 0.30, "2": 0.40, "3": 0.20, "4": 0.07, "5": 0.03}
  },
  "quality": {"mean": 0.78, "p10": 0.65, "p90": 0.91},
  "warnings": ["3 target topics have zero coverage", "..."]
}
```

This JSON is the input format for Phase 3's `syntheta analyze` and `syntheta fill` commands.

---

## 11. Config system and CLI

### 11.1 Design principles

Layered overrides:

```
Defaults (in code)
  ↓ overridden by
Recipe (built-in preset)
  ↓ overridden by
Config file (user's YAML)
  ↓ overridden by
CLI flags (quick overrides)
```

### 11.2 Config file format (YAML)

```yaml
# example_sft.yaml
config_version: "0.1"
task: generate
seed: 42

domain: "Arabic culture and traditions"
n: 10000
method: topic_tree

languages: [ar, en]
task_types: [qa, explain, compare, creative]
difficulty: [1, 5]

topic_tree:
  depth: 3
  breadth: 5

evolve:
  enabled: true
  rounds: 2
  strategies: [deepen, concretize, complicate]

responses:
  enabled: true
  use_cot: true

filters:
  safety:
    enabled: true
    cultural_context: "islamic"
  quality:
    enabled: true
    min_score: 0.7
  dedup:
    enabled: true
    threshold: 0.85
  template_collapse:
    enabled: true
    max_repeat: 3

llm:
  base_url: null
  api_key_env: OPENAI_API_KEY
  model: gpt-4o-mini
  response_model: gpt-4o
  embedding_model: text-embedding-3-small
  max_concurrent: 10

output: arabic_culture_10k.jsonl
checkpoint_path: ./checkpoints
log_prompts: false
batch_size: 100                          # samples per streaming batch
over_generate_factor: 1.2               # generate 20% extra to account for filtering losses
```

### 11.3 CLI commands

```bash
# ─── Generation ───
syntheta generate --config config.yaml
syntheta generate --recipe sft --domain "Python" --n 500
syntheta generate --domain "Math" --n 100 --output out.jsonl

# ─── Dry run (estimate cost before committing) ───
syntheta generate --config config.yaml --dry-run
# Output: "Estimated: ~10,000 LLM calls, ~5M tokens, ~$0.75 at gpt-4o-mini pricing"

# ─── Augmentation ───
syntheta augment --source data.jsonl --n 5000 --config augment.yaml

# ─── Translation ───
syntheta translate --source data.jsonl --target-lang ko --config translate.yaml

# ─── Utilities ───
syntheta inspect data.jsonl
syntheta validate data.jsonl --format sft
syntheta convert data.jsonl --to sharegpt --output out.json

# ─── Recipes ───
syntheta recipes list
syntheta recipes show sft
syntheta recipes export sft --output my_config.yaml

# ─── Config overrides (dot notation) ───
syntheta generate --config base.yaml --n 2000 --llm.model gpt-4o --filters.quality.min_score 0.8
```

### 11.4 Built-in recipes

Recipes are standard YAML config files bundled inside the package at `syntheta/recipes/`. Each recipe is a complete config with sensible defaults — users override only what they need.

```
syntheta/recipes/
├── sft.yaml
├── sft_persona.yaml
├── sft_multilingual.yaml
├── chat.yaml
├── preference.yaml
├── rlvr_judge.yaml
├── pretrain.yaml
├── pretrain_mix.yaml
├── augment.yaml
└── translate.yaml
```

| Recipe | Description |
|--------|-------------|
| `sft` | Basic SFT dataset: topic tree + responses + quality filter + safety filter |
| `sft-persona` | Persona-driven SFT generation |
| `sft-multilingual` | SFT with multiple languages |
| `chat` | Multi-turn conversation generation |
| `preference` | DPO preference pairs from instructions |
| `rlvr-judge` | RLVR judge environment data (question + info) |
| `pretrain` | Pretraining data: rephrase source text into FAQ/Math/Table/Tutorial |
| `pretrain-mix` | Pretrain + automatic mixing with original data (default 30/70) |
| `augment` | Expand existing dataset |
| `translate` | Translate existing dataset with cultural adaptation |

### 11.5 Dry-run cost estimation

`--dry-run` runs the planning phase (topic tree generation, distribution allocation) without executing the full pipeline:

- Counts planned samples and pipeline stages
- Estimates tokens per call using **built-in technique averages** (no tokenizer needed — these are historical averages per technique, not text-level token counting)
- Looks up model pricing from the bundled `model_prices.json` (sourced from LiteLLM, cached locally, refreshed weekly)
- Reports estimated total cost and time

```
$ syntheta generate --config arabic_sft.yaml --dry-run

Dry-run estimate (not a commitment — actual costs will vary):
  Planned samples:     12,000 (10,000 target × 1.2 over-generation)
  Pipeline stages:     4 (topic_tree → evol_instruct → response_generator → quality_filter)
  Estimated LLM calls: ~48,000
  Estimated tokens:    ~28M (prompt: ~19M / completion: ~9M)
  Estimated cost:      ~$4.20 (at gpt-4o-mini pricing)
  Estimated time:      ~8 min (at max_concurrent: 10)
```

**How technique averages work:** Syntheta ships with a table of average prompt/completion tokens per technique (e.g., `topic_tree: ~800 prompt / ~400 completion`, `response_generator: ~500 prompt / ~800 completion`). These averages are updated automatically from actual run data — each post-run summary feeds back into the estimates, so they improve over time.

### 11.6 `syntheta inspect` output spec

`syntheta inspect data.jsonl` provides a quick overview of any dataset:

```
Dataset: data.jsonl
═══════════════════════════════════════════════════

Basics:
  Samples:    5,000
  File size:  12.3 MB
  Format:     JSONL (auto-detected: SFT/Alpaca)

Field population:
  instruction:   5,000 / 5,000  (100%)
  response:      4,812 / 5,000  (96.2%)
  system_prompt:   500 / 5,000  (10.0%)
  quality_score: 4,812 / 5,000  (96.2%)
  domain:        5,000 / 5,000  (100%)
  topic:         4,998 / 5,000  (99.9%)
  language:      5,000 / 5,000  (100%)
  difficulty:    5,000 / 5,000  (100%)

Distributions:
  Languages:   ar: 2,480 (49.6%) | en: 2,520 (50.4%)
  Task types:  qa: 1,400 (28%) | explain: 1,250 (25%) | compare: 1,200 (24%) | creative: 1,150 (23%)
  Difficulty:  1: 1,500 (30%) | 2: 2,000 (40%) | 3: 1,000 (20%) | 4: 350 (7%) | 5: 150 (3%)
  Top topics:  history (312) | religion (289) | geography (245) | traditions (231) | language (198) | ...

Quality (if scored):
  Mean: 0.78 | Median: 0.79 | p10: 0.65 | p90: 0.91

Sample preview (first 3):
  [1] "What are the main ingredients in traditional Saudi Arabian kabsa?" → "Kabsa is a..."
  [2] "Explain the significance of the Hajj pilgrimage..." → "The Hajj is one of..."
  [3] "Compare the architectural styles of mosques in..." → "Mosque architecture varies..."
```

Flags: `--json` for machine-readable output, `--full` for all distributions (not just top-5), `--samples N` to show N preview samples.

### 11.7 `syntheta validate` rules

`syntheta validate data.jsonl --format sft` checks that a dataset meets format requirements:

**Structural checks (all formats):**

- File is valid JSONL / JSON / CSV / Parquet (parseable without errors)
- Every row has a valid `id` field (or one can be auto-assigned)
- No completely empty rows
- UTF-8 encoding (no invalid byte sequences)

**Format-specific checks:**

| Format | Required fields | Checks |
|--------|----------------|--------|
| `sft` | instruction + response | Both non-empty, non-whitespace-only, response > 10 chars |
| `chat` | turns | At least 2 turns, alternating user/assistant roles, no empty content |
| `dpo` | instruction + chosen + rejected | All three non-empty, chosen ≠ rejected |
| `rlvr` | instruction + info | Instruction non-empty, info is valid JSON dict |
| `pretrain` | text | Non-empty, > 50 chars |

**Quality checks (optional, with `--strict`):**

- No duplicate instructions (exact match)
- No excessively short responses (< 20 tokens)
- No responses that are just the instruction repeated
- Language field matches detected language (if `langdetect` available)

**Output:**

```
Validation: data.jsonl (format: sft)
  ✓ 4,987 / 5,000 samples passed
  ✗ 13 samples failed:
    - 7 empty response fields (rows: 234, 891, ...)
    - 4 whitespace-only instruction (rows: 12, 1033, ...)
    - 2 response shorter than 10 chars (rows: 4521, 4522)
```

Exit code 0 if all pass, exit code 1 if any fail (for CI integration).

### 11.8 Config versioning

Every config file and recipe includes a `config_version` field:

```yaml
config_version: "0.1"    # matches the Syntheta version that created it
task: generate
...
```

**Compatibility rules:**

- Syntheta loads configs from the same major.minor version without warnings
- Configs from older versions produce a deprecation warning listing what changed, but still run (best-effort backward compatibility)
- If a config uses a removed field, Syntheta errors with a clear message: "Field 'X' was removed in v0.2. Use 'Y' instead."
- Built-in recipes always use the current version's schema — they're never stale
- `syntheta recipes export` always writes the current version's format

---

## 12. Cost tracking

### 12.1 Two approaches, no tokenizer needed

| When | Source of token counts | Source of pricing | Accuracy |
|------|----------------------|-------------------|----------|
| **Post-run** (actual) | API response `usage.prompt_tokens` / `usage.completion_tokens` | `model_prices.json` or user override | Exact tokens, accurate pricing |
| **Dry-run** (estimate) | Technique average table (arithmetic, no tokenizer) | `model_prices.json` or user override | Rough estimate (~±30%) |

No tokenizer library is needed. Post-run token counts come directly from the API response (authoritative). Dry-run estimates use historical averages per technique (pure arithmetic).

### 12.2 Pricing lookup

Pricing is resolved in this priority order:

1. **User config override** — if the user specifies pricing in YAML, use that (they know their deal):
   ```yaml
   llm:
     model: my-custom-model
     pricing:
       prompt_per_million: 0.15
       completion_per_million: 0.60
   ```
2. **Bundled `model_prices.json`** — sourced from LiteLLM's community-maintained pricing database (300+ models). Bundled at release time, auto-refreshed from GitHub weekly and cached locally in `~/.syntheta/model_prices.json`. Falls back to bundled version if fetch fails (offline, firewall).
3. **Unknown model** — report token counts only, skip dollar estimate, log a warning: "Model 'xyz' not found in pricing database. Set `llm.pricing` in config for cost estimates."

### 12.3 Post-run report

```
Pipeline complete: 5,000 samples in 342.1s

Cost summary:
  Total tokens:     2,847,320 (prompt: 1,923,000 / completion: 924,320)
  Estimated cost:   $4.27 (at gpt-4o-mini pricing)

  By stage:
    topic_tree:           412,000 tokens  ($0.62)    14.5%
    evol_instruct:        387,000 tokens  ($0.58)    13.6%
    response_generator: 1,648,320 tokens  ($2.47)    57.9%
    quality_filter:       400,000 tokens  ($0.60)    14.0%
```

Per-stage breakdown and total saved to `run_summary.json` for programmatic access.

---

## 13. Checkpointing and fault tolerance

### 13.1 Pipeline checkpointing

Integrated with the streaming architecture. After each batch:

1. Completed samples written to JSONL
2. Running stats (count, cost, filter reasons) saved to checkpoint
3. Batch number recorded for resume

```
checkpoints/
├── batch_0001.jsonl              # samples from batch 1
├── batch_0002.jsonl              # samples from batch 2
├── ...
├── state.json                    # current batch, running stats, cost tracker, config hash
├── run_summary.json              # filter summary (updated after each batch)
├── health_report.json            # dataset health (updated at end)
└── prompts.jsonl                 # prompt log (if enabled)
```

**Consolidation:** When the pipeline completes successfully, all `batch_*.jsonl` files are concatenated into the final output file (specified by `output` in config). If a two-pass filter (DiversityFilter) is configured, consolidation happens after pass 2. The checkpoint directory can then be deleted (or kept for debugging). The final output path is always a single JSONL file — users never need to interact with batch files.

### 13.2 Resume

If a pipeline crashes, `syntheta generate --config config.yaml --resume` picks up from the last completed batch. The `state.json` file contains everything needed to resume.

**Config change detection:** On resume, Syntheta computes a hash of the current config and compares it to the hash stored in `state.json`. If the config changed (different model, different filters, different n), Syntheta warns and asks for confirmation before continuing. This prevents silently mixing outputs from different configurations in the same run.

### 13.3 Fault tolerance

- Failed LLM calls retry with exponential backoff
- Malformed responses logged and skipped
- Pipeline over-generates by configurable factor (default: 1.2x) to account for filtering losses
- Rate limit events trigger adaptive concurrency reduction

---

## 14. Extensibility

### 14.1 Custom generators, transformers, and filters

Syntheta provides a simple registration API for user-defined components:

```python
from syntheta import register_generator, register_transformer, register_filter

@register_generator("my_custom_gen")
class MyGenerator(BaseGenerator):
    def generate(self, n: int, **kwargs) -> Iterator[list[Sample]]:
        ...

@register_transformer("my_custom_transform")
class MyTransformer(BaseTransformer):
    def transform(self, samples: list[Sample]) -> list[Sample]:
        ...

@register_filter("my_custom_filter")
class MyFilter(BaseFilter):
    def filter(self, samples: list[Sample]) -> list[Sample]:
        ...
```

Registered components are available in YAML configs and CLI:

```yaml
method: my_custom_gen
filters:
  my_custom_filter:
    enabled: true
    my_param: 42
```

### 14.2 Custom prompt templates

Any built-in prompt can be overridden per the prompt management system (Section 4.5).

### 14.3 Custom safety presets

Users can define cultural presets as YAML files:

```yaml
# presets/conservative_arabic.yaml
name: conservative_arabic
extends: islamic
blocklist_topics:
  - mixed_gender_socializing
  - music_instruments
blocklist_words:
  - ...
custom_instruction: "Content must be appropriate for a conservative Gulf Arab audience"
```

---

## 15. Research references

| Technique | Paper | Year | Venue |
|-----------|-------|------|-------|
| TopicTreeGenerator | TreeSynth | 2025 | arXiv |
| TopicTreeGenerator | Seed-Free SDG | 2024 | arXiv |
| PersonaGenerator | Scaling Synthetic Data with 1B Personas | 2024 | arXiv |
| PersonaGenerator | MATRIX-Gen | 2025 | ACL |
| MagpieGenerator | Magpie | 2025 | ICLR |
| SeedDatasetGenerator | Self-Instruct | 2023 | ACL |
| InstructionBacktranslation | Instruction Backtranslation | 2023 | arXiv |
| Genstruct | Genstruct | 2024 | arXiv |
| EvolInstruct | WizardLM / Evol-Instruct | 2023 | arXiv |
| PreferencePairGenerator | UltraFeedback | 2024 | arXiv |
| PreferencePairGenerator | RLAIF | 2024 | arXiv |
| ArenaGenerator | Arena Learning | 2024 | arXiv |
| PretrainRewriter | WRAP / Rephrasing the Web | 2024 | ACL |
| PretrainRewriter | Phi series | 2023-2024 | arXiv |
| PretrainRewriter | FinePhrase / Synthetic Data Playbook | 2026 | HuggingFace |
| TemplateCollapseDetector | FinePhrase / Synthetic Data Playbook | 2026 | HuggingFace |
| QualityFilter | UltraFeedback (multi-aspect judging) | 2024 | arXiv |
| DiversityFilter | MMR: Maximal Marginal Relevance | 1998 | SIGIR |
| CulturalTranslation | PALM: Culturally Inclusive Arabic Dataset | 2025 | ACL |
| RLVR environments | DeepSeek-R1 | 2025 | Nature |
| BootstrappedDPO | Bootstrapping LMs with DPO | 2025 | ICLR |
| ConstitutionalAI | Constitutional AI | 2022 | arXiv |
| EntiGraph | EntiGraph: Entity-relation synthesis | 2025 | ICLR |
| Model collapse (non-issue) | Demystifying Synthetic Data — 1000 LLM study | 2025 | EMNLP |

---

## 16. Phased roadmap

### Phase 1: v0.1 — "Generate" (MVP)

The sharp MVP: generate data from scratch, filter it, export it.

**Generators:**
1. TopicTreeGenerator
2. PersonaGenerator
3. SeedDatasetGenerator (Self-Instruct)

**Transformers:**
4. EvolInstruct (6 strategies)
5. ResponseGenerator (with CoT)

**Pretrain:**
6. PretrainRewriter (4 FinePhrase strategies)

**Filters:**
7. SafetyFilter (universal + cultural presets)
8. QualityFilter (LLM-as-judge)
9. TemplateCollapseDetector

**Infrastructure:**
10. Unified OpenAI-compat backend with adaptive rate limiting
11. Streaming pipeline with chunked checkpointing
12. Unified Sample schema + ColumnMapper
13. YAML config with `config_version` + CLI with recipes
14. JSONL output (primary format)
15. External prompt templates (user-overridable)
16. Seed / reproducibility
17. Structured logging + filter summary
18. Dataset health report (auto-generated)
19. Cost tracking (post-run)
20. `--dry-run` estimation
21. `syntheta inspect` (basic: field stats, distributions, previews)
22. `syntheta validate` (structural + format-specific checks)

**Deferred to v0.1.1 (quick follow-up):**
- DocumentQAGenerator (requires PDF parsing dependency)
- Multi-turn chat generation
- DiversityFilter (requires embedding model — adds complexity to first install)
- HuggingFace Hub push
- CSV/Parquet/JSON export (JSONL first)
- `pretrain-mix` recipe

### Phase 2: v0.2 — "Expand + Translate"

**Generators:**
23. MagpieGenerator
24. InstructionBacktranslation
25. Genstruct
26. DocumentQAGenerator (if not shipped in v0.1.1)

**Transformers:**
27. PreferencePairGenerator
28. ArenaGenerator
29. RLVRInfoBuilder

**Filters:**
30. DiversityFilter (embedding-based, if not shipped in v0.1.1)
31. BackTranslationFilter
32. NaturalnessJudge
33. ContaminationFilter (lightweight 8-gram overlap check against curated list of common benchmarks: GSM8K, MMLU, HumanEval, ARC, HellaSwag, TruthfulQA). Flags samples that overlap with known eval sets — critical for anyone generating math/code/reasoning data.

**Translation module:**
34. CulturalTranslator + CrossLingualTransfer + VariantAdapter
35. ReferenceLocalizer + GlossaryEnforcer

**Infrastructure:**
36. Multi-turn chat generation (if not shipped in v0.1.1)
37. All export formats (Parquet, CSV, JSON, HF Hub)
38. Plugin architecture (register_generator / register_transformer / register_filter)
39. `syntheta inspect` enhanced with quality histograms and embedding visualization (v0.1 ships basic inspect per Section 11.6)

### Phase 3: v0.3 — "Analyze + Fill"

**Analysis:**
40. `syntheta analyze` — gap detection against target spec
41. `syntheta fill` — targeted generation to fill detected gaps
42. Cross-lingual gap detection ("your Arabic split is missing 30% of English topics")
43. Embedding hole detection ("these regions have no coverage")

**Generation (research frontier):**
44. BootstrappedDPO
45. ConstitutionalAI
46. RejectionSampler
47. EntiGraph augmenter

**The closed loop:** analyze → fill → analyze again → done.

### Phase 4: v1.0 — "Platform"

**Audit and compare:**
48. `syntheta compare` — cross-dataset overlap, coverage diff, quality delta
49. `syntheta audit` — full audit: quality, diversity, safety, contamination, bias
50. Deep contamination detection (extends Phase 2's lightweight filter with fuzzy matching, embedding similarity, and user-supplied custom benchmark sets)
51. Public dataset analysis (`syntheta audit HuggingFaceTB/smoltalk`)

**Additional capabilities:**
52. Additional LLM backends (Anthropic, LiteLLM)
53. Multimodal generation (image-text pairs)
54. Agentic data generation (tool-use, multi-step reasoning traces)
55. Curriculum design (suggest training data ordering)

At Phase 4, Syntheta evolves from "synthetic data generator" to "training data engineering platform."

---

## 17. Python API reference (v1 surface)

### 17.1 High-level API

```python
import syntheta

# ─── Generate from scratch ───
dataset = syntheta.generate(
    domain="Arabic culture and traditions",
    n=1000,
    method="topic_tree",
    task_types=["qa", "explain", "compare", "creative"],
    languages=["ar", "en"],
    difficulty_range=(1, 5),
    evolve=True,
    generate_responses=True,
    filter_quality=True,
    safety_context="islamic",
    seed=42,
)

# ─── Rephrase for pretraining (FinePhrase-style) ───
dataset = syntheta.rephrase_pretrain(
    source="fineweb-edu-sample.jsonl",
    strategies=["faq", "math", "table", "tutorial"],
)

# ─── Augment existing data ───
dataset = syntheta.augment(
    source="my_data.jsonl",
    n=5000,
    techniques=["self_instruct", "evol_instruct"],
)

# ─── Translate ───
dataset = syntheta.translate(
    source="english_data.jsonl",
    target_lang="ar",
    strategy="culturally_adapted",
    safety_context="islamic",
)
```

### 17.2 Pipeline API (fine-grained control)

```python
from syntheta.pipeline import Pipeline
from syntheta.generators import TopicTreeGenerator
from syntheta.transformers import EvolInstruct, ResponseGenerator
from syntheta.filters import SafetyFilter, QualityFilter, TemplateCollapseDetector
from syntheta.llms import OpenAICompatibleLLM

llm = OpenAICompatibleLLM(model="gpt-4o-mini")

pipe = Pipeline(
    generator=TopicTreeGenerator(
        domain="Cardiology",
        task_types=["qa", "explain"],
    ),
    transformers=[
        EvolInstruct(rounds=1, strategies=["deepen", "concretize"]),
        ResponseGenerator(use_cot=True),
    ],
    filters=[
        SafetyFilter(cultural_context="academic"),
        QualityFilter(min_score=0.7),
        TemplateCollapseDetector(max_repeat=3),
    ],
    llm=llm,
    seed=42,
)

dataset = pipe.run(n=2000, output="cardiology_sft.jsonl", checkpoint_path="./checkpoints")
# Returns SynthDataset for further analysis; data already written to disk via streaming
```

### 17.3 I/O and utilities API

```python
# ─── Loading ───
ds = syntheta.load("data.jsonl")
ds = syntheta.load("data.csv", column_map={"q": "instruction", "a": "response"})
ds = syntheta.load("org/dataset", source="huggingface", split="train")

# ─── Saving ───
ds.save("out.jsonl", format="sft")
ds.save("out.jsonl", format="dpo", rename={"instruction": "prompt"})

# ─── Health report ───
report = ds.health_report()
print(report)
report.save("report.json")

# ─── Inspect (same output as CLI `syntheta inspect`) ───
info = syntheta.inspect("data.jsonl")
print(info)                          # human-readable summary
info.to_json("info.json")           # machine-readable

# ─── Validate (same rules as CLI `syntheta validate`) ───
result = syntheta.validate("data.jsonl", format="sft", strict=False)
print(result.passed)                 # True / False
print(result.errors)                 # list of {row, field, reason}

# ─── Filtering ───
high_quality = ds.filter(lambda s: s.quality_score and s.quality_score > 0.8)
arabic_only = ds.filter(lambda s: s.language == "ar")
```

### 17.4 Pipeline control (dry-run, resume)

```python
from syntheta.pipeline import Pipeline

pipe = Pipeline(...)

# ─── Dry run (estimate cost without executing) ───
estimate = pipe.dry_run(n=10000)
print(estimate.total_tokens)         # ~28M
print(estimate.estimated_cost)       # ~$4.20
print(estimate.estimated_time)       # ~8 min

# ─── Normal run ───
dataset = pipe.run(n=10000, output="out.jsonl", checkpoint_path="./checkpoints")

# ─── Resume after crash ───
dataset = pipe.run(n=10000, output="out.jsonl", checkpoint_path="./checkpoints", resume=True)
```

---

## 18. Technical requirements

### 18.1 Python version

Python 3.10+.

### 18.2 Core dependencies

- `pydantic>=2.0` — schema validation
- `httpx>=0.27` — async HTTP for LLM calls (or `openai` SDK)
- `tqdm>=4.66` — progress bars
- `pyyaml>=6.0` — config file parsing
- `click>=8.0` — CLI framework

### 18.3 Optional dependencies

- `openai>=1.30` — OpenAI SDK convenience wrapper
- `datasets>=2.18` — HuggingFace dataset integration and Hub push
- `pymupdf` or `pdfplumber` — PDF parsing for DocumentQAGenerator

### 18.4 Install profiles

```bash
pip install syntheta                    # core only
pip install "syntheta[openai]"          # + openai SDK
pip install "syntheta[documents]"       # + PDF/EPUB parsing
pip install "syntheta[hub]"             # + HuggingFace datasets/Hub
pip install "syntheta[all]"             # everything
```

No local model downloads, no PyTorch, no GPU required. Everything runs through the OpenAI-compatible API.

---

## 19. Non-goals (explicitly out of scope)

- **Model training.** Syntheta generates data. It does not train models, compute gradients, or run RL loops.
- **RL environment runtime.** Syntheta does not run GRPO/PPO loops or serve as a reward function during training.
- **Reward model training.** Syntheta can generate preference data to train reward models, but does not train them.
- **Ground-truth verification for RLVR.** v1 supports judge-based RLVR only. Verifiable ground truth deferred to v2+.
- **Web UI / Gradio interface.** CLI + Python API only for v1.
- **Multimodal data.** Text only for v1.
- **Downstream evaluation.** Syntheta does not fine-tune models or run benchmarks. It validates data quality through its health report, filters, and (in Phase 3+) gap analysis — not through training runs.

---

## 20. Success metrics

- **Adoption:** GitHub stars, PyPI downloads, community contributions
- **Quality:** Community-reported results: models trained on Syntheta-generated data match or exceed models trained on comparable manually-curated datasets
- **Coverage:** Support for 10+ languages with cultural adaptation by v0.2
- **Reliability:** <1% pipeline failure rate on standard generation runs
- **Safety:** Zero reports of harmful content slipping through SafetyFilter with cultural presets enabled
- **Developer experience:** A new user should be able to generate their first 100-sample dataset in under 5 minutes with `syntheta generate --recipe sft --domain "..." --n 100`