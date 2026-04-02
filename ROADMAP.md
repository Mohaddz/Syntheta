# Syntheta Roadmap

## Phase 1: v0.1 — "Generate" (MVP)

### Core Pipeline — DONE

- [x] TopicTreeGenerator (TreeSynth/Seed-Free SDG)
- [x] PersonaGenerator (1B Personas / MATRIX-Gen)
- [x] SeedDatasetGenerator (Self-Instruct)
- [x] EvolInstruct transformer (WizardLM — 6 strategies, multi-round)
- [x] ResponseGenerator transformer (optional CoT)
- [x] PretrainRewriter transformer (FinePhrase — faq, math, table, tutorial)
- [x] SafetyFilter (blocklist + LLM cultural check, 3 presets)
- [x] QualityFilter (LLM-as-judge multi-dimensional scoring)
- [x] TemplateCollapseDetector (prefix-based dedup)
- [x] Streaming pipeline with sample-level concurrency
- [x] Adaptive rate limiter with exponential backoff
- [x] Checkpoint every 100 samples with resume support
- [x] JSONL writer with append mode
- [x] Unified Sample schema (SFT, chat, DPO, RLVR, pretrain)
- [x] Exporters (SFT, chat, DPO, RLVR, pretrain) with validation
- [x] Column mapper with auto-detect and fuzzy synonym mapping
- [x] Config layering (defaults → recipe → YAML → CLI dot-notation)
- [x] Recipes: sft, sft_persona, augment, pretrain
- [x] Rich live progress display with stats bar
- [x] Cost tracking with model_prices.json (300+ models)
- [x] Filter summary, health report, cost report
- [x] Exception hierarchy (config, LLM, pipeline errors)
- [x] CLI: generate, inspect, validate, convert, recipes

### Algorithm Audit — P0

Audit every generator, transformer, and filter against the source papers. The current implementations were built to match the paper abstracts and high-level descriptions. A line-by-line comparison with the actual methodology sections is needed to ensure correctness and completeness.

- [x] **TopicTreeGenerator** — compare against TreeSynth (2025) and Seed-Free SDG (2024). Verify: tree construction prompt design, diversity of topic sampling strategy, plan distribution balancing, instruction generation prompt faithfulness.
- [ ] **PersonaGenerator** — compare against "Scaling Synthetic Data with 1B Personas" (2024) and MATRIX-Gen (ACL 2025). Verify: persona attribute diversity (demographics, expertise, perspective), question generation grounding in persona context, deduplication across personas.
- [ ] **SeedDatasetGenerator** — compare against Self-Instruct (Wang et al., 2023). Verify: few-shot pool construction, similarity filtering against seed pool (currently missing?), iterative pool expansion, classification vs generation task split.
- [ ] **EvolInstruct** — compare against WizardLM (Xu et al., 2023) and WizardLM2. Verify: all 6 strategy prompts match paper descriptions, elimination criteria (too short, copy-paste, "sorry" responses), multi-round evolution chain integrity.
- [ ] **ResponseGenerator** — verify CoT prompting follows best practices. Check: does it strip thinking tokens from final response? Does it handle refusals?
- [ ] **PretrainRewriter** — compare against FinePhrase (2024). Verify: all 4 strategy prompts match paper methodology, output format correctness, quality of rewrites at scale.
- [ ] **QualityFilter** — compare against LLM-as-judge literature (MT-Bench, AlpacaEval). Verify: scoring dimensions, prompt design, score extraction robustness, threshold calibration.
- [ ] **SafetyFilter** — verify blocklist coverage, LLM safety check prompt design, cultural preset completeness. Test with adversarial inputs.
- [ ] **TemplateCollapseDetector** — verify prefix detection logic handles edge cases (short samples, multilingual text, code blocks).

### Remaining MVP Items — P1

- [ ] **Prompt logging** — wire `log_prompts` config flag through to the LLM backend. Every call logs to `prompts.jsonl` with: timestamp, stage, model, prompt, response, tokens, latency.
- [ ] **run_summary.json** — auto-save after every pipeline run: input/output counts, per-filter rejections, retry/rate-limit stats, cost breakdown, elapsed time.
- [ ] **Auto-save health report** — call `DatasetHealthReport.save()` post-run, writing `{output_path}.health.json`.
- [ ] **Dry-run dollar cost** — wire CostTracker pricing lookup into `_dry_run()` so it outputs estimated $ cost alongside token counts.
- [ ] **`--strict` validate** — implement strict checks: no duplicate instructions, minimum response length, quality score distribution, field completeness.
- [ ] **High-level Python API** — expose `syntheta.generate()`, `syntheta.augment()`, `syntheta.rephrase_pretrain()` in `__init__.py` as thin wrappers over Pipeline.
- [ ] **`syntheta augment` CLI command** — dedicated command that loads an existing JSONL and runs it through transformers + filters.

---

## Phase 2: v0.2 — "Expand + Translate"

### New Generators — P1

- [ ] **MagpieGenerator** — feed aligned LLM only the pre-query chat template to extract instructions. Paper: Magpie (2024).
- [ ] **InstructionBacktranslation** — reverse-engineer instructions from high-quality responses. Paper: Li et al. (2023).
- [ ] **Genstruct** — generate instruction-response pairs grounded in a source text passage. Paper: Genstruct (2024).
- [ ] **DocumentQAGenerator** — parse PDFs/EPUBs, chunk, generate diverse QA pairs per chunk. Requires `pymupdf` or `pdfplumber` optional dependency.

### New Transformers — P1

- [ ] **PreferencePairGenerator** — generate K candidate responses, score with judge LLM, pick best/worst for DPO pairs. Paper: UltraFeedback (2024).
- [ ] **ArenaGenerator** — multiple LLMs answer the same prompt, pairs ranked by judge. Paper: Chatbot Arena / LMSYS.
- [ ] **RLVRInfoBuilder** — enrich samples with ground-truth `info` dict for RLVR judge environments.

### Multi-turn Chat Generation — P1

- [ ] **Simulated conversation** — two LLM personas role-play a multi-turn dialogue.
- [ ] **Single-turn expansion** — take a single-turn sample and add follow-ups, clarifications, topic pivots.
- [ ] **Tree-branching** — branch a first turn into multiple continuations for diversity.

### New Filters — P2

- [ ] **DiversityFilter** — embedding-based deduplication + MMR (maximal marginal relevance) selection. Requires embedding model support.
- [ ] **BackTranslationFilter** — translate to target language and back, compare semantic similarity to detect translation drift.
- [ ] **NaturalnessJudge** — LLM judges whether text sounds native vs. machine-generated.
- [ ] **ContaminationFilter** — 8-gram overlap check against benchmark datasets to prevent data contamination.

### Translation Module — P2

- [ ] Source quality filter
- [ ] Safety filter with cultural context
- [ ] Field policy assignment (translate vs. preserve vs. localize)
- [ ] Chunked thematic batching
- [ ] Structure-aware translation (preserve markdown, code, formatting)
- [ ] Refinement pass
- [ ] Glossary enforcement
- [ ] Reference localization
- [ ] Back-translation consistency check
- [ ] Naturalness + difficulty scoring
- [ ] Format preservation check
- [ ] Three strategies: culturally-adapted, cross-lingual structure transfer, variant adaptation

### Infrastructure — P2

- [ ] `syntheta translate` CLI command
- [ ] HuggingFace Hub push (`syntheta push`)
- [ ] CSV and Parquet export formats
- [ ] Plugin architecture for custom generators/transformers/filters
- [ ] Config versioning with deprecation warnings

---

## Phase 3: v0.3 — "Analyze + Fill"

### Gap Detection — P1

- [ ] `syntheta analyze` — detect coverage gaps across topics, task types, difficulty, languages.
- [ ] Cross-lingual gap detection — identify under-represented language/domain combinations.
- [ ] Embedding hole detection — find sparse regions in the embedding space.

### Targeted Generation — P1

- [ ] `syntheta fill` — auto-generate samples to fill detected gaps with appropriate generators.

### Advanced Generation Techniques — P2

- [ ] BootstrappedDPO — iterative DPO pair generation with self-improving judge.
- [ ] ConstitutionalAI — critique-and-revise loop for alignment data.
- [ ] RejectionSampler — generate many, keep only top-scored.
- [ ] EntiGraph — entity-centric knowledge graph for structured data generation.

---

## Phase 4: v1.0 — "Platform"

- [ ] `syntheta compare` — cross-dataset analysis and diff.
- [ ] `syntheta audit` — comprehensive quality, diversity, safety, contamination, and bias audit.
- [ ] Deep contamination detection (beyond 8-gram).
- [ ] Additional LLM backend adapters (Bedrock, Vertex, etc.).
- [ ] Multimodal data generation (image-text pairs).
- [ ] Agentic data generation (tool-use, multi-step reasoning traces).
- [ ] Curriculum design — ordered training data for progressive difficulty.

---

## Priority Legend

| Tag | Meaning |
|-----|---------|
| P0 | Must do before any new features — correctness of existing code |
| P1 | Core value — directly enables new use cases |
| P2 | Important but not blocking — quality of life, completeness |
