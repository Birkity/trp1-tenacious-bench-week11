# Datasheet for Tenacious-Bench v0.1

*Following Gebru et al. (2018) "Datasheets for Datasets" and Pushkarna et al. (2022) "Data Cards" templates.*

---

## 1. Motivation

**Purpose.** Tenacious-Bench is a domain-specific evaluation benchmark for B2B outbound sales agents. It measures whether a (brief, email) pair satisfies five machine-verifiable rubric dimensions: grounding fidelity, ICP pitch alignment, signal directionality, tone compliance, and format compliance.

**Why this gap exists.** Standard NLP benchmarks (MMLU, HellaSwag, BIG-Bench) do not cover the narrow failure mode observed in Week 10 traces: an agent that generates grammatically clean emails with correct surface-level compliance but incorrect semantic reasoning — e.g., applying a growth-pitch frame to a decelerating signal, or claiming capability gaps that are absent from the hiring brief.

**Who created it.** Birkity Mekasha, Research Partner at Tenacious Intelligence Corporation, under the 10 Academy TRP1 Week 11 challenge. No external funding. No third-party annotators.

**Intended use.** Evaluate Path B (preference-tuned judge/critic) trained via SimPO on Qwen 3.5 0.8B. Secondary use: benchmark regression testing for any future iteration of the outbound email generation pipeline.

---

## 2. Composition

### 2.1 Task Structure

Each task is a JSON object with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Unique ID with prefix `TB-TRACE`, `TB-PROG`, `TB-ADV`, or `TB-SEM` |
| `brief` | object | Company signal bundle: company name, ICP segment, hiring velocity, grounding facts, AI maturity, bench availability, capability gaps |
| `email` | object | Subject line, body, word count, tone warnings |
| `prior_thread` | string | Preceding conversation context (empty for cold outbound) |
| `bench_summary` | string | Current available engineer roster (fixed across all tasks) |
| `rubric` | object | Dimension list and pass thresholds |
| `difficulty` | string | `easy` / `medium` / `hard` (assigned by LLM judge filter) |
| `judge_filter` | object | LLM judge scores: IC, GTV, RAC (1–5 each) |
| `brief.source_mode` | string | Authoring mode (see §2.2) |

### 2.2 Authoring Modes and Task Counts

| Source Mode | Prefix | Count | Description |
|-------------|--------|-------|-------------|
| `trace_derived` | TB-TRACE | 75 | Derived from 25 Week 10 company events × 3 variants (A/B/C) |
| `programmatic` | TB-PROG | 75 | Deterministic templates, 25 parameter combos × 3 variants |
| `adversarial_hand` | TB-ADV | 40 | Hand-authored edge cases targeting each rubric dimension |
| `synthetic_semantic_edge_cases` | TB-SEM | 60 | LLM-generated, evaluator-PASS, semantically unjustified |
| **Total** | | **250** | |

### 2.3 Task Variants

Each authoring mode (except adversarial) produces three task variants:

| Variant | Suffix | Verdict | Failing Dimension |
|---------|--------|---------|-------------------|
| A | `A` | PASS | — |
| B | `B` | REJECT | D1 grounding_fidelity (corrupted numeric) |
| C | `C` | REJECT | D2 icp_pitch_alignment (Ambiguous segment + product claim) |

Adversarial tasks (`TB-ADV`) cover additional failure modes: D3 (signal directionality), D4 (tone compliance), D5 (format compliance), and multi-dimension failures.

### 2.4 Label Distribution

| Overall Verdict | Count | % |
|----------------|-------|---|
| PASS | ~110 | ~44% |
| REJECT | ~140 | ~56% |

Difficulty distribution (approximate, post judge-filter):

| Difficulty | Count | % |
|-----------|-------|---|
| easy | ~55 | ~22% |
| medium | ~90 | ~36% |
| hard | ~105 | ~42% |

### 2.5 Rubric Dimensions

| Code | Dimension | Description | Type |
|------|-----------|-------------|------|
| D1 | grounding_fidelity | All numerics in email appear in brief; `bench_available` check | deterministic |
| D2 | icp_pitch_alignment | Segment="Ambiguous" → no product claim permitted (Phase 1) | deterministic |
| D3 | signal_directionality | `delta_pct < -20` → no growth-frame terms in body | deterministic |
| D4 | tone_compliance | No banned phrases (hyperbole, urgency, meeting-push) | deterministic |
| D5 | format_compliance | Subject ≤60 chars, body ≤120 words, ≤1 `?`, no URLs, no booking links | deterministic |

All five dimensions are computed deterministically by `benchmark/scoring_evaluator.py`. No LLM inference is required at evaluation time.

---

## 3. Collection Process

### 3.1 Source Data

**Trace-derived tasks** are grounded in Week 10 company signal traces:
- 9 companies with hiring velocity signals observed Feb–Apr 2026
- 25 distinct company events selected for clear rubric coverage
- Each event produces one PASS email (A), one D1-corrupt email (B), one D2-isolate email (C)

**Programmatic tasks** use no external data — all briefs are constructed from a fixed parameter table (25 combinations of ICP segment × velocity bucket × AI maturity × bench availability).

**Adversarial hand-authored tasks** were written directly by the primary author with deliberate rubric violations, targeting one or more specific dimensions per task.

**Synthetic semantic edge cases** are LLM-generated using DeepSeek V3.2 via OpenRouter. The model is prompted to produce emails that PASS all D1–D5 programmatic checks but are semantically unjustified given the brief. Generation and filtering use separate model calls (see §3.3).

### 3.2 Observation Window

All hiring velocity signals reference the Feb–Apr 2026 observation window. The base model's knowledge cutoff (August 2025) predates this window by 6+ months. This prevents any evaluation model from relying on pre-training memorization of these specific company signals.

### 3.3 LLM Rotation Policy

To prevent preference leakage between generation and quality filtering:

| Stage | Model | Role |
|-------|-------|------|
| Trace-derived task drafting | Google Gemini (free tier) | Generation |
| Synthetic semantic edge case generation | DeepSeek V3.2 (OpenRouter) | Generation |
| Quality filtering (all batches) | DeepSeek V3.2 (OpenRouter) | Judge |
| Deterministic scoring | `scoring_evaluator.py` (no LLM) | Evaluation |

The generation model and judge model belong to different model families for trace-derived tasks (Gemini ≠ DeepSeek). For synthetic tasks, the same model family is used for generation and filtering — this is acceptable because the judge evaluates *email quality* (IC/GTV/RAC dimensions), not semantic correctness, and the evaluation-time model is the trained judge being benchmarked.

### 3.4 LLM-as-a-Judge Filter

All LLM-generated and trace-derived tasks are passed through a quality gate before inclusion. The judge (DeepSeek V3.2) scores three dimensions:

| Dimension | Code | Description | Threshold |
|-----------|------|-------------|-----------|
| Instruction Compliance | IC | Does the email follow brief constraints? | ≥ 3/5 |
| Grounded Truth Value | GTV | Are claims grounded in the brief? | ≥ 3/5 |
| Rubric Alignment Confidence | RAC | How confident is the judge in the rubric call? | ≥ 3/5 |

Tasks scoring below threshold on any dimension are discarded. Difficulty labels are assigned from the judge scores: `easy` (all ≥4), `medium` (any 3), `hard` (any ≤2 after filter pass).

### 3.5 Dataset Partitioning

The 250 tasks are split deterministically (seed=42):

| Partition | Ratio | Count | Purpose |
|-----------|-------|-------|---------|
| `train/` | 50% | 125 | Preference pair generation for SimPO training |
| `dev/` | 30% | 75 | Public evaluation set |
| `held_out/` | 20% | 50 | Sealed; used only for final evaluation |

The shuffle uses `random.seed(42)` applied after sorting by `task_id`. Partition checksums (SHA-256 of sorted task IDs) are recorded for reproducibility.

---

## 4. Preprocessing and Labeling

### 4.1 Deterministic Labels

All D1–D5 labels and overall verdicts are computed by `benchmark/scoring_evaluator.py`. The evaluator is deterministic — no model inference, no temperature. Given the same task JSON, it returns the same scores. The source code and full rubric are published in `benchmark/dimensions.md`.

### 4.2 Human Calibration

A 30-task subset was hand-labeled by the primary author (Birkity) twice with a 24-hour gap, following the inter-rater agreement protocol documented in `docs/inter_rater_agreement.md`.

| Metric | Result | Threshold |
|--------|--------|-----------|
| Min per-dimension agreement | 97% | ≥ 80% ✓ |
| Min Cohen's κ | 0.92 | ≥ 0.80 ✓ |
| Human-evaluator agreement | 93% | — |

The two human-evaluator disagreements were both human attention errors, not evaluator bugs.

### 4.3 No Crowd-Sourced Annotation

This benchmark does not use Mechanical Turk, Scale AI, or any other third-party annotation service. All labels derive from the deterministic evaluator, cross-checked by a single human annotator.

---

## 5. Uses

### 5.1 Intended Uses

- **Primary**: Benchmark evaluation of `(brief, email)` scoring models, specifically the Path B preference-tuned judge trained on this dataset's `train/` partition.
- **Secondary**: Regression testing for Tenacious Intelligence Corporation's outbound email generation pipeline.
- **Research**: Study of B2B sales agent alignment failures, specifically grounding fidelity and ICP pitch calibration.

### 5.2 Out-of-Scope Uses

- **Production sales automation**: Tasks are constructed for evaluation purposes with deliberately wrong emails. Using this dataset to train a production email generator would produce incorrect emails.
- **General NLP benchmarks**: Tenacious-Bench is narrow-domain (B2B SaaS outbound). It does not measure general language understanding.
- **Held-out leakage**: The `held_out/` partition must not be used for training or hyperparameter search. It is gitignored and sealed as of 2026-04-30.

### 5.3 Future Uses

- **SimPO preference training (Act III)**: The `train/` partition is the source for preference pair generation. Chosen = correct REJECT with rubric reasoning; Rejected = surface PASS with no semantic grounding.
- **Phase 2 semantic evaluation**: The 60 synthetic semantic edge cases in `dev_synthetic/` are designed to test LLM judges beyond Phase 1 deterministic checks.

---

## 6. Distribution

### 6.1 License

Tenacious-Bench v0.1 is released under **CC-BY-4.0**. Attribution required. Commercial use permitted. Derivatives must credit the original dataset.

### 6.2 Repository Structure

```
data/tenacious_bench_v0.1/
├── dev/
│   ├── trace_derived_batch1.jsonl   (75 tasks)
│   ├── programmatic_batch1.jsonl    (75 tasks)
│   └── adversarial_hand_batch1.jsonl (40 tasks)
├── dev_synthetic/
│   └── semantic_edge_cases_batch1.jsonl (60 tasks)
├── train/
│   └── tasks.jsonl                  (125 tasks, 50%)
├── dev/
│   └── tasks.jsonl                  (75 tasks, 30%)
└── held_out/                        ← gitignored
    └── tasks.jsonl                  (50 tasks, 20%)
```

### 6.3 Held-Out Partition

The `held_out/` directory is listed in `.gitignore`. It is not published in the repository. It is sealed as of 2026-04-30 and may only be used for final evaluation. Checksums are recorded in the partition summary output from `scripts/generation/partition.py`.

### 6.4 Contamination Verification

Three automated contamination checks are recorded in `data/contamination_check.json`:

| Check | Threshold | Result |
|-------|-----------|--------|
| N-gram overlap (n=8) | 0 shared 8-grams | PASS |
| Embedding cosine similarity | < 0.85 | PASS |
| Time-shift (observation window) | ≥ 6 months ahead of model cutoff | PASS |

---

## 7. Maintenance

### 7.1 Versioning

This is v0.1. Future versions will be tagged with semantic versioning. Changes to the rubric (dimensions.md) or scoring evaluator that affect labels will increment the major version number.

### 7.2 Contact

Birkity Mekasha — Birkity@10academy.org  
Research Partner, Tenacious Intelligence Corporation

### 7.3 Known Limitations

1. **Single human annotator**: The inter-rater agreement study uses the same person in two rounds. A true multi-annotator study would involve at least two independent labelers.

2. **Phase 1 only**: The deterministic evaluator (Phase 1) does not catch semantic errors — emails that PASS D1–D5 but make unjustified inferences. The 60 synthetic semantic edge cases (`dev_synthetic/`) are designed for Phase 2 evaluation, which requires an LLM judge.

3. **Narrow domain**: The benchmark covers only B2B outbound email for a specific staffing-augmentation ICP. Performance on this benchmark does not generalize to other sales domains.

4. **Observation window**: All signals reference Feb–Apr 2026. Tasks referencing real company data may become stale as companies change.

---

## 8. Human Calibration Summary

Full protocol: `docs/inter_rater_agreement.md`

| Item | Value |
|------|-------|
| Labeler | Birkity (primary author) |
| Date Round 1 | 2026-04-29 |
| Date Round 2 | 2026-04-30 |
| Tasks sampled | 30 (stratified: 10 easy, 10 medium, 10 hard) |
| Min dimension agreement | 97% |
| Min Cohen's κ | 0.92 |
| Human-evaluator agreement | 93% |
| Rubric revisions required | 1 clarification (D3 note) |
| Held-out sealed | 2026-04-30 |

---

## 9. Contamination Controls Summary

Full results: `data/contamination_check.json`

**Check 1 — N-gram Overlap (n=8)**  
No 8-gram from the held-out partition appears in the train partition. Identical company names and common phrases are filtered by the 8-gram window requirement — no pair of train/held-out tasks shares an 8-token sequence.

**Check 2 — Embedding Similarity**  
Maximum cosine similarity between any held-out task and any train task is below 0.85. Method: sentence-transformers `all-MiniLM-L6-v2` if available; TF-IDF cosine fallback otherwise. TF-IDF is conservative (overestimates similarity for short texts), so a pass under TF-IDF is a strong pass.

**Check 3 — Time-Shift**  
All hiring velocity observations reference the Feb–Apr 2026 observation window. The base model's knowledge cutoff (August 2025) predates this window by 6 months. Models cannot have memorized these specific company signals from pre-training data.

---

*Datasheet authored 2026-04-29 by Birkity Mekasha. Updated 2026-04-30 after held-out sealing.*
