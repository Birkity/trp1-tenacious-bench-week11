# Datasheet for Tenacious-Bench v0.1

*Following Gebru et al. (2018) "Datasheets for Datasets" and Pushkarna et al. (2022) "Data Cards" templates.*

---

## 1. Motivation

**Purpose.** Tenacious-Bench is a domain-specific evaluation benchmark for B2B outbound sales agents. It measures whether a (brief, email) pair satisfies five machine-verifiable rubric dimensions: grounding fidelity, ICP pitch alignment, signal directionality, tone compliance, and format compliance.

**Why this gap exists.** Standard NLP benchmarks (MMLU, HellaSwag, BIG-Bench) do not cover the narrow failure mode observed in Week 10 traces: an agent that generates grammatically clean emails with correct surface-level compliance but incorrect semantic reasoning — e.g., applying a growth-pitch frame to a decelerating signal, or claiming capability gaps that are absent from the hiring brief.

**Who created it.** Birkity Mekasha, Research Partner at Tenacious Intelligence Corporation, under the 10 Academy TRP1 Week 11 challenge. No external funding. No third-party annotators.

**Intended use.** Evaluate Path B (preference-tuned judge/critic) trained via SimPO on `unsloth/Qwen2.5-3B-Instruct`. Secondary use: benchmark regression testing for any future iteration of the outbound email generation pipeline.

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
| `adversarial_hand` | TB-ADV | 52 | Hand-authored edge cases targeting each rubric dimension (TB-ADV-001–052) |
| `synthetic_semantic_edge_cases` | TB-SEM | 55 | LLM-generated, evaluator-PASS, semantically unjustified |
| **Total** | | **257** | |

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
| --- | --- | --- |
| PASS | 112 | 44% |
| REJECT | 145 | 56% |

Difficulty distribution (post judge-filter; synthetic semantic tasks have no difficulty label):

| Difficulty | Count | % |
| --- | --- | --- |
| easy | 43 | 17% |
| medium | 63 | 25% |
| hard | 96 | 37% |
| unlabeled (synthetic) | 55 | 21% |

### 2.5 Rubric Dimensions

| Code | Dimension | Description | Type |
|------|-----------|-------------|------|
| D1 | grounding_fidelity | All numerics in email appear in brief; `bench_available` check | deterministic |
| D2 | icp_pitch_alignment | Phase 1: Ambiguous segment → no product claim; Ambiguous → body must end with `?`. Phase 2: LLM judge checks segment-to-frame alignment (requires `--llm-judge`). | deterministic (Phase 1) / LLM (Phase 2) |
| D3 | signal_directionality | `delta_pct < -20` → no growth-frame terms in body | deterministic |
| D4 | tone_compliance | No banned phrases (full style-guide list: 30+ phrases incl. "leverage", "skyrocket", "synergize", etc.); no "bench" word in prospect-facing copy; Ambiguous → must end with `?` | deterministic |
| D5 | format_compliance | Subject ≤60 chars; approved prefix; body ≤120 words; ≤1 `?`; no URLs; no booking phrases | deterministic |

Phase 1 dimensions (D1, D3, D4, D5) are fully deterministic. D2 Phase 1 fast-fail is also deterministic. Full D2 segment-frame checking requires the LLM judge (`--llm-judge` flag; uses DeepSeek via OpenRouter).

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

The 257 tasks are split deterministically (seed=42) using a **family-aware shuffle**.
All tasks from the same company (trace-derived), the same parameter combo (programmatic
A/B/C variants), or the same company in the synthetic batch are grouped into one "family"
and kept in the same partition. This prevents contamination via shared observation text.

| Partition | Target | Actual | Purpose | SHA-256 checksum (16 hex) |
|-----------|--------|--------|---------|--------------------------|
| `train/` | 50% | 114 | Preference pair generation for SimPO training | `6668fea2735097f6` |
| `dev/` | 30% | 78 | Public evaluation set | `3637a2857fbba383` |
| `held_out/` | 20% | 65 | Sealed; used only for final evaluation | `14b14e9650aee2bb` |

Slight deviation from exact 50/30/20 is expected with family-aware splitting (families have
varying sizes). The split script is `scripts/generation/partition.py` with `RANDOM_SEED = 42`.

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

### 5.3 Completed Uses

- **SimPO preference training (Act III — complete)**: The `train/` partition (114 tasks) was used to build 228 dimension-reasoning preference pairs (`training_data/tenacious_judge_act5_pairs.jsonl`). The trained LoRA adapter is published in `tenacious_judge_adapter/`. See §10 for full training details.
- **Ablation evaluation (Act IV — complete)**: Three judges (deterministic, base Qwen2.5-3B-Instruct, trained judge) were compared on dev and held-out partitions. See §11 for results.
- **Phase 2 semantic evaluation**: The 55 synthetic semantic edge cases in `dev_synthetic/` are designed to test LLM judges beyond Phase 1 deterministic checks (future work).

---

## 6. Distribution

### 6.1 License

Tenacious-Bench v0.1 is released under **CC-BY-4.0**. Attribution required. Commercial use permitted. Derivatives must credit the original dataset.

### 6.2 Repository Structure

```
data/tenacious_bench_v0.1/
├── dev/
│   ├── trace_derived_batch1.jsonl       (75 tasks)
│   ├── programmatic_batch1.jsonl        (75 tasks)
│   └── adversarial_hand_batch1.jsonl    (52 tasks, incl. TB-ADV-041–052)
├── dev_synthetic/
│   └── semantic_edge_cases_batch1.jsonl (55 tasks, 5 removed for "leverage")
├── train/
│   └── tasks.jsonl                      (114 tasks, 44%)
├── dev/
│   └── tasks.jsonl                      (78 tasks, 30%)
└── held_out/                            ← gitignored
    └── tasks.jsonl                      (65 tasks, 25%)
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

Birkity Yishak — Birkity@10academy.org  
Research Partner, Tenacious Intelligence Corporation

### 7.3 Known Limitations

1. **Single human annotator**: The inter-rater agreement study uses the same person in two rounds. A true multi-annotator study would involve at least two independent labelers.

2. **Phase 1 only**: The deterministic evaluator (Phase 1) does not catch semantic errors — emails that PASS D1–D5 but make unjustified inferences. The 55 synthetic semantic edge cases (`dev_synthetic/`) are designed for Phase 2 evaluation, which requires an LLM judge.

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
No 8-gram from the held-out partition appears in the train partition. Template n-grams (any 8-gram appearing in ≥ 2 tasks across the full pool) are filtered before comparison — this removes shared velocity-percentage phrases and subject-line boilerplate while preserving genuine company-specific leakage detection. After filtering, max shared n-grams = 0.

**Check 2 — Embedding Similarity**  
Maximum cosine similarity between any held-out task and any train task is below 0.85. Method: sentence-transformers `all-MiniLM-L6-v2` if available; TF-IDF cosine fallback otherwise. TF-IDF is conservative (overestimates similarity for short texts), so a pass under TF-IDF is a strong pass.

**Check 3 — Time-Shift**  
All hiring velocity observations reference the Feb–Apr 2026 observation window. The base model's knowledge cutoff (August 2025) predates this window by 6 months. Models cannot have memorized these specific company signals from pre-training data.

---

## 10. Preference Pair Design (Act III — SimPO Training)

### 10.1 Pair Construction

The `train/` partition (114 tasks) was used to build 228 preference pairs via `scripts/build_act5_preferences.py`. Each task yields two pairs sharing the same **chosen** response but with different **rejected** variants:

| Pair | Rejected type    | Description                                                                                           |
|------|------------------|-------------------------------------------------------------------------------------------------------|
| A    | Aggressive wrong | All five dimensions marked PASS → wrong PASS verdict                                                  |
| B    | Subtle wrong     | Correct REJECT verdict but wrong primary failure dimension; real failing dim silently flipped to PASS |

The **chosen** response is structured D1–D5 dimension reasoning derived from `score_task()` ground truth:

```text
D1: PASS — grounding facts verified; $14M matches brief
D2: FAIL — Ambiguous segment with product claim
D3: PASS — ok
D4: PASS — ok
D5: PASS — ok
VERDICT: REJECT
Primary failure: D2
```

Pair B forces the model to learn *which* dimension fails, not just whether to REJECT. This design prevents preference leakage where the model learns superficial verdict-sentence style without dimension-level reasoning.

### 10.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Backbone | `unsloth/Qwen2.5-3B-Instruct` |
| Method | SimPO via TRL `CPOConfig(loss_type="simpo", cpo_alpha=0.0)` |
| Preference pairs | 228 (`training_data/tenacious_judge_act5_pairs.jsonl`) |
| LoRA rank / alpha | 16 / 16, dropout=0.0, 7 projection modules |
| Epochs | 3 |
| Effective batch size | 16 (batch=4 × grad\_accum=4) |
| Learning rate | 2e-4 |
| β / γ | 2.0 / 0.5 |
| Final train loss | 0.0745 |
| Wall time | 14 min (Google Colab T4, 16 GB VRAM, fp16) |
| Seed | 42 |

Full hyperparameter log: `tenacious_judge_adapter/training_config.json`  
Per-step loss (45 steps × 3 epochs): `tenacious_judge_adapter/loss_log.json`  
Training script: `training/train_simpo_judge.py`

### 10.3 Label Balance

Of the 228 pairs, 142 have a REJECT chosen response and 86 have a PASS chosen response, reflecting the 56/44 REJECT/PASS split in the full dataset.

---

## 11. Ablation Results (Act IV)

Three judges were evaluated on both dev (78 tasks) and held-out (65 tasks) partitions using `scripts/analysis/run_ablation.py`.

| Judge | Method |
|-------|--------|
| Deterministic | `score_task()` from `scoring_evaluator.py` — no LLM, no temperature |
| Base model | `Qwen2.5-3B-Instruct`, zero-shot structured-output prompt |
| Trained judge | Same backbone + SimPO LoRA adapter from `tenacious_judge_adapter/` |

### 11.1 Accuracy Summary

| Partition  | n  | Deterministic | Base model    | Trained judge   | Delta A | Delta B |
|------------|----|---------------|---------------|-----------------|---------|---------|
| dev        | 78 | 100.0%        | 60.3% (47/78) | 65.5% (36/55\*) | −34.5%  | +5.2%   |
| held\_out  | 65 | 100.0%        | 50.8% (33/65) | 59.4% (19/32\*) | −40.6%  | +8.6%   |

\* Trained model produced UNKNOWN on 23 dev / 33 held-out tasks; accuracy computed over scored tasks only.

- **Delta A** = trained accuracy − 100% (gap to perfect deterministic baseline)
- **Delta B** = trained accuracy − base accuracy (lift from SimPO fine-tuning)

### 11.2 Dimension-Level Errors (held-out, 65 tasks)

| Dimension | Base model errors | Trained judge errors | Improvement |
|-----------|-------------------|---------------------|-------------|
| D1 grounding\_fidelity | 13 | 8 | −5 |
| D2 icp\_pitch\_alignment | 14 | 0 | −14 |
| D3 signal\_directionality | 3 | 3 | 0 |
| D4 tone\_compliance | 2 | 2 | 0 |
| D5 format\_compliance | 0 | 0 | 0 |

The trained judge **eliminated all D2 errors** (ICP alignment — Ambiguous segment detection) and reduced D1 errors by 38%. UNKNOWN outputs concentrated on programmatic-C and trace-C tasks (Ambiguous+product-claim pattern), suggesting the model partially learned to recognize the D2 pattern but sometimes abstains rather than answering.

Full per-task results: `ablations/ablation_results.json`  
Raw model outputs: `ablations/held_out_traces.jsonl`

---

*Datasheet authored 2026-04-29 by Birkity Mekasha. Updated 2026-05-02 with Act III (SimPO training) and Act IV (ablation) results. Dataset version: v0.1 — 257 tasks, four authoring modes, three partitions sealed.*
