# Week 11 Architecture — Tenacious-Bench + Path B Judge

## Overview

Six layers. Each layer feeds the next. Phase 0 builds the Evidence and Benchmark layers.
Acts I–II build the Dataset layer. Acts III–IV build the Judge and Evaluation layers.
Act V is the Publication layer.

```
EVIDENCE LAYER
──────────────────────────────────────────────────────────────────────
  traces/{slug}/hiring_signal_brief.json    (9 companies)
  artifacts/{slug}/email_log.jsonl          (5 companies have emails)
  artifacts/{slug}/conversation_state.json  (5 companies have turns)
  docs/probe_library.md                     (32 probes, 10 categories)
  docs/failure_taxonomy.md                  (75% → 97% pass rate delta)
        │
        ▼
BENCHMARK LAYER
──────────────────────────────────────────────────────────────────────
  week11/schema.json                  Task input/output schema (v0.1)
  week11/scoring_evaluator.py         D1–D5 programmatic + LLM wrapper
  week11/dimensions.md                Rubrics, lookup tables, thresholds
        │
        ▼
DATASET LAYER  (Acts I–II, Days 1–3)
──────────────────────────────────────────────────────────────────────
  Target: 250 tasks total

  Generation pipeline:
    generation_scripts/trace_derived.py           75 tasks (30%)
    generation_scripts/programmatic_templates.py  75 tasks (30%)
    generation_scripts/synthesis_router.py        60 tasks (25%)
    generation_scripts/adversarial_hand.py        40 tasks (15%)

  Partition (sealed after contamination check):
    tenacious_bench_v0.1/train/     50%  — 125 tasks → preference pairs
    tenacious_bench_v0.1/dev/       30%  — 75 tasks,  public
    tenacious_bench_v0.1/held_out/  20%  — 50 tasks,  gitignored, sealed
        │
        ▼
JUDGE LAYER  (Act III, Day 4)
──────────────────────────────────────────────────────────────────────
  training_data/simpo_pairs.jsonl
    Format: {prompt, chosen, rejected}
    prompt   = system_prompt + "\n\n" + brief_json + email_json
    chosen   = correct REJECT/PASS with dimension reasoning
    rejected = surface-only judgment (passes D4/D5, misses D1/D2)

  Training stack:
    Backbone  : Qwen 3.5 0.8B  (pinned version)
    Method    : SimPO via Unsloth
    Adapter   : LoRA rank=16, alpha=32, target_modules=["q_proj","v_proj"]
    Hardware  : Colab T4 free tier  (16-bit LoRA, ~30–90 min wall time)
    Publish   : LoRA adapter only → HuggingFace
        │
        ▼
EVALUATION LAYER  (Act IV, Days 5–6)
──────────────────────────────────────────────────────────────────────
  ablations/ablation_results.json
  ablations/held_out_traces.jsonl

  Delta A : trained judge vs. regex-only baseline on held-out (primary metric)
  Delta B : trained judge vs. prompt-engineered judge, same backbone, no training
            (tests whether training beat a careful prompt alone)
  Pareto  : per-task cost + latency with/without trained component

  Eval model: DeepSeek V3.2 via OpenRouter
  Passes: ≤4 total on sealed slice
        │
        ▼
PUBLICATION LAYER  (Act V, Day 7)
──────────────────────────────────────────────────────────────────────
  HuggingFace dataset : birkity/tenacious-bench-v0.1
    - dev/ + train/ public at launch
    - held_out/ released after leaderboard

  HuggingFace model   : birkity/tenacious-judge-qwen3.5-0.8b-lora
    - LoRA adapter only (backbone not merged)
    - Full model card

  Blog post           : 1200–2000 words on HuggingFace community blog
  Community artifact  : GitHub issue on τ²-Bench repo with gap findings
  CEO/CFO memo        : 2-page PDF  (memo.pdf)
```

---

## Preference Pair Construction — Data Flow

```
  hiring_signal_brief.json
        +
  generated_email (from email_log.jsonl or generated)
        │
        ▼
  scoring_evaluator.py  →  dimension scores D1–D5  →  verdict
        │
        ├─ FAIL on D1 or D2 (semantic failure) ──────────────────┐
        │                                                          │
        │   chosen  = "VERDICT: REJECT                            │
        │              Primary: D2_signal_directionality          │
        │              Reason: velocity -60%, growth-frame pitch"  │
        │                                                          │
        │   rejected = "VERDICT: PASS                             │
        │               D4 clean. D5 clean. Recommend: send."     │
        │                                                         ─┤
        └─ PASS all (correct email) ─────────────────────────────┐│
                                                                  ││
            chosen  = "VERDICT: PASS                             ││
                        D1 Seg 1 + scaling frame aligned.        ││
                        D2 +100% velocity, growth pitch OK.      ││
                        D3 all numbers verified. Recommend: send."││
                                                                  ││
            rejected = "VERDICT: REJECT                          ││
                         Grounding sparse, cannot verify claims." ││
                         [false positive — this is the rejected   ││
                          response for a passing email]           ││
        ──────────────────────────────────────────────────────────┘┘
                        ▼
        training_data/simpo_pairs.jsonl
```

---

## Cost Envelope

| Bucket | Model | Days active | Budget |
|---|---|---|---|
| Dataset synthesis + filter | DeepSeek V3.2 | 2–3 | $3–5 |
| Training | Qwen 3.5 0.8B, Colab T4 | 5 | $0 (free) |
| Held-out eval | DeepSeek V3.2 | 5–6 | $2–3 |
| Reserve | — | any | $1–2 |
| **Total** | | | **≤ $10** |

---

## Hard Constraints

1. No Claude Sonnet 4.6 or GPT-class models at any stage.
2. No τ²-Bench retail re-runs (Week 10 score reused as Delta C reference only).
3. Eval-tier (DeepSeek V3.2) used only on held-out slice, Days 5–6. Not during authoring.
4. LoRA adapter only published — backbone not merged, not uploaded.
5. `held_out/` never committed to public repo in unencrypted form before leaderboard launch.

---

## File Map

```
week11/
├── methodology.md              Path B justification + model/method choices
├── dimensions.md               D1–D5 specification with rubrics and code
├── week11_architecture.md      This file
├── cost_log.md                 Running cost tracker (every charge logged)
├── schema.json                 JSON schema for task records
├── scoring_evaluator.py        D2–D5 programmatic + D1 LLM wrapper
├── audit_memo.md               (Act I) τ²-Bench gap analysis, ≤600 words
├── datasheet.md                (Act II) Gebru + Pushkarna documentation
├── inter_rater_agreement.md    (Act II) 30-task hand-label agreement matrix
├── tenacious_bench_v0.1/
│   ├── train/                  125 preference pairs (SimPO format)
│   ├── dev/                    75 scored tasks (public)
│   └── held_out/               50 tasks (gitignored, sealed)
├── generation_scripts/
│   ├── trace_derived.py        Converts 9 briefs × variants into tasks
│   ├── programmatic_templates.py  Parameter sweep generator
│   ├── synthesis_router.py     DeepSeek V3.2 synthesis + judge filter
│   └── adversarial_hand.py     Probe library → adversarial tasks
├── training_data/
│   └── simpo_pairs.jsonl       SimPO training pairs (Act III output)
├── ablations/
│   ├── ablation_results.json   Delta A / Delta B / Pareto table
│   └── held_out_traces.jsonl   Raw scoring traces from held-out eval
└── synthesis_memos/
    ├── synthetic_data_liu2024.md     (required before Day 2)
    ├── datasheets_gebru2021.md
    ├── contamination_chen2025.md
    ├── llm_judge_gu2024.md
    ├── dpo_rafailov2023.md           (Path B required)
    ├── simpo_meng2024.md             (Path B — chosen method)
    └── prometheus2_kim2024.md        (Path B required)
```
