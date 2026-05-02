# Cost Log — Week 11

All API and compute charges logged here. Every entry requires: date, time (UTC),
provider, model, bucket, task description, units, and cost in USD.

> **Note on timestamps**: Exact UTC call times were not captured at execution time.
> Dates are accurate (derived from inter_rater_agreement.md protocol log).
> Token counts are derived from prompt structure × task count (see methodology below).

---

## Entries

| Date | UTC | Provider | Model | Bucket | Task | Input tok | Output tok | Cost ($) |
|------|-----|----------|-------|--------|------|-----------|------------|----------|
| 2026-04-29 | — | — | — | — | Phase 0: folder setup, no model calls | 0 | 0 | 0.00 |
| 2026-04-29 | ~ | OpenRouter | google/gemini-2.5-flash | dataset | trace_derived email drafting — 75 tasks × ~1,200 in / ~350 out tok | 90,000 | 26,250 | 0.03 |
| 2026-04-29 | ~ | OpenRouter | openai/gpt-4o-mini | dataset | judge_filter: trace_derived_batch1 — 75 tasks × ~1,200 in / ~100 out tok | 90,000 | 7,500 | 0.02 |
| 2026-04-29 | ~ | OpenRouter | openai/gpt-4o-mini | dataset | judge_filter: programmatic_batch1 — 75 tasks × ~1,200 in / ~100 out tok | 90,000 | 7,500 | 0.02 |
| 2026-04-29 | ~ | OpenRouter | openai/gpt-4o-mini | dataset | judge_filter: adversarial_hand_batch1 (original 40 tasks) × ~1,200 in / ~100 out tok | 48,000 | 4,000 | 0.01 |
| 2026-04-30 | ~ | OpenRouter | openai/gpt-4o-mini | dataset | synthetic semantic edge-case generation — ~70 attempts, 60 kept; ~1,100 in / ~265 out tok | 77,000 | 18,550 | 0.02 |

**No API calls for:**
- Programmatic templates generation (fully deterministic, zero LLM calls)
- Adversarial hand-authored tasks TB-ADV-001–040 (human-written, judge scores assigned after manual review)
- Adversarial expansion TB-ADV-041–052 (hardcoded judge scores in gen_adversarial_12.py)
- partition.py and contamination_check.py (no model inference)
- synthesis_router.py (stub file, never executed)

---

## Token Cost Methodology

Costs estimated from prompt structure × task count at OpenRouter April 2026 pricing:

| Model | Input $/1M | Output $/1M | Why chosen |
|-------|-----------|------------|------------|
| google/gemini-2.5-flash | $0.15 | $0.60 | Trace-derived generation (different family from judge) |
| openai/gpt-4o-mini      | $0.15 | $0.60 | Judge-filter and synthesis — cheapest capable tier on OpenRouter |

gpt-4o-mini was selected over deepseek/deepseek-v3.2 ($0.27/M in, $1.10/M out) for 45–55% cost reduction while maintaining IC/GTV/RAC scoring quality at the 1–5 scale required by the judge filter.

**judge_filter.py per-task token breakdown:**
- System prompt (`_JUDGE_SYSTEM`): ~650 tokens
- User prompt: brief JSON (~350 tok) + email subject+body (~165 tok) + scores JSON (~80 tok) + headers (~50 tok) = ~645 tokens
- Response: `{"input_coherence": N, "ground_truth_verifiability": N, "rubric_application_clarity": N, "notes": "..."}` = ~90 tokens
- **Per task: ~1,295 input / ~90 output**

**synthetic_semantic_edge_cases.py per-task token breakdown:**
- Instructions (format rules + semantic failure type): ~800 tokens
- Brief JSON: ~300 tokens
- Response: subject + body + explanation + JSON overhead = ~265 tokens
- **Per task: ~1,100 input / ~265 output**

**trace_derived.py per-task token breakdown:**
- System message + format rules: ~500 tokens
- User message (trace event JSON + instructions): ~700 tokens
- Response (email JSON with subject + body): ~350 tokens
- **Per task: ~1,200 input / ~350 output**

---

## Bucket Definitions

| Bucket | Allowed days | Purpose |
|--------|-------------|---------|
| **dataset** | Days 2–3 only | Synthesis, dedup, judge-filter during dataset authoring |
| **training** | Day 5 | Colab T4 or RunPod compute for LoRA training run |
| **eval** | Days 5–6 only | Held-out slice scoring, ≤4 passes on sealed partition |
| **reserve** | Any | Bug fixes, re-runs, late probe additions |

---

## Running Total

| Bucket | Spent ($) | Budget ($) | Remaining ($) |
|--------|-----------|-----------|--------------|
| dataset  | 0.10 | 5.00 | 4.90 |
| training | 0.00 | 0.00 (free Colab T4) | 0.00 |
| eval     | 0.00 | 3.00 | 3.00 |
| reserve  | 0.00 | 2.00 | 2.00 |
| **total** | **0.10** | **10.00** | **9.90** |

---

## Hard Rules

- **No eval-tier spend on Days 2–3.** All dataset authoring uses dev-tier models only.
- **No τ²-Bench retail validation runs.** Week 10 score reused as reference; re-running
  is a cost-discipline failure.
- **No Claude Sonnet 4.6 / GPT-5 class models at any stage.** Dev-tier = gpt-4o-mini
  or Qwen3-Next-80B-A3B via OpenRouter.
- **Log every charge before the next session.** Missing charges invalidate the cost artifact.
- **RunPod cap: $5.** Use only if Colab T4 session limits force it on Day 5.
