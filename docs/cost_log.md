# Cost Log — Week 11

All API and compute charges logged here. Every entry requires: date, time (UTC),
provider, model, bucket, task description, units, and cost in USD.

## Entries

| Date | UTC | Provider | Model | Bucket | Task | Units | Cost ($) |
|---|---|---|---|---|---|---|---|
| 2026-04-29 | — | — | — | — | Phase 0: folder setup, no model calls | 0 | 0.00 |

---

## Bucket Definitions

| Bucket | Allowed days | Purpose |
|---|---|---|
| **dataset** | Days 2–3 only | Synthesis, dedup, judge-filter during dataset authoring |
| **training** | Day 5 | Colab T4 or RunPod compute for LoRA training run |
| **eval** | Days 5–6 only | Held-out slice scoring, ≤4 passes on sealed partition |
| **reserve** | Any | Bug fixes, re-runs, late probe additions |

---

## Running Total

| Bucket | Spent ($) | Budget ($) | Remaining ($) |
|---|---|---|---|
| dataset | 0.00 | 5.00 | 5.00 |
| training | 0.00 | 0.00 (free) | 0.00 |
| eval | 0.00 | 3.00 | 3.00 |
| reserve | 0.00 | 2.00 | 2.00 |
| **total** | **0.00** | **10.00** | **10.00** |

---

## Hard Rules

- **No eval-tier spend on Days 2–3.** All dataset authoring uses dev-tier models only.
- **No τ²-Bench retail validation runs.** Week 10 score reused as reference; re-running
  is a cost-discipline failure.
- **No Claude Sonnet 4.6 / GPT-5 class models at any stage.** Dev-tier = DeepSeek V3.2
  or Qwen3-Next-80B-A3B via OpenRouter.
- **Log every charge before the next session.** Missing charges invalidate the cost artifact.
- **RunPod cap: $5.** Use only if Colab T4 session limits force it on Day 5.
