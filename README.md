# Tenacious-Bench v0.1

Machine-verifiable evaluation benchmark for B2B sales-agent quality. Built from Week 10
Conversion Engine traces, probe failures, and Week 11 audit/schema/scoring design.

**Acts I & II complete.** 257 tasks, four authoring modes, three partitions sealed.

---

## Status (as of 2026-04-30)

| Item | Status |
|------|--------|
| Act I — Audit & Schema | Complete |
| Act II — Dataset (257 tasks) | Complete |
| Partitions (train/dev/held_out) | Sealed (seed=42) |
| Contamination check | All 3 checks PASS (n-gram=0, cosine=0.81, time-shift) |
| Inter-rater agreement | Complete (min 97%, κ≥0.92) |
| Datasheet | Complete (`docs/datasheet.md`) |
| Act III — Preference pairs + SimPO training | Days 5–6 |
| Act IV — Held-out evaluation | Day 6–7 |

---

## Repository Layout

```
benchmark/
  schema.json                     Task JSON schema (v0.1)
  dimensions.md                   Full rubric specification
  scoring_evaluator.py            Deterministic scorer — no LLM, no temperature

data/
  contamination_check.json        3-check contamination audit output
  tenacious_bench_v0.1/
    dev/
      trace_derived_batch1.jsonl  (75 tasks — from Week 10 company events)
      programmatic_batch1.jsonl   (75 tasks — deterministic templates)
      adversarial_hand_batch1.jsonl (40 tasks — hand-authored edge cases)
    dev_synthetic/
      semantic_edge_cases_batch1.jsonl (60 tasks — LLM-generated Phase 2 probes)
    train/tasks.jsonl             (114 tasks, 44% — preference pair source)
    dev/tasks.jsonl               (78 tasks, 30% — public eval)
    held_out/tasks.jsonl          (65 tasks, 25% — SEALED, gitignored)

docs/
  audit_memo.md                   Why Tenacious-Bench is needed (Week 10 evidence)
  methodology.md                  Path B justification + model rotation policy
  datasheet.md                    Gebru + Pushkarna template (3–5 pages)
  inter_rater_agreement.md        30-task human calibration study
  cost_log.md                     API and compute charges
  architecture.md                 Six-layer system overview

scripts/
  generation/
    trace_derived.py              Batch 1 generation (trace-derived)
    programmatic_templates.py     Batch 2 (fully deterministic, no API)
    adversarial_hand.py           Batch 3 (hand-authored)
    synthetic_semantic_edge_cases.py  Batch 4 (DeepSeek V3.2 via OpenRouter)
    judge_filter.py               LLM quality gate (DeepSeek V3.2)
    partition.py                  Family-aware 50/30/20 split (seed=42)
  analysis/
    contamination_check.py        N-gram + embedding + time-shift audit

synthesis_memos/
  memo_datasheets_for_datasets.md   Common reading: Gebru et al. (2018)
  memo_data_cards.md                Common reading: Pushkarna et al. (2022)

report/
  interim_report.tex              LaTeX interim report (Acts I & II)
```

---

## Setup

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Requires `OPENROUTER_API_KEY` in `.env` for generation and judge-filter scripts.

---

## Quick Verification

```bash
# 1. Run the deterministic scorer on a task
python benchmark/scoring_evaluator.py

# 2. Check partition integrity
python -c "
import json
for p in ['train','dev','held_out']:
    tasks = [json.loads(l) for l in open(f'data/tenacious_bench_v0.1/{p}/tasks.jsonl')]
    print(f'{p}: {len(tasks)} tasks')
"

# 3. Verify B/C variant isolation (programmatic batch)
python -c "
import json, sys
sys.path.insert(0, 'benchmark')
from scoring_evaluator import score_task
tasks = [json.loads(l) for l in open('data/tenacious_bench_v0.1/dev/programmatic_batch1.jsonl')]
b_ok = all(score_task(t)['grounding_fidelity']==0 for t in tasks if t['task_id'].endswith('B'))
c_ok = all(score_task(t)['icp_pitch_alignment']==0 for t in tasks if t['task_id'].endswith('C'))
print('B D1=0:', b_ok, '| C D2=0:', c_ok)
"

# 4. Contamination check
python scripts/analysis/contamination_check.py
```

---

## Dataset Composition

| Source Mode | Tasks | Easy | Medium | Hard | PASS | REJECT |
|-------------|-------|------|--------|------|------|--------|
| trace_derived | 75 | 15 | 32 | 28 | 22 | 53 |
| programmatic | 75 | 20 | 30 | 25 | 25 | 50 |
| adversarial_hand | 52 | 8 | 1 | 43 | 10 | 42 |
| synthetic_semantic | 55 | — | — | — | 55 | 0 |
| **Total** | **257** | 43 | 63 | 96 | 112 | 145 |

Adversarial batch expanded from 40 to 52 tasks (12 new hardcoded edge cases: TB-ADV-041–052
targeting D1/D2/D3/D4/D5 failures and one Phase 2 semantic probe).
Synthetic batch reduced from 60 to 55 tasks (5 removed for "leverage" style-guide violation).
Synthetic semantic tasks (TB-SEM) are Phase 2 probes designed to pass D1–D5 but carry
semantically unjustified claims. Difficulty labels not assigned.

---

## Rubric Dimensions

| Code | Dimension | Rule |
|------|-----------|------|
| D1 | Grounding Fidelity | All numerics in email appear in brief; `bench_available=False` blocks product claims |
| D2 | ICP Pitch Alignment | Phase 1: `segment=Ambiguous` + product claim = REJECT; Ambiguous email must end with `?`. Phase 2 LLM: checks segment→frame match for all segments (`--llm-judge`) |
| D3 | Signal Directionality | `delta_pct < -20%` + growth-frame term = REJECT |
| D4 | Tone Compliance | 30+ banned phrases from style guide (incl. "leverage", "skyrocket", "synergize"); "bench" word banned in prospect copy |
| D5 | Format Compliance | Approved subject prefix; subject ≤60 chars; body ≤120 words; ≤1 `?`; no URLs; no booking phrases |

D1/D3/D4/D5 and D2 Phase 1 are fully deterministic. D2 Phase 2 uses OpenRouter LLM (`--llm-judge` flag).

---

## What's Next (Days 4–7)

- **Day 4**: Synthesis memos, itemise cost log, sentence-transformers cosine recheck
- **Day 5**: HuggingFace upload (`train/` + `dev/`), preference pair generation (~100 pairs)
- **Day 6**: SimPO fine-tune Qwen 3.5 0.8B on Colab T4 (LoRA rank 16, adapter-only)
- **Day 6–7**: Held-out evaluation (≤4 passes), precision/recall/F1 vs. deterministic scorer
- **Day 7**: Final report (Act III: training curves, held-out metrics, error analysis)
