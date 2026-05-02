# Tenacious-Bench v0.1

Machine-verifiable evaluation benchmark for B2B sales-agent quality. Built from Week 10
Conversion Engine traces, probe failures, and Week 11 audit/schema/scoring design.

**Acts I–IV complete.** 257 tasks · four authoring modes · three partitions sealed ·
SimPO-trained judge adapter in `tenacious_judge_adapter/`.

---

## Status (as of 2026-05-02)

| Item | Status |
|------|--------|
| Act I — Audit & Schema | ✅ Complete |
| Act II — Dataset (257 tasks) | ✅ Complete |
| Partitions (train/dev/held\_out) | ✅ Sealed (seed=42) |
| Contamination check | ✅ All 3 checks PASS (n-gram=0, cosine=0.81, time-shift) |
| Inter-rater agreement | ✅ Complete (min 97%, κ≥0.92) |
| Datasheet | ✅ Complete (`docs/datasheet.md`) |
| Act III — SimPO preference pairs + training | ✅ Complete |
| Act IV — Ablation (dev + held-out) | ✅ Complete |

### Act III — Training summary

| Parameter | Value |
|-----------|-------|
| Backbone | `unsloth/Qwen2.5-3B-Instruct` |
| Method | SimPO via TRL `CPOConfig(loss_type="simpo")` |
| Preference pairs | 228 — dimension-reasoning format (`training_data/tenacious_judge_act5_pairs.jsonl`) |
| LoRA rank / alpha | 16 / 16, dropout=0.0, all 7 projection modules |
| Epochs | 3 |
| Effective batch | 16 (batch=4 × grad\_accum=4) |
| Learning rate | 2e-4 |
| β / γ | 2.0 / 0.5 |
| Final train loss | 0.0745 |
| Wall time | 14 min (Google Colab T4, 16 GB) |
| Adapter | `tenacious_judge_adapter/` |

### Act IV — Ablation

Three judges compared on dev (78 tasks) and held-out (65 tasks) partitions:

1. **Deterministic** — `score_task()` from `scoring_evaluator.py` (100% by construction, ground truth)
2. **Base model** — `Qwen2.5-3B-Instruct`, no adapter, structured-output prompt
3. **Trained judge** — same backbone + SimPO LoRA adapter

| Partition  | n  | Deterministic | Base model    | Trained judge   | Delta A | Delta B |
|------------|----|---------------|---------------|-----------------|---------|---------|
| dev        | 78 | 100.0%        | 60.3% (47/78) | 65.5% (36/55\*) | −34.5%  | +5.2%   |
| held\_out  | 65 | 100.0%        | 50.8% (33/65) | 59.4% (19/32\*) | −40.6%  | +8.6%   |

\* Trained model produced UNKNOWN on 23 dev / 33 held-out tasks; accuracy computed over scored tasks only.

**Dimension-level findings (held-out):** Trained judge eliminated D2 (ICP alignment) errors entirely
(base: 14 errors → trained: 0), reduced D1 (grounding) errors from 13 → 8.
UNKNOWN outputs concentrated on programmatic-C and trace-C tasks (Ambiguous+product-claim pattern).

Results in `ablations/ablation_results.json`.
Per-task traces in `ablations/held_out_traces.jsonl`.

---

## Repository Layout

```
benchmark/
  schema.json                       Task JSON schema (v0.1)
  dimensions.md                     Full rubric specification
  scoring_evaluator.py              Deterministic scorer — no LLM, no temperature

data/
  contamination_check.json          3-check contamination audit output
  tenacious_bench_v0.1/
    dev/
      trace_derived_batch1.jsonl    75 tasks — Week 10 company event traces
      programmatic_batch1.jsonl     75 tasks — deterministic parameter templates
      adversarial_hand_batch1.jsonl 52 tasks — hand-authored edge cases
    dev_synthetic/
      semantic_edge_cases_batch1.jsonl  55 tasks — LLM-generated Phase 2 probes
    train/tasks.jsonl               114 tasks (44%) — preference pair source
    dev/dev_tasks.jsonl             78 tasks  (30%) — public eval partition
    held_out/held_tasks.jsonl       65 tasks  (25%) — SEALED, not committed

docs/
  audit_memo.md                     Week 10 failure evidence (why Tenacious-Bench)
  methodology.md                    Path B rationale + model rotation policy
  datasheet.md                      Gebru + Pushkarna datasheet (7 sections)
  inter_rater_agreement.md          30-task human calibration (97%, κ≥0.92)
  cost_log.md                       Full API and compute cost log ($0.10 total)
  architecture.md                   Six-layer system overview

scripts/
  generation/
    trace_derived.py                Batch 1 — trace-derived generation
    programmatic_templates.py       Batch 2 — deterministic templates, no API
    adversarial_hand.py             Batch 3 — hand-authored adversarial tasks
    synthetic_semantic_edge_cases.py Batch 4 — LLM semantic probes
    judge_filter.py                 LLM quality gate
    partition.py                    50/30/20 split (seed=42)
    augment_preference_pairs.py     Expand pairs v1→v2 (Phase A deterministic)
  analysis/
    contamination_check.py          N-gram + embedding + time-shift audit
    run_ablation.py                 Three-judge ablation (Act IV)
  build_act5_preferences.py         Dimension-reasoning pair builder (Act V fix)

training/
  train_simpo_judge.py              SimPO LoRA training script (Colab-ready)
  requirements.txt                  Pinned training dependencies

training_data/
  tenacious_judge_train.jsonl       114 original preference pairs (v1)
  tenacious_judge_train_v2.jsonl    200 pairs (v1 + Phase A multi-negative)
  tenacious_judge_act5_pairs.jsonl  228 pairs — dimension-reasoning format (used for training)

tenacious_judge_adapter/
  adapter_config.json               PEFT LoRA adapter config
  adapter_model.safetensors         Trained weights
  training_config.json              Full hyperparameter log
  loss_log.json                     Per-step loss (45 steps × 3 epochs)
  loss_curve.png                    Training loss plot

ablations/
  ablation_results.json             Per-task verdicts + summary metrics (dev + held_out)
  held_out_traces.jsonl             Raw model outputs for audit

evidence_graph.json                 Maps every numeric claim to its source file
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
# 1. Deterministic scorer smoke-test
python benchmark/scoring_evaluator.py

# 2. Partition counts
python -c "
import json
for p, f in [('train','tasks.jsonl'),('dev','dev_tasks.jsonl'),('held_out','held_tasks.jsonl')]:
    tasks = [json.loads(l) for l in open(f'data/tenacious_bench_v0.1/{p}/{f}')]
    print(f'{p}: {len(tasks)} tasks')
"

# 3. Verify B/C variant isolation
python -c "
import json, sys; sys.path.insert(0, 'benchmark')
from scoring_evaluator import score_task
tasks = [json.loads(l) for l in open('data/tenacious_bench_v0.1/dev/programmatic_batch1.jsonl')]
b_ok = all(score_task(t)['grounding_fidelity']==0 for t in tasks if t['task_id'].endswith('B'))
c_ok = all(score_task(t)['icp_pitch_alignment']==0 for t in tasks if t['task_id'].endswith('C'))
print('B D1=0:', b_ok, '| C D2=0:', c_ok)
"

# 4. Ablation dry-run (no GPU needed)
python scripts/analysis/run_ablation.py --dry-run

# 5. Build Act V preference pairs
python scripts/build_act5_preferences.py
```

---

## Dataset Composition

| Source Mode | Tasks | Easy | Medium | Hard | PASS | REJECT |
|-------------|-------|------|--------|------|------|--------|
| trace\_derived | 75 | 15 | 32 | 28 | 22 | 53 |
| programmatic | 75 | 20 | 30 | 25 | 25 | 50 |
| adversarial\_hand | 52 | 8 | 1 | 43 | 10 | 42 |
| synthetic\_semantic | 55 | — | — | — | 55 | 0 |
| **Total** | **257** | 43 | 63 | 96 | 112 | 145 |

---

## Rubric Dimensions

| Code | Dimension | Rule |
|------|-----------|------|
| D1 | Grounding Fidelity | All numerics in email traceable to brief; `bench_available=False` blocks product claims |
| D2 | ICP Pitch Alignment | Phase 1: `Ambiguous`+product claim=REJECT; Ambiguous must end with `?`. Phase 2 LLM: full segment→frame check (`--llm-judge`) |
| D3 | Signal Directionality | `delta_pct < -20%` + growth-frame term = REJECT |
| D4 | Tone Compliance | 30+ banned phrases (style guide); "bench" banned in prospect copy |
| D5 | Format Compliance | Approved subject prefix; ≤60 chars subject; ≤120 words body; ≤1 `?`; no URLs; no booking phrases |

D1/D3/D4/D5 and D2 Phase 1 are fully deterministic (no LLM, no temperature).

---

## Preference Pair Design (Act V — dimension-reasoning format)

The final training set (`tenacious_judge_act5_pairs.jsonl`) uses structured D1–D5 reasoning
instead of short verdict sentences, specifically to prevent preference leakage:

**Chosen** (ground truth from `score_task()`):
```
D1: PASS — grounding facts verified; $14M matches brief
D2: FAIL — Ambiguous segment with product claim
D3: PASS — ok
D4: PASS — ok
D5: PASS — ok
VERDICT: REJECT
Primary failure: D2
```

**Rejected A** (aggressive wrong — all PASS, wrong verdict):
```
D1: PASS — numerics align with available evidence
D2: PASS — segment-appropriate framing applied
...
VERDICT: PASS
```

**Rejected B** (subtle wrong — right verdict, wrong primary dimension):
```
D1: FAIL — dollar amount in email does not appear in brief grounding facts
D2: PASS — ...
...
VERDICT: REJECT
Primary failure: D1   ← wrong; real failure was D2
```

---

## License

CC-BY-4.0. See `docs/datasheet.md` §6 for full distribution terms.
