# Tenacious-Bench Week 11

Machine-verifiable sales-agent evaluation benchmark for Tenacious-Bench v0.1, built from Week 10 Conversion Engine traces, probes, and Week 11 audit/schema/scoring design.

GitHub: https://github.com/Birkity/trp1-tenacious-bench-week11

## Short Week 11 Challenge Description

Week 11 turns the Week 10 Conversion Engine into a domain-specific evaluation system. Instead of re-running tau2-Bench retail, this repo designs Tenacious-Bench: a benchmark for B2B sales-agent quality focused on grounding fidelity, ICP-pitch alignment, signal directionality, tone compliance, and format compliance. The benchmark is built from private local source material, Week 10 traces, probe failures, and Tenacious sales rules, then used to construct datasets and train a small judge/critic in later phases.

## Repository Layout

```text
benchmark/                  Schema, dimensions, and runnable scoring evaluator
docs/                       Audit memo, methodology, architecture, cost log, reading memos
docs/challenge/             Local Week 11 challenge doc + style guide + examples
evidence/                   Selected Week 10 traces, probes, generated emails, tau2 references
scripts/generation/         Dataset construction scripts (trace-derived + programmatic)
data/tenacious_bench_v0.1/  Train/dev/held-out benchmark partitions (in progress)
training_data/              Preference data for Path B judge training
evaluation/ablations/       Evaluation and ablation outputs
tests/                      Future tests for schema and evaluator behavior
```

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python benchmark\scoring_evaluator.py
```

The evaluator should run three local dummy tasks and return one `PASS` plus two `REJECT` verdicts.

## Act II (Dataset Authoring) — Current Outputs

Dev batch files (JSONL tasks) live in:

- data/tenacious_bench_v0.1/dev/trace_derived_batch1.jsonl (75 tasks)
- data/tenacious_bench_v0.1/dev/programmatic_batch1.jsonl (75 tasks)
- data/tenacious_bench_v0.1/dev/programmatic_phase2_1_batch1.jsonl (75 tasks)

Naming/structure convention used in this repo:

- Dataset partitions are only: data/tenacious_bench_v0.1/{train,dev,held_out}/
- Within a partition, files follow: <source>_<phase?>_batch<N>.jsonl
	- Example: programmatic_phase2_1_batch1.jsonl
	- Reason: avoids creating extra pseudo-partitions

### Regenerate Batch 2 (Programmatic, deterministic)

This does not require any API keys.

```powershell
python scripts/generation/programmatic_templates.py
```

### Regenerate Phase 2.1 (Programmatic grid, deterministic)

This does not require any API keys.

```powershell
python scripts/generation/programmatic_phase2_1.py
```

### Run judge-filter (adds judge_filter + difficulty)

This requires an OpenRouter key in `OPENROUTER_API_KEY` (see `.env.example`).

```powershell
python scripts/generation/judge_filter.py --input-file data/tenacious_bench_v0.1/dev/programmatic_batch1.jsonl
```

### Spot-check intended failure isolations

```powershell
python -c "import json
from collections import Counter

tasks = [json.loads(l) for l in open('data/tenacious_bench_v0.1/dev/programmatic_batch1.jsonl', encoding='utf-8')]
print('Total:', len(tasks))
print('Difficulty:', Counter(t['difficulty'] for t in tasks))
from benchmark.scoring_evaluator import score_task
b_ok = all(score_task(t)['grounding_fidelity']==0 for t in tasks if t['task_id'].endswith('B'))
c_ok = all(score_task(t)['icp_pitch_alignment']==0 for t in tasks if t['task_id'].endswith('C'))
print('B D1=0:', b_ok, '| C D2=0:', c_ok)
"
```

## Interim Submission Readiness (Acts I–II)

This repo contains the core Act I/II implementation, but is not fully interim-ready yet.

Present:

- Act I audit memo: docs/audit_memo.md
- Methodology draft: docs/methodology.md
- Schema + evaluator: benchmark/schema.json, benchmark/scoring_evaluator.py
- Dev tasks (225 total so far): data/tenacious_bench_v0.1/dev/*.jsonl
- Generation scripts: scripts/generation/
- Synthesis memos (common readings): docs/synthesis_memos/
- Cost log: docs/cost_log.md

Missing (required for interim):

- Populated dataset partitions:
	- data/tenacious_bench_v0.1/train/
	- data/tenacious_bench_v0.1/held_out/
- Datasheet (3–5 pages) for Tenacious-Bench v0.1 (e.g., datasheet.md)
- Contamination check outputs (e.g., contamination_check.json) + script
- Inter-rater agreement artifact (e.g., inter_rater_agreement.md)

