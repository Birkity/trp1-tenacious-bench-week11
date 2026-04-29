# Tenacious-Bench Week 11

Machine-verifiable sales-agent evaluation benchmark for Tenacious-Bench v0.1, built from Week 10 Conversion Engine traces, probes, and Week 11 audit/schema/scoring design.

GitHub: https://github.com/Birkity/trp1-tenacious-bench-week11

## Short Week 11 Challenge Description

Week 11 turns the Week 10 Conversion Engine into a domain-specific evaluation system. Instead of re-running tau2-Bench retail, this repo designs Tenacious-Bench: a benchmark for B2B sales-agent quality focused on grounding fidelity, ICP-pitch alignment, signal directionality, tone compliance, and format compliance. The benchmark is built from private local source material, Week 10 traces, probe failures, and Tenacious sales rules, then used to construct datasets and train a small judge/critic in later phases.

## Repository Layout

```text
benchmark/                  Schema, dimensions, and runnable scoring evaluator
docs/                       Audit memo, methodology, architecture, cost log, reading memos
docs/challenge/             Local Week 11 challenge doc, ignored by git
evidence/                   Selected Week 10 traces, probes, generated emails, tau2 references
scripts/generation/         Future dataset construction scripts
data/tenacious_bench_v0.1/  Train/dev/held-out benchmark partitions
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
