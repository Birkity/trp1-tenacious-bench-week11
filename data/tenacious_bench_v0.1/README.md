---
license: cc-by-4.0
task_categories:
  - text-classification
  - token-classification
language:
  - en
tags:
  - evaluation
  - benchmark
  - sales
  - B2B
  - preference-learning
  - simpo
  - judge
  - outbound-email
pretty_name: Tenacious-Bench v0.1
size_categories:
  - n<1K
---

# Tenacious-Bench v0.1

Machine-verifiable evaluation benchmark for B2B sales-agent quality.
257 tasks · four authoring modes · three sealed partitions · SimPO-trained judge adapter.

## Dataset Summary

Tenacious-Bench measures whether a `(brief, email)` pair satisfies five rubric dimensions
that are fully machine-verifiable without an LLM:

| Code | Dimension | Rule |
|------|-----------|------|
| D1 | grounding_fidelity | All numerics in email traceable to brief |
| D2 | icp_pitch_alignment | Ambiguous segment → no product claim; must end with `?` |
| D3 | signal_directionality | Negative velocity → no growth-frame terms |
| D4 | tone_compliance | No banned phrases (30+ style-guide terms) |
| D5 | format_compliance | Subject ≤60 chars; body ≤120 words; ≤1 `?`; no URLs |

## Dataset Structure

```
tenacious_bench_v0.1/
├── dev/
│   ├── trace_derived_batch1.jsonl       75 tasks — Week 10 company event traces
│   ├── programmatic_batch1.jsonl        75 tasks — deterministic parameter templates
│   └── adversarial_hand_batch1.jsonl    52 tasks — hand-authored edge cases
├── dev_synthetic/
│   └── semantic_edge_cases_batch1.jsonl 55 tasks — LLM-generated semantic probes
├── train/
│   └── tasks.jsonl                      114 tasks (44%) — preference pair source
└── dev/
    └── dev_tasks.jsonl                  78 tasks  (30%) — public eval partition
```

> **Note:** The `held_out/` partition (65 tasks, 25%) is sealed and not distributed.

## Task Schema

Each task is a JSON object:

```json
{
  "task_id": "TB-TRACE-006A",
  "source_mode": "trace_derived",
  "difficulty": "easy",
  "brief": {
    "company": "...",
    "icp_segment": "Segment 1 | 2 | 3 | 4 | Ambiguous",
    "hiring_velocity": { "direction": "up", "delta_pct": 45, "signal_strength": "strong" },
    "bench_match": { "bench_available": true },
    "grounding_facts": ["...", "..."],
    "ai_maturity": 2
  },
  "email": { "subject": "...", "body": "..." },
  "rubric": { "dimensions": [...] },
  "judge_filter": { "IC": 4, "GTV": 4, "RAC": 3 }
}
```

## Dataset Composition

| Source Mode | Tasks | Easy | Medium | Hard | PASS | REJECT |
|-------------|-------|------|--------|------|------|--------|
| trace_derived | 75 | 15 | 32 | 28 | 22 | 53 |
| programmatic | 75 | 20 | 30 | 25 | 25 | 50 |
| adversarial_hand | 52 | 8 | 1 | 43 | 10 | 42 |
| synthetic_semantic | 55 | — | — | — | 55 | 0 |
| **Total** | **257** | 43 | 63 | 96 | 112 | 145 |

## Partitions

| Partition | Tasks | Split | SHA-256 (16 hex) |
|-----------|-------|-------|-----------------|
| train | 114 | 44% | `6668fea2735097f6` |
| dev | 78 | 30% | `3637a2857fbba383` |
| held_out | 65 | 25% | `14b14e9650aee2bb` (sealed) |

Split is deterministic with `seed=42`, family-aware (A/B/C variants of same task kept together).

## Contamination Checks

All three checks PASS:

| Check | Result |
|-------|--------|
| N-gram overlap (n=8) | 0 shared 8-grams between held_out and train |
| Embedding cosine similarity | Max 0.81 (threshold 0.85) |
| Time-shift | Feb–Apr 2026 signals vs Aug 2025 model cutoff |

## SimPO Judge Adapter (Act III)

The `train/` partition was used to build 228 dimension-reasoning preference pairs
(`tenacious_judge_act5_pairs.jsonl`). A LoRA adapter was trained on
`unsloth/Qwen2.5-3B-Instruct` via SimPO (TRL `CPOConfig(loss_type="simpo")`).

| Parameter | Value |
|-----------|-------|
| Backbone | `unsloth/Qwen2.5-3B-Instruct` |
| Preference pairs | 228 (114 tasks × 2 pairs) |
| Final train loss | 0.0745 |
| Wall time | 14 min (Colab T4) |

Adapter: `tenacious_judge_adapter/` in the companion model repo.

## Ablation Results (Act IV)

| Partition | Base model | Trained judge | Delta B |
|-----------|-----------|---------------|---------|
| dev (78) | 60.3% | 65.5% | +5.2% |
| held_out (65) | 50.8% | 59.4% | +8.6% |

Trained judge eliminated all D2 (ICP alignment) errors on held-out (14 → 0)
and reduced D1 errors by 38% (13 → 8).

## Inter-Rater Agreement

30-task calibration study by primary author (two rounds, 24-hour gap):
min per-dimension agreement 97%, min Cohen's κ = 0.92.

## License

CC-BY-4.0. See `DATASHEET.md` for full distribution terms.

## Citation

```bibtex
@misc{tenacious-bench-2026,
  author    = {Birkity Yishak},
  title     = {Tenacious-Bench v0.1: A Machine-Verifiable Benchmark for B2B Sales Agent Quality},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/Birkity/tenacious_bench_v0.1}
}
```
