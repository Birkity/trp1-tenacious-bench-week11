# Tenacious-Bench — Artifact URLs

## HuggingFace Dataset (Artifact #1)
https://huggingface.co/datasets/Birkity/tenacious_bench_v0.1

## HuggingFace Model — Tenacious Judge LoRA Adapter (Artifact #2)
https://huggingface.co/Birkity/tenacious-judge

## Community Post (Artifact #3)
https://www.reddit.com/r/FunMachineLearning/comments/1t1mlby/built_a_sales_judgment_benchmark_tenacious_bench/

## GitHub Repository
Branch: feat/act4
https://github.com/10academy-trp1/trp1-tenacious-bench-week11

---

## Key Numbers for Report

| Metric | Value |
| --- | --- |
| Total benchmark tasks | 257 |
| Training pairs (SimPO) | 228 |
| Backbone | unsloth/Qwen2.5-3B-Instruct |
| Final train loss | 0.0745 |
| Wall time | 14 min (Colab T4) |
| Base model accuracy (held-out) | 50.8% |
| Trained judge accuracy (held-out) | 59.4% (over scored tasks) |
| Delta B (lift from fine-tuning) | +8.6% |
| D2 errors eliminated | 14 → 0 |
| D1 errors reduced | 13 → 8 |
| Total API cost | $0.10 |
| Training compute cost | $0.00 (Colab free tier) |
