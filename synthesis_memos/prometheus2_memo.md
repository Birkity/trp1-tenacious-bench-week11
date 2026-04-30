# Prometheus 2 — Engineering Memo

*Kim et al. "Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models." EMNLP 2024.*

---

## 1. What problem the paper solves

Open-source LLM judges give inconsistent verdicts because they score against vague, implicit criteria — "is this a good response?" — instead of explicit, enumerable rubric dimensions. Prometheus 2 addresses this by training a dedicated evaluator fine-tuned on both direct scoring data (score 1–5 with rubric) and pairwise ranking data (which of two responses is better, and why), then merging the two fine-tuned checkpoints into a single model. The merged model can follow an arbitrary evaluation rubric written in the prompt, produces a score anchored to that rubric, and agrees with human / GPT-4 judgements more reliably than generic chat LLMs used as judges.

The core contribution is not the weight merging (which is an implementation detail) — it is the **output contract**: a judge should always (a) quote the criterion it is scoring against, (b) give a one-sentence reason, then (c) emit the score. This discipline is what makes the model's outputs machine-parseable and auditable.

---

## 2. What idea you are taking from it

The **criterion-anchored output format**.

Every scoring call should produce: `criterion → reason → score`. Not just a number. The reason must reference the specific dimension that passed or failed, not a generic quality statement.

**Disagreement with the paper's architecture**: Prometheus 2 achieves this via 7B weight merging — entirely infeasible on Colab T4 with a 0.8B model and no multi-GPU setup. We achieve the same output contract through a **deterministic evaluator** rather than a fine-tuned judge, which costs zero inference and is perfectly reproducible. The format discipline is preserved; the LLM component is not needed for Phase 1.

For Phase 2 (semantic evaluation), where a deterministic check is insufficient, the output format is enforced via the system prompt to gpt-4o-mini in `_llm_judge_d2()` — the model is instructed to output `{"segment_frame_match": bool, "reason": "..."}` and nothing else, directly implementing Prometheus 2's criterion-anchored contract.

---

## 3. Where it appears in your system

- **`benchmark/scoring_evaluator.py` `score_task()` return value** — returns `{"verdict": "REJECT", "failed_dimension": "D1", "reason": "Numeric token '$5M' not found in brief", ...}`. The `failed_dimension` + `reason` fields are a direct implementation of the criterion-anchored output format.
- **`benchmark/dimensions.md`** — the five rubric dimensions (D1–D5) serve as the evaluation criteria Prometheus 2 says must be explicit and enumerable. The scorer always names the dimension before emitting the score.
- **`_llm_judge_d2()` in scoring_evaluator.py** — the Phase 2 LLM judge enforces the criterion-anchored format via prompt instruction: the model must state the `segment_frame_match` boolean and give a `reason` string before the call is accepted.
- **`training_data/simpo_pairs.jsonl` chosen-response format** — the `chosen` email in each preference pair includes the rubric critique (which dimension failed and why), training the judge to produce criterion-anchored reasoning at inference time.
