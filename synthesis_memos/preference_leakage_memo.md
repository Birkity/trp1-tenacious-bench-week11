# Preference Leakage — Engineering Memo

*Li et al. "Preference Leakage: A Contamination Problem in LLM-as-a-Judge." ICLR 2026 (arXiv 2025).*

---

## 1. What problem the paper solves

When a model generates training data and a related model judges it, the judge has a systematic bias toward outputs that resemble its own generation style. This is preference leakage: the judge rewards familiarity, not quality. The paper defines "relatedness" precisely — same model, same family (parent/child fine-tunes), or sibling checkpoints — and shows empirically that related judges inflate scores for related generators by a consistent, measurable margin across tasks. The implication is that any benchmark or training pipeline that uses one LLM family for both generation and evaluation is measuring that model's self-consistency, not task performance.

---

## 2. What idea you are taking from it

The **model rotation policy**: every stage of the pipeline must document which model is used, and generation models and judge models must come from different families.

Concretely, the paper gives us the audit checklist:
1. List every model at every stage (generation, quality filter, evaluation).
2. Check relatedness: same family = leak risk.
3. If same family is unavoidable, document it explicitly and constrain the judge to criterion-anchored output (which reduces style bias compared to open-ended scoring).

**Disagreement with the paper's proposed fix of "use an unrelated judge"**: In practice, the available cheap inference tier on OpenRouter is dominated by a small number of families. For synthetic generation and quality filtering, we used `openai/gpt-4o-mini` for both — same family, which the paper flags as a leakage risk. Our counter-argument: the judge scores *structural quality* (IC/GTV/RAC on a 1–5 scale), not semantic correctness. The evaluation-time model is the trained Qwen 0.8B judge, not gpt-4o-mini. The leakage concern applies when the same model judges the training data that trains the final evaluation model; here, the training target is Qwen 0.8B, which was trained by a different lab entirely. The residual risk is that gpt-4o-mini may inflate IC/GTV/RAC scores for gpt-4o-mini-style prose — we document this as a known limitation in `docs/datasheet.md §7.3`.

---

## 3. Where it appears in your system

- **`docs/methodology.md` §Model Rotation Policy** — the full table documenting generation model vs judge model at every stage, with the explicit family-separation rationale:
  - Trace-derived generation: `google/gemini-2.5-flash` (Google family)
  - Quality filter: `openai/gpt-4o-mini` (OpenAI family) → different family from Gemini ✓
  - Synthetic generation: `openai/gpt-4o-mini`
  - Synthetic quality filter: `openai/gpt-4o-mini` → same family, documented as acceptable (structural scoring only, see §3.3 of datasheet)
  - Deterministic scoring: `benchmark/scoring_evaluator.py` — no LLM, zero leakage risk
- **`docs/cost_log.md`** — every model and call logged by name so the rotation policy is auditable.
- **`scripts/analysis/contamination_check.py`** — the n-gram overlap and cosine similarity checks are a direct implementation of the paper's recommendation to run automated leakage detection after dataset construction.
- **`scripts/generation/partition.py` family-aware split** — grouping tasks by company (trace-derived and synthetic) ensures that model-specific phrasing patterns for a given company do not leak across the train/held-out boundary, which is the partition-level version of the paper's contamination control.
