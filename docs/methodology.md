# Methodology: Path B — Preference-Tuned Judge for Tenacious-Bench

## Path Selection: B (Judge/Critic)

Path B is selected. Justification is grounded in Week 10 trace evidence.

### Trace IDs Cited

- **snaptrade-2026-04-28** (`artifacts/snaptrade/email_log.jsonl`): velocity=-60%,
  segment=Ambiguous, pitch="bottlenecks integrating new APIs" → semantic mismatch,
  `tone_warnings: []` — the existing checker saw nothing wrong.

- **wiseitech-2026-04-27** (`artifacts/wiseitech/email_log.jsonl`): velocity=-100%,
  segment=Ambiguous, pitch="find themselves needing to augment existing Python, Big Data,
  and ML capabilities" → zero hiring signal; augmentation ask has no evidential basis.

- **arcana-2026-04-27** (`artifacts/arcana/email_log.jsonl`): velocity=+100%,
  segment=Segment 1, pitch="scaling bottleneck post-Series A" → correct baseline.
  `tone_warnings: []` — correct and clean.

### Why Not Path A (SFT Generation)

Email prose quality is adequate across all five companies with logged emails. `tone_warnings`
fires zero times. The failure is not that the generator writes bad sentences — it is that the
inputs fed to the generator are wrong: ICP segment misclassified, hiring velocity direction
ignored, confidence inflated by generic stack presence. Training a better generator on
better-written outputs does not fix wrong inputs. The failure is upstream of generation.

### Why Not Path C (PRM)

Process reward modeling requires rich multi-step trajectory data with step-level correctness
labels. The Week 10 corpus has 9 companies with 2–6 turns each. Sparse, noisy step labels
would dominate any training signal. More importantly, the primary failure occurs at Turn 0
(the outbound email), before any multi-step trajectory begins. This is a classification
failure at one step, not a compounding trajectory failure across steps.

### Why Path B

The Week 10 system already has the judgment scaffolding: `tone_warnings`, `confidence`
thresholds, `honesty_flags`, `weak_hiring_velocity_signal`. The infrastructure reports but
never enforces. The gap is a trained component that can evaluate a `(brief, email)` pair and
return a structured PASS/REJECT with reasoning — not just a surface checklist.

Path B trains exactly this: a preference-tuned critic where:
- **Chosen** = correct REJECT with semantic reasoning ("contraction signal + growth frame")
- **Rejected** = surface-only PASS ("no banned phrases, word count clean, recommend send")

This mirrors the actual failure mode: the agent gets the surface right but the substance wrong,
and nothing currently catches it.

### Model and Training Choices

**Backbone: Qwen 3.5 0.8B** (pinned, not merged)
Fits on free Colab T4 in 16-bit LoRA with headroom. Sufficient capacity for binary
classification + one-sentence reasoning output. Fast iteration during ablations.

**Method: SimPO** (Simple Preference Optimization, reference-free)
Reference-free: no frozen reference model held in memory alongside the training model.
Fits Colab T4 without VRAM spill. SimPO's length-normalized reward handles variable-length
judge outputs better than token-level DPO loss. Lighter than DPO, more stable than ORPO on
short classification-style outputs.

**Adapter: LoRA only** (rank 16, alpha 32)
Not merged. Published as adapter only on HuggingFace. Budget constraint: full fine-tune
does not fit in $10 compute envelope.

### Cost Rules

- No Claude Sonnet 4.6 / GPT-class models at any stage.
- No τ²-Bench retail re-runs.
- Synthesis and judge filtering: DeepSeek V3.2 via OpenRouter (dev-tier only, Days 2–3).
- Held-out evaluation: DeepSeek V3.2 (Days 5–6 only, sealed slice, ≤4 passes total).
- Training: Unsloth on Colab T4 (free). RunPod only if Colab session caps force it (cap $5).

### Path Declaration

Path B. Committed 2026-04-29. Primary failure modes targeted: ICP misclassification and
signal over-claiming. Backbone: Qwen 3.5 0.8B. Training method: SimPO via Unsloth.

---

## Model Rotation Policy (Act II Dataset Authoring)

To prevent preference leakage — where the model used to generate tasks scores favorably under the judge because they share the same biases — generation and quality filtering use separate model families wherever possible.

### Rotation Table

| Authoring Mode | Generation Model | Judge Model | Same Family? |
| --- | --- | --- | --- |
| Trace-derived (batch 1) | Google Gemini (free tier) | DeepSeek V3.2 | **No** ✓ |
| Programmatic templates | None (deterministic) | DeepSeek V3.2 | N/A ✓ |
| Adversarial hand-authored | None (human-written) | DeepSeek V3.2 | N/A ✓ |
| Synthetic semantic edge cases | DeepSeek V3.2 | DeepSeek V3.2 | Yes* |

*For synthetic semantic edge cases, the same model family is used for generation and filtering because the judge scores **email quality** (instruction compliance, grounded truth value, rubric alignment confidence) — not semantic correctness. The semantic incorrectness is verified by construction (the brief explicitly encodes the intended failure type). An adversary cannot "game" the judge by generating semantically wrong emails that the judge scores as high-quality, because the judge is evaluating different dimensions than the intended failure.

### What Preference Leakage Means Here

If the generation model and judge model are identical, the generator may learn stylistic patterns that the judge rewards regardless of rubric compliance. This is the preference leakage risk. It is mitigated by:

1. **Cross-family rotation** for the largest batches (trace-derived, where prose quality is most variable).
2. **Deterministic generation** for programmatic tasks (no LLM, no leakage possible).
3. **Human authorship** for adversarial tasks (no LLM generation bias).
4. **Structural separation** for synthetic tasks: the judge evaluates surface quality; the semantic failure is embedded in the brief structure, not in how the email is written.

### Claude API Exclusion

Claude Sonnet 4.6 and GPT-class models are excluded from all generation and judging stages per the cost rules in §Cost Rules above. All LLM calls in the pipeline use DeepSeek V3.2 via OpenRouter (dev-tier pricing) or Google Gemini (free tier).
