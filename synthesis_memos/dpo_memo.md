# DPO: Direct Preference Optimization — Engineering Memo

*Rafailov et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS 2023.*

---

## 1. What problem the paper solves

RLHF requires three components running simultaneously: a policy model, a frozen reference model, and a separately-trained reward model whose outputs feed a PPO loop. At every training step, PPO samples from the current policy, scores samples with the reward model, and updates via clipped importance-weighted gradients. This is expensive, numerically unstable, and requires careful reward-model calibration. DPO collapses all three into a single fine-tuning objective by showing that, under the Bradley–Terry preference model, the optimal policy can be recovered directly from (prompt, chosen, rejected) pairs without ever explicitly constructing a reward model. The implicit reward is the ratio of the policy's log-probability to the reference policy's log-probability.

---

## 2. What idea you are taking from it

The **preference pair as the atomic unit of alignment training**.

DPO establishes that all alignment signal can be expressed as a (prompt, chosen, rejected) triple and that training on these triples directly adjusts model behaviour without reward modelling or RL. This defines the data contract that every method in this reading list — SimPO, ORPO — inherits. We do not use DPO's algorithm, but we use its data structure.

Specifically: the task of Acts I and II is to produce high-quality (brief, email_correct_reject, email_surface_pass) triples. The correctness of that labelling — which response is chosen, which is rejected — must be machine-verifiable and criterion-anchored. DPO is the paper that proves this data structure is sufficient for alignment; every subsequent paper is an efficiency improvement on top of it.

**Disagreement with DPO's algorithm**: DPO requires a reference model for the KL term. We do not use DPO — we use SimPO, which is strictly better suited to our 114-pair, 0.8B, T4-constrained setup. DPO's contribution to this project is the conceptual proof that preference pairs work; SimPO's contribution is how to train on them without a reference model.

---

## 3. Where it appears in your system

- **`training_data/simpo_pairs.jsonl` schema** — the (prompt, chosen, rejected) structure is DPO's contribution. Each record has: `prompt` = the signal brief, `chosen` = the correct REJECT email with full rubric reasoning, `rejected` = the surface-PASS email with no grounding. This format is DPO-defined even though the training algorithm is SimPO.
- **`docs/methodology.md` §Path B Justification** — cites DPO as the foundational algorithm and explains the step from DPO → SimPO as a memory and stability improvement, not a conceptual departure.
- **Act I dataset design** — the A/B/C task variant structure (A=PASS, B=D1-fail, C=D2-fail) exists to generate clean (chosen=A, rejected=B) and (chosen=A, rejected=C) pairs where the signal of which is better is unambiguous. This is DPO's core requirement: preference pairs must have a clear ground-truth ordering.
