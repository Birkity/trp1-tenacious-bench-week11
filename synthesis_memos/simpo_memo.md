# SimPO: Simple Preference Optimization — Engineering Memo

*Meng, Xia, and Chen. "SimPO: Simple Preference Optimization with a Reference-Free Reward." NeurIPS 2024.*

---

## 1. What problem the paper solves

DPO requires a frozen reference model to compute the KL penalty term. On constrained hardware this means two model copies in GPU memory simultaneously — for a 0.8B model on Colab T4 that is manageable, but any growth in model size or batch size risks OOM. More importantly, the reference model introduces a hyperparameter (β, the KL coefficient) that requires careful tuning and decays in relevance as the policy drifts. SimPO removes the reference entirely by replacing the per-token log-probability reward with **mean log-probability** — the average of log p(y_t | x, y_{<t}) across the response tokens. This one change eliminates the reference model, adds implicit length normalisation (short responses are no longer trivially preferred), and replaces KL regularisation with a **target margin M** in the Bradley–Terry objective. The margin forces a minimum separation between chosen and rejected reward rather than just requiring chosen > rejected.

---

## 2. What idea you are taking from it

The **reference-free mean-log-prob reward with a target margin**.

Concretely: for each preference pair (brief, email_chosen, email_rejected), the SimPO loss encourages:

```
mean_logp(email_chosen) - mean_logp(email_rejected) >= M
```

No reference model. No β to tune. The only new hyperparameter is M (target margin), which has a direct interpretation: how far apart must the model's confidence in the chosen vs rejected response be before the pair is considered "learned"?

**Disagreement with the paper's default M=2.0**: The authors tune M on AlpacaEval 2 with thousands of pairs and 7B+ models. With 114 preference pairs on Qwen 0.8B, M=2.0 is too aggressive — it forces near-perfect separation on a tiny dataset and risks gradient spikes that collapse outputs. We use **M=0.5** as the starting point, calibrated to the pair count, and anneal upward if the training loss does not decrease.

---

## 3. Where it appears in your system

- **`training_data/simpo_pairs.jsonl`** — the (prompt, chosen, rejected) pairs that SimPO trains on. `chosen` = a REJECT email with a complete D1–D5 rubric critique attached as reasoning chain. `rejected` = the same brief paired with a surface-PASS email that has no rubric grounding.
- **Act III training script** (Colab) — LoRA rank 16 on `Qwen/Qwen2.5-0.5B-Instruct`, SimPO loss with M=0.5, batch size 4, gradient accumulation 8, ~3 epochs on the 114-pair train partition.
- **`docs/methodology.md` §Path B Justification** — documents why SimPO was selected over DPO (no reference model, length normalisation, stable cross-entropy base) and over ORPO (see orpo_memo.md for the full rejection argument).
