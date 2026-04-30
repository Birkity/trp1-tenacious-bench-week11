# ORPO: Monolithic Preference Optimization — Engineering Memo

*Hong, Lee, and Thorne. "ORPO: Monolithic Preference Optimization without Reference Model." EMNLP 2024.*

---

## 1. What problem the paper solves

DPO and SimPO both require a separate preference-training phase after SFT. ORPO merges these into a single pass: it adds an odds-ratio penalty term directly to the standard SFT cross-entropy loss, penalising rejected outputs inline rather than in a second stage. The penalty is `log(1 - σ(log OR))` where OR = p(chosen|x) / p(rejected|x). One training loop produces a model that both learns to generate correctly (SFT) and learns to prefer better outputs (preference), with no reference model and no two-phase orchestration.

---

## 2. What idea you are taking from it

Nothing. ORPO is documented here as the **rejected alternative** — the method we explicitly evaluated and chose not to use.

The core rejection argument is:

**ORPO's odds-ratio penalty coefficient is not self-calibrating.** SimPO's target margin M has a direct, interpretable meaning: the minimum log-probability gap between chosen and rejected. If M=0.5 and the model is already achieving a gap of 0.5, the pair contributes zero gradient — the model has "learned" that pair. This makes M a natural early-stopping signal on small datasets.

ORPO's penalty coefficient λ does not have this property. It scales the odds-ratio penalty relative to the SFT loss, and the SFT loss magnitude changes throughout training as the model improves. With 114 preference pairs on a 0.8B model, λ miscalibration causes one of two failure modes: (a) λ too small — the preference signal is drowned out by the SFT objective and the model learns correct generation but not correct preference; (b) λ too large — the model over-penalises rejected outputs and collapses to a narrow distribution. Neither failure mode is detectable without held-out evaluation after training, which costs eval-partition budget.

Additionally, ORPO's empirical gains in the paper appear primarily at 1B+ models with thousands of preference pairs. With 114 pairs and 0.8B, the paper's own ablations suggest the odds-ratio contribution is marginal — essentially SFT with a small perturbation.

---

## 3. Where it appears in your system

Nowhere. SimPO was selected.

The decision is documented in `docs/methodology.md` §Path B Justification: "ORPO was evaluated and rejected. Its odds-ratio penalty coefficient requires calibration against SFT loss magnitude, which changes throughout training and cannot be reliably tuned on 114 preference pairs. SimPO's target margin M provides a directly interpretable stopping criterion and has been shown to converge stably on small datasets."
