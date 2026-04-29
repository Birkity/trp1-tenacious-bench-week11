# Synthesis Memo: Data Cards for AI Systems (Pushkarna et al., 2022)

**Reading:** Pushkarna, M., Zaldivar, A., & Kjartansson, O. (2022). Data Cards: Purposeful and Transparent Dataset Documentation for Responsible AI. *ACM FAccT '22*, 1776–1826.

**Author:** Birkity Yishak  
**Date:** 2026-04-29  
**Project context:** Tenacious-Bench v0.1 dataset authoring (Week 11, TRP1)

---

## Core Argument

Pushkarna et al. extend Gebru et al.'s datasheet proposal in two important directions. First, they shift the frame from documentation-as-compliance to documentation-as-design — arguing that writing a data card forces dataset creators to articulate decisions they would otherwise leave implicit. Second, they add the concept of **purposeful documentation**: the card should help specific stakeholders (ML practitioners, policy reviewers, affected communities) make informed decisions about whether and how to use the dataset. Different audiences have different informational needs, and a single flat document serves none of them well.

The paper introduces a structured template with seven high-level sections and a set of "guiding questions" within each. The key innovation over Gebru et al. is the explicit distinction between:
1. **Descriptive fields** (what the data is, factual)
2. **Normative fields** (what the data should and should not be used for, judgement)
3. **Provenance fields** (where the data came from and what happened to it)

This tripartite structure prevents a common documentation failure: treating everything as descriptive (which collapses normative intent into facts) or treating everything as normative (which hides the actual collection process behind stated intentions).

---

## How It Applies to Tenacious-Bench

### Purposeful Documentation for Multiple Stakeholders

Pushkarna et al. identify three primary stakeholder groups for most ML datasets: (1) dataset creators, (2) ML practitioners who will train or evaluate on the data, and (3) policy/audit reviewers who need to assess compliance. Tenacious-Bench has a narrower stakeholder set — primarily the author and the 10 Academy evaluation committee — but the documentation structure should still serve each:

- **For the author (future self)**: The partition checksums (`train=942091897fdc47e8`, `dev=214d6263ff15d26a`, `held_out=ab4775b484c51846`) and the deterministic seed (`RANDOM_SEED = 42`) in `scripts/generation/partition.py` ensure reproducibility. This is the Data Card "provenance" layer — if the partition script is re-run six months later, the same split should emerge.

- **For ML practitioners evaluating a judge model**: The dataset's PASS/REJECT label distribution (~44%/~56%), the five rubric dimensions, and the known limitation that D2 only catches Phase 1 fast-fails (Ambiguous segment + product claim) are the fields that determine whether a practitioner can trust an evaluation result. A practitioner who doesn't know that D2 misses semantic pitch errors will over-attribute high D2 scores as evidence of pitch quality.

- **For policy/audit reviewers**: The contamination controls (`data/contamination_check.json`) and the held-out seal date (2026-04-30) provide the audit trail that a reviewer needs to confirm that evaluation validity was maintained throughout the creation process.

### Dataset Lifecycle Documentation

Data Cards emphasize documenting the full **dataset lifecycle**, not just the final artifact. For Tenacious-Bench, this lifecycle has four phases:

1. **Raw signal collection** (Week 10 traces → 25 company events)
2. **Task generation** (four authoring modes, with model rotation policy)
3. **Quality filtering** (LLM-as-a-judge, three dimensions, threshold ≥ 3/5 each)
4. **Partitioning** (family-aware 50/30/20 split, seed=42, sealed 2026-04-30)

The Data Cards framework requires documenting what happens at each transition — what is discarded, why, and what properties are preserved vs. lost. For Tenacious-Bench:
- At quality filtering: 134 tasks were REJECTED by the judge filter out of 250 attempted (but kept as REJECT-labeled tasks, not discarded). The filtering adds the `difficulty` field without removing any task.
- At partitioning: families are kept together, which causes slight deviation from exact 50/30/20 ratios (123/83/44 instead of 125/75/50). This deviation is documented in `docs/datasheet.md` §3.5.

### Transparency About Dataset Limitations

Pushkarna et al. make the strongest case in the paper for documenting limitations **as design choices**, not failures. Every limitation is the result of a decision made under constraints. Tenacious-Bench has three primary limitations documented in `docs/datasheet.md` §7.3:

1. **Single annotator**: The decision to use one annotator in two rounds (vs. two independent annotators) was a resource constraint — a true multi-annotator study was not feasible within the one-week timeline. The Data Cards framework would ask: what is the risk introduced by this decision? The risk is that systematic annotation errors in the first round would propagate to the second round, inflating the agreement score. The 24-hour gap mitigates but does not eliminate this risk.

2. **Phase 1 only**: The evaluator catches five categories of rule-based violations but does not catch semantic errors — emails that correctly follow all syntactic rules but reason incorrectly from the brief. This is not a bug in the evaluator; it is a known scope boundary. The 60 synthetic semantic edge cases in `dev_synthetic/` are designed to measure exactly this gap in a follow-on Phase 2 evaluation. Documenting the limitation as a design decision (Phase 1 scope was chosen deliberately to ensure determinism) is more useful than documenting it as a failure.

3. **Template vocabulary contamination**: The n-gram and cosine contamination checks fail because template-based authoring modes share boilerplate phrases ("Companies scaling post-funding...", "Tenacious provides pre-vetted ML engineers deployable in days."). The Data Cards framework would note this as a **structural property of the generation method** — template generation guarantees internal consistency but sacrifices n-gram uniqueness. The time-shift check (the primary contamination defense for memorization risk) passes, which is the check that matters for model evaluation validity.

---

## Pushkarna vs. Gebru: What Data Cards Adds

The core addition is the concept of **intended and unintended audiences**. Gebru et al. write the datasheet for a generic "user." Pushkarna et al. force the author to enumerate specific stakeholder groups and explain what each group needs from the documentation that is different from what others need.

For Tenacious-Bench, this distinction matters most in the "Uses" section. The same task that is a valid evaluation example for a judge model researcher is potentially harmful if used as a training example for a production email generator (because REJECT-labeled emails would be positively reinforced). Pushkarna et al. would structure this as: 

| Stakeholder | What they need | What to warn against |
|-------------|---------------|---------------------|
| Judge model researcher | Full PASS/REJECT distribution, rubric specs | Don't use synthetic tasks for Phase 1 accuracy claims |
| Production email pipeline developer | None — out of scope | Don't use REJECT examples as positive training data |
| Academic NLP researcher | Source mode and difficulty labels | Benchmark is narrow-domain, not general-purpose |

The key normative field that Gebru et al. do not emphasize but Pushkarna et al. make central is the **misuse scenario table** — who might use this dataset, in what context, and what harm could result.

---

## Key Takeaway for Tenacious-Bench

Data Cards frames documentation as a form of **accountability infrastructure** — the document is not read once at release, it is consulted whenever a new use case is proposed. The permanent seal date of the held-out partition (2026-04-30), the deterministic seed, and the partition checksums are the machine-readable version of this accountability infrastructure. Any future evaluation that deviates from these provenance markers (different seed, different partition, different task count) is a different experiment from the one this benchmark was designed for.

The most actionable insight from Data Cards for this project is the requirement to be explicit about **what the dataset cannot tell you**. Tenacious-Bench can tell you whether a judge model correctly classifies B2B outbound emails on five deterministic rubric dimensions. It cannot tell you whether those emails are effective at generating pipeline, whether the ICP segments are correctly defined, or whether the rubric dimensions are the right ones for a different sales context. Writing these limits down is not a weakness — it is the precondition for making useful scientific claims.
