# Synthesis Memo: Datasheets for Datasets (Gebru et al., 2018)

**Reading:** Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J.W., Wallach, H., Daumé III, H., & Crawford, K. (2018). Datasheets for Datasets. *Communications of the ACM*, 64(12), 86–92.

**Author:** Birkity Yishak 
**Date:** 2026-04-29  
**Project context:** Tenacious-Bench v0.1 dataset authoring (Week 11, TRP1)

---

## Core Argument

Gebru et al. argue that ML datasets lack the standardized documentation that hardware components have had for decades ("datasheets"). Without structured transparency, dataset creators and consumers operate with incompatible assumptions about intended use, collection methodology, and known limitations — creating silent failure modes that surface only after deployment. The datasheet template addresses this by mandating structured disclosure across seven questions (motivation, composition, collection, preprocessing, uses, distribution, maintenance) for every released dataset.

The analogy to hardware datasheets is precise: a capacitor's datasheet specifies operating conditions, failure modes, and rated tolerances so that an engineer can make an informed decision about whether to use it in a circuit. A dataset without equivalent documentation invites misuse at best and catastrophic deployment at worst.

---

## How It Applies to Tenacious-Bench

### Motivation (§1 of the datasheet template)

The paper asks: who funded the dataset? For what purpose? Tenacious-Bench answers both clearly — no external funding, single researcher (Birkity Mekasha / Tenacious Intelligence Corporation), primary purpose is evaluating a Path B preference-tuned judge for a specific B2B outbound email generation pipeline. This narrow scope is not a limitation to hide; it is the core claim of applicability. A dataset designed for one narrow purpose is more useful than an under-documented general dataset precisely because users know whether they are in scope.

### Composition (§2)

Gebru et al. require explicit documentation of: what the instances represent, how many there are, any missing data, any known errors, and whether instances can be considered independent. Tenacious-Bench addresses all five:
- **What instances represent**: (brief, email) pairs tested against five rubric dimensions
- **Count**: 250 tasks across four authoring modes
- **Missing data**: explicitly none — all tasks are complete; the `difficulty` field is absent only for synthetic tasks (by design, not omission)
- **Known errors**: single-annotator inter-rater study is a known limitation (§7.1 of `docs/datasheet.md`)
- **Independence**: explicitly not independent — A/B/C variants share observation text, which drove the family-aware partitioning design

The family-aware partitioning in `scripts/generation/partition.py` is a direct implementation of the independence concern raised in the composition section. Without grouping variants into families, train/held_out contamination via shared observation text would undermine the evaluation validity that the datasheet promises.

### Collection Process (§3)

The paper specifically asks whether subjects were notified and whether they consented. For Tenacious-Bench, the "subjects" are synthetic (LLM-generated) or derived from publicly available hiring signals — no consent mechanism applies. But the paper's deeper question is: what real-world entities are represented, and could they be harmed? The benchmark references real-sounding company names in the trace-derived batch, but names used are from Week 10 training traces, not companies whose actual emails are being evaluated in production. This distinction is worth making explicit.

The model rotation policy (§3.3 of `docs/datasheet.md`) directly addresses Gebru et al.'s concern about annotation bias. When the same model generates and filters a dataset, the distribution of "passing" examples is skewed toward that model's output priors. Using Gemini for trace-derived generation and DeepSeek V3.2 for filtering breaks this circular dependency for three of four authoring modes.

### Preprocessing (§4)

Gebru et al. note that many datasets fail to document their labeling process — who labeled, under what instructions, with what quality check. Tenacious-Bench uses a deterministic scorer (`benchmark/scoring_evaluator.py`) which eliminates annotator subjectivity by construction. The inter-rater agreement study in `docs/inter_rater_agreement.md` documents that human review of the deterministic labels achieves κ ≥ 0.92, but the labels themselves are computed, not assigned. This is more transparent than crowd-sourced labeling because the labeling logic is published in full source code.

### Uses (§5)

The paper emphasizes distinguishing intended uses from out-of-scope uses, and listing uses that should be discouraged. `docs/datasheet.md` §5 follows this structure explicitly:
- **Intended**: benchmark evaluation of Path B judge, regression testing for the email pipeline
- **Out-of-scope**: training a production email generator (deliberately wrong emails in the dataset would reinforce errors), general NLP benchmarks (narrow domain)
- **Future**: SimPO preference training from `train/` partition; Phase 2 semantic evaluation from `dev_synthetic/`

The "out-of-scope uses" section is where most dataset documentation fails. Gebru et al. note that omitting this section implicitly licenses any use. Tenacious-Bench documents the specific misuse scenario (using REJECT-labeled emails as positive training examples) that would result from ignoring the intended use boundaries.

---

## Gaps and Honest Limitations

Gebru et al. recommend a multi-annotator study with at least two independent labelers. Tenacious-Bench uses a single annotator (the primary author) in two rounds with a 24-hour gap. This satisfies the letter of the inter-rater agreement requirement but not its spirit — a true multi-annotator study would require an independent second labeler. The single-annotator design is documented as a known limitation in `docs/datasheet.md` §7.1 and is the highest-priority issue for v0.2.

The paper also recommends documenting the "individual economic or other costs" of collection. `docs/cost_log.md` exists but is not fully itemised — approximately 190 API calls to DeepSeek V3.2 via OpenRouter have occurred but exact dollar amounts have not been tallied. This should be completed before final submission.

---

## Key Takeaway for Tenacious-Bench

The most important contribution of Gebru et al. is framing dataset documentation as a **social contract** between creator and consumer, not a checkbox exercise. The datasheet is not for the creator — it is for the person who will use the dataset in a context the creator never imagined. For Tenacious-Bench, the most likely misuse scenario is using this benchmark to evaluate a general-purpose email assistant that operates outside the B2B SaaS staffing-augmentation ICP. The datasheet's narrow-domain limitation notice (§7.3) is the mechanism that prevents this misapplication. Without it, a well-meaning researcher could publish a misleading "our model achieves 92% on Tenacious-Bench" headline that means nothing outside the specific domain.
