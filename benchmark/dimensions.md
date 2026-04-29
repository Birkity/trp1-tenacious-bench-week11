# Tenacious-Bench Dimensions

Tenacious-Bench scores a `(brief, email, prior_thread, bench_summary, rubric)` task. The binding rule is simple: if any dimension fails, the email is rejected. The benchmark prefers deterministic checks where possible and reserves LLM judgment for semantic ICP-pitch alignment.

## D1 Grounding Fidelity

**What it means:** Every concrete claim in the email must be supported by the supplied brief, prior thread, bench summary, or pricing/rubric facts. Concrete claims include numbers, dates, funding amounts, hiring-velocity statements, stack claims, availability claims, and claims about Tenacious capability.

**Why it matters:** Week 10's main trust risk was not bad prose; it was confident claims built on weak or contradictory evidence. SnapTrade had `delta_pct=-60.0` and `icp_segment=ambiguous`, yet the email moved from a real decrease signal to an unsupported API acceleration pitch. WiseiTech had `delta_pct=-100.0`, `weak_hiring_velocity_signal=true`, and still received an augmentation claim. Probes 11-13 show that prospects challenge weak or stale signal directly.

**PASS looks like:** The email repeats or paraphrases at least one supplied grounding fact, and all numeric/date/funding/capacity tokens are present in the evidence. If evidence is weak, the email asks rather than asserts.

**FAIL looks like:** The email invents a funding amount, percentage, year, role type, stack availability, customer claim, or hiring state not present in the task evidence. It also fails if it makes no concrete reference to the brief at all.

**Scoring type:** Programmatic first. Numeric/date/currency tokens are checked against the evidence text; at least one supplied grounding fact must appear or be matched by its key terms. A later LLM judge may be added for paraphrase-heavy claims, but the Phase 1 evaluator stays deterministic.

## D2 ICP-Pitch Alignment

**What it means:** The email's primary pitch frame must match the ICP segment assigned in the brief.

**Why it matters:** The hardest Week 10 failure is confusing "some technical stack match exists" with "this company is in-market for this Tenacious pitch." SnapTrade and WiseiTech were `Ambiguous`, but both received product claims. The evaluation rule is that when no segment is justified, the system must abstain and send an exploratory email rather than segment-specific pitch.

**PASS looks like:** Segment 1 receives a growth, post-funding, scaling, or hiring-capacity pitch. Segment 2 receives a restructuring, continuity, cost-discipline, or retained-delivery pitch. Segment 3 receives a leadership-transition or vendor-mix reassessment pitch. Segment 4 receives a specific capability-gap pitch grounded in AI readiness and bench feasibility. Ambiguous receives a qualifying question and no product claim.

**FAIL looks like:** Ambiguous receives "Tenacious can provide..." or any direct product/capacity claim. Segment 1 receives a layoff/cost-cutting pitch. Segment 2 receives an aggressive scaling pitch. Segment 4 receives a generic engineering-shortage pitch or a capability pitch unsupported by AI readiness or bench availability.

**Scoring type:** LLM-judged later, with deterministic fast-fail for obvious Ambiguous-plus-product-claim cases. The Phase 1 evaluator includes a placeholder that catches the obvious fast-fail and otherwise returns pass until the trained or prompted judge is attached.

## D3 Signal Directionality

**What it means:** The pitch direction must match hiring velocity. Positive velocity can support scaling language; negative velocity requires caution, restructuring framing, or a qualifying question.

**Why it matters:** The confirmed Week 10 failure mode is signal over-claiming when job velocity is negative. SnapTrade decreased 60% and WiseiTech decreased 100%, but both emails used growth or augmentation frames. A generic benchmark would reward clean language; Tenacious-Bench rejects the direction mismatch.

**PASS looks like:** A company with `delta_pct >= -20` can receive a growth-frame pitch if other ICP evidence supports it. A company with `delta_pct < -20` avoids terms such as "scaling," "bottleneck," "accelerate," "augment your team," or "increased demand" unless the brief explicitly supplies a restructuring/cost-preservation rationale.

**FAIL looks like:** A company with materially negative hiring velocity receives a growth, acceleration, scale-up, or headcount-expansion pitch.

**Scoring type:** Programmatic. The evaluator checks `brief.hiring_velocity.delta_pct` and rejects negative-velocity emails containing growth-frame terms.

## D4 Tone Compliance

**What it means:** The email must obey the Tenacious style guide: direct, grounded, honest, professional, and non-condescending.

**Why it matters:** Week 10 already produced `tone_warnings`, but warnings did not block bad messages. Probes 18-20 show why tone matters: self-disclosed AI gaps can be mishandled as condescension, and a prospect asking "are you calling us unsophisticated?" is a relationship-risk event.

**PASS looks like:** No filler openers, no offshore-vendor cliches, no condescending gap language, no invented urgency, and no internal jargon such as "bench" in prospect-facing copy.

**FAIL looks like:** The body contains banned phrases such as "top talent," "world-class," "rockstar," "ninja," "aggressive hiring," "guaranteed ROI," "falling behind," "you lack," "left behind," "quick," "just," or "hope this finds." It also fails on prospect-facing "bench" jargon.

**Scoring type:** Programmatic banned-phrase and jargon check.

## D5 Format Compliance

**What it means:** The email must follow the cold-outreach structure from the style guide.

**Why it matters:** Format is not the deepest failure, but it is a reliable hard gate. The style guide caps cold email bodies at 120 words, subject lines at 60 characters, one ask per message, and no cold booking links. Week 10 also showed that booking links belong after an interested or scheduling reply, not in a first-touch email.

**PASS looks like:** Subject line is 60 characters or fewer and starts with an approved prefix: "Context:", "Note on", "Congrats on", or "Question on". Body is 120 words or fewer. The email contains no URL or Cal.com link and does not stack multiple asks.

**FAIL looks like:** Overlong subject, overlong body, URL in cold outreach, direct booking ask, multiple question marks, or unsupported meeting language such as "schedule a call" in first-touch copy.

**Scoring type:** Programmatic.

## Overall Verdict

The evaluator returns five binary dimension scores. `PASS` requires all five dimensions to score `1`. Any `0` produces `REJECT`.

## What This Enables for Phase 2 (Dataset Construction)

These dimensions define the labels for trace-derived tasks, programmatic sweeps, synthetic hard cases, and hand-authored adversarial examples. Phase 2 can now vary ICP segment, hiring velocity, grounding facts, tone traps, and format traps independently while keeping every task reproducible and scoreable by a stranger.
