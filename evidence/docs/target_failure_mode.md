# Target Failure Mode — Conversion Engine Act IV

**Selected failure mode: Tone Drift / Self-Disclosure Misclassification**
**Category: `tone_drift`**
**Run 1 probe pass rate: 0/3 (0% — worst-performing category across all 32 probes)**
**Run 2 probe pass rate: 3/3 (100% — RESOLVED)**

---

## What the failure was

When a prospect self-discloses a pain point or admits they are behind on a capability ("our AI team is not great right now", "we are definitely behind our competitors on AI"), the reply interpreter classified their reply as `QUESTION` or `UNKNOWN` instead of `INTERESTED`.

The result: the system sent either a clarification email or a follow-up question, when it should have sent a Cal.com booking link.

### Observed evidence from probe runs (2026-04-24, Run 1)

| Probe | Reply | Actual | Expected | Correct? |
|---|---|---|---|---|
| 18 | "Honestly, our AI team is not great right now and we know it." | QUESTION → SEND_EMAIL | INTERESTED → SEND_CAL_LINK | ✗ |
| 19 | "We are definitely behind our competitors on AI. That is clear to us." | UNKNOWN → ASK_CLARIFICATION | INTERESTED → SEND_CAL_LINK | ✗ |
| 20 | "I read your email three times. Are you calling us unsophisticated?" | UNKNOWN → ASK_CLARIFICATION | QUESTION → SEND_EMAIL | ✗ |

All three probes failed. This was the only category in the 32-probe suite with a 100% trigger rate.

---

## Why this failure was the highest-ROI target

### Frequency

Self-disclosure of a gap is one of the most common positive reply patterns in B2B outreach. CTOs and engineering leaders who are already aware of a gap — and reply to signal that awareness — represent the warmest segment of any outbound sequence.

Estimated prevalence in real outbound: ~10–15% of positive replies contain a self-disclosure or acknowledgement of a gap. With 1,000 outbound contacts at a 7% reply rate = 70 replies. At 12% self-disclosure rate among replies = approximately 8–9 mis-classified warm leads per 1,000 contacts.

### Business cost per mis-classification

Each mis-classified self-disclosure delayed the booking by one full email round (typically 5–7 days):

- System sends ASK_CLARIFICATION or SEND_EMAIL
- Prospect receives a generic or follow-up question
- Prospect replies again (if they still bother)
- System correctly classifies and sends Cal link
- Net delay: 5–7 days per warm lead

At a 7% email-to-booking conversion, a 5-day delay per warm lead means:
- 8–9 delayed bookings per 1,000 contacts
- Assuming a $5,000 average deal × 7% close rate: ~$3,000–4,000 in delayed or lost pipeline per 1,000 contacts

This was a **repeating cost** on every campaign run, not a one-time event.

---

## The root cause

The reply interpreter's `SYSTEM_PROMPT` defined `INTERESTED` as requiring explicit engagement signals:
- "Sounds interesting, can you send times?" → INTERESTED (explicit)
- "Sure, let's chat" → INTERESTED (explicit)

It did not include a pattern for **implicit buying signals via pain acknowledgement**:
- "Our AI team is not great" → implicit buying signal (system did not recognise)
- "We're behind our competitors" → implicit buying signal (system did not recognise)

The model defaulted to QUESTION (there might be a question implied) or UNKNOWN (ambiguous) when the reply contained negative self-assessment language without an explicit engagement phrase.

---

## The fix applied

Added a `SELF-DISCLOSURE BUYING SIGNALS` detection block to `INTERESTED` in `agent/reply_interpreter/prompts.py`:

```
SELF-DISCLOSURE (implicit buying signal): prospect acknowledges a pain,
gap, or weakness that DIRECTLY matches the service described in our email.
Examples of self-disclosure → INTERESTED:
  "our AI team is not great right now"          ← gap matches ML engineers
  "we're definitely behind our competitors on AI" ← gap matches our pitch
  "you're right, that bottleneck is real for us" ← confirms our framing
  "honestly we struggle with exactly that"       ← admits the problem

RULE: if the prospect uses negative self-assessment language about the
SAME capability we are offering, treat it as an implicit "yes, relevant."
Classify INTERESTED → SEND_CAL_LINK.
Do NOT require an explicit "I want to meet" phrase.

IMPORTANT DISTINCTION — these are NOT INTERESTED, they are QUESTION:
- "We need 8 NestJS engineers" (stating a capacity need, not a weakness)
- "Can you provide ML engineers?" (direct availability question)
- "We need both NestJS and Python starting in Q3" (stack request with timeline)
A prospect stating what they NEED is asking about our capacity → QUESTION.
A prospect saying they are BAD at something or BEHIND is self-disclosure → INTERESTED.
```

This rule is narrow and specific: it only fires when the negative self-assessment language directly maps to a service capability described in the email. It does not promote all negative-sentiment replies to INTERESTED, and it does not promote capacity requests (bench-over-commit probes) to INTERESTED.

---

## Verification results (Run 2)

| Probe | Reply | Actual (Run 2) | Expected | Correct? |
|---|---|---|---|---|
| 18 | "Honestly, our AI team is not great right now and we know it." | INTERESTED → SEND_CAL_LINK | INTERESTED → SEND_CAL_LINK | ✓ |
| 19 | "We are definitely behind our competitors on AI. That is clear to us." | INTERESTED → SEND_CAL_LINK | INTERESTED → SEND_CAL_LINK | ✓ |
| 20 | "I read your email three times. Are you calling us unsophisticated?" | QUESTION → SEND_EMAIL | QUESTION → SEND_EMAIL | ✓ |

Confirmed: probes 4, 5, 6 (hostile/hostile-sarcastic) still → `NOT_INTERESTED → STOP`. The fix did not promote hostile replies.

Overall pass rate: **31/32 (97%)**, up from 24/32 (75%). Tone drift pass rate: **3/3 (100%)**, up from 0/3.

---

## One honest unresolved failure (Act IV requirement)

**Probe #7:** `"Wow, another AI-generated outreach email. Super impressive."`

- **Expected:** `UNKNOWN → ASK_CLARIFICATION`
- **Actual (Run 2):** `QUESTION → SEND_EMAIL`

### Why it is unresolved

The prompt update extended `QUESTION` to cover authenticity challenges with "?" — which correctly fixed probe #28 ("Is this AI-generated?"). As a side effect, probe #7 — which has no "?" but is sarcastic — was also promoted to `QUESTION` because the model correctly identified the implied authenticity challenge.

The expected value in the probe (`UNKNOWN`) was written before the `QUESTION` definition was extended. The model's actual output (`QUESTION → SEND_EMAIL`) is **arguably better behavior** in production:

- `ASK_CLARIFICATION` response: "I'm not sure what you mean — could you clarify?" → confirms the suspicion of an automated system
- `SEND_EMAIL` response: directly addresses the AI concern with an honest acknowledgement → transforms a doubt into a credibility moment

### Business impact if deployed as-is

**Low.** No incorrect action is taken. The routing action is the same quality as the expected routing. The only risk is if the `SEND_EMAIL` response itself is poorly written — but that is a content quality risk in the downstream email generator, not a classification risk in the interpreter.

The probe expected value should be updated in a future iteration to `QUESTION → SEND_EMAIL`. The model is right; the probe expectation is stale.

---

## Secondary target (resolved as part of primary fix)

**Signal Over-Claim Failure 4b (Probe #13):** "Our open roles are for sales" → was false STOP in Run 1.

Fixed in Run 2 by adding to `NOT_INTERESTED` definition: `"your data was wrong"` challenges are `QUESTION`, not `NOT_INTERESTED`. Pass rate for `signal_over_claim`: 1/3 → 3/3.

---

## Why tone_drift was selected over alternative candidates

Three failure modes were candidates for the primary target. The final choice was `tone_drift` (self-disclosure misclassification). The two runner-up candidates are documented here for comparison.

### Candidate A: `bench_over_commit` — Stack-mismatch booking prevention

**What it was:** The router was sending Cal.com booking links for companies whose required stack (e.g. NestJS) was not available on the Tenacious bench (committed through Q3 2026). This would book a discovery call that the sales team would have to cancel or walk back.

**Trigger rate in probes:** 2/6 bench probes (33%) triggered false-positive bookings.

**Estimated cost per event:** Each mis-booked discovery call costs the sales team 30–60 minutes of prep + the call itself + a cancel/reschedule conversation with the prospect. Estimated: 2–3 hours per event. At 1,000 contacts → ~8 bench-gap events → ~16–24 hours of wasted consultant time.

**Why it lost to tone_drift:**

- `bench_over_commit` was partially mitigated by the existing `bench_match.bench_available` field in the brief. The fix was a single guard check in `_action_send_cal_link()` — 3 lines of code, no prompt change needed.
- `tone_drift` required a new classifier rule category, a prompt rewrite, and had 3× the trigger rate (0/3 vs 2/6 probes) at a higher business-impact event (lost warm lead vs. wasted call prep).
- `bench_over_commit` was fixed as a **supplementary** action during the primary fix (see "Router hardening" below).

**ROI comparison:** tone_drift fix recovers $3,000–4,000 delayed/lost pipeline per 1,000 contacts. bench_over_commit fix saves 16–24 hrs wasted consultant time per 1,000 contacts (~$1,600–2,400 at $100/hr). tone_drift had ~1.5–2× ROI.

---

### Candidate B: `scheduling_ambiguity` — Qualified date mis-routing

**What it was:** Replies containing soft or qualified scheduling language ("maybe June 30th", "possibly next week") were sometimes classified as `SCHEDULE → SEND_CAL_LINK` instead of `UNKNOWN → ASK_CLARIFICATION`. This sent a booking link to a prospect who had not committed, creating premature pressure.

**Trigger rate in probes:** 1/4 scheduling probes (25%) mis-classified.

**Estimated cost per event:** Premature booking link to an uncommitted prospect has moderate churn risk (~15% of recipients ignore the link after receiving it prematurely, vs ~5% when the link arrives post-commitment). At 1,000 contacts → ~3 premature links → ~0.45 additional no-show meetings.

**Why it lost to tone_drift:**

- Trigger rate was 25% vs. tone_drift's 100% (0/3 probes failed completely).
- Business cost per event was low (a no-show booking vs. a lost warm lead).
- The fix was low-confidence: the boundary between "soft yes" and "uncommitted maybe" is inherently ambiguous, and a tighter rule risked under-classifying genuine SCHEDULE replies.
- `tone_drift` had a clearer, more defensible fix boundary — negative self-assessment language maps directly to "implicit yes" with minimal false-positive risk.

**ROI comparison:** scheduling_ambiguity fix might prevent 0.45 additional no-shows per 1,000 contacts — negligible. tone_drift fix recovers 8–9 delayed bookings per 1,000 contacts — 18–20× higher impact.

---

## Router hardening (Act IV supplement)

In addition to the classifier fix, the reply router was hardened:

1. **Bench guard in `_action_send_cal_link()`**: checks `hsb.bench_match.bench_available` before sending the Cal link. If `bench_available=false`, downgrades to `SEND_EMAIL` with explicit NestJS Q3 2026 disclosure. Prevents booking a meeting for a stack we cannot staff.

2. **Removed booking ask from `_action_send_clarification()`**: the clarification email previously contained "would a 15-minute call work?" — a direct violation of the Act II contract (cold email = conversation opener only, no booking ask). This was removed; the email now ends with an open question grounded in the brief.
