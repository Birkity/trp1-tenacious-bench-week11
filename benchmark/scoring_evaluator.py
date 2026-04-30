"""
Tenacious-Bench Phase 1 scoring evaluator.

Deterministic across D1, D3, D4, D5. D2 (ICP-pitch alignment) uses an
OpenRouter LLM judge when OPENROUTER_API_KEY is set; falls back to the
deterministic Phase 1 fast-fail (Ambiguous + product claim) otherwise.

Usage:
    python benchmark/scoring_evaluator.py                          # smoke-test dummy tasks
    python benchmark/scoring_evaluator.py path/to/task.json        # score one task
    python benchmark/scoring_evaluator.py path/to/dir/             # score all *.jsonl in dir
    python benchmark/scoring_evaluator.py path/to/dir/ --no-llm    # deterministic-only mode
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Default rubric — all configurable overrides live here
# ---------------------------------------------------------------------------

DEFAULT_RUBRIC: dict[str, Any] = {
    "negative_velocity_threshold_pct": -20.0,
    "max_subject_chars": 60,
    "max_body_words": 120,
    "approved_subject_prefixes": [
        "context:",
        "note on",
        "congrats on",
        "question on",
        "question:",
        "re:",
        "resource:",
        "new:",
    ],
    # Full Tenacious style-guide banned phrase list (tenacious_style_guide_v2.md).
    # Any of these in subject or body → D4 FAIL.
    "banned_phrases": [
        # Offshore/vendor clichés
        "top talent",
        "world-class",
        "a-players",
        "rockstar",
        "ninja",
        "wizard",
        # Hyperbolic growth promises
        "skyrocket",
        "supercharge",
        "10x",
        "aggressive hiring",
        # Opener clichés
        "i hope this email finds you well",
        "hope this finds",
        "hey there",
        # Re-engagement guilt
        "just following up",
        "circling back",
        "per my last email",
        "per my last",
        # Fake-concision opener
        "i'll keep this brief",
        "i will keep this brief",
        # Condescension
        "falling behind",
        "you're behind",
        "you are behind",
        "you lack",
        "you're missing",
        "you are missing",
        "you need to catch up",
        "left behind",
        "behind the curve",
        # Junk urgency / scarcity
        "you'll regret",
        "you will regret",
        "don't miss out",
        "do not miss out",
        "last chance",
        # Consultant jargon
        "synergize",
        "synergy",
        "leverage",
        "ecosystem",
        "game-changer",
        "game changer",
        "disruptor",
        "paradigm shift",
        # Black-box claims
        "our proprietary",
        "ai-powered",
        # Generic openers
        "quick ",
        "just ",
        # Fabricated/invented claims guard
        "cost savings of",
        "guaranteed roi",
        "proven track record",
    ],
    "growth_frame_terms": [
        "bottleneck",
        "bottlenecks",
        "scaling",
        "scale your",
        "accelerate",
        "accelerating",
        "rapid growth",
        "increased demand",
        "augment your team",
        "augment existing",
        "expand the team",
        "staff up",
    ],
    "product_claim_terms": [
        "tenacious can",
        "tenacious provides",
        "tenacious offers",
        "we can",
        "we provide",
        "our engineers",
        "available on-demand",
        "pre-vetted",
        "project-ready",
        "augment your",
        "deploy in days",
    ],
    "icp_pitch_alignment_policy": (
        "Phase 1 fast-fails Ambiguous+product-claim and Ambiguous+no-question-mark. "
        "Full segment-frame check requires the D2 LLM judge (--llm-judge / OPENROUTER_API_KEY)."
    ),
}

# ---------------------------------------------------------------------------
# Segment → required pitch frame  (D2 LLM judge reference table)
# ---------------------------------------------------------------------------

SEGMENT_FRAMES: dict[str, str] = {
    "segment 1": "scaling bottleneck post funding",
    "segment 2": "restaffing after reduction",
    "segment 3": "new leader building stack",
    "segment 4": "specific AI capability gap",
    "ambiguous":  "qualifying question only — no product claims permitted",
}

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

NUMERIC_TOKEN_RE = re.compile(
    r"\$[\d,.]+[A-Za-z]*|\b\d+(?:\.\d+)?%|\b\d{4}\b|\bQ[1-4]\s+\d{4}\b",
    re.IGNORECASE,
)
URL_RE = re.compile(r"https?://|cal\.com", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z0-9$%]+")


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def _rubric(task_or_rubric: dict[str, Any] | None = None) -> dict[str, Any]:
    rubric = dict(DEFAULT_RUBRIC)
    if task_or_rubric:
        source = task_or_rubric.get("_rubric") or task_or_rubric.get("rubric", task_or_rubric)
        if isinstance(source, dict):
            for key, value in source.items():
                if value in (None, ""):
                    continue
                # List fields (banned_phrases, growth_frame_terms, etc.) are additive —
                # we take the union so that new entries added to DEFAULT_RUBRIC always
                # apply even when tasks carry an older baked-in rubric.
                if isinstance(value, list) and isinstance(rubric.get(key), list):
                    combined = list(rubric[key])
                    lower_existing = {v.lower() for v in combined}
                    for item in value:
                        if item.lower() not in lower_existing:
                            combined.append(item)
                    rubric[key] = combined
                else:
                    rubric[key] = value
    return rubric


def _evidence_text(brief: dict[str, Any]) -> str:
    chunks = [json.dumps(brief, sort_keys=True)]
    velocity = brief.get("hiring_velocity", {})
    delta = velocity.get("delta_pct")
    if isinstance(delta, (int, float)):
        chunks.append(f"{abs(delta):g}%")
        chunks.append(f"{abs(round(delta)):g}%")
        chunks.append(f"{delta:g}%")
    return _normalize(" ".join(chunks))


def _contains_grounding_fact(email_body: str, brief: dict[str, Any]) -> bool:
    body = _normalize(email_body)
    facts = [f for f in brief.get("grounding_facts", []) if isinstance(f, str)]
    if not facts:
        return False

    for fact in facts:
        fact_norm = _normalize(fact)
        if fact_norm and fact_norm in body:
            return True
        # Allow compact paraphrases via meaningful token overlap.
        tokens = [
            t
            for t in WORD_RE.findall(fact_norm)
            if len(t) > 3 and t not in {"have", "with", "that", "this", "from"}
        ]
        if tokens:
            hits = sum(1 for token in set(tokens) if token in body)
            if hits >= min(3, len(set(tokens))):
                return True

    return False


# ---------------------------------------------------------------------------
# D1 — Grounding Fidelity
# ---------------------------------------------------------------------------

def score_grounding_fidelity(email: dict[str, Any], brief: dict[str, Any]) -> tuple[int, str]:
    """Return (score, reason). Score=1 means the email's claims are grounded in the brief."""
    rubric = _rubric(brief)
    body = email.get("body", "")

    if not _contains_grounding_fact(body, brief):
        return 0, "No grounding fact from brief found in email body"

    evidence = _evidence_text(brief)
    numeric_tokens = NUMERIC_TOKEN_RE.findall(body)
    for token in numeric_tokens:
        token_norm = _normalize(token)
        if token_norm not in evidence:
            return 0, f"Numeric token {token!r} not found in brief"

    bench = brief.get("bench_match", {})
    unavailable = [s.lower() for s in bench.get("unavailable_stacks", [])]
    bench_available = bench.get("bench_available", True)
    body_lower = body.lower()

    if bench_available is False and any(term in body_lower for term in rubric["product_claim_terms"]):
        return 0, "Product claim made when bench_available=False"

    if unavailable and any(stack in body_lower for stack in unavailable):
        if any(term in body_lower for term in ("available", "can provide", "can staff", "deploy")):
            return 0, f"Claimed capacity for unavailable stack"

    return 1, "ok"


# ---------------------------------------------------------------------------
# D2 — ICP-Pitch Alignment  (deterministic Phase 1 + optional LLM Phase 2)
# ---------------------------------------------------------------------------

def _llm_judge_d2(segment: str, email: dict[str, Any]) -> tuple[bool, str]:
    """
    Call OpenRouter to check whether the email pitch frame matches the
    required frame for the given ICP segment.

    Returns (passes: bool, reason: str).
    Raises RuntimeError on API failure so the caller can fall back gracefully.
    """
    import urllib.request

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    required_frame = SEGMENT_FRAMES.get(segment.lower(), "unknown")
    subject = email.get("subject", "")
    body = email.get("body", "")

    prompt = f"""You are a B2B sales-email evaluator for Tenacious Intelligence Corporation.

ICP segment: {segment}
Required pitch frame for this segment: "{required_frame}"

Email subject: {subject}
Email body:
{body}

Question: Does the email's pitch frame match the required frame for this segment?
- For "scaling bottleneck post funding": email should frame Tenacious as solving a hiring/capacity bottleneck following a funding event.
- For "restaffing after reduction": email should acknowledge a headcount reduction and frame Tenacious as a cost-efficient delivery option.
- For "new leader building stack": email should reference a leadership change and frame Tenacious as a vendor-reassessment opportunity.
- For "specific AI capability gap": email should reference a specific ML/AI capability gap relative to peers.
- For "qualifying question only": email must ask a qualifying question and must NOT make a product claim.

Respond with exactly one word: true (frame matches) or false (frame does not match).
No explanation."""

    payload = json.dumps({
        "model": "deepseek/deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 0,
    }).encode()

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/tenacious-bench",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    content = data["choices"][0]["message"]["content"].strip().lower()
    passes = content.startswith("true")
    reason = "LLM judge: frame matches" if passes else f"LLM judge: frame mismatch for {segment} (expected: {required_frame})"
    return passes, reason


def score_icp_pitch_alignment(
    brief: dict[str, Any],
    email: dict[str, Any],
    use_llm: bool = False,
) -> tuple[int, str]:
    """
    Phase 1 deterministic fast-fails + optional LLM check.

    Phase 1 catches:
      - Ambiguous segment + any product claim → FAIL
      - Ambiguous segment + body does not end with ? → FAIL

    Phase 2 (use_llm=True): additionally calls the LLM judge for full frame-check.
    """
    segment = str(brief.get("icp_segment", "")).strip().lower()
    body = email.get("body", "").lower()
    rubric = _rubric(brief)

    # Phase 1a: Ambiguous + product claim
    if segment == "ambiguous":
        if any(term.lower() in body for term in rubric["product_claim_terms"]):
            return 0, "Ambiguous segment with product claim (Phase 1 fast-fail)"

    # Phase 1b: Ambiguous + no closing question mark
    if segment == "ambiguous":
        # last non-empty, non-signature line must end with ?
        sig_tokens = {"birkity", "research partner", "tenacious intelligence", "gettenacious.com"}
        lines = [l.strip() for l in email.get("body", "").splitlines() if l.strip()]
        content_lines = [l for l in lines if not any(tok in l.lower() for tok in sig_tokens)]
        if content_lines:
            last = content_lines[-1]
            if not last.endswith("?"):
                return 0, "Ambiguous segment: email body must end with a qualifying question (?)"

    # Phase 2: LLM judge (all segments including Ambiguous)
    if use_llm:
        try:
            passes, reason = _llm_judge_d2(segment, email)
            if not passes:
                return 0, reason
        except Exception as exc:
            # Graceful fallback: log warning, do not fail on API error
            print(f"  [D2 LLM warning] {exc} — falling back to Phase 1 deterministic", file=sys.stderr)

    return 1, "ok"


# ---------------------------------------------------------------------------
# D3 — Signal Directionality
# ---------------------------------------------------------------------------

def score_signal_directionality(brief: dict[str, Any], email: dict[str, Any]) -> tuple[int, str]:
    """Return (score, reason). Fails when negative hiring velocity is paired with growth-frame copy."""
    rubric = _rubric(brief)
    velocity = brief.get("hiring_velocity", {})
    delta = velocity.get("delta_pct", 0)
    try:
        delta_value = float(delta)
    except (TypeError, ValueError):
        delta_value = 0.0

    threshold = rubric["negative_velocity_threshold_pct"]
    body = email.get("body", "").lower()
    growth_terms = rubric["growth_frame_terms"]
    matched = [t for t in growth_terms if t.lower() in body]

    if delta_value < threshold and matched:
        return 0, f"Negative velocity ({delta_value}%) with growth-frame term(s): {matched[:2]}"
    return 1, "ok"


# ---------------------------------------------------------------------------
# D4 — Tone Compliance
# ---------------------------------------------------------------------------

def score_tone_compliance(email: dict[str, Any], brief: dict[str, Any] | None = None) -> tuple[int, str]:
    """Return (score, reason). Fails on any banned phrase or external 'bench' jargon."""
    rubric = _rubric(email)
    text = _normalize(f"{email.get('subject', '')} {email.get('body', '')}")
    banned = rubric["banned_phrases"]

    for phrase in banned:
        if phrase.lower() in text:
            return 0, f"Banned phrase found: {phrase!r}"

    # "bench" is internal jargon — must not appear in prospect-facing copy.
    if re.search(r"\bbench\b", text):
        return 0, "Word 'bench' used in prospect-facing email (internal jargon)"

    return 1, "ok"


# ---------------------------------------------------------------------------
# D5 — Format Compliance
# ---------------------------------------------------------------------------

def score_format(email: dict[str, Any]) -> tuple[int, str]:
    """Return (score, reason). Fails on format constraint violations."""
    rubric = _rubric(email)
    subject = email.get("subject", "")
    body = email.get("body", "")
    subject_lower = subject.lower()

    if len(subject) > rubric["max_subject_chars"]:
        return 0, f"Subject too long ({len(subject)} chars > {rubric['max_subject_chars']})"

    if not any(subject_lower.startswith(prefix) for prefix in rubric["approved_subject_prefixes"]):
        return 0, f"Subject does not start with an approved prefix: {subject!r}"

    word_count = len(body.split())
    if word_count > rubric["max_body_words"]:
        return 0, f"Body too long ({word_count} words > {rubric['max_body_words']})"

    if URL_RE.search(body):
        return 0, "URL found in email body (not permitted in cold outreach)"

    if body.count("?") > 1:
        return 0, f"Multiple question marks found ({body.count('?')} > 1)"

    meeting_phrases = ("schedule a", "book a", "15 minutes", "book time")
    for phrase in meeting_phrases:
        if phrase in body.lower():
            return 0, f"Meeting-booking phrase found: {phrase!r}"

    return 1, "ok"


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

_DIM_CODES = {
    "grounding_fidelity":    "D1",
    "icp_pitch_alignment":   "D2",
    "signal_directionality": "D3",
    "tone_compliance":       "D4",
    "format_compliance":     "D5",
}


def score_task(task: dict[str, Any], use_llm: bool = False) -> dict[str, Any]:
    """
    Score one Tenacious-Bench task and return the verdict object.

    Returns:
      {
        "verdict":          "PASS" | "REJECT",
        "failed_dimension": "D1" | "D2" | "D3" | "D4" | "D5" | null,
        "reason":           "one-line explanation",
        # per-dimension binary scores (1=pass, 0=fail):
        "grounding_fidelity":    int,
        "icp_pitch_alignment":   int,
        "signal_directionality": int,
        "tone_compliance":       int,
        "format_compliance":     int,
        # legacy key kept for backwards-compat:
        "overall_verdict":  "PASS" | "REJECT",
      }
    """
    brief = {
        **task["brief"],
        "_bench_summary": task.get("bench_summary", ""),
        "_prior_thread":  task.get("prior_thread", ""),
        "_rubric":        task.get("rubric", {}),
    }
    email = {**task["email"], "_rubric": task.get("rubric", {})}

    d1_score, d1_reason = score_grounding_fidelity(email, brief)
    d2_score, d2_reason = score_icp_pitch_alignment(brief, email, use_llm=use_llm)
    d3_score, d3_reason = score_signal_directionality(brief, email)
    d4_score, d4_reason = score_tone_compliance(email, brief)
    d5_score, d5_reason = score_format(email)

    dim_scores = {
        "grounding_fidelity":    d1_score,
        "icp_pitch_alignment":   d2_score,
        "signal_directionality": d3_score,
        "tone_compliance":       d4_score,
        "format_compliance":     d5_score,
    }
    dim_reasons = {
        "grounding_fidelity":    d1_reason,
        "icp_pitch_alignment":   d2_reason,
        "signal_directionality": d3_reason,
        "tone_compliance":       d4_reason,
        "format_compliance":     d5_reason,
    }

    verdict = "PASS" if all(v == 1 for v in dim_scores.values()) else "REJECT"

    failed_dimension: str | None = None
    reason = "All dimensions passed"
    for dim_name, score in dim_scores.items():
        if score == 0:
            failed_dimension = _DIM_CODES[dim_name]
            reason = dim_reasons[dim_name]
            break  # report first failure

    return {
        "verdict":          verdict,
        "failed_dimension": failed_dimension,
        "reason":           reason,
        # per-dimension scores
        **dim_scores,
        # legacy key
        "overall_verdict":  verdict,
    }


# ---------------------------------------------------------------------------
# Rubric builder (used by generation scripts)
# ---------------------------------------------------------------------------

def _base_rubric() -> dict[str, Any]:
    return {
        "dimensions": list(_DIM_CODES.keys()),
        **DEFAULT_RUBRIC,
    }


# ---------------------------------------------------------------------------
# Dummy smoke-test tasks
# ---------------------------------------------------------------------------

def dummy_tasks() -> list[dict[str, Any]]:
    """Three local smoke-test tasks. Not dataset examples."""
    rubric = _base_rubric()
    return [
        # TB-DUMMY-001: should PASS all dimensions
        {
            "task_id": "TB-DUMMY-001",
            "brief": {
                "company": "Arcana Analytics",
                "icp_segment": "Segment 1",
                "confidence": 0.9,
                "hiring_velocity": {
                    "direction": "accelerating",
                    "delta_pct": 100.0,
                    "signal_strength": "strong",
                    "observation": "Arcana Analytics has doubled its open job postings in the last 60 days.",
                },
                "budget_urgency": {"level": "high", "signal": "Series A $14M closed March 2026"},
                "grounding_facts": [
                    "Arcana Analytics has doubled its open job postings in the last 60 days.",
                    "Series A $14M closed March 2026",
                ],
                "bench_match": {"required_stacks": ["ml", "python"], "bench_available": True},
            },
            "email": {
                "subject": "Context: Series A $14M closed March 2026",
                "body": (
                    "Jordan,\n\n"
                    "Arcana Analytics has doubled its open job postings in the last 60 days.\n\n"
                    "Companies that close a Series A often face bottlenecks integrating new ML engineers.\n\n"
                    "Tenacious provides pre-vetted engineers ready to support Python and ML delivery.\n\n"
                    "What challenges are you encountering scaling your ML engineering team?\n\n"
                    "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
                ),
            },
            "prior_thread": "",
            "bench_summary": "ML and Python engineers are available.",
            "rubric": rubric,
        },
        # TB-DUMMY-002: should REJECT on D2 (Ambiguous + product claim) and D3 (neg velocity + growth frame)
        {
            "task_id": "TB-DUMMY-002",
            "brief": {
                "company": "SnapTrade",
                "icp_segment": "Ambiguous",
                "confidence": 0.7,
                "hiring_velocity": {
                    "direction": "decelerating",
                    "delta_pct": -60.0,
                    "signal_strength": "moderate",
                    "observation": "Job postings have decreased in the last 60 days.",
                },
                "budget_urgency": {"level": "low", "signal": "Seed round $3.2M in 2021"},
                "grounding_facts": [
                    "Job postings have decreased 60% in the last 60 days.",
                    "Seed round $3.2M in 2021",
                ],
                "bench_match": {"required_stacks": ["python", "aws"], "bench_available": True},
            },
            "email": {
                "subject": "Context: Job postings decreased 60%",
                "body": (
                    "SnapTrade Contact,\n\n"
                    "Job postings have decreased 60% in the last 60 days.\n\n"
                    "Companies in your position often face bottlenecks integrating new APIs.\n\n"
                    "Tenacious provides engineers who can augment your team and accelerate API development.\n\n"
                    "What are your biggest challenges in maintaining API integrations?\n\n"
                    "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
                ),
            },
            "prior_thread": "",
            "bench_summary": "Python and AWS engineers are available.",
            "rubric": rubric,
        },
        # TB-DUMMY-003: should REJECT on D4 (world-class, top talent), D5 (bad prefix, URL), D1 (wrong numeric)
        {
            "task_id": "TB-DUMMY-003",
            "brief": {
                "company": "PulseSight",
                "icp_segment": "Segment 1",
                "confidence": 0.85,
                "hiring_velocity": {
                    "direction": "accelerating",
                    "delta_pct": 133.33,
                    "signal_strength": "strong",
                    "observation": "PulseSight's open job postings have increased significantly.",
                },
                "budget_urgency": {"level": "high", "signal": "Series A $9M"},
                "grounding_facts": [
                    "PulseSight's open job postings have increased significantly.",
                    "Series A $9M",
                ],
                "bench_match": {"required_stacks": ["python", "infra"], "bench_available": True},
            },
            "email": {
                "subject": "Quick note about world-class hiring support for PulseSight",
                "body": (
                    "Hey there,\n\n"
                    "PulseSight raised $99M and is clearly falling behind competitors.\n\n"
                    "Our world-class top talent can fix that quickly. "
                    "Book a 15 minutes call at https://cal.com/example.\n\n"
                    "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
                ),
            },
            "prior_thread": "",
            "bench_summary": "Python and infrastructure engineers are available.",
            "rubric": rubric,
        },
    ]


def _run_dummy_tasks(use_llm: bool = False) -> None:
    expected_verdicts = {
        "TB-DUMMY-001": "PASS",
        "TB-DUMMY-002": "REJECT",
        "TB-DUMMY-003": "REJECT",
    }
    all_ok = True
    for task in dummy_tasks():
        result = score_task(task, use_llm=use_llm)
        print(json.dumps({"task_id": task["task_id"], **result}, indent=2))
        exp = expected_verdicts[task["task_id"]]
        if result["verdict"] != exp:
            print(f"  !! EXPECTED {exp}, GOT {result['verdict']}", file=sys.stderr)
            all_ok = False
    if all_ok:
        print("\nAll smoke-test tasks matched expected verdicts.")
    else:
        raise SystemExit("One or more smoke-test tasks produced unexpected verdicts.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _score_directory(dir_path: Path, use_llm: bool, output_name: str = "scored_tasks.jsonl") -> None:
    """Score every task in every *.jsonl file under dir_path and write results."""
    jsonl_files = sorted(dir_path.glob("*.jsonl"))
    if not jsonl_files:
        raise SystemExit(f"No *.jsonl files found in {dir_path}")

    output_path = dir_path / output_name
    total = passed = failed = errors = 0

    with output_path.open("w", encoding="utf-8") as out_fh:
        for jf in jsonl_files:
            if jf.name == output_name:
                continue
            for line in jf.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    task = json.loads(line)
                    result = score_task(task, use_llm=use_llm)
                    record = {"task_id": task.get("task_id", "?"), **result}
                    out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total += 1
                    if result["verdict"] == "PASS":
                        passed += 1
                    else:
                        failed += 1
                except Exception as exc:
                    errors += 1
                    print(f"  ERROR on task: {exc}", file=sys.stderr)

    print(f"Scored {total} tasks -> {output_path}")
    print(f"  PASS: {passed}  REJECT: {failed}  Errors: {errors}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tenacious-Bench deterministic scorer")
    parser.add_argument(
        "target", nargs="?", default=None,
        help="Path to a task JSON file or a directory of JSONL files. "
             "Omit to run smoke-test dummy tasks.",
    )
    parser.add_argument(
        "--llm-judge", action="store_true",
        help="Enable D2 LLM judge via OpenRouter (requires OPENROUTER_API_KEY).",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Force deterministic-only mode even if OPENROUTER_API_KEY is set.",
    )
    args = parser.parse_args()

    use_llm = args.llm_judge and not args.no_llm

    if args.target is None:
        _run_dummy_tasks(use_llm=use_llm)
    else:
        target = Path(args.target)
        if target.is_dir():
            _score_directory(target, use_llm=use_llm)
        elif target.is_file():
            task = json.loads(target.read_text(encoding="utf-8"))
            print(json.dumps(score_task(task, use_llm=use_llm), indent=2))
        else:
            raise SystemExit(f"Path not found: {target}")
