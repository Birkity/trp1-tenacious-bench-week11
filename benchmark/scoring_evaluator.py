"""
Tenacious-Bench Phase 1 scoring evaluator.

No model calls are made here. ICP-pitch alignment has an LLM-judge placeholder
with a deterministic fast-fail for the most obvious Week 10 failure:
Ambiguous segment plus a product/capacity claim.

Usage:
    python week11/scoring_evaluator.py
    python week11/scoring_evaluator.py path/to/task.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


DEFAULT_RUBRIC: dict[str, Any] = {
    "negative_velocity_threshold_pct": -20.0,
    "max_subject_chars": 60,
    "max_body_words": 120,
    "approved_subject_prefixes": [
        "context:",
        "note on",
        "congrats on",
        "question on",
    ],
    "banned_phrases": [
        "top talent",
        "world-class",
        "a-players",
        "rockstar",
        "ninja",
        "aggressive hiring",
        "cost savings of",
        "guaranteed roi",
        "proven track record",
        "falling behind",
        "you're behind",
        "you lack",
        "you're missing",
        "you need to catch up",
        "left behind",
        "hope this finds",
        "hey there",
        "quick ",
        "just ",
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
}

NUMERIC_TOKEN_RE = re.compile(
    r"\$[\d,.]+[A-Za-z]*|\b\d+(?:\.\d+)?%|\b\d{4}\b|\bQ[1-4]\s+\d{4}\b",
    re.IGNORECASE,
)
URL_RE = re.compile(r"https?://|cal\.com", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z0-9$%]+")


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
                if value not in (None, ""):
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

        # Allow compact paraphrases by requiring several meaningful fact tokens.
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


def score_grounding_fidelity(email: dict[str, Any], brief: dict[str, Any]) -> int:
    """Return 1 when the email's concrete claims are supported by the brief."""
    rubric = _rubric(brief)
    body = email.get("body", "")
    if not _contains_grounding_fact(body, brief):
        return 0

    evidence = _evidence_text(brief)
    numeric_tokens = NUMERIC_TOKEN_RE.findall(body)
    for token in numeric_tokens:
        token_norm = _normalize(token)
        if token_norm not in evidence:
            return 0

    bench = brief.get("bench_match", {})
    unavailable = [s.lower() for s in bench.get("unavailable_stacks", [])]
    bench_available = bench.get("bench_available", True)
    body_lower = body.lower()
    if bench_available is False and any(term in body_lower for term in rubric["product_claim_terms"]):
        return 0
    if unavailable and any(stack in body_lower for stack in unavailable):
        if any(term in body_lower for term in ("available", "can provide", "can staff", "deploy")):
            return 0

    return 1


def score_signal_directionality(brief: dict[str, Any], email: dict[str, Any]) -> int:
    """Return 0 when negative hiring velocity is paired with growth-frame copy."""
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
    has_growth_frame = any(term.lower() in body for term in growth_terms)

    if delta_value < threshold and has_growth_frame:
        return 0
    return 1


def score_tone_compliance(email: dict[str, Any]) -> int:
    """Return 1 when no Tenacious style-guide banned phrase or jargon appears."""
    rubric = _rubric(email)
    text = _normalize(f"{email.get('subject', '')} {email.get('body', '')}")
    banned = rubric["banned_phrases"]
    if any(phrase.lower() in text for phrase in banned):
        return 0

    # "bench" is internal jargon in prospect-facing copy.
    if re.search(r"\bbench\b", text):
        return 0
    return 1


def score_format(email: dict[str, Any]) -> int:
    """Return 1 when the email follows cold-outreach format constraints."""
    rubric = _rubric(email)
    subject = email.get("subject", "")
    body = email.get("body", "")
    subject_lower = subject.lower()

    if len(subject) > rubric["max_subject_chars"]:
        return 0
    if not any(subject_lower.startswith(prefix) for prefix in rubric["approved_subject_prefixes"]):
        return 0
    if len(body.split()) > rubric["max_body_words"]:
        return 0
    if URL_RE.search(body):
        return 0
    if body.count("?") > 1:
        return 0

    meeting_phrases = ("schedule a", "book a", "15 minutes", "book time")
    if any(phrase in body.lower() for phrase in meeting_phrases):
        return 0

    return 1


def score_icp_pitch_alignment(brief: dict[str, Any], email: dict[str, Any]) -> int:
    """
    Placeholder for the later LLM-based ICP-pitch judge.

    Phase 1 keeps this deterministic and conservative: it only fails the
    confirmed obvious Week 10 error where Ambiguous receives a product claim.
    All subtler segment-frame judgments should be handled by the future judge.
    """
    segment = str(brief.get("icp_segment", "")).strip().lower()
    body = email.get("body", "").lower()
    rubric = _rubric(brief)

    if segment == "ambiguous":
        if any(term.lower() in body for term in rubric["product_claim_terms"]):
            return 0
    return 1


def score_task(task: dict[str, Any]) -> dict[str, Any]:
    """Score one Tenacious-Bench task and return the required verdict object."""
    brief = {
        **task["brief"],
        "_bench_summary": task.get("bench_summary", ""),
        "_prior_thread": task.get("prior_thread", ""),
        "_rubric": task.get("rubric", {}),
    }
    email = {**task["email"], "_rubric": task.get("rubric", {})}

    result = {
        "grounding_fidelity": score_grounding_fidelity(email, brief),
        "signal_directionality": score_signal_directionality(brief, email),
        "tone_compliance": score_tone_compliance(email),
        "format_compliance": score_format(email),
        "icp_pitch_alignment": score_icp_pitch_alignment(brief, email),
    }
    result["overall_verdict"] = "PASS" if all(value == 1 for value in result.values()) else "REJECT"
    return result


def _base_rubric() -> dict[str, Any]:
    return {
        "dimensions": [
            "grounding_fidelity",
            "icp_pitch_alignment",
            "signal_directionality",
            "tone_compliance",
            "format_compliance",
        ],
        **DEFAULT_RUBRIC,
        "icp_pitch_alignment_policy": (
            "Later LLM judge checks whether the primary pitch frame matches the "
            "brief ICP segment. Phase 1 only fast-fails Ambiguous plus product claim."
        ),
    }


def dummy_tasks() -> list[dict[str, Any]]:
    """Three local smoke-test tasks. These are not dataset examples."""
    rubric = _base_rubric()
    return [
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
                "budget_urgency": {
                    "level": "high",
                    "signal": "Series A $14M closed March 2026",
                },
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
                    "Companies scaling after a Series A often face bottlenecks integrating new ML engineers.\n\n"
                    "Tenacious provides pre-vetted engineers ready to support Python and ML delivery.\n\n"
                    "What challenges are you encountering scaling your ML engineering team?\n\n"
                    "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
                ),
            },
            "prior_thread": "",
            "bench_summary": "ML and Python engineers are available.",
            "rubric": rubric,
        },
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
                "budget_urgency": {
                    "level": "low",
                    "signal": "Seed round $3.2M in 2021",
                },
                "grounding_facts": [
                    "Job postings have decreased 60% in the last 60 days.",
                    "Seed round $3.2M in 2021",
                ],
                "bench_match": {"required_stacks": ["python", "aws"], "bench_available": True},
            },
            "email": {
                "subject": "Context: Job postings decreased -60%",
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
                    "observation": "PulseSight's open job postings have increased significantly in the last 60 days.",
                },
                "budget_urgency": {
                    "level": "high",
                    "signal": "Series A $9M",
                },
                "grounding_facts": [
                    "PulseSight's open job postings have increased significantly in the last 60 days.",
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


def _run_dummy_tasks() -> None:
    expected = {
        "TB-DUMMY-001": "PASS",
        "TB-DUMMY-002": "REJECT",
        "TB-DUMMY-003": "REJECT",
    }
    for task in dummy_tasks():
        result = score_task(task)
        print(json.dumps({"task_id": task["task_id"], **result}, indent=2))
        if result["overall_verdict"] != expected[task["task_id"]]:
            raise SystemExit(f"{task['task_id']} expected {expected[task['task_id']]}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        task = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(score_task(task), indent=2))
    else:
        _run_dummy_tasks()
