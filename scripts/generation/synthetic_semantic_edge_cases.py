"""scripts/generation/synthetic_semantic_edge_cases.py

Semantic edge cases (LLM-generated, evaluator-pass, semantically wrong).

Goal:
- Generate 60 tasks where D1/D2/D3/D4/D5 all PASS,
  but the email is semantically unjustified given the brief.

Pipeline:
1) Generate structured briefs programmatically (no trace usage).
2) Use a small OpenRouter model to draft emails that are *technically compliant*
   but logically over-claim.
3) Filter with scoring_evaluator.py; keep only tasks with overall PASS.

Output:
- data/tenacious_bench_v0.1/dev_synthetic/semantic_edge_cases_batch1.jsonl

Requires:
- OPENROUTER_API_KEY in .env
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).parents[2]

load_dotenv()

SCHEMA_PATH = ROOT / "benchmark" / "schema.json"
OUT_DIR = ROOT / "data" / "tenacious_bench_v0.1" / "dev_synthetic"
OUT_FILE = OUT_DIR / "semantic_edge_cases_batch1.jsonl"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v3.2"  # small/cheap model tier on OpenRouter

BENCH_SUMMARY = (
    "Python: 7 available (7-day deploy). "
    "Data: 9 available (7-day deploy). "
    "ML: 5 available (10-day deploy). "
    "Go: 3 available (14-day deploy). "
    "Infra: 4 available (14-day deploy). "
    "Frontend: 6 available (7-day deploy). "
    "NestJS: constrained through Q3 2026 (Modo Compass engagement — do not commit)."
)

APPROVED_PREFIXES = ("context:", "note on", "congrats on", "question on")


@dataclass(frozen=True)
class Params:
    company: str
    icp_segment: str
    confidence: float
    velocity_delta: int
    ai_maturity_score: int
    bench_available: bool
    gap_count: int


SEMANTIC_FAILURES = [
    "overclaim_growth_from_weak_signal",
    "infer_ai_maturity_from_single_signal",
    "claim_capability_gap_when_none",
    "overstate_confidence",
    "assert_strong_need_from_ambiguous_signal",
]


def make_client() -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set in .env")
    return OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)


def _direction(delta: int) -> str:
    if delta > 20:
        return "accelerating"
    if delta < -20:
        return "decelerating"
    if delta == 0:
        return "stable"
    return "unknown"


def _strength(delta: int) -> str:
    if abs(delta) >= 80:
        return "strong"
    if abs(delta) >= 40:
        return "moderate"
    return "weak"


def build_brief(p: Params) -> dict[str, Any]:
    observation = (
        f"{p.company} open role postings changed by {p.velocity_delta}% over the last 60 days."
    )
    gaps = [] if p.gap_count == 0 else [f"gap-{i+1}" for i in range(p.gap_count)]

    brief: dict[str, Any] = {
        "company": p.company,
        "icp_segment": p.icp_segment,
        "confidence": p.confidence,
        "hiring_velocity": {
            "direction": _direction(p.velocity_delta),
            "delta_pct": float(p.velocity_delta),
            "signal_strength": _strength(p.velocity_delta),
            "observation": observation,
        },
        "grounding_facts": [
            observation,
            f"AI maturity score: {p.ai_maturity_score}.",
            f"Signal confidence: {p.confidence}.",
        ],
        "ai_maturity_score": p.ai_maturity_score,
        "bench_match": {
            "required_stacks": ["python", "data", "ml"],
            "bench_available": p.bench_available,
            "unavailable_stacks": [],
        },
        "capability_gaps": gaps,
    }
    return brief


def build_prompt(brief: dict[str, Any], semantic_type: str) -> str:
    brief_json = json.dumps(brief, ensure_ascii=False, indent=2)

    semantic_instructions = {
        "overclaim_growth_from_weak_signal": (
            "Make a strong growth/bottleneck claim even though the hiring delta is small."
        ),
        "infer_ai_maturity_from_single_signal": (
            "Claim advanced AI maturity even though the brief only shows a low score."
        ),
        "claim_capability_gap_when_none": (
            "Assert a capability gap even though capability_gaps is an empty list."
        ),
        "overstate_confidence": (
            "Use language of certainty despite low signal confidence."
        ),
        "assert_strong_need_from_ambiguous_signal": (
            "Make a strong need claim despite the signal being ambiguous or weak."
        ),
    }[semantic_type]

    return f"""
You are generating adversarial benchmark tasks for Tenacious-Bench semantic edge cases.

Your goal is to produce emails that PASS all programmatic checks (D1–D5)
but are still semantically incorrect when compared to the hiring brief.

This means:
- All numbers in the email must appear in the brief (D1 pass)
- Pitch frame must appear to match the ICP segment (D2 superficial pass)
- Tone must avoid all banned phrases (D4 pass)
- Word count <= 120 and no booking links (D5 pass)

But the email must make an inference or claim that the brief does NOT justify.

Examples of semantic failures:
- Inferring a scaling bottleneck from +10% growth (too weak evidence)
- Claiming AI maturity from a single ML job post
- Claiming capability gaps when gaps[] is empty
- Overstating confidence when brief confidence = 0.5
- Turning ambiguous signals into strong claims

Specific instruction for this task:
- {semantic_instructions}

Use this brief JSON exactly (do not edit values):
{brief_json}

Output JSON with:
- brief_json (exactly as provided)
- email_subject
- email_body
- explanation_of_why_this_is_semantically_wrong
""".strip()


def parse_llm_json(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
    return None


def validate_email_format(subject: str, body: str) -> list[str]:
    issues: list[str] = []
    if len(subject) > 60:
        issues.append("subject too long")
    if not any(subject.lower().startswith(p) for p in APPROVED_PREFIXES):
        issues.append("subject missing approved prefix")
    if len(body.split()) > 120:
        issues.append("body too long")
    if re.search(r"https?://|cal\.com", body, re.I):
        issues.append("body contains URL")
    if body.count("?") > 1:
        issues.append("more than one question mark")
    for phrase in ("schedule a", "book a", "15 minutes", "book time"):
        if phrase in body.lower():
            issues.append(f"meeting phrase: {phrase}")
    if re.search(r"\bbench\b", body.lower()):
        issues.append("uses 'bench'")
    return issues


def call_llm(client: OpenAI, model: str, prompt: str, retries: int = 3) -> dict[str, Any]:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise JSON generator."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.6,
                max_tokens=900,
            )
            raw = resp.choices[0].message.content or ""
            data = parse_llm_json(raw)
            if not data:
                raise ValueError("No valid JSON in model response")
            return data
        except Exception as exc:
            if attempt < retries - 1:
                print(f"    -> attempt {attempt + 1} failed: {exc}")
                time.sleep(2)
            else:
                raise
    raise RuntimeError("LLM call failed")


def make_task(task_id: str, brief: dict[str, Any], email: dict[str, Any]) -> dict[str, Any]:
    body = email["body"]
    return {
        "task_id": task_id,
        "brief": brief,
        "email": {
            "subject": email["subject"],
            "body": body,
            "word_count": len(body.split()),
            "tone_warnings": [],
        },
        "prior_thread": "",
        "bench_summary": BENCH_SUMMARY,
        "rubric": {
            "dimensions": [
                "grounding_fidelity",
                "icp_pitch_alignment",
                "signal_directionality",
                "tone_compliance",
                "format_compliance",
            ],
        },
    }


def build_params_pool(seed: int) -> list[Params]:
    random.seed(seed)
    companies = [
        "Arcana Systems",
        "PulseSight",
        "SnapTrade",
        "WiseiTech",
        "StreamlineOps",
        "Helix Cloud",
        "Atlas Ledger",
        "Nimbus Forge",
        "QuantaLoop",
        "CivicStack",
    ]
    deltas = [5, 10, 15, 20, 40, -5, -10, -15]
    confidences = [0.5, 0.65, 0.8]
    segments = ["Segment 1", "Segment 2", "Segment 3", "Segment 4"]
    pool = [
        Params(
            company=company,
            icp_segment=seg,
            confidence=conf,
            velocity_delta=delta,
            ai_maturity_score=1 if seg in ("Segment 1", "Segment 2") else 2,
            bench_available=True,
            gap_count=0,
        )
        for company in companies
        for delta in deltas
        for conf in confidences
        for seg in segments
    ]
    random.shuffle(pool)
    return pool


def _attempt_candidate(
    client: OpenAI,
    model: str,
    brief: dict[str, Any],
    semantic_type: str,
) -> tuple[dict[str, Any], str, str, str] | None:
    prompt = build_prompt(brief, semantic_type)
    data = call_llm(client, model, prompt)

    brief_out = data.get("brief_json")
    subject = data.get("email_subject") or ""
    body = data.get("email_body") or ""
    explanation = data.get("explanation_of_why_this_is_semantically_wrong") or ""

    try:
        echoed = json.loads(brief_out) if isinstance(brief_out, str) else brief_out
    except Exception:
        echoed = None
    if echoed != brief:
        return None

    if validate_email_format(subject, body):
        return None

    if not explanation.strip():
        return None

    return brief, subject, body, explanation.strip()


def main() -> None:
    import sys
    sys.path.insert(0, str(ROOT / "benchmark"))
    from scoring_evaluator import score_task  # type: ignore

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=60)
    parser.add_argument("--max-candidates", type=int, default=240)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds between API calls")
    args = parser.parse_args()

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    client = make_client()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    params_pool = build_params_pool(args.seed)
    tasks: list[dict[str, Any]] = []
    attempts = 0

    while len(tasks) < args.target and attempts < args.max_candidates:
        attempts += 1
        p = params_pool[attempts % len(params_pool)]
        semantic_type = SEMANTIC_FAILURES[attempts % len(SEMANTIC_FAILURES)]
        brief = build_brief(p)

        candidate = _attempt_candidate(client, args.model, brief, semantic_type)
        if not candidate:
            print(f"    -> skip: invalid candidate on attempt {attempts}")
            continue

        brief, subject, body, explanation = candidate

        # Store semantic annotations inside the brief (schema allows extra props).
        brief_enriched = dict(brief)
        brief_enriched["semantic_error_type"] = semantic_type
        brief_enriched["semantic_error_reason"] = explanation.strip()
        brief_enriched["source_mode"] = "synthetic_semantic_edge_cases"

        task_id = f"TB-SEM-{len(tasks)+1:03d}"
        task = make_task(task_id, brief_enriched, {"subject": subject, "body": body})

        scores = score_task(task)
        if scores["overall_verdict"] != "PASS":
            print(f"    -> skip: evaluator rejected {task_id} {scores}")
            continue

        jsonschema.validate(instance=task, schema=schema)
        tasks.append(task)

        if args.sleep:
            time.sleep(args.sleep)

        if len(tasks) % 10 == 0:
            print(f"Collected {len(tasks)} / {args.target} tasks...")

    if len(tasks) < args.target:
        raise SystemExit(
            f"Only collected {len(tasks)} tasks after {attempts} attempts. "
            "Increase --max-candidates or adjust prompts."
        )

    with OUT_FILE.open("w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"Wrote {len(tasks)} tasks -> {OUT_FILE}")


if __name__ == "__main__":
    main()
