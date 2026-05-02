"""scripts/generation/programmatic_templates.py

Act II — Batch 2 (programmatic template tasks).

This script generates 75 deterministic Tenacious-Bench tasks:
  - 25 parameter combos
  - 3 variants per combo (A/B/C)

No LLM calls are made. Every task is validated against the JSON schema and
scored using the deterministic evaluator.

Variant semantics (aligned with judge_filter.py expectations):
  - A: intended PASS task
  - B: intended D1 failure (numeric corruption) with all other dims passing
  - C: intended D2 failure (Ambiguous brief + product claim) with all other dims passing

Usage (from repo root):
  python scripts/generation/programmatic_templates.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))

from scoring_evaluator import (  # type: ignore
    DEFAULT_RUBRIC,
    NUMERIC_TOKEN_RE,
    _evidence_text,
    score_task,
)


OUTPUT_FILE = ROOT / "data" / "tenacious_bench_v0.1" / "dev" / "programmatic_batch1.jsonl"
SCHEMA_PATH = ROOT / "benchmark" / "schema.json"


BENCH_SUMMARY = (
    "Python: 7 available (7-day deploy). Data: 9 available (7-day deploy). "
    "ML: 5 available (10-day deploy). Go: 3 available (14-day deploy). "
    "Infra: 4 available (14-day deploy). Frontend: 6 available (7-day deploy). "
    "NestJS: constrained through Q3 2026 (Modo Compass engagement — do not commit)."
)


@dataclass(frozen=True)
class Combo:
    idx: int
    segment: str
    delta_pct: float
    ai_maturity: int
    bench_available: bool
    company: str
    contact: str
    key_numeric: str


COMBO_TABLE: list[Combo] = [
    Combo(1,  "Segment 1",  120, 2, True,  "Velox Systems",   "Marcus",  "$28M, 120%"),
    Combo(2,  "Segment 1",   60, 1, True,  "Praxis AI",       "Lena",    "$16M, 60%"),
    Combo(3,  "Segment 1",    0, 3, True,  "Navix Tech",      "Chris",   "$11M"),
    Combo(4,  "Segment 1",  -40, 0, False, "Kinetra Health",  "Morgan",  "$9M, 40%"),
    Combo(5,  "Segment 1",  -80, 2, True,  "Fortica Labs",    "Jamie",   "$18M, 80%"),
    Combo(6,  "Segment 2",  120, 1, True,  "Crestline Ops",   "Taylor",  "120%"),
    Combo(7,  "Segment 2",   60, 0, True,  "Meridian Ops",    "Sarah",   "60%"),
    Combo(8,  "Segment 2",    0, 2, False, "Fulcrum Systems", "Devon",   "Q4 2025"),
    Combo(9,  "Segment 2",  -40, 3, True,  "Northgate Corp",  "Reed",    "40%"),
    Combo(10, "Segment 2",  -80, 1, True,  "Altera Group",    "Blair",   "80%"),
    Combo(11, "Segment 3",  120, 2, True,  "Corvex Networks", "Dana",    "120%"),
    Combo(12, "Segment 3",   60, 0, True,  "Zephyr Tech",     "Quinn",   "60%"),
    Combo(13, "Segment 3",    0, 3, True,  "Helia Software",  "River",   "Q4 2025"),
    Combo(14, "Segment 3",  -40, 1, False, "Axon Platforms",  "Casey",   "$14M, 40%"),
    Combo(15, "Segment 3",  -80, 2, True,  "Pragma Tech",     "Sydney",  "80%"),
    Combo(16, "Segment 4",  120, 3, True,  "Luminar AI",      "Priya",   "$55M, 120%"),
    Combo(17, "Segment 4",   60, 2, True,  "Synapse Labs",    "Kenji",   "$32M, 60%"),
    Combo(18, "Segment 4",    0, 1, True,  "Koda Systems",    "Avery",   "$4M"),
    Combo(19, "Segment 4",  -40, 0, True,  "Vertex AI Inc",   "Sage",    "40%"),
    Combo(20, "Segment 4",  -80, 3, False, "Arcturus ML",     "Jordan",  "$22M, 80%"),
    Combo(21, "Ambiguous",  120, 1, True,  "Optivex",         "Alex",    "120%"),
    Combo(22, "Ambiguous",   60, 0, True,  "Dualpath Inc",    "Sam",     "60%"),
    Combo(23, "Ambiguous",    0, 2, True,  "Solaris Core",    "Robin",   "Q4 2025"),
    Combo(24, "Ambiguous",  -40, 3, True,  "Paravox",         "Skyler",  "40%"),
    Combo(25, "Ambiguous",  -80, 1, True,  "Nivo Systems",    "Cam",     "80%"),
]


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


def _velocity_direction(delta_pct: float) -> str:
    if delta_pct > 20:
        return "accelerating"
    if delta_pct < -20:
        return "decelerating"
    if delta_pct == 0:
        return "stable"
    return "unknown"


def _signal_strength(delta_pct: float) -> str:
    if abs(delta_pct) >= 80:
        return "strong"
    if abs(delta_pct) >= 40:
        return "moderate"
    return "weak"


def _velocity_fact(company: str, delta_pct: float) -> str:
    if delta_pct > 0:
        return f"{company} has increased open roles by {int(delta_pct)}% in the last 60 days."
    if delta_pct < 0:
        return f"{company} has reduced open roles by {abs(int(delta_pct))}% in the last 60 days."
    return f"{company} has held open roles steady (0% change) in the last 60 days."


def build_brief(
    combo: Combo,
    icp_override: str | None = None,
    bench_available_override: bool | None = None,
) -> dict[str, Any]:
    icp_segment = icp_override or combo.segment
    bench_available = combo.bench_available if bench_available_override is None else bench_available_override

    confidence = 0.85
    if str(icp_segment).strip().lower() == "ambiguous":
        confidence = 0.65

    velocity_fact = _velocity_fact(combo.company, combo.delta_pct)

    # Segment-anchored event strings. These are intentionally simple but verifiable.
    if combo.segment == "Segment 1":
        event_fact = f"Funding context: Series round noted at {combo.key_numeric} (public)."
        pitch = "growth/post-funding"
    elif combo.segment == "Segment 2":
        event_fact = "Cost context: operating plan pressure noted in Q1 2026 (public)."
        pitch = "cost-discipline"
    elif combo.segment == "Segment 3":
        event_fact = "Leadership context: engineering leadership change noted in March 2026 (public)."
        pitch = "leadership transition"
    elif combo.segment == "Segment 4":
        event_fact = "Capability context: peer AI/ML hiring activity noted in Q1 2026 (public)."
        pitch = "capability gap"
    else:
        event_fact = "ICP context: public signals are mixed; ICP is unclear."
        pitch = "qualify"

    brief: dict[str, Any] = {
        "company": combo.company,
        "icp_segment": icp_segment,
        "confidence": confidence,
        "hiring_velocity": {
            "direction": _velocity_direction(combo.delta_pct),
            "delta_pct": float(combo.delta_pct),
            "signal_strength": _signal_strength(combo.delta_pct),
            "observation": velocity_fact,
        },
        "grounding_facts": [event_fact, velocity_fact],
        "ai_maturity_score": int(combo.ai_maturity),
        "recommended_pitch_angle": pitch,
        "bench_match": {
            "required_stacks": ["python", "data", "ml", "infra"],
            "bench_available": bool(bench_available),
        },
        "honesty_flags": {
            "weak_hiring_velocity_signal": _signal_strength(combo.delta_pct) == "weak",
            "bench_gap_detected": not bool(bench_available),
        },
    }

    # Ensure we embed combo.key_numeric somewhere in the brief so the Task B corruption
    # ladder always has an eligible numeric grounding token.
    brief["budget_urgency"] = {"level": "unknown", "signal": f"Key numeric: {combo.key_numeric}"}
    return brief


def _signature() -> str:
    return (
        "Birkity\n"
        "Research Partner, Tenacious Intelligence Corporation\n"
        "gettenacious.com"
    )


def _make_email(subject: str, paragraphs: list[str]) -> dict[str, Any]:
    body = "\n\n".join(paragraphs + [_signature()])
    return {
        "subject": subject,
        "body": body,
        "word_count": len(body.split()),
        "tone_warnings": [],
    }


def build_seg1_email(combo: Combo) -> dict[str, Any]:
    velocity_line = _velocity_fact(combo.company, combo.delta_pct)

    if combo.delta_pct >= -20:
        subject = f"Congrats on {combo.company} context"
        paragraphs = [
            f"{combo.contact},",
            velocity_line,
            "Companies scaling post-funding often need engineers faster than traditional hiring.",
        ]
        if combo.bench_available:
            paragraphs.append("Tenacious provides pre-vetted ML engineers deployable in days.")
        paragraphs.append("What engineering priorities are you scaling into next quarter?")
        return _make_email(subject, paragraphs)

    # delta < -20: no growth-frame terms.
    if not combo.bench_available:
        subject = f"Context: engineering shift at {combo.company}"
        paragraphs = [
            f"{combo.contact},",
            velocity_line,
            "Post-funding teams often consolidate delivery on fewer, higher-priority workstreams.",
            "What engineering initiatives are you prioritizing to make that capital go further?",
        ]
        return _make_email(subject, paragraphs)

    subject = f"Note on {combo.company} engineering context"
    paragraphs = [
        f"{combo.contact},",
        velocity_line,
        "Post-funding teams often consolidate delivery during headcount pauses.",
        "Tenacious provides engineers on defined scopes to cover delivery gaps during those pauses.",
        "What delivery commitments are at risk this quarter?",
    ]
    return _make_email(subject, paragraphs)


def build_seg2_email(combo: Combo) -> dict[str, Any]:
    velocity_line = _velocity_fact(combo.company, combo.delta_pct)
    subject = f"Note on engineering continuity at {combo.company}"
    paragraphs = [
        f"{combo.contact},",
        velocity_line,
        "Mid-market teams often need reliable delivery coverage without long-term headcount commitments.",
    ]
    if combo.bench_available:
        paragraphs.append("Tenacious provides engineers with defined scopes and predictable costs.")
    paragraphs.append("What delivery risks are top of mind for your team this quarter?")
    return _make_email(subject, paragraphs)


def build_seg3_email(combo: Combo) -> dict[str, Any]:
    velocity_line = _velocity_fact(combo.company, combo.delta_pct)
    subject = f"Context: vendor strategy at {combo.company}"
    paragraphs = [
        f"{combo.contact},",
        velocity_line,
        "Leadership transitions often trigger a vendor reassessment in the first 90 days.",
    ]
    if combo.bench_available:
        paragraphs.append("Tenacious provides engineers for defined project scopes during transitions.")
    paragraphs.append("Which delivery commitments are most critical to maintain right now?")
    return _make_email(subject, paragraphs)


def build_seg4_email(combo: Combo) -> dict[str, Any]:
    velocity_line = _velocity_fact(combo.company, combo.delta_pct)
    subject = f"Context: ML capability at {combo.company}"
    paragraphs = [
        f"{combo.contact},",
        velocity_line,
        "At your stage, the challenge is matching specialized engineers to your architecture, not volume hiring.",
    ]
    if combo.bench_available:
        paragraphs.append("Tenacious provides pre-vetted ML engineers for high-specificity gaps.")
    paragraphs.append("Which ML capability gaps are most critical for your roadmap right now?")
    return _make_email(subject, paragraphs)


def build_ambiguous_email(combo: Combo) -> dict[str, Any]:
    velocity_line = _velocity_fact(combo.company, combo.delta_pct)
    subject = f"Question on engineering priorities at {combo.company}"
    paragraphs = [
        f"{combo.contact},",
        velocity_line,
        "Before making a specific recommendation, I want to understand your current priorities.",
        "Are you primarily focused on reducing costs, increasing capacity, or building a new technical capability?",
    ]
    return _make_email(subject, paragraphs)


def build_task_a_email(combo: Combo) -> dict[str, Any]:
    if combo.segment == "Segment 1":
        return build_seg1_email(combo)
    if combo.segment == "Segment 2":
        return build_seg2_email(combo)
    if combo.segment == "Segment 3":
        return build_seg3_email(combo)
    if combo.segment == "Segment 4":
        return build_seg4_email(combo)
    return build_ambiguous_email(combo)


def _pick_absent_replacement(evidence: str, replacement: str) -> str:
    # Extremely defensive: if a replacement collides with evidence, deterministically choose a nearby value.
    if replacement.lower() not in evidence:
        return replacement
    if replacement.startswith("$"):
        for amt in ("$98M", "$97M", "$96M"):
            if amt.lower() not in evidence:
                return amt
    if replacement.endswith("%"):
        for pct in ("199%", "201%", "198%"):
            if pct.lower() not in evidence:
                return pct
    if replacement.isdigit() and len(replacement) == 4:
        for yr in ("2018", "2017", "2020"):
            if yr.lower() not in evidence:
                return yr
    return replacement


def corrupt_numeric(body: str, brief: dict[str, Any]) -> str:
    evidence = _evidence_text(brief)
    tokens = NUMERIC_TOKEN_RE.findall(body)

    eligible: list[str] = []
    for tok in tokens:
        if tok and tok.lower() in evidence:
            eligible.append(tok)

    def _priority(tok: str) -> int:
        t = tok.strip()
        if t.startswith("$"):
            return 0
        if t.endswith("%"):
            return 1
        if t.lower().startswith("q"):
            return 2
        if t.isdigit() and len(t) == 4:
            return 3
        return 9

    eligible.sort(key=_priority)

    for tok in eligible:
        if tok.startswith("$"):
            replacement = _pick_absent_replacement(evidence, "$99M")
            return body.replace(tok, replacement, 1)
        if tok.endswith("%"):
            replacement = _pick_absent_replacement(evidence, "200%")
            return body.replace(tok, replacement, 1)
        # year or quarter
        replacement = _pick_absent_replacement(evidence, "2019")
        return body.replace(tok, replacement, 1)

    # Fallback: inject an obviously wrong but absent numeric.
    injected = "(Public filings note a recent round of $99M.) "
    if injected.lower() in evidence:
        injected = "(Public filings note a recent round of $98M.) "
    return injected + body


def inject_product_claim(body: str) -> str:
    # Insert the product claim before the final question paragraph.
    marker = "\n\nWhat"
    if marker in body:
        return body.replace(marker, "\n\nTenacious provides pre-vetted engineers for defined scopes.\n\nWhat", 1)
    # If we can't find the expected structure, prepend (still safe).
    return "Tenacious provides pre-vetted engineers for defined scopes.\n\n" + body


def build_task_c(combo: Combo, brief_a: dict[str, Any], email_a: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if combo.segment != "Ambiguous":
        # Flip to Ambiguous and ensure bench_available True to avoid D1 bench/product-claim trap.
        brief_c = build_brief(combo, icp_override="Ambiguous", bench_available_override=True)
        email_c = dict(email_a)
        if not combo.bench_available:
            # If A had no product claim, inject one so D2 reliably fails.
            email_c = {**email_c, "body": inject_product_claim(email_c["body"])}
        return brief_c, email_c

    # Ambiguous combos: keep brief; mutate email to add product claim while keeping the qualifying question.
    brief_c = dict(brief_a)
    body = email_a["body"]
    # Insert product claim before the qualifying question (the only '?').
    body = body.replace(
        "Before making a specific recommendation, I want to understand your current priorities.",
        "Tenacious provides pre-vetted engineers ready to deploy.\n\nBefore making a specific recommendation, I want to understand your current priorities.",
        1,
    )
    email_c = {**email_a, "body": body}
    return brief_c, email_c


def make_task(task_id: str, brief: dict[str, Any], email: dict[str, Any]) -> dict[str, Any]:
    task = {
        "task_id": task_id,
        "brief": brief,
        "email": email,
        "prior_thread": "",
        "bench_summary": BENCH_SUMMARY,
        "rubric": _base_rubric(),
    }
    return task


def assign_difficulty(task: dict[str, Any], scores: dict[str, Any]) -> str:
    variant = task["task_id"][-1]
    segment = str(task["brief"].get("icp_segment", ""))

    if variant == "C":
        return "hard"
    if variant == "B":
        return "medium"
    if scores["overall_verdict"] == "REJECT":
        return "hard"
    if segment.strip().lower() == "ambiguous":
        return "medium"
    return "easy"


def verify_all(tasks: list[dict[str, Any]], schema: dict[str, Any]) -> list[str]:
    import jsonschema

    errors: list[str] = []

    if len(tasks) != 75:
        errors.append(f"Expected 75 tasks, got {len(tasks)}")

    # Basic counts
    a = [t for t in tasks if t["task_id"].endswith("A")]
    b = [t for t in tasks if t["task_id"].endswith("B")]
    c = [t for t in tasks if t["task_id"].endswith("C")]
    if (len(a), len(b), len(c)) != (25, 25, 25):
        errors.append(f"Expected 25/25/25 variants, got {len(a)}/{len(b)}/{len(c)}")

    for t in tasks:
        tid = t["task_id"]
        try:
            jsonschema.validate(instance=t, schema=schema)
        except jsonschema.ValidationError as e:
            errors.append(f"{tid}: schema: {e.message}")
            continue

        # Global constraints
        subject = t["email"]["subject"]
        body = t["email"]["body"]
        if len(subject) > 60:
            errors.append(f"{tid}: subject too long ({len(subject)})")
        if len(body.split()) > 120:
            errors.append(f"{tid}: body too long ({len(body.split())} words)")
        if body.count("?") != 1:
            errors.append(f"{tid}: expected exactly 1 '?', got {body.count('?')}")
        if "bench" in (subject + " " + body).lower():
            errors.append(f"{tid}: contains forbidden word 'bench'")
        if "nestjs" in (subject + " " + body).lower():
            errors.append(f"{tid}: contains unexpected 'NestJS' mention")

        scores = score_task(t)
        variant = tid[-1]
        if variant == "A":
            if scores["overall_verdict"] != "PASS":
                errors.append(f"{tid}: expected PASS, got {scores}")
        elif variant == "B":
            # D1 should be the only failure.
            if not (
                scores["grounding_fidelity"] == 0
                and scores["icp_pitch_alignment"] == 1
                and scores["signal_directionality"] == 1
                and scores["tone_compliance"] == 1
                and scores["format_compliance"] == 1
            ):
                errors.append(f"{tid}: expected only grounding_fidelity=0, got {scores}")
        elif variant == "C":
            # D2 should be the only failure.
            if not (
                scores["icp_pitch_alignment"] == 0
                and scores["grounding_fidelity"] == 1
                and scores["signal_directionality"] == 1
                and scores["tone_compliance"] == 1
                and scores["format_compliance"] == 1
            ):
                errors.append(f"{tid}: expected only icp_pitch_alignment=0, got {scores}")

    return errors


def main() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    tasks: list[dict[str, Any]] = []

    for combo in COMBO_TABLE:
        base_id = f"TB-PROG-{combo.idx:03d}"

        brief_a = build_brief(combo)
        email_a = build_task_a_email(combo)
        email_a = {**email_a, "icp_segment_used": combo.segment}
        task_a = make_task(base_id + "A", brief_a, email_a)

        # B: numeric corruption in body
        email_b = {**email_a, "body": corrupt_numeric(email_a["body"], brief_a)}
        task_b = make_task(base_id + "B", brief_a, email_b)

        # C: D2 isolation
        brief_c, email_c = build_task_c(combo, brief_a, email_a)
        task_c = make_task(base_id + "C", brief_c, email_c)

        tasks.extend([task_a, task_b, task_c])

    # Assign difficulty deterministically (same logic as judge_filter.py)
    for t in tasks:
        scores = score_task(t)
        t["difficulty"] = assign_difficulty(t, scores)

    errors = verify_all(tasks, schema)
    if errors:
        print("Programmatic batch generation FAILED verification:")
        for e in errors:
            print("  *", e)
        raise SystemExit(1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"Wrote {len(tasks)} tasks -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
