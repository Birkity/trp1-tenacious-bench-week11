"""
Trace-derived task generator — Act II, Days 2-3.

Produces 75 tasks (25 events × 3 variants A/B/C) from 5 real company
hiring_signal_brief.json files plus LLM-synthesised emails.

Variants per event:
  A — original email as-is (PASS or REJECT depending on content)
  B — same email with one numeric fact corrupted → D1 (grounding_fidelity) = 0
  C — Ambiguous brief + product-claim email → D2 (icp_pitch_alignment) = 0

Models (via OpenRouter, dev-tier only):
  Primary : google/gemini-2.5-flash-preview
  Fallback : openai/gpt-4o-mini

Usage (from project root):
    python scripts/generation/trace_derived.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import jsonschema
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).parents[2]

SCHEMA_PATH = ROOT / "benchmark" / "schema.json"
EVAL_PATH = ROOT / "benchmark" / "scoring_evaluator.py"
TRACES_DIR = ROOT / "evidence" / "traces"
ARTIFACTS_DIR = ROOT / "evidence" / "artifacts"
OUTPUT_DIR = ROOT / "data" / "tenacious_bench_v0.1" / "dev"
OUTPUT_FILE = OUTPUT_DIR / "trace_derived_batch1.jsonl"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PRIMARY_MODEL = "google/gemini-2.5-flash"
FALLBACK_MODEL = "openai/gpt-4o-mini"

BENCH_SUMMARY = (
    "Python: 7 available (7-day deploy). "
    "Data: 9 available (7-day deploy). "
    "ML: 5 available (10-day deploy). "
    "Go: 3 available (14-day deploy). "
    "Infra: 4 available (14-day deploy). "
    "Frontend: 6 available (7-day deploy). "
    "NestJS: constrained through Q3 2026 (Modo Compass engagement — do not commit)."
)

# 5 companies with real evidence; each supplies 5 events (real + synthetic).
COMPANIES = [
    {"slug": "arcana",        "target_events": 5},
    {"slug": "pulsesight",    "target_events": 5},
    {"slug": "streamlineops", "target_events": 5},
    {"slug": "snaptrade",     "target_events": 5},
    {"slug": "wiseitech",     "target_events": 5},
]

# Must match scoring_evaluator.py DEFAULT_RUBRIC exactly.
DEFAULT_RUBRIC: dict[str, Any] = {
    "dimensions": [
        "grounding_fidelity",
        "icp_pitch_alignment",
        "signal_directionality",
        "tone_compliance",
        "format_compliance",
    ],
    "negative_velocity_threshold_pct": -20.0,
    "max_subject_chars": 60,
    "max_body_words": 120,
    "approved_subject_prefixes": ["context:", "note on", "congrats on", "question on"],
    "banned_phrases": [
        "top talent", "world-class", "a-players", "rockstar", "ninja",
        "aggressive hiring", "cost savings of", "guaranteed roi",
        "proven track record", "falling behind", "you're behind", "you lack",
        "you're missing", "you need to catch up", "left behind", "hope this finds",
        "hey there", "quick ", "just ",
    ],
    "growth_frame_terms": [
        "bottleneck", "bottlenecks", "scaling", "scale your", "accelerate",
        "accelerating", "rapid growth", "increased demand", "augment your team",
        "augment existing", "expand the team", "staff up",
    ],
    "product_claim_terms": [
        "tenacious can", "tenacious provides", "tenacious offers",
        "we can", "we provide", "our engineers", "available on-demand",
        "pre-vetted", "project-ready", "augment your", "deploy in days",
    ],
    "icp_pitch_alignment_policy": (
        "Later LLM judge checks whether the primary pitch frame matches the "
        "brief ICP segment. Phase 1 only fast-fails Ambiguous plus product claim."
    ),
}

NUMERIC_RE = re.compile(
    r"\$[\d,.]+[A-Za-z]*|\b\d+(?:\.\d+)?%|\b\d{4}\b|\bQ[1-4]\s+\d{4}\b",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[A-Za-z0-9$%]+")


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_SYSTEM = """You are a Tenacious Intelligence Corporation sales researcher.
Write cold outreach emails following these non-negotiable rules:
- Subject: ≤60 characters, must start with exactly one of: "Context:", "Note on", "Congrats on", "Question on"
- Body: ≤120 words
- Exactly one "?" in the entire email
- No URLs ("https://", "cal.com", etc.)
- No meeting phrases ("schedule a", "book a", "15 minutes", "book time")
- No banned phrases: "top talent", "world-class", "a-players", "rockstar", "ninja",
  "aggressive hiring", "hope this finds", "hey there", "quick ", "just ", "falling behind",
  "you're behind", "you lack", "you need to catch up", "left behind"
- Never use "bench" — use "engineering team" or "available capacity" instead
- Signature must end exactly with:
  Birkity
  Research Partner, Tenacious Intelligence Corporation
  gettenacious.com

Respond with ONLY a JSON object with keys "subject" and "body"."""

_PASS_SEG1_PROMPT = """\
Write a PASS cold email for {company} (Segment 1 — recently funded startup).

Brief:
{brief_json}

Requirements:
- Reference at least one grounding fact from brief.grounding_facts verbatim or closely paraphrased
- Include a product claim using one of: "Tenacious provides", "Tenacious can", "pre-vetted engineers"
- Pitch frame: growth / post-funding / hiring velocity angle
- Hiring delta_pct = {delta_pct}% — use growth language only if delta >= -20

Output: JSON {{"subject": "...", "body": "..."}}"""

_QUALIFYING_PROMPT = """\
Write a PASS qualifying cold email for {company} (Ambiguous segment — unclear ICP).

Brief:
{brief_json}

CRITICAL rules:
- Do NOT include any product claim: no "Tenacious provides", "Tenacious can", "we provide",
  "pre-vetted", "available on-demand", "our engineers", "augment your", "deploy in days"
- Ask exactly one qualifying question — do not pitch services or bench capacity
- Reference at least one grounding fact from brief.grounding_facts in the body
- No growth-frame terms: "bottleneck", "scaling", "accelerate", "augment your team",
  "augment existing", "expand the team", "staff up"
- This email must score PASS on all 5 dimensions

Output: JSON {{"subject": "...", "body": "..."}}"""

_D2_FAIL_PROMPT = """\
Write a D2-FAIL test email for {company} (Ambiguous segment — delta_pct = {delta_pct}%).

Brief:
{brief_json}

Requirements:
- MUST include a product claim (triggers D2 fail): "Tenacious provides" or "Tenacious can"
- MUST reference at least one grounding fact from brief.grounding_facts (D1 must pass)
- MUST NOT use growth-frame terms — avoid ALL of:
  "bottleneck", "bottlenecks", "scaling", "scale your", "accelerate", "accelerating",
  "rapid growth", "increased demand", "augment your team", "augment existing",
  "expand the team", "staff up"
  (D3 must pass even though delta_pct = {delta_pct}%)
- No banned phrases (D4 passes), correct format (D5 passes)
- Expected scores: D1=1, D2=0, D3=1, D4=1, D5=1

Output: JSON {{"subject": "...", "body": "..."}}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def load_brief(slug: str) -> dict:
    return json.loads((TRACES_DIR / slug / "hiring_signal_brief.json").read_text(encoding="utf-8"))


def load_real_emails(slug: str) -> list[dict]:
    path = ARTIFACTS_DIR / slug / "email_log.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def normalize_segment(seg: str) -> str:
    s = str(seg).strip()
    return "Ambiguous" if s.lower() == "ambiguous" else s


def make_client() -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set in .env")
    return OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)


def brief_for_task(raw: dict, icp_override: str | None = None) -> dict:
    """Build a schema-compliant brief dict from a raw hiring_signal_brief.json."""
    vel = raw.get("hiring_velocity", {})
    budget = raw.get("budget_urgency", {})

    grounding_facts: list[str] = []
    if budget.get("signal"):
        grounding_facts.append(str(budget["signal"]))
    if vel.get("observation"):
        grounding_facts.append(str(vel["observation"]))
    if not grounding_facts:
        grounding_facts = ["Signal data derived from public job-posting activity."]

    segment = normalize_segment(raw.get("icp_segment", "Ambiguous"))
    if icp_override:
        segment = normalize_segment(icp_override)

    brief: dict[str, Any] = {
        "company": raw["company"],
        "icp_segment": segment,
        "confidence": float(raw.get("confidence", 0.7)),
        "hiring_velocity": {
            "direction": vel.get("direction", "unknown"),
            "delta_pct": float(vel.get("delta_pct", 0)),
            "signal_strength": vel.get("signal_strength", "unknown"),
            "observation": str(vel.get("observation", "")),
        },
        "grounding_facts": grounding_facts,
    }
    for opt in ("budget_urgency", "bench_match", "engineering_maturity", "honesty_flags"):
        if raw.get(opt):
            brief[opt] = raw[opt]
    if raw.get("ai_maturity_score") is not None:
        brief["ai_maturity_score"] = int(raw["ai_maturity_score"])
    return brief


def _evidence_text(brief: dict) -> str:
    chunks = [json.dumps(brief, sort_keys=True)]
    delta = brief.get("hiring_velocity", {}).get("delta_pct")
    if isinstance(delta, (int, float)):
        chunks += [f"{abs(delta):g}%", f"{abs(round(delta)):g}%", f"{delta:g}%"]
    return re.sub(r"\s+", " ", " ".join(chunks)).lower().strip()


def corrupt_numeric(body: str, brief: dict) -> str:
    """
    Return a Task-B body: grounding-fact tokens still match (so D1 check-1 passes)
    but one numeric token in the body does not appear in evidence (D1 check-2 fails).
    """
    evidence = _evidence_text(brief)

    # Try to replace an existing numeric that IS in evidence
    for token in NUMERIC_RE.findall(body):
        tn = token.lower().strip()
        if tn in evidence:
            if token.startswith("$"):
                wrong = "$99M"
                if "$99m" in evidence:
                    wrong = "$500K"
            elif "%" in token:
                wrong = "200%"
                if "200%" in evidence:
                    wrong = "99%"
            else:  # 4-digit year
                wrong = "2019"
                if "2019" in evidence:
                    wrong = "2015"
            if wrong.lower() not in evidence:
                return body.replace(token, wrong, 1)

    # No matching numeric — inject a wrong dollar amount
    wrong_amount = "$99M"
    if "$99m" in evidence:
        wrong_amount = "$500K"
    if "$500k" in evidence:
        wrong_amount = "$77M"

    # Insert as a parenthetical in the first body paragraph (after first double-newline)
    parts = body.split("\n\n", 2)
    if len(parts) >= 2:
        inject = f"(Public filings note a recent round of {wrong_amount}.)"
        parts[1] = inject + " " + parts[1]
        return "\n\n".join(parts)
    return body.rstrip() + f"\n\n(Public filings note a recent round of {wrong_amount}.)"


def call_llm(client: OpenAI, prompt: str, model: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=700,
            )
            raw = resp.choices[0].message.content or ""
            # Try direct JSON parse
            try:
                data = json.loads(raw)
                if "subject" in data and "body" in data:
                    return data
            except json.JSONDecodeError:
                pass
            # Extract JSON block from prose
            m = re.search(r"\{[^{}]*\"subject\"[^{}]*\"body\"[^{}]*\}", raw, re.DOTALL)
            if not m:
                m = re.search(r"\{.*?\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                if "subject" in data and "body" in data:
                    return data
            raise ValueError(f"No valid JSON in LLM response:\n{raw[:300]}")
        except Exception as exc:
            if attempt < retries - 1:
                print(f"    -> attempt {attempt + 1} failed ({exc.__class__.__name__}): {exc}")
                time.sleep(2)
            else:
                raise
    raise RuntimeError("LLM generation failed")


def validate_email_format(subject: str, body: str) -> list[str]:
    """Return list of format issues (empty = OK)."""
    issues = []
    if len(subject) > 60:
        issues.append(f"subject too long ({len(subject)} chars)")
    prefixes = ["context:", "note on", "congrats on", "question on"]
    if not any(subject.lower().startswith(p) for p in prefixes):
        issues.append(f"subject lacks approved prefix: {subject!r}")
    if len(body.split()) > 120:
        issues.append(f"body too long ({len(body.split())} words)")
    if re.search(r"https?://|cal\.com", body, re.I):
        issues.append("body contains URL")
    if body.count("?") != 1:
        issues.append(f"body has {body.count('?')} question marks (need exactly 1)")
    for phrase in ("schedule a", "book a", "15 minutes", "book time"):
        if phrase in body.lower():
            issues.append(f"meeting phrase in body: {phrase!r}")
    return issues


def generate_email(
    client: OpenAI,
    model: str,
    prompt: str,
    label: str,
    max_attempts: int = 4,
) -> dict:
    for attempt in range(max_attempts):
        result = call_llm(client, prompt, model)
        issues = validate_email_format(result["subject"], result["body"])
        if not issues:
            return result
        print(f"    -> {label} attempt {attempt + 1}: format issues {issues}")
        if attempt < max_attempts - 1:
            # Ask the model to fix
            fix_note = "Fix these issues: " + "; ".join(issues)
            prompt = prompt + f"\n\nPREVIOUS ATTEMPT FAILED:\nsubject: {result['subject']}\nbody: {result['body']}\n\n{fix_note}"
    raise RuntimeError(f"Could not generate valid {label} email after {max_attempts} attempts")


# ---------------------------------------------------------------------------
# Task builders
# ---------------------------------------------------------------------------

def make_task(
    event_id: int,
    variant: str,
    brief: dict,
    email: dict,
    source: str,
) -> dict:
    body = email.get("body", "")
    return {
        "task_id": f"TB-TRACE-{event_id:03d}{variant}",
        "brief": brief,
        "email": {
            "subject": email.get("subject", ""),
            "body": body,
            "word_count": len(WORD_RE.findall(body)),
            "icp_segment_used": email.get("icp_segment_used", brief["icp_segment"]),
            "tone_warnings": email.get("tone_warnings", []),
        },
        "prior_thread": "",
        "bench_summary": BENCH_SUMMARY,
        "rubric": dict(DEFAULT_RUBRIC),
        # _meta stripped before output and schema validation
        "_meta": {"variant": variant, "source": source},
    }


# ---------------------------------------------------------------------------
# Per-company generation
# ---------------------------------------------------------------------------

def process_company(
    slug: str,
    target_events: int,
    event_start: int,
    schema: dict,
    client: OpenAI,
    model: str,
) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"  {slug.upper()}  events {event_start}–{event_start + target_events - 1}")
    print(f"{'='*60}")

    raw_brief = load_brief(slug)
    real_emails = load_real_emails(slug)
    segment = normalize_segment(raw_brief.get("icp_segment", "Ambiguous"))
    is_ambiguous = segment.lower() == "ambiguous"
    delta_pct = float(raw_brief.get("hiring_velocity", {}).get("delta_pct", 0))
    company = raw_brief["company"]

    # ------------------------------------------------------------------
    # Collect Task-A source emails
    # ------------------------------------------------------------------
    source_emails: list[dict] = []
    for em in real_emails[:target_events]:
        source_emails.append({**em, "_src": "trace_real"})

    needed = target_events - len(source_emails)
    for i in range(needed):
        lbl = f"{slug} synthetic {i + 1}/{needed}"
        print(f"  Generating {lbl}...")
        b = brief_for_task(raw_brief)
        if is_ambiguous:
            prompt = _QUALIFYING_PROMPT.format(
                company=company,
                brief_json=json.dumps(b, indent=2),
            )
        else:
            prompt = _PASS_SEG1_PROMPT.format(
                company=company,
                brief_json=json.dumps(b, indent=2),
                delta_pct=delta_pct,
            )
        llm = generate_email(client, model, prompt, lbl)
        source_emails.append({
            "subject": llm["subject"],
            "body": llm["body"],
            "icp_segment_used": segment,
            "tone_warnings": [],
            "_src": "llm_synthetic",
        })

    # ------------------------------------------------------------------
    # Build A/B/C triads
    # ------------------------------------------------------------------
    all_tasks: list[dict] = []

    for idx, src in enumerate(source_emails):
        eid = event_start + idx
        src_label = src.get("_src", "trace")

        # ---- Task A ------------------------------------------------
        brief_a = brief_for_task(raw_brief)
        email_a = {
            "subject": src.get("subject", ""),
            "body": src.get("body", ""),
            "icp_segment_used": src.get("icp_segment_used", segment),
            "tone_warnings": src.get("tone_warnings", []),
        }
        task_a = make_task(eid, "A", brief_a, email_a, src_label)

        # ---- Task B (D1 corrupt) ------------------------------------
        brief_b = brief_for_task(raw_brief)  # identical to A
        body_b = corrupt_numeric(email_a["body"], brief_b)
        email_b = {**email_a, "body": body_b}
        task_b = make_task(eid, "B", brief_b, email_b, src_label + "_d1corrupt")

        # ---- Task C (D2 fail) ---------------------------------------
        if not is_ambiguous:
            # Segment 1/2/3/4: change brief to Ambiguous, keep email (has product claim)
            brief_c = brief_for_task(raw_brief, icp_override="Ambiguous")
            email_c = {**email_a, "icp_segment_used": "Ambiguous"}
            task_c = make_task(eid, "C", brief_c, email_c, src_label + "_d2fail_briefswap")
        else:
            # Ambiguous: generate a clean D2-fail email (product claim, no growth frame)
            lbl_c = f"{slug} event {eid} Task-C D2-fail"
            print(f"  Generating {lbl_c}...")
            brief_c = brief_for_task(raw_brief)  # stays Ambiguous
            prompt_c = _D2_FAIL_PROMPT.format(
                company=company,
                brief_json=json.dumps(brief_c, indent=2),
                delta_pct=delta_pct,
            )
            llm_c = generate_email(client, model, prompt_c, lbl_c)
            email_c = {
                "subject": llm_c["subject"],
                "body": llm_c["body"],
                "icp_segment_used": "Ambiguous",
                "tone_warnings": [],
            }
            task_c = make_task(eid, "C", brief_c, email_c, "llm_d2fail")

        all_tasks.extend([task_a, task_b, task_c])

    return all_tasks


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _score(task: dict) -> dict:
    clean = {k: v for k, v in task.items() if not k.startswith("_")}
    sys.path.insert(0, str(ROOT / "benchmark"))
    from scoring_evaluator import score_task  # type: ignore
    return score_task(clean)


def verify_all(tasks: list[dict], schema: dict) -> list[str]:
    errors: list[str] = []

    for task in tasks:
        tid = task["task_id"]
        variant = task.get("_meta", {}).get("variant", "?")
        clean = {k: v for k, v in task.items() if not k.startswith("_")}

        # Schema
        try:
            jsonschema.validate(instance=clean, schema=schema)
        except jsonschema.ValidationError as e:
            errors.append(f"{tid}: schema — {e.message}")
            continue

        # Scoring assertions
        result = _score(task)

        if variant == "B":
            if result["grounding_fidelity"] != 0:
                errors.append(
                    f"{tid}: expected D1=0 but got D1={result['grounding_fidelity']}. "
                    f"Body: {clean['email']['body'][:120]!r}"
                )

        elif variant == "C":
            if result["icp_pitch_alignment"] != 0:
                errors.append(
                    f"{tid}: expected D2=0 but got D2={result['icp_pitch_alignment']}. "
                    f"Segment={clean['brief']['icp_segment']!r}  "
                    f"Body: {clean['email']['body'][:120]!r}"
                )

    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    schema = load_schema()
    client = make_client()
    model = PRIMARY_MODEL
    print(f"Model: {model}")

    all_tasks: list[dict] = []
    event_id = 1

    for cfg in COMPANIES:
        batch = process_company(
            slug=cfg["slug"],
            target_events=cfg["target_events"],
            event_start=event_id,
            schema=schema,
            client=client,
            model=model,
        )
        all_tasks.extend(batch)
        event_id += cfg["target_events"]

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  VERIFICATION")
    print(f"{'='*60}")
    errors = verify_all(all_tasks, schema)

    if errors:
        print(f"\n✗ {len(errors)} verification error(s):\n")
        for e in errors:
            print(f"  • {e}")
        raise SystemExit("Fix errors before writing output.")

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        for task in all_tasks:
            clean = {k: v for k, v in task.items() if not k.startswith("_")}
            fh.write(json.dumps(clean, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    a_tasks = [t for t in all_tasks if t.get("_meta", {}).get("variant") == "A"]
    b_tasks = [t for t in all_tasks if t.get("_meta", {}).get("variant") == "B"]
    c_tasks = [t for t in all_tasks if t.get("_meta", {}).get("variant") == "C"]

    a_pass = sum(1 for t in a_tasks if _score(t)["overall_verdict"] == "PASS")
    a_rej = len(a_tasks) - a_pass
    b_d1_ok = sum(1 for t in b_tasks if _score(t)["grounding_fidelity"] == 0)
    c_d2_ok = sum(1 for t in c_tasks if _score(t)["icp_pitch_alignment"] == 0)

    print(f"\n{'='*60}")
    print("  BATCH 1 SUMMARY")
    print(f"{'='*60}")
    print(f"Total tasks written : {len(all_tasks)}")
    print(f"Schema validation   : {len(all_tasks)}/{len(all_tasks)} ✓")
    print(f"Task A (original)   : {len(a_tasks)} tasks  [PASS: {a_pass}, REJECT: {a_rej}]")
    print(f"Task B (D1 fail)    : D1=0 on {b_d1_ok}/{len(b_tasks)} ✓")
    print(f"Task C (D2 fail)    : D2=0 on {c_d2_ok}/{len(c_tasks)} ✓")
    print(f"\nOutput → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
