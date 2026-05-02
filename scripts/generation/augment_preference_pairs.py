"""
augment_preference_pairs.py  —  Expand training pairs from 114 to ~250.

Three phases:

  Phase A  PASS multi-negative augmentation  (+86 pairs, 0 API calls)
           Each of the 43 PASS tasks gets 2 additional rejected responses
           drawn from a 12-template pool (5 original + 7 new from style-guide
           BAD-example taxonomy). Rotation is deterministic via md5 hash.

  Phase B  Style-guide scaffolded pairs  (+20 pairs, gpt-4o-mini)
           10 GOOD-style + 10 BAD-style scenarios generated with briefs
           anchored to Tenacious style-guide segment/signal taxonomy.
           score_task() verifies every generated task before inclusion.

  Phase C  Category coverage pairs  (+30 pairs, gpt-4o-mini)
           Targets underrepresented dimensions in train:
             D3 (signal_directionality): 3 in train  → +15
             D4 (tone_compliance):       5 in train  → +15
           score_task() verifies every generated task before inclusion.

Output:
  training_data/tenacious_judge_train_v2.jsonl   (~250 pairs)

Usage:
  python scripts/generation/augment_preference_pairs.py
  python scripts/generation/augment_preference_pairs.py --dry-run
  python scripts/generation/augment_preference_pairs.py --phases A    # Phase A only, no API
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))
from scoring_evaluator import score_task  # noqa: E402

TRAIN_FILE  = ROOT / "data" / "tenacious_bench_v0.1" / "train" / "tasks.jsonl"
BASE_FILE   = ROOT / "training_data" / "tenacious_judge_train.jsonl"
OUTPUT_FILE = ROOT / "training_data" / "tenacious_judge_train_v2.jsonl"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL           = "openai/gpt-4o-mini"

DIM_NAMES = {
    "D1": "grounding_fidelity",
    "D2": "icp_pitch_alignment",
    "D3": "signal_directionality",
    "D4": "tone_compliance",
    "D5": "format_compliance",
}

BENCH_SUMMARY = (
    "Python: 7 available (7-day deploy). "
    "Data: 9 available (7-day deploy). "
    "ML: 5 available (10-day deploy). "
    "Go: 3 available (14-day deploy). "
    "Infra: 4 available (14-day deploy). "
    "Frontend: 6 available (7-day deploy). "
    "NestJS: constrained through Q3 2026."
)

# ── Judge prompt (matches build_preference_pairs.py exactly) ──────────────────
_JUDGE_PROMPT = (
    "You are a sales quality judge for Tenacious Intelligence Corporation.\n"
    "Given a hiring signal brief and a generated outreach email,\n"
    "return a structured verdict.\n"
    "\n"
    "Score these dimensions:\n"
    "grounding_fidelity, icp_pitch_alignment,\n"
    "signal_directionality, tone_compliance, format_compliance.\n"
    "\n"
    "Hiring Brief:\n"
    "{brief_text}\n"
    "\n"
    "Generated Email:\n"
    "Subject: {subject}\n"
    "{body}\n"
    "\n"
    "Velocity delta: {delta:+.0f}%\n"
    "ICP segment: {segment}"
)


def _brief_text(task: dict) -> str:
    brief = task.get("brief", {})
    hv    = brief.get("hiring_velocity", {})
    bm    = brief.get("bench_match", {})
    facts = brief.get("grounding_facts", [])
    lines = [
        f"Company: {brief.get('company', 'Unknown')}",
        f"ICP Segment: {brief.get('icp_segment', 'Unknown')}",
        f"AI Maturity: {brief.get('ai_maturity', '?')}/3",
        (
            f"Hiring Velocity: {hv.get('direction', '?')} "
            f"({hv.get('delta_pct', 0):+.0f}% in 60 days, "
            f"{hv.get('signal_strength', '?')} signal)"
        ),
        f"Observation: {hv.get('observation', '').strip()}",
        f"Bench Available: {bm.get('bench_available', '?')}",
        "Grounding Facts:",
    ]
    for fact in facts:
        lines.append(f"  - {fact.strip()}")
    return "\n".join(lines)


def _build_judge_prompt(task: dict) -> str:
    brief = task.get("brief", {})
    email = task.get("email", {})
    hv    = brief.get("hiring_velocity", {})
    return _JUDGE_PROMPT.format(
        brief_text = _brief_text(task),
        subject    = email.get("subject", ""),
        body       = email.get("body", "").strip(),
        delta      = float(hv.get("delta_pct", 0)),
        segment    = brief.get("icp_segment", "Unknown"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase A — 12-template pool for PASS multi-negative augmentation
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (dimension_name, plausible-but-wrong reason string)
# Templates 0–4: original 5 from build_preference_pairs.py
# Templates 5–11: new, drawn from style-guide BAD-example taxonomy
_REJECT_TEMPLATES: list[tuple[str, str]] = [
    # ── original 5 ───────────────────────────────────────────────────────────
    (
        "grounding_fidelity",
        "A numeric reference in the email body cannot be traced to a grounding "
        "fact in the brief.",
    ),
    (
        "icp_pitch_alignment",
        "Pitch frame implies a growth posture inconsistent with the stated ICP "
        "segment context.",
    ),
    (
        "signal_directionality",
        "Email body uses acceleration language while the hiring velocity "
        "direction is ambiguous or declining.",
    ),
    (
        "tone_compliance",
        "Phrasing positions the prospect's situation as a failure, approaching "
        "a condescending frame.",
    ),
    (
        "format_compliance",
        "Subject line does not cleanly separate the approved prefix from the "
        "company reference within the character limit.",
    ),
    # ── new 7 from style-guide BAD-example taxonomy ───────────────────────────
    (
        "grounding_fidelity",
        "Funding event referenced in the email (amount or stage) does not match "
        "what the brief's grounding facts record.",
    ),
    (
        "icp_pitch_alignment",
        "Brief shows AI maturity score below 2 yet the email pitches a "
        "Segment 4 capability-gap frame — the maturity threshold for that "
        "framing is not met.",
    ),
    (
        "signal_directionality",
        "Email asserts strong hiring momentum using phrases such as 'scaling "
        "aggressively' or 'rapid growth', but the brief velocity is flat or "
        "declining.",
    ),
    (
        "tone_compliance",
        "Email body contains a phrase from the Tenacious banned-phrase list "
        "(e.g., 'top talent', 'world-class', 'synergize') that fails the "
        "Professional tone marker.",
    ),
    (
        "tone_compliance",
        "Email asserts strategic need at higher confidence than the brief "
        "signal supports — weak or medium-confidence signals require "
        "interrogative phrasing, not assertions.",
    ),
    (
        "format_compliance",
        "Email body contains more than one explicit question, violating the "
        "single-ask formatting constraint.",
    ),
    (
        "grounding_fidelity",
        "Email commits to specific capacity (engineer count or start timeline) "
        "that is not supported by the bench summary field in the brief.",
    ),
]


def _pick_template(task_id: str, seed_suffix: str, exclude_idx: int | None) -> tuple[str, str]:
    """Pick a template deterministically, avoiding exclude_idx."""
    n = len(_REJECT_TEMPLATES)
    for attempt in range(n):
        raw = f"{task_id}_{seed_suffix}_{attempt}"
        idx = int(hashlib.md5(raw.encode()).hexdigest(), 16) % n
        if idx != exclude_idx:
            return _REJECT_TEMPLATES[idx]
    return _REJECT_TEMPLATES[(exclude_idx + 1) % n]


def _original_template_idx(task_id: str) -> int:
    """Return the template index used by the original build_preference_pairs.py."""
    return int(hashlib.md5(task_id.encode()).hexdigest(), 16) % len(_REJECT_TEMPLATES)


# ─────────────────────────────────────────────────────────────────────────────
# Phase B — style-guide scaffolded generation prompts
# ─────────────────────────────────────────────────────────────────────────────

_STYLE_SCENARIOS: list[dict] = [
    # ── GOOD-style (should score PASS) ────────────────────────────────────────
    {
        "label": "good_seg1_funding_velocity",
        "expected": "PASS",
        "brief_seed": {
            "company": "Orinda Labs", "icp_segment": "Segment 1",
            "funding": "Series A $18M Jan 2026", "velocity_direction": "accelerating",
            "delta_pct": 180, "open_roles": "Python and data engineers tripled",
            "ai_maturity": 1, "bench_available": True,
        },
        "style_note": (
            "GOOD #1 pattern — Series A funding + strong role velocity. "
            "Ground the email in the exact funding amount and role-count trend. "
            "Use 'engineering team' not 'bench'. One ask: 15 minutes. "
            "Body <=120 words."
        ),
    },
    {
        "label": "good_seg2_layoff_cost",
        "expected": "PASS",
        "brief_seed": {
            "company": "Fulcrum Ops", "icp_segment": "Segment 2",
            "signal": "12% headcount reduction March 2026",
            "velocity_direction": "decelerating", "delta_pct": -45,
            "ai_maturity": 1, "bench_available": True,
        },
        "style_note": (
            "GOOD #2 pattern — post-layoff mid-market cost-discipline pitch. "
            "Acknowledge contraction respectfully, frame as common pattern. "
            "Use conditional language. No growth-frame terms. Body <=120 words."
        ),
    },
    {
        "label": "good_seg3_cto_transition",
        "expected": "PASS",
        "brief_seed": {
            "company": "Navex Systems", "icp_segment": "Segment 3",
            "signal": "new CTO hired Feb 2026",
            "velocity_direction": "stable", "delta_pct": 0,
            "ai_maturity": 2, "bench_available": True,
        },
        "style_note": (
            "GOOD #3 pattern — new engineering leader, 90-day vendor reassessment. "
            "Name the announcement date specifically. Lower the ask. "
            "Offer value-add, not a pitch. Body <=120 words."
        ),
    },
    {
        "label": "good_seg4_capability_gap",
        "expected": "PASS",
        "brief_seed": {
            "company": "Lumex AI", "icp_segment": "Segment 4",
            "signal": "3 peer companies posted MLOps roles, Lumex has none",
            "velocity_direction": "stable", "delta_pct": 5,
            "ai_maturity": 2, "bench_available": True,
        },
        "style_note": (
            "GOOD #4 pattern — capability gap, AI maturity 2, peer comparison. "
            "Frame gap as 'two readings', not a deficiency. Name specific peers. "
            "One ask: 15-minute walk-through. Body <=120 words."
        ),
    },
    {
        "label": "good_weak_signal_asks",
        "expected": "PASS",
        "brief_seed": {
            "company": "Kestrel Tech", "icp_segment": "Segment 1",
            "signal": "2 open data engineer roles, ambiguous demand",
            "velocity_direction": "stable", "delta_pct": 10,
            "ai_maturity": 0, "bench_available": True,
        },
        "style_note": (
            "GOOD #5 pattern — weak signal, asks rather than asserts. "
            "Say explicitly you cannot tell from outside. Use conditional framing. "
            "Give explicit out: 'if two roles is actual demand, ignore this'. "
            "Body <=120 words."
        ),
    },
    {
        "label": "good_seg1_declining_honest",
        "expected": "PASS",
        "brief_seed": {
            "company": "Driftwood Labs", "icp_segment": "Segment 1",
            "funding": "Series B $22M Oct 2025",
            "velocity_direction": "decelerating", "delta_pct": -30,
            "ai_maturity": 1, "bench_available": False,
        },
        "style_note": (
            "PASS pattern — declining velocity, no bench available. "
            "Do NOT pitch capacity (bench not available). "
            "No growth-frame terms (velocity is negative). "
            "Ask a qualifying question only. Body <=120 words."
        ),
    },
    {
        "label": "good_seg2_flat_cost_discipline",
        "expected": "PASS",
        "brief_seed": {
            "company": "Meridian Platforms", "icp_segment": "Segment 2",
            "signal": "hiring flat, cost optimisation underway Q1 2026",
            "velocity_direction": "stable", "delta_pct": -5,
            "ai_maturity": 1, "bench_available": True,
        },
        "style_note": (
            "PASS pattern — mid-market, flat velocity, cost-discipline pitch. "
            "No growth-frame terms. Conditional language. "
            "Product claim acceptable (bench available). Body <=120 words."
        ),
    },
    {
        "label": "good_ambiguous_qualifying",
        "expected": "PASS",
        "brief_seed": {
            "company": "Proxima Group", "icp_segment": "Ambiguous",
            "velocity_direction": "accelerating", "delta_pct": 40,
            "ai_maturity": 1, "bench_available": True,
        },
        "style_note": (
            "PASS pattern — Ambiguous ICP segment. "
            "MUST use qualifying-question only. NO product claim. "
            "Ask about cost reduction vs capacity scaling vs new capability. "
            "Body <=120 words."
        ),
    },
    {
        "label": "good_seg4_low_ai_reframe",
        "expected": "PASS",
        "brief_seed": {
            "company": "Vertex AI Labs", "icp_segment": "Segment 1",
            "funding": "Series A $9M March 2026",
            "velocity_direction": "stable", "delta_pct": 0,
            "ai_maturity": 1, "bench_available": True,
        },
        "style_note": (
            "GOOD #10 pattern — AI maturity 0-1, gentle Segment 1 reframe. "
            "Do NOT pitch Segment 4 capability gap (maturity too low). "
            "Frame AI as optional next step, not a gap. Conditional language. "
            "Body <=120 words."
        ),
    },
    {
        "label": "good_seg3_declining_vendor",
        "expected": "PASS",
        "brief_seed": {
            "company": "Crestline Corp", "icp_segment": "Segment 3",
            "signal": "VP Engineering departed Jan 2026, interim in place",
            "velocity_direction": "decelerating", "delta_pct": -20,
            "ai_maturity": 2, "bench_available": True,
        },
        "style_note": (
            "PASS pattern — leadership transition + declining velocity. "
            "Reference the departure/transition date. "
            "No growth-frame terms (velocity negative at threshold). "
            "Frame as vendor reassessment window. Body <=120 words."
        ),
    },
    # ── BAD-style (should score REJECT) ───────────────────────────────────────
    {
        "label": "bad_d4_banned_phrases",
        "expected": "REJECT",
        "expected_dim": "D4",
        "brief_seed": {
            "company": "Solaris Systems", "icp_segment": "Segment 1",
            "funding": "Series A $12M Feb 2026",
            "velocity_direction": "accelerating", "delta_pct": 90,
            "ai_maturity": 1, "bench_available": True,
        },
        "style_note": (
            "BAD #1/#6 pattern — include at least one banned phrase from the "
            "Tenacious banned list: 'top talent', 'world-class', 'rockstar', "
            "'synergize', or 'skyrocket'. Otherwise looks like a normal email. "
            "Body <=120 words."
        ),
    },
    {
        "label": "bad_d4_condescending",
        "expected": "REJECT",
        "expected_dim": "D4",
        "brief_seed": {
            "company": "Nexar Platforms", "icp_segment": "Segment 4",
            "signal": "3 peers posted MLOps roles, Nexar has none",
            "velocity_direction": "stable", "delta_pct": 8,
            "ai_maturity": 2, "bench_available": True,
        },
        "style_note": (
            "BAD #4 pattern — condescending competitor gap framing. "
            "Use phrases like 'falling behind', 'you lack', 'you need to catch up'. "
            "Frame the prospect's situation as a failure. "
            "Body <=120 words. Subject <=60 chars."
        ),
    },
    {
        "label": "bad_d3_wrong_direction_growth_claim",
        "expected": "REJECT",
        "expected_dim": "D3",
        "brief_seed": {
            "company": "Arcturus ML", "icp_segment": "Segment 1",
            "funding": "Series B $28M Nov 2025",
            "velocity_direction": "decelerating", "delta_pct": -40,
            "ai_maturity": 2, "bench_available": True,
        },
        "style_note": (
            "BAD signal-direction pattern — velocity is DECLINING (-40%) "
            "but the email uses growth-frame terms: 'scaling', 'bottleneck', "
            "'accelerate', 'expand the team'. "
            "Otherwise grounded correctly. Body <=120 words."
        ),
    },
    {
        "label": "bad_d3_aggressive_on_declining",
        "expected": "REJECT",
        "expected_dim": "D3",
        "brief_seed": {
            "company": "Kinetra Data", "icp_segment": "Segment 2",
            "signal": "hiring dropped 60% since Q4 2025",
            "velocity_direction": "decelerating", "delta_pct": -60,
            "ai_maturity": 1, "bench_available": True,
        },
        "style_note": (
            "BAD signal-direction pattern — velocity declining strongly. "
            "Email uses 'augment your team', 'scaling', 'accelerating growth'. "
            "These growth-frame terms are banned on negative velocity. "
            "Body <=120 words."
        ),
    },
    {
        "label": "bad_d4_fake_urgency",
        "expected": "REJECT",
        "expected_dim": "D4",
        "brief_seed": {
            "company": "Polaris Labs", "icp_segment": "Segment 1",
            "funding": "Series A $16M Feb 2026",
            "velocity_direction": "accelerating", "delta_pct": 70,
            "ai_maturity": 1, "bench_available": True,
        },
        "style_note": (
            "BAD #7 pattern — fake urgency. Include: 'last open slot', "
            "'strong demand', 'don't miss out', artificial deadline. "
            "Also include 'just ' or 'quick ' to trigger banned-phrase check. "
            "Body <=120 words."
        ),
    },
    {
        "label": "bad_d1_wrong_funding",
        "expected": "REJECT",
        "expected_dim": "D1",
        "brief_seed": {
            "company": "Veloris AI", "icp_segment": "Segment 1",
            "funding": "Series A $11M Jan 2026",
            "velocity_direction": "accelerating", "delta_pct": 120,
            "ai_maturity": 1, "bench_available": True,
        },
        "style_note": (
            "BAD #12 pattern — signal fabrication. "
            "Brief says Series A $11M but email says '$25M Series B' or similar. "
            "Wrong funding amount in the email. Otherwise looks professional. "
            "Body <=120 words."
        ),
    },
    {
        "label": "bad_d4_bench_word",
        "expected": "REJECT",
        "expected_dim": "D4",
        "brief_seed": {
            "company": "Praxis Cloud", "icp_segment": "Segment 2",
            "signal": "headcount cut 20% Q1 2026",
            "velocity_direction": "decelerating", "delta_pct": -50,
            "ai_maturity": 1, "bench_available": True,
        },
        "style_note": (
            "BAD Professional-marker pattern — use the word 'bench' externally. "
            "Write 'our bench is deep', 'from our bench', or 'bench engineers'. "
            "This is banned in prospect-facing outreach. "
            "Body <=120 words."
        ),
    },
    {
        "label": "bad_d2_wrong_segment_ai_maturity",
        "expected": "REJECT",
        "expected_dim": "D2",
        "brief_seed": {
            "company": "Ironclad SaaS", "icp_segment": "Segment 1",
            "funding": "Series A $8M Dec 2025",
            "velocity_direction": "stable", "delta_pct": 5,
            "ai_maturity": 0, "bench_available": True,
        },
        "style_note": (
            "BAD #8 pattern — wrong segment pitch. AI maturity is 0 but email "
            "pitches 'agentic systems roadmap', 'MLOps', 'Segment 4 capability gap'. "
            "Should have used Segment 1 reframe for AI maturity 0. "
            "Body <=120 words."
        ),
    },
    {
        "label": "bad_d3_growth_frame_on_stable",
        "expected": "REJECT",
        "expected_dim": "D3",
        "brief_seed": {
            "company": "Quorum Tech", "icp_segment": "Segment 1",
            "funding": "Series B $19M Mar 2026",
            "velocity_direction": "decelerating", "delta_pct": -25,
            "ai_maturity": 1, "bench_available": True,
        },
        "style_note": (
            "BAD signal-direction — velocity is -25% (clearly declining). "
            "Email uses 'bottleneck', 'scale your team', 'rapid growth'. "
            "Growth-frame terms on negative velocity fail D3. Body <=120 words."
        ),
    },
    {
        "label": "bad_d5_over_word_count",
        "expected": "REJECT",
        "expected_dim": "D5",
        "brief_seed": {
            "company": "Apexion Labs", "icp_segment": "Segment 1",
            "funding": "Series A $14M Jan 2026",
            "velocity_direction": "accelerating", "delta_pct": 80,
            "ai_maturity": 1, "bench_available": True,
        },
        "style_note": (
            "BAD #1 pattern — wall of self-promotion that exceeds 120 words. "
            "Describe Tenacious at length, list all service lines, ask for a "
            "45-minute discovery call. Must be 130+ words in the body. "
            "One question still OK."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Phase C — underrepresented dimension categories
# ─────────────────────────────────────────────────────────────────────────────

_D3_SEEDS: list[dict] = [
    {"company": f"VeloxCorp {i}", "delta_pct": delta, "segment": seg, "ai_maturity": am}
    for i, (delta, seg, am) in enumerate([
        (-35, "Segment 1", 1), (-55, "Segment 2", 1), (-28, "Segment 3", 2),
        (-45, "Segment 1", 0), (-70, "Segment 2", 1), (-30, "Segment 3", 1),
        (-22, "Segment 1", 2), (-60, "Segment 2", 0), (-40, "Segment 3", 1),
        (-38, "Segment 1", 1), (-52, "Segment 2", 2), (-25, "Segment 3", 0),
        (-33, "Segment 1", 1), (-48, "Segment 2", 1), (-42, "Segment 3", 2),
    ], 1)
]

_D4_SEEDS: list[dict] = [
    {"company": f"ToneFail {i}", "phrase": phrase, "segment": seg, "delta_pct": delta}
    for i, (phrase, seg, delta) in enumerate([
        ("top talent",    "Segment 1", 80),  ("world-class",   "Segment 2", -30),
        ("rockstar",      "Segment 3",  0),  ("synergize",     "Segment 4", 10),
        ("falling behind","Segment 4", 15),  ("you're behind", "Segment 1", 60),
        ("skyrocket",     "Segment 2", -20), ("aggressive hiring","Segment 3", 5),
        ("you lack",      "Segment 4",  8),  ("just ",         "Segment 1", 90),
        ("quick ",        "Segment 2", -40), ("top talent",    "Segment 3", 30),
        ("world-class",   "Segment 1", 50),  ("synergize",     "Segment 2", -55),
        ("you need to catch up", "Segment 4", 12),
    ], 1)
]


# ─────────────────────────────────────────────────────────────────────────────
# LLM helpers
# ─────────────────────────────────────────────────────────────────────────────

def _llm_client():
    from openai import OpenAI
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment/.env")
    return OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)


_TASK_SYSTEM = f"""You are a dataset engineer building Tenacious-Bench, a B2B sales-email
evaluation dataset. Your output is a JSON task object following this exact schema:

{{
  "task_id": "TB-AUG-XXX",
  "brief": {{
    "company": "...",
    "icp_segment": "Segment 1" | "Segment 2" | "Segment 3" | "Segment 4" | "Ambiguous",
    "confidence": 0.7-0.95,
    "ai_maturity": 0-3,
    "hiring_velocity": {{
      "direction": "accelerating" | "decelerating" | "stable",
      "delta_pct": <float>,
      "signal_strength": "strong" | "moderate" | "weak",
      "observation": "<one sentence>"
    }},
    "grounding_facts": ["<fact1>", "<fact2>"],
    "bench_match": {{
      "required_stacks": ["python", "ml"],
      "bench_available": true | false
    }},
    "source_mode": "augmented_style_guide"
  }},
  "email": {{
    "subject": "<subject <=60 chars, starts with Context:/Note on/Congrats on/Question on>",
    "body": "<body — specific word count depends on scenario instructions>",
    "word_count": <int>,
    "tone_warnings": []
  }},
  "prior_thread": "",
  "bench_summary": "{BENCH_SUMMARY}",
  "rubric": {{
    "dimensions": ["grounding_fidelity","icp_pitch_alignment","signal_directionality","tone_compliance","format_compliance"],
    "negative_velocity_threshold_pct": -20.0,
    "max_subject_chars": 60,
    "max_body_words": 120,
    "approved_subject_prefixes": ["context:","note on","congrats on","question on"],
    "banned_phrases": ["top talent","world-class","a-players","rockstar","ninja","aggressive hiring","cost savings of","guaranteed roi","proven track record","falling behind","you're behind","you lack","you're missing","you need to catch up","left behind","hope this finds","hey there","quick ","just "],
    "growth_frame_terms": ["bottleneck","bottlenecks","scaling","scale your","accelerate","accelerating","rapid growth","increased demand","augment your team","augment existing","expand the team","staff up"],
    "product_claim_terms": ["tenacious can","tenacious provides","tenacious offers","we can","we provide","our engineers","available on-demand","pre-vetted","project-ready","augment your","deploy in days"],
    "icp_pitch_alignment_policy": "Phase 1 only fast-fails Ambiguous plus product claim."
  }},
  "difficulty": "medium",
  "judge_filter": {{"passed": true, "model": "augmented"}}
}}

Output ONLY the JSON object. No markdown. No explanation."""


def _generate_task(client, prompt_user: str, task_id: str) -> dict | None:
    """Call LLM to generate a task JSON; validate schema basics."""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model    = LLM_MODEL,
                messages = [
                    {"role": "system", "content": _TASK_SYSTEM},
                    {"role": "user",   "content": prompt_user},
                ],
                temperature = 0.8,
                max_tokens  = 900,
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown fences if present
            raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
            task = json.loads(raw)
            task["task_id"] = task_id
            # Basic schema checks
            assert "brief" in task and "email" in task
            assert task["email"].get("subject", "")
            assert task["email"].get("body", "")
            return task
        except Exception as exc:
            print(f"    attempt {attempt+1} failed: {exc}")
            time.sleep(2)
    return None


def _scenario_to_prompt(scenario: dict) -> str:
    bs = scenario["brief_seed"]
    return (
        f"Generate a Tenacious outreach task with task_id 'TB-AUG-PLACEHOLDER'.\n\n"
        f"Brief seed: {json.dumps(bs, indent=2)}\n\n"
        f"Style note: {scenario['style_note']}\n\n"
        f"Expected verdict: {scenario['expected']}\n"
        + (f"Expected failing dimension: {scenario.get('expected_dim','')}\n" if scenario.get("expected_dim") else "")
        + "\nOutput only the JSON task object."
    )


def _d3_seed_to_prompt(seed: dict, task_id: str) -> str:
    return (
        f"Generate a Tenacious outreach task.\n\n"
        f"Requirements:\n"
        f"- Company: {seed['company']}\n"
        f"- ICP Segment: {seed['segment']}\n"
        f"- Hiring velocity delta: {seed['delta_pct']:+.0f}% (DECLINING — do NOT use growth-frame terms in the email)\n"
        f"- AI maturity: {seed['ai_maturity']}\n"
        f"- The email MUST use at least one growth-frame term from this list: "
        f"bottleneck, bottlenecks, scaling, scale your, accelerate, accelerating, "
        f"rapid growth, increased demand, augment your team, augment existing, "
        f"expand the team, staff up\n"
        f"- This creates a D3 (signal_directionality) failure: negative velocity "
        f"+ growth-frame language\n"
        f"- Everything else (subject <=60, body <=120 words, 1 question, "
        f"no banned phrases except growth-frame, grounded numerics) should PASS\n"
        f"- bench_match.bench_available: true\n"
        f"- Include one grounding fact with the exact delta percentage\n\n"
        f"Output only the JSON task object with task_id '{task_id}'."
    )


def _d4_seed_to_prompt(seed: dict, task_id: str) -> str:
    return (
        f"Generate a Tenacious outreach task.\n\n"
        f"Requirements:\n"
        f"- Company: {seed['company']}\n"
        f"- ICP Segment: {seed['segment']}\n"
        f"- Hiring velocity delta: {seed['delta_pct']:+.0f}%\n"
        f"- The email MUST contain the exact phrase: '{seed['phrase']}'\n"
        f"- This creates a D4 (tone_compliance) failure\n"
        f"- Everything else (subject <=60, body <=120 words, 1 question, "
        f"grounded numerics, correct velocity framing) should PASS\n"
        f"- bench_match.bench_available: true\n"
        f"- Include one grounding fact referencing the velocity\n\n"
        f"Output only the JSON task object with task_id '{task_id}'."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pair builder
# ─────────────────────────────────────────────────────────────────────────────

def _task_to_pair(task: dict, scores: dict) -> dict:
    verdict  = scores["verdict"]
    dim_code = scores.get("failed_dimension")
    reason   = scores.get("reason", "Rubric dimension check failed.")
    prompt   = _build_judge_prompt(task)

    if verdict == "REJECT":
        dim_name = DIM_NAMES.get(dim_code, dim_code or "unknown")
        chosen  = (
            f"VERDICT: REJECT\n"
            f"Primary failure: {dim_name}\n"
            f"Reason: {reason}"
        )
        rejected = "VERDICT: PASS\nAll checks satisfied. Email is acceptable to send."
    else:
        orig_idx = _original_template_idx(task["task_id"])
        fake_dim, fake_reason = _pick_template(task["task_id"], "style_guide", orig_idx)
        chosen   = "VERDICT: PASS\nAll grounding facts verified. No pitch or signal mismatch."
        rejected = (
            f"VERDICT: REJECT\n"
            f"Primary failure: {fake_dim}\n"
            f"Reason: {fake_reason}"
        )

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


# ─────────────────────────────────────────────────────────────────────────────
# Phase runners
# ─────────────────────────────────────────────────────────────────────────────

def run_phase_a(tasks: list[dict]) -> list[dict]:
    """PASS multi-negative: 2 additional rejected responses per PASS task."""
    print("\n[Phase A] PASS multi-negative augmentation...")
    pass_tasks = [t for t in tasks if score_task(t)["verdict"] == "PASS"]
    print(f"  PASS tasks in train: {len(pass_tasks)}")

    new_pairs: list[dict] = []
    for task in pass_tasks:
        prompt   = _build_judge_prompt(task)
        chosen   = "VERDICT: PASS\nAll grounding facts verified. No pitch or signal mismatch."
        orig_idx = _original_template_idx(task["task_id"])

        for suffix in ("_extra1", "_extra2"):
            dim, reason = _pick_template(task["task_id"], suffix, orig_idx)
            rejected    = f"VERDICT: REJECT\nPrimary failure: {dim}\nReason: {reason}"
            new_pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    print(f"  Generated: {len(new_pairs)} pairs")
    return new_pairs


def run_phase_b(dry_run: bool) -> list[dict]:
    """Style-guide scaffolded generation."""
    print("\n[Phase B] Style-guide scaffolded pairs...")
    if dry_run:
        print("  dry-run: skipping LLM calls")
        return []

    client    = _llm_client()
    new_pairs: list[dict] = []
    ok = err = 0

    for i, scenario in enumerate(_STYLE_SCENARIOS):
        task_id = f"TB-AUG-B{i+1:02d}"
        prompt  = _scenario_to_prompt(scenario)
        print(f"  {task_id} [{scenario['label']}]...", end=" ", flush=True)

        task = _generate_task(client, prompt, task_id)
        if task is None:
            print("FAILED (generation)")
            err += 1
            continue

        try:
            scores = score_task(task)
        except Exception as exc:
            print(f"FAILED (score_task: {exc})")
            err += 1
            continue

        actual   = scores["verdict"]
        expected = scenario["expected"]

        if actual != expected:
            exp_dim = scenario.get("expected_dim")
            if expected == "REJECT" and exp_dim and scores.get("failed_dimension") != exp_dim:
                print(f"DIM MISMATCH (wanted {exp_dim}, got {scores.get('failed_dimension')}) — kept")
            elif actual != expected:
                print(f"VERDICT MISMATCH (wanted {expected}, got {actual}) — skipped")
                err += 1
                continue

        pair = _task_to_pair(task, scores)
        new_pairs.append(pair)
        print(f"OK [{actual}]")
        ok += 1
        time.sleep(0.5)

    print(f"  Phase B: {ok} OK, {err} failed/skipped")
    return new_pairs


def run_phase_c(dry_run: bool) -> list[dict]:
    """Category coverage: D3 + D4 underrepresented dimensions."""
    print("\n[Phase C] Category coverage pairs (D3 + D4)...")
    if dry_run:
        print("  dry-run: skipping LLM calls")
        return []

    client    = _llm_client()
    new_pairs: list[dict] = []
    ok = err = 0

    # D3 failures
    for i, seed in enumerate(_D3_SEEDS):
        task_id = f"TB-AUG-D3-{i+1:02d}"
        prompt  = _d3_seed_to_prompt(seed, task_id)
        print(f"  {task_id} [D3]...", end=" ", flush=True)

        task = _generate_task(client, prompt, task_id)
        if task is None:
            print("FAILED"); err += 1; continue

        try:
            scores = score_task(task)
        except Exception as exc:
            print(f"FAILED ({exc})"); err += 1; continue

        if scores["verdict"] != "REJECT" or scores.get("failed_dimension") != "D3":
            print(f"WRONG ({scores['verdict']}/{scores.get('failed_dimension')}) — skipped")
            err += 1
            continue

        new_pairs.append(_task_to_pair(task, scores))
        print("OK"); ok += 1
        time.sleep(0.5)

    # D4 failures
    for i, seed in enumerate(_D4_SEEDS):
        task_id = f"TB-AUG-D4-{i+1:02d}"
        prompt  = _d4_seed_to_prompt(seed, task_id)
        print(f"  {task_id} [D4 '{seed['phrase']}']...", end=" ", flush=True)

        task = _generate_task(client, prompt, task_id)
        if task is None:
            print("FAILED"); err += 1; continue

        try:
            scores = score_task(task)
        except Exception as exc:
            print(f"FAILED ({exc})"); err += 1; continue

        if scores["verdict"] != "REJECT" or scores.get("failed_dimension") != "D4":
            print(f"WRONG ({scores['verdict']}/{scores.get('failed_dimension')}) — skipped")
            err += 1
            continue

        new_pairs.append(_task_to_pair(task, scores))
        print("OK"); ok += 1
        time.sleep(0.5)

    print(f"  Phase C: {ok} OK, {err} failed/skipped")
    return new_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Phase A only (no LLM calls)")
    parser.add_argument("--phases", default="ABC",
                        help="Which phases to run, e.g. 'A', 'AB', 'ABC' (default: ABC)")
    args = parser.parse_args()

    # Load base data
    base_pairs = [
        json.loads(l) for l in BASE_FILE.read_text("utf-8").splitlines() if l.strip()
    ]
    tasks = [
        json.loads(l) for l in TRAIN_FILE.read_text("utf-8").splitlines() if l.strip()
    ]
    print(f"Base pairs : {len(base_pairs)}")
    print(f"Train tasks: {len(tasks)}")

    new_pairs: list[dict] = []

    if "A" in args.phases:
        new_pairs += run_phase_a(tasks)

    if "B" in args.phases:
        new_pairs += run_phase_b(args.dry_run)

    if "C" in args.phases:
        new_pairs += run_phase_c(args.dry_run)

    all_pairs = base_pairs + new_pairs
    print(f"\nTotal pairs: {len(base_pairs)} original + {len(new_pairs)} new = {len(all_pairs)}")

    # Label distribution
    reject_chosen = sum(1 for p in all_pairs if p["chosen"].startswith("VERDICT: REJECT"))
    pass_chosen   = len(all_pairs) - reject_chosen
    print(f"REJECT-chosen={reject_chosen}  PASS-chosen={pass_chosen}")

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        for pair in all_pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nWritten -> {OUTPUT_FILE}  ({len(all_pairs)} pairs)")


if __name__ == "__main__":
    main()
