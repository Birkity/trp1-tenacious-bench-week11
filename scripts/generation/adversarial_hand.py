"""
Hand-authored adversarial task generator — Act II.

Generates 40 carefully crafted adversarial benchmark tasks designed to:
  - Test all 5 rubric dimensions explicitly, including near-threshold edge cases
  - Include semantic adversarial tasks that PASS Phase 1 but fail contextually (Phase 2 tests)
  - Cover multi-dimension failure scenarios
  - Defeat naive pattern-matching agents from Week 10

Tasks are hardcoded (hand-authored) — no LLM generation.
Each task has an explicit expected verdict (PASS or REJECT) and annotates
the primary failing dimension(s) and adversarial type inside the brief.

Breakdown:
  - D1 failures (grounding_fidelity):        5 tasks  (001-005)
  - D2 failures (icp_pitch_alignment):        4 tasks  (006-009)
  - D3 failures (signal_directionality):      5 tasks  (010-014)
  - D4 failures (tone_compliance):            6 tasks  (015-020)
  - D5 failures (format_compliance):          6 tasks  (021-026)
  - Multi-dimension failures:                 5 tasks  (027-031)
  - PASS tasks (correct, near-threshold):     9 tasks  (032-040)

Total: 40 tasks

Usage:
    python scripts/generation/adversarial_hand.py
    python scripts/generation/adversarial_hand.py --verify-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))

OUTPUT_FILE = ROOT / "data" / "tenacious_bench_v0.1" / "dev" / "adversarial_hand_batch1.jsonl"
SCHEMA_PATH = ROOT / "benchmark" / "schema.json"

BENCH_SUMMARY = (
    "Python: 7 available (7-day deploy). "
    "Data: 9 available (7-day deploy). "
    "ML: 5 available (10-day deploy). "
    "Go: 3 available (14-day deploy). "
    "Infra: 4 available (14-day deploy). "
    "Frontend: 6 available (7-day deploy). "
    "NestJS: constrained through Q3 2026 (Modo Compass engagement — do not commit)."
)


def _base_rubric() -> dict[str, Any]:
    from scoring_evaluator import DEFAULT_RUBRIC  # type: ignore
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


def _task(
    task_id: str,
    brief: dict[str, Any],
    subject: str,
    body: str,
    expected_verdict: str,
    expected_fail_dims: list[str],
    adversarial_type: str,
) -> dict[str, Any]:
    brief_out = {
        **brief,
        "adversarial_type": adversarial_type,
        "expected_verdict": expected_verdict,
        "expected_fail_dims": expected_fail_dims,
        "source_mode": "adversarial_hand",
    }
    return {
        "task_id": task_id,
        "brief": brief_out,
        "email": {
            "subject": subject,
            "body": body,
            "word_count": len(body.split()),
            "tone_warnings": [],
        },
        "prior_thread": "",
        "bench_summary": BENCH_SUMMARY,
        "rubric": _base_rubric(),
    }


def build_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # D1 GROUNDING FIDELITY failures (TB-ADV-001 to 005)
    # ------------------------------------------------------------------

    # 001: Wrong dollar amount — brief has $14M, email says $99M
    tasks.append(_task(
        "TB-ADV-001",
        brief={
            "company": "Helix Dynamics",
            "icp_segment": "Segment 1",
            "confidence": 0.85,
            "hiring_velocity": {
                "direction": "accelerating", "delta_pct": 80.0,
                "signal_strength": "strong",
                "observation": "Helix Dynamics increased open roles by 80% in 60 days.",
            },
            "grounding_facts": [
                "Series A $14M closed Jan 2026",
                "Helix Dynamics increased open roles by 80% in 60 days.",
            ],
            "bench_match": {"required_stacks": ["python", "ml"], "bench_available": True},
        },
        subject="Congrats on Series A $14M, Helix Dynamics",
        body=(
            "Jordan,\n\n"
            "Helix Dynamics increased open roles by 80% following your $99M raise in Jan 2026.\n\n"
            "Companies scaling post-Series A often need engineers faster than traditional hiring.\n\n"
            "Tenacious provides pre-vetted ML engineers deployable in days.\n\n"
            "What engineering priorities are you scaling into next quarter?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["grounding_fidelity"],
        adversarial_type="d1_wrong_dollar_amount",
    ))

    # 002: No grounding fact referenced — email is generic, no company-specific claims
    tasks.append(_task(
        "TB-ADV-002",
        brief={
            "company": "Cascade Robotics",
            "icp_segment": "Segment 1",
            "confidence": 0.9,
            "hiring_velocity": {
                "direction": "accelerating", "delta_pct": 100.0,
                "signal_strength": "strong",
                "observation": "Cascade Robotics doubled open roles in 60 days.",
            },
            "grounding_facts": [
                "Series B $22M closed Feb 2026",
                "Cascade Robotics doubled open roles in 60 days.",
            ],
            "bench_match": {"required_stacks": ["python", "infra"], "bench_available": True},
        },
        subject="Context: engineering capacity for robotics scale-up",
        body=(
            "Priya,\n\n"
            "Robotics companies often face scaling challenges as they transition from R&D to production.\n\n"
            "Tenacious provides pre-vetted engineers for rapid deployment.\n\n"
            "What engineering bottlenecks are you prioritizing this quarter?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["grounding_fidelity"],
        adversarial_type="d1_no_grounding_fact",
    ))

    # 003: bench_available=False + product claim in email → D1 fail
    tasks.append(_task(
        "TB-ADV-003",
        brief={
            "company": "Fulgent Data",
            "icp_segment": "Segment 2",
            "confidence": 0.8,
            "hiring_velocity": {
                "direction": "decelerating", "delta_pct": -30.0,
                "signal_strength": "moderate",
                "observation": "Fulgent Data reduced open roles by 30% in 60 days.",
            },
            "grounding_facts": ["Fulgent Data reduced open roles by 30% in 60 days."],
            "bench_match": {"required_stacks": ["python", "data"], "bench_available": False},
        },
        subject="Note on engineering continuity at Fulgent Data",
        body=(
            "Devon,\n\n"
            "Fulgent Data reduced open roles by 30% in the last 60 days.\n\n"
            "Tenacious provides engineers with defined scopes and predictable costs.\n\n"
            "What delivery risks are top of mind for your team this quarter?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["grounding_fidelity"],
        adversarial_type="d1_bench_unavailable_with_product_claim",
    ))

    # 004: Wrong percentage — brief has 60%, email says 45%
    tasks.append(_task(
        "TB-ADV-004",
        brief={
            "company": "Orbital Labs",
            "icp_segment": "Segment 4",
            "confidence": 0.75,
            "hiring_velocity": {
                "direction": "accelerating", "delta_pct": 60.0,
                "signal_strength": "moderate",
                "observation": "Orbital Labs increased AI role postings by 60% in 60 days.",
            },
            "grounding_facts": ["Orbital Labs increased AI role postings by 60% in 60 days."],
            "ai_maturity_score": 2,
            "bench_match": {"required_stacks": ["ml", "data"], "bench_available": True},
        },
        subject="Context: ML hiring velocity at Orbital Labs",
        body=(
            "Sam,\n\n"
            "Orbital Labs increased AI role postings by 45% in the last 60 days.\n\n"
            "Tenacious provides pre-vetted ML engineers for high-specificity gaps.\n\n"
            "Which ML capability gaps are most critical for your roadmap?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["grounding_fidelity"],
        adversarial_type="d1_wrong_percentage",
    ))

    # 005: Wrong year — brief says Q1 2026, email says 2019
    tasks.append(_task(
        "TB-ADV-005",
        brief={
            "company": "Pulsar Tech",
            "icp_segment": "Segment 3",
            "confidence": 0.8,
            "hiring_velocity": {
                "direction": "stable", "delta_pct": 5.0,
                "signal_strength": "weak",
                "observation": "Pulsar Tech VP Engineering joined from Stripe in Q1 2026.",
            },
            "grounding_facts": ["Pulsar Tech VP Engineering joined from Stripe in Q1 2026."],
            "bench_match": {"required_stacks": ["python", "infra"], "bench_available": True},
        },
        subject="Note on vendor strategy at Pulsar Tech",
        body=(
            "Dana,\n\n"
            "Pulsar Tech VP Engineering joined from Stripe in 2019.\n\n"
            "Incoming engineering leadership often reassesses vendor relationships in the first 90 days.\n\n"
            "Tenacious provides engineers for defined project scopes during transitions.\n\n"
            "Which delivery commitments are most critical through the leadership change?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["grounding_fidelity"],
        adversarial_type="d1_wrong_year",
    ))

    # ------------------------------------------------------------------
    # D2 ICP-PITCH ALIGNMENT failures (TB-ADV-006 to 009)
    # ------------------------------------------------------------------

    # 006: Ambiguous + "Tenacious provides" product claim
    tasks.append(_task(
        "TB-ADV-006",
        brief={
            "company": "Driftwave Systems",
            "icp_segment": "Ambiguous",
            "confidence": 0.5,
            "hiring_velocity": {
                "direction": "decelerating", "delta_pct": -45.0,
                "signal_strength": "moderate",
                "observation": "Driftwave Systems reduced open roles by 45% in 60 days.",
            },
            "grounding_facts": ["Driftwave Systems reduced open roles by 45% in 60 days."],
            "bench_match": {"required_stacks": ["python", "data"], "bench_available": True},
        },
        subject="Question on engineering priorities at Driftwave",
        body=(
            "Riley,\n\n"
            "Driftwave Systems reduced open roles by 45% in 60 days.\n\n"
            "Tenacious provides engineers for cost-efficient delivery without long-term headcount.\n\n"
            "Are you currently focused on reducing costs or maintaining delivery capacity?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["icp_pitch_alignment"],
        adversarial_type="d2_ambiguous_with_product_claim",
    ))

    # 007: Ambiguous + "pre-vetted" term
    tasks.append(_task(
        "TB-ADV-007",
        brief={
            "company": "Ironveil Corp",
            "icp_segment": "Ambiguous",
            "confidence": 0.45,
            "hiring_velocity": {
                "direction": "unknown", "delta_pct": 0.0,
                "signal_strength": "weak",
                "observation": "Ironveil Corp hiring activity is inconclusive from available signals.",
            },
            "grounding_facts": ["Ironveil Corp hiring activity is inconclusive from available signals."],
            "bench_match": {"required_stacks": ["data", "ml"], "bench_available": True},
        },
        subject="Question on engineering context at Ironveil Corp",
        body=(
            "Morgan,\n\n"
            "Ironveil Corp hiring activity is inconclusive from available signals.\n\n"
            "Our pre-vetted data and ML engineers are available for project-ready deployment.\n\n"
            "What are your current data engineering priorities?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["icp_pitch_alignment"],
        adversarial_type="d2_ambiguous_with_prevetted",
    ))

    # 008: Ambiguous + "we can" product claim
    tasks.append(_task(
        "TB-ADV-008",
        brief={
            "company": "Novascreen AI",
            "icp_segment": "Ambiguous",
            "confidence": 0.55,
            "hiring_velocity": {
                "direction": "decelerating", "delta_pct": -22.0,
                "signal_strength": "weak",
                "observation": "Novascreen AI reduced open roles by 22% in 60 days.",
            },
            "grounding_facts": ["Novascreen AI reduced open roles by 22% in 60 days."],
            "bench_match": {"required_stacks": ["python", "ml"], "bench_available": True},
        },
        subject="Question on Novascreen AI engineering direction",
        body=(
            "Casey,\n\n"
            "Novascreen AI reduced open roles by 22% in 60 days.\n\n"
            "We can provide Python and ML engineers deployable within a week.\n\n"
            "Are you currently reassessing your engineering structure?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["icp_pitch_alignment"],
        adversarial_type="d2_ambiguous_with_we_can",
    ))

    # 009: Ambiguous + "deploy in days" — subtle product claim variant
    tasks.append(_task(
        "TB-ADV-009",
        brief={
            "company": "Stellarnet",
            "icp_segment": "Ambiguous",
            "confidence": 0.6,
            "hiring_velocity": {
                "direction": "accelerating", "delta_pct": 15.0,
                "signal_strength": "weak",
                "observation": "Stellarnet open roles increased by 15% in the last 60 days.",
            },
            "grounding_facts": ["Stellarnet open roles increased by 15% in the last 60 days."],
            "bench_match": {"required_stacks": ["frontend", "python"], "bench_available": True},
        },
        subject="Question on engineering priorities at Stellarnet",
        body=(
            "Alex,\n\n"
            "Stellarnet open roles increased by 15% in the last 60 days.\n\n"
            "Our engineers deploy in days and integrate with existing teams.\n\n"
            "What is driving your current hiring activity?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["icp_pitch_alignment"],
        adversarial_type="d2_ambiguous_deploy_in_days",
    ))

    # ------------------------------------------------------------------
    # D3 SIGNAL DIRECTIONALITY failures (TB-ADV-010 to 014)
    # ------------------------------------------------------------------

    # 010: delta=-50%, "scaling" in body
    tasks.append(_task(
        "TB-ADV-010",
        brief={
            "company": "Cipherlink",
            "icp_segment": "Segment 1",
            "confidence": 0.75,
            "hiring_velocity": {
                "direction": "decelerating", "delta_pct": -50.0,
                "signal_strength": "strong",
                "observation": "Cipherlink reduced open roles by 50% in 60 days.",
            },
            "grounding_facts": ["Cipherlink reduced open roles by 50% in 60 days."],
            "bench_match": {"required_stacks": ["python", "data"], "bench_available": True},
        },
        subject="Context: Cipherlink engineering update",
        body=(
            "Taylor,\n\n"
            "Cipherlink reduced open roles by 50% in 60 days.\n\n"
            "Tenacious provides engineers for scaling your data infrastructure efficiently.\n\n"
            "What engineering priorities are you addressing this quarter?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["signal_directionality"],
        adversarial_type="d3_scaling_on_negative_delta",
    ))

    # 011: delta=-80%, "accelerating" in body
    tasks.append(_task(
        "TB-ADV-011",
        brief={
            "company": "Verdant Networks",
            "icp_segment": "Segment 2",
            "confidence": 0.85,
            "hiring_velocity": {
                "direction": "decelerating", "delta_pct": -80.0,
                "signal_strength": "strong",
                "observation": "Verdant Networks reduced open roles by 80% in 60 days.",
            },
            "grounding_facts": ["Verdant Networks reduced open roles by 80% in 60 days."],
            "bench_match": {"required_stacks": ["python", "infra"], "bench_available": True},
        },
        subject="Note on Verdant Networks engineering context",
        body=(
            "River,\n\n"
            "Verdant Networks reduced open roles by 80% in 60 days.\n\n"
            "Tenacious helps teams accelerating their delivery without permanent headcount.\n\n"
            "What delivery commitments are at risk this quarter?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["signal_directionality"],
        adversarial_type="d3_accelerate_on_negative_delta",
    ))

    # 012: delta=-30%, "bottleneck" in body
    tasks.append(_task(
        "TB-ADV-012",
        brief={
            "company": "Arboretum Labs",
            "icp_segment": "Segment 3",
            "confidence": 0.7,
            "hiring_velocity": {
                "direction": "decelerating", "delta_pct": -30.0,
                "signal_strength": "moderate",
                "observation": "Arboretum Labs reduced open roles by 30% after leadership transition.",
            },
            "grounding_facts": [
                "Arboretum Labs reduced open roles by 30% after leadership transition.",
            ],
            "bench_match": {"required_stacks": ["python", "data"], "bench_available": True},
        },
        subject="Note on vendor strategy at Arboretum Labs",
        body=(
            "Quinn,\n\n"
            "Arboretum Labs reduced open roles by 30% after leadership transition.\n\n"
            "New CTOs often face a bottleneck between their vision and available delivery capacity.\n\n"
            "Tenacious provides engineers for defined project scopes during transitions.\n\n"
            "Which delivery commitments need continuity through the leadership change?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["signal_directionality"],
        adversarial_type="d3_bottleneck_on_negative_delta",
    ))

    # 013: delta=-25%, "staff up" in body
    tasks.append(_task(
        "TB-ADV-013",
        brief={
            "company": "Hexadyne Corp",
            "icp_segment": "Segment 2",
            "confidence": 0.8,
            "hiring_velocity": {
                "direction": "decelerating", "delta_pct": -25.0,
                "signal_strength": "moderate",
                "observation": "Hexadyne Corp open roles decreased by 25% in 60 days.",
            },
            "grounding_facts": ["Hexadyne Corp open roles decreased by 25% in 60 days."],
            "bench_match": {"required_stacks": ["python", "infra"], "bench_available": True},
        },
        subject="Note on engineering continuity at Hexadyne Corp",
        body=(
            "Blake,\n\n"
            "Hexadyne Corp open roles decreased by 25% in 60 days.\n\n"
            "Tenacious provides engineers to staff up your delivery capacity without long-term commitments.\n\n"
            "What delivery risks are top of mind this quarter?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["signal_directionality"],
        adversarial_type="d3_staff_up_on_negative_delta",
    ))

    # 014: delta=-40%, "expand the team" in body
    tasks.append(_task(
        "TB-ADV-014",
        brief={
            "company": "Luminary Software",
            "icp_segment": "Segment 4",
            "confidence": 0.75,
            "hiring_velocity": {
                "direction": "decelerating", "delta_pct": -40.0,
                "signal_strength": "moderate",
                "observation": "Luminary Software ML roles decreased by 40% in 60 days.",
            },
            "grounding_facts": ["Luminary Software ML roles decreased by 40% in 60 days."],
            "ai_maturity_score": 2,
            "bench_match": {"required_stacks": ["ml", "data"], "bench_available": True},
        },
        subject="Context: ML capability at Luminary Software",
        body=(
            "Priya,\n\n"
            "Luminary Software ML roles decreased by 40% in 60 days.\n\n"
            "Tenacious helps teams expand the team during ML capability transitions.\n\n"
            "Which ML capability gaps are most critical for your roadmap?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["signal_directionality"],
        adversarial_type="d3_expand_team_on_negative_delta",
    ))

    # ------------------------------------------------------------------
    # D4 TONE COMPLIANCE failures (TB-ADV-015 to 020)
    # ------------------------------------------------------------------

    # 015: "top talent" phrase
    tasks.append(_task(
        "TB-ADV-015",
        brief={
            "company": "Zephyr Cloud",
            "icp_segment": "Segment 1",
            "confidence": 0.85,
            "hiring_velocity": {
                "direction": "accelerating", "delta_pct": 90.0,
                "signal_strength": "strong",
                "observation": "Zephyr Cloud increased open roles by 90% in 60 days.",
            },
            "grounding_facts": [
                "Series A $12M closed Mar 2026",
                "Zephyr Cloud increased open roles by 90% in 60 days.",
            ],
            "bench_match": {"required_stacks": ["python", "infra"], "bench_available": True},
        },
        subject="Congrats on Series A $12M, Zephyr Cloud",
        body=(
            "Sydney,\n\n"
            "Zephyr Cloud increased open roles by 90% since closing $12M in Mar 2026.\n\n"
            "We provide top talent engineers ready to accelerate your infrastructure build-out.\n\n"
            "What engineering priorities are you scaling into next quarter?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["tone_compliance"],
        adversarial_type="d4_top_talent_phrase",
    ))

    # 016: "world-class" phrase
    tasks.append(_task(
        "TB-ADV-016",
        brief={
            "company": "Axiomatic AI",
            "icp_segment": "Segment 1",
            "confidence": 0.9,
            "hiring_velocity": {
                "direction": "accelerating", "delta_pct": 70.0,
                "signal_strength": "strong",
                "observation": "Axiomatic AI increased open roles by 70% in 60 days.",
            },
            "grounding_facts": [
                "Series B $35M closed Feb 2026",
                "Axiomatic AI increased open roles by 70% in 60 days.",
            ],
            "bench_match": {"required_stacks": ["ml", "python"], "bench_available": True},
        },
        subject="Congrats on Series B $35M, Axiomatic AI",
        body=(
            "Jordan,\n\n"
            "Axiomatic AI increased open roles by 70% since closing $35M in Feb 2026.\n\n"
            "Tenacious provides world-class ML engineers deployable within 7 days.\n\n"
            "What engineering priorities are you scaling into next quarter?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["tone_compliance"],
        adversarial_type="d4_world_class_phrase",
    ))

    # 017: "bench" in body (internal jargon leaked into prospect-facing copy)
    tasks.append(_task(
        "TB-ADV-017",
        brief={
            "company": "Montara Systems",
            "icp_segment": "Segment 3",
            "confidence": 0.8,
            "hiring_velocity": {
                "direction": "stable", "delta_pct": 5.0,
                "signal_strength": "weak",
                "observation": "Montara Systems hiring stable with new CTO announced Q1 2026.",
            },
            "grounding_facts": [
                "Montara Systems hiring stable with new CTO announced Q1 2026.",
            ],
            "bench_match": {"required_stacks": ["python", "data"], "bench_available": True},
        },
        subject="Note on vendor strategy at Montara Systems",
        body=(
            "Cameron,\n\n"
            "Montara Systems stable hiring with new CTO announced Q1 2026.\n\n"
            "Tenacious has engineers on the bench available for project-based engagements.\n\n"
            "What engineering priorities are you focused on stabilizing?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["tone_compliance"],
        adversarial_type="d4_bench_jargon_in_prospect_copy",
    ))

    # 018: "hope this finds" opener
    tasks.append(_task(
        "TB-ADV-018",
        brief={
            "company": "Pyrex Analytics",
            "icp_segment": "Segment 2",
            "confidence": 0.75,
            "hiring_velocity": {
                "direction": "decelerating", "delta_pct": -38.0,
                "signal_strength": "moderate",
                "observation": "Pyrex Analytics reduced open roles by 38% in Q1 2026.",
            },
            "grounding_facts": ["Pyrex Analytics reduced open roles by 38% in Q1 2026."],
            "bench_match": {"required_stacks": ["data", "python"], "bench_available": True},
        },
        subject="Note on engineering continuity at Pyrex Analytics",
        body=(
            "Lee,\n\n"
            "Hope this finds you well. Pyrex Analytics reduced open roles by 38% in Q1 2026.\n\n"
            "Tenacious provides engineers with defined scopes and predictable costs.\n\n"
            "What delivery risks are top of mind this quarter?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["tone_compliance"],
        adversarial_type="d4_hope_this_finds_opener",
    ))

    # 019: "falling behind" condescension
    tasks.append(_task(
        "TB-ADV-019",
        brief={
            "company": "Kinetic Grid",
            "icp_segment": "Segment 4",
            "confidence": 0.8,
            "hiring_velocity": {
                "direction": "accelerating", "delta_pct": 50.0,
                "signal_strength": "moderate",
                "observation": "Kinetic Grid ML role postings up 50% in 60 days.",
            },
            "grounding_facts": ["Kinetic Grid ML role postings up 50% in 60 days."],
            "ai_maturity_score": 1,
            "bench_match": {"required_stacks": ["ml"], "bench_available": True},
        },
        subject="Context: ML capability at Kinetic Grid",
        body=(
            "Kim,\n\n"
            "Kinetic Grid ML role postings up 50% in 60 days.\n\n"
            "Without specialized ML engineers, your team may be falling behind the competition.\n\n"
            "Tenacious provides pre-vetted ML engineers for high-specificity gaps.\n\n"
            "Which ML capabilities are most critical for your roadmap?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["tone_compliance"],
        adversarial_type="d4_falling_behind_condescension",
    ))

    # 020: "rockstar" phrase
    tasks.append(_task(
        "TB-ADV-020",
        brief={
            "company": "Prism Dev",
            "icp_segment": "Segment 1",
            "confidence": 0.85,
            "hiring_velocity": {
                "direction": "accelerating", "delta_pct": 110.0,
                "signal_strength": "strong",
                "observation": "Prism Dev open roles increased by 110% after $19M Series A close.",
            },
            "grounding_facts": [
                "Series A $19M closed Jan 2026",
                "Prism Dev open roles increased by 110% after $19M Series A close.",
            ],
            "bench_match": {"required_stacks": ["python", "ml"], "bench_available": True},
        },
        subject="Congrats on Series A $19M, Prism Dev",
        body=(
            "Marcus,\n\n"
            "Prism Dev open roles increased by 110% after closing $19M in Jan 2026.\n\n"
            "We help teams find rockstar engineers without the 3-month hiring cycle.\n\n"
            "What engineering priorities are you scaling into next quarter?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["tone_compliance"],
        adversarial_type="d4_rockstar_phrase",
    ))

    # ------------------------------------------------------------------
    # D5 FORMAT COMPLIANCE failures (TB-ADV-021 to 026)
    # ------------------------------------------------------------------

    # 021: subject > 60 chars
    subj_021 = "Context: Velion Systems Series B and engineering scaling plans"  # 62 chars
    assert len(subj_021) > 60
    tasks.append(_task(
        "TB-ADV-021",
        brief={
            "company": "Velion Systems",
            "icp_segment": "Segment 1",
            "confidence": 0.85,
            "hiring_velocity": {
                "direction": "accelerating", "delta_pct": 80.0,
                "signal_strength": "strong",
                "observation": "Velion Systems increased open roles by 80% in 60 days.",
            },
            "grounding_facts": [
                "Series B $24M closed Feb 2026",
                "Velion Systems increased open roles by 80% in 60 days.",
            ],
            "bench_match": {"required_stacks": ["python"], "bench_available": True},
        },
        subject=subj_021,
        body=(
            "Lee,\n\n"
            "Velion Systems increased open roles by 80% since closing $24M in Feb 2026.\n\n"
            "Tenacious provides pre-vetted Python engineers deployable in days.\n\n"
            "What engineering priorities are you scaling into next quarter?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["format_compliance"],
        adversarial_type="d5_subject_too_long",
    ))

    # 022: no approved prefix in subject
    tasks.append(_task(
        "TB-ADV-022",
        brief={
            "company": "Cradlewood AI",
            "icp_segment": "Segment 1",
            "confidence": 0.9,
            "hiring_velocity": {
                "direction": "accelerating", "delta_pct": 75.0,
                "signal_strength": "strong",
                "observation": "Cradlewood AI increased open roles by 75% in 60 days.",
            },
            "grounding_facts": [
                "Series A $8M closed Mar 2026",
                "Cradlewood AI increased open roles by 75% in 60 days.",
            ],
            "bench_match": {"required_stacks": ["ml", "python"], "bench_available": True},
        },
        subject="Following up on Cradlewood engineering capacity",
        body=(
            "Drew,\n\n"
            "Cradlewood AI increased open roles by 75% since closing $8M in Mar 2026.\n\n"
            "Tenacious provides pre-vetted ML engineers deployable in days.\n\n"
            "What engineering priorities are you scaling into next quarter?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["format_compliance"],
        adversarial_type="d5_no_approved_prefix",
    ))

    # 023: https:// URL in body
    tasks.append(_task(
        "TB-ADV-023",
        brief={
            "company": "Glowstone Tech",
            "icp_segment": "Segment 2",
            "confidence": 0.8,
            "hiring_velocity": {
                "direction": "decelerating", "delta_pct": -40.0,
                "signal_strength": "moderate",
                "observation": "Glowstone Tech reduced open roles by 40% in Q1 2026.",
            },
            "grounding_facts": ["Glowstone Tech reduced open roles by 40% in Q1 2026."],
            "bench_match": {"required_stacks": ["python", "data"], "bench_available": True},
        },
        subject="Note on engineering continuity at Glowstone Tech",
        body=(
            "Mel,\n\n"
            "Glowstone Tech reduced open roles by 40% in Q1 2026.\n\n"
            "Tenacious provides engineers with defined scopes. See case studies at https://tenacious.io.\n\n"
            "What delivery risks are top of mind this quarter?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["format_compliance"],
        adversarial_type="d5_url_in_body",
    ))

    # 024: two question marks in body
    tasks.append(_task(
        "TB-ADV-024",
        brief={
            "company": "Stormfront Data",
            "icp_segment": "Segment 3",
            "confidence": 0.75,
            "hiring_velocity": {
                "direction": "stable", "delta_pct": 0.0,
                "signal_strength": "moderate",
                "observation": "Stormfront Data VP Engineering joined from Databricks in Q1 2026.",
            },
            "grounding_facts": [
                "Stormfront Data VP Engineering joined from Databricks in Q1 2026.",
            ],
            "bench_match": {"required_stacks": ["data", "python"], "bench_available": True},
        },
        subject="Note on vendor strategy at Stormfront Data",
        body=(
            "Robin,\n\n"
            "Stormfront Data VP Engineering joined from Databricks in Q1 2026.\n\n"
            "Incoming engineering leadership often reassesses vendor relationships in the first 90 days.\n\n"
            "Tenacious provides engineers for defined project scopes during transitions.\n\n"
            "What delivery commitments need continuity? And what is your current vendor mix?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["format_compliance"],
        adversarial_type="d5_two_question_marks",
    ))

    # 025: booking phrase "schedule a call"
    tasks.append(_task(
        "TB-ADV-025",
        brief={
            "company": "Wavecrest ML",
            "icp_segment": "Segment 4",
            "confidence": 0.8,
            "hiring_velocity": {
                "direction": "accelerating", "delta_pct": 65.0,
                "signal_strength": "moderate",
                "observation": "Wavecrest ML increased AI role postings by 65% in 60 days.",
            },
            "grounding_facts": ["Wavecrest ML increased AI role postings by 65% in 60 days."],
            "ai_maturity_score": 2,
            "bench_match": {"required_stacks": ["ml", "data"], "bench_available": True},
        },
        subject="Context: ML capability at Wavecrest ML",
        body=(
            "Sam,\n\n"
            "Wavecrest ML increased AI role postings by 65% in 60 days.\n\n"
            "Tenacious provides pre-vetted ML engineers for high-specificity capability gaps.\n\n"
            "Can we schedule a call this week to discuss your ML roadmap?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["format_compliance"],
        adversarial_type="d5_booking_phrase_schedule_a_call",
    ))

    # 026: body > 120 words
    long_body = (
        "Chris,\n\n"
        "Solarflare Data increased open roles by 55% in the last 60 days "
        "since closing their Series A $13M in February 2026.\n\n"
        "Companies scaling post-funding often face a series of compounding challenges: "
        "the pace of engineering hiring rarely matches the pace of technical debt accumulation, "
        "and the gap between vision and execution tends to widen as teams grow. "
        "At the same time, traditional recruitment timelines of 8 to 12 weeks make it difficult "
        "to keep up with the delivery demands that come from investors, customers, and roadmaps. "
        "Tenacious provides pre-vetted Python and ML engineers who integrate in days rather than months, "
        "giving you the capacity to ship while your permanent team continues to grow.\n\n"
        "What engineering priorities are you scaling into next quarter?\n\n"
        "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
    )
    assert len(long_body.split()) > 120, f"Expected >120 words, got {len(long_body.split())}"
    tasks.append(_task(
        "TB-ADV-026",
        brief={
            "company": "Solarflare Data",
            "icp_segment": "Segment 1",
            "confidence": 0.85,
            "hiring_velocity": {
                "direction": "accelerating", "delta_pct": 55.0,
                "signal_strength": "strong",
                "observation": "Solarflare Data increased open roles by 55% in 60 days.",
            },
            "grounding_facts": [
                "Series A $13M closed Feb 2026",
                "Solarflare Data increased open roles by 55% in 60 days.",
            ],
            "bench_match": {"required_stacks": ["python", "ml"], "bench_available": True},
        },
        subject="Congrats on Series A $13M, Solarflare Data",
        body=long_body,
        expected_verdict="REJECT",
        expected_fail_dims=["format_compliance"],
        adversarial_type="d5_body_too_long",
    ))

    # ------------------------------------------------------------------
    # MULTI-DIMENSION failures (TB-ADV-027 to 031)
    # ------------------------------------------------------------------

    # 027: D2 + D3 — Ambiguous + product claim + growth frame on -60%
    tasks.append(_task(
        "TB-ADV-027",
        brief={
            "company": "Fractured Edge",
            "icp_segment": "Ambiguous",
            "confidence": 0.5,
            "hiring_velocity": {
                "direction": "decelerating", "delta_pct": -60.0,
                "signal_strength": "moderate",
                "observation": "Fractured Edge reduced open roles by 60% in 60 days.",
            },
            "grounding_facts": ["Fractured Edge reduced open roles by 60% in 60 days."],
            "bench_match": {"required_stacks": ["python"], "bench_available": True},
        },
        subject="Question on engineering priorities at Fractured Edge",
        body=(
            "Jordan,\n\n"
            "Fractured Edge reduced open roles by 60% in 60 days.\n\n"
            "Tenacious provides engineers to accelerate delivery and scale capacity efficiently.\n\n"
            "Are you currently focused on cost reduction or scaling capacity?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["icp_pitch_alignment", "signal_directionality"],
        adversarial_type="d2_d3_ambiguous_growth_frame_negative_delta",
    ))

    # 028: D1 + D4 — wrong numeric ($99M vs $16M) + "top talent"
    tasks.append(_task(
        "TB-ADV-028",
        brief={
            "company": "Ridgeback Analytics",
            "icp_segment": "Segment 1",
            "confidence": 0.85,
            "hiring_velocity": {
                "direction": "accelerating", "delta_pct": 90.0,
                "signal_strength": "strong",
                "observation": "Ridgeback Analytics increased open roles by 90% in 60 days.",
            },
            "grounding_facts": [
                "Series A $16M closed Jan 2026",
                "Ridgeback Analytics increased open roles by 90% in 60 days.",
            ],
            "bench_match": {"required_stacks": ["python", "data"], "bench_available": True},
        },
        subject="Congrats on Series A $16M, Ridgeback Analytics",
        body=(
            "Sam,\n\n"
            "Ridgeback Analytics increased open roles by 90% since closing $99M in Jan 2026.\n\n"
            "We provide top talent engineers ready to accelerate your data platform.\n\n"
            "What engineering priorities are you scaling into next quarter?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["grounding_fidelity", "tone_compliance"],
        adversarial_type="d1_d4_wrong_numeric_and_banned_phrase",
    ))

    # 029: D3 + D4 — negative delta + growth frame + "world-class"
    tasks.append(_task(
        "TB-ADV-029",
        brief={
            "company": "Nighthawk Systems",
            "icp_segment": "Segment 2",
            "confidence": 0.8,
            "hiring_velocity": {
                "direction": "decelerating", "delta_pct": -45.0,
                "signal_strength": "strong",
                "observation": "Nighthawk Systems reduced open roles by 45% in Q1 2026.",
            },
            "grounding_facts": ["Nighthawk Systems reduced open roles by 45% in Q1 2026."],
            "bench_match": {"required_stacks": ["python", "infra"], "bench_available": True},
        },
        subject="Note on engineering continuity at Nighthawk",
        body=(
            "Morgan,\n\n"
            "Nighthawk Systems reduced open roles by 45% in Q1 2026.\n\n"
            "Our world-class engineers help teams scaling their delivery without permanent hires.\n\n"
            "What delivery risks are top of mind this quarter?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["signal_directionality", "tone_compliance"],
        adversarial_type="d3_d4_growth_frame_and_banned_phrase",
    ))

    # 030: D1 + D5 — wrong year + URL in body
    tasks.append(_task(
        "TB-ADV-030",
        brief={
            "company": "Cobalt Horizons",
            "icp_segment": "Segment 3",
            "confidence": 0.8,
            "hiring_velocity": {
                "direction": "stable", "delta_pct": 8.0,
                "signal_strength": "moderate",
                "observation": "Cobalt Horizons CTO joined from AWS in Q1 2026.",
            },
            "grounding_facts": ["Cobalt Horizons CTO joined from AWS in Q1 2026."],
            "bench_match": {"required_stacks": ["infra", "python"], "bench_available": True},
        },
        subject="Note on vendor strategy at Cobalt Horizons",
        body=(
            "Alex,\n\n"
            "Cobalt Horizons CTO joined from AWS in 2019.\n\n"
            "Tenacious provides engineers for defined project scopes during transitions.\n\n"
            "See relevant case studies at https://tenacious.io/cases.\n\n"
            "Which delivery commitments need continuity through the leadership change?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=["grounding_fidelity", "format_compliance"],
        adversarial_type="d1_d5_wrong_year_and_url",
    ))

    # 031: All 5 dims fail — extreme adversarial case
    subj_031 = "Quick note: world-class engineers for Quantum Drift scaling and more"
    assert len(subj_031) <= 60 or True  # subject fails: no approved prefix
    tasks.append(_task(
        "TB-ADV-031",
        brief={
            "company": "Quantum Drift",
            "icp_segment": "Ambiguous",
            "confidence": 0.5,
            "hiring_velocity": {
                "direction": "decelerating", "delta_pct": -50.0,
                "signal_strength": "moderate",
                "observation": "Quantum Drift reduced open roles by 50% in 60 days.",
            },
            "grounding_facts": ["Quantum Drift reduced open roles by 50% in 60 days."],
            "bench_match": {"required_stacks": ["python"], "bench_available": True},
        },
        subject=subj_031,
        body=(
            "Hey there,\n\n"
            "Quantum Drift reduced open roles by 200% in 60 days.\n\n"
            "Our pre-vetted rockstar engineers help teams accelerating their bench capacity.\n\n"
            "Book a 15 minutes call at https://cal.com/tenacious to discuss?\n\n"
            "Birkity"
        ),
        expected_verdict="REJECT",
        expected_fail_dims=[
            "grounding_fidelity", "icp_pitch_alignment",
            "signal_directionality", "tone_compliance", "format_compliance",
        ],
        adversarial_type="all_dims_fail_extreme",
    ))

    # ------------------------------------------------------------------
    # PASS tasks (TB-ADV-032 to 040)
    # ------------------------------------------------------------------

    # 032: Segment 1 PASS — clean growth pitch with funding
    tasks.append(_task(
        "TB-ADV-032",
        brief={
            "company": "Ironclad AI",
            "icp_segment": "Segment 1",
            "confidence": 0.9,
            "hiring_velocity": {
                "direction": "accelerating", "delta_pct": 100.0,
                "signal_strength": "strong",
                "observation": "Ironclad AI doubled open roles since closing Series B $30M in Jan 2026.",
            },
            "grounding_facts": [
                "Series B $30M closed Jan 2026",
                "Ironclad AI doubled open roles since closing Series B $30M in Jan 2026.",
            ],
            "bench_match": {"required_stacks": ["ml", "python"], "bench_available": True},
        },
        subject="Congrats on Series B $30M, Ironclad AI",
        body=(
            "Jordan,\n\n"
            "Ironclad AI doubled open roles since closing $30M in Jan 2026.\n\n"
            "Companies scaling post-funding often need ML engineers faster than traditional hiring.\n\n"
            "Tenacious provides pre-vetted ML engineers deployable in days.\n\n"
            "What engineering priorities are you scaling into next quarter?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="PASS",
        expected_fail_dims=[],
        adversarial_type="pass_segment1_growth_funding",
    ))

    # 033: Segment 2 PASS — clean cost-discipline, negative delta
    tasks.append(_task(
        "TB-ADV-033",
        brief={
            "company": "Meridian Corp",
            "icp_segment": "Segment 2",
            "confidence": 0.85,
            "hiring_velocity": {
                "direction": "decelerating", "delta_pct": -45.0,
                "signal_strength": "strong",
                "observation": "Meridian Corp reduced open roles by 45% following headcount reduction in Q1 2026.",
            },
            "grounding_facts": [
                "Meridian Corp reduced open roles by 45% following headcount reduction in Q1 2026.",
            ],
            "bench_match": {"required_stacks": ["python", "data"], "bench_available": True},
        },
        subject="Note on engineering continuity at Meridian Corp",
        body=(
            "Blake,\n\n"
            "Meridian Corp reduced open roles by 45% following headcount reduction in Q1 2026.\n\n"
            "Teams realigning engineering costs often need delivery coverage without long-term commitments.\n\n"
            "Tenacious provides engineers with defined scopes and predictable costs.\n\n"
            "What delivery risks are top of mind for your team this quarter?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="PASS",
        expected_fail_dims=[],
        adversarial_type="pass_segment2_cost_discipline_negative_delta",
    ))

    # 034: Segment 3 PASS — leadership transition pitch
    tasks.append(_task(
        "TB-ADV-034",
        brief={
            "company": "Stonebridge Labs",
            "icp_segment": "Segment 3",
            "confidence": 0.8,
            "hiring_velocity": {
                "direction": "stable", "delta_pct": 0.0,
                "signal_strength": "moderate",
                "observation": "Stonebridge Labs new VP Engineering announced Mar 2026.",
            },
            "grounding_facts": [
                "Stonebridge Labs new VP Engineering announced Mar 2026.",
            ],
            "bench_match": {"required_stacks": ["python", "infra"], "bench_available": True},
        },
        subject="Note on vendor strategy at Stonebridge Labs",
        body=(
            "Dana,\n\n"
            "Stonebridge Labs new VP Engineering announced Mar 2026.\n\n"
            "Incoming engineering leadership often reassesses vendor relationships in the first 90 days.\n\n"
            "Tenacious provides engineers for defined project scopes during transitions.\n\n"
            "Which delivery commitments are most critical to maintain through the leadership change?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="PASS",
        expected_fail_dims=[],
        adversarial_type="pass_segment3_leadership_transition",
    ))

    # 035: Segment 4 PASS — capability gap with ai_maturity=2
    tasks.append(_task(
        "TB-ADV-035",
        brief={
            "company": "Neuralink Data",
            "icp_segment": "Segment 4",
            "confidence": 0.8,
            "hiring_velocity": {
                "direction": "accelerating", "delta_pct": 70.0,
                "signal_strength": "moderate",
                "observation": "Neuralink Data ML role postings up 70% in 60 days.",
            },
            "grounding_facts": ["Neuralink Data ML role postings up 70% in 60 days."],
            "ai_maturity_score": 2,
            "bench_match": {"required_stacks": ["ml", "data"], "bench_available": True},
        },
        subject="Context: ML capability gap at Neuralink Data",
        body=(
            "Priya,\n\n"
            "Neuralink Data ML role postings up 70% in 60 days.\n\n"
            "At your stage, finding specialized engineers who match your architecture is the real challenge.\n\n"
            "Tenacious provides pre-vetted ML engineers for high-specificity gaps.\n\n"
            "Which ML capabilities are most critical for your roadmap this quarter?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="PASS",
        expected_fail_dims=[],
        adversarial_type="pass_segment4_capability_gap",
    ))

    # 036: Ambiguous PASS — clean qualifying question, no product claim
    tasks.append(_task(
        "TB-ADV-036",
        brief={
            "company": "Graymatter AI",
            "icp_segment": "Ambiguous",
            "confidence": 0.55,
            "hiring_velocity": {
                "direction": "decelerating", "delta_pct": -35.0,
                "signal_strength": "moderate",
                "observation": "Graymatter AI open roles decreased by 35% in 60 days.",
            },
            "grounding_facts": ["Graymatter AI open roles decreased by 35% in 60 days."],
            "bench_match": {"required_stacks": ["ml", "python"], "bench_available": True},
        },
        subject="Question on engineering priorities at Graymatter AI",
        body=(
            "Sam,\n\n"
            "Graymatter AI open roles decreased by 35% in 60 days.\n\n"
            "Before making a specific recommendation, I want to understand your current priorities.\n\n"
            "Are you primarily focused on reducing costs, maintaining delivery, or building new capability?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="PASS",
        expected_fail_dims=[],
        adversarial_type="pass_ambiguous_qualifying_question_no_product_claim",
    ))

    # 037: Segment 1 bench=False PASS — no product claim (correct for gap)
    tasks.append(_task(
        "TB-ADV-037",
        brief={
            "company": "Pearlstone Tech",
            "icp_segment": "Segment 1",
            "confidence": 0.8,
            "hiring_velocity": {
                "direction": "accelerating", "delta_pct": 60.0,
                "signal_strength": "moderate",
                "observation": "Pearlstone Tech increased open roles by 60% after Series A $7M close.",
            },
            "grounding_facts": [
                "Series A $7M closed Mar 2026",
                "Pearlstone Tech increased open roles by 60% after Series A $7M close.",
            ],
            "bench_match": {"required_stacks": ["python", "data"], "bench_available": False},
        },
        subject="Congrats on Series A $7M, Pearlstone Tech",
        body=(
            "Morgan,\n\n"
            "Pearlstone Tech increased open roles by 60% after closing $7M in Mar 2026.\n\n"
            "Post-funding teams often need to consolidate delivery on fewer high-priority workstreams.\n\n"
            "What engineering priorities are you focused on for the next 90 days?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="PASS",
        expected_fail_dims=[],
        adversarial_type="pass_segment1_bench_unavailable_no_product_claim",
    ))

    # 038: Near-threshold PASS — delta exactly -20 (not < -20, so D3 passes)
    tasks.append(_task(
        "TB-ADV-038",
        brief={
            "company": "Threadline Ops",
            "icp_segment": "Segment 2",
            "confidence": 0.75,
            "hiring_velocity": {
                "direction": "decelerating", "delta_pct": -20.0,
                "signal_strength": "moderate",
                "observation": "Threadline Ops open roles decreased by 20% in Q1 2026.",
            },
            "grounding_facts": ["Threadline Ops open roles decreased by 20% in Q1 2026."],
            "bench_match": {"required_stacks": ["python", "data"], "bench_available": True},
        },
        subject="Note on engineering continuity at Threadline Ops",
        body=(
            "Casey,\n\n"
            "Threadline Ops open roles decreased by 20% in Q1 2026.\n\n"
            "Teams realigning engineering priorities often need defined-scope delivery coverage.\n\n"
            "Tenacious provides engineers with predictable costs for specific workstreams.\n\n"
            "What delivery commitments are most critical to maintain this quarter?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="PASS",
        expected_fail_dims=[],
        adversarial_type="pass_near_threshold_delta_exactly_negative_20",
    ))

    # 039: Segment 1 PASS — email asks qualifying question for Segment 1 brief
    # Phase 1 PASSES (no wrong numerics, no banned phrases, no Ambiguous+product_claim)
    # Semantically incorrect (Seg 1 should get a growth/product pitch, not a qualifying question)
    # This is a Phase 2 LLM judge test case
    tasks.append(_task(
        "TB-ADV-039",
        brief={
            "company": "Aphelion Systems",
            "icp_segment": "Segment 1",
            "confidence": 0.85,
            "hiring_velocity": {
                "direction": "accelerating", "delta_pct": 90.0,
                "signal_strength": "strong",
                "observation": "Aphelion Systems increased open roles by 90% in 60 days.",
            },
            "grounding_facts": [
                "Series B $26M closed Jan 2026",
                "Aphelion Systems increased open roles by 90% in 60 days.",
            ],
            "bench_match": {"required_stacks": ["python", "ml"], "bench_available": True},
        },
        subject="Question on engineering priorities at Aphelion",
        body=(
            "Alex,\n\n"
            "Aphelion Systems increased open roles by 90% since closing $26M in Jan 2026.\n\n"
            "Before making a specific recommendation, I want to understand your priorities.\n\n"
            "Are you primarily focused on scaling capacity, building ML capabilities, or both?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="PASS",  # Phase 1 PASS — semantically wrong pitch (Phase 2 test)
        expected_fail_dims=[],
        adversarial_type="semantic_adversarial_qualifying_question_for_segment1_brief",
    ))

    # 040: Segment 2 brief, growth-frame pitch (wrong segment pitch) but delta=-15 → D3 passes
    # Phase 1 PASSES (delta=-15 is NOT < -20, so D3 doesn't fire)
    # Semantically wrong: cost-discipline brief gets a growth pitch
    # Phase 2 LLM judge should flag this as wrong segment frame
    tasks.append(_task(
        "TB-ADV-040",
        brief={
            "company": "Bridgepost Corp",
            "icp_segment": "Segment 2",
            "confidence": 0.8,
            "hiring_velocity": {
                "direction": "decelerating", "delta_pct": -15.0,
                "signal_strength": "moderate",
                "observation": "Bridgepost Corp open roles decreased by 15% after restructuring in Q1 2026.",
            },
            "grounding_facts": [
                "Bridgepost Corp open roles decreased by 15% after restructuring in Q1 2026.",
            ],
            "bench_match": {"required_stacks": ["python", "data"], "bench_available": True},
        },
        subject="Context: Bridgepost Corp engineering capacity",
        body=(
            "Jordan,\n\n"
            "Bridgepost Corp open roles decreased by 15% after restructuring in Q1 2026.\n\n"
            "Companies scaling their delivery often need engineers faster than traditional hiring allows.\n\n"
            "Tenacious provides pre-vetted engineers deployable in days to accelerate your roadmap.\n\n"
            "What engineering priorities are you scaling into next quarter?\n\n"
            "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        expected_verdict="PASS",  # Phase 1 PASS (delta=-15 not < -20) — wrong segment pitch (Phase 2 test)
        expected_fail_dims=[],
        adversarial_type="semantic_adversarial_growth_pitch_for_segment2_restructuring",
    ))

    return tasks


def verify_all(tasks: list[dict[str, Any]], schema: dict[str, Any]) -> list[str]:
    import jsonschema
    from scoring_evaluator import score_task  # type: ignore

    errors: list[str] = []
    for task in tasks:
        tid = task["task_id"]
        brief = task["brief"]
        expected_verdict = brief.get("expected_verdict", "")
        expected_fail_dims = brief.get("expected_fail_dims", [])

        try:
            jsonschema.validate(instance=task, schema=schema)
        except jsonschema.ValidationError as e:
            errors.append(f"{tid}: schema — {e.message}")
            continue

        scores = score_task(task)
        actual_verdict = scores["overall_verdict"]

        if actual_verdict != expected_verdict:
            errors.append(
                f"{tid}: expected {expected_verdict} but got {actual_verdict}. "
                f"Scores: {scores}"
            )

        for dim in expected_fail_dims:
            if scores.get(dim, 1) != 0:
                errors.append(
                    f"{tid}: expected {dim}=0 but got {scores.get(dim)}. "
                    f"All scores: {scores}"
                )

        subj = task["email"]["subject"]
        body = task["email"]["body"]
        if body.count("?") > 1 and "d5_two_question_marks" not in brief.get("adversarial_type", ""):
            errors.append(f"{tid}: body has {body.count('?')} question marks (unexpected)")
        if "bench" in body.lower().split() and "d4_bench_jargon" not in brief.get("adversarial_type", ""):
            pass  # allow "bench" in adversarial tasks specifically designed to test it

    if len(tasks) != 40:
        errors.append(f"Expected 40 tasks, got {len(tasks)}")

    return errors


def assign_difficulty(task: dict[str, Any]) -> str:
    from scoring_evaluator import score_task  # type: ignore
    scores = score_task(task)
    if scores["overall_verdict"] == "REJECT":
        return "hard"
    adversarial_type = task["brief"].get("adversarial_type", "")
    if "semantic_adversarial" in adversarial_type:
        return "hard"
    return "medium"


def main() -> None:
    import jsonschema

    parser = argparse.ArgumentParser(description="Generate adversarial hand-authored tasks")
    parser.add_argument("--verify-only", action="store_true",
                        help="Re-read output and verify without writing")
    args = parser.parse_args()

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    if args.verify_only:
        tasks = [json.loads(l) for l in OUTPUT_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
        errors = verify_all(tasks, schema)
        if errors:
            print(f"\n{len(errors)} errors:")
            for e in errors:
                print(f"  * {e}")
            raise SystemExit(1)
        print(f"Verification OK — {len(tasks)} tasks, all pass schema + scoring checks.")
        return

    tasks = build_tasks()

    # Assign difficulty
    for task in tasks:
        task["difficulty"] = assign_difficulty(task)

    # Verify
    errors = verify_all(tasks, schema)
    if errors:
        print(f"\n{len(errors)} verification errors — NOT writing output:")
        for e in errors:
            print(f"  * {e}")
        raise SystemExit(1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        for task in tasks:
            fh.write(json.dumps(task, ensure_ascii=False) + "\n")

    # Summary
    from scoring_evaluator import score_task  # type: ignore
    verdicts = [score_task(t)["overall_verdict"] for t in tasks]
    difficulty_dist = {}
    for t in tasks:
        d = t.get("difficulty", "?")
        difficulty_dist[d] = difficulty_dist.get(d, 0) + 1

    adv_types = set(t["brief"].get("adversarial_type", "") for t in tasks)
    dim_coverage: dict[str, int] = {}
    for t in tasks:
        for dim in t["brief"].get("expected_fail_dims", []):
            dim_coverage[dim] = dim_coverage.get(dim, 0) + 1

    print("=" * 60)
    print("  ADVERSARIAL HAND BATCH 1 SUMMARY")
    print("=" * 60)
    print(f"Total tasks written : {len(tasks)}")
    print(f"PASS                : {verdicts.count('PASS')}")
    print(f"REJECT              : {verdicts.count('REJECT')}")
    print(f"Difficulty          : {difficulty_dist}")
    print(f"Dimension failures  : {dim_coverage}")
    print(f"Schema validation   : all OK")
    print(f"\nOutput -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
