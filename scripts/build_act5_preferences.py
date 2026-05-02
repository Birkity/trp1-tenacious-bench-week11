"""
build_act5_preferences.py — Dimension-reasoning preference pairs for Act V SimPO retraining.

Fixes preference leakage in the Act III/IV training data where the model learned
short verdict-sentence style rather than dimension-level judging logic.

Source:  data/tenacious_bench_v0.1/train/tasks.jsonl  (114 tasks)
Output:  training_data/tenacious_judge_act5_pairs.jsonl  (~228 pairs)

Per task, builds TWO pairs that share the same chosen (correct D1–D5 reasoning)
but have different rejected variants:

  Pair A — aggressive wrong
    REJECT tasks: all dims PASS → wrong PASS verdict
    PASS tasks:   one dim fabricated FAIL → wrong REJECT verdict

  Pair B — subtle wrong (higher signal)
    REJECT tasks: correct REJECT verdict but wrong primary failure dim
    PASS tasks:   different fabricated FAIL dim from Pair A → wrong REJECT verdict

This forces the model to learn WHICH dimension fails, not just whether to REJECT.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))

from scoring_evaluator import (  # noqa: E402
    score_grounding_fidelity,
    score_icp_pitch_alignment,
    score_signal_directionality,
    score_tone_compliance,
    score_format,
)

TRAIN_FILE  = ROOT / "data" / "tenacious_bench_v0.1" / "train" / "tasks.jsonl"
OUTPUT_FILE = ROOT / "training_data" / "tenacious_judge_act5_pairs.jsonl"

DIM_ORDER = ("D1", "D2", "D3", "D4", "D5")

# ── Judge prompt (matches run_ablation.py exactly) ───────────────────────────

_PROMPT_TEMPLATE = (
    "You are a sales quality judge for Tenacious Intelligence Corporation.\n"
    "Given a hiring signal brief and a generated outreach email,\n"
    "return a structured verdict.\n"
    "\n"
    "Score these dimensions:\n"
    "D1 — grounding_fidelity: do the email's numerics and claims trace to the brief?\n"
    "D2 — icp_pitch_alignment: does the pitch frame match the ICP segment?\n"
    "D3 — signal_directionality: does the growth/contraction frame match hiring velocity?\n"
    "D4 — tone_compliance: are there banned phrases or tone violations?\n"
    "D5 — format_compliance: does the email meet subject/body/structure rules?\n"
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
    for fact in brief.get("grounding_facts", []):
        lines.append(f"  - {fact.strip()}")
    return "\n".join(lines)


def _build_prompt(task: dict) -> str:
    brief = task.get("brief", {})
    email = task.get("email", {})
    hv    = brief.get("hiring_velocity", {})
    return _PROMPT_TEMPLATE.format(
        brief_text = _brief_text(task),
        subject    = email.get("subject", ""),
        body       = email.get("body", "").strip(),
        delta      = float(hv.get("delta_pct", 0)),
        segment    = brief.get("icp_segment", "Unknown"),
    )


# ── Plausible-but-wrong reason pools ─────────────────────────────────────────

_FAKE_PASS: dict[str, list[str]] = {
    "D1": [
        "email references specific company context consistent with brief",
        "numerics in the body align with available grounding evidence",
        "grounding facts are represented in the email body",
        "no fabricated claims detected; evidence traceable to brief",
    ],
    "D2": [
        "ICP segment frame appears consistent with expected approach",
        "pitch aligns with segment requirements for this company type",
        "segment-appropriate framing applied throughout the email",
        "product claim is appropriate given the segment and context",
    ],
    "D3": [
        "velocity framing is broadly consistent with hiring signal",
        "growth context is proportionate to the available delta data",
        "signal directionality is consistent across brief and email",
        "email framing is neutral enough to apply regardless of delta",
    ],
    "D4": [
        "no compliance issues detected in subject or body",
        "tone is professional and acceptable throughout",
        "no banned phrases present in the copy",
        "language is within acceptable Tenacious style standards",
    ],
    "D5": [
        "format is within all specified bounds",
        "structure meets subject, body, and prefix requirements",
        "subject and body comply with length and formatting rules",
        "single question present; no structural violations found",
    ],
}

_FAKE_FAIL: dict[str, list[str]] = {
    "D1": [
        "dollar amount in email does not appear in brief grounding facts",
        "email references a funding round not documented in the brief",
        "numeric claim in body cannot be traced to evidence in the brief",
        "company detail cited in email is absent from grounding_facts field",
    ],
    "D2": [
        "pitch frame does not match the required approach for this ICP segment",
        "product claim is present but segment context does not support it here",
        "segment-appropriate framing is absent from the email body",
        "email uses wrong segment frame — contraction context requires different pitch",
    ],
    "D3": [
        "growth-frame language used despite clearly declining hiring signal",
        "email implies acceleration while brief shows velocity contraction",
        "velocity direction in brief contradicts the email's narrative frame",
        "scaling terminology present despite negative delta exceeding threshold",
    ],
    "D4": [
        "phrase in subject or body matches a Tenacious style-guide banned term",
        "tone approaches a condescending or urgency-pressure framing",
        "language in body violates the professional tone requirement",
        "email body contains internal jargon not appropriate for prospect-facing copy",
    ],
    "D5": [
        "subject line exceeds 60-character limit or lacks an approved prefix",
        "email body contains more than one question mark",
        "meeting-booking or scheduling language detected in email body",
        "body word count exceeds the 120-word maximum",
    ],
}


def _pick(pool: list[str], task_id: str, seed: str) -> str:
    """Deterministic pick from pool using md5 hash of task_id + seed."""
    idx = int(hashlib.md5(f"{task_id}_{seed}".encode()).hexdigest(), 16) % len(pool)
    return pool[idx]


# ── Dimension scorer ─────────────────────────────────────────────────────────

def _score_all_dims(task: dict) -> dict[str, tuple[int, str]]:
    """Run all 5 individual scorers. Returns {D1: (score, reason), ...}."""
    brief = {
        **task["brief"],
        "_bench_summary": task.get("bench_summary", ""),
        "_prior_thread":  task.get("prior_thread", ""),
        "_rubric":        task.get("rubric", {}),
    }
    email = {**task["email"], "_rubric": task.get("rubric", {})}
    return {
        "D1": score_grounding_fidelity(email, brief),
        "D2": score_icp_pitch_alignment(brief, email, use_llm=False),
        "D3": score_signal_directionality(brief, email),
        "D4": score_tone_compliance(email, brief),
        "D5": score_format(email),
    }


# ── Chosen / rejected builders ───────────────────────────────────────────────

def _build_chosen(dims: dict[str, tuple[int, str]]) -> str:
    """Ground-truth: all five dimensions with actual PASS/FAIL + scorer reason."""
    lines = []
    for d in DIM_ORDER:
        score, reason = dims[d]
        lines.append(f"{d}: {'PASS' if score else 'FAIL'} — {reason}")

    failed = [d for d in DIM_ORDER if dims[d][0] == 0]
    verdict = "REJECT" if failed else "PASS"
    lines.append(f"VERDICT: {verdict}")
    if failed:
        lines.append(f"Primary failure: {failed[0]}")
    return "\n".join(lines)


def _build_rejected_aggressive(
    dims: dict[str, tuple[int, str]],
    task_id: str,
    verdict: str,
) -> str:
    """
    Pair A rejected.
    REJECT tasks: claim every dim passes → wrong PASS verdict.
    PASS tasks:   fabricate one failure on a random dim → wrong REJECT verdict.
    """
    lines = []
    if verdict == "REJECT":
        for d in DIM_ORDER:
            reason = _pick(_FAKE_PASS[d], task_id, f"aggP_{d}")
            lines.append(f"{d}: PASS — {reason}")
        lines.append("VERDICT: PASS")
    else:
        fake_fail = _pick(list(DIM_ORDER), task_id, "aggF_target")
        for d in DIM_ORDER:
            if d == fake_fail:
                reason = _pick(_FAKE_FAIL[d], task_id, f"aggF_{d}")
                lines.append(f"{d}: FAIL — {reason}")
            else:
                reason = _pick(_FAKE_PASS[d], task_id, f"aggP_{d}")
                lines.append(f"{d}: PASS — {reason}")
        lines.append("VERDICT: REJECT")
        lines.append(f"Primary failure: {fake_fail}")
    return "\n".join(lines)


def _build_rejected_subtle(
    dims: dict[str, tuple[int, str]],
    task_id: str,
    verdict: str,
) -> str:
    """
    Pair B rejected — higher training signal.
    REJECT tasks: correct REJECT verdict but blames a DIFFERENT (wrong) primary dim.
                  The real failing dim is quietly flipped to PASS.
    PASS tasks:   fabricate a failure on a DIFFERENT dim than Pair A → wrong REJECT.
    """
    lines = []
    if verdict == "REJECT":
        real_fail = next(d for d in DIM_ORDER if dims[d][0] == 0)
        other_dims = [d for d in DIM_ORDER if d != real_fail]
        fake_fail = _pick(other_dims, task_id, "subF_target")

        for d in DIM_ORDER:
            if d == real_fail:
                # Flip the actual failure to PASS
                reason = _pick(_FAKE_PASS[d], task_id, f"subFlip_{d}")
                lines.append(f"{d}: PASS — {reason}")
            elif d == fake_fail:
                # Falsely blame a different dim
                reason = _pick(_FAKE_FAIL[d], task_id, f"subF_{d}")
                lines.append(f"{d}: FAIL — {reason}")
            else:
                score, reason = dims[d]
                lines.append(f"{d}: {'PASS' if score else 'FAIL'} — {reason}")

        lines.append("VERDICT: REJECT")  # verdict is right but primary failure is wrong
        lines.append(f"Primary failure: {fake_fail}")
    else:
        # PASS task: different fake failure dim than Pair A
        a_fake = _pick(list(DIM_ORDER), task_id, "aggF_target")  # same seed as Pair A
        other_dims = [d for d in DIM_ORDER if d != a_fake]
        fake_fail = _pick(other_dims, task_id, "subF_target")

        for d in DIM_ORDER:
            if d == fake_fail:
                reason = _pick(_FAKE_FAIL[d], task_id, f"subF_{d}")
                lines.append(f"{d}: FAIL — {reason}")
            else:
                reason = _pick(_FAKE_PASS[d], task_id, f"aggP_{d}")
                lines.append(f"{d}: PASS — {reason}")
        lines.append("VERDICT: REJECT")
        lines.append(f"Primary failure: {fake_fail}")
    return "\n".join(lines)


# ── Per-task pair builder ─────────────────────────────────────────────────────

def build_pairs(task: dict) -> list[dict]:
    task_id = task.get("task_id", "UNKNOWN")
    prompt  = _build_prompt(task)
    dims    = _score_all_dims(task)
    verdict = "REJECT" if any(s == 0 for s, _ in dims.values()) else "PASS"

    chosen    = _build_chosen(dims)
    rejected_a = _build_rejected_aggressive(dims, task_id, verdict)
    rejected_b = _build_rejected_subtle(dims, task_id, verdict)

    return [
        {"prompt": prompt, "chosen": chosen, "rejected": rejected_a},
        {"prompt": prompt, "chosen": chosen, "rejected": rejected_b},
    ]


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    tasks = [
        json.loads(l)
        for l in TRAIN_FILE.read_text("utf-8").splitlines()
        if l.strip()
    ]
    print(f"Loaded {len(tasks)} tasks from {TRAIN_FILE.relative_to(ROOT)}")

    all_pairs: list[dict] = []
    errors: list[str]    = []

    for task in tasks:
        try:
            all_pairs.extend(build_pairs(task))
        except Exception as exc:
            errors.append(f"{task.get('task_id', '?')}: {exc}")

    if errors:
        print(f"\n{len(errors)} errors:")
        for e in errors[:5]:
            print(f"  {e}")

    reject_chosen = sum(1 for p in all_pairs if "VERDICT: REJECT" in p["chosen"])
    pass_chosen   = len(all_pairs) - reject_chosen

    print(f"\nTotal pairs written : {len(all_pairs)}")
    print(f"REJECT-chosen       : {reject_chosen}")
    print(f"PASS-chosen         : {pass_chosen}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        for pair in all_pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nWritten -> {OUTPUT_FILE.relative_to(ROOT)}")

    # ── Sample inspection ────────────────────────────────────────────────────
    sample_task = tasks[0]
    sample_dims = _score_all_dims(sample_task)
    sample_verdict = "REJECT" if any(s == 0 for s, _ in sample_dims.values()) else "PASS"
    pairs_a_b = build_pairs(sample_task)

    sep = "=" * 64
    print(f"\n{sep}")
    print(f"SAMPLE PAIR  task_id={sample_task.get('task_id')}  verdict={sample_verdict}")
    print(sep)
    print("[CHOSEN — ground truth reasoning]")
    print(pairs_a_b[0]["chosen"])
    print()
    print("[REJECTED A — aggressive wrong]")
    print(pairs_a_b[0]["rejected"])
    print()
    print("[REJECTED B — subtle wrong (right verdict, wrong primary dim)]")
    print(pairs_a_b[1]["rejected"])
    print(sep)


if __name__ == "__main__":
    main()
