"""
Build SimPO preference pairs from the Tenacious-Bench train partition.

Reads:  data/tenacious_bench_v0.1/train/tasks.jsonl  (114 tasks)
Writes: training_data/tenacious_judge_train.jsonl     (114 JSONL lines)

Each output line has three keys: prompt, chosen, rejected.

REJECT tasks (71):
  chosen   = VERDICT: REJECT  + dimension name + brief-grounded reason
  rejected = VERDICT: PASS    + surface assertion (no grounding)

PASS tasks (43):
  chosen   = VERDICT: PASS    + grounding confirmation
  rejected = VERDICT: REJECT  + plausible-but-wrong dimension/reason
             (deterministically rotated via hash of task_id — reproducible)

Usage:
    python scripts/generation/build_preference_pairs.py
    python scripts/generation/build_preference_pairs.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))
from scoring_evaluator import score_task  # noqa: E402

TRAIN_FILE  = ROOT / "data" / "tenacious_bench_v0.1" / "train" / "tasks.jsonl"
OUTPUT_FILE = ROOT / "training_data" / "tenacious_judge_train.jsonl"

# Maps the D-code returned by score_task() to the full dimension name
DIM_NAMES: dict[str, str] = {
    "D1": "grounding_fidelity",
    "D2": "icp_pitch_alignment",
    "D3": "signal_directionality",
    "D4": "tone_compliance",
    "D5": "format_compliance",
}

# Plausible-but-wrong critique templates used as the *rejected* response for PASS tasks.
# Deliberately surface-level and not grounded in any specific brief fact.
# Rotated deterministically via hash(task_id) so output is reproducible.
_FAKE: list[tuple[str, str]] = [
    (
        "grounding_fidelity",
        "A numeric reference in the email body cannot be traced to a grounding fact in the brief.",
    ),
    (
        "icp_pitch_alignment",
        "Pitch frame implies a growth posture inconsistent with the stated ICP segment context.",
    ),
    (
        "signal_directionality",
        "Email body uses acceleration language while the hiring velocity direction is ambiguous or declining.",
    ),
    (
        "tone_compliance",
        "Phrasing positions the prospect's situation as a failure, approaching a condescending frame.",
    ),
    (
        "format_compliance",
        "Subject line does not cleanly separate the approved prefix from the company reference within the character limit.",
    ),
]


def _fake_rejection(task_id: str) -> tuple[str, str]:
    idx = int(hashlib.md5(task_id.encode()).hexdigest(), 16) % len(_FAKE)
    return _FAKE[idx]


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


_PROMPT = (
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


def build_pair(task: dict) -> dict:
    scores  = score_task(task)
    verdict = scores["verdict"]
    brief   = task.get("brief", {})
    email   = task.get("email", {})
    hv      = brief.get("hiring_velocity", {})

    prompt = _PROMPT.format(
        brief_text=_brief_text(task),
        subject=email.get("subject", ""),
        body=email.get("body", "").strip(),
        delta=float(hv.get("delta_pct", 0)),
        segment=brief.get("icp_segment", "Unknown"),
    )

    if verdict == "REJECT":
        dim_code = scores.get("failed_dimension") or "D1"
        dim_name = DIM_NAMES.get(dim_code, dim_code)
        reason   = scores.get("reason", "Rubric dimension check failed.")

        chosen = (
            f"VERDICT: REJECT\n"
            f"Primary failure: {dim_name}\n"
            f"Reason: {reason}"
        )
        rejected = (
            "VERDICT: PASS\n"
            "All checks satisfied. Email is acceptable to send."
        )
    else:
        fake_dim, fake_reason = _fake_rejection(task["task_id"])

        chosen = (
            "VERDICT: PASS\n"
            "All grounding facts verified. No pitch or signal mismatch."
        )
        rejected = (
            f"VERDICT: REJECT\n"
            f"Primary failure: {fake_dim}\n"
            f"Reason: {fake_reason}"
        )

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print first 3 pairs without writing output file",
    )
    args = parser.parse_args()

    tasks = [
        json.loads(line)
        for line in TRAIN_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"Loaded {len(tasks)} train tasks from {TRAIN_FILE}")

    pairs: list[dict] = []
    reject_chosen = 0
    pass_chosen   = 0
    errors: list[str] = []

    for task in tasks:
        try:
            pair = build_pair(task)
            pairs.append(pair)
            if pair["chosen"].startswith("VERDICT: REJECT"):
                reject_chosen += 1
            else:
                pass_chosen += 1
        except Exception as exc:
            errors.append(f"{task.get('task_id', '?')}: {exc}")

    print(
        f"Pairs built : {len(pairs)}  "
        f"(REJECT-chosen={reject_chosen}, PASS-chosen={pass_chosen})"
    )
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")

    if args.dry_run:
        print("\n--- Sample pairs (first 3) ---")
        for p in pairs[:3]:
            prompt_preview = "\n".join(p["prompt"].splitlines()[:5])
            print(f"\nPROMPT (first 5 lines):\n{prompt_preview}")
            print(f"CHOSEN  : {p['chosen'].replace(chr(10), ' | ')}")
            print(f"REJECTED: {p['rejected'].replace(chr(10), ' | ')}")
        print("\nDry run — no files written.")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Written  -> {OUTPUT_FILE}")
    print(f"Lines    -> {len(pairs)}")


if __name__ == "__main__":
    main()
