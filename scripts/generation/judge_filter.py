"""
LLM-as-a-Judge quality filter — Act II, Days 2-3.

Reads every task in trace_derived_batch1.jsonl and scores it on three
pointwise quality dimensions (1-5 each).  Adds `judge_filter` and
`difficulty` fields then rewrites the file in-place.

Model: deepseek/deepseek-v3.2  (different family from Gemini used for generation)
Preference-leakage prevention: generation model != judge model (per methodology.md).

Pass threshold: all three judge scores >= 3.
Tasks scoring < 3 on any dimension are flagged but kept (human review required).

Usage (from project root):
    python scripts/generation/judge_filter.py
    python scripts/generation/judge_filter.py --dry-run   # print prompts, no API calls
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))
from scoring_evaluator import score_task  # type: ignore

INPUT_FILE = ROOT / "data" / "tenacious_bench_v0.1" / "dev" / "trace_derived_batch1.jsonl"
SCHEMA_PATH = ROOT / "benchmark" / "schema.json"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL = "openai/gpt-4o-mini"   # Different family from Gemini (generation model); cheaper than deepseek-v3.2
PASS_THRESHOLD = 3                        # score >= 3 on all 3 dims = passes filter

# ---------------------------------------------------------------------------
# Difficulty assignment (deterministic, no LLM needed)
# ---------------------------------------------------------------------------

def assign_difficulty(task: dict, scores: dict) -> str:
    """
    Assign difficulty based on variant type and which dimensions fail.

    easy   — Task A, all 5 dims pass, Segment 1/2/3/4 brief (clear signal, aligned pitch)
    medium — Task A PASS with Ambiguous brief (qualifying-question pattern, subtle)
             OR Task B (D1 numeric-corruption test)
    hard   — Task A REJECT on any dimension (real failure modes from Week 10)
             OR Task C (D2 isolation test: Ambiguous brief + product claim)
    """
    variant = task["task_id"][-1]          # "A", "B", or "C"
    segment = task["brief"].get("icp_segment", "")

    if variant == "C":
        return "hard"

    if variant == "B":
        return "medium"

    # variant == "A"
    if scores["overall_verdict"] == "REJECT":
        return "hard"

    # PASS — easy unless the brief is Ambiguous (qualifying question is subtler to judge)
    if segment.lower() == "ambiguous":
        return "medium"

    return "easy"


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

# Context injected once so individual task prompts stay short.
_JUDGE_SYSTEM = """\
You are a benchmark quality auditor for Tenacious-Bench v0.1 — a machine-verifiable
B2B sales-agent evaluation dataset built from Tenacious Intelligence Corporation
outbound sales traces.

== WHAT THE BENCHMARK TESTS ==
Each task is an (input, candidate_output) pair:
  - input  : a signal brief (hiring velocity, ICP segment, grounding facts, bench summary)
  - output : a cold outreach email the sales agent drafted

The email is evaluated on FIVE rubric dimensions (all machine-checkable in Phase 1):
  D1 grounding_fidelity      — every numeric/factual claim traces to the brief
  D2 icp_pitch_alignment     — pitch frame matches ICP segment (fast-fail: Ambiguous + product claim)
  D3 signal_directionality   — pitch direction matches hiring velocity
  D4 tone_compliance         — no Tenacious banned phrases; no "bench" externally
  D5 format_compliance       — subject ≤60 chars, body ≤120 words, no URL, 1 question only

Verdict: PASS if all five = 1; REJECT if any = 0.

== FOUR ICP SEGMENTS ==
  Segment 1 — Series A/B startup, growth/post-funding pitch
  Segment 2 — Mid-market restructuring, cost-discipline pitch
  Segment 3 — Engineering leadership transition, vendor-reassessment pitch
  Segment 4 — Specialized capability gap (requires AI maturity >= 2)
  Ambiguous — unclear ICP; correct response = qualifying question, NO product claim

== TENACIOUS TONE RULES (summarised) ==
Direct: ≤120 words cold, ≤60 char subject with approved prefix (Context:/Note on/Congrats on/Question on)
Grounded: every claim supported by hiring signal brief or competitor data
Honest: never over-commit bench capacity; never invent funding rounds
Professional: no "bench" externally; no "top talent"/"world-class"/"rockstar" etc.
Non-condescending: frame gaps as research findings, not prospect failures

== TASK VARIANTS YOU WILL SEE ==
  A  — original email (may be PASS or REJECT)
  B  — intentional D1 failure: one numeric fact in the email contradicts the brief
  C  — intentional D2 failure: brief is Ambiguous but email makes a product claim (or
       brief segment was changed to Ambiguous while email retains Segment-1 pitch claims)

== YOUR JOB ==
Score the QUALITY OF THE TASK AS A BENCHMARK ITEM (not whether the email is "good").
Ask: "Would a benchmark researcher include this task in an evaluation set?"

Score each dimension 1–5:
  5 = excellent benchmark item
  4 = good, minor quibbles
  3 = acceptable, passes the filter
  2 = borderline, human review needed
  1 = reject from dataset

Dimensions:
  input_coherence            — Is the signal brief plausible and internally consistent?
                               Are the hiring velocity, ICP segment, grounding facts, and bench
                               summary a coherent scenario a real B2B sales researcher might face?

  ground_truth_verifiability — Given the brief + rubric, can the correct PASS/REJECT verdict
                               and the specific failing dimension(s) be determined objectively
                               from the provided evidence alone, without guessing?
                               For variant B/C (designed failures), is the intended failure
                               clearly traceable?

  rubric_application_clarity — Would two independent reviewers applying the five rubric
                               dimensions mechanically reach the same verdict on this task?
                               Deduct for ambiguity in which dimension fires.

Output ONLY valid JSON — no markdown, no prose:
{"input_coherence": <1-5>, "ground_truth_verifiability": <1-5>, "rubric_application_clarity": <1-5>, "notes": "<one sentence max>"}
"""

_JUDGE_USER = """\
Rate this Tenacious-Bench task.  Variant: {variant}  Expected primary failure dimension: {expected_fail}

--- BRIEF ---
{brief_json}

--- EMAIL ---
Subject: {subject}
Body:
{body}

--- SCORING RESULT (from deterministic evaluator) ---
{scores_json}

Score the task quality on the three dimensions.  Output only JSON.
"""


def _expected_fail(variant: str, scores: dict) -> str:
    if variant == "B":
        return "D1 grounding_fidelity (numeric fact in email contradicts brief)"
    if variant == "C":
        return "D2 icp_pitch_alignment (Ambiguous brief + product claim fast-fail)"
    failing = [k for k, v in scores.items() if k != "overall_verdict" and v == 0]
    return ", ".join(failing) if failing else "none (PASS task)"


def build_prompt(task: dict, scores: dict) -> str:
    variant = task["task_id"][-1]
    return _JUDGE_USER.format(
        variant=variant,
        expected_fail=_expected_fail(variant, scores),
        brief_json=json.dumps(task["brief"], indent=2),
        subject=task["email"]["subject"],
        body=task["email"]["body"],
        scores_json=json.dumps(scores, indent=2),
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_judge(client: OpenAI, prompt: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,   # deterministic judge
                max_tokens=200,
            )
            raw = (resp.choices[0].message.content or "").strip()

            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\s*```$",          "", raw, flags=re.MULTILINE)

            data = json.loads(raw)
            # Validate required keys
            for key in ("input_coherence", "ground_truth_verifiability", "rubric_application_clarity"):
                if key not in data:
                    raise ValueError(f"Missing key: {key}")
                val = int(data[key])
                if not (1 <= val <= 5):
                    raise ValueError(f"{key}={val} out of range")
                data[key] = val
            data.setdefault("notes", "")
            return data

        except Exception as exc:
            if attempt < retries - 1:
                print(f"    -> judge attempt {attempt + 1} failed: {exc}")
                time.sleep(3)
            else:
                raise
    raise RuntimeError("Judge failed after retries")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = False, input_file: str | None = None) -> None:
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    target = Path(input_file) if input_file else INPUT_FILE

    tasks: list[dict] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            tasks.append(json.loads(line))

    print(f"Loaded {len(tasks)} tasks from {target}")
    print(f"Judge model : {JUDGE_MODEL}")
    print(f"Pass threshold : all dims >= {PASS_THRESHOLD}")
    print(f"Dry run : {dry_run}\n")

    client: OpenAI | None = None
    if not dry_run:
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise SystemExit("OPENROUTER_API_KEY not set in .env")
        client = OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)

    enriched: list[dict] = []
    flagged: list[str] = []

    for i, task in enumerate(tasks, 1):
        tid = task["task_id"]
        variant = tid[-1]

        # Deterministic scores (already computed by scoring_evaluator)
        scores = score_task(task)

        # Difficulty label (no LLM needed)
        difficulty = assign_difficulty(task, scores)

        if dry_run:
            prompt = build_prompt(task, scores)
            print(f"[{i:03d}] {tid}  variant={variant}  difficulty={difficulty}")
            print("  prompt preview:", prompt[:120].replace("\n", " "), "...")
            task_out = {**task, "difficulty": difficulty}
            enriched.append(task_out)
            continue

        print(f"[{i:03d}/{len(tasks)}] Judging {tid}  ({variant}, {difficulty})...", end=" ", flush=True)

        prompt = build_prompt(task, scores)
        jf = call_judge(client, prompt)

        passed = all(jf[k] >= PASS_THRESHOLD for k in
                     ("input_coherence", "ground_truth_verifiability", "rubric_application_clarity"))
        jf["passed"] = passed
        jf["model"]  = JUDGE_MODEL

        status = "PASS" if passed else "FLAG"
        print(f"IC={jf['input_coherence']} GTV={jf['ground_truth_verifiability']} RAC={jf['rubric_application_clarity']}  [{status}]")

        if not passed:
            flagged.append(tid)

        task_out = {**task, "difficulty": difficulty, "judge_filter": jf}
        enriched.append(task_out)

        # Polite rate-limit pause
        if i % 10 == 0:
            time.sleep(1)

    # ------------------------------------------------------------------
    # Schema-validate enriched tasks
    # ------------------------------------------------------------------
    schema_errors: list[str] = []
    if not dry_run:
        for t in enriched:
            try:
                jsonschema.validate(instance=t, schema=schema)
            except jsonschema.ValidationError as e:
                schema_errors.append(f"{t['task_id']}: {e.message}")

    # ------------------------------------------------------------------
    # Write back in-place
    # ------------------------------------------------------------------
    with target.open("w", encoding="utf-8") as fh:
        for t in enriched:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  JUDGE FILTER SUMMARY")
    print(f"{'='*60}")
    print(f"Total tasks processed : {len(enriched)}")

    if not dry_run:
        passed_count = sum(1 for t in enriched if t.get("judge_filter", {}).get("passed", False))
        flagged_count = len(flagged)

        # Score distribution
        for dim in ("input_coherence", "ground_truth_verifiability", "rubric_application_clarity"):
            vals = [t["judge_filter"][dim] for t in enriched]
            avg = sum(vals) / len(vals)
            dist = {s: vals.count(s) for s in range(1, 6)}
            print(f"  {dim:<32} avg={avg:.2f}  dist={dist}")

        # Difficulty distribution
        diff_dist = {}
        for t in enriched:
            d = t.get("difficulty", "?")
            diff_dist[d] = diff_dist.get(d, 0) + 1
        print(f"\nDifficulty distribution : {diff_dist}")

        print(f"\nPassed filter ({PASS_THRESHOLD}+ on all dims) : {passed_count}/{len(enriched)}")
        if flagged_count:
            print(f"Flagged for review : {flagged_count}")
            for tid in flagged:
                t = next(x for x in enriched if x["task_id"] == tid)
                jf = t["judge_filter"]
                print(f"  * {tid}  IC={jf['input_coherence']} GTV={jf['ground_truth_verifiability']} RAC={jf['rubric_application_clarity']}  notes: {jf['notes']}")
        else:
            print("No tasks flagged.")

        if schema_errors:
            print(f"\nSchema validation errors ({len(schema_errors)}):")
            for e in schema_errors:
                print(f"  * {e}")
        else:
            print("Schema validation : all OK")
    else:
        diff_dist = {}
        for t in enriched:
            d = t.get("difficulty", "?")
            diff_dist[d] = diff_dist.get(d, 0) + 1
        print(f"Difficulty distribution (dry run) : {diff_dist}")

    print(f"\nOutput -> {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Assign difficulty labels only; skip LLM judge calls")
    parser.add_argument("--input-file", default=None,
                        help="Path to a .jsonl file to judge-filter (default: trace_derived_batch1.jsonl)")
    args = parser.parse_args()
    main(dry_run=args.dry_run, input_file=args.input_file)
