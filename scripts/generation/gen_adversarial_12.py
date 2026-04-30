"""
Generate 12 hardcoded adversarial edge-case tasks (TB-ADV-041 to TB-ADV-052).

Each task targets exactly one rubric dimension failure:
  TB-ADV-041 D1 — fabricated numeric ($5M) not in brief
  TB-ADV-042 D3 — negative velocity (-30%) paired with growth-frame copy
  TB-ADV-043 D1 — invented headcount reduction (45%) with no grounding in brief
  TB-ADV-044 D4 — banned phrase "falling behind" (condescension violation)
  TB-ADV-045 D2 — Ambiguous segment, body does not end with ?
  TB-ADV-046 D1 — fabricated productivity statistic (340%) not in brief
  TB-ADV-047 D4 — banned phrases "world-class" + "just "
  TB-ADV-048 D4 — word "bench" used in prospect-facing copy
  TB-ADV-049 D1 — wrong funding amount ($99M) vs brief ($9M)
  TB-ADV-050 D5 — meeting-booking phrase "book a call"
  TB-ADV-051 D1 — misleading metric ($30M) vs brief ($15M)
  TB-ADV-052 PASS — Phase 2 semantic probe: passes D1-D5 but is generically weak

Usage:
    python scripts/generation/gen_adversarial_12.py
    python scripts/generation/gen_adversarial_12.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))
from scoring_evaluator import score_task, _base_rubric  # noqa: E402

OUTPUT = ROOT / "data" / "tenacious_bench_v0.1" / "dev" / "adversarial_hand_batch1.jsonl"

RUBRIC = _base_rubric()
BENCH_SUMMARY = (
    "Python: 7 available (7-day deploy). "
    "ML: 4 available (14-day deploy). "
    "Go: 3 available (7-day deploy). "
    "Infra: 5 available (7-day deploy)."
)
JUDGE_FILTER = {
    "input_coherence": 4,
    "ground_truth_verifiability": 5,
    "rubric_application_clarity": 5,
}


def _task(
    tid: str,
    company: str,
    segment: str,
    delta_pct: float,
    observation: str,
    grounding_facts: list[str],
    subject: str,
    body: str,
    bench_available: bool = True,
    unavailable_stacks: list[str] | None = None,
    ai_maturity: int = 1,
    semantic_note: str | None = None,
) -> dict:
    t: dict = {
        "task_id": tid,
        "brief": {
            "company": company,
            "icp_segment": segment,
            "ai_maturity": ai_maturity,
            "confidence": 0.8,
            "hiring_velocity": {
                "direction": (
                    "accelerating" if delta_pct > 0
                    else ("stable" if delta_pct == 0 else "decelerating")
                ),
                "delta_pct": delta_pct,
                "signal_strength": "strong",
                "observation": observation,
            },
            "grounding_facts": grounding_facts,
            "bench_match": {
                "bench_available": bench_available,
                "unavailable_stacks": unavailable_stacks or [],
            },
            "source_mode": "adversarial_hand",
        },
        "email": {"subject": subject, "body": body},
        "prior_thread": "",
        "bench_summary": BENCH_SUMMARY,
        "rubric": RUBRIC,
        "difficulty": "hard",
        "judge_filter": JUDGE_FILTER,
    }
    if semantic_note:
        t["brief"]["semantic_note"] = semantic_note
    return t


def build_tasks() -> list[dict]:
    return [

        # ------------------------------------------------------------------ #
        # TB-ADV-041  D1 FAIL: fabricated numeric not in brief                #
        # Segment 4, Helix Cloud, -15% delta. Email introduces "$5M AI        #
        # investment" — that number does not appear anywhere in the brief.     #
        # ------------------------------------------------------------------ #
        _task(
            "TB-ADV-041",
            company="Helix Cloud", segment="Segment 4", delta_pct=-15,
            observation="Helix Cloud reduced open roles by 15% in the last 60 days.",
            grounding_facts=[
                "Helix Cloud reduced open roles by 15% in the last 60 days.",
            ],
            subject="Context: AI pipeline at Helix Cloud",
            body=(
                "Helix Cloud reduced open roles by 15% recently.\n\n"
                "Your recent $5M AI investment signals a critical pipeline gap "
                "that needs immediate attention.\n\n"
                "Tenacious can deploy specialized AI engineers to address this.\n\n"
                "What AI delivery commitments are at risk this quarter?\n\n"
                "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
            ),
        ),

        # ------------------------------------------------------------------ #
        # TB-ADV-042  D3 FAIL: negative velocity + growth-frame terms         #
        # WidgetCo, -30% delta. Email uses "scaling", "bottlenecks",          #
        # "accelerate" — growth-frame vocabulary violates D3.                 #
        # ------------------------------------------------------------------ #
        _task(
            "TB-ADV-042",
            company="WidgetCo", segment="Segment 1", delta_pct=-30,
            observation="WidgetCo reduced open engineering roles by 30% in the last 60 days.",
            grounding_facts=[
                "WidgetCo reduced open engineering roles by 30% in the last 60 days.",
            ],
            subject="Context: WidgetCo engineering velocity",
            body=(
                "WidgetCo reduced open roles by 30% recently.\n\n"
                "Companies scaling post-Series A often face bottlenecks and need "
                "to accelerate engineering capacity to stay competitive.\n\n"
                "Tenacious provides pre-vetted engineers for teams in rapid growth.\n\n"
                "What engineering priorities are you scaling into next quarter?\n\n"
                "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
            ),
        ),

        # ------------------------------------------------------------------ #
        # TB-ADV-043  D1 FAIL: invented headcount reduction percentage        #
        # DataSync, 0% delta — no reduction signal. Email introduces "45%     #
        # workforce reduction" which does not appear in any grounding fact.   #
        # ------------------------------------------------------------------ #
        _task(
            "TB-ADV-043",
            company="DataSync", segment="Segment 2", delta_pct=0,
            observation="DataSync maintained a stable open role count in the last 60 days.",
            grounding_facts=[
                "DataSync maintained a stable open role count in the last 60 days.",
            ],
            subject="Context: DataSync engineering restructure",
            body=(
                "DataSync maintained stable hiring in the last 60 days.\n\n"
                "After your 45% workforce reduction earlier this year, teams at your stage "
                "need to rebuild delivery capacity without long-term headcount commitments.\n\n"
                "Tenacious provides managed engineering teams for post-restructure delivery.\n\n"
                "What delivery commitments are most critical to maintain this quarter?\n\n"
                "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
            ),
        ),

        # ------------------------------------------------------------------ #
        # TB-ADV-044  D4 FAIL: banned phrase "falling behind"                 #
        # CloudEdge, Segment 3, new CTO, +40%. Email uses condescending       #
        # "falling behind" framing — Non-condescending marker violation.      #
        # ------------------------------------------------------------------ #
        _task(
            "TB-ADV-044",
            company="CloudEdge", segment="Segment 3", delta_pct=40,
            observation="CloudEdge increased open roles by 40% since new CTO joined in Q1 2026.",
            grounding_facts=[
                "CloudEdge increased open roles by 40% since new CTO joined in Q1 2026.",
            ],
            subject="Context: leadership change at CloudEdge",
            body=(
                "CloudEdge open roles increased 40% since new CTO joined in Q1 2026.\n\n"
                "Without the right vendor stack, companies in this transition risk "
                "falling behind peers who moved faster on engineering capacity.\n\n"
                "Tenacious specializes in supporting engineering leadership transitions.\n\n"
                "What delivery gaps are highest priority for the new leadership?\n\n"
                "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
            ),
        ),

        # ------------------------------------------------------------------ #
        # TB-ADV-045  D2 FAIL: Ambiguous segment, body does not end with ?   #
        # Apollo, Ambiguous, +20%. Email ends with a declarative statement   #
        # instead of a qualifying question — Phase 1b D2 fast-fail.          #
        # ------------------------------------------------------------------ #
        _task(
            "TB-ADV-045",
            company="Apollo", segment="Ambiguous", delta_pct=20,
            observation="Apollo increased open roles by 20% in Q1 2026.",
            grounding_facts=[
                "Apollo increased open roles by 20% in Q1 2026.",
            ],
            subject="Note on engineering growth at Apollo",
            body=(
                "Apollo increased open roles by 20% in Q1 2026.\n\n"
                "Companies at this stage commonly reassess their engineering vendor "
                "mix when hiring accelerates.\n\n"
                "It sounds like your growth is driving new engineering priorities "
                "and Tenacious is well positioned to support this expansion.\n\n"
                "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
            ),
        ),

        # ------------------------------------------------------------------ #
        # TB-ADV-046  D1 FAIL: fabricated statistic (340%) not in brief      #
        # ProtoTech, Segment 4, +5% delta. Email invents a "340% productivity #
        # boost" benchmark that has no grounding in the brief.               #
        # ------------------------------------------------------------------ #
        _task(
            "TB-ADV-046",
            company="ProtoTech", segment="Segment 4", delta_pct=5,
            observation="ProtoTech posted 3 ML engineer roles in Q1 2026.",
            grounding_facts=[
                "ProtoTech posted 3 ML engineer roles in Q1 2026.",
            ],
            subject="Context: AI capability at ProtoTech",
            body=(
                "ProtoTech posted 3 ML engineer roles in Q1 2026.\n\n"
                "Our data shows teams that adopt specialized AI squads see a 340% "
                "productivity boost within 90 days of deployment.\n\n"
                "Tenacious can deliver that for your ML function.\n\n"
                "Which ML capability gaps are most critical for your roadmap right now?\n\n"
                "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
            ),
        ),

        # ------------------------------------------------------------------ #
        # TB-ADV-047  D4 FAIL: banned phrases "world-class" and "just "      #
        # Cypher Inc, Segment 2, -30% delta. Email trivializes the            #
        # restructure with style-guide-banned openers.                        #
        # ------------------------------------------------------------------ #
        _task(
            "TB-ADV-047",
            company="Cypher Inc", segment="Segment 2", delta_pct=-30,
            observation="Cypher Inc reduced open roles by 30% in the last 60 days.",
            grounding_facts=[
                "Cypher Inc reduced open roles by 30% in the last 60 days.",
            ],
            subject="Context: Cypher Inc engineering capacity",
            body=(
                "Cypher Inc reduced open roles by 30% recently.\n\n"
                "Just wanted to check in — our world-class engineers can help "
                "maintain delivery output through your restructure without "
                "long-term headcount commitments.\n\n"
                "What delivery commitments are highest priority this quarter?\n\n"
                "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
            ),
        ),

        # ------------------------------------------------------------------ #
        # TB-ADV-048  D4 FAIL: "bench" used in prospect-facing copy          #
        # MetaNet, Segment 3, new CTO, +50%. Email uses "bench" which is     #
        # internal jargon — Professional marker violation.                   #
        # ------------------------------------------------------------------ #
        _task(
            "TB-ADV-048",
            company="MetaNet", segment="Segment 3", delta_pct=50,
            observation="MetaNet increased open roles by 50% since new CTO joined in Q1 2026.",
            grounding_facts=[
                "MetaNet increased open roles by 50% since new CTO joined in Q1 2026.",
            ],
            subject="Note on vendor strategy at MetaNet",
            body=(
                "MetaNet increased open roles by 50% since new CTO joined in Q1 2026.\n\n"
                "Incoming engineering leaders typically reassess vendor mix in the first 90 days.\n\n"
                "Our ML bench is deep across Python and data engineering, "
                "ready to fill your highest-priority gaps.\n\n"
                "Which delivery commitments are most critical to maintain through the transition?\n\n"
                "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
            ),
        ),

        # ------------------------------------------------------------------ #
        # TB-ADV-049  D1 FAIL: wrong funding amount ($99M vs $9M in brief)   #
        # NanoHealth, Segment 1, +10% delta, Series A $9M. Email claims      #
        # "$99M" which is not in the brief — corrupted numeric.              #
        # ------------------------------------------------------------------ #
        _task(
            "TB-ADV-049",
            company="NanoHealth", segment="Segment 1", delta_pct=10,
            observation="NanoHealth increased open roles by 10% since Series A $9M closed in Q1 2026.",
            grounding_facts=[
                "NanoHealth increased open roles by 10% since Series A $9M closed in Q1 2026.",
            ],
            subject="Congrats on Series A, NanoHealth",
            body=(
                "NanoHealth raised $99M and increased open roles by 10% in Q1 2026.\n\n"
                "Companies at this funding stage often find that recruiting velocity "
                "cannot keep pace with engineering demand.\n\n"
                "Tenacious provides pre-vetted engineers deployable in days.\n\n"
                "What engineering priorities are you scaling into next quarter?\n\n"
                "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
            ),
        ),

        # ------------------------------------------------------------------ #
        # TB-ADV-050  D5 FAIL: meeting-booking phrase "book a call"          #
        # OmniData, Segment 4, +5% delta. Email closes with a direct         #
        # booking CTA — Format Compliance violation.                         #
        # ------------------------------------------------------------------ #
        _task(
            "TB-ADV-050",
            company="OmniData", segment="Segment 4", delta_pct=5,
            observation="OmniData posted 4 ML engineer roles in Q1 2026.",
            grounding_facts=[
                "OmniData posted 4 ML engineer roles in Q1 2026.",
            ],
            subject="Context: AI capability at OmniData",
            body=(
                "OmniData posted 4 ML engineer roles in Q1 2026.\n\n"
                "This signals a specific AI capability gap in specialized ML functions.\n\n"
                "Tenacious staffs ML squads on fixed-scope engagements, "
                "typically 3 to 4 months.\n\n"
                "Book a call with our team today to discuss your ML roadmap.\n\n"
                "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
            ),
        ),

        # ------------------------------------------------------------------ #
        # TB-ADV-051  D1 FAIL: misleading metric ($30M vs $15M in brief)     #
        # AeroSys, Segment 1, +20% delta, Series A $15M. Email claims "$30M" #
        # — a fabricated funding amount that fails grounding fidelity.        #
        # ------------------------------------------------------------------ #
        _task(
            "TB-ADV-051",
            company="AeroSys", segment="Segment 1", delta_pct=20,
            observation="AeroSys increased open roles by 20% since Series A $15M in Feb 2026.",
            grounding_facts=[
                "AeroSys increased open roles by 20% since Series A $15M in Feb 2026.",
            ],
            subject="Congrats on Series A, AeroSys",
            body=(
                "AeroSys raised $30M and increased open engineering roles by 20% since Feb 2026.\n\n"
                "Companies at this stage typically need engineering capacity "
                "faster than traditional hiring allows.\n\n"
                "Tenacious provides pre-vetted engineers ready to deploy.\n\n"
                "What engineering priorities are you scaling into next quarter?\n\n"
                "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
            ),
        ),

        # ------------------------------------------------------------------ #
        # TB-ADV-052  PASS (Phase 2 semantic probe)                           #
        # ZetaCorp, Ambiguous, stable signal. Passes ALL Phase 1 rules but   #
        # is semantically weak: the qualifying question ignores the specific  #
        # signal (2 open roles) and defaults to a generic three-option frame. #
        # Intended to catch LLM judges that penalize low-specificity probes.  #
        # ------------------------------------------------------------------ #
        _task(
            "TB-ADV-052",
            company="ZetaCorp", segment="Ambiguous", delta_pct=0,
            observation="ZetaCorp posted 2 open roles in Q1 2026.",
            grounding_facts=[
                "ZetaCorp posted 2 open roles in Q1 2026.",
            ],
            subject="Question on engineering priorities at ZetaCorp",
            body=(
                "ZetaCorp posted 2 open roles in Q1 2026.\n\n"
                "Before making a specific recommendation, I want to understand "
                "your current engineering priorities.\n\n"
                "Are you primarily focused on reducing costs, scaling capacity, "
                "or building a new technical capability?\n\n"
                "Birkity\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
            ),
            semantic_note=(
                "PASS Phase 1 (D1-D5 deterministic). Phase 2 semantic probe: "
                "email ignores the specific weak signal (2 open roles) and asks "
                "a generic three-option question. A trained LLM judge should flag "
                "this as low-signal outreach that fails to ground the qualifying "
                "question in observable evidence from the brief."
            ),
        ),

    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify tasks without writing to disk.")
    args = parser.parse_args()

    tasks = build_tasks()

    expected: dict[str, tuple[str, str | None]] = {
        "TB-ADV-041": ("REJECT", "D1"),
        "TB-ADV-042": ("REJECT", "D3"),
        "TB-ADV-043": ("REJECT", "D1"),
        "TB-ADV-044": ("REJECT", "D4"),
        "TB-ADV-045": ("REJECT", "D2"),
        "TB-ADV-046": ("REJECT", "D1"),
        "TB-ADV-047": ("REJECT", "D4"),
        "TB-ADV-048": ("REJECT", "D4"),
        "TB-ADV-049": ("REJECT", "D1"),
        "TB-ADV-050": ("REJECT", "D5"),
        "TB-ADV-051": ("REJECT", "D1"),
        "TB-ADV-052": ("PASS",   None),
    }

    print("Verifying 12 adversarial edge-case tasks...")
    all_ok = True
    for t in tasks:
        r = score_task(t)
        exp_v, exp_d = expected[t["task_id"]]
        ok = (r["verdict"] == exp_v and r["failed_dimension"] == exp_d)
        mark = "OK  " if ok else "FAIL"
        print(
            f"  {mark}  {t['task_id']}  "
            f"expected={exp_v}/{exp_d}  "
            f"got={r['verdict']}/{r['failed_dimension']}  "
            f"reason={r['reason']!r}"
        )
        if not ok:
            all_ok = False

    if not all_ok:
        raise SystemExit("One or more tasks failed verification. Not writing output.")

    print(f"\nAll 12 tasks verified OK.")

    if args.dry_run:
        print("Dry run — not writing to disk.")
        return

    # Check for duplicate IDs before appending
    existing_ids: set[str] = set()
    if OUTPUT.exists():
        for line in OUTPUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_ids.add(json.loads(line)["task_id"])

    new_tasks = [t for t in tasks if t["task_id"] not in existing_ids]
    skipped = len(tasks) - len(new_tasks)
    if skipped:
        print(f"Skipped {skipped} already-present task IDs.")

    with OUTPUT.open("a", encoding="utf-8") as f:
        for t in new_tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    total = len(existing_ids) + len(new_tasks)
    print(f"Appended {len(new_tasks)} tasks. adversarial_hand_batch1.jsonl now has {total} tasks.")


if __name__ == "__main__":
    main()
