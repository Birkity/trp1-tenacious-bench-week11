"""
Programmatic template generator — Act II, Days 2–3.

Parameter sweep over brief fields to produce ~75 tasks with known ground truth.
Ground truth is fully deterministic — scoring_evaluator.py verifies each task.

Sweep dimensions:
    velocity_delta   : [-80, -40, 0, +60, +120]   (5 values)
    icp_segment      : [Segment 1, Segment 2, Segment 3, Segment 4, ambiguous]  (5)
    ai_maturity      : [0, 1, 2, 3]   (4)
    bench_available  : [True, False]   (2)

Total combinations: 5 × 5 × 4 × 2 = 200 → subsample 75 after dedup / near-dupe filter.

For each combination:
    - Generate brief dict from parameters
    - Generate one "correct" email (matching segment pitch + velocity direction)
    - Generate one "incorrect" email (mismatched pitch) via template swap
    - Score both with scoring_evaluator.py
    - Keep the pair as (correct→PASS, incorrect→REJECT) task

Usage:
    python programmatic_templates.py --output ../tenacious_bench_v0.1/raw_programmatic.jsonl --n 75
"""

# TODO (Act II): implement

VELOCITY_BUCKETS = [-80, -40, 0, 60, 120]
SEGMENTS = ["Segment 1", "Segment 2", "Segment 3", "Segment 4", "ambiguous"]
AI_MATURITY_LEVELS = [0, 1, 2, 3]
