"""
Hand-authored adversarial task generator — Act I/II.

Converts the 32 probes from docs/probe_library.md into ~40 benchmark tasks.
These carry the highest originality weight at grading.

Probe → Task mapping:
    Each probe has: input specification, expected behavior, pass/fail condition.
    Each becomes a task where:
        - brief = the probe's input context
        - email = an email that violates the probe's expected behavior
        - ground_truth = REJECT with the probe's failure mode as reason

Additional 8 tasks are hand-authored specifically to defeat:
    - Emails that pass all programmatic checks (D2–D5) but fail D1 semantically
    - Examples: correct numbers, correct tone, but wrong pitch for the segment
      ("We understand you're going through a leadership transition" for Segment 1)

This file is a scaffold. Actual task content is hand-written, not generated.
See: week11/tenacious_bench_v0.1/train/adversarial/ for raw task files.

Usage:
    python adversarial_hand.py --probe-library ../../docs/probe_library.md \
                               --output ../tenacious_bench_v0.1/raw_adversarial.jsonl
"""

# TODO (Act I): expand probe library entries into task schema
# Source probe library: docs/probe_library.md (32 probes, 10 categories)

PROBE_LIBRARY_PATH = "../../docs/probe_library.md"
TARGET_TASK_COUNT = 40
