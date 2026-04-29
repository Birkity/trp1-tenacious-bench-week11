"""
Trace-derived task generator — Act II, Days 2–3.

Converts 9 company hiring signal briefs × 3 variants into ~75 tasks.

Variants per company:
  1. original    — real email from artifacts/{slug}/email_log.jsonl, scored as-is
  2. corrupted   — grounding fact numeric replaced with wrong value → D3 FAIL
  3. counterfact — velocity sign flipped (pos→neg or neg→pos) → D2 FAIL or PASS

Usage:
    python trace_derived.py --output ../tenacious_bench_v0.1/raw_trace.jsonl
"""

# TODO (Act II): implement

SLUGS = [
    "arcana", "brightpath", "coraltech", "kinanalytics",
    "novaspark", "pulsesight", "snaptrade", "streamlineops", "wiseitech",
]
