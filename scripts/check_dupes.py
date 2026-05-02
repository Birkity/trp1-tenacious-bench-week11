"""Check for duplicate company-observation pairs in trace batch."""
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent
path = ROOT / "data/tenacious_bench_v0.1/dev/trace_derived_batch1.jsonl"
tasks = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

# Only look at A variants
a_tasks = [t for t in tasks if t["task_id"].endswith("A")]
print(f"A-variant tasks: {len(a_tasks)}")

obs_to_ids = {}
for t in a_tasks:
    obs = t["brief"]["hiring_velocity"]["observation"]
    company = t["brief"]["company"]
    key = f"{company}::{obs}"
    if key not in obs_to_ids:
        obs_to_ids[key] = []
    obs_to_ids[key].append(t["task_id"])

dupes = {k: v for k, v in obs_to_ids.items() if len(v) > 1}
print(f"Duplicate company-observation pairs: {len(dupes)}")
for k, ids in dupes.items():
    print(f"  {ids} -> {k[:60]}")
