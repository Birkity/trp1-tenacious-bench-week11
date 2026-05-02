"""Debug contamination check violations."""
import json, sys
sys.path.insert(0, "benchmark")
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Load all tasks
task_index = {}
for pname in ["train", "dev", "held_out"]:
    path = ROOT / f"data/tenacious_bench_v0.1/{pname}/tasks.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            t = json.loads(line)
            task_index[t["task_id"]] = t

# Look at the cosine 1.0 pair
for tid in ["TB-TRACE-005A", "TB-TRACE-003A"]:
    t = task_index.get(tid, {})
    b = t.get("brief", {})
    company = b.get("company", "")
    segment = b.get("icp_segment", "")
    delta = b.get("hiring_velocity", {}).get("delta_pct", "")
    obs = b.get("hiring_velocity", {}).get("observation", "")
    subject = t.get("email", {}).get("subject", "")
    print(f"{tid}: company={company!r} segment={segment!r} delta={delta} obs={obs!r} subject={subject!r}")
    print()

# Check n-gram violations
r = json.loads((ROOT / "data/contamination_check.json").read_text(encoding="utf-8"))
ngram_viols = r["checks"]["ngram_overlap"]["violations"]
print(f"\nN-gram violations (first 3):")
for v in ngram_viols[:3]:
    print(f"  {v['held_out']} <-> {v['train']}: {v['shared_ngrams']} shared, examples={v['examples'][:1]}")
