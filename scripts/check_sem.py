"""Check synthetic semantic task filter results."""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
path = ROOT / "data/tenacious_bench_v0.1/dev_synthetic/semantic_edge_cases_batch1.jsonl"
tasks = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
passed = [t for t in tasks if t.get("judge_filter", {}).get("passed", True)]
failed = [t for t in tasks if not t.get("judge_filter", {}).get("passed", True)]
print(f"Total: {len(tasks)}  Passed: {len(passed)}  Failed: {len(failed)}")

if passed:
    t = passed[0]
    tid = t["task_id"]
    notes = t.get("judge_filter", {}).get("notes", "")
    print(f"Example pass: {tid}  notes: {notes[:80]}")

if failed:
    t = failed[0]
    tid = t["task_id"]
    notes = t.get("judge_filter", {}).get("notes", "")
    ic = t.get("judge_filter", {}).get("input_coherence", "?")
    gtv = t.get("judge_filter", {}).get("ground_truth_verifiability", "?")
    rac = t.get("judge_filter", {}).get("rubric_application_clarity", "?")
    print(f"Example fail: {tid} IC={ic} GTV={gtv} RAC={rac} notes: {notes[:100]}")
