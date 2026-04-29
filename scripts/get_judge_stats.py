"""Extract judge filter statistics from all batches."""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent

files = {
    "trace_derived": ROOT / "data/tenacious_bench_v0.1/dev/trace_derived_batch1.jsonl",
    "programmatic": ROOT / "data/tenacious_bench_v0.1/dev/programmatic_batch1.jsonl",
    "adversarial_hand": ROOT / "data/tenacious_bench_v0.1/dev/adversarial_hand_batch1.jsonl",
}

for name, path in files.items():
    tasks = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    jf_list = [t.get("judge_filter", {}) for t in tasks if t.get("judge_filter")]
    if jf_list:
        ic = sum(j.get("input_coherence", 0) for j in jf_list) / len(jf_list)
        gtv = sum(j.get("ground_truth_verifiability", 0) for j in jf_list) / len(jf_list)
        rac = sum(j.get("rubric_application_clarity", 0) for j in jf_list) / len(jf_list)
        passed = sum(1 for j in jf_list if j.get("passed"))
        print(f"{name}: n={len(jf_list)} passed={passed}/{len(jf_list)}")
        print(f"  input_coherence={ic:.2f}  ground_truth_verifiability={gtv:.2f}  rubric_application_clarity={rac:.2f}")
