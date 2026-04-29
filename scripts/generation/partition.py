"""
Dataset partitioner — Act II.

Combines all generated JSONL batches into a single shuffled pool,
then splits deterministically into:
  train/     50%  → preference pair generation (Act III)
  dev/       30%  → public evaluation set
  held_out/  20%  → sealed, gitignored, final eval

Source files (all merged):
  dev/trace_derived_batch1.jsonl       (75 tasks)
  dev/programmatic_batch1.jsonl        (75 tasks)
  dev/adversarial_hand_batch1.jsonl    (40 tasks)
  dev_synthetic/semantic_edge_cases_batch1.jsonl  (60 tasks, if present)

Output:
  train/tasks.jsonl
  dev/tasks.jsonl
  held_out/tasks.jsonl

Adds source_mode to brief if missing (inferred from task_id prefix).

Usage:
    python scripts/generation/partition.py
    python scripts/generation/partition.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).parents[2]
DATA_DIR = ROOT / "data" / "tenacious_bench_v0.1"

SOURCE_FILES = [
    DATA_DIR / "dev" / "trace_derived_batch1.jsonl",
    DATA_DIR / "dev" / "programmatic_batch1.jsonl",
    DATA_DIR / "dev" / "adversarial_hand_batch1.jsonl",
    DATA_DIR / "dev_synthetic" / "semantic_edge_cases_batch1.jsonl",
]

TRAIN_RATIO = 0.50
DEV_RATIO   = 0.30
# held_out = remainder (0.20)

RANDOM_SEED = 42


def infer_source_mode(task: dict) -> str:
    tid = task.get("task_id", "")
    if tid.startswith("TB-TRACE"):
        return "trace_derived"
    if tid.startswith("TB-PROG"):
        return "programmatic"
    if tid.startswith("TB-ADV"):
        return "adversarial_hand"
    if tid.startswith("TB-SEM"):
        return "synthetic_semantic"
    return "unknown"


def load_all_tasks() -> list[dict]:
    tasks: list[dict] = []
    for path in SOURCE_FILES:
        if not path.exists():
            print(f"  [SKIP] {path.name} not found")
            continue
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            # Ensure source_mode is in brief
            if "source_mode" not in task.get("brief", {}):
                task["brief"]["source_mode"] = infer_source_mode(task)
            tasks.append(task)
            count += 1
        print(f"  [LOAD] {path.name}: {count} tasks")
    return tasks


def _task_family(task: dict) -> str:
    """
    Return the contamination family of a task — the group of tasks that share
    the same company signal and must stay in the same partition.

    For trace-derived tasks: all events from the same company share the same
    observation text (5 events × same company). Group by company name.

    For programmatic tasks: A/B/C variants share the same observation. Group
    by the numeric prefix (TB-PROG-007A → TB-PROG-007).

    For adversarial and synthetic tasks: each task is independent.
    """
    import re
    task_id = task.get("task_id", "")
    source_mode = task.get("brief", {}).get("source_mode", "")

    # Trace-derived: group by company (all events from same company share observation)
    if task_id.startswith("TB-TRACE") or source_mode == "trace_derived":
        company = task.get("brief", {}).get("company", task_id)
        return f"company::{company}"

    # Programmatic: group A/B/C variants together
    m = re.match(r"^(TB-PROG-\d+)[ABC]$", task_id)
    if m:
        return m.group(1)

    # Adversarial and synthetic: each task is its own family
    return task_id


def deterministic_split(
    tasks: list[dict], seed: int
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Family-aware split: tasks that share observation text (same company, or
    A/B/C variants) are kept in the same partition to prevent contamination.
    """
    from collections import defaultdict
    families: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        families[_task_family(t)].append(t)

    rng = random.Random(seed)
    family_keys = sorted(families.keys())
    rng.shuffle(family_keys)

    n_families = len(family_keys)
    n_train_fam = int(n_families * TRAIN_RATIO)
    n_dev_fam   = int(n_families * DEV_RATIO)

    train_keys    = family_keys[:n_train_fam]
    dev_keys      = family_keys[n_train_fam:n_train_fam + n_dev_fam]
    held_out_keys = family_keys[n_train_fam + n_dev_fam:]

    def flatten(keys: list[str]) -> list[dict]:
        result: list[dict] = []
        for k in sorted(keys):
            result.extend(sorted(families[k], key=lambda t: t["task_id"]))
        return result

    return flatten(train_keys), flatten(dev_keys), flatten(held_out_keys)


def write_jsonl(path: Path, tasks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for task in tasks:
            fh.write(json.dumps(task, ensure_ascii=False) + "\n")


def partition_checksum(tasks: list[dict]) -> str:
    ids = sorted(t["task_id"] for t in tasks)
    return hashlib.sha256("|".join(ids).encode()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print split statistics without writing files")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    print("Loading tasks...")
    all_tasks = load_all_tasks()
    print(f"Total loaded: {len(all_tasks)}\n")

    if not all_tasks:
        raise SystemExit("No tasks loaded — check source files.")

    # Check for duplicates
    ids = [t["task_id"] for t in all_tasks]
    dupes = [tid for tid in ids if ids.count(tid) > 1]
    if dupes:
        print(f"WARNING: {len(set(dupes))} duplicate task IDs: {set(dupes)}")

    train, dev, held_out = deterministic_split(all_tasks, args.seed)

    # Difficulty distribution per partition
    def diff_dist(tasks: list[dict]) -> dict:
        d: dict[str, int] = {}
        for t in tasks:
            k = t.get("difficulty", "?")
            d[k] = d.get(k, 0) + 1
        return d

    def source_dist(tasks: list[dict]) -> dict:
        d: dict[str, int] = {}
        for t in tasks:
            k = t.get("brief", {}).get("source_mode", "unknown")
            d[k] = d.get(k, 0) + 1
        return d

    print("=" * 60)
    print("  PARTITION SUMMARY")
    print("=" * 60)
    print(f"Total tasks          : {len(all_tasks)}")
    print(f"Train ({TRAIN_RATIO*100:.0f}%)            : {len(train)} tasks")
    print(f"  difficulty         : {diff_dist(train)}")
    print(f"  source_mode        : {source_dist(train)}")
    print(f"Dev ({DEV_RATIO*100:.0f}%)              : {len(dev)} tasks")
    print(f"  difficulty         : {diff_dist(dev)}")
    print(f"  source_mode        : {source_dist(dev)}")
    print(f"Held-out (20%)       : {len(held_out)} tasks")
    print(f"  difficulty         : {diff_dist(held_out)}")
    print(f"  source_mode        : {source_dist(held_out)}")
    print()
    print(f"Train checksum       : {partition_checksum(train)}")
    print(f"Dev checksum         : {partition_checksum(dev)}")
    print(f"Held-out checksum    : {partition_checksum(held_out)}")

    if args.dry_run:
        print("\nDry run — no files written.")
        return

    write_jsonl(DATA_DIR / "train" / "tasks.jsonl", train)
    write_jsonl(DATA_DIR / "dev" / "tasks.jsonl", dev)
    write_jsonl(DATA_DIR / "held_out" / "tasks.jsonl", held_out)

    print(f"\nWrote:")
    print(f"  {DATA_DIR / 'train' / 'tasks.jsonl'}")
    print(f"  {DATA_DIR / 'dev' / 'tasks.jsonl'}")
    print(f"  {DATA_DIR / 'held_out' / 'tasks.jsonl'}")


if __name__ == "__main__":
    main()
