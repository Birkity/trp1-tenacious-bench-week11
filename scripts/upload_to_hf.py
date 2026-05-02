"""
upload_to_hf.py — Push Tenacious-Bench v0.1 to HuggingFace Hub.

Usage:
    python scripts/upload_to_hf.py --token hf_xxx
    # or set HF_TOKEN env var and run without --token
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, login, upload_folder

ROOT = Path(__file__).parents[1]
DATASET_DIR = ROOT / "data" / "tenacious_bench_v0.1"
REPO_ID = "Birkity/tenacious_bench_v0.1"

# Files/dirs to exclude from the upload
IGNORE_PATTERNS = [
    "held_out/*",        # sealed partition — never distributed
    ".gitkeep",
    "__pycache__",
    "*.pyc",
    ".DS_Store",
    "scored_tasks.jsonl",  # intermediate scoring artifact
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--dry-run", action="store_true",
                        help="List files that would be uploaded without uploading")
    args = parser.parse_args()

    if not args.token:
        print("ERROR: provide --token hf_xxx or set HF_TOKEN env var")
        raise SystemExit(1)

    if args.dry_run:
        print(f"Dry run — would upload from: {DATASET_DIR}")
        for f in sorted(DATASET_DIR.rglob("*")):
            if f.is_file():
                rel = f.relative_to(DATASET_DIR)
                skip = any(
                    f.match(p.replace("/*", "/**")) or str(rel).startswith(p.rstrip("/*"))
                    for p in IGNORE_PATTERNS
                )
                tag = "SKIP" if skip else "  OK"
                print(f"  [{tag}] {rel}")
        return

    login(token=args.token)

    print(f"Uploading {DATASET_DIR} -> {REPO_ID} ...")
    url = upload_folder(
        folder_path=str(DATASET_DIR),
        repo_id=REPO_ID,
        repo_type="dataset",
        ignore_patterns=IGNORE_PATTERNS,
        commit_message="Upload Tenacious-Bench v0.1 (Acts I–IV complete)",
    )
    print(f"\nDone! Dataset live at: {url}")


if __name__ == "__main__":
    main()
