"""Inspect contamination check results."""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
r = json.loads((ROOT / "data/contamination_check.json").read_text(encoding="utf-8"))

ng = r["checks"]["ngram_overlap"]
print(f"N-gram: violations={ng['violating_pairs']} max_shared={ng['max_shared_ngrams']}")
for v in ng["violations"][:5]:
    ho = v["held_out"]
    tr = v["train"]
    shared = v["shared_ngrams"]
    ex = v["examples"][0] if v["examples"] else []
    print(f"  {ho} <-> {tr}: {shared} shared, example: {' '.join(ex)!r}")

emb = r["checks"]["embedding_similarity"]
print(f"\nCosine: violations={emb['violating_pairs']} max_sim={emb['max_cosine_similarity']}")
for v in emb["violations"][:5]:
    ho = v["held_out"]
    tr = v["train"]
    sim = v["cosine_similarity"]
    print(f"  {ho} <-> {tr}: sim={sim}")
