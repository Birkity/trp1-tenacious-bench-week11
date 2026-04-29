# Synthesis Memo: Chen et al. (2025) — Contamination Prevention

**Paper:** Recent Advances in Large Language Model Benchmarks against Data Contamination: From Static to Dynamic Evaluation  
**Authors:** Chen et al.  
**Venue:** EMNLP 2025  
**Status:** STUB — complete before sealing held_out/ partition

---

## Summary (to write)

<!-- 1 page. Focus on the three contamination checks we must implement. -->

## Three Checks to Implement (from the paper)

1. **N-gram overlap** — < 8-gram overlap between held_out and train on input fields
2. **Embedding similarity** — cosine < 0.85 for any held_out/train pair
3. **Time-shift verification** — public signal references must be documentable

## Implementation Plan (to complete)

<!-- Where does contamination_check.py go? What model for embeddings? -->
<!-- How do we handle the small private source corpus without leaking it? -->

## Our Disagreement (to complete)

<!-- Our dataset is constructed from a small closed set of companies, not web-scraped data.
     How does this change which contamination checks are most critical? -->
