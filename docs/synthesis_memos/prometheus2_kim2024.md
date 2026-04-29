# Synthesis Memo: Kim et al. (2024) — Prometheus 2

**Paper:** Prometheus 2: An Open-Source Language Model Specialized in Evaluating Other Language Models  
**Authors:** Kim et al.  
**Venue:** 2024  
**Status:** STUB — complete before Day 4 (Path B required reading)

---

## Summary (to write)

<!-- 1 page. Focus on what Prometheus 2 teaches us about training a small open judge. -->

## What Prometheus 2 Did That We Are Doing Similarly

- Small backbone (we: 0.8B vs Prometheus: 7B/13B) fine-tuned as a judge
- Preference data from existing strong judges → training a weaker judge to match
- Explicit rubric-grounded scoring (our D1–D5 rubrics)

## Key Differences in Our Setup

- We have domain-specific rubrics (Tenacious sales), not general quality
- Our judge is binary (PASS/REJECT), not 1–5 scale
- Our training data is smaller (~125 preference pairs vs Prometheus's 100K+)

## LIMA Effect (connects to Zhou et al. 2023)

<!-- Prometheus 2 + LIMA together: does quality dominate quantity at 100–1000 pairs?
     What does this imply for our 125-pair training partition? -->

## Our Disagreement (to complete)

<!-- Prometheus 2 uses absolute scoring with reference answers. Our task is relative
     (no gold email, just rubric adherence). Does this change the preference pair
     construction strategy? -->
