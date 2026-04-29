# Synthesis Memo: Meng, Xia, Chen (2024) — SimPO

**Paper:** SimPO: Simple Preference Optimization with a Reference-Free Reward  
**Authors:** Meng, Xia, and Chen  
**Venue:** NeurIPS 2024  
**Status:** STUB — complete before Day 4 (Path B chosen method)

---

## Summary (to write)

<!-- 1 page. This is our chosen training method — highest priority Path B memo. -->
<!-- Must explain: length-normalized reward, gamma margin, why reference-free matters. -->

## Why SimPO for Our Judge Task

- Reference-free: no frozen reference model in memory alongside Qwen 3.5 0.8B
- Length normalization: our judge outputs vary from 1 sentence to 4 sentences
- Margin parameter gamma: controls separation between chosen/rejected reward
- Fits Colab T4 16GB without VRAM spill

## Hyperparameter Choices (to decide before training)

<!-- beta: KL penalty weight -->
<!-- gamma: reward margin -->
<!-- From the paper's ablation table, what are the recommended defaults? -->

## Our Disagreement (to complete)

<!-- Does SimPO's implicit reward formulation handle the case where our rejected
     responses are deliberately "almost right" (surface-passing but semantically wrong)?
     Is this a limitation? -->
