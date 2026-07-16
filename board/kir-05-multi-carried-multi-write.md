---
title: "kir-05 — multi-carried registers and multiple writes per iteration"
status: todo
claimed_by:
created: 2026-07-16T22:00:00Z
updated: 2026-07-16T22:00:00Z
---

## Description

Generalize the loop template: k carried registers (model threading = k+1
recursive fns each passing all carried values; tuple-free) and multiple WriteOuts
(argmin assign carries best-key AND best-seed; some kernels write two buffers
worth per iteration — v1 target: multiple writes to the one out buffer at
distinct indices, side condition distinctness).

- Test kernel: 2-carried running argmin over a buffer (best_key, best_idx) —
  the exact shape protolith assign needs.

Done when: the argmin toy generates + verifies; single-carried path unchanged
(regression: add_limbs still first-run green).

## Progress

- (2026-07-16T22:00:00Z) created

## Writeup

