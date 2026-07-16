---
title: "kir-12 — the once-proved parallel-map dispatch lemma"
status: todo
claimed_by:
created: 2026-07-16T22:00:00Z
updated: 2026-07-16T22:00:00Z
---

## Description

DESIGN 4.4: v1 kernels are maps; prove once at the KIR level that N
disjoint-writing per-thread executions equal the sequential loop over tids
(own(tid) = {tid} first; contiguous slices second). This is what makes the
per-thread certificates speak for the parallel dispatch.

Done when: lemma verified in verus-kir; DESIGN 4.4 references it; add_limbs'
per-tid reading documented (each tid = one limb index i in the map form).

## Progress

- (2026-07-16T22:00:00Z) created

## Writeup

