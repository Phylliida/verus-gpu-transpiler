---
title: "kir-16 — customer summit: protolith assign pass, source to pixel-identical GPU"
status: todo
claimed_by:
created: 2026-07-16T22:00:00Z
updated: 2026-07-16T22:00:00Z
---

## Description

The G3 first summit (DESIGN 9): protolith M0's per-voxel argmin assign pass —
verified CPU source (material-synthesis board m0-04) -> generated certificate
(needs kir-04 indexing + kir-05 two-carried argmin) -> WGSL (kir-10) -> harness
(kir-11) -> bit-identical assign buffer on an M0 preview slab.

Composite keys on GPU: (dist2, index) as lexicographic u32 pair — no u64
(DESIGN 7 note).

Done when: CPU and GPU produce bit-identical per-voxel (key, owner) buffers for a
256x256x16 preview slab; recorded as the program's first end-to-end kernel.

## Progress

- (2026-07-16T22:00:00Z) created

## Writeup

