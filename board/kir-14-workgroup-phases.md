---
title: "kir-14 — v1.5 barrier-phased workgroup memory"
status: todo
claimed_by:
created: 2026-07-16T22:00:00Z
updated: 2026-07-16T22:00:00Z
---

## Description

Shipped mandelbrot shaders use var<workgroup> wg_mem (32KB tile-shared ref
orbit), so parity needs shared memory. Discipline (DESIGN 4.5): phases separated
by workgroupBarrier; within a phase, per-thread writes to shared arrays are
disjoint (side condition); phase-composition lemma proved once. No weak-memory
model, ever.

Done when: design note + KIR extension + a toy two-phase kernel certified and
running; phase composition lemma 0 errors.

## Progress

- (2026-07-16T22:00:00Z) created

## Writeup

