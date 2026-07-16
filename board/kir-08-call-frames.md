---
title: "kir-08 — KIR function calls (or principled inlining) for composed kernels"
status: todo
claimed_by:
created: 2026-07-16T22:00:00Z
updated: 2026-07-16T22:00:00Z
---

## Description

torus_dist2 = wrap_delta + wrap_delta + abs_delta + sum_sq3. Two routes; decide
and do one:

- (a) KIR Call statement + u_ call-frame unfold shape + per-callee certificates
  composing (structure-preserving, matches shipped-WGSL style, printer emits fns).
- (b) generator inlining with certificates against the composed spec (simpler,
  code-duplicating; fine at this size).

DESIGN 4.3 prefers functions-stay-functions. Probe REPORT guessed one more u_
shape suffices — measure it.

Done when: torus_dist2 certified end-to-end by the chosen route; decision + measurement
recorded here and in REPORT.md.

## Progress

- (2026-07-16T22:00:00Z) created

## Writeup

