---
title: "kir-10 — trusted WGSL printer + naga gate"
status: todo
claimed_by:
created: 2026-07-16T22:00:00Z
updated: 2026-07-16T22:00:00Z
---

## Description

KIR -> WGSL text. One construct one line, whole printer one file, reviewable in
a sitting (DESIGN 7). Mine verus-gpu-parser/src/emit.rs for calling-convention
knowledge (struct returns, ptr-to-local-array params, var<workgroup> decls — the
latter NOT emitted yet, v1.5). select()/u32() for compare-bits; u32 wrapping ops
map directly. naga parse+validate in CI as sanity gate (not a soundness dep).

Done when: wrap_delta.wgsl, sum_sq3.wgsl, add_limbs.wgsl emitted from KIR literals
and naga-clean; printer + emitted files committed.

## Progress

- (2026-07-16T22:00:00Z) created

## Writeup

