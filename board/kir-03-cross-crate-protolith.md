---
title: "kir-03 — cross-crate certificates: target protolith spec fns in place"
status: todo
claimed_by:
created: 2026-07-16T22:00:00Z
updated: 2026-07-16T22:00:00Z
---

## Description

Drop the spec_kernels verbatim copies; generated certificates import
protolith::slab::{spec_wrap_delta, spec_abs_delta} (and a named spec_sum_sq3 added
to protolith) directly.

- Multi-crate wiring under the Lean backend (tactus-group-theory export pattern,
  crate-local check.sh precedent).
- kirgen grows a --target-crate config (import lines + spec paths).
- protolith gains the named sum_sq3 spec fn (currently inline in its ensures).

Done when: cert_wrap_delta's ensures references protolith's spec fn by path and the
combined check is 0 errors; spec_kernels.rs deleted.

## Progress

- (2026-07-16T22:00:00Z) created

## Writeup

