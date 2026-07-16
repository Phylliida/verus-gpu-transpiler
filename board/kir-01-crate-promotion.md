---
title: "kir-01 — promote KIR out of probe-g1 into a verus-kir crate"
status: todo
claimed_by:
created: 2026-07-16T22:00:00Z
updated: 2026-07-16T22:00:00Z
---

## Description

probe-g1 currently owns kir.rs (AST/eval/exec/kloop), the generic u_* layer
(scattered in addloop.rs), mk_state, and gen_certs.rs. Promote the reusable parts
into a proper `verus-kir` crate with its own check.sh; probe-g1 becomes a historical
consumer; kirgen output imports verus-kir.

- Move: kir.rs; generic u_keval_*/u_kexec_*/u_kloop_* unfolds + mk_state (out of
  addloop.rs into a kir-owned unfolds module); gen_certs prelude unfolds.
- Keep probe-g1's addloop/certificate as the golden hand-written reference (they
  re-import from verus-kir).
- Watch: module paths in generated simp lists (gen_certs.stmt_* / addloop.mk_state)
  must follow the move — kirgen templates change too.

Done when: verus-kir + probe-g1 + regenerated gen_certs all verify (75/0-equivalent)
with kir types imported from the new crate.

## Progress

- (2026-07-16T22:00:00Z) created

## Writeup

