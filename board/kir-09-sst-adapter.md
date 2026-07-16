---
title: "kir-09 — the fork --emit-gpu pass: SST -> kirgen AST (kill the hand-encoding seam)"
status: todo
claimed_by:
created: 2026-07-16T22:00:00Z
updated: 2026-07-16T22:00:00Z
---

## Description

The big one; replaces hand-encoded typed ASTs. Staged:

1. Fork flag + kernel discovery (attribute or CLI fn list); locate SST access
   point (sst_to_lean.rs neighborhood; bootstrap R2 SST-literal machinery is the
   sibling to crib from).
2. SST walk -> kirgen AST for the expression subset (u32 ops, lets, ite); emit
   via kirgen as a library.
3. Loop subset (while with counter, slice reads/writes).
4. Unsupported constructs = hard, named errors (never silent fallbacks — the
   verus-gpu-parser lesson).

Done when: protolith wrap_delta's certificate is generated from its actual SST
with the hand-encoded AST deleted for that kernel; mis-lowering demonstrably fails
the certificate (mutation test).

## Progress

- (2026-07-16T22:00:00Z) created

## Writeup

