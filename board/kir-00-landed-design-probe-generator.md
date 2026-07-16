---
title: "kir-00 — LANDED: DESIGN v0.1 + G1 probe + kirgen (expr & loop certs)"
status: done
claimed_by:
created: 2026-07-16T22:00:00Z
updated: 2026-07-16T22:00:00Z
---

## Description

Record of the 2026-07-16 arc (commits bd962dc, 7553bed, 7fa4650, 5c15473, de8ea26):

- DESIGN.md v0.1: certificate-checked transpiler; three-artifact audit (verus-gpu-parser
  bug ledger, v1 interpreter vacuities, ArithLimb wf_spec hole); four-link trust chain;
  no-optimizer commitment; G0-G3 ladder.
- probe-g1/: KIR fragment + add-limbs certificate hand-proven (37/0, closer-free);
  five-shape mechanical discharge schema; REPORT.md.
- kirgen/: generator for expression kernels (abs_delta, wrap_delta, sum_sq3 -> targets
  spec_kernels copies of protolith specs) and loop kernels (add_limbs from an 8-line
  typed def). gen_certs.rs = 1138 generated lines, crate 75/0, first-run unedited.
- Design amendment: certificates target functional-spec fns; exec == spec == KIR
  closes without SST reflection for kernels with exact functional postconditions.

Key discovered constraints are in probe-g1/REPORT.md (tactus_auto attr on proof fns;
u_* unfold layer mandatory-but-free; vstd Seq axioms suffice; tactic vocabulary of six).

## Progress

- (2026-07-16T22:00:00Z) created

## Writeup

