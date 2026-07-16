---
title: "kir-13 — CUDA printer"
status: todo
claimed_by:
created: 2026-07-16T22:00:00Z
updated: 2026-07-16T22:00:00Z
---

## Description

Second printer, same KIR: native u32 (and later u64) ops, one thread per
element. Protolith DESIGN 11 committed to CUDA for phase 2; the printer is small
next to kir-10. Same differential harness (kir-11) gains a CUDA backend.

Done when: one kernel runs bit-identical under CUDA on the golden suite.

## Progress

- (2026-07-16T22:00:00Z) created

## Writeup

