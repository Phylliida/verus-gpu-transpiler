---
title: "kir-07 — witness instantiation against vacuous requires"
status: todo
claimed_by:
created: 2026-07-16T22:00:00Z
updated: 2026-07-16T22:00:00Z
---

## Description

A certificate with contradictory requires verifies vacuously. Guard: kirgen takes
a concrete witness (param values + tiny buffers) per kernel and emits a witness
lemma instantiating the certificate at it (plus, with kir-06, ksafe at the witness).
Witness values can come from the kernel def or a tiny evaluator in kirgen.

Done when: every generated cert has a witness lemma; a test kernel with
contradictory requires is caught (witness lemma fails).

## Progress

- (2026-07-16T22:00:00Z) created

## Writeup

