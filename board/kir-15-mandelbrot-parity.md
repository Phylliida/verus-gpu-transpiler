---
title: "kir-15 — mandelbrot retro-suite: regenerate a shipped kernel"
status: todo
claimed_by:
created: 2026-07-16T22:00:00Z
updated: 2026-07-16T22:00:00Z
---

## Description

The empirical anchor: push one function from the shipped perturbation shaders
(web/mandelbrot_perturbation_n*.wgsl) through the new pipeline and diff.
Divergence = new-pipeline bug or latent old-transpiler bug; both are wins. Convert
the old bug ledger (hex literals, dropped blocks, return-stripping, fn_id) into
named regression tests against kirgen/the adapter.

Depends: kir-04/05/08/10, realistically kir-14 for the full kernel.

Done when: signed_add_to-shape function reproduced + diffed; bug-ledger regression
tests exist and pass.

## Progress

- (2026-07-16T22:00:00Z) created

## Writeup

