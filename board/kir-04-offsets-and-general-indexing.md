---
title: "kir-04 — lift v1 indexing restrictions: offsets, non-counter reads"
status: todo
claimed_by:
created: 2026-07-16T22:00:00Z
updated: 2026-07-16T22:00:00Z
---

## Description

kirgen v1 panics on: buffer reads not indexed by the counter, counter-as-value,
offsets. Protolith's assign kernel reads seed data at seed-indexed positions and
the fixed-point *_to fns use a_off/out_off. Lift:

- Read index = arbitrary KIR expression; certificate carries the in-bounds side
  condition (interacts with kir-06).
- Offset params as scalar locals; counter usable as an arithmetic value (cast
  bridge emission generalizes).
- Generate + verify an offset variant of add_limbs (add_limbs_to shape).

Done when: offset add_limbs generates and verifies unedited; v1 panics replaced by
supported paths or precise unsupported-construct errors (no silent fallbacks).

## Progress

- (2026-07-16T22:00:00Z) created

## Writeup

