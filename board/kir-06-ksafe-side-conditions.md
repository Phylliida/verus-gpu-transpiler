---
title: "kir-06 — partial semantics: ksafe conjunct in certificates"
status: todo
claimed_by:
created: 2026-07-16T22:00:00Z
updated: 2026-07-16T22:00:00Z
---

## Description

DESIGN 4.2: production KIR semantics is partial. Add:

- ksafe spec fns (expression + statement + loop level): all buffer/local accesses
  in bounds; Div/Mod (add the ops) require nonzero divisor.
- Certificates gain the ksafe ensures conjunct; kirgen emits the (mostly already
  established) facts.
- Negative test: a deliberately out-of-bounds kernel must FAIL its certificate
  (this is the test that the conjunct has teeth).

Done when: all generated certs carry ksafe; the OOB kernel fails; REPORT updated.

## Progress

- (2026-07-16T22:00:00Z) created

## Writeup

