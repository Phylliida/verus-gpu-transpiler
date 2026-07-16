---
title: "kir-02 — source-side anchor: exec add_limbs ensures the generated model"
status: todo
claimed_by:
created: 2026-07-16T22:00:00Z
updated: 2026-07-16T22:00:00Z
---

## Description

The generated loop model (model_out_add_limbs / model_c0_add_limbs) is currently
unanchored — no independently-verified artifact ensures against it. Close the chain:

- Write u32 exec fns: add3_gpu (compare-trick: wrapping adds + (sum < a) carries)
  and add_limbs_gpu (loop over slices, writes out, returns carry).
- ensures: out@ == model_out_add_limbs(a@, b@, old-out@, 0, n) and
  carry == model_c0_add_limbs(...). The wrap-vs-plain arithmetic seam lives HERE
  (this is where split_ifs/omega work belongs per DESIGN).
- This also validates the kernel-safe-source contract on a loop kernel: exact
  functional postcondition against the generated canonical model.

Done when: exec fns verify 0 errors in the kir/probe crate against the GENERATED
model fns (not a hand copy), tactus exec-loop idioms.

## Progress

- (2026-07-16T22:00:00Z) created

## Writeup

