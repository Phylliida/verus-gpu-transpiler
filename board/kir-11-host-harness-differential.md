---
title: "kir-11 — host shell + bit-for-bit differential harness"
status: todo
claimed_by:
created: 2026-07-16T22:00:00Z
updated: 2026-07-16T22:00:00Z
---

## Description

Unverified host (wgpu or the existing mandelbrot web harness): create buffers,
dispatch one thread per element for v1 map kernels, read back.

- goldens/ directory: inputs + CPU-reference outputs per kernel (CPU reference =
  the verified exec fns from kir-02/protolith).
- Bit-for-bit comparison; any mismatch is a printer/driver bug by construction
  (certificates cover everything upstream).
- Reuse profile_shader.rs plumbing + NixOS Vulkan setup notes from verus-mandelbrot.

Done when: add_limbs and wrap_delta run on GPU bit-identical to CPU over the
golden suite, wired into a script.

## Progress

- (2026-07-16T22:00:00Z) created

## Writeup

