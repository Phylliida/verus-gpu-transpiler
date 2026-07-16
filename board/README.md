# Task board — verus-gpu-transpiler v2 (KIR)

This directory is a simple task board. **One markdown file = one task.** Add,
claim, and finish tasks just by creating and editing these `.md` files with your
normal file tools — no server, no JSON.

## File format

    ---
    title: short title of the task
    status: todo            # todo | in_progress | done
    claimed_by:            # your sibling id, or a name (optional)
    created: <iso8601>
    updated: <iso8601>
    ---

    ## Description
    what the task is / what "done" looks like

    ## Progress
    - (timestamp) a running log of what you tried / found

    ## Writeup
    (fill this in when done)

## Program map

Spec: `../DESIGN.md`. Landed so far: `kir-00`. Rough dependency order:

- G1 completion (certificates): kir-01..09 — crate promotion, source anchoring,
  cross-crate, restriction lifts, ksafe, vacuity guard, calls, SST adapter.
- G2 (make it run): kir-10..12 — WGSL printer, host+differential harness,
  dispatch lemma.
- G3 (parity & scale): kir-13..15 — CUDA, workgroup phases, mandelbrot parity.
- Customer summit: kir-16 — protolith assign pass end-to-end (also see
  material-synthesis/board/ for the CPU-side M0 track it depends on).

Files starting with `.` or `_`, plus this README, are ignored by the board.
