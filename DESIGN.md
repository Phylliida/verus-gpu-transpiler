# verus-gpu-transpiler v2 — a certificate-checked GPU transpiler for tactus kernels

*Design document v0.1, 2026-07-16. Supersedes `docs/design.md` (v3, April 2026), which is retained as history. Grounded in the 2026-07-16 audit of the three GPU-codegen artifacts in this workspace; the audit is §2, and everything after it is downstream of what the audit found.*

## 0. Thesis and scope

We have verified CPU kernels (tactus/Lean backend) and we want the same computations on the GPU without re-trusting a compiler we wrote ourselves. The v1 attempt split into three artifacts, and the one that actually shipped pixels — a tree-sitter text-to-WGSL transpiler — is precisely the one with no verification story and a bug ledger to show for it (§2.1).

The v2 design keeps what v1 empirically validated and repairs the trust chain with one architectural move: **translation validation via per-kernel certificates**. The transpiler is not proved correct; each of its *outputs* is, automatically, on every build. A kernel's emitted IR comes with a machine-checked theorem that it computes the same function as the verified source. A compiler bug then cannot produce a wrong shader; it produces a build failure.

The claim we can honestly make at the end (trusted base enumerated in §3.3):

> For every emitted kernel, the kernel IR provably computes the same function as its verified tactus source, machine-checked per build, under semantics that exactly match the GPU target on the operation subset used. The IR-to-text printer is a transparent 1:1 mapping, and CPU/GPU outputs are bit-identical on the golden suite.

Out of scope, permanently: an optimizer (§6 — the evidence says we never need one), floats (until lean-flocq earns its keep on a real kernel), atomics and general shared-memory concurrency (a weak memory model buys nothing that our kernel discipline doesn't get for free).

First customers: protolith (material-synthesis) phase-2 GPU port; mandelbrot-parity regeneration of the shipped perturbation shaders; cutedsl layout-heavy kernels later.

## 1. Architecture at a glance

```
verified tactus kernel fn  (#[kernel], kernel-safe subset §4.1)
        │
        ▼
[fork pass] SST → KIR lowering            structural, monomorphizing, dumb
        │           │
        │           └──► certificate obligation:  kir_result(K_f, ·) == f(·)
        │                 discharged by the Lean backend on every build (§5)
        ▼
KIR  (statement-level kernel IR, u32, partial semantics §4)
        │
        ▼
[trusted printer] KIR → WGSL / CUDA text   1:1, tiny, naga-validated (§7)
        │
        ▼
[unverified host shell] buffers, dispatch, readback — protolith-style I/O shell
        │
        ▼
bit-for-bit differential harness vs the CPU reference (§8)
```

Two once-proved theorems bracket the per-kernel certificates: the dispatch lemma (parallel map over disjoint writes equals the sequential loop, §4.4) and, later, the barrier-phase composition lemma (v1.5, §4.5).

## 2. The audit — what v1 actually proved (receipts)

Three artifacts, three different trust stories. This section exists so v2 never accidentally re-tells v1's story about itself.

### 2.1 verus-gpu-parser — what shipped, and its bug ledger

The shipped mandelbrot shaders (`verus-mandelbrot/web/mandelbrot_perturbation_n{4,8,16,32,64}.wgsl`, 1769 lines each) were produced by `verus-gpu-parser` (~4.2k lines): tree-sitter parses the Verus *source text*, builds a kernel IR, `emit.rs` prints WGSL. It is statement-level and structure-preserving — real WGSL functions, loops, `var` locals, tuple-return structs, per-size monomorphization (`signed_add_to___local_8`), workgroup shared memory. It self-describes as "the trusted component."

Its bug ledger, from git history and session notes, all found by *visual* debugging (pink tiles, zoom noise), each costing days:

- `0xFFFF` parsed as `0` (`i64::parse` doesn't do hex) — silently corrupted all multi-precision arithmetic;
- bare `{ }` blocks silently dropped;
- `return expr;` emitted as bare `return;`;
- inline comments in call arguments parsed as variable names;
- fn_id corruption from a stale remap pass;
- a "text-fallback extraction" path — when parsing failed, source text was pasted through.

Every one of these is caught at build time by a certificate. That is the empirical case for this design; nothing in §3–§5 is speculative safety, it is this list, prevented.

### 2.2 verus-gpu-transpiler v1 — the semantics that verified the wrong thing

The v1 crate proved `gpu_eval == wgsl_semantics` — two interpreters agree. But: no theorem connects the emitted *text* to either interpreter; the two interpreters are same-author near-copies (the binop agreement lemma closes with an empty body — definitional equality, a consistency check, not grounding); `GpuValue::Int(int)` is unbounded, so plain ops don't model hardware wrapping and Euclidean-vs-truncated division diverges on negatives; edge cases are total-ized wrongly (model says div-by-zero is 0; WGSL defines `x/0 == x`); `MatMul`, `Determinant`, `Pack4x8`, and all subgroup ops evaluate to literal 0 — vacuous for any kernel touching them. And there is no front end: kernels were to be hand-authored in IR with hand proofs, which is the labor a transpiler exists to delete.

Salvage: the statement-level IR *shape* (Seq/If/For/Call with a fuel interpreter, structural-decreases discipline) is right, and survives into KIR.

### 2.3 ArithLimb staging — elegant, and unsound at the joint

`verus-fractals/gpu_codegen.rs` instantiates the `LimbOps` trait at symbolic expression trees: run the verified generic algorithm at the symbolic type and the IR falls out. Beautiful idea — the verified program is its own front end. But the joint is unsound twice over: `wf_spec()` is literally `{ true }`, so nothing ties the built expression to the ghost `model` that carries all the proofs (one could emit `Const(0)` per limb and still verify — postconditions are satisfied by ghost bookkeeping alone); and cutedsl's `arith_eval` gives `Add` unbounded-int semantics while the built trees assume u32 wrapping (`carry = (sum < a)` is only correct under wrap). It also never shipped, and being expression-level it cannot express the loops, early exits, and local arrays the real kernels needed.

Verdict: deferred, not dead (appendix A). The genuinely-verified precedent to keep citing is cutedsl-codegen's ArithExpr lemmas (`lemma_offset_expr_correct` et al.), where a *constructed expression* is proved to match a spec function — that is a real certificate, just for a fragment.

## 3. The trust chain

Four links, each with a different verification story. Naming them prevents the v1 failure mode of verifying hard at one link while another silently carries the load.

**Link 1 — source.** A kernel is a verified tactus exec fn in the kernel-safe subset (§4.1). This link is free; it is what this workspace does all day.

**Link 2 — lowering.** A pass in the tactus fork lowers the kernel's SST to a KIR literal and *generates* the certificate obligation (§5). The pass itself is unverified and allowed to be wrong; wrongness fails the build.

**Link 3 — IR semantics.** KIR's semantics are honest by *restriction and partiality*, not by modeling all of WGSL (§4.2). Side conditions (division nonzero, indices in bounds) are part of the certificate, so the kernel provably never reaches the edges where GPU semantics are weird — and then the model needs no opinion about those edges.

**Link 4 — printer and host.** KIR → target text stays trusted-but-tiny (1:1, one construct one line, reviewable in one sitting, naga-validated for WGSL), backstopped by the bit-for-bit differential harness. The host shell (buffers, dispatch, readback) is the protolith-style deliberately-unverified surface, enumerated in one module.

### 3.3 The trusted base, enumerated

The Lean kernel and the tactus toolchain (shrinking under the bootstrap program's W-ladder); the KIR printer (§7); naga only as a sanity gate, not a soundness dependency; the host shell; the GPU driver, shader compiler, and silicon. Everything else is checked per build.

## 4. KIR — the kernel IR

### 4.1 The kernel-safe source subset

A `#[kernel]` fn must be: integer/fixed-point only (u32 values; u64 only where the target has it, §7); allocation-free (outputs and scratch are caller-provided slices with offsets); loops bounded with `decreases`; no recursion (bounded algorithm recursion is unrolled at source level — the shipped one-level Karatsuba is the pattern); no traits in the kernel path after monomorphization; writes confined to `out[own(tid)]` for a statically-declared disjoint ownership map (v1: `own(tid) = {tid}` or a contiguous slice per tid).

Everything on this list is a restriction the optimized mandelbrot kernels *already obeyed* — the subset is descriptive, not aspirational.

### 4.2 Values and semantics: u32, wrapping, partial

- Values are `u32` with exact mod-2³² wrapping on add/sub/mul — matching WGSL and CUDA hardware semantics precisely. No unbounded `int` anywhere in the target-facing semantics (the v1 hole, §2.2).
- Comparisons produce 0/1 (printed as `select(0u, 1u, cond)` or `u32(cond)`); `select` is a first-class op.
- `/` and `%` carry a side condition: divisor provably nonzero. Array reads and writes carry in-bounds side conditions. The semantics is *partial*; the certificate includes the side conditions; the model deliberately says nothing about div-by-zero or out-of-bounds because certified kernels cannot reach them. This single move deletes the need to formalize WGSL's edge behavior, which is good, because no trustworthy mechanization of WGSL exists to borrow.
- Signed values: sign-magnitude at source level (u32 magnitude limbs + u32 sign word), exactly as `verus-fixed-point` already does. No i32 in KIR v1, which kills the Euclidean-vs-truncated division mismatch outright.
- No floats, no atomics, no subgroup ops. Not "modeled as 0" (the v1 vacuity) — *absent from the AST*. Unsupported constructs in a kernel are a hard, named build error. No silent fallbacks of any kind (the text-fallback lesson, §2.1).

### 4.3 Statements — the feature set optimization actually demanded

The mandelbrot sessions are the requirements document; every item below is load-bearing in the shipped 72x-optimized shaders:

| KIR construct | validated by |
|---|---|
| let-bound locals; `var` mutables | everywhere |
| functions, kept as functions (not inlined) | shipped shaders stay 1769 lines at n=64 because structure survives |
| tuple returns as small structs | `sub_borrow` → `R2 { f0, f1 }` |
| for-loops, runtime bounds (`n` as parameter) | all limb loops |
| early exit: `break` / early `return` | escape check, periodicity detection |
| thread-local fixed arrays; pointers to them as fn params, monomorphized per size | `___local_8` mangling; direct-mode fallback |
| scratch buffers with offset discipline | Karatsuba `tmp1/tmp2` + offsets, copybuf |
| storage buffers (read, read_write); params buffer; `gid` builtin | GPU-side `c = center + (gid − w/2)·step` |
| cmp-as-carry, select, `Shr`/`Mod` by constants (16-bit-halves mulhi) | `add3`, `mul2` — no u64 on WGSL |

Multi-kernel pipelines (refine rounds, vote buffers) stay host-side orchestration of individually-certified kernels; the pipeline is the shell's job, like protolith's C↔D age loop.

### 4.4 Dispatch model: certified map, once-proved scheduling

v1 kernels are pure maps: thread `tid` reads anything, writes only `own(tid)`. The dispatch lemma is proved once at the KIR level: for pairwise-disjoint ownership, any interleaving of thread executions produces the same buffers as the sequential `for tid in 0..N` loop. Per-kernel certificates then only ever speak about one thread's sequential execution. All source-level parallel structure (8-color checkerboards, JFA ping-pong, per-column ownership) already fits this shape — protolith's pass inventory was designed race-free, and the mandelbrot kernels are per-pixel maps.

### 4.5 v1.5: barrier-phased workgroup memory

The shipped perturbation shader uses `var<workgroup> wg_mem: array<u32, 8192>` (32 KB, tile-shared reference orbit), so mandelbrot parity needs shared memory. The v1.5 extension: workgroup arrays with execution split into *phases* separated by `workgroupBarrier()`; within each phase, writes to the shared array are disjoint per thread (a per-phase side condition, same machinery as §4.4); the phase-composition lemma is proved once. This stays a data-race-free discipline — no weak memory model, ever. Protolith needs none of this for M0–M2, so it gates nothing early.

## 5. The certificate front end

### 5.1 Obligation shape

For each kernel `f` the fork pass emits (a) the KIR literal `K_f`, (b) a generated proof fn:

```
proof fn certificate_f(inputs: ..., tid: u32)
    requires f_requires(inputs, tid)
    ensures
        kir_safe(K_f, env(inputs, tid)),            // side conditions: div≠0, in-bounds
        kir_result(K_f, env(inputs, tid)) == f(inputs, tid),
```

discharged by the Lean backend like any other obligation, on every build, cached like any other function (unchanged kernel + unchanged pass version = cache hit).

### 5.2 Why discharge should be mostly automatic — and the gate if it isn't

The lowering is structural: source expression nodes map 1:1 to KIR nodes, so straight-line bodies should close by interpreter unfolding (`simp`/definitional). Loops use *state-equality bisimulation*: each source loop maps to one KIR `For`, and the induction invariant is full-state equality — deliberately stronger and dumber than the source's functional invariants, so the discharge never needs to understand what the loop *means*, only that both sides take the same step. Recursive-induction discharge under the Lean backend is a solved idiom (probe32, 2026-07-15), which is the reason to believe this paragraph.

**The G1 probe gate (§9):** if bisimulation discharge is not mostly-automatic on `add_limbs_to` (loop + carry chain + locals — deliberately not a toy), the architecture gets revisited before any further investment: fallback options are fuel-bounded unrolling for static-bound loops, per-construct tactic support in the fork, or narrowing v1 to straight-line kernels plus hand loop lemmas. The design does not proceed on hope past this gate.

### 5.3 Monomorphization

Per-size kernel instances (the `___local_N` pattern, protolith's per-preset dims) each get their own KIR literal and certificate — certificates are per-monomorphization, which is strictly stronger than one generic proof and matches how the cache thinks anyway. `generate_shaders.sh`'s multi-N flow survives as the driver.

### 5.4 Relationship to the bootstrap program

This is a sibling of DESIGN-bootstrap.md's R2 certificate architecture — SST-literal reflection plus a checked equivalence, on a target radically smaller than refWp. Idioms, and possibly plumbing, transfer in both directions; a working G1 is incidental evidence for the bootstrap reflection route on easy ground.

## 6. What this transpiler will never do: optimize

The strategic finding of the audit, stated as a design commitment. Every optimization in the 72x mandelbrot speedup lives in the *verified source*: re²/im² reuse across the escape check (40% fewer multiplies), O(n) `scalar_mul_int` replacing O(n²) schoolbook for the offset multiply, the Karatsuba n≥8 threshold, periodicity checkpointing with early exit, first-iteration-by-copy. The transpiler that shipped all of that is dumb and faithful; structure preservation (functions stay functions, loops stay loops) is the only "optimization" codegen needs.

Consequences: no optimizer pass, no CSE, no scheduling, no cost model — the hardest parts of a verified compiler are simply not in this project. When a kernel is slow, the fix is a better verified source algorithm, measured by the existing profiling harness (`profile_shader.rs`, RTX 3090 baselines, NixOS Vulkan setup already documented).

## 7. Targets and printers

**WGSL first.** The browser runtime plumbing exists (mandelbrot viewer, buffer layouts, JS shell), and WGSL's strictness — no u64 — forces the portable discipline early. The u64 gap is closed at source level, not in codegen: 32×32→64 goes via the 16-bit-halves `mul2` (validated), and wide sort/selection keys are `(hi, lo)` u32 pairs under lexicographic compare — protolith's composite keys (`dist2`, `voxel index`) decompose this way naturally and never need u64 at all.

**CUDA second.** Native u64 and sane integer semantics make the CUDA printer the easy port once KIR exists; protolith's DESIGN §11 CUDA commitment is served through it. SPIR-V via naga's WGSL path covers the Vulkan compute route (the `.comp.spv` heritage in verus-mandelbrot).

**Printer discipline (the residual trust):** one KIR construct → one target construct; no conditionals in the printer that depend on kernel content beyond the construct at hand; the whole printer reviewable in one sitting, in one file; emitted WGSL validated by naga in CI as a sanity gate. The printer inherits `emit.rs`'s knowledge (struct returns, pointer-to-local-array calling convention, workgroup declarations) with its parser half deleted.

## 8. Differential testing and goldens

Bit-for-bit CPU-vs-GPU equality is meaningful because everything is integer (protolith DESIGN §12.2 reasoning, inherited here):

- **Retro-suite:** regenerate the five shipped perturbation shaders through the new pipeline and diff against `web/mandelbrot_perturbation_n*.wgsl`. Any divergence is a new-pipeline bug or a latent old-transpiler bug; both are wins. The old bug ledger (§2.1) becomes named regression tests.
- **Golden outputs per kernel per preset**, recorded from the verified CPU reference from day one (protolith M0 goldens start this suite).
- **Property replay on GPU output:** cheap Tier-2-style checks (quota counts, partition spot-checks) run on read-back buffers in the harness — catching driver/printer faults the certificates cannot see.

## 9. Milestones

- **G0 — KIR core.** Gut this crate: KIR AST (statement-level, §4.3 feature set), u32 wrapping partial semantics, fuel interpreter, `kir_safe`, dispatch lemma (§4.4). Port to the tactus Lean backend with a crate-local `check.sh` (group-theory pattern). Exit: `torus_dist2` hand-lowered, hand-certified, printed, executed, bit-identical to CPU. *(No fork work yet.)*
- **G1 — the probe, then the pass.** Probe: hand-generate the certificate obligation for `add_limbs_to` and measure discharge automation under the Lean backend — this is the §5.2 gate. Then: the fork pass (`--emit-gpu`) lowering SST → KIR literal + obligation, straight-line fns first, loops second. Exit: protolith `sum_sq3`/`torus_dist2`/`wrap_delta` certificates with zero hand proof; `add_limbs_to` with at most localized hand lemmas.
- **G2 — printers, shell, harness.** WGSL printer + naga CI + host shell + differential harness; retro-suite diff (§8). Exit: one protolith M0 kernel (the assign pass) end-to-end on GPU, bit-identical image.
- **G3 — parity and customers.** Mandelbrot perturbation kernel re-certified end-to-end (requires v1.5 shared-memory phases, §4.5); CUDA printer; protolith phase-2 passes migrate as they land on the CPU side.

Sizing honestly: G0 is a slab.rs-scale arc times two or three; G1's probe is days and its pass is the M6-scale heart of the project; G2 is engineering with existing parts; G3 rides on G1. After G1, every future kernel across protolith, mandelbrot, and cutedsl costs one attribute.

## 10. Risks and open questions

**Discharge automation (the real one).** Gated at G1 (§5.2) with named fallbacks; the project does not proceed past a failed gate. **Certificate build cost:** expect seconds per kernel per package-check timings; per-function caching applies; if a monster kernel appears, split it at source level like any other slow proof. **KIR/SST impedance:** tactus SST details (temporaries, place expressions) may not map 1:1 to KIR lets; the lowering may need a normalization step, which grows the surface the certificate must absorb — acceptable, since the certificate absorbs it by construction, but watch for blowup. **Monomorphization explosion:** per-size certificates multiply check time by the size menu (n ∈ {4..64} is ×5); fine at current scale, revisit if a customer needs dozens of sizes. **Printer drift vs WGSL evolution:** naga in CI catches syntax; semantics drift is covered by the differential harness; WGSL's integer core is stable. **v1.5 shared memory:** the phase-discipline proof is new machinery; scoped to G3 so it cannot block protolith. **The `own(tid)` declaration language:** starts as `{tid}`/contiguous-slice; JFA ping-pong and checkerboard passes need offset patterns — extend the vocabulary when the first real pass demands it, not before.

## 11. Prior art, in-workspace and out

In-workspace: DESIGN-bootstrap.md R2 (SST-literal certificates — the sibling architecture); cutedsl-codegen ArithExpr lemmas (a real construction-certificate, fragment-scale); verus-fixed-point LimbOps (the generic-algorithm discipline, and the honest carry/mulhi recipes KIR inherits); tactus decide-checker idioms; protolith DESIGN §11–§12 (the differential-testing contract and the fixed-point/integer commitment this design inherits). Out-of-workspace: CompCert (the trusted-printer precedent and the translation-validation literature around it — Necula & Lee's proof-carrying code, Pnueli's translation validation); Alive2 as the modern per-output-validation exemplar.

---

## Appendix A — the staging front end, deferred

The LimbOps-at-symbolic-type idea (§2.3) is worth keeping on the shelf: for straight-line arithmetic fragments it produces IR from the verified algorithm with no fork machinery at all. Reviving it honestly requires: a wrapping-u32 evaluation model (not unbounded int); `wf_spec` becoming a real invariant `arith_eval_u32(expr, env) == model` maintained by every constructor; and let-binding/sharing to bound tree growth. If G1's fork-pass route stalls at the gate, this is fallback ground for the arithmetic-only subset — protolith's per-voxel key computations would fit. It can never cover loops or early exit, so it cannot be the whole story.

## Appendix B — disposition of the v1 artifacts

`verus-gpu-transpiler` (this crate): gutted in place at G0; v1 sources retained under `docs/v1/` for reference. `verus-gpu-parser`: retired from the build; `emit.rs` mined for printer knowledge; its bug ledger converted to regression tests in the G2 harness; the tree-sitter front half deleted from the trust story entirely. `verus-fractals` ArithLimb path: left as-is (appendix A governs any revival). Shipped shaders in `verus-mandelbrot/web/`: frozen as the retro-suite oracles (§8).
