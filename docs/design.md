# verus-gpu-transpiler: Design Document

**Status**: Draft v3 (final review pass)  
**Last updated**: 2026-04-03

## 1. Motivation

verus-cutedsl represents GPU kernels as `ArithExpr` trees — a pure expression
language with no functions, control flow, or let bindings. Every kernel is one
giant expression. This prevents writing modular, performant GPU code (e.g.,
tiled GEMM with helper functions, conditional epilogue, multi-barrier stages).

**Goal**: Write GPU kernels as normal Verus exec functions. Verus verifies them.
A transpiler emits WGSL. The transpiler is proved semantics-preserving.

```rust
// What we want to write:
#[gpu_kernel(workgroup_size(256))]
fn vector_add(
    #[gpu_builtin(thread_id_x)] tid: u32,
    #[gpu_buffer(0, read)] a: &[i32],
    #[gpu_buffer(1, read)] b: &[i32],
    #[gpu_buffer(2, read_write)] out: &mut [i32],
)
    requires tid < a@.len(), a@.len() == b@.len(), a@.len() == out@.len()
    ensures out@[tid as int] == old(a)@[tid as int] + old(b)@[tid as int]
{
    out[tid] = a[tid] + b[tid];
}

// Instead of:
let kernel = KernelSpec {
    guard: ArithExpr::Cmp(CmpOp::Lt, Box::new(ArithExpr::Var(0)),
                          Box::new(ArithExpr::Const(1024))),
    outputs: seq![OutputSpec {
        scatter: ArithExpr::Var(0),
        compute: ArithExpr::Add(
            Box::new(ArithExpr::Index(0, Box::new(ArithExpr::Var(0)))),
            Box::new(ArithExpr::Index(1, Box::new(ArithExpr::Var(0))))),
    }],
};
```

---

## 2. Related Work

### 2.1 No existing system does what we're doing

No prior work combines: (1) a verified source language with proofs (Verus),
(2) verified transpilation to GPU shaders, and (3) verified layout algebra.

Closest:
- **Liu, Bernstein, Chlipala, Ragan-Kelley (POPL 2022)** — verified
  tensor-program *schedule transformations* in Coq. Proves rewrites preserve
  semantics of a pure functional array language. Generates competitive GPU code.
  But verifies *transformations*, not a *transpiler*.
- **Volta (Microsoft Research, 2025)** — first equivalence checker for GPU
  kernels. Black-box verification of optimized PTX against reference. Handles
  tensor cores. But works post-hoc on compiled code, not source-level.
- **HaliVer (2024)** — deductive verification (separation logic) for Halide
  programs. Proves memory safety and functional correctness. But trusts the
  Halide compiler.

### 2.2 Translation validation > verified compilation

**Key insight from the research**: For our use case (tiny source language,
structural transpilation), *translation validation* is far simpler than
CompCert-style verified compilation, with equivalent guarantees.

| Approach | What it proves | Effort | Trust boundary |
|----------|---------------|--------|----------------|
| **Verified compiler** (CompCert) | Compiler correct for ALL inputs | Enormous (100K+ lines Coq) | Proof checker + source/target semantics |
| **Translation validation** (Alive2) | This PARTICULAR output matches input | Per-compilation check | Validator + source/target semantics |
| **Our approach** | Structural map preserves eval semantics | Moderate (~600 lines total, structural induction) | Verus + source/target semantics + string rendering |

Our transpiler is a *structural map* — each source AST node maps to exactly one
target AST node. Proving this preserves semantics is a simple structural
induction, not a simulation relation. This is essentially translation validation
baked into the transpiler itself.

### 2.3 GPU verification landscape

- **GPUVerify** (Imperial) — race/divergence freedom via barrier-interval
  analysis. We already use this approach in our Stage model.
- **Faial** (CAV 2021) — first Coq-mechanized GPU DRF analysis. Memory access
  protocols.
- **DarthShader** (CCS 2024) — fuzzer that found 39 bugs (15 CVEs) in Naga,
  Tint, and wgslc. Demonstrates the verification gap in shader compilers.
- **rust-gpu** (Embark) — Rust to SPIR-V via rustc backend. No formal
  verification.

### 2.4 Our verus-cutedsl is the first mechanized CuTe verification

Jay Shah's "Note on the Algebra of CuTe Layouts" (2024) and "Categorical
Foundations for CuTe Layouts" (Carlisle, Shah, Stern, 2026) provide rigorous
math but are not mechanized. Our ~840 verified functions are the first
machine-checked formalization.

---

## 3. Architecture

### 3.1 Design principles

1. **One IR.** GpuIR is the only custom intermediate representation. No WgslIR,
   no SPIR-V formalization. Minimal moving parts.

2. **Prove emit correct once, for all programs.** A single general theorem
   (structural induction on GpuIR) proves the emitter preserves semantics.
   No per-kernel translation proofs.

3. **Emit WGSL, not SPIR-V.** Let naga/tint optimize (constant folding, dead
   code elimination, loop transformations) when compiling WGSL → SPIR-V.
   Emitting SPIR-V directly would bypass these optimizations. WGSL is also
   structurally closer to GpuIR (both have mutable variables, structured
   if/else/for) vs SPIR-V's SSA form.

4. **SPIR-V was considered and rejected.** SPIR-V is Vulkan's IR and has a
   more formal spec than WGSL, but: (a) formalizing even a minimal subset
   requires ~45 opcodes + SSA + structured control flow merge semantics,
   (b) no existing mechanized SPIR-V formalization exists (POPL 2023 only
   covers control flow structure), (c) we'd miss shader compiler optimizations.

### 3.2 Pipeline

```
Verus source ──> [parse] ──> GpuIR ──> [emit] ──> WGSL ──> [naga] ──> SPIR-V ──> GPU
                  trusted      │        trusted     optimized
                  ~300 lines   │        ~80 lines   by naga/tint
                               │
                    VERIFIED (one proof, all programs):
                    - GpuIR eval semantics
                    - emit preserves eval (structural induction)
                    - overflow safety, bounds safety
                    - race freedom (via Stage)
```

The key: `emit` is a structural map (each GpuIR node → one WGSL construct).
The general correctness proof is structural induction — one case per AST
variant, proved once, covers all well-formed programs.

### 3.3 Trust boundary

| Component | Status | Size |
|-----------|--------|------|
| GpuIR types + eval semantics | **Verified** | ~1000 lines |
| GpuIR well-formedness + safety | **Verified** | ~400 lines |
| `wgsl_semantics` + emit correctness proof | **Verified** | ~600 lines |
| Performance property specs (coalescing, bank conflicts) | **Verified** | ~200 lines |
| tree-sitter front-end | Trusted | ~700 lines |
| WGSL string rendering | Trusted | ~150 lines |
| naga/tint (WGSL → SPIR-V) | Trusted | — |
| GPU driver + silicon | Trusted | — |

**No per-kernel proofs needed.** The general emit-correctness theorem covers
all well-formed GpuIR programs. Per-kernel, Verus verifies the source function
satisfies its spec (standard Verus). The parser (trusted) connects them.

### 3.4 What changes, what stays

| Component | Status |
|-----------|--------|
| CuTe layout algebra (shape, layout, composition, tiling, swizzle) | **Unchanged** |
| Stage parallel reasoning (barriers, race freedom, SharedState) | **Adapted** — works with GpuIR |
| Scan/sort/GEMM spec-level correctness properties | **Unchanged** |
| ArithExpr + KernelSpec | **Superseded** — backward-compat embedding provided |
| WgslExpr + emit() + codegen crate | **Replaced** — by verified transpiler |

---

## 4. GpuIR Design

### 4.1 Types

See **Section 8.4** for the complete, authoritative GpuIR type definitions
(updated after the hardware faithfulness audit). Key types:

- **ScalarType**: `I32, U32, F32, F16, Bool`
- **GpuType**: `Scalar(ScalarType), Vec2/3/4(ScalarType), Mat{cols,rows}, Void`
- **GpuValue**: `Int, Float, Half, Bool, Vec, Mat` — spec-level value model
- **GpuExpr** (~25 variants): constants, vars, builtins, arithmetic,
  memory access, function calls, casts, vector/matrix ops, packed types,
  subgroup ops
- **GpuStmt** (~12 variants): `Assign, BufWrite, TextureStore, AtomicRMW,
  Block, If, For, Break, Continue, Barrier, Return, Noop`
- **GpuKernel**: typed buffer/texture bindings with `MemorySpace`, helper
  functions, body, workgroup_size, feature flags
- **GpuBinOp** includes wrapping variants (`WrappingAdd`, etc.) for
  faithful-to-hardware wrapping arithmetic

### 4.2 Value model

GpuIR has multiple types (scalars, vectors, matrices). The spec-level value
type represents all possible runtime values:

```rust
pub enum GpuValue {
    Int(int),                                   // i32, u32
    Float(f32),                                 // f32
    Half(f32),                                  // f16 (stored as f32 in spec)
    Bool(bool),
    Vec(Seq<GpuValue>),                         // vec2..vec4 (2-4 components)
    Mat(Seq<Seq<GpuValue>>),                    // mat columns (2-4 cols of vec)
}
```

State model:

```rust
pub struct GpuState {
    pub locals: Seq<GpuValue>,                  // typed local variables
    pub bufs: Seq<Seq<GpuValue>>,               // buffer contents
    pub returned: bool,
}
```

### 4.3 Eval semantics

Expression eval — returns `GpuValue`:

```rust
spec fn gpu_eval_expr(e: &GpuExpr, state: &GpuState,
                      fns: &Seq<GpuFunction>) -> GpuValue
```

Statement eval — follows `staged_eval` / `eval_loop` pattern:

```rust
spec fn gpu_eval_stmt(s: &GpuStmt, state: GpuState,
                      fns: &Seq<GpuFunction>) -> GpuState
```

Key semantics:
- `Assign`: update `locals[var]`
- `BufWrite`: update `bufs[buf][eval(idx)]` (idx evaluates to Int)
- `Block`: eval statements left to right (unless returned)
- `If`: branch on `eval(cond)` being truthy (non-zero int or true bool)
- `For`: bounded loop via `gpu_eval_loop` (mutual recursion, same as
  `eval_loop` in stage.rs). Supports `start..end` range.
- `Barrier`: spec-level no-op on per-thread state (like Stage::Barrier).
  The barrier's effect is at the *parallel* level — it's the point where
  all threads' writes become visible. The Stage framework handles this:
  a kernel body with Barrier nodes is decomposed into phases, and each
  phase is verified independently for race freedom. The barrier scope
  determines which threads synchronize (workgroup vs storage vs subgroup).
- `Call`: substitute args into function body, eval, extract return value
- `Break`: exit innermost loop (set a `broken` flag in eval state)
- `Continue`: skip to next iteration
- `Return`: set `returned = true`

### 4.4 Well-formedness

- `gpu_expr_wf(e, n_locals, n_bufs)` — structural (indices in range, types match)
- `gpu_expr_safe(e, state)` — runtime safety: integer overflow (for
  non-wrapping ops), float finiteness (for `add_req`), array bounds
- `gpu_stmt_safe(s, state)` — runtime safety propagated through execution

### 4.4 Backward compatibility

```rust
spec fn arith_to_gpu_expr(e: &ArithExpr) -> GpuExpr { /* 1:1 */ }

proof fn lemma_arith_gpu_equiv(e: &ArithExpr, env: Seq<int>, arrays: Seq<Seq<int>>)
    ensures gpu_eval_expr(&arith_to_gpu_expr(e), env, arrays, seq![])
         == arith_eval_with_arrays(e, env, arrays)
```

---

## 5. Verified Emit — One Proof, All Programs

### 5.1 The structural correspondence

The emitter is a structural map: each GpuIR node maps to exactly one WGSL
construct with identical integer semantics. We prove this correspondence
ONCE, by structural induction, and it covers all well-formed programs.

The key observation: GpuIR and our WGSL subset are *isomorphic languages*.
Both have:
- Integer constants and variables
- Binary arithmetic (`+`, `-`, `*`, `/`, `%`, `>>`)
- Comparisons returning 0/1
- Conditional expressions (`select`)
- Array read/write
- `if`/`else`, bounded `for` loops
- Function calls, `return`

They differ only in syntax (Rust-like vs C-like) and representation (Verus
spec types vs text strings). The semantics are identical by construction.

### 5.2 How the general proof works

We define a spec function `emit_expr_correct` that expresses the structural
correspondence, then prove it by induction:

```rust
/// GpuIR eval and WGSL semantics agree on all well-formed expressions.
/// KEY theorem — proved once by structural induction, covers all programs.
proof fn lemma_eval_matches_wgsl(e: &GpuExpr)
    requires gpu_expr_wf(e, n_locals, n_bufs)
    ensures
        forall |state: &GpuState|
            gpu_expr_safe(e, state) ==>
            wgsl_semantics_expr(e, state) == gpu_eval_expr(e, state, &seq![])
    decreases e
{
    match e {
        GpuExpr::Const(c, _) => {},                  // both return GpuValue::Int(c) ✓
        GpuExpr::Var(i, _) => {},                     // both return state.locals[i] ✓
        GpuExpr::BinOp(op, a, b) => {                 // both apply op to sub-evals
            lemma_eval_matches_wgsl(a);               //   by IH on a
            lemma_eval_matches_wgsl(b);               //   by IH on b
        },
        GpuExpr::VecConstruct(components) => {        // component-wise IH
            /* IH on each component */
        },
        // ... one case per variant, each trivial by IH ...
    }
}
```

Similarly for statements:

```rust
proof fn lemma_eval_stmt_matches_wgsl(s: &GpuStmt)
    requires gpu_stmt_wf(s, n_locals, n_bufs)
    ensures forall |state: GpuState| gpu_stmt_safe(s, state) ==>
        wgsl_semantics_stmt(s, state) == gpu_eval_stmt(s, state, &seq![])
    decreases s
```

### 5.3 What is `wgsl_semantics`?

The emit proof needs a "WGSL-side" interpretation to compare against
`gpu_eval`. Key insight: since emit is a structural map, we define
`wgsl_semantics` **directly on GpuIR nodes**, not on strings. It captures
"what WGSL would compute for this construct":

```rust
/// WGSL semantics for a GpuExpr — defined on the IR, not on strings.
/// Captures what a conformant WGSL implementation computes.
spec fn wgsl_semantics_expr(e: &GpuExpr, state: &GpuState) -> GpuValue
```

Then the general proof is:

```
gpu_eval_expr(e, state) == wgsl_semantics_expr(e, state)   // for all e
```

Both are spec functions on GpuIR. The proof is structural induction showing
they agree on every variant. The trusted claim is: "`wgsl_semantics_expr`
faithfully models what a WGSL implementation does." For integer addition,
this is `a + b = a + b`. For bounded for-loops, this is iteration. For
float ops, this is IEEE 754 arithmetic. Uncontroversial.

The **string rendering** (`emit: GpuIR → WGSL text`) is a separate trusted
layer (~80 lines). It's correct iff it renders each GpuIR node as the WGSL
syntax that `wgsl_semantics` describes. This is auditable by inspection.

This avoids the awkwardness of parsing strings in Verus spec functions.
The analog: CompCert trusts its C semantics spec, CakeML trusts its ISA
spec. We trust `wgsl_semantics` — a ~100-line spec on well-typed IR nodes.

### 5.4 What gets proved (general, once)

1. **Eval matches WGSL semantics (exprs)** — `wgsl_semantics_expr(e) == gpu_eval_expr(e)` for all `e`
2. **Eval matches WGSL semantics (stmts)** — `wgsl_semantics_stmt(s) == gpu_eval_stmt(s)` for all `s`
3. **GpuIR eval is total and deterministic**
4. **Overflow safety propagates** — well-formed + fits-i64 is maintained through eval
5. **Array bounds safety** — well-formed implies all accesses in bounds
6. **ArithExpr embeds faithfully** — `arith_to_gpu_expr` preserves eval

### 5.5 No per-kernel translation proofs

The general theorems (5.4) cover all well-formed programs. Per-kernel, the
only proof obligation is what Verus already does: verify the source function
satisfies its `requires`/`ensures`. No auto-generated bridging proofs needed.

The trust chain for any kernel:
1. Verus proves source function satisfies its spec (standard Verus)
2. Parser produces GpuIR from source (trusted, ~300 lines)
3. General theorem: emit preserves GpuIR eval (proved once)
4. Therefore: WGSL output has same semantics as source function

The gap is step 2 (parser correctness). A parser bug would produce wrong GpuIR.
But the GpuIR is inspectable, and parser bugs are caught by testing (emit the
WGSL, run it on test data, compare against the Verus exec function on the same
data).

---

## 6. Front-End

### 6.1 Parser architecture

Uses tree-sitter-verus (already in the workspace) to parse `#[gpu_kernel]`
functions. The parser is ~300 lines of CST walking — trusted but auditable.

Supported Rust subset:
- `let x: T = expr;` / `let mut x: T = expr;` → `Assign`
- `x = expr;` → `Assign`
- `buf[idx] = val;` → `BufWrite`
- `if cond { ... } else { ... }` → `If`
- `while cond { ... }` with `invariant` → `For`
- `for i in 0..n { ... }` with `invariant` → `For`
- `return;` → `Return`
- Arithmetic: `+`, `-`, `*`, `/`, `%`, `>>`, `&`, `|`, `^`
- Comparisons: `<`, `<=`, `>`, `>=`, `==`, `!=`
- Array read: `buf[expr]`
- Function call: `helper(args)` → `Call`

Rejected: closures, trait methods, `&`/`&mut` (except buffer params), heap
allocation, recursion, unbounded loops, pattern matching.

### 6.2 GPU annotations

```rust
#[gpu_kernel(workgroup_size(X, Y, Z))]     // required on kernel fn
#[gpu_builtin(thread_id_x)]                // maps to gid.x
#[gpu_builtin(workgroup_id_x)]             // maps to wgid.x
#[gpu_buffer(binding = N, read)]           // storage buffer, read-only
#[gpu_buffer(binding = N, read_write)]     // storage buffer, mutable
#[gpu_shared]                              // workgroup shared memory
```

Barriers are called directly in kernel code with explicit scope:

```rust
gpu_workgroup_barrier();   // workgroupBarrier() — sync all threads in workgroup
gpu_storage_barrier();     // storageBarrier() — flush storage buffer writes
gpu_subgroup_barrier();    // subgroup barrier (warp sync)
```

The scope matters for both correctness and performance:
- Workgroup barrier: all threads in the workgroup reach the barrier before
  any proceed. Shared memory writes become visible. ~20 cycles.
- Storage barrier: storage buffer writes become visible to subsequent reads
  within the workgroup. No thread synchronization.
- Subgroup barrier: threads within a subgroup (warp) synchronize. Cheapest
  but smallest scope.

Example with barriers:

```rust
#[gpu_kernel(workgroup_size(256))]
fn reduce_kernel(
    #[gpu_builtin(thread_id_x)] tid: u32,
    #[gpu_buffer(0, read)] input: &[f32],
    #[gpu_buffer(1, read_write)] output: &mut [f32],
    #[gpu_shared] smem: &mut [f32; 256],
)
    requires tid < 256
{
    smem[tid] = input[tid];
    gpu_workgroup_barrier();     // all threads have written to smem

    // tree reduction: 8 rounds (log2(256) = 8), stride = 128, 64, ..., 1
    for round in 0..8u32
        invariant /* partial reduction correct after `round` rounds */
    {
        let stride: u32 = 128 >> round;
        if tid < stride {
            smem[tid] = smem[tid] + smem[tid + stride];
        }
        gpu_workgroup_barrier(); // all threads see updated smem
    }

    if tid == 0 {
        output[0] = smem[0];
    }
}
```

These are Rust outer attributes — Verus ignores them (treats as custom
attributes), the transpiler front-end reads them.

### 6.3 Helper functions

The user writes helper functions as normal Rust `fn`s in the same module.
The parser recognizes them (no `#[gpu_kernel]` attribute) and includes them
in `GpuKernel.functions`. They're emitted as WGSL `fn` definitions.

```rust
fn clamp(x: i32, lo: i32, hi: i32) -> i32
    requires lo <= hi
    ensures lo <= result <= hi
{
    if x < lo { lo }
    else if x > hi { hi }
    else { x }
}

#[gpu_kernel(workgroup_size(256))]
fn my_kernel(...) {
    let val = clamp(buf[tid], 0, 255);  // calls helper
    out[tid] = val;
}
```

---

## 7. Parallel Correctness

The Stage framework adapts to GpuIR-based kernels with minimal changes.

### 7.1 Simple map kernels

A GpuKernel with a single `BufWrite` and no barriers maps to `Stage::Map`.
The scatter expression is the write index; the compute expression is the write
value. Race freedom requires scatter injectivity (same as today).

### 7.2 Multi-barrier kernels

`gpu_barrier()` calls in the source decompose the kernel body into phases
separated by `Stage::Barrier`. The existing barrier-interval analysis applies.

### 7.3 Race freedom predicate

```rust
spec fn gpu_writes_injective(k: &GpuKernel, bufs: Seq<Seq<int>>,
                              tid1: nat, tid2: nat) -> bool
{
    // for each BufWrite in k.body reachable under the control flow,
    // the write index differs for tid1 vs tid2
}
```

Per D7: the user proves race freedom explicitly via companion proof functions.

---

## 8. Hardware Faithfulness Audit

Comprehensive check of WGSL compute shader features against our GpuIR model.
Goal: don't abstract away anything the user might need for correct, performant
kernels.

### 8.1 What we model (covered in GpuIR)

| Hardware feature | GpuIR representation | Faithful? |
|---|---|---|
| Integer arithmetic (i32, u32) | GpuExpr::BinOp + GpuType | Yes |
| Float arithmetic (f32) | GpuExpr::BinOp(FAdd, ..) + GpuType::F32 | Yes |
| Storage buffers (read, read_write) | `#[gpu_buffer]` annotation + BufWrite/ArrayRead | Yes |
| Workgroup shared memory | `#[gpu_shared]` annotation + GpuState.bufs | Yes |
| Barriers (workgroup, storage, subgroup) | GpuStmt::Barrier { scope } | Yes |
| Bounded loops | GpuStmt::For | Yes |
| Conditionals | GpuStmt::If | Yes |
| Function calls | GpuExpr::Call | Yes |
| Thread builtins (global_id, local_id, etc.) | `#[gpu_builtin]` params | Yes |
| Type casts | GpuExpr::Cast | Yes |
| No recursion | Enforced by parser | Yes |
| No unbounded loops | For-only, no while | Yes |

### 8.2 What's MISSING — adding now

**Atomic operations** (CORRECTNESS — essential for cross-workgroup algorithms):

```rust
pub enum AtomicOp {
    Load, Store, Add, Sub, Max, Min,
    And, Or, Xor, Exchange, CompareExchangeWeak,
}

// New GpuStmt variant:
AtomicRMW { buf: nat, idx: GpuExpr, op: AtomicOp, val: GpuExpr, result_var: nat },
```

WGSL atomics use **relaxed ordering only** — no acquire/release/seq_cst.
This is a hardware limitation we model faithfully. Cross-workgroup sync
requires atomics + careful algorithm design (e.g., decoupled lookback).
User must prove: atomic operations are well-typed (only on `atomic<i32>`
or `atomic<u32>` in storage/workgroup space).

**Subgroup (warp) operations** (PERFORMANCE — essential for efficient reductions):

```rust
pub enum SubgroupOp {
    Add, Mul, Min, Max, And, Or, Xor,       // reductions
    ExclusiveAdd, InclusiveAdd,              // scans
    Broadcast, BroadcastFirst,               // communication
    Shuffle, ShuffleXor, ShuffleUp, ShuffleDown,
    Ballot, All, Any, Elect,                 // voting
}

// New GpuExpr variant:
SubgroupOp(SubgroupOp, Box<GpuExpr>),
```

Requires `enable subgroups;` in WGSL. Must be called in **subgroup-uniform
control flow** — user proves this as an additional obligation.

**Uniform buffers** (CORRECTNESS — different semantics from storage):

```rust
#[gpu_uniform(binding = N)]   // maps to var<uniform>
```

Read-only, all threads see same data. Hardware broadcasts efficiently.
Important for kernel parameters (dimensions, strides, constants).

**Wrapping arithmetic** (CORRECTNESS — faithful to hardware):

```rust
pub enum GpuBinOp {
    // ... existing ops (default: proven overflow-free) ...
    // Wrapping variants: no overflow proof needed, wraps at 32 bits
    WrappingAdd, WrappingSub, WrappingMul,
}
```

WGSL i32/u32 arithmetic wraps silently. By default we require the user to
prove no overflow (catches bugs). But for algorithms that intentionally use
wrapping (hash functions, checksums), we provide explicit wrapping ops.

**Uniform control flow obligation** (CORRECTNESS — UB if violated):

Barriers and subgroup operations require **uniform control flow**: all threads
in the scope must reach the call. Violating this is UB in WGSL. We model this
as a proof obligation the user must satisfy:

```rust
proof fn barrier_uniform_control_flow(tid1: nat, tid2: nat)
    requires
        tid1 < workgroup_size, tid2 < workgroup_size,
    ensures
        // both threads reach the barrier (same branch taken)
        reaches_barrier(kernel_body, tid1, barrier_id)
        == reaches_barrier(kernel_body, tid2, barrier_id)
```

**All builtins** (CORRECTNESS — complete set):

```rust
pub enum GpuBuiltin {
    GlobalInvocationId { dim: nat },    // gid.x/y/z
    LocalInvocationId { dim: nat },     // lid.x/y/z
    LocalInvocationIndex,               // linearized local id
    WorkgroupId { dim: nat },           // wgid.x/y/z
    NumWorkgroups { dim: nat },         // dispatch dimensions
    SubgroupId,                         // which subgroup within workgroup
    SubgroupInvocationId,               // thread index within subgroup
    SubgroupSize,                       // subgroup width (typically 32 or 64)
}
```

### 8.3 Additional features (included, not deferred)

All features below are included in GpuIR to avoid leaving performance or
correctness gaps. The principle: if real GPU kernels need it, we model it.

**f16 (half precision)**:

Requires `enable f16;` in WGSL. Critical for ML workloads (transformer
inference, mixed-precision training). GpuType gains `F16` variant.
Arithmetic ops apply to f16 the same as f32. Verus f16 support TBD —
may need external_body wrappers until Verus adds native f16.

**vec2/vec3/vec4 types** (hardware SIMD):

WGSL has `vec2<T>`, `vec3<T>`, `vec4<T>` for T in {f32, f16, i32, u32}.
These map to hardware SIMD lanes. A vec4 load is up to 4x faster than
four scalar loads. GpuIR support:

```rust
pub enum GpuType {
    I32, U32, F32, F16, Bool,
    Vec2(Box<GpuType>),   // vec2<f32>, vec2<i32>, etc.
    Vec3(Box<GpuType>),
    Vec4(Box<GpuType>),
    Mat(nat, nat),        // mat2x2 through mat4x4 (always f32 or f16)
}

// New GpuExpr variants for vectors:
VecConstruct(Seq<GpuExpr>),              // vec3(x, y, z)
VecComponent(Box<GpuExpr>, nat),         // v.x (0), v.y (1), v.z (2), v.w (3)
Swizzle(Box<GpuExpr>, Seq<nat>),         // v.xyz, v.xz, v.ww, etc.
```

Component-wise arithmetic works automatically: `vec3 + vec3` applies `+`
per component. The emit proof extends by structural induction — vec cases
are component-wise applications of the scalar case.

**Matrix types**:

`mat2x2<f32>` through `mat4x4<f32>` (and f16). Matrix-vector multiply
(`mat * vec`) and matrix-matrix multiply (`mat * mat`) are single WGSL
expressions. GpuIR support:

```rust
// New GpuExpr variants for matrices:
MatConstruct(nat, nat, Seq<GpuExpr>),    // mat3x3(col0, col1, col2)
MatMul(Box<GpuExpr>, Box<GpuExpr>),      // mat * mat or mat * vec
Transpose(Box<GpuExpr>),                  // transpose(m)
Determinant(Box<GpuExpr>),               // determinant(m) (2x2, 3x3, 4x4)
```

**Texture load/store**:

`textureLoad(tex, coords)` and `textureStore(tex, coords, value)` on storage
textures. Used in image processing compute kernels. GpuIR support:

```rust
pub enum TextureOp { Load, Store }

// New GpuExpr variant:
TextureOp { tex: nat, coords: Box<GpuExpr>, value: Option<Box<GpuExpr>> },
```

User must prove: coords in bounds for texture dimensions.

**Indirect dispatch**:

Dispatch dimensions read from a buffer instead of host-specified constants.
Not a GpuIR change — it's a runtime/dispatch-level feature. The kernel code
is identical; only the dispatch call changes. Modeled as a variant in the
dispatch metadata, not in GpuIR itself.

**Packed types**:

`pack4x8snorm(vec4<f32>) -> u32` and `unpack4x8snorm(u32) -> vec4<f32>`.
Compact encoding for normalized values. GpuIR support:

```rust
pub enum PackFormat { SNorm, UNorm, SInt, UInt }

// New GpuExpr variants:
Pack4x8(PackFormat, Box<GpuExpr>),       // vec4<f32> → u32
Unpack4x8(PackFormat, Box<GpuExpr>),     // u32 → vec4<f32>
```

**Memory coalescing proofs** (performance verification):

Global memory loads are coalesced when adjacent threads access adjacent
addresses. Uncoalesced access can be 10-32x slower. We model this as a
provable property:

```rust
/// Adjacent threads access adjacent memory (coalesced load).
spec fn is_coalesced_access(
    buf: nat, idx_expr: &GpuExpr, workgroup_size: nat
) -> bool {
    forall |tid: nat| tid + 1 < workgroup_size ==>
        gpu_eval_expr(idx_expr, env_for(tid+1), bufs)
        == gpu_eval_expr(idx_expr, env_for(tid), bufs) + 1
}
```

Not required for correctness, but the user CAN prove it as an optimization
guarantee. We provide library lemmas for common patterns (linear access,
strided access with known stride).

**Bank conflict proofs** (performance verification):

Shared memory has 32 banks (4-byte stride). Two threads in a warp accessing
the same bank (different address) causes serialization. We model this:

```rust
/// No bank conflicts within a warp for shared memory access.
spec fn bank_conflict_free(
    smem_idx_expr: &GpuExpr, warp_size: nat
) -> bool {
    forall |t1: nat, t2: nat|
        t1 < warp_size && t2 < warp_size && t1 != t2 ==>
        // different bank OR same address (broadcast)
        bank(gpu_eval_expr(smem_idx_expr, env_for(t1), bufs))
        != bank(gpu_eval_expr(smem_idx_expr, env_for(t2), bufs))
        || gpu_eval_expr(smem_idx_expr, env_for(t1), bufs)
        == gpu_eval_expr(smem_idx_expr, env_for(t2), bufs)
}

spec fn bank(addr: int) -> nat { (addr / 4) % 32 }
```

Again, optional for correctness but provable for performance. The existing
swizzle infrastructure in verus-cutedsl was designed exactly for this —
`swizzled_offset` eliminates bank conflicts by permuting addresses.

### 8.4 Updated GpuIR types (complete)

```rust
// ── Scalar and composite types ──────────────────────────────

pub enum ScalarType { I32, U32, F32, F16, Bool }

pub enum GpuType {
    Scalar(ScalarType),
    Vec2(ScalarType),
    Vec3(ScalarType),
    Vec4(ScalarType),
    Mat { cols: nat, rows: nat, elem: ScalarType },  // mat2x2..mat4x4
    Void,                                             // for kernel/helper return
}

// ── Operators ───────────────────────────────────────────────

pub enum GpuBinOp {
    // Integer arithmetic (proven overflow-free)
    Add, Sub, Mul, Div, Mod, Shr, Shl,
    // Wrapping integer arithmetic (no overflow proof, wraps at 32 bits)
    WrappingAdd, WrappingSub, WrappingMul,
    // Float arithmetic (f32 and f16)
    FAdd, FSub, FMul, FDiv,
    // Comparisons (return Bool)
    Lt, Le, Gt, Ge, Eq, Ne,
    // Bitwise (integer only)
    BitAnd, BitOr, BitXor,
    // Logical (bool only)
    LogicalAnd, LogicalOr,
}

pub enum GpuUnaryOp { Neg, FNeg, Not, LogicalNot }

pub enum AtomicOp {
    Load, Store, Add, Sub, Max, Min,
    And, Or, Xor, Exchange, CompareExchangeWeak,
}

pub enum SubgroupOp {
    // Reductions
    Add, Mul, Min, Max, And, Or, Xor,
    // Scans
    ExclusiveAdd, InclusiveAdd, ExclusiveMul, InclusiveMul,
    // Communication
    Broadcast(nat),  // lane index
    BroadcastFirst,
    Shuffle, ShuffleXor, ShuffleUp, ShuffleDown,
    // Voting
    Ballot, All, Any, Elect,
}

pub enum BarrierScope { Workgroup, Storage, Subgroup }

pub enum PackFormat { SNorm, UNorm, SInt, UInt }

// ── Builtins ────────────────────────────────────────────────

pub enum GpuBuiltin {
    GlobalInvocationId { dim: nat },    // vec3<u32>
    LocalInvocationId { dim: nat },     // vec3<u32>
    LocalInvocationIndex,               // u32 (linearized)
    WorkgroupId { dim: nat },           // vec3<u32>
    NumWorkgroups { dim: nat },         // vec3<u32>
    SubgroupId,                         // u32
    SubgroupInvocationId,               // u32
    SubgroupSize,                       // u32
}

// ── Expressions ─────────────────────────────────────────────

pub enum GpuExpr {
    // Constants
    Const(int, ScalarType),             // integer/bool constant with type
    FConst(f32),                        // f32 constant
    HConst(f32),                        // f16 constant (stored as f32, emitted as f16)

    // Variables and builtins
    Var(nat, GpuType),                  // local variable with type
    Builtin(GpuBuiltin),               // thread/workgroup builtins

    // Arithmetic and logic
    BinOp(GpuBinOp, Box<GpuExpr>, Box<GpuExpr>),
    UnaryOp(GpuUnaryOp, Box<GpuExpr>),
    Select(Box<GpuExpr>, Box<GpuExpr>, Box<GpuExpr>),  // cond ? a : b

    // Memory access
    ArrayRead(nat, Box<GpuExpr>),       // buf[idx]
    TextureLoad(nat, Box<GpuExpr>),     // textureLoad(tex, coords)

    // Function call
    Call(nat, Seq<GpuExpr>),            // fn_table[id](args...)

    // Type conversion
    Cast(GpuType, Box<GpuExpr>),        // e.g., i32 → f32, f32 → f16

    // Vector operations
    VecConstruct(Seq<GpuExpr>),         // vec3(x, y, z)
    VecComponent(Box<GpuExpr>, nat),    // v.x (0), v.y (1), v.z (2), v.w (3)
    Swizzle(Box<GpuExpr>, Seq<nat>),    // v.xyz, v.xz, v.ww

    // Matrix operations
    MatConstruct(nat, nat, Seq<GpuExpr>),  // mat3x3(col0, col1, col2)
    MatMul(Box<GpuExpr>, Box<GpuExpr>),    // mat*mat or mat*vec
    Transpose(Box<GpuExpr>),
    Determinant(Box<GpuExpr>),

    // Packed types
    Pack4x8(PackFormat, Box<GpuExpr>),     // vec4<f32> → u32
    Unpack4x8(PackFormat, Box<GpuExpr>),   // u32 → vec4<f32>

    // Subgroup operations
    SubgroupOp(SubgroupOp, Box<GpuExpr>),
}

// ── Statements ──────────────────────────────────────────────

pub enum GpuStmt {
    Assign { var: nat, rhs: GpuExpr },
    BufWrite { buf: nat, idx: GpuExpr, val: GpuExpr },
    TextureStore { tex: nat, coords: GpuExpr, val: GpuExpr },
    AtomicRMW { buf: nat, idx: GpuExpr, op: AtomicOp,
                val: GpuExpr, result_var: Option<nat> },  // None = fire-and-forget
    Block(Seq<GpuStmt>),                                   // flat statement list
    If { cond: GpuExpr, then_body: Box<GpuStmt>, else_body: Box<GpuStmt> },
    For { var: nat, start: GpuExpr, end: GpuExpr,          // for var in start..end
          body: Box<GpuStmt> },
    Break,
    Continue,
    Barrier { scope: BarrierScope },
    Return,
    Noop,
}

// ── Program structure ───────────────────────────────────────

pub struct GpuFunction {
    pub params: Seq<(nat, GpuType)>,
    pub ret_type: GpuType,
    pub body: GpuStmt,
    pub ret_var: nat,
}

pub enum MemorySpace { Storage, Workgroup, Uniform }

pub struct BufferBinding {
    pub binding: nat,
    pub space: MemorySpace,
    pub read_only: bool,
    pub elem_type: GpuType,
}

pub struct TextureBinding {
    pub binding: nat,
    pub read_only: bool,
    pub texel_type: GpuType,
    pub dimensions: nat,           // 1D, 2D, 3D
}

pub struct GpuKernel {
    pub n_locals: nat,
    pub buffers: Seq<BufferBinding>,
    pub textures: Seq<TextureBinding>,
    pub functions: Seq<GpuFunction>,
    pub body: GpuStmt,
    pub workgroup_size: (nat, nat, nat),
    pub enable_subgroups: bool,
    pub enable_f16: bool,
}
```

### 8.5 User proof obligations (complete list)

The user must prove, per kernel:

**Correctness (required):**

1. **Buffer bounds** — all reads/writes within buffer length
2. **Texture bounds** — all texture load/store coords within dimensions
3. **Overflow safety** — non-wrapping arithmetic doesn't overflow i32/u32
4. **Race freedom** — no two threads write to same non-atomic location
   within a barrier interval
5. **Barrier uniformity** — all threads in scope reach each barrier
6. **Subgroup uniformity** — all active threads in subgroup reach each
   subgroup operation
7. **Atomic type safety** — atomics only on i32/u32 in storage/workgroup
8. **Type safety** — operations applied to correct types (vec/mat
   dimensions match, cast targets are valid)
9. **Functional correctness** — the ensures clause (application-specific)

**Performance (optional but provable):**

10. **Memory coalescing** — adjacent threads access adjacent global memory
11. **Bank conflict freedom** — no shared memory bank conflicts within a warp
12. **Occupancy bounds** — shared memory + register usage within SM limits

We provide library lemmas for common patterns (identity scatter, strided
scatter, tree reduction, coalesced linear access, bank-conflict-free
swizzled access, etc.) so users don't re-prove boilerplate.

---

## 9. Resolved Decisions

### D1: WgslIR — DROPPED

One IR (GpuIR). No WgslIR. `wgsl_semantics` is defined directly on GpuIR
nodes (not on strings). The emit function renders GpuIR to WGSL strings
(trusted ~150 lines). The correctness proof shows `gpu_eval == wgsl_semantics`
by structural induction.

### D2: SPIR-V — EMIT WGSL INSTEAD

Emitting SPIR-V directly would bypass naga/tint optimizations (constant
folding, dead code elimination, loop transforms). WGSL also has mutable
variables and structured control flow matching GpuIR, while SPIR-V requires
SSA conversion. If we need SPIR-V later, add a second backend from GpuIR.

### D3: Per-kernel proofs — NOT NEEDED

The general emit-correctness theorem covers all well-formed programs.
No auto-generated per-kernel translation proofs. Per-kernel, Verus does its
normal job (verify source function satisfies requires/ensures).

### D4: Scope of "verified"

**Theorem**: *For all well-formed GpuKernel `k` and valid GpuState `s`,
`wgsl_semantics_kernel(k, s) == gpu_eval_kernel(k, s)`.*

Proved once by structural induction. Combined with Verus verification of the
source function, this gives end-to-end: source spec holds → WGSL output
satisfies the same spec.

**Trusted base** (irreducible):
- `wgsl_semantics` spec matches real WGSL behavior (uncontroversial for
  integer/float arithmetic + bounded loops + vectors + atomics)
- Parser: source → GpuIR (~300 lines, bugs caught by testing)
- naga/tint: WGSL → SPIR-V
- GPU driver + silicon

### D5: Float support — NATIVE f32 via Verus

**Tested (2026-04-03):** Verus supports f32/f64 natively. Results:

| Level | f32 support | Status |
|-------|------------|--------|
| Spec | Full arithmetic (`+`, `*`, `-`, `/`), sequences, recursive specs | Works |
| Proof | Bit representation (`to_bits_spec`), finiteness, NaN/infinity | Works |
| Exec | Passthrough (assign, return) | Works |
| Exec arithmetic | `a + b` requires `add_req` precondition; result is nondeterministic (IEEE rounding modes) | Works with precondition |

**Design**: GpuIR supports both integer and float types natively (see
Section 8.4 for the full `ScalarType`/`GpuType` hierarchy). At the spec level, float
arithmetic is fully available for writing specs (dot products, GEMM
accumulation, etc.). At the exec level, float arithmetic goes through Verus's
`add_req`/`mul_req` preconditions — the user must prove the operation is
well-defined (e.g., finite operands). Float results are nondeterministic
at the exec level (IEEE rounding), which is faithful to hardware.

This means: users can write f32 GPU kernels, specify exact mathematical
behavior in ensures clauses using spec-level f32 arithmetic, and verify that
their index computations / buffer accesses / race freedom are correct. The
numerical precision of float ops is trusted (hardware-faithful), not verified.

### D6: Shared memory — FAITHFUL TO HARDWARE

Principle: model what the hardware actually does. Don't abstract away details.
Users should be able to reason about bank conflicts and precise memory layout.

**Model**: Shared memory is modeled as a flat buffer in GpuState (same as
global buffers, but with `#[gpu_shared]` annotation). Barrier semantics:

```rust
// Before barrier: each thread sees only its own writes to shared memory
// After barrier: all threads' writes become visible
// This matches __syncthreads() / workgroupBarrier() exactly
```

**Spec-level model**: The Stage framework's existing barrier-interval analysis
applies directly. Between barriers, shared memory reads see the pre-barrier
snapshot. After a barrier, the declarative `map_output_declarative` spec
gives the post-barrier state.

**User obligations**: The user must prove:
1. No data races on shared memory within a barrier interval
2. Shared memory accesses are in bounds (array size is compile-time known)
3. Bank conflict freedom (optional — for performance, not correctness)

We provide library lemmas for common patterns (e.g., "linear tid access
to shared memory is race-free and bank-conflict-free").

### D7: Race freedom — USER PROVES IT

Principle: more user proofs = more correctness. The user must prove race
freedom for every buffer write, not rely on inference.

**Mechanism**: For each `#[gpu_kernel]` function that writes to buffers, the
user writes a companion `proof fn` establishing scatter injectivity:

```rust
#[gpu_kernel(workgroup_size(256))]
fn my_kernel(
    #[gpu_builtin(thread_id_x)] tid: u32,
    #[gpu_buffer(0, read_write)] out: &mut [i32],
) { ... }

proof fn my_kernel_race_free(tid1: nat, tid2: nat)
    requires tid1 != tid2, tid1 < 256, tid2 < 256
    ensures
        // write index for tid1 != write index for tid2
        write_index(tid1) != write_index(tid2)
{ ... }
```

For multi-barrier kernels, race freedom must be proved per barrier interval.
The Stage framework provides the compositional reasoning structure.

**Library support**: We provide verified helper lemmas for common patterns:
- `lemma_identity_scatter_injective` — tid → tid writes are race-free
- `lemma_strided_scatter_injective` — tid*stride+offset writes are race-free
- `lemma_tiled_scatter_injective` — tiled GEMM output patterns

### D8: Front-end tooling — MCP + CLI

Both paths:
1. **MCP tool**: `verus_transpile("kernel.rs")` — integrated into the
   development workflow via verus-mcp
2. **CLI**: `verus-gpu-transpile kernel.rs -o kernel.wgsl` — for build scripts
   and CI

Both use the same tree-sitter front-end + GpuIR construction internally.

---

### D9: Type tags — CARRY ON EVERY NODE

GpuExpr carries `GpuType` on every node. Simplifies the emit proof (each
node knows its WGSL type) and matches WGSL's explicitly-typed semantics.
The parser infers types from the Verus source and tags each node.

### D10: No automatic behavior — EVERYTHING EXPLICIT

No implicit transformations, no auto-splitting, no inference of patterns.
The user writes exactly what they mean. The parser does a 1:1 structural
translation from Verus source to GpuIR. If the user wants a barrier, they
write `gpu_workgroup_barrier()`. If they want a type cast, they write
`x as f32`.

### D11: No while loops — FOR ONLY

GPU hardware has no concept of unbounded iteration. Infinite loops are
undefined behavior in WGSL. Only bounded `for` loops are supported:
`for i in 0..n { ... }`. The user provides the bound explicitly. This is
faithful to hardware and simplifies the GpuIR eval semantics (termination
is trivially guaranteed by the bound).

---

## 10. Phased Implementation Plan

| Phase | Deliverable | Depends on |
|-------|-------------|------------|
| **1a** | GpuValue + scalar GpuExpr/GpuStmt + eval semantics | — |
| **1b** | Vec/mat/texture/atomic/subgroup GpuExpr extensions | 1a |
| **1c** | Well-formedness + safety predicates | 1a |
| **2** | `wgsl_semantics` spec + general correctness proof | 1a |
| **3** | WGSL string rendering + naga validation tests | 2 |
| **4** | ArithExpr → GpuExpr embedding + equivalence proof | 1a |
| **5** | tree-sitter front-end parser | 1b |
| **6** | End-to-end demo: vector_add as `#[gpu_kernel]` → WGSL | 3, 5 |
| **7** | Stage integration (race freedom, barrier uniformity) | 1a |
| **8** | Performance proof library (coalescing, bank conflicts) | 1a |
| **9** | Port tiled GEMM, scan, radix sort | 6, 7 |

**Milestone 1** (Phases 1-3): GpuIR with formal semantics, emit to WGSL,
general correctness proof. Validates core architecture — the hardest part.
Start with scalar types, extend to vec/mat/subgroup.

**Milestone 2** (Phases 4-6): Full developer experience. Write `#[gpu_kernel]`
Verus function, parse, emit, run on GPU. ArithExpr backward compat.

**Milestone 3** (Phases 7-9): Tiled GEMM in natural Verus, verified and
transpiled, with race freedom + coalescing proofs. The motivating use case.

**Milestone 4** (Phase 10): CUDA backend with tensor core support.

---

## 11. Dual Target: WGSL + CUDA

### 11.1 Why both

WGSL: portable (WebGPU, all vendors), naga/tint optimize for us, good for
deployment. But WGSL can't express tensor cores, async copy, warp
specialization — ceiling on performance for NVIDIA hardware.

CUDA: state-of-the-art performance on NVIDIA (tensor cores, cp.async, thread
block clusters). But NVIDIA-only, requires nvcc/ptxas toolchain.

The GpuIR is already backend-neutral — it represents GPU *concepts*, not
WGSL-specific syntax. Adding CUDA is a second emitter + second semantics spec.

### 11.2 Architecture

```
                          ┌──> [emit_wgsl] ──> WGSL ──> [naga] ──> SPIR-V
                          │     trusted         portable
GpuIR (one IR) ──────────┤
  verified eval semantics │
  + safety proofs         └──> [emit_cuda] ──> CUDA ──> [nvcc] ──> PTX
                                trusted         NVIDIA, tensor cores
```

Both emitters are trusted (structural maps, ~150 lines each). Both have a
corresponding `*_semantics` spec. Both are proved correct once:

```
gpu_eval(ir) == wgsl_semantics(ir)     // for WGSL backend
gpu_eval(ir) == cuda_semantics(ir)     // for CUDA backend
```

Since both equal `gpu_eval`, they're also equal to each other — the two
backends produce semantically identical results for shared GpuIR programs.

### 11.3 CUDA-specific GpuIR extensions

For features that CUDA has but WGSL doesn't, we add GpuIR variants that
the CUDA backend emits and the WGSL backend rejects:

```rust
pub enum GpuExpr {
    // ... all existing variants (work on both backends) ...

    // CUDA-only: tensor core matrix multiply-accumulate
    TensorCoreMMA {
        m: nat, n: nat, k: nat,              // tile dimensions (e.g., 16x16x16)
        a_frag: Box<GpuExpr>,                // A fragment
        b_frag: Box<GpuExpr>,                // B fragment
        c_frag: Box<GpuExpr>,                // accumulator
    },

    // CUDA-only: async global→shared memory copy
    AsyncCopy {
        dst_smem: nat,                        // shared memory buffer
        src_gmem: nat,                        // global memory buffer
        idx: Box<GpuExpr>,
        size: nat,                            // bytes per thread
    },
}

pub enum GpuStmt {
    // ... all existing variants ...

    // CUDA-only: async copy fence/wait
    AsyncCopyCommit,                          // cp.async.commit_group
    AsyncCopyWait { groups: nat },            // cp.async.wait_group

    // CUDA-only: warp-level matrix store
    TensorCoreStore {
        buf: nat,
        idx: GpuExpr,
        frag: GpuExpr,
    },
}

pub struct GpuKernel {
    // ... existing fields ...
    pub target: GpuTarget,                    // which backend(s) this kernel supports
}

pub enum GpuTarget {
    Both,         // portable: WGSL + CUDA
    WgslOnly,     // WGSL-specific features
    CudaOnly,     // tensor cores, async copy, etc.
}
```

### 11.4 CUDA emit mapping

| GpuIR | CUDA output |
|-------|-------------|
| `Barrier { Workgroup }` | `__syncthreads()` |
| `Barrier { Subgroup }` | `__syncwarp()` |
| `SubgroupOp(ShuffleXor, e)` | `__shfl_xor_sync(0xffffffff, e, lane)` |
| `SubgroupOp(Add, e)` | warp reduction via `__shfl_down_sync` tree |
| `AtomicRMW { Add, .. }` | `atomicAdd(&buf[idx], val)` |
| `TensorCoreMMA { 16,16,16, .. }` | `wmma::mma_sync(d, a, b, c)` |
| `AsyncCopy { .. }` | `cp.async.cg.shared.global` |
| `For { var, start, end, body }` | `for (int var = start; var < end; var++)` |
| `VecConstruct([x,y,z,w])` | `make_float4(x, y, z, w)` |

### 11.5 What this means for the proof

The general correctness proof becomes two proofs (both structural induction):
1. `gpu_eval == wgsl_semantics` (covers portable GpuIR)
2. `gpu_eval == cuda_semantics` (covers portable + CUDA-specific GpuIR)

For CUDA-specific variants (tensor cores, async copy), we define their
`gpu_eval` semantics (what the MMA mathematically computes) and their
`cuda_semantics` (what the CUDA intrinsic computes). The proof connects them.

Tensor core semantics are interesting: `TensorCoreMMA(16,16,16, A, B, C)`
computes `C += A * B` where A is 16x16, B is 16x16. This connects directly
to the existing GEMM correctness specs in `verus-cutedsl/src/gemm.rs`.

### 11.6 Implementation: CUDA backend as Phase 10

| Phase | Deliverable | Depends on |
|-------|-------------|------------|
| **10a** | `cuda_semantics` spec for portable GpuIR | Phase 2 |
| **10b** | CUDA string emitter + nvcc validation tests | 10a |
| **10c** | CUDA correctness proof (structural induction) | 10a |
| **10d** | TensorCoreMMA + AsyncCopy GpuIR extensions | 10a |
| **10e** | Tensor core GEMM in Verus → CUDA | 10d, Phase 9 |

The WGSL backend ships first (Milestone 1-3). CUDA backend follows as
Milestone 4. Both share the same GpuIR, same parser, same proof obligations.

---

## 12. Risks and Limitations

Honest assessment of where this project could be misguided.

### R1: The verification gap is in the wrong place

DarthShader (CCS 2024) found 39 bugs in naga/tint — the shader *compilers*,
not kernel source code. We verify source→IR→WGSL/CUDA but trust naga/tint/nvcc,
which is where the real bugs live. If the goal is "fewer GPU bugs in practice,"
verifying naga would have more impact.

**Mitigation**: Our verification catches a different class of bugs —
algorithmic errors (wrong index, race condition, overflow) rather than
compiler bugs (miscompilation). Both matter. And our verified CuTe layout
algebra prevents the specific class of index computation bugs that are
notorious in GPU code.

### R2: The parser is a 700-line trusted gap

The trust chain is: Verus Rust → (parser, ~700 lines trusted) → GpuIR →
(verified) → WGSL/CUDA. A parser bug silently produces wrong GpuIR. The
general emit proof doesn't help — it proves the wrong GpuIR is emitted
correctly.

**Mitigation**: (a) Parser bugs are caught by differential testing (run WGSL
on GPU, compare against Verus exec on CPU). (b) GpuIR is inspectable — a
human can read it. (c) The parser is structural (one CST node → one GpuIR
node), limiting the space for subtle bugs. (d) Long-term: could verify the
parser against a formal Rust subset semantics, but this is a large effort.

### R3: `wgsl_semantics` / `cuda_semantics` could be wrong

Both `gpu_eval` and `wgsl_semantics` are defined by us. If `wgsl_semantics`
is wrong, the proof is vacuously true. GPU memory models (relaxed atomics,
memory visibility, subgroup semantics) are subtle and differ across vendors.

**Mitigation**: (a) For integer/float arithmetic, the semantics are obvious
(addition is addition). (b) For atomics and barriers, we follow the WGSL/CUDA
spec text carefully. (c) The semantics spec is small (~100 lines per backend)
and auditable. (d) Testing: if `wgsl_semantics` disagrees with what GPUs
actually do, differential testing catches it.

### R4: Float verification is mostly trusted

Verus exec-level f32 arithmetic is nondeterministic (`add_req` but no
`ensures result == a + b`). For float-heavy kernels, the "verified" part
only covers indices, bounds, and races — not the actual computation.

**Mitigation**: This is correct and by design. IEEE 754 float arithmetic
has implementation-defined rounding, fused multiply-add, denormal handling.
Verifying exact float results would require a formal IEEE 754 model (large
effort, exists in Coq but not Verus). Our approach: verify everything except
float precision, which is faithful to what the hardware actually guarantees.

### R5: Proof effort may be disproportionate

For a simple vector_add: 5 lines of kernel, ~20 lines of proofs. For tiled
GEMM: ~50 lines of kernel, potentially hundreds of lines of proofs.
Meanwhile, differential testing (GPU vs CPU reference) catches most bugs
with zero proof effort.

**Mitigation**: (a) The proof effort amortizes — library lemmas for common
patterns (identity scatter, strided scatter, tree reduction) are proved once
and reused. (b) Formal verification catches bugs that testing misses: rare
thread interleavings, edge cases in integer overflow, subtle index errors
that only manifest at specific tensor dimensions. (c) For safety-critical
GPU code (autonomous driving, medical imaging), the proof effort is
justified.

### R6: The sequential eval model doesn't capture parallel bugs

`gpu_eval_stmt` evaluates one thread sequentially. Real GPU bugs involve
thread interactions: races, deadlocks, barrier divergence. The Stage model
addresses some of this, but complex patterns (producer-consumer, warp
specialization, decoupled lookback) may not fit.

**Mitigation**: (a) The Stage framework is extensible — new patterns can be
added as new Stage variants with new proof obligations. (b) The barrier
uniformity proof obligation catches divergence bugs. (c) The race freedom
proof obligation catches data races. Together these cover the most common
parallel bugs. (d) More exotic patterns (warp specialization) can be
modeled as they arise.

### R7: Scope is large

The original problem was "add functions to ArithExpr." The solution grew to
a complete verified GPU transpiler with ~25 GpuExpr variants, dual WGSL/CUDA
backends, performance proofs, and a full hardware audit. This is a
multi-month project.

**Mitigation**: (a) The phased plan delivers value incrementally — Milestone 1
(scalar GpuIR + WGSL) is useful on its own. (b) The scope is large because
the problem is large — half-measures (just adding `Let` to ArithExpr) would
hit the same walls again soon. (c) The design doc front-loads the hard
thinking; implementation should be faster because decisions are made.

### R8: Tree-sitter parser maintenance burden

tree-sitter-verus must track Verus language evolution. Verus syntax changes
(new keywords, attribute formats) will break the parser.

**Mitigation**: (a) We already maintain tree-sitter-verus in this workspace.
(b) The GPU kernel subset of Verus is small and stable (basic control flow,
arithmetic, buffer access). (c) Changes to Verus spec syntax (ghost, proof,
requires/ensures) don't affect us — we strip those in parsing.

### R9: Target audience is small

The intersection of "writes GPU kernels" and "uses formal verification" is
tiny. Even CompCert is rarely used in industry.

**Mitigation**: (a) The CuTe layout algebra is already the first of its kind
— we're building for a new category. (b) As AI/ML accelerator code becomes
safety-critical (autonomous vehicles, medical devices), the audience grows.
(c) The verified transpiler could be used as infrastructure by other tools
(e.g., a verified ML compiler) even if individual developers don't use it
directly.

---

## 13. References

1. Liu, Bernstein, Chlipala, Ragan-Kelley. "Verified Tensor-Program
   Optimization via High-Level Scheduling Rewrites." POPL 2022.
2. Leroy. "Formal Verification of a Realistic Compiler." CACM 2009. (CompCert)
3. DarthShader: Fuzzing WebGPU Shader Translators. CCS 2024.
4. Betts et al. "GPUVerify: A Verifier for GPU Kernels." OOPSLA 2012.
5. Cogumbreiro et al. "Faial: Memory Access Protocols." CAV 2021.
6. Volta: Equivalence Checking of ML GPU Kernels. Microsoft Research, 2025.
7. HaliVer: Deductive Verification for Halide. 2024.
8. Shah. "A Note on the Algebra of CuTe Layouts." 2024.
9. Carlisle, Shah, Stern. "Categorical Foundations for CuTe Layouts." 2026.
10. Pnueli, Siegel, Singerman. "Translation Validation." TACAS 1998.
11. Necula. "Translation Validation for an Optimizing Compiler." PLDI 2000.
