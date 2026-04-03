# verus-gpu-transpiler: Design Document

**Status**: Draft v2, open for iteration  
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
| **Our approach** | Structural map preserves eval semantics | Small (~50 lines proof per AST variant) | Verus + source/target semantics + string emission |

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
| GpuIR types + eval semantics | **Verified** | ~400 lines |
| GpuIR well-formedness / overflow | **Verified** | ~200 lines |
| Emit correctness (general, all programs) | **Verified** | ~300 lines |
| tree-sitter front-end | Trusted | ~300 lines |
| WGSL string rendering | Trusted | ~80 lines |
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

```rust
/// Type tag for GpuIR values — maps directly to WGSL types.
pub enum GpuType { I32, U32, F32 }

pub enum GpuBinOp {
    // Integer arithmetic
    Add, Sub, Mul, Div, Mod, Shr,
    // Comparisons (return i32: 1 or 0)
    Lt, Le, Gt, Ge, Eq, Ne,
    // Bitwise (integer only)
    BitAnd, BitOr, BitXor,
    // Float arithmetic (same names, dispatched by GpuType)
    FAdd, FSub, FMul, FDiv,
}

pub enum GpuExpr {
    Const(int),                                         // integer constant
    FConst(f32),                                        // float constant
    Var(nat),                                           // locals[i]
    BinOp(GpuBinOp, Box<GpuExpr>, Box<GpuExpr>),
    Select(Box<GpuExpr>, Box<GpuExpr>, Box<GpuExpr>),  // c != 0 ? a : b
    ArrayRead(nat, Box<GpuExpr>),                       // bufs[arr][idx]
    Call(nat, Seq<GpuExpr>),                            // fn_table[id](args)
    Cast(GpuType, Box<GpuExpr>),                        // type cast (e.g., i32 → f32)
}

/// Barrier scope — matches hardware synchronization primitives.
pub enum BarrierScope {
    /// workgroupBarrier() — all threads in workgroup sync.
    /// Cheapest (~20 cycles). Most common.
    Workgroup,
    /// storageBarrier() — ensures storage buffer writes are visible
    /// to subsequent reads within the workgroup.
    Storage,
    /// Subgroup-level barrier (warp sync). Not available in all
    /// WGSL implementations; maps to __syncwarp() in CUDA.
    Subgroup,
}

pub enum GpuStmt {
    Assign { var: nat, rhs: GpuExpr },
    BufWrite { buf: nat, idx: GpuExpr, val: GpuExpr },
    Seq { first: Box<GpuStmt>, then: Box<GpuStmt> },
    If { cond: GpuExpr, then_body: Box<GpuStmt>, else_body: Box<GpuStmt> },
    For { var: nat, bound: GpuExpr, body: Box<GpuStmt> },
    Barrier { scope: BarrierScope },
    Return,
    Noop,
}

pub struct GpuFunction {
    pub params: Seq<nat>,       // which locals are parameters
    pub body: GpuStmt,
    pub ret_var: nat,           // which local holds the return value
}

pub struct GpuKernel {
    pub n_locals: nat,
    pub n_bufs: nat,
    pub functions: Seq<GpuFunction>,   // helper functions
    pub body: GpuStmt,
    pub thread_dim: ThreadDim,
    pub workgroup_size: (nat, nat, nat),
}
```

### 4.2 Eval semantics

State model (extends ArithExpr's env + arrays):

```rust
pub struct GpuState {
    pub locals: Seq<int>,
    pub bufs: Seq<Seq<int>>,
    pub returned: bool,
}
```

Expression eval — pure, follows `arith_eval_with_arrays` pattern:

```rust
spec fn gpu_eval_expr(e: &GpuExpr, locals: Seq<int>, bufs: Seq<Seq<int>>,
                      fns: Seq<GpuFunction>) -> int
```

Statement eval — follows `staged_eval` / `eval_loop` pattern:

```rust
spec fn gpu_eval_stmt(s: &GpuStmt, state: GpuState, fns: Seq<GpuFunction>) -> GpuState
```

Key semantics:
- `Assign`: update `locals[var]`
- `BufWrite`: update `bufs[buf][eval(idx)]`
- `Seq`: eval first, then eval second (unless returned)
- `If`: branch on `eval(cond) != 0`
- `For`: bounded loop via `gpu_eval_loop` (mutual recursion, same as `eval_loop` in stage.rs)
- `Barrier`: spec-level no-op on per-thread state (like Stage::Barrier).
  The barrier's effect is at the *parallel* level — it's the point where
  all threads' writes become visible. The Stage framework handles this:
  a kernel body with Barrier nodes is decomposed into phases, and each
  phase is verified independently for race freedom. The barrier scope
  determines which threads synchronize (workgroup vs storage vs subgroup).
- `Call`: substitute args into function body, eval, extract return value
- `Return`: set `returned = true`

### 4.3 Well-formedness

- `gpu_expr_wf(e, n_locals, n_bufs)` — structural (indices in range)
- `gpu_expr_fits_i64(e, locals, bufs)` — overflow safety (all intermediates in i64)
- `gpu_stmt_safe(s, state)` — runtime safety through execution

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
/// The WGSL text for an expression computes the same value as gpu_eval_expr.
/// This is the KEY theorem — proved once, covers all expressions.
proof fn lemma_emit_expr_preserves_eval(e: &GpuExpr)
    requires gpu_expr_wf(e, n_locals, n_bufs)
    ensures
        // For all environments where the expression is safe:
        forall |locals: Seq<int>, bufs: Seq<Seq<int>>|
            gpu_expr_fits_i64(e, locals, bufs) ==>
            wgsl_interp(emit_expr(e), locals, bufs) == gpu_eval_expr(e, locals, bufs)
    decreases e
{
    match e {
        GpuExpr::Const(c) => {},                    // emit "42" → interprets as 42 ✓
        GpuExpr::Var(i) => {},                      // emit "v_i" → interprets as locals[i] ✓
        GpuExpr::BinOp(op, a, b) => {               // emit "(a + b)" → interp = eval(a) + eval(b)
            lemma_emit_expr_preserves_eval(a);       //   by IH on a
            lemma_emit_expr_preserves_eval(b);       //   by IH on b
        },
        GpuExpr::Select(c, t, f) => { /* IH on c, t, f */ },
        GpuExpr::ArrayRead(buf, idx) => { /* IH on idx */ },
        GpuExpr::Call(fn_id, args) => { /* IH on each arg */ },
    }
}
```

Similarly for statements:

```rust
proof fn lemma_emit_stmt_preserves_eval(s: &GpuStmt)
    requires gpu_stmt_wf(s, n_locals, n_bufs)
    ensures forall |state: GpuState| gpu_stmt_safe(s, state) ==>
        wgsl_interp_stmt(emit_stmt(s), state) == gpu_eval_stmt(s, state)
    decreases s
```

### 5.3 What is `wgsl_interp`?

This is the one subtlety. We need to say "the WGSL text computes X" without
formalizing all of WGSL. Our approach: define `wgsl_interp` as a Verus spec
function that captures the semantics of ONLY the WGSL constructs we emit.

This is NOT a general WGSL interpreter. It's a ~100-line spec function
mirroring our ~80-line emitter, defining the obvious semantics:
- `"(a + b)"` means `interp(a) + interp(b)`
- `"if (c) { ... } else { ... }"` means branch on `interp(c)`
- `"for (var i = 0; i < n; i++) { ... }"` means bounded iteration
- etc.

The claim "this spec matches real WGSL behavior" is our minimal axiom — it's
the analog of CompCert trusting the C semantics or CakeML trusting the
ISA spec. For integer arithmetic with bounded loops, this is about as
uncontroversial as axioms get.

### 5.4 What gets proved (general, once)

1. **Emit preserves expression eval** — `wgsl_interp(emit(e)) == gpu_eval_expr(e)` for all `e`
2. **Emit preserves statement eval** — `wgsl_interp_stmt(emit(s)) == gpu_eval_stmt(s)` for all `s`
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

    // tree reduction
    let mut stride: u32 = 128;
    while stride > 0
        invariant /* partial reduction correct for current stride */
    {
        if tid < stride {
            smem[tid] = smem[tid] + smem[tid + stride];
        }
        gpu_workgroup_barrier(); // all threads see updated smem
        stride = stride / 2;
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

**Open question**: Extracting write indices from a statement tree (with control
flow) is more complex than from ArithExpr scatter expressions. Should we require
users to express the scatter pattern explicitly via an attribute, or can we infer
it from the GpuStmt structure?

---

## 8. Resolved Decisions

### D1: WgslIR — DROPPED

One IR (GpuIR). No WgslIR. The emit function goes directly from GpuIR to
WGSL strings. The general correctness proof uses `wgsl_interp` (a ~100-line
spec defining the obvious semantics of our WGSL subset) — no separate IR type.

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
`wgsl_interp(emit(k), s) == gpu_eval_kernel(k, s)`.*

Proved once by structural induction. Combined with Verus verification of the
source function, this gives end-to-end: source spec holds → WGSL output
satisfies the same spec.

**Trusted base** (irreducible):
- `wgsl_interp` spec matches real WGSL behavior (uncontroversial for integer
  arithmetic + bounded loops)
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

**Design**: GpuIR supports both integer and float types natively. The GpuExpr
type uses `GpuType { I32, U32, F32 }` tags. At the spec level, float
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

## 9. Open Questions

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
| **1** | GpuExpr + GpuStmt + eval semantics + well-formedness | — |
| **2** | `wgsl_interp` spec + `emit` function + general correctness proof | Phase 1 |
| **3** | WGSL string rendering + naga validation tests | Phase 2 |
| **4** | ArithExpr → GpuExpr embedding + equivalence proof | Phase 1 |
| **5** | tree-sitter front-end parser | Phase 1 |
| **6** | End-to-end demo: vector_add as `#[gpu_kernel]` → WGSL | Phase 3, 5 |
| **7** | Stage integration (race freedom for GpuIR) | Phase 1 |
| **8** | Port tiled GEMM, scan, radix sort | Phase 6, 7 |

**Milestone 1** (Phases 1-3): GpuIR with formal semantics, emit to WGSL,
general correctness proof. Validates core architecture — the hardest part.

**Milestone 2** (Phases 4-6): Full developer experience. Write `#[gpu_kernel]`
Verus function, parse, emit, run on GPU. ArithExpr backward compat.

**Milestone 3** (Phases 7-8): Tiled GEMM in natural Verus, verified and
transpiled. The motivating use case.

---

## 11. References

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
