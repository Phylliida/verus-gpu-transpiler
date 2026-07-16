//! Minimal KIR fragment for the G1 probe (DESIGN.md §4).
//!
//! Just enough AST to express the add-limbs loop body: locals, two read-only
//! input buffers, one output buffer, wrapping u32 add, compare-as-0/1.
//! Values are u32; wrapping arithmetic is mod-then-cast (the mod result is
//! always in [0, 2^32), so the cast is lossless and needs no truncation
//! reasoning). Eval is totalized with 0-defaults for the probe; production
//! KIR makes the semantics partial with side conditions (§4.2) — the probe
//! measures result-equality discharge only.
//!
//! Spec fns are in dependency order (Lean-backend structural-equality
//! convention).

use vstd::prelude::*;

verus! {

/// 2^32, the wrap modulus.
pub open spec fn kwrap() -> int {
    0x1_0000_0000
}

/// Wrapping u32 add: (x + y) mod 2^32. The mod lands in [0, 2^32), so the
/// cast back to u32 is exact.
pub open spec fn wadd(x: u32, y: u32) -> u32 {
    (((x as int) + (y as int)) % kwrap()) as u32
}

/// Compare-as-0/1: the KIR encoding of a bool used arithmetically.
pub open spec fn ltu(x: u32, y: u32) -> u32 {
    if x < y { 1u32 } else { 0u32 }
}

/// Wrapping u32 subtract: Euclidean mod keeps the result in [0, 2^32).
pub open spec fn wsub(x: u32, y: u32) -> u32 {
    (((x as int) - (y as int)) % kwrap()) as u32
}

/// Wrapping u32 multiply.
pub open spec fn wmul(x: u32, y: u32) -> u32 {
    (((x as int) * (y as int)) % kwrap()) as u32
}

/// (x >= y) as 0/1.
pub open spec fn geu(x: u32, y: u32) -> u32 {
    if x >= y { 1u32 } else { 0u32 }
}

/// (x <= y) as 0/1.
pub open spec fn leu(x: u32, y: u32) -> u32 {
    if x <= y { 1u32 } else { 0u32 }
}

/// KIR expressions (probe fragment).
pub enum KExpr {
    /// Literal constant.
    Const(u32),
    /// Read local variable by index.
    Loc(nat),
    /// Read input buffer A at computed index.
    ReadA(Box<KExpr>),
    /// Read input buffer B at computed index.
    ReadB(Box<KExpr>),
    /// Wrapping u32 add.
    AddW(Box<KExpr>, Box<KExpr>),
    /// (lhs < rhs) as 0/1.
    LtU(Box<KExpr>, Box<KExpr>),
    /// Wrapping u32 subtract.
    SubW(Box<KExpr>, Box<KExpr>),
    /// Wrapping u32 multiply.
    MulW(Box<KExpr>, Box<KExpr>),
    /// (lhs >= rhs) as 0/1.
    GeU(Box<KExpr>, Box<KExpr>),
    /// (lhs <= rhs) as 0/1.
    LeU(Box<KExpr>, Box<KExpr>),
    /// Branchless conditional: cond != 0 picks the second argument.
    Select(Box<KExpr>, Box<KExpr>, Box<KExpr>),
}

/// KIR statements (probe fragment).
pub enum KStmt {
    /// locals[loc] := rhs
    Assign { loc: nat, rhs: KExpr },
    /// out[idx] := val
    WriteOut { idx: KExpr, val: KExpr },
    /// first; then
    Seq2(Box<KStmt>, Box<KStmt>),
}

/// Per-thread interpreter state: mutable locals and the output buffer.
/// Input buffers are read-only and passed alongside.
pub struct KState {
    pub locals: Seq<u32>,
    pub out: Seq<u32>,
}

/// Expression evaluation. Totalized: out-of-range local/buffer reads give 0
/// (never reached by certified programs; the probe proves result equality,
/// under which the in-bounds branches are forced).
pub open spec fn keval(e: KExpr, st: KState, a: Seq<u32>, b: Seq<u32>) -> u32
    decreases e,
{
    match e {
        KExpr::Const(c) => c,
        KExpr::Loc(i) => {
            if (i as int) < st.locals.len() { st.locals[i as int] } else { 0u32 }
        },
        KExpr::ReadA(ie) => {
            let i = keval(*ie, st, a, b);
            if (i as int) < a.len() { a[i as int] } else { 0u32 }
        },
        KExpr::ReadB(ie) => {
            let i = keval(*ie, st, a, b);
            if (i as int) < b.len() { b[i as int] } else { 0u32 }
        },
        KExpr::AddW(x, y) => wadd(keval(*x, st, a, b), keval(*y, st, a, b)),
        KExpr::LtU(x, y) => ltu(keval(*x, st, a, b), keval(*y, st, a, b)),
        KExpr::SubW(x, y) => wsub(keval(*x, st, a, b), keval(*y, st, a, b)),
        KExpr::MulW(x, y) => wmul(keval(*x, st, a, b), keval(*y, st, a, b)),
        KExpr::GeU(x, y) => geu(keval(*x, st, a, b), keval(*y, st, a, b)),
        KExpr::LeU(x, y) => leu(keval(*x, st, a, b), keval(*y, st, a, b)),
        KExpr::Select(c, t, e) => {
            if keval(*c, st, a, b) != 0 {
                keval(*t, st, a, b)
            } else {
                keval(*e, st, a, b)
            }
        },
    }
}

/// Statement execution.
pub open spec fn kexec(s: KStmt, st: KState, a: Seq<u32>, b: Seq<u32>) -> KState
    decreases s,
{
    match s {
        KStmt::Assign { loc, rhs } => {
            let v = keval(rhs, st, a, b);
            if (loc as int) < st.locals.len() {
                KState { locals: st.locals.update(loc as int, v), ..st }
            } else {
                st
            }
        },
        KStmt::WriteOut { idx, val } => {
            let i = keval(idx, st, a, b);
            let v = keval(val, st, a, b);
            if (i as int) < st.out.len() {
                KState { out: st.out.update(i as int, v), ..st }
            } else {
                st
            }
        },
        KStmt::Seq2(s1, s2) => kexec(*s2, kexec(*s1, st, a, b), a, b),
    }
}

/// Loop runner: the interpreter's For semantics. Before each iteration the
/// loop counter is written into local `ivar` (mirrors the v1 GpuStmt::For
/// contract), then the body runs. Counts lo..n.
pub open spec fn kloop(
    body: KStmt, ivar: nat, lo: nat, n: nat,
    st: KState, a: Seq<u32>, b: Seq<u32>,
) -> KState
    decreases n - lo,
{
    if lo >= n {
        st
    } else {
        let st1 = if (ivar as int) < st.locals.len() {
            KState { locals: st.locals.update(ivar as int, lo as u32), ..st }
        } else {
            st
        };
        let st2 = kexec(body, st1, a, b);
        kloop(body, ivar, lo + 1, n, st2, a, b)
    }
}

} // verus!
