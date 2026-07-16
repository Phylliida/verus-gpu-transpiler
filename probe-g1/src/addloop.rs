//! G1 probe, kernel side: the add-limbs loop.
//!
//! `model_*` spec fns mirror the *source shape* (the u32 compare-trick add3,
//! per DESIGN.md §7 — WGSL-target sources are u32-only). `stmt_*`/`body_kir`
//! are the hand-lowered KIR literal the --emit-gpu pass would produce. The
//! certificate lemmas (step bisimulation, loop induction) live at the bottom
//! of this file — they are the discharge-automation measurement.
//!
//! Locals layout (the pass's register allocation):
//!   0 = carry   1 = i   2 = av   3 = bv   4 = ab   5 = c1   6 = abc   7 = c2

use vstd::prelude::*;
use crate::kir::*;

verus! {

// ══════════════════════════════════════════════════════════════
// Source-shape model (what the SST reflection stands for)
// ══════════════════════════════════════════════════════════════

/// One iteration's output limb: abc = wrap(wrap(a[i] + b[i]) + carry).
pub open spec fn model_abc(a: Seq<u32>, b: Seq<u32>, carry: u32, i: nat) -> u32 {
    wadd(wadd(a[i as int], b[i as int]), carry)
}

/// One iteration's carry-out: c1 + c2, the compare-trick pair.
/// Source computes this as a plain (proven in-range) u32 add; the sum is
/// at most 2, so the `as u32` cast is exact.
pub open spec fn model_carry_next(a: Seq<u32>, b: Seq<u32>, carry: u32, i: nat) -> u32 {
    let av = a[i as int];
    let ab = wadd(av, b[i as int]);
    let abc = wadd(ab, carry);
    (ltu(ab, av) + ltu(abc, ab)) as u32
}

/// Carry after processing limbs lo..n.
pub open spec fn model_carry_loop(
    a: Seq<u32>, b: Seq<u32>, carry: u32, lo: nat, n: nat,
) -> u32
    decreases n - lo,
{
    if lo >= n {
        carry
    } else {
        model_carry_loop(a, b, model_carry_next(a, b, carry, lo), lo + 1, n)
    }
}

/// Output buffer after processing limbs lo..n (carry threaded alongside).
pub open spec fn model_out_loop(
    a: Seq<u32>, b: Seq<u32>, out: Seq<u32>, carry: u32, lo: nat, n: nat,
) -> Seq<u32>
    decreases n - lo,
{
    if lo >= n {
        out
    } else {
        model_out_loop(
            a, b,
            out.update(lo as int, model_abc(a, b, carry, lo)),
            model_carry_next(a, b, carry, lo),
            lo + 1, n,
        )
    }
}

// ══════════════════════════════════════════════════════════════
// The KIR literal (what --emit-gpu would generate)
// ══════════════════════════════════════════════════════════════

/// av := a[i]
pub open spec fn stmt_av() -> KStmt {
    KStmt::Assign { loc: 2, rhs: KExpr::ReadA(Box::new(KExpr::Loc(1))) }
}

/// bv := b[i]
pub open spec fn stmt_bv() -> KStmt {
    KStmt::Assign { loc: 3, rhs: KExpr::ReadB(Box::new(KExpr::Loc(1))) }
}

/// ab := av + bv (wrapping)
pub open spec fn stmt_ab() -> KStmt {
    KStmt::Assign {
        loc: 4,
        rhs: KExpr::AddW(Box::new(KExpr::Loc(2)), Box::new(KExpr::Loc(3))),
    }
}

/// c1 := (ab < av) as u32
pub open spec fn stmt_c1() -> KStmt {
    KStmt::Assign {
        loc: 5,
        rhs: KExpr::LtU(Box::new(KExpr::Loc(4)), Box::new(KExpr::Loc(2))),
    }
}

/// abc := ab + carry (wrapping)
pub open spec fn stmt_abc() -> KStmt {
    KStmt::Assign {
        loc: 6,
        rhs: KExpr::AddW(Box::new(KExpr::Loc(4)), Box::new(KExpr::Loc(0))),
    }
}

/// c2 := (abc < ab) as u32
pub open spec fn stmt_c2() -> KStmt {
    KStmt::Assign {
        loc: 7,
        rhs: KExpr::LtU(Box::new(KExpr::Loc(6)), Box::new(KExpr::Loc(4))),
    }
}

/// out[i] := abc
pub open spec fn stmt_write() -> KStmt {
    KStmt::WriteOut { idx: KExpr::Loc(1), val: KExpr::Loc(6) }
}

/// carry := c1 + c2 (wrapping; sum is at most 2, so wrap is identity)
pub open spec fn stmt_carry() -> KStmt {
    KStmt::Assign {
        loc: 0,
        rhs: KExpr::AddW(Box::new(KExpr::Loc(5)), Box::new(KExpr::Loc(7))),
    }
}

/// Right-nested statement chain, named per level so the certificate can
/// unfold one sequencing step at a time (generator-emitted schema).
pub open spec fn chain7() -> KStmt {
    KStmt::Seq2(Box::new(stmt_write()), Box::new(stmt_carry()))
}

pub open spec fn chain6() -> KStmt {
    KStmt::Seq2(Box::new(stmt_c2()), Box::new(chain7()))
}

pub open spec fn chain5() -> KStmt {
    KStmt::Seq2(Box::new(stmt_abc()), Box::new(chain6()))
}

pub open spec fn chain4() -> KStmt {
    KStmt::Seq2(Box::new(stmt_c1()), Box::new(chain5()))
}

pub open spec fn chain3() -> KStmt {
    KStmt::Seq2(Box::new(stmt_ab()), Box::new(chain4()))
}

pub open spec fn chain2() -> KStmt {
    KStmt::Seq2(Box::new(stmt_bv()), Box::new(chain3()))
}

/// The loop body: the eight statements right-nested (head-first unfolding).
pub open spec fn body_kir() -> KStmt {
    KStmt::Seq2(Box::new(stmt_av()), Box::new(chain2()))
}

/// Spec constructor for KState (struct literals in assert positions are a
/// known parser hazard — CLAUDE.md pitfall).
pub open spec fn mk_state(locals: Seq<u32>, out: Seq<u32>) -> KState {
    KState { locals, out }
}

// ══════════════════════════════════════════════════════════════
// One-step unfold lemmas (the u_* pattern): empty-body proof fns
// giving the recursive interpreters' equations as callable facts.
// The --emit-gpu pass would emit these mechanically alongside the
// literal; they are kernel-independent schema.
// ══════════════════════════════════════════════════════════════

pub proof fn u_kexec_seq2(s1: KStmt, s2: KStmt, st: KState, a: Seq<u32>, b: Seq<u32>)
    ensures
        kexec(KStmt::Seq2(Box::new(s1), Box::new(s2)), st, a, b)
            == kexec(s2, kexec(s1, st, a, b), a, b),
{
}

pub proof fn u_kexec_assign(loc: nat, rhs: KExpr, st: KState, a: Seq<u32>, b: Seq<u32>)
    requires
        (loc as int) < st.locals.len(),
    ensures
        kexec(KStmt::Assign { loc, rhs }, st, a, b)
            == mk_state(st.locals.update(loc as int, keval(rhs, st, a, b)), st.out),
{
}

pub proof fn u_kexec_writeout(idx: KExpr, val: KExpr, st: KState, a: Seq<u32>, b: Seq<u32>)
    requires
        (keval(idx, st, a, b) as int) < st.out.len(),
    ensures
        kexec(KStmt::WriteOut { idx, val }, st, a, b)
            == mk_state(st.locals, st.out.update(
                keval(idx, st, a, b) as int, keval(val, st, a, b))),
{
}

pub proof fn u_keval_loc(i: nat, st: KState, a: Seq<u32>, b: Seq<u32>)
    requires
        (i as int) < st.locals.len(),
    ensures
        keval(KExpr::Loc(i), st, a, b) == st.locals[i as int],
{
}

pub proof fn u_keval_reada(ie: KExpr, st: KState, a: Seq<u32>, b: Seq<u32>)
    requires
        (keval(ie, st, a, b) as int) < a.len(),
    ensures
        keval(KExpr::ReadA(Box::new(ie)), st, a, b) == a[keval(ie, st, a, b) as int],
{
}

pub proof fn u_keval_readb(ie: KExpr, st: KState, a: Seq<u32>, b: Seq<u32>)
    requires
        (keval(ie, st, a, b) as int) < b.len(),
    ensures
        keval(KExpr::ReadB(Box::new(ie)), st, a, b) == b[keval(ie, st, a, b) as int],
{
}

pub proof fn u_keval_addw(x: KExpr, y: KExpr, st: KState, a: Seq<u32>, b: Seq<u32>)
    ensures
        keval(KExpr::AddW(Box::new(x), Box::new(y)), st, a, b)
            == wadd(keval(x, st, a, b), keval(y, st, a, b)),
{
}

pub proof fn u_keval_ltu(x: KExpr, y: KExpr, st: KState, a: Seq<u32>, b: Seq<u32>)
    ensures
        keval(KExpr::LtU(Box::new(x), Box::new(y)), st, a, b)
            == ltu(keval(x, st, a, b), keval(y, st, a, b)),
{
}

/// The compare result is a bit.
pub proof fn u_ltu_bit(x: u32, y: u32)
    ensures
        ltu(x, y) == 0 || ltu(x, y) == 1,
{
}

/// kloop unfold, one iteration (ivar in range).
pub proof fn u_kloop_step(
    body: KStmt, ivar: nat, lo: nat, n: nat, st: KState, a: Seq<u32>, b: Seq<u32>,
)
    requires
        lo < n,
        (ivar as int) < st.locals.len(),
    ensures
        kloop(body, ivar, lo, n, st, a, b)
            == kloop(body, ivar, lo + 1, n,
                kexec(body, mk_state(st.locals.update(ivar as int, lo as u32), st.out), a, b),
                a, b),
{
}

/// kloop unfold, empty range.
pub proof fn u_kloop_done(
    body: KStmt, ivar: nat, lo: nat, n: nat, st: KState, a: Seq<u32>, b: Seq<u32>,
)
    requires
        lo >= n,
    ensures
        kloop(body, ivar, lo, n, st, a, b) == st,
{
}

/// model_out_loop unfold, one iteration.
pub proof fn u_model_out_step(
    a: Seq<u32>, b: Seq<u32>, out: Seq<u32>, carry: u32, lo: nat, n: nat,
)
    requires
        lo < n,
    ensures
        model_out_loop(a, b, out, carry, lo, n)
            == model_out_loop(
                a, b,
                out.update(lo as int, model_abc(a, b, carry, lo)),
                model_carry_next(a, b, carry, lo),
                lo + 1, n),
{
}

/// model_carry_loop unfold, one iteration.
pub proof fn u_model_carry_step(
    a: Seq<u32>, b: Seq<u32>, carry: u32, lo: nat, n: nat,
)
    requires
        lo < n,
    ensures
        model_carry_loop(a, b, carry, lo, n)
            == model_carry_loop(a, b, model_carry_next(a, b, carry, lo), lo + 1, n),
{
}

/// model_out_loop unfold, empty range.
pub proof fn u_model_out_done(
    a: Seq<u32>, b: Seq<u32>, out: Seq<u32>, carry: u32, lo: nat, n: nat,
)
    requires
        lo >= n,
    ensures
        model_out_loop(a, b, out, carry, lo, n) == out,
{
}

/// model_carry_loop unfold, empty range.
pub proof fn u_model_carry_done(
    a: Seq<u32>, b: Seq<u32>, carry: u32, lo: nat, n: nat,
)
    requires
        lo >= n,
    ensures
        model_carry_loop(a, b, carry, lo, n) == carry,
{
}

} // verus!
