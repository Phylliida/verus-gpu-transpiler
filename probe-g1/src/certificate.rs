//! The certificate obligations (the G1 measurement).
//!
//! `u_seq_*`: per-level sequencing unfolds of the body chain (schema).
//! `step_after_*`: symbolic execution of the body, one statement at a time,
//! each giving the exact intermediate state (generator-emitted shape).
//! `lemma_step_bisim` / `lemma_loop_bisim`: the certificate proper.

use vstd::prelude::*;
use crate::kir::*;
use crate::addloop::*;

verus! {

// ══════════════════════════════════════════════════════════════
// Sequencing unfolds: body → fold of the eight statements
// ══════════════════════════════════════════════════════════════

pub proof fn u_seq_body(st: KState, a: Seq<u32>, b: Seq<u32>)
    ensures
        kexec(body_kir(), st, a, b) == kexec(chain2(), kexec(stmt_av(), st, a, b), a, b),
{
}

pub proof fn u_seq_chain2(st: KState, a: Seq<u32>, b: Seq<u32>)
    ensures
        kexec(chain2(), st, a, b) == kexec(chain3(), kexec(stmt_bv(), st, a, b), a, b),
{
}

pub proof fn u_seq_chain3(st: KState, a: Seq<u32>, b: Seq<u32>)
    ensures
        kexec(chain3(), st, a, b) == kexec(chain4(), kexec(stmt_ab(), st, a, b), a, b),
{
}

pub proof fn u_seq_chain4(st: KState, a: Seq<u32>, b: Seq<u32>)
    ensures
        kexec(chain4(), st, a, b) == kexec(chain5(), kexec(stmt_c1(), st, a, b), a, b),
{
}

pub proof fn u_seq_chain5(st: KState, a: Seq<u32>, b: Seq<u32>)
    ensures
        kexec(chain5(), st, a, b) == kexec(chain6(), kexec(stmt_abc(), st, a, b), a, b),
{
}

pub proof fn u_seq_chain6(st: KState, a: Seq<u32>, b: Seq<u32>)
    ensures
        kexec(chain6(), st, a, b) == kexec(chain7(), kexec(stmt_c2(), st, a, b), a, b),
{
}

pub proof fn u_seq_chain7(st: KState, a: Seq<u32>, b: Seq<u32>)
    ensures
        kexec(chain7(), st, a, b) == kexec(stmt_carry(), kexec(stmt_write(), st, a, b), a, b),
{
}

// ══════════════════════════════════════════════════════════════
// Per-statement execution: exact successor states
// ══════════════════════════════════════════════════════════════

/// Statement 1: av := a[i]. Locals index 2 receives a[i].
#[verifier::tactus_auto]
pub proof fn step_av(st: KState, a: Seq<u32>, b: Seq<u32>, i: nat)
    requires
        st.locals.len() == 8,
        st.locals[1] == i as u32,
        i < 0x1_0000_0000,
        (i as int) < a.len(),
    ensures
        kexec(stmt_av(), st, a, b)
            == mk_state(st.locals.update(2, a[i as int]), st.out),
{
    u_keval_loc(1, st, a, b);
    assert((i as u32) as int == i as int) by {
        intros
        push_cast
        omega
    };
    u_keval_reada(KExpr::Loc(1), st, a, b);
    u_kexec_assign(2, KExpr::ReadA(Box::new(KExpr::Loc(1))), st, a, b);
    assert(kexec(stmt_av(), st, a, b)
        == mk_state(st.locals.update(2, a[i as int]), st.out)) by {
        intros
        simp_all (config := { zetaDelta := true }) [addloop.stmt_av]
    };
}

/// Statement 2: bv := b[i].
#[verifier::tactus_auto]
pub proof fn step_bv(st: KState, a: Seq<u32>, b: Seq<u32>, i: nat)
    requires
        st.locals.len() == 8,
        st.locals[1] == i as u32,
        i < 0x1_0000_0000,
        (i as int) < b.len(),
    ensures
        kexec(stmt_bv(), st, a, b)
            == mk_state(st.locals.update(3, b[i as int]), st.out),
{
    u_keval_loc(1, st, a, b);
    assert((i as u32) as int == i as int) by {
        intros
        push_cast
        omega
    };
    u_keval_readb(KExpr::Loc(1), st, a, b);
    u_kexec_assign(3, KExpr::ReadB(Box::new(KExpr::Loc(1))), st, a, b);
    assert(kexec(stmt_bv(), st, a, b)
        == mk_state(st.locals.update(3, b[i as int]), st.out)) by {
        intros
        simp_all (config := { zetaDelta := true }) [addloop.stmt_bv]
    };
}

/// Statement 3: ab := av + bv (wrapping).
#[verifier::tactus_auto]
pub proof fn step_ab(st: KState, a: Seq<u32>, b: Seq<u32>)
    requires
        st.locals.len() == 8,
    ensures
        kexec(stmt_ab(), st, a, b)
            == mk_state(st.locals.update(4, wadd(st.locals[2], st.locals[3])), st.out),
{
    u_keval_loc(2, st, a, b);
    u_keval_loc(3, st, a, b);
    u_keval_addw(KExpr::Loc(2), KExpr::Loc(3), st, a, b);
    u_kexec_assign(4, KExpr::AddW(Box::new(KExpr::Loc(2)), Box::new(KExpr::Loc(3))), st, a, b);
    assert(kexec(stmt_ab(), st, a, b)
        == mk_state(st.locals.update(4, wadd(st.locals[2], st.locals[3])), st.out)) by {
        intros
        simp_all (config := { zetaDelta := true }) [addloop.stmt_ab]
    };
}

/// Statement 4: c1 := (ab < av) as bit.
#[verifier::tactus_auto]
pub proof fn step_c1(st: KState, a: Seq<u32>, b: Seq<u32>)
    requires
        st.locals.len() == 8,
    ensures
        kexec(stmt_c1(), st, a, b)
            == mk_state(st.locals.update(5, ltu(st.locals[4], st.locals[2])), st.out),
{
    u_keval_loc(4, st, a, b);
    u_keval_loc(2, st, a, b);
    u_keval_ltu(KExpr::Loc(4), KExpr::Loc(2), st, a, b);
    u_kexec_assign(5, KExpr::LtU(Box::new(KExpr::Loc(4)), Box::new(KExpr::Loc(2))), st, a, b);
    assert(kexec(stmt_c1(), st, a, b)
        == mk_state(st.locals.update(5, ltu(st.locals[4], st.locals[2])), st.out)) by {
        intros
        simp_all (config := { zetaDelta := true }) [addloop.stmt_c1]
    };
}

/// Statement 5: abc := ab + carry (wrapping).
#[verifier::tactus_auto]
pub proof fn step_abc(st: KState, a: Seq<u32>, b: Seq<u32>)
    requires
        st.locals.len() == 8,
    ensures
        kexec(stmt_abc(), st, a, b)
            == mk_state(st.locals.update(6, wadd(st.locals[4], st.locals[0])), st.out),
{
    u_keval_loc(4, st, a, b);
    u_keval_loc(0, st, a, b);
    u_keval_addw(KExpr::Loc(4), KExpr::Loc(0), st, a, b);
    u_kexec_assign(6, KExpr::AddW(Box::new(KExpr::Loc(4)), Box::new(KExpr::Loc(0))), st, a, b);
    assert(kexec(stmt_abc(), st, a, b)
        == mk_state(st.locals.update(6, wadd(st.locals[4], st.locals[0])), st.out)) by {
        intros
        simp_all (config := { zetaDelta := true }) [addloop.stmt_abc]
    };
}

/// Statement 6: c2 := (abc < ab) as bit.
#[verifier::tactus_auto]
pub proof fn step_c2(st: KState, a: Seq<u32>, b: Seq<u32>)
    requires
        st.locals.len() == 8,
    ensures
        kexec(stmt_c2(), st, a, b)
            == mk_state(st.locals.update(7, ltu(st.locals[6], st.locals[4])), st.out),
{
    u_keval_loc(6, st, a, b);
    u_keval_loc(4, st, a, b);
    u_keval_ltu(KExpr::Loc(6), KExpr::Loc(4), st, a, b);
    u_kexec_assign(7, KExpr::LtU(Box::new(KExpr::Loc(6)), Box::new(KExpr::Loc(4))), st, a, b);
    assert(kexec(stmt_c2(), st, a, b)
        == mk_state(st.locals.update(7, ltu(st.locals[6], st.locals[4])), st.out)) by {
        intros
        simp_all (config := { zetaDelta := true }) [addloop.stmt_c2]
    };
}

/// Statement 7: out[i] := abc.
#[verifier::tactus_auto]
pub proof fn step_write(st: KState, a: Seq<u32>, b: Seq<u32>, i: nat)
    requires
        st.locals.len() == 8,
        st.locals[1] == i as u32,
        i < 0x1_0000_0000,
        (i as int) < st.out.len(),
    ensures
        kexec(stmt_write(), st, a, b)
            == mk_state(st.locals, st.out.update(i as int, st.locals[6])),
{
    u_keval_loc(1, st, a, b);
    u_keval_loc(6, st, a, b);
    assert((i as u32) as int == i as int) by {
        intros
        push_cast
        omega
    };
    u_kexec_writeout(KExpr::Loc(1), KExpr::Loc(6), st, a, b);
    assert(kexec(stmt_write(), st, a, b)
        == mk_state(st.locals, st.out.update(i as int, st.locals[6]))) by {
        intros
        simp_all (config := { zetaDelta := true }) [addloop.stmt_write]
    };
}

/// Statement 8: carry := c1 + c2 (wrapping).
#[verifier::tactus_auto]
pub proof fn step_carry(st: KState, a: Seq<u32>, b: Seq<u32>)
    requires
        st.locals.len() == 8,
    ensures
        kexec(stmt_carry(), st, a, b)
            == mk_state(st.locals.update(0, wadd(st.locals[5], st.locals[7])), st.out),
{
    u_keval_loc(5, st, a, b);
    u_keval_loc(7, st, a, b);
    u_keval_addw(KExpr::Loc(5), KExpr::Loc(7), st, a, b);
    u_kexec_assign(0, KExpr::AddW(Box::new(KExpr::Loc(5)), Box::new(KExpr::Loc(7))), st, a, b);
    assert(kexec(stmt_carry(), st, a, b)
        == mk_state(st.locals.update(0, wadd(st.locals[5], st.locals[7])), st.out)) by {
        intros
        simp_all (config := { zetaDelta := true }) [addloop.stmt_carry]
    };
}

// ══════════════════════════════════════════════════════════════
// The step certificate: one body execution == one model step
// ══════════════════════════════════════════════════════════════

/// Full-body bisimulation: executing the eight-statement body from any
/// well-formed state performs exactly one model step — out gets the model
/// limb at i, local 0 gets the model carry, local 1 (the loop counter) is
/// untouched, locals stay 8 wide.
#[verifier::tactus_auto]
pub proof fn lemma_step_bisim(st: KState, a: Seq<u32>, b: Seq<u32>, i: nat)
    requires
        st.locals.len() == 8,
        st.locals[1] == i as u32,
        i < 0x1_0000_0000,
        (i as int) < a.len(),
        (i as int) < b.len(),
        (i as int) < st.out.len(),
    ensures
        kexec(body_kir(), st, a, b).locals.len() == 8,
        kexec(body_kir(), st, a, b).locals[0]
            == model_carry_next(a, b, st.locals[0], i),
        kexec(body_kir(), st, a, b).locals[1] == st.locals[1],
        kexec(body_kir(), st, a, b).out
            == st.out.update(i as int, model_abc(a, b, st.locals[0], i)),
{
    let l = st.locals;
    let av = a[i as int];
    let bv = b[i as int];
    let ab = wadd(av, bv);
    let c1v = ltu(ab, av);
    let abc = wadd(ab, l[0]);
    let c2v = ltu(abc, ab);
    let cnew = wadd(c1v, c2v);

    let ll1 = l.update(2, av);
    let ll2 = ll1.update(3, bv);
    let ll3 = ll2.update(4, ab);
    let ll4 = ll3.update(5, c1v);
    let ll5 = ll4.update(6, abc);
    let ll6 = ll5.update(7, c2v);
    let ll7 = ll6.update(0, cnew);

    // Seq bookkeeping: lengths and index-through-update (schema block).
    vstd::seq::axiom_seq_update_len::<u32>(l, 2, av);
    vstd::seq::axiom_seq_update_len::<u32>(ll1, 3, bv);
    vstd::seq::axiom_seq_update_len::<u32>(ll2, 4, ab);
    vstd::seq::axiom_seq_update_len::<u32>(ll3, 5, c1v);
    vstd::seq::axiom_seq_update_len::<u32>(ll4, 6, abc);
    vstd::seq::axiom_seq_update_len::<u32>(ll5, 7, c2v);
    vstd::seq::axiom_seq_update_len::<u32>(ll6, 0, cnew);
    assert(ll1.len() == 8 && ll2.len() == 8 && ll3.len() == 8 && ll4.len() == 8
        && ll5.len() == 8 && ll6.len() == 8 && ll7.len() == 8) by {
        intros
        simp_all (config := { zetaDelta := true }) []
    };

    // Loop counter survives every update (index 1 never written).
    vstd::seq::axiom_seq_update_different::<u32>(l, 1, 2, av);
    vstd::seq::axiom_seq_update_different::<u32>(ll1, 1, 3, bv);
    vstd::seq::axiom_seq_update_different::<u32>(ll2, 1, 4, ab);
    vstd::seq::axiom_seq_update_different::<u32>(ll3, 1, 5, c1v);
    vstd::seq::axiom_seq_update_different::<u32>(ll4, 1, 6, abc);
    vstd::seq::axiom_seq_update_different::<u32>(ll5, 1, 7, c2v);
    vstd::seq::axiom_seq_update_different::<u32>(ll6, 1, 0, cnew);
    assert(ll1[1] == i as u32 && ll2[1] == i as u32 && ll3[1] == i as u32
        && ll4[1] == i as u32 && ll5[1] == i as u32 && ll6[1] == i as u32
        && ll7[1] == i as u32) by {
        intros
        simp_all (config := { zetaDelta := true }) []
    };

    // Value tracking: each written slot read back, later writes elsewhere.
    vstd::seq::axiom_seq_update_same::<u32>(l, 2, av);
    vstd::seq::axiom_seq_update_different::<u32>(ll1, 2, 3, bv);
    vstd::seq::axiom_seq_update_same::<u32>(ll1, 3, bv);
    vstd::seq::axiom_seq_update_different::<u32>(l, 0, 2, av);
    vstd::seq::axiom_seq_update_different::<u32>(ll1, 0, 3, bv);
    vstd::seq::axiom_seq_update_different::<u32>(ll2, 0, 4, ab);
    vstd::seq::axiom_seq_update_same::<u32>(ll2, 4, ab);
    vstd::seq::axiom_seq_update_different::<u32>(ll2, 2, 4, ab);
    vstd::seq::axiom_seq_update_same::<u32>(ll3, 5, c1v);
    vstd::seq::axiom_seq_update_different::<u32>(ll3, 4, 5, c1v);
    vstd::seq::axiom_seq_update_different::<u32>(ll3, 0, 5, c1v);
    vstd::seq::axiom_seq_update_same::<u32>(ll4, 6, abc);
    vstd::seq::axiom_seq_update_different::<u32>(ll4, 4, 6, abc);
    vstd::seq::axiom_seq_update_different::<u32>(ll4, 5, 6, abc);
    vstd::seq::axiom_seq_update_same::<u32>(ll5, 7, c2v);
    vstd::seq::axiom_seq_update_different::<u32>(ll5, 6, 7, c2v);
    vstd::seq::axiom_seq_update_different::<u32>(ll5, 5, 7, c2v);
    vstd::seq::axiom_seq_update_same::<u32>(ll6, 0, cnew);
    assert(ll2[2] == av && ll2[3] == bv && ll2[0] == l[0]
        && ll3[4] == ab && ll3[2] == av && ll3[0] == l[0]
        && ll4[4] == ab && ll4[5] == c1v && ll4[0] == l[0]
        && ll5[6] == abc && ll5[4] == ab && ll5[5] == c1v
        && ll6[6] == abc && ll6[5] == c1v && ll6[7] == c2v
        && ll7[0] == cnew) by {
        intros
        simp_all (config := { zetaDelta := true }) []
    };

    // Execution chain: apply each step lemma at the concrete successor state.
    u_seq_body(st, a, b);
    step_av(st, a, b, i);
    let st1 = mk_state(ll1, st.out);
    assert(kexec(stmt_av(), st, a, b) == st1 && st1.locals == ll1 && st1.out == st.out) by {
        intros
        simp_all (config := { zetaDelta := true }) [addloop.mk_state]
    };

    u_seq_chain2(st1, a, b);
    step_bv(st1, a, b, i);
    let st2 = mk_state(ll2, st.out);
    assert(kexec(stmt_bv(), st1, a, b) == st2 && st2.locals == ll2 && st2.out == st.out) by {
        intros
        simp_all (config := { zetaDelta := true }) [addloop.mk_state]
    };

    u_seq_chain3(st2, a, b);
    step_ab(st2, a, b);
    let st3 = mk_state(ll3, st.out);
    assert(kexec(stmt_ab(), st2, a, b) == st3 && st3.locals == ll3 && st3.out == st.out) by {
        intros
        simp_all (config := { zetaDelta := true }) [addloop.mk_state]
    };

    u_seq_chain4(st3, a, b);
    step_c1(st3, a, b);
    let st4 = mk_state(ll4, st.out);
    assert(kexec(stmt_c1(), st3, a, b) == st4 && st4.locals == ll4 && st4.out == st.out) by {
        intros
        simp_all (config := { zetaDelta := true }) [addloop.mk_state]
    };

    u_seq_chain5(st4, a, b);
    step_abc(st4, a, b);
    let st5 = mk_state(ll5, st.out);
    assert(kexec(stmt_abc(), st4, a, b) == st5 && st5.locals == ll5 && st5.out == st.out) by {
        intros
        simp_all (config := { zetaDelta := true }) [addloop.mk_state]
    };

    u_seq_chain6(st5, a, b);
    step_c2(st5, a, b);
    let st6 = mk_state(ll6, st.out);
    assert(kexec(stmt_c2(), st5, a, b) == st6 && st6.locals == ll6 && st6.out == st.out) by {
        intros
        simp_all (config := { zetaDelta := true }) [addloop.mk_state]
    };

    u_seq_chain7(st6, a, b);
    step_write(st6, a, b, i);
    let outw = st.out.update(i as int, abc);
    let st7 = mk_state(ll6, outw);
    assert(kexec(stmt_write(), st6, a, b) == st7 && st7.locals == ll6 && st7.out == outw) by {
        intros
        simp_all (config := { zetaDelta := true }) [addloop.mk_state]
    };

    step_carry(st7, a, b);
    let st8 = mk_state(ll7, outw);
    assert(kexec(stmt_carry(), st7, a, b) == st8 && st8.locals == ll7 && st8.out == outw) by {
        intros
        simp_all (config := { zetaDelta := true }) [addloop.mk_state]
    };

    // The whole body lands on st8.
    assert(kexec(body_kir(), st, a, b) == st8) by {
        intros
        simp_all (config := { zetaDelta := true }) []
    };

    // Model connection: the tracked values are exactly one model step.
    u_ltu_bit(ab, av);
    u_ltu_bit(abc, ab);
    assert(cnew == model_carry_next(a, b, l[0], i)) by {
        intros
        simp_all (config := { zetaDelta := true })
            [addloop.model_carry_next, kir.wadd, kir.ltu]
        <;> split_ifs <;> (try push_cast at *) <;> omega
    };
    assert(abc == model_abc(a, b, l[0], i)) by {
        intros
        simp_all (config := { zetaDelta := true }) [addloop.model_abc]
    };
}

// ══════════════════════════════════════════════════════════════
// The loop certificate: kloop over the body == the model loop
// ══════════════════════════════════════════════════════════════

/// Loop bisimulation by induction on the trip count: running the KIR loop
/// from lo..n matches the model loop — out buffers equal, final carry in
/// local 0 equals the model carry, buffer length and locals width preserved.
#[verifier::tactus_auto]
pub proof fn lemma_loop_bisim(st: KState, a: Seq<u32>, b: Seq<u32>, lo: nat, n: nat)
    requires
        st.locals.len() == 8,
        lo <= n,
        n <= a.len(),
        n <= b.len(),
        (n as int) <= st.out.len(),
        n < 0x1_0000_0000,
    ensures
        kloop(body_kir(), 1, lo, n, st, a, b).locals.len() == 8,
        kloop(body_kir(), 1, lo, n, st, a, b).locals[0]
            == model_carry_loop(a, b, st.locals[0], lo, n),
        kloop(body_kir(), 1, lo, n, st, a, b).out
            == model_out_loop(a, b, st.out, st.locals[0], lo, n),
        kloop(body_kir(), 1, lo, n, st, a, b).out.len() == st.out.len(),
    decreases n - lo,
{
    if lo >= n {
        u_kloop_done(body_kir(), 1, lo, n, st, a, b);
        u_model_out_done(a, b, st.out, st.locals[0], lo, n);
        u_model_carry_done(a, b, st.locals[0], lo, n);
    } else {
        // The runner writes the counter, then the body runs.
        let stl = mk_state(st.locals.update(1, lo as u32), st.out);
        u_kloop_step(body_kir(), 1, lo, n, st, a, b);
        vstd::seq::axiom_seq_update_len::<u32>(st.locals, 1, lo as u32);
        vstd::seq::axiom_seq_update_same::<u32>(st.locals, 1, lo as u32);
        vstd::seq::axiom_seq_update_different::<u32>(st.locals, 0, 1, lo as u32);
        assert(stl.locals.len() == 8 && stl.locals[1] == lo as u32
            && stl.locals[0] == st.locals[0] && stl.out == st.out) by {
            intros
            simp_all (config := { zetaDelta := true }) [addloop.mk_state]
        };
        assert(lo < 0x1_0000_0000 && (lo as int) < a.len() && (lo as int) < b.len()
            && (lo as int) < stl.out.len()) by {
            intros
            omega
        };

        // One body execution is one model step.
        lemma_step_bisim(stl, a, b, lo);
        let st2 = kexec(body_kir(), stl, a, b);
        vstd::seq::axiom_seq_update_len::<u32>(
            st.out, lo as int, model_abc(a, b, st.locals[0], lo));
        assert(st2.locals.len() == 8 && (n as int) <= st2.out.len()
            && st2.out.len() == st.out.len()
            && st2.locals[0] == model_carry_next(a, b, st.locals[0], lo)
            && st2.out == st.out.update(lo as int, model_abc(a, b, st.locals[0], lo))) by {
            intros
            simp_all (config := { zetaDelta := true }) []
        };

        // Induction hypothesis on the remaining iterations.
        lemma_loop_bisim(st2, a, b, lo + 1, n);

        // The model loop takes the same step.
        u_model_out_step(a, b, st.out, st.locals[0], lo, n);
        u_model_carry_step(a, b, st.locals[0], lo, n);
        assert(kloop(body_kir(), 1, lo, n, st, a, b)
            == kloop(body_kir(), 1, lo + 1, n, st2, a, b)) by {
            intros
            simp_all (config := { zetaDelta := true }) [addloop.mk_state]
        };
    }
}

} // verus!
