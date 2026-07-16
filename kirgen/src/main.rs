//! kirgen — G1a certificate generator.
//!
//! Input: typed kernel definitions (in-file for now; the fork's SST adapter
//! replaces this seam later — DESIGN.md §5). Output: probe-g1/src/gen_certs.rs,
//! a self-contained Verus module with KIR literals and certificate proof fns
//! following the probe-g1 schema exactly. Generated output is never
//! hand-edited: if a certificate fails to verify, fix the generator.
//!
//! Trust note: hand-transcription errors in the kernel definitions are caught
//! by verification, because every certificate's ensures references the real
//! spec fn (spec_kernels::*) by name.

use std::fmt::Write as _;

// ── Typed kernel AST ────────────────────────────────────────────

#[derive(Clone)]
enum E {
    Param(usize),
    Temp(usize),
    Add(Box<E>, Box<E>),
    Sub(Box<E>, Box<E>),
    Mul(Box<E>, Box<E>),
    Ge(Box<E>, Box<E>),
    Le(Box<E>, Box<E>),
    Ite(Box<E>, Box<E>, Box<E>),
}

struct Kernel {
    name: &'static str,
    params: Vec<&'static str>,
    /// verbatim Verus requires clauses over the params
    requires: Vec<&'static str>,
    /// verbatim spec-fn application the result must equal (as int)
    spec_call: &'static str,
    /// statement k assigns temps[k] (local params.len()+k) := expr
    stmts: Vec<E>,
}

// ── Lowering: AST -> KIR literal text ───────────────────────────

fn loc(k: &Kernel, e: &E) -> usize {
    match e {
        E::Param(i) => *i,
        E::Temp(t) => k.params.len() + t,
        _ => unreachable!("loc of non-leaf"),
    }
}

fn kexpr(k: &Kernel, e: &E) -> String {
    match e {
        E::Param(_) | E::Temp(_) => format!("KExpr::Loc({})", loc(k, e)),
        E::Add(x, y) => format!("KExpr::AddW(Box::new({}), Box::new({}))", kexpr(k, x), kexpr(k, y)),
        E::Sub(x, y) => format!("KExpr::SubW(Box::new({}), Box::new({}))", kexpr(k, x), kexpr(k, y)),
        E::Mul(x, y) => format!("KExpr::MulW(Box::new({}), Box::new({}))", kexpr(k, x), kexpr(k, y)),
        E::Ge(x, y) => format!("KExpr::GeU(Box::new({}), Box::new({}))", kexpr(k, x), kexpr(k, y)),
        E::Le(x, y) => format!("KExpr::LeU(Box::new({}), Box::new({}))", kexpr(k, x), kexpr(k, y)),
        E::Ite(c, t, e2) => format!(
            "KExpr::Select(Box::new({}), Box::new({}), Box::new({}))",
            kexpr(k, c), kexpr(k, t), kexpr(k, e2)),
    }
}

/// Semantic value of e with locals resolved through `env` (index -> text).
fn sem(k: &Kernel, e: &E, env: &dyn Fn(usize) -> String) -> String {
    match e {
        E::Param(_) | E::Temp(_) => env(loc(k, e)),
        E::Add(x, y) => format!("wadd({}, {})", sem(k, x, env), sem(k, y, env)),
        E::Sub(x, y) => format!("wsub({}, {})", sem(k, x, env), sem(k, y, env)),
        E::Mul(x, y) => format!("wmul({}, {})", sem(k, x, env), sem(k, y, env)),
        E::Ge(x, y) => format!("geu({}, {})", sem(k, x, env), sem(k, y, env)),
        E::Le(x, y) => format!("leu({}, {})", sem(k, x, env), sem(k, y, env)),
        E::Ite(c, t, e2) => format!(
            "(if {} != 0 {{ {} }} else {{ {} }})",
            sem(k, c, env), sem(k, t, env), sem(k, e2, env)),
    }
}

/// Emit bottom-up u_keval_* calls for every non-leaf node of e at state `st`.
fn keval_calls(k: &Kernel, e: &E, st: &str, out: &mut String) {
    let bin = |out: &mut String, name: &str, x: &E, y: &E| {
        writeln!(out, "    {}({}, {}, {}, bufa, bufb);", name, kexpr(k, x), kexpr(k, y), st).unwrap();
    };
    match e {
        E::Param(_) | E::Temp(_) => {
            writeln!(out, "    u_keval_loc({}, {}, bufa, bufb);", loc(k, e), st).unwrap();
        }
        E::Add(x, y) => { keval_calls(k, x, st, out); keval_calls(k, y, st, out); bin(out, "u_keval_addw", x, y); }
        E::Sub(x, y) => { keval_calls(k, x, st, out); keval_calls(k, y, st, out); bin(out, "u_keval_subw", x, y); }
        E::Mul(x, y) => { keval_calls(k, x, st, out); keval_calls(k, y, st, out); bin(out, "u_keval_mulw", x, y); }
        E::Ge(x, y) => { keval_calls(k, x, st, out); keval_calls(k, y, st, out); bin(out, "u_keval_geu", x, y); }
        E::Le(x, y) => { keval_calls(k, x, st, out); keval_calls(k, y, st, out); bin(out, "u_keval_leu", x, y); }
        E::Ite(c, t, e2) => {
            keval_calls(k, c, st, out); keval_calls(k, t, st, out); keval_calls(k, e2, st, out);
            writeln!(out, "    u_keval_select({}, {}, {}, {}, bufa, bufb);",
                kexpr(k, c), kexpr(k, t), kexpr(k, e2), st).unwrap();
        }
    }
}

/// Collect every Mul node's operand pair (for range bound asserts).
fn mul_nodes<'a>(e: &'a E, acc: &mut Vec<(&'a E, &'a E)>) {
    match e {
        E::Param(_) | E::Temp(_) => {}
        E::Add(x, y) | E::Sub(x, y) | E::Mul(x, y) | E::Ge(x, y) | E::Le(x, y) => {
            mul_nodes(x, acc); mul_nodes(y, acc);
            if let E::Mul(a, b) = e { let _ = (a, b); }
            if matches!(e, E::Mul(..)) {
                if let E::Mul(a, b) = e { acc.push((a, b)); }
            }
        }
        E::Ite(c, t, e2) => { mul_nodes(c, acc); mul_nodes(t, acc); mul_nodes(e2, acc); }
    }
}

// ── Certificate emission ────────────────────────────────────────

fn emit_kernel(k: &Kernel, out: &mut String) {
    let p = k.params.len();
    let ns = k.stmts.len();
    let width = p + ns;
    let result_loc = width - 1;
    let n = k.name;

    // Symbolic values per temp (in terms of param names).
    let mut vals: Vec<String> = Vec::new();
    for e in &k.stmts {
        let env = |i: usize| -> String {
            if i < p { k.params[i].to_string() } else { format!("v{}_{}", n, i - p) }
        };
        vals.push(sem(k, e, &env));
    }

    writeln!(out, "// ══════════ kernel: {} ══════════\n", n).unwrap();

    // Statement literals and chain fns.
    for (i, e) in k.stmts.iter().enumerate() {
        writeln!(out, "pub open spec fn stmt_{}_{}() -> KStmt {{", n, i).unwrap();
        writeln!(out, "    KStmt::Assign {{ loc: {}, rhs: {} }}", p + i, kexpr(k, e)).unwrap();
        writeln!(out, "}}\n").unwrap();
    }
    // chain_i = Seq2(stmt_i, chain_{i+1}); last chain is the last stmt.
    for i in (0..ns.saturating_sub(1)).rev() {
        let tail = if i + 1 == ns - 1 {
            format!("stmt_{}_{}()", n, ns - 1)
        } else {
            format!("chain_{}_{}()", n, i + 1)
        };
        writeln!(out, "pub open spec fn chain_{}_{}() -> KStmt {{", n, i).unwrap();
        writeln!(out, "    KStmt::Seq2(Box::new(stmt_{}_{}()), Box::new({}))", n, i, tail).unwrap();
        writeln!(out, "}}\n").unwrap();
    }
    let body = if ns == 1 { format!("stmt_{}_0()", n) } else { format!("chain_{}_0()", n) };
    writeln!(out, "pub open spec fn body_{}() -> KStmt {{ {} }}\n", n, body).unwrap();

    // Sequencing unfolds (empty bodies).
    for i in 0..ns.saturating_sub(1) {
        let this = if i == 0 { format!("body_{}()", n) } else { format!("chain_{}_{}()", n, i) };
        let rest = if i + 1 == ns - 1 {
            format!("stmt_{}_{}()", n, ns - 1)
        } else {
            format!("chain_{}_{}()", n, i + 1)
        };
        writeln!(out, "pub proof fn u_seq_{}_{}(st: KState, bufa: Seq<u32>, bufb: Seq<u32>)", n, i).unwrap();
        writeln!(out, "    ensures").unwrap();
        writeln!(out, "        kexec({}, st, bufa, bufb)", this).unwrap();
        writeln!(out, "            == kexec({}, kexec(stmt_{}_{}(), st, bufa, bufb), bufa, bufb),", rest, n, i).unwrap();
        writeln!(out, "{{\n}}\n").unwrap();
    }
    // Trivial body unfold when the body is a single statement.
    if ns == 1 {
        writeln!(out, "pub proof fn u_seq_{}_0(st: KState, bufa: Seq<u32>, bufb: Seq<u32>)", n).unwrap();
        writeln!(out, "    ensures kexec(body_{}(), st, bufa, bufb) == kexec(stmt_{}_0(), st, bufa, bufb),", n, n).unwrap();
        writeln!(out, "{{\n}}\n").unwrap();
    }

    // The certificate.
    writeln!(out, "#[verifier::tactus_auto]").unwrap();
    write!(out, "pub proof fn cert_{}(st: KState, bufa: Seq<u32>, bufb: Seq<u32>", n).unwrap();
    for pn in &k.params { write!(out, ", {}: u32", pn).unwrap(); }
    writeln!(out, ")").unwrap();
    writeln!(out, "    requires").unwrap();
    writeln!(out, "        st.locals.len() == {},", width).unwrap();
    for (i, pn) in k.params.iter().enumerate() {
        writeln!(out, "        st.locals[{}] == {},", i, pn).unwrap();
    }
    for r in &k.requires { writeln!(out, "        {},", r).unwrap(); }
    writeln!(out, "    ensures").unwrap();
    writeln!(out, "        kexec(body_{}(), st, bufa, bufb).locals.len() == {},", n, width).unwrap();
    writeln!(out, "        kexec(body_{}(), st, bufa, bufb).locals[{}] as int == {},", n, result_loc, k.spec_call).unwrap();
    writeln!(out, "{{").unwrap();

    // Value lets.
    for (t, v) in vals.iter().enumerate() {
        writeln!(out, "    let v{}_{} = {};", n, t, v).unwrap();
    }
    // Locals chains.
    for t in 0..ns {
        let prev = if t == 0 { "st.locals".to_string() } else { format!("ll{}_{}", n, t - 1) };
        writeln!(out, "    let ll{}_{} = {}.update({}, v{}_{});", n, t, prev, p + t, n, t).unwrap();
    }
    writeln!(out).unwrap();

    // Seq bookkeeping: lengths, then live-index tracking per level.
    for t in 0..ns {
        let prev = if t == 0 { "st.locals".to_string() } else { format!("ll{}_{}", n, t - 1) };
        writeln!(out, "    vstd::seq::axiom_seq_update_len::<u32>({}, {}, v{}_{});", prev, p + t, n, t).unwrap();
    }
    {
        let mut conds: Vec<String> = Vec::new();
        for t in 0..ns { conds.push(format!("ll{}_{}.len() == {}", n, t, width)); }
        writeln!(out, "    assert({}) by {{", conds.join(" && ")).unwrap();
        writeln!(out, "        intros").unwrap();
        writeln!(out, "        simp_all (config := {{ zetaDelta := true }}) []").unwrap();
        writeln!(out, "    }};\n").unwrap();
    }
    // Index tracking: at each level t, every live index j (params and temps
    // written so far) keeps/gets its symbolic value.
    for t in 0..ns {
        let prev = if t == 0 { "st.locals".to_string() } else { format!("ll{}_{}", n, t - 1) };
        for j in 0..(p + t + 1) {
            if j == p + t {
                writeln!(out, "    vstd::seq::axiom_seq_update_same::<u32>({}, {}, v{}_{});", prev, p + t, n, t).unwrap();
            } else {
                writeln!(out, "    vstd::seq::axiom_seq_update_different::<u32>({}, {}, {}, v{}_{});", prev, j, p + t, n, t).unwrap();
            }
        }
        let mut conds: Vec<String> = Vec::new();
        for j in 0..(p + t + 1) {
            let val = if j < p { k.params[j].to_string() } else { format!("v{}_{}", n, j - p) };
            conds.push(format!("ll{}_{}[{}] == {}", n, t, j, val));
        }
        writeln!(out, "    assert({}) by {{", conds.join(" && ")).unwrap();
        writeln!(out, "        intros").unwrap();
        writeln!(out, "        simp_all (config := {{ zetaDelta := true }}) []").unwrap();
        writeln!(out, "    }};\n").unwrap();
    }

    // Execution chain.
    for t in 0..ns {
        let stv = if t == 0 { "st".to_string() } else { format!("s{}_{}", n, t - 1) };
        if t < ns.saturating_sub(1) || ns == 1 {
            if t == 0 {
                writeln!(out, "    u_seq_{}_0(st, bufa, bufb);", n).unwrap();
            } else {
                writeln!(out, "    u_seq_{}_{}({}, bufa, bufb);", n, t, stv).unwrap();
            }
        }
        keval_calls(k, &k.stmts[t], &stv, out);
        writeln!(out, "    u_kexec_assign({}, {}, {}, bufa, bufb);", p + t, kexpr(k, &k.stmts[t]), stv).unwrap();
        writeln!(out, "    let s{}_{} = mk_state(ll{}_{}, st.out);", n, t, n, t).unwrap();
        writeln!(out, "    assert(kexec(stmt_{}_{}(), {}, bufa, bufb) == s{}_{} && s{}_{}.locals == ll{}_{} && s{}_{}.out == st.out) by {{", n, t, stv, n, t, n, t, n, t, n, t).unwrap();
        writeln!(out, "        intros").unwrap();
        writeln!(out, "        simp_all (config := {{ zetaDelta := true }}) [gen_certs.stmt_{}_{}, addloop.mk_state]", n, t).unwrap();
        writeln!(out, "    }};\n").unwrap();
    }
    let last = format!("s{}_{}", n, ns - 1);
    writeln!(out, "    assert(kexec(body_{}(), st, bufa, bufb) == {}) by {{", n, last).unwrap();
    writeln!(out, "        intros").unwrap();
    writeln!(out, "        simp_all (config := {{ zetaDelta := true }}) []").unwrap();
    writeln!(out, "    }};\n").unwrap();

    // Multiplication range bounds (nlinarith food for the seam).
    let mut muls = Vec::new();
    for e in &k.stmts { mul_nodes(e, &mut muls); }
    for (x, y) in &muls {
        let env = |i: usize| -> String {
            if i < p { k.params[i].to_string() } else { format!("v{}_{}", n, i - p) }
        };
        let (sx, sy) = (sem(k, x, &env), sem(k, y, &env));
        writeln!(out, "    assert(({} as int) * ({} as int) < 0x1_0000_0000) by {{", sx, sy).unwrap();
        writeln!(out, "        intros").unwrap();
        writeln!(out, "        simp_all (config := {{ zetaDelta := true }}) [kir.wadd, kir.wsub, kir.wmul, kir.geu, kir.leu, kir.ltu]").unwrap();
        writeln!(out, "        <;> (try split_ifs)").unwrap();
        writeln!(out, "        <;> (try push_cast at *)").unwrap();
        writeln!(out, "        <;> (first | omega | nlinarith)").unwrap();
        writeln!(out, "    }};").unwrap();
    }

    // The arithmetic seam: symbolic result equals the spec.
    writeln!(out, "    assert(v{}_{} as int == {}) by {{", n, ns - 1, k.spec_call).unwrap();
    writeln!(out, "        intros").unwrap();
    writeln!(out, "        simp_all (config := {{ zetaDelta := true }}) [spec_kernels.{}, kir.wadd, kir.wsub, kir.wmul, kir.geu, kir.leu, kir.ltu]",
        k.spec_call.split('(').next().unwrap()).unwrap();
    writeln!(out, "        <;> (try split_ifs)").unwrap();
    writeln!(out, "        <;> (try push_cast at *)").unwrap();
    writeln!(out, "        <;> (first | omega | nlinarith)").unwrap();
    writeln!(out, "    }};").unwrap();
    writeln!(out, "}}\n").unwrap();
}

fn prelude() -> &'static str {
    r#"//! GENERATED by kirgen — do not edit. Regenerate: cargo run --manifest-path ../kirgen/Cargo.toml
//!
//! Certificates for expression kernels: for each kernel, the KIR literal
//! provably computes the kernel's functional spec (spec_kernels::*), which
//! the verified source exec fn already provably returns. Chain complete:
//! source exec == spec == KIR.

use vstd::prelude::*;
use crate::kir::*;
use crate::addloop::mk_state;
use crate::addloop::{u_keval_loc, u_keval_addw, u_kexec_assign};
use crate::spec_kernels::*;

verus! {

// Generic one-step unfolds for the extended op set (schema; empty bodies).

pub proof fn u_keval_subw(x: KExpr, y: KExpr, st: KState, bufa: Seq<u32>, bufb: Seq<u32>)
    ensures
        keval(KExpr::SubW(Box::new(x), Box::new(y)), st, bufa, bufb)
            == wsub(keval(x, st, bufa, bufb), keval(y, st, bufa, bufb)),
{
}

pub proof fn u_keval_mulw(x: KExpr, y: KExpr, st: KState, bufa: Seq<u32>, bufb: Seq<u32>)
    ensures
        keval(KExpr::MulW(Box::new(x), Box::new(y)), st, bufa, bufb)
            == wmul(keval(x, st, bufa, bufb), keval(y, st, bufa, bufb)),
{
}

pub proof fn u_keval_geu(x: KExpr, y: KExpr, st: KState, bufa: Seq<u32>, bufb: Seq<u32>)
    ensures
        keval(KExpr::GeU(Box::new(x), Box::new(y)), st, bufa, bufb)
            == geu(keval(x, st, bufa, bufb), keval(y, st, bufa, bufb)),
{
}

pub proof fn u_keval_leu(x: KExpr, y: KExpr, st: KState, bufa: Seq<u32>, bufb: Seq<u32>)
    ensures
        keval(KExpr::LeU(Box::new(x), Box::new(y)), st, bufa, bufb)
            == leu(keval(x, st, bufa, bufb), keval(y, st, bufa, bufb)),
{
}

pub proof fn u_keval_select(c: KExpr, t: KExpr, e: KExpr, st: KState, bufa: Seq<u32>, bufb: Seq<u32>)
    ensures
        keval(KExpr::Select(Box::new(c), Box::new(t), Box::new(e)), st, bufa, bufb)
            == (if keval(c, st, bufa, bufb) != 0 {
                keval(t, st, bufa, bufb)
            } else {
                keval(e, st, bufa, bufb)
            }),
{
}

"#
}

fn main() {
    use E::*;
    let p = |i| Box::new(Param(i));
    let t = |i| Box::new(Temp(i));

    // abs_delta: d := if a >= b { a - b } else { b - a }
    let abs_delta = Kernel {
        name: "abs_delta",
        params: vec!["a", "b", "n"],
        requires: vec!["1 <= n", "n <= 1024", "a < n", "b < n"],
        spec_call: "spec_abs_delta(a as int, b as int)",
        stmts: vec![Ite(Box::new(Ge(p(0), p(1))), Box::new(Sub(p(0), p(1))), Box::new(Sub(p(1), p(0))))],
    };

    // wrap_delta: d := |a-b|; nd := n - d; r := if d <= nd { d } else { nd }
    let wrap_delta = Kernel {
        name: "wrap_delta",
        params: vec!["a", "b", "n"],
        requires: vec!["1 <= n", "n <= 1024", "a < n", "b < n"],
        spec_call: "spec_wrap_delta(a as int, b as int, n as int)",
        stmts: vec![
            Ite(Box::new(Ge(p(0), p(1))), Box::new(Sub(p(0), p(1))), Box::new(Sub(p(1), p(0)))),
            Sub(p(2), t(0)),
            Ite(Box::new(Le(t(0), t(1))), t(0), t(1)),
        ],
    };

    // sum_sq3: x2 := dx*dx; y2 := dy*dy; z2 := dz*dz; s := x2+y2; r := s+z2
    let sum_sq3 = Kernel {
        name: "sum_sq3",
        params: vec!["dx", "dy", "dz"],
        requires: vec!["dx <= 512", "dy <= 512", "dz <= 1023"],
        spec_call: "spec_sum_sq3(dx as int, dy as int, dz as int)",
        stmts: vec![
            Mul(p(0), p(0)),
            Mul(p(1), p(1)),
            Mul(p(2), p(2)),
            Add(t(0), t(1)),
            Add(t(3), t(2)),
        ],
    };

    let mut out = String::new();
    out.push_str(prelude());
    for k in [&abs_delta, &wrap_delta, &sum_sq3] {
        emit_kernel(k, &mut out);
    }
    out.push_str("} // verus!\n");

    let dest = concat!(env!("CARGO_MANIFEST_DIR"), "/../probe-g1/src/gen_certs.rs");
    std::fs::write(dest, &out).unwrap();
    println!("wrote {} ({} lines)", dest, out.lines().count());
}
