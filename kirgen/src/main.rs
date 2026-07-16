//! kirgen — G1a/G1b certificate generator.
//!
//! Input: typed kernel definitions (in-file for now; the fork's SST adapter
//! replaces this seam later — DESIGN.md §5). Output: probe-g1/src/gen_certs.rs,
//! a self-contained Verus module with KIR literals and certificate proof fns
//! following the probe-g1 schema exactly. Generated output is never
//! hand-edited: if a certificate fails to verify, fix the generator.
//!
//! Two kernel classes:
//! - expression kernels (G1a): straight-line temps over scalar params;
//!   certificate targets an existing spec fn (spec_kernels::*).
//! - loop kernels (G1b): per-iteration body over carried registers, a loop
//!   counter, counter-indexed buffer reads and one counter-indexed output
//!   write; kirgen emits the recursive model fns (the kernel's canonical
//!   functional spec — the thing source exec fns must `ensures` against),
//!   the step bisimulation, and the loop induction, templated from
//!   probe-g1/src/certificate.rs.
//!
//! v1 loop restrictions (enforced by panics): single carried register,
//! single WriteOut, buffer reads/writes indexed by the counter only.
//!
//! Trust note: hand-transcription errors in the kernel definitions are
//! caught by verification for expression kernels (the ensures references the
//! real spec fn). For loop kernels the generated model IS the spec; it is
//! anchored when a source exec fn verifies `ensures out@ == model_out_*`.

use std::fmt::Write as _;

// ── Typed kernel AST ────────────────────────────────────────────

#[derive(Clone)]
enum E {
    Param(usize),
    Temp(usize),
    Carried(usize),
    Counter,
    ReadA(Box<E>),
    ReadB(Box<E>),
    Add(Box<E>, Box<E>),
    Sub(Box<E>, Box<E>),
    Mul(Box<E>, Box<E>),
    Lt(Box<E>, Box<E>),
    Ge(Box<E>, Box<E>),
    Le(Box<E>, Box<E>),
    Ite(Box<E>, Box<E>, Box<E>),
}

/// Local-slot layout. Expression kernels: params at 0..P, temps after.
/// Loop kernels: carried at 0..C, counter at C, temps at C+1..
struct Layout {
    tempbase: usize,
    counter: Option<usize>,
}

struct Kernel {
    name: &'static str,
    params: Vec<&'static str>,
    requires: Vec<&'static str>,
    spec_call: &'static str,
    stmts: Vec<E>,
}

enum LStmt {
    /// next temp := expr
    Temp(E),
    /// out[Counter] := expr
    WriteOut(E),
    /// carried register r := expr
    SetCarried(usize, E),
}

struct LoopKernel {
    name: &'static str,
    carried: Vec<&'static str>,
    stmts: Vec<LStmt>,
    /// locals width = carried + counter + temps
    width: usize,
}

// ── Shared lowering helpers ─────────────────────────────────────

fn loc(lay: &Layout, e: &E) -> usize {
    match e {
        E::Param(i) => *i,
        E::Carried(r) => *r,
        E::Counter => lay.counter.expect("counter in expression kernel"),
        E::Temp(t) => lay.tempbase + t,
        _ => unreachable!("loc of non-leaf"),
    }
}

fn is_leaf(e: &E) -> bool {
    matches!(e, E::Param(_) | E::Temp(_) | E::Carried(_) | E::Counter)
}

fn kexpr(lay: &Layout, e: &E) -> String {
    let b = |x: &E| format!("Box::new({})", kexpr(lay, x));
    match e {
        _ if is_leaf(e) => format!("KExpr::Loc({})", loc(lay, e)),
        E::ReadA(x) => format!("KExpr::ReadA({})", b(x)),
        E::ReadB(x) => format!("KExpr::ReadB({})", b(x)),
        E::Add(x, y) => format!("KExpr::AddW({}, {})", b(x), b(y)),
        E::Sub(x, y) => format!("KExpr::SubW({}, {})", b(x), b(y)),
        E::Mul(x, y) => format!("KExpr::MulW({}, {})", b(x), b(y)),
        E::Lt(x, y) => format!("KExpr::LtU({}, {})", b(x), b(y)),
        E::Ge(x, y) => format!("KExpr::GeU({}, {})", b(x), b(y)),
        E::Le(x, y) => format!("KExpr::LeU({}, {})", b(x), b(y)),
        E::Ite(c, t, e2) => format!("KExpr::Select({}, {}, {})", b(c), b(t), b(e2)),
        _ => unreachable!(),
    }
}

/// Semantic value with leaf slots resolved through `env` (slot index -> text)
/// and counter-indexed reads resolved to buffer indexing text.
fn sem(lay: &Layout, e: &E, env: &dyn Fn(usize) -> String) -> String {
    match e {
        _ if is_leaf(e) => env(loc(lay, e)),
        E::ReadA(x) => {
            assert!(matches!(**x, E::Counter), "v1: ReadA index must be the counter");
            "bufa[i as int]".to_string()
        }
        E::ReadB(x) => {
            assert!(matches!(**x, E::Counter), "v1: ReadB index must be the counter");
            "bufb[i as int]".to_string()
        }
        E::Add(x, y) => format!("wadd({}, {})", sem(lay, x, env), sem(lay, y, env)),
        E::Sub(x, y) => format!("wsub({}, {})", sem(lay, x, env), sem(lay, y, env)),
        E::Mul(x, y) => format!("wmul({}, {})", sem(lay, x, env), sem(lay, y, env)),
        E::Lt(x, y) => format!("ltu({}, {})", sem(lay, x, env), sem(lay, y, env)),
        E::Ge(x, y) => format!("geu({}, {})", sem(lay, x, env), sem(lay, y, env)),
        E::Le(x, y) => format!("leu({}, {})", sem(lay, x, env), sem(lay, y, env)),
        E::Ite(c, t, e2) => format!(
            "(if {} != 0 {{ {} }} else {{ {} }})",
            sem(lay, c, env), sem(lay, t, env), sem(lay, e2, env)),
        _ => unreachable!(),
    }
}

/// Emit bottom-up u_keval_* calls for every node of e at state `st`.
fn keval_calls(lay: &Layout, e: &E, st: &str, out: &mut String) {
    let bin = |out: &mut String, name: &str, x: &E, y: &E| {
        writeln!(out, "    {}({}, {}, {}, bufa, bufb);",
            name, kexpr(lay, x), kexpr(lay, y), st).unwrap();
    };
    match e {
        _ if is_leaf(e) => {
            writeln!(out, "    u_keval_loc({}, {}, bufa, bufb);", loc(lay, e), st).unwrap();
        }
        E::ReadA(x) => {
            keval_calls(lay, x, st, out);
            writeln!(out, "    u_keval_reada({}, {}, bufa, bufb);", kexpr(lay, x), st).unwrap();
        }
        E::ReadB(x) => {
            keval_calls(lay, x, st, out);
            writeln!(out, "    u_keval_readb({}, {}, bufa, bufb);", kexpr(lay, x), st).unwrap();
        }
        E::Add(x, y) => { keval_calls(lay, x, st, out); keval_calls(lay, y, st, out); bin(out, "u_keval_addw", x, y); }
        E::Sub(x, y) => { keval_calls(lay, x, st, out); keval_calls(lay, y, st, out); bin(out, "u_keval_subw", x, y); }
        E::Mul(x, y) => { keval_calls(lay, x, st, out); keval_calls(lay, y, st, out); bin(out, "u_keval_mulw", x, y); }
        E::Lt(x, y) => { keval_calls(lay, x, st, out); keval_calls(lay, y, st, out); bin(out, "u_keval_ltu", x, y); }
        E::Ge(x, y) => { keval_calls(lay, x, st, out); keval_calls(lay, y, st, out); bin(out, "u_keval_geu", x, y); }
        E::Le(x, y) => { keval_calls(lay, x, st, out); keval_calls(lay, y, st, out); bin(out, "u_keval_leu", x, y); }
        E::Ite(c, t, e2) => {
            keval_calls(lay, c, st, out); keval_calls(lay, t, st, out); keval_calls(lay, e2, st, out);
            writeln!(out, "    u_keval_select({}, {}, {}, {}, bufa, bufb);",
                kexpr(lay, c), kexpr(lay, t), kexpr(lay, e2), st).unwrap();
        }
        _ => unreachable!(),
    }
}

fn mul_nodes<'a>(e: &'a E, acc: &mut Vec<(&'a E, &'a E)>) {
    match e {
        _ if is_leaf(e) => {}
        E::ReadA(x) | E::ReadB(x) => mul_nodes(x, acc),
        E::Add(x, y) | E::Sub(x, y) | E::Mul(x, y) | E::Lt(x, y) | E::Ge(x, y) | E::Le(x, y) => {
            mul_nodes(x, acc); mul_nodes(y, acc);
            if let E::Mul(a, b) = e { acc.push((a, b)); }
        }
        E::Ite(c, t, e2) => { mul_nodes(c, acc); mul_nodes(t, acc); mul_nodes(e2, acc); }
        _ => unreachable!(),
    }
}

fn reads(e: &E, ra: &mut bool, rb: &mut bool) {
    match e {
        E::ReadA(_) => *ra = true,
        E::ReadB(_) => *rb = true,
        E::Add(x, y) | E::Sub(x, y) | E::Mul(x, y) | E::Lt(x, y) | E::Ge(x, y) | E::Le(x, y) => {
            reads(x, ra, rb); reads(y, ra, rb);
        }
        E::Ite(c, t, e2) => { reads(c, ra, rb); reads(t, ra, rb); reads(e2, ra, rb); }
        _ => {}
    }
}

/// The uniform Lean seam cascade (probe schema shape 5).
fn cascade(out: &mut String, simp_names: &str) {
    writeln!(out, "        intros").unwrap();
    writeln!(out, "        simp_all (config := {{ zetaDelta := true }}) [{}]", simp_names).unwrap();
    writeln!(out, "        <;> (try split_ifs)").unwrap();
    writeln!(out, "        <;> (try push_cast at *)").unwrap();
    writeln!(out, "        <;> (first | omega | nlinarith)").unwrap();
}

fn plain_simp(out: &mut String, simp_names: &str) {
    writeln!(out, "        intros").unwrap();
    writeln!(out, "        simp_all (config := {{ zetaDelta := true }}) [{}]", simp_names).unwrap();
}

// ── Expression-kernel emission (G1a, unchanged schema) ─────────

fn emit_kernel(k: &Kernel, out: &mut String) {
    let lay = Layout { tempbase: k.params.len(), counter: None };
    let p = k.params.len();
    let ns = k.stmts.len();
    let width = p + ns;
    let result_loc = width - 1;
    let n = k.name;

    let mut vals: Vec<String> = Vec::new();
    for e in &k.stmts {
        let env = |i: usize| -> String {
            if i < p { k.params[i].to_string() } else { format!("v{}_{}", n, i - p) }
        };
        vals.push(sem(&lay, e, &env));
    }

    writeln!(out, "// ══════════ expression kernel: {} ══════════\n", n).unwrap();

    for (i, e) in k.stmts.iter().enumerate() {
        writeln!(out, "pub open spec fn stmt_{}_{}() -> KStmt {{", n, i).unwrap();
        writeln!(out, "    KStmt::Assign {{ loc: {}, rhs: {} }}", p + i, kexpr(&lay, e)).unwrap();
        writeln!(out, "}}\n").unwrap();
    }
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
    if ns == 1 {
        writeln!(out, "pub proof fn u_seq_{}_0(st: KState, bufa: Seq<u32>, bufb: Seq<u32>)", n).unwrap();
        writeln!(out, "    ensures kexec(body_{}(), st, bufa, bufb) == kexec(stmt_{}_0(), st, bufa, bufb),", n, n).unwrap();
        writeln!(out, "{{\n}}\n").unwrap();
    }

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

    for (t, v) in vals.iter().enumerate() {
        writeln!(out, "    let v{}_{} = {};", n, t, v).unwrap();
    }
    for t in 0..ns {
        let prev = if t == 0 { "st.locals".to_string() } else { format!("ll{}_{}", n, t - 1) };
        writeln!(out, "    let ll{}_{} = {}.update({}, v{}_{});", n, t, prev, p + t, n, t).unwrap();
    }
    writeln!(out).unwrap();

    for t in 0..ns {
        let prev = if t == 0 { "st.locals".to_string() } else { format!("ll{}_{}", n, t - 1) };
        writeln!(out, "    vstd::seq::axiom_seq_update_len::<u32>({}, {}, v{}_{});", prev, p + t, n, t).unwrap();
    }
    {
        let mut conds: Vec<String> = Vec::new();
        for t in 0..ns { conds.push(format!("ll{}_{}.len() == {}", n, t, width)); }
        writeln!(out, "    assert({}) by {{", conds.join(" && ")).unwrap();
        plain_simp(out, "");
        writeln!(out, "    }};\n").unwrap();
    }
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
        plain_simp(out, "");
        writeln!(out, "    }};\n").unwrap();
    }

    for t in 0..ns {
        let stv = if t == 0 { "st".to_string() } else { format!("s{}_{}", n, t - 1) };
        if t < ns.saturating_sub(1) || ns == 1 {
            if t == 0 {
                writeln!(out, "    u_seq_{}_0(st, bufa, bufb);", n).unwrap();
            } else {
                writeln!(out, "    u_seq_{}_{}({}, bufa, bufb);", n, t, stv).unwrap();
            }
        }
        keval_calls(&lay, &k.stmts[t], &stv, out);
        writeln!(out, "    u_kexec_assign({}, {}, {}, bufa, bufb);", p + t, kexpr(&lay, &k.stmts[t]), stv).unwrap();
        writeln!(out, "    let s{}_{} = mk_state(ll{}_{}, st.out);", n, t, n, t).unwrap();
        writeln!(out, "    assert(kexec(stmt_{}_{}(), {}, bufa, bufb) == s{}_{} && s{}_{}.locals == ll{}_{} && s{}_{}.out == st.out) by {{", n, t, stv, n, t, n, t, n, t, n, t).unwrap();
        plain_simp(out, &format!("gen_certs.stmt_{}_{}, addloop.mk_state", n, t));
        writeln!(out, "    }};\n").unwrap();
    }
    let last = format!("s{}_{}", n, ns - 1);
    writeln!(out, "    assert(kexec(body_{}(), st, bufa, bufb) == {}) by {{", n, last).unwrap();
    plain_simp(out, "");
    writeln!(out, "    }};\n").unwrap();

    let mut muls = Vec::new();
    for e in &k.stmts { mul_nodes(e, &mut muls); }
    for (x, y) in &muls {
        let env = |i: usize| -> String {
            if i < p { k.params[i].to_string() } else { format!("v{}_{}", n, i - p) }
        };
        let (sx, sy) = (sem(&lay, x, &env), sem(&lay, y, &env));
        writeln!(out, "    assert(({} as int) * ({} as int) < 0x1_0000_0000) by {{", sx, sy).unwrap();
        cascade(out, "kir.wadd, kir.wsub, kir.wmul, kir.geu, kir.leu, kir.ltu");
        writeln!(out, "    }};").unwrap();
    }

    writeln!(out, "    assert(v{}_{} as int == {}) by {{", n, ns - 1, k.spec_call).unwrap();
    cascade(out, &format!("spec_kernels.{}, kir.wadd, kir.wsub, kir.wmul, kir.geu, kir.leu, kir.ltu",
        k.spec_call.split('(').next().unwrap()));
    writeln!(out, "    }};").unwrap();
    writeln!(out, "}}\n").unwrap();
}

// ── Loop-kernel emission (G1b: the probe schema, templated) ────

fn emit_loop_kernel(k: &LoopKernel, out: &mut String) {
    assert!(k.carried.len() == 1, "v1: exactly one carried register");
    let nc = k.carried.len();
    let ctr = nc; // counter local
    let lay = Layout { tempbase: nc + 1, counter: Some(ctr) };
    let n = k.name;
    let ns = k.stmts.len();
    let width = k.width;

    let nwrites = k.stmts.iter().filter(|s| matches!(s, LStmt::WriteOut(_))).count();
    assert!(nwrites == 1, "v1: exactly one WriteOut");

    // Inline semantic env for model fns: carried -> c0, temps inlined.
    fn inline_sem(k: &LoopKernel, lay: &Layout, e: &E) -> String {
        let lk = k as *const LoopKernel;
        let lay2 = Layout { tempbase: lay.tempbase, counter: lay.counter };
        let env = move |slot: usize| -> String {
            let k = unsafe { &*lk };
            if slot < k.carried.len() {
                format!("c{}", slot)
            } else if slot == k.carried.len() {
                unreachable!("counter used as value in v1")
            } else {
                let t = slot - k.carried.len() - 1;
                match &k.stmts[temp_stmt_index(k, t)] {
                    LStmt::Temp(te) => inline_sem(k, &lay2, te),
                    _ => unreachable!(),
                }
            }
        };
        sem(lay, e, &env)
    }
    fn temp_stmt_index(k: &LoopKernel, t: usize) -> usize {
        let mut seen = 0;
        for (i, s) in k.stmts.iter().enumerate() {
            if matches!(s, LStmt::Temp(_)) {
                if seen == t { return i; }
                seen += 1;
            }
        }
        unreachable!()
    }

    let write_val: &E = k.stmts.iter().find_map(|s| match s {
        LStmt::WriteOut(v) => Some(v), _ => None }).unwrap();
    let carried_next: &E = k.stmts.iter().find_map(|s| match s {
        LStmt::SetCarried(0, v) => Some(v), _ => None })
        .expect("v1: carried register must be assigned");

    writeln!(out, "// ══════════ loop kernel: {} ══════════\n", n).unwrap();

    // Model step fns (the canonical per-iteration spec).
    writeln!(out, "/// Written output value for iteration i (canonical spec).").unwrap();
    writeln!(out, "pub open spec fn step_val_{}(bufa: Seq<u32>, bufb: Seq<u32>, c0: u32, i: nat) -> u32 {{", n).unwrap();
    writeln!(out, "    {}", inline_sem(k, &lay, write_val)).unwrap();
    writeln!(out, "}}\n").unwrap();
    writeln!(out, "/// Carried-register value after iteration i (canonical spec).").unwrap();
    writeln!(out, "pub open spec fn step_c0_{}(bufa: Seq<u32>, bufb: Seq<u32>, c0: u32, i: nat) -> u32 {{", n).unwrap();
    writeln!(out, "    {}", inline_sem(k, &lay, carried_next)).unwrap();
    writeln!(out, "}}\n").unwrap();

    // Recursive model loops.
    writeln!(out, "pub open spec fn model_out_{}(bufa: Seq<u32>, bufb: Seq<u32>, out: Seq<u32>, c0: u32, lo: nat, n: nat) -> Seq<u32>", n).unwrap();
    writeln!(out, "    decreases n - lo,").unwrap();
    writeln!(out, "{{").unwrap();
    writeln!(out, "    if lo >= n {{ out }} else {{").unwrap();
    writeln!(out, "        model_out_{}(bufa, bufb, out.update(lo as int, step_val_{}(bufa, bufb, c0, lo)), step_c0_{}(bufa, bufb, c0, lo), lo + 1, n)", n, n, n).unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}\n").unwrap();
    writeln!(out, "pub open spec fn model_c0_{}(bufa: Seq<u32>, bufb: Seq<u32>, c0: u32, lo: nat, n: nat) -> u32", n).unwrap();
    writeln!(out, "    decreases n - lo,").unwrap();
    writeln!(out, "{{").unwrap();
    writeln!(out, "    if lo >= n {{ c0 }} else {{").unwrap();
    writeln!(out, "        model_c0_{}(bufa, bufb, step_c0_{}(bufa, bufb, c0, lo), lo + 1, n)", n, n).unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}\n").unwrap();

    // Model unfold lemmas.
    for (fnname, sig, step, done) in [
        ("out", "out: Seq<u32>, ", true, false),
        ("out", "out: Seq<u32>, ", false, true),
        ("c0", "", true, false),
        ("c0", "", false, true),
    ] {
        let which = if step { "step" } else { "done" };
        let _ = done;
        writeln!(out, "pub proof fn u_model_{}_{}_{}(bufa: Seq<u32>, bufb: Seq<u32>, {}c0: u32, lo: nat, n: nat)", fnname, which, n, sig).unwrap();
        writeln!(out, "    requires lo {} n,", if step { "<" } else { ">=" }).unwrap();
        writeln!(out, "    ensures").unwrap();
        if fnname == "out" {
            if step {
                writeln!(out, "        model_out_{}(bufa, bufb, out, c0, lo, n)", n).unwrap();
                writeln!(out, "            == model_out_{}(bufa, bufb, out.update(lo as int, step_val_{}(bufa, bufb, c0, lo)), step_c0_{}(bufa, bufb, c0, lo), lo + 1, n),", n, n, n).unwrap();
            } else {
                writeln!(out, "        model_out_{}(bufa, bufb, out, c0, lo, n) == out,", n).unwrap();
            }
        } else {
            if step {
                writeln!(out, "        model_c0_{}(bufa, bufb, c0, lo, n)", n).unwrap();
                writeln!(out, "            == model_c0_{}(bufa, bufb, step_c0_{}(bufa, bufb, c0, lo), lo + 1, n),", n, n).unwrap();
            } else {
                writeln!(out, "        model_c0_{}(bufa, bufb, c0, lo, n) == c0,", n).unwrap();
            }
        }
        writeln!(out, "{{\n}}\n").unwrap();
    }

    // Statement literals, chains, u_seq unfolds.
    let stmt_text = |s: &LStmt| -> String {
        match s {
            LStmt::Temp(_) | LStmt::SetCarried(..) => unreachable!(),
            LStmt::WriteOut(v) => format!(
                "KStmt::WriteOut {{ idx: KExpr::Loc({}), val: {} }}", ctr, kexpr(&lay, v)),
        }
    };
    let mut tempno = 0usize;
    let mut assign_loc: Vec<usize> = Vec::new(); // per stmt: target local (usize::MAX = writeout)
    for (i, s) in k.stmts.iter().enumerate() {
        write!(out, "pub open spec fn stmt_{}_{}() -> KStmt {{\n    ", n, i).unwrap();
        match s {
            LStmt::Temp(e) => {
                let l = lay.tempbase + tempno;
                tempno += 1;
                assign_loc.push(l);
                writeln!(out, "KStmt::Assign {{ loc: {}, rhs: {} }}", l, kexpr(&lay, e)).unwrap();
            }
            LStmt::SetCarried(r, e) => {
                assign_loc.push(*r);
                writeln!(out, "KStmt::Assign {{ loc: {}, rhs: {} }}", r, kexpr(&lay, e)).unwrap();
            }
            LStmt::WriteOut(_) => {
                assign_loc.push(usize::MAX);
                writeln!(out, "{}", stmt_text(s)).unwrap();
            }
        }
        writeln!(out, "}}\n").unwrap();
    }
    for i in (0..ns - 1).rev() {
        let tail = if i + 1 == ns - 1 {
            format!("stmt_{}_{}()", n, ns - 1)
        } else {
            format!("chain_{}_{}()", n, i + 1)
        };
        writeln!(out, "pub open spec fn chain_{}_{}() -> KStmt {{", n, i).unwrap();
        writeln!(out, "    KStmt::Seq2(Box::new(stmt_{}_{}()), Box::new({}))", n, i, tail).unwrap();
        writeln!(out, "}}\n").unwrap();
    }
    writeln!(out, "pub open spec fn body_{}() -> KStmt {{ chain_{}_0() }}\n", n, n).unwrap();
    for i in 0..ns - 1 {
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

    // Per-statement step lemmas (uniform signature).
    for (i, s) in k.stmts.iter().enumerate() {
        let (mut ra, mut rb) = (false, false);
        let e_opt: Option<&E> = match s {
            LStmt::Temp(e) | LStmt::SetCarried(_, e) | LStmt::WriteOut(e) => Some(e),
        };
        if let Some(e) = e_opt { reads(e, &mut ra, &mut rb); }
        let wo = matches!(s, LStmt::WriteOut(_));

        writeln!(out, "#[verifier::tactus_auto]").unwrap();
        writeln!(out, "pub proof fn lstep_{}_{}(st: KState, bufa: Seq<u32>, bufb: Seq<u32>, i: nat)", n, i).unwrap();
        writeln!(out, "    requires").unwrap();
        writeln!(out, "        st.locals.len() == {},", width).unwrap();
        writeln!(out, "        st.locals[{}] == i as u32,", ctr).unwrap();
        writeln!(out, "        i < 0x1_0000_0000,").unwrap();
        if ra { writeln!(out, "        (i as int) < bufa.len(),").unwrap(); }
        if rb { writeln!(out, "        (i as int) < bufb.len(),").unwrap(); }
        if wo { writeln!(out, "        (i as int) < st.out.len(),").unwrap(); }
        writeln!(out, "    ensures").unwrap();
        let stenv = |j: usize| format!("st.locals[{}]", j);
        match s {
            LStmt::Temp(e) | LStmt::SetCarried(_, e) => {
                writeln!(out, "        kexec(stmt_{}_{}(), st, bufa, bufb)", n, i).unwrap();
                writeln!(out, "            == mk_state(st.locals.update({}, {}), st.out),",
                    assign_loc[i], sem(&lay, e, &stenv)).unwrap();
            }
            LStmt::WriteOut(v) => {
                writeln!(out, "        kexec(stmt_{}_{}(), st, bufa, bufb)", n, i).unwrap();
                writeln!(out, "            == mk_state(st.locals, st.out.update(i as int, {})),",
                    sem(&lay, v, &stenv)).unwrap();
            }
        }
        writeln!(out, "{{").unwrap();
        // cast bridge (counter is used by every lemma's requires; emit when
        // the counter actually indexes something)
        if ra || rb || wo {
            writeln!(out, "    assert((i as u32) as int == i as int) by {{").unwrap();
            writeln!(out, "        intros").unwrap();
            writeln!(out, "        push_cast").unwrap();
            writeln!(out, "        omega").unwrap();
            writeln!(out, "    }};").unwrap();
        }
        match s {
            LStmt::Temp(e) | LStmt::SetCarried(_, e) => {
                keval_calls(&lay, e, "st", out);
                writeln!(out, "    u_kexec_assign({}, {}, st, bufa, bufb);", assign_loc[i], kexpr(&lay, e)).unwrap();
            }
            LStmt::WriteOut(v) => {
                writeln!(out, "    u_keval_loc({}, st, bufa, bufb);", ctr).unwrap();
                keval_calls(&lay, v, "st", out);
                writeln!(out, "    u_kexec_writeout(KExpr::Loc({}), {}, st, bufa, bufb);", ctr, kexpr(&lay, v)).unwrap();
            }
        }
        writeln!(out, "    assert(kexec(stmt_{}_{}(), st, bufa, bufb) == {}) by {{", n, i,
            match s {
                LStmt::Temp(e) | LStmt::SetCarried(_, e) =>
                    format!("mk_state(st.locals.update({}, {}), st.out)", assign_loc[i], sem(&lay, e, &stenv)),
                LStmt::WriteOut(v) =>
                    format!("mk_state(st.locals, st.out.update(i as int, {}))", sem(&lay, v, &stenv)),
            }).unwrap();
        plain_simp(out, &format!("gen_certs.stmt_{}_{}, addloop.mk_state", n, i));
        writeln!(out, "    }};").unwrap();
        writeln!(out, "}}\n").unwrap();
    }

    // ── Composed step bisimulation ──
    writeln!(out, "#[verifier::tactus_auto]").unwrap();
    writeln!(out, "pub proof fn lemma_step_bisim_{}(st: KState, bufa: Seq<u32>, bufb: Seq<u32>, i: nat)", n).unwrap();
    writeln!(out, "    requires").unwrap();
    writeln!(out, "        st.locals.len() == {},", width).unwrap();
    writeln!(out, "        st.locals[{}] == i as u32,", ctr).unwrap();
    writeln!(out, "        i < 0x1_0000_0000,").unwrap();
    writeln!(out, "        (i as int) < bufa.len(),").unwrap();
    writeln!(out, "        (i as int) < bufb.len(),").unwrap();
    writeln!(out, "        (i as int) < st.out.len(),").unwrap();
    writeln!(out, "    ensures").unwrap();
    writeln!(out, "        kexec(body_{}(), st, bufa, bufb).locals.len() == {},", n, width).unwrap();
    writeln!(out, "        kexec(body_{}(), st, bufa, bufb).locals[0] == step_c0_{}(bufa, bufb, st.locals[0], i),", n, n).unwrap();
    writeln!(out, "        kexec(body_{}(), st, bufa, bufb).locals[{}] == st.locals[{}],", n, ctr, ctr).unwrap();
    writeln!(out, "        kexec(body_{}(), st, bufa, bufb).out == st.out.update(i as int, step_val_{}(bufa, bufb, st.locals[0], i)),", n, n).unwrap();
    writeln!(out, "{{").unwrap();
    writeln!(out, "    let l = st.locals;").unwrap();

    // value lets (temps + carried-next), tracked env
    let lenv = |k2: &LoopKernel, j: usize| -> String {
        if j < k2.carried.len() { format!("l[{}]", j) }
        else if j == k2.carried.len() { unreachable!() }
        else { format!("tv{}", j - k2.carried.len() - 1) }
    };
    let mut tno = 0usize;
    let mut cnew_sym = String::new();
    let mut wval_sym = String::new();
    for s in &k.stmts {
        match s {
            LStmt::Temp(e) => {
                writeln!(out, "    let tv{} = {};", tno, sem(&lay, e, &|j| lenv(k, j))).unwrap();
                tno += 1;
            }
            LStmt::SetCarried(_, e) => {
                cnew_sym = "cnew".to_string();
                writeln!(out, "    let cnew = {};", sem(&lay, e, &|j| lenv(k, j))).unwrap();
            }
            LStmt::WriteOut(v) => {
                wval_sym = sem(&lay, v, &|j| lenv(k, j));
            }
        }
    }
    // locals chain lets + out let
    let mut cur = "l".to_string();
    let mut lvl = 0usize;
    let mut ll_of_stmt: Vec<Option<usize>> = Vec::new();
    let mut val_of_lvl: Vec<String> = Vec::new();
    let mut loc_of_lvl: Vec<usize> = Vec::new();
    {
        let mut tno2 = 0usize;
        for s in &k.stmts {
            match s {
                LStmt::Temp(_) => {
                    let v = format!("tv{}", tno2);
                    tno2 += 1;
                    writeln!(out, "    let ll{} = {}.update({}, {});", lvl, cur, lay.tempbase + (tno2 - 1), v).unwrap();
                    val_of_lvl.push(v);
                    loc_of_lvl.push(lay.tempbase + (tno2 - 1));
                    cur = format!("ll{}", lvl);
                    ll_of_stmt.push(Some(lvl));
                    lvl += 1;
                }
                LStmt::SetCarried(r, _) => {
                    writeln!(out, "    let ll{} = {}.update({}, {});", lvl, cur, r, cnew_sym).unwrap();
                    val_of_lvl.push(cnew_sym.clone());
                    loc_of_lvl.push(*r);
                    cur = format!("ll{}", lvl);
                    ll_of_stmt.push(Some(lvl));
                    lvl += 1;
                }
                LStmt::WriteOut(_) => {
                    writeln!(out, "    let outw = st.out.update(i as int, {});", wval_sym).unwrap();
                    ll_of_stmt.push(None);
                }
            }
        }
    }
    writeln!(out).unwrap();

    // len axioms + assert
    for t in 0..lvl {
        let prev = if t == 0 { "l".to_string() } else { format!("ll{}", t - 1) };
        writeln!(out, "    vstd::seq::axiom_seq_update_len::<u32>({}, {}, {});", prev, loc_of_lvl[t], val_of_lvl[t]).unwrap();
    }
    {
        let conds: Vec<String> = (0..lvl).map(|t| format!("ll{}.len() == {}", t, width)).collect();
        writeln!(out, "    assert({}) by {{", conds.join(" && ")).unwrap();
        plain_simp(out, "");
        writeln!(out, "    }};\n").unwrap();
    }
    // live-index tracking: for each level, counter + carried + temps-so-far
    for t in 0..lvl {
        let prev = if t == 0 { "l".to_string() } else { format!("ll{}", t - 1) };
        let mut live: Vec<usize> = vec![ctr];
        for r in 0..nc { live.push(r); }
        for u in 0..lvl {
            if u <= t && loc_of_lvl[u] >= lay.tempbase { live.push(loc_of_lvl[u]); }
        }
        live.sort(); live.dedup();
        for &j in &live {
            if j == loc_of_lvl[t] {
                writeln!(out, "    vstd::seq::axiom_seq_update_same::<u32>({}, {}, {});", prev, j, val_of_lvl[t]).unwrap();
            } else {
                writeln!(out, "    vstd::seq::axiom_seq_update_different::<u32>({}, {}, {}, {});", prev, j, loc_of_lvl[t], val_of_lvl[t]).unwrap();
            }
        }
        let conds: Vec<String> = live.iter().map(|&j| {
            let v = if j == ctr { "i as u32".to_string() }
                else if j < nc {
                    // carried: value is l[j] until its Set level, then cnew
                    let set_lvl = (0..lvl).find(|&u| loc_of_lvl[u] == j);
                    match set_lvl {
                        Some(u) if u <= t => val_of_lvl[u].clone(),
                        _ => format!("l[{}]", j),
                    }
                }
                else {
                    let u = (0..=t).rev().find(|&u| loc_of_lvl[u] == j).unwrap();
                    val_of_lvl[u].clone()
                };
            format!("ll{}[{}] == {}", t, j, v)
        }).collect();
        writeln!(out, "    assert({}) by {{", conds.join(" && ")).unwrap();
        plain_simp(out, "");
        writeln!(out, "    }};\n").unwrap();
    }

    // execution chain
    let mut cur_state = "st".to_string();
    let mut cur_out = "st.out".to_string();
    for (si, _s) in k.stmts.iter().enumerate() {
        if si < ns - 1 {
            if si == 0 {
                writeln!(out, "    u_seq_{}_0(st, bufa, bufb);", n).unwrap();
            } else {
                writeln!(out, "    u_seq_{}_{}({}, bufa, bufb);", n, si, cur_state).unwrap();
            }
        }
        writeln!(out, "    lstep_{}_{}({}, bufa, bufb, i);", n, si, cur_state).unwrap();
        let (locals_sym, out_sym) = match ll_of_stmt[si] {
            Some(l) => (format!("ll{}", l), cur_out.clone()),
            None => {
                cur_out = "outw".to_string();
                // locals unchanged: previous locals symbol
                let prevloc = (0..si).rev().find_map(|u| ll_of_stmt[u].map(|l| format!("ll{}", l)))
                    .unwrap_or_else(|| "l".to_string());
                (prevloc, "outw".to_string())
            }
        };
        writeln!(out, "    let gs{} = mk_state({}, {});", si, locals_sym, out_sym).unwrap();
        writeln!(out, "    assert(kexec(stmt_{}_{}(), {}, bufa, bufb) == gs{} && gs{}.locals == {} && gs{}.out == {}) by {{",
            n, si, cur_state, si, si, locals_sym, si, out_sym).unwrap();
        plain_simp(out, "addloop.mk_state");
        writeln!(out, "    }};\n").unwrap();
        cur_state = format!("gs{}", si);
    }
    writeln!(out, "    assert(kexec(body_{}(), st, bufa, bufb) == {}) by {{", n, cur_state).unwrap();
    plain_simp(out, "");
    writeln!(out, "    }};\n").unwrap();

    // model connection (near-definitional: same AST both sides)
    writeln!(out, "    assert({} == step_c0_{}(bufa, bufb, l[0], i) && {} == step_val_{}(bufa, bufb, l[0], i)) by {{",
        cnew_sym, n, wval_sym, n).unwrap();
    cascade(out, &format!("gen_certs.step_c0_{}, gen_certs.step_val_{}", n, n));
    writeln!(out, "    }};").unwrap();
    writeln!(out, "}}\n").unwrap();

    // ── Loop induction (probe lemma_loop_bisim, templated) ──
    writeln!(out, "#[verifier::tactus_auto]").unwrap();
    writeln!(out, "pub proof fn lemma_loop_bisim_{}(st: KState, bufa: Seq<u32>, bufb: Seq<u32>, lo: nat, n: nat)", n).unwrap();
    writeln!(out, "    requires").unwrap();
    writeln!(out, "        st.locals.len() == {},", width).unwrap();
    writeln!(out, "        lo <= n,").unwrap();
    writeln!(out, "        n <= bufa.len(),").unwrap();
    writeln!(out, "        n <= bufb.len(),").unwrap();
    writeln!(out, "        (n as int) <= st.out.len(),").unwrap();
    writeln!(out, "        n < 0x1_0000_0000,").unwrap();
    writeln!(out, "    ensures").unwrap();
    writeln!(out, "        kloop(body_{}(), {}, lo, n, st, bufa, bufb).locals.len() == {},", n, ctr, width).unwrap();
    writeln!(out, "        kloop(body_{}(), {}, lo, n, st, bufa, bufb).locals[0]", n, ctr).unwrap();
    writeln!(out, "            == model_c0_{}(bufa, bufb, st.locals[0], lo, n),", n).unwrap();
    writeln!(out, "        kloop(body_{}(), {}, lo, n, st, bufa, bufb).out", n, ctr).unwrap();
    writeln!(out, "            == model_out_{}(bufa, bufb, st.out, st.locals[0], lo, n),", n).unwrap();
    writeln!(out, "        kloop(body_{}(), {}, lo, n, st, bufa, bufb).out.len() == st.out.len(),", n, ctr).unwrap();
    writeln!(out, "    decreases n - lo,").unwrap();
    writeln!(out, "{{").unwrap();
    writeln!(out, "    if lo >= n {{").unwrap();
    writeln!(out, "        u_kloop_done(body_{}(), {}, lo, n, st, bufa, bufb);", n, ctr).unwrap();
    writeln!(out, "        u_model_out_done_{}(bufa, bufb, st.out, st.locals[0], lo, n);", n).unwrap();
    writeln!(out, "        u_model_c0_done_{}(bufa, bufb, st.locals[0], lo, n);", n).unwrap();
    writeln!(out, "    }} else {{").unwrap();
    writeln!(out, "        let stl = mk_state(st.locals.update({}, lo as u32), st.out);", ctr).unwrap();
    writeln!(out, "        u_kloop_step(body_{}(), {}, lo, n, st, bufa, bufb);", n, ctr).unwrap();
    writeln!(out, "        vstd::seq::axiom_seq_update_len::<u32>(st.locals, {}, lo as u32);", ctr).unwrap();
    writeln!(out, "        vstd::seq::axiom_seq_update_same::<u32>(st.locals, {}, lo as u32);", ctr).unwrap();
    for r in 0..nc {
        writeln!(out, "        vstd::seq::axiom_seq_update_different::<u32>(st.locals, {}, {}, lo as u32);", r, ctr).unwrap();
    }
    writeln!(out, "        assert(stl.locals.len() == {} && stl.locals[{}] == lo as u32", width, ctr).unwrap();
    writeln!(out, "            && stl.locals[0] == st.locals[0] && stl.out == st.out) by {{").unwrap();
    writeln!(out, "            intros").unwrap();
    writeln!(out, "            simp_all (config := {{ zetaDelta := true }}) [addloop.mk_state]").unwrap();
    writeln!(out, "        }};").unwrap();
    writeln!(out, "        assert(lo < 0x1_0000_0000 && (lo as int) < bufa.len() && (lo as int) < bufb.len()").unwrap();
    writeln!(out, "            && (lo as int) < stl.out.len()) by {{").unwrap();
    writeln!(out, "            intros").unwrap();
    writeln!(out, "            omega").unwrap();
    writeln!(out, "        }};\n").unwrap();
    writeln!(out, "        lemma_step_bisim_{}(stl, bufa, bufb, lo);", n).unwrap();
    writeln!(out, "        let st2 = kexec(body_{}(), stl, bufa, bufb);", n).unwrap();
    writeln!(out, "        vstd::seq::axiom_seq_update_len::<u32>(st.out, lo as int, step_val_{}(bufa, bufb, st.locals[0], lo));", n).unwrap();
    writeln!(out, "        assert(st2.locals.len() == {} && (n as int) <= st2.out.len()", width).unwrap();
    writeln!(out, "            && st2.out.len() == st.out.len()").unwrap();
    writeln!(out, "            && st2.locals[0] == step_c0_{}(bufa, bufb, st.locals[0], lo)", n).unwrap();
    writeln!(out, "            && st2.out == st.out.update(lo as int, step_val_{}(bufa, bufb, st.locals[0], lo))) by {{", n).unwrap();
    writeln!(out, "            intros").unwrap();
    writeln!(out, "            simp_all (config := {{ zetaDelta := true }}) []").unwrap();
    writeln!(out, "        }};\n").unwrap();
    writeln!(out, "        lemma_loop_bisim_{}(st2, bufa, bufb, lo + 1, n);", n).unwrap();
    writeln!(out, "        u_model_out_step_{}(bufa, bufb, st.out, st.locals[0], lo, n);", n).unwrap();
    writeln!(out, "        u_model_c0_step_{}(bufa, bufb, st.locals[0], lo, n);", n).unwrap();
    writeln!(out, "        assert(kloop(body_{}(), {}, lo, n, st, bufa, bufb)", n, ctr).unwrap();
    writeln!(out, "            == kloop(body_{}(), {}, lo + 1, n, st2, bufa, bufb)) by {{", n, ctr).unwrap();
    writeln!(out, "            intros").unwrap();
    writeln!(out, "            simp_all (config := {{ zetaDelta := true }}) [addloop.mk_state]").unwrap();
    writeln!(out, "        }};").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}\n").unwrap();
}

fn prelude() -> &'static str {
    r#"//! GENERATED by kirgen — do not edit. Regenerate: cargo run --manifest-path ../kirgen/Cargo.toml
//!
//! Certificates for expression kernels (target spec_kernels::*) and loop
//! kernels (target the generated model_* recursive spec fns, which are the
//! kernels' canonical functional specs). Chain: source exec == spec == KIR.

use vstd::prelude::*;
use crate::kir::*;
use crate::addloop::mk_state;
use crate::addloop::{u_keval_loc, u_keval_addw, u_keval_ltu, u_keval_reada, u_keval_readb};
use crate::addloop::{u_kexec_assign, u_kexec_writeout, u_kloop_step, u_kloop_done};
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

    let abs_delta = Kernel {
        name: "abs_delta",
        params: vec!["a", "b", "n"],
        requires: vec!["1 <= n", "n <= 1024", "a < n", "b < n"],
        spec_call: "spec_abs_delta(a as int, b as int)",
        stmts: vec![Ite(Box::new(Ge(p(0), p(1))), Box::new(Sub(p(0), p(1))), Box::new(Sub(p(1), p(0))))],
    };

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

    // add_limbs, G1b: the probe's loop kernel, now generated.
    // carried: carry (local 0); counter: i (local 1); temps av..c2 (2..7).
    let c = |r| Box::new(Carried(r));
    let add_limbs = LoopKernel {
        name: "add_limbs",
        carried: vec!["carry"],
        width: 8,
        stmts: vec![
            LStmt::Temp(ReadA(Box::new(Counter))),            // av   (tv0, local 2)
            LStmt::Temp(ReadB(Box::new(Counter))),            // bv   (tv1, local 3)
            LStmt::Temp(Add(t(0), t(1))),                     // ab   (tv2, local 4)
            LStmt::Temp(Lt(t(2), t(0))),                      // c1   (tv3, local 5)
            LStmt::Temp(Add(t(2), c(0))),                     // abc  (tv4, local 6)
            LStmt::Temp(Lt(t(4), t(2))),                      // c2   (tv5, local 7)
            LStmt::WriteOut(Temp(4)),                         // out[i] := abc
            LStmt::SetCarried(0, Add(t(3), t(5))),            // carry := c1 + c2
        ],
    };

    let mut out = String::new();
    out.push_str(prelude());
    for k in [&abs_delta, &wrap_delta, &sum_sq3] {
        emit_kernel(k, &mut out);
    }
    emit_loop_kernel(&add_limbs, &mut out);
    out.push_str("} // verus!\n");

    let dest = concat!(env!("CARGO_MANIFEST_DIR"), "/../probe-g1/src/gen_certs.rs");
    std::fs::write(dest, &out).unwrap();
    println!("wrote {} ({} lines)", dest, out.lines().count());
}
