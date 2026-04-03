///  General correctness proof: gpu_eval == wgsl_semantics for all well-formed programs.
///
///  This is THE key theorem of the verified transpiler. Proved once by structural
///  induction, it covers all well-formed GpuIR programs — no per-kernel proofs needed.
///
///  The trusted claim: `wgsl_semantics` faithfully models what a conformant WGSL
///  implementation computes. The WGSL string rendering (~150 lines, trusted) renders
///  GpuIR to WGSL text matching the semantics described here.

use vstd::prelude::*;
use crate::gpu_ir::*;
use crate::wgsl_semantics::*;

verus! {

//  ══════════════════════════════════════════════════════════════
//  Expression equivalence — the core theorem
//  ══════════════════════════════════════════════════════════════

///  For all well-formed expressions: gpu_eval_expr == wgsl_semantics_expr.
///
///  Proof: structural induction on e.
///  Each case unfolds both definitions and shows they compute the same value.
///  Currently both specs are definitionally equal (they use the same helper
///  functions), so the proof is trivial — Verus sees they're the same.
pub proof fn lemma_expr_eval_eq_wgsl(e: &GpuExpr, state: &GpuState)
    ensures
        gpu_eval_expr(e, state) == wgsl_semantics_expr(e, state),
    decreases e,
{
    match e {
        GpuExpr::Const(_, _) => {},
        GpuExpr::FConst(_) => {},
        GpuExpr::Var(_, _) => {},
        GpuExpr::Builtin(_) => {},
        GpuExpr::BinOp(_, a, b) => {
            lemma_expr_eval_eq_wgsl(a, state);
            lemma_expr_eval_eq_wgsl(b, state);
        },
        GpuExpr::UnaryOp(_, a) => {
            lemma_expr_eval_eq_wgsl(a, state);
        },
        GpuExpr::Select(c, t, f) => {
            lemma_expr_eval_eq_wgsl(c, state);
            lemma_expr_eval_eq_wgsl(t, state);
            lemma_expr_eval_eq_wgsl(f, state);
        },
        GpuExpr::ArrayRead(_, idx) => {
            lemma_expr_eval_eq_wgsl(idx, state);
        },
        GpuExpr::TextureLoad(_, coords) => {
            lemma_expr_eval_eq_wgsl(coords, state);
        },
        GpuExpr::Cast(_, inner) => {
            lemma_expr_eval_eq_wgsl(inner, state);
        },
        GpuExpr::VecConstruct(components) => {
            if components.len() >= 2 {
                lemma_expr_eval_eq_wgsl(&components[0], state);
                lemma_expr_eval_eq_wgsl(&components[1], state);
            }
            if components.len() >= 3 {
                lemma_expr_eval_eq_wgsl(&components[2], state);
            }
            if components.len() >= 4 {
                lemma_expr_eval_eq_wgsl(&components[3], state);
            }
        },
        GpuExpr::VecComponent(vec_expr, _) => {
            lemma_expr_eval_eq_wgsl(vec_expr, state);
        },
        GpuExpr::Swizzle(vec_expr, _) => {
            lemma_expr_eval_eq_wgsl(vec_expr, state);
        },
        GpuExpr::MatConstruct(_, _, col_exprs) => {
            if col_exprs.len() >= 2 {
                lemma_expr_eval_eq_wgsl(&col_exprs[0], state);
                lemma_expr_eval_eq_wgsl(&col_exprs[1], state);
            }
            if col_exprs.len() >= 3 {
                lemma_expr_eval_eq_wgsl(&col_exprs[2], state);
            }
            if col_exprs.len() >= 4 {
                lemma_expr_eval_eq_wgsl(&col_exprs[3], state);
            }
        },
        GpuExpr::MatMul(a, b) => {
            lemma_expr_eval_eq_wgsl(a, state);
            lemma_expr_eval_eq_wgsl(b, state);
        },
        GpuExpr::Transpose(m) => {
            lemma_expr_eval_eq_wgsl(m, state);
        },
        GpuExpr::Determinant(m) => {
            lemma_expr_eval_eq_wgsl(m, state);
        },
        GpuExpr::Pack4x8(_, v) => {
            lemma_expr_eval_eq_wgsl(v, state);
        },
        GpuExpr::Unpack4x8(_, v) => {
            lemma_expr_eval_eq_wgsl(v, state);
        },
        GpuExpr::SubgroupReduce(_, v) => {
            lemma_expr_eval_eq_wgsl(v, state);
        },
        GpuExpr::SubgroupComm(_, v) => {
            lemma_expr_eval_eq_wgsl(v, state);
        },
        GpuExpr::SubgroupVote(_, v) => {
            lemma_expr_eval_eq_wgsl(v, state);
        },
    }
}

//  ══════════════════════════════════════════════════════════════
//  Statement equivalence — mutual induction with loop
//  ══════════════════════════════════════════════════════════════

///  For all well-formed statements: gpu_eval_stmt == wgsl_semantics_stmt.
pub proof fn lemma_stmt_eval_eq_wgsl(
    s: &GpuStmt, state: GpuState, fns: &Seq<GpuFunction>,
)
    ensures
        gpu_eval_stmt(s, state, fns) == wgsl_semantics_stmt(s, state, fns),
    decreases s, 0nat,
{
    if state.returned || state.broken { return; }
    match s {
        GpuStmt::Assign { var, rhs } => {
            lemma_expr_eval_eq_wgsl(rhs, &state);
        },
        GpuStmt::BufWrite { buf, idx, val } => {
            lemma_expr_eval_eq_wgsl(idx, &state);
            lemma_expr_eval_eq_wgsl(val, &state);
        },
        GpuStmt::TextureStore { tex, coords, val } => {
            lemma_expr_eval_eq_wgsl(coords, &state);
            lemma_expr_eval_eq_wgsl(val, &state);
        },
        GpuStmt::AtomicRMW { buf, idx, op: _, val, old_val_var } => {
            lemma_expr_eval_eq_wgsl(idx, &state);
            lemma_expr_eval_eq_wgsl(val, &state);
        },
        GpuStmt::CallStmt { fn_id, args, result_var } => {
            //  Prove arg evaluation agrees
            if args.len() >= 1 { lemma_expr_eval_eq_wgsl(&args[0], &state); }
            if args.len() >= 2 { lemma_expr_eval_eq_wgsl(&args[1], &state); }
            if args.len() >= 3 { lemma_expr_eval_eq_wgsl(&args[2], &state); }
            if args.len() >= 4 { lemma_expr_eval_eq_wgsl(&args[3], &state); }
        },
        GpuStmt::Seq { first, then } => {
            lemma_stmt_eval_eq_wgsl(first, state, fns);
            let mid = gpu_eval_stmt(first, state, fns);
            lemma_stmt_eval_eq_wgsl(then, mid, fns);
        },
        GpuStmt::If { cond, then_body, else_body } => {
            lemma_expr_eval_eq_wgsl(cond, &state);
            lemma_stmt_eval_eq_wgsl(then_body, state, fns);
            lemma_stmt_eval_eq_wgsl(else_body, state, fns);
        },
        GpuStmt::For { var, start, end, body } => {
            lemma_expr_eval_eq_wgsl(start, &state);
            lemma_expr_eval_eq_wgsl(end, &state);
            let s_val = gpu_value_to_int(&gpu_eval_expr(start, &state));
            let e_val = gpu_value_to_int(&gpu_eval_expr(end, &state));
            lemma_loop_eval_eq_wgsl(*var, s_val, e_val, body, state, fns);
        },
        GpuStmt::Break => {},
        GpuStmt::Continue => {},
        GpuStmt::Barrier { .. } => {},
        GpuStmt::Return => {},
        GpuStmt::Noop => {},
    }
}

///  For all loops: gpu_eval_loop == wgsl_semantics_loop.
pub proof fn lemma_loop_eval_eq_wgsl(
    var: nat, current: int, end: int,
    body: &GpuStmt, state: GpuState,
    fns: &Seq<GpuFunction>,
)
    ensures
        gpu_eval_loop(var, current, end, body, state, fns)
        == wgsl_semantics_loop(var, current, end, body, state, fns),
    decreases body, (if end > current { (end - current) as nat } else { 0nat }),
{
    if current >= end || state.returned || state.broken { return; }
    let s = if (var as int) < state.locals.len() {
        GpuState {
            locals: state.locals.update(var as int, GpuValue::Int(current)),
            ..state
        }
    } else { state };
    lemma_stmt_eval_eq_wgsl(body, s, fns);
    let s2 = gpu_eval_stmt(body, s, fns);
    lemma_loop_eval_eq_wgsl(var, current + 1, end, body, s2, fns);
}

} // verus!
