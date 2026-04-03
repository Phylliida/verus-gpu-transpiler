///  General correctness proof: gpu_eval == wgsl_semantics.
///
///  Both specs are now INDEPENDENTLY defined (wgsl_semantics does NOT delegate
///  to gpu_eval helpers). The proof checks structural correspondence case by case.

use vstd::prelude::*;
use crate::gpu_ir::*;
use crate::wgsl_semantics::*;

verus! {

//  ══════════════════════════════════════════════════════════════
//  Helper: binop and unaryop agree
//  ══════════════════════════════════════════════════════════════

proof fn lemma_binop_agree(op: &GpuBinOp, a: &GpuValue, b: &GpuValue)
    ensures gpu_eval_binop(op, a, b) == wgsl_binop(op, a, b)
{}

proof fn lemma_unaryop_agree(op: &GpuUnaryOp, a: &GpuValue)
    ensures gpu_eval_unaryop(op, a) == wgsl_unaryop(op, a)
{}

proof fn lemma_atomic_op_agree(op: &AtomicOp, old: &GpuValue, operand: &GpuValue)
    ensures gpu_eval_atomic_op(op, old, operand) == wgsl_atomic_op(op, old, operand)
{}

//  ══════════════════════════════════════════════════════════════
//  Expression equivalence
//  ══════════════════════════════════════════════════════════════

pub proof fn lemma_expr_eval_eq_wgsl(e: &GpuExpr, state: &GpuState)
    ensures gpu_eval_expr(e, state) == wgsl_semantics_expr(e, state)
    decreases e
{
    match e {
        GpuExpr::Const(_, _) => {},
        GpuExpr::FConst(_) => {},
        GpuExpr::Var(_, _) => {},
        GpuExpr::Builtin { .. } => {},
        GpuExpr::BinOp(op, a, b) => {
            lemma_expr_eval_eq_wgsl(a, state);
            lemma_expr_eval_eq_wgsl(b, state);
            lemma_binop_agree(op, &gpu_eval_expr(a, state), &gpu_eval_expr(b, state));
        },
        GpuExpr::UnaryOp(op, a) => {
            lemma_expr_eval_eq_wgsl(a, state);
            lemma_unaryop_agree(op, &gpu_eval_expr(a, state));
        },
        GpuExpr::Select(c, t, f) => {
            lemma_expr_eval_eq_wgsl(c, state);
            lemma_expr_eval_eq_wgsl(t, state);
            lemma_expr_eval_eq_wgsl(f, state);
        },
        GpuExpr::ArrayRead(_, idx) => { lemma_expr_eval_eq_wgsl(idx, state); },
        GpuExpr::TextureLoad(_, coords) => { lemma_expr_eval_eq_wgsl(coords, state); },
        GpuExpr::Cast(_, inner) => { lemma_expr_eval_eq_wgsl(inner, state); },
        GpuExpr::VecConstruct(components) => {
            if components.len() >= 2 {
                lemma_expr_eval_eq_wgsl(&components[0], state);
                lemma_expr_eval_eq_wgsl(&components[1], state);
            }
            if components.len() >= 3 { lemma_expr_eval_eq_wgsl(&components[2], state); }
            if components.len() >= 4 { lemma_expr_eval_eq_wgsl(&components[3], state); }
        },
        GpuExpr::VecComponent(v, _) => { lemma_expr_eval_eq_wgsl(v, state); },
        GpuExpr::Swizzle(v, _) => { lemma_expr_eval_eq_wgsl(v, state); },
        GpuExpr::MatConstruct(_, _, col_exprs) => {
            if col_exprs.len() >= 2 {
                lemma_expr_eval_eq_wgsl(&col_exprs[0], state);
                lemma_expr_eval_eq_wgsl(&col_exprs[1], state);
            }
            if col_exprs.len() >= 3 { lemma_expr_eval_eq_wgsl(&col_exprs[2], state); }
            if col_exprs.len() >= 4 { lemma_expr_eval_eq_wgsl(&col_exprs[3], state); }
        },
        GpuExpr::MatMul(a, b) => {
            lemma_expr_eval_eq_wgsl(a, state);
            lemma_expr_eval_eq_wgsl(b, state);
        },
        GpuExpr::Transpose(m) => { lemma_expr_eval_eq_wgsl(m, state); },
        GpuExpr::Determinant(m) => { lemma_expr_eval_eq_wgsl(m, state); },
        GpuExpr::Pack4x8(_, v) => { lemma_expr_eval_eq_wgsl(v, state); },
        GpuExpr::Unpack4x8(_, v) => { lemma_expr_eval_eq_wgsl(v, state); },
        GpuExpr::SubgroupReduce(_, v) => { lemma_expr_eval_eq_wgsl(v, state); },
        GpuExpr::SubgroupComm(_, v) => { lemma_expr_eval_eq_wgsl(v, state); },
        GpuExpr::SubgroupVote(_, v) => { lemma_expr_eval_eq_wgsl(v, state); },
    }
}

//  ══════════════════════════════════════════════════════════════
//  Statement equivalence (mutual with loop)
//  ══════════════════════════════════════════════════════════════

pub proof fn lemma_stmt_eval_eq_wgsl(
    s: &GpuStmt, state: GpuState, fns: &Seq<GpuFunction>, fuel: nat,
)
    ensures gpu_eval_stmt(s, state, fns, fuel) == wgsl_semantics_stmt(s, state, fns, fuel)
    decreases fuel, s, 0nat,
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
        GpuStmt::AtomicRMW { buf, idx, op: atomic_op, val, old_val_var } => {
            lemma_expr_eval_eq_wgsl(idx, &state);
            lemma_expr_eval_eq_wgsl(val, &state);
            //  Prove atomic op agrees
            let i = gpu_value_to_int(&gpu_eval_expr(idx, &state));
            if (*buf as int) < state.bufs.len() && 0 <= i && i < state.bufs[*buf as int].len() {
                let old_val = state.bufs[*buf as int][i];
                let operand = gpu_eval_expr(val, &state);
                lemma_atomic_op_agree(atomic_op, &old_val, &operand);
            }
        },
        GpuStmt::CallStmt { fn_id, args, result_var } => {
            if fuel == 0 || !((*fn_id as int) < fns.len()) { return; }
            let f = &fns[*fn_id as int];
            //  Prove args agree
            if args.len() >= 1 { lemma_expr_eval_eq_wgsl(&args[0], &state); }
            if args.len() >= 2 { lemma_expr_eval_eq_wgsl(&args[1], &state); }
            if args.len() >= 3 { lemma_expr_eval_eq_wgsl(&args[2], &state); }
            if args.len() >= 4 { lemma_expr_eval_eq_wgsl(&args[3], &state); }
            if args.len() >= 5 { lemma_expr_eval_eq_wgsl(&args[4], &state); }
            if args.len() >= 6 { lemma_expr_eval_eq_wgsl(&args[5], &state); }
            if args.len() >= 7 { lemma_expr_eval_eq_wgsl(&args[6], &state); }
            if args.len() >= 8 { lemma_expr_eval_eq_wgsl(&args[7], &state); }
            //  Both sides build arg_vals the same way (proved equal above),
            //  so fn_locals and fn_state are identical. IH on body at fuel-1.
            //  Construct the fn_state that both sides would build:
            let arg_vals: Seq<GpuValue> =
                if args.len() == 0 { seq![] }
                else if args.len() == 1 { seq![gpu_eval_expr(&args[0], &state)] }
                else if args.len() == 2 { seq![gpu_eval_expr(&args[0], &state),
                    gpu_eval_expr(&args[1], &state)] }
                else if args.len() == 3 { seq![gpu_eval_expr(&args[0], &state),
                    gpu_eval_expr(&args[1], &state), gpu_eval_expr(&args[2], &state)] }
                else if args.len() == 4 { seq![gpu_eval_expr(&args[0], &state),
                    gpu_eval_expr(&args[1], &state), gpu_eval_expr(&args[2], &state),
                    gpu_eval_expr(&args[3], &state)] }
                else if args.len() == 5 { seq![gpu_eval_expr(&args[0], &state),
                    gpu_eval_expr(&args[1], &state), gpu_eval_expr(&args[2], &state),
                    gpu_eval_expr(&args[3], &state), gpu_eval_expr(&args[4], &state)] }
                else if args.len() == 6 { seq![gpu_eval_expr(&args[0], &state),
                    gpu_eval_expr(&args[1], &state), gpu_eval_expr(&args[2], &state),
                    gpu_eval_expr(&args[3], &state), gpu_eval_expr(&args[4], &state),
                    gpu_eval_expr(&args[5], &state)] }
                else if args.len() == 7 { seq![gpu_eval_expr(&args[0], &state),
                    gpu_eval_expr(&args[1], &state), gpu_eval_expr(&args[2], &state),
                    gpu_eval_expr(&args[3], &state), gpu_eval_expr(&args[4], &state),
                    gpu_eval_expr(&args[5], &state), gpu_eval_expr(&args[6], &state)] }
                else if args.len() == 8 { seq![gpu_eval_expr(&args[0], &state),
                    gpu_eval_expr(&args[1], &state), gpu_eval_expr(&args[2], &state),
                    gpu_eval_expr(&args[3], &state), gpu_eval_expr(&args[4], &state),
                    gpu_eval_expr(&args[5], &state), gpu_eval_expr(&args[6], &state),
                    gpu_eval_expr(&args[7], &state)] }
                else { seq![] };
            let fn_locals = Seq::new(f.n_locals, |i: int|
                if i < arg_vals.len() { arg_vals[i] } else { GpuValue::Int(0) });
            let fn_state = GpuState {
                locals: fn_locals, bufs: state.bufs,
                returned: false, broken: false };
            lemma_stmt_eval_eq_wgsl(&f.body, fn_state, fns, (fuel - 1) as nat);
        },
        GpuStmt::Seq { first, then } => {
            lemma_stmt_eval_eq_wgsl(first, state, fns, fuel);
            let mid = gpu_eval_stmt(first, state, fns, fuel);
            lemma_stmt_eval_eq_wgsl(then, mid, fns, fuel);
        },
        GpuStmt::If { cond, then_body, else_body } => {
            lemma_expr_eval_eq_wgsl(cond, &state);
            lemma_stmt_eval_eq_wgsl(then_body, state, fns, fuel);
            lemma_stmt_eval_eq_wgsl(else_body, state, fns, fuel);
        },
        GpuStmt::For { var, start, end, body } => {
            lemma_expr_eval_eq_wgsl(start, &state);
            lemma_expr_eval_eq_wgsl(end, &state);
            let s_val = gpu_value_to_int(&gpu_eval_expr(start, &state));
            let e_val = gpu_value_to_int(&gpu_eval_expr(end, &state));
            lemma_loop_eval_eq_wgsl(*var, s_val, e_val, body, state, fns, fuel);
        },
        GpuStmt::Break => {},
        GpuStmt::Continue => {},
        GpuStmt::Barrier { .. } => {},
        GpuStmt::Return => {},
        GpuStmt::Noop => {},
    }
}

pub proof fn lemma_loop_eval_eq_wgsl(
    var: nat, current: int, end: int,
    body: &GpuStmt, state: GpuState,
    fns: &Seq<GpuFunction>, fuel: nat,
)
    ensures gpu_eval_loop(var, current, end, body, state, fns, fuel)
         == wgsl_semantics_loop(var, current, end, body, state, fns, fuel)
    decreases fuel, body, (if end > current { (end - current) as nat } else { 0nat }),
{
    if current >= end || state.returned || state.broken { return; }
    let s = if (var as int) < state.locals.len() {
        GpuState { locals: state.locals.update(var as int, GpuValue::Int(current)), ..state }
    } else { state };
    lemma_stmt_eval_eq_wgsl(body, s, fns, fuel);
    let s2 = gpu_eval_stmt(body, s, fns, fuel);
    lemma_loop_eval_eq_wgsl(var, current + 1, end, body, s2, fns, fuel);
}

} // verus!
