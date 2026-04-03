///  WGSL semantics — defines what WGSL computes for each GpuIR construct.
///
///  This spec is our minimal axiom: "this is what a conformant WGSL implementation
///  does." It's defined on GpuIR nodes (not on strings) so the correctness proof
///  is structural induction in Verus, not string parsing.
///
///  The general correctness theorem (in `emit_proof.rs`):
///    forall e. wgsl_semantics_expr(e, state) == gpu_eval_expr(e, state)
///    forall s. wgsl_semantics_stmt(s, state) == gpu_eval_stmt(s, state)
///
///  Currently these are definitionally equal (both model the same integer/float
///  semantics). They are kept as separate specs to:
///  1. Provide the "two independent authorities agree" structure
///  2. Allow future divergence where WGSL semantics differ (e.g., wrapping
///     is always-on in WGSL but opt-in in gpu_eval)
///  3. Make the trusted claim explicit and auditable

use vstd::prelude::*;
use crate::gpu_ir::*;

verus! {

//  ══════════════════════════════════════════════════════════════
//  WGSL expression semantics
//  ══════════════════════════════════════════════════════════════

///  WGSL semantics for a binary operation.
///  Currently identical to gpu_eval_binop — kept separate as the
///  "WGSL authority" for the correctness proof.
pub open spec fn wgsl_binop(op: &GpuBinOp, a: &GpuValue, b: &GpuValue) -> GpuValue {
    gpu_eval_binop(op, a, b)
}

///  WGSL semantics for a unary operation.
pub open spec fn wgsl_unaryop(op: &GpuUnaryOp, a: &GpuValue) -> GpuValue {
    gpu_eval_unaryop(op, a)
}

///  WGSL semantics for expression evaluation.
///  Structurally parallel to gpu_eval_expr.
pub open spec fn wgsl_semantics_expr(
    e: &GpuExpr, state: &GpuState,
) -> GpuValue
    decreases e,
{
    match e {
        GpuExpr::Const(c, _) => GpuValue::Int(*c),
        GpuExpr::FConst(f) => GpuValue::Float(*f),
        GpuExpr::Var(i, _) => {
            if (*i as int) < state.locals.len() {
                state.locals[*i as int]
            } else { GpuValue::Int(0) }
        },
        GpuExpr::Builtin(_) => GpuValue::Int(0),
        GpuExpr::BinOp(op, a, b) => {
            let va = wgsl_semantics_expr(a, state);
            let vb = wgsl_semantics_expr(b, state);
            wgsl_binop(op, &va, &vb)
        },
        GpuExpr::UnaryOp(op, a) => {
            let va = wgsl_semantics_expr(a, state);
            wgsl_unaryop(op, &va)
        },
        GpuExpr::Select(cond, t, f) => {
            if gpu_value_truthy(&wgsl_semantics_expr(cond, state)) {
                wgsl_semantics_expr(t, state)
            } else {
                wgsl_semantics_expr(f, state)
            }
        },
        GpuExpr::ArrayRead(buf_idx, idx_expr) => {
            let idx = gpu_value_to_int(&wgsl_semantics_expr(idx_expr, state));
            if (*buf_idx as int) < state.bufs.len()
                && 0 <= idx
                && idx < state.bufs[*buf_idx as int].len()
            {
                state.bufs[*buf_idx as int][idx]
            } else { GpuValue::Int(0) }
        },
        GpuExpr::TextureLoad(tex_idx, coords_expr) => {
            let coords = gpu_value_to_int(&wgsl_semantics_expr(coords_expr, state));
            if (*tex_idx as int) < state.bufs.len()
                && 0 <= coords
                && coords < state.bufs[*tex_idx as int].len()
            {
                state.bufs[*tex_idx as int][coords]
            } else { GpuValue::Int(0) }
        },
        GpuExpr::Cast(ty, inner) => {
            let v = wgsl_semantics_expr(inner, state);
            match ty {
                GpuType::Scalar(ScalarType::F32) =>
                    GpuValue::Float((gpu_value_to_int(&v) as i32) as f32),
                GpuType::Scalar(ScalarType::F16) =>
                    GpuValue::Float((gpu_value_to_int(&v) as i32) as f32),
                GpuType::Scalar(ScalarType::I32) =>
                    GpuValue::Int(gpu_value_to_int(&v)),
                GpuType::Scalar(ScalarType::U32) =>
                    GpuValue::Int(gpu_value_to_int(&v)),
                GpuType::Scalar(ScalarType::Bool) =>
                    GpuValue::Bool(gpu_value_truthy(&v)),
                _ => v,
            }
        },
        GpuExpr::VecConstruct(components) => {
            if components.len() == 2 {
                GpuValue::Vec(seq![
                    wgsl_semantics_expr(&components[0], state),
                    wgsl_semantics_expr(&components[1], state),
                ])
            } else if components.len() == 3 {
                GpuValue::Vec(seq![
                    wgsl_semantics_expr(&components[0], state),
                    wgsl_semantics_expr(&components[1], state),
                    wgsl_semantics_expr(&components[2], state),
                ])
            } else if components.len() == 4 {
                GpuValue::Vec(seq![
                    wgsl_semantics_expr(&components[0], state),
                    wgsl_semantics_expr(&components[1], state),
                    wgsl_semantics_expr(&components[2], state),
                    wgsl_semantics_expr(&components[3], state),
                ])
            } else { GpuValue::Vec(seq![]) }
        },
        GpuExpr::VecComponent(vec_expr, idx) => {
            gpu_value_vec_component(&wgsl_semantics_expr(vec_expr, state), *idx)
        },
        GpuExpr::Swizzle(vec_expr, indices) => {
            let v = wgsl_semantics_expr(vec_expr, state);
            GpuValue::Vec(Seq::new(indices.len(), |i: int|
                gpu_value_vec_component(&v, indices[i])))
        },
        GpuExpr::MatConstruct(_cols, _rows, col_exprs) => {
            if col_exprs.len() == 2 {
                let c0 = wgsl_semantics_expr(&col_exprs[0], state);
                let c1 = wgsl_semantics_expr(&col_exprs[1], state);
                GpuValue::Mat(seq![
                    match c0 { GpuValue::Vec(v) => v, _ => seq![c0] },
                    match c1 { GpuValue::Vec(v) => v, _ => seq![c1] },
                ])
            } else if col_exprs.len() == 3 {
                let c0 = wgsl_semantics_expr(&col_exprs[0], state);
                let c1 = wgsl_semantics_expr(&col_exprs[1], state);
                let c2 = wgsl_semantics_expr(&col_exprs[2], state);
                GpuValue::Mat(seq![
                    match c0 { GpuValue::Vec(v) => v, _ => seq![c0] },
                    match c1 { GpuValue::Vec(v) => v, _ => seq![c1] },
                    match c2 { GpuValue::Vec(v) => v, _ => seq![c2] },
                ])
            } else if col_exprs.len() == 4 {
                let c0 = wgsl_semantics_expr(&col_exprs[0], state);
                let c1 = wgsl_semantics_expr(&col_exprs[1], state);
                let c2 = wgsl_semantics_expr(&col_exprs[2], state);
                let c3 = wgsl_semantics_expr(&col_exprs[3], state);
                GpuValue::Mat(seq![
                    match c0 { GpuValue::Vec(v) => v, _ => seq![c0] },
                    match c1 { GpuValue::Vec(v) => v, _ => seq![c1] },
                    match c2 { GpuValue::Vec(v) => v, _ => seq![c2] },
                    match c3 { GpuValue::Vec(v) => v, _ => seq![c3] },
                ])
            } else { GpuValue::Mat(seq![]) }
        },
        GpuExpr::MatMul(a_expr, b_expr) => {
            let _a = wgsl_semantics_expr(a_expr, state);
            let _b = wgsl_semantics_expr(b_expr, state);
            GpuValue::Int(0)  //  TODO: mat mul spec
        },
        GpuExpr::Transpose(m_expr) => {
            let m = wgsl_semantics_expr(m_expr, state);
            match m {
                GpuValue::Mat(cols) => {
                    if cols.len() > 0 && cols[0].len() > 0 {
                        let n_rows = cols[0].len();
                        let n_cols = cols.len();
                        GpuValue::Mat(Seq::new(n_rows, |r: int|
                            Seq::new(n_cols, |c: int| cols[c][r])))
                    } else { GpuValue::Mat(seq![]) }
                },
                _ => m,
            }
        },
        GpuExpr::Determinant(_m_expr) => {
            let _m = wgsl_semantics_expr(_m_expr, state);
            GpuValue::Float(0.0f32)  //  TODO: determinant spec
        },
        GpuExpr::Pack4x8(_fmt, vec_expr) => {
            let _v = wgsl_semantics_expr(vec_expr, state);
            GpuValue::Int(0)  //  TODO: pack spec
        },
        GpuExpr::Unpack4x8(_fmt, int_expr) => {
            let _v = wgsl_semantics_expr(int_expr, state);
            GpuValue::Vec(seq![
                GpuValue::Float(0.0f32), GpuValue::Float(0.0f32),
                GpuValue::Float(0.0f32), GpuValue::Float(0.0f32),
            ])
        },
        GpuExpr::SubgroupReduce(_op, val_expr) => {
            let _v = wgsl_semantics_expr(val_expr, state);
            GpuValue::Int(0)
        },
        GpuExpr::SubgroupComm(_op, val_expr) => {
            let _v = wgsl_semantics_expr(val_expr, state);
            GpuValue::Int(0)
        },
        GpuExpr::SubgroupVote(_op, val_expr) => {
            let _v = wgsl_semantics_expr(val_expr, state);
            GpuValue::Bool(false)
        },
    }
}

//  ══════════════════════════════════════════════════════════════
//  WGSL statement semantics
//  ══════════════════════════════════════════════════════════════

///  WGSL semantics for statement evaluation.
///  Structurally parallel to gpu_eval_stmt.
pub open spec fn wgsl_semantics_stmt(
    s: &GpuStmt, state: GpuState, fns: &Seq<GpuFunction>,
) -> GpuState
    decreases s, 0nat,
{
    if state.returned || state.broken { state }
    else {
        match s {
            GpuStmt::Assign { var, rhs } => {
                let v = wgsl_semantics_expr(rhs, &state);
                if (*var as int) < state.locals.len() {
                    GpuState { locals: state.locals.update(*var as int, v), ..state }
                } else { state }
            },
            GpuStmt::BufWrite { buf, idx, val } => {
                let i = gpu_value_to_int(&wgsl_semantics_expr(idx, &state));
                let v = wgsl_semantics_expr(val, &state);
                if (*buf as int) < state.bufs.len()
                    && 0 <= i && i < state.bufs[*buf as int].len()
                {
                    GpuState {
                        bufs: state.bufs.update(*buf as int,
                            state.bufs[*buf as int].update(i, v)),
                        ..state
                    }
                } else { state }
            },
            GpuStmt::TextureStore { tex, coords, val } => {
                let c = gpu_value_to_int(&wgsl_semantics_expr(coords, &state));
                let v = wgsl_semantics_expr(val, &state);
                if (*tex as int) < state.bufs.len()
                    && 0 <= c && c < state.bufs[*tex as int].len()
                {
                    GpuState {
                        bufs: state.bufs.update(*tex as int,
                            state.bufs[*tex as int].update(c, v)),
                        ..state
                    }
                } else { state }
            },
            GpuStmt::AtomicRMW { buf, idx, op: _, val, old_val_var } => {
                let i = gpu_value_to_int(&wgsl_semantics_expr(idx, &state));
                let new_val = wgsl_semantics_expr(val, &state);
                if (*buf as int) < state.bufs.len()
                    && 0 <= i && i < state.bufs[*buf as int].len()
                {
                    let old_val = state.bufs[*buf as int][i];
                    let state_with_result = match old_val_var {
                        Option::Some(rv) => {
                            if (*rv as int) < state.locals.len() {
                                GpuState {
                                    locals: state.locals.update(*rv as int, old_val),
                                    ..state
                                }
                            } else { state }
                        },
                        Option::None => state,
                    };
                    GpuState {
                        bufs: state_with_result.bufs.update(*buf as int,
                            state_with_result.bufs[*buf as int].update(i, new_val)),
                        ..state_with_result
                    }
                } else { state }
            },
            GpuStmt::CallStmt { fn_id, args, result_var } => {
                if (*fn_id as int) < fns.len() {
                    let f = &fns[*fn_id as int];
                    let arg_vals: Seq<GpuValue> =
                        if args.len() == 0 { seq![] }
                        else if args.len() == 1 { seq![wgsl_semantics_expr(&args[0], &state)] }
                        else if args.len() == 2 { seq![wgsl_semantics_expr(&args[0], &state),
                            wgsl_semantics_expr(&args[1], &state)] }
                        else if args.len() == 3 { seq![wgsl_semantics_expr(&args[0], &state),
                            wgsl_semantics_expr(&args[1], &state),
                            wgsl_semantics_expr(&args[2], &state)] }
                        else if args.len() == 4 { seq![wgsl_semantics_expr(&args[0], &state),
                            wgsl_semantics_expr(&args[1], &state),
                            wgsl_semantics_expr(&args[2], &state),
                            wgsl_semantics_expr(&args[3], &state)] }
                        else { seq![] };
                    let fn_locals = Seq::new(f.n_locals, |i: int|
                        if i < arg_vals.len() { arg_vals[i] }
                        else { GpuValue::Int(0) });
                    let fn_state = GpuState {
                        locals: fn_locals, bufs: state.bufs,
                        returned: false, broken: false,
                    };
                    let result_state = fn_state;  //  placeholder
                    let ret_val = if (f.ret_var as int) < result_state.locals.len() {
                        result_state.locals[f.ret_var as int]
                    } else { GpuValue::Int(0) };
                    if (*result_var as int) < state.locals.len() {
                        GpuState {
                            locals: state.locals.update(*result_var as int, ret_val),
                            ..state
                        }
                    } else { state }
                } else { state }
            },
            GpuStmt::Seq { first, then } => {
                let mid = wgsl_semantics_stmt(first, state, fns);
                wgsl_semantics_stmt(then, mid, fns)
            },
            GpuStmt::If { cond, then_body, else_body } => {
                if gpu_value_truthy(&wgsl_semantics_expr(cond, &state)) {
                    wgsl_semantics_stmt(then_body, state, fns)
                } else {
                    wgsl_semantics_stmt(else_body, state, fns)
                }
            },
            GpuStmt::For { var, start, end, body } => {
                let s_val = gpu_value_to_int(&wgsl_semantics_expr(start, &state));
                let e_val = gpu_value_to_int(&wgsl_semantics_expr(end, &state));
                let result = wgsl_semantics_loop(*var, s_val, e_val, body, state, fns);
                GpuState { broken: false, ..result }
            },
            GpuStmt::Break => GpuState { broken: true, ..state },
            GpuStmt::Continue => state,
            GpuStmt::Barrier { .. } => state,
            GpuStmt::Return => GpuState { returned: true, ..state },
            GpuStmt::Noop => state,
        }
    }
}

///  WGSL semantics for loop evaluation.
pub open spec fn wgsl_semantics_loop(
    var: nat, current: int, end: int,
    body: &GpuStmt, state: GpuState,
    fns: &Seq<GpuFunction>,
) -> GpuState
    decreases body, (if end > current { (end - current) as nat } else { 0nat }),
{
    if current >= end || state.returned || state.broken {
        state
    } else {
        let s = if (var as int) < state.locals.len() {
            GpuState {
                locals: state.locals.update(var as int, GpuValue::Int(current)),
                ..state
            }
        } else { state };
        let s2 = wgsl_semantics_stmt(body, s, fns);
        wgsl_semantics_loop(var, current + 1, end, body, s2, fns)
    }
}

} // verus!
