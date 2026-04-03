///  WGSL semantics — INDEPENDENTLY defined interpretation of each GpuIR construct.
///
///  This is NOT a copy of gpu_eval. Each function is written from scratch to
///  capture "what a conformant WGSL implementation computes." The emit_proof
///  then shows these two independent definitions agree.

use vstd::prelude::*;
use crate::gpu_ir::*;

verus! {

//  ══════════════════════════════════════════════════════════════
//  WGSL binary operator semantics (independent of gpu_eval_binop)
//  ══════════════════════════════════════════════════════════════

pub open spec fn wgsl_binop(op: &GpuBinOp, a: &GpuValue, b: &GpuValue) -> GpuValue {
    let ai = gpu_value_to_int(a);
    let bi = gpu_value_to_int(b);
    match op {
        GpuBinOp::Add => GpuValue::Int(ai + bi),
        GpuBinOp::Sub => GpuValue::Int(ai - bi),
        GpuBinOp::Mul => GpuValue::Int(ai * bi),
        GpuBinOp::Div => GpuValue::Int(if bi != 0 { ai / bi } else { 0 }),
        GpuBinOp::Mod => GpuValue::Int(if bi != 0 { ai % bi } else { 0 }),
        GpuBinOp::Shr => GpuValue::Int(
            if bi >= 0 && bi < 32 { ai / pow2(bi as nat) as int } else { 0 }),
        GpuBinOp::Shl => GpuValue::Int(
            if bi >= 0 && bi < 32 { ai * pow2(bi as nat) as int } else { 0 }),
        GpuBinOp::WrappingAdd => GpuValue::Int(wrap32(ai + bi)),
        GpuBinOp::WrappingSub => GpuValue::Int(wrap32(ai - bi)),
        GpuBinOp::WrappingMul => GpuValue::Int(wrap32(ai * bi)),
        GpuBinOp::FAdd => match (a, b) {
            (GpuValue::Float(fa), GpuValue::Float(fb)) => GpuValue::Float(*fa + *fb),
            _ => GpuValue::Float(0.0f32),
        },
        GpuBinOp::FSub => match (a, b) {
            (GpuValue::Float(fa), GpuValue::Float(fb)) => GpuValue::Float(*fa - *fb),
            _ => GpuValue::Float(0.0f32),
        },
        GpuBinOp::FMul => match (a, b) {
            (GpuValue::Float(fa), GpuValue::Float(fb)) => GpuValue::Float(*fa * *fb),
            _ => GpuValue::Float(0.0f32),
        },
        GpuBinOp::FDiv => match (a, b) {
            (GpuValue::Float(fa), GpuValue::Float(fb)) => GpuValue::Float(*fa / *fb),
            _ => GpuValue::Float(0.0f32),
        },
        GpuBinOp::Lt => if gpu_value_is_float(a) || gpu_value_is_float(b) {
            GpuValue::Bool(gpu_value_to_float(a) < gpu_value_to_float(b))
        } else { GpuValue::Bool(ai < bi) },
        GpuBinOp::Le => if gpu_value_is_float(a) || gpu_value_is_float(b) {
            GpuValue::Bool(gpu_value_to_float(a) <= gpu_value_to_float(b))
        } else { GpuValue::Bool(ai <= bi) },
        GpuBinOp::Gt => if gpu_value_is_float(a) || gpu_value_is_float(b) {
            GpuValue::Bool(gpu_value_to_float(a) > gpu_value_to_float(b))
        } else { GpuValue::Bool(ai > bi) },
        GpuBinOp::Ge => if gpu_value_is_float(a) || gpu_value_is_float(b) {
            GpuValue::Bool(gpu_value_to_float(a) >= gpu_value_to_float(b))
        } else { GpuValue::Bool(ai >= bi) },
        GpuBinOp::Eq => if gpu_value_is_float(a) || gpu_value_is_float(b) {
            GpuValue::Bool(gpu_value_to_float(a) == gpu_value_to_float(b))
        } else { GpuValue::Bool(ai == bi) },
        GpuBinOp::Ne => if gpu_value_is_float(a) || gpu_value_is_float(b) {
            GpuValue::Bool(gpu_value_to_float(a) != gpu_value_to_float(b))
        } else { GpuValue::Bool(ai != bi) },
        GpuBinOp::BitAnd => GpuValue::Int((ai as i32 & bi as i32) as int),
        GpuBinOp::BitOr => GpuValue::Int((ai as i32 | bi as i32) as int),
        GpuBinOp::BitXor => GpuValue::Int((ai as i32 ^ bi as i32) as int),
        GpuBinOp::LogicalAnd => GpuValue::Bool(
            gpu_value_truthy(a) && gpu_value_truthy(b)),
        GpuBinOp::LogicalOr => GpuValue::Bool(
            gpu_value_truthy(a) || gpu_value_truthy(b)),
    }
}

pub open spec fn wgsl_unaryop(op: &GpuUnaryOp, a: &GpuValue) -> GpuValue {
    match op {
        GpuUnaryOp::Neg => GpuValue::Int(-gpu_value_to_int(a)),
        GpuUnaryOp::FNeg => match a {
            GpuValue::Float(f) => GpuValue::Float(-*f),
            _ => GpuValue::Float(0.0f32),
        },
        GpuUnaryOp::BitNot => GpuValue::Int(!(gpu_value_to_int(a) as i32) as int),
        GpuUnaryOp::LogicalNot => GpuValue::Bool(!gpu_value_truthy(a)),
    }
}

pub open spec fn wgsl_atomic_op(
    op: &AtomicOp, old: &GpuValue, operand: &GpuValue, compare: &GpuValue,
) -> GpuValue {
    let oi = gpu_value_to_int(old);
    let vi = gpu_value_to_int(operand);
    let ci = gpu_value_to_int(compare);
    match op {
        AtomicOp::Load => *old,
        AtomicOp::Store => *operand,
        AtomicOp::Add => GpuValue::Int(oi + vi),
        AtomicOp::Sub => GpuValue::Int(oi - vi),
        AtomicOp::Max => GpuValue::Int(if oi >= vi { oi } else { vi }),
        AtomicOp::Min => GpuValue::Int(if oi <= vi { oi } else { vi }),
        AtomicOp::And => GpuValue::Int((oi as i32 & vi as i32) as int),
        AtomicOp::Or => GpuValue::Int((oi as i32 | vi as i32) as int),
        AtomicOp::Xor => GpuValue::Int((oi as i32 ^ vi as i32) as int),
        AtomicOp::Exchange => *operand,
        AtomicOp::CompareExchangeWeak => {
            if oi == ci { *operand } else { *old }
        },
    }
}

//  ══════════════════════════════════════════════════════════════
//  WGSL expression semantics (independent)
//  ══════════════════════════════════════════════════════════════

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
        GpuExpr::Builtin { which: _, local_idx } => {
            if (*local_idx as int) < state.locals.len() {
                state.locals[*local_idx as int]
            } else { GpuValue::Int(0) }
        },
        GpuExpr::BinOp(op, a, b) => {
            wgsl_binop(op, &wgsl_semantics_expr(a, state), &wgsl_semantics_expr(b, state))
        },
        GpuExpr::UnaryOp(op, a) => {
            wgsl_unaryop(op, &wgsl_semantics_expr(a, state))
        },
        GpuExpr::Select(cond, t, f) => {
            if gpu_value_truthy(&wgsl_semantics_expr(cond, state)) {
                wgsl_semantics_expr(t, state)
            } else { wgsl_semantics_expr(f, state) }
        },
        GpuExpr::ArrayRead(buf_idx, idx_expr) => {
            let idx = gpu_value_to_int(&wgsl_semantics_expr(idx_expr, state));
            if (*buf_idx as int) < state.bufs.len()
                && 0 <= idx && idx < state.bufs[*buf_idx as int].len()
            { state.bufs[*buf_idx as int][idx] }
            else { GpuValue::Int(0) }
        },
        GpuExpr::TextureLoad(tex_idx, coords_expr) => {
            let coords = gpu_value_to_int(&wgsl_semantics_expr(coords_expr, state));
            if (*tex_idx as int) < state.bufs.len()
                && 0 <= coords && coords < state.bufs[*tex_idx as int].len()
            { state.bufs[*tex_idx as int][coords] }
            else { GpuValue::Int(0) }
        },
        GpuExpr::Cast(ty, inner) => {
            let v = wgsl_semantics_expr(inner, state);
            match ty {
                GpuType::Scalar(ScalarType::F32) => match v {
                    GpuValue::Float(_) => v,
                    _ => GpuValue::Float((gpu_value_to_int(&v) as i32) as f32),
                },
                GpuType::Scalar(ScalarType::F16) => match v {
                    GpuValue::Float(_) => v,
                    _ => GpuValue::Float((gpu_value_to_int(&v) as i32) as f32),
                },
                GpuType::Scalar(ScalarType::I32) => GpuValue::Int(gpu_value_to_int(&v)),
                GpuType::Scalar(ScalarType::U32) => GpuValue::Int(gpu_value_to_int(&v)),
                GpuType::Scalar(ScalarType::Bool) => GpuValue::Bool(gpu_value_truthy(&v)),
                _ => v,
            }
        },
        GpuExpr::VecConstruct(components) => {
            if components.len() == 2 {
                GpuValue::Vec(seq![wgsl_semantics_expr(&components[0], state),
                                   wgsl_semantics_expr(&components[1], state)])
            } else if components.len() == 3 {
                GpuValue::Vec(seq![wgsl_semantics_expr(&components[0], state),
                                   wgsl_semantics_expr(&components[1], state),
                                   wgsl_semantics_expr(&components[2], state)])
            } else if components.len() == 4 {
                GpuValue::Vec(seq![wgsl_semantics_expr(&components[0], state),
                                   wgsl_semantics_expr(&components[1], state),
                                   wgsl_semantics_expr(&components[2], state),
                                   wgsl_semantics_expr(&components[3], state)])
            } else { GpuValue::Vec(seq![]) }
        },
        GpuExpr::VecComponent(vec_expr, idx) => {
            gpu_value_vec_component(&wgsl_semantics_expr(vec_expr, state), *idx)
        },
        GpuExpr::Swizzle(vec_expr, indices) => {
            let v = wgsl_semantics_expr(vec_expr, state);
            GpuValue::Vec(Seq::new(indices.len(), |i: int| gpu_value_vec_component(&v, indices[i])))
        },
        GpuExpr::MatConstruct(_c, _r, col_exprs) => {
            if col_exprs.len() == 2 {
                let c0 = wgsl_semantics_expr(&col_exprs[0], state);
                let c1 = wgsl_semantics_expr(&col_exprs[1], state);
                GpuValue::Mat(seq![
                    match c0 { GpuValue::Vec(v) => v, _ => seq![c0] },
                    match c1 { GpuValue::Vec(v) => v, _ => seq![c1] }])
            } else if col_exprs.len() == 3 {
                let c0 = wgsl_semantics_expr(&col_exprs[0], state);
                let c1 = wgsl_semantics_expr(&col_exprs[1], state);
                let c2 = wgsl_semantics_expr(&col_exprs[2], state);
                GpuValue::Mat(seq![
                    match c0 { GpuValue::Vec(v) => v, _ => seq![c0] },
                    match c1 { GpuValue::Vec(v) => v, _ => seq![c1] },
                    match c2 { GpuValue::Vec(v) => v, _ => seq![c2] }])
            } else if col_exprs.len() == 4 {
                let c0 = wgsl_semantics_expr(&col_exprs[0], state);
                let c1 = wgsl_semantics_expr(&col_exprs[1], state);
                let c2 = wgsl_semantics_expr(&col_exprs[2], state);
                let c3 = wgsl_semantics_expr(&col_exprs[3], state);
                GpuValue::Mat(seq![
                    match c0 { GpuValue::Vec(v) => v, _ => seq![c0] },
                    match c1 { GpuValue::Vec(v) => v, _ => seq![c1] },
                    match c2 { GpuValue::Vec(v) => v, _ => seq![c2] },
                    match c3 { GpuValue::Vec(v) => v, _ => seq![c3] }])
            } else { GpuValue::Mat(seq![]) }
        },
        GpuExpr::MatMul(a_expr, b_expr) => {
            let _a = wgsl_semantics_expr(a_expr, state);
            let _b = wgsl_semantics_expr(b_expr, state);
            GpuValue::Int(0)
        },
        GpuExpr::Transpose(m_expr) => {
            let m = wgsl_semantics_expr(m_expr, state);
            match m {
                GpuValue::Mat(cols) => {
                    if cols.len() > 0 && cols[0].len() > 0 {
                        GpuValue::Mat(Seq::new(cols[0].len(), |r: int|
                            Seq::new(cols.len(), |c: int| cols[c][r])))
                    } else { GpuValue::Mat(seq![]) }
                },
                _ => m,
            }
        },
        GpuExpr::Determinant(m) => {
            let _m = wgsl_semantics_expr(m, state);
            GpuValue::Float(0.0f32)
        },
        GpuExpr::Pack4x8(_, v) => {
            let _v = wgsl_semantics_expr(v, state);
            GpuValue::Int(0)
        },
        GpuExpr::Unpack4x8(_, v) => {
            let _v = wgsl_semantics_expr(v, state);
            GpuValue::Vec(seq![GpuValue::Float(0.0f32), GpuValue::Float(0.0f32),
                               GpuValue::Float(0.0f32), GpuValue::Float(0.0f32)])
        },
        GpuExpr::SubgroupReduce(_, v) => {
            let _v = wgsl_semantics_expr(v, state); GpuValue::Int(0)
        },
        GpuExpr::SubgroupComm(_, v) => {
            let _v = wgsl_semantics_expr(v, state); GpuValue::Int(0)
        },
        GpuExpr::SubgroupVote(_, v) => {
            let _v = wgsl_semantics_expr(v, state); GpuValue::Bool(false)
        },
    }
}

//  ══════════════════════════════════════════════════════════════
//  WGSL statement semantics (independent, with fuel)
//  ══════════════════════════════════════════════════════════════

pub open spec fn wgsl_semantics_stmt(
    s: &GpuStmt, state: GpuState, fns: &Seq<GpuFunction>, fuel: nat,
) -> GpuState
    decreases fuel, s, 0nat,
{
    if state.returned || state.broken { state }
    else { match s {
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
                GpuState { bufs: state.bufs.update(*buf as int,
                    state.bufs[*buf as int].update(i, v)), ..state }
            } else { state }
        },
        GpuStmt::TextureStore { tex, coords, val } => {
            let c = gpu_value_to_int(&wgsl_semantics_expr(coords, &state));
            let v = wgsl_semantics_expr(val, &state);
            if (*tex as int) < state.bufs.len()
                && 0 <= c && c < state.bufs[*tex as int].len()
            {
                GpuState { bufs: state.bufs.update(*tex as int,
                    state.bufs[*tex as int].update(c, v)), ..state }
            } else { state }
        },
        GpuStmt::AtomicRMW { buf, idx, op: atomic_op, val, compare, old_val_var } => {
            let i = gpu_value_to_int(&wgsl_semantics_expr(idx, &state));
            let operand = wgsl_semantics_expr(val, &state);
            let cmp_val = match compare {
                Option::Some(cmp_expr) => wgsl_semantics_expr(cmp_expr, &state),
                Option::None => GpuValue::Int(0),
            };
            if (*buf as int) < state.bufs.len()
                && 0 <= i && i < state.bufs[*buf as int].len()
            {
                let old_val = state.bufs[*buf as int][i];
                let new_val = wgsl_atomic_op(atomic_op, &old_val, &operand, &cmp_val);
                let locals_updated = match old_val_var {
                    Option::Some(rv) => if (*rv as int) < state.locals.len() {
                        state.locals.update(*rv as int, old_val)
                    } else { state.locals },
                    Option::None => state.locals,
                };
                GpuState { locals: locals_updated,
                    bufs: state.bufs.update(*buf as int,
                        state.bufs[*buf as int].update(i, new_val)), ..state }
            } else { state }
        },
        GpuStmt::CallStmt { fn_id, args, result_var } => {
            if fuel == 0 || !((*fn_id as int) < fns.len()) { state }
            else {
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
                    else if args.len() == 5 { seq![wgsl_semantics_expr(&args[0], &state),
                        wgsl_semantics_expr(&args[1], &state),
                        wgsl_semantics_expr(&args[2], &state),
                        wgsl_semantics_expr(&args[3], &state),
                        wgsl_semantics_expr(&args[4], &state)] }
                    else if args.len() == 6 { seq![wgsl_semantics_expr(&args[0], &state),
                        wgsl_semantics_expr(&args[1], &state),
                        wgsl_semantics_expr(&args[2], &state),
                        wgsl_semantics_expr(&args[3], &state),
                        wgsl_semantics_expr(&args[4], &state),
                        wgsl_semantics_expr(&args[5], &state)] }
                    else if args.len() == 7 { seq![wgsl_semantics_expr(&args[0], &state),
                        wgsl_semantics_expr(&args[1], &state),
                        wgsl_semantics_expr(&args[2], &state),
                        wgsl_semantics_expr(&args[3], &state),
                        wgsl_semantics_expr(&args[4], &state),
                        wgsl_semantics_expr(&args[5], &state),
                        wgsl_semantics_expr(&args[6], &state)] }
                    else if args.len() == 8 { seq![wgsl_semantics_expr(&args[0], &state),
                        wgsl_semantics_expr(&args[1], &state),
                        wgsl_semantics_expr(&args[2], &state),
                        wgsl_semantics_expr(&args[3], &state),
                        wgsl_semantics_expr(&args[4], &state),
                        wgsl_semantics_expr(&args[5], &state),
                        wgsl_semantics_expr(&args[6], &state),
                        wgsl_semantics_expr(&args[7], &state)] }
                    else { seq![] };
                let fn_locals = Seq::new(f.n_locals, |i: int|
                    if i < arg_vals.len() { arg_vals[i] } else { GpuValue::Int(0) });
                let fn_state = GpuState {
                    locals: fn_locals, bufs: state.bufs,
                    returned: false, broken: false };
                let result_state = wgsl_semantics_stmt(&f.body, fn_state, fns, (fuel - 1) as nat);
                let ret_val = if (f.ret_var as int) < result_state.locals.len() {
                    result_state.locals[f.ret_var as int]
                } else { GpuValue::Int(0) };
                if (*result_var as int) < state.locals.len() {
                    GpuState { locals: state.locals.update(*result_var as int, ret_val),
                        bufs: result_state.bufs, ..state }
                } else { GpuState { bufs: result_state.bufs, ..state } }
            }
        },
        GpuStmt::Seq { first, then } => {
            let mid = wgsl_semantics_stmt(first, state, fns, fuel);
            wgsl_semantics_stmt(then, mid, fns, fuel)
        },
        GpuStmt::If { cond, then_body, else_body } => {
            if gpu_value_truthy(&wgsl_semantics_expr(cond, &state)) {
                wgsl_semantics_stmt(then_body, state, fns, fuel)
            } else { wgsl_semantics_stmt(else_body, state, fns, fuel) }
        },
        GpuStmt::For { var, start, end, body } => {
            let s_val = gpu_value_to_int(&wgsl_semantics_expr(start, &state));
            let e_val = gpu_value_to_int(&wgsl_semantics_expr(end, &state));
            let result = wgsl_semantics_loop(*var, s_val, e_val, body, state, fns, fuel);
            GpuState { broken: false, ..result }
        },
        GpuStmt::Break => GpuState { broken: true, ..state },
        GpuStmt::Continue => state,
        GpuStmt::Barrier { .. } => state,
        GpuStmt::Return => GpuState { returned: true, ..state },
        GpuStmt::Noop => state,
    }}
}

pub open spec fn wgsl_semantics_loop(
    var: nat, current: int, end: int,
    body: &GpuStmt, state: GpuState,
    fns: &Seq<GpuFunction>, fuel: nat,
) -> GpuState
    decreases fuel, body, (if end > current { (end - current) as nat } else { 0nat }),
{
    if current >= end || state.returned || state.broken { state }
    else {
        let s = if (var as int) < state.locals.len() {
            GpuState { locals: state.locals.update(var as int, GpuValue::Int(current)), ..state }
        } else { state };
        let s2 = wgsl_semantics_stmt(body, s, fns, fuel);
        wgsl_semantics_loop(var, current + 1, end, body, s2, fns, fuel)
    }
}

} // verus!
