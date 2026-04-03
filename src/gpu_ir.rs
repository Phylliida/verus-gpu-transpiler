///  GpuIR: Verified intermediate representation for GPU compute kernels.
///
///  Key design choices for Verus compatibility:
///  - GpuExpr is purely self-recursive (no stmt calls) → simple `decreases e`
///  - GpuStmt uses binary Seq { first, then } (not Block(Seq)) → structural recursion
///  - gpu_eval_stmt + gpu_eval_loop are a mutual recursion pair → matches stage.rs pattern
///  - Function calls are statements (CallStmt), not expressions → avoids expr↔stmt recursion

use vstd::prelude::*;

verus! {

//  ══════════════════════════════════════════════════════════════
//  Helpers
//  ══════════════════════════════════════════════════════════════

///  Power of 2.
pub open spec fn pow2(n: nat) -> nat
    decreases n,
{
    if n == 0 { 1 }
    else { 2 * pow2((n - 1) as nat) }
}

//  ══════════════════════════════════════════════════════════════
//  Types
//  ══════════════════════════════════════════════════════════════

pub enum ScalarType { I32, U32, F32, Bool }

pub enum GpuType {
    Scalar(ScalarType),
    Void,
}

///  Runtime value at the spec level.
pub enum GpuValue {
    Int(int),
    Float(f32),
    Bool(bool),
}

//  ══════════════════════════════════════════════════════════════
//  Value helpers
//  ══════════════════════════════════════════════════════════════

pub open spec fn gpu_value_to_int(v: &GpuValue) -> int {
    match v {
        GpuValue::Int(i) => *i,
        GpuValue::Float(_) => 0,
        GpuValue::Bool(b) => if *b { 1 } else { 0 },
    }
}

pub open spec fn gpu_value_truthy(v: &GpuValue) -> bool {
    match v {
        GpuValue::Int(i) => *i != 0,
        GpuValue::Float(_) => true,
        GpuValue::Bool(b) => *b,
    }
}

//  ══════════════════════════════════════════════════════════════
//  Operators
//  ══════════════════════════════════════════════════════════════

pub enum GpuBinOp {
    //  Integer (proven overflow-free)
    Add, Sub, Mul, Div, Mod, Shr, Shl,
    //  Wrapping integer (wraps at 32 bits, no overflow proof)
    WrappingAdd, WrappingSub, WrappingMul,
    //  Float
    FAdd, FSub, FMul, FDiv,
    //  Comparison (returns Bool)
    Lt, Le, Gt, Ge, Eq, Ne,
    //  Bitwise (integer, operates on i32 width)
    BitAnd, BitOr, BitXor,
    //  Logical (bool)
    LogicalAnd, LogicalOr,
}

pub enum GpuUnaryOp {
    Neg,
    FNeg,
    BitNot,
    LogicalNot,
}

pub enum BarrierScope {
    Workgroup,
    Storage,
}

//  ══════════════════════════════════════════════════════════════
//  Expressions — purely self-recursive, no stmt calls
//  ══════════════════════════════════════════════════════════════

pub enum GpuExpr {
    Const(int, ScalarType),
    FConst(f32),
    Var(nat, GpuType),
    BinOp(GpuBinOp, Box<GpuExpr>, Box<GpuExpr>),
    UnaryOp(GpuUnaryOp, Box<GpuExpr>),
    Select(Box<GpuExpr>, Box<GpuExpr>, Box<GpuExpr>),
    ArrayRead(nat, Box<GpuExpr>),
    Cast(GpuType, Box<GpuExpr>),
}

//  ══════════════════════════════════════════════════════════════
//  Statements — binary Seq tree (matches Stage pattern)
//  ══════════════════════════════════════════════════════════════

pub enum GpuStmt {
    Assign { var: nat, rhs: GpuExpr },
    BufWrite { buf: nat, idx: GpuExpr, val: GpuExpr },
    ///  Function call: evaluate args, run body, store result in result_var.
    ///  Lives in GpuStmt (not GpuExpr) to avoid expr↔stmt mutual recursion.
    ///  Usage: `call(f, args, tmp); let x = tmp + 1;`
    CallStmt { fn_id: nat, args: Seq<GpuExpr>, result_var: nat },
    ///  Binary sequential composition. Both children are structural sub-terms.
    ///  Build flat sequences with `seq_stmts` helper or nested Seq.
    Seq { first: Box<GpuStmt>, then: Box<GpuStmt> },
    If { cond: GpuExpr, then_body: Box<GpuStmt>, else_body: Box<GpuStmt> },
    For { var: nat, start: GpuExpr, end: GpuExpr, body: Box<GpuStmt> },
    Break,
    Continue,
    Barrier { scope: BarrierScope },
    Return,
    Noop,
}

//  ══════════════════════════════════════════════════════════════
//  Sequence builder — flat list → nested binary Seq
//  ══════════════════════════════════════════════════════════════

///  Build a GpuStmt from a flat list. Mirrors `seq_stages` from stage.rs.
pub open spec fn seq_stmts(stmts: Seq<GpuStmt>) -> GpuStmt
    decreases stmts.len(),
{
    if stmts.len() == 0 {
        GpuStmt::Noop
    } else if stmts.len() == 1 {
        stmts[0]
    } else {
        GpuStmt::Seq {
            first: Box::new(stmts[0]),
            then: Box::new(seq_stmts(stmts.skip(1))),
        }
    }
}

//  ══════════════════════════════════════════════════════════════
//  Program structure
//  ══════════════════════════════════════════════════════════════

pub struct GpuFunction {
    pub params: Seq<(nat, GpuType)>,
    pub ret_type: GpuType,
    pub n_locals: nat,
    pub body: GpuStmt,
    pub ret_var: nat,
}

pub struct GpuKernel {
    pub n_locals: nat,
    pub n_bufs: nat,
    pub functions: Seq<GpuFunction>,
    pub body: GpuStmt,
    pub workgroup_size: (nat, nat, nat),
}

//  ══════════════════════════════════════════════════════════════
//  State model
//  ══════════════════════════════════════════════════════════════

pub struct GpuState {
    pub locals: Seq<GpuValue>,
    pub bufs: Seq<Seq<GpuValue>>,
    pub returned: bool,
    pub broken: bool,
}

//  ══════════════════════════════════════════════════════════════
//  Wrapping arithmetic
//  ══════════════════════════════════════════════════════════════

///  Wrap to signed 32-bit range [-2^31, 2^31).
pub open spec fn wrap32(x: int) -> int {
    let m: int = 0x1_0000_0000;
    ((x + 0x8000_0000) % m + m) % m - 0x8000_0000
}

//  ══════════════════════════════════════════════════════════════
//  Binary operator evaluation
//  ══════════════════════════════════════════════════════════════

pub open spec fn gpu_eval_binop(op: &GpuBinOp, a: &GpuValue, b: &GpuValue) -> GpuValue {
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
        GpuBinOp::Lt => GpuValue::Bool(ai < bi),
        GpuBinOp::Le => GpuValue::Bool(ai <= bi),
        GpuBinOp::Gt => GpuValue::Bool(ai > bi),
        GpuBinOp::Ge => GpuValue::Bool(ai >= bi),
        GpuBinOp::Eq => GpuValue::Bool(ai == bi),
        GpuBinOp::Ne => GpuValue::Bool(ai != bi),
        GpuBinOp::BitAnd => GpuValue::Int((ai as i32 & bi as i32) as int),
        GpuBinOp::BitOr => GpuValue::Int((ai as i32 | bi as i32) as int),
        GpuBinOp::BitXor => GpuValue::Int((ai as i32 ^ bi as i32) as int),
        GpuBinOp::LogicalAnd => GpuValue::Bool(
            gpu_value_truthy(a) && gpu_value_truthy(b)),
        GpuBinOp::LogicalOr => GpuValue::Bool(
            gpu_value_truthy(a) || gpu_value_truthy(b)),
    }
}

pub open spec fn gpu_eval_unaryop(op: &GpuUnaryOp, a: &GpuValue) -> GpuValue {
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

//  ══════════════════════════════════════════════════════════════
//  Expression evaluation — self-recursive only
//  ══════════════════════════════════════════════════════════════

pub open spec fn gpu_eval_expr(
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
        GpuExpr::BinOp(op, a, b) => {
            let va = gpu_eval_expr(a, state);
            let vb = gpu_eval_expr(b, state);
            gpu_eval_binop(op, &va, &vb)
        },
        GpuExpr::UnaryOp(op, a) => {
            let va = gpu_eval_expr(a, state);
            gpu_eval_unaryop(op, &va)
        },
        GpuExpr::Select(cond, t, f) => {
            if gpu_value_truthy(&gpu_eval_expr(cond, state)) {
                gpu_eval_expr(t, state)
            } else {
                gpu_eval_expr(f, state)
            }
        },
        GpuExpr::ArrayRead(buf_idx, idx_expr) => {
            let idx = gpu_value_to_int(&gpu_eval_expr(idx_expr, state));
            if (*buf_idx as int) < state.bufs.len()
                && 0 <= idx
                && idx < state.bufs[*buf_idx as int].len()
            {
                state.bufs[*buf_idx as int][idx]
            } else { GpuValue::Int(0) }
        },
        GpuExpr::Cast(ty, inner) => {
            let v = gpu_eval_expr(inner, state);
            match ty {
                GpuType::Scalar(ScalarType::F32) =>
                    GpuValue::Float((gpu_value_to_int(&v) as i32) as f32),
                GpuType::Scalar(ScalarType::I32) => GpuValue::Int(gpu_value_to_int(&v)),
                GpuType::Scalar(ScalarType::U32) => GpuValue::Int(gpu_value_to_int(&v)),
                GpuType::Scalar(ScalarType::Bool) => GpuValue::Bool(gpu_value_truthy(&v)),
                _ => v,
            }
        },
    }
}

//  ══════════════════════════════════════════════════════════════
//  Helper: evaluate a Seq of expressions (for function call args)
//  ══════════════════════════════════════════════════════════════

pub open spec fn gpu_eval_args(
    args: &Seq<GpuExpr>, state: &GpuState, idx: nat,
) -> Seq<GpuValue>
    decreases args.len() - idx,
{
    if idx >= args.len() {
        seq![]
    } else {
        seq![gpu_eval_expr(&args[idx as int], state)]
            + gpu_eval_args(args, state, idx + 1)
    }
}

//  ══════════════════════════════════════════════════════════════
//  Statement evaluation — mutual recursion with gpu_eval_loop
//
//  Follows the staged_eval / eval_loop pattern from stage.rs:
//  - gpu_eval_stmt(s, ..): decreases (s, 0nat)
//  - gpu_eval_loop(body, ..): decreases (body, remaining_iterations)
//  Both first args are GpuStmt, both second args are nat.
//  loop→stmt: body is same, 0 < remaining, so (body, 0) < (body, n). ✓
//  stmt→loop (For): body is sub-term of For, so (body, _) < (For, _). ✓
//  stmt→stmt (Seq/If): children are sub-terms, so strictly smaller. ✓
//  ══════════════════════════════════════════════════════════════

pub open spec fn gpu_eval_stmt(
    s: &GpuStmt, state: GpuState, fns: &Seq<GpuFunction>,
) -> GpuState
    decreases s, 0nat,
{
    if state.returned || state.broken { state }
    else {
        match s {
            GpuStmt::Assign { var, rhs } => {
                let v = gpu_eval_expr(rhs, &state);
                if (*var as int) < state.locals.len() {
                    GpuState { locals: state.locals.update(*var as int, v), ..state }
                } else { state }
            },
            GpuStmt::BufWrite { buf, idx, val } => {
                let i = gpu_value_to_int(&gpu_eval_expr(idx, &state));
                let v = gpu_eval_expr(val, &state);
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
            GpuStmt::CallStmt { fn_id, args, result_var } => {
                if (*fn_id as int) < fns.len() {
                    let f = &fns[*fn_id as int];
                    let arg_vals = gpu_eval_args(args, &state, 0);
                    let fn_locals = Seq::new(f.n_locals, |i: int|
                        if i < arg_vals.len() { arg_vals[i] }
                        else { GpuValue::Int(0) });
                    let fn_state = GpuState {
                        locals: fn_locals, bufs: state.bufs,
                        returned: false, broken: false,
                    };
                    //  NOTE: f.body is NOT a sub-term of s. This call is NOT
                    //  structurally decreasing. We need a separate mechanism
                    //  (fuel/depth counter) to handle function calls.
                    //  For Phase 1a: support non-recursive calls only by
                    //  using a depth counter.
                    //  TODO: add fuel parameter for function call depth
                    let result_state = fn_state;  //  placeholder: no eval yet
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
                let mid = gpu_eval_stmt(first, state, fns);
                gpu_eval_stmt(then, mid, fns)
            },
            GpuStmt::If { cond, then_body, else_body } => {
                if gpu_value_truthy(&gpu_eval_expr(cond, &state)) {
                    gpu_eval_stmt(then_body, state, fns)
                } else {
                    gpu_eval_stmt(else_body, state, fns)
                }
            },
            GpuStmt::For { var, start, end, body } => {
                let s_val = gpu_value_to_int(&gpu_eval_expr(start, &state));
                let e_val = gpu_value_to_int(&gpu_eval_expr(end, &state));
                let result = gpu_eval_loop(*var, s_val, e_val, body, state, fns);
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

///  Evaluate a for loop: iterate var from `current` to `end - 1`.
pub open spec fn gpu_eval_loop(
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
        let s2 = gpu_eval_stmt(body, s, fns);
        gpu_eval_loop(var, current + 1, end, body, s2, fns)
    }
}

//  ══════════════════════════════════════════════════════════════
//  Kernel-level evaluation
//  ══════════════════════════════════════════════════════════════

///  Evaluate a kernel for a single thread. Thread ID in locals[0].
pub open spec fn gpu_eval_kernel_thread(
    k: &GpuKernel, tid: nat, bufs: Seq<Seq<GpuValue>>,
) -> GpuState {
    let locals = Seq::new(k.n_locals, |_i: int| GpuValue::Int(0))
        .update(0, GpuValue::Int(tid as int));
    let init = GpuState { locals, bufs, returned: false, broken: false };
    gpu_eval_stmt(&k.body, init, &k.functions)
}

} // verus!
