///  GpuIR: Verified intermediate representation for GPU compute kernels.
///
///  This is the core IR that replaces ArithExpr for GPU kernel representation.
///  It supports statements, control flow, functions, and multiple types —
///  everything needed to write real GPU kernels in natural Verus code.
///
///  Spec-level eval functions define the semantics. A separate `wgsl_semantics`
///  spec (Phase 2) is proved equivalent by structural induction, giving a
///  general correctness theorem for the WGSL emitter.

use vstd::prelude::*;

verus! {

///  Power of 2. Local definition to avoid dependency on verus-cutedsl in core IR.
pub open spec fn pow2(n: nat) -> nat
    decreases n,
{
    if n == 0 { 1 }
    else { 2 * pow2((n - 1) as nat) }
}

//  ══════════════════════════════════════════════════════════════
//  Types
//  ══════════════════════════════════════════════════════════════

///  Scalar types — maps directly to WGSL scalar types.
pub enum ScalarType {
    I32,
    U32,
    F32,
    Bool,
}

///  GpuIR type system. Phase 1a: scalar only. Vec/Mat/F16 in Phase 1b.
pub enum GpuType {
    Scalar(ScalarType),
    Void,
}

///  Runtime value at the spec level.
///  All GPU computations reduce to these values.
pub enum GpuValue {
    ///  Integer value (i32 or u32 at the spec level — uses mathematical int).
    Int(int),
    ///  Float value (f32).
    Float(f32),
    ///  Boolean value.
    Bool(bool),
}

//  ══════════════════════════════════════════════════════════════
//  Value helpers
//  ══════════════════════════════════════════════════════════════

///  Extract integer from a GpuValue, defaulting to 0.
pub open spec fn gpu_value_to_int(v: &GpuValue) -> int {
    match v {
        GpuValue::Int(i) => *i,
        GpuValue::Float(_) => 0,
        GpuValue::Bool(b) => if *b { 1 } else { 0 },
    }
}

///  Truthiness: non-zero int or true bool.
pub open spec fn gpu_value_truthy(v: &GpuValue) -> bool {
    match v {
        GpuValue::Int(i) => *i != 0,
        GpuValue::Float(_) => true,
        GpuValue::Bool(b) => *b,
    }
}

///  Default value for a scalar type.
pub open spec fn gpu_default_value(ty: &ScalarType) -> GpuValue {
    match ty {
        ScalarType::I32 => GpuValue::Int(0),
        ScalarType::U32 => GpuValue::Int(0),
        ScalarType::F32 => GpuValue::Float(0.0f32),
        ScalarType::Bool => GpuValue::Bool(false),
    }
}

//  ══════════════════════════════════════════════════════════════
//  Operators
//  ══════════════════════════════════════════════════════════════

///  Binary operators.
pub enum GpuBinOp {
    //  Integer arithmetic (proven overflow-free)
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Shr,
    Shl,
    //  Wrapping integer arithmetic (no overflow proof, wraps at 32 bits)
    WrappingAdd,
    WrappingSub,
    WrappingMul,
    //  Float arithmetic
    FAdd,
    FSub,
    FMul,
    FDiv,
    //  Comparisons (return Bool)
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    //  Bitwise (integer only)
    BitAnd,
    BitOr,
    BitXor,
    //  Logical (bool only)
    LogicalAnd,
    LogicalOr,
}

///  Unary operators.
pub enum GpuUnaryOp {
    Neg,
    FNeg,
    Not,
    LogicalNot,
}

///  Barrier synchronization scope.
pub enum BarrierScope {
    ///  workgroupBarrier() — all threads in workgroup sync.
    Workgroup,
    ///  storageBarrier() — storage buffer writes become visible.
    Storage,
}

//  ══════════════════════════════════════════════════════════════
//  Expressions
//  ══════════════════════════════════════════════════════════════

///  GPU expression — pure (no side effects).
pub enum GpuExpr {
    ///  Integer/bool constant with type tag.
    Const(int, ScalarType),
    ///  Float constant.
    FConst(f32),
    ///  Local variable reference (index into GpuState.locals).
    Var(nat, GpuType),
    ///  Binary operation.
    BinOp(GpuBinOp, Box<GpuExpr>, Box<GpuExpr>),
    ///  Unary operation.
    UnaryOp(GpuUnaryOp, Box<GpuExpr>),
    ///  Conditional: if cond is truthy then true_val else false_val.
    Select(Box<GpuExpr>, Box<GpuExpr>, Box<GpuExpr>),
    ///  Buffer read: bufs[buf_idx][eval(index)].
    ArrayRead(nat, Box<GpuExpr>),
    ///  Function call: fns[fn_id](args...).
    Call(nat, Seq<GpuExpr>),
    ///  Type cast (e.g., i32 → f32).
    Cast(GpuType, Box<GpuExpr>),
}

//  ══════════════════════════════════════════════════════════════
//  Statements
//  ══════════════════════════════════════════════════════════════

///  GPU statement — has side effects on GpuState.
pub enum GpuStmt {
    ///  Assign to a local variable: locals[var] = eval(rhs).
    Assign { var: nat, rhs: GpuExpr },
    ///  Write to buffer: bufs[buf][eval(idx)] = eval(val).
    BufWrite { buf: nat, idx: GpuExpr, val: GpuExpr },
    ///  Sequential block of statements.
    Block(Seq<GpuStmt>),
    ///  Conditional execution.
    If { cond: GpuExpr, then_body: Box<GpuStmt>, else_body: Box<GpuStmt> },
    ///  Bounded for loop: for var in eval(start)..eval(end) { body }.
    For { var: nat, start: GpuExpr, end: GpuExpr, body: Box<GpuStmt> },
    ///  Break out of innermost loop.
    Break,
    ///  Continue to next iteration of innermost loop.
    Continue,
    ///  Barrier synchronization.
    Barrier { scope: BarrierScope },
    ///  Early return.
    Return,
    ///  No-op.
    Noop,
}

//  ══════════════════════════════════════════════════════════════
//  Program structure
//  ══════════════════════════════════════════════════════════════

///  A helper function callable from kernel code.
pub struct GpuFunction {
    pub params: Seq<(nat, GpuType)>,
    pub ret_type: GpuType,
    pub n_locals: nat,
    pub body: GpuStmt,
    pub ret_var: nat,
}

///  A complete GPU compute kernel.
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

///  Per-thread execution state.
pub struct GpuState {
    ///  Local variable values.
    pub locals: Seq<GpuValue>,
    ///  Buffer contents (shared across threads at spec level).
    pub bufs: Seq<Seq<GpuValue>>,
    ///  Whether a Return statement has been executed.
    pub returned: bool,
    ///  Whether a Break statement has been executed (cleared on loop entry).
    pub broken: bool,
}

//  ══════════════════════════════════════════════════════════════
//  Binary operator evaluation
//  ══════════════════════════════════════════════════════════════

///  Wrapping addition at 32 bits (faithful to i32/u32 hardware behavior).
pub open spec fn wrap32(x: int) -> int {
    //  Wrap to [-2^31, 2^31) range
    let m = 0x1_0000_0000int;
    ((x + 0x8000_0000) % m + m) % m - 0x8000_0000
}

///  Evaluate a binary operator on two values.
pub open spec fn gpu_eval_binop(op: &GpuBinOp, a: &GpuValue, b: &GpuValue) -> GpuValue {
    let ai = gpu_value_to_int(a);
    let bi = gpu_value_to_int(b);
    match op {
        //  Integer arithmetic
        GpuBinOp::Add => GpuValue::Int(ai + bi),
        GpuBinOp::Sub => GpuValue::Int(ai - bi),
        GpuBinOp::Mul => GpuValue::Int(ai * bi),
        GpuBinOp::Div => GpuValue::Int(if bi != 0 { ai / bi } else { 0 }),
        GpuBinOp::Mod => GpuValue::Int(if bi != 0 { ai % bi } else { 0 }),
        GpuBinOp::Shr => GpuValue::Int(if bi >= 0 && bi < 32 {
            ai / pow2(bi as nat) as int
        } else { 0 }),
        GpuBinOp::Shl => GpuValue::Int(if bi >= 0 && bi < 32 {
            ai * pow2(bi as nat) as int
        } else { 0 }),
        //  Wrapping arithmetic
        GpuBinOp::WrappingAdd => GpuValue::Int(wrap32(ai + bi)),
        GpuBinOp::WrappingSub => GpuValue::Int(wrap32(ai - bi)),
        GpuBinOp::WrappingMul => GpuValue::Int(wrap32(ai * bi)),
        //  Float arithmetic
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
        //  Comparisons
        GpuBinOp::Lt => GpuValue::Bool(ai < bi),
        GpuBinOp::Le => GpuValue::Bool(ai <= bi),
        GpuBinOp::Gt => GpuValue::Bool(ai > bi),
        GpuBinOp::Ge => GpuValue::Bool(ai >= bi),
        GpuBinOp::Eq => GpuValue::Bool(ai == bi),
        GpuBinOp::Ne => GpuValue::Bool(ai != bi),
        //  Bitwise — operate on i32 representation, then widen back to int
        GpuBinOp::BitAnd => GpuValue::Int((ai as i32 & bi as i32) as int),
        GpuBinOp::BitOr => GpuValue::Int((ai as i32 | bi as i32) as int),
        GpuBinOp::BitXor => GpuValue::Int((ai as i32 ^ bi as i32) as int),
        //  Logical
        GpuBinOp::LogicalAnd => GpuValue::Bool(
            gpu_value_truthy(a) && gpu_value_truthy(b)),
        GpuBinOp::LogicalOr => GpuValue::Bool(
            gpu_value_truthy(a) || gpu_value_truthy(b)),
    }
}

///  Evaluate a unary operator.
pub open spec fn gpu_eval_unaryop(op: &GpuUnaryOp, a: &GpuValue) -> GpuValue {
    match op {
        GpuUnaryOp::Neg => GpuValue::Int(-gpu_value_to_int(a)),
        GpuUnaryOp::FNeg => match a {
            GpuValue::Float(f) => GpuValue::Float(-*f),
            _ => GpuValue::Float(0.0f32),
        },
        GpuUnaryOp::Not => GpuValue::Int(!(gpu_value_to_int(a) as i32) as int),
        GpuUnaryOp::LogicalNot => GpuValue::Bool(!gpu_value_truthy(a)),
    }
}

//  ══════════════════════════════════════════════════════════════
//  Expression evaluation
//  ══════════════════════════════════════════════════════════════

///  Evaluate an expression, producing a GpuValue.
///  Pure — no side effects on state.
pub open spec fn gpu_eval_expr(
    e: &GpuExpr, state: &GpuState, fns: &Seq<GpuFunction>,
) -> GpuValue
    decreases e,
{
    match e {
        GpuExpr::Const(c, _) => GpuValue::Int(*c),
        GpuExpr::FConst(f) => GpuValue::Float(*f),
        GpuExpr::Var(i, _) => {
            if (*i as int) < state.locals.len() {
                state.locals[*i as int]
            } else {
                GpuValue::Int(0)
            }
        },
        GpuExpr::BinOp(op, a, b) => {
            let va = gpu_eval_expr(a, state, fns);
            let vb = gpu_eval_expr(b, state, fns);
            gpu_eval_binop(op, &va, &vb)
        },
        GpuExpr::UnaryOp(op, a) => {
            let va = gpu_eval_expr(a, state, fns);
            gpu_eval_unaryop(op, &va)
        },
        GpuExpr::Select(cond, t, f) => {
            let vc = gpu_eval_expr(cond, state, fns);
            if gpu_value_truthy(&vc) {
                gpu_eval_expr(t, state, fns)
            } else {
                gpu_eval_expr(f, state, fns)
            }
        },
        GpuExpr::ArrayRead(buf_idx, idx_expr) => {
            let idx = gpu_value_to_int(&gpu_eval_expr(idx_expr, state, fns));
            if (*buf_idx as int) < state.bufs.len()
                && 0 <= idx
                && idx < state.bufs[*buf_idx as int].len()
            {
                state.bufs[*buf_idx as int][idx]
            } else {
                GpuValue::Int(0)
            }
        },
        GpuExpr::Call(fn_id, args) => {
            if (*fn_id as int) < fns.len() {
                let f = &fns[*fn_id as int];
                //  Evaluate arguments
                let arg_vals = Seq::new(args.len(), |i: int|
                    gpu_eval_expr(&args[i], state, fns));
                //  Build function-local state with args bound to param slots
                let fn_locals = Seq::new(
                    f.n_locals,
                    |i: int| if i < arg_vals.len() { arg_vals[i] } else { GpuValue::Int(0) },
                );
                let fn_state = GpuState {
                    locals: fn_locals,
                    bufs: state.bufs,
                    returned: false,
                    broken: false,
                };
                //  Evaluate function body, extract return value
                let result_state = gpu_eval_stmt(&f.body, fn_state, fns);
                if (f.ret_var as int) < result_state.locals.len() {
                    result_state.locals[f.ret_var as int]
                } else {
                    GpuValue::Int(0)
                }
            } else {
                GpuValue::Int(0)
            }
        },
        GpuExpr::Cast(ty, inner) => {
            let v = gpu_eval_expr(inner, state, fns);
            //  Type casts at spec level — identity for same-width types
            match ty {
                GpuType::Scalar(ScalarType::F32) => {
                    //  int → i32 → f32 (Verus requires concrete type for float cast)
                    GpuValue::Float((gpu_value_to_int(&v) as i32) as f32)
                },
                GpuType::Scalar(ScalarType::I32) => GpuValue::Int(gpu_value_to_int(&v)),
                GpuType::Scalar(ScalarType::U32) => GpuValue::Int(gpu_value_to_int(&v)),
                GpuType::Scalar(ScalarType::Bool) => GpuValue::Bool(gpu_value_truthy(&v)),
                _ => v,
            }
        },
    }
}

//  ══════════════════════════════════════════════════════════════
//  Statement evaluation
//  ══════════════════════════════════════════════════════════════

///  Evaluate a statement, producing an updated GpuState.
pub open spec fn gpu_eval_stmt(
    s: &GpuStmt, state: GpuState, fns: &Seq<GpuFunction>,
) -> GpuState
    decreases s, 0int,
{
    if state.returned || state.broken { state }
    else {
        match s {
            GpuStmt::Assign { var, rhs } => {
                let v = gpu_eval_expr(rhs, &state, fns);
                if (*var as int) < state.locals.len() {
                    GpuState { locals: state.locals.update(*var as int, v), ..state }
                } else {
                    state
                }
            },
            GpuStmt::BufWrite { buf, idx, val } => {
                let i = gpu_value_to_int(&gpu_eval_expr(idx, &state, fns));
                let v = gpu_eval_expr(val, &state, fns);
                if (*buf as int) < state.bufs.len()
                    && 0 <= i
                    && i < state.bufs[*buf as int].len()
                {
                    GpuState {
                        bufs: state.bufs.update(*buf as int,
                            state.bufs[*buf as int].update(i, v)),
                        ..state
                    }
                } else {
                    state
                }
            },
            GpuStmt::Block(stmts) => {
                gpu_eval_block(stmts, state, fns, 0)
            },
            GpuStmt::If { cond, then_body, else_body } => {
                let cv = gpu_eval_expr(cond, &state, fns);
                if gpu_value_truthy(&cv) {
                    gpu_eval_stmt(then_body, state, fns)
                } else {
                    gpu_eval_stmt(else_body, state, fns)
                }
            },
            GpuStmt::For { var, start, end, body } => {
                let s_val = gpu_value_to_int(&gpu_eval_expr(start, &state, fns));
                let e_val = gpu_value_to_int(&gpu_eval_expr(end, &state, fns));
                let result = gpu_eval_loop(*var, s_val, e_val, body, state, fns);
                //  Clear broken flag after loop exits (break only exits innermost)
                GpuState { broken: false, ..result }
            },
            GpuStmt::Break => GpuState { broken: true, ..state },
            GpuStmt::Continue => state,  //  Continue handled in loop eval
            GpuStmt::Barrier { .. } => state,  //  No-op on per-thread state
            GpuStmt::Return => GpuState { returned: true, ..state },
            GpuStmt::Noop => state,
        }
    }
}

///  Evaluate a block of statements sequentially.
///  NOTE: Block is not mutually recursive with gpu_eval_stmt — it's called
///  from within the Block arm. We use a separate function to avoid deep nesting.
///  Termination: stmts.len() - idx decreases each step.
pub open spec fn gpu_eval_block(
    stmts: &Seq<GpuStmt>, state: GpuState,
    fns: &Seq<GpuFunction>, idx: nat,
) -> GpuState
    decreases stmts.len() - idx, 0int,
{
    if idx >= stmts.len() || state.returned || state.broken {
        state
    } else {
        let s2 = gpu_eval_stmt(&stmts[idx as int], state, fns);
        gpu_eval_block(stmts, s2, fns, idx + 1)
    }
}

///  Evaluate a for loop body from `current` to `end - 1`.
pub open spec fn gpu_eval_loop(
    var: nat, current: int, end: int,
    body: &GpuStmt, state: GpuState,
    fns: &Seq<GpuFunction>,
) -> GpuState
    decreases body, (if end > current { end - current } else { 0 }),
{
    if current >= end || state.returned || state.broken {
        state
    } else {
        let s = if (var as int) < state.locals.len() {
            GpuState {
                locals: state.locals.update(var as int, GpuValue::Int(current)),
                ..state
            }
        } else {
            state
        };
        let s2 = gpu_eval_stmt(body, s, fns);
        gpu_eval_loop(var, current + 1, end, body, s2, fns)
    }
}

//  ══════════════════════════════════════════════════════════════
//  Kernel-level evaluation
//  ══════════════════════════════════════════════════════════════

///  Evaluate a kernel for a single thread.
///  Thread ID is placed in locals[0].
pub open spec fn gpu_eval_kernel_thread(
    k: &GpuKernel, tid: nat, bufs: Seq<Seq<GpuValue>>,
) -> GpuState {
    let locals = Seq::new(k.n_locals, |_i: int| GpuValue::Int(0))
        .update(0, GpuValue::Int(tid as int));
    let init = GpuState { locals, bufs, returned: false, broken: false };
    gpu_eval_stmt(&k.body, init, &k.functions)
}

} // verus!
