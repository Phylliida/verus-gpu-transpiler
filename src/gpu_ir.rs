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

pub enum ScalarType { I32, U32, F32, F16, Bool }

pub enum GpuType {
    Scalar(ScalarType),
    Vec2(ScalarType),
    Vec3(ScalarType),
    Vec4(ScalarType),
    Mat { cols: nat, rows: nat, elem: ScalarType },
    Void,
}

///  Runtime value at the spec level.
pub enum GpuValue {
    Int(int),
    Float(f32),
    Bool(bool),
    ///  Vector: 2-4 components.
    Vec(Seq<GpuValue>),
    ///  Matrix: columns of vectors (column-major, matching WGSL).
    Mat(Seq<Seq<GpuValue>>),
}

//  ══════════════════════════════════════════════════════════════
//  Value helpers
//  ══════════════════════════════════════════════════════════════

pub open spec fn gpu_value_to_int(v: &GpuValue) -> int {
    match v {
        GpuValue::Int(i) => *i,
        GpuValue::Float(_) => 0,
        GpuValue::Bool(b) => if *b { 1 } else { 0 },
        GpuValue::Vec(_) => 0,
        GpuValue::Mat(_) => 0,
    }
}

pub open spec fn gpu_value_truthy(v: &GpuValue) -> bool {
    match v {
        GpuValue::Int(i) => *i != 0,
        GpuValue::Float(_) => true,
        GpuValue::Bool(b) => *b,
        _ => false,
    }
}

///  Get a component of a vec value. Returns Int(0) if out of range or not a vec.
pub open spec fn gpu_value_vec_component(v: &GpuValue, idx: nat) -> GpuValue {
    match v {
        GpuValue::Vec(components) => {
            if (idx as int) < components.len() {
                components[idx as int]
            } else { GpuValue::Int(0) }
        },
        _ => GpuValue::Int(0),
    }
}

///  Get the number of components in a vec, or 1 for scalars.
pub open spec fn gpu_value_width(v: &GpuValue) -> nat {
    match v {
        GpuValue::Vec(c) => c.len(),
        _ => 1,
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
    Subgroup,
}

pub enum AtomicOp {
    Load, Store, Add, Sub, Max, Min,
    And, Or, Xor, Exchange, CompareExchangeWeak,
}

pub enum SubgroupReduceOp {
    Add, Mul, Min, Max, And, Or, Xor,
    ExclusiveAdd, InclusiveAdd, ExclusiveMul, InclusiveMul,
}

pub enum SubgroupCommOp {
    Broadcast(nat),
    BroadcastFirst,
    Shuffle, ShuffleXor, ShuffleUp, ShuffleDown,
}

pub enum SubgroupVoteOp {
    Ballot, All, Any, Elect,
}

pub enum PackFormat { SNorm, UNorm, SInt, UInt }

pub enum GpuBuiltin {
    GlobalInvocationId { dim: nat },
    LocalInvocationId { dim: nat },
    LocalInvocationIndex,
    WorkgroupId { dim: nat },
    NumWorkgroups { dim: nat },
    SubgroupId,
    SubgroupInvocationId,
    SubgroupSize,
}

pub enum MemorySpace { Storage, Workgroup, Uniform }

pub struct BufferBinding {
    pub binding: nat,
    pub space: MemorySpace,
    pub read_only: bool,
    pub elem_type: GpuType,
}

pub struct TextureBinding {
    pub binding: nat,
    pub read_only: bool,
    pub texel_type: GpuType,
    pub dimensions: nat,
}

//  ══════════════════════════════════════════════════════════════
//  Expressions — purely self-recursive, no stmt calls
//  ══════════════════════════════════════════════════════════════

pub enum GpuExpr {
    //  Scalar constants
    Const(int, ScalarType),
    FConst(f32),
    //  Variables and builtins
    Var(nat, GpuType),
    Builtin(GpuBuiltin),
    //  Arithmetic and logic
    BinOp(GpuBinOp, Box<GpuExpr>, Box<GpuExpr>),
    UnaryOp(GpuUnaryOp, Box<GpuExpr>),
    Select(Box<GpuExpr>, Box<GpuExpr>, Box<GpuExpr>),
    //  Memory
    ArrayRead(nat, Box<GpuExpr>),
    TextureLoad(nat, Box<GpuExpr>),
    //  Type conversion
    Cast(GpuType, Box<GpuExpr>),
    //  Vector ops
    VecConstruct(Seq<GpuExpr>),
    VecComponent(Box<GpuExpr>, nat),
    Swizzle(Box<GpuExpr>, Seq<nat>),
    //  Matrix ops
    MatConstruct(nat, nat, Seq<GpuExpr>),
    MatMul(Box<GpuExpr>, Box<GpuExpr>),
    Transpose(Box<GpuExpr>),
    Determinant(Box<GpuExpr>),
    //  Packed types
    Pack4x8(PackFormat, Box<GpuExpr>),
    Unpack4x8(PackFormat, Box<GpuExpr>),
    //  Subgroup ops
    SubgroupReduce(SubgroupReduceOp, Box<GpuExpr>),
    SubgroupComm(SubgroupCommOp, Box<GpuExpr>),
    SubgroupVote(SubgroupVoteOp, Box<GpuExpr>),
}

//  ══════════════════════════════════════════════════════════════
//  Statements — binary Seq tree (matches Stage pattern)
//  ══════════════════════════════════════════════════════════════

pub enum GpuStmt {
    Assign { var: nat, rhs: GpuExpr },
    BufWrite { buf: nat, idx: GpuExpr, val: GpuExpr },
    TextureStore { tex: nat, coords: GpuExpr, val: GpuExpr },
    AtomicRMW { buf: nat, idx: GpuExpr, op: AtomicOp,
                val: GpuExpr, old_val_var: Option<nat> },
    ///  Function call: evaluate args, run body, store result in result_var.
    CallStmt { fn_id: nat, args: Seq<GpuExpr>, result_var: nat },
    ///  Binary sequential composition. Both children are structural sub-terms.
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
    pub buffers: Seq<BufferBinding>,
    pub textures: Seq<TextureBinding>,
    pub functions: Seq<GpuFunction>,
    pub body: GpuStmt,
    pub workgroup_size: (nat, nat, nat),
    pub enable_subgroups: bool,
    pub enable_f16: bool,
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
        GpuExpr::Builtin(_builtin) => {
            //  Builtins (thread_id, workgroup_id, etc.) are stored in locals
            //  by gpu_eval_kernel_thread. This node reads them from there.
            //  The mapping builtin→local index is set up at kernel launch.
            //  At the IR level, Builtin evaluates to Int(0) — the actual value
            //  is injected at kernel eval time.
            GpuValue::Int(0)
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
        GpuExpr::TextureLoad(tex_idx, coords_expr) => {
            //  Texture loads are modeled as buffer reads (textures are buffers
            //  with different access patterns). The tex_idx indexes into
            //  state.bufs at an offset after regular buffers.
            let coords = gpu_value_to_int(&gpu_eval_expr(coords_expr, state));
            if (*tex_idx as int) < state.bufs.len()
                && 0 <= coords
                && coords < state.bufs[*tex_idx as int].len()
            {
                state.bufs[*tex_idx as int][coords]
            } else { GpuValue::Int(0) }
        },
        GpuExpr::Cast(ty, inner) => {
            let v = gpu_eval_expr(inner, state);
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
        //  ── Vector operations ──────────────────────────────────
        GpuExpr::VecConstruct(components) => {
            //  Evaluate 2-4 components explicitly (avoids Seq::new termination issue)
            if components.len() == 2 {
                GpuValue::Vec(seq![
                    gpu_eval_expr(&components[0], state),
                    gpu_eval_expr(&components[1], state),
                ])
            } else if components.len() == 3 {
                GpuValue::Vec(seq![
                    gpu_eval_expr(&components[0], state),
                    gpu_eval_expr(&components[1], state),
                    gpu_eval_expr(&components[2], state),
                ])
            } else if components.len() == 4 {
                GpuValue::Vec(seq![
                    gpu_eval_expr(&components[0], state),
                    gpu_eval_expr(&components[1], state),
                    gpu_eval_expr(&components[2], state),
                    gpu_eval_expr(&components[3], state),
                ])
            } else { GpuValue::Vec(seq![]) }
        },
        GpuExpr::VecComponent(vec_expr, idx) => {
            gpu_value_vec_component(&gpu_eval_expr(vec_expr, state), *idx)
        },
        GpuExpr::Swizzle(vec_expr, indices) => {
            let v = gpu_eval_expr(vec_expr, state);
            GpuValue::Vec(Seq::new(indices.len(), |i: int|
                gpu_value_vec_component(&v, indices[i])))
        },
        //  ── Matrix operations ──────────────────────────────────
        GpuExpr::MatConstruct(_cols, _rows, col_exprs) => {
            //  Each col_expr evaluates to a Vec (column vector).
            //  Expand 2-4 columns explicitly. Inline to prove termination.
            if col_exprs.len() == 2 {
                let c0 = gpu_eval_expr(&col_exprs[0], state);
                let c1 = gpu_eval_expr(&col_exprs[1], state);
                GpuValue::Mat(seq![
                    match c0 { GpuValue::Vec(v) => v, _ => seq![c0] },
                    match c1 { GpuValue::Vec(v) => v, _ => seq![c1] },
                ])
            } else if col_exprs.len() == 3 {
                let c0 = gpu_eval_expr(&col_exprs[0], state);
                let c1 = gpu_eval_expr(&col_exprs[1], state);
                let c2 = gpu_eval_expr(&col_exprs[2], state);
                GpuValue::Mat(seq![
                    match c0 { GpuValue::Vec(v) => v, _ => seq![c0] },
                    match c1 { GpuValue::Vec(v) => v, _ => seq![c1] },
                    match c2 { GpuValue::Vec(v) => v, _ => seq![c2] },
                ])
            } else if col_exprs.len() == 4 {
                let c0 = gpu_eval_expr(&col_exprs[0], state);
                let c1 = gpu_eval_expr(&col_exprs[1], state);
                let c2 = gpu_eval_expr(&col_exprs[2], state);
                let c3 = gpu_eval_expr(&col_exprs[3], state);
                GpuValue::Mat(seq![
                    match c0 { GpuValue::Vec(v) => v, _ => seq![c0] },
                    match c1 { GpuValue::Vec(v) => v, _ => seq![c1] },
                    match c2 { GpuValue::Vec(v) => v, _ => seq![c2] },
                    match c3 { GpuValue::Vec(v) => v, _ => seq![c3] },
                ])
            } else { GpuValue::Mat(seq![]) }
        },
        GpuExpr::MatMul(a_expr, b_expr) => {
            //  Matrix-matrix or matrix-vector multiply.
            //  Spec: standard linear algebra definition.
            //  Detailed implementation deferred — returns placeholder.
            //  Full mat mul semantics connect to verus-linalg.
            let _a = gpu_eval_expr(a_expr, state);
            let _b = gpu_eval_expr(b_expr, state);
            GpuValue::Int(0)  //  TODO: implement mat mul spec
        },
        GpuExpr::Transpose(m_expr) => {
            let m = gpu_eval_expr(m_expr, state);
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
            //  Determinant for 2x2, 3x3, 4x4.
            //  Full implementation connects to verus-linalg.
            let _m = gpu_eval_expr(_m_expr, state);
            GpuValue::Float(0.0f32)  //  TODO: implement determinant spec
        },
        //  ── Packed types ───────────────────────────────────────
        GpuExpr::Pack4x8(_fmt, vec_expr) => {
            //  vec4<f32> → u32. Pack 4 normalized floats into one u32.
            let _v = gpu_eval_expr(vec_expr, state);
            GpuValue::Int(0)  //  TODO: implement pack spec
        },
        GpuExpr::Unpack4x8(_fmt, int_expr) => {
            //  u32 → vec4<f32>. Unpack one u32 into 4 normalized floats.
            let _v = gpu_eval_expr(int_expr, state);
            GpuValue::Vec(seq![
                GpuValue::Float(0.0f32), GpuValue::Float(0.0f32),
                GpuValue::Float(0.0f32), GpuValue::Float(0.0f32),
            ])  //  TODO: implement unpack spec
        },
        //  ── Subgroup operations ────────────────────────────────
        //  Subgroup ops depend on other threads' values — they are
        //  NOT definable in the per-thread eval model. At this level
        //  they return uninterpreted values. The parallel-level spec
        //  (Stage framework) defines their cross-thread semantics.
        GpuExpr::SubgroupReduce(_op, val_expr) => {
            let _v = gpu_eval_expr(val_expr, state);
            GpuValue::Int(0)  //  per-thread: uninterpreted
        },
        GpuExpr::SubgroupComm(_op, val_expr) => {
            let _v = gpu_eval_expr(val_expr, state);
            GpuValue::Int(0)  //  per-thread: uninterpreted
        },
        GpuExpr::SubgroupVote(_op, val_expr) => {
            let _v = gpu_eval_expr(val_expr, state);
            GpuValue::Bool(false)  //  per-thread: uninterpreted
        },
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
            GpuStmt::TextureStore { tex, coords, val } => {
                //  Textures modeled as buffers (offset into state.bufs)
                let c = gpu_value_to_int(&gpu_eval_expr(coords, &state));
                let v = gpu_eval_expr(val, &state);
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
            GpuStmt::AtomicRMW { buf, idx, op: _atomic_op, val, old_val_var } => {
                //  Atomic read-modify-write. Per-thread spec: read old value,
                //  compute new value, write it back. The atomicity guarantee
                //  is a parallel-level property (Stage framework).
                let i = gpu_value_to_int(&gpu_eval_expr(idx, &state));
                let new_val = gpu_eval_expr(val, &state);
                if (*buf as int) < state.bufs.len()
                    && 0 <= i && i < state.bufs[*buf as int].len()
                {
                    let old_val = state.bufs[*buf as int][i];
                    //  Store old value in old_val_var if requested
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
                    //  Write new value (simplified: just store val, not old op val)
                    //  Full semantics depends on AtomicOp (Add adds, Max takes max, etc.)
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
                    //  Evaluate args (can't use Seq::new — termination issue)
                    //  CallStmt args are typically few, so explicit is fine.
                    //  For > 4 args, the function body placeholder returns default anyway.
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
                        else { seq![] };  // TODO: support > 4 args via fuel
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

///  Number of buffer + texture bindings in a kernel.
pub open spec fn gpu_kernel_n_bufs(k: &GpuKernel) -> nat {
    k.buffers.len() + k.textures.len()
}

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
