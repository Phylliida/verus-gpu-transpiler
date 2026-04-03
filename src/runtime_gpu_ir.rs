///  Runtime (exec) types for GpuIR — mirrors of the spec types using concrete
///  Rust types (u32, i64, Vec) instead of ghost types (nat, int, Seq).
///
///  Each type has a `view_spec()` that converts to the corresponding spec type,
///  establishing the structural correspondence used by the verified emitter.

use vstd::prelude::*;
use crate::gpu_ir::*;

verus! {

//  ══════════════════════════════════════════════════════════════
//  Runtime scalar/type enums
//  ══════════════════════════════════════════════════════════════

#[derive(Clone, Copy)]
pub enum RtScalarType { I32, U32, F32, F16, Bool }

impl RtScalarType {
    pub open spec fn view_spec(&self) -> ScalarType {
        match self {
            RtScalarType::I32 => ScalarType::I32,
            RtScalarType::U32 => ScalarType::U32,
            RtScalarType::F32 => ScalarType::F32,
            RtScalarType::F16 => ScalarType::F16,
            RtScalarType::Bool => ScalarType::Bool,
        }
    }
}

//  ══════════════════════════════════════════════════════════════
//  Runtime binary operator
//  ══════════════════════════════════════════════════════════════

#[derive(Clone, Copy)]
pub enum RtBinOp {
    Add, Sub, Mul, Div, Mod, Shr, Shl,
    WrappingAdd, WrappingSub, WrappingMul,
    FAdd, FSub, FMul, FDiv,
    Lt, Le, Gt, Ge, Eq, Ne,
    BitAnd, BitOr, BitXor,
    LogicalAnd, LogicalOr,
}

impl RtBinOp {
    pub open spec fn view_spec(&self) -> GpuBinOp {
        match self {
            RtBinOp::Add => GpuBinOp::Add, RtBinOp::Sub => GpuBinOp::Sub,
            RtBinOp::Mul => GpuBinOp::Mul, RtBinOp::Div => GpuBinOp::Div,
            RtBinOp::Mod => GpuBinOp::Mod, RtBinOp::Shr => GpuBinOp::Shr,
            RtBinOp::Shl => GpuBinOp::Shl,
            RtBinOp::WrappingAdd => GpuBinOp::WrappingAdd,
            RtBinOp::WrappingSub => GpuBinOp::WrappingSub,
            RtBinOp::WrappingMul => GpuBinOp::WrappingMul,
            RtBinOp::FAdd => GpuBinOp::FAdd, RtBinOp::FSub => GpuBinOp::FSub,
            RtBinOp::FMul => GpuBinOp::FMul, RtBinOp::FDiv => GpuBinOp::FDiv,
            RtBinOp::Lt => GpuBinOp::Lt, RtBinOp::Le => GpuBinOp::Le,
            RtBinOp::Gt => GpuBinOp::Gt, RtBinOp::Ge => GpuBinOp::Ge,
            RtBinOp::Eq => GpuBinOp::Eq, RtBinOp::Ne => GpuBinOp::Ne,
            RtBinOp::BitAnd => GpuBinOp::BitAnd, RtBinOp::BitOr => GpuBinOp::BitOr,
            RtBinOp::BitXor => GpuBinOp::BitXor,
            RtBinOp::LogicalAnd => GpuBinOp::LogicalAnd,
            RtBinOp::LogicalOr => GpuBinOp::LogicalOr,
        }
    }

    ///  WGSL operator string for this binary op.
    pub fn wgsl_op_str(&self) -> (s: &'static str)
    {
        match self {
            RtBinOp::Add => "+", RtBinOp::Sub => "-",
            RtBinOp::Mul => "*", RtBinOp::Div => "/",
            RtBinOp::Mod => "%", RtBinOp::Shr => ">>",
            RtBinOp::Shl => "<<",
            //  Wrapping: WGSL always wraps, so same ops
            RtBinOp::WrappingAdd => "+", RtBinOp::WrappingSub => "-",
            RtBinOp::WrappingMul => "*",
            RtBinOp::FAdd => "+", RtBinOp::FSub => "-",
            RtBinOp::FMul => "*", RtBinOp::FDiv => "/",
            RtBinOp::Lt => "<", RtBinOp::Le => "<=",
            RtBinOp::Gt => ">", RtBinOp::Ge => ">=",
            RtBinOp::Eq => "==", RtBinOp::Ne => "!=",
            RtBinOp::BitAnd => "&", RtBinOp::BitOr => "|",
            RtBinOp::BitXor => "^",
            RtBinOp::LogicalAnd => "&&", RtBinOp::LogicalOr => "||",
        }
    }
}

//  ══════════════════════════════════════════════════════════════
//  Runtime unary operator
//  ══════════════════════════════════════════════════════════════

#[derive(Clone, Copy)]
pub enum RtUnaryOp { Neg, FNeg, BitNot, LogicalNot }

impl RtUnaryOp {
    pub fn wgsl_op_str(&self) -> (s: &'static str)
    {
        match self {
            RtUnaryOp::Neg => "-",
            RtUnaryOp::FNeg => "-",
            RtUnaryOp::BitNot => "~",
            RtUnaryOp::LogicalNot => "!",
        }
    }
}

//  ══════════════════════════════════════════════════════════════
//  Runtime barrier scope
//  ══════════════════════════════════════════════════════════════

#[derive(Clone, Copy)]
pub enum RtBarrierScope { Workgroup, Storage, Subgroup }

impl RtBarrierScope {
    pub fn wgsl_str(&self) -> (s: &'static str)
    {
        match self {
            RtBarrierScope::Workgroup => "workgroupBarrier()",
            RtBarrierScope::Storage => "storageBarrier()",
            RtBarrierScope::Subgroup => "subgroupBarrier()",
        }
    }
}

//  ══════════════════════════════════════════════════════════════
//  Runtime expression (exec — uses u32, i64, Vec, Box)
//  ══════════════════════════════════════════════════════════════

pub enum RtExpr {
    Const(i64, RtScalarType),
    FConst(u32),                    //  f32 bits — emits as bitcast<f32>(0xNNNNNNNNu)
    Var(u32),                       //  local index
    Builtin(u32),                   //  local_idx (which builtin is metadata)
    BinOp(RtBinOp, Box<RtExpr>, Box<RtExpr>),
    UnaryOp(RtUnaryOp, Box<RtExpr>),
    Select(Box<RtExpr>, Box<RtExpr>, Box<RtExpr>),
    ArrayRead(u32, Box<RtExpr>),    //  buf index
    Cast(RtScalarType, Box<RtExpr>),
    //  Vec/Mat/Texture/Subgroup variants can be added incrementally
}

//  ══════════════════════════════════════════════════════════════
//  Runtime statement
//  ══════════════════════════════════════════════════════════════

pub enum RtStmt {
    Assign { var: u32, rhs: RtExpr },
    BufWrite { buf: u32, idx: RtExpr, val: RtExpr },
    Seq { first: Box<RtStmt>, then: Box<RtStmt> },
    If { cond: RtExpr, then_body: Box<RtStmt>, else_body: Box<RtStmt> },
    For { var: u32, start: RtExpr, end: RtExpr, body: Box<RtStmt> },
    Break,
    Continue,
    Barrier { scope: RtBarrierScope },
    Return,
    Noop,
}

//  ══════════════════════════════════════════════════════════════
//  Runtime buffer declaration
//  ══════════════════════════════════════════════════════════════

pub struct RtBufDecl {
    pub binding: u32,
    pub name: Vec<char>,
    pub read_only: bool,
    pub elem_type: RtScalarType,
}

//  ══════════════════════════════════════════════════════════════
//  Runtime kernel
//  ══════════════════════════════════════════════════════════════

pub struct RtKernel {
    pub name: Vec<char>,
    pub var_names: Vec<Vec<char>>,  //  local variable names for emission
    pub buf_decls: Vec<RtBufDecl>,
    pub body: RtStmt,
    pub workgroup_size: (u32, u32, u32),
    pub builtin_names: Vec<Vec<char>>,  //  e.g., ["gid.x", "gid.y"]
}

} // verus!
