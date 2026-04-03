///  Well-formedness predicates for GpuIR.
///
///  Structural checks: variable indices in range, buffer indices in range,
///  sub-expressions/statements well-formed. These are preconditions for
///  gpu_eval_expr/gpu_eval_stmt to produce meaningful results.

use vstd::prelude::*;
use crate::gpu_ir::*;

verus! {

//  ══════════════════════════════════════════════════════════════
//  Expression well-formedness
//  ══════════════════════════════════════════════════════════════

///  An expression is well-formed if all variable and buffer indices are in range
///  and all sub-expressions are well-formed.
pub open spec fn gpu_expr_wf(e: &GpuExpr, n_locals: nat, n_bufs: nat) -> bool
    decreases e,
{
    match e {
        GpuExpr::Const(_, _) => true,
        GpuExpr::FConst(_) => true,
        GpuExpr::Var(i, _) => *i < n_locals,
        GpuExpr::Builtin { which: _, local_idx } => *local_idx < n_locals,
        GpuExpr::BinOp(_, a, b) =>
            gpu_expr_wf(a, n_locals, n_bufs)
            && gpu_expr_wf(b, n_locals, n_bufs),
        GpuExpr::UnaryOp(_, a) =>
            gpu_expr_wf(a, n_locals, n_bufs),
        GpuExpr::Select(c, t, f) =>
            gpu_expr_wf(c, n_locals, n_bufs)
            && gpu_expr_wf(t, n_locals, n_bufs)
            && gpu_expr_wf(f, n_locals, n_bufs),
        GpuExpr::ArrayRead(buf_idx, idx_expr) =>
            *buf_idx < n_bufs
            && gpu_expr_wf(idx_expr, n_locals, n_bufs),
        GpuExpr::TextureLoad(tex_idx, coords_expr) =>
            *tex_idx < n_bufs
            && gpu_expr_wf(coords_expr, n_locals, n_bufs),
        GpuExpr::Cast(_, inner) =>
            gpu_expr_wf(inner, n_locals, n_bufs),
        GpuExpr::VecConstruct(components) =>
            components.len() >= 2 && components.len() <= 4
            && forall|i: int| 0 <= i < components.len() ==>
                gpu_expr_wf(&components[i], n_locals, n_bufs),
        GpuExpr::VecComponent(vec_expr, idx) =>
            *idx < 4
            && gpu_expr_wf(vec_expr, n_locals, n_bufs),
        GpuExpr::Swizzle(vec_expr, indices) =>
            indices.len() >= 1 && indices.len() <= 4
            && gpu_expr_wf(vec_expr, n_locals, n_bufs),
        GpuExpr::MatConstruct(cols, rows, col_exprs) =>
            *cols >= 2 && *cols <= 4 && *rows >= 2 && *rows <= 4
            && col_exprs.len() == *cols as int
            && forall|i: int| 0 <= i < col_exprs.len() ==>
                gpu_expr_wf(&col_exprs[i], n_locals, n_bufs),
        GpuExpr::MatMul(a, b) =>
            gpu_expr_wf(a, n_locals, n_bufs)
            && gpu_expr_wf(b, n_locals, n_bufs),
        GpuExpr::Transpose(m) =>
            gpu_expr_wf(m, n_locals, n_bufs),
        GpuExpr::Determinant(m) =>
            gpu_expr_wf(m, n_locals, n_bufs),
        GpuExpr::Pack4x8(_, vec_expr) =>
            gpu_expr_wf(vec_expr, n_locals, n_bufs),
        GpuExpr::Unpack4x8(_, int_expr) =>
            gpu_expr_wf(int_expr, n_locals, n_bufs),
        GpuExpr::SubgroupReduce(_, val) =>
            gpu_expr_wf(val, n_locals, n_bufs),
        GpuExpr::SubgroupComm(_, val) =>
            gpu_expr_wf(val, n_locals, n_bufs),
        GpuExpr::SubgroupVote(_, val) =>
            gpu_expr_wf(val, n_locals, n_bufs),
    }
}

//  ══════════════════════════════════════════════════════════════
//  Statement well-formedness
//  ══════════════════════════════════════════════════════════════

///  A statement is well-formed if all indices are in range and all
///  sub-expressions/statements are well-formed.
pub open spec fn gpu_stmt_wf(
    s: &GpuStmt, n_locals: nat, n_bufs: nat, n_fns: nat,
) -> bool
    decreases s,
{
    match s {
        GpuStmt::Assign { var, rhs } =>
            *var < n_locals
            && gpu_expr_wf(rhs, n_locals, n_bufs),
        GpuStmt::BufWrite { buf, idx, val } =>
            *buf < n_bufs
            && gpu_expr_wf(idx, n_locals, n_bufs)
            && gpu_expr_wf(val, n_locals, n_bufs),
        GpuStmt::TextureStore { tex, coords, val } =>
            *tex < n_bufs
            && gpu_expr_wf(coords, n_locals, n_bufs)
            && gpu_expr_wf(val, n_locals, n_bufs),
        GpuStmt::AtomicRMW { buf, idx, op: _, val, old_val_var } =>
            *buf < n_bufs
            && gpu_expr_wf(idx, n_locals, n_bufs)
            && gpu_expr_wf(val, n_locals, n_bufs)
            && match old_val_var {
                Option::Some(v) => *v < n_locals,
                Option::None => true,
            },
        GpuStmt::CallStmt { fn_id, args, result_var } =>
            *fn_id < n_fns
            && *result_var < n_locals
            && gpu_args_wf(args, n_locals, n_bufs),
        GpuStmt::Seq { first, then } =>
            gpu_stmt_wf(first, n_locals, n_bufs, n_fns)
            && gpu_stmt_wf(then, n_locals, n_bufs, n_fns),
        GpuStmt::If { cond, then_body, else_body } =>
            gpu_expr_wf(cond, n_locals, n_bufs)
            && gpu_stmt_wf(then_body, n_locals, n_bufs, n_fns)
            && gpu_stmt_wf(else_body, n_locals, n_bufs, n_fns),
        GpuStmt::For { var, start, end, body } =>
            *var < n_locals
            && gpu_expr_wf(start, n_locals, n_bufs)
            && gpu_expr_wf(end, n_locals, n_bufs)
            && gpu_stmt_wf(body, n_locals, n_bufs, n_fns),
        GpuStmt::Break => true,
        GpuStmt::Continue => true,
        GpuStmt::Barrier { .. } => true,
        GpuStmt::Return => true,
        GpuStmt::Noop => true,
    }
}

///  Check well-formedness of a Seq of call arguments.
pub open spec fn gpu_args_wf(
    args: &Seq<GpuExpr>, n_locals: nat, n_bufs: nat,
) -> bool {
    forall|i: int| 0 <= i < args.len() ==>
        gpu_expr_wf(&args[i], n_locals, n_bufs)
}

//  ══════════════════════════════════════════════════════════════
//  Function and kernel well-formedness
//  ══════════════════════════════════════════════════════════════

///  A function is well-formed if its body is well-formed within its own
///  local variable space, and parameter/return indices are in range.
pub open spec fn gpu_function_wf(
    f: &GpuFunction, n_bufs: nat, n_fns: nat,
) -> bool {
    &&& f.ret_var < f.n_locals
    &&& forall|i: int| 0 <= i < f.params.len() ==>
            #[trigger] f.params[i].0 < f.n_locals
    &&& gpu_stmt_wf(&f.body, f.n_locals, n_bufs, n_fns)
}

///  A kernel is well-formed if its body and all functions are well-formed.
pub open spec fn gpu_kernel_wf(k: &GpuKernel) -> bool {
    let n_bufs = gpu_kernel_n_bufs(k);
    &&& k.n_locals > 0  //  at least locals[0] for thread ID
    &&& gpu_stmt_wf(&k.body, k.n_locals, n_bufs, k.functions.len())
    &&& forall|i: int| 0 <= i < k.functions.len() ==>
            #[trigger] gpu_function_wf(
                &k.functions[i], n_bufs, k.functions.len())
}

} // verus!
