///  Verified WGSL emitter — converts runtime GpuIR to WGSL compute shader text.
///  Fully verified in Verus. No external_body.

use vstd::prelude::*;
use vstd::string::*;
use crate::runtime_gpu_ir::*;
use crate::string_util::*;

verus! {

//  ══════════════════════════════════════════════════════════════
//  Helper: Vec<char> → String
//  ══════════════════════════════════════════════════════════════

///  Convert a Vec<char> name to a String by appending chars one at a time.
fn chars_to_string(chars: &Vec<char>) -> (s: String)
{
    let mut s = String::from_str("");
    let mut i: usize = 0;
    while i < chars.len()
        invariant i <= chars@.len(),
        decreases chars.len() - i,
    {
        //  Build a single-char &str via substring
        //  We use a workaround: collect into the string char by char
        //  Since we can't push single chars, we build tiny strings
        let c = chars[i];
        //  Map common chars to &str literals
        let cs = char_to_str(c);
        s.append(cs);
        i = i + 1;
    }
    s
}

///  Map a char to a single-char &str. Covers ASCII printable range.
fn char_to_str(c: char) -> (s: &'static str)
{
    //  Alphanumeric + common symbols
    if c == 'a' { "a" } else if c == 'b' { "b" } else if c == 'c' { "c" }
    else if c == 'd' { "d" } else if c == 'e' { "e" } else if c == 'f' { "f" }
    else if c == 'g' { "g" } else if c == 'h' { "h" } else if c == 'i' { "i" }
    else if c == 'j' { "j" } else if c == 'k' { "k" } else if c == 'l' { "l" }
    else if c == 'm' { "m" } else if c == 'n' { "n" } else if c == 'o' { "o" }
    else if c == 'p' { "p" } else if c == 'q' { "q" } else if c == 'r' { "r" }
    else if c == 's' { "s" } else if c == 't' { "t" } else if c == 'u' { "u" }
    else if c == 'v' { "v" } else if c == 'w' { "w" } else if c == 'x' { "x" }
    else if c == 'y' { "y" } else if c == 'z' { "z" }
    else if c == 'A' { "A" } else if c == 'B' { "B" } else if c == 'C' { "C" }
    else if c == 'D' { "D" } else if c == 'E' { "E" } else if c == 'F' { "F" }
    else if c == 'G' { "G" } else if c == 'H' { "H" } else if c == 'I' { "I" }
    else if c == 'J' { "J" } else if c == 'K' { "K" } else if c == 'L' { "L" }
    else if c == 'M' { "M" } else if c == 'N' { "N" } else if c == 'O' { "O" }
    else if c == 'P' { "P" } else if c == 'Q' { "Q" } else if c == 'R' { "R" }
    else if c == 'S' { "S" } else if c == 'T' { "T" } else if c == 'U' { "U" }
    else if c == 'V' { "V" } else if c == 'W' { "W" } else if c == 'X' { "X" }
    else if c == 'Y' { "Y" } else if c == 'Z' { "Z" }
    else if c == '0' { "0" } else if c == '1' { "1" } else if c == '2' { "2" }
    else if c == '3' { "3" } else if c == '4' { "4" } else if c == '5' { "5" }
    else if c == '6' { "6" } else if c == '7' { "7" } else if c == '8' { "8" }
    else if c == '9' { "9" }
    else if c == '_' { "_" } else if c == '.' { "." } else if c == ' ' { " " }
    else if c == '(' { "(" } else if c == ')' { ")" }
    else if c == '[' { "[" } else if c == ']' { "]" }
    else if c == '{' { "{" } else if c == '}' { "}" }
    else if c == '<' { "<" } else if c == '>' { ">" }
    else if c == ',' { "," } else if c == ';' { ";" } else if c == ':' { ":" }
    else if c == '+' { "+" } else if c == '-' { "-" } else if c == '*' { "*" }
    else if c == '/' { "/" } else if c == '%' { "%" } else if c == '=' { "=" }
    else if c == '!' { "!" } else if c == '&' { "&" } else if c == '|' { "|" }
    else if c == '^' { "^" } else if c == '~' { "~" } else if c == '@' { "@" }
    else if c == '#' { "#" } else if c == '\n' { "\n" } else if c == '\t' { "\t" }
    else { "?" }
}

//  ══════════════════════════════════════════════════════════════
//  Helper: indent string
//  ══════════════════════════════════════════════════════════════

fn indent_str(depth: u32) -> (s: String)
{
    let mut s = String::from_str("");
    let mut i: u32 = 0;
    while i < depth
        invariant i <= depth,
        decreases depth - i,
    {
        s.append("  ");
        i = i + 1;
    }
    s
}

//  ══════════════════════════════════════════════════════════════
//  Helper: variable name from index
//  ══════════════════════════════════════════════════════════════

fn var_name(names: &Vec<Vec<char>>, idx: u32) -> (s: String)
{
    if (idx as usize) < names.len() {
        chars_to_string(&names[idx as usize])
    } else {
        let mut s = String::from_str("v_");
        let n = u32_to_string(idx);
        s.append(n.as_str());
        s
    }
}

fn buf_name(decls: &Vec<RtBufDecl>, idx: u32) -> (s: String)
{
    if (idx as usize) < decls.len() {
        chars_to_string(&decls[idx as usize].name)
    } else {
        let mut s = String::from_str("buf_");
        let n = u32_to_string(idx);
        s.append(n.as_str());
        s
    }
}

fn scalar_type_str(ty: &RtScalarType) -> (s: &'static str)
{
    match ty {
        RtScalarType::I32 => "i32",
        RtScalarType::U32 => "u32",
        RtScalarType::F32 => "f32",
        RtScalarType::F16 => "f16",
        RtScalarType::Bool => "bool",
    }
}

//  ══════════════════════════════════════════════════════════════
//  Expression emission
//  ══════════════════════════════════════════════════════════════

pub fn emit_expr(e: &RtExpr, var_names: &Vec<Vec<char>>, buf_decls: &Vec<RtBufDecl>) -> (s: String)
    decreases *e,
{
    match e {
        RtExpr::Const(val, ty) => {
            match ty {
                RtScalarType::F32 => {
                    let mut s = i64_to_string(*val);
                    s.append(".0f");
                    s
                },
                RtScalarType::I32 => {
                    let mut s = String::from_str("i32(");
                    let n = i64_to_string(*val);
                    s.append(n.as_str());
                    s.append(")");
                    s
                },
                _ => {
                    //  u32 default
                    let mut s = i64_to_string(*val);
                    s.append("u");
                    s
                },
            }
        },
        RtExpr::FConst(bits) => {
            //  Emit float as bitcast from exact bit representation
            let mut s = String::from_str("bitcast<f32>(");
            let hex = u32_to_hex(*bits);
            s.append(hex.as_str());
            s.append("u)");
            s
        },
        RtExpr::Var(idx) => var_name(var_names, *idx),
        RtExpr::Builtin(local_idx) => var_name(var_names, *local_idx),
        RtExpr::BinOp(op, a, b) => {
            let mut s = String::from_str("(");
            let left = emit_expr(a, var_names, buf_decls);
            s.append(left.as_str());
            s.append(" ");
            s.append(op.wgsl_op_str());
            s.append(" ");
            let right = emit_expr(b, var_names, buf_decls);
            s.append(right.as_str());
            s.append(")");
            s
        },
        RtExpr::UnaryOp(op, a) => {
            let mut s = String::from_str("(");
            s.append(op.wgsl_op_str());
            let inner = emit_expr(a, var_names, buf_decls);
            s.append(inner.as_str());
            s.append(")");
            s
        },
        RtExpr::Select(cond, t, f) => {
            let mut s = String::from_str("select(");
            let fs = emit_expr(f, var_names, buf_decls);
            s.append(fs.as_str());
            s.append(", ");
            let ts = emit_expr(t, var_names, buf_decls);
            s.append(ts.as_str());
            s.append(", ");
            let cs = emit_expr(cond, var_names, buf_decls);
            s.append(cs.as_str());
            s.append(")");
            s
        },
        RtExpr::ArrayRead(buf_idx, idx_expr) => {
            let mut s = buf_name(buf_decls, *buf_idx);
            s.append("[");
            let idx_s = emit_expr(idx_expr, var_names, buf_decls);
            s.append(idx_s.as_str());
            s.append("]");
            s
        },
        RtExpr::Cast(ty, inner) => {
            let mut s = String::from_str(scalar_type_str(ty));
            s.append("(");
            let inner_s = emit_expr(inner, var_names, buf_decls);
            s.append(inner_s.as_str());
            s.append(")");
            s
        },
    }
}

//  ══════════════════════════════════════════════════════════════
//  Statement emission
//  ══════════════════════════════════════════════════════════════

pub fn emit_stmt(
    s: &RtStmt, var_names: &Vec<Vec<char>>, buf_decls: &Vec<RtBufDecl>, depth: u32,
) -> (out: String)
    requires depth < 1000,  //  nesting limit — recursive calls use depth+1 ≤ 1000
    decreases *s,
{
    match s {
        RtStmt::Assign { var, rhs } => {
            let mut out = indent_str(depth);
            let name = var_name(var_names, *var);
            out.append(name.as_str());
            out.append(" = ");
            let rhs_s = emit_expr(rhs, var_names, buf_decls);
            out.append(rhs_s.as_str());
            out.append(";\n");
            out
        },
        RtStmt::BufWrite { buf, idx, val } => {
            let mut out = indent_str(depth);
            let bn = buf_name(buf_decls, *buf);
            out.append(bn.as_str());
            out.append("[");
            let idx_s = emit_expr(idx, var_names, buf_decls);
            out.append(idx_s.as_str());
            out.append("] = ");
            let val_s = emit_expr(val, var_names, buf_decls);
            out.append(val_s.as_str());
            out.append(";\n");
            out
        },
        RtStmt::Seq { first, then } => {
            let mut out = emit_stmt(first, var_names, buf_decls, depth);
            let then_s = emit_stmt(then, var_names, buf_decls, depth);
            out.append(then_s.as_str());
            out
        },
        RtStmt::If { cond, then_body, else_body } => {
            let mut out = indent_str(depth);
            out.append("if (");
            let cond_s = emit_expr(cond, var_names, buf_decls);
            out.append(cond_s.as_str());
            out.append(") {\n");
            let then_s = emit_stmt(then_body, var_names, buf_decls, if depth < 999 { depth + 1 } else { depth });
            out.append(then_s.as_str());
            let pad = indent_str(depth);
            out.append(pad.as_str());
            out.append("} else {\n");
            let else_s = emit_stmt(else_body, var_names, buf_decls, if depth < 999 { depth + 1 } else { depth });
            out.append(else_s.as_str());
            let pad2 = indent_str(depth);
            out.append(pad2.as_str());
            out.append("}\n");
            out
        },
        RtStmt::For { var, start, end, body } => {
            let mut out = indent_str(depth);
            let vn = var_name(var_names, *var);
            out.append("for (var ");
            out.append(vn.as_str());
            out.append(": u32 = ");
            let start_s = emit_expr(start, var_names, buf_decls);
            out.append(start_s.as_str());
            out.append("; ");
            let vn2 = var_name(var_names, *var);
            out.append(vn2.as_str());
            out.append(" < ");
            let end_s = emit_expr(end, var_names, buf_decls);
            out.append(end_s.as_str());
            out.append("; ");
            let vn3 = var_name(var_names, *var);
            out.append(vn3.as_str());
            out.append("++) {\n");
            let body_s = emit_stmt(body, var_names, buf_decls, if depth < 999 { depth + 1 } else { depth });
            out.append(body_s.as_str());
            let pad = indent_str(depth);
            out.append(pad.as_str());
            out.append("}\n");
            out
        },
        RtStmt::Break => {
            let mut out = indent_str(depth);
            out.append("break;\n");
            out
        },
        RtStmt::Continue => {
            let mut out = indent_str(depth);
            out.append("continue;\n");
            out
        },
        RtStmt::Barrier { scope } => {
            let mut out = indent_str(depth);
            out.append(scope.wgsl_str());
            out.append(";\n");
            out
        },
        RtStmt::Return => {
            let mut out = indent_str(depth);
            out.append("return;\n");
            out
        },
        RtStmt::Noop => String::from_str(""),
    }
}

//  ══════════════════════════════════════════════════════════════
//  Kernel emission
//  ══════════════════════════════════════════════════════════════

pub fn emit_kernel(k: &RtKernel) -> (out: String)
{
    let mut out = String::from_str("");

    //  Buffer declarations
    let mut bi: usize = 0;
    while bi < k.buf_decls.len()
        invariant bi <= k.buf_decls@.len(),
        decreases k.buf_decls.len() - bi,
    {
        let buf = &k.buf_decls[bi];
        out.append("@group(0) @binding(");
        let bn = u32_to_string(buf.binding);
        out.append(bn.as_str());
        out.append(") var<storage, ");
        if buf.read_only { out.append("read"); } else { out.append("read_write"); }
        out.append("> ");
        let name = chars_to_string(&buf.name);
        out.append(name.as_str());
        out.append(": array<");
        out.append(scalar_type_str(&buf.elem_type));
        out.append(">;\n");
        bi = bi + 1;
    }
    out.append("\n");

    //  Entry point
    out.append("@compute @workgroup_size(");
    let ws0 = u32_to_string(k.workgroup_size.0);
    out.append(ws0.as_str());
    out.append(", ");
    let ws1 = u32_to_string(k.workgroup_size.1);
    out.append(ws1.as_str());
    out.append(", ");
    let ws2 = u32_to_string(k.workgroup_size.2);
    out.append(ws2.as_str());
    out.append(")\n");

    out.append("fn ");
    let kname = chars_to_string(&k.name);
    out.append(kname.as_str());
    out.append("(\n  @builtin(global_invocation_id) gid: vec3<u32>,\n) {\n");

    //  Builtin variable extraction
    let mut bui: usize = 0;
    while bui < k.builtin_names.len()
        invariant bui <= k.builtin_names@.len(),
        decreases k.builtin_names.len() - bui,
    {
        out.append("  let ");
        let vn = var_name(&k.var_names, bui as u32);
        out.append(vn.as_str());
        out.append(" = ");
        let bn = chars_to_string(&k.builtin_names[bui]);
        out.append(bn.as_str());
        out.append(";\n");
        bui = bui + 1;
    }

    //  Body
    let body_s = emit_stmt(&k.body, &k.var_names, &k.buf_decls, 1);
    out.append(body_s.as_str());

    out.append("}\n");
    out
}

} // verus!
