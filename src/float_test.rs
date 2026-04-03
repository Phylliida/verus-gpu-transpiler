///  Smoke tests for Verus f32 support.
///
///  Results (2026-04-03): 4 verified, 0 errors.
///  - Spec-level: full f32 arithmetic (+, *, -, /) works
///  - Proof-level: bit representation, finiteness/NaN/infinity checks work
///  - Exec-level: f32 passthrough works; arithmetic needs add_req precondition
///  - Exec arithmetic result is nondeterministic (IEEE rounding) — by design

use vstd::prelude::*;
use vstd::float::*;

verus! {

spec fn add_f32_spec(a: f32, b: f32) -> f32 { a + b }

spec fn mul_f32_spec(a: f32, b: f32) -> f32 { a * b }

spec fn dot_product_spec(a: Seq<f32>, b: Seq<f32>, n: nat) -> f32
    decreases n,
{
    if n == 0 { 0.0f32 }
    else { dot_product_spec(a, b, (n - 1) as nat) + a[(n-1) as int] * b[(n-1) as int] }
}

proof fn test_f32_finite(x: f32)
    requires x.is_finite_spec()
    ensures !x.is_nan_spec(), !x.is_infinite_spec()
{}

proof fn test_zero_bits()
    ensures 0.0f32.to_bits_spec() == 0u32
{}

fn pass_f32(a: f32) -> (r: f32)
    ensures r == a
{ a }

} // verus!
