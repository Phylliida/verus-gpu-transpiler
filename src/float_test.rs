use vstd::prelude::*;
use vstd::float::*;

verus! {

/// Test 1: Can we use f32 as a type in spec functions?
spec fn add_f32_spec(a: f32, b: f32) -> f32 {
    a + b
}

/// Test 2: Can we use f32 in exec functions?
fn add_f32_exec(a: f32, b: f32) -> (r: f32)
    requires a.is_finite_spec(), b.is_finite_spec()
{
    a + b
}

/// Test 3: Can we reason about f32 finiteness?
proof fn test_f32_finite(x: f32)
    requires x.is_finite_spec()
    ensures !x.is_nan_spec(), !x.is_infinite_spec()
{
}

/// Test 4: Can we use f32 in arrays/sequences?
spec fn sum_f32_spec(s: Seq<f32>, n: nat) -> f32
    decreases n,
{
    if n == 0 { 0.0f32 }
    else { sum_f32_spec(s, (n - 1) as nat) + s[(n - 1) as int] }
}

} // verus!
