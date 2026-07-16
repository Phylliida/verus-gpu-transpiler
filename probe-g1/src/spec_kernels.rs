//! Certificate targets: the kernels' functional specs.
//!
//! These are verbatim copies of protolith's spec fns (material-synthesis/
//! protolith/src/slab.rs) so the probe stays single-crate; the production
//! generator references the originals cross-crate instead. The trust
//! argument: protolith's exec fns already carry `ensures result == <these>`
//! (verified 12/0), so a certificate `keval(kir) == <these>` completes the
//! chain exec == spec == KIR with no SST reflection.
//!
//! spec_sum_sq3 is named here because protolith states sum_sq3's spec inline
//! in its ensures; same formula.

use vstd::prelude::*;

verus! {

/// Toroidal separation along one wrapped axis (protolith spec_wrap_delta).
pub open spec fn spec_wrap_delta(a: int, b: int, n: int) -> int {
    let d = if a >= b { a - b } else { b - a };
    if d <= n - d { d } else { n - d }
}

/// Open-axis separation (protolith spec_abs_delta).
pub open spec fn spec_abs_delta(a: int, b: int) -> int {
    if a >= b { a - b } else { b - a }
}

/// Sum of three squares (protolith sum_sq3's inline ensures formula).
pub open spec fn spec_sum_sq3(dx: int, dy: int, dz: int) -> int {
    dx * dx + dy * dy + dz * dz
}

} // verus!
