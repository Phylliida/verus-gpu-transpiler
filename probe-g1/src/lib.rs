//! G1 probe (DESIGN.md §5.2, §9): measure how automatically the Lean backend
//! discharges a hand-generated KIR certificate for the add-limbs loop —
//! straight-line step bisimulation + loop induction with state equality.
//!
//! This crate is deliberately spec/proof only: the certificate obligations of
//! the future --emit-gpu pass are proof fns, so the probe measures exactly
//! that surface. Probe simplifications vs production KIR are listed in
//! REPORT.md (no offsets, totalized eval with result-equality only).

pub mod kir;
pub mod addloop;
pub mod certificate;
