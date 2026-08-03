//! Selection crate: choose exactly one candidate identifier from a
//! per-candidate score vector under a validated shaping configuration, an
//! accumulated history, and a step position.
//!
//! Numeric parity with any upstream implementation is out of scope; only
//! the observable input, output, ordering, and error contract is exercised.
//! The internal random-number algorithm is likewise out of scope; only its
//! external contract (seeded reproducibility, per-position independence,
//! non-degenerate output) is fixed.

pub mod choose;
pub mod derive;
pub mod penalty;
pub mod shaping;
pub mod truncation;

pub use choose::select;
pub use shaping::{SelectionError, ShapingConfig};
