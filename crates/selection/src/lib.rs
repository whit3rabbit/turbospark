//! Selection crate: choose exactly one candidate identifier from a
//! per-candidate score vector under a validated shaping configuration, an
//! accumulated history, and a step position.
//!
//! Numeric parity with any upstream implementation is out of scope; only
//! the observable input, output, ordering, and error contract is exercised.
//! The internal random-number algorithm is likewise out of scope; only its
//! external contract (seeded reproducibility, per-position independence,
//! non-degenerate output) is fixed.

/// Token selection strategy and candidate sampling logic.
pub mod choose;
/// Utility functions for deriving sampling states.
pub mod derive;
/// The materialized shaped distribution and the speculative residual sampler.
pub mod distribution;
/// Repetition and presence penalty applications.
pub mod penalty;
/// Logit distribution shaping, temperature scaling, and selection errors.
pub mod shaping;
/// Top-k, top-p, and min-p truncation filters.
pub mod truncation;

pub use choose::{select, shaped_distribution};
pub use distribution::{residual, ShapedDistribution};
pub use shaping::{SelectionError, ShapingConfig};
