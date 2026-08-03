//! Destination-selected compute strategy (skeleton).
//!
//! Concrete forward-pass and per-layer attention compute are designed
//! independently and land in later slices. Numerics parity with any upstream
//! implementation is explicitly out of scope; only the structural contracts of
//! the decode and prefill areas are exercised.
//!
//! The `core` dependency is brought in under the alias `foundation` to avoid
//! colliding with the standard library `core` crate in the extern prelude.

/// Marker for the destination-selected compute strategy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ComputeStrategy {
    _private: (),
}

impl ComputeStrategy {
    /// Construct the default compute strategy.
    pub fn new() -> Self {
        Self::default()
    }
}

// Token id width consumed from the core primitives, keeping the dependency
// edge live and documenting the interchange type used in later slices.
pub use foundation::TokenId;
