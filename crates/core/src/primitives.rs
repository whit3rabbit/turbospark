//! Shared primitives exchanged across the generation crates.
//!
//! Half-precision logits are backed by the maintained `half` crate rather than
//! the unstable native `f16` type, so the destination does not hand-roll
//! IEEE-754 binary16 storage or arithmetic.

use half::f16 as Half;

/// Token id interchange type. Signed 32-bit integer to match the downstream
/// buffer element width.
pub type TokenId = i32;

/// Half-precision logit element (IEEE-754 binary16).
pub type LogitValue = Half;

/// A view over a contiguous run of half-precision logits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogitsView<'a> {
    values: &'a [LogitValue],
}

impl<'a> LogitsView<'a> {
    /// Wrap a slice of logits as a view.
    pub fn new(values: &'a [LogitValue]) -> Self {
        Self { values }
    }

    /// The underlying logits slice.
    pub fn as_slice(&self) -> &[LogitValue] {
        self.values
    }

    /// Number of logits in the view.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the view contains no logits.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}
