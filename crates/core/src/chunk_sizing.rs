//! Automatic prompt-processing chunk-size resolution.
//!
//! Turns an automatic chunk-size request into one concrete size drawn from
//! the allowed chunk-size set declared in [`crate::runtime_config`]. The
//! allowed-set literals and the fixed default stay declared exactly once
//! there; this module only selects among them, so there is exactly one
//! place that could disagree with the runtime-configuration boundary.

use crate::runtime_config::{ALLOWED_CHUNK_SIZES, DEFAULT_CHUNK_SIZE};

/// Known or not-yet-known length of the input a chunk size is being chosen
/// for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputLength {
    /// The input length has not been determined yet.
    Unknown,
    /// The input length is known, counted in the same units as the chunk
    /// size (input elements).
    Known(u64),
}

/// Resolve an automatic chunk-size request to one concrete allowed size.
///
/// Three-state resolution:
/// - Known length covered by at least one allowed size: the smallest
///   allowed size that covers it.
/// - Known length that exceeds every allowed size: the largest allowed
///   size.
/// - Length not yet known: the fixed documented default chunk size.
pub fn resolve_automatic_chunk_size(length: InputLength) -> u32 {
    match length {
        InputLength::Unknown => DEFAULT_CHUNK_SIZE,
        InputLength::Known(len) => smallest_covering_size(len).unwrap_or_else(largest_allowed_size),
    }
}

fn smallest_covering_size(len: u64) -> Option<u32> {
    ALLOWED_CHUNK_SIZES
        .iter()
        .copied()
        .find(|&size| u64::from(size) >= len)
}

fn largest_allowed_size() -> u32 {
    *ALLOWED_CHUNK_SIZES
        .iter()
        .max()
        .expect("allowed chunk size set is declared non-empty")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_length_resolves_to_the_fixed_default() {
        assert_eq!(
            resolve_automatic_chunk_size(InputLength::Unknown),
            DEFAULT_CHUNK_SIZE
        );
    }

    #[test]
    fn known_small_length_resolves_to_the_smallest_covering_size() {
        assert_eq!(resolve_automatic_chunk_size(InputLength::Known(1)), 32);
        assert_eq!(resolve_automatic_chunk_size(InputLength::Known(32)), 32);
        assert_eq!(resolve_automatic_chunk_size(InputLength::Known(33)), 64);
    }

    #[test]
    fn length_beyond_every_allowed_size_resolves_to_the_largest() {
        let largest = largest_allowed_size();
        assert_eq!(
            resolve_automatic_chunk_size(InputLength::Known(u64::from(largest) + 1)),
            largest
        );
        assert_eq!(
            resolve_automatic_chunk_size(InputLength::Known(u64::MAX)),
            largest
        );
    }
}
