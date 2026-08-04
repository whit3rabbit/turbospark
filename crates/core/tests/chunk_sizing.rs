//! Integration tests for automatic chunk-size resolution.
//!
//! The core crate is referenced by its package name `core`; within this
//! integration test crate the path `core::` resolves to the dependency
//! because there is no conflicting `extern crate core` here.

use core::chunk_sizing::{resolve_automatic_chunk_size, InputLength};
use core::runtime_config::{ALLOWED_CHUNK_SIZES, DEFAULT_CHUNK_SIZE};

#[test]
fn a_request_made_before_the_length_is_known_uses_the_fixed_default() {
    assert_eq!(
        resolve_automatic_chunk_size(InputLength::Unknown),
        DEFAULT_CHUNK_SIZE
    );
}

#[test]
fn every_allowed_size_is_reachable_as_the_smallest_covering_size() {
    let mut previous_upper_bound = 0u64;
    for &size in ALLOWED_CHUNK_SIZES.iter() {
        let probe = InputLength::Known(previous_upper_bound + 1);
        assert_eq!(
            resolve_automatic_chunk_size(probe),
            size,
            "length {} should resolve to allowed size {size}",
            previous_upper_bound + 1
        );
        previous_upper_bound = u64::from(size);
    }
}

#[test]
fn a_length_exceeding_every_allowed_size_resolves_to_the_largest_allowed_size() {
    let largest = *ALLOWED_CHUNK_SIZES.iter().max().unwrap();
    let probe = InputLength::Known(u64::from(largest) + 1);
    assert_eq!(resolve_automatic_chunk_size(probe), largest);
}

#[test]
fn the_resolved_size_is_always_a_member_of_the_allowed_set() {
    for probe_len in [0u64, 1, 31, 32, 33, 4096, 4097, 1_000_000, u64::MAX] {
        let resolved = resolve_automatic_chunk_size(InputLength::Known(probe_len));
        assert!(
            ALLOWED_CHUNK_SIZES.contains(&resolved),
            "resolved size {resolved} for length {probe_len} is not in the allowed set"
        );
    }
}

#[test]
fn an_exact_boundary_length_resolves_to_that_same_allowed_size() {
    for &size in ALLOWED_CHUNK_SIZES.iter() {
        let probe = InputLength::Known(u64::from(size));
        assert_eq!(resolve_automatic_chunk_size(probe), size);
    }
}
