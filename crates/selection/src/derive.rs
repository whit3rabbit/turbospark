//! Deterministic per-position value derivation from an optional seed.
//!
//! The internal random-number algorithm is explicitly out of scope per the
//! approved spec; only its external contract is fixed: supplying an
//! explicit seed makes selection reproducible, each step position gets its
//! own derived sub-value so a replayed sequence matches element by element,
//! and omitting the seed falls back to a clock-derived value that varies
//! between process runs. A fixed 64-bit mixing step satisfies that contract
//! without adding a random-number dependency.

/// Derive a 64-bit value for one selection step from an optional base seed
/// and the step position.
///
/// When `seed` is `Some`, the same `(seed, position)` pair always derives
/// the same value, and different positions under the same seed derive
/// independent-looking values, so an entire replayed sequence of positions
/// reproduces identically. When `seed` is `None`, a clock-derived value is
/// produced instead, varying between calls.
pub fn derive_step_value(seed: Option<u64>, position: u64) -> u64 {
    match seed {
        Some(base) => mix(base ^ position.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        None => {
            let clock = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);
            mix(clock ^ position)
        }
    }
}

/// A fixed deterministic 64-bit mixing step (splitmix64 finalizer shape).
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Produce a value in `[0.0, 1.0)` from a 64-bit derived value, for use as a
/// uniform categorical draw.
pub fn to_unit_interval(value: u64) -> f64 {
    // Use the top 53 bits so the result is exactly representable as f64.
    (value >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

#[cfg(test)]
mod tests {
    //! Light inline checks; scenario-level determinism coverage lives in the
    //! integration test suite.
    use super::*;

    #[test]
    fn same_seed_and_position_derive_the_same_value() {
        assert_eq!(derive_step_value(Some(7), 3), derive_step_value(Some(7), 3));
    }

    #[test]
    fn different_positions_under_the_same_seed_derive_different_values() {
        assert_ne!(derive_step_value(Some(7), 3), derive_step_value(Some(7), 4));
    }

    #[test]
    fn unit_interval_stays_within_bounds() {
        for position in 0..50 {
            let v = to_unit_interval(derive_step_value(Some(1), position));
            assert!((0.0..1.0).contains(&v));
        }
    }
}
