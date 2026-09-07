//! Integration tests for the allowed runtime-knob sets.
//!
//! The core crate is referenced by its package name `turbospark-core`
//! (`turbospark_core`).

use turbospark_core::runtime_config::{ALLOWED_CACHE_SLOTS, ALLOWED_CHUNK_SIZES};

#[test]
fn allowed_sets_are_nonempty_positive_and_strictly_increasing() {
    for set in [&ALLOWED_CACHE_SLOTS[..], &ALLOWED_CHUNK_SIZES[..]] {
        assert!(!set.is_empty());
        for &value in set {
            assert!(value > 0);
        }
        for pair in set.windows(2) {
            assert!(
                pair[0] < pair[1],
                "set is not strictly increasing at {pair:?}"
            );
        }
    }
}

#[test]
fn primitives_are_re_exported_with_documented_widths() {
    use turbospark_core::{LogitValue, LogitsView, TokenId};

    let id: TokenId = 0;
    let logit: LogitValue = LogitValue::from_bits(0);
    let _ = (id, logit);

    let empty: [LogitValue; 0] = [];
    let view = LogitsView::new(&empty);
    assert!(view.is_empty());
    assert_eq!(view.len(), 0);

    // Token id interchange width is 32 bits.
    assert_eq!(std::mem::size_of::<TokenId>(), 4);
}
