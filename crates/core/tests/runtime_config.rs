//! Integration tests for the runtime configuration value-set contract.
//!
//! Covers foundation scenarios test-005, test-006, test-007, test-008, and the
//! value-set edge case. The core crate is referenced by its package name
//! `mrefrust-core` (`mrefrust_core`).

use mrefrust_core::runtime_config::{
    AttentionStrategy, CacheReplacement, HeadProjection, RuntimeConfig, RuntimeConfigBuilder,
    ALLOWED_CACHE_SLOTS, ALLOWED_CHUNK_SIZES, DEFAULT_CACHE_SLOTS, DEFAULT_CHUNK_SIZE,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

#[test]
fn each_allowed_cache_slot_is_accepted() {
    for &value in ALLOWED_CACHE_SLOTS.iter() {
        let cfg = RuntimeConfig::builder().cache_slots(value).build();
        assert_eq!(cfg.cache_slots(), value);
    }
}

#[test]
fn disallowed_cache_slot_aborts_construction() {
    // Values just outside the allowed set plus clearly invalid values.
    let invalid = [0u32, 1, 7, 9, 15, 17, 23, 25, 31, 33, 64, 100, u32::MAX];
    for &value in invalid.iter() {
        let result = catch_unwind(AssertUnwindSafe(|| {
            RuntimeConfig::builder().cache_slots(value).build()
        }));
        assert!(
            result.is_err(),
            "cache slots {value} should have aborted construction",
        );
    }
}

#[test]
fn each_allowed_chunk_size_is_accepted() {
    for &value in ALLOWED_CHUNK_SIZES.iter() {
        let cfg = RuntimeConfig::builder().chunk_size(value).build();
        assert_eq!(cfg.chunk_size(), value);
    }
}

#[test]
fn disallowed_chunk_size_aborts_construction() {
    let invalid = [
        0u32,
        1,
        16,
        31,
        33,
        63,
        65,
        127,
        129,
        1000,
        4095,
        4097,
        8192,
        u32::MAX,
    ];
    for &value in invalid.iter() {
        let result = catch_unwind(AssertUnwindSafe(|| {
            RuntimeConfig::builder().chunk_size(value).build()
        }));
        assert!(
            result.is_err(),
            "chunk size {value} should have aborted construction",
        );
    }
}

#[test]
fn default_config_carries_documented_defaults() {
    let cfg = RuntimeConfig::default();
    assert_eq!(cfg.cache_slots(), DEFAULT_CACHE_SLOTS);
    assert_eq!(cfg.chunk_size(), DEFAULT_CHUNK_SIZE);
    assert_eq!(cfg.cache_replacement(), CacheReplacement::Primary);
    assert!(cfg.prompt_processing_enabled());
    assert_eq!(cfg.attention_strategy(), AttentionStrategy::Standard);
    assert_eq!(cfg.head_projection(), HeadProjection::Combined);
}

#[test]
fn builder_with_no_overrides_matches_default() {
    let built = RuntimeConfigBuilder::new().build();
    assert_eq!(built, RuntimeConfig::default());
}

#[test]
fn identical_overrides_produce_equal_configs() {
    let a = RuntimeConfig::builder()
        .cache_slots(32)
        .chunk_size(2048)
        .cache_replacement(CacheReplacement::Alternate)
        .prompt_processing_enabled(false)
        .attention_strategy(AttentionStrategy::Standard)
        .head_projection(HeadProjection::Separate)
        .build();
    let b = RuntimeConfig::builder()
        .cache_slots(32)
        .chunk_size(2048)
        .cache_replacement(CacheReplacement::Alternate)
        .prompt_processing_enabled(false)
        .attention_strategy(AttentionStrategy::Standard)
        .head_projection(HeadProjection::Separate)
        .build();
    assert_eq!(a, b);
}

#[test]
fn different_overrides_produce_unequal_configs() {
    let a = RuntimeConfig::builder().cache_slots(8).build();
    let b = RuntimeConfig::builder().cache_slots(32).build();
    assert_ne!(a, b);
}

// Guards against a builder setter silently dropping an override: each non-default
// override must be visible through its getter after build, not merely preserve
// equality between two identically built configs.
#[test]
fn non_default_overrides_round_trip_through_getters() {
    let cfg = RuntimeConfig::builder()
        .cache_slots(32)
        .chunk_size(2048)
        .cache_replacement(CacheReplacement::Alternate)
        .prompt_processing_enabled(false)
        .head_projection(HeadProjection::Separate)
        .build();
    assert_eq!(cfg.cache_slots(), 32);
    assert_eq!(cfg.chunk_size(), 2048);
    assert_eq!(cfg.cache_replacement(), CacheReplacement::Alternate);
    assert!(!cfg.prompt_processing_enabled());
    assert_eq!(cfg.head_projection(), HeadProjection::Separate);
}

#[test]
fn primitives_are_re_exported_with_documented_widths() {
    use mrefrust_core::{LogitValue, LogitsView, TokenId};

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
