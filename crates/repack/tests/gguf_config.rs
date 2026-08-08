//! GGUF metadata to `ArchConfig`.
//!
//! The authoritative check is in `gguf_checkpoint_network.rs`, which derives
//! an `ArchConfig` from the real published GGUF and asserts it equals the one
//! this port's own MLX-derived install declares. These tests cover what that
//! one cannot: the failure paths, and the two inversions that would produce a
//! plausible-looking but wrong config.

use model_io::ModelFamily;
use mrefrust_repack::{
    arch_from_gguf, build_synthetic_gemma4_gguf, parse_gguf_header, GgufBuilder, GgufConfigError,
    SyntheticGgufShape, GGUF_DEFAULT_MAX_HEADER_BYTES,
};

fn arch_of(shape: SyntheticGgufShape) -> model_io::ArchConfig {
    let (bytes, _) = build_synthetic_gemma4_gguf(shape);
    let h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).expect("parse");
    arch_from_gguf(&h).expect("arch")
}

#[test]
fn reads_the_shape_fields_off_the_file() {
    let s = SyntheticGgufShape::default();
    let a = arch_of(s);

    assert_eq!(a.family, ModelFamily::Gemma4);
    assert_eq!(a.num_layers, s.num_layers as i64);
    assert_eq!(a.hidden_size, s.hidden as i64);
    assert_eq!(a.num_heads, s.num_heads as i64);
    assert_eq!(a.num_experts, s.num_experts as i64);
    assert_eq!(a.top_k_experts, s.top_k as i64);
    assert_eq!(a.moe_intermediate_size, s.moe_intermediate as i64);
    assert_eq!(a.intermediate_size, s.intermediate as i64);
    assert_eq!(a.sliding_window, s.sliding_window as i64);
    assert_eq!(a.head_dim, s.head_dim as i64);
    assert_eq!(a.full_head_dim, s.full_head_dim as i64);
    // From the embedding tensor, not from the tokenizer's token list.
    assert_eq!(a.vocab_size, s.vocab as i64);
    // Gemma ships no output.weight.
    assert!(a.tie_word_embeddings);
}

/// GGUF says "true means this layer slides"; the mask says "1 means full
/// attention". Getting the polarity backwards swaps every layer's attention
/// kind AND its rope base, and still produces a structurally valid config.
#[test]
fn inverts_the_sliding_window_pattern_into_the_layer_mask() {
    let a = arch_of(SyntheticGgufShape::default());
    // The fixture slides on every layer but the last.
    assert_eq!(a.full_attention_layer_mask, vec![0, 1]);
}

/// `freq_base` is the GLOBAL layers' base and `freq_base_swa` the sliding
/// ones'; this port's field names read the other way round, so a
/// same-looking-name assignment swaps them.
#[test]
fn assigns_the_two_rope_bases_to_the_right_fields() {
    let a = arch_of(SyntheticGgufShape::default());
    assert_eq!(a.rope_theta, 10_000.0, "sliding layers");
    assert_eq!(a.full_rope_theta, 1_000_000.0, "global layers");
}

/// Gemma publishes `head_count_kv` as a per-layer array because its two
/// layer kinds differ. The right entry has to be picked per kind.
#[test]
fn resolves_per_layer_kv_head_counts_by_layer_kind() {
    let s = SyntheticGgufShape::default();
    let a = arch_of(s);
    assert_eq!(a.num_kv_heads, s.num_kv_heads as i64);
    assert_eq!(a.num_full_kv_heads, s.num_full_kv_heads as i64);
    assert_ne!(a.num_kv_heads, a.num_full_kv_heads, "fixture must differ");
}

/// Behavioral fields are absent from GGUF metadata because llama.cpp
/// hardcodes them in its graph builder. They must come from the family
/// baseline untouched, NOT be invented.
#[test]
fn takes_behavioural_fields_from_the_family_baseline() {
    let a = arch_of(SyntheticGgufShape::default());
    let baseline = model_io::gemma4_26b_a4b();

    assert_eq!(a.hidden_activation, baseline.hidden_activation);
    assert_eq!(a.attention_scale, baseline.attention_scale);
    assert_eq!(a.attn_output_gate, baseline.attn_output_gate);
    assert_eq!(a.ffn_sandwich_norms, baseline.ffn_sandwich_norms);
    assert_eq!(a.router_scaled, baseline.router_scaled);
    assert_eq!(
        a.embedding_scaled_by_sqrt_hidden,
        baseline.embedding_scaled_by_sqrt_hidden
    );
    assert_eq!(a.router_scoring_func, baseline.router_scoring_func);
    // The worked example from the module doc: GGUF's rope.dimension_count
    // equals the head dim, so reading a rotary fraction off it would give
    // 1.0 where the truth is 0.25.
    assert_eq!(a.partial_rotary_factor, baseline.partial_rotary_factor);
    assert_ne!(a.partial_rotary_factor, 1.0);
}

#[test]
fn rejects_a_file_with_no_architecture() {
    let (bytes, _) = GgufBuilder::new()
        .q8_0_tensor("token_embd.weight", &[64, 128], 1)
        .build();
    let h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    assert!(matches!(
        arch_from_gguf(&h),
        Err(GgufConfigError::MissingArchitecture)
    ));
}

#[test]
fn rejects_an_architecture_with_no_family() {
    let (bytes, _) = GgufBuilder::new()
        .metadata_str("general.architecture", "llama")
        .q8_0_tensor("token_embd.weight", &[64, 128], 1)
        .build();
    let h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    match arch_from_gguf(&h) {
        Err(GgufConfigError::UnsupportedArchitecture { architecture }) => {
            assert_eq!(architecture, "llama");
        }
        other => panic!("expected UnsupportedArchitecture, got {other:?}"),
    }
}

#[test]
fn rejects_a_missing_shape_key() {
    let (bytes, _) = GgufBuilder::new()
        .metadata_str("general.architecture", "gemma4")
        .q8_0_tensor("token_embd.weight", &[64, 128], 1)
        .build();
    let h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    match arch_from_gguf(&h) {
        Err(GgufConfigError::MissingKey { key }) => assert_eq!(key, "gemma4.block_count"),
        other => panic!("expected MissingKey, got {other:?}"),
    }
}

/// A pattern whose length disagrees with `block_count` must be an error,
/// not a silently truncated or zero-extended mask.
#[test]
fn rejects_a_layer_pattern_of_the_wrong_length() {
    let (bytes, _) = build_synthetic_gemma4_gguf(SyntheticGgufShape::default());
    let mut h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    // Claim four layers while the pattern still describes two.
    h.metadata.insert(
        "gemma4.block_count".to_string(),
        mrefrust_repack::GgufValue::U32(4),
    );
    match arch_from_gguf(&h) {
        Err(GgufConfigError::BadValue { key, .. }) => {
            assert_eq!(key, "gemma4.attention.sliding_window_pattern");
        }
        other => panic!("expected BadValue, got {other:?}"),
    }
}

#[test]
fn rejects_a_file_with_no_embedding_tensor() {
    let (bytes, _) = GgufBuilder::new()
        .metadata_str("general.architecture", "gemma4")
        .metadata_u32("gemma4.block_count", 1)
        .metadata_u32("gemma4.embedding_length", 64)
        .metadata_u32("gemma4.attention.head_count", 4)
        .metadata_u32("gemma4.expert_count", 4)
        .metadata_u32("gemma4.expert_used_count", 2)
        .metadata_u32("gemma4.expert_feed_forward_length", 16)
        .q8_0_tensor("output_norm.weight", &[64], 1)
        .build();
    let h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    match arch_from_gguf(&h) {
        Err(GgufConfigError::MissingTensor { name }) => assert_eq!(name, "token_embd.weight"),
        other => panic!("expected MissingTensor, got {other:?}"),
    }
}
