//! GGUF metadata to `ArchConfig`.
//!
//! The authoritative check is in `gguf_checkpoint_network.rs`, which derives
//! an `ArchConfig` from the real published GGUF and asserts it equals the one
//! this port's own MLX-derived install declares. These tests cover what that
//! one cannot: the failure paths, and the two inversions that would produce a
//! plausible-looking but wrong config.

use model_io::ModelFamily;
use turbospark_repack::{
    arch_from_gguf, build_synthetic_gemma4_gguf, build_synthetic_gpt_oss_gguf, parse_gguf_header,
    GgufBuilder, GgufConfigError, SyntheticGgufShape, SyntheticGptOssShape,
    GGUF_DEFAULT_MAX_HEADER_BYTES,
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

/// A `qwen3moe` header, with the exact keys the published
/// `Qwen3-30B-A3B-Q4_K_M.gguf` carries (read off it by
/// `gguf_checkpoint_network.rs::scopes_qwen3moe_...`), at a fixture's scale.
///
/// Deliberately does NOT publish `attention.sliding_window`,
/// `attention.key_length_swa` or `rope.freq_base_swa`, because the real file
/// does not: this is the single-attention-kind, single-rope-base case, and
/// asserting the derivation against a fixture that invented those keys would
/// test the wrong file.
fn qwen3moe_header() -> Vec<u8> {
    let (bytes, _) = GgufBuilder::new()
        .metadata_str("general.architecture", "qwen3moe")
        .metadata_u32("qwen3moe.block_count", 2)
        .metadata_u32("qwen3moe.embedding_length", 64)
        .metadata_u32("qwen3moe.attention.head_count", 8)
        .metadata_u32("qwen3moe.attention.head_count_kv", 2)
        .metadata_u32("qwen3moe.attention.key_length", 16)
        .metadata_u32("qwen3moe.attention.value_length", 16)
        .metadata_u32("qwen3moe.expert_count", 16)
        .metadata_u32("qwen3moe.expert_used_count", 4)
        .metadata_u32("qwen3moe.expert_feed_forward_length", 32)
        .metadata_u32("qwen3moe.feed_forward_length", 96)
        .metadata_f32("qwen3moe.rope.freq_base", 1_000_000.0)
        // Q8_0 rather than the real file's Q4_K/Q6_K: `arch_from_gguf` reads
        // DIMS and never block types, and a K-quant row cannot be a partial
        // 256-element superblock, so matching the real types here would force
        // a 256-wide hidden and buy nothing.
        .q8_0_tensor("token_embd.weight", &[64, 256], 1)
        .q8_0_tensor("output.weight", &[64, 256], 2)
        .build();
    bytes
}

/// The whole family, off one header: shapes from the file, behaviour from
/// the baseline, and the two head-dim fields agreeing.
#[test]
fn derives_qwen3moe_from_its_own_metadata() {
    let bytes = qwen3moe_header();
    let h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    let a = arch_from_gguf(&h).expect("arch");

    assert_eq!(a.family, ModelFamily::Qwen3Moe);
    assert_eq!(a.num_layers, 2);
    assert_eq!(a.hidden_size, 64);
    assert_eq!(a.num_heads, 8);
    assert_eq!(a.num_kv_heads, 2);
    assert_eq!(a.num_full_kv_heads, 2);
    assert_eq!(a.num_experts, 16);
    assert_eq!(a.top_k_experts, 4);
    // The expert width comes from its OWN key, not from
    // `feed_forward_length`; the `llama` architecture is the one that has to
    // fall back, and taking 96 here would size every routed dispatch wrong.
    assert_eq!(a.moe_intermediate_size, 32);
    assert_eq!(a.intermediate_size, 96);
    assert_eq!(a.vocab_size, 256);
    // `output.weight` present, so the head is untied.
    assert!(!a.tie_word_embeddings);
    // One attention kind, so every layer is mask 1 and both rope bases and
    // both head-dim fields take the single published value. `head_dim`
    // keeping a stale baseline value here is invisible in this family (no
    // sliding layer reads it) and would surface on the next one.
    assert_eq!(a.full_attention_layer_mask, vec![1, 1]);
    assert_eq!(a.rope_theta, 1_000_000.0);
    assert_eq!(a.full_rope_theta, 1_000_000.0);
    assert_eq!(a.head_dim, 16);
    assert_eq!(a.full_head_dim, 16);

    // Behavioural fields, none of which GGUF publishes.
    let baseline = model_io::qwen3_30b_a3b();
    assert_eq!(a.attention_scale, baseline.attention_scale);
    assert_eq!(a.partial_rotary_factor, 1.0);
    assert_eq!(a.hidden_activation, "silu");
    assert_eq!(a.final_logit_softcap, 0.0);
    assert!(!a.ffn_sandwich_norms);
    assert!(!a.attn_output_gate);
    assert!(!a.shared_expert_gated);
    assert!(!a.embedding_scaled_by_sqrt_hidden);
    assert!(!a.rope_neox_subdim);
    assert!(!a.router_scaled);
    assert!(!a.attention_k_eq_v);
}

/// `qwen3moe` and `qwen35moe` are different models that this port runs
/// through different flows, and the strings are one character apart.
#[test]
fn qwen3moe_is_not_the_qwen36_family() {
    let bytes = qwen3moe_header();
    let h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    let a = arch_from_gguf(&h).expect("arch");
    assert_ne!(a.family, ModelFamily::QwenGdnMoe);
    // Qwen 3.6's derivation would demand the `ssm.*` keys this file has
    // none of, so a misrouted family fails loudly rather than silently.
    assert_eq!(a.linear_attention, model_io::LinearAttentionConfig::NONE);
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
    // The exemplar has to be a string with NO family, and it has to be
    // re-picked whenever one gains a flow: this used to be "llama", which
    // ROADMAP Phase M2 made supported. Same maintenance as
    // `names_an_unsupported_but_real_ggml_type`. `phi3` is a registry row
    // with no decode flow; if that ever changes, pick another.
    let (bytes, _) = GgufBuilder::new()
        .metadata_str("general.architecture", "phi3")
        .q8_0_tensor("token_embd.weight", &[64, 128], 1)
        .build();
    let h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    match arch_from_gguf(&h) {
        Err(GgufConfigError::UnsupportedArchitecture { architecture }) => {
            assert_eq!(architecture, "phi3");
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

/// A required `i64` field above `i64::MAX` must be refused rather than
/// wrapped into a plausible-looking negative shape.
#[test]
fn rejects_a_required_field_that_exceeds_i64() {
    let (bytes, _) = build_synthetic_gemma4_gguf(SyntheticGgufShape::default());
    let mut h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    h.metadata.insert(
        "gemma4.block_count".to_string(),
        turbospark_repack::GgufValue::U64(u64::MAX),
    );
    match arch_from_gguf(&h) {
        Err(GgufConfigError::BadValue { key, detail }) => {
            assert_eq!(key, "gemma4.block_count");
            assert!(detail.contains("i64"), "{detail}");
        }
        other => panic!("expected BadValue, got {other:?}"),
    }
}

/// A representable but implausibly large depth must be refused before the
/// gpt-oss mask builder allocates one byte per claimed layer.
#[test]
fn rejects_a_gpt_oss_layer_count_above_the_allocation_bound() {
    let (bytes, _) = build_synthetic_gpt_oss_gguf(SyntheticGptOssShape::default());
    let mut h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    h.metadata.insert(
        "gpt-oss.block_count".to_string(),
        turbospark_repack::GgufValue::U64(4097),
    );
    match arch_from_gguf(&h) {
        Err(GgufConfigError::BadValue { key, detail }) => {
            assert_eq!(key, "gpt-oss.block_count");
            assert!(detail.contains("between 1 and 4096"), "{detail}");
        }
        other => panic!("expected bounded block_count error, got {other:?}"),
    }
}

#[test]
fn rejects_a_zero_layer_model_before_building_its_mask() {
    let (bytes, _) = build_synthetic_gpt_oss_gguf(SyntheticGptOssShape::default());
    let mut h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    h.metadata.insert(
        "gpt-oss.block_count".to_string(),
        turbospark_repack::GgufValue::U32(0),
    );
    match arch_from_gguf(&h) {
        Err(GgufConfigError::BadValue { key, detail }) => {
            assert_eq!(key, "gpt-oss.block_count");
            assert!(detail.contains("between 1 and 4096"), "{detail}");
        }
        other => panic!("expected positive block_count error, got {other:?}"),
    }
}

/// Layer counts size per-layer allocations during import, so malformed GGUF
/// metadata must be rejected before any mask allocation is attempted.
#[test]
fn rejects_llama_block_counts_outside_the_model_limit() {
    for block_count in [0, 4097] {
        let (bytes, _) = GgufBuilder::new()
            .metadata_str("general.architecture", "llama")
            .metadata_u32("llama.block_count", block_count)
            .build();
        let h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
        match arch_from_gguf(&h) {
            Err(GgufConfigError::BadValue { key, detail }) => {
                assert_eq!(key, "llama.block_count");
                assert!(detail.contains("between 1 and 4096"), "{detail}");
            }
            other => panic!("expected bounded block_count error, got {other:?}"),
        }
    }
}

/// A PRESENT optional `i64` field that cannot be read (here: above
/// `i64::MAX`) must be refused, not silently treated as an ABSENT key with
/// its default applied. Collapsing the two would read a corrupt or hostile
/// value as though the checkpoint had simply not published it.
///
/// Targets `attention.sliding_window` rather than `nextn_predict_layers`:
/// the latter is ALSO checked downstream (`mtp_blocks < 0`), so a silently
/// wrapped negative value would still redden that assertion for an
/// unrelated reason and the test could not tell the two checks apart. A
/// wrapped `sliding_window` has no such downstream check -- it would be
/// accepted and stored as -1 with no error at all.
#[test]
fn rejects_an_optional_field_that_is_present_but_exceeds_i64_rather_than_defaulting() {
    let (bytes, _) = build_synthetic_gemma4_gguf(SyntheticGgufShape::default());
    let mut h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    h.metadata.insert(
        "gemma4.attention.sliding_window".to_string(),
        turbospark_repack::GgufValue::U64(u64::MAX),
    );
    match arch_from_gguf(&h) {
        Err(GgufConfigError::BadValue { key, detail }) => {
            assert_eq!(key, "gemma4.attention.sliding_window");
            assert!(detail.contains("i64"), "{detail}");
        }
        other => panic!("expected BadValue naming attention.sliding_window, got {other:?}"),
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
        turbospark_repack::GgufValue::U32(4),
    );
    match arch_from_gguf(&h) {
        Err(GgufConfigError::BadValue { key, .. }) => {
            assert_eq!(key, "gemma4.attention.sliding_window_pattern");
        }
        other => panic!("expected BadValue, got {other:?}"),
    }
}

/// A minimal `qwen35moe` (QwenGdnMoe) header, the family whose
/// `linear_attention` derivation reads the `ssm.*` keys.
fn qwen_gdn_moe_header(inner_size: u32, time_step_rank: u32) -> Vec<u8> {
    let (bytes, _) = GgufBuilder::new()
        .metadata_str("general.architecture", "qwen35moe")
        .metadata_u32("qwen35moe.block_count", 1)
        .metadata_u32("qwen35moe.embedding_length", 64)
        .metadata_u32("qwen35moe.attention.head_count", 4)
        .metadata_u32("qwen35moe.attention.head_count_kv", 2)
        .metadata_u32("qwen35moe.expert_count", 4)
        .metadata_u32("qwen35moe.expert_used_count", 2)
        .metadata_u32("qwen35moe.expert_feed_forward_length", 16)
        .metadata_u32("qwen35moe.full_attention_interval", 4)
        .metadata_u32("qwen35moe.ssm.group_count", 1)
        .metadata_u32("qwen35moe.ssm.time_step_rank", time_step_rank)
        .metadata_u32("qwen35moe.ssm.state_size", 32)
        .metadata_u32("qwen35moe.ssm.inner_size", inner_size)
        .metadata_u32("qwen35moe.ssm.conv_kernel", 2)
        .q8_0_tensor("token_embd.weight", &[64, 256], 1)
        .build();
    bytes
}

/// `value_head_dim` is DERIVED (`ssm.inner_size / ssm.time_step_rank`), not
/// published, so a file whose `inner_size` is not a whole multiple of
/// `time_step_rank` must be refused rather than silently floor-divided into
/// a smaller, plausible-looking head width.
#[test]
fn rejects_an_inner_size_that_is_not_a_whole_multiple_of_the_head_count() {
    let bytes = qwen_gdn_moe_header(130, 4); // 130 / 4 floors to 32, remainder 2
    let h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    match arch_from_gguf(&h) {
        Err(GgufConfigError::BadValue { key, .. }) => {
            assert_eq!(key, "qwen35moe.ssm.inner_size");
        }
        other => panic!("expected BadValue, got {other:?}"),
    }
}

/// The ordinary, evenly-divisible case must still derive correctly.
#[test]
fn derives_value_head_dim_when_inner_size_divides_evenly() {
    let bytes = qwen_gdn_moe_header(128, 4);
    let h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    let a = arch_from_gguf(&h).expect("arch");
    assert_eq!(a.linear_attention.value_head_dim, 32);
}

#[test]
fn rejects_zero_linear_attention_inner_size() {
    let bytes = qwen_gdn_moe_header(0, 4);
    let h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    match arch_from_gguf(&h) {
        Err(GgufConfigError::BadValue { key, detail }) => {
            assert_eq!(key, "qwen35moe.ssm.inner_size");
            assert_eq!(detail, "must be positive");
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

#[test]
fn rejects_an_embedding_whose_width_disagrees_with_metadata() {
    let (bytes, _) = GgufBuilder::new()
        .metadata_str("general.architecture", "llama")
        .metadata_u32("llama.block_count", 1)
        .metadata_u32("llama.embedding_length", 96)
        .metadata_u32("llama.attention.head_count", 8)
        .metadata_u32("llama.feed_forward_length", 128)
        .metadata_f32("llama.rope.freq_base", 10_000.0)
        .q8_0_tensor("token_embd.weight", &[64, 2], 1)
        .build();
    let h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    let error = arch_from_gguf(&h).unwrap_err().to_string();
    assert!(
        error.contains("expected [96, vocab], got [64, 2]"),
        "{error}"
    );
}

#[test]
fn rejects_quantized_embedding_blocks_that_straddle_rows() {
    let (bytes, _) = GgufBuilder::new()
        .metadata_str("general.architecture", "llama")
        .metadata_u32("llama.block_count", 1)
        .metadata_u32("llama.embedding_length", 384)
        .metadata_u32("llama.attention.head_count", 8)
        .metadata_u32("llama.feed_forward_length", 512)
        .metadata_f32("llama.rope.freq_base", 10_000.0)
        // Three complete Q4_K blocks in total, but one and a half per row.
        .tensor("token_embd.weight", 12, &[384, 2], vec![0; 3 * 144])
        .build();
    let h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    let error = arch_from_gguf(&h).unwrap_err().to_string();
    assert!(
        error.contains("row width 384 is not divisible by the type-12 block size 256"),
        "{error}"
    );
}
