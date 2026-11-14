//! Granite 4.x's GGUF intake, offline. The authoritative checks are in
//! `granite_gguf_network.rs` (real bytes); these cover the registry, the
//! derivation, and the refusals for what the Llama flow cannot express.

use model_io::ModelFamily;
use turbospark_repack::{
    arch_from_gguf, family_for_architecture, parse_gguf_header, GgufBuilder, GgufConfigError,
    GGUF_DEFAULT_MAX_HEADER_BYTES,
};

fn builder(eps: f32, scales: [Option<f32>; 3], attn_scale: Option<f32>) -> GgufBuilder {
    let mut b = GgufBuilder::new()
        .metadata_str("general.architecture", "granite")
        .metadata_u32("granite.block_count", 2)
        .metadata_u32("granite.embedding_length", 128)
        .metadata_u32("granite.feed_forward_length", 256)
        .metadata_u32("granite.attention.head_count", 2)
        .metadata_u32("granite.attention.head_count_kv", 1)
        .metadata_f32("granite.rope.freq_base", 10_000_000.0)
        .metadata_f32("granite.attention.layer_norm_rms_epsilon", eps)
        .metadata_u32("granite.rope.dimension_count", 64);
    for (key, v) in ["embedding_scale", "residual_scale", "logit_scale"]
        .iter()
        .zip(scales)
    {
        if let Some(v) = v {
            b = b.metadata_f32(&format!("granite.{key}"), v);
        }
    }
    if let Some(s) = attn_scale {
        b = b.metadata_f32("granite.attention.scale", s);
    }
    b.q8_0_tensor("token_embd.weight", &[128, 256], 1)
        .q8_0_tensor("output.weight", &[128, 256], 2)
}

fn arch(b: GgufBuilder) -> Result<model_io::ArchConfig, GgufConfigError> {
    let (bytes, _) = b.build();
    let h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).expect("parse");
    arch_from_gguf(&h)
}

const ONES: [Option<f32>; 3] = [Some(1.0), Some(1.0), Some(1.0)];

#[test]
fn granite_is_the_llama_flow_with_the_files_attention_scale() {
    assert_eq!(family_for_architecture("granite"), Some(ModelFamily::Llama));
    let a = arch(builder(1e-5, ONES, Some(0.015625))).expect("derives");
    assert_eq!(a.family, ModelFamily::Llama);
    // 1/head_dim, not Llama's 1/sqrt(head_dim).
    assert_eq!(a.attention_scale, 0.015625);
    assert_eq!(a.head_dim, 64);
    assert!(!a.tie_word_embeddings);
    assert_eq!(a.num_experts, 0);
    assert_eq!(a.full_attention_layer_mask, vec![1, 1]);
}

#[test]
fn a_missing_scalar_means_identity_and_a_missing_attention_scale_is_refused() {
    assert!(arch(builder(1e-5, [None, None, None], Some(0.125))).is_ok());
    match arch(builder(1e-5, ONES, None)) {
        Err(GgufConfigError::MissingKey { key }) => assert_eq!(key, "granite.attention.scale"),
        other => panic!("expected MissingKey, got {other:?}"),
    }
}

#[test]
fn non_identity_multipliers_are_refused_not_decoded_fluently_wrong() {
    // Granite 3.x: embedding 12.0, residual 0.22, logits 16.0.
    for (i, key) in ["embedding_scale", "residual_scale", "logit_scale"]
        .iter()
        .enumerate()
    {
        let mut scales = ONES;
        scales[i] = Some([12.0, 0.22, 16.0][i]);
        match arch(builder(1e-5, scales, Some(0.0078125))) {
            Err(GgufConfigError::BadValue { key: k, .. }) => {
                assert_eq!(k, format!("granite.{key}"))
            }
            other => panic!("{key}: expected BadValue, got {other:?}"),
        }
    }
}

#[test]
fn another_epsilon_is_refused() {
    assert!(matches!(
        arch(builder(1e-6, ONES, Some(0.015625))),
        Err(GgufConfigError::BadValue { .. })
    ));
}
