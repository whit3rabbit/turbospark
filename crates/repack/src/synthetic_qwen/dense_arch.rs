//! Architecture definition and shape helpers for synthetic Qwen GDN dense test installs.

use model_io::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig,
    ModelFamily, RopeScalingConfig, VisionConfig,
};

pub(crate) const HIDDEN: usize = 128;
pub(crate) const NUM_HEADS: usize = 4;
pub(crate) const HEAD_DIM: usize = 32;
pub(crate) const NUM_KV_HEADS: usize = 2;
/// The DENSE FFN width. Qwen 3.6's fixture calls the same constant the
/// shared-expert-and-routed-expert width; here there are no experts and this
/// is the only FFN there is.
pub(crate) const INTER: usize = 128;

pub(crate) const LA_K_HEADS: usize = 2;
pub(crate) const LA_V_HEADS: usize = 4;
pub(crate) const LA_KEY_DIM: usize = 32;
pub(crate) const LA_VALUE_DIM: usize = 32;
pub(crate) const LA_CONV_K: usize = 4;

/// The checkpoint's group size. Named here rather than imported so the
/// fixture states the shape it is building.
pub(crate) const GROUP: usize = 128;
pub(crate) const DFLASH_LAYERS: usize = 5;
pub(crate) const DFLASH_RANK: usize = 32;

/// The group size each width is published at, which is a TABLE of what real
/// checkpoints carry rather than a rule about narrow quantization.
///
/// `config.rs`'s `is_supported_affine_shape` accepts `(1, 128)`, `(2, 128)`
/// and `(4|8, 64)` as one conjunction, so the cross-products are refused at
/// open -- a 4-bit fixture at 128 is not a slightly-off fixture, it is an
/// install that cannot load.
pub(crate) fn group_for(bits: u32) -> usize {
    match bits {
        1 | 2 => GROUP,
        // `compute::quant::GROUP_SIZE`, which that crate does not re-export.
        // Stated rather than imported for the reason GROUP above is: the
        // fixture declares the shape it builds, and `quantize_int4_affine`
        // asserts the row is a multiple of it, so a disagreement is a
        // panic in the fixture rather than a wrong install.
        _ => 64,
    }
}

/// The companion dtype each width is published at, and the axis that fails
/// SILENTLY: the two are the same width and share no exponent field, so
/// accepting either produces an install of exactly the right size whose
/// scales are wrong by orders of magnitude.
pub(crate) fn companion_dtype(bits: u32) -> &'static str {
    match bits {
        1 | 2 => "F16",
        _ => "BF16",
    }
}

fn linear_attention() -> LinearAttentionConfig {
    LinearAttentionConfig {
        num_k_heads: LA_K_HEADS as i64,
        num_v_heads: LA_V_HEADS as i64,
        key_head_dim: LA_KEY_DIM as i64,
        value_head_dim: LA_VALUE_DIM as i64,
        conv_kernel_size: LA_CONV_K as i64,
    }
}

/// A tiny `qwen3_5`-shaped architecture. Every non-shape field takes
/// [`model_io::qwen_gdn_dense_27b`]'s own value, for the reason `tiny_qwen_gdn_moe_arch`
/// pins Qwen 3.6's: the manifest's optional family extensions fall back to a
/// baseline, so anything else has to be written and matched explicitly.
pub fn tiny_qwen_gdn_dense_arch(vocab_size: i64, num_layers: i64) -> ArchConfig {
    ArchConfig {
        hidden_size: HIDDEN as i64,
        intermediate_size: INTER as i64,
        // Dense: no routed width at all, matching the real baseline.
        moe_intermediate_size: 0,
        num_heads: NUM_HEADS as i64,
        num_kv_heads: NUM_KV_HEADS as i64,
        num_full_kv_heads: NUM_KV_HEADS as i64,
        head_dim: HEAD_DIM as i64,
        full_head_dim: HEAD_DIM as i64,
        vocab_size,
        sliding_window: 0,
        final_logit_softcap: 0.0,
        rope_theta: 10_000_000.0,
        full_rope_theta: 10_000_000.0,
        partial_rotary_factor: 0.25,
        num_layers,
        num_experts: 0,
        top_k_experts: 0,
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        // Layer 0 linear, matching the real 64-layer model; alternating
        // keeps a toy small while exercising both flows and their
        // interleaving.
        full_attention_layer_mask: (0..num_layers)
            .map(|l| if l % 2 == 1 { 1u8 } else { 2u8 })
            .collect(),
        hidden_activation: "silu".to_string(),
        family: ModelFamily::QwenGdnDense,
        attn_output_gate: true,
        // 0.125, not the mathematically-right 32^-0.5: `validate_arch`
        // compares this f64 EXACTLY against the manifest's, and serde_json's
        // default parser is only correct to ~1 ULP -- so a scale that is not
        // a binary fraction cannot survive the round trip (AGENTS.md Gotcha
        // 24). The real baseline's 0.0625 is a power of two and is unaffected.
        attention_scale: 0.125,
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        // Nothing to gate: the FFN is dense.
        shared_expert_gated: false,
        rope_neox_subdim: true,
        linear_attention: linear_attention(),
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: RopeScalingConfig::NONE,
        vision: VisionConfig::NONE,
    }
}
