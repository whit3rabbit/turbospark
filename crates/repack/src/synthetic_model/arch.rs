//! Architecture parameters, specs, and helper functions for synthetic models.

use compute::quantize_int4_affine;
use model_io::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig, MlaConfig,
    ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};

use crate::resident_writer::ResidentTensorSpec;

/// Hidden size, per-head dim, and dense FFN width are all fixed at 64: the
/// smallest value satisfying `turbospark_compute::quant::GROUP_SIZE`'s
/// multiple-of-64 requirement on every GEMV's contraction dimension.
pub const HIDDEN_SIZE: i64 = 64;
/// Number of attention heads in the synthetic model.
pub const NUM_HEADS: i64 = 2;
/// Dimension per attention head (NUM_HEADS * FULL_HEAD_DIM == HIDDEN_SIZE).
pub const FULL_HEAD_DIM: i64 = 32;
/// Intermediate hidden dimension for the feed-forward network.
pub const INTERMEDIATE_SIZE: i64 = 64;

/// A tiny, dense (no MoE, no sliding-window/linear/compressed layers)
/// Gemma-4-shaped architecture. `vocab_size` and `num_layers` are the only
/// caller-chosen dimensions; everything else matches
/// `turbospark_model_io::gemma4_26b_a4b()`'s non-shape fields exactly, since
/// `manifest.json`'s optional family-extension fields fall back to the
/// Gemma 4 baseline's values when omitted (see `arch_validation.rs`).
pub fn tiny_gemma4_arch(vocab_size: i64, num_layers: i64) -> ArchConfig {
    ArchConfig {
        hidden_size: HIDDEN_SIZE,
        intermediate_size: INTERMEDIATE_SIZE,
        moe_intermediate_size: 0,
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_HEADS,
        num_full_kv_heads: NUM_HEADS,
        head_dim: FULL_HEAD_DIM,
        full_head_dim: FULL_HEAD_DIM,
        vocab_size,
        sliding_window: 0,
        final_logit_softcap: 30.0,
        rope_theta: 10_000.0,
        full_rope_theta: 10_000.0,
        partial_rotary_factor: 1.0,
        num_layers,
        dense_lead_intermediate_size: 0,
        num_dense_leading_layers: 0,
        num_experts: 0,
        top_k_experts: 0,
        tie_word_embeddings: true,
        attention_k_eq_v: true,
        full_attention_layer_mask: vec![1u8; num_layers as usize],
        hidden_activation: "gelu_pytorch_tanh".to_string(),
        family: ModelFamily::Gemma4,
        attn_output_gate: false,
        attention_scale: 1.0,
        embedding_scaled_by_sqrt_hidden: true,
        router_scaled: true,
        ffn_sandwich_norms: true,
        shared_expert_gated: false,
        rope_neox_subdim: false,
        linear_attention: LinearAttentionConfig::NONE,
        mla: MlaConfig::NONE,
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: RopeScalingConfig::NONE,
        vision: VisionConfig::NONE,
        ple: PleConfig::NONE,
    }
}

/// A cheap deterministic xorshift stream, seeded per row so every
/// generated tensor is reproducible without relying on any RNG crate.
pub(crate) fn deterministic_row(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(0x9E37_79B9);
    (0..n)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state = state.wrapping_add(i as u64);
            ((state % 2000) as f32 / 1000.0) - 1.0
        })
        .collect()
}

pub(crate) fn quantized_tensor(
    name: &str,
    rows: usize,
    cols: usize,
    seed: u64,
) -> ResidentTensorSpec {
    let mut packed = Vec::with_capacity(rows * cols / 2);
    let mut scales = Vec::with_capacity(rows * cols / 64);
    let mut biases = Vec::with_capacity(rows * cols / 64);
    for r in 0..rows {
        let row = deterministic_row(seed.wrapping_add(r as u64 * 97 + 1), cols);
        let q = quantize_int4_affine(&row);
        packed.extend_from_slice(&q.packed);
        scales.extend_from_slice(&q.scales);
        biases.extend_from_slice(&q.biases);
    }
    ResidentTensorSpec {
        name: name.to_string(),
        packed,
        scales,
        biases,
        rows: rows as u32,
        cols: cols as u32,
    }
}

/// Resident tensor names `RealForwardRunner` looks up by. Kept in sync
/// with `build_synthetic_gemma4_install`'s writes.
pub fn embed_lm_head_name() -> String {
    "embed_lm_head".to_string()
}
/// Returns resident query projection tensor name for a given layer.
pub fn q_proj_name(layer: i64) -> String {
    format!("layer{layer}.q_proj")
}
/// Returns resident key projection tensor name for a given layer.
pub fn k_proj_name(layer: i64) -> String {
    format!("layer{layer}.k_proj")
}
/// Returns resident output projection tensor name for a given layer.
pub fn o_proj_name(layer: i64) -> String {
    format!("layer{layer}.o_proj")
}
/// Returns resident gate projection tensor name for a given layer.
pub fn gate_proj_name(layer: i64) -> String {
    format!("layer{layer}.gate_proj")
}
/// Returns resident up projection tensor name for a given layer.
pub fn up_proj_name(layer: i64) -> String {
    format!("layer{layer}.up_proj")
}
/// Returns resident down projection tensor name for a given layer.
pub fn down_proj_name(layer: i64) -> String {
    format!("layer{layer}.down_proj")
}
/// Returns resident router tensor name for a given layer.
pub fn router_name(layer: i64) -> String {
    format!("layer{layer}.router")
}
/// Returns expert gate projection tensor name for a given layer and expert index.
pub fn expert_gate_proj_name(layer: i64, expert: i64) -> String {
    format!("layer{layer}.expert{expert}.gate_proj")
}
/// Returns expert up projection tensor name for a given layer and expert index.
pub fn expert_up_proj_name(layer: i64, expert: i64) -> String {
    format!("layer{layer}.expert{expert}.up_proj")
}
/// Returns expert down projection tensor name for a given layer and expert index.
pub fn expert_down_proj_name(layer: i64, expert: i64) -> String {
    format!("layer{layer}.expert{expert}.down_proj")
}
