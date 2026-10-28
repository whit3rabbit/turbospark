//! Builds a tiny `lfm3` install through the REAL GGUF walk.
//!
//! The spark fixture's approach, for a different reason: `lfm3` HAS an MLX
//! intake, but the GGUF walk is the one with family-specific logic worth
//! driving end to end -- the per-layer `head_count_kv` mask builder (0 marks
//! a short-conv layer), the `shortconv.*` name table, and the
//! `rope_freqs.weight` value check, which refuses the whole install unless
//! the vector is the exact zero-tail pattern. The derived `ArchConfig` and
//! the manifest come from the walk itself, so the returned arch is what a
//! test must pass to `RealForwardRunner::open`.
//!
//! Shapes: `HEAD_DIM` 64 with the real 0.5 partial factor gives 16 rotated
//! pairs of 32, both even; a small nonzero `SLIDING_WINDOW` so the KV ring
//! wraps; the real 7 conv taps. `attention_scale` is the BASELINE's
//! `128^-0.5` rather than this head_dim's, for the reason the spark fixture
//! records: no GGUF key carries it, so the walk takes the baseline's value
//! and the fixture must agree. The per-class thetas are invented, positive
//! and different.

use model_io::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig, MlaConfig,
    ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};

use crate::gguf_header::{parse_header as parse_gguf_header, GgufValue, DEFAULT_MAX_HEADER_BYTES};
use crate::ranged_download::MemoryRangeSource;
use crate::synthetic_gguf::GgufBuilder;
use crate::write_gguf_install_streamed;

const HIDDEN: usize = 128;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = 64;
const NUM_KV_HEADS: usize = 2;
const INTER: usize = 128;
const TAPS: usize = 7;
const SLIDING_WINDOW: i64 = 8;
const ROPE_THETA_SWA: f64 = 1_000.0;
const ROPE_THETA_FULL: f64 = 5_000.0;
const PARTIAL_ROTARY_FACTOR: f64 = 0.5;
const GGML_TYPE_F32: u32 = 0;

/// The real checkpoint's layer pattern as a rule: layers 0 and 1 and every
/// odd layer are short-conv, `i % 4 == 2` is full attention, the rest
/// slide. At 24 layers this IS `model_io::lfm3_layer_mask()` (asserted in
/// the tests below), so a fixture of any depth keeps the real phase.
pub fn tiny_lfm3_layer_mask(num_layers: i64) -> Vec<u8> {
    (0..num_layers)
        .map(|i| {
            if i < 2 || i % 2 == 1 {
                6u8
            } else if i % 4 == 2 {
                1
            } else {
                0
            }
        })
        .collect()
}

/// A tiny `lfm3`-shaped architecture. Every non-shape field takes
/// [`model_io::lfm3_1_7b`]'s own value.
pub fn tiny_lfm3_arch(vocab_size: i64, num_layers: i64) -> ArchConfig {
    assert!(num_layers >= 3, "lfm3 needs a conv and an attention layer");
    ArchConfig {
        hidden_size: HIDDEN as i64,
        intermediate_size: INTER as i64,
        moe_intermediate_size: 0,
        num_heads: NUM_HEADS as i64,
        num_kv_heads: NUM_KV_HEADS as i64,
        num_full_kv_heads: NUM_KV_HEADS as i64,
        head_dim: HEAD_DIM as i64,
        full_head_dim: HEAD_DIM as i64,
        vocab_size,
        sliding_window: SLIDING_WINDOW,
        final_logit_softcap: 0.0,
        rope_theta: ROPE_THETA_SWA,
        full_rope_theta: ROPE_THETA_FULL,
        partial_rotary_factor: PARTIAL_ROTARY_FACTOR,
        num_layers,
        dense_lead_intermediate_size: 0,
        num_dense_leading_layers: 0,
        num_experts: 0,
        top_k_experts: 0,
        tie_word_embeddings: true,
        attention_k_eq_v: false,
        full_attention_layer_mask: tiny_lfm3_layer_mask(num_layers),
        hidden_activation: "silu".to_string(),
        family: ModelFamily::Lfm3,
        attn_output_gate: false,
        attention_scale: 0.088_388_347_648_318_45,
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
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
        short_conv_kernel_size: TAPS as i64,
        ple: PleConfig::NONE,
    }
}

/// The zero-tail `rope_freqs.weight` for this fixture's head: ones over the
/// rotated pairs, 1e30 over the rest, exactly the published encoding.
pub fn tiny_lfm3_rope_freqs() -> Vec<f32> {
    let pairs = HEAD_DIM / 2;
    let rotated = (pairs as f64 * PARTIAL_ROTARY_FACTOR) as usize;
    (0..pairs)
        .map(|i| if i < rotated { 1.0 } else { 1e30 })
        .collect()
}

/// Writes a tiny `lfm3` `.gturbo` install via the REAL GGUF walk and returns
/// the derived `ArchConfig`. Q8_0 for every matrix, F32 for the norms and
/// the conv taps (the walk narrows both to BF16).
pub fn build_synthetic_lfm3_install(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_synthetic_lfm3_install_with_rope_freqs(
        dir,
        vocab_size,
        num_layers,
        model_id,
        Some(&tiny_lfm3_rope_freqs()),
    )
}

/// [`build_synthetic_lfm3_install`] with the `rope_freqs.weight` vector
/// under the caller's control (`None` omits it), so a test can prove the
/// walk refuses a wrong or missing one.
pub fn build_synthetic_lfm3_install_with_rope_freqs(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
    rope_freqs: Option<&[f32]>,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    let expected = tiny_lfm3_arch(vocab_size, num_layers);
    let mask = &expected.full_attention_layer_mask;
    let q_dim = (NUM_HEADS * HEAD_DIM) as u64;
    let kv_dim = (NUM_KV_HEADS * HEAD_DIM) as u64;
    let h = HIDDEN as u64;
    let inter = INTER as u64;

    let kv_per_layer: Vec<GgufValue> = mask
        .iter()
        .map(|&m| GgufValue::U32(if m == 6 { 0 } else { NUM_KV_HEADS as u32 }))
        .collect();
    let pattern: Vec<GgufValue> = mask.iter().map(|&m| GgufValue::Bool(m == 0)).collect();

    let mut b = GgufBuilder::new()
        .metadata_str("general.architecture", "lfm3")
        .metadata_u32("lfm3.block_count", num_layers as u32)
        .metadata_u32("lfm3.embedding_length", HIDDEN as u32)
        .metadata_u32("lfm3.feed_forward_length", INTER as u32)
        .metadata_u32("lfm3.attention.head_count", NUM_HEADS as u32)
        .metadata(
            "lfm3.attention.head_count_kv",
            GgufValue::Array(kv_per_layer),
        )
        .metadata_u32("lfm3.attention.key_length", HEAD_DIM as u32)
        .metadata_u32("lfm3.attention.value_length", HEAD_DIM as u32)
        .metadata_f32("lfm3.attention.layer_norm_rms_epsilon", 1e-5)
        .metadata_f32("lfm3.rope.freq_base", ROPE_THETA_FULL as f32)
        .metadata_f32("lfm3.rope.freq_base_swa", ROPE_THETA_SWA as f32)
        // The whole head, as the published conversion writes it; the partial
        // factor lives in `rope_freqs.weight`.
        .metadata_u32("lfm3.rope.dimension_count", HEAD_DIM as u32)
        .metadata_u32("lfm3.attention.sliding_window", SLIDING_WINDOW as u32)
        .metadata(
            "lfm3.attention.sliding_window_pattern",
            GgufValue::Array(pattern),
        )
        .metadata_u32("lfm3.shortconv.l_cache", TAPS as u32)
        .q8_0_tensor("token_embd.weight", &[h, vocab_size as u64], 1)
        .f32_tensor("output_norm.weight", &[h], 2);
    if let Some(freqs) = rope_freqs {
        let data = freqs.iter().flat_map(|v| v.to_le_bytes()).collect();
        b = b.tensor(
            "rope_freqs.weight",
            GGML_TYPE_F32,
            &[freqs.len() as u64],
            data,
        );
    }

    // Seeds are u8 and wrap across layers; within a layer the slots stay
    // distinct, which is what the perturbation tests need.
    for (l, &kind) in mask.iter().enumerate() {
        let p = format!("blk.{l}");
        let base = (l as u8).wrapping_mul(16);
        let s = |i: u8| base.wrapping_add(i);
        b = b
            .f32_tensor(&format!("{p}.attn_norm.weight"), &[h], s(1))
            .f32_tensor(&format!("{p}.post_attention_norm.weight"), &[h], s(2))
            .f32_tensor(&format!("{p}.ffn_norm.weight"), &[h], s(3))
            .f32_tensor(&format!("{p}.post_ffw_norm.weight"), &[h], s(4))
            .q8_0_tensor(&format!("{p}.ffn_gate.weight"), &[h, inter], s(5))
            .q8_0_tensor(&format!("{p}.ffn_up.weight"), &[h, inter], s(6))
            .q8_0_tensor(&format!("{p}.ffn_down.weight"), &[inter, h], s(7));
        b = if kind == 6 {
            b.q8_0_tensor(&format!("{p}.shortconv.in_proj.weight"), &[h, 3 * h], s(8))
                // ggml `[taps, hidden]`: logical `[hidden, taps]`, channel-major.
                .f32_tensor(
                    &format!("{p}.shortconv.conv.weight"),
                    &[TAPS as u64, h],
                    s(9),
                )
                .q8_0_tensor(&format!("{p}.shortconv.out_proj.weight"), &[h, h], s(10))
        } else {
            b.q8_0_tensor(&format!("{p}.attn_q.weight"), &[h, q_dim], s(8))
                .q8_0_tensor(&format!("{p}.attn_k.weight"), &[h, kv_dim], s(9))
                .q8_0_tensor(&format!("{p}.attn_v.weight"), &[h, kv_dim], s(10))
                .q8_0_tensor(&format!("{p}.attn_output.weight"), &[q_dim, h], s(11))
                // Q8_0 rather than F32, the spark fixture's reason: BF16 is a
                // norm dtype and gets no matrix GEMV.
                .q8_0_tensor(
                    &format!("{p}.attn_gate.weight"),
                    &[h, NUM_HEADS as u64],
                    s(12),
                )
                .f32_tensor(
                    &format!("{p}.attn_q_norm.weight"),
                    &[HEAD_DIM as u64],
                    s(13),
                )
                .f32_tensor(
                    &format!("{p}.attn_k_norm.weight"),
                    &[HEAD_DIM as u64],
                    s(14),
                )
        };
    }

    let (bytes, _) = b.build();
    let header = parse_gguf_header(&bytes, DEFAULT_MAX_HEADER_BYTES)?;
    let arch = write_gguf_install_streamed(
        dir,
        &header,
        &MemoryRangeSource::new(&bytes),
        model_id,
        |_| {},
    )?;
    assert_eq!(arch, expected, "the walk derived a different arch");
    Ok(arch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tiny_mask_rule_reproduces_the_real_layer_types() {
        assert_eq!(tiny_lfm3_layer_mask(24), model_io::lfm3_layer_mask());
    }
}
