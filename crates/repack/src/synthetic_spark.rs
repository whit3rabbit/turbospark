//! Builds a tiny `spark2_5` install through the REAL GGUF walk.
//!
//! The sibling fixtures (`synthetic_muse`, `synthetic_qwen`) build a
//! safetensors blob and go through the MLX-side writers, but this family has
//! NO safetensors intake this pass -- GGUF is its only way in. So this
//! builder constructs a small `GgufBuilder` file shaped like the real
//! `XHToken/Spark-X2.5-4B-GGUF` header (fused `attn_qkv`, per-head
//! `attn_gate`, bool-array `sliding_window_pattern`, per-class rope keys)
//! and pushes it through `write_gguf_install_streamed` exactly as a real
//! `pull` would. The derived `ArchConfig` and the manifest therefore come
//! from the walk itself, which is also why the returned arch is what a test
//! must pass to `RealForwardRunner::open`.
//!
//! Constants chosen against the same traps the muse fixture documents:
//! `NUM_LAYERS` a multiple of 4 (the window pattern needs its full layer),
//! `HEAD_DIM` 64 so the per-class rotary widths stay even and small, and a
//! small nonzero `SLIDING_WINDOW` so the KV ring is exercised.
//! The per-class rope thetas are invented POSITIVE values that differ -- the
//! real checkpoint's 5e6/1e4 are its own -- and `PARTIAL_ROTARY_FACTOR` is
//! the real 0.25, which at head_dim 64 gives a 16-dim rotary width, even.

use model_io::{
    spark_layer_mask, ArchConfig, CompressedAttentionConfig, HyperConnectionConfig,
    LinearAttentionConfig, MlaConfig, ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};

use crate::gguf_header::{parse_header as parse_gguf_header, GgufValue, DEFAULT_MAX_HEADER_BYTES};
use crate::ranged_download::MemoryRangeSource;
use crate::synthetic_gguf::GgufBuilder;
use crate::write_gguf_install_streamed;

const HIDDEN: usize = 128;
const NUM_HEADS: usize = 4;
/// 64, so `attention_scale` is 0.125: a binary fraction.
const HEAD_DIM: usize = 64;
const NUM_KV_HEADS: usize = 2;
const INTER: usize = 128;
/// Small and nonzero, so the sliding layers' ring actually wraps.
const SLIDING_WINDOW: i64 = 8;
/// The two per-class thetas: invented, positive, and DIFFERENT -- which is
/// the family's shape the flow has to honor.
const ROPE_THETA_SWA: f64 = 1_000.0;
const ROPE_THETA_FULL: f64 = 5_000.0;
/// The real checkpoint's, and the reason the full layers' rotary width is
/// `head_dim / 4` here.
const PARTIAL_ROTARY_FACTOR: f64 = 0.25;

/// A tiny `spark2_5`-shaped architecture.
///
/// Every non-shape field takes [`model_io::spark_x25_4b`]'s own value (the
/// manifest's optional family-extension fields fall back to GEMMA's whatever
/// family the manifest claims, AGENTS.md Gotcha 24), with the shapes scaled
/// down and the binary-fraction substitutions the muse fixture's header
/// records.
pub fn tiny_spark_arch(vocab_size: i64, num_layers: i64) -> ArchConfig {
    assert!(
        num_layers > 0 && num_layers % 4 == 0,
        "spark2_5's window period is 4 with the FULL layer last; \
         {num_layers} layers would leave the per-class rope branch uncovered or truncated"
    );
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
        // TIED: the GGUF fixture ships no `output.weight` and the head
        // re-reads the embedding, exactly as the real file does.
        tie_word_embeddings: true,
        attention_k_eq_v: false,
        full_attention_layer_mask: spark_layer_mask(num_layers),
        hidden_activation: "gelu".to_string(),
        family: ModelFamily::Spark25,
        // FALSE: the gate is its own headwise `g_proj` tensor, not a packed
        // `q_proj`.
        attn_output_gate: false,
        // THE BASELINE'S VALUE, NOT THE TOY SHAPE'S: this install is derived
        // through the GGUF walk, which takes behavioural fields from
        // `known_architecture(Spark25)` (attention_scale is not in any GGUF
        // key), so the fixture must agree with the baseline it derives from
        // or `arch_from_gguf` and `tiny_spark_arch` disagree. It is a
        // constant multiplier on the attention logits; untrained weights
        // make its exact value immaterial, and 0.0625 is a binary fraction
        // anyway.
        attention_scale: 0.0625,
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
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

/// Writes a tiny `spark2_5` `.gturbo` install via the REAL GGUF walk and
/// returns the derived `ArchConfig`.
///
/// Dense, so the walk writes zero packed-expert files; Q8_0 everywhere a
/// matrix exists (executable, and the real Q8_0 GGUF variant uses exactly
/// that width), F32 for the norms and the tiny gate vector (the walk narrows
/// both to BF16, Gotcha 45).
pub fn build_synthetic_spark_install(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    let _ = tiny_spark_arch(vocab_size, num_layers);
    let vocab = vocab_size as usize;
    let q_dim = NUM_HEADS * HEAD_DIM;
    let kv_dim = NUM_KV_HEADS * HEAD_DIM;
    let qkv_out = q_dim + 2 * kv_dim;

    // The bool array is the INVERSE of this port's mask (1/true = sliding),
    // which is the whole inversion `gguf_config`'s mask builder exists for.
    let pattern: Vec<GgufValue> = (0..num_layers)
        .map(|i| GgufValue::Bool(i % 4 != 3))
        .collect();

    let mut b = GgufBuilder::new()
        .metadata_str("general.architecture", "spark2_5")
        .metadata_u32("spark2_5.block_count", num_layers as u32)
        .metadata_u32("spark2_5.embedding_length", HIDDEN as u32)
        .metadata_u32("spark2_5.feed_forward_length", INTER as u32)
        .metadata_u32("spark2_5.attention.head_count", NUM_HEADS as u32)
        .metadata_u32("spark2_5.attention.head_count_kv", NUM_KV_HEADS as u32)
        .metadata_u32("spark2_5.attention.key_length", HEAD_DIM as u32)
        .metadata_f32("spark2_5.rope.freq_base", ROPE_THETA_FULL as f32)
        .metadata_f32("spark2_5.rope.freq_base_swa", ROPE_THETA_SWA as f32)
        .metadata_u32("spark2_5.attention.sliding_window", SLIDING_WINDOW as u32)
        .metadata(
            "spark2_5.attention.sliding_window_pattern",
            GgufValue::Array(pattern),
        )
        // The converter encodes the per-class partial factors as rotary
        // dims; `arch_from_gguf` cross-checks these against the factor it
        // derives, so a fixture that disagreed with itself would refuse.
        .metadata_u32(
            "spark2_5.rope.dimension_count",
            (HEAD_DIM as f64 * PARTIAL_ROTARY_FACTOR) as u32,
        )
        .metadata_u32("spark2_5.rope.dimension_count_swa", HEAD_DIM as u32)
        // The vocabulary comes off the embedding tensor's own dims.
        .q8_0_tensor("token_embd.weight", &[HIDDEN as u64, vocab as u64], 1);

    // `GgufBuilder`'s seeds are u8, so the per-layer base is taken modulo
    // the range and the eight slots stay distinct within a layer. Colliding
    // across LAYERS is harmless: every perturbation test targets layer 0,
    // and the frozen digest needs determinism, not global uniqueness.
    for l in 0..num_layers as usize {
        // GGUF names, not canonical ones: this file is the walk's INPUT,
        // and the walk maps `blk.N.*` through `gguf_names::spark`.
        let p = format!("blk.{l}");
        let base = (l as u8).wrapping_mul(8);
        let slot = |i: u8| base.wrapping_add(i);
        b = b
            .f32_tensor(&format!("{p}.attn_norm.weight"), &[HIDDEN as u64], slot(1))
            .q8_0_tensor(
                &format!("{p}.attn_qkv.weight"),
                &[HIDDEN as u64, qkv_out as u64],
                slot(2),
            )
            // Q8_0, not F32: the walk narrows F32 to BF16, and BF16 is a
            // NORM-weight dtype here -- `encode_gemv_any` dispatches no
            // matrix GEMV for it. The real file carries this tensor as
            // Q4_K, which is the same "executable matrix dtype" class.
            .q8_0_tensor(
                &format!("{p}.attn_gate.weight"),
                &[HIDDEN as u64, NUM_HEADS as u64],
                slot(3),
            )
            .q8_0_tensor(
                &format!("{p}.attn_output.weight"),
                &[q_dim as u64, HIDDEN as u64],
                slot(4),
            )
            .f32_tensor(&format!("{p}.ffn_norm.weight"), &[HIDDEN as u64], slot(5))
            .q8_0_tensor(
                &format!("{p}.ffn_gate.weight"),
                &[HIDDEN as u64, INTER as u64],
                slot(6),
            )
            .q8_0_tensor(
                &format!("{p}.ffn_up.weight"),
                &[HIDDEN as u64, INTER as u64],
                slot(7),
            )
            .q8_0_tensor(
                &format!("{p}.ffn_down.weight"),
                &[INTER as u64, HIDDEN as u64],
                slot(8),
            );
    }
    let b = b.f32_tensor("output_norm.weight", &[HIDDEN as u64], 9);

    let (bytes, _) = b.build();
    let header = parse_gguf_header(&bytes, DEFAULT_MAX_HEADER_BYTES)?;
    let arch = write_gguf_install_streamed(
        dir,
        &header,
        &MemoryRangeSource::new(&bytes),
        model_id,
        |_| {},
    )?;
    // The walk derives the config from the file; a mismatch against the
    // intended tiny arch means the fixture's metadata is wrong, not a
    // tolerance to paper over.
    let expected = tiny_spark_arch(vocab_size, num_layers);
    assert_eq!(
        arch.full_attention_layer_mask, expected.full_attention_layer_mask,
        "the walk derived a different window than the fixture intended"
    );
    assert_eq!(arch, expected, "the walk derived a different arch");
    Ok(arch)
}
