//! Builds a tiny Qwen 3.6 install through the REAL checkpoint repack
//! pipeline, the sibling of [`crate::build_synthetic_gemma4_real_install`]:
//! an in-memory safetensors blob using the verbatim Qwen naming
//! (`language_model.` prefix, `.mlp.switch_mlp.` routed experts,
//! `linear_attn.*` gated-DeltaNet tensors, INT8 router + shared expert,
//! BF16 norms), pushed through [`crate::write_qwen36_install`]. This is
//! what `crates/runtime`'s Qwen decode flow is exercised against, since no
//! trained Qwen `.gturbo` checkpoint exists in this environment.
//!
//! Shape mirrors the Swift `QwenToySynthetic` fixture: hidden 64, 4 query
//! heads of 32 over 2 KV heads, linear attention (2 K heads, 4 V heads,
//! 32/32 head dims, 4 conv taps -> 256 conv channels, 128 value dim),
//! layers alternating linear (mask 2) and full attention (mask 1), 8
//! routed experts.
//!
//! Weights are deterministic but NOT trained: generated tokens are
//! structurally real and semantically meaningless, and a short generation
//! can decode to the empty string. Assert on token counts and stop
//! reasons, never on text (see AGENTS.md Gotcha 12).

use model_io::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig,
    ModelFamily,
};

use crate::gemma4_checkpoint::{write_qwen36_install, Gemma4Quant};
use crate::ranged_download::MemoryRangeSource;
use crate::safetensors_header::parse_header;
use crate::synthetic_real::{
    assemble_safetensors, bf16_vector, expert_int4_triple, int4_triple, int8_triple, Tensor,
};

const HIDDEN: usize = 64;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = 32;
const NUM_KV_HEADS: usize = 2;
/// Shared-expert and per-expert FFN width (both, as on the real 35B).
const INTER: usize = 64;

const LA_K_HEADS: usize = 2;
const LA_V_HEADS: usize = 4;
const LA_KEY_DIM: usize = 32;
const LA_VALUE_DIM: usize = 32;
const LA_CONV_K: usize = 4;

fn linear_attention() -> LinearAttentionConfig {
    LinearAttentionConfig {
        num_k_heads: LA_K_HEADS as i64,
        num_v_heads: LA_V_HEADS as i64,
        key_head_dim: LA_KEY_DIM as i64,
        value_head_dim: LA_VALUE_DIM as i64,
        conv_kernel_size: LA_CONV_K as i64,
    }
}

/// A tiny Qwen-3.6-shaped architecture. Every non-shape field takes
/// `mrefrust_model_io::qwen36_35b_a3b()`'s own value, for the same reason
/// `tiny_gemma4_arch` pins Gemma's: the manifest's optional family
/// extensions fall back to a baseline, so anything else has to be written
/// explicitly and matched explicitly.
pub fn tiny_qwen36_arch(vocab_size: i64, num_layers: i64, num_experts: i64) -> ArchConfig {
    ArchConfig {
        hidden_size: HIDDEN as i64,
        intermediate_size: INTER as i64,
        moe_intermediate_size: INTER as i64,
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
        num_experts,
        top_k_experts: num_experts.min(8),
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        // Layer 0 is linear, matching the real 40-layer model (full
        // attention on every 4th layer); alternating keeps a toy model
        // small while still exercising both flows and their interleaving.
        full_attention_layer_mask: (0..num_layers)
            .map(|l| if l % 2 == 1 { 1u8 } else { 2u8 })
            .collect(),
        hidden_activation: "silu".to_string(),
        family: ModelFamily::Qwen36,
        attn_output_gate: true,
        // 0.125, not the mathematically-right 32^-0.5. `validate_arch`
        // compares this f64 EXACTLY against the manifest's, and
        // serde_json's default float parser is only correct to ~1 ULP
        // (exactness needs its `float_roundtrip` feature), so a scale that
        // is not a binary fraction cannot survive the round trip. The real
        // 35B's 0.0625 is a power of two and is unaffected; these weights
        // are untrained, so any finite scale is equally meaningless.
        attention_scale: 0.125,
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        shared_expert_gated: true,
        rope_neox_subdim: true,
        linear_attention: linear_attention(),
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
    }
}

/// The depthwise conv kernel: BF16, rank 3 `[channels, taps, 1]`. Rank 3
/// with a trailing 1 is what the real checkpoint ships (a PyTorch
/// `Conv1d` weight), and `shape4` pads it out, so the resident entry
/// records `(C, K, 1, 0)`.
fn conv1d_weight(name: &str, channels: usize, taps: usize, seed: u64) -> Tensor {
    let flat = bf16_vector(name, channels * taps, 0.0, seed);
    Tensor {
        name: flat.name,
        dtype: flat.dtype,
        shape: vec![channels as u64, taps as u64, 1],
        bytes: flat.bytes,
    }
}

/// Writes a tiny Qwen 3.6 `.gturbo` install and returns the `ArchConfig`
/// needed to open it. `num_experts` must be positive; `top_k` is
/// `min(num_experts, 8)` (the MoE kernels' fixed slot count).
pub fn build_synthetic_qwen36_real_install(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    num_experts: i64,
    model_id: &str,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    let arch = tiny_qwen36_arch(vocab_size, num_layers, num_experts);
    let experts = num_experts as usize;
    let vocab = vocab_size as usize;
    let la = &arch.linear_attention;
    let qkv_dim = la.qkv_dim() as usize;
    let value_dim = la.value_dim() as usize;
    let v_heads = LA_V_HEADS;

    let mut ts: Vec<Tensor> = Vec::new();
    // Untied head: embed_tokens and lm_head are separate tensors.
    ts.extend(int4_triple(
        "language_model.model.embed_tokens.weight",
        vocab,
        HIDDEN,
        1,
    ));
    ts.extend(int4_triple(
        "language_model.lm_head.weight",
        vocab,
        HIDDEN,
        2,
    ));

    // `Gemma4Quant` overrides are keyed by the tensor base path (no
    // `.weight`); everything not listed here stays at the default 4 bits.
    let mut overrides = std::collections::HashMap::new();
    for l in 0..num_layers as usize {
        let p = format!("language_model.model.layers.{l}");
        let seed = 1000 * (l as u64 + 1);
        let is_full = arch.layer_is_full(l);

        for (i, norm) in ["input_layernorm", "post_attention_layernorm"]
            .iter()
            .enumerate()
        {
            ts.push(bf16_vector(
                &format!("{p}.{norm}.weight"),
                HIDDEN,
                1.0,
                seed + 40 + i as u64,
            ));
        }

        if is_full {
            // attn_output_gate: q_proj emits per-head [query; gate] pairs,
            // so 2x the usual row count.
            ts.extend(int4_triple(
                &format!("{p}.self_attn.q_proj.weight"),
                2 * NUM_HEADS * HEAD_DIM,
                HIDDEN,
                seed + 1,
            ));
            for (i, role) in ["k_proj", "v_proj"].iter().enumerate() {
                ts.extend(int4_triple(
                    &format!("{p}.self_attn.{role}.weight"),
                    NUM_KV_HEADS * HEAD_DIM,
                    HIDDEN,
                    seed + 2 + i as u64,
                ));
            }
            ts.extend(int4_triple(
                &format!("{p}.self_attn.o_proj.weight"),
                HIDDEN,
                NUM_HEADS * HEAD_DIM,
                seed + 4,
            ));
            for (i, norm) in ["q_norm", "k_norm"].iter().enumerate() {
                ts.push(bf16_vector(
                    &format!("{p}.self_attn.{norm}.weight"),
                    HEAD_DIM,
                    1.0,
                    seed + 20 + i as u64,
                ));
            }
        } else {
            for (name, rows, cols, s) in [
                ("in_proj_qkv", qkv_dim, HIDDEN, 1u64),
                ("in_proj_z", value_dim, HIDDEN, 2),
                ("in_proj_a", v_heads, HIDDEN, 3),
                ("in_proj_b", v_heads, HIDDEN, 4),
                ("out_proj", HIDDEN, value_dim, 5),
            ] {
                ts.extend(int4_triple(
                    &format!("{p}.linear_attn.{name}.weight"),
                    rows,
                    cols,
                    seed + s,
                ));
            }
            ts.push(conv1d_weight(
                &format!("{p}.linear_attn.conv1d.weight"),
                qkv_dim,
                LA_CONV_K,
                seed + 10,
            ));
            // NOTE: A_log and dt_bias carry NO `.weight` suffix in the real
            // checkpoint. They are plain BF16 [num_v_heads] parameters.
            ts.push(bf16_vector(
                &format!("{p}.linear_attn.A_log"),
                v_heads,
                0.0,
                seed + 11,
            ));
            ts.push(bf16_vector(
                &format!("{p}.linear_attn.dt_bias"),
                v_heads,
                0.0,
                seed + 12,
            ));
            ts.push(bf16_vector(
                &format!("{p}.linear_attn.norm.weight"),
                LA_VALUE_DIM,
                1.0,
                seed + 13,
            ));
        }

        // MoE on every layer: INT8 router, INT8 sigmoid-gated shared
        // expert, INT4 routed experts under `.mlp.switch_mlp.`.
        ts.extend(int8_triple(
            &format!("{p}.mlp.gate.weight"),
            experts,
            HIDDEN,
            seed + 50,
        ));
        overrides.insert(format!("{p}.mlp.gate"), 8u32);
        ts.extend(int8_triple(
            &format!("{p}.mlp.shared_expert_gate.weight"),
            1,
            HIDDEN,
            seed + 51,
        ));
        overrides.insert(format!("{p}.mlp.shared_expert_gate"), 8u32);
        for (i, role) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
            let (rows, cols) = if *role == "down_proj" {
                (HIDDEN, INTER)
            } else {
                (INTER, HIDDEN)
            };
            ts.extend(int8_triple(
                &format!("{p}.mlp.shared_expert.{role}.weight"),
                rows,
                cols,
                seed + 60 + i as u64,
            ));
            overrides.insert(format!("{p}.mlp.shared_expert.{role}"), 8u32);
            ts.extend(expert_int4_triple(
                &format!("{p}.mlp.switch_mlp.{role}.weight"),
                experts,
                rows,
                cols,
                seed + 70 + i as u64,
            ));
        }
    }
    ts.push(bf16_vector(
        "language_model.model.norm.weight",
        HIDDEN,
        1.0,
        7,
    ));

    let blob = assemble_safetensors(&ts);
    let source = MemoryRangeSource::new(&blob);
    let header = parse_header(&blob, crate::safetensors_header::DEFAULT_MAX_HEADER_BYTES)?;
    let quant = Gemma4Quant {
        default_bits: 4,
        group_size: 64,
        bits_overrides: overrides,
    };
    write_qwen36_install(dir, &arch, model_id, &header, &source, &quant)?;
    Ok(arch)
}
