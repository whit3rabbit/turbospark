//! Builds a tiny Flint-shaped install through the REAL checkpoint repack
//! pipeline, in the checkpoint's own SOURCE form rather than the canonical
//! one: `model.*` names, U8-packed affine weights, an unquantized BF16 router,
//! an INT8 embedding under an INT4 default, and the `softmax_offset` sink
//! with no `.weight` suffix. It then runs `canonicalize_flint_header`, so the
//! namespace rename, the U8 relabel and the router force-quantize are all on
//! the path a test exercises, which a canonical-form fixture would skip.
//!
//! Shape: hidden 64, 4 query heads of 32 over 2 KV heads, 8 routed experts at
//! top-2 of width 64 plus a 64-wide shared expert, a 128-wide dense lead.
//! Layers: `0, 1` dense and full-attention NoPE, `2` sliding MoE, `3` full
//! MoE, which is Flint's own order (`[1, 1, 0, ..., 1]`) at fixture scale.
//!
//! Weights are deterministic but NOT trained: assert on structure, never on
//! generated text (AGENTS.md Gotcha 12).

use model_io::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig, MlaConfig,
    ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};

use crate::gemma4_checkpoint::{canonicalize_flint_header, write_gemma4_install, Gemma4Quant};
use crate::ranged_download::MemoryRangeSource;
use crate::safetensors_header::parse_header;
use crate::synthetic_tensors::{
    assemble_safetensors, bf16_vector, expert_int4_triple, int4_triple, int8_triple, Tensor,
};

const HIDDEN: usize = 64;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = 32;
const NUM_KV_HEADS: usize = 2;
/// The routed AND shared expert width. 64 so every down projection's input
/// dimension is a whole affine group.
const MOE_INTER: usize = 64;
/// The dense lead's width, distinct from the shared expert's on purpose: one
/// scalar cannot carry both, and a flow that read the wrong one is the bug.
const DENSE_INTER: usize = 128;
const NUM_EXPERTS: usize = 8;
const TOP_K: i64 = 2;
const WINDOW: i64 = 8;
const LAYERS: usize = 4;
/// `[full, full, sliding, full]`: the dense lead is also NoPE, as in Flint.
const MASK: [u8; LAYERS] = [1, 1, 0, 1];
const DENSE_LEAD: usize = 2;

/// A tiny Flint config. Every behavioural field takes the real baseline's
/// value; only shapes shrink. `attention_scale` is a binary fraction
/// (AGENTS.md Gotcha 24), not the real 128^-0.5.
pub fn tiny_flint_arch(vocab_size: i64) -> ArchConfig {
    ArchConfig {
        hidden_size: HIDDEN as i64,
        intermediate_size: MOE_INTER as i64,
        moe_intermediate_size: MOE_INTER as i64,
        num_heads: NUM_HEADS as i64,
        num_kv_heads: NUM_KV_HEADS as i64,
        num_full_kv_heads: NUM_KV_HEADS as i64,
        head_dim: HEAD_DIM as i64,
        full_head_dim: HEAD_DIM as i64,
        vocab_size,
        sliding_window: WINDOW,
        final_logit_softcap: 0.0,
        rope_theta: 10_000.0,
        full_rope_theta: 0.0,
        partial_rotary_factor: 1.0,
        num_layers: LAYERS as i64,
        num_dense_leading_layers: DENSE_LEAD as i64,
        dense_lead_intermediate_size: DENSE_INTER as i64,
        num_experts: NUM_EXPERTS as i64,
        top_k_experts: TOP_K,
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        full_attention_layer_mask: MASK.to_vec(),
        hidden_activation: "silu".to_string(),
        family: ModelFamily::Flint,
        attn_output_gate: false,
        attention_scale: 0.25,
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        shared_expert_gated: false,
        rope_neox_subdim: true,
        linear_attention: LinearAttentionConfig::NONE,
        mla: MlaConfig::NONE,
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "sigmoid".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: RopeScalingConfig::NONE,
        vision: VisionConfig::NONE,
        ple: PleConfig::NONE,
    }
}

/// Re-labels a canonical `U32`-packed fixture tensor as the checkpoint's
/// `U8` packing: same bytes, last dimension times four.
fn as_u8_packed(mut tensors: Vec<Tensor>) -> Vec<Tensor> {
    for t in &mut tensors {
        if t.dtype == "U32" {
            t.dtype = "U8";
            let n = t.shape.len();
            t.shape[n - 1] *= 4;
        }
    }
    tensors
}

/// Writes a tiny Flint `.gturbo` install and returns the config to open it.
pub fn build_synthetic_flint_install(
    dir: &std::path::Path,
    vocab_size: i64,
    model_id: &str,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    let arch = tiny_flint_arch(vocab_size);
    let vocab = vocab_size as usize;
    let mut ts: Vec<Tensor> = Vec::new();

    // The embedding is INT8 under the INT4 default, the head INT4.
    ts.extend(as_u8_packed(int8_triple(
        "model.embed_tokens.weight",
        vocab,
        HIDDEN,
        1,
    )));
    ts.extend(as_u8_packed(int4_triple(
        "lm_head.weight",
        vocab,
        HIDDEN,
        2,
    )));

    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        let seed = 1000 * (l as u64 + 1);
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
        for (i, norm) in ["q_norm", "k_norm"].iter().enumerate() {
            ts.push(bf16_vector(
                &format!("{p}.self_attn.{norm}.weight"),
                HEAD_DIM,
                1.0,
                seed + 60 + i as u64,
            ));
        }
        // The learned sink: one logit per query head, NO `.weight` suffix.
        ts.push(bf16_vector(
            &format!("{p}.self_attn.softmax_offset"),
            NUM_HEADS,
            0.0,
            seed + 65,
        ));
        ts.extend(as_u8_packed(int4_triple(
            &format!("{p}.self_attn.q_proj.weight"),
            NUM_HEADS * HEAD_DIM,
            HIDDEN,
            seed + 1,
        )));
        for (i, role) in ["k_proj", "v_proj"].iter().enumerate() {
            ts.extend(as_u8_packed(int4_triple(
                &format!("{p}.self_attn.{role}.weight"),
                NUM_KV_HEADS * HEAD_DIM,
                HIDDEN,
                seed + 2 + i as u64,
            )));
        }
        ts.extend(as_u8_packed(int4_triple(
            &format!("{p}.self_attn.o_proj.weight"),
            HIDDEN,
            NUM_HEADS * HEAD_DIM,
            seed + 4,
        )));

        let ffn = |ts: &mut Vec<Tensor>, prefix: &str, inter: usize, seed: u64| {
            for (i, role) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
                let (rows, cols) = if *role == "down_proj" {
                    (HIDDEN, inter)
                } else {
                    (inter, HIDDEN)
                };
                ts.extend(as_u8_packed(int4_triple(
                    &format!("{prefix}.{role}.weight"),
                    rows,
                    cols,
                    seed + i as u64,
                )));
            }
        };
        if l < DENSE_LEAD {
            ffn(&mut ts, &format!("{p}.mlp"), DENSE_INTER, seed + 70);
        } else {
            // The router ships RAW BF16 [experts, hidden] and no companions.
            let mut gate = bf16_vector(
                &format!("{p}.mlp.gate.weight"),
                NUM_EXPERTS * HIDDEN,
                0.0,
                seed + 50,
            );
            gate.shape = vec![NUM_EXPERTS as u64, HIDDEN as u64];
            ts.push(gate);
            ffn(
                &mut ts,
                &format!("{p}.mlp.shared_expert"),
                MOE_INTER,
                seed + 80,
            );
            for (i, role) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
                let (rows, cols) = if *role == "down_proj" {
                    (HIDDEN, MOE_INTER)
                } else {
                    (MOE_INTER, HIDDEN)
                };
                ts.extend(as_u8_packed(expert_int4_triple(
                    &format!("{p}.mlp.switch_mlp.{role}.weight"),
                    NUM_EXPERTS,
                    rows,
                    cols,
                    seed + 90 + i as u64,
                )));
            }
        }
    }
    ts.push(bf16_vector("model.norm.weight", HIDDEN, 1.0, 7));

    let blob = assemble_safetensors(&ts);
    let source = MemoryRangeSource::new(&blob);
    let mut header = parse_header(&blob, crate::safetensors_header::DEFAULT_MAX_HEADER_BYTES)?;
    canonicalize_flint_header(&mut header)?;
    // The embedding override is what `parse_flint_quantization` derives from
    // `mlx_export`; the fixture states it directly.
    let mut quant = Gemma4Quant {
        default_bits: 4,
        group_size: 64,
        ..Default::default()
    };
    quant
        .bits_overrides
        .insert("language_model.model.embed_tokens".to_string(), 8);
    write_gemma4_install(dir, &arch, model_id, &header, &source, &quant)?;
    // The walk applied the up/down rebalance; the real writer records it too.
    crate::flint_rebalance::record_shift(dir, crate::flint_rebalance::FLINT_SWIGLU_SHIFT)?;
    Ok(arch)
}
