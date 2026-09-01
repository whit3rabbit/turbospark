//! Builds a tiny Qwen 3.6 MoE install through the REAL checkpoint repack pipeline.

use model_io::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig,
    ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};

use crate::gemma4_checkpoint::{write_qwen_gdn_moe_install, Gemma4Quant};
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
        output_gate_sigmoid: false,
    }
}

/// A tiny Qwen-3.6-shaped architecture. Every non-shape field takes
/// `turbospark_model_io::qwen_gdn_moe_35b_a3b()`'s own value, for the same reason
/// `tiny_gemma4_arch` pins Gemma's: the manifest's optional family
/// extensions fall back to a baseline, so anything else has to be written
/// explicitly and matched explicitly.
pub fn tiny_qwen_gdn_moe_arch(vocab_size: i64, num_layers: i64, num_experts: i64) -> ArchConfig {
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
        // Layer 0 is linear attention on the real 48-layer model;
        // alternating keeps a toy small while exercising both flows and
        // their interleaving.
        full_attention_layer_mask: (0..num_layers)
            .map(|l| if l % 2 == 1 { 1u8 } else { 2u8 })
            .collect(),
        hidden_activation: "silu".to_string(),
        family: ModelFamily::QwenGdnMoe,
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
        shared_expert_gated: true,
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
        ple: PleConfig::NONE,
    }
}

/// Builds an in-memory synthetic Qwen checkpoint, runs it through the
/// real repack pipeline, and writes a `.gturbo` install to `dir`.
pub fn build_synthetic_qwen_gdn_moe_install(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    num_experts: i64,
    model_id: &str,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_synthetic_qwen_gdn_moe_install_inner(
        dir,
        vocab_size,
        num_layers,
        num_experts,
        model_id,
        false,
    )
}

/// The same install PLUS a multi-token-prediction head, which is what
/// makes the batched routed verify (`families/qwen/moe_batch.rs`)
/// reachable at all: `produce_batched` allocates its M-row scratch beside
/// a drafter, so a MoE install with no head has nowhere to put it.
///
/// **THE HEAD IS DENSE AND THE TRUNK IS MoE, which is deliberate and is
/// not what a real Ornith checkpoint looks like.** That model's head is
/// itself MoE -- 256 per-expert `gate/up/down`, a router and a shared
/// expert, all in its BF16 repo's last shard -- while `MtpState::REQUIRED`
/// names the DENSE FFN tensors a `qwen3_5` head has. Adapting the drafter
/// to an MoE head is its own piece of work. What this fixture exists to
/// reach is the TRUNK's routed batched half; the head's only job is to
/// allocate the verify scratch, and it is the smallest thing that will.
pub fn build_synthetic_qwen_gdn_moe_install_with_mtp(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    num_experts: i64,
    model_id: &str,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_synthetic_qwen_gdn_moe_install_inner(
        dir,
        vocab_size,
        num_layers,
        num_experts,
        model_id,
        true,
    )
}

fn build_synthetic_qwen_gdn_moe_install_inner(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    num_experts: i64,
    model_id: &str,
    with_mtp: bool,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    let arch = tiny_qwen_gdn_moe_arch(vocab_size, num_layers, num_experts);
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
    if with_mtp {
        // At THIS fixture's widths, not the dense fixture's: a head whose
        // hidden size disagrees with its trunk's fails at `open()` on a
        // length check rather than producing a smaller head.
        ts.extend(crate::synthetic_qwen::dense_tensors::mtp_head_tensors_at(
            HIDDEN,
            NUM_HEADS,
            HEAD_DIM,
            NUM_KV_HEADS,
            INTER,
        ));
    }

    let blob = assemble_safetensors(&ts);
    let source = MemoryRangeSource::new(&blob);
    let header = parse_header(&blob, crate::safetensors_header::DEFAULT_MAX_HEADER_BYTES)?;
    let quant = Gemma4Quant {
        default_bits: 4,
        group_size: 64,
        bits_overrides: overrides,
    };
    write_qwen_gdn_moe_install(dir, &arch, model_id, &header, &source, &quant)?;
    Ok(arch)
}

fn conv1d_weight(name: &str, channels: usize, taps: usize, seed: u64) -> Tensor {
    let flat = bf16_vector(name, channels * taps, 0.0, seed);
    Tensor {
        name: flat.name,
        dtype: flat.dtype,
        shape: vec![channels as u64, taps as u64, 1],
        bytes: flat.bytes,
    }
}
