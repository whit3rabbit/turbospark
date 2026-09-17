//! Builds a tiny `muse_glimmer` install through the REAL checkpoint repack
//! pipeline.
//!
//! Reuses `synthetic_real.rs`'s `pub(crate)` tensor helpers, as
//! `synthetic_qwen/` does, because this checkpoint's quantization is the
//! ORDINARY INT4 affine at group 64 with BF16 companions -- the shape those
//! helpers were written for. Nothing about this family is new on the
//! quantization axis; all ten of its differences are in the decode flow.
//!
//! **WHAT THIS FIXTURE MUST GET RIGHT THAT A SHAPE-ONLY ONE WOULD NOT.**
//! Three of its constants are load-bearing and each is chosen against a trap
//! recorded elsewhere in this repo:
//!
//! 1. `NUM_LAYERS` in the caller must be a multiple of 4 and at least 4, or
//!    the `[0, 0, 0, 1]` window pattern has no FULL layer in it and the
//!    fixture cannot exercise the NoPE branch at all -- which is the single
//!    most dangerous thing in this family's flow, because a full layer that
//!    rotates produces fluent wrong text rather than an error.
//! 2. `HEAD_DIM` is 64 rather than the Qwen fixtures' 32, so
//!    `attention_scale` is `64^-0.5 = 0.125`, a BINARY FRACTION. The real
//!    model's `128^-0.5 = 2^-3.5` is not one and survives anyway (Mixtral
//!    carries the identical value), but an invented fixture value has no such
//!    evidence behind it and `validate_arch` compares this `f64` with `!=`
//!    (AGENTS.md Gotcha 24).
//! 3. `SLIDING_WINDOW` is small and NONZERO, so the KV ring is actually
//!    exercised. A fixture with a window at or above its test's context
//!    length runs a linear cache and proves nothing about the ring
//!    (AGENTS.md Gotcha 18).

use model_io::{
    muse_glimmer_layer_mask, ArchConfig, CompressedAttentionConfig, HyperConnectionConfig,
    LinearAttentionConfig, MlaConfig, ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};

use crate::gemma4_checkpoint::{write_muse_glimmer_install, Gemma4Quant};
use crate::ranged_download::MemoryRangeSource;
use crate::safetensors_header::parse_header;
use crate::synthetic_real::{assemble_safetensors, bf16_vector, int4_triple, Tensor};

/// A whole number of 64-element groups, which is all the quantizer requires
/// of a COLUMN count.
const HIDDEN: usize = 128;
const NUM_HEADS: usize = 4;
/// 64, not 32: see the module header's point 2.
const HEAD_DIM: usize = 64;
/// Two, matching the real model's aggressive GQA (32 q over 2 kv).
const NUM_KV_HEADS: usize = 2;
/// The DENSE FFN width. There are no experts, so this is the only FFN there
/// is and `moe_intermediate_size` stays 0.
const INTER: usize = 128;
/// Small and nonzero: see the module header's point 3.
const SLIDING_WINDOW: i64 = 8;
/// The real checkpoint's, and exactly representable.
const ROPE_THETA: f64 = 500_000.0;
/// The real checkpoint's `final_logit_softcapping`.
const SOFTCAP: f64 = 20.0;

/// A tiny `muse_glimmer`-shaped architecture.
///
/// Every non-shape field takes [`model_io::muse_glimmer_30b`]'s own value,
/// for the reason the Qwen fixtures pin theirs: the manifest's optional
/// family-extension fields fall back to the GEMMA baseline whatever family
/// the manifest claims (AGENTS.md Gotcha 24), so anything else has to be
/// written explicitly and matched explicitly.
///
/// # Panics
///
/// If `num_layers` is not a positive multiple of 4. That is not fussiness:
/// the window pattern is `[sliding, sliding, sliding, full]`, so 3 layers
/// have no full layer and 5 have a truncated period, and in both cases the
/// fixture silently stops covering the branch it exists to cover.
pub fn tiny_muse_glimmer_arch(vocab_size: i64, num_layers: i64) -> ArchConfig {
    assert!(
        num_layers > 0 && num_layers % 4 == 0,
        "muse_glimmer's window period is 4 with the FULL layer last; \
         {num_layers} layers would leave the NoPE branch uncovered or truncated"
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
        final_logit_softcap: SOFTCAP,
        rope_theta: ROPE_THETA,
        // NoPE on the full layers, exactly as the real checkpoint declares.
        full_rope_theta: 0.0,
        partial_rotary_factor: 1.0,
        num_layers,
        dense_lead_intermediate_size: 0,
        num_dense_leading_layers: 0,
        num_experts: 0,
        top_k_experts: 0,
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        full_attention_layer_mask: muse_glimmer_layer_mask(num_layers),
        hidden_activation: "silu".to_string(),
        family: ModelFamily::MuseGlimmer,
        // FALSE: the gate is a separate tensor, not a packed `q_proj`.
        attn_output_gate: false,
        // 64^-0.5 = 0.125, a binary fraction. See the module header.
        attention_scale: 0.125,
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
        ple: PleConfig::NONE,
    }
}

/// A CENTERED norm weight vector, i.e. one centred at ZERO.
///
/// **The centre is the whole difference from every other fixture's norm
/// helper and it is not cosmetic.** Those pass `center: 1.0`, because their
/// families' norms apply the stored weight directly. This family's four
/// per-layer norms apply `1 + w`, so a fixture that stored values near 1.0
/// would build a model whose effective norm scale is near 2.0 -- which still
/// decodes, still produces finite deterministic logits, and is not the
/// architecture. `language_model.model.norm.weight` is the exception and
/// takes 1.0, because that one really is a plain `nn.RMSNorm`.
fn centered_norm(name: &str, n: usize, seed: u64) -> Tensor {
    bf16_vector(name, n, 0.0, seed)
}

/// Writes a tiny `muse_glimmer` `.gturbo` install and returns the
/// `ArchConfig` needed to open it.
///
/// Dense, so it writes ZERO packed-expert files -- the routed marker in
/// `classify_for_family` never fires.
pub fn build_synthetic_muse_glimmer_install(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    let arch = tiny_muse_glimmer_arch(vocab_size, num_layers);
    let vocab = vocab_size as usize;
    let q_dim = NUM_HEADS * HEAD_DIM;
    let kv_dim = NUM_KV_HEADS * HEAD_DIM;

    let mut ts: Vec<Tensor> = Vec::new();
    // Untied: the real checkpoint ships a separate `lm_head`, and its
    // `tie_word_embeddings` is false.
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

    for l in 0..num_layers as usize {
        let p = format!("language_model.model.layers.{l}");
        let seed = 1000 * (l as u64 + 1);

        // ALL FOUR sandwich norms, on EVERY layer. Unlike the Qwen fixtures'
        // two, and unlike Gemma's, this family applies a norm at all four
        // positions and all four are centered.
        for (i, norm) in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ]
        .iter()
        .enumerate()
        {
            ts.push(centered_norm(
                &format!("{p}.{norm}.weight"),
                HIDDEN,
                seed + 40 + i as u64,
            ));
        }

        // EVERY layer carries the same five projections, sliding or full.
        // The window and the NoPE branch change what the FLOW does with
        // them, never which tensors exist -- which is why this loop has no
        // `is_full` fork, unlike the Qwen fixture's.
        ts.extend(int4_triple(
            &format!("{p}.self_attn.q_proj.weight"),
            q_dim,
            HIDDEN,
            seed + 1,
        ));
        for (i, role) in ["k_proj", "v_proj"].iter().enumerate() {
            ts.extend(int4_triple(
                &format!("{p}.self_attn.{role}.weight"),
                kv_dim,
                HIDDEN,
                seed + 2 + i as u64,
            ));
        }
        // The ATTENTION OUTPUT GATE, a separate `[q_dim, hidden]` projection
        // fed from the normed input. Qwen packs its gate into `q_proj`; this
        // family does not, which is why `attn_output_gate` is false above and
        // this tensor exists.
        ts.extend(int4_triple(
            &format!("{p}.self_attn.gate_proj.weight"),
            q_dim,
            HIDDEN,
            seed + 4,
        ));
        ts.extend(int4_triple(
            &format!("{p}.self_attn.o_proj.weight"),
            HIDDEN,
            q_dim,
            seed + 5,
        ));
        // NO `q_norm` / `k_norm` tensors: this family's q/k norms are
        // NO-SCALE, so there is nothing to store. Confirmed absent in the
        // real checkpoint's index.

        for (i, role) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
            let (rows, cols) = if *role == "down_proj" {
                (HIDDEN, INTER)
            } else {
                (INTER, HIDDEN)
            };
            ts.extend(int4_triple(
                &format!("{p}.mlp.{role}.weight"),
                rows,
                cols,
                seed + 60 + i as u64,
            ));
        }
    }

    // The FINAL norm, and the ONE norm in this model that is not centered:
    // `TextModel.norm` is a plain `nn.RMSNorm`. Centre 1.0, not 0.0.
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
        bits_overrides: std::collections::HashMap::new(),
    };
    write_muse_glimmer_install(dir, &arch, model_id, &header, &source, &quant)?;
    Ok(arch)
}
