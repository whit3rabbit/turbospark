//! Builds a tiny Mixtral-shaped install through the REAL checkpoint repack
//! pipeline (ROADMAP Phase M2), the sibling of
//! [`crate::build_synthetic_qwen_gdn_moe_install`].
//!
//! Shorter than either sibling because the `llama` architecture is defined by
//! what it lacks: no linear-attention layers, no per-head q/k norms, no
//! packed query/gate rows, no shared expert, no sandwich norms, no softcap.
//! One layer kind, one RoPE base, one norm before attention and one before
//! the MoE.
//!
//! Shape: hidden 64, 4 query heads of 16 over 2 KV heads, expert FFN 64,
//! every layer full attention, `num_experts` routed experts at top-2 -- the
//! real 8x7B's ratio rather than a copy of its size.
//!
//! Weights are deterministic but NOT trained: generated tokens are
//! structurally real and semantically meaningless, and a short generation can
//! decode to the empty string. Assert on token counts and stop reasons, never
//! on text (AGENTS.md Gotcha 12).

use model_io::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig,
    ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};

use crate::gemma4_checkpoint::{write_gemma4_install, Gemma4Quant};
use crate::ranged_download::MemoryRangeSource;
use crate::safetensors_header::parse_header;
use crate::synthetic_real::{
    assemble_safetensors, bf16_vector, expert_int4_triple, int4_triple, int8_triple, Tensor,
};

const HIDDEN: usize = 64;
const NUM_HEADS: usize = 4;
// 32 rather than the smaller round number a synthetic fixture would
// otherwise pick: `--kv-bits`'s TurboQuant codec requires a head_dim that
// is a power of two in 32..=512 (`model_io::rht_supported`), and this
// fixture is what `real_forward_llama_kv_quant.rs` opens with quantization
// on. No existing test in this family asserts a frozen digest or a
// specific head_dim (per this file's own doc: weights are untrained, so
// only token counts and structural invariants are ever checked), which is
// what makes widening it here safe -- confirmed by re-running the full
// existing llama/qwen3moe suite after this change, not assumed.
const HEAD_DIM: usize = 32;
const NUM_KV_HEADS: usize = 2;
/// Per-expert FFN width. The real Mixtral publishes ONE
/// `feed_forward_length` and has no shared expert, so `intermediate_size`
/// and `moe_intermediate_size` are the same number here as there.
const INTER: usize = 64;

/// A tiny `llama`-architecture (Mixtral-shaped) config. Every non-shape field
/// takes `model_io::mixtral_8x7b()`'s own value, for the reason its two
/// siblings pin theirs: the manifest's optional family-extension fields fall
/// back to the GEMMA baseline whatever family the manifest claims (AGENTS.md
/// Gotcha 24), so anything else has to be written explicitly and matched
/// explicitly.
pub fn tiny_llama_arch(vocab_size: i64, num_layers: i64, num_experts: i64) -> ArchConfig {
    tiny_gqa_moe_arch(vocab_size, num_layers, num_experts, ModelFamily::Llama)
}

/// A tiny DENSE `llama` config: the same architecture with `num_experts` 0,
/// which is how one `general.architecture` string covers Mistral and Llama
/// 2/3.x as well as the Mixtral MoEs (ROADMAP M4).
///
/// Nothing else changes. That is the point of the pair: a dense Llama's
/// family-EXTENSION fields are Mixtral's exactly (no output gate, no sandwich
/// norms, no subdim rope), and those are the only fields
/// `arch_validation` really binds to a baseline (AGENTS.md Gotcha 24).
pub fn tiny_dense_llama_arch(vocab_size: i64, num_layers: i64) -> ArchConfig {
    tiny_gqa_moe_arch(vocab_size, num_layers, 0, ModelFamily::Llama)
}

/// The same shape for either family that runs this layer graph.
///
/// `ModelFamily::Qwen3Moe` is the identical config with a different `family`
/// tag: the two architectures share every shape and behavioural field, and
/// differ only in the per-head q/k norms (extra TENSORS, written by the
/// builder below) and the RMS epsilon (not an `ArchConfig` field at all).
///
/// `num_experts == 0` is the DENSE case and gives `top_k_experts` 0 with it.
pub fn tiny_gqa_moe_arch(
    vocab_size: i64,
    num_layers: i64,
    num_experts: i64,
    family: ModelFamily,
) -> ArchConfig {
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
        rope_theta: 1_000_000.0,
        full_rope_theta: 1_000_000.0,
        // Full rotary, as the real file's `rope.dimension_count == head_dim`
        // says.
        partial_rotary_factor: 1.0,
        num_layers,
        num_experts,
        top_k_experts: num_experts.min(2),
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        full_attention_layer_mask: vec![1u8; num_layers as usize],
        hidden_activation: "silu".to_string(),
        family,
        attn_output_gate: false,
        // 0.25, not the real model's 128^-0.5. `validate_arch` compares this
        // f64 EXACTLY against the manifest's and serde_json's default parser
        // is only correct to ~1 ULP, so a scale that is not a binary fraction
        // cannot survive the round trip. These weights are untrained, so any
        // finite scale is equally meaningless. The REAL baseline's
        // `attention_scale` is not a binary fraction and is exercised by the
        // real install instead.
        attention_scale: 0.25,
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        shared_expert_gated: false,
        rope_neox_subdim: false,
        linear_attention: LinearAttentionConfig::NONE,
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

/// Writes a tiny Mixtral-shaped `.gturbo` install and returns the
/// `ArchConfig` needed to open it.
///
/// Goes through [`write_gemma4_install`] rather than a family-specific
/// wrapper, because that walk is family-PARAMETERIZED: it classifies routed
/// tensors by `arch.family`'s marker, and this architecture shares Gemma's
/// (`.experts.switch_glu.`). Only Qwen needed its own name, and only because
/// its marker differs.
pub fn build_synthetic_llama_real_install(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    num_experts: i64,
    model_id: &str,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_synthetic_gqa_moe_install(
        dir,
        vocab_size,
        num_layers,
        num_experts,
        model_id,
        ModelFamily::Llama,
    )
}

/// Writes a tiny DENSE `llama` install: the same attention block with a
/// plain gated FFN (`mlp.gate_proj` / `mlp.up_proj` / `mlp.down_proj`) in
/// place of the router and the routed experts (ROADMAP M4).
///
/// Those three names are not invented for the fixture: they are exactly what
/// `gguf_names.rs` maps a dense `llama` file's `ffn_gate` / `ffn_up` /
/// `ffn_down` to, and exactly what
/// `families/gemma4/mod.rs::encode_shared_expert_branch` already reads.
///
/// The install comes out with ZERO packed-expert layer files, which the
/// writer already handles (`write_gemma4_install` routes an empty
/// `out.layers` to `write_gturbo_install_with_resident_index`).
pub fn build_synthetic_dense_llama_install(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_gqa_install(dir, vocab_size, num_layers, 0, model_id, ModelFamily::Llama)
}

/// The same builder for either family, which is what makes the pair a real
/// test of the shared flow: pass `ModelFamily::Qwen3Moe` and it additionally
/// writes the two `[head_dim]` q/k norm vectors that architecture carries.
pub fn build_synthetic_gqa_moe_install(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    num_experts: i64,
    model_id: &str,
    family: ModelFamily,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    assert!(
        num_experts > 0,
        "build_synthetic_gqa_moe_install is the MoE half; \
         use build_synthetic_dense_llama_install for num_experts == 0"
    );
    build_gqa_install(dir, vocab_size, num_layers, num_experts, model_id, family)
}

/// The shared body. `num_experts == 0` writes the dense FFN and omits the
/// router; anything above writes the router and the routed experts.
fn build_gqa_install(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    num_experts: i64,
    model_id: &str,
    family: ModelFamily,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    let arch = tiny_gqa_moe_arch(vocab_size, num_layers, num_experts, family);
    let experts = num_experts as usize;
    let vocab = vocab_size as usize;

    let mut ts: Vec<Tensor> = Vec::new();
    // Untied head, as every real `llama` checkpoint here is.
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

    let mut overrides = std::collections::HashMap::new();
    for l in 0..num_layers as usize {
        let p = format!("language_model.model.layers.{l}");
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

        // Plain GQA: q_proj emits exactly `num_heads * head_dim` rows, with
        // no gate half. The per-head q/k norms exist on `qwen3moe` and not
        // on `llama`, which is one of the two differences the shared decode
        // flow keys on the family for.
        if family == ModelFamily::Qwen3Moe {
            for (i, norm) in ["q_norm", "k_norm"].iter().enumerate() {
                ts.push(bf16_vector(
                    &format!("{p}.self_attn.{norm}.weight"),
                    HEAD_DIM,
                    1.0,
                    seed + 60 + i as u64,
                ));
            }
        }
        ts.extend(int4_triple(
            &format!("{p}.self_attn.q_proj.weight"),
            NUM_HEADS * HEAD_DIM,
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

        if experts == 0 {
            // The DENSE half: one gated FFN, resident like every other
            // non-routed tensor, and no router at all.
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
                    seed + 70 + i as u64,
                ));
            }
        } else {
            // INT8 router, INT4 routed experts, and NO shared expert.
            ts.extend(int8_triple(
                &format!("{p}.mlp.gate.weight"),
                experts,
                HIDDEN,
                seed + 50,
            ));
            overrides.insert(format!("{p}.mlp.gate"), 8u32);
            for (i, role) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
                let (rows, cols) = if *role == "down_proj" {
                    (HIDDEN, INTER)
                } else {
                    (INTER, HIDDEN)
                };
                ts.extend(expert_int4_triple(
                    &format!("{p}.experts.switch_glu.{role}.weight"),
                    experts,
                    rows,
                    cols,
                    seed + 70 + i as u64,
                ));
            }
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
    write_gemma4_install(dir, &arch, model_id, &header, &source, &quant)?;
    Ok(arch)
}
