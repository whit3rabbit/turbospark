//! Builds a tiny `qwen3_5` install through the REAL checkpoint repack pipeline.

use model_io::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig,
    ModelFamily, RopeScalingConfig,
};

use super::dense_tensors::{dflash_drafter_tensors, f16_vector, mtp_head_tensors, packed_triple};
use crate::gemma4_checkpoint::{write_qwen_gdn_dense_install, Gemma4Quant};
use crate::ranged_download::MemoryRangeSource;
use crate::safetensors_header::parse_header;
use crate::synthetic_real::{assemble_safetensors, Tensor};

pub(crate) const HIDDEN: usize = 128;
pub(crate) const NUM_HEADS: usize = 4;
pub(crate) const HEAD_DIM: usize = 32;
pub(crate) const NUM_KV_HEADS: usize = 2;
/// The DENSE FFN width. Qwen 3.6's fixture calls the same constant the
/// shared-expert-and-routed-expert width; here there are no experts and this
/// is the only FFN there is.
pub(crate) const INTER: usize = 128;

const LA_K_HEADS: usize = 2;
const LA_V_HEADS: usize = 4;
const LA_KEY_DIM: usize = 32;
const LA_VALUE_DIM: usize = 32;
const LA_CONV_K: usize = 4;

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
    }
}

/// Writes a tiny `qwen3_5` `.gturbo` install and returns the `ArchConfig`
/// needed to open it.
pub fn build_synthetic_qwen_gdn_dense_install(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_synthetic_qwen_gdn_dense_install_at_bits(dir, vocab_size, num_layers, model_id, 1)
}

/// [`build_synthetic_qwen_gdn_dense_install`] at an explicit affine width:
/// 1 for `prism-ml/Bonsai-27B-mlx-1bit`, 2 for
/// `prism-ml/Ternary-Bonsai-27B-mlx-2bit` (ROADMAP's ternary entry).
pub fn build_synthetic_qwen_gdn_dense_install_at_bits(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
    bits: u32,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_synthetic_qwen_gdn_dense_install_inner(
        dir, vocab_size, num_layers, model_id, bits, false, false, false,
    )
}

/// [`build_synthetic_qwen_gdn_dense_install_at_bits`] with the
/// multi-token-prediction head attached (`docs/MTP_SPECULATIVE.md`, step 1).
///
/// A third entry point rather than a widened signature, following this file's
/// own delegation chain: five callers take the two existing forms and none of
/// them wants a head, so adding a parameter to those would edit five call
/// sites to say `false`.
pub fn build_synthetic_qwen_gdn_dense_install_with_mtp(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
    bits: u32,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_synthetic_qwen_gdn_dense_install_inner(
        dir, vocab_size, num_layers, model_id, bits, true, false, false,
    )
}

/// The same install through the STREAMED writer, which is the one every real
/// checkpoint takes.
///
/// **This entry point exists because its absence shipped a bug.** The head's
/// ingest landed in `orchestrate_gemma4_checkpoint_sharded` alone, every
/// fixture went through that non-streamed path, and
/// `write_gemma4_install_streamed` classified `mtp.*` correctly and then
/// never read it -- so the first real stream that asked for a head wrote a
/// byte-identical HEADLESS install and said nothing. A fixture has to
/// exercise the WRITER the download will use, not just the walk it shares.
pub fn build_synthetic_qwen_gdn_dense_install_with_mtp_streamed(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
    bits: u32,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_synthetic_qwen_gdn_dense_install_inner(
        dir, vocab_size, num_layers, model_id, bits, true, true, false,
    )
}

/// [`build_synthetic_qwen_gdn_dense_install`] with the DFlash2 drafter
/// attached (`docs/DFLASH2.md`), through the NON-streamed writer.
pub fn build_synthetic_qwen_gdn_dense_install_with_dflash(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
    bits: u32,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_synthetic_qwen_gdn_dense_install_inner(
        dir, vocab_size, num_layers, model_id, bits, false, false, true,
    )
}

/// BOTH drafters in one install, which no real checkpoint ships and which
/// exists to make a DISAMBIGUATING fixture possible.
///
/// `crates/cli`'s `resolve_drafter` picks between them off the resident
/// index, and the clause that matters is that an install carrying both keeps
/// the pre-existing MTP behaviour with no DFlash2 note. A dflash-only
/// fixture cannot see that clause: drop it and the dflash-only case still
/// passes, so the mutation survives and the test reads stronger than it is.
/// The install this builds is the only input on which the two rules differ.
pub fn build_synthetic_qwen_gdn_dense_install_with_both_drafters(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
    bits: u32,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_synthetic_qwen_gdn_dense_install_inner(
        dir, vocab_size, num_layers, model_id, bits, true, false, true,
    )
}

/// The same drafter through the STREAMED writer, which is the one the real
/// download takes. Exists for the head's reason verbatim: a drafter arm
/// that only the non-streamed path read would stream a drafterless install
/// and say nothing (see `both_writers_carry_the_mtp_head`'s header).
pub fn build_synthetic_qwen_gdn_dense_install_with_dflash_streamed(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
    bits: u32,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_synthetic_qwen_gdn_dense_install_inner(
        dir, vocab_size, num_layers, model_id, bits, false, true, true,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_synthetic_qwen_gdn_dense_install_inner(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
    bits: u32,
    with_mtp: bool,
    streamed: bool,
    with_dflash: bool,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    let arch = tiny_qwen_gdn_dense_arch(vocab_size, num_layers);
    let vocab = vocab_size as usize;
    let la = &arch.linear_attention;
    let qkv_dim = la.qkv_dim() as usize;
    let value_dim = la.value_dim() as usize;
    let v_heads = LA_V_HEADS;

    let mut ts: Vec<Tensor> = Vec::new();
    ts.extend(packed_triple(
        "language_model.model.embed_tokens.weight",
        vocab,
        HIDDEN,
        1,
        bits,
    ));
    ts.extend(packed_triple(
        "language_model.lm_head.weight",
        vocab,
        HIDDEN,
        2,
        bits,
    ));

    for l in 0..num_layers as usize {
        let p = format!("language_model.model.layers.{l}");
        let seed = 1000 * (l as u64 + 1);
        let is_full = arch.layer_is_full(l);

        for (i, norm) in ["input_layernorm", "post_attention_layernorm"]
            .iter()
            .enumerate()
        {
            ts.push(f16_vector(
                &format!("{p}.{norm}.weight"),
                HIDDEN,
                1.0,
                seed + 40 + i as u64,
            ));
        }

        if is_full {
            ts.extend(packed_triple(
                &format!("{p}.self_attn.q_proj.weight"),
                2 * NUM_HEADS * HEAD_DIM,
                HIDDEN,
                seed + 1,
                bits,
            ));
            for (i, role) in ["k_proj", "v_proj"].iter().enumerate() {
                ts.extend(packed_triple(
                    &format!("{p}.self_attn.{role}.weight"),
                    NUM_KV_HEADS * HEAD_DIM,
                    HIDDEN,
                    seed + 2 + i as u64,
                    bits,
                ));
            }
            ts.extend(packed_triple(
                &format!("{p}.self_attn.o_proj.weight"),
                HIDDEN,
                NUM_HEADS * HEAD_DIM,
                seed + 4,
                bits,
            ));
            for (i, norm) in ["q_norm", "k_norm"].iter().enumerate() {
                ts.push(f16_vector(
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
                ts.extend(packed_triple(
                    &format!("{p}.linear_attn.{name}.weight"),
                    rows,
                    cols,
                    seed + s,
                    bits,
                ));
            }
            ts.push(conv1d_weight(
                &format!("{p}.linear_attn.conv1d.weight"),
                qkv_dim,
                LA_CONV_K,
                seed + 10,
            ));
            ts.push(f16_vector(
                &format!("{p}.linear_attn.A_log"),
                v_heads,
                0.0,
                seed + 11,
            ));
            ts.push(f16_vector(
                &format!("{p}.linear_attn.dt_bias"),
                v_heads,
                0.0,
                seed + 12,
            ));
            ts.push(f16_vector(
                &format!("{p}.linear_attn.norm.weight"),
                LA_VALUE_DIM,
                1.0,
                seed + 13,
            ));
        }

        for (i, role) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
            let (rows, cols) = if *role == "down_proj" {
                (HIDDEN, INTER)
            } else {
                (INTER, HIDDEN)
            };
            ts.extend(packed_triple(
                &format!("{p}.mlp.{role}.weight"),
                rows,
                cols,
                seed + 60 + i as u64,
                bits,
            ));
        }
    }
    ts.push(f16_vector(
        "language_model.model.norm.weight",
        HIDDEN,
        1.0,
        7,
    ));
    if with_mtp {
        ts.extend(mtp_head_tensors());
    }
    if with_dflash {
        ts.extend(dflash_drafter_tensors(vocab));
    }

    let blob = assemble_safetensors(&ts);
    let source = MemoryRangeSource::new(&blob);
    let header = parse_header(&blob, crate::safetensors_header::DEFAULT_MAX_HEADER_BYTES)?;
    let quant = Gemma4Quant {
        default_bits: bits,
        group_size: group_for(bits) as u32,
        bits_overrides: std::collections::HashMap::new(),
    };
    if streamed {
        let shards = crate::gemma4_checkpoint::Gemma4Shards::single(&header, &source);
        crate::gemma4_checkpoint::write_qwen_gdn_dense_install_streamed(
            dir,
            &arch,
            model_id,
            &shards,
            &quant,
            |_| {},
        )?;
    } else {
        write_qwen_gdn_dense_install(dir, &arch, model_id, &header, &source, &quant)?;
    }
    Ok(arch)
}

fn conv1d_weight(name: &str, channels: usize, taps: usize, seed: u64) -> Tensor {
    let flat = f16_vector(name, channels * taps, 0.0, seed);
    Tensor {
        name: flat.name,
        dtype: flat.dtype,
        shape: vec![channels as u64, taps as u64, 1],
        bytes: flat.bytes,
    }
}
