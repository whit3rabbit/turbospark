//! Builds a tiny `qwen3_5` install through the REAL checkpoint repack
//! pipeline (ROADMAP's 1-bit entry, step 3): the DENSE, ONE-BIT sibling of
//! [`crate::build_synthetic_qwen_gdn_moe_install`].
//!
//! **This fixture exists to be built BEFORE the 4.78 GiB stream rather than
//! after it**, which is `crates/repack` Gotcha 8's rule and what M4's dense
//! `llama` half paid three five-minute re-streams for. Every place the walk
//! assumed routed experts, or assumed BF16 companions at group 64, is
//! exercised here in milliseconds.
//!
//! Two things differ from the Qwen 3.6 fixture and they are the two the
//! checkpoint's own header and config establish:
//!
//! - **It is DENSE.** One `mlp.{gate,up,down}_proj` per layer, and NO
//!   router, shared expert or `.mlp.switch_mlp.` routed experts anywhere.
//!   The install therefore comes out with ZERO packed-expert layer files.
//! - **It is 1-BIT AT GROUP 128 WITH FP16 COMPANIONS.** All three axes
//!   together, because that is how a checkpoint carries them; the FP16 one
//!   is the axis nothing else can catch, since FP16 and BF16 are the same
//!   width.
//!
//! **The shape constraint that decided the constants: only COLUMN counts
//! have to be multiples of the group size.** `pass_through_packed` checks
//! `cols % group == 0` and leaves rows free, so this fixture is the Qwen 3.6
//! one with `HIDDEN` doubled from 64 to 128 and nothing else moved -- the
//! three other column dims (`o_proj`'s `NUM_HEADS * HEAD_DIM`, `down_proj`'s
//! `INTER`, `out_proj`'s `value_dim`) already came to 128.
//!
//! Weights are deterministic but NOT trained: generated tokens are
//! structurally real and semantically meaningless (AGENTS.md Gotcha 12).

use compute::quantize_int1_affine_symmetric;
use model_io::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig,
    ModelFamily, RopeScalingConfig,
};

use crate::gemma4_checkpoint::{write_qwen_gdn_dense_install, Gemma4Quant};
use crate::ranged_download::MemoryRangeSource;
use crate::safetensors_header::parse_header;
use crate::synthetic_real::{assemble_safetensors, deterministic_row, u16_le, Tensor};

/// 128 rather than the Qwen 3.6 fixture's 64: every quantized tensor's
/// COLUMN count must be a whole number of 128-element groups.
const HIDDEN: usize = 128;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = 32;
const NUM_KV_HEADS: usize = 2;
/// The DENSE FFN width. Qwen 3.6's fixture calls the same constant the
/// shared-expert-and-routed-expert width; here there are no experts and this
/// is the only FFN there is.
const INTER: usize = 128;

const LA_K_HEADS: usize = 2;
const LA_V_HEADS: usize = 4;
const LA_KEY_DIM: usize = 32;
const LA_VALUE_DIM: usize = 32;
const LA_CONV_K: usize = 4;

/// The checkpoint's group size. Named here rather than imported so the
/// fixture states the shape it is building.
const GROUP: usize = 128;

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
        // 0.125, not the mathematically-right 32^-0.5, for the reason
        // `tiny_qwen_gdn_moe_arch` gives: `validate_arch` compares this f64
        // EXACTLY against the manifest's and serde_json's default parser is
        // only correct to ~1 ULP, so a scale that is not a binary fraction
        // cannot survive the round trip (AGENTS.md Gotcha 24). The real
        // baseline's 0.0625 is a power of two and is unaffected.
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

/// An UNQUANTIZED vector, F16 like every unquantized tensor in the real
/// checkpoint -- and unlike the Qwen 3.6 fixture's BF16, which this file was
/// forked from.
///
/// **That difference is the whole reason this helper exists rather than
/// reusing `bf16_vector`, and it is the second time this fixture has earned
/// its place.** The 1-bit entry's step 3 got the quantized triple's F16
/// companions right off the real header and left the RAW tensors at the
/// sibling's BF16, so the fixture said nothing about a walk that wrote F16
/// bytes under a dtype tag no reader in `crates/runtime` honours. Every
/// consumer of an unquantized tensor decodes BF16 off the byte size, so the
/// install would have opened, decoded, and been wrong by a factor of 2^112 on
/// every norm.
///
/// The values go through F16 rather than being BF16 values relabelled,
/// because a BF16 value narrows back losslessly and a fixture that cannot
/// lose a bit cannot exercise `narrow_raw_to_bf16`'s counting at all.
fn f16_vector(name: &str, n: usize, center: f32, seed: u64) -> Tensor {
    let bits: Vec<u16> = deterministic_row(seed, n)
        .iter()
        .map(|&j| compute::f32_to_f16(center + j * 0.05))
        .collect();
    Tensor {
        name: name.to_string(),
        dtype: "F16",
        shape: vec![n as u64],
        bytes: u16_le(&bits),
    }
}

/// One 1-bit-affine weight plus its two FP16 companions.
///
/// The sibling of `int4_triple`, and it differs on all three axes at once:
/// 32 elements per packed `u32` word rather than 8, one companion per 128
/// elements rather than per 64, and `F16` rather than `BF16`. The dtype
/// string is the one that matters here -- `pass_through_packed` requires it
/// per bit width, and a fixture writing `BF16` would produce an install of
/// exactly the right size whose scales are wrong by orders of magnitude.
fn int1_triple(name: &str, rows: usize, cols: usize, seed: u64) -> Vec<Tensor> {
    assert_eq!(
        cols % GROUP,
        0,
        "{name}: {cols} columns is not a whole number of {GROUP}-element groups"
    );
    let base = name.strip_suffix(".weight").unwrap();
    let mut packed = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    for r in 0..rows {
        let q = quantize_int1_affine_symmetric(
            &deterministic_row(seed.wrapping_add(r as u64 * 97 + 1), cols),
            GROUP,
        );
        packed.extend_from_slice(&q.packed);
        scales.extend_from_slice(&q.scales);
        biases.extend_from_slice(&q.biases);
    }
    let groups = cols / GROUP;
    vec![
        Tensor {
            name: name.to_string(),
            dtype: "U32",
            shape: vec![rows as u64, (cols / 32) as u64],
            bytes: packed,
        },
        Tensor {
            name: format!("{base}.scales"),
            dtype: "F16",
            shape: vec![rows as u64, groups as u64],
            bytes: u16_le(&scales),
        },
        Tensor {
            name: format!("{base}.biases"),
            dtype: "F16",
            shape: vec![rows as u64, groups as u64],
            bytes: u16_le(&biases),
        },
    ]
}

/// Writes a tiny `qwen3_5` `.gturbo` install and returns the `ArchConfig`
/// needed to open it.
///
/// Takes no `num_experts`: the family is dense and a fixture that could be
/// asked for experts would be a fixture no real file corresponds to.
pub fn build_synthetic_qwen_gdn_dense_install(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    let arch = tiny_qwen_gdn_dense_arch(vocab_size, num_layers);
    let vocab = vocab_size as usize;
    let la = &arch.linear_attention;
    let qkv_dim = la.qkv_dim() as usize;
    let value_dim = la.value_dim() as usize;
    let v_heads = LA_V_HEADS;

    let mut ts: Vec<Tensor> = Vec::new();
    // Untied head, and BOTH quantized at one bit -- which is what the real
    // checkpoint's safetensors header says (`embed_tokens` and `lm_head` are
    // two of its 498 tensors carrying `.scales`).
    ts.extend(int1_triple(
        "language_model.model.embed_tokens.weight",
        vocab,
        HIDDEN,
        1,
    ));
    ts.extend(int1_triple(
        "language_model.lm_head.weight",
        vocab,
        HIDDEN,
        2,
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
            // attn_output_gate: q_proj emits per-head [query; gate] pairs.
            ts.extend(int1_triple(
                &format!("{p}.self_attn.q_proj.weight"),
                2 * NUM_HEADS * HEAD_DIM,
                HIDDEN,
                seed + 1,
            ));
            for (i, role) in ["k_proj", "v_proj"].iter().enumerate() {
                ts.extend(int1_triple(
                    &format!("{p}.self_attn.{role}.weight"),
                    NUM_KV_HEADS * HEAD_DIM,
                    HIDDEN,
                    seed + 2 + i as u64,
                ));
            }
            ts.extend(int1_triple(
                &format!("{p}.self_attn.o_proj.weight"),
                HIDDEN,
                NUM_HEADS * HEAD_DIM,
                seed + 4,
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
                ts.extend(int1_triple(
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
            // A_log and dt_bias carry NO `.weight` suffix in the real
            // checkpoint (AGENTS.md Gotcha 26).
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

        // THE DENSE FFN, and the whole reason this fixture exists beside the
        // Qwen 3.6 one: no `mlp.gate`, no `mlp.shared_expert*`, no
        // `.mlp.switch_mlp.`. Three projections and nothing else.
        for (i, role) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
            let (rows, cols) = if *role == "down_proj" {
                (HIDDEN, INTER)
            } else {
                (INTER, HIDDEN)
            };
            ts.extend(int1_triple(
                &format!("{p}.mlp.{role}.weight"),
                rows,
                cols,
                seed + 60 + i as u64,
            ));
        }
    }
    ts.push(f16_vector(
        "language_model.model.norm.weight",
        HIDDEN,
        1.0,
        7,
    ));

    let blob = assemble_safetensors(&ts);
    let source = MemoryRangeSource::new(&blob);
    let header = parse_header(&blob, crate::safetensors_header::DEFAULT_MAX_HEADER_BYTES)?;
    // The real checkpoint's `quantization` object, verbatim: two keys and no
    // per-tensor overrides.
    let quant = Gemma4Quant {
        default_bits: 1,
        group_size: GROUP as u32,
        bits_overrides: std::collections::HashMap::new(),
    };
    write_qwen_gdn_dense_install(dir, &arch, model_id, &header, &source, &quant)?;
    Ok(arch)
}

/// The depthwise conv kernel: BF16, rank 3 `[channels, taps, 1]`, exactly as
/// the Qwen 3.6 fixture writes it. Unquantized in the real checkpoint too --
/// its 48 `conv1d` tensors carry no `.scales`.
fn conv1d_weight(name: &str, channels: usize, taps: usize, seed: u64) -> Tensor {
    let flat = f16_vector(name, channels * taps, 0.0, seed);
    Tensor {
        name: flat.name,
        dtype: flat.dtype,
        shape: vec![channels as u64, taps as u64, 1],
        bytes: flat.bytes,
    }
}
