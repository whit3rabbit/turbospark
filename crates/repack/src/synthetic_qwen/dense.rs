//! Builds a tiny `qwen3_5` install through the REAL checkpoint repack pipeline.

use compute::{quantize_int1_affine_symmetric, quantize_int2_affine_ternary};
use model_io::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig,
    ModelFamily, RopeScalingConfig,
};

use crate::gemma4_checkpoint::{write_qwen_gdn_dense_install, Gemma4Quant};
use crate::ranged_download::MemoryRangeSource;
use crate::safetensors_header::parse_header;
use crate::synthetic_real::{assemble_safetensors, bf16_vector, deterministic_row, u16_le, Tensor};

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

/// An UNQUANTIZED vector, F16 like every unquantized tensor in the real
/// checkpoint -- and unlike the Qwen 3.6 fixture's BF16, which this file was
/// forked from.
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

/// One sub-4-bit-affine weight plus its two FP16 companions, at `bits` = 1 or
/// 2.
fn packed_triple(name: &str, rows: usize, cols: usize, seed: u64, bits: u32) -> Vec<Tensor> {
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
        let row = deterministic_row(seed.wrapping_add(r as u64 * 97 + 1), cols);
        // Each width takes its OWN quantizer, not a shared one with a level
        // count: at one bit the rule is a sign and at two it is a ternary
        // threshold, and both are stated where the layout they produce is.
        let (p, s, b) = match bits {
            1 => {
                let q = quantize_int1_affine_symmetric(&row, GROUP);
                (q.packed, q.scales, q.biases)
            }
            2 => {
                let q = quantize_int2_affine_ternary(&row, GROUP);
                (q.packed, q.scales, q.biases)
            }
            other => panic!("this fixture builds 1- or 2-bit installs, not {other}"),
        };
        packed.extend_from_slice(&p);
        scales.extend_from_slice(&s);
        biases.extend_from_slice(&b);
    }
    let groups = cols / GROUP;
    vec![
        Tensor {
            name: name.to_string(),
            dtype: "U32",
            shape: vec![rows as u64, (cols * bits as usize / 32) as u64],
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

/// A BF16 rank-2 matrix, the dtype and rank the MTP head's projections
/// carry in the official checkpoint.
///
/// Rank-2 is the whole discriminator the walk uses to decide "quantize this"
/// against "narrow this", so a helper that produced rank-1 would silently
/// route a projection down the norm path.
fn bf16_matrix(name: &str, rows: usize, cols: usize, seed: u64) -> Tensor {
    let mut bits: Vec<u16> = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for &j in deterministic_row(seed.wrapping_add(r as u64 * 131 + 1), cols).iter() {
            bits.push(compute::f32_to_bf16(j * 0.05));
        }
    }
    Tensor {
        name: name.to_string(),
        dtype: "BF16",
        shape: vec![rows as u64, cols as u64],
        bytes: u16_le(&bits),
    }
}

/// The multi-token-prediction head, as `Qwen/Qwen3.8-27B` publishes it:
/// fifteen tensors, BF16 throughout, name for name against
/// `crates/repack/tests/mtp_head_network.rs`'s `EXPECTED` table.
///
/// **BF16 where the trunk around it is F16-and-packed, and that asymmetry is
/// the real situation rather than fixture sloppiness.** The trunk install is
/// streamed from an mlx-community conversion that DROPS `mtp.*` entirely, so
/// the head can only come from the official checkpoint -- two publishers, two
/// dtypes, one install. It is also what makes the narrowing assertion mean
/// something: a BF16 source narrows to BF16 losslessly, so this fixture's
/// head must report ZERO lossy values while Bonsai's F16 trunk reports many.
///
/// Three shapes here are load-bearing and each defeats a mutation that a
/// tidier fixture would pass:
///
/// - `fc` is `[hidden, 2 * hidden]`, the only tensor in the model with that
///   width. It takes the CONCATENATION of a normalized next-token embedding
///   and a normalized trunk hidden state, so a fixture with `2 * hidden`
///   anywhere else could not catch a transposed or half-width read.
/// - `q_proj` has `2 * num_heads * head_dim` rows while `o_proj`'s INPUT is
///   the unhalved `num_heads * head_dim`. That asymmetry is `attn_output_gate`
///   being true; a non-gated head would have the same number on both, so
///   equal widths would make the gate unobservable.
/// - the two `pre_fc_norm_*` vectors take DIFFERENT seeds, so swapping them
///   changes the output. Seeded alike they are interchangeable and the
///   embedding/hidden ordering is untestable.
fn mtp_head_tensors() -> Vec<Tensor> {
    let q_out = NUM_HEADS * HEAD_DIM;
    let kv_out = NUM_KV_HEADS * HEAD_DIM;
    let mut ts = vec![
        // The head's own structure, above the block.
        bf16_matrix("mtp.fc.weight", HIDDEN, 2 * HIDDEN, 9_100),
        bf16_vector("mtp.pre_fc_norm_embedding.weight", HIDDEN, 1.0, 9_101),
        bf16_vector("mtp.pre_fc_norm_hidden.weight", HIDDEN, 1.0, 9_102),
        bf16_vector("mtp.norm.weight", HIDDEN, 1.0, 9_103),
    ];
    // The block: every shape a TRUNK full-attention layer's, which is what
    // makes the draft step reuse `families/qwen/attn.rs` with no new kernel.
    ts.push(bf16_vector(
        "mtp.layers.0.input_layernorm.weight",
        HIDDEN,
        1.0,
        9_110,
    ));
    ts.push(bf16_vector(
        "mtp.layers.0.post_attention_layernorm.weight",
        HIDDEN,
        1.0,
        9_111,
    ));
    ts.push(bf16_matrix(
        "mtp.layers.0.self_attn.q_proj.weight",
        2 * q_out,
        HIDDEN,
        9_120,
    ));
    ts.push(bf16_matrix(
        "mtp.layers.0.self_attn.k_proj.weight",
        kv_out,
        HIDDEN,
        9_121,
    ));
    ts.push(bf16_matrix(
        "mtp.layers.0.self_attn.v_proj.weight",
        kv_out,
        HIDDEN,
        9_122,
    ));
    ts.push(bf16_matrix(
        "mtp.layers.0.self_attn.o_proj.weight",
        HIDDEN,
        q_out,
        9_123,
    ));
    ts.push(bf16_vector(
        "mtp.layers.0.self_attn.q_norm.weight",
        HEAD_DIM,
        1.0,
        9_124,
    ));
    ts.push(bf16_vector(
        "mtp.layers.0.self_attn.k_norm.weight",
        HEAD_DIM,
        1.0,
        9_125,
    ));
    ts.push(bf16_matrix(
        "mtp.layers.0.mlp.gate_proj.weight",
        INTER,
        HIDDEN,
        9_130,
    ));
    ts.push(bf16_matrix(
        "mtp.layers.0.mlp.up_proj.weight",
        INTER,
        HIDDEN,
        9_131,
    ));
    ts.push(bf16_matrix(
        "mtp.layers.0.mlp.down_proj.weight",
        HIDDEN,
        INTER,
        9_132,
    ));
    ts
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
    build_synthetic_qwen_gdn_dense_install_inner(dir, vocab_size, num_layers, model_id, bits, false)
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
    build_synthetic_qwen_gdn_dense_install_inner(dir, vocab_size, num_layers, model_id, bits, true)
}

fn build_synthetic_qwen_gdn_dense_install_inner(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
    bits: u32,
    with_mtp: bool,
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

    let blob = assemble_safetensors(&ts);
    let source = MemoryRangeSource::new(&blob);
    let header = parse_header(&blob, crate::safetensors_header::DEFAULT_MAX_HEADER_BYTES)?;
    let quant = Gemma4Quant {
        default_bits: bits,
        group_size: GROUP as u32,
        bits_overrides: std::collections::HashMap::new(),
    };
    write_qwen_gdn_dense_install(dir, &arch, model_id, &header, &source, &quant)?;
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
