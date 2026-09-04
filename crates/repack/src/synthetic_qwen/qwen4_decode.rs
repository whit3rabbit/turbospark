//! Builds a DECODE-CAPABLE tiny `qwen4_exp` install: every layer carries
//! real hyper-connection, GDN-or-attention, and MoE weights, plus a fully
//! populated PLE layer (key/value/norm/conv1d projections, not just the
//! n-gram table `synthetic_qwen::qwen4`'s fixture stops at). This is what
//! `crates/runtime`'s `real_forward_qwen4.rs` exercises the decode flow
//! against -- that other fixture's own doc says explicitly it is not built
//! for this.
//!
//! Untrained, deterministic weights (the same xorshift scheme every other
//! synthetic fixture here uses), so nothing about the OUTPUT is meaningful;
//! what a test built on this can prove is that the flow runs, produces
//! finite logits, and reads the tensors it claims to (perturbation cases
//! plus a frozen digest, per `docs/NEW_MODEL.md` and AGENTS.md Gotcha 23).

use std::collections::HashMap;

use model_io::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig,
    ModelFamily, PleConfig,
};

use crate::gemma4_checkpoint::{write_gemma4_install, Gemma4Quant, Gemma4Shards};
use crate::ranged_download::MemoryRangeSource;
use crate::safetensors_header::parse_header;
use crate::synthetic_tensors::{
    assemble_safetensors, bf16_vector, deterministic_row, expert_int4_triple, int4_triple,
    int8_triple, u16_le, Tensor,
};

pub const HIDDEN: usize = 128;
pub const HC_COUNT: usize = 2;
pub const HC_LOWRANK: usize = 64;
pub const NUM_HEADS: usize = 4;
pub const HEAD_DIM: usize = 32;
pub const NUM_KV_HEADS: usize = 2;

const LA_K_HEADS: usize = 2;
const LA_V_HEADS: usize = 4;
const LA_KEY_DIM: usize = 32;
const LA_VALUE_DIM: usize = 32;
const LA_CONV_K: usize = 4;

pub const NUM_EXPERTS: usize = 4;
pub const TOP_K: usize = 2;
const MOE_INTER: usize = 64;
/// The shared expert's width (`ArchConfig::intermediate_size` on this
/// family, matching `families/qwen/moe.rs`'s own convention).
const SHARED_INTER: usize = 64;

/// `(ngram_size - 1) * heads_per_ngram` = `2 * 2`.
const NGRAM_HEADS: usize = 4;
/// One whole group and nothing more (`NGRAM_GROUP_SIZE` in
/// `gemma4_checkpoint::ngram`), so `ple_embed_dim` comes out to
/// `NGRAM_HEADS * NGRAM_HEAD_DIM` = 128 = [`HIDDEN`].
const NGRAM_HEAD_DIM: usize = 32;
const NGRAM_GROUP_SIZE: usize = 32;
pub const PLE_LAYER: usize = 1;
pub const NGRAM_EOS_TOKEN_ID: i64 = 999;
/// Small vocabulary base so the derived prime head-vocab sizes stay in the
/// tens rather than the tens of millions -- the fixture's table has to
/// physically hold every row the hash can address.
const NGRAM_VOCAB_BASE: i64 = 50;

/// `num_layers` this fixture always builds: enough to cover GDN, the PLE
/// layer (also GDN, matching the real checkpoint's own layer 1), and one
/// QSA-as-dense-attention layer, with MoE on every one of them.
pub const NUM_LAYERS: usize = 4;
/// `[GDN, GDN+PLE, attention, GDN]` -- `mask[l] == 1` is full attention,
/// `2` is linear (GDN).
fn layer_mask() -> Vec<u8> {
    vec![2, 2, 1, 2]
}

fn wide_dim() -> usize {
    HIDDEN * HC_COUNT
}

fn linear_attention() -> LinearAttentionConfig {
    LinearAttentionConfig {
        num_k_heads: LA_K_HEADS as i64,
        num_v_heads: LA_V_HEADS as i64,
        key_head_dim: LA_KEY_DIM as i64,
        value_head_dim: LA_VALUE_DIM as i64,
        conv_kernel_size: LA_CONV_K as i64,
        // The SIGMOID variant -- this fixture's whole reason for not being
        // `synthetic_qwen::dense`'s GDN shape reused verbatim.
        output_gate_sigmoid: true,
    }
}

/// A tiny but fully decode-shaped `qwen4_exp` architecture.
pub fn tiny_qwen4_exp_decode_arch(vocab_size: i64) -> ArchConfig {
    let num_layers = NUM_LAYERS as i64;
    ArchConfig {
        hidden_size: HIDDEN as i64,
        // The shared expert's width on this family (`families/qwen/moe.rs`'s
        // convention, matching Qwen 3.6's).
        intermediate_size: SHARED_INTER as i64,
        moe_intermediate_size: MOE_INTER as i64,
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
        num_experts: NUM_EXPERTS as i64,
        top_k_experts: TOP_K as i64,
        // The real checkpoint's own value: `lm_head` is its own tensor.
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        full_attention_layer_mask: layer_mask(),
        hidden_activation: "silu".to_string(),
        family: ModelFamily::Qwen4Exp,
        attn_output_gate: true,
        // A binary fraction, not `HEAD_DIM^-0.5` (irrational at this width):
        // `arch_validation` compares manifest floats with `!=` against a
        // ~1-ULP parser (AGENTS.md Gotcha 24), matching
        // `synthetic_qwen::dense_arch`'s own workaround.
        attention_scale: 0.125,
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        shared_expert_gated: true,
        rope_neox_subdim: true,
        linear_attention: linear_attention(),
        // Real baseline's own values (`model_io::qwen4_exp_125b_a6b()`):
        // this flow reads only `index_budget` today (the QSA-indexer
        // refusal), so the rest are carried for a config that is otherwise
        // self-consistent rather than because anything dispatches on them.
        compressed_attention: CompressedAttentionConfig {
            index_n_heads: 4,
            index_kv_heads: 1,
            index_head_dim: 128,
            index_top_k: 512,
            index_budget: 2048,
            csa_compress_rate: 4,
            q_lora_rank: 0,
            o_lora_rank: 0,
            o_groups: 0,
            rope_head_dim: 0,
            hca_compress_rate: 0,
            compress_rope_theta: 0.0,
            rope_scaling_factor: 0.0,
            rope_scaling_original_max: 0,
            rope_scaling_beta_fast: 0.0,
            rope_scaling_beta_slow: 0.0,
        },
        hyper_connections: HyperConnectionConfig {
            mult: HC_COUNT as i64,
            lowrank: HC_LOWRANK as i64,
            sinkhorn_iters: 0,
            eps: 0.0,
        },
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: model_io::RopeScalingConfig::NONE,
        vision: model_io::VisionConfig::NONE,
        ple: PleConfig {
            ngram_size: 3,
            heads_per_ngram: 2,
            ngram_vocab_size_base: NGRAM_VOCAB_BASE,
            make_divisible_by: 1,
            split_ngram_parts: 1,
            ple_embed_dim: (NGRAM_HEADS * NGRAM_HEAD_DIM) as i64,
            conv_kernel_size: 4,
            layer_ids: vec![PLE_LAYER as i64 + 1],
            seed: 1234,
            eos_token_id: NGRAM_EOS_TOKEN_ID,
        },
    }
}

/// One hyper-connection block's four tensors under `prefix` (either
/// `attn_hyper_connection` or `mlp_hyper_connection`, per layer).
fn hyper_connection_tensors(prefix: &str, seed: u64) -> Vec<Tensor> {
    let wide = wide_dim();
    let mut ts = vec![bf16_vector(
        &format!("{prefix}.hc_norm.weight"),
        wide,
        // Centered convention: weight is an OFFSET FROM UNITY, initialized
        // near zero in the real checkpoint.
        0.0,
        seed,
    )];
    ts.extend(int4_triple(
        &format!("{prefix}.input_mix_weight_down.weight"),
        HC_LOWRANK,
        wide,
        seed + 1,
    ));
    ts.extend(int4_triple(
        &format!("{prefix}.input_mix_weight_up.weight"),
        wide,
        HC_LOWRANK,
        seed + 2,
    ));
    ts.extend(int4_triple(
        &format!("{prefix}.block_inject_weight.weight"),
        HC_COUNT,
        wide,
        seed + 3,
    ));
    ts
}

/// The final `hyper_connection_mixer`: the same `hc_norm` +
/// `input_mix_weight_{down,up}` triple, with NO `block_inject_weight`
/// (`use_combine=False`).
fn final_mixer_tensors() -> Vec<Tensor> {
    let wide = wide_dim();
    let prefix = "language_model.model.hyper_connection_mixer";
    let mut ts = vec![bf16_vector(
        &format!("{prefix}.hc_norm.weight"),
        wide,
        0.0,
        8_000,
    )];
    ts.extend(int4_triple(
        &format!("{prefix}.input_mix_weight_down.weight"),
        HC_LOWRANK,
        wide,
        8_001,
    ));
    ts.extend(int4_triple(
        &format!("{prefix}.input_mix_weight_up.weight"),
        wide,
        HC_LOWRANK,
        8_002,
    ));
    ts
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

fn gdn_layer_tensors(p: &str, seed: u64) -> Vec<Tensor> {
    let qkv_dim = 2 * LA_K_HEADS * LA_KEY_DIM + LA_V_HEADS * LA_VALUE_DIM;
    let value_dim = LA_V_HEADS * LA_VALUE_DIM;
    let mut ts = Vec::new();
    for (name, rows, cols, s) in [
        ("in_proj_qkv", qkv_dim, HIDDEN, 1u64),
        ("in_proj_z", value_dim, HIDDEN, 2),
        ("in_proj_a", LA_V_HEADS, HIDDEN, 3),
        ("in_proj_b", LA_V_HEADS, HIDDEN, 4),
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
    // NO `.weight` suffix on these two, matching the real checkpoint
    // (`crates/repack/CLAUDE.md` Gotcha 15).
    ts.push(bf16_vector(
        &format!("{p}.linear_attn.A_log"),
        LA_V_HEADS,
        0.0,
        seed + 11,
    ));
    ts.push(bf16_vector(
        &format!("{p}.linear_attn.dt_bias"),
        LA_V_HEADS,
        0.0,
        seed + 12,
    ));
    // PLAIN (not centered) gated norm, per `docs/QWEN4_PHASE0.md` item 9's
    // one exception -- weight initialized near ONE, not zero.
    ts.push(bf16_vector(
        &format!("{p}.linear_attn.norm.weight"),
        LA_VALUE_DIM,
        1.0,
        seed + 13,
    ));
    ts
}

fn attention_layer_tensors(p: &str, seed: u64) -> Vec<Tensor> {
    let q_out = NUM_HEADS * HEAD_DIM;
    let kv_out = NUM_KV_HEADS * HEAD_DIM;
    let mut ts = Vec::new();
    ts.extend(int4_triple(
        &format!("{p}.self_attn.q_proj.weight"),
        2 * q_out,
        HIDDEN,
        seed + 1,
    ));
    for (i, role) in ["k_proj", "v_proj"].iter().enumerate() {
        ts.extend(int4_triple(
            &format!("{p}.self_attn.{role}.weight"),
            kv_out,
            HIDDEN,
            seed + 2 + i as u64,
        ));
    }
    ts.extend(int4_triple(
        &format!("{p}.self_attn.o_proj.weight"),
        HIDDEN,
        q_out,
        seed + 4,
    ));
    // CENTERED unconditionally -- this family has no plain-norm sibling
    // sharing this call (`families/qwen4/attn.rs`'s own doc).
    for (i, norm) in ["q_norm", "k_norm"].iter().enumerate() {
        ts.push(bf16_vector(
            &format!("{p}.self_attn.{norm}.weight"),
            HEAD_DIM,
            0.0,
            seed + 20 + i as u64,
        ));
    }
    ts
}

/// A raw (unquantized) BF16 router matrix, matching what the real
/// `qwen4_exp` checkpoint ships instead of the pre-packed INT8 [`int8_triple`]
/// produces. Router-shaped only: `int8_triple`'s `.weight`/`.scales`/`.biases`
/// triple is what `pass_through_packed` reads, and a raw tensor has no
/// companions to build.
fn bf16_matrix(name: &str, rows: usize, cols: usize, seed: u64) -> Tensor {
    let mut bits = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        let row = deterministic_row(seed.wrapping_add(r as u64 * 97 + 1), cols);
        bits.extend(row.iter().map(|&v| compute::f32_to_bf16(v)));
    }
    Tensor {
        name: name.to_string(),
        dtype: "BF16",
        shape: vec![rows as u64, cols as u64],
        bytes: u16_le(&bits),
    }
}

/// `router_raw` ships the router as [`bf16_matrix`] instead of pre-packed
/// [`int8_triple`], matching the real REAP-288 checkpoint
/// (`crates/repack/CLAUDE.md` Gotcha 6's router-transcode pattern applied on
/// the safetensors side, `orchestrate.rs`'s `quantize_router_int8`) rather
/// than every OTHER safetensors MoE checkpoint this walk has seen, which
/// ships it already `U32`-packed. No `bits_overrides` entry is inserted for
/// it either, matching the real checkpoint's `config.json`, which declares
/// none.
fn moe_tensors(
    p: &str,
    seed: u64,
    overrides: &mut HashMap<String, u32>,
    router_raw: bool,
) -> Vec<Tensor> {
    let mut ts = Vec::new();
    if router_raw {
        ts.push(bf16_matrix(
            &format!("{p}.mlp.gate.weight"),
            NUM_EXPERTS,
            HIDDEN,
            seed + 50,
        ));
    } else {
        ts.extend(int8_triple(
            &format!("{p}.mlp.gate.weight"),
            NUM_EXPERTS,
            HIDDEN,
            seed + 50,
        ));
        overrides.insert(format!("{p}.mlp.gate"), 8);
    }
    ts.extend(int8_triple(
        &format!("{p}.mlp.shared_expert_gate.weight"),
        1,
        HIDDEN,
        seed + 51,
    ));
    overrides.insert(format!("{p}.mlp.shared_expert_gate"), 8);
    for (i, role) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
        let (rows, cols) = if *role == "down_proj" {
            (HIDDEN, SHARED_INTER)
        } else {
            (SHARED_INTER, HIDDEN)
        };
        ts.extend(int8_triple(
            &format!("{p}.mlp.shared_expert.{role}.weight"),
            rows,
            cols,
            seed + 60 + i as u64,
        ));
        overrides.insert(format!("{p}.mlp.shared_expert.{role}"), 8);
        let (erows, ecols) = if *role == "down_proj" {
            (HIDDEN, MOE_INTER)
        } else {
            (MOE_INTER, HIDDEN)
        };
        ts.extend(expert_int4_triple(
            &format!("{p}.mlp.switch_mlp.{role}.weight"),
            NUM_EXPERTS,
            erows,
            ecols,
            seed + 70 + i as u64,
        ));
    }
    ts
}

/// One group's INT4-affine-at-[`NGRAM_GROUP_SIZE`] quantization, the
/// n-gram table's own layout -- distinct from
/// `compute::quantize_int4_affine`, which is fixed at group 64 and cannot
/// express this table's group 32. The inverse of
/// `compute::dequant_ngram_row`: same nibble order (low nibble first,
/// `byte_idx = elem / 2`, group-independent since every group here is a
/// whole number of bytes).
fn quantize_ngram_row(row: &[f32]) -> (Vec<u8>, Vec<u16>, Vec<u16>) {
    assert_eq!(row.len() % NGRAM_GROUP_SIZE, 0);
    let groups = row.len() / NGRAM_GROUP_SIZE;
    let mut packed = vec![0u8; row.len().div_ceil(2)];
    let mut scales = Vec::with_capacity(groups);
    let mut biases = Vec::with_capacity(groups);
    for g in 0..groups {
        let slice = &row[g * NGRAM_GROUP_SIZE..(g + 1) * NGRAM_GROUP_SIZE];
        let lo = slice.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let scale = ((hi - lo) / 15.0).max(1e-6);
        scales.push(compute::f32_to_bf16(scale));
        biases.push(compute::f32_to_bf16(lo));
        for (k, &v) in slice.iter().enumerate() {
            let q = (((v - lo) / scale).round().clamp(0.0, 15.0)) as u8;
            let elem = g * NGRAM_GROUP_SIZE + k;
            let byte_idx = elem / 2;
            if elem & 1 == 0 {
                packed[byte_idx] = (packed[byte_idx] & 0xF0) | q;
            } else {
                packed[byte_idx] = (packed[byte_idx] & 0x0F) | (q << 4);
            }
        }
    }
    (packed, scales, biases)
}

/// The n-gram table: ONE shard (the row/shard split is proven by
/// `synthetic_qwen::qwen4`'s own fixture; this one's job is REAL,
/// dequantizable data), with real quantized rows and prime-sized
/// per-head vocabularies small enough that the fixture's single shard can
/// address every row the hash reaches.
fn ngram_tensors(p: &str, vocab_size: i64) -> Vec<Tensor> {
    let head_vocab_sizes: Vec<i64> = (0..NGRAM_HEADS as i64)
        .map(|i| model_io::find_nth_prime_after(NGRAM_VOCAB_BASE - 1, i + 1))
        .collect();
    let mut running = 0i64;
    let head_offsets: Vec<i64> = head_vocab_sizes
        .iter()
        .map(|&s| {
            let o = running;
            running += s;
            o
        })
        .collect();
    let total_rows = running as usize;

    let mut weight = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    for r in 0..total_rows {
        let row = deterministic_row(9_000 + r as u64, NGRAM_HEAD_DIM);
        let (pk, sc, bi) = quantize_ngram_row(&row);
        weight.extend_from_slice(&pk);
        scales.extend_from_slice(&u16_le(&sc));
        biases.extend_from_slice(&u16_le(&bi));
    }
    let weight_bytes_per_row = NGRAM_HEAD_DIM / 2;
    let groups_per_row = NGRAM_HEAD_DIM / NGRAM_GROUP_SIZE;

    let prefix = format!("{p}.ple.ple_embedding");
    let mut ts = vec![
        Tensor {
            name: format!("{prefix}.ngram_embedding.shard_0.weight"),
            dtype: "U32",
            shape: vec![total_rows as u64, (weight_bytes_per_row / 4).max(1) as u64],
            bytes: weight,
        },
        Tensor {
            name: format!("{prefix}.ngram_embedding.shard_0.scales"),
            dtype: "BF16",
            shape: vec![total_rows as u64, groups_per_row as u64],
            bytes: scales,
        },
        Tensor {
            name: format!("{prefix}.ngram_embedding.shard_0.biases"),
            dtype: "BF16",
            shape: vec![total_rows as u64, groups_per_row as u64],
            bytes: biases,
        },
    ];

    // The DERIVED multipliers, matching the frozen-constant cross-check
    // (`crates/model-io/src/ngram_hash.rs`): using the real derivation
    // here rather than arbitrary constants is what lets a test compute the
    // expected hash independently and compare.
    let multipliers = model_io::build_layer_multipliers(vocab_size, 3, 0, 1234);
    ts.push(i64_tensor(
        &format!("{prefix}.layer_multipliers"),
        &multipliers,
    ));
    ts.push(i64_tensor(
        &format!("{prefix}.ngram_heads_offsets"),
        &head_offsets,
    ));
    ts.push(i64_tensor(
        &format!("{prefix}.ngram_heads_vocab_sizes"),
        &head_vocab_sizes,
    ));
    ts
}

fn i64_tensor(name: &str, values: &[i64]) -> Tensor {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    Tensor {
        name: name.to_string(),
        dtype: "I64",
        shape: vec![values.len() as u64],
        bytes,
    }
}

fn ple_layer_tensors(p: &str, vocab_size: i64) -> Vec<Tensor> {
    let wide = wide_dim();
    let mut ts = Vec::new();
    ts.extend(int4_triple(
        &format!("{p}.ple.key_proj.weight"),
        wide,
        HIDDEN,
        9_200,
    ));
    ts.extend(int4_triple(
        &format!("{p}.ple.value_proj.weight"),
        HIDDEN,
        HIDDEN,
        9_201,
    ));
    ts.push(conv1d_weight(
        &format!("{p}.ple.conv1d.weight"),
        wide,
        4,
        9_202,
    ));
    for (i, norm) in ["norm_key", "norm_query", "norm_conv"].iter().enumerate() {
        ts.push(bf16_vector(
            &format!("{p}.ple.{norm}.weight"),
            wide,
            0.0,
            9_210 + i as u64,
        ));
    }
    ts.extend(ngram_tensors(p, vocab_size));
    ts
}

fn build_tensors(vocab_size: i64, router_raw: bool) -> (Vec<Tensor>, HashMap<String, u32>) {
    let vocab = vocab_size as usize;
    let mut overrides = HashMap::new();
    let mut ts = Vec::new();
    ts.extend(int4_triple(
        "language_model.model.embed_tokens.weight",
        vocab,
        HIDDEN,
        1001,
    ));
    ts.extend(int4_triple(
        "language_model.lm_head.weight",
        vocab,
        HIDDEN,
        1002,
    ));

    let mask = layer_mask();
    for (l, &m) in mask.iter().enumerate() {
        let p = format!("language_model.model.layers.{l}");
        let seed = 1000 * (l as u64 + 1);
        ts.extend(hyper_connection_tensors(
            &format!("{p}.attn_hyper_connection"),
            seed + 100,
        ));
        ts.extend(hyper_connection_tensors(
            &format!("{p}.mlp_hyper_connection"),
            seed + 200,
        ));
        if m == 1 {
            ts.extend(attention_layer_tensors(&p, seed));
        } else {
            ts.extend(gdn_layer_tensors(&p, seed));
        }
        ts.extend(moe_tensors(&p, seed, &mut overrides, router_raw));
        if l == PLE_LAYER {
            ts.extend(ple_layer_tensors(&p, vocab_size));
        }
    }
    ts.extend(final_mixer_tensors());
    (ts, overrides)
}

/// Writes a decode-capable `qwen4_exp` install through the real repack
/// pipeline and returns the `ArchConfig` needed to open it.
pub fn build_synthetic_qwen4_exp_decode_install(
    dir: &std::path::Path,
    vocab_size: i64,
    model_id: &str,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_synthetic_qwen4_exp_decode_install_inner(dir, vocab_size, model_id, false)
}

/// [`build_synthetic_qwen4_exp_decode_install`], with the router shipped raw
/// (unquantized BF16) rather than pre-packed INT8, matching the real
/// REAP-288 checkpoint that exposed the router dtype bug
/// (`crates/runtime/src/families/qwen4/moe.rs`'s dtype-5 refusal). This is
/// the fixture that actually exercises `orchestrate.rs`'s
/// `quantize_router_int8`: the default fixture above ships the router
/// already `U32`-packed and takes the pre-existing `pass_through_packed`
/// branch, so it would pass identically whether or not that fix exists.
pub fn build_synthetic_qwen4_exp_decode_install_raw_router(
    dir: &std::path::Path,
    vocab_size: i64,
    model_id: &str,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_synthetic_qwen4_exp_decode_install_inner(dir, vocab_size, model_id, true)
}

fn build_synthetic_qwen4_exp_decode_install_inner(
    dir: &std::path::Path,
    vocab_size: i64,
    model_id: &str,
    router_raw: bool,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    let arch = tiny_qwen4_exp_decode_arch(vocab_size);
    let (ts, bits_overrides) = build_tensors(vocab_size, router_raw);
    let blob = assemble_safetensors(&ts);
    let source = MemoryRangeSource::new(&blob);
    let header = parse_header(&blob, crate::safetensors_header::DEFAULT_MAX_HEADER_BYTES)?;
    let quant = Gemma4Quant {
        default_bits: 4,
        group_size: 64,
        bits_overrides,
    };
    let _ = Gemma4Shards::single(&header, &source);
    write_gemma4_install(dir, &arch, model_id, &header, &source, &quant)?;
    Ok(arch)
}
