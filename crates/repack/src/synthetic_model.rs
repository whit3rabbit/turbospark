//! Builds a small, fully in-memory-generated "tiny Gemma 4" `.gturbo`
//! install: real INT4-affine-quantized weights (deterministic, not
//! trained), written through the real resident-tensor writer and readable
//! back through every `mrefrust_model_io` loader. This is what
//! `crates/runtime`'s `RealForwardRunner` (macOS/GPU only) drives for its
//! end-to-end real-forward-pass test, since no trained `.gturbo` checkpoint
//! is available in this environment.
//!
//! Every non-shape architecture field is set to Gemma 4's own real
//! canonical baseline value (`mrefrust_model_io::gemma4_26b_a4b`): this is
//! honestly a tiny Gemma-4-architecture model, not an invented one.

use compute::quantize_int4_affine;
use model_io::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig,
    ModelFamily,
};

use crate::gturbo_writer::{write_gturbo_install_with_resident_index, WriterError};
use crate::resident_writer::{build_resident_weights_bin, ResidentTensorSpec};

/// Hidden size, per-head dim, and dense FFN width are all fixed at 64: the
/// smallest value satisfying `mrefrust_compute::quant::GROUP_SIZE`'s
/// multiple-of-64 requirement on every GEMV's contraction dimension.
const HIDDEN_SIZE: i64 = 64;
const NUM_HEADS: i64 = 2;
const FULL_HEAD_DIM: i64 = 32; // NUM_HEADS * FULL_HEAD_DIM == HIDDEN_SIZE
const INTERMEDIATE_SIZE: i64 = 64;

/// A tiny, dense (no MoE, no sliding-window/linear/compressed layers)
/// Gemma-4-shaped architecture. `vocab_size` and `num_layers` are the only
/// caller-chosen dimensions; everything else matches
/// `mrefrust_model_io::gemma4_26b_a4b()`'s non-shape fields exactly, since
/// `manifest.json`'s optional family-extension fields fall back to the
/// Gemma 4 baseline's values when omitted (see `arch_validation.rs`).
pub fn tiny_gemma4_arch(vocab_size: i64, num_layers: i64) -> ArchConfig {
    ArchConfig {
        hidden_size: HIDDEN_SIZE,
        intermediate_size: INTERMEDIATE_SIZE,
        moe_intermediate_size: 0,
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_HEADS,
        num_full_kv_heads: NUM_HEADS,
        head_dim: FULL_HEAD_DIM,
        full_head_dim: FULL_HEAD_DIM,
        vocab_size,
        sliding_window: 0,
        final_logit_softcap: 30.0,
        rope_theta: 10_000.0,
        full_rope_theta: 10_000.0,
        partial_rotary_factor: 1.0,
        num_layers,
        num_experts: 0,
        top_k_experts: 0,
        tie_word_embeddings: true,
        attention_k_eq_v: true,
        full_attention_layer_mask: vec![1u8; num_layers as usize],
        hidden_activation: "gelu_pytorch_tanh".to_string(),
        family: ModelFamily::Gemma4,
        attn_output_gate: false,
        attention_scale: 1.0,
        embedding_scaled_by_sqrt_hidden: true,
        router_scaled: true,
        ffn_sandwich_norms: true,
        shared_expert_gated: false,
        rope_neox_subdim: false,
        linear_attention: LinearAttentionConfig::NONE,
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
    }
}

/// A cheap deterministic xorshift stream, seeded per row so every
/// generated tensor is reproducible without relying on any RNG crate.
fn deterministic_row(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(0x9E37_79B9);
    (0..n)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state = state.wrapping_add(i as u64);
            ((state % 2000) as f32 / 1000.0) - 1.0
        })
        .collect()
}

fn quantized_tensor(name: &str, rows: usize, cols: usize, seed: u64) -> ResidentTensorSpec {
    let mut packed = Vec::with_capacity(rows * cols / 2);
    let mut scales = Vec::with_capacity(rows * cols / 64);
    let mut biases = Vec::with_capacity(rows * cols / 64);
    for r in 0..rows {
        let row = deterministic_row(seed.wrapping_add(r as u64 * 97 + 1), cols);
        let q = quantize_int4_affine(&row);
        packed.extend_from_slice(&q.packed);
        scales.extend_from_slice(&q.scales);
        biases.extend_from_slice(&q.biases);
    }
    ResidentTensorSpec {
        name: name.to_string(),
        packed,
        scales,
        biases,
        rows: rows as u32,
        cols: cols as u32,
    }
}

/// Resident tensor names `RealForwardRunner` looks up by. Kept in sync
/// with `build_synthetic_gemma4_install`'s writes.
pub fn embed_lm_head_name() -> String {
    "embed_lm_head".to_string()
}
pub fn q_proj_name(layer: i64) -> String {
    format!("layer{layer}.q_proj")
}
pub fn k_proj_name(layer: i64) -> String {
    format!("layer{layer}.k_proj")
}
pub fn o_proj_name(layer: i64) -> String {
    format!("layer{layer}.o_proj")
}
pub fn gate_proj_name(layer: i64) -> String {
    format!("layer{layer}.gate_proj")
}
pub fn up_proj_name(layer: i64) -> String {
    format!("layer{layer}.up_proj")
}
pub fn down_proj_name(layer: i64) -> String {
    format!("layer{layer}.down_proj")
}

/// Writes a full tiny-Gemma4 `.gturbo` install to `dir` and returns the
/// `ArchConfig` it was built against (the caller needs this to open the
/// install back up, since `mrefrust_model_io::load_manifest` validates
/// against a caller-supplied expected architecture rather than inferring
/// one for non-canonical shapes).
pub fn build_synthetic_gemma4_install(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
) -> Result<ArchConfig, WriterError> {
    let arch = tiny_gemma4_arch(vocab_size, num_layers);
    let hidden = HIDDEN_SIZE as usize;
    let inter = INTERMEDIATE_SIZE as usize;
    let qk_dim = (NUM_HEADS * FULL_HEAD_DIM) as usize;
    let vocab = vocab_size as usize;

    let mut specs = Vec::with_capacity(1 + num_layers as usize * 6);
    specs.push(quantized_tensor(&embed_lm_head_name(), vocab, hidden, 1));
    for l in 0..num_layers {
        let base = 1000u64 * (l as u64 + 1);
        specs.push(quantized_tensor(&q_proj_name(l), qk_dim, hidden, base + 1));
        specs.push(quantized_tensor(&k_proj_name(l), qk_dim, hidden, base + 2));
        specs.push(quantized_tensor(&o_proj_name(l), hidden, qk_dim, base + 3));
        specs.push(quantized_tensor(
            &gate_proj_name(l),
            inter,
            hidden,
            base + 4,
        ));
        specs.push(quantized_tensor(&up_proj_name(l), inter, hidden, base + 5));
        specs.push(quantized_tensor(
            &down_proj_name(l),
            hidden,
            inter,
            base + 6,
        ));
    }

    let resident_bytes = build_resident_weights_bin(&specs);
    write_gturbo_install_with_resident_index(dir, &arch, model_id, &resident_bytes)?;
    Ok(arch)
}

/// A dense tiny-Gemma4 install whose layers ALTERNATE sliding-window
/// (mask 0) and full attention (mask 1), with `sliding_window` positions
/// of window — the mixed attention-kind shape real Gemma 4 has (25 SWA +
/// 5 full layers), at toy size. Weights are identical to
/// [`build_synthetic_gemma4_install`] (same seeds).
pub fn build_synthetic_gemma4_swa_install(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    sliding_window: i64,
    model_id: &str,
) -> Result<ArchConfig, WriterError> {
    let mut arch = tiny_gemma4_arch(vocab_size, num_layers);
    arch.sliding_window = sliding_window;
    arch.full_attention_layer_mask = (0..num_layers).map(|l| (l % 2 == 1) as u8).collect();

    let hidden = HIDDEN_SIZE as usize;
    let inter = INTERMEDIATE_SIZE as usize;
    let qk_dim = (NUM_HEADS * FULL_HEAD_DIM) as usize;
    let vocab = vocab_size as usize;

    let mut specs = Vec::with_capacity(1 + num_layers as usize * 6);
    specs.push(quantized_tensor(&embed_lm_head_name(), vocab, hidden, 1));
    for l in 0..num_layers {
        let base = 1000u64 * (l as u64 + 1);
        specs.push(quantized_tensor(&q_proj_name(l), qk_dim, hidden, base + 1));
        specs.push(quantized_tensor(&k_proj_name(l), qk_dim, hidden, base + 2));
        specs.push(quantized_tensor(&o_proj_name(l), hidden, qk_dim, base + 3));
        specs.push(quantized_tensor(
            &gate_proj_name(l),
            inter,
            hidden,
            base + 4,
        ));
        specs.push(quantized_tensor(&up_proj_name(l), inter, hidden, base + 5));
        specs.push(quantized_tensor(
            &down_proj_name(l),
            hidden,
            inter,
            base + 6,
        ));
    }

    let resident_bytes = build_resident_weights_bin(&specs);
    write_gturbo_install_with_resident_index(dir, &arch, model_id, &resident_bytes)?;
    Ok(arch)
}

/// Like [`build_synthetic_gemma4_moe_install`], with IDENTICAL weights
/// (same deterministic seeds), but the routed experts live in
/// `packed_experts/layer_NN.bin` blob files instead of the resident
/// region — the streamed layout `mrefrust-streaming`'s
/// `PreadExpertStreamer` reads at decode time. A runner over this install
/// must produce exactly the tokens the resident-expert variant produces.
pub fn build_synthetic_gemma4_moe_streamed_install(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    num_experts: i64,
    top_k: i64,
    model_id: &str,
) -> Result<ArchConfig, WriterError> {
    let mut arch = tiny_gemma4_arch(vocab_size, num_layers);
    arch.num_experts = num_experts;
    arch.top_k_experts = top_k;
    arch.moe_intermediate_size = INTERMEDIATE_SIZE;

    let hidden = HIDDEN_SIZE as usize;
    let inter = INTERMEDIATE_SIZE as usize;
    let qk_dim = (NUM_HEADS * FULL_HEAD_DIM) as usize;
    let vocab = vocab_size as usize;

    let mut specs = Vec::with_capacity(1 + num_layers as usize * 4);
    specs.push(quantized_tensor(&embed_lm_head_name(), vocab, hidden, 1));
    for l in 0..num_layers {
        let base = 1000u64 * (l as u64 + 1);
        specs.push(quantized_tensor(&q_proj_name(l), qk_dim, hidden, base + 1));
        specs.push(quantized_tensor(&k_proj_name(l), qk_dim, hidden, base + 2));
        specs.push(quantized_tensor(&o_proj_name(l), hidden, qk_dim, base + 3));
        specs.push(quantized_tensor(
            &router_name(l),
            num_experts as usize,
            hidden,
            base + 4,
        ));
    }
    let resident_bytes = build_resident_weights_bin(&specs);

    // Expert blobs: the same nine sub-tensors per expert the real format
    // packs ({gate,up,down} x {weights,scales,biases}), with the same
    // deterministic seeds the resident-expert builder uses.
    let u16_le = |v: &[u16]| -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 2);
        for x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    };
    let mut layers = Vec::with_capacity(num_layers as usize);
    let mut max_blob = 0u64;
    for l in 0..num_layers {
        let base = 1000u64 * (l as u64 + 1);
        let mut experts = Vec::with_capacity(num_experts as usize);
        for e in 0..num_experts {
            let ebase = base + 100 + 10 * e as u64;
            let gate = quantized_tensor("gate", inter, hidden, ebase + 1);
            let up = quantized_tensor("up", inter, hidden, ebase + 2);
            let down = quantized_tensor("down", hidden, inter, ebase + 3);
            let mut sub_tensors = Vec::with_capacity(9);
            let mut used = 0u64;
            for (role, spec) in [("gate", &gate), ("up", &up), ("down", &down)] {
                for (suffix, bytes, dtype) in [
                    ("", spec.packed.clone(), "u32"),
                    ("_scales", u16_le(&spec.scales), "bf16"),
                    ("_biases", u16_le(&spec.biases), "bf16"),
                ] {
                    used += bytes.len() as u64;
                    sub_tensors.push(crate::gturbo_writer::SubTensor {
                        role: format!("{role}{suffix}"),
                        bytes,
                        dtype: dtype.to_string(),
                        shape: vec![spec.rows as u64, spec.cols as u64],
                    });
                }
            }
            max_blob = max_blob.max(used);
            experts.push(crate::gturbo_writer::ExpertBlob {
                expert: e as usize,
                sub_tensors,
            });
        }
        layers.push(crate::gturbo_writer::LayerBlobs {
            layer: l as usize,
            experts,
        });
    }
    // One page-rounded stride for the whole model (16 KiB pages), matching
    // the Swift repacker's roundUpToPage(expertStride).
    let expert_stride = max_blob.div_ceil(16_384) * 16_384;

    crate::gturbo_writer::write_gturbo_install_with_resident_index_and_experts(
        dir,
        &arch,
        model_id,
        &resident_bytes,
        expert_stride,
        num_experts as usize,
        &layers,
    )?;
    Ok(arch)
}

pub fn router_name(layer: i64) -> String {
    format!("layer{layer}.router")
}
pub fn expert_gate_proj_name(layer: i64, expert: i64) -> String {
    format!("layer{layer}.expert{expert}.gate_proj")
}
pub fn expert_up_proj_name(layer: i64, expert: i64) -> String {
    format!("layer{layer}.expert{expert}.up_proj")
}
pub fn expert_down_proj_name(layer: i64, expert: i64) -> String {
    format!("layer{layer}.expert{expert}.down_proj")
}

/// Writes a tiny Gemma4-shaped `.gturbo` install with routed-expert FFN
/// layers instead of a dense FFN: same attention shape as
/// [`build_synthetic_gemma4_install`] (dense q/k/o projections, no
/// separate V weights since `attention_k_eq_v`), but each layer's FFN is
/// `num_experts` independently-weighted experts of width
/// [`INTERMEDIATE_SIZE`] selected `top_k` at a time by a router GEMV — no
/// dense/shared FFN branch is written (`crates/runtime`'s
/// `RealForwardRunner` runs routed-only when `num_experts > 0`; real
/// Gemma 4 sums a dense branch alongside the routed one, which this port
/// does not do — see `DEVIATIONS.md`).
pub fn build_synthetic_gemma4_moe_install(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    num_experts: i64,
    top_k: i64,
    model_id: &str,
) -> Result<ArchConfig, WriterError> {
    let mut arch = tiny_gemma4_arch(vocab_size, num_layers);
    arch.num_experts = num_experts;
    arch.top_k_experts = top_k;
    arch.moe_intermediate_size = INTERMEDIATE_SIZE;

    let hidden = HIDDEN_SIZE as usize;
    let inter = INTERMEDIATE_SIZE as usize;
    let qk_dim = (NUM_HEADS * FULL_HEAD_DIM) as usize;
    let vocab = vocab_size as usize;
    let experts = num_experts as usize;

    let mut specs = Vec::with_capacity(1 + num_layers as usize * (4 + experts * 3));
    specs.push(quantized_tensor(&embed_lm_head_name(), vocab, hidden, 1));
    for l in 0..num_layers {
        let base = 1000u64 * (l as u64 + 1);
        specs.push(quantized_tensor(&q_proj_name(l), qk_dim, hidden, base + 1));
        specs.push(quantized_tensor(&k_proj_name(l), qk_dim, hidden, base + 2));
        specs.push(quantized_tensor(&o_proj_name(l), hidden, qk_dim, base + 3));
        specs.push(quantized_tensor(&router_name(l), experts, hidden, base + 4));
        for e in 0..num_experts {
            let ebase = base + 100 + 10 * e as u64;
            specs.push(quantized_tensor(
                &expert_gate_proj_name(l, e),
                inter,
                hidden,
                ebase + 1,
            ));
            specs.push(quantized_tensor(
                &expert_up_proj_name(l, e),
                inter,
                hidden,
                ebase + 2,
            ));
            specs.push(quantized_tensor(
                &expert_down_proj_name(l, e),
                hidden,
                inter,
                ebase + 3,
            ));
        }
    }

    let resident_bytes = build_resident_weights_bin(&specs);
    write_gturbo_install_with_resident_index(dir, &arch, model_id, &resident_bytes)?;
    Ok(arch)
}
