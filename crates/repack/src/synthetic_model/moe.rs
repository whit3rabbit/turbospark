//! MoE synthetic Gemma 4 install builders (resident and streamed).

use model_io::ArchConfig;

use super::arch::{
    embed_lm_head_name, expert_down_proj_name, expert_gate_proj_name, expert_up_proj_name,
    k_proj_name, o_proj_name, q_proj_name, quantized_tensor, router_name, tiny_gemma4_arch,
    FULL_HEAD_DIM, HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_HEADS,
};
use crate::gturbo_writer::{write_gturbo_install_with_resident_index, WriterError};
use crate::resident_writer::build_resident_weights_bin;

/// Like [`build_synthetic_gemma4_moe_install`], with IDENTICAL weights
/// (same deterministic seeds), but the routed experts live in
/// `packed_experts/layer_NN.bin` blob files instead of the resident
/// region — the streamed layout `turbospark-streaming`'s
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
            // The companions are per-GROUP (64 wide, the affine group the
            // quantizer above uses), so their logical shape is [rows, groups]
            // and not the weight's [rows, cols] -- the layout validator sizes
            // every bf16 companion off its declared shape, and a [rows, cols]
            // declaration on a 64-row, 64-col weight demands 8192 bytes where
            // one group of 64 rows is 128.
            let groups = |cols: usize| cols.div_ceil(64) as u64;
            let mut sub_tensors = Vec::with_capacity(9);
            let mut used = 0u64;
            for (role, spec) in [("gate", &gate), ("up", &up), ("down", &down)] {
                for (suffix, bytes, dtype, shape) in [
                    (
                        "",
                        spec.packed.clone(),
                        "u32",
                        vec![spec.rows as u64, spec.cols as u64],
                    ),
                    (
                        "_scales",
                        u16_le(&spec.scales),
                        "bf16",
                        vec![spec.rows as u64, groups(spec.cols as usize)],
                    ),
                    (
                        "_biases",
                        u16_le(&spec.biases),
                        "bf16",
                        vec![spec.rows as u64, groups(spec.cols as usize)],
                    ),
                ] {
                    used += bytes.len() as u64;
                    sub_tensors.push(crate::gturbo_writer::SubTensor {
                        role: format!("{role}{suffix}"),
                        bytes,
                        dtype: dtype.to_string(),
                        shape,
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

/// Writes a tiny Gemma4-shaped `.gturbo` install with routed-expert FFN
/// layers instead of a dense FFN: same attention shape as
/// [`super::dense::build_synthetic_gemma4_install`] (dense q/k/o projections, no
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
