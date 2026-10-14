//! A tiny Qwen 3.6 MoE install laid out the way
//! `mlx-community/MiniCPM-V-4.7-35B-A3B-mxfp4` is, through the REAL repack
//! pipeline:
//!
//! - MLX mxfp4 (u32 codes plus u8 E8M0 scales, no biases) on the routed
//!   experts, the shared expert and the full-attention projections;
//! - affine INT8 at group 64 on the router and shared-expert gate, as
//!   per-tensor overrides under the 32-element mxfp4 default;
//! - BF16 unquantized embed, head and GDN projections;
//! - decoy `vision_tower.`, `vit_merger.` and `merger.` tensors, which the
//!   walk must drop rather than ingest or refuse.
//!
//! The shapes are [`tiny_qwen_gdn_moe_arch`]'s, so a runtime test can hold
//! this install beside the affine one.

use crate::gemma4_checkpoint::{write_qwen_gdn_moe_install, Gemma4Quant, QuantMode};
use crate::ranged_download::MemoryRangeSource;
use crate::safetensors_header::parse_header;
use crate::synthetic_tensors::{
    assemble_safetensors, bf16_vector, deterministic_row, int8_triple, Tensor,
};
use model_io::ArchConfig;

use super::moe::tiny_qwen_gdn_moe_arch;

/// The MLX planes of one `rows x cols` mxfp4 matrix (`cols % 32 == 0`).
/// Exponents sit at 2^-5..2^-3 so decoded weights stay under 0.75, the
/// magnitude the affine fixtures use.
fn mxfp4_planes(rows: usize, cols: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
    let noise = deterministic_row(seed, rows * cols);
    let codes: Vec<u8> = noise
        .iter()
        .map(|v| (((v + 1.0) * 8.0) as u8).min(15))
        .collect();
    let mut weight = Vec::with_capacity(rows * cols / 2);
    for pair in codes.chunks_exact(2) {
        weight.push(pair[0] | (pair[1] << 4));
    }
    let scales = deterministic_row(seed ^ 0x5CA1E, rows * cols / 32)
        .iter()
        .map(|v| 122 + (((v + 1.0) * 1.5) as u8).min(2))
        .collect();
    (weight, scales)
}

/// A rank-2 MLX mxfp4 pair: `U32 [rows, cols/8]` plus `U8 [rows, cols/32]`.
fn mxfp4_pair(name: &str, rows: usize, cols: usize, seed: u64) -> Vec<Tensor> {
    let base = name.strip_suffix(".weight").unwrap_or(name);
    let (weight, scales) = mxfp4_planes(rows, cols, seed);
    vec![
        Tensor {
            name: name.to_string(),
            dtype: "U32",
            shape: vec![rows as u64, (cols / 8) as u64],
            bytes: weight,
        },
        Tensor {
            name: format!("{base}.scales"),
            dtype: "U8",
            shape: vec![rows as u64, (cols / 32) as u64],
            bytes: scales,
        },
    ]
}

/// The rank-3 routed-expert form of [`mxfp4_pair`], expert axis first.
fn expert_mxfp4_pair(
    name: &str,
    experts: usize,
    rows: usize,
    cols: usize,
    seed: u64,
) -> Vec<Tensor> {
    let base = name.strip_suffix(".weight").unwrap_or(name);
    let (weight, scales) = mxfp4_planes(experts * rows, cols, seed);
    vec![
        Tensor {
            name: name.to_string(),
            dtype: "U32",
            shape: vec![experts as u64, rows as u64, (cols / 8) as u64],
            bytes: weight,
        },
        Tensor {
            name: format!("{base}.scales"),
            dtype: "U8",
            shape: vec![experts as u64, rows as u64, (cols / 32) as u64],
            bytes: scales,
        },
    ]
}

/// An unquantized BF16 matrix with values in about +-0.05.
fn bf16_matrix(name: &str, rows: usize, cols: usize, seed: u64) -> Tensor {
    let flat = bf16_vector(name, rows * cols, 0.0, seed);
    Tensor {
        name: flat.name,
        dtype: flat.dtype,
        shape: vec![rows as u64, cols as u64],
        bytes: flat.bytes,
    }
}

/// Builds the mxfp4 fixture and writes a `.gturbo` install to `dir`.
pub fn build_synthetic_qwen_gdn_moe_mxfp4_install(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    num_experts: i64,
    model_id: &str,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    let arch = tiny_qwen_gdn_moe_arch(vocab_size, num_layers, num_experts);
    let hidden = arch.hidden_size as usize;
    let inter = arch.moe_intermediate_size as usize;
    let heads = arch.num_heads as usize;
    let kv_heads = arch.num_kv_heads as usize;
    let head_dim = arch.head_dim as usize;
    let experts = num_experts as usize;
    let vocab = vocab_size as usize;
    let la = &arch.linear_attention;
    let qkv_dim = la.qkv_dim() as usize;
    let value_dim = la.value_dim() as usize;
    let v_heads = la.num_v_heads as usize;
    let conv_k = la.conv_kernel_size as usize;

    let mut ts: Vec<Tensor> = vec![
        bf16_matrix("language_model.model.embed_tokens.weight", vocab, hidden, 1),
        bf16_matrix("language_model.lm_head.weight", vocab, hidden, 2),
    ];
    let mut quant = Gemma4Quant {
        default_bits: 4,
        group_size: 32,
        mode: QuantMode::Mxfp4,
        ..Default::default()
    };
    for l in 0..num_layers as usize {
        let p = format!("language_model.model.layers.{l}");
        let seed = 1000 * (l as u64 + 1);
        for (i, norm) in ["input_layernorm", "post_attention_layernorm"]
            .iter()
            .enumerate()
        {
            ts.push(bf16_vector(
                &format!("{p}.{norm}.weight"),
                hidden,
                1.0,
                seed + 40 + i as u64,
            ));
        }
        if arch.layer_is_full(l) {
            // attn_output_gate: per-head [query; gate], so twice the rows.
            for (role, rows, cols, s) in [
                ("q_proj", 2 * heads * head_dim, hidden, 1u64),
                ("k_proj", kv_heads * head_dim, hidden, 2),
                ("v_proj", kv_heads * head_dim, hidden, 3),
                ("o_proj", hidden, heads * head_dim, 4),
            ] {
                ts.extend(mxfp4_pair(
                    &format!("{p}.self_attn.{role}.weight"),
                    rows,
                    cols,
                    seed + s,
                ));
            }
            for (i, norm) in ["q_norm", "k_norm"].iter().enumerate() {
                ts.push(bf16_vector(
                    &format!("{p}.self_attn.{norm}.weight"),
                    head_dim,
                    1.0,
                    seed + 20 + i as u64,
                ));
            }
        } else {
            for (name, rows, cols, s) in [
                ("in_proj_qkv", qkv_dim, hidden, 1u64),
                ("in_proj_z", value_dim, hidden, 2),
                ("in_proj_a", v_heads, hidden, 3),
                ("in_proj_b", v_heads, hidden, 4),
                ("out_proj", hidden, value_dim, 5),
            ] {
                ts.push(bf16_matrix(
                    &format!("{p}.linear_attn.{name}.weight"),
                    rows,
                    cols,
                    seed + s,
                ));
            }
            let conv = bf16_vector(
                &format!("{p}.linear_attn.conv1d.weight"),
                qkv_dim * conv_k,
                0.0,
                seed + 10,
            );
            ts.push(Tensor {
                shape: vec![qkv_dim as u64, conv_k as u64, 1],
                ..conv
            });
            for (name, center, s) in [("A_log", 0.0, 11u64), ("dt_bias", 0.0, 12)] {
                ts.push(bf16_vector(
                    &format!("{p}.linear_attn.{name}"),
                    v_heads,
                    center,
                    seed + s,
                ));
            }
            ts.push(bf16_vector(
                &format!("{p}.linear_attn.norm.weight"),
                la.value_head_dim as usize,
                1.0,
                seed + 13,
            ));
        }
        for (name, rows, s) in [("gate", experts, 50u64), ("shared_expert_gate", 1, 51)] {
            ts.extend(int8_triple(
                &format!("{p}.mlp.{name}.weight"),
                rows,
                hidden,
                seed + s,
            ));
            quant.bits_overrides.insert(format!("{p}.mlp.{name}"), 8);
            quant.group_overrides.insert(format!("{p}.mlp.{name}"), 64);
        }
        for (i, role) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
            let (rows, cols) = if *role == "down_proj" {
                (hidden, inter)
            } else {
                (inter, hidden)
            };
            ts.extend(mxfp4_pair(
                &format!("{p}.mlp.shared_expert.{role}.weight"),
                rows,
                cols,
                seed + 60 + i as u64,
            ));
            ts.extend(expert_mxfp4_pair(
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
        hidden,
        1.0,
        7,
    ));
    // MiniCPM's three non-text prefixes, one tensor each.
    for name in [
        "vision_tower.encoder.layers.0.self_attn.q_proj.weight",
        "vit_merger.0.mlp.0.weight",
        "merger.mlp.0.weight",
    ] {
        ts.push(bf16_matrix(name, 4, 32, 99));
    }

    let blob = assemble_safetensors(&ts);
    let source = MemoryRangeSource::new(&blob);
    let header = parse_header(&blob, crate::safetensors_header::DEFAULT_MAX_HEADER_BYTES)?;
    write_qwen_gdn_moe_install(dir, &arch, model_id, &header, &source, &quant)?;
    Ok(arch)
}
