//! A tiny Fish Audio Drama 3 checkpoint, and an install built from it through
//! the real streamed Drama intake (`crate::fish_drama3`).
//!
//! Starts from the Qwen 3.6 MoE fixture's tensors (`super::moe::trunk_tensors`)
//! and re-encodes the ones the real checkpoint stores differently, so the
//! intake meets the storage it will meet on the real download
//! (`docs/FISH_DRAMA3_PHASE0.md`, tensor inventory):
//!
//! - trunk names under `backbone.model.`, no `lm_head`;
//! - embedding, router, shared-expert gate and `in_proj_a/b` raw BF16;
//! - attention and linear-attention projections 8-bit, routed experts 4-bit
//!   (the qwen flow sends a GDN layer whose `in_proj_qkv` is INT4 down a
//!   fused path that needs all four input projections INT4, which the real
//!   checkpoint's BF16 `in_proj_a/b` are not);
//! - layer, q/k and final norms F32 and shifted to `1 + w`;
//! - a handful of `tower.*` and `semantic_head.*` side tensors;
//! - a quantization block whose GLOBAL default is 8 bits, with every 4-bit
//!   tensor listed per path under its `backbone.model.` name. That is what
//!   makes the intake's quantization-key rename observable: without it every
//!   4-bit tensor would be read as 8-bit.
//!
//! Four layers, three linear then one full, the 3:1 pattern the intake
//! requires.

use model_io::ArchConfig;

use crate::gemma4_checkpoint::Gemma4Shards;
use crate::ranged_download::MemoryRangeSource;
use crate::safetensors_header::parse_header;
use crate::synthetic_tensors::{
    assemble_safetensors, deterministic_row, int8_triple, u16_le, Tensor,
};

use super::moe::{tiny_qwen_gdn_moe_arch, trunk_tensors};

/// Layers in the fixture: three linear-attention layers, then one full.
pub const DRAMA3_FIXTURE_LAYERS: i64 = 4;

/// A synthetic Drama 3 checkpoint: one safetensors blob and its `config.json`.
pub struct SyntheticDrama3 {
    pub blob: Vec<u8>,
    pub config_text: String,
}

/// How the fixture stores its trunk norms.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Drama3NormStorage {
    /// `1 + w`, as the real conversion stores them.
    Shifted,
    /// `w` alone: the export the intake must refuse.
    Unshifted,
}

fn raw_bf16(name: &str, rows: usize, cols: usize, seed: u64) -> Tensor {
    let bits: Vec<u16> = deterministic_row(seed, rows * cols)
        .iter()
        .map(|v| compute::f32_to_bf16(v * 0.05))
        .collect();
    Tensor {
        name: name.to_string(),
        dtype: "BF16",
        shape: if rows == 1 {
            vec![1, cols as u64]
        } else {
            vec![rows as u64, cols as u64]
        },
        bytes: u16_le(&bits),
    }
}

fn f32_norm(name: &str, n: usize, center: f32, seed: u64) -> Tensor {
    let bytes = deterministic_row(seed, n)
        .iter()
        .flat_map(|j| (center + j * 0.05).to_le_bytes())
        .collect();
    Tensor {
        name: name.to_string(),
        dtype: "F32",
        shape: vec![n as u64],
        bytes,
    }
}

fn is_norm(name: &str) -> bool {
    name.ends_with(".input_layernorm.weight")
        || name.ends_with(".post_attention_layernorm.weight")
        || name.ends_with(".self_attn.q_norm.weight")
        || name.ends_with(".self_attn.k_norm.weight")
        || name == "language_model.model.norm.weight"
}

/// The `text_config` a Drama 3 conversion would declare for `arch`.
fn text_config(arch: &ArchConfig) -> serde_json::Value {
    let la = &arch.linear_attention;
    let layer_types: Vec<&str> = (0..arch.num_layers as usize)
        .map(|l| {
            if arch.layer_is_full(l) {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    serde_json::json!({
        "model_type": "fish_omni_text",
        "attn_output_gate": true,
        "hidden_act": "silu",
        "hidden_size": arch.hidden_size,
        "head_dim": arch.head_dim,
        "num_attention_heads": arch.num_heads,
        "num_key_value_heads": arch.num_kv_heads,
        "num_hidden_layers": arch.num_layers,
        "layer_types": layer_types,
        "mlp_layer_types": vec!["moe"; arch.num_layers as usize],
        "layernorm_zero_centered_gamma": true,
        "linear_conv_kernel_dim": la.conv_kernel_size,
        "linear_key_head_dim": la.key_head_dim,
        "linear_num_key_heads": la.num_k_heads,
        "linear_num_value_heads": la.num_v_heads,
        "linear_value_head_dim": la.value_head_dim,
        "max_position_embeddings": 4096,
        "moe_intermediate_size": arch.moe_intermediate_size,
        "num_experts": arch.num_experts,
        "num_experts_per_tok": arch.top_k_experts,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "partial_rotary_factor": arch.partial_rotary_factor,
            "rope_theta": arch.rope_theta,
            "rope_type": "default",
        },
        "rotary_interleaved": false,
        "shared_expert_gate": true,
        "shared_expert_intermediate_size": arch.intermediate_size,
        "tie_word_embeddings": false,
        "use_qk_norm": true,
        "vocab_size": arch.vocab_size,
    })
}

/// Builds the fixture checkpoint (see the module doc).
pub fn synthetic_fish_drama3_checkpoint(
    vocab_size: i64,
    num_experts: i64,
    norms: Drama3NormStorage,
) -> SyntheticDrama3 {
    let mut arch = tiny_qwen_gdn_moe_arch(vocab_size, DRAMA3_FIXTURE_LAYERS, num_experts);
    arch.full_attention_layer_mask = vec![2, 2, 2, 1];
    let hidden = arch.hidden_size as usize;
    let experts = num_experts as usize;
    let vocab = vocab_size as usize;
    let v_heads = arch.linear_attention.num_v_heads as usize;
    let (source, overrides) = trunk_tensors(&arch, false);

    // Every U32 tensor's base path at its stated bits. The global default is
    // 8, so the 4-bit ones each need an entry, as on the real checkpoint.
    let mut quant = serde_json::Map::new();
    quant.insert("group_size".into(), 64.into());
    quant.insert("bits".into(), 8.into());
    let raw_bases = [
        "language_model.model.embed_tokens",
        ".mlp.gate",
        ".mlp.shared_expert_gate",
        ".linear_attn.in_proj_a",
        ".linear_attn.in_proj_b",
    ];
    let is_raw_base = |base: &str| raw_bases.iter().any(|r| base.ends_with(r));

    // Dense projections the Qwen fixture writes INT4, rewritten INT8 here.
    let dense_int4: std::collections::HashMap<String, (usize, usize)> = source
        .iter()
        .filter(|t| t.dtype == "U32" && t.name.ends_with(".weight"))
        .filter(|t| t.name.contains(".self_attn.") || t.name.contains(".linear_attn."))
        .filter_map(|t| {
            let base = t.name.strip_suffix(".weight")?;
            (!is_raw_base(base) && !overrides.contains_key(base)).then(|| {
                (
                    base.to_string(),
                    (t.shape[0] as usize, t.shape[1] as usize * 8),
                )
            })
        })
        .collect();

    let mut out: Vec<Tensor> = Vec::new();
    for (i, t) in source.into_iter().enumerate() {
        let seed = 9000 + i as u64;
        if t.name.starts_with("language_model.lm_head") {
            continue;
        }
        let base = t
            .name
            .strip_suffix(".weight")
            .or_else(|| t.name.strip_suffix(".scales"))
            .or_else(|| t.name.strip_suffix(".biases"))
            .unwrap_or(&t.name)
            .to_string();
        let renamed = |name: &str| name.replacen("language_model.model.", "backbone.model.", 1);
        if is_raw_base(&base) {
            // One raw BF16 matrix replaces the packed triple.
            if t.name.ends_with(".weight") {
                let rows = if base.ends_with(".mlp.shared_expert_gate") {
                    1
                } else if base.ends_with(".mlp.gate") {
                    experts
                } else if base.ends_with("embed_tokens") {
                    vocab
                } else {
                    v_heads
                };
                out.push(raw_bf16(
                    &renamed(&format!("{base}.weight")),
                    rows,
                    hidden,
                    seed,
                ));
            }
            continue;
        }
        if let Some(&(rows, cols)) = dense_int4.get(&base) {
            // The global default is 8, so these need no per-path entry.
            if t.name.ends_with(".weight") {
                out.extend(int8_triple(
                    &renamed(&format!("{base}.weight")),
                    rows,
                    cols,
                    seed,
                ));
            }
            continue;
        }
        if is_norm(&t.name) {
            let center = match norms {
                Drama3NormStorage::Shifted => 1.0,
                Drama3NormStorage::Unshifted => 0.0,
            };
            let n = t.shape[0] as usize;
            out.push(f32_norm(&renamed(&t.name), n, center, seed));
            continue;
        }
        if t.dtype == "U32" && t.name.ends_with(".weight") {
            let bits = overrides.get(&base).copied().unwrap_or(4);
            quant.insert(
                renamed(&base),
                serde_json::json!({"group_size": 64, "bits": bits}),
            );
        }
        out.push(Tensor {
            name: renamed(&t.name),
            ..t
        });
    }

    // The side tensors: a representative few of each storage the real tower
    // and head use, under their real names.
    out.push(raw_bf16("tower.codebook_embeddings.weight", 20, hidden, 1));
    out.extend(int8_triple(
        "tower.shared_layers.0.eh_proj.weight",
        hidden,
        2 * hidden,
        2,
    ));
    out.push(f32_norm(
        "tower.shared_layers.0.transformer_layer.attention_norm.weight",
        hidden,
        1.0,
        3,
    ));
    out.extend(int8_triple("semantic_head.weight", 17, hidden, 4));
    for side in ["tower.shared_layers.0.eh_proj", "semantic_head"] {
        quant.insert(
            side.into(),
            serde_json::json!({"group_size": 64, "bits": 8}),
        );
    }

    let config = serde_json::json!({
        "model_type": "fish_drama3",
        "architectures": ["FishOmniModel"],
        "speech_template_version": "reference-bracket-v1",
        "text_config": text_config(&arch),
        "quantization": serde_json::Value::Object(quant),
    });
    SyntheticDrama3 {
        blob: assemble_safetensors(&out),
        config_text: config.to_string(),
    }
}

/// Builds the fixture and streams it through
/// [`crate::fish_drama3::write_fish_drama3_install_streamed`] into `dir`.
/// Returns the trunk's `ArchConfig` as the intake parsed it.
pub fn build_synthetic_fish_drama3_install(
    dir: &std::path::Path,
    vocab_size: i64,
    num_experts: i64,
    model_id: &str,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    let checkpoint =
        synthetic_fish_drama3_checkpoint(vocab_size, num_experts, Drama3NormStorage::Shifted);
    write_synthetic_fish_drama3(dir, &checkpoint, model_id)
}

/// Streams an already-built fixture checkpoint into `dir`.
pub fn write_synthetic_fish_drama3(
    dir: &std::path::Path,
    checkpoint: &SyntheticDrama3,
    model_id: &str,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    let source = MemoryRangeSource::new(&checkpoint.blob);
    let mut header = parse_header(
        &checkpoint.blob,
        crate::safetensors_header::DEFAULT_MAX_HEADER_BYTES,
    )?;
    crate::fish_drama3::canonicalize_drama3_header(&mut header)?;
    let shards = Gemma4Shards::single(&header, &source);
    crate::fish_drama3::write_fish_drama3_install_streamed(
        dir,
        &checkpoint.config_text,
        model_id,
        &shards,
        |_| {},
    )
}
