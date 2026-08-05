//! `manifest.json` decode and validation against a resolved [`ArchConfig`].
//! Ported from `Infrastructure/ModelIO/ManifestReader.swift`.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use serde::Deserialize;

use crate::arch_baselines::all_known_architectures;
use crate::arch_config::{ArchConfig, ModelFamily};
use crate::error::ModelError;

/// Affine-quant group size (matches `mrefrust_compute::quant::GROUP_SIZE`;
/// duplicated here rather than adding a compute dependency to this crate).
const QUANT_GROUP_SIZE: i64 = 64;

pub const DEFAULT_MAX_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ManifestFileEntry {
    pub size: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestArch {
    pub hidden_size: i64,
    pub ffn_intermediate: i64,
    pub moe_intermediate_size: i64,
    pub num_heads: i64,
    #[serde(rename = "numKVHeads")]
    pub num_kv_heads: i64,
    #[serde(rename = "numFullKVHeads")]
    pub num_full_kv_heads: i64,
    pub head_dim: i64,
    pub full_head_dim: i64,
    pub vocab_size: i64,
    pub sliding_window: i64,
    pub final_logit_softcap: f64,
    pub rope_theta: f64,
    pub full_rope_theta: f64,
    pub partial_rotary_factor: f64,
    pub num_layers: i64,
    pub num_experts: i64,
    pub top_k_experts: i64,
    pub tie_word_embeddings: bool,
    pub attention_k_eq_v: bool,
    pub hidden_activation: String,
    pub full_attention_layer_mask: Vec<i64>,

    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub attn_output_gate: Option<bool>,
    #[serde(default)]
    pub attention_scale: Option<f64>,
    #[serde(default)]
    pub embedding_scaled_by_sqrt_hidden: Option<bool>,
    #[serde(default)]
    pub router_scaled: Option<bool>,
    #[serde(default)]
    pub ffn_sandwich_norms: Option<bool>,
    #[serde(default)]
    pub shared_expert_gated: Option<bool>,
    #[serde(default)]
    pub rope_neox_subdim: Option<bool>,
    #[serde(default)]
    pub linear_num_k_heads: Option<i64>,
    #[serde(default)]
    pub linear_num_v_heads: Option<i64>,
    #[serde(default)]
    pub linear_key_head_dim: Option<i64>,
    #[serde(default)]
    pub linear_value_head_dim: Option<i64>,
    #[serde(default)]
    pub linear_conv_kernel_size: Option<i64>,

    #[serde(default)]
    pub ca_q_lora_rank: Option<i64>,
    #[serde(default)]
    pub ca_o_lora_rank: Option<i64>,
    #[serde(default)]
    pub ca_o_groups: Option<i64>,
    #[serde(default)]
    pub ca_rope_head_dim: Option<i64>,
    #[serde(default)]
    pub ca_index_n_heads: Option<i64>,
    #[serde(default)]
    pub ca_index_head_dim: Option<i64>,
    #[serde(default)]
    pub ca_index_top_k: Option<i64>,
    #[serde(default, rename = "caCSACompressRate")]
    pub ca_csa_compress_rate: Option<i64>,
    #[serde(default, rename = "caHCACompressRate")]
    pub ca_hca_compress_rate: Option<i64>,
    #[serde(default)]
    pub ca_compress_rope_theta: Option<f64>,
    #[serde(default)]
    pub ca_rope_scaling_factor: Option<f64>,
    #[serde(default)]
    pub ca_rope_scaling_original_max: Option<i64>,
    #[serde(default)]
    pub ca_rope_scaling_beta_fast: Option<f64>,
    #[serde(default)]
    pub ca_rope_scaling_beta_slow: Option<f64>,
    #[serde(default)]
    pub hc_mult: Option<i64>,
    #[serde(default)]
    pub hc_sinkhorn_iters: Option<i64>,
    #[serde(default)]
    pub hc_eps: Option<f64>,
    #[serde(default)]
    pub num_hash_routed_layers: Option<i64>,
    #[serde(default)]
    pub router_scoring_func: Option<String>,
    #[serde(default)]
    pub routed_scaling_factor: Option<f64>,
    #[serde(default)]
    pub swiglu_limit: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestQuantSlot {
    pub weight_bits: i64,
    pub scheme: String,
    pub scale_type: String,
    pub bias_type: String,
    pub group_size: i64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestQuant {
    pub embedding: ManifestQuantSlot,
    pub attention: ManifestQuantSlot,
    pub router: ManifestQuantSlot,
    pub shared_expert: ManifestQuantSlot,
    pub routed_expert: ManifestQuantSlot,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Manifest {
    pub magic: String,
    pub version_major: i64,
    pub version_minor: i64,
    pub flags: BTreeMap<String, bool>,
    #[serde(rename = "modelID")]
    pub model_id: String,
    #[serde(default)]
    pub source_snapshot_hash: Option<String>,
    pub arch: ManifestArch,
    #[serde(default)]
    pub quant: Option<ManifestQuant>,
    pub files: BTreeMap<String, ManifestFileEntry>,
    pub experts_per_layer: i64,
    pub num_layers: i64,
    pub expert_stride: u64,
}

/// Recognized flag keys. Anything else in `manifest.flags` is an error.
pub fn known_flags() -> HashSet<&'static str> {
    ["streamingPresent", "turboQuantKV", "aneSharedExpert"]
        .into_iter()
        .collect()
}

/// Required file entries (relative to `model.gturbo/`).
pub const REQUIRED_FILES: [&str; 2] = ["model_weights.bin", "packed_experts/layout.json"];

pub fn load(dir: &Path, expecting: &ArchConfig, max_bytes: u64) -> Result<Manifest, ModelError> {
    let manifest = decode_manifest(dir, max_bytes)?;
    validate(&manifest, expecting)?;
    Ok(manifest)
}

/// Read and parse `manifest.json` without validating it against an
/// expected [`ArchConfig`] -- the step [`load`] and
/// [`arch_from_manifest_dir`] share.
fn decode_manifest(dir: &Path, max_bytes: u64) -> Result<Manifest, ModelError> {
    let manifest_path = dir.join("manifest.json");
    if !manifest_path.exists() {
        return Err(ModelError::PartialInstall {
            path: dir.display().to_string(),
        });
    }
    let size = file_size(&manifest_path)?;
    if size > max_bytes {
        return Err(ModelError::IndexCorrupt {
            detail: format!("manifest.json size {size} exceeds metadata cap {max_bytes}"),
        });
    }
    let data = std::fs::read(&manifest_path).map_err(|e| ModelError::IoFailed {
        call: "read".to_string(),
        detail: e.to_string(),
    })?;
    let manifest: Manifest =
        serde_json::from_slice(&data).map_err(|e| ModelError::IndexCorrupt {
            detail: format!("manifest.json: {e}"),
        })?;
    Ok(manifest)
}

fn file_size(path: &Path) -> Result<u64, ModelError> {
    std::fs::metadata(path)
        .map(|m| m.len())
        .map_err(|e| ModelError::IoFailed {
            call: "stat".to_string(),
            detail: e.to_string(),
        })
}

pub fn validate(m: &Manifest, expected: &ArchConfig) -> Result<(), ModelError> {
    if m.magic != "GTURBO" {
        return Err(ModelError::NotAGTurboDirectory);
    }
    if m.version_major != 1 {
        return Err(ModelError::UnsupportedVersion {
            major: m.version_major,
            minor: m.version_minor,
        });
    }
    let known = known_flags();
    for key in m.flags.keys() {
        if !known.contains(key.as_str()) {
            return Err(ModelError::UnknownFlag { name: key.clone() });
        }
    }
    if m.flags.get("turboQuantKV") == Some(&true) {
        return Err(ModelError::IndexCorrupt {
            detail: "manifest requests removed TurboQuant KV runtime support".to_string(),
        });
    }
    crate::arch_validation::validate_arch(&m.arch, expected)?;
    if let Some(quant) = &m.quant {
        validate_quant(quant)?;
    } else if is_production_arch(expected) {
        return Err(ModelError::IndexCorrupt {
            detail: "manifest.quant is required for the production architecture".to_string(),
        });
    }
    let page_size = page_size_bytes();
    if m.expert_stride % page_size != 0 {
        return Err(ModelError::ExpertStrideNotPageAligned {
            stride: m.expert_stride,
            page_size,
        });
    }
    for f in REQUIRED_FILES {
        if !m.files.contains_key(f) {
            return Err(ModelError::MissingFile {
                name: f.to_string(),
            });
        }
    }
    for layer in 0..m.num_layers {
        let padded = format!("packed_experts/layer_{layer:02}.bin");
        let plain = format!("packed_experts/layer_{layer}.bin");
        if !m.files.contains_key(&padded) && !m.files.contains_key(&plain) {
            return Err(ModelError::MissingFile { name: padded });
        }
    }
    Ok(())
}

/// The page size this format's writer aligns `expertStride` to. Hardcoded
/// rather than queried from the OS: both macOS/arm64 and Linux/x86_64 (the
/// platforms this workspace targets) use 4 KiB pages, and the manifest
/// format itself has no per-install page size field to validate against.
fn page_size_bytes() -> u64 {
    4096
}

/// A manifest matching one of the shipped production baselines must carry
/// quantization metadata; toy/synthetic manifests may omit it.
fn is_production_arch(expected: &ArchConfig) -> bool {
    all_known_architectures()
        .iter()
        .any(|b| b.num_layers == expected.num_layers && b.hidden_size == expected.hidden_size)
}

fn validate_quant(quant: &ManifestQuant) -> Result<(), ModelError> {
    let slots: [(&str, &ManifestQuantSlot, &[i64]); 5] = [
        ("embedding", &quant.embedding, &[4]),
        ("attention", &quant.attention, &[4]),
        ("router", &quant.router, &[8]),
        ("sharedExpert", &quant.shared_expert, &[4, 8]),
        // Routed experts additionally allow 2-bit: the DeepSeek-V4-Flash
        // dynamic-quant checkpoint ships Q2 experts under a Q4 core.
        ("routedExpert", &quant.routed_expert, &[2, 4]),
    ];
    for (name, slot, allowed_bits) in slots {
        let ok = allowed_bits.contains(&slot.weight_bits)
            && slot.scheme.to_lowercase() == "affine"
            && slot.scale_type.to_lowercase() == "bf16"
            && slot.bias_type.to_lowercase() == "bf16"
            && slot.group_size == QUANT_GROUP_SIZE;
        if !ok {
            return Err(ModelError::IndexCorrupt {
                detail: format!("unsupported quantization for {name}"),
            });
        }
    }
    Ok(())
}

/// Decode just enough of `manifest.json` to identify the model family,
/// without arch validation.
pub fn peek_family(dir: &Path, max_bytes: u64) -> Result<ModelFamily, ModelError> {
    let manifest_path = dir.join("manifest.json");
    if !manifest_path.exists() {
        return Err(ModelError::PartialInstall {
            path: dir.display().to_string(),
        });
    }
    let size = file_size(&manifest_path)?;
    if size > max_bytes {
        return Err(ModelError::IndexCorrupt {
            detail: format!("manifest.json size {size} exceeds metadata cap {max_bytes}"),
        });
    }
    let data = std::fs::read(&manifest_path).map_err(|e| ModelError::IoFailed {
        call: "read".to_string(),
        detail: e.to_string(),
    })?;
    let manifest: Manifest =
        serde_json::from_slice(&data).map_err(|e| ModelError::IndexCorrupt {
            detail: format!("manifest.json: {e}"),
        })?;
    let Some(raw) = manifest.arch.family else {
        return Ok(ModelFamily::Gemma4);
    };
    ModelFamily::parse(&raw).ok_or_else(|| ModelError::IndexCorrupt {
        detail: format!("unknown arch.family \"{raw}\""),
    })
}

/// Reconstruct a full [`ArchConfig`] from an install's `manifest.json`.
///
/// Starts from the family baseline (the same fallback rule
/// [`crate::validate`] applies to omitted family-extension fields) and
/// overwrites every shape field with what the manifest actually says. Only
/// Gemma 4 installs are resolved; anything else is rejected here rather
/// than failing later inside the runner.
pub fn arch_from_manifest_dir(dir: &Path) -> Result<ArchConfig, ModelError> {
    let manifest = decode_manifest(dir, DEFAULT_MAX_BYTES)?;
    let m = &manifest.arch;
    if m.family.as_deref().is_some_and(|f| f != "gemma4") {
        return Err(ModelError::IndexCorrupt {
            detail: format!(
                "manifest family {:?} is not supported by real generation yet",
                m.family
            ),
        });
    }

    let mut arch = crate::arch_baselines::gemma4_26b_a4b();
    arch.hidden_size = m.hidden_size;
    arch.intermediate_size = m.ffn_intermediate;
    arch.moe_intermediate_size = m.moe_intermediate_size;
    arch.num_heads = m.num_heads;
    arch.num_kv_heads = m.num_kv_heads;
    arch.num_full_kv_heads = m.num_full_kv_heads;
    arch.head_dim = m.head_dim;
    arch.full_head_dim = m.full_head_dim;
    arch.vocab_size = m.vocab_size;
    arch.num_layers = m.num_layers;
    arch.sliding_window = m.sliding_window;
    arch.final_logit_softcap = m.final_logit_softcap;
    arch.rope_theta = m.rope_theta;
    arch.full_rope_theta = m.full_rope_theta;
    arch.partial_rotary_factor = m.partial_rotary_factor;
    arch.num_experts = m.num_experts;
    arch.top_k_experts = m.top_k_experts;
    arch.tie_word_embeddings = m.tie_word_embeddings;
    arch.attention_k_eq_v = m.attention_k_eq_v;
    arch.hidden_activation = m.hidden_activation.clone();
    arch.full_attention_layer_mask = m
        .full_attention_layer_mask
        .iter()
        .map(|&v| v as u8)
        .collect();
    if let Some(scale) = m.attention_scale {
        arch.attention_scale = scale;
    }
    Ok(arch)
}
