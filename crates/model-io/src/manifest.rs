//! `manifest.json` decode and validation against a resolved [`ArchConfig`].
//! Ported from `Infrastructure/ModelIO/ManifestReader.swift`.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use serde::Deserialize;

use crate::arch_baselines::all_known_architectures;
use crate::arch_config::{ArchConfig, ModelFamily};
use crate::error::ModelError;

/// Affine-quant group size (matches `turbospark_compute::quant::GROUP_SIZE`;
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
    /// The ggml block type, present only on `scheme: "gguf"` slots (ROADMAP
    /// Phase G). Optional because every affine install predates it and none
    /// writes it.
    #[serde(default)]
    pub ggml_type: Option<String>,
    /// Every ggml block type this slot carries, when it carries more than one
    /// (ROADMAP Phase S). Optional, and absent means "exactly `ggml_type`".
    ///
    /// A mixed sub-4-bit checkpoint needs this: the Phase S candidate's routed
    /// experts are IQ3_XXS gate/up over IQ4_NL down, with IQ4_XS and Q8_0 on
    /// layer 29, so no single type describes the slot. `ggml_type` stays as
    /// the DOMINANT one, which keeps a hand-read of the manifest informative;
    /// this is what the gate actually checks, member by member.
    #[serde(default)]
    pub ggml_types: Option<Vec<String>>,
}

impl ManifestQuantSlot {
    /// The block types this slot claims, dominant one first.
    fn declared_types(&self) -> Vec<&str> {
        match (&self.ggml_types, &self.ggml_type) {
            (Some(all), _) if !all.is_empty() => all.iter().map(String::as_str).collect(),
            (_, Some(one)) => vec![one.as_str()],
            _ => Vec::new(),
        }
    }
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
    validate(&manifest, expecting)?;
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

/// The GGUF block types this port can EXECUTE, as they are spelled in a
/// manifest's `ggmlType` (ROADMAP Phase G Stage 2).
///
/// A GGUF-sourced install is written for every block type the parser knows,
/// which is deliberately more than the set with kernels behind it: the repack
/// walk's job is to carry bytes, and refusing to install would lose the
/// artifact. This list is what decides whether one can be OPENED, and it
/// grows only when a kernel plus its parity test land. `crates/runtime`
/// applies the same rule again to the resident index's dtype tags, which is
/// the backstop for a hand-edited manifest.
/// Q6_K is here on weaker grounds than the other two and the difference is
/// worth knowing: it has a resident GEMV and nothing else, because the only
/// real file that uses it puts it in `output.weight`. An install that carried
/// Q6_K experts would pass this gate and fail at the routed dispatch instead,
/// which is a worse error message but not a wrong answer.
/// The three IQ types (ROADMAP Phase S) join on the same terms as the rest,
/// and one of them is narrower than it looks: IQ3_XXS and IQ4_XS have a
/// routed phase-1 kernel, IQ4_NL a routed phase-2 one, and all three a
/// resident GEMV, but there is no IQ4_NL phase 1 and no IQ3_XXS phase 2
/// because no real file asks for either. That is the same weaker footing
/// Q6_K stands on, and it fails the same way: at the dispatch site, by name.
pub const EXECUTABLE_GGUF_TYPES: [&str; 6] =
    ["q8_0", "q4_k", "q6_k", "iq3_xxs", "iq4_nl", "iq4_xs"];

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
        let affine = allowed_bits.contains(&slot.weight_bits)
            && slot.scheme.to_lowercase() == "affine"
            && slot.scale_type.to_lowercase() == "bf16"
            && slot.bias_type.to_lowercase() == "bf16"
            && slot.group_size == QUANT_GROUP_SIZE;
        // A GGUF slot carries no bits, no group size and no companion types:
        // the scale lives inside each block. What it does carry is the block
        // type -- possibly SEVERAL, since ROADMAP Phase S -- and that is the
        // whole question: a Q4_K install and a Q8_0 one are equally
        // well-formed here and only one of them has kernels.
        //
        // Every declared type must be executable, not just the dominant one.
        // Checking only `ggmlType` would let a mixed install through on the
        // strength of its majority and fail at a dispatch thirty layers in.
        let declared = slot.declared_types();
        let gguf = slot.scheme.to_lowercase() == "gguf"
            && !declared.is_empty()
            && declared
                .iter()
                .all(|t| EXECUTABLE_GGUF_TYPES.contains(&t.to_lowercase().as_str()));
        if !(affine || gguf) {
            let detail = match slot.scheme.to_lowercase().as_str() {
                "gguf" => {
                    let offending: Vec<&str> = declared
                        .iter()
                        .copied()
                        .filter(|t| !EXECUTABLE_GGUF_TYPES.contains(&t.to_lowercase().as_str()))
                        .collect();
                    let named = if offending.is_empty() {
                        "unspecified".to_string()
                    } else {
                        offending.join(", ")
                    };
                    format!(
                        "unsupported quantization for {name}: GGUF block type {named} has no \
                         kernel in this port (executable types: {})",
                        EXECUTABLE_GGUF_TYPES.join(", ")
                    )
                }
                _ => format!("unsupported quantization for {name}"),
            };
            return Err(ModelError::IndexCorrupt { detail });
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
