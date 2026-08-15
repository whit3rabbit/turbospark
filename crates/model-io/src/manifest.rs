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

/// Affine-quant group size at ONE bit (matches
/// `turbospark_compute::quant_1bit::BONSAI_GROUP_SIZE`, duplicated for the
/// reason above).
///
/// A separate constant rather than a widening of [`QUANT_GROUP_SIZE`],
/// because the two are not alternatives a caller picks between: 64 is what
/// every INT4/INT8 GEMV kernel is compiled against, and 128 is what the one
/// published 1-bit checkpoint declares. See [`validate_quant`] for why the
/// bit width, the group size and the companion dtype are checked as one
/// conjunction rather than three independent axes.
const QUANT_1BIT_GROUP_SIZE: i64 = 128;

/// Affine-quant group size at TWO bits (matches
/// `turbospark_compute::quant_2bit::TERNARY_GROUP_SIZE`, duplicated for the
/// reason above).
///
/// Equal to [`QUANT_1BIT_GROUP_SIZE`] and a separate constant for the same
/// reason that one is separate from [`QUANT_GROUP_SIZE`]: the two happen to
/// agree because one publisher chose 128 for both of its checkpoints, not
/// because sub-4-bit implies 128.
const QUANT_2BIT_GROUP_SIZE: i64 = 128;

/// Default maximum byte limit for reading `manifest.json`.
pub const DEFAULT_MAX_BYTES: u64 = 4 * 1024 * 1024;

/// File size and SHA-256 entry in manifest.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ManifestFileEntry {
    /// Expected file size in bytes.
    pub size: u64,
    /// Expected SHA-256 digest string.
    pub sha256: String,
}

/// Serialized architecture specification in `manifest.json`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestArch {
    /// Hidden dimension size.
    pub hidden_size: i64,
    /// Shared expert intermediate size.
    pub ffn_intermediate: i64,
    /// Routed expert intermediate size.
    pub moe_intermediate_size: i64,
    /// Number of query attention heads.
    pub num_heads: i64,
    /// Number of key/value heads for SWA layers.
    #[serde(rename = "numKVHeads")]
    pub num_kv_heads: i64,
    /// Number of key/value heads for full attention layers.
    #[serde(rename = "numFullKVHeads")]
    pub num_full_kv_heads: i64,
    /// Head dimension size for SWA layers.
    pub head_dim: i64,
    /// Head dimension size for full attention layers.
    pub full_head_dim: i64,
    /// Vocabulary size.
    pub vocab_size: i64,
    /// Sliding window context limit.
    pub sliding_window: i64,
    /// Final logit soft-capping value.
    pub final_logit_softcap: f64,
    /// RoPE base theta value.
    pub rope_theta: f64,
    /// Full-attention RoPE base theta value.
    pub full_rope_theta: f64,
    /// Partial rotary dimension factor.
    pub partial_rotary_factor: f64,
    /// Total layer count.
    pub num_layers: i64,
    /// Total routed experts count per MoE layer.
    pub num_experts: i64,
    /// Selected expert count per token.
    pub top_k_experts: i64,
    /// True if word embeddings are tied to output lm_head.
    pub tie_word_embeddings: bool,
    /// True if key and value attention projections share memory structures.
    pub attention_k_eq_v: bool,
    /// Activation function name for MLP layers.
    pub hidden_activation: String,
    /// Per-layer attention type mask.
    pub full_attention_layer_mask: Vec<i64>,

    /// Model family identifier string.
    #[serde(default)]
    pub family: Option<String>,
    /// True if attention output is gated.
    #[serde(default)]
    pub attn_output_gate: Option<bool>,
    /// Attention scale factor.
    #[serde(default)]
    pub attention_scale: Option<f64>,
    /// True if embedding is scaled by sqrt(hidden_size).
    #[serde(default)]
    pub embedding_scaled_by_sqrt_hidden: Option<bool>,
    /// True if router weights are scaled.
    #[serde(default)]
    pub router_scaled: Option<bool>,
    /// True if sandwich RMSNorms are used in FFN.
    #[serde(default)]
    pub ffn_sandwich_norms: Option<bool>,
    /// True if shared expert output is scalar-gated.
    #[serde(default)]
    pub shared_expert_gated: Option<bool>,
    /// True if RoPE subdim uses NeoX ordering.
    #[serde(default)]
    pub rope_neox_subdim: Option<bool>,
    /// Linear attention key heads count.
    #[serde(default)]
    pub linear_num_k_heads: Option<i64>,
    /// Linear attention value heads count.
    #[serde(default)]
    pub linear_num_v_heads: Option<i64>,
    /// Linear attention key head dimension.
    #[serde(default)]
    pub linear_key_head_dim: Option<i64>,
    /// Linear attention value head dimension.
    #[serde(default)]
    pub linear_value_head_dim: Option<i64>,
    /// Linear attention depthwise conv kernel size.
    #[serde(default)]
    pub linear_conv_kernel_size: Option<i64>,

    /// Compressed attention Q LoRA rank.
    #[serde(default)]
    pub ca_q_lora_rank: Option<i64>,
    /// Compressed attention Out LoRA rank.
    #[serde(default)]
    pub ca_o_lora_rank: Option<i64>,
    /// Compressed attention output groups.
    #[serde(default)]
    pub ca_o_groups: Option<i64>,
    /// Compressed attention RoPE head dim.
    #[serde(default)]
    pub ca_rope_head_dim: Option<i64>,
    /// Compressed attention index heads count.
    #[serde(default)]
    pub ca_index_n_heads: Option<i64>,
    /// Compressed attention index head dim.
    #[serde(default)]
    pub ca_index_head_dim: Option<i64>,
    /// Compressed attention index top-k selection.
    #[serde(default)]
    pub ca_index_top_k: Option<i64>,
    /// Compressed attention CSA compress rate.
    #[serde(default, rename = "caCSACompressRate")]
    pub ca_csa_compress_rate: Option<i64>,
    /// Compressed attention HCA compress rate.
    #[serde(default, rename = "caHCACompressRate")]
    pub ca_hca_compress_rate: Option<i64>,
    /// Compressed attention RoPE theta.
    #[serde(default)]
    pub ca_compress_rope_theta: Option<f64>,
    /// Compressed attention RoPE scaling factor.
    #[serde(default)]
    pub ca_rope_scaling_factor: Option<f64>,
    /// Compressed attention RoPE scaling original max.
    #[serde(default)]
    pub ca_rope_scaling_original_max: Option<i64>,
    /// Compressed attention RoPE scaling beta fast.
    #[serde(default)]
    pub ca_rope_scaling_beta_fast: Option<f64>,
    /// Compressed attention RoPE scaling beta slow.
    #[serde(default)]
    pub ca_rope_scaling_beta_slow: Option<f64>,
    /// Hyper-connection multiplier.
    #[serde(default)]
    pub hc_mult: Option<i64>,
    /// Hyper-connection Sinkhorn iterations.
    #[serde(default)]
    pub hc_sinkhorn_iters: Option<i64>,
    /// Hyper-connection epsilon.
    #[serde(default)]
    pub hc_eps: Option<f64>,
    /// Hash-routed leading MoE layer count.
    #[serde(default)]
    pub num_hash_routed_layers: Option<i64>,
    /// Router scoring function name.
    #[serde(default)]
    pub router_scoring_func: Option<String>,
    /// Routed expert output scaling factor.
    #[serde(default)]
    pub routed_scaling_factor: Option<f64>,
    /// SwiGLU activation clamp limit.
    #[serde(default)]
    pub swiglu_limit: Option<f64>,
    /// YaRN rope scaling (ROADMAP M5, `gpt-oss`). Absent means no scaling,
    /// which is what `RopeScalingConfig::NONE` says and what every family
    /// before this one declares.
    #[serde(default)]
    pub rope_scaling_factor: Option<f64>,
    #[serde(default)]
    pub rope_scaling_original_context: Option<i64>,
    #[serde(default)]
    pub rope_scaling_beta_fast: Option<f64>,
    #[serde(default)]
    pub rope_scaling_beta_slow: Option<f64>,
}

/// Quantization parameters for a model component slot in `manifest.json`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestQuantSlot {
    /// Weight quantization bit width (e.g. 4 or 8).
    pub weight_bits: i64,
    /// Quantization scheme identifier (e.g. "affine" or "gguf").
    pub scheme: String,
    /// Scale data type name (e.g. "BF16").
    pub scale_type: String,
    /// Bias data type name (e.g. "BF16").
    pub bias_type: String,
    /// Quantization group element count.
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

/// Grouped quantization configuration across model component slots.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestQuant {
    /// Embedding layer quantization slot.
    pub embedding: ManifestQuantSlot,
    /// Attention projections quantization slot.
    pub attention: ManifestQuantSlot,
    /// Router projection quantization slot.
    pub router: ManifestQuantSlot,
    /// Shared expert quantization slot.
    pub shared_expert: ManifestQuantSlot,
    /// Routed expert quantization slot.
    pub routed_expert: ManifestQuantSlot,
}

/// Top-level model manifest deserialized from `manifest.json`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Manifest {
    /// Format magic string ("GTURBO").
    pub magic: String,
    /// Major format version number.
    pub version_major: i64,
    /// Minor format version number.
    pub version_minor: i64,
    /// Feature flags map.
    pub flags: BTreeMap<String, bool>,
    /// Model identifier string.
    #[serde(rename = "modelID")]
    pub model_id: String,
    /// Source HF repository snapshot hash if recorded.
    #[serde(default)]
    pub source_snapshot_hash: Option<String>,
    /// Architecture configuration table.
    pub arch: ManifestArch,
    /// Quantization configuration table if present.
    #[serde(default)]
    pub quant: Option<ManifestQuant>,
    /// Installed files map matching filename to size and hash.
    pub files: BTreeMap<String, ManifestFileEntry>,
    /// Number of experts per layer.
    pub experts_per_layer: i64,
    /// Number of layers.
    pub num_layers: i64,
    /// Stride in bytes per expert in stream storage.
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
///
/// MXFP4 (ROADMAP M5, `gpt-oss`) joins on the narrowest footing yet, and it
/// is narrow in a NEW DIRECTION: it has both routed phases and NO resident
/// GEMV, where Q6_K and Q5_K have a resident GEMV and (almost) no routed
/// kernels. That is the real file's shape rather than a choice -- the one
/// checkpoint carrying MXFP4 puts it in `ffn_{gate,up,down}_exps` and keeps
/// attention, `token_embd` and `output` at Q8_0. This list is what the
/// manifest's per-slot `ggmlType` is read against, so MXFP4 belongs in it;
/// `RealForwardRunner`'s `EXECUTABLE_GGUF_DTYPES`, which reads RESIDENT
/// tensors, deliberately omits it, and its doc explains why the two are twins
/// rather than copies.
pub const EXECUTABLE_GGUF_TYPES: [&str; 8] = [
    "q8_0", "q4_k", "q5_k", "q6_k", "iq3_xxs", "iq4_nl", "iq4_xs", "mxfp4",
];

/// Accepts a quant block iff every slot's shape has kernels behind it.
///
/// Four shapes are accepted; the third came with ROADMAP's 1-bit entry and
/// the fourth with its ternary one.
///
/// 1. **INT4/INT8 affine**: BF16 companions at group 64, per-slot bit widths.
/// 2. **GGUF**: every declared block type in [`EXECUTABLE_GGUF_TYPES`].
/// 3. **1-bit affine**: FP16 companions at group 128.
/// 4. **2-bit affine**: FP16 companions at group 128.
///
/// **EACH SUB-4-BIT SHAPE IS CHECKED AS ONE CONJUNCTION, NOT AS WIDENINGS OF
/// THE FIRST, and that is the point of writing them as separate predicates.**
/// It would have been shorter to add `1` and `2` to the affine bit lists,
/// `128` to the group sizes and `fp16` to the companion types, and the result
/// would accept a dozen combinations that no kernel implements -- 1-bit at
/// group 64, 4-bit with FP16 companions, and so on. The real shapes are
/// `(4|8, bf16, 64)`, `(1, fp16, 128)` and `(2, fp16, 128)`, because a
/// checkpoint's bit width, companion dtype and group size travel together,
/// and the FP16-versus-BF16 axis is the dangerous one: the two planes are the
/// same width, so a wrong reading passes every length check and decodes these
/// checkpoints' 0.027 and 0.0137 scales as ~1e-16.
///
/// Note the fourth shape does NOT subsume the `weight_bits == 2` the affine
/// arm already allows on `routedExpert`: that one is BF16 at group 64, for
/// the DeepSeek-V4-Flash dynamic-quant checkpoint, and the two 2-bit shapes
/// share nothing but their width.
///
/// **Both sub-4-bit shapes are accepted on ALL FIVE SLOTS, including
/// `routedExpert`, though neither has a routed-expert kernel.** That is not
/// an oversight and it is not a claim that such an MoE install would run.
/// `manifest.quant` has five fixed slots and no architecture fills all five;
/// both published sub-4-bit checkpoints are DENSE, so their router,
/// shared-expert and routed-expert probes find nothing and fall back to the
/// type the rest of the model uses
/// (`crates/repack` Gotcha 8: refusing a slot for a component the install
/// does not have is how a runnable model fails to open). An install that
/// really did carry 1-bit routed experts passes here and fails at the routed
/// dispatch, by name -- the same weaker footing Q6_K and the IQ types stand
/// on, stated in [`EXECUTABLE_GGUF_TYPES`]'s doc.
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
    // A slot that is byte-for-byte the ATTENTION slot is a DEFAULTED
    // statement about a component the model does not have, not a claim about
    // bytes. The repack walks write it that way on purpose
    // (`crates/repack`'s `manifest_quant_for` and Gotcha 8's `or_attention`):
    // `manifest.quant` has five fixed slots and no architecture fills all
    // five, so refusing one is how a runnable dense install fails to open
    // with a message about something it never had -- which cost M4 one
    // five-minute re-stream per slot.
    //
    // The residual is worth naming: this admits an MoE install whose ROUTER
    // genuinely is 4-bit and happens to match its attention slot, which the
    // INT8-only `router_gemv_gemma4_r4` would misread. Nothing writes one --
    // Gemma and Qwen both override the router to 8 bits, and a checkpoint
    // that did not would be a new family's problem -- but it is admitted
    // here rather than refused, unlike the bit lists below.
    let attention = &quant.attention;
    for (name, slot, allowed_bits) in slots {
        if !std::ptr::eq(slot, attention) && slot == attention {
            continue;
        }
        let affine = allowed_bits.contains(&slot.weight_bits)
            && slot.scheme.to_lowercase() == "affine"
            && slot.scale_type.to_lowercase() == "bf16"
            && slot.bias_type.to_lowercase() == "bf16"
            && slot.group_size == QUANT_GROUP_SIZE;
        // The 1-bit shape, whole. See this function's doc for why the three
        // fields are one conjunction and why every slot accepts it.
        let affine_1bit = slot.weight_bits == 1
            && slot.scheme.to_lowercase() == "affine"
            && slot.scale_type.to_lowercase() == "fp16"
            && slot.bias_type.to_lowercase() == "fp16"
            && slot.group_size == QUANT_1BIT_GROUP_SIZE;
        // The 2-bit shape (ROADMAP's ternary entry), a FOURTH conjunction and
        // not a widening of the third: it is a separate `(bits, companions,
        // group)` triple that happens to share two of its three fields with
        // the 1-bit one. Note `routedExpert` already admits `weight_bits == 2`
        // through the affine arm above, at BF16 and group 64 -- a different
        // shape entirely, for the DeepSeek-V4 dynamic-quant checkpoint -- so
        // the two must not be collapsed into one bit list.
        let affine_2bit = slot.weight_bits == 2
            && slot.scheme.to_lowercase() == "affine"
            && slot.scale_type.to_lowercase() == "fp16"
            && slot.bias_type.to_lowercase() == "fp16"
            && slot.group_size == QUANT_2BIT_GROUP_SIZE;
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
        if !(affine || affine_1bit || affine_2bit || gguf) {
            let detail = match slot.scheme.to_lowercase().as_str() {
                // An affine slot has four fields that can each be wrong and a
                // bare "unsupported" names none of them. It matters most on
                // the companion dtype, where the failure is otherwise
                // invisible: FP16 and BF16 are the same width, so nothing
                // downstream notices, and this message is the only place the
                // mismatch is ever spelled out.
                "affine" => format!(
                    "unsupported quantization for {name}: affine slot is \
                     {}-bit with {}/{} companions at group {}, and the shapes \
                     with kernels are {allowed_bits:?}-bit bf16 at group \
                     {QUANT_GROUP_SIZE}, 1-bit fp16 at group \
                     {QUANT_1BIT_GROUP_SIZE} and 2-bit fp16 at group \
                     {QUANT_2BIT_GROUP_SIZE}",
                    slot.weight_bits, slot.scale_type, slot.bias_type, slot.group_size
                ),
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
