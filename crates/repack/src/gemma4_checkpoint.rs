//! Real Gemma 4 checkpoint repack mapping: classifies the
//! `mlx-community/gemma-4-*-4bit` conversion's tensor names, orders the
//! resident LM tensors the way the Swift repacker does
//! (`RepackPlanner.swift`), passes the already-quantized u32 weights and
//! BF16 scales/biases through byte-for-byte (no re-quantization — the MLX
//! affine int4 packing is exactly this port's packed layout when viewed as
//! little-endian bytes), carries unquantized tensors (norms, `router.scale`,
//! `layer_scalar`) as raw BF16/FP16/FP32 entries, and slices the per-layer
//! `.experts.switch_glu.` routed-expert tensors into per-expert blobs with
//! ONE page-rounded (16 KiB) expert stride for the whole model.
//!
//! Tensor names are kept VERBATIM from the source checkpoint (the Swift
//! original's behavior): the resident index addresses
//! `language_model.model.layers.N.self_attn.q_proj.weight`, not a renamed
//! alias. The synthetic installs' short names (`layerN.q_proj`) remain a
//! separate, synthetic-only convention.

use std::collections::BTreeMap;

use model_io::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig,
    ModelFamily,
};

use crate::gturbo_writer::{ExpertBlob, LayerBlobs, SubTensor};
use crate::ranged_download::{DownloadError, RangeSource};
use crate::resident_writer::{
    RawTensorSpec, ResidentEntrySpec, ResidentTensorSpec, DTYPE_BF16, DTYPE_FP16, DTYPE_FP32,
};
use crate::safetensors_header::{SafetensorsHeader, TensorInfo};

/// On-disk page alignment unit for `.gturbo` files (the Swift repacker's
/// `Layout.pageBytes`): fixed at 16 KiB regardless of host page size.
pub const GTURBO_PAGE_BYTES: u64 = 16_384;

#[derive(Debug)]
pub enum Gemma4Error {
    Config(String),
    MissingTensor(String),
    MissingCompanion(String),
    UnsupportedDtype { tensor: String, dtype: String },
    ShapeMismatch { tensor: String, detail: String },
    UnknownTensor(String),
    Download(DownloadError),
}

impl std::fmt::Display for Gemma4Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Gemma4Error::Config(detail) => write!(f, "config.json invalid: {detail}"),
            Gemma4Error::MissingTensor(name) => write!(f, "missing tensor: {name}"),
            Gemma4Error::MissingCompanion(name) => {
                write!(f, "missing .scales/.biases companion for: {name}")
            }
            Gemma4Error::UnsupportedDtype { tensor, dtype } => {
                write!(f, "tensor {tensor} has unsupported dtype {dtype}")
            }
            Gemma4Error::ShapeMismatch { tensor, detail } => {
                write!(f, "tensor {tensor}: {detail}")
            }
            Gemma4Error::UnknownTensor(name) => write!(f, "unclassifiable tensor: {name}"),
            Gemma4Error::Download(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Gemma4Error {}

impl From<DownloadError> for Gemma4Error {
    fn from(e: DownloadError) -> Self {
        Gemma4Error::Download(e)
    }
}

// ---------------------------------------------------------------------------
// config.json -> ArchConfig
// ---------------------------------------------------------------------------

/// Parses a Gemma 4 `config.json` (the `text_config` wrapper form the
/// multimodal checkpoints ship) into an [`ArchConfig`], mirroring the Swift
/// `ArchInfo.loadGemma4`: `layer_types` maps to the layer mask
/// (`full_attention` = 1, anything else = 0), the two `rope_parameters`
/// sub-objects supply the full/SWA thetas and the partial rotary factor,
/// and every family-extension field takes Gemma 4's fixed value.
pub fn parse_gemma4_config(json: &str) -> Result<ArchConfig, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    let tc = root
        .get("text_config")
        .ok_or_else(|| Gemma4Error::Config("no text_config".to_string()))?;

    let i = |k: &str| -> Result<i64, Gemma4Error> {
        tc.get(k)
            .and_then(|v| v.as_i64())
            .ok_or_else(|| Gemma4Error::Config(format!("missing {k}")))
    };
    let d = |k: &str| -> Result<f64, Gemma4Error> {
        tc.get(k)
            .and_then(|v| v.as_f64())
            .ok_or_else(|| Gemma4Error::Config(format!("missing {k}")))
    };

    let mask: Vec<u8> = tc
        .get("layer_types")
        .and_then(|v| v.as_array())
        .map(|types| {
            types
                .iter()
                .map(|t| (t.as_str() == Some("full_attention")) as u8)
                .collect()
        })
        .unwrap_or_default();

    let rope = tc.get("rope_parameters");
    let rope_sub = |kind: &str, key: &str, fallback: f64| -> f64 {
        rope.and_then(|r| r.get(kind))
            .and_then(|s| s.get(key))
            .and_then(|v| v.as_f64())
            .unwrap_or(fallback)
    };
    let prf = rope_sub("full_attention", "partial_rotary_factor", 0.25);
    let full_theta = rope_sub("full_attention", "rope_theta", 1_000_000.0);
    let swa_theta = rope_sub("sliding_attention", "rope_theta", 10_000.0);

    let b = |k: &str| tc.get(k).and_then(|v| v.as_bool()).unwrap_or(false);
    let act = tc
        .get("hidden_activation")
        .and_then(|v| v.as_str())
        .unwrap_or("gelu_pytorch_tanh")
        .to_string();

    Ok(ArchConfig {
        hidden_size: i("hidden_size")?,
        intermediate_size: i("intermediate_size")?,
        moe_intermediate_size: i("moe_intermediate_size")?,
        num_heads: i("num_attention_heads")?,
        num_kv_heads: i("num_key_value_heads")?,
        num_full_kv_heads: i("num_global_key_value_heads")?,
        head_dim: i("head_dim")?,
        full_head_dim: i("global_head_dim")?,
        vocab_size: i("vocab_size")?,
        sliding_window: i("sliding_window")?,
        final_logit_softcap: d("final_logit_softcapping")?,
        rope_theta: swa_theta,
        full_rope_theta: full_theta,
        partial_rotary_factor: prf,
        num_layers: i("num_hidden_layers")?,
        num_experts: i("num_experts")?,
        top_k_experts: i("top_k_experts")?,
        tie_word_embeddings: b("tie_word_embeddings"),
        attention_k_eq_v: b("attention_k_eq_v"),
        full_attention_layer_mask: mask,
        hidden_activation: act,
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
    })
}

/// Per-tensor quantization widths, parsed from the checkpoint's own
/// `config.json -> quantization` object (MLX convention: a global
/// `group_size`/`bits` pair plus per-tensor override entries keyed by the
/// tensor base path — an object with its own `bits`, or `false` for
/// tensors left unquantized, which pass-through handles by dtype anyway).
#[derive(Debug, Clone)]
pub struct Gemma4Quant {
    pub default_bits: u32,
    pub group_size: u32,
    pub bits_overrides: std::collections::HashMap<String, u32>,
}

impl Default for Gemma4Quant {
    /// 4-bit, 64-element groups, no overrides: the base mode this port's
    /// GEMV kernels assume.
    fn default() -> Self {
        Gemma4Quant {
            default_bits: 4,
            group_size: 64,
            bits_overrides: std::collections::HashMap::new(),
        }
    }
}

impl Gemma4Quant {
    fn bits_for(&self, base_name: &str) -> u32 {
        // Overrides are keyed by the source path without the `.weight`
        // suffix; try both to be safe.
        self.bits_overrides
            .get(base_name)
            .copied()
            .unwrap_or(self.default_bits)
    }
}

/// Parses `config.json -> quantization` into a [`Gemma4Quant`]. A missing
/// `quantization` object yields the 4-bit/group-64 default.
pub fn parse_gemma4_quantization(json: &str) -> Result<Gemma4Quant, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    let Some(q) = root.get("quantization").and_then(|v| v.as_object()) else {
        return Ok(Gemma4Quant::default());
    };
    let default_bits = q.get("bits").and_then(|v| v.as_u64()).unwrap_or(4) as u32;
    let group_size = q.get("group_size").and_then(|v| v.as_u64()).unwrap_or(64) as u32;
    let mut bits_overrides = std::collections::HashMap::new();
    for (key, value) in q {
        if key == "bits" || key == "group_size" || key == "mode" {
            continue;
        }
        if let Some(obj) = value.as_object() {
            if let Some(bits) = obj.get("bits").and_then(|v| v.as_u64()) {
                bits_overrides.insert(key.clone(), bits as u32);
            }
            if let Some(gs) = obj.get("group_size").and_then(|v| v.as_u64()) {
                if gs as u32 != group_size {
                    return Err(Gemma4Error::Config(format!(
                        "per-tensor group_size {gs} for {key} differs from the \
                         global {group_size}; this port's kernels assume one group size"
                    )));
                }
            }
        }
    }
    if group_size != 64 {
        return Err(Gemma4Error::Config(format!(
            "group_size {group_size} unsupported: this port's GEMV kernels assume 64"
        )));
    }
    Ok(Gemma4Quant {
        default_bits,
        group_size,
        bits_overrides,
    })
}

// ---------------------------------------------------------------------------
// Tensor-name classification and ordering
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gemma4Bucket {
    LmResident,
    RoutedExpert { role: &'static str, layer: usize },
    ExcludedMultimodal,
    Unknown,
}

fn layer_index(name: &str) -> Option<usize> {
    let tail = &name[name.find(".layers.")? + ".layers.".len()..];
    tail[..tail.find('.')?].parse().ok()
}

/// The container path a family's routed (per-expert) weights live under.
/// Everything else about the walk is family-agnostic, so this substring
/// plus `ArchConfig.family` is the whole of the Qwen 3.6 delta here.
///
/// Getting this wrong is silent and expensive: an unrecognized routed
/// marker makes every expert a RESIDENT tensor, which loads and runs but
/// blows the footprint up by the entire expert table. `tests/synthetic_qwen.rs`
/// pins both families' markers for that reason.
fn routed_marker(family: ModelFamily) -> &'static str {
    match family {
        ModelFamily::Qwen36 => ".mlp.switch_mlp.",
        // DeepSeek V4 has no repack path yet; Gemma's marker is the default.
        ModelFamily::Gemma4 | ModelFamily::DeepseekV4Flash => ".experts.switch_glu.",
    }
}

/// [`classify_gemma4`] for any family: same `language_model.` text-tower
/// contract and the same multimodal exclusions, with the routed-expert
/// container taken from [`routed_marker`].
pub fn classify_for_family(name: &str, num_layers: usize, family: ModelFamily) -> Gemma4Bucket {
    if name.starts_with("language_model.") {
        if name.contains(routed_marker(family)) {
            let role = if name.contains(".gate_proj.") {
                Some("gate")
            } else if name.contains(".up_proj.") {
                Some("up")
            } else if name.contains(".down_proj.") {
                Some("down")
            } else {
                None
            };
            if let (Some(role), Some(layer)) = (role, layer_index(name)) {
                if layer < num_layers {
                    return Gemma4Bucket::RoutedExpert { role, layer };
                }
            }
        }
        return Gemma4Bucket::LmResident;
    }
    if name.starts_with("vision_tower.")
        || name.starts_with("embed_vision.")
        || name.starts_with("audio_tower.")
    {
        return Gemma4Bucket::ExcludedMultimodal;
    }
    Gemma4Bucket::Unknown
}

/// Classify a source tensor name under the Gemma 4 family contract: the
/// text tower lives under `language_model.`, routed experts under
/// `.experts.switch_glu.`, and the vision/audio towers are excluded.
pub fn classify_gemma4(name: &str, num_layers: usize) -> Gemma4Bucket {
    classify_for_family(name, num_layers, ModelFamily::Gemma4)
}

/// Within-layer slot order, mirroring `RepackPlanner.swift`'s Gemma table.
fn slot_rank(n: &str) -> usize {
    const CONTAINS: [&str; 12] = [
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
        ".self_attn.o_proj.weight",
        ".self_attn.q_norm.weight",
        ".self_attn.k_norm.weight",
        ".router.proj.weight",
        ".router.scale",
        ".router.per_expert_scale",
        ".mlp.gate_proj.weight",
        ".mlp.up_proj.weight",
        ".mlp.down_proj.weight",
    ];
    for (rank, pat) in CONTAINS.iter().enumerate() {
        if n.contains(pat) {
            return rank;
        }
    }
    const SUFFIXES: [&str; 8] = [
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".pre_feedforward_layernorm.weight",
        ".pre_feedforward_layernorm_2.weight",
        ".post_feedforward_layernorm.weight",
        ".post_feedforward_layernorm_1.weight",
        ".post_feedforward_layernorm_2.weight",
        ".layer_scalar",
    ];
    for (j, pat) in SUFFIXES.iter().enumerate() {
        if n.ends_with(pat) {
            return CONTAINS.len() + j;
        }
    }
    // Names from other families (Qwen's `linear_attn.*`, `mlp.shared_expert.*`)
    // land here and tie at 100, broken by name in `lm_order_key`. Resident
    // order is locality only -- the index is name-keyed and every entry is
    // independently 4-byte aligned -- so a family-specific table buys nothing
    // that is worth another 20 lines of string matching.
    100
}

/// Stable resident order: embedding first, then per-layer groups in layer
/// order (slot-ranked within a layer), then top-level extras, the final
/// norm, and `lm_head` last.
fn lm_order_key(n: &str) -> (usize, usize, usize, &str) {
    match n {
        "language_model.model.embed_tokens.weight" => (0, 0, 0, n),
        "language_model.model.norm.weight" => (3, 0, 0, n),
        "language_model.lm_head.weight" => (4, 0, 0, n),
        _ => match layer_index(n) {
            Some(li) => (1, li, slot_rank(n), n),
            None => (2, 0, 0, n),
        },
    }
}

// ---------------------------------------------------------------------------
// Orchestration
// ---------------------------------------------------------------------------

/// Everything the writer needs for a full Gemma 4 `.gturbo` install:
/// ordered resident entries (pass-through quantized + raw), per-layer
/// routed-expert blobs, and the one model-wide page-rounded expert stride
/// (0 when the checkpoint has no routed experts).
pub struct Gemma4RepackOutput {
    pub resident: Vec<ResidentEntrySpec>,
    pub layers: Vec<LayerBlobs>,
    pub expert_stride: u64,
    pub excluded_multimodal: Vec<String>,
}

/// A multi-shard checkpoint view: real HF checkpoints split their tensors
/// across several `model-NNNNN-of-NNNNN.safetensors` files (the
/// `model.safetensors.index.json` weight map), and companion tensors may
/// live in a different shard than their weight — so lookups go through one
/// merged name registry, exactly like the Swift planner's `registry`.
pub struct Gemma4Shards<'a> {
    shards: Vec<(&'a SafetensorsHeader, &'a dyn RangeSource)>,
    by_name: std::collections::HashMap<&'a str, usize>,
}

impl<'a> Gemma4Shards<'a> {
    pub fn new(shards: Vec<(&'a SafetensorsHeader, &'a dyn RangeSource)>) -> Self {
        let mut by_name = std::collections::HashMap::new();
        for (i, (header, _)) in shards.iter().enumerate() {
            for name in header.tensors.keys() {
                by_name.insert(name.as_str(), i);
            }
        }
        Self { shards, by_name }
    }

    pub fn single(header: &'a SafetensorsHeader, source: &'a dyn RangeSource) -> Self {
        Self::new(vec![(header, source)])
    }

    fn shard_of(
        &self,
        name: &str,
    ) -> Result<&(&'a SafetensorsHeader, &'a dyn RangeSource), Gemma4Error> {
        let i = *self
            .by_name
            .get(name)
            .ok_or_else(|| Gemma4Error::MissingTensor(name.to_string()))?;
        Ok(&self.shards[i])
    }

    fn info(&self, name: &str) -> Result<&'a TensorInfo, Gemma4Error> {
        let (header, _) = self.shard_of(name)?;
        header
            .tensors
            .get(name)
            .ok_or_else(|| Gemma4Error::MissingTensor(name.to_string()))
    }

    fn contains(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, Gemma4Error> {
        let (header, source) = self.shard_of(name)?;
        let (start, end) = header
            .absolute_range(name)
            .ok_or_else(|| Gemma4Error::MissingTensor(name.to_string()))?;
        Ok(source.read_range(start, end)?)
    }

    fn names(&self) -> impl Iterator<Item = &'a String> + '_ {
        self.shards.iter().flat_map(|(h, _)| h.tensors.keys())
    }
}

fn raw_dtype_tag(tensor: &str, dtype: &str) -> Result<u8, Gemma4Error> {
    match dtype {
        "BF16" => Ok(DTYPE_BF16),
        "F16" => Ok(DTYPE_FP16),
        "F32" => Ok(DTYPE_FP32),
        other => Err(Gemma4Error::UnsupportedDtype {
            tensor: tensor.to_string(),
            dtype: other.to_string(),
        }),
    }
}

fn shape4(shape: &[u64]) -> (u32, u32, u32, u32) {
    let get = |i: usize| shape.get(i).copied().unwrap_or(0) as u32;
    (get(0), get(1), get(2), get(3))
}

/// One quantized tensor's u32 weight bytes + BF16 companions, passed
/// through verbatim, tagged INT4 or INT8 per the checkpoint's quantization
/// spec. The MLX affine layout viewed as LE bytes is exactly this port's
/// packed layout: for 4-bit, value `i` of a u32 word occupies bits
/// `[4i, 4i+4)`, so byte `k` holds values `2k` (low nibble) and `2k+1`
/// (high nibble); for 8-bit, byte `k` is value `k`. Scale/bias sizes are
/// validated against the 64-element group the GPU kernels assume.
fn pass_through_packed(
    shards: &Gemma4Shards<'_>,
    name: &str,
    quant: &Gemma4Quant,
) -> Result<ResidentEntrySpec, Gemma4Error> {
    let w = shards.info(name)?;
    if w.shape.len() != 2 {
        return Err(Gemma4Error::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!("expected rank-2 packed weight, got {:?}", w.shape),
        });
    }
    let base = name.strip_suffix(".weight").unwrap_or(name);
    let bits = quant.bits_for(base);
    let factor = match bits {
        4 | 8 => 32 / bits as u64,
        other => {
            return Err(Gemma4Error::UnsupportedDtype {
                tensor: name.to_string(),
                dtype: format!("{other}-bit quantization"),
            })
        }
    };
    let scales_name = format!("{base}.scales");
    let biases_name = format!("{base}.biases");
    for companion in [&scales_name, &biases_name] {
        if !shards.contains(companion) {
            return Err(Gemma4Error::MissingCompanion(name.to_string()));
        }
        let c = shards.info(companion)?;
        if c.dtype != "BF16" {
            return Err(Gemma4Error::UnsupportedDtype {
                tensor: companion.to_string(),
                dtype: c.dtype.clone(),
            });
        }
    }
    let rows = w.shape[0];
    let cols = w.shape[1] * factor;
    let packed = shards.read(name)?;
    let scales = le_u16(&shards.read(&scales_name)?);
    let biases = le_u16(&shards.read(&biases_name)?);
    let expected_groups = (rows * cols / 64) as usize;
    if cols % 64 != 0 || scales.len() != expected_groups || biases.len() != expected_groups {
        return Err(Gemma4Error::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!(
                "shape {rows}x{cols} with {} scales / {} biases does not match \
                 the 64-element groups this port's kernels assume",
                scales.len(),
                biases.len()
            ),
        });
    }
    let spec = ResidentTensorSpec {
        name: name.to_string(),
        packed,
        scales,
        biases,
        rows: rows as u32,
        cols: cols as u32,
    };
    Ok(match bits {
        4 => ResidentEntrySpec::Int4(spec),
        _ => ResidentEntrySpec::Int8(spec),
    })
}

fn le_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Walks a Gemma 4 checkpoint's tensors: classifies every name, orders and
/// reads the resident LM set (pass-through for `U32` `.weight` tensors with
/// their BF16 companions, raw bytes for everything else), and slices each
/// layer's routed-expert `gate/up/down` bundles into per-expert blobs.
pub fn orchestrate_gemma4_checkpoint(
    header: &SafetensorsHeader,
    source: &dyn RangeSource,
    arch: &ArchConfig,
    quant: &Gemma4Quant,
) -> Result<Gemma4RepackOutput, Gemma4Error> {
    orchestrate_gemma4_checkpoint_sharded(&Gemma4Shards::single(header, source), arch, quant)
}

/// Multi-shard variant of [`orchestrate_gemma4_checkpoint`] — what a real
/// (three-shard) checkpoint goes through.
pub fn orchestrate_gemma4_checkpoint_sharded(
    shards: &Gemma4Shards<'_>,
    arch: &ArchConfig,
    quant: &Gemma4Quant,
) -> Result<Gemma4RepackOutput, Gemma4Error> {
    let plan = classify_all(shards, arch)?;
    let resident = read_resident_entries(shards, &plan.resident_bases, quant)?;
    let expert_stride = expert_stride_from_headers(shards, arch, quant, &plan.routed)?;
    let mut layers = Vec::new();
    if !plan.routed.is_empty() {
        for layer in 0..arch.num_layers as usize {
            let (blobs, _used) = plan_one_expert_layer(shards, arch, quant, &plan.routed, layer)?;
            layers.push(blobs);
        }
    }
    Ok(Gemma4RepackOutput {
        resident,
        layers,
        expert_stride,
        excluded_multimodal: plan.excluded,
    })
}

struct ClassifiedNames<'a> {
    resident_bases: Vec<&'a str>,
    routed: BTreeMap<usize, BTreeMap<&'static str, &'a str>>,
    excluded: Vec<String>,
}

fn classify_all<'a>(
    shards: &Gemma4Shards<'a>,
    arch: &ArchConfig,
) -> Result<ClassifiedNames<'a>, Gemma4Error> {
    let num_layers = arch.num_layers as usize;
    let mut resident_bases: Vec<&str> = Vec::new();
    let mut excluded: Vec<String> = Vec::new();
    let mut routed: BTreeMap<usize, BTreeMap<&'static str, &str>> = BTreeMap::new();

    for name in shards.names() {
        if name.ends_with(".scales") || name.ends_with(".biases") {
            continue;
        }
        match classify_for_family(name, num_layers, arch.family) {
            Gemma4Bucket::LmResident => resident_bases.push(name),
            Gemma4Bucket::RoutedExpert { role, layer } => {
                if routed
                    .entry(layer)
                    .or_default()
                    .insert(role, name.as_str())
                    .is_some()
                {
                    return Err(Gemma4Error::ShapeMismatch {
                        tensor: name.to_string(),
                        detail: format!("two routed-expert tensors for layer {layer} role {role}"),
                    });
                }
            }
            Gemma4Bucket::ExcludedMultimodal => excluded.push(name.clone()),
            Gemma4Bucket::Unknown => return Err(Gemma4Error::UnknownTensor(name.clone())),
        }
    }
    resident_bases.sort_by(|a, b| lm_order_key(a).cmp(&lm_order_key(b)));
    excluded.sort();
    Ok(ClassifiedNames {
        resident_bases,
        routed,
        excluded,
    })
}

fn read_resident_entries(
    shards: &Gemma4Shards<'_>,
    resident_bases: &[&str],
    quant: &Gemma4Quant,
) -> Result<Vec<ResidentEntrySpec>, Gemma4Error> {
    let mut resident = Vec::with_capacity(resident_bases.len());
    for &name in resident_bases {
        let t = shards.info(name)?;
        if t.dtype == "U32" && name.ends_with(".weight") {
            resident.push(pass_through_packed(shards, name, quant)?);
        } else {
            resident.push(ResidentEntrySpec::Raw(RawTensorSpec {
                name: name.to_string(),
                dtype: raw_dtype_tag(name, &t.dtype)?,
                bytes: shards.read(name)?,
                shape: shape4(&t.shape),
            }));
        }
    }
    Ok(resident)
}

/// The one model-wide expert stride, computed from shard HEADERS alone
/// (per-expert weight+scales+biases byte totals, max across layers,
/// rounded to 16 KiB) — so a streaming writer knows the stride before any
/// expert byte downloads.
fn expert_stride_from_headers(
    shards: &Gemma4Shards<'_>,
    arch: &ArchConfig,
    quant: &Gemma4Quant,
    routed: &BTreeMap<usize, BTreeMap<&'static str, &str>>,
) -> Result<u64, Gemma4Error> {
    if routed.is_empty() {
        return Ok(0);
    }
    let expert_count = arch.num_experts as u64;
    let mut max_blob = 0u64;
    for layer in 0..arch.num_layers as usize {
        let bundle = routed.get(&layer).ok_or_else(|| {
            Gemma4Error::MissingTensor(format!("layer {layer} routed-expert bundle"))
        })?;
        let mut blob = 0u64;
        for role in ["gate", "up", "down"] {
            let name = *bundle
                .get(role)
                .ok_or_else(|| Gemma4Error::MissingTensor(format!("layer {layer} {role}_proj")))?;
            let base = name.strip_suffix(".weight").unwrap_or(name);
            let bits = quant.bits_for(base);
            if bits != 4 {
                return Err(Gemma4Error::UnsupportedDtype {
                    tensor: name.to_string(),
                    dtype: format!("{bits}-bit routed expert (kernels are int4-only)"),
                });
            }
            for suffix in ["", ".scales", ".biases"] {
                let full = if suffix.is_empty() {
                    name.to_string()
                } else {
                    format!("{base}{suffix}")
                };
                let t = shards.info(&full)?;
                let bytes = t.data_offsets.1 - t.data_offsets.0;
                if bytes % expert_count != 0 {
                    return Err(Gemma4Error::ShapeMismatch {
                        tensor: full,
                        detail: format!("bytes not divisible by {expert_count} experts"),
                    });
                }
                blob += bytes / expert_count;
            }
        }
        max_blob = max_blob.max(blob);
    }
    Ok(max_blob.div_ceil(GTURBO_PAGE_BYTES) * GTURBO_PAGE_BYTES)
}

/// Builds one routed layer's per-expert blobs. Blob layout matches the
/// synthetic MoE installs (and `RealForwardRunner`'s `MoeExpertOffsets`):
/// `gate, gate_scales, gate_biases, up, ..., down, ...` back to back. The
/// down projection's blob offset must land 4-byte aligned (the phase-2
/// kernel reads its weights with `uint` loads); u32 weights and even-sized
/// BF16 companions keep that true for any real shape, and we verify it.
/// Returns the blobs plus the per-expert bytes used (callers pad to the
/// model-wide stride).
fn plan_one_expert_layer(
    shards: &Gemma4Shards<'_>,
    arch: &ArchConfig,
    quant: &Gemma4Quant,
    routed: &BTreeMap<usize, BTreeMap<&'static str, &str>>,
    layer: usize,
) -> Result<(LayerBlobs, u64), Gemma4Error> {
    let expert_count = arch.num_experts as usize;
    {
        let bundle = routed.get(&layer).ok_or_else(|| {
            Gemma4Error::MissingTensor(format!("layer {layer} routed-expert bundle"))
        })?;
        let mut experts: Vec<ExpertBlob> = (0..expert_count)
            .map(|e| ExpertBlob {
                expert: e,
                sub_tensors: Vec::with_capacity(9),
            })
            .collect();
        let mut blob_used = 0u64;

        for role in ["gate", "up", "down"] {
            let name = *bundle
                .get(role)
                .ok_or_else(|| Gemma4Error::MissingTensor(format!("layer {layer} {role}_proj")))?;
            let w = shards.info(name)?;
            if w.dtype != "U32" || w.shape.len() != 3 || w.shape[0] as usize != expert_count {
                return Err(Gemma4Error::ShapeMismatch {
                    tensor: name.to_string(),
                    detail: format!(
                        "expected U32 rank-3 with leading {expert_count}, got {} {:?}",
                        w.dtype, w.shape
                    ),
                });
            }
            let base = name.strip_suffix(".weight").unwrap_or(name);
            // Bits already validated by expert_stride_from_headers, but
            // this function is also reachable on its own.
            let bits = quant.bits_for(base);
            if bits != 4 {
                return Err(Gemma4Error::UnsupportedDtype {
                    tensor: name.to_string(),
                    dtype: format!("{bits}-bit routed expert (kernels are int4-only)"),
                });
            }
            let s_name = format!("{base}.scales");
            let b_name = format!("{base}.biases");
            let s = shards.info(&s_name)?;
            let b = shards.info(&b_name)?;
            if s.dtype != "BF16" || b.dtype != "BF16" {
                return Err(Gemma4Error::UnsupportedDtype {
                    tensor: name.to_string(),
                    dtype: format!("{}/{} companions", s.dtype, b.dtype),
                });
            }

            let w_bytes = shards.read(name)?;
            let s_bytes = shards.read(&s_name)?;
            let b_bytes = shards.read(&b_name)?;
            let per = |total: usize, what: &str| -> Result<usize, Gemma4Error> {
                if total % expert_count != 0 {
                    return Err(Gemma4Error::ShapeMismatch {
                        tensor: name.to_string(),
                        detail: format!("{what} bytes not divisible by {expert_count} experts"),
                    });
                }
                Ok(total / expert_count)
            };
            let w_per = per(w_bytes.len(), "weight")?;
            let s_per = per(s_bytes.len(), "scales")?;
            let b_per = per(b_bytes.len(), "biases")?;

            // Per-expert logical shape: [rows, cols_packed * 8].
            let rows = w.shape[1];
            let cols = w.shape[2] * 8;
            let comp_shape = |t: &TensorInfo| t.shape[1..].to_vec();
            if role == "down" && blob_used % 4 != 0 {
                return Err(Gemma4Error::ShapeMismatch {
                    tensor: name.to_string(),
                    detail: format!("down offset {blob_used} not 4-byte aligned"),
                });
            }
            for (e, blob) in experts.iter_mut().enumerate() {
                for (suffix, bytes, dtype, shape) in [
                    (
                        "",
                        w_bytes[e * w_per..(e + 1) * w_per].to_vec(),
                        "u32",
                        vec![rows, cols],
                    ),
                    (
                        "_scales",
                        s_bytes[e * s_per..(e + 1) * s_per].to_vec(),
                        "bf16",
                        comp_shape(s),
                    ),
                    (
                        "_biases",
                        b_bytes[e * b_per..(e + 1) * b_per].to_vec(),
                        "bf16",
                        comp_shape(b),
                    ),
                ] {
                    blob.sub_tensors.push(SubTensor {
                        role: format!("{role}{suffix}"),
                        bytes,
                        dtype: dtype.to_string(),
                        shape,
                    });
                }
            }
            blob_used += (w_per + s_per + b_per) as u64;
        }
        Ok((LayerBlobs { layer, experts }, blob_used))
    }
}

/// The `manifest.json -> quant` object for a Gemma 4 install, derived
/// from the checkpoint's own per-tensor bits (slot bits are read from the
/// layer-0 base names; every slot is affine/BF16/group-64 in this format).
/// Production-shape manifests are rejected by `mrefrust_model_io` without
/// this object.
pub fn gemma4_manifest_quant(quant: &Gemma4Quant) -> serde_json::Value {
    manifest_quant(quant, ModelFamily::Gemma4)
}

/// [`gemma4_manifest_quant`] for any family. The probe names are the only
/// difference: Qwen 3.6's layer 0 is a LINEAR layer, so its "attention"
/// slot has to be probed at `linear_attn.in_proj_qkv` -- there is no
/// `self_attn.q_proj` under layer 0 at all.
pub fn manifest_quant(quant: &Gemma4Quant, family: ModelFamily) -> serde_json::Value {
    let slot = |bits: u32| {
        serde_json::json!({
            "weightBits": bits,
            "scheme": "affine",
            "scaleType": "bf16",
            "biasType": "bf16",
            "groupSize": 64,
        })
    };
    let l0 = "language_model.model.layers.0";
    let (attention, router, shared, routed) = match family {
        ModelFamily::Qwen36 => (
            format!("{l0}.linear_attn.in_proj_qkv"),
            format!("{l0}.mlp.gate"),
            format!("{l0}.mlp.shared_expert.gate_proj"),
            format!("{l0}.mlp.switch_mlp.gate_proj"),
        ),
        ModelFamily::Gemma4 | ModelFamily::DeepseekV4Flash => (
            format!("{l0}.self_attn.q_proj"),
            format!("{l0}.router.proj"),
            format!("{l0}.mlp.gate_proj"),
            format!("{l0}.experts.switch_glu.gate_proj"),
        ),
    };
    serde_json::json!({
        "embedding": slot(quant.bits_for("language_model.model.embed_tokens")),
        "attention": slot(quant.bits_for(&attention)),
        "router": slot(quant.bits_for(&router)),
        "sharedExpert": slot(quant.bits_for(&shared)),
        "routedExpert": slot(quant.bits_for(&routed)),
    })
}

/// Streamed install write for real (multi-GB) checkpoints: the expert
/// stride comes from shard headers alone, the resident set is read and
/// written first, then each layer's expert blobs download, hit disk, and
/// drop before the next layer starts — peak memory is one layer's blobs,
/// not thirty. `progress` gets one call per completed stage.
pub fn write_gemma4_install_streamed(
    dir: &std::path::Path,
    arch: &ArchConfig,
    model_id: &str,
    shards: &Gemma4Shards<'_>,
    quant: &Gemma4Quant,
    mut progress: impl FnMut(&str),
) -> Result<(), Box<dyn std::error::Error>> {
    let plan = classify_all(shards, arch)?;
    let expert_stride = expert_stride_from_headers(shards, arch, quant, &plan.routed)?;
    progress(&format!(
        "classified {} resident tensors, {} routed layers, expert stride {expert_stride}",
        plan.resident_bases.len(),
        plan.routed.len(),
    ));

    let resident = read_resident_entries(shards, &plan.resident_bases, quant)?;
    let resident_bytes = crate::resident_writer::build_resident_weights_bin_mixed(&resident);
    drop(resident);
    progress(&format!(
        "resident region built ({} bytes)",
        resident_bytes.len()
    ));

    if plan.routed.is_empty() {
        crate::gturbo_writer::write_gturbo_install_with_resident_index(
            dir,
            arch,
            model_id,
            &resident_bytes,
        )?;
        progress("install written (no routed experts)");
        return Ok(());
    }

    let mut writer = crate::gturbo_writer::StreamingGturboWriter::new(
        dir,
        expert_stride,
        arch.num_experts as usize,
    )?;
    writer.set_quant(manifest_quant(quant, arch.family));
    for layer in 0..arch.num_layers as usize {
        let (blobs, used) = plan_one_expert_layer(shards, arch, quant, &plan.routed, layer)?;
        writer.write_layer(&blobs)?;
        progress(&format!(
            "layer {layer} written ({} experts, {used} bytes/expert)",
            blobs.experts.len()
        ));
    }
    writer.finish(arch, model_id, &resident_bytes)?;
    progress("manifest written");
    Ok(())
}

/// Convenience: orchestrate + build the resident index + write the full
/// install directory in one call.
pub fn write_gemma4_install(
    dir: &std::path::Path,
    arch: &ArchConfig,
    model_id: &str,
    header: &SafetensorsHeader,
    source: &dyn RangeSource,
    quant: &Gemma4Quant,
) -> Result<Gemma4RepackOutput, Box<dyn std::error::Error>> {
    let out = orchestrate_gemma4_checkpoint(header, source, arch, quant)?;
    let resident_bytes = crate::resident_writer::build_resident_weights_bin_mixed(&out.resident);
    if out.layers.is_empty() {
        crate::gturbo_writer::write_gturbo_install_with_resident_index(
            dir,
            arch,
            model_id,
            &resident_bytes,
        )?;
    } else {
        // Through the same streaming writer the real-checkpoint path uses,
        // so the two stay byte-identical (including the manifest's quant
        // object, which production-shape loads require).
        let mut writer = crate::gturbo_writer::StreamingGturboWriter::new(
            dir,
            out.expert_stride,
            arch.num_experts as usize,
        )?;
        writer.set_quant(manifest_quant(quant, arch.family));
        for layer in &out.layers {
            writer.write_layer(layer)?;
        }
        writer.finish(arch, model_id, &resident_bytes)?;
    }
    Ok(out)
}

/// Qwen 3.6 install write. The walk itself is family-agnostic (see
/// [`classify_for_family`] and [`manifest_quant`]), so this is
/// [`write_gemma4_install`] plus the guard that `arch.family` actually says
/// Qwen -- passing a Gemma arch here would silently classify
/// `.mlp.switch_mlp.` tensors as resident.
pub fn write_qwen36_install(
    dir: &std::path::Path,
    arch: &ArchConfig,
    model_id: &str,
    header: &SafetensorsHeader,
    source: &dyn RangeSource,
    quant: &Gemma4Quant,
) -> Result<Gemma4RepackOutput, Box<dyn std::error::Error>> {
    if arch.family != ModelFamily::Qwen36 {
        return Err(Box::new(Gemma4Error::Config(format!(
            "write_qwen36_install needs arch.family = qwen36, got {}",
            arch.family.as_str()
        ))));
    }
    write_gemma4_install(dir, arch, model_id, header, source, quant)
}

/// [`write_qwen36_install`] for a real multi-GB checkpoint: the same family
/// guard in front of [`write_gemma4_install_streamed`].
///
/// The guard matters more here than on the in-memory path, because this is
/// the one the ~20 GB repack actually runs: a Gemma `arch.family` would
/// classify every `.mlp.switch_mlp.` tensor as RESIDENT, and the failure
/// is a footprint blowup at load time, not an error here.
pub fn write_qwen36_install_streamed(
    dir: &std::path::Path,
    arch: &ArchConfig,
    model_id: &str,
    shards: &Gemma4Shards<'_>,
    quant: &Gemma4Quant,
    progress: impl FnMut(&str),
) -> Result<(), Box<dyn std::error::Error>> {
    if arch.family != ModelFamily::Qwen36 {
        return Err(Box::new(Gemma4Error::Config(format!(
            "write_qwen36_install_streamed needs arch.family = qwen36, got {}",
            arch.family.as_str()
        ))));
    }
    write_gemma4_install_streamed(dir, arch, model_id, shards, quant, progress)
}
