//! The GGUF repack walk: a GGUF file in, a `.gturbo` install out.
//!
//! Sibling of [`crate::gemma4_checkpoint`]'s walk over an mlx-community
//! safetensors checkpoint, and it reuses that walk's output types
//! (`ResidentEntrySpec`, `LayerBlobs`, `StreamingGturboWriter`) rather than
//! inventing parallel ones. What it does NOT reuse is the quantization step,
//! because there isn't one: GGUF bytes arrive already quantized and are
//! copied through untouched, which is the roadmap's lossless-repack rule
//! taken literally.
//!
//! ## What makes this walk different from the safetensors one
//!
//! **The quant layout is block-interleaved, not planar.** MLX affine stores
//! three separate regions per tensor (nibbles, BF16 scales, BF16 biases) at
//! group 64. A Q8_0 block is 34 self-contained bytes: an f16 scale followed
//! by its own 32 weights. So a GGUF tensor becomes ONE sub-tensor with no
//! `_scales`/`_biases` companions, where the safetensors walk emits three.
//!
//! **Gemma 4 fuses gate and up into a single routed tensor.** llama.cpp's
//! converter writes `ffn_gate_up_exps` as `[hidden, 2 * ffn, experts]` where
//! MLX keeps `gate_proj` and `up_proj` apart. Splitting it is byte-exact
//! rather than arithmetic -- blocks tile along the fastest-varying dim, so
//! whole output rows are contiguous byte ranges -- but the ORDER of the two
//! halves is an assumption this stage cannot verify. See
//! [`FUSED_GATE_FIRST`].
//!
//! **The install this produces does not load yet, on purpose.** Its manifest
//! declares `scheme: "gguf"` and `mrefrust_model_io::validate_quant` accepts
//! only `"affine"`, so `load_manifest` rejects it by name. That rejection IS
//! the Stage 1 boundary: the on-disk shape is settled and verifiable here,
//! while the kernels that could read these blocks are Stage 2 work. Flipping
//! one validation rule is what promotes the artifact, and nothing about the
//! bytes changes when that happens.

use std::collections::BTreeMap;
use std::path::Path;

use model_io::{ArchConfig, ModelFamily};

use crate::gguf_config::{arch_from_gguf, GgufConfigError};
use crate::gguf_header::{ggml_type_name, GgufHeader, GgufHeaderError};
use crate::gguf_names::{map_gguf_name, GgufMapping, GgufNameError};
use crate::gturbo_writer::{ExpertBlob, LayerBlobs, SubTensor, WriterError};
use crate::ranged_download::{DownloadError, RangeSource};
use crate::resident_writer::{RawTensorSpec, ResidentEntrySpec};

/// Which half of a fused `ffn_gate_up_exps` tensor is the gate.
///
/// **This is an assumption, not a measurement.** Both halves have identical
/// shapes, so nothing in the file distinguishes them, and Stage 1 has no
/// dequantization reference to decode them with. `true` follows the
/// prevailing convention (HF's own `gate_up_proj` fusion puts gate first),
/// and it is a named constant so that flipping it is a one-line change.
///
/// How to settle it in Stage 2, once a Q8_0 CPU reference exists: dequantize
/// expert 0 of layer 0 from the GGUF and compare against the same expert's
/// `gate_proj` in the MLX-derived install. They are different quantizations
/// of the same trained weights, so they will not match to the bit, but gate
/// and up are unrelated matrices and the correlation will be unambiguous.
pub const FUSED_GATE_FIRST: bool = true;

#[derive(Debug)]
pub enum GgufRepackError {
    Config(GgufConfigError),
    Header(GgufHeaderError),
    Name(GgufNameError),
    Download(DownloadError),
    Writer(WriterError),
    UnsupportedFamily {
        family: &'static str,
    },
    /// A ggml type with no `.gturbo` dtype tag.
    UnsupportedType {
        tensor: String,
        ggml_type: u32,
    },
    ShapeMismatch {
        tensor: String,
        detail: String,
    },
    MissingTensor {
        name: String,
    },
    Io {
        path: String,
        detail: String,
    },
}

impl std::fmt::Display for GgufRepackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GgufRepackError::Config(e) => write!(f, "{e}"),
            GgufRepackError::Header(e) => write!(f, "{e}"),
            GgufRepackError::Name(e) => write!(f, "{e}"),
            GgufRepackError::Download(e) => write!(f, "{e}"),
            GgufRepackError::Writer(e) => write!(f, "{e}"),
            GgufRepackError::UnsupportedFamily { family } => {
                write!(f, "no GGUF repack path for family {family}")
            }
            GgufRepackError::UnsupportedType { tensor, ggml_type } => write!(
                f,
                "tensor {tensor}: ggml type {} (id {ggml_type}) has no .gturbo dtype tag",
                ggml_type_name(*ggml_type).unwrap_or("?")
            ),
            GgufRepackError::ShapeMismatch { tensor, detail } => {
                write!(f, "tensor {tensor}: {detail}")
            }
            GgufRepackError::MissingTensor { name } => write!(f, "missing tensor {name}"),
            GgufRepackError::Io { path, detail } => write!(f, "{path}: {detail}"),
        }
    }
}

impl std::error::Error for GgufRepackError {}

macro_rules! from_error {
    ($($ty:ty => $variant:ident),* $(,)?) => {
        $(impl From<$ty> for GgufRepackError {
            fn from(e: $ty) -> Self {
                GgufRepackError::$variant(e)
            }
        })*
    };
}
from_error! {
    GgufConfigError => Config,
    GgufHeaderError => Header,
    GgufNameError => Name,
    DownloadError => Download,
    WriterError => Writer,
}

/// `.gturbo` resident-index dtype tag for a ggml type.
///
/// The GGUF block types get their own tags rather than being squeezed into
/// the affine ones: a consumer that mistook a Q8_0 block for an INT8-affine
/// row would read the f16 scale as two weights and be silently wrong.
pub fn dtype_tag_for_ggml_type(ggml_type: u32) -> Option<u8> {
    use crate::resident_writer::{
        DTYPE_BF16, DTYPE_FP16, DTYPE_FP32, DTYPE_GGUF_Q4_0, DTYPE_GGUF_Q4_K, DTYPE_GGUF_Q6_K,
        DTYPE_GGUF_Q8_0,
    };
    Some(match ggml_type {
        0 => DTYPE_FP32,
        1 => DTYPE_FP16,
        30 => DTYPE_BF16,
        2 => DTYPE_GGUF_Q4_0,
        8 => DTYPE_GGUF_Q8_0,
        12 => DTYPE_GGUF_Q4_K,
        14 => DTYPE_GGUF_Q6_K,
        _ => return None,
    })
}

/// The `manifest.json -> quant` slot name for a ggml type, for the manifest
/// this walk writes.
fn ggml_scheme_name(ggml_type: u32) -> &'static str {
    ggml_type_name(ggml_type).unwrap_or("unknown")
}

/// GGUF stores dims fastest-varying first; the resident index stores logical
/// shape. Reversing is the whole conversion, and getting it wrong produces
/// an index whose shapes are transposed but whose bytes are fine, which no
/// byte-level assertion would catch.
fn logical_shape(dims: &[u64]) -> (u32, u32, u32, u32) {
    let mut out = [0u32; 4];
    for (slot, d) in out.iter_mut().zip(dims.iter().rev()) {
        *slot = *d as u32;
    }
    (out[0], out[1], out[2], out[3])
}

fn read_tensor(
    header: &GgufHeader,
    source: &dyn RangeSource,
    name: &str,
) -> Result<Vec<u8>, GgufRepackError> {
    let (start, end) =
        header
            .absolute_range(name)
            .ok_or_else(|| GgufRepackError::MissingTensor {
                name: name.to_string(),
            })??;
    Ok(source.read_range(start, end)?)
}

/// One routed-expert source tensor, resolved but not yet read.
struct RoutedSource<'a> {
    /// GGUF tensor name.
    name: &'a str,
    /// Roles this tensor supplies, in blob order. Two for a fused gate/up.
    roles: Vec<&'static str>,
}

struct Plan<'a> {
    resident: Vec<&'a str>,
    /// layer -> role-bearing source tensors.
    routed: BTreeMap<usize, Vec<RoutedSource<'a>>>,
    ignored: Vec<String>,
}

fn classify<'a>(header: &'a GgufHeader, family: ModelFamily) -> Result<Plan<'a>, GgufRepackError> {
    let mut plan = Plan {
        resident: Vec::new(),
        routed: BTreeMap::new(),
        ignored: Vec::new(),
    };
    for name in header.tensors.keys() {
        match map_gguf_name(name, family)? {
            GgufMapping::Resident(_) => plan.resident.push(name.as_str()),
            GgufMapping::Routed { layer, role } => {
                plan.routed.entry(layer).or_default().push(RoutedSource {
                    name,
                    roles: vec![role],
                });
            }
            GgufMapping::RoutedFusedGateUp { layer } => {
                let roles = if FUSED_GATE_FIRST {
                    vec!["gate", "up"]
                } else {
                    vec!["up", "gate"]
                };
                plan.routed
                    .entry(layer)
                    .or_default()
                    .push(RoutedSource { name, roles });
            }
            GgufMapping::Ignored { reason } => {
                plan.ignored.push(format!("{name} ({reason})"));
            }
        }
    }
    // Blob order is gate, up, down, matching the safetensors walk and
    // `RealForwardRunner`'s `MoeExpertOffsets`. Sorting by role rather than
    // by source name keeps a fused tensor's two halves adjacent and ahead
    // of the down projection.
    const ORDER: [&str; 3] = ["gate", "up", "down"];
    let rank = |r: &str| ORDER.iter().position(|o| *o == r).unwrap_or(ORDER.len());
    for sources in plan.routed.values_mut() {
        sources.sort_by_key(|s| rank(s.roles[0]));
    }
    plan.resident.sort_unstable();
    Ok(plan)
}

/// Per-expert byte size of one routed source tensor, plus the number of
/// experts it carries. The expert index is GGUF's SLOWEST-varying dimension
/// (last, as stored), so each expert's bytes are one contiguous range.
fn per_expert_bytes(
    header: &GgufHeader,
    name: &str,
    num_experts: u64,
) -> Result<u64, GgufRepackError> {
    let info = header
        .tensors
        .get(name)
        .ok_or_else(|| GgufRepackError::MissingTensor {
            name: name.to_string(),
        })?;
    if info.dims.len() != 3 {
        return Err(GgufRepackError::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!("expected a rank-3 routed tensor, got {:?}", info.dims),
        });
    }
    if info.dims[2] != num_experts {
        return Err(GgufRepackError::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!(
                "trailing dim {} is not the expert count {num_experts}",
                info.dims[2]
            ),
        });
    }
    let total = info.byte_size(name)?;
    if total % num_experts != 0 {
        return Err(GgufRepackError::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!("{total} bytes is not divisible by {num_experts} experts"),
        });
    }
    Ok(total / num_experts)
}

/// The one model-wide expert stride, from the header alone, so a streaming
/// writer knows it before any expert byte is read.
fn expert_stride(
    header: &GgufHeader,
    arch: &ArchConfig,
    plan: &Plan<'_>,
) -> Result<u64, GgufRepackError> {
    if plan.routed.is_empty() {
        return Ok(0);
    }
    let experts = arch.num_experts as u64;
    let mut max_blob = 0u64;
    for sources in plan.routed.values() {
        let mut blob = 0u64;
        for s in sources {
            blob += per_expert_bytes(header, s.name, experts)?;
        }
        max_blob = max_blob.max(blob);
    }
    Ok(max_blob.div_ceil(crate::GTURBO_PAGE_BYTES) * crate::GTURBO_PAGE_BYTES)
}

/// Build one layer's per-expert blobs. Returns the blobs and the per-expert
/// bytes used (the caller pads to the model-wide stride).
fn plan_one_layer(
    header: &GgufHeader,
    source: &dyn RangeSource,
    arch: &ArchConfig,
    plan: &Plan<'_>,
    layer: usize,
) -> Result<(LayerBlobs, u64), GgufRepackError> {
    let experts = arch.num_experts as usize;
    let sources = plan
        .routed
        .get(&layer)
        .ok_or_else(|| GgufRepackError::MissingTensor {
            name: format!("layer {layer} routed experts"),
        })?;

    let mut blobs: Vec<ExpertBlob> = (0..experts)
        .map(|e| ExpertBlob {
            expert: e,
            sub_tensors: Vec::new(),
        })
        .collect();
    let mut used = 0u64;

    for s in sources {
        let info = &header.tensors[s.name];
        let per = per_expert_bytes(header, s.name, experts as u64)? as usize;
        let bytes = read_tensor(header, source, s.name)?;
        if bytes.len() != per * experts {
            return Err(GgufRepackError::ShapeMismatch {
                tensor: s.name.to_string(),
                detail: format!("read {} bytes, expected {}", bytes.len(), per * experts),
            });
        }

        // Split a fused tensor by whole output rows. Blocks tile along the
        // fastest-varying dim, so row r is exactly [r * row, (r + 1) * row)
        // -- but only if that dim is a whole number of blocks, which is
        // checked rather than assumed.
        let parts = s.roles.len();
        if per % parts != 0 {
            return Err(GgufRepackError::ShapeMismatch {
                tensor: s.name.to_string(),
                detail: format!("{per} bytes per expert does not split into {parts} roles"),
            });
        }
        let part_bytes = per / parts;
        // Logical [in, out_total] per expert; each role takes out_total/parts
        // output rows.
        let out_total = info.dims[1];
        if out_total % parts as u64 != 0 {
            return Err(GgufRepackError::ShapeMismatch {
                tensor: s.name.to_string(),
                detail: format!("output dim {out_total} does not split into {parts} roles"),
            });
        }
        let part_shape = vec![out_total / parts as u64, info.dims[0]];
        let dtype = ggml_scheme_name(info.ggml_type).to_lowercase();

        for (e, blob) in blobs.iter_mut().enumerate() {
            let expert = &bytes[e * per..(e + 1) * per];
            for (p, role) in s.roles.iter().enumerate() {
                blob.sub_tensors.push(SubTensor {
                    role: (*role).to_string(),
                    bytes: expert[p * part_bytes..(p + 1) * part_bytes].to_vec(),
                    dtype: dtype.clone(),
                    shape: part_shape.clone(),
                });
            }
        }
        used += per as u64;
    }

    Ok((
        LayerBlobs {
            layer,
            experts: blobs,
        },
        used,
    ))
}

fn resident_entries(
    header: &GgufHeader,
    source: &dyn RangeSource,
    family: ModelFamily,
    names: &[&str],
) -> Result<Vec<ResidentEntrySpec>, GgufRepackError> {
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let info = &header.tensors[*name];
        let dtype = dtype_tag_for_ggml_type(info.ggml_type).ok_or_else(|| {
            GgufRepackError::UnsupportedType {
                tensor: (*name).to_string(),
                ggml_type: info.ggml_type,
            }
        })?;
        let GgufMapping::Resident(canonical) = map_gguf_name(name, family)? else {
            continue;
        };
        out.push(ResidentEntrySpec::Raw(RawTensorSpec {
            name: canonical,
            dtype,
            bytes: read_tensor(header, source, name)?,
            shape: logical_shape(&info.dims),
        }));
    }
    Ok(out)
}

/// The `manifest.json -> quant` object for a GGUF-sourced install.
///
/// Deliberately NOT `"affine"`. `mrefrust_model_io::validate_quant` accepts
/// only that scheme, so this value is what makes `load_manifest` refuse a
/// Stage 1 install by name instead of loading one whose expert bytes no
/// kernel can read.
fn gguf_manifest_quant(header: &GgufHeader, plan: &Plan<'_>) -> serde_json::Value {
    let type_of = |name: &str| {
        header
            .tensors
            .get(name)
            .map(|i| ggml_scheme_name(i.ggml_type))
            .unwrap_or("absent")
    };
    let routed = plan
        .routed
        .values()
        .next()
        .and_then(|s| s.first())
        .map(|s| type_of(s.name))
        .unwrap_or("absent");
    let slot = |ggml: &str| {
        serde_json::json!({
            "weightBits": 0,
            "scheme": "gguf",
            "ggmlType": ggml,
            "scaleType": "inline",
            "biasType": "inline",
            "groupSize": 0,
        })
    };
    serde_json::json!({
        "embedding": slot(type_of("token_embd.weight")),
        "attention": slot(type_of("blk.0.attn_q.weight")),
        "router": slot(type_of("blk.0.ffn_gate_inp.weight")),
        "sharedExpert": slot(type_of("blk.0.ffn_gate.weight")),
        "routedExpert": slot(routed),
    })
}

/// Everything the writer needs, for callers that want to inspect the plan
/// before it hits disk.
pub struct GgufRepackOutput {
    pub arch: ArchConfig,
    pub resident: Vec<ResidentEntrySpec>,
    pub layers: Vec<LayerBlobs>,
    pub expert_stride: u64,
    /// Tensors recognized and deliberately not carried, with the reason.
    pub ignored: Vec<String>,
}

/// Plan a whole GGUF repack in memory. Fine for a fixture; use
/// [`write_gguf_install_streamed`] for a real multi-GB checkpoint.
pub fn orchestrate_gguf_checkpoint(
    header: &GgufHeader,
    source: &dyn RangeSource,
) -> Result<GgufRepackOutput, GgufRepackError> {
    let arch = arch_from_gguf(header)?;
    if arch.family == ModelFamily::DeepseekV4Flash {
        return Err(GgufRepackError::UnsupportedFamily {
            family: arch.family.as_str(),
        });
    }
    let plan = classify(header, arch.family)?;
    let stride = expert_stride(header, &arch, &plan)?;
    let resident = resident_entries(header, source, arch.family, &plan.resident)?;

    let mut layers = Vec::with_capacity(plan.routed.len());
    for layer in plan.routed.keys().copied() {
        layers.push(plan_one_layer(header, source, &arch, &plan, layer)?.0);
    }

    Ok(GgufRepackOutput {
        arch,
        resident,
        layers,
        expert_stride: stride,
        ignored: plan.ignored,
    })
}

/// Streamed install write: the stride comes from the header alone, the
/// resident set is read and written first, then one layer at a time, so a
/// 27 GB checkpoint never has to be materialized locally.
pub fn write_gguf_install_streamed(
    dir: &Path,
    header: &GgufHeader,
    source: &dyn RangeSource,
    model_id: &str,
    mut progress: impl FnMut(&str),
) -> Result<ArchConfig, GgufRepackError> {
    let arch = arch_from_gguf(header)?;
    if arch.family == ModelFamily::DeepseekV4Flash {
        return Err(GgufRepackError::UnsupportedFamily {
            family: arch.family.as_str(),
        });
    }
    let plan = classify(header, arch.family)?;
    let stride = expert_stride(header, &arch, &plan)?;
    progress(&format!(
        "classified {} resident tensors, {} routed layers, expert stride {stride}",
        plan.resident.len(),
        plan.routed.len()
    ));
    for note in &plan.ignored {
        progress(&format!("ignored {note}"));
    }

    let resident = resident_entries(header, source, arch.family, &plan.resident)?;
    let resident_bytes = crate::resident_writer::build_resident_weights_bin_mixed(&resident);
    drop(resident);
    progress(&format!(
        "resident region built ({} bytes)",
        resident_bytes.len()
    ));

    if plan.routed.is_empty() {
        crate::gturbo_writer::write_gturbo_install_with_resident_index(
            dir,
            &arch,
            model_id,
            &resident_bytes,
        )?;
        progress("install written (no routed experts)");
        return Ok(arch);
    }

    let mut writer =
        crate::gturbo_writer::StreamingGturboWriter::new(dir, stride, arch.num_experts as usize)?;
    writer.set_quant(gguf_manifest_quant(header, &plan));
    for layer in plan.routed.keys().copied() {
        let (blobs, used) = plan_one_layer(header, source, &arch, &plan, layer)?;
        writer.write_layer(&blobs)?;
        progress(&format!(
            "layer {layer} written ({} experts, {used} bytes/expert)",
            blobs.experts.len()
        ));
    }
    writer.finish(&arch, model_id, &resident_bytes)?;
    progress("manifest written");
    Ok(arch)
}
