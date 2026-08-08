//! The GGUF repack walk: a GGUF file in, a `.gturbo` install out.
//!
//! Sibling of [`crate::gemma4_checkpoint`]'s walk over an mlx-community
//! safetensors checkpoint, and it reuses that walk's output types
//! (`ResidentEntrySpec`, `LayerBlobs`, `StreamingGturboWriter`) rather than
//! inventing parallel ones.
//!
//! **Quantized bytes are copied through untouched**, which is the roadmap's
//! lossless-repack rule taken literally: every routed expert byte written to
//! disk is a byte the GGUF held. The one exception is the resident F32 core,
//! which is TRANSCODED here rather than carried (see [`transcode_f32`]):
//! llama.cpp ships norms and the router as F32 where this port's kernels
//! read BF16 and INT8-affine, and no F32 kernel exists or is planned.
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
//! declares `scheme: "gguf"` on every slot but the transcoded router, and
//! `turbospark_model_io::validate_quant` accepts only `"affine"`, so
//! `load_manifest` rejects it by name. That rejection IS
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
use crate::resident_writer::{RawTensorSpec, ResidentEntrySpec, ResidentTensorSpec};

/// ggml's F32 type id. The one source type this walk does not carry through
/// verbatim; see [`transcode_f32`].
const GGML_TYPE_F32: u32 = 0;

/// Which half of a fused `ffn_gate_up_exps` tensor is the gate.
///
/// **MEASURED 2026-08-07, no longer an assumption.** Stage 1 could only guess
/// (both halves have identical shapes and nothing in the file distinguishes
/// them), and a wrong guess swaps two unrelated matrices inside every routed
/// expert without crashing: the model keeps generating fluent, wrong text.
///
/// Settled by dequantizing layer 0 expert 0 out of the real
/// `gemma-4-26B-A4B-it-Q8_0.gguf` and correlating it against the same expert
/// in the MLX-derived install. Different quantizations of the same trained
/// weights, so the two agree in direction rather than to the bit, and gate
/// and up are unrelated matrices, so the separation is not subtle:
///
/// ```text
/// first half  vs install gate +0.9957   vs install up -0.0084
/// second half vs install gate -0.0085   vs install up +0.9957
/// ```
///
/// `crates/repack/tests/gguf_fused_gate_network.rs` is that measurement,
/// kept as a standing test rather than a note. It costs a few KB of ranged
/// reads, not a download.
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

/// Canonical-name suffixes whose F32 GGUF tensor becomes an INT8-affine
/// resident entry. Every other F32 tensor narrows to BF16.
///
/// Keyed by the CANONICAL name (what `RealForwardRunner` looks up), not by
/// rank, because rank does not decide: Qwen's `linear_attn.conv1d.weight` is
/// a rank-2 F32 tensor the runtime reads through `norm_view` (BF16), while
/// `mlp.gate.weight` is rank-2 and must arrive as dtype 5. Each row below is
/// a name the runtime already asserts a dtype on, not a guess:
/// `real_forward_gemma4.rs` rejects a router that is not dtype 5, and
/// `real_forward_qwen.rs` does the same for `mlp.gate.weight`.
fn int8_transcode_targets(family: ModelFamily) -> &'static [&'static str] {
    match family {
        ModelFamily::Gemma4 => &["router.proj.weight"],
        // `mlp.shared_expert_gate.weight` goes through `encode_gemv_any`,
        // which takes 4 or 8 bits; INT8 is the less lossy of the two and
        // matches the sibling router next to it.
        ModelFamily::Qwen36 => &["mlp.gate.weight", "mlp.shared_expert_gate.weight"],
        ModelFamily::DeepseekV4Flash => &[],
    }
}

/// `(rows, cols)` as the V-head helpers below read a shape. A rank-1 tensor
/// is a column of single-element rows, because that is what its V-head axis
/// is; note this is NOT how [`transcode_f32`]'s INT8 branch reads rank 1,
/// where a `[hidden]` tensor is one ROW of a `1 x hidden` projection. The two
/// readings answer different questions about the same dims.
fn row_and_col(dims: &[u64]) -> (usize, usize) {
    let (r, c, _, _) = logical_shape(dims);
    if dims.len() == 1 {
        (r as usize, 1)
    } else {
        (r as usize, c as usize)
    }
}

/// Where a tensor's V-HEAD AXIS sits, in rows or in columns of its logical
/// shape. `base` and `span` count along that axis: the first V head and how
/// wide one head is.
struct VHeadAxis {
    /// The axis is COLUMNS (`out_proj` alone), so the permutation happens
    /// inside every row instead of across whole rows.
    columns: bool,
    base: usize,
    span: usize,
}

/// Every Qwen tensor carrying llama.cpp's V-head ordering, keyed by
/// canonical-name suffix.
///
/// **THE CONVENTION IS A PROPERTY OF THE AXIS, NOT OF A TENSOR LIST**, which
/// is the finding that cost this item its second session. The three tensors
/// GGUF ships as F32 (`A_log`, `dt_bias`, `conv1d.weight`) were characterized
/// first only because a BF16 probe can compare them directly; fixing just
/// those three left the model generating gibberish, because five more tensors
/// index the same axis and are quantized
/// (`tests/gguf_qwen_quant_probe.rs`: `in_proj_b`'s map recovers exactly, and
/// `in_proj_qkv` / `in_proj_z` / `out_proj` read 0.996 de-interleaved against
/// 0.17 / 0.09 / 0.13 at the same index). If a new tensor has a V-head
/// dimension, it belongs in this table; asking whether it is F32 is asking
/// the wrong question.
///
/// llama.cpp orders V heads interleaved where MLX orders them contiguously:
/// GGUF head `h` is MLX head `2h` for the first half and `2(h - heads/2) + 1`
/// for the second, which is Qwen's two-V-heads-per-K-head grouping written
/// one way by each side. Measured, not read off the converter
/// (`tests/gguf_qwen_core_probe.rs`, worst relative error 0.000000).
///
/// The SPANS differ per tensor and that is the trap in one shared helper
/// call: a head is one element of `A_log`, `value_head_dim` rows of
/// `in_proj_z`, and `value_head_dim` COLUMNS of `out_proj`. An element-stride
/// pass over the wrong one still correlates high and is still wrong.
fn v_head_axis(canonical: &str, arch: &ArchConfig) -> Option<VHeadAxis> {
    if arch.family != ModelFamily::Qwen36 {
        return None;
    }
    let la = &arch.linear_attention;
    let rows = |base: usize, span: usize| {
        Some(VHeadAxis {
            columns: false,
            base,
            span,
        })
    };
    // Where the V region starts inside the fused `[q | k | v]` run.
    let v_at = 2 * la.num_k_heads as usize * la.key_head_dim as usize;
    let width = la.value_head_dim as usize;
    match canonical.rsplit_once("linear_attn.")?.1 {
        // One value, or one output row, per V head.
        "A_log" | "dt_bias" | "in_proj_a.weight" | "in_proj_b.weight" => rows(0, 1),
        // A `[q | k | v]` channel run; q and k do not move.
        "conv1d.weight" | "in_proj_qkv.weight" => rows(v_at, width),
        // The value stream alone.
        "in_proj_z.weight" => rows(0, width),
        // The only one whose V axis is its INPUT dimension.
        "out_proj.weight" => Some(VHeadAxis {
            columns: true,
            base: 0,
            span: width,
        }),
        _ => None,
    }
}

/// De-interleave one V-head axis in place, over units of whatever `data`
/// holds: f32 values for a transcoded tensor, raw bytes for a quantized one.
///
/// Direction: the probe measured `gguf[h] == mlx[from(h)]`, and this walk
/// writes the MLX convention, so source head `h` lands at `from(h)`.
fn permute_v_heads<T: Copy>(data: &mut [T], base: usize, span: usize, heads: usize) {
    let source = data.to_owned();
    for h in 0..heads {
        let to = if h < heads / 2 {
            2 * h
        } else {
            2 * (h - heads / 2) + 1
        };
        data[base + to * span..base + (to + 1) * span]
            .copy_from_slice(&source[base + h * span..base + (h + 1) * span]);
    }
}

fn even_v_heads(name: &str, arch: &ArchConfig) -> Result<usize, GgufRepackError> {
    let heads = arch.linear_attention.num_v_heads as usize;
    if heads < 2 || heads % 2 != 0 {
        return Err(GgufRepackError::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!("the V-head de-interleave needs an even num_v_heads, got {heads}"),
        });
    }
    Ok(heads)
}

/// The convention applied to a tensor this walk transcodes to BF16, where the
/// unit is one f32 value. Also the one place `A_log`'s value transform lives.
///
/// Applied at REPACK time, exactly like [`transcode_f32`] and
/// [`int8_transcode_targets`]: a per-tensor decision made once against a
/// canonical name, never a runtime branch and never a kernel.
fn apply_source_convention(
    name: &str,
    canonical: &str,
    arch: &ArchConfig,
    cols: usize,
    values: &mut [f32],
) -> Result<(), GgufRepackError> {
    let Some(axis) = v_head_axis(canonical, arch) else {
        return Ok(());
    };
    let heads = even_v_heads(name, arch)?;
    let shape_err = |detail: String| GgufRepackError::ShapeMismatch {
        tensor: name.to_string(),
        detail,
    };
    if axis.columns {
        // No F32 tensor takes the column axis today, and handling one would
        // need the row stride threaded in; refuse rather than half-do it.
        return Err(shape_err(
            "a column-axis V-head tensor is not expected to arrive as F32".to_string(),
        ));
    }
    // The axis counts ROWS, so scale into elements by the row width.
    let (base, span) = (axis.base * cols, axis.span * cols);
    if base + heads * span != values.len() {
        return Err(shape_err(format!(
            "{} values do not fill {base} + {heads} x {span}",
            values.len()
        )));
    }

    // `ssm_a` holds `-exp(A_log)`; the install carries `A_log` itself. Done
    // before the permutation only because it reads better; it is element-wise
    // and the two commute.
    if canonical.ends_with("linear_attn.A_log") {
        for v in values.iter_mut() {
            if !v.is_finite() || *v >= 0.0 {
                return Err(shape_err(format!(
                    "ssm_a holds -exp(A_log) and must be finite and negative, found {v}"
                )));
            }
            *v = (-*v).ln();
        }
    }
    permute_v_heads(values, base, span, heads);
    Ok(())
}

/// The same convention applied to a tensor carried through VERBATIM, where
/// the unit is one byte.
///
/// This stays inside the lossless-repack rule and does not dequantize
/// anything: every block layout here tiles along the fastest-varying dim, so
/// a logical row is a contiguous byte run and a V head is a whole number of
/// blocks. That is checked rather than assumed -- a head narrower than a
/// block (Q4_K's 256-element superblock against a 128-wide head, say) cannot
/// be permuted byte-wise, and this refuses by name instead of shuffling
/// half-blocks.
fn apply_source_convention_bytes(
    name: &str,
    canonical: &str,
    arch: &ArchConfig,
    (rows, cols): (usize, usize),
    bytes: &mut [u8],
) -> Result<(), GgufRepackError> {
    let Some(axis) = v_head_axis(canonical, arch) else {
        return Ok(());
    };
    let heads = even_v_heads(name, arch)?;
    let shape_err = |detail: String| GgufRepackError::ShapeMismatch {
        tensor: name.to_string(),
        detail,
    };
    let along = if axis.columns { cols } else { rows };
    if axis.base + heads * axis.span != along {
        return Err(shape_err(format!(
            "a {rows}x{cols} tensor's V axis does not fill {} + {heads} x {}",
            axis.base, axis.span
        )));
    }
    if rows == 0 || bytes.len() % rows != 0 {
        return Err(shape_err(format!(
            "{} bytes is not a whole number of {rows} rows",
            bytes.len()
        )));
    }
    let row_bytes = bytes.len() / rows;

    if !axis.columns {
        permute_v_heads(bytes, axis.base * row_bytes, axis.span * row_bytes, heads);
        return Ok(());
    }
    // The V axis is this tensor's INPUT dimension, so the permutation happens
    // inside every row. A head is only addressable byte-wise if it is a whole
    // number of blocks: Q8_0 tiles 32 elements to 34 bytes, so a 128-wide head
    // is 4 blocks and lands exactly, while a Q4_K superblock spans 256
    // elements and would put two heads in one block.
    if row_bytes * axis.span % cols != 0 {
        return Err(shape_err(format!(
            "a {row_bytes}-byte row of {cols} columns has no whole-byte \
             {}-column V head: the block is wider than one head",
            axis.span
        )));
    }
    let head_bytes = row_bytes * axis.span / cols;
    let base_bytes = row_bytes * axis.base / cols;
    for r in 0..rows {
        permute_v_heads(
            &mut bytes[r * row_bytes..(r + 1) * row_bytes],
            base_bytes,
            head_bytes,
            heads,
        );
    }
    Ok(())
}

/// One transcoded F32 tensor, plus how many of its values a BF16 could not
/// hold exactly.
struct Transcoded {
    spec: ResidentEntrySpec,
    lossy: usize,
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// GGUF's F32 resident core, converted to the dtypes this port's kernels
/// read. DECIDED BY MEASUREMENT, not preference, and not a tradeoff:
/// `tests/gguf_f32_transcode_network.rs` on the real published file found
/// that llama.cpp UPCASTS norms that are BF16 in the original checkpoint (all
/// 15,592 values across seven norm tensors narrow back with zero bit loss),
/// and that the router's INT8 transcode is the same affine the MLX path
/// already applies to the same tensor and leaves top-1 routing unchanged on
/// 32 of 32 activations. The alternative cost two more kernels, each needing
/// its own CPU reference and parity test, to buy nothing. See
/// `crates/repack/CLAUDE.md` Gotcha 6 before re-opening it.
///
/// The BF16 default is the safe one: a tensor sent to the wrong dtype fails
/// LOUDLY at `open()` (`norm_view`'s byte-size check, or `encode_gemv_any`'s
/// "no dispatched GEMV kernel"), never silently.
fn transcode_f32(
    name: &str,
    canonical: String,
    bytes: &[u8],
    dims: &[u64],
    arch: &ArchConfig,
) -> Result<Transcoded, GgufRepackError> {
    let family = arch.family;
    let mut values = f32_values(bytes);
    if values.len() * 4 != bytes.len() {
        return Err(GgufRepackError::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!("{} bytes is not a whole number of F32 values", bytes.len()),
        });
    }
    // Before either dtype branch: the convention is about what the values
    // MEAN, and both branches would otherwise carry the wrong ones faithfully.
    apply_source_convention(name, &canonical, arch, row_and_col(dims).1, &mut values)?;

    let wants_int8 = int8_transcode_targets(family)
        .iter()
        .any(|suffix| canonical.ends_with(suffix));
    if !wants_int8 {
        let mut lossy = 0usize;
        let mut out = Vec::with_capacity(values.len() * 2);
        for &v in &values {
            let bits = compute::f32_to_bf16(v);
            if compute::bf16_to_f32(bits) != v {
                lossy += 1;
            }
            out.extend_from_slice(&bits.to_le_bytes());
        }
        return Ok(Transcoded {
            spec: ResidentEntrySpec::Raw(RawTensorSpec {
                name: canonical,
                dtype: crate::resident_writer::DTYPE_BF16,
                bytes: out,
                shape: logical_shape(dims),
            }),
            lossy,
        });
    }

    // INT8 affine, per logical row, at the 64-element group every GEMV
    // kernel here assumes.
    //
    // Rank 1 is a SINGLE-ROW matrix, not an error: llama.cpp drops the
    // degenerate output dimension, so Qwen's `ffn_gate_inp_shexp.weight`
    // arrives as `[hidden]` where the runtime reads it as a `1 x hidden`
    // projection through `encode_gemv_any`. Rejecting it here was the first
    // thing the real Q4_K_M install hit, and a rank-1 tensor sent to the BF16
    // default instead would have failed at `open()` rather than silently, so
    // this arm is about accepting the file rather than about safety.
    let (rows, cols) = match dims.len() {
        1 => (1usize, dims[0] as usize),
        2 => {
            let (r, c, _, _) = logical_shape(dims);
            (r as usize, c as usize)
        }
        _ => {
            return Err(GgufRepackError::ShapeMismatch {
                tensor: name.to_string(),
                detail: format!("INT8 transcode needs a rank-1 or rank-2 tensor, got {dims:?}"),
            })
        }
    };
    if rows * cols != values.len() {
        return Err(GgufRepackError::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!("{} values do not fill {rows}x{cols}", values.len()),
        });
    }
    if cols % 64 != 0 {
        return Err(GgufRepackError::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!("row length {cols} is not a multiple of the 64-element group"),
        });
    }
    let mut packed = Vec::with_capacity(rows * cols);
    let mut scales = Vec::with_capacity(rows * cols / 64);
    let mut biases = Vec::with_capacity(rows * cols / 64);
    for r in 0..rows {
        let row = compute::quantize_int8_affine(&values[r * cols..(r + 1) * cols]);
        packed.extend_from_slice(&row.packed);
        scales.extend_from_slice(&row.scales);
        biases.extend_from_slice(&row.biases);
    }
    Ok(Transcoded {
        spec: ResidentEntrySpec::Int8(ResidentTensorSpec {
            name: canonical,
            packed,
            scales,
            biases,
            rows: rows as u32,
            cols: cols as u32,
        }),
        lossy: 0,
    })
}

/// Resident entries plus one `(GGUF tensor name, value count)` row per
/// tensor whose BF16 narrowing was not exact.
type ResidentSet = (Vec<ResidentEntrySpec>, Vec<(String, usize)>);

/// The resident set, with every F32 tensor transcoded. A lossy narrowing
/// comes back as a reported fact rather than being silently swallowed.
fn resident_entries(
    header: &GgufHeader,
    source: &dyn RangeSource,
    arch: &ArchConfig,
    names: &[&str],
) -> Result<ResidentSet, GgufRepackError> {
    let family = arch.family;
    let mut out = Vec::with_capacity(names.len());
    let mut lossy = Vec::new();
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
        let bytes = read_tensor(header, source, name)?;
        if info.ggml_type == GGML_TYPE_F32 {
            let t = transcode_f32(name, canonical, &bytes, &info.dims, arch)?;
            if t.lossy > 0 {
                lossy.push(((*name).to_string(), t.lossy));
            }
            out.push(t.spec);
            continue;
        }
        // The one thing done to a verbatim tensor's bytes, and it moves them
        // rather than changing them: Qwen's V-head order. A no-op for every
        // other family and every tensor with no V-head axis.
        let mut bytes = bytes;
        apply_source_convention_bytes(name, &canonical, arch, row_and_col(&info.dims), &mut bytes)?;
        out.push(ResidentEntrySpec::Raw(RawTensorSpec {
            name: canonical,
            dtype,
            bytes,
            shape: logical_shape(&info.dims),
        }));
    }
    Ok((out, lossy))
}

/// The `manifest.json -> quant` object for a GGUF-sourced install.
///
/// Deliberately NOT `"affine"` for the block-quantized slots.
/// `turbospark_model_io::validate_quant` accepts only that scheme, so those
/// values are what make `load_manifest` refuse a Stage 1 install by name
/// instead of loading one whose expert bytes no kernel can read.
///
/// The ROUTER slot is the exception, and it is not a relaxation: after
/// [`transcode_f32`] the router really is INT8 affine at group 64 with BF16
/// companions, so saying anything else would be a lie about the bytes on
/// disk. The install stays refused on the other four slots.
fn gguf_manifest_quant(header: &GgufHeader, plan: &Plan<'_>) -> serde_json::Value {
    // Searched by SUFFIX across every tensor, not read off `blk.0.`, because
    // a hybrid model's layer 0 need not carry the tensor a slot is named for.
    // Qwen 3.6 puts linear attention on layer 0, so it has no `attn_q` there
    // (and no `ffn_gate` anywhere -- its shared expert is `ffn_gate_shexp`),
    // and both slots came out "absent", which `validate_quant` then refused
    // as a block type with no kernel. Found on the real Q4_K_M install, after
    // the walk had written 19 GB of correct bytes.
    let type_of = |suffixes: &[&str]| {
        header
            .tensors
            .iter()
            .find(|(name, _)| suffixes.iter().any(|s| name.ends_with(s)))
            .map(|(_, i)| ggml_scheme_name(i.ggml_type))
            .unwrap_or("absent")
    };
    let routed = plan
        .routed
        .values()
        .next()
        .and_then(|s| s.first())
        .map(|s| type_of(&[s.name]))
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
    // What the router slot says has to match what the transcode wrote.
    let router_source = type_of(&["ffn_gate_inp.weight"]);
    let router = if router_source == "F32" {
        serde_json::json!({
            "weightBits": 8,
            "scheme": "affine",
            "scaleType": "bf16",
            "biasType": "bf16",
            "groupSize": 64,
        })
    } else {
        slot(router_source)
    };
    serde_json::json!({
        "embedding": slot(type_of(&["token_embd.weight"])),
        // Qwen's hybrid layers fuse Q/K/V into `attn_qkv`; Gemma has only
        // `attn_q`. Either answers "what block type is the attention core".
        "attention": slot(type_of(&["attn_q.weight", "attn_qkv.weight"])),
        "router": router,
        // Gemma's shared expert is `ffn_gate`, Qwen's is `ffn_gate_shexp`.
        "sharedExpert": slot(type_of(&["ffn_gate.weight", "ffn_gate_shexp.weight"])),
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
    /// `(GGUF tensor name, value count)` for every F32 tensor whose BF16
    /// narrowing lost bits. Empty for Gemma, because llama.cpp upcasts norms
    /// that were BF16 upstream (`tests/gguf_f32_transcode_network.rs`); a row
    /// there means this converter did not, and the install carries less
    /// precision than the source did.
    ///
    /// QWEN'S `ssm_a` ROWS ARE THE EXCEPTION AND ARE EXPECTED, roughly one
    /// per GDN layer: [`apply_source_convention`] rewrites that tensor as
    /// `ln(-ssm_a)`, and a logarithm's output is not generally BF16-exact
    /// whatever the source was. Those rows are about the TRANSFORM, not about
    /// the converter, so they are a different population from the raw
    /// narrowing this field was written for. The real Q4_K_M reports 130: 101
    /// norms plus 29 of its 30 `ssm_a` (layer 0's 32 values happen to land on
    /// the BF16 grid; the rest lose 3 to 14 each). The norms are proven
    /// bit-identical to the MLX install (ROADMAP item 10).
    pub lossy_narrowing: Vec<(String, usize)>,
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
    let (resident, lossy_narrowing) = resident_entries(header, source, &arch, &plan.resident)?;

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
        lossy_narrowing,
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

    let (resident, lossy) = resident_entries(header, source, &arch, &plan.resident)?;
    let resident_bytes = crate::resident_writer::build_resident_weights_bin_mixed(&resident);
    drop(resident);
    for (name, count) in &lossy {
        progress(&format!(
            "WARNING {name}: {count} F32 values lost bits narrowing to BF16 \
             (this converter did not upcast from BF16)"
        ));
    }
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
