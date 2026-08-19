//! GGUF tensor classification, expert stride calculation, and layer planning.

use std::collections::BTreeMap;

use model_io::{ArchConfig, ModelFamily};

use super::types::{ggml_scheme_name, read_tensor, GgufRepackError, FUSED_GATE_FIRST};
use crate::gguf_header::GgufHeader;
use crate::gguf_names::{map_gguf_name, GgufMapping};
use crate::gturbo_writer::{ExpertBlob, LayerBlobs, SubTensor};
use crate::ranged_download::RangeSource;

/// One routed-expert source tensor, resolved but not yet read.
pub struct RoutedSource<'a> {
    /// GGUF tensor name.
    pub name: &'a str,
    /// Roles this tensor supplies, in blob order. Two for a fused gate/up.
    pub roles: Vec<&'static str>,
}

pub struct Plan<'a> {
    pub resident: Vec<&'a str>,
    /// layer -> role-bearing source tensors.
    pub routed: BTreeMap<usize, Vec<RoutedSource<'a>>>,
    pub ignored: Vec<String>,
}

/// The block index of a `blk.<n>.` tensor that sits ABOVE the trunk, or
/// `None` for anything the trunk owns.
///
/// `arch_from_gguf` subtracts `nextn_predict_layers` from `block_count`,
/// because this port's `num_layers` is the trunk alone and llama.cpp writes
/// a multi-token-prediction head as one more `blk.` block. That subtraction
/// has to reach the WALK too: without it the head's tensors map through the
/// trunk table into a phantom `layers.<num_layers>` group (silently, on a
/// dense file) or hit `Unmapped` on a head-only suffix and abort a multi-GB
/// stream naming the tensor rather than the head.
///
/// A file declaring no head has `num_layers == block_count`, so this is
/// `None` for every tensor of every GGUF this port installed before the key
/// was read.
fn head_block_index(name: &str, num_layers: usize) -> Option<usize> {
    let (idx, _) = name.strip_prefix("blk.")?.split_once('.')?;
    let layer: usize = idx.parse().ok()?;
    (layer >= num_layers).then_some(layer)
}

pub fn classify<'a>(
    header: &'a GgufHeader,
    family: ModelFamily,
    num_layers: usize,
) -> Result<Plan<'a>, GgufRepackError> {
    let mut plan = Plan {
        resident: Vec::new(),
        routed: BTreeMap::new(),
        ignored: Vec::new(),
    };
    for name in header.tensors.keys() {
        if let Some(layer) = head_block_index(name, num_layers) {
            plan.ignored.push(format!(
                "{name} (block {layer} is above the {num_layers}-block trunk: a \
                 multi-token-prediction head, which this port ingests from the safetensors \
                 checkpoint rather than from a GGUF)"
            ));
            continue;
        }
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
    // Each bias sits immediately after the projection it belongs to, which is
    // the order `gemma4_checkpoint/orchestrate.rs` already documents for the
    // affine layout (`gate, gate_scales, gate_biases, up, ...`). Nothing
    // ADDRESSES a sub-tensor by position -- `moe_offsets_from_layout` resolves
    // every one by role through `layout.json` -- so this buys byte
    // reproducibility and readability rather than correctness. Leaving the
    // three M5 roles off the list would have sorted them all to the end in
    // whatever order the header happened to list them.
    const ORDER: [&str; 6] = [
        "gate",
        "gate_biases",
        "up",
        "up_biases",
        "down",
        "down_biases",
    ];
    let rank = |r: &str| ORDER.iter().position(|o| *o == r).unwrap_or(ORDER.len());
    for sources in plan.routed.values_mut() {
        sources.sort_by_key(|s| rank(s.roles[0]));
    }
    plan.resident.sort_unstable();
    Ok(plan)
}

/// One routed source's dims with the trailing EXPERT dimension removed.
///
/// Rank 3 (`[in, out, experts]`) is a matrix per expert and was the only shape
/// any routed tensor had before ROADMAP M5. `gpt-oss` adds RANK 2
/// (`[out, experts]`), its per-expert biases, so the walk can no longer assume
/// a routed tensor has an input dimension at all. Both are handled the same
/// way once the expert axis is off: whatever is left is that expert's payload,
/// stored fastest-varying first.
fn routed_body_dims<'a>(
    info: &'a crate::gguf_header::GgufTensorInfo,
    name: &str,
    num_experts: u64,
) -> Result<&'a [u64], GgufRepackError> {
    if !matches!(info.dims.len(), 2 | 3) {
        return Err(GgufRepackError::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!(
                "expected a rank-2 or rank-3 routed tensor, got {:?}",
                info.dims
            ),
        });
    }
    let last = info.dims[info.dims.len() - 1];
    if last != num_experts {
        return Err(GgufRepackError::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!("trailing dim {last} is not the expert count {num_experts}"),
        });
    }
    Ok(&info.dims[..info.dims.len() - 1])
}

/// Per-expert byte size of one routed source tensor, plus the number of
/// experts it carries. The expert index is GGUF's SLOWEST-varying dimension
/// (last, as stored), so each expert's bytes are one contiguous range.
pub fn per_expert_bytes(
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
    routed_body_dims(info, name, num_experts)?;
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
pub fn expert_stride(
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
pub fn plan_one_layer(
    header: &GgufHeader,
    source: &dyn RangeSource,
    arch: &ArchConfig,
    plan: &Plan<'_>,
    layer: usize,
) -> Result<(LayerBlobs, u64), GgufRepackError> {
    plan_one_layer_inner(header, Some(source), arch, plan, layer)
}

/// The same layer plan with NO network read: every sub-tensor gets a
/// correctly-SIZED run of zeros instead of its real bytes.
///
/// This exists for resume (ROADMAP Phase M2). A layer file already on disk
/// at its expected size does not need re-fetching, but the walk still has to
/// produce that layer's `layout.json` entry -- offsets, strides, dtypes and
/// shapes -- and every one of those is a function of the HEADER rather than
/// of the bytes. Zero-filling is what lets the entry be built by the same
/// `build_layer_file` the real path uses, so the two cannot disagree about a
/// layout; the zeros are allocated and dropped without ever being written.
pub fn plan_one_layer_shape(
    header: &GgufHeader,
    arch: &ArchConfig,
    plan: &Plan<'_>,
    layer: usize,
) -> Result<(LayerBlobs, u64), GgufRepackError> {
    plan_one_layer_inner(header, None, arch, plan, layer)
}

/// Bytes one layer's expert FILE occupies, from the header alone.
///
/// Needed BEFORE deciding whether to fetch a layer, so the decision cannot
/// depend on the fetch. Mirrors `build_layer_file`'s stride arithmetic; the
/// two are held together by the resume path asserting the size it predicted
/// against the size the writer produces.
pub fn layer_file_bytes(
    header: &GgufHeader,
    arch: &ArchConfig,
    plan: &Plan<'_>,
    layer: usize,
    max_stride: u64,
) -> Result<u64, GgufRepackError> {
    let experts = arch.num_experts as u64;
    let sources = plan
        .routed
        .get(&layer)
        .ok_or_else(|| GgufRepackError::MissingTensor {
            name: format!("layer {layer} routed experts"),
        })?;
    let mut used = 0u64;
    for s in sources {
        used += per_expert_bytes(header, s.name, experts)?;
    }
    let stride =
        (used.div_ceil(crate::GTURBO_PAGE_BYTES) * crate::GTURBO_PAGE_BYTES).min(max_stride);
    Ok(stride * experts)
}

fn plan_one_layer_inner(
    header: &GgufHeader,
    source: Option<&dyn RangeSource>,
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
        let bytes = match source {
            Some(source) => read_tensor(header, source, s.name)?,
            None => vec![0u8; per * experts],
        };
        if bytes.len() != per * experts {
            return Err(GgufRepackError::ShapeMismatch {
                tensor: s.name.to_string(),
                detail: format!("read {} bytes, expected {}", bytes.len(), per * experts),
            });
        }

        let parts = s.roles.len();
        if per % parts != 0 {
            return Err(GgufRepackError::ShapeMismatch {
                tensor: s.name.to_string(),
                detail: format!("{per} bytes per expert does not split into {parts} roles"),
            });
        }
        let part_bytes = per / parts;
        // The OUTPUT dim is the slowest-varying one of the per-expert body,
        // which is `dims[1]` for a rank-3 weight and `dims[0]` for a rank-2
        // bias. Indexing `dims[1]` unconditionally read the EXPERT COUNT on a
        // bias, which is a plausible number and would have produced a
        // correctly-sized blob with a nonsense recorded shape.
        let body = routed_body_dims(info, s.name, experts as u64)?;
        let out_total = body[body.len() - 1];
        if out_total % parts as u64 != 0 {
            return Err(GgufRepackError::ShapeMismatch {
                tensor: s.name.to_string(),
                detail: format!("output dim {out_total} does not split into {parts} roles"),
            });
        }
        // Logical shape, i.e. GGUF's reversed: `[out, in]` for a weight and
        // `[out]` for a bias. Only Gemma's fused gate/up ever has `parts > 1`
        // and it is rank 3, so the division always lands on the output dim.
        let mut part_shape = vec![out_total / parts as u64];
        part_shape.extend(body[..body.len() - 1].iter().rev().copied());
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
