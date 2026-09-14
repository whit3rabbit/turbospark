//! MoE layer blob planning and expert stride calculation.

use std::collections::BTreeMap;

use model_io::ArchConfig;

use super::config::{Gemma4Error, Gemma4Quant};
use super::shards::{Gemma4Shards, GTURBO_PAGE_BYTES};
use crate::gturbo_writer::{ExpertBlob, LayerBlobs, SubTensor};
use crate::safetensors_header::TensorInfo;

/// The one model-wide expert stride, computed from shard HEADERS alone
/// (per-expert weight+scales+biases byte totals, max across layers,
/// rounded to 16 KiB) -- so a streaming writer knows the stride before any
/// expert byte downloads.
pub fn expert_stride_from_headers(
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
pub fn plan_one_expert_layer(
    shards: &Gemma4Shards<'_>,
    arch: &ArchConfig,
    quant: &Gemma4Quant,
    routed: &BTreeMap<usize, BTreeMap<&'static str, &str>>,
    layer: usize,
) -> Result<(LayerBlobs, u64), Gemma4Error> {
    let expert_count = arch.num_experts as usize;
    let bundle = routed
        .get(&layer)
        .ok_or_else(|| Gemma4Error::MissingTensor(format!("layer {layer} routed-expert bundle")))?;
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
        for (companion_name, companion) in [(&s_name, s), (&b_name, b)] {
            if companion.shape.len() != 3 || companion.shape[0] as usize != expert_count {
                return Err(Gemma4Error::ShapeMismatch {
                    tensor: companion_name.to_string(),
                    detail: format!(
                        "expected rank-3 with leading {expert_count}, got {:?}",
                        companion.shape
                    ),
                });
            }
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
        // `w_per` comes purely from the byte range; tie it to the tensor's
        // OWN declared shape too, or a shard whose `data_offsets` disagree
        // with `shape` writes a correctly-sized blob at a nonsense logical
        // shape (`rows`/`cols` below are read from `w.shape`, independently
        // of `w_per`).
        let expected_w_per = (w.shape[1] * w.shape[2] * 4) as usize;
        if w_per != expected_w_per {
            return Err(Gemma4Error::ShapeMismatch {
                tensor: name.to_string(),
                detail: format!(
                    "per-expert weight blob is {w_per} bytes, expected {expected_w_per} for \
                     shape {:?}",
                    w.shape
                ),
            });
        }

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
