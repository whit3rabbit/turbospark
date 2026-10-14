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
    if arch.num_experts <= 0 {
        return Err(Gemma4Error::Config(
            "routed expert tensors require a positive num_experts".to_string(),
        ));
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
            if quant.is_mxfp4(base) {
                // Re-blocked to GGUF MXFP4 at write time: the 17-byte block
                // carries its own exponent, so there are no companions.
                let t = shards.info(name)?;
                let [_, rows, words] = t.shape[..] else {
                    return Err(Gemma4Error::ShapeMismatch {
                        tensor: name.to_string(),
                        detail: format!("expected rank-3 mxfp4 experts, got {:?}", t.shape),
                    });
                };
                blob += rows * super::mlx_mxfp4::gguf_mxfp4_row_bytes(words as usize * 8) as u64;
                continue;
            }
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
    if arch.num_experts <= 0 {
        return Err(Gemma4Error::Config(
            "routed expert tensors require a positive num_experts".to_string(),
        ));
    }
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

    let base_of = |role: &str| -> Result<&str, Gemma4Error> {
        let name = *bundle
            .get(role)
            .ok_or_else(|| Gemma4Error::MissingTensor(format!("layer {layer} {role}_proj")))?;
        Ok(name.strip_suffix(".weight").unwrap_or(name))
    };
    let mxfp4 = ["gate", "up", "down"]
        .iter()
        .map(|r| base_of(r).map(|b| quant.is_mxfp4(b)))
        .collect::<Result<Vec<_>, _>>()?;
    if mxfp4.iter().all(|&m| m) {
        return plan_mxfp4_expert_layer(shards, bundle, experts, layer, expert_count);
    }
    if mxfp4.iter().any(|&m| m) {
        return Err(Gemma4Error::UnsupportedDtype {
            tensor: format!("layer {layer} routed experts"),
            dtype: "mxfp4 mixed with affine across gate/up/down".to_string(),
        });
    }

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

/// The MLX mxfp4 sibling of the affine walk above: three sub-tensors per
/// expert, `gate`, `up`, `down`, each re-blocked to GGUF MXFP4 with dtype
/// `"mxfp4"` and logical shape `[out, in]` -- what the GGUF writer records
/// for `gpt-oss`, so `routed_layouts_from_layout` classifies the layer as
/// `GgufMxfp4` (no `gate_scales`) and resolves no bias planes.
fn plan_mxfp4_expert_layer(
    shards: &Gemma4Shards<'_>,
    bundle: &BTreeMap<&'static str, &str>,
    mut experts: Vec<ExpertBlob>,
    layer: usize,
    expert_count: usize,
) -> Result<(LayerBlobs, u64), Gemma4Error> {
    let mut blob_used = 0u64;
    for role in ["gate", "up", "down"] {
        let name = *bundle
            .get(role)
            .ok_or_else(|| Gemma4Error::MissingTensor(format!("layer {layer} {role}_proj")))?;
        let base = name.strip_suffix(".weight").unwrap_or(name);
        let w = shards.info(name)?;
        let [count, rows, words] = w.shape[..] else {
            return Err(Gemma4Error::ShapeMismatch {
                tensor: name.to_string(),
                detail: format!("expected rank-3 mxfp4 experts, got {:?}", w.shape),
            });
        };
        if w.dtype != "U32" || count as usize != expert_count {
            return Err(Gemma4Error::ShapeMismatch {
                tensor: name.to_string(),
                detail: format!(
                    "expected U32 rank-3 with leading {expert_count}, got {} {:?}",
                    w.dtype, w.shape
                ),
            });
        }
        // What mxfp4 writes beside it: U8 exponents, one per 32, and no
        // `.biases` (whose presence would mean a mislabelled affine tensor).
        let scales_name = format!("{base}.scales");
        let s = shards.info(&scales_name)?;
        let cols = words * 8;
        let expected = [count, rows, cols / super::mlx_mxfp4::MXFP4_GROUP as u64];
        if s.dtype != "U8" || s.shape != expected {
            return Err(Gemma4Error::ShapeMismatch {
                tensor: scales_name,
                detail: format!(
                    "expected U8 {expected:?} mxfp4 expert scales, got {} {:?}",
                    s.dtype, s.shape
                ),
            });
        }
        if shards.contains(&format!("{base}.biases")) {
            return Err(Gemma4Error::ShapeMismatch {
                tensor: name.to_string(),
                detail: "mxfp4 experts carry a .biases plane, so they are affine".to_string(),
            });
        }
        let (rows, cols) = (rows as usize, cols as usize);
        let all_rows = rows * expert_count;
        let gguf = super::mlx_mxfp4::reblock_to_gguf_mxfp4(
            name,
            &shards.read(name)?,
            &shards.read(&format!("{base}.scales"))?,
            all_rows,
            cols,
        )?;
        let per = rows * super::mlx_mxfp4::gguf_mxfp4_row_bytes(cols);
        // The phase-2 row helper reads its weights with `uint` loads.
        if role == "down" && blob_used % 4 != 0 {
            return Err(Gemma4Error::ShapeMismatch {
                tensor: name.to_string(),
                detail: format!("down offset {blob_used} not 4-byte aligned"),
            });
        }
        for (e, blob) in experts.iter_mut().enumerate() {
            blob.sub_tensors.push(SubTensor {
                role: role.to_string(),
                bytes: gguf[e * per..(e + 1) * per].to_vec(),
                dtype: "mxfp4".to_string(),
                shape: vec![rows as u64, cols as u64],
            });
        }
        blob_used += per as u64;
    }
    Ok((LayerBlobs { layer, experts }, blob_used))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routed_layout_refuses_zero_experts_before_division() {
        let arch = crate::tiny_muse_glimmer_arch(64, 4);
        let shards = Gemma4Shards::new(Vec::new()).expect("empty shard registry");
        let routed = BTreeMap::from([(
            0,
            BTreeMap::from([(
                "gate",
                "language_model.model.layers.0.experts.switch_glu.gate_proj.weight",
            )]),
        )]);

        let err = expert_stride_from_headers(&shards, &arch, &Gemma4Quant::default(), &routed)
            .expect_err("a routed map cannot be laid out with zero experts");
        assert!(
            err.to_string().contains("positive num_experts"),
            "unexpected error: {err}"
        );
    }
}
