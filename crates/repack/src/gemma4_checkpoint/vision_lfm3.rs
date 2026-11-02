//! The LFM3-VL vision tower's config and ingest (`docs/LFM3_VL_VISION_PHASE0.md`).
//!
//! A sibling of [`super::vision`] and [`super::vision_gemma4`], not a variant
//! of either. The tower alternates two block kinds, so a blob's role set
//! depends on the block:
//!
//! - every block: `operator_norm`, `ffn_norm` (RMSNorm) and a SwiGLU MLP
//!   (`w1`, `w3`, `w2`, no biases);
//! - a `conv` block adds `in_proj`, the depthwise 3x3 `spatial_conv` and
//!   `out_proj`;
//! - a `full_attention` block adds grouped-query `q`/`k`/`v`/`out`
//!   projections and per-head `q_layernorm`/`k_layernorm`.
//!
//! The kind is read off which tensors a layer carries, never from a position
//! formula, and a layer carrying both or neither kind's set is refused.
//!
//! # Two transformations that change no arithmetic
//!
//! **The patchifier's columns are permuted.** The checkpoint's `Linear`
//! consumes patch rows flattened `(C, T, P_h, P_w)`, as the reference
//! processor emits them. `turbospark_vision_io` emits `(T, P_h, P_w, C)` for
//! every tower, so the columns are reordered here, once, rather than a second
//! patch layout being carried through preprocessing.
//!
//! **Grouped-query K and V are expanded to full width.** The vision attention
//! kernel is multi-head, so each KV head's `head_dim` rows of `k_proj` and
//! `v_proj` are written `heads / kv_heads` times in a row. The projection then
//! produces the repeated heads directly, which is what `repeat_kv` does
//! after the projection (head `h` reads KV head `h / n_rep`), for 2 MiB more
//! per attention block. The per-head k norm sees identical inputs for the
//! copies, so it is unchanged.

use model_io::{VisionConfig, VisionTowerKind};

use super::classify::{VISION_INSTALL_PREFIX, VISION_PREFIX};
use super::config::Gemma4Error;
use super::narrow::convert_raw_to_fp16;
use super::shards::{shape4, Gemma4Shards, GTURBO_PAGE_BYTES};
use super::vision::VisionRead;
use crate::gturbo_writer::{ExpertBlob, LayerBlobs, SubTensor};
use crate::resident_writer::{RawTensorSpec, ResidentEntrySpec};

/// The projector's checkpoint prefix, beside the tower's rather than under it.
pub const LFM3_PROJECTOR_PREFIX: &str = "multi_modal_projector.";

/// Where a block's tensors live under [`VISION_PREFIX`].
const BLOCK_PATH: &str = "layers.";

/// Roles every block carries, `(role in the blob, suffix in the checkpoint)`.
pub const LFM3_COMMON_ROLES: [(&str, &str); 5] = [
    ("op_norm", "operator_norm.weight"),
    ("ffn_norm", "ffn_norm.weight"),
    ("w1", "feed_forward.w1.weight"),
    ("w3", "feed_forward.w3.weight"),
    ("w2", "feed_forward.w2.weight"),
];

/// Roles only a conv block carries.
pub const LFM3_CONV_ROLES: [(&str, &str); 3] = [
    ("in_proj", "conv.in_proj.weight"),
    ("conv_w", "conv.spatial_conv.weight"),
    ("out_proj", "conv.out_proj.weight"),
];

/// Roles only an attention block carries.
pub const LFM3_ATTN_ROLES: [(&str, &str); 6] = [
    ("q_w", "self_attn.q_proj.weight"),
    ("k_w", "self_attn.k_proj.weight"),
    ("v_w", "self_attn.v_proj.weight"),
    ("o_w", "self_attn.out_proj.weight"),
    ("q_norm", "self_attn.q_layernorm.weight"),
    ("k_norm", "self_attn.k_layernorm.weight"),
];

/// The tower's non-block tensors under [`VISION_PREFIX`], resident at FP16.
pub const LFM3_RESIDENT_TENSORS: [&str; 2] = ["patchifier.weight", "embedding_norm.weight"];

/// The projector's six tensors under [`LFM3_PROJECTOR_PREFIX`].
pub const LFM3_PROJECTOR_TENSORS: [&str; 6] = [
    "layer_norm.weight",
    "layer_norm.bias",
    "linear_1.weight",
    "linear_1.bias",
    "linear_2.weight",
    "linear_2.bias",
];

/// The install-name level the projector lands under, beside the tower's own.
pub const LFM3_PROJECTOR_INSTALL_LEVEL: &str = "projector.";

/// The only depth a published LFM3-VL tower declares.
const SUPPORTED_DEPTHS: [i64; 1] = [14];

/// Parse an LFM3-VL `config.json`'s vision half into a [`VisionConfig`].
///
/// Absent `vision_config` is [`VisionConfig::NONE`]. Every value the runtime
/// does not store is checked here and refused when it differs from the one
/// tower this port implements: the axial rope at theta 1e6, norm eps 1e-5,
/// the 3x3 conv, the projector's exact-GELU, bias and eps, and the layer
/// pattern's vocabulary.
pub fn parse_lfm3_vl_vision_config(json: &str) -> Result<VisionConfig, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    let vc = match root.get("vision_config") {
        None | Some(serde_json::Value::Null) => return Ok(VisionConfig::NONE),
        Some(v) => v,
    };
    let bad = |m: String| Gemma4Error::Config(m);
    let vi = |k: &str| -> Result<i64, Gemma4Error> {
        vc.get(k)
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| bad(format!("missing vision_config.{k}")))
    };
    let ri = |k: &str| -> Result<i64, Gemma4Error> {
        root.get(k)
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| bad(format!("missing {k}")))
    };
    let text_hidden = root
        .get("text_config")
        .and_then(|t| t.get("hidden_size"))
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| bad("missing text_config.hidden_size".into()))?;

    let hidden = vi("hidden_size")?;
    let heads = vi("num_attention_heads")?;
    let kv_heads = vi("num_key_value_heads")?;
    if heads <= 0 || kv_heads <= 0 || hidden % heads != 0 || heads % kv_heads != 0 {
        return Err(bad(format!(
            "vision hidden {hidden} / heads {heads} / kv heads {kv_heads} is not a grouped-query \
             shape"
        )));
    }
    let head_dim = hidden / heads;
    if head_dim % 4 != 0 {
        return Err(bad(format!(
            "vision head_dim {head_dim} is not a multiple of 4; the 2-D rope splits it in two"
        )));
    }
    let rope = vc.get("rope_parameters");
    if rope
        .and_then(|r| r.get("rope_theta"))
        .and_then(serde_json::Value::as_f64)
        != Some(1_000_000.0)
    {
        return Err(bad(
            "vision rope_theta must be 1e6; the implemented tower is theta 1e6".into(),
        ));
    }
    if rope
        .and_then(|r| r.get("rope_type"))
        .and_then(serde_json::Value::as_str)
        != Some("axial")
    {
        return Err(bad("vision rope_type must be axial".into()));
    }
    if vc.get("norm_eps").and_then(serde_json::Value::as_f64) != Some(1e-5) {
        return Err(bad("vision norm_eps must be 1e-5".into()));
    }
    if vi("conv_kernel_size")? != 3 {
        return Err(bad("only a 3x3 depthwise vision conv is implemented".into()));
    }
    let layer_types = vc
        .get("layer_types")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| bad("missing vision_config.layer_types".into()))?;
    let depth = vi("num_hidden_layers")?;
    if layer_types.len() as i64 != depth
        || layer_types
            .iter()
            .any(|t| !matches!(t.as_str(), Some("conv" | "full_attention")))
    {
        return Err(bad(format!(
            "vision layer_types must name {depth} layers, each conv or full_attention"
        )));
    }
    // The projector is part of the tower kind: exact GELU, biases, eps 1e-6,
    // an unshuffle factor equal to the merge size, and the 4096-wide hidden
    // layer the runtime sizes its scratch for.
    if root
        .get("projector_hidden_act")
        .and_then(serde_json::Value::as_str)
        != Some("gelu")
    {
        return Err(bad("projector_hidden_act must be gelu".into()));
    }
    if root
        .get("projector_bias")
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        return Err(bad("projector_bias must be true".into()));
    }
    if root
        .get("projector_layer_norm_eps")
        .and_then(serde_json::Value::as_f64)
        != Some(1e-6)
    {
        return Err(bad("projector_layer_norm_eps must be 1e-6".into()));
    }
    let merge = vi("spatial_merge_size")?;
    if ri("downsample_factor")? != merge {
        return Err(bad(format!(
            "downsample_factor {} differs from spatial_merge_size {merge}",
            ri("downsample_factor")?
        )));
    }
    if ri("projector_hidden_size")? != 4096 {
        return Err(bad("projector_hidden_size must be 4096".into()));
    }

    let vision = VisionConfig {
        depth,
        hidden_size: hidden,
        intermediate_size: vi("intermediate_size")?,
        num_heads: heads,
        patch_size: vi("patch_size")?,
        temporal_patch_size: vi("temporal_patch_size")?,
        in_channels: vi("in_channels")?,
        spatial_merge_size: merge,
        num_position_embeddings: 0,
        out_hidden_size: text_hidden,
        mrope_section: [0, 0, 0],
        deepstack_visual_indexes: Vec::new(),
        vision_start_token_id: ri("vision_start_token_id")?,
        vision_end_token_id: ri("vision_end_token_id")?,
        image_token_id: ri("image_token_id")?,
        video_token_id: ri("video_token_id")?,
        tower: VisionTowerKind::Lfm3Vl,
    };
    if !SUPPORTED_DEPTHS.contains(&vision.depth) {
        return Err(bad(format!(
            "lfm3_vl vision depth {} is unsupported; expected one of {SUPPORTED_DEPTHS:?}",
            vision.depth
        )));
    }
    if (
        vision.patch_size,
        vision.temporal_patch_size,
        vision.spatial_merge_size,
    ) != (16, 2, 2)
    {
        return Err(bad(format!(
            "lfm3_vl vision patch {} temporal {} merge {}; the implemented preprocessing is \
             patch 16, temporal 2, merge 2",
            vision.patch_size, vision.temporal_patch_size, vision.spatial_merge_size
        )));
    }
    Ok(vision)
}

/// Reorder a `[rows, T*P*P*C]` patchifier from the reference's `(C, T, P_h,
/// P_w)` column order to this port's `(T, P_h, P_w, C)`. `elem` is the
/// element width in bytes.
fn permute_patchifier_columns(
    name: &str,
    bytes: &[u8],
    rows: usize,
    vision: &VisionConfig,
    elem: usize,
) -> Result<Vec<u8>, Gemma4Error> {
    let (t, p, c) = (
        vision.temporal_patch_size as usize,
        vision.patch_size as usize,
        vision.in_channels as usize,
    );
    let cols = t * p * p * c;
    if bytes.len() != rows * cols * elem {
        return Err(Gemma4Error::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!(
                "{} bytes do not fit [{rows}, {cols}] at {elem} bytes per element",
                bytes.len()
            ),
        });
    }
    let mut out = vec![0u8; bytes.len()];
    for r in 0..rows {
        for ti in 0..t {
            for pi in 0..p * p {
                for ci in 0..c {
                    // Reference column: c * (T*P*P) + t * (P*P) + (ph*P + pw).
                    let old = ci * (t * p * p) + ti * (p * p) + pi;
                    // This port's: t * (P*P*C) + (ph*P + pw) * C + c.
                    let new = ti * (p * p * c) + pi * c + ci;
                    let src = (r * cols + old) * elem;
                    let dst = (r * cols + new) * elem;
                    out[dst..dst + elem].copy_from_slice(&bytes[src..src + elem]);
                }
            }
        }
    }
    Ok(out)
}

/// Repeat each group of `head_dim` rows `n_rep` times in a row, so a
/// `[kv_heads * head_dim, cols]` projection becomes `[kv_heads * n_rep *
/// head_dim, cols]` with head `h` reading KV head `h / n_rep`.
fn expand_kv_heads(bytes: &[u8], row_bytes: usize, head_dim: usize, n_rep: usize) -> Vec<u8> {
    let group = head_dim * row_bytes;
    let mut out = Vec::with_capacity(bytes.len() * n_rep);
    for head in bytes.chunks_exact(group) {
        for _ in 0..n_rep {
            out.extend_from_slice(head);
        }
    }
    out
}

/// One layer's tensor names, bucketed by role.
#[derive(Default)]
struct LayerNames<'a> {
    common: [Option<&'a str>; 5],
    conv: [Option<&'a str>; 3],
    attn: [Option<&'a str>; 6],
}

/// Reads the LFM3-VL tower and projector, converting every tensor to FP16.
///
/// Every name under either prefix must be a known block role, a known
/// resident, or a projector tensor; an unknown one is refused, as the qwen
/// reader refuses one, because a publisher who adds a tensor has changed the
/// tower.
pub fn read_lfm3_vl_vision_entries(
    shards: &Gemma4Shards<'_>,
    vision_bases: &[&str],
    vision: &VisionConfig,
) -> Result<VisionRead, Gemma4Error> {
    if vision.tower != VisionTowerKind::Lfm3Vl || !SUPPORTED_DEPTHS.contains(&vision.depth) {
        return Err(Gemma4Error::Config(format!(
            "not a supported lfm3_vl tower (kind {:?}, depth {})",
            vision.tower, vision.depth
        )));
    }
    let depth = vision.depth as usize;
    let mut layers: Vec<LayerNames<'_>> = (0..depth).map(|_| LayerNames::default()).collect();
    let mut residents: [Option<&str>; 2] = [None; 2];
    let mut projector: [Option<&str>; 6] = [None; 6];

    for &name in vision_bases {
        if let Some(tail) = name.strip_prefix(LFM3_PROJECTOR_PREFIX) {
            let slot = LFM3_PROJECTOR_TENSORS
                .iter()
                .position(|p| *p == tail)
                .ok_or_else(|| {
                    Gemma4Error::UnknownTensor(format!("{name}: not a projector tensor"))
                })?;
            if projector[slot].replace(name).is_some() {
                return Err(Gemma4Error::UnknownTensor(format!("{name}: duplicate")));
            }
            continue;
        }
        let tail = name.strip_prefix(VISION_PREFIX).ok_or_else(|| {
            Gemma4Error::UnknownTensor(format!("{name}: not under an lfm3_vl vision prefix"))
        })?;
        if let Some(rest) = tail.strip_prefix(BLOCK_PATH) {
            let (index, suffix) = rest.split_once('.').ok_or_else(|| {
                Gemma4Error::UnknownTensor(format!("{name}: no block index and suffix"))
            })?;
            let index: usize = index.parse().map_err(|_| {
                Gemma4Error::UnknownTensor(format!("{name}: block index is not a number"))
            })?;
            if index >= depth {
                return Err(Gemma4Error::ShapeMismatch {
                    tensor: name.to_string(),
                    detail: format!("block {index} is past the declared depth of {depth}"),
                });
            }
            let layer = &mut layers[index];
            let placed = if let Some(r) = LFM3_COMMON_ROLES.iter().position(|(_, s)| *s == suffix) {
                layer.common[r].replace(name)
            } else if let Some(r) = LFM3_CONV_ROLES.iter().position(|(_, s)| *s == suffix) {
                layer.conv[r].replace(name)
            } else if let Some(r) = LFM3_ATTN_ROLES.iter().position(|(_, s)| *s == suffix) {
                layer.attn[r].replace(name)
            } else {
                return Err(Gemma4Error::UnknownTensor(format!(
                    "{name}: {suffix} is not a tensor of either lfm3_vl block kind"
                )));
            };
            if placed.is_some() {
                return Err(Gemma4Error::UnknownTensor(format!("{name}: duplicate")));
            }
        } else {
            let slot = LFM3_RESIDENT_TENSORS
                .iter()
                .position(|t| *t == tail)
                .ok_or_else(|| {
                    Gemma4Error::UnknownTensor(format!(
                        "{name}: not a block tensor and not an lfm3_vl tower resident"
                    ))
                })?;
            if residents[slot].replace(name).is_some() {
                return Err(Gemma4Error::UnknownTensor(format!("{name}: duplicate")));
            }
        }
    }

    // Every slot filled or refused by name, and each block exactly one kind.
    for (index, layer) in layers.iter().enumerate() {
        let full = |names: &[Option<&str>], table: &[(&str, &str)]| -> Result<(), Gemma4Error> {
            for (slot, name) in names.iter().enumerate() {
                if name.is_none() {
                    return Err(Gemma4Error::MissingTensor(format!(
                        "{VISION_PREFIX}{BLOCK_PATH}{index}.{}",
                        table[slot].1
                    )));
                }
            }
            Ok(())
        };
        full(&layer.common, &LFM3_COMMON_ROLES)?;
        let (conv, attn) = (
            layer.conv.iter().filter(|n| n.is_some()).count(),
            layer.attn.iter().filter(|n| n.is_some()).count(),
        );
        match (conv, attn) {
            (3, 0) | (0, 6) => {}
            (c, a) if c > 0 && a > 0 => {
                return Err(Gemma4Error::UnknownTensor(format!(
                    "{VISION_PREFIX}{BLOCK_PATH}{index}: carries both conv and attention tensors"
                )))
            }
            (0, 0) => {
                return Err(Gemma4Error::MissingTensor(format!(
                    "{VISION_PREFIX}{BLOCK_PATH}{index}: neither a conv nor an attention block"
                )))
            }
            // A partial set of one kind: `full` names the first missing tensor.
            (c, _) if c > 0 => full(&layer.conv, &LFM3_CONV_ROLES)?,
            _ => full(&layer.attn, &LFM3_ATTN_ROLES)?,
        }
    }
    for (slot, name) in residents.iter().enumerate() {
        if name.is_none() {
            return Err(Gemma4Error::MissingTensor(format!(
                "{VISION_PREFIX}{}",
                LFM3_RESIDENT_TENSORS[slot]
            )));
        }
    }
    for (slot, name) in projector.iter().enumerate() {
        if name.is_none() {
            return Err(Gemma4Error::MissingTensor(format!(
                "{LFM3_PROJECTOR_PREFIX}{}",
                LFM3_PROJECTOR_TENSORS[slot]
            )));
        }
    }

    let hidden = vision.hidden_size as usize;
    let heads = vision.num_heads as usize;
    let head_dim = hidden / heads;
    let mut lossy_conversion = Vec::new();
    let mut entries =
        Vec::with_capacity(LFM3_RESIDENT_TENSORS.len() + LFM3_PROJECTOR_TENSORS.len());

    // Residents: the tower's two, then the projector's six. The patchifier's
    // columns are permuted to this port's patch-row order.
    for (slot, name) in residents.iter().enumerate() {
        let name = name.expect("checked above");
        let t = shards.info(name)?;
        let converted = convert_raw_to_fp16(name, &t.dtype, shards.read(name)?)?;
        if converted.lossy > 0 {
            lossy_conversion.push((name.to_string(), converted.lossy));
        }
        let bytes = if LFM3_RESIDENT_TENSORS[slot] == "patchifier.weight" {
            if t.shape.len() != 2 || t.shape[0] as usize != hidden {
                return Err(Gemma4Error::ShapeMismatch {
                    tensor: name.to_string(),
                    detail: format!("expected [{hidden}, patch values], got {:?}", t.shape),
                });
            }
            permute_patchifier_columns(name, &converted.bytes, hidden, vision, 2)?
        } else {
            converted.bytes
        };
        entries.push(ResidentEntrySpec::Raw(RawTensorSpec {
            name: format!("{VISION_INSTALL_PREFIX}{}", LFM3_RESIDENT_TENSORS[slot]),
            dtype: converted.dtype,
            bytes,
            shape: shape4(&t.shape),
        }));
    }
    for (slot, name) in projector.iter().enumerate() {
        let name = name.expect("checked above");
        let t = shards.info(name)?;
        let converted = convert_raw_to_fp16(name, &t.dtype, shards.read(name)?)?;
        if converted.lossy > 0 {
            lossy_conversion.push((name.to_string(), converted.lossy));
        }
        entries.push(ResidentEntrySpec::Raw(RawTensorSpec {
            name: format!(
                "{VISION_INSTALL_PREFIX}{LFM3_PROJECTOR_INSTALL_LEVEL}{}",
                LFM3_PROJECTOR_TENSORS[slot]
            ),
            dtype: converted.dtype,
            bytes: converted.bytes,
            shape: shape4(&t.shape),
        }));
    }

    let mut experts = Vec::with_capacity(depth);
    let mut widest = 0u64;
    for (index, layer) in layers.iter().enumerate() {
        let is_conv = layer.conv.iter().all(Option::is_some);
        let mut subs = Vec::new();
        let mut used = 0u64;
        let mut push =
            |role: &str, name: &str, expand: Option<(usize, usize)>| -> Result<(), Gemma4Error> {
                let t = shards.info(name)?;
                let converted = convert_raw_to_fp16(name, &t.dtype, shards.read(name)?)?;
                if converted.lossy > 0 {
                    lossy_conversion.push((name.to_string(), converted.lossy));
                }
                let (bytes, shape) = match expand {
                    Some((head_dim, n_rep)) => {
                        let cols = *t.shape.get(1).ok_or_else(|| Gemma4Error::ShapeMismatch {
                            tensor: name.to_string(),
                            detail: "a grouped K/V projection must be 2-D".into(),
                        })? as usize;
                        let rows = t.shape[0] as usize;
                        if rows % head_dim != 0 {
                            return Err(Gemma4Error::ShapeMismatch {
                                tensor: name.to_string(),
                                detail: format!("{rows} rows are not whole {head_dim}-row heads"),
                            });
                        }
                        (
                            expand_kv_heads(&converted.bytes, cols * 2, head_dim, n_rep),
                            vec![(rows * n_rep) as u64, cols as u64],
                        )
                    }
                    None => (converted.bytes, t.shape.clone()),
                };
                used += bytes.len() as u64;
                subs.push(SubTensor {
                    role: role.to_string(),
                    bytes,
                    dtype: "fp16".to_string(),
                    shape,
                });
                Ok(())
            };
        for (slot, (role, _)) in LFM3_COMMON_ROLES.iter().enumerate() {
            push(role, layer.common[slot].expect("checked"), None)?;
        }
        if is_conv {
            for (slot, (role, _)) in LFM3_CONV_ROLES.iter().enumerate() {
                push(role, layer.conv[slot].expect("checked"), None)?;
            }
        } else {
            for (slot, (role, _)) in LFM3_ATTN_ROLES.iter().enumerate() {
                let name = layer.attn[slot].expect("checked");
                let expand = if *role == "k_w" || *role == "v_w" {
                    let kv_rows = shards.info(name)?.shape[0] as usize;
                    if kv_rows == 0 || kv_rows % head_dim != 0 || (heads * head_dim) % kv_rows != 0
                    {
                        return Err(Gemma4Error::ShapeMismatch {
                            tensor: name.to_string(),
                            detail: format!(
                                "{kv_rows} rows do not make a divisor of {heads} {head_dim}-wide heads"
                            ),
                        });
                    }
                    Some((head_dim, heads * head_dim / kv_rows))
                } else {
                    None
                };
                push(role, name, expand)?;
            }
        }
        let _ = index;
        widest = widest.max(used);
        experts.push(ExpertBlob {
            expert: index,
            sub_tensors: subs,
        });
    }

    Ok(VisionRead {
        entries,
        blocks: LayerBlobs { layer: 0, experts },
        block_stride: widest.div_ceil(GTURBO_PAGE_BYTES) * GTURBO_PAGE_BYTES,
        lossy_conversion,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ranged_download::MemoryRangeSource;
    use crate::safetensors_header::{parse_header, DEFAULT_MAX_HEADER_BYTES};
    use crate::synthetic_tensors::{assemble_safetensors, bf16_vector, Tensor};

    /// The real checkpoint's config, trimmed to the keys this parser reads,
    /// with values from `LiquidAI/LFM3-VL-2B-MLX-4bit@4146d7fa`.
    fn real_config() -> serde_json::Value {
        let layer_types = [
            "conv",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
        ];
        serde_json::json!({
            "image_token_id": 125030, "video_token_id": 125029,
            "vision_start_token_id": 125009, "vision_end_token_id": 125010,
            "downsample_factor": 2, "projector_hidden_size": 4096,
            "projector_hidden_act": "gelu", "projector_bias": true,
            "projector_layer_norm_eps": 1e-6,
            "text_config": {"hidden_size": 2048},
            "vision_config": {
                "hidden_size": 1024, "intermediate_size": 2560, "num_hidden_layers": 14,
                "num_attention_heads": 16, "num_key_value_heads": 8, "in_channels": 3,
                "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2,
                "conv_kernel_size": 3, "norm_eps": 1e-5,
                "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "axial"},
                "layer_types": layer_types
            }
        })
    }

    #[test]
    fn the_real_config_parses_to_the_lfm3_vl_tower() {
        let v = parse_lfm3_vl_vision_config(&real_config().to_string()).unwrap();
        assert_eq!(v.tower, VisionTowerKind::Lfm3Vl);
        assert_eq!(
            (v.depth, v.hidden_size, v.num_heads, v.head_dim()),
            (14, 1024, 16, 64)
        );
        assert_eq!((v.out_hidden_size, v.patches_per_token()), (2048, 4));
        assert_eq!(
            (
                v.vision_start_token_id,
                v.vision_end_token_id,
                v.image_token_id
            ),
            (125009, 125010, 125030)
        );
    }

    #[test]
    fn no_vision_config_is_no_tower() {
        let mut c = real_config();
        c.as_object_mut().unwrap().remove("vision_config");
        assert_eq!(
            parse_lfm3_vl_vision_config(&c.to_string()).unwrap(),
            VisionConfig::NONE
        );
    }

    /// Each value the tower kind implies is refused when it moves.
    #[test]
    fn a_changed_tower_is_refused() {
        for (key, value) in [
            (
                "rope_parameters",
                serde_json::json!({"rope_theta": 10000.0, "rope_type": "axial"}),
            ),
            (
                "rope_parameters",
                serde_json::json!({"rope_theta": 1000000.0, "rope_type": "default"}),
            ),
            ("norm_eps", serde_json::json!(1e-6)),
            ("conv_kernel_size", serde_json::json!(5)),
            ("num_key_value_heads", serde_json::json!(5)),
            ("num_hidden_layers", serde_json::json!(16)),
            ("patch_size", serde_json::json!(14)),
        ] {
            let mut c = real_config();
            c["vision_config"][key] = value.clone();
            assert!(
                parse_lfm3_vl_vision_config(&c.to_string()).is_err(),
                "{key} = {value} must be refused"
            );
        }
        for (key, value) in [
            ("projector_hidden_act", serde_json::json!("silu")),
            ("projector_bias", serde_json::json!(false)),
            ("projector_layer_norm_eps", serde_json::json!(1e-5)),
            ("projector_hidden_size", serde_json::json!(2048)),
            ("downsample_factor", serde_json::json!(3)),
        ] {
            let mut c = real_config();
            c[key] = value.clone();
            assert!(
                parse_lfm3_vl_vision_config(&c.to_string()).is_err(),
                "{key} = {value} must be refused"
            );
        }
        let mut c = real_config();
        c["vision_config"]["layer_types"][3] = serde_json::json!("sliding_attention");
        assert!(parse_lfm3_vl_vision_config(&c.to_string()).is_err());
    }

    #[test]
    fn the_patchifier_permutation_moves_each_column_to_t_p_p_c_order() {
        let vision = parse_lfm3_vl_vision_config(&real_config().to_string()).unwrap();
        let (t, p, c) = (2usize, 16usize, 3usize);
        let cols = t * p * p * c;
        // One row whose value at reference column k is k.
        let old: Vec<u8> = (0..cols as u16).flat_map(|k| k.to_le_bytes()).collect();
        let new = permute_patchifier_columns("w", &old, 1, &vision, 2).unwrap();
        let at = |i: usize| u16::from_le_bytes([new[2 * i], new[2 * i + 1]]) as usize;
        // Port column (t=1, ph=2, pw=3, c=2) reads reference column
        // c*(T*P*P) + t*P*P + (ph*P + pw).
        let (ti, ci) = (1usize, 2usize);
        let port = ti * (p * p * c) + (2 * p + 3) * c + ci;
        assert_eq!(at(port), ci * (t * p * p) + ti * (p * p) + (2 * p + 3));
        // A bijection: every reference column lands exactly once.
        let mut seen: Vec<usize> = (0..cols).map(at).collect();
        seen.sort_unstable();
        assert!(seen.iter().copied().eq(0..cols));
    }

    #[test]
    fn kv_heads_repeat_consecutively() {
        // 2 KV heads of 3 rows, 1 byte per column, 2 columns.
        let kv: Vec<u8> = (0..12).collect();
        let out = expand_kv_heads(&kv, 2, 3, 2);
        assert_eq!(
            out,
            [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11]
        );
    }

    /// A tiny tower at the real depth and layer pattern with every name the
    /// real checkpoint uses. Returns the config, the names and the assembled
    /// safetensors blob.
    fn tiny_checkpoint(
        drop: Option<&str>,
        extra: Option<&str>,
    ) -> (VisionConfig, Vec<String>, Vec<u8>) {
        let (hid, inter, text, heads, kv, hd) = (8usize, 12usize, 16usize, 2usize, 1usize, 4usize);
        let mut vision = parse_lfm3_vl_vision_config(&real_config().to_string()).unwrap();
        vision.hidden_size = hid as i64;
        vision.num_heads = heads as i64;
        vision.intermediate_size = inter as i64;
        vision.out_hidden_size = text as i64;
        let shaped = |name: String, shape: Vec<u64>, seed: u64| {
            let n: u64 = shape.iter().product();
            let mut t = bf16_vector(&name, n as usize, 0.5, seed);
            t.shape = shape;
            t
        };
        let layer_is_attn = [2usize, 4, 6, 8, 10, 12];
        let mut ts: Vec<Tensor> = Vec::new();
        for b in 0..14usize {
            let base = format!("{VISION_PREFIX}{BLOCK_PATH}{b}.");
            let (h, i) = (hid as u64, inter as u64);
            for (suffix, shape) in [
                ("operator_norm.weight", vec![h]),
                ("ffn_norm.weight", vec![h]),
                ("feed_forward.w1.weight", vec![i, h]),
                ("feed_forward.w3.weight", vec![i, h]),
                ("feed_forward.w2.weight", vec![h, i]),
            ] {
                ts.push(shaped(format!("{base}{suffix}"), shape, b as u64));
            }
            if layer_is_attn.contains(&b) {
                for (suffix, shape) in [
                    ("self_attn.q_proj.weight", vec![h, h]),
                    ("self_attn.k_proj.weight", vec![(kv * hd) as u64, h]),
                    ("self_attn.v_proj.weight", vec![(kv * hd) as u64, h]),
                    ("self_attn.out_proj.weight", vec![h, h]),
                    ("self_attn.q_layernorm.weight", vec![hd as u64]),
                    ("self_attn.k_layernorm.weight", vec![hd as u64]),
                ] {
                    ts.push(shaped(format!("{base}{suffix}"), shape, 50 + b as u64));
                }
            } else {
                for (suffix, shape) in [
                    ("conv.in_proj.weight", vec![3 * h, h]),
                    ("conv.spatial_conv.weight", vec![h, 3, 3, 1]),
                    ("conv.out_proj.weight", vec![h, h]),
                ] {
                    ts.push(shaped(format!("{base}{suffix}"), shape, 80 + b as u64));
                }
            }
        }
        ts.push(shaped(
            format!("{VISION_PREFIX}patchifier.weight"),
            vec![hid as u64, 1536],
            90,
        ));
        ts.push(shaped(
            format!("{VISION_PREFIX}embedding_norm.weight"),
            vec![hid as u64],
            91,
        ));
        let pin = (hid * 4) as u64;
        for (suffix, shape) in [
            ("layer_norm.weight", vec![hid as u64]),
            ("layer_norm.bias", vec![hid as u64]),
            ("linear_1.weight", vec![40, pin]),
            ("linear_1.bias", vec![40]),
            ("linear_2.weight", vec![text as u64, 40]),
            ("linear_2.bias", vec![text as u64]),
        ] {
            ts.push(shaped(
                format!("{LFM3_PROJECTOR_PREFIX}{suffix}"),
                shape,
                95,
            ));
        }
        if let Some(name) = extra {
            ts.push(shaped(name.to_string(), vec![hid as u64], 99));
        }
        if let Some(d) = drop {
            ts.retain(|t| t.name != d);
        }
        let names = ts.iter().map(|t| t.name.clone()).collect();
        (vision, names, assemble_safetensors(&ts))
    }

    fn read(drop: Option<&str>, extra: Option<&str>) -> Result<VisionRead, Gemma4Error> {
        let (vision, names, blob) = tiny_checkpoint(drop, extra);
        let header = parse_header(&blob, DEFAULT_MAX_HEADER_BYTES).unwrap();
        let source = MemoryRangeSource::new(&blob);
        let shards = Gemma4Shards::single(&header, &source);
        let bases: Vec<&str> = names.iter().map(String::as_str).collect();
        // Through the shared entry point, so the tower-kind dispatch is what
        // is tested.
        super::super::vision::read_vision_entries(&shards, &bases, &vision)
    }

    #[test]
    fn a_complete_tower_reads_into_per_kind_blocks() {
        let r = read(None, None).unwrap();
        assert_eq!(r.blocks.experts.len(), 14);
        let roles = |i: usize| -> Vec<&str> {
            r.blocks.experts[i]
                .sub_tensors
                .iter()
                .map(|t| t.role.as_str())
                .collect()
        };
        assert_eq!(
            roles(0),
            ["op_norm", "ffn_norm", "w1", "w3", "w2", "in_proj", "conv_w", "out_proj"]
        );
        assert_eq!(
            roles(2),
            [
                "op_norm", "ffn_norm", "w1", "w3", "w2", "q_w", "k_w", "v_w", "o_w", "q_norm",
                "k_norm"
            ]
        );
        // K and V were expanded to the full 2 heads x 4 rows.
        let k = r.blocks.experts[2]
            .sub_tensors
            .iter()
            .find(|t| t.role == "k_w")
            .unwrap();
        assert_eq!(k.shape, vec![8, 8]);
        assert_eq!(k.bytes.len(), 8 * 8 * 2);
        let names: Vec<String> = r
            .entries
            .iter()
            .map(|e| match e {
                ResidentEntrySpec::Raw(t) => t.name.clone(),
                _ => panic!("vision residents are raw FP16"),
            })
            .collect();
        assert_eq!(
            names,
            [
                "vision.patchifier.weight",
                "vision.embedding_norm.weight",
                "vision.projector.layer_norm.weight",
                "vision.projector.layer_norm.bias",
                "vision.projector.linear_1.weight",
                "vision.projector.linear_1.bias",
                "vision.projector.linear_2.weight",
                "vision.projector.linear_2.bias",
            ]
        );
        assert_eq!(r.block_stride % GTURBO_PAGE_BYTES, 0);
    }

    #[test]
    fn a_missing_unknown_or_mixed_tensor_is_refused_by_name() {
        let missing = format!("{VISION_PREFIX}{BLOCK_PATH}4.self_attn.k_layernorm.weight");
        let err = read(Some(&missing), None).err().expect("missing role");
        assert!(
            err.to_string().contains("4.self_attn.k_layernorm.weight"),
            "{err}"
        );
        let err = read(Some("multi_modal_projector.linear_2.bias"), None)
            .err()
            .expect("missing projector tensor");
        assert!(err.to_string().contains("linear_2.bias"), "{err}");
        let err = read(None, Some("vision_tower.layers.0.conv.extra.weight"))
            .err()
            .expect("unknown role");
        assert!(err.to_string().contains("conv.extra"), "{err}");
        // A conv block that also carries one attention tensor is two kinds.
        let err = read(None, Some("vision_tower.layers.0.self_attn.q_proj.weight"))
            .err()
            .expect("mixed block");
        assert!(err.to_string().contains("both conv and attention"), "{err}");
    }
}
