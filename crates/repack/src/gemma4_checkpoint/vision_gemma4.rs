//! The Gemma 4 vision tower's config and ingest.
//!
//! A sibling of [`super::vision`], not a variant of it. The two towers share
//! their shapes and nothing else (`docs/GEMMA4_VISION_PHASE0.md`): different
//! tensor names, thirteen roles per block instead of twelve, no biases, a
//! rank-3 position table instead of a learned grid, and a 4-bit projector
//! under a second prefix. What IS shared is the destination: blocks stream
//! from `packed_vision/` exactly as the qwen tower's do, residents land under
//! [`VISION_INSTALL_PREFIX`] at FP16, and the FP16 overflow refusal applies.
//!
//! **The projector is dequantized here.** It ships MLX affine 4-bit, group 64
//! (`[2816, 1152]` packed as `[2816, 144]` u32). Keeping it quantized would
//! give the tower a second weight format for one 6.5 MiB matrix used once per
//! image. FP16 is exact for every value the 4-bit grid can produce at these
//! scales, and the overflow refusal still guards it.

use model_io::{VisionConfig, VisionTowerKind};

use super::classify::{VISION_INSTALL_PREFIX, VISION_PREFIX};
use super::config::Gemma4Error;
use super::narrow::convert_raw_to_fp16;
use super::shards::{shape4, Gemma4Shards, GTURBO_PAGE_BYTES};
use super::vision::VisionRead;
use crate::gturbo_writer::{ExpertBlob, LayerBlobs, SubTensor};
use crate::resident_writer::{RawTensorSpec, ResidentEntrySpec};

/// The projector's checkpoint prefix, beside the tower's rather than under it.
pub const EMBED_VISION_PREFIX: &str = "embed_vision.";

/// Where a block's tensors live under [`VISION_PREFIX`].
const BLOCK_PATH: &str = "encoder.layers.";

/// One block's thirteen sub-tensors: `(role in the blob, suffix in the
/// checkpoint)`, in forward order. Read off the real checkpoint header.
pub const GEMMA4_BLOCK_ROLES: [(&str, &str); 13] = [
    ("ln_in", "input_layernorm.weight"),
    ("q_w", "self_attn.q_proj.linear.weight"),
    ("k_w", "self_attn.k_proj.linear.weight"),
    ("v_w", "self_attn.v_proj.linear.weight"),
    ("q_norm", "self_attn.q_norm.weight"),
    ("k_norm", "self_attn.k_norm.weight"),
    ("o_w", "self_attn.o_proj.linear.weight"),
    ("ln_post_attn", "post_attention_layernorm.weight"),
    ("ln_pre_ffw", "pre_feedforward_layernorm.weight"),
    ("gate_w", "mlp.gate_proj.linear.weight"),
    ("up_w", "mlp.up_proj.linear.weight"),
    ("down_w", "mlp.down_proj.linear.weight"),
    ("ln_post_ffw", "post_feedforward_layernorm.weight"),
];

/// The tower's non-block tensors, under [`VISION_PREFIX`], resident at FP16.
pub const GEMMA4_RESIDENT_TENSORS: [&str; 4] = [
    "patch_embedder.input_proj.weight",
    "patch_embedder.position_embedding_table",
    "std_bias",
    "std_scale",
];

/// The projector's three quantized parts, under [`EMBED_VISION_PREFIX`].
const PROJECTOR: [&str; 3] = [
    "embedding_projection.weight",
    "embedding_projection.scales",
    "embedding_projection.biases",
];

/// The install name of the dequantized projector, `[text_hidden, hidden]`.
pub const GEMMA4_PROJECTOR_INSTALL_NAME: &str = "embedding_projection.weight";

/// The only depth a published Gemma 4 tower declares.
const SUPPORTED_DEPTHS: [i64; 1] = [27];

/// Parse a Gemma 4 `config.json`'s vision half into a [`VisionConfig`].
///
/// Absent `vision_config` is [`VisionConfig::NONE`]. Everything the runtime
/// does NOT store is checked here instead and refused when it differs from
/// the one tower this port implements: the axial rope at theta 100, the
/// standardize step, no clipped linears, full (non-grouped) attention, the
/// tanh GELU, and RMS epsilon 1e-6. Each is a value the tower kind implies,
/// so a checkpoint that changes one has changed the tower.
pub fn parse_gemma4_vision_config(json: &str) -> Result<VisionConfig, Gemma4Error> {
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
    // The runtime implements ONE trunk mask for image spans: bidirectional on
    // sliding layers, causal on full ones (`use_bidirectional_attention:
    // "vision"`). A checkpoint declaring anything else needs a different
    // mask, and the hard-coded one would answer fluently and wrongly.
    match root
        .get("text_config")
        .and_then(|t| t.get("use_bidirectional_attention"))
        .and_then(serde_json::Value::as_str)
    {
        Some("vision") => {}
        other => {
            return Err(bad(format!(
                "text_config.use_bidirectional_attention is {other:?}; the implemented image \
                 mask is the \"vision\" one"
            )))
        }
    }
    let text_hidden = root
        .get("text_config")
        .and_then(|t| t.get("hidden_size"))
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| bad("missing text_config.hidden_size".into()))?;

    let hidden = vi("hidden_size")?;
    let heads = vi("num_attention_heads")?;
    let head_dim = vi("head_dim")?;
    if heads <= 0 || hidden != heads * head_dim {
        return Err(bad(format!(
            "vision hidden_size {hidden} is not num_attention_heads {heads} x head_dim \
             {head_dim}; the runtime derives head_dim and would read the wrong width"
        )));
    }
    if head_dim % 4 != 0 {
        return Err(bad(format!(
            "vision head_dim {head_dim} is not a multiple of 4; the axial rope splits each \
             head into two rotated halves"
        )));
    }
    if vi("num_key_value_heads")? != heads {
        return Err(bad("grouped vision attention is not implemented".into()));
    }
    let rope = vc.get("rope_parameters");
    let theta = rope
        .and_then(|r| r.get("rope_theta"))
        .and_then(serde_json::Value::as_f64);
    if theta != Some(100.0) {
        return Err(bad(format!(
            "vision rope_theta {theta:?}; the implemented tower is theta 100"
        )));
    }
    // `"default"` is what the published config says; the reference forces
    // the vision rope to axial regardless (Phase 0 trap 3). Anything else is
    // a rope this port has not read.
    match rope
        .and_then(|r| r.get("rope_type"))
        .and_then(serde_json::Value::as_str)
    {
        Some("default" | "axial") => {}
        other => return Err(bad(format!("vision rope_type {other:?} is not axial"))),
    }
    if vc.get("standardize").and_then(serde_json::Value::as_bool) != Some(true) {
        return Err(bad("vision standardize must be true".into()));
    }
    if vc
        .get("use_clipped_linears")
        .and_then(serde_json::Value::as_bool)
        == Some(true)
    {
        return Err(bad("clipped vision linears are not implemented".into()));
    }
    match vc
        .get("hidden_activation")
        .and_then(serde_json::Value::as_str)
    {
        Some("gelu_pytorch_tanh") => {}
        other => return Err(bad(format!("vision hidden_activation {other:?}"))),
    }
    if vc.get("rms_norm_eps").and_then(serde_json::Value::as_f64) != Some(1e-6) {
        return Err(bad("vision rms_norm_eps must be 1e-6".into()));
    }

    let vision = VisionConfig {
        depth: vi("num_hidden_layers")?,
        hidden_size: hidden,
        intermediate_size: vi("intermediate_size")?,
        num_heads: heads,
        patch_size: vi("patch_size")?,
        temporal_patch_size: 1,
        in_channels: 3,
        spatial_merge_size: vi("pooling_kernel_size")?,
        num_position_embeddings: vi("position_embedding_size")?,
        out_hidden_size: text_hidden,
        mrope_section: [0, 0, 0],
        deepstack_visual_indexes: Vec::new(),
        vision_start_token_id: ri("boi_token_id")?,
        vision_end_token_id: ri("eoi_token_id")?,
        image_token_id: ri("image_token_id")?,
        video_token_id: ri("video_token_id")?,
        tower: VisionTowerKind::Gemma4,
    };
    if !SUPPORTED_DEPTHS.contains(&vision.depth) {
        return Err(bad(format!(
            "gemma4 vision depth {} is unsupported; expected one of {SUPPORTED_DEPTHS:?}",
            vision.depth
        )));
    }
    if (vision.patch_size, vision.spatial_merge_size) != (16, 3) {
        return Err(bad(format!(
            "gemma4 vision patch {} pool {}; the implemented preprocessing is patch 16 pool 3",
            vision.patch_size, vision.spatial_merge_size
        )));
    }
    Ok(vision)
}

/// Dequantize an MLX affine 4-bit matrix to FP16 bytes.
///
/// `packed` is the raw U32 tensor, which on a little-endian host is the
/// nibble order `compute::quant` uses (low nibble first), so its bytes are an
/// `Int4AffineRow`'s `packed` row by row. `scales` and `biases` are BF16, one
/// per 64 values.
fn dequant_projector(
    name: &str,
    packed: &[u8],
    scales: &[u8],
    biases: &[u8],
    rows: usize,
    cols: usize,
) -> Result<Vec<u8>, Gemma4Error> {
    let group = compute::quant::GROUP_SIZE;
    let mismatch = |detail: String| Gemma4Error::ShapeMismatch {
        tensor: name.to_string(),
        detail,
    };
    if cols % group != 0 {
        return Err(mismatch(format!(
            "{cols} columns are not whole groups of {group}"
        )));
    }
    let groups = cols / group;
    if packed.len() != rows * cols / 2
        || scales.len() != rows * groups * 2
        || biases.len() != rows * groups * 2
    {
        return Err(mismatch(format!(
            "packed {} / scales {} / biases {} bytes do not fit [{rows}, {cols}] at 4 bits",
            packed.len(),
            scales.len(),
            biases.len()
        )));
    }
    let bf16 = |b: &[u8], i: usize| u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
    let mut out = Vec::with_capacity(rows * cols * 2);
    for r in 0..rows {
        let row = compute::quant::Int4AffineRow {
            packed: packed[r * cols / 2..(r + 1) * cols / 2].to_vec(),
            scales: (0..groups).map(|g| bf16(scales, r * groups + g)).collect(),
            biases: (0..groups).map(|g| bf16(biases, r * groups + g)).collect(),
        };
        for v in compute::quant::dequantize_int4_affine(&row, cols) {
            if !v.is_finite() || v.abs() > 65_504.0 {
                return Err(mismatch(format!("value {v} does not fit FP16")));
            }
            out.extend_from_slice(&compute::f32_to_f16(v).to_le_bytes());
        }
    }
    Ok(out)
}

/// Reads the Gemma 4 tower and projector, converting every tensor to FP16.
///
/// Every name under either prefix must be a known block role, a known
/// resident, or a projector part. An unknown name is refused, as the qwen
/// reader refuses one, because a publisher who adds a tensor has changed the
/// tower.
pub fn read_gemma4_vision_entries(
    shards: &Gemma4Shards<'_>,
    vision_bases: &[&str],
    vision: &VisionConfig,
) -> Result<VisionRead, Gemma4Error> {
    if vision.tower != VisionTowerKind::Gemma4 || !SUPPORTED_DEPTHS.contains(&vision.depth) {
        return Err(Gemma4Error::Config(format!(
            "not a supported gemma4 tower (kind {:?}, depth {})",
            vision.tower, vision.depth
        )));
    }
    let depth = vision.depth as usize;
    let mut blocks: Vec<[Option<&str>; 13]> = vec![[None; 13]; depth];
    let mut residents: [Option<&str>; 4] = [None; 4];
    let mut projector: [Option<&str>; 3] = [None; 3];

    for &name in vision_bases {
        if let Some(tail) = name.strip_prefix(EMBED_VISION_PREFIX) {
            let slot = PROJECTOR.iter().position(|p| *p == tail).ok_or_else(|| {
                Gemma4Error::UnknownTensor(format!("{name}: not a projector tensor"))
            })?;
            if projector[slot].replace(name).is_some() {
                return Err(Gemma4Error::UnknownTensor(format!("{name}: duplicate")));
            }
            continue;
        }
        let tail = name.strip_prefix(VISION_PREFIX).ok_or_else(|| {
            Gemma4Error::UnknownTensor(format!("{name}: not under a gemma4 vision prefix"))
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
            let role = GEMMA4_BLOCK_ROLES
                .iter()
                .position(|(_, s)| *s == suffix)
                .ok_or_else(|| {
                    Gemma4Error::UnknownTensor(format!(
                        "{name}: {suffix} is not one of the thirteen per-block tensors"
                    ))
                })?;
            if blocks[index][role].replace(name).is_some() {
                return Err(Gemma4Error::UnknownTensor(format!("{name}: duplicate")));
            }
        } else {
            let slot = GEMMA4_RESIDENT_TENSORS
                .iter()
                .position(|t| *t == tail)
                .ok_or_else(|| {
                    Gemma4Error::UnknownTensor(format!(
                        "{name}: not a block tensor and not a gemma4 tower resident"
                    ))
                })?;
            if residents[slot].replace(name).is_some() {
                return Err(Gemma4Error::UnknownTensor(format!("{name}: duplicate")));
            }
        }
    }

    // Every slot filled, or refused by name: a missing tensor would stream
    // as zeros and the tower would run with a sublayer silently off.
    for (index, roles) in blocks.iter().enumerate() {
        for (role, name) in roles.iter().enumerate() {
            if name.is_none() {
                return Err(Gemma4Error::MissingTensor(format!(
                    "{VISION_PREFIX}{BLOCK_PATH}{index}.{}",
                    GEMMA4_BLOCK_ROLES[role].1
                )));
            }
        }
    }
    for (slot, name) in residents.iter().enumerate() {
        if name.is_none() {
            return Err(Gemma4Error::MissingTensor(format!(
                "{VISION_PREFIX}{}",
                GEMMA4_RESIDENT_TENSORS[slot]
            )));
        }
    }
    // The walk (`orchestrate::classify_all`) hands over the `.weight` alone:
    // it skips every `.scales`/`.biases` name and leaves a quantized
    // tensor's reader to fetch its companions. So they are resolved here from
    // the shards, and taken from `vision_bases` only if a caller passed them.
    let projector: Vec<String> = projector
        .iter()
        .enumerate()
        .map(|(slot, name)| {
            let full = format!("{EMBED_VISION_PREFIX}{}", PROJECTOR[slot]);
            match name {
                Some(n) => Ok(n.to_string()),
                None if slot > 0 && projector[0].is_some() && shards.contains(&full) => Ok(full),
                None => Err(Gemma4Error::MissingTensor(full)),
            }
        })
        .collect::<Result<_, _>>()?;

    let mut lossy_conversion = Vec::new();
    let mut entries = Vec::with_capacity(GEMMA4_RESIDENT_TENSORS.len() + 1);
    for (slot, name) in residents.iter().enumerate() {
        let name = name.expect("checked above");
        let t = shards.info(name)?;
        let converted = convert_raw_to_fp16(name, &t.dtype, shards.read(name)?)?;
        if converted.lossy > 0 {
            lossy_conversion.push((name.to_string(), converted.lossy));
        }
        entries.push(ResidentEntrySpec::Raw(RawTensorSpec {
            name: format!("{VISION_INSTALL_PREFIX}{}", GEMMA4_RESIDENT_TENSORS[slot]),
            dtype: converted.dtype,
            bytes: converted.bytes,
            shape: shape4(&t.shape),
        }));
    }

    let (w, s, b) = (
        projector[0].as_str(),
        projector[1].as_str(),
        projector[2].as_str(),
    );
    let wi = shards.info(w)?;
    if wi.dtype != "U32" || wi.shape.len() != 2 {
        return Err(Gemma4Error::ShapeMismatch {
            tensor: w.to_string(),
            detail: format!(
                "expected a 2-D U32 packed matrix, got {} {:?}",
                wi.dtype, wi.shape
            ),
        });
    }
    let (rows, cols) = (wi.shape[0] as usize, wi.shape[1] as usize * 8);
    if rows as i64 != vision.out_hidden_size || cols as i64 != vision.hidden_size {
        return Err(Gemma4Error::ShapeMismatch {
            tensor: w.to_string(),
            detail: format!(
                "unpacks to [{rows}, {cols}], expected [{}, {}]",
                vision.out_hidden_size, vision.hidden_size
            ),
        });
    }
    for part in [s, b] {
        if shards.info(part)?.dtype != "BF16" {
            return Err(Gemma4Error::ShapeMismatch {
                tensor: part.to_string(),
                detail: "projector scales and biases must be BF16".into(),
            });
        }
    }
    let bytes = dequant_projector(
        w,
        &shards.read(w)?,
        &shards.read(s)?,
        &shards.read(b)?,
        rows,
        cols,
    )?;
    entries.push(ResidentEntrySpec::Raw(RawTensorSpec {
        name: format!("{VISION_INSTALL_PREFIX}{GEMMA4_PROJECTOR_INSTALL_NAME}"),
        dtype: crate::DTYPE_FP16,
        bytes,
        shape: (rows as u32, cols as u32, 0, 0),
    }));

    let mut experts = Vec::with_capacity(depth);
    let mut widest = 0u64;
    for (index, roles) in blocks.iter().enumerate() {
        let mut sub_tensors = Vec::with_capacity(GEMMA4_BLOCK_ROLES.len());
        let mut used = 0u64;
        for (role, name) in roles.iter().enumerate() {
            let name = name.expect("checked above");
            let t = shards.info(name)?;
            let converted = convert_raw_to_fp16(name, &t.dtype, shards.read(name)?)?;
            if converted.lossy > 0 {
                lossy_conversion.push((name.to_string(), converted.lossy));
            }
            used += converted.bytes.len() as u64;
            sub_tensors.push(SubTensor {
                role: GEMMA4_BLOCK_ROLES[role].0.to_string(),
                bytes: converted.bytes,
                dtype: "fp16".to_string(),
                shape: t.shape.clone(),
            });
        }
        widest = widest.max(used);
        experts.push(ExpertBlob {
            expert: index,
            sub_tensors,
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

    /// The real checkpoint's config, trimmed to the keys this parser reads,
    /// with values from `mlx-community/gemma-4-26b-a4b-it-4bit@0d77464e`.
    fn real_config() -> serde_json::Value {
        serde_json::json!({
            "boi_token_id": 255999, "eoi_token_id": 258882,
            "image_token_id": 258880, "video_token_id": 258884,
            "text_config": {"hidden_size": 2816, "use_bidirectional_attention": "vision"},
            "vision_config": {
                "hidden_size": 1152, "intermediate_size": 4304, "num_hidden_layers": 27,
                "num_attention_heads": 16, "num_key_value_heads": 16, "head_dim": 72,
                "patch_size": 16, "pooling_kernel_size": 3, "position_embedding_size": 10240,
                "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"},
                "standardize": true, "use_clipped_linears": false,
                "hidden_activation": "gelu_pytorch_tanh", "rms_norm_eps": 1e-6
            }
        })
    }

    #[test]
    fn the_real_config_parses_to_the_gemma4_tower() {
        let v = parse_gemma4_vision_config(&real_config().to_string()).unwrap();
        assert_eq!(v.tower, VisionTowerKind::Gemma4);
        assert_eq!(
            (v.depth, v.hidden_size, v.num_heads, v.head_dim()),
            (27, 1152, 16, 72)
        );
        assert_eq!(v.out_hidden_size, 2816);
        assert_eq!(v.patches_per_token(), 9);
        assert_eq!(v.num_position_embeddings, 10240);
        assert_eq!(
            (
                v.vision_start_token_id,
                v.vision_end_token_id,
                v.image_token_id
            ),
            (255999, 258882, 258880)
        );
    }

    /// The runtime's image mask is the `"vision"` one; any other declared
    /// mask is refused at intake rather than run with the wrong one.
    #[test]
    fn a_trunk_mask_other_than_vision_is_refused() {
        for value in [serde_json::json!(null), serde_json::json!("all")] {
            let mut c = real_config();
            c["text_config"]["use_bidirectional_attention"] = value.clone();
            assert!(
                parse_gemma4_vision_config(&c.to_string()).is_err(),
                "{value} must be refused"
            );
        }
    }

    #[test]
    fn no_vision_config_is_no_tower() {
        let mut c = real_config();
        c.as_object_mut().unwrap().remove("vision_config");
        assert_eq!(
            parse_gemma4_vision_config(&c.to_string()).unwrap(),
            VisionConfig::NONE
        );
    }

    /// Each value the tower kind implies is refused when it moves.
    #[test]
    fn a_changed_tower_is_refused() {
        for (key, value) in [
            (
                "rope_parameters",
                serde_json::json!({"rope_theta": 10000.0, "rope_type": "default"}),
            ),
            (
                "rope_parameters",
                serde_json::json!({"rope_theta": 100.0, "rope_type": "yarn"}),
            ),
            ("standardize", serde_json::json!(false)),
            ("use_clipped_linears", serde_json::json!(true)),
            ("num_key_value_heads", serde_json::json!(8)),
            ("hidden_activation", serde_json::json!("gelu")),
            ("head_dim", serde_json::json!(64)),
            ("num_hidden_layers", serde_json::json!(16)),
            ("pooling_kernel_size", serde_json::json!(2)),
        ] {
            let mut c = real_config();
            c["vision_config"][key] = value.clone();
            assert!(
                parse_gemma4_vision_config(&c.to_string()).is_err(),
                "{key} = {value} must be refused"
            );
        }
    }

    /// A tiny tower at the real depth, every name the real checkpoint uses,
    /// read through the shard reader. Returns the config, the names, and the
    /// assembled blob.
    fn tiny_checkpoint(
        drop: Option<&str>,
        extra: Option<&str>,
    ) -> (VisionConfig, Vec<String>, Vec<u8>) {
        use crate::synthetic_tensors::{assemble_safetensors, bf16_vector, Tensor};
        let (hid, inter, text) = (8usize, 12usize, 64usize);
        let mut vision = parse_gemma4_vision_config(&real_config().to_string()).unwrap();
        vision.hidden_size = hid as i64;
        vision.num_heads = 2;
        vision.intermediate_size = inter as i64;
        vision.out_hidden_size = text as i64;
        vision.num_position_embeddings = 4;
        let shaped = |name: String, shape: Vec<u64>, seed: u64| {
            let n: u64 = shape.iter().product();
            let mut t = bf16_vector(&name, n as usize, 0.5, seed);
            t.shape = shape;
            t
        };
        let mut ts: Vec<Tensor> = Vec::new();
        for b in 0..27u64 {
            for (_, suffix) in GEMMA4_BLOCK_ROLES {
                let shape = match suffix {
                    s if s.ends_with("norm.weight")
                        && s.contains("_norm")
                        && s.starts_with("self_attn") =>
                    {
                        vec![4]
                    }
                    s if s.ends_with("layernorm.weight") => vec![hid as u64],
                    "mlp.down_proj.linear.weight" => vec![hid as u64, inter as u64],
                    s if s.starts_with("mlp.") => vec![inter as u64, hid as u64],
                    _ => vec![hid as u64, hid as u64],
                };
                ts.push(shaped(
                    format!("{VISION_PREFIX}{BLOCK_PATH}{b}.{suffix}"),
                    shape,
                    b,
                ));
            }
        }
        ts.push(shaped(
            format!("{VISION_PREFIX}patch_embedder.input_proj.weight"),
            vec![hid as u64, 768],
            90,
        ));
        ts.push(shaped(
            format!("{VISION_PREFIX}patch_embedder.position_embedding_table"),
            vec![2, 4, hid as u64],
            91,
        ));
        ts.push(shaped(
            format!("{VISION_PREFIX}std_bias"),
            vec![hid as u64],
            92,
        ));
        ts.push(shaped(
            format!("{VISION_PREFIX}std_scale"),
            vec![hid as u64],
            93,
        ));
        // The projector at hidden 64 so it is whole groups: one group per row.
        let mut q_packed = Vec::new();
        let (mut sc, mut bi) = (Vec::new(), Vec::new());
        for r in 0..text {
            let row: Vec<f32> = (0..64)
                .map(|i| ((r * 64 + i) as f32 * 0.11).cos())
                .collect();
            let q = compute::quant::quantize_int4_affine(&row);
            q_packed.extend_from_slice(&q.packed);
            sc.extend(q.scales.iter().flat_map(|v| v.to_le_bytes()));
            bi.extend(q.biases.iter().flat_map(|v| v.to_le_bytes()));
        }
        ts.push(Tensor {
            name: format!("{EMBED_VISION_PREFIX}embedding_projection.weight"),
            dtype: "U32",
            shape: vec![text as u64, 8],
            bytes: q_packed,
        });
        ts.push(Tensor {
            name: format!("{EMBED_VISION_PREFIX}embedding_projection.scales"),
            dtype: "BF16",
            shape: vec![text as u64, 1],
            bytes: sc,
        });
        ts.push(Tensor {
            name: format!("{EMBED_VISION_PREFIX}embedding_projection.biases"),
            dtype: "BF16",
            shape: vec![text as u64, 1],
            bytes: bi,
        });
        if let Some(name) = extra {
            ts.push(shaped(name.to_string(), vec![hid as u64], 99));
        }
        if let Some(d) = drop {
            ts.retain(|t| t.name != d);
        }
        // The projector reads at hidden 64 against a tower at hidden 8: only
        // the shape check below cares, so give it the width it unpacks to.
        vision.hidden_size = 64;
        let names = ts.iter().map(|t| t.name.clone()).collect();
        (vision, names, assemble_safetensors(&ts))
    }

    fn read(drop: Option<&str>, extra: Option<&str>) -> Result<VisionRead, Gemma4Error> {
        use crate::ranged_download::MemoryRangeSource;
        use crate::safetensors_header::{parse_header, DEFAULT_MAX_HEADER_BYTES};
        let (vision, names, blob) = tiny_checkpoint(drop, extra);
        let header = parse_header(&blob, DEFAULT_MAX_HEADER_BYTES).unwrap();
        let source = MemoryRangeSource::new(&blob);
        let shards = Gemma4Shards::single(&header, &source);
        // Only what the real walk hands over: `classify_all` skips every
        // `.scales`/`.biases` name, so the reader must find the projector's
        // companions in the shards itself.
        let bases: Vec<&str> = names
            .iter()
            .map(String::as_str)
            .filter(|n| !n.ends_with(".scales") && !n.ends_with(".biases"))
            .collect();
        // Through the shared entry point, so the tower-kind dispatch is
        // what is tested.
        super::super::vision::read_vision_entries(&shards, &bases, &vision)
    }

    #[test]
    fn a_complete_tower_reads_into_blocks_and_residents() {
        let r = read(None, None).unwrap();
        assert_eq!(r.blocks.experts.len(), 27);
        assert!(r.blocks.experts.iter().all(|e| e.sub_tensors.len() == 13));
        let roles: Vec<&str> = r.blocks.experts[0]
            .sub_tensors
            .iter()
            .map(|t| t.role.as_str())
            .collect();
        let want: Vec<&str> = GEMMA4_BLOCK_ROLES.iter().map(|(r, _)| *r).collect();
        assert_eq!(roles, want, "blob order is the forward order");
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
                "vision.patch_embedder.input_proj.weight",
                "vision.patch_embedder.position_embedding_table",
                "vision.std_bias",
                "vision.std_scale",
                "vision.embedding_projection.weight",
            ]
        );
        assert_eq!(r.block_stride % GTURBO_PAGE_BYTES, 0);
    }

    #[test]
    fn a_missing_or_unknown_tensor_is_refused_by_name() {
        let missing = format!("{VISION_PREFIX}{BLOCK_PATH}13.self_attn.k_norm.weight");
        let err = read(Some(&missing), None)
            .err()
            .expect("a missing role must refuse");
        assert!(
            err.to_string().contains("13.self_attn.k_norm.weight"),
            "{err}"
        );
        let err = read(Some("embed_vision.embedding_projection.scales"), None)
            .err()
            .expect("a missing projector part must refuse");
        assert!(
            err.to_string().contains("embedding_projection.scales"),
            "{err}"
        );
        let err = read(
            None,
            Some("vision_tower.encoder.layers.0.self_attn.v_norm.weight"),
        )
        .err()
        .expect("an unknown role must refuse");
        assert!(err.to_string().contains("v_norm"), "{err}");
    }

    #[test]
    fn the_projector_dequantizes_like_compute_quant() {
        let (rows, cols) = (2usize, 128usize);
        let src: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut packed = Vec::new();
        let mut scales = Vec::new();
        let mut biases = Vec::new();
        let mut want = Vec::new();
        for r in 0..rows {
            let q = compute::quant::quantize_int4_affine(&src[r * cols..(r + 1) * cols]);
            want.extend(compute::quant::dequantize_int4_affine(&q, cols));
            packed.extend_from_slice(&q.packed);
            scales.extend(q.scales.iter().flat_map(|s| s.to_le_bytes()));
            biases.extend(q.biases.iter().flat_map(|s| s.to_le_bytes()));
        }
        let got = dequant_projector("p", &packed, &scales, &biases, rows, cols).unwrap();
        for (i, w) in want.iter().enumerate() {
            let g = compute::f16_to_f32(u16::from_le_bytes([got[2 * i], got[2 * i + 1]]));
            assert!(
                (g - w).abs() <= w.abs() * 1e-3 + 1e-4,
                "value {i}: {g} vs {w}"
            );
        }
        assert!(dequant_projector("p", &packed[1..], &scales, &biases, rows, cols).is_err());
    }
}
