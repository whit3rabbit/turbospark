//! The LLaVA-OneVision SigLIP tower's config and ingest
//! (`docs/LLAVA_ONEVISION_VISION_PHASE0.md`).
//!
//! A sibling of [`super::vision_clip`] with the same source names
//! (`vision_tower.vision_model.*`, `multi_modal_projector.*`) and the same
//! destination (`packed_vision/` in the qwen tower's twelve-role blob, q/k/v
//! fused by row concatenation). What differs is the tower:
//!
//! - SigLIP has no class embedding and no pre-norm, and its patch embedding
//!   carries a bias;
//! - the feature layer is `-1`, so EVERY block runs (26 of 26) and the only
//!   dropped tensors are `post_layernorm` (never read: no head, no final
//!   norm) and any block at or past `depth`;
//! - the projector is applied per patch row (`hidden -> out_hidden`, GELU,
//!   `out_hidden -> out_hidden`), and a top-level `image_newline` `[out_hidden]`
//!   rides along as a resident.

use model_io::{VisionConfig, VisionTowerKind};

use super::classify::VISION_INSTALL_PREFIX;
use super::config::Gemma4Error;
use super::narrow::convert_raw_to_fp16;
use super::shards::{shape4, Gemma4Shards, GTURBO_PAGE_BYTES};
use super::vision::VisionRead;
use super::vision_clip::{
    mlx_conv_to_channel_major, CLIP_MODEL_PREFIX as TOWER_PREFIX,
    CLIP_PROJECTOR_PREFIX as PROJECTOR_PREFIX,
};
use crate::gturbo_writer::{ExpertBlob, LayerBlobs, SubTensor};
use crate::resident_writer::{RawTensorSpec, ResidentEntrySpec};

/// The HF `model_type` of a LLaVA-OneVision checkpoint.
pub const ONEVISION_MODEL_TYPE: &str = "llava_onevision";

/// The top-level learned row appended to every feature-map row.
pub const ONEVISION_NEWLINE: &str = "image_newline";

/// Every source prefix (or exact name) a OneVision tower fetch needs.
pub const ONEVISION_SOURCE_PREFIXES: [&str; 3] =
    [TOWER_PREFIX, PROJECTOR_PREFIX, ONEVISION_NEWLINE];

const BLOCK_ROLE_NAMES: [&str; 12] = [
    "ln1_w", "ln1_b", "qkv_w", "qkv_b", "proj_w", "proj_b", "ln2_w", "ln2_b", "fc1_w", "fc1_b",
    "fc2_w", "fc2_b",
];

/// The sixteen per-block tensors a SigLIP layer carries.
const LAYER_SUFFIXES: [&str; 16] = [
    "layer_norm1.weight",
    "layer_norm1.bias",
    "self_attn.q_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.k_proj.weight",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.weight",
    "self_attn.v_proj.bias",
    "self_attn.out_proj.weight",
    "self_attn.out_proj.bias",
    "layer_norm2.weight",
    "layer_norm2.bias",
    "mlp.fc1.weight",
    "mlp.fc1.bias",
    "mlp.fc2.weight",
    "mlp.fc2.bias",
];

/// The tower's non-block tensors, under [`TOWER_PREFIX`].
const EMBED_TENSORS: [&str; 3] = [
    "embeddings.patch_embedding.weight",
    "embeddings.patch_embedding.bias",
    "embeddings.position_embedding.weight",
];

const PROJECTOR_TENSORS: [&str; 4] = [
    "linear_1.weight",
    "linear_1.bias",
    "linear_2.weight",
    "linear_2.bias",
];

/// The install-side resident names under [`VISION_INSTALL_PREFIX`], in the
/// order they are written. The runtime resolves exactly these.
pub const ONEVISION_RESIDENT_TENSORS: [&str; 8] = [
    "patch_embed.proj.weight",
    "patch_embed.proj.bias",
    "pos_embed.weight",
    "projector.layers.0.weight",
    "projector.layers.0.bias",
    "projector.layers.2.weight",
    "projector.layers.2.bias",
    "image_newline",
];

const SUPPORTED_PATCH: i64 = 14;

/// Parse a `llava_onevision` `config.json` into the tower's [`VisionConfig`].
///
/// Everything the runtime does not store is checked here and refused when it
/// differs from the one tower this port implements: a SigLIP vision model
/// with the tanh-GELU activation and epsilon 1e-6 (the HF defaults, which the
/// file omits), no pooling head, the `full` feature-select strategy (all 729
/// patches kept), a GELU projector, `anyres_max_9` and the learned
/// `image_newline`.
pub fn parse_llava_onevision_vision_config(json: &str) -> Result<VisionConfig, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    let bad = |m: String| Gemma4Error::Config(m);
    if root.get("model_type").and_then(serde_json::Value::as_str) != Some(ONEVISION_MODEL_TYPE) {
        return Err(bad(format!(
            "model_type is not {ONEVISION_MODEL_TYPE:?}; this reader is for LLaVA-OneVision"
        )));
    }
    let vc = match root.get("vision_config") {
        None | Some(serde_json::Value::Null) => return Ok(VisionConfig::NONE),
        Some(v) => v,
    };
    let vi = |k: &str| -> Result<i64, Gemma4Error> {
        vc.get(k)
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| bad(format!("missing vision_config.{k}")))
    };
    if vc.get("model_type").and_then(serde_json::Value::as_str) != Some("siglip_vision_model") {
        return Err(bad(
            "vision_config.model_type is not siglip_vision_model".to_string()
        ));
    }
    if let Some(act) = vc.get("hidden_act").and_then(serde_json::Value::as_str) {
        if act != "gelu_pytorch_tanh" {
            return Err(bad(format!(
                "vision hidden_act {act:?}; the implemented tower is gelu_pytorch_tanh"
            )));
        }
    }
    if let Some(eps) = vc.get("layer_norm_eps").and_then(serde_json::Value::as_f64) {
        if eps != 1e-6 {
            return Err(bad(format!(
                "vision layer_norm_eps {eps}; the implemented tower is 1e-6"
            )));
        }
    }
    if vc
        .get("vision_use_head")
        .and_then(serde_json::Value::as_bool)
        != Some(false)
    {
        return Err(bad(
            "vision_use_head must be false (no pooling head)".to_string()
        ));
    }
    let text = |k: &str, want: &str| -> Result<(), Gemma4Error> {
        match root.get(k).and_then(serde_json::Value::as_str) {
            Some(v) if v == want => Ok(()),
            other => Err(bad(format!("{k} {other:?}; this reader needs {want:?}"))),
        }
    };
    text("vision_feature_select_strategy", "full")?;
    text("projector_hidden_act", "gelu")?;
    text("vision_aspect_ratio", "anyres_max_9")?;
    if root
        .get("use_image_newline_parameter")
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        return Err(bad(
            "use_image_newline_parameter must be true (the packing appends image_newline)"
                .to_string(),
        ));
    }

    let layers = vi("num_hidden_layers")?;
    let feature_layer = root
        .get("vision_feature_layer")
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| bad("missing vision_feature_layer".to_string()))?;
    // `hidden_states` has `layers + 1` entries; a negative index counts from
    // the end, so -1 is after the last block.
    let depth = if feature_layer < 0 {
        layers + 1 + feature_layer
    } else {
        feature_layer
    };
    if depth < 1 || depth > layers {
        return Err(bad(format!(
            "vision_feature_layer {feature_layer} selects {depth} block(s) of {layers}"
        )));
    }
    let hidden = vi("hidden_size")?;
    let heads = vi("num_attention_heads")?;
    if heads <= 0 || hidden % heads != 0 {
        return Err(bad(format!(
            "vision hidden_size {hidden} is not divisible by num_attention_heads {heads}"
        )));
    }
    let patch = vi("patch_size")?;
    if patch != SUPPORTED_PATCH {
        return Err(bad(format!(
            "vision patch_size {patch}; the implemented preprocessing is patch {SUPPORTED_PATCH}"
        )));
    }
    let image = vi("image_size")?;
    if image < patch {
        return Err(bad(format!(
            "vision image_size {image} is smaller than the patch {patch}"
        )));
    }
    // A valid-padding stride-14 conv over a 384 px tile reads 27 x 27 patches;
    // the 6 trailing pixels never reach it (the DeepSeek-VL2 tower's case).
    let side = image / patch;
    let text_hidden = root
        .get("text_config")
        .and_then(|t| t.get("hidden_size"))
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| bad("missing text_config.hidden_size".to_string()))?;
    let image_token = root
        .get("image_token_index")
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| bad("missing image_token_index".to_string()))?;

    Ok(VisionConfig {
        depth,
        hidden_size: hidden,
        intermediate_size: vi("intermediate_size")?,
        num_heads: heads,
        patch_size: patch,
        temporal_patch_size: 1,
        in_channels: 3,
        spatial_merge_size: 1,
        num_position_embeddings: side * side,
        out_hidden_size: text_hidden,
        mrope_section: [0, 0, 0],
        deepstack_visual_indexes: Vec::new(),
        vision_start_token_id: 0,
        vision_end_token_id: 0,
        image_token_id: image_token,
        video_token_id: root
            .get("video_token_index")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(0),
        tower: VisionTowerKind::SigLipOv,
    })
}

/// Reads the OneVision tower, its projector and `image_newline`, converting
/// every tensor to FP16.
///
/// Every name must be a known layer tensor, a known resident, a projector
/// part, or one of the two documented drops (blocks at or past `depth`, and
/// `post_layernorm`); an unknown name is refused, a packed (quantized) tensor
/// is refused by dtype.
pub fn read_onevision_vision_entries(
    shards: &Gemma4Shards<'_>,
    vision_bases: &[&str],
    vision: &VisionConfig,
) -> Result<VisionRead, Gemma4Error> {
    if vision.tower != VisionTowerKind::SigLipOv || vision.depth < 1 {
        return Err(Gemma4Error::Config(format!(
            "not a OneVision tower (kind {:?}, depth {})",
            vision.tower, vision.depth
        )));
    }
    let depth = vision.depth as usize;
    let hidden = vision.hidden_size as usize;
    let inter = vision.intermediate_size as usize;
    let out_hidden = vision.out_hidden_size as usize;
    let patch = vision.patch_size as usize;
    let pos_rows = vision.num_position_embeddings as usize;

    let mut blocks: Vec<[Option<&str>; 16]> = vec![[None; 16]; depth];
    let mut embed: [Option<&str>; 3] = [None; 3];
    let mut projector: [Option<&str>; 4] = [None; 4];
    let mut newline: Option<&str> = None;

    for &name in vision_bases {
        if name == ONEVISION_NEWLINE {
            if newline.replace(name).is_some() {
                return Err(Gemma4Error::UnknownTensor(format!("{name}: duplicate")));
            }
            continue;
        }
        if let Some(tail) = name.strip_prefix(PROJECTOR_PREFIX) {
            let slot = PROJECTOR_TENSORS
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
        let tail = name.strip_prefix(TOWER_PREFIX).ok_or_else(|| {
            Gemma4Error::UnknownTensor(format!("{name}: not under a OneVision vision prefix"))
        })?;
        if tail.ends_with("position_ids") || tail.starts_with("post_layernorm.") {
            // Never read: no head and no final norm after the feature layer.
            continue;
        }
        if let Some(rest) = tail.strip_prefix("encoder.layers.") {
            let (index, suffix) = rest.split_once('.').ok_or_else(|| {
                Gemma4Error::UnknownTensor(format!("{name}: no layer index and suffix"))
            })?;
            let index: usize = index.parse().map_err(|_| {
                Gemma4Error::UnknownTensor(format!("{name}: layer index is not a number"))
            })?;
            let role = LAYER_SUFFIXES
                .iter()
                .position(|s| *s == suffix)
                .ok_or_else(|| {
                    Gemma4Error::UnknownTensor(format!(
                        "{name}: {suffix} is not one of the sixteen per-layer tensors"
                    ))
                })?;
            if index >= depth {
                continue;
            }
            if blocks[index][role].replace(name).is_some() {
                return Err(Gemma4Error::UnknownTensor(format!("{name}: duplicate")));
            }
        } else {
            let slot = EMBED_TENSORS
                .iter()
                .position(|t| *t == tail)
                .ok_or_else(|| {
                    Gemma4Error::UnknownTensor(format!(
                        "{name}: not a layer tensor and not a SigLIP tower resident"
                    ))
                })?;
            if embed[slot].replace(name).is_some() {
                return Err(Gemma4Error::UnknownTensor(format!("{name}: duplicate")));
            }
        }
    }

    // Every slot filled, or refused by name: a missing tensor would stream as
    // zeros and the tower would run with a sublayer silently off.
    for (index, roles) in blocks.iter().enumerate() {
        for (role, name) in roles.iter().enumerate() {
            if name.is_none() {
                return Err(Gemma4Error::MissingTensor(format!(
                    "{TOWER_PREFIX}encoder.layers.{index}.{}",
                    LAYER_SUFFIXES[role]
                )));
            }
        }
    }
    for (slot, name) in embed.iter().enumerate() {
        if name.is_none() {
            return Err(Gemma4Error::MissingTensor(format!(
                "{TOWER_PREFIX}{}",
                EMBED_TENSORS[slot]
            )));
        }
    }
    for (slot, name) in projector.iter().enumerate() {
        if name.is_none() {
            return Err(Gemma4Error::MissingTensor(format!(
                "{PROJECTOR_PREFIX}{}",
                PROJECTOR_TENSORS[slot]
            )));
        }
    }
    let newline =
        newline.ok_or_else(|| Gemma4Error::MissingTensor(ONEVISION_NEWLINE.to_string()))?;

    let mut lossy_conversion = Vec::new();
    let mut load = |name: &str, want: &[usize]| -> Result<(Vec<u8>, Vec<u64>), Gemma4Error> {
        let t = shards.info(name)?;
        if t.dtype == "U32" {
            return Err(Gemma4Error::ShapeMismatch {
                tensor: name.to_string(),
                detail: "a packed (quantized) tensor; the SigLIP tower is read unquantized"
                    .to_string(),
            });
        }
        let got: Vec<usize> = t.shape.iter().map(|&d| d as usize).collect();
        if got != want {
            return Err(Gemma4Error::ShapeMismatch {
                tensor: name.to_string(),
                detail: format!("shape {got:?}, expected {want:?}"),
            });
        }
        let converted = convert_raw_to_fp16(name, &t.dtype, shards.read(name)?)?;
        if converted.lossy > 0 {
            lossy_conversion.push((name.to_string(), converted.lossy));
        }
        Ok((converted.bytes, t.shape.clone()))
    };

    let mut entries = Vec::with_capacity(ONEVISION_RESIDENT_TENSORS.len());
    let mut push_resident = |slot: usize, bytes: Vec<u8>, shape: (u32, u32, u32, u32)| {
        entries.push(ResidentEntrySpec::Raw(RawTensorSpec {
            name: format!(
                "{VISION_INSTALL_PREFIX}{}",
                ONEVISION_RESIDENT_TENSORS[slot]
            ),
            dtype: crate::DTYPE_FP16,
            bytes,
            shape,
        }));
    };

    // Patch embedding: HF stores [out, C, kh, kw]; an MLX conversion
    // [out, kh, kw, C]. Both land as [out, C*kh*kw], channels outermost, the
    // order vision-io emits a patch row in.
    let patch_name = embed[0].expect("checked above");
    let pt = shards.info(patch_name)?;
    let pshape: Vec<usize> = pt.shape.iter().map(|&d| d as usize).collect();
    let patch_dim = 3 * patch * patch;
    let patch_bytes = match pshape.as_slice() {
        [o, 3, kh, kw] if (*o, *kh, *kw) == (hidden, patch, patch) => load(patch_name, &pshape)?.0,
        [o, kh, kw, 3] if (*o, *kh, *kw) == (hidden, patch, patch) => {
            let (b, _) = load(patch_name, &pshape)?;
            mlx_conv_to_channel_major(&b, hidden, patch, patch, 3)
        }
        other => {
            return Err(Gemma4Error::ShapeMismatch {
                tensor: patch_name.to_string(),
                detail: format!(
                    "shape {other:?} is neither [{hidden}, 3, {patch}, {patch}] nor \
                     [{hidden}, {patch}, {patch}, 3]"
                ),
            })
        }
    };
    push_resident(0, patch_bytes, (hidden as u32, patch_dim as u32, 0, 0));
    push_resident(
        1,
        load(embed[1].expect("checked above"), &[hidden])?.0,
        shape4(&[hidden as u64]),
    );
    push_resident(
        2,
        load(embed[2].expect("checked above"), &[pos_rows, hidden])?.0,
        (pos_rows as u32, hidden as u32, 0, 0),
    );

    // The projector's output width IS the trunk width; the tensor decides and
    // a disagreement with the config is refused by the shape check.
    let p = |slot: usize| projector[slot].expect("checked above");
    push_resident(
        3,
        load(p(0), &[out_hidden, hidden])?.0,
        (out_hidden as u32, hidden as u32, 0, 0),
    );
    push_resident(
        4,
        load(p(1), &[out_hidden])?.0,
        shape4(&[out_hidden as u64]),
    );
    push_resident(
        5,
        load(p(2), &[out_hidden, out_hidden])?.0,
        (out_hidden as u32, out_hidden as u32, 0, 0),
    );
    push_resident(
        6,
        load(p(3), &[out_hidden])?.0,
        shape4(&[out_hidden as u64]),
    );
    push_resident(
        7,
        load(newline, &[out_hidden])?.0,
        shape4(&[out_hidden as u64]),
    );

    let mut experts = Vec::with_capacity(depth);
    let mut widest = 0u64;
    for (index, roles) in blocks.iter().enumerate() {
        let n = |role: usize| roles[role].expect("checked above");
        let mut sub_tensors = Vec::with_capacity(BLOCK_ROLE_NAMES.len());
        let mut used = 0u64;
        let mut add = |role: &str, bytes: Vec<u8>, shape: Vec<u64>| {
            used += bytes.len() as u64;
            sub_tensors.push(SubTensor {
                role: role.to_string(),
                bytes,
                dtype: "fp16".to_string(),
                shape,
            });
        };
        let h = hidden as u64;
        // Indexes into LAYER_SUFFIXES.
        add("ln1_w", load(n(0), &[hidden])?.0, vec![h]);
        add("ln1_b", load(n(1), &[hidden])?.0, vec![h]);
        let mut qkv_w = Vec::with_capacity(3 * hidden * hidden * 2);
        let mut qkv_b = Vec::with_capacity(3 * hidden * 2);
        for part in [2usize, 4, 6] {
            qkv_w.extend_from_slice(&load(n(part), &[hidden, hidden])?.0);
            qkv_b.extend_from_slice(&load(n(part + 1), &[hidden])?.0);
        }
        add("qkv_w", qkv_w, vec![3 * h, h]);
        add("qkv_b", qkv_b, vec![3 * h]);
        add("proj_w", load(n(8), &[hidden, hidden])?.0, vec![h, h]);
        add("proj_b", load(n(9), &[hidden])?.0, vec![h]);
        add("ln2_w", load(n(10), &[hidden])?.0, vec![h]);
        add("ln2_b", load(n(11), &[hidden])?.0, vec![h]);
        add(
            "fc1_w",
            load(n(12), &[inter, hidden])?.0,
            vec![inter as u64, h],
        );
        add("fc1_b", load(n(13), &[inter])?.0, vec![inter as u64]);
        add(
            "fc2_w",
            load(n(14), &[hidden, inter])?.0,
            vec![h, inter as u64],
        );
        add("fc2_b", load(n(15), &[hidden])?.0, vec![h]);
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

    /// The published `llava-hf/llava-onevision-qwen2-7b-ov-hf` config, trimmed
    /// to the keys this parser reads.
    fn real_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "llava_onevision",
            "image_token_index": 151646,
            "video_token_index": 151647,
            "projector_hidden_act": "gelu",
            "vision_feature_layer": -1,
            "vision_feature_select_strategy": "full",
            "vision_aspect_ratio": "anyres_max_9",
            "use_image_newline_parameter": true,
            "text_config": {"model_type": "qwen2", "hidden_size": 3584, "vocab_size": 152128},
            "vision_config": {
                "model_type": "siglip_vision_model", "hidden_size": 1152, "image_size": 384,
                "intermediate_size": 4304, "num_attention_heads": 16, "num_hidden_layers": 26,
                "patch_size": 14, "vision_use_head": false
            }
        })
    }

    #[test]
    fn the_published_config_parses_to_a_26_block_siglip_tower() {
        let v = parse_llava_onevision_vision_config(&real_config().to_string()).unwrap();
        assert_eq!(v.tower, VisionTowerKind::SigLipOv);
        // 26 layers, feature layer -1: hidden_states[-1] is after block 25.
        assert_eq!(v.depth, 26);
        assert_eq!((v.hidden_size, v.num_heads, v.head_dim()), (1152, 16, 72));
        // No class row: 27 x 27 positions.
        assert_eq!(v.num_position_embeddings, 729);
        assert_eq!(v.out_hidden_size, 3584);
        assert_eq!((v.image_token_id, v.video_token_id), (151646, 151647));
        assert_eq!((v.temporal_patch_size, v.spatial_merge_size), (1, 1));
    }

    #[test]
    fn a_config_this_tower_does_not_implement_is_refused() {
        let cases: [(&str, serde_json::Value); 8] = [
            ("model_type", serde_json::json!("llava_next")),
            ("projector_hidden_act", serde_json::json!("relu")),
            (
                "vision_feature_select_strategy",
                serde_json::json!("default"),
            ),
            ("vision_aspect_ratio", serde_json::json!("anyres")),
            ("use_image_newline_parameter", serde_json::json!(false)),
            ("vision_feature_layer", serde_json::json!(-30)),
            ("vision_feature_layer", serde_json::json!(0)),
            ("image_token_index", serde_json::Value::Null),
        ];
        for (key, value) in cases {
            let mut c = real_config();
            c[key] = value;
            assert!(
                parse_llava_onevision_vision_config(&c.to_string()).is_err(),
                "{key} must be refused"
            );
        }
        for (key, value) in [
            ("patch_size", serde_json::json!(16)),
            ("vision_use_head", serde_json::json!(true)),
            ("hidden_act", serde_json::json!("quick_gelu")),
            ("layer_norm_eps", serde_json::json!(1e-5)),
            ("model_type", serde_json::json!("clip_vision_model")),
        ] {
            let mut c = real_config();
            c["vision_config"][key] = value;
            assert!(
                parse_llava_onevision_vision_config(&c.to_string()).is_err(),
                "vision_config.{key} must be refused"
            );
        }
    }

    #[test]
    fn a_text_only_config_has_no_tower() {
        let mut c = real_config();
        c["vision_config"] = serde_json::Value::Null;
        let v = parse_llava_onevision_vision_config(&c.to_string()).unwrap();
        assert!(!v.is_active());
    }
}
