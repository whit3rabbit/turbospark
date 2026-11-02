//! The CLIP vision tower's config and ingest (LLaVA-1.5,
//! `docs/LLAVA_VISION_PHASE0.md`).
//!
//! A sibling of [`super::vision`] and [`super::vision_gemma4`]. What it shares
//! with the qwen tower is the destination: blocks pack into `packed_vision/`
//! as the SAME twelve-role blob (`ln1_w ln1_b qkv_w qkv_b proj_w proj_b ln2_w
//! ln2_b fc1_w fc1_b fc2_w fc2_b`), so the runtime reuses the qwen tower's
//! role table and streamer. What it does not share is the source: CLIP keeps
//! q, k and v as three tensors, which are FUSED here by row concatenation
//! (the fused weight's first `hidden` rows are q, the next k, the next v, the
//! partition the qwen block already reads), and names the layers
//! `vision_tower.vision_model.encoder.layers.N.*`.
//!
//! **Only `depth` blocks are packed, and `depth` is not the checkpoint's
//! layer count.** LLaVA-1.5 takes `vision_feature_layer = -2`: the hidden
//! state after the second-to-last block. A 24-block checkpoint therefore runs
//! 23 blocks, and block 23 and `post_layernorm` are never read by anything.
//! They are recognized and DROPPED here, by name, rather than refused (an
//! unknown name is refused) or packed (they would cost 25 MiB of streamed
//! bytes per image to read nothing).

use model_io::{VisionConfig, VisionTowerKind};

use super::classify::VISION_INSTALL_PREFIX;
use super::config::Gemma4Error;
use super::narrow::convert_raw_to_fp16;
use super::shards::{shape4, Gemma4Shards, GTURBO_PAGE_BYTES};
use super::vision::VisionRead;
use crate::gturbo_writer::{ExpertBlob, LayerBlobs, SubTensor};
use crate::resident_writer::{RawTensorSpec, ResidentEntrySpec};

/// The HF `model_type` of a LLaVA-1.5 checkpoint.
pub const LLAVA_MODEL_TYPE: &str = "llava";

/// The CLIP tower's prefix in a LLaVA checkpoint.
pub const CLIP_MODEL_PREFIX: &str = "vision_tower.vision_model.";

/// The projector's prefix in a LLaVA checkpoint, beside the tower's.
pub const CLIP_PROJECTOR_PREFIX: &str = "multi_modal_projector.";

/// Both source prefixes a LLaVA tower fetch needs.
pub const CLIP_SOURCE_PREFIXES: [&str; 2] = [CLIP_MODEL_PREFIX, CLIP_PROJECTOR_PREFIX];

/// The blob's roles, in the qwen tower's order (the runtime's role table).
const BLOCK_ROLE_NAMES: [&str; 12] = [
    "ln1_w", "ln1_b", "qkv_w", "qkv_b", "proj_w", "proj_b", "ln2_w", "ln2_b", "fc1_w", "fc1_b",
    "fc2_w", "fc2_b",
];

/// The sixteen per-block tensors a CLIP layer carries.
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

/// The tower's non-block tensors, under [`CLIP_MODEL_PREFIX`].
const EMBED_TENSORS: [&str; 4] = [
    "embeddings.patch_embedding.weight",
    "embeddings.class_embedding",
    "embeddings.position_embedding.weight",
    "pre_layrnorm.weight",
];
const PRE_NORM_BIAS: &str = "pre_layrnorm.bias";

/// The projector's four tensors, under [`CLIP_PROJECTOR_PREFIX`].
const PROJECTOR_TENSORS: [&str; 4] = [
    "linear_1.weight",
    "linear_1.bias",
    "linear_2.weight",
    "linear_2.bias",
];

/// The install-side resident names, under [`VISION_INSTALL_PREFIX`], in the
/// order they are written. The runtime resolves exactly these.
pub const CLIP_RESIDENT_TENSORS: [&str; 9] = [
    "patch_embed.proj.weight",
    "class_embedding",
    "pos_embed.weight",
    "ln_pre.weight",
    "ln_pre.bias",
    "projector.linear_1.weight",
    "projector.linear_1.bias",
    "projector.linear_2.weight",
    "projector.linear_2.bias",
];

/// Patch edge this port's preprocessing implements.
const SUPPORTED_PATCH: i64 = 14;

/// Parse a LLaVA `config.json` into the tower's [`VisionConfig`].
///
/// Everything the runtime does not store is checked here and refused when it
/// differs from the one tower this port implements: `clip_vision_model`, the
/// `quick_gelu` activation, LayerNorm epsilon 1e-5, a patch of 14, the
/// `default` feature-select strategy (drop the class row) and a GELU
/// projector. `out_hidden_size` is the trunk width, `text_config.hidden_size`
/// or the Llama default of 4096 when the file omits it (`llava-hf` does); the
/// reader cross-checks it against the projector's own shape either way.
pub fn parse_llava_vision_config(json: &str) -> Result<VisionConfig, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    let bad = |m: String| Gemma4Error::Config(m);
    if root.get("model_type").and_then(serde_json::Value::as_str) != Some(LLAVA_MODEL_TYPE) {
        return Err(bad(format!(
            "model_type is not {LLAVA_MODEL_TYPE:?}; this reader is for LLaVA-1.5"
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
    if vc.get("model_type").and_then(serde_json::Value::as_str) != Some("clip_vision_model") {
        return Err(bad(
            "vision_config.model_type is not clip_vision_model".to_string()
        ));
    }
    if let Some(act) = vc.get("hidden_act").and_then(serde_json::Value::as_str) {
        if act != "quick_gelu" {
            return Err(bad(format!(
                "vision hidden_act {act:?}; the implemented tower is quick_gelu"
            )));
        }
    }
    if let Some(eps) = vc.get("layer_norm_eps").and_then(serde_json::Value::as_f64) {
        if eps != 1e-5 {
            return Err(bad(format!(
                "vision layer_norm_eps {eps}; the implemented tower is 1e-5"
            )));
        }
    }
    if root
        .get("vision_feature_select_strategy")
        .and_then(serde_json::Value::as_str)
        != Some("default")
    {
        return Err(bad(
            "vision_feature_select_strategy must be \"default\" (drop the class row)".to_string(),
        ));
    }
    match root
        .get("projector_hidden_act")
        .and_then(serde_json::Value::as_str)
    {
        Some("gelu") => {}
        other => return Err(bad(format!("projector_hidden_act {other:?}; need gelu"))),
    }

    let layers = vi("num_hidden_layers")?;
    let feature_layer = root
        .get("vision_feature_layer")
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| bad("missing vision_feature_layer".to_string()))?;
    // `hidden_states` has `layers + 1` entries (the embeddings, then each
    // block); a negative index counts from the end.
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
    if image <= 0 || image % patch != 0 {
        return Err(bad(format!(
            "vision image_size {image} is not a multiple of the patch {patch}"
        )));
    }
    let side = image / patch;
    let text_hidden = root
        .get("text_config")
        .and_then(|t| t.get("hidden_size"))
        .and_then(serde_json::Value::as_i64)
        .unwrap_or(4096);
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
        num_position_embeddings: side * side + 1,
        out_hidden_size: text_hidden,
        mrope_section: [0, 0, 0],
        deepstack_visual_indexes: Vec::new(),
        vision_start_token_id: 0,
        vision_end_token_id: 0,
        image_token_id: image_token,
        video_token_id: 0,
        tower: VisionTowerKind::Clip,
    })
}

/// `[out, kh, kw, C]` (MLX's conv layout) to `[out, C, kh, kw]` (the layout the
/// patch row `(C, P_h, P_w)` flattens against), on FP16 bytes.
fn mlx_conv_to_channel_major(bytes: &[u8], out: usize, kh: usize, kw: usize, c: usize) -> Vec<u8> {
    let mut dst = vec![0u8; bytes.len()];
    for o in 0..out {
        for y in 0..kh {
            for x in 0..kw {
                for ch in 0..c {
                    let s = (((o * kh + y) * kw + x) * c + ch) * 2;
                    let d = (((o * c + ch) * kh + y) * kw + x) * 2;
                    dst[d..d + 2].copy_from_slice(&bytes[s..s + 2]);
                }
            }
        }
    }
    dst
}

/// Reads the CLIP tower and its projector, converting every tensor to FP16.
///
/// Every name under either prefix must be a known layer tensor, a known
/// resident, a projector part, or one of the two documented drops (blocks at
/// or past `depth`, and `post_layernorm`). An unknown name is refused, as the
/// other readers refuse one: a publisher who adds a tensor has changed the
/// tower. A packed (quantized) tensor is refused by dtype.
pub fn read_clip_vision_entries(
    shards: &Gemma4Shards<'_>,
    vision_bases: &[&str],
    vision: &VisionConfig,
) -> Result<VisionRead, Gemma4Error> {
    if vision.tower != VisionTowerKind::Clip || vision.depth < 1 {
        return Err(Gemma4Error::Config(format!(
            "not a CLIP tower (kind {:?}, depth {})",
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
    let mut embed: [Option<&str>; 4] = [None; 4];
    let mut pre_bias: Option<&str> = None;
    let mut projector: [Option<&str>; 4] = [None; 4];

    for &name in vision_bases {
        if let Some(tail) = name.strip_prefix(CLIP_PROJECTOR_PREFIX) {
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
        let tail = name.strip_prefix(CLIP_MODEL_PREFIX).ok_or_else(|| {
            Gemma4Error::UnknownTensor(format!("{name}: not under a CLIP vision prefix"))
        })?;
        if tail.ends_with("position_ids") {
            continue;
        }
        if tail.starts_with("post_layernorm.") {
            // Never read: the feature layer is before it.
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
                // The unread tail of the tower; see the module doc.
                continue;
            }
            if blocks[index][role].replace(name).is_some() {
                return Err(Gemma4Error::UnknownTensor(format!("{name}: duplicate")));
            }
        } else if tail == PRE_NORM_BIAS {
            if pre_bias.replace(name).is_some() {
                return Err(Gemma4Error::UnknownTensor(format!("{name}: duplicate")));
            }
        } else {
            let slot = EMBED_TENSORS
                .iter()
                .position(|t| *t == tail)
                .ok_or_else(|| {
                    Gemma4Error::UnknownTensor(format!(
                        "{name}: not a layer tensor and not a CLIP tower resident"
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
                    "{CLIP_MODEL_PREFIX}encoder.layers.{index}.{}",
                    LAYER_SUFFIXES[role]
                )));
            }
        }
    }
    for (slot, name) in embed.iter().enumerate() {
        if name.is_none() {
            return Err(Gemma4Error::MissingTensor(format!(
                "{CLIP_MODEL_PREFIX}{}",
                EMBED_TENSORS[slot]
            )));
        }
    }
    let pre_bias = pre_bias
        .ok_or_else(|| Gemma4Error::MissingTensor(format!("{CLIP_MODEL_PREFIX}{PRE_NORM_BIAS}")))?;
    for (slot, name) in projector.iter().enumerate() {
        if name.is_none() {
            return Err(Gemma4Error::MissingTensor(format!(
                "{CLIP_PROJECTOR_PREFIX}{}",
                PROJECTOR_TENSORS[slot]
            )));
        }
    }

    let mut lossy_conversion = Vec::new();
    let mut load = |name: &str, want: &[usize]| -> Result<(Vec<u8>, Vec<u64>), Gemma4Error> {
        let t = shards.info(name)?;
        if t.dtype == "U32" {
            return Err(Gemma4Error::ShapeMismatch {
                tensor: name.to_string(),
                detail: "a packed (quantized) tensor; the CLIP tower is read unquantized"
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

    let mut entries = Vec::with_capacity(CLIP_RESIDENT_TENSORS.len());
    let mut push_resident = |slot: usize, bytes: Vec<u8>, shape: (u32, u32, u32, u32)| {
        entries.push(ResidentEntrySpec::Raw(RawTensorSpec {
            name: format!("{VISION_INSTALL_PREFIX}{}", CLIP_RESIDENT_TENSORS[slot]),
            dtype: crate::DTYPE_FP16,
            bytes,
            shape,
        }));
    };

    // Patch embedding. HF stores [out, C, kh, kw]; an MLX conversion stores
    // [out, kh, kw, C]. Both land as [out, C*kh*kw] with channels outermost,
    // the order vision-io emits a patch row in.
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
    push_resident(
        3,
        load(embed[3].expect("checked above"), &[hidden])?.0,
        shape4(&[hidden as u64]),
    );
    push_resident(4, load(pre_bias, &[hidden])?.0, shape4(&[hidden as u64]));

    let p = |slot: usize| projector[slot].expect("checked above");
    // The projector's output width IS the trunk width: it writes straight
    // into the residual stream. The config's `out_hidden_size` may be a
    // default (`llava-hf` omits `text_config.hidden_size`), so the tensor
    // decides and a disagreement is refused.
    push_resident(
        5,
        load(p(0), &[out_hidden, hidden])?.0,
        (out_hidden as u32, hidden as u32, 0, 0),
    );
    push_resident(
        6,
        load(p(1), &[out_hidden])?.0,
        shape4(&[out_hidden as u64]),
    );
    push_resident(
        7,
        load(p(2), &[out_hidden, out_hidden])?.0,
        (out_hidden as u32, out_hidden as u32, 0, 0),
    );
    push_resident(
        8,
        load(p(3), &[out_hidden])?.0,
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

    /// The published `llava-hf/llava-1.5-7b-hf` config, trimmed to the keys
    /// this parser reads.
    fn real_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "llava",
            "image_token_index": 32000,
            "projector_hidden_act": "gelu",
            "vision_feature_layer": -2,
            "vision_feature_select_strategy": "default",
            "text_config": {"model_type": "llama", "vocab_size": 32064},
            "vision_config": {
                "model_type": "clip_vision_model", "hidden_size": 1024, "image_size": 336,
                "intermediate_size": 4096, "num_attention_heads": 16, "num_hidden_layers": 24,
                "patch_size": 14, "projection_dim": 768
            }
        })
    }

    #[test]
    fn the_published_config_parses_to_a_23_block_clip_tower() {
        let v = parse_llava_vision_config(&real_config().to_string()).unwrap();
        assert_eq!(v.tower, VisionTowerKind::Clip);
        // 24 layers, feature layer -2: hidden_states[-2] is after block 22.
        assert_eq!(v.depth, 23);
        assert_eq!((v.hidden_size, v.num_heads, v.head_dim()), (1024, 16, 64));
        assert_eq!(v.num_position_embeddings, 577);
        assert_eq!(v.out_hidden_size, 4096);
        assert_eq!(v.image_token_id, 32000);
        assert_eq!((v.temporal_patch_size, v.spatial_merge_size), (1, 1));
    }

    #[test]
    fn a_config_this_tower_does_not_implement_is_refused() {
        let cases: [(&str, serde_json::Value); 6] = [
            ("model_type", serde_json::json!("llava_next")),
            ("projector_hidden_act", serde_json::json!("relu")),
            ("vision_feature_select_strategy", serde_json::json!("full")),
            ("vision_feature_layer", serde_json::json!(-30)),
            ("vision_feature_layer", serde_json::json!(0)),
            ("image_token_index", serde_json::Value::Null),
        ];
        for (key, value) in cases {
            let mut c = real_config();
            c[key] = value;
            assert!(
                parse_llava_vision_config(&c.to_string()).is_err(),
                "{key} must be refused"
            );
        }
        let mut c = real_config();
        c["vision_config"]["patch_size"] = serde_json::json!(16);
        assert!(parse_llava_vision_config(&c.to_string()).is_err());
        let mut c = real_config();
        c["vision_config"]["hidden_act"] = serde_json::json!("gelu");
        assert!(parse_llava_vision_config(&c.to_string()).is_err());
    }

    #[test]
    fn an_mlx_conv_layout_is_permuted_to_channel_major() {
        // out=1, kh=2, kw=2, C=3 with value = 100*y + 10*x + c.
        let mut mlx = Vec::new();
        for y in 0..2u16 {
            for x in 0..2u16 {
                for c in 0..3u16 {
                    mlx.extend_from_slice(&(100 * y + 10 * x + c).to_le_bytes());
                }
            }
        }
        let out = mlx_conv_to_channel_major(&mlx, 1, 2, 2, 3);
        let v: Vec<u16> = out
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        // [C][kh][kw]: channel 0 first.
        assert_eq!(v, vec![0, 10, 100, 110, 1, 11, 101, 111, 2, 12, 102, 112]);
    }

    const HID: usize = 8;
    const INTER: usize = 12;
    const TEXT: usize = 16;
    const POS: usize = 5; // a 2x2 grid plus the class row

    fn tiny_vision() -> VisionConfig {
        let mut v = parse_llava_vision_config(&real_config().to_string()).unwrap();
        v.depth = 2;
        v.hidden_size = HID as i64;
        v.num_heads = 2;
        v.intermediate_size = INTER as i64;
        v.out_hidden_size = TEXT as i64;
        v.num_position_embeddings = POS as i64;
        v
    }

    fn tiny_checkpoint(
        drop: Option<&str>,
        extra: Option<&str>,
        mlx_conv: bool,
    ) -> (Vec<String>, Vec<u8>) {
        use crate::synthetic_tensors::{assemble_safetensors, bf16_vector, Tensor};
        let shaped = |name: String, shape: Vec<u64>, seed: u64| {
            let n: u64 = shape.iter().product();
            let mut t = bf16_vector(&name, n as usize, 0.5, seed);
            t.shape = shape;
            t
        };
        let (h, i) = (HID as u64, INTER as u64);
        let mut ts: Vec<Tensor> = Vec::new();
        // Three layers in the file against a depth of 2: layer 2 is the
        // documented drop.
        for b in 0..3u64 {
            for suffix in LAYER_SUFFIXES {
                let shape = match suffix {
                    s if s.starts_with("layer_norm") || s.ends_with("proj.bias") => vec![h],
                    "mlp.fc1.weight" => vec![i, h],
                    "mlp.fc1.bias" => vec![i],
                    "mlp.fc2.weight" => vec![h, i],
                    "mlp.fc2.bias" => vec![h],
                    s if s.ends_with(".bias") => vec![h],
                    _ => vec![h, h],
                };
                ts.push(shaped(
                    format!("{CLIP_MODEL_PREFIX}encoder.layers.{b}.{suffix}"),
                    shape,
                    b * 100 + suffix.len() as u64,
                ));
            }
        }
        let conv_shape = if mlx_conv {
            vec![h, 14, 14, 3]
        } else {
            vec![h, 3, 14, 14]
        };
        ts.push(shaped(
            format!("{CLIP_MODEL_PREFIX}embeddings.patch_embedding.weight"),
            conv_shape,
            1,
        ));
        ts.push(shaped(
            format!("{CLIP_MODEL_PREFIX}embeddings.class_embedding"),
            vec![h],
            2,
        ));
        ts.push(shaped(
            format!("{CLIP_MODEL_PREFIX}embeddings.position_embedding.weight"),
            vec![POS as u64, h],
            3,
        ));
        for (n, seed) in [("pre_layrnorm.weight", 4), ("pre_layrnorm.bias", 5)] {
            ts.push(shaped(format!("{CLIP_MODEL_PREFIX}{n}"), vec![h], seed));
        }
        for (n, seed) in [("post_layernorm.weight", 6), ("post_layernorm.bias", 7)] {
            ts.push(shaped(format!("{CLIP_MODEL_PREFIX}{n}"), vec![h], seed));
        }
        let t = TEXT as u64;
        ts.push(shaped(
            format!("{CLIP_PROJECTOR_PREFIX}linear_1.weight"),
            vec![t, h],
            8,
        ));
        ts.push(shaped(
            format!("{CLIP_PROJECTOR_PREFIX}linear_1.bias"),
            vec![t],
            9,
        ));
        ts.push(shaped(
            format!("{CLIP_PROJECTOR_PREFIX}linear_2.weight"),
            vec![t, t],
            10,
        ));
        ts.push(shaped(
            format!("{CLIP_PROJECTOR_PREFIX}linear_2.bias"),
            vec![t],
            11,
        ));
        if let Some(name) = extra {
            ts.push(shaped(name.to_string(), vec![h], 99));
        }
        if let Some(d) = drop {
            ts.retain(|t| t.name != d);
        }
        let names = ts.iter().map(|t| t.name.clone()).collect();
        (names, assemble_safetensors(&ts))
    }

    fn read(
        drop: Option<&str>,
        extra: Option<&str>,
        mlx_conv: bool,
    ) -> Result<VisionRead, Gemma4Error> {
        use crate::ranged_download::MemoryRangeSource;
        use crate::safetensors_header::{parse_header, DEFAULT_MAX_HEADER_BYTES};
        let (names, blob) = tiny_checkpoint(drop, extra, mlx_conv);
        let header = parse_header(&blob, DEFAULT_MAX_HEADER_BYTES).unwrap();
        let source = MemoryRangeSource::new(&blob);
        let shards = Gemma4Shards::single(&header, &source);
        let bases: Vec<&str> = names.iter().map(String::as_str).collect();
        // Through the shared entry point, so the tower-kind dispatch is what
        // is tested.
        super::super::vision::read_vision_entries(&shards, &bases, &tiny_vision())
    }

    #[test]
    fn a_complete_tower_reads_into_depth_blocks_and_nine_residents() {
        let r = read(None, None, false).unwrap();
        assert_eq!(
            r.blocks.experts.len(),
            2,
            "the unread third layer is dropped"
        );
        assert!(r.blocks.experts.iter().all(|e| e.sub_tensors.len() == 12));
        let roles: Vec<&str> = r.blocks.experts[0]
            .sub_tensors
            .iter()
            .map(|t| t.role.as_str())
            .collect();
        assert_eq!(roles, BLOCK_ROLE_NAMES, "the qwen blob's role order");
        let qkv = &r.blocks.experts[0].sub_tensors[2];
        assert_eq!(qkv.shape, vec![3 * HID as u64, HID as u64]);
        assert_eq!(qkv.bytes.len(), 3 * HID * HID * 2);
        let names: Vec<String> = r
            .entries
            .iter()
            .map(|e| match e {
                ResidentEntrySpec::Raw(t) => t.name.clone(),
                _ => panic!("vision residents are raw FP16"),
            })
            .collect();
        let want: Vec<String> = CLIP_RESIDENT_TENSORS
            .iter()
            .map(|n| format!("vision.{n}"))
            .collect();
        assert_eq!(names, want);
        assert_eq!(r.block_stride % GTURBO_PAGE_BYTES, 0);
    }

    #[test]
    fn q_k_and_v_fuse_in_that_row_order() {
        let r = read(None, None, false).unwrap();
        // Rebuild the expected fused bytes from the source tensors.
        use crate::ranged_download::MemoryRangeSource;
        use crate::safetensors_header::{parse_header, DEFAULT_MAX_HEADER_BYTES};
        let (_, blob) = tiny_checkpoint(None, None, false);
        let header = parse_header(&blob, DEFAULT_MAX_HEADER_BYTES).unwrap();
        let source = MemoryRangeSource::new(&blob);
        let shards = Gemma4Shards::single(&header, &source);
        let mut want = Vec::new();
        for p in ["q", "k", "v"] {
            let name = format!("{CLIP_MODEL_PREFIX}encoder.layers.1.self_attn.{p}_proj.weight");
            let raw = shards.read(&name).unwrap();
            want.extend(convert_raw_to_fp16(&name, "BF16", raw).unwrap().bytes);
        }
        assert_eq!(r.blocks.experts[1].sub_tensors[2].bytes, want);
    }

    #[test]
    fn an_mlx_layout_conv_lands_channel_major_like_the_hf_one() {
        // The same logical weights, stored two ways, must install differently
        // only by the permutation: the HF one verbatim, the MLX one permuted.
        let hf = read(None, None, false).unwrap();
        let mlx = read(None, None, true).unwrap();
        let bytes = |r: &VisionRead| match &r.entries[0] {
            ResidentEntrySpec::Raw(t) => (t.bytes.clone(), t.shape),
            _ => unreachable!(),
        };
        let (a, sa) = bytes(&hf);
        let (b, sb) = bytes(&mlx);
        assert_eq!(sa, (HID as u32, 588, 0, 0));
        assert_eq!(sa, sb);
        assert_eq!(a.len(), b.len());
        assert_ne!(a, b, "a permutation of different source bytes");
    }

    #[test]
    fn a_missing_unknown_or_packed_tensor_is_refused_by_name() {
        let missing = format!("{CLIP_MODEL_PREFIX}encoder.layers.1.mlp.fc2.bias");
        let err = read(Some(&missing), None, false).err().expect("missing");
        assert!(err.to_string().contains("layers.1.mlp.fc2.bias"), "{err}");
        let err = read(Some("multi_modal_projector.linear_2.weight"), None, false)
            .err()
            .expect("missing projector part");
        assert!(err.to_string().contains("linear_2.weight"), "{err}");
        let err = read(
            None,
            Some("vision_tower.vision_model.encoder.layers.0.self_attn.rotary.weight"),
            false,
        )
        .err()
        .expect("unknown role");
        assert!(err.to_string().contains("rotary"), "{err}");
    }

    #[test]
    fn the_projector_width_is_decided_by_the_tensor() {
        // A config claiming a different trunk width than the projector ships
        // is refused rather than read at the wrong stride.
        use crate::ranged_download::MemoryRangeSource;
        use crate::safetensors_header::{parse_header, DEFAULT_MAX_HEADER_BYTES};
        let (names, blob) = tiny_checkpoint(None, None, false);
        let header = parse_header(&blob, DEFAULT_MAX_HEADER_BYTES).unwrap();
        let source = MemoryRangeSource::new(&blob);
        let shards = Gemma4Shards::single(&header, &source);
        let bases: Vec<&str> = names.iter().map(String::as_str).collect();
        let mut v = tiny_vision();
        v.out_hidden_size = 24;
        let err = read_clip_vision_entries(&shards, &bases, &v)
            .err()
            .expect("width mismatch");
        assert!(err.to_string().contains("linear_1.weight"), "{err}");
    }
}
