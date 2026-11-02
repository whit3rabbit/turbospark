//! The Pixtral vision tower's config and ingest (Pixtral-12B,
//! `docs/PIXTRAL_VISION_PHASE0.md`).
//!
//! A sibling of [`super::vision_clip`]. What it shares with the CLIP reader is
//! the shape of the job: a tower whose q, k and v are three tensors that are
//! FUSED here by row concatenation (q block, k block, v block: the partition
//! the fused-qkv matmul reads), a patch convolution copied as a channel-major
//! row matrix, and a two-layer projector, all written FP16. What differs is
//! the block: Pixtral's has NO BIASES anywhere (so seven roles, not twelve),
//! RMSNorm instead of LayerNorm, and a gated MLP.
//!
//! Two source spellings of the same tensors exist and both are read: the
//! `mlx-community` conversion keeps the tower under
//! `vision_tower.vision_model.` and the original `mistral-community` HF export
//! under `vision_tower.` directly. The projector is `multi_modal_projector.`
//! in both.
//!
//! A packed (quantized) tensor is REFUSED by dtype: `mlx-community/pixtral-12b-4bit`
//! quantizes the tower, and the unquantized `pixtral-12b-bf16` is the source
//! this reader is for.

use model_io::{VisionConfig, VisionTowerKind};

use super::classify::VISION_INSTALL_PREFIX;
use super::config::Gemma4Error;
use super::narrow::convert_raw_to_fp16;
use super::shards::{shape4, Gemma4Shards, GTURBO_PAGE_BYTES};
use super::vision::VisionRead;
use super::vision_clip::mlx_conv_to_channel_major;
use crate::gturbo_writer::{ExpertBlob, LayerBlobs, SubTensor};
use crate::resident_writer::{RawTensorSpec, ResidentEntrySpec};

/// The HF `model_type` of a Pixtral checkpoint.
pub const PIXTRAL_MODEL_TYPE: &str = "pixtral";

/// The projector's prefix, beside the tower's.
pub const PIXTRAL_PROJECTOR_PREFIX: &str = "multi_modal_projector.";

/// The tower's prefix, both spellings sharing this stem. The `vision_model.`
/// level the mlx conversion adds is stripped by the reader.
pub const PIXTRAL_TOWER_PREFIX: &str = "vision_tower.";

/// Both source prefixes a Pixtral tower fetch needs.
pub const PIXTRAL_SOURCE_PREFIXES: [&str; 2] = [PIXTRAL_TOWER_PREFIX, PIXTRAL_PROJECTOR_PREFIX];

/// The blob's roles, in byte order. The runtime restates this table.
pub const PIXTRAL_BLOCK_ROLES: [&str; 7] = [
    "n1_w", "qkv_w", "proj_w", "n2_w", "gate_w", "up_w", "down_w",
];

/// The nine per-block tensors a Pixtral layer carries.
const LAYER_SUFFIXES: [&str; 9] = [
    "attention_norm.weight",
    "attention.q_proj.weight",
    "attention.k_proj.weight",
    "attention.v_proj.weight",
    "attention.o_proj.weight",
    "ffn_norm.weight",
    "feed_forward.gate_proj.weight",
    "feed_forward.up_proj.weight",
    "feed_forward.down_proj.weight",
];

/// The tower's non-block tensors, under the tower prefix (after the optional
/// `vision_model.` level).
const TOWER_TENSORS: [&str; 2] = ["patch_conv.weight", "ln_pre.weight"];

/// The projector's four tensors.
const PROJECTOR_TENSORS: [&str; 4] = [
    "linear_1.weight",
    "linear_1.bias",
    "linear_2.weight",
    "linear_2.bias",
];

/// The install-side resident names, under [`VISION_INSTALL_PREFIX`], in the
/// order they are written. The runtime resolves exactly these.
pub const PIXTRAL_RESIDENT_TENSORS: [&str; 6] = [
    "patch_embed.proj.weight",
    "ln_pre.weight",
    "projector.linear_1.weight",
    "projector.linear_1.bias",
    "projector.linear_2.weight",
    "projector.linear_2.bias",
];

/// Patch edge this port's preprocessing implements.
const SUPPORTED_PATCH: i64 = 16;

/// `PixtralVisionConfig`'s own defaults. The `mlx-community` config omits
/// `hidden_size`, `num_hidden_layers` and `num_attention_heads` (it states
/// `head_dim` and `intermediate_size`), so silence means the class defaults,
/// which the reader then checks against the tensor shapes.
const DEFAULT_HIDDEN: i64 = 1024;
const DEFAULT_LAYERS: i64 = 24;
const DEFAULT_HEADS: i64 = 16;

/// Parse a Pixtral `config.json` into the tower's [`VisionConfig`].
///
/// `vision_start_token_id` and `vision_end_token_id` come back 0: the
/// `[IMG_BREAK]` and `[IMG_END]` ids live in the tokenizer, not the config, and
/// the caller sets them with [`pixtral_special_token_ids`].
///
/// Everything the runtime does not store is checked and refused when it
/// differs from the one tower this port implements: the `pixtral` model type,
/// patch 16, RoPE theta 10000 (a kind constant), a GELU projector and a
/// `head_dim` that agrees with the width. `hidden_act` may say `gelu` or
/// `silu`: the published config says `gelu` and the tower runs SiLU
/// (`docs/PIXTRAL_VISION_PHASE0.md`), so the field is not read as a switch.
pub fn parse_pixtral_vision_config(json: &str) -> Result<VisionConfig, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    let bad = |m: String| Gemma4Error::Config(m);
    if root.get("model_type").and_then(serde_json::Value::as_str) != Some(PIXTRAL_MODEL_TYPE) {
        return Err(bad(format!(
            "model_type is not {PIXTRAL_MODEL_TYPE:?}; this reader is for Pixtral"
        )));
    }
    let vc = match root.get("vision_config") {
        None | Some(serde_json::Value::Null) => return Ok(VisionConfig::NONE),
        Some(v) => v,
    };
    let vi_or = |k: &str, default: i64| -> i64 {
        vc.get(k)
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(default)
    };
    if let Some(t) = vc.get("model_type").and_then(serde_json::Value::as_str) {
        if t != "pixtral" {
            return Err(bad(format!("vision_config.model_type {t:?}, not pixtral")));
        }
    }
    if let Some(theta) = vc.get("rope_theta").and_then(serde_json::Value::as_f64) {
        if theta != 10_000.0 {
            return Err(bad(format!(
                "vision rope_theta {theta}; the implemented tower is 10000"
            )));
        }
    }
    match root
        .get("projector_hidden_act")
        .and_then(serde_json::Value::as_str)
    {
        Some("gelu") | None => {}
        Some(other) => return Err(bad(format!("projector_hidden_act {other:?}; need gelu"))),
    }
    let strategy = root
        .get("vision_feature_select_strategy")
        .and_then(serde_json::Value::as_str);
    if strategy.is_some_and(|s| s != "full") {
        return Err(bad(format!(
            "vision_feature_select_strategy {strategy:?}; Pixtral uses full (no class row)"
        )));
    }
    if root
        .get("vision_feature_layer")
        .and_then(serde_json::Value::as_i64)
        .is_some_and(|l| l != -1)
    {
        return Err(bad(
            "vision_feature_layer must be -1: Pixtral reads the last block".to_string(),
        ));
    }

    let hidden = vi_or("hidden_size", DEFAULT_HIDDEN);
    let layers = vi_or("num_hidden_layers", DEFAULT_LAYERS);
    let heads = vi_or("num_attention_heads", DEFAULT_HEADS);
    if heads <= 0 || layers <= 0 || hidden % heads != 0 {
        return Err(bad(format!(
            "vision hidden_size {hidden} / {heads} heads / {layers} layers is not a tower"
        )));
    }
    let head_dim = hidden / heads;
    if let Some(stated) = vc.get("head_dim").and_then(serde_json::Value::as_i64) {
        if stated != head_dim {
            return Err(bad(format!(
                "vision head_dim {stated} against hidden_size {hidden} over {heads} heads"
            )));
        }
    }
    if head_dim % 4 != 0 {
        return Err(bad(format!(
            "vision head_dim {head_dim} must be a multiple of 4 for the 2-D rope"
        )));
    }
    let patch = vi_or("patch_size", SUPPORTED_PATCH);
    if patch != SUPPORTED_PATCH {
        return Err(bad(format!(
            "vision patch_size {patch}; the implemented preprocessing is patch {SUPPORTED_PATCH}"
        )));
    }
    let channels = vi_or("num_channels", 3);
    if channels != 3 {
        return Err(bad(format!("vision num_channels {channels}, need 3")));
    }
    let image_size = vi_or("image_size", 1024);
    if image_size <= 0 || image_size % patch != 0 {
        return Err(bad(format!(
            "vision image_size {image_size} is not a multiple of the patch {patch}"
        )));
    }
    let text_hidden = root
        .get("text_config")
        .and_then(|t| t.get("hidden_size"))
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| bad("missing text_config.hidden_size".to_string()))?;
    let image_token = root
        .get("image_token_index")
        .or_else(|| root.get("image_token_id"))
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| bad("missing image_token_index".to_string()))?;

    Ok(VisionConfig {
        depth: layers,
        hidden_size: hidden,
        intermediate_size: vi_or("intermediate_size", 4096),
        num_heads: heads,
        patch_size: patch,
        temporal_patch_size: 1,
        in_channels: 3,
        spatial_merge_size: 1,
        num_position_embeddings: 0,
        out_hidden_size: text_hidden,
        mrope_section: [0, 0, 0],
        deepstack_visual_indexes: Vec::new(),
        vision_start_token_id: 0,
        vision_end_token_id: 0,
        image_token_id: image_token,
        video_token_id: 0,
        tower: VisionTowerKind::Pixtral,
    })
}

/// `[IMG]`, `[IMG_BREAK]` and `[IMG_END]` ids, read off a `tokenizer.json`'s
/// `added_tokens`. The ids are the checkpoint's own (10, 12 and 13 on the
/// published Tekken vocabulary), and a tokenizer missing any of the three is a
/// checkpoint whose prompt layout this port cannot build.
pub fn pixtral_special_token_ids(tokenizer_json: &str) -> Result<(i64, i64, i64), Gemma4Error> {
    let v: serde_json::Value = serde_json::from_str(tokenizer_json)
        .map_err(|e| Gemma4Error::Config(format!("tokenizer.json: {e}")))?;
    let added = v
        .get("added_tokens")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Gemma4Error::Config("tokenizer.json has no added_tokens".to_string()))?;
    let find = |content: &str| -> Result<i64, Gemma4Error> {
        added
            .iter()
            .find(|t| t.get("content").and_then(serde_json::Value::as_str) == Some(content))
            .and_then(|t| t.get("id").and_then(serde_json::Value::as_i64))
            .ok_or_else(|| {
                Gemma4Error::Config(format!("tokenizer.json has no added token {content:?}"))
            })
    };
    Ok((find("[IMG]")?, find("[IMG_BREAK]")?, find("[IMG_END]")?))
}

/// Reads the Pixtral tower and its projector, converting every tensor to FP16.
///
/// Every name under either prefix must be a known layer tensor, a tower
/// resident or a projector part. An unknown name is refused, as the other
/// readers refuse one: a publisher who adds a tensor has changed the tower.
pub fn read_pixtral_vision_entries(
    shards: &Gemma4Shards<'_>,
    vision_bases: &[&str],
    vision: &VisionConfig,
) -> Result<VisionRead, Gemma4Error> {
    if vision.tower != VisionTowerKind::Pixtral || vision.depth < 1 {
        return Err(Gemma4Error::Config(format!(
            "not a Pixtral tower (kind {:?}, depth {})",
            vision.tower, vision.depth
        )));
    }
    let depth = vision.depth as usize;
    let hidden = vision.hidden_size as usize;
    let inter = vision.intermediate_size as usize;
    let out_hidden = vision.out_hidden_size as usize;
    let patch = vision.patch_size as usize;

    let mut blocks: Vec<[Option<&str>; 9]> = vec![[None; 9]; depth];
    let mut tower: [Option<&str>; 2] = [None; 2];
    let mut projector: [Option<&str>; 4] = [None; 4];

    for &name in vision_bases {
        if let Some(tail) = name.strip_prefix(PIXTRAL_PROJECTOR_PREFIX) {
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
        let tail = name.strip_prefix(PIXTRAL_TOWER_PREFIX).ok_or_else(|| {
            Gemma4Error::UnknownTensor(format!("{name}: not under a Pixtral vision prefix"))
        })?;
        let tail = tail.strip_prefix("vision_model.").unwrap_or(tail);
        if let Some(rest) = tail.strip_prefix("transformer.layers.") {
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
                        "{name}: {suffix} is not one of the nine per-layer tensors"
                    ))
                })?;
            if index >= depth {
                return Err(Gemma4Error::ShapeMismatch {
                    tensor: name.to_string(),
                    detail: format!("layer {index} is past the declared depth of {depth}"),
                });
            }
            if blocks[index][role].replace(name).is_some() {
                return Err(Gemma4Error::UnknownTensor(format!("{name}: duplicate")));
            }
        } else {
            let slot = TOWER_TENSORS
                .iter()
                .position(|t| *t == tail)
                .ok_or_else(|| {
                    Gemma4Error::UnknownTensor(format!(
                        "{name}: not a layer tensor and not a Pixtral tower resident"
                    ))
                })?;
            if tower[slot].replace(name).is_some() {
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
                    "{PIXTRAL_TOWER_PREFIX}transformer.layers.{index}.{}",
                    LAYER_SUFFIXES[role]
                )));
            }
        }
    }
    for (slot, name) in tower.iter().enumerate() {
        if name.is_none() {
            return Err(Gemma4Error::MissingTensor(format!(
                "{PIXTRAL_TOWER_PREFIX}{}",
                TOWER_TENSORS[slot]
            )));
        }
    }
    for (slot, name) in projector.iter().enumerate() {
        if name.is_none() {
            return Err(Gemma4Error::MissingTensor(format!(
                "{PIXTRAL_PROJECTOR_PREFIX}{}",
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
                detail: "a packed (quantized) tensor; the Pixtral tower is read unquantized \
                         (use the bf16 conversion, not the 4-bit one)"
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

    let mut entries = Vec::with_capacity(PIXTRAL_RESIDENT_TENSORS.len());
    let mut push_resident = |slot: usize, bytes: Vec<u8>, shape: (u32, u32, u32, u32)| {
        entries.push(ResidentEntrySpec::Raw(RawTensorSpec {
            name: format!("{VISION_INSTALL_PREFIX}{}", PIXTRAL_RESIDENT_TENSORS[slot]),
            dtype: crate::DTYPE_FP16,
            bytes,
            shape,
        }));
    };

    // Patch convolution. HF stores [out, C, kh, kw]; an MLX conversion stores
    // [out, kh, kw, C]. Both land as [out, C*kh*kw] with channels outermost,
    // the order `crates/vision-io` emits a patch row in.
    let patch_name = tower[0].expect("checked above");
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
        load(tower[1].expect("checked above"), &[hidden])?.0,
        shape4(&[hidden as u64]),
    );

    let p = |slot: usize| projector[slot].expect("checked above");
    // The projector's output width IS the trunk width: it writes straight into
    // the residual stream, and the tensor shape cross-checks the config.
    push_resident(
        2,
        load(p(0), &[out_hidden, hidden])?.0,
        (out_hidden as u32, hidden as u32, 0, 0),
    );
    push_resident(
        3,
        load(p(1), &[out_hidden])?.0,
        shape4(&[out_hidden as u64]),
    );
    push_resident(
        4,
        load(p(2), &[out_hidden, out_hidden])?.0,
        (out_hidden as u32, out_hidden as u32, 0, 0),
    );
    push_resident(
        5,
        load(p(3), &[out_hidden])?.0,
        shape4(&[out_hidden as u64]),
    );

    let mut experts = Vec::with_capacity(depth);
    let mut widest = 0u64;
    for (index, roles) in blocks.iter().enumerate() {
        let n = |role: usize| roles[role].expect("checked above");
        let mut sub_tensors = Vec::with_capacity(PIXTRAL_BLOCK_ROLES.len());
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
        let i = inter as u64;
        // Indexes into LAYER_SUFFIXES.
        add("n1_w", load(n(0), &[hidden])?.0, vec![h]);
        let mut qkv_w = Vec::with_capacity(3 * hidden * hidden * 2);
        for part in [1usize, 2, 3] {
            qkv_w.extend_from_slice(&load(n(part), &[hidden, hidden])?.0);
        }
        add("qkv_w", qkv_w, vec![3 * h, h]);
        add("proj_w", load(n(4), &[hidden, hidden])?.0, vec![h, h]);
        add("n2_w", load(n(5), &[hidden])?.0, vec![h]);
        add("gate_w", load(n(6), &[inter, hidden])?.0, vec![i, h]);
        add("up_w", load(n(7), &[inter, hidden])?.0, vec![i, h]);
        add("down_w", load(n(8), &[hidden, inter])?.0, vec![h, i]);
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

    /// `mlx-community/pixtral-12b-bf16`'s config, as published (the vision
    /// block states no width, depth or head count).
    fn real_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "pixtral",
            "image_token_index": 10,
            "projector_hidden_act": "gelu",
            "vision_feature_layer": -1,
            "vision_feature_select_strategy": "full",
            "text_config": {"model_type": "mistral", "hidden_size": 5120},
            "vision_config": {
                "head_dim": 64, "hidden_act": "gelu", "image_size": 1024,
                "model_type": "pixtral", "patch_size": 16, "rope_theta": 10000.0,
                "intermediate_size": 4096
            }
        })
    }

    #[test]
    fn the_published_config_parses_to_a_24_block_pixtral_tower() {
        let v = parse_pixtral_vision_config(&real_config().to_string()).unwrap();
        assert_eq!(v.tower, VisionTowerKind::Pixtral);
        assert_eq!(
            (v.depth, v.hidden_size, v.num_heads, v.head_dim()),
            (24, 1024, 16, 64)
        );
        assert_eq!((v.intermediate_size, v.patch_size), (4096, 16));
        assert_eq!(v.out_hidden_size, 5120);
        // No class row, no position table, no merge.
        assert_eq!((v.num_position_embeddings, v.spatial_merge_size), (0, 1));
        assert_eq!(v.image_token_id, 10);
        // The break and end ids are the tokenizer's to supply.
        assert_eq!((v.vision_start_token_id, v.vision_end_token_id), (0, 0));
    }

    #[test]
    fn a_config_this_port_does_not_implement_is_refused() {
        for (key, value) in [
            ("model_type", serde_json::json!("llava")),
            ("vision_feature_layer", serde_json::json!(-2)),
            (
                "vision_feature_select_strategy",
                serde_json::json!("default"),
            ),
            ("projector_hidden_act", serde_json::json!("silu")),
        ] {
            let mut c = real_config();
            c[key] = value;
            assert!(
                parse_pixtral_vision_config(&c.to_string()).is_err(),
                "{key}"
            );
        }
        for (key, value) in [
            ("patch_size", serde_json::json!(14)),
            ("rope_theta", serde_json::json!(1_000_000.0)),
            ("head_dim", serde_json::json!(128)),
        ] {
            let mut c = real_config();
            c["vision_config"][key] = value;
            assert!(
                parse_pixtral_vision_config(&c.to_string()).is_err(),
                "{key}"
            );
        }
    }

    const HID: usize = 8;
    const INTER: usize = 12;
    const TEXT: usize = 16;

    fn tiny_vision() -> VisionConfig {
        let mut v = parse_pixtral_vision_config(&real_config().to_string()).unwrap();
        v.depth = 2;
        v.hidden_size = HID as i64;
        v.num_heads = 2;
        v.intermediate_size = INTER as i64;
        v.out_hidden_size = TEXT as i64;
        v
    }

    /// A tiny two-layer checkpoint. `mlx_names` picks the
    /// `vision_tower.vision_model.` spelling over the bare `vision_tower.` one,
    /// `mlx_conv` the `[out, kh, kw, C]` conv layout over `[out, C, kh, kw]`.
    fn tiny_checkpoint(
        drop: Option<&str>,
        extra: Option<&str>,
        mlx_names: bool,
        mlx_conv: bool,
    ) -> (Vec<String>, Vec<u8>) {
        use crate::synthetic_tensors::{assemble_safetensors, bf16_vector, Tensor};
        let shaped = |name: String, shape: Vec<u64>, seed: u64| {
            let n: u64 = shape.iter().product();
            let mut t = bf16_vector(&name, n as usize, 0.5, seed);
            t.shape = shape;
            t
        };
        let tower = if mlx_names {
            "vision_tower.vision_model."
        } else {
            "vision_tower."
        };
        let (h, i) = (HID as u64, INTER as u64);
        let mut ts: Vec<Tensor> = Vec::new();
        for b in 0..2u64 {
            for suffix in LAYER_SUFFIXES {
                let shape = match suffix {
                    s if s.ends_with("norm.weight") => vec![h],
                    s if s.contains("gate_proj") || s.contains("up_proj") => vec![i, h],
                    s if s.contains("down_proj") => vec![h, i],
                    _ => vec![h, h],
                };
                ts.push(shaped(
                    format!("{tower}transformer.layers.{b}.{suffix}"),
                    shape,
                    b * 100 + suffix.len() as u64,
                ));
            }
        }
        let conv = if mlx_conv {
            vec![h, 16, 16, 3]
        } else {
            vec![h, 3, 16, 16]
        };
        ts.push(shaped(format!("{tower}patch_conv.weight"), conv, 1));
        ts.push(shaped(format!("{tower}ln_pre.weight"), vec![h], 2));
        let t = TEXT as u64;
        for (n, shape, seed) in [
            ("linear_1.weight", vec![t, h], 3),
            ("linear_1.bias", vec![t], 4),
            ("linear_2.weight", vec![t, t], 5),
            ("linear_2.bias", vec![t], 6),
        ] {
            ts.push(shaped(
                format!("{PIXTRAL_PROJECTOR_PREFIX}{n}"),
                shape,
                seed,
            ));
        }
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
        mlx_names: bool,
        mlx_conv: bool,
    ) -> Result<VisionRead, Gemma4Error> {
        use crate::ranged_download::MemoryRangeSource;
        use crate::safetensors_header::{parse_header, DEFAULT_MAX_HEADER_BYTES};
        let (names, blob) = tiny_checkpoint(drop, extra, mlx_names, mlx_conv);
        let header = parse_header(&blob, DEFAULT_MAX_HEADER_BYTES).unwrap();
        let source = MemoryRangeSource::new(&blob);
        let shards = Gemma4Shards::single(&header, &source);
        let bases: Vec<&str> = names.iter().map(String::as_str).collect();
        // Through the shared entry point, so the tower-kind dispatch is what
        // is tested.
        super::super::vision::read_vision_entries(&shards, &bases, &tiny_vision())
    }

    #[test]
    fn a_complete_tower_reads_into_depth_blocks_and_six_residents() {
        let r = read(None, None, true, false).unwrap();
        assert_eq!(r.blocks.experts.len(), 2);
        assert!(r.blocks.experts.iter().all(|e| e.sub_tensors.len() == 7));
        let roles: Vec<&str> = r.blocks.experts[0]
            .sub_tensors
            .iter()
            .map(|t| t.role.as_str())
            .collect();
        assert_eq!(roles, PIXTRAL_BLOCK_ROLES, "the blob's role order");
        // No bias anywhere: the fused qkv is three weights and nothing else.
        let qkv = &r.blocks.experts[0].sub_tensors[1];
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
        let want: Vec<String> = PIXTRAL_RESIDENT_TENSORS
            .iter()
            .map(|n| format!("vision.{n}"))
            .collect();
        assert_eq!(names, want);
        assert_eq!(r.block_stride % GTURBO_PAGE_BYTES, 0);
    }

    #[test]
    fn both_source_spellings_install_the_same_bytes() {
        let a = read(None, None, true, false).unwrap();
        let b = read(None, None, false, false).unwrap();
        // Different seeds per name length, so compare structure and sizes: the
        // two spellings have different name lengths and so different bytes,
        // but the same roles in the same order and the same total size.
        let sizes = |r: &VisionRead| -> Vec<usize> {
            r.blocks
                .experts
                .iter()
                .flat_map(|e| e.sub_tensors.iter().map(|t| t.bytes.len()))
                .collect()
        };
        assert_eq!(sizes(&a), sizes(&b));
        assert_eq!(a.entries.len(), b.entries.len());
    }

    #[test]
    fn q_k_and_v_fuse_in_that_row_order() {
        let r = read(None, None, true, false).unwrap();
        use crate::ranged_download::MemoryRangeSource;
        use crate::safetensors_header::{parse_header, DEFAULT_MAX_HEADER_BYTES};
        let (_, blob) = tiny_checkpoint(None, None, true, false);
        let header = parse_header(&blob, DEFAULT_MAX_HEADER_BYTES).unwrap();
        let source = MemoryRangeSource::new(&blob);
        let shards = Gemma4Shards::single(&header, &source);
        let mut want = Vec::new();
        for p in ["q", "k", "v"] {
            let name =
                format!("vision_tower.vision_model.transformer.layers.1.attention.{p}_proj.weight");
            let raw = shards.read(&name).unwrap();
            want.extend(convert_raw_to_fp16(&name, "BF16", raw).unwrap().bytes);
        }
        assert_eq!(r.blocks.experts[1].sub_tensors[1].bytes, want);
    }

    #[test]
    fn an_mlx_layout_conv_lands_channel_major_like_the_hf_one() {
        let hf = read(None, None, true, false).unwrap();
        let mlx = read(None, None, true, true).unwrap();
        let bytes = |r: &VisionRead| match &r.entries[0] {
            ResidentEntrySpec::Raw(t) => (t.bytes.clone(), t.shape),
            _ => unreachable!(),
        };
        let (a, sa) = bytes(&hf);
        let (b, sb) = bytes(&mlx);
        assert_eq!(sa, (HID as u32, 768, 0, 0));
        assert_eq!(sa, sb);
        assert_ne!(a, b, "a permutation of different source bytes");
    }

    #[test]
    fn a_missing_unknown_or_packed_tensor_is_refused_by_name() {
        let missing =
            "vision_tower.vision_model.transformer.layers.1.feed_forward.down_proj.weight";
        let err = read(Some(missing), None, true, false)
            .err()
            .expect("missing");
        assert!(
            err.to_string().contains("layers.1.feed_forward.down_proj"),
            "{err}"
        );
        let err = read(
            Some("multi_modal_projector.linear_2.bias"),
            None,
            true,
            false,
        )
        .err()
        .expect("missing projector part");
        assert!(err.to_string().contains("linear_2.bias"), "{err}");
        let err = read(
            None,
            Some("vision_tower.vision_model.transformer.layers.0.attention.rotary.weight"),
            true,
            false,
        )
        .err()
        .expect("unknown role");
        assert!(err.to_string().contains("rotary"), "{err}");
        // A layer past the declared depth is a different tower, not a tail to drop.
        let err = read(
            None,
            Some("vision_tower.vision_model.transformer.layers.2.attention_norm.weight"),
            true,
            false,
        )
        .err()
        .expect("layer past depth");
        assert!(err.to_string().contains("past the declared depth"), "{err}");
    }

    #[test]
    fn the_special_ids_come_off_the_tokenizers_added_tokens() {
        let tok = r#"{"added_tokens": [
            {"id": 10, "content": "[IMG]"}, {"id": 12, "content": "[IMG_BREAK]"},
            {"id": 13, "content": "[IMG_END]"}, {"id": 1, "content": "<s>"}]}"#;
        assert_eq!(pixtral_special_token_ids(tok).unwrap(), (10, 12, 13));
        let missing = r#"{"added_tokens": [{"id": 10, "content": "[IMG]"}]}"#;
        assert!(pixtral_special_token_ids(missing).is_err());
    }
}
