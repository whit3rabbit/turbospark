//! The InternViT tower the InternVL wrappers carry: config and ingest
//! (`docs/INTERNVL_VISION_PHASE0.md`).
//!
//! A sibling of [`super::vision_siglip`], and it lands in the same place: each
//! block packs into `packed_vision/` as the qwen tower's twelve-role blob PLUS
//! the two layer-scale rows (`ls1`, `ls2`), and the residents and projector go
//! in under `vision.` at FP16, dequantized where the source is an MLX affine
//! plane (the tower bodies are unquantized on every published conversion; the
//! projector Linears are 4-bit).
//!
//! **Two wrappers, one tower.** Both are read here because the tower is the
//! same InternViT and the tensors differ only in naming:
//!
//! - `internvl_chat` (root `model_type`, trunk `qwen2` per
//!   `mlx-community/InternVL3-1B-4bit` @ 7927bb6b86987f19398a14eee20a73c45fdb1867
//!   and `mlx-community/InternVL2_5-1B-4bit` @
//!   2f49cb7e2a06e34ae8f1880735cd1fdc1bf21359): tower prefix
//!   `vision_model.` (fused `attn.qkv` already), projector prefix `mlp1.`
//!   (the mlx-vlm module list: `mlp1.0` LayerNorm, `mlp1.1` Linear, `mlp1.3`
//!   Linear -- there is no `mlp1.2`, that slot is the activation).
//! - `internvl` (trunk `qwen3` per `mlx-community/InternVL3_5-1B-4bit` @
//!   f9d179a8be8ac53e96c6ee5cce8493856d4b8f09): tower prefix `vision_tower.`
//!   (separate `attention.{q,k,v}_proj`, fused here into one `qkv` blob in q,
//!   k, v row order), projector prefix `multi_modal_projector.`.
//!
//! **The wrapper rows deliberately DO NOT live in `SUPPORTED_HF`.** An
//! InternVL checkpoint's trunk is whatever `text_config.model_type` says
//! (`qwen2` on the three published 1B builds read for this port; other
//! publishers wrap `internlm2` and friends), so a row mapping the WRAPPER to
//! one family would resolve a config whose trunk the registry does not know
//! into that family and then parse a foreign trunk as Qwen. The trunk-first
//! resolution (`arch_registry::config_json_resolution` consults the root
//! `model_type` and then `text_config.model_type`) already resolves both
//! wrappers, so these constants name the wrappers for the VISION reader and
//! the classifier only.
//!
//! Both published flavors have NO final norm after the encoder (no
//! `vision_model.layernorm` / `vision_tower.layernorm` tensor in either
//! index, matching mlx-vlm's `VisionModel`), and the runtime does not run
//! one. The runtime tower also runs no `qk_norm`: `qk_normalization`
//! (`internvl_chat`) and `use_qk_norm` (`internvl`) are false on all three
//! published 1B configs, and a config that declares them true is refused by
//! name at parse.

use model_io::{VisionConfig, VisionTowerKind};

use super::classify::VISION_INSTALL_PREFIX;
use super::config::{Gemma4Error, Gemma4Quant};
use super::shards::{shape4, Gemma4Shards, GTURBO_PAGE_BYTES};
use super::vision::{read_tower_tensor, VisionRead};
use crate::gturbo_writer::{ExpertBlob, LayerBlobs, SubTensor};
use crate::resident_writer::{RawTensorSpec, ResidentEntrySpec};

/// The root `model_type` of the `internvl_chat` wrapper (the InternVL2.5 and
/// InternVL3 line over a `qwen2` trunk on the published 1B builds).
pub const INTERNVL_CHAT_MODEL_TYPE: &str = "internvl_chat";
/// The root `model_type` of the `internvl` wrapper (InternVL3.5 and later,
/// trunk `qwen3` on the published 1B build).
pub const INTERNVL_MODEL_TYPE: &str = "internvl";

/// The tower's prefix in an `internvl_chat` checkpoint.
pub const INTERNVL_CHAT_TOWER_PREFIX: &str = "vision_model.";
/// The projector's prefix in an `internvl_chat` checkpoint: mlx-vlm's
/// `mlp1` module list. Slot 2 is the GELU and carries no tensors.
pub const INTERNVL_CHAT_PROJECTOR_PREFIX: &str = "mlp1.";
/// The tower's prefix in an `internvl` checkpoint.
pub const INTERNVL_TOWER_PREFIX: &str = "vision_tower.";
/// The projector's prefix in an `internvl` checkpoint.
pub const INTERNVL_PROJECTOR_PREFIX: &str = "multi_modal_projector.";

/// The block's fourteen roles, in forward-pass order: the qwen tower's
/// twelve plus the two layer-scale rows the residual adds scale by. The
/// runtime tower resolves exactly these.
pub const INTERNVL_BLOCK_ROLES: [&str; 14] = [
    "ln1_w", "ln1_b", "qkv_w", "qkv_b", "proj_w", "proj_b", "ln2_w", "ln2_b", "fc1_w", "fc1_b",
    "fc2_w", "fc2_b", "ls1", "ls2",
];

/// The source suffix under the tower prefix for each role, in the same
/// order, for BOTH flavors (the `internvl` flavor's attention submodule is
/// spelled the same once its separate q/k/v are under `attn.`-equivalent
/// names -- the mapping to source names happens in [`read_internvl_vision_entries`]).
pub const INTERNVL_CHAT_BLOCK_SUFFIXES: [&str; 14] = [
    "norm1.weight",
    "norm1.bias",
    "attn.qkv.weight",
    "attn.qkv.bias",
    "attn.proj.weight",
    "attn.proj.bias",
    "norm2.weight",
    "norm2.bias",
    "mlp.fc1.weight",
    "mlp.fc1.bias",
    "mlp.fc2.weight",
    "mlp.fc2.bias",
    "ls1",
    "ls2",
];

/// The `internvl` flavor's per-block source suffixes. The block container is
/// `encoder.layer.N.` (singular), the norms are `layernorm_before/after`, the
/// output projection is `projection_layer`, and the layer scale is
/// `lambda_1`/`lambda_2`.
pub const INTERNVL_BLOCK_SUFFIXES: [&str; 14] = [
    "layernorm_before.weight",
    "layernorm_before.bias",
    "attention.qkv.weight",
    "attention.qkv.bias",
    "attention.projection_layer.weight",
    "attention.projection_layer.bias",
    "layernorm_after.weight",
    "layernorm_after.bias",
    "mlp.fc1.weight",
    "mlp.fc1.bias",
    "mlp.fc2.weight",
    "mlp.fc2.bias",
    "lambda_1",
    "lambda_2",
];

/// The install-side resident names under [`VISION_INSTALL_PREFIX`], in the
/// order they are written. The runtime resolves exactly these.
pub const INTERNVL_RESIDENT_TENSORS: [&str; 3] =
    ["cls_token", "patch_embed.weight", "patch_embed.bias"];

/// The position table's install name, kept apart from [`INTERNVL_RESIDENT_TENSORS`]
/// because it is the one resident whose expected SHAPE is `[rows, width]`
/// rather than `[width]`.
pub const INTERNVL_POS_EMBED_INSTALL: &str = "pos_embed.weight";

/// The projector's six install names under `vision.projector.`, the LFM3
/// projector's layout.
pub const INTERNVL_PROJECTOR_INSTALL_LEVEL: &str = "projector.";
pub const INTERNVL_PROJECTOR_TENSORS: [&str; 6] = [
    "layer_norm.weight",
    "layer_norm.bias",
    "linear_1.weight",
    "linear_1.bias",
    "linear_2.weight",
    "linear_2.bias",
];

/// The source names of the six projector tensors for each flavor, as TAILS
/// under the flavor's own projector prefix. The `internvl_chat` flavor
/// numbers the module list (`mlp1.0`, `mlp1.1`, `mlp1.3`; 2 is the
/// activation), so under `mlp1.` its tails are the bare indices; the
/// `internvl` prefix's tails are the module names.
const INTERNVL_CHAT_PROJECTOR_SOURCES: [&str; 6] = [
    "0.weight", "0.bias", "1.weight", "1.bias", "3.weight", "3.bias",
];
const INTERNVL_PROJECTOR_SOURCES: [&str; 6] = [
    "layer_norm.weight",
    "layer_norm.bias",
    "linear_1.weight",
    "linear_1.bias",
    "linear_2.weight",
    "linear_2.bias",
];

/// The only normalization the implemented tower runs. `rms_norm` is a config
/// spelling on both wrappers (`norm_type`) that no published 1B checkpoint
/// uses and no kernel here serves; refused by name rather than run as
/// LayerNorm.
const SUPPORTED_NORM_TYPE: &str = "layer_norm";
const SUPPORTED_LAYER_NORM_EPS: f64 = 1e-6;

/// Parse an InternVL `config.json`'s vision half into a [`VisionConfig`].
///
/// Accepts BOTH wrapper spellings the published checkpoints use (see the
/// module doc) and refuses any other root `model_type` BY NAME: a bare
/// `qwen2`/`qwen3` trunk config belongs to its own parser, and an unknown
/// wrapper is a checkpoint this reader has never seen. A config that omits
/// `model_type` entirely is refused too (unlike the lfm3 reader's trimmed
/// fixtures, the callers here always route a wrapper by its root string).
///
/// Absent `vision_config` is [`VisionConfig::NONE`]. Every value the runtime
/// does not store is checked here and refused when it differs from the one
/// tower this port implements: LayerNorm (not RMSNorm) blocks, no qk
/// normalization, the downsample ratio the pixel shuffle factor implies,
/// one class row, and the image/patch edge pair the position table is sized
/// from.
pub fn parse_internvl_vision_config(json: &str) -> Result<VisionConfig, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    let bad = |m: String| Gemma4Error::Config(m);
    let flavor = match root.get("model_type").and_then(|v| v.as_str()) {
        Some(INTERNVL_CHAT_MODEL_TYPE) => InternVlFlavor::Chat,
        Some(INTERNVL_MODEL_TYPE) => InternVlFlavor::Plain,
        Some(other) => {
            return Err(bad(format!(
                "config.json model_type {other:?} is neither {INTERNVL_CHAT_MODEL_TYPE:?} nor \
                 {INTERNVL_MODEL_TYPE:?}; this reader accepts only the two InternVL wrappers the \
                 published mlx-community checkpoints use"
            )))
        }
        None => {
            return Err(bad(
                "config.json declares no model_type; this reader accepts only \
                 internvl_chat and internvl"
                    .into(),
            ))
        }
    };
    let vc = match root.get("vision_config") {
        None | Some(serde_json::Value::Null) => return Ok(VisionConfig::NONE),
        Some(v) => v,
    };
    let vi = |k: &str| -> Result<i64, Gemma4Error> {
        vc.get(k)
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| bad(format!("missing vision_config.{k}")))
    };
    let text_hidden = root
        .get("text_config")
        .and_then(|t| t.get("hidden_size"))
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| bad("missing text_config.hidden_size".into()))?;

    let hidden = vi("hidden_size")?;
    let heads = vi("num_attention_heads")?;
    if heads <= 0 || hidden % heads != 0 {
        return Err(bad(format!(
            "vision hidden {hidden} is not divisible by {heads} heads"
        )));
    }
    let depth = vi("num_hidden_layers")?;
    let image = dimension(vc, "image_size")?;
    let patch = dimension(vc, "patch_size")?;
    if patch <= 0 || image <= 0 || image % patch != 0 {
        return Err(bad(format!(
            "vision image_size {image} is not a whole number of {patch}-pixel patches"
        )));
    }
    let channels = vi("num_channels")?;
    if channels != 3 {
        return Err(bad(format!(
            "vision num_channels {channels}; the implemented preprocessing is RGB"
        )));
    }
    if let Some(norm) = vc.get("norm_type").and_then(serde_json::Value::as_str) {
        if norm != SUPPORTED_NORM_TYPE {
            return Err(bad(format!(
                "vision norm_type {norm:?}; the implemented tower is {SUPPORTED_NORM_TYPE}"
            )));
        }
    }
    // The runtime hardcodes the LayerNorm epsilon and the GELU, so a config
    // stating anything else is refused rather than run with different math.
    if let Some(eps) = vc.get("layer_norm_eps") {
        if eps.as_f64() != Some(SUPPORTED_LAYER_NORM_EPS) {
            return Err(bad(format!(
                "vision layer_norm_eps {eps}; the implemented tower runs {SUPPORTED_LAYER_NORM_EPS:e}"
            )));
        }
    }
    if let Some(act) = vc.get("hidden_act") {
        if act.as_str() != Some("gelu") {
            return Err(bad(format!(
                "vision hidden_act {act}; the implemented tower runs gelu"
            )));
        }
    }
    // The two qk-normalization spellings, one per wrapper. Both must be off:
    // no kernel here runs the RMSNorm they would add to the attention path,
    // and every published 1B config declares them false or absent.
    let qk_norm = match flavor {
        InternVlFlavor::Chat => vc
            .get("qk_normalization")
            .and_then(serde_json::Value::as_bool),
        InternVlFlavor::Plain => vc.get("use_qk_norm").and_then(serde_json::Value::as_bool),
    };
    if qk_norm == Some(true) {
        return Err(bad(match flavor {
            InternVlFlavor::Chat => "vision qk_normalization true is not implemented".to_string(),
            InternVlFlavor::Plain => "vision use_qk_norm true is not implemented".to_string(),
        }));
    }
    // THE PIXEL SHUFFLE FACTOR. `downsample_ratio` is a non-binary float
    // (0.5 is, but the field is not guaranteed one), so it is not stored:
    // the manifest carries its square root as `spatial_merge_size` and this
    // check ties the two together. `int(1 / downsample_ratio)` is what the
    // reference sizes the projector's LayerNorm with.
    let downsample = root
        .get("downsample_ratio")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| bad("missing downsample_ratio".into()))?;
    if downsample <= 0.0 || (1.0 / downsample).fract() != 0.0 {
        return Err(bad(format!(
            "downsample_ratio {downsample} does not divide 1 into a whole pixel-shuffle factor"
        )));
    }
    let merge = (1.0 / downsample) as i64;
    let side = image / patch;
    if (side * side) % (merge * merge) != 0 {
        return Err(bad(format!(
            "a {side}x{side} patch grid does not shuffle whole {merge}x{merge} blocks"
        )));
    }

    // The span ids. `image_token_id` is `<IMG_CONTEXT>` (151671 on both
    // published tokenizers); the `<img>` / `</img>` ends are NOT config keys
    // and are resolved from the tokenizer by every front end, which is why
    // they are recorded here as ZERO and never read as ids by the runtime.
    let image_token_id = match flavor {
        InternVlFlavor::Chat => root
            .get("image_token_index")
            .or_else(|| root.get("image_token_id"))
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(151_667),
        InternVlFlavor::Plain => root
            .get("image_token_id")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(151_671),
    };
    let vision = VisionConfig {
        depth,
        hidden_size: hidden,
        intermediate_size: vi("intermediate_size")?,
        num_heads: heads,
        patch_size: patch,
        temporal_patch_size: 1,
        in_channels: channels,
        spatial_merge_size: merge,
        num_position_embeddings: side * side + 1,
        out_hidden_size: text_hidden,
        mrope_section: [0, 0, 0],
        deepstack_visual_indexes: Vec::new(),
        vision_start_token_id: 0,
        vision_end_token_id: 0,
        image_token_id,
        video_token_id: 0,
        tower: VisionTowerKind::InternViT,
    };
    if !vision.is_active() {
        return Err(bad("a parsed InternVL vision config must be active".into()));
    }
    Ok(vision)
}

/// Which wrapper a config declares, deciding the tensor prefixes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InternVlFlavor {
    /// `internvl_chat`: `vision_model.` tower, `mlp1.` projector.
    Chat,
    /// `internvl`: `vision_tower.` tower, `multi_modal_projector.` projector.
    Plain,
}

/// An `image_size` / `patch_size` that a config may state as `[448, 448]`
/// (the `internvl` wrapper's list form, read off the InternVL3.5 config) or
/// as a bare int. A list is required to be one value long after the pair
/// check; the reference's `_dimension` takes element 0 of any sequence.
fn dimension(vc: &serde_json::Value, key: &str) -> Result<i64, Gemma4Error> {
    let bad = |m: String| Gemma4Error::Config(m);
    match vc.get(key) {
        Some(serde_json::Value::Number(n)) => n
            .as_i64()
            .ok_or_else(|| bad(format!("vision_config.{key} is not an integer"))),
        Some(serde_json::Value::Array(items)) => {
            if items.len() != 2 {
                return Err(bad(format!(
                    "vision_config.{key} is a {}-element pair; the published configs carry two",
                    items.len()
                )));
            }
            for item in items {
                if item.as_i64().is_none() {
                    return Err(bad(format!(
                        "vision_config.{key} carries a non-integer element"
                    )));
                }
            }
            if items[0] != items[1] {
                return Err(bad(format!(
                    "vision_config.{key} = {:?} is not square; the learned position table and \
                     the tiling are sized from one edge",
                    items
                )));
            }
            items[0]
                .as_i64()
                .ok_or_else(|| bad(format!("vision_config.{key} is not an integer")))
        }
        other => Err(bad(format!(
            "missing or malformed vision_config.{key} ({other:?})"
        ))),
    }
}

/// Reads the InternViT tower and projector, converting every tensor to FP16
/// (dequantizing the packed planes), for either wrapper flavor.
///
/// Every name under either prefix must be a known block role, a known
/// resident, or a projector tensor; an unknown one is refused, as every other
/// reader refuses one, because a publisher who adds a tensor has changed the
/// tower. The `internvl` flavor's separate `attention.{q,k,v}_proj` are
/// concatenated here into one `qkv` blob in q, k, v row order, which is the
/// layout the runtime's attention dispatch reads at rows 0, hidden, 2*hidden.
pub fn read_internvl_vision_entries(
    shards: &Gemma4Shards<'_>,
    vision_bases: &[&str],
    vision: &VisionConfig,
    quant: Option<&Gemma4Quant>,
) -> Result<VisionRead, Gemma4Error> {
    if vision.tower != VisionTowerKind::InternViT || !vision.is_active() {
        return Err(Gemma4Error::Config(format!(
            "not a supported internvit tower (kind {:?}, depth {})",
            vision.tower, vision.depth
        )));
    }
    // The flavor is decided by what the header carries, not by a parameter:
    // the caller routes by config, and a config and a header that disagree is
    // a checkpoint worth refusing rather than guessing at.
    let chat = vision_bases
        .iter()
        .any(|n| n.starts_with(INTERNVL_CHAT_TOWER_PREFIX));
    let plain = vision_bases
        .iter()
        .any(|n| n.starts_with(INTERNVL_TOWER_PREFIX));
    if chat == plain {
        return Err(Gemma4Error::UnknownTensor(format!(
            "the header carries tower tensors under neither or both of \
             {INTERNVL_CHAT_TOWER_PREFIX:?} and {INTERNVL_TOWER_PREFIX:?}; the two InternVL \
             flavors cannot be told apart"
        )));
    }
    let flavor = if chat {
        InternVlFlavor::Chat
    } else {
        InternVlFlavor::Plain
    };
    let (tower_prefix, block_suffixes, projector_sources): (&str, &[&str; 14], &[&str; 6]) =
        match flavor {
            InternVlFlavor::Chat => (
                INTERNVL_CHAT_TOWER_PREFIX,
                &INTERNVL_CHAT_BLOCK_SUFFIXES,
                &INTERNVL_CHAT_PROJECTOR_SOURCES,
            ),
            InternVlFlavor::Plain => (
                INTERNVL_TOWER_PREFIX,
                &INTERNVL_BLOCK_SUFFIXES,
                &INTERNVL_PROJECTOR_SOURCES,
            ),
        };

    let depth = vision.depth as usize;
    let hidden = vision.hidden_size as usize;
    let inter = vision.intermediate_size as usize;
    let out_hidden = vision.out_hidden_size as usize;
    let patch = vision.patch_size as usize;
    let pos_rows = vision.num_position_embeddings as usize;
    let merge = vision.spatial_merge_size as usize;

    // Bucket every name first, so an unknown one is refused before any bytes
    // move. The separate q/k/v of the `internvl` flavor are recognized as one
    // logical `qkv` slot, asserted below.
    let qkv_weight_sources: [&str; 3] = [
        "attention.q_proj.weight",
        "attention.k_proj.weight",
        "attention.v_proj.weight",
    ];
    let qkv_bias_sources: [&str; 3] = [
        "attention.q_proj.bias",
        "attention.k_proj.bias",
        "attention.v_proj.bias",
    ];
    let mut block_seen = vec![[false; 14]; depth];
    // Per block, [weight, bias] x q/k/v: the parts belong to ONE block's
    // fused slot, never to the tower.
    let mut qkv_parts_seen = vec![[[false; 3]; 2]; depth];
    let mut resident_seen = [false; 3];
    let mut pos_seen = false;
    let mut projector_seen = [false; 6];
    // The two residents' source names differ per flavor; the position table
    // is a fourth resident keyed by its own slot.
    let resident_sources: [&str; 3] = match flavor {
        InternVlFlavor::Chat => [
            "embeddings.class_embedding",
            "embeddings.patch_embedding.weight",
            "embeddings.patch_embedding.bias",
        ],
        InternVlFlavor::Plain => [
            "embeddings.cls_token",
            "embeddings.patch_embeddings.projection.weight",
            "embeddings.patch_embeddings.projection.bias",
        ],
    };
    let pos_source = match flavor {
        InternVlFlavor::Chat => "embeddings.position_embedding",
        InternVlFlavor::Plain => "embeddings.position_embeddings",
    };

    for &name in vision_bases {
        if let Some(rest) = name.strip_prefix(tower_prefix) {
            if let Some((index_str, suffix)) = split_block_index(rest, flavor) {
                let index: usize = index_str.parse().map_err(|_| {
                    Gemma4Error::UnknownTensor(format!("{name}: block index is not a number"))
                })?;
                if index >= depth {
                    return Err(Gemma4Error::ShapeMismatch {
                        tensor: name.to_string(),
                        detail: format!("block {index} is past the declared depth of {depth}"),
                    });
                }
                // The `internvl` flavor's q/k/v map onto the fused qkv slots
                // (roles 2 = weight, 3 = bias), tracked per part so the
                // missing-slot report can name the source spelling.
                let (role, part) = if let Some(r) = block_suffixes.iter().position(|s| *s == suffix)
                {
                    (r, None)
                } else if let Some(i) = qkv_weight_sources.iter().position(|s| *s == suffix) {
                    (2, Some(i))
                } else if let Some(i) = qkv_bias_sources.iter().position(|s| *s == suffix) {
                    (3, Some(i))
                } else {
                    return Err(Gemma4Error::UnknownTensor(format!(
                        "{name}: {suffix} is not one of this tower's block tensors"
                    )));
                };
                if let Some(part_index) = part {
                    // One of the three qkv parts: the slot fills when the
                    // LAST part arrives, so the block_seen duplicate check
                    // fires on a fourth part, never on the second part.
                    if std::mem::replace(&mut qkv_parts_seen[index][role - 2][part_index], true) {
                        return Err(Gemma4Error::UnknownTensor(format!("{name}: duplicate")));
                    }
                    if qkv_parts_seen[index][role - 2].iter().all(|&seen| seen)
                        && std::mem::replace(&mut block_seen[index][role], true)
                    {
                        return Err(Gemma4Error::UnknownTensor(format!("{name}: duplicate")));
                    }
                    continue;
                }
                if std::mem::replace(&mut block_seen[index][role], true) {
                    return Err(Gemma4Error::UnknownTensor(format!("{name}: duplicate")));
                }
                continue;
            }
            let rest_is_pos = match flavor {
                // `internvl_chat`: `embeddings.position_embedding`.
                InternVlFlavor::Chat => rest == "embeddings.position_embedding",
                // `internvl`: `embeddings.position_embeddings`.
                InternVlFlavor::Plain => rest == "embeddings.position_embeddings",
            };
            if rest_is_pos {
                if std::mem::replace(&mut pos_seen, true) {
                    return Err(Gemma4Error::UnknownTensor(format!("{name}: duplicate")));
                }
                continue;
            }
            let slot = resident_sources
                .iter()
                .position(|r| *r == rest)
                .ok_or_else(|| {
                    Gemma4Error::UnknownTensor(format!(
                        "{name}: not a block tensor and not an InternViT tower resident"
                    ))
                })?;
            if std::mem::replace(&mut resident_seen[slot], true) {
                return Err(Gemma4Error::UnknownTensor(format!("{name}: duplicate")));
            }
            continue;
        }
        let projector_prefix = match flavor {
            InternVlFlavor::Chat => INTERNVL_CHAT_PROJECTOR_PREFIX,
            InternVlFlavor::Plain => INTERNVL_PROJECTOR_PREFIX,
        };
        if let Some(tail) = name.strip_prefix(projector_prefix) {
            let slot = projector_sources
                .iter()
                .position(|s| *s == tail)
                .ok_or_else(|| {
                    Gemma4Error::UnknownTensor(format!("{name}: not a projector tensor"))
                })?;
            if std::mem::replace(&mut projector_seen[slot], true) {
                return Err(Gemma4Error::UnknownTensor(format!("{name}: duplicate")));
            }
            continue;
        }
        return Err(Gemma4Error::UnknownTensor(format!(
            "{name}: not under an InternViT tower or projector prefix"
        )));
    }

    // Every slot filled or refused by name. The block container is spelled
    // `layers.N.` on the chat flavor and `layer.N.` on the plain one; a
    // missing fused qkv part names its own source spelling.
    let container = |index: usize| {
        format!(
            "{tower_prefix}{}{index}.",
            match flavor {
                InternVlFlavor::Chat => "encoder.layers.",
                InternVlFlavor::Plain => "encoder.layer.",
            }
        )
    };
    for (index, roles) in block_seen.iter().enumerate() {
        for (role, seen) in roles.iter().enumerate() {
            if !seen {
                let missing = match (role, flavor) {
                    (2 | 3, InternVlFlavor::Plain) => {
                        let slot = role - 2;
                        let sources = if role == 2 {
                            &qkv_weight_sources
                        } else {
                            &qkv_bias_sources
                        };
                        let part = sources
                            .iter()
                            .enumerate()
                            .find(|(i, _)| !qkv_parts_seen[index][slot][*i])
                            .map(|(_, s)| *s)
                            .unwrap_or(sources[0]);
                        format!("{}{part}", container(index))
                    }
                    _ => format!("{}{}", container(index), block_suffixes[role]),
                };
                return Err(Gemma4Error::MissingTensor(missing));
            }
        }
    }
    for (slot, seen) in resident_seen.iter().enumerate() {
        if !seen {
            return Err(Gemma4Error::MissingTensor(format!(
                "{tower_prefix}{}",
                resident_sources[slot]
            )));
        }
    }
    if !pos_seen {
        return Err(Gemma4Error::MissingTensor(format!(
            "{tower_prefix}{pos_source}"
        )));
    }
    for (slot, seen) in projector_seen.iter().enumerate() {
        if !seen {
            let projector_prefix = match flavor {
                InternVlFlavor::Chat => INTERNVL_CHAT_PROJECTOR_PREFIX,
                InternVlFlavor::Plain => INTERNVL_PROJECTOR_PREFIX,
            };
            return Err(Gemma4Error::MissingTensor(format!(
                "{projector_prefix}{}",
                projector_sources[slot]
            )));
        }
    }

    // The closure and the fused-qkv loop both report lossy conversions, so
    // the list sits behind a RefCell rather than inside the closure's borrow.
    let lossy_cell = std::cell::RefCell::new(Vec::<(String, usize)>::new());
    let load = |name: &str, want: &[usize]| -> Result<Vec<u8>, Gemma4Error> {
        let (converted, shape) = read_tower_tensor(shards, name, quant)?;
        let got: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        // Checkpoints state tables with leading batch rows ([1, rows, width]
        // for the position table, [1, 1, width] for the class token); only
        // the element count and trailing dims matter to a row-major reader,
        // so EVERY leading 1 is squeezed before the comparison.
        let all = got.as_slice();
        let stripped: &[usize] = &all[all.iter().take_while(|&&d| d == 1).count()..];
        let squeezed: &[usize] = if stripped.len() == want.len() {
            stripped
        } else {
            all
        };
        if squeezed != want {
            return Err(Gemma4Error::ShapeMismatch {
                tensor: name.to_string(),
                detail: format!("shape {got:?}, expected {want:?}"),
            });
        }
        if converted.lossy > 0 {
            lossy_cell
                .borrow_mut()
                .push((name.to_string(), converted.lossy));
        }
        Ok(converted.bytes)
    };

    // The conv weight is stored [out, kh, kw, C] (MLX's own conv layout; the
    // mlx-vlm sanitize transposes PyTorch's [out, C, kh, kw] into it), and
    // flattened row-major that is already the (P_h, P_w, C) order the
    // preprocessing emits a patch row in.
    let patch_dim = 3 * patch * patch;
    let mut entries = Vec::new();
    let h32 = hidden as u32;
    for (slot, name) in resident_sources.iter().enumerate() {
        let full = format!("{tower_prefix}{name}");
        let want: &[usize] = match slot {
            0 => &[hidden],
            1 => &[hidden, patch, patch, 3],
            _ => &[hidden],
        };
        let bytes = load(&full, want)?;
        let shape = match slot {
            0 => shape4(&[hidden as u64]),
            1 => (h32, patch_dim as u32, 0, 0),
            _ => shape4(&[hidden as u64]),
        };
        entries.push(ResidentEntrySpec::Raw(RawTensorSpec {
            name: format!("{VISION_INSTALL_PREFIX}{}", INTERNVL_RESIDENT_TENSORS[slot]),
            dtype: crate::DTYPE_FP16,
            bytes,
            shape,
        }));
    }
    let pos_bytes = load(&format!("{tower_prefix}{pos_source}"), &[pos_rows, hidden])?;
    entries.push(ResidentEntrySpec::Raw(RawTensorSpec {
        name: format!("{VISION_INSTALL_PREFIX}{INTERNVL_POS_EMBED_INSTALL}"),
        dtype: crate::DTYPE_FP16,
        bytes: pos_bytes,
        shape: (pos_rows as u32, h32, 0, 0),
    }));
    for (slot, source) in projector_sources.iter().enumerate() {
        let full = format!(
            "{}{source}",
            match flavor {
                InternVlFlavor::Chat => INTERNVL_CHAT_PROJECTOR_PREFIX,
                InternVlFlavor::Plain => INTERNVL_PROJECTOR_PREFIX,
            }
        );
        let unfolded = hidden * merge * merge;
        let want: &[usize] = match slot {
            0 | 1 => &[unfolded],
            2 => &[out_hidden, unfolded],
            3 => &[out_hidden],
            4 => &[out_hidden, out_hidden],
            _ => &[out_hidden],
        };
        let bytes = load(&full, want)?;
        let shape = match slot {
            2 => (out_hidden as u32, unfolded as u32, 0, 0),
            4 => (out_hidden as u32, out_hidden as u32, 0, 0),
            // The norm/bias/linear_2-bias shapes are one-dimensional; the
            // writer's shape4 records a single rank-1 extent.
            slot => shape4(&[match slot {
                0 | 1 => unfolded as u64,
                _ => out_hidden as u64,
            }]),
        };
        entries.push(ResidentEntrySpec::Raw(RawTensorSpec {
            name: format!(
                "{VISION_INSTALL_PREFIX}{INTERNVL_PROJECTOR_INSTALL_LEVEL}{}",
                INTERNVL_PROJECTOR_TENSORS[slot]
            ),
            dtype: crate::DTYPE_FP16,
            bytes,
            shape,
        }));
    }

    // The blocks. The `internvl` flavor's q/k/v are concatenated in q, k, v
    // row order into one `qkv_w` / `qkv_b` blob, the layout the runtime
    // reads at rows 0, hidden, 2*hidden.
    let mut experts = Vec::with_capacity(depth);
    let mut widest = 0u64;
    for index in 0..depth {
        let mut sub_tensors = Vec::with_capacity(INTERNVL_BLOCK_ROLES.len());
        let mut used = 0u64;
        for (role, role_name) in INTERNVL_BLOCK_ROLES.iter().enumerate() {
            let (bytes, shape) = if flavor == InternVlFlavor::Plain && (role == 2 || role == 3) {
                // THE FUSED QKV. q, then k, then v, in row order, so the
                // runtime's attention dispatch reads them at rows 0, hidden,
                // 2*hidden exactly as the fused `attn.qkv` sits on the chat
                // flavor. The recorded shape is the fused one.
                let sources: &[&str; 3] = if role == 2 {
                    &qkv_weight_sources
                } else {
                    &qkv_bias_sources
                };
                let mut fused = Vec::new();
                let mut rows = 0u64;
                let mut cols = 0u64;
                for source in sources {
                    let full = format!(
                        "{tower_prefix}encoder.{}{index}.{source}",
                        if flavor == InternVlFlavor::Chat {
                            "layers."
                        } else {
                            "layer."
                        }
                    );
                    let t = shards.info(&full)?;
                    // The weight sources are rank-2 [heads*head_dim, hidden];
                    // the bias sources are rank-1 [heads*head_dim].
                    // The weight sources are rank-2 [heads*head_dim, hidden];
                    // the bias sources are rank-1 [heads*head_dim].
                    let (expect_rank, trailing) = if role == 3 {
                        (1, hidden as u64)
                    } else {
                        (2, hidden as u64)
                    };
                    if t.shape.len() != expect_rank || *t.shape.last().unwrap() != trailing {
                        return Err(Gemma4Error::ShapeMismatch {
                            tensor: full.clone(),
                            detail: format!(
                                "shape {:?} is not a fused qkv part with trailing {trailing}",
                                t.shape
                            ),
                        });
                    }
                    rows += t.shape[0];
                    cols = t.shape.get(1).copied().unwrap_or(1);
                    let (converted, _) = read_tower_tensor(shards, &full, quant)?;
                    if converted.lossy > 0 {
                        lossy_cell
                            .borrow_mut()
                            .push((full.clone(), converted.lossy));
                    }
                    fused.extend_from_slice(&converted.bytes);
                }
                (fused, vec![rows, cols])
            } else {
                let suffix = block_suffixes[role];
                let full = format!(
                    "{tower_prefix}encoder.{}{index}.{suffix}",
                    if flavor == InternVlFlavor::Chat {
                        "layers."
                    } else {
                        "layer."
                    }
                );
                let t = shards.info(&full)?;
                let want: &[usize] = match role {
                    0 | 1 | 5 | 6 | 7 | 11 | 12 | 13 => &[hidden],
                    9 => &[inter],
                    // The chat flavor's fused qkv: 3*hidden rows. The plain
                    // flavor reaches this match only for the other roles.
                    2 => &[3 * hidden, hidden],
                    3 => &[3 * hidden],
                    4 => &[hidden, hidden],
                    8 => &[inter, hidden],
                    10 => &[hidden, inter],
                    _ => unreachable!("qkv slots handled above"),
                };
                let bytes = load(&full, want)?;
                (bytes, t.shape.clone())
            };
            used += bytes.len() as u64;
            sub_tensors.push(SubTensor {
                role: role_name.to_string(),
                bytes,
                dtype: "fp16".to_string(),
                shape,
            });
        }
        widest = widest.max(used);
        experts.push(ExpertBlob {
            expert: index,
            sub_tensors,
        });
    }

    let lossy_conversion = lossy_cell.into_inner();
    Ok(VisionRead {
        entries,
        blocks: LayerBlobs { layer: 0, experts },
        block_stride: widest.div_ceil(GTURBO_PAGE_BYTES) * GTURBO_PAGE_BYTES,
        lossy_conversion,
    })
}

/// Splits a tower-relative name into its block index and per-block suffix:
/// `encoder.layers.0.norm1.weight` on the chat flavor,
/// `encoder.layer.0.attention.qkv.weight` on the plain one (the published
/// indexes carry the `encoder.` container). `None` for anything that is not
/// a block tensor.
fn split_block_index(rest: &str, flavor: InternVlFlavor) -> Option<(&str, &str)> {
    let tail = match flavor {
        InternVlFlavor::Chat => rest.strip_prefix("encoder.layers.")?,
        InternVlFlavor::Plain => rest.strip_prefix("encoder.layer.")?,
    };
    tail.split_once('.')
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published `mlx-community/InternVL3-1B-4bit` config, trimmed to the
    /// keys this parser reads (@ 7927bb6b).
    fn chat_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "internvl_chat",
            "downsample_ratio": 0.5,
            "image_token_index": 151667,
            "text_config": {"hidden_size": 896, "model_type": "qwen2"},
            "vision_config": {
                "model_type": "intern_vit_6b",
                "hidden_size": 1024, "intermediate_size": 4096,
                "num_hidden_layers": 24, "num_attention_heads": 16,
                "image_size": 448, "patch_size": 14, "num_channels": 3,
                "layer_norm_eps": 1e-6, "drop_path_rate": 0.1,
                "qkv_bias": true, "qk_normalization": false,
                "norm_type": "layer_norm", "hidden_act": "gelu"
            }
        })
    }

    /// The published `mlx-community/InternVL3_5-1B-4bit` config, trimmed
    /// (@ f9d179a8): the `internvl` wrapper, list-form image/patch sizes.
    fn plain_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "internvl",
            "downsample_ratio": 0.5,
            "image_token_id": 151671,
            "text_config": {"hidden_size": 1024, "model_type": "qwen3"},
            "vision_config": {
                "model_type": "internvl_vision",
                "hidden_size": 1024, "intermediate_size": 4096,
                "num_hidden_layers": 24, "num_attention_heads": 16,
                "image_size": [448, 448], "patch_size": [14, 14],
                "num_channels": 3, "layer_norm_eps": 1e-6,
                "norm_type": "layer_norm", "attention_bias": true,
                "use_qk_norm": false, "layer_scale_init_value": 0.1,
                "hidden_act": "gelu"
            }
        })
    }

    fn parse(v: &serde_json::Value) -> Result<VisionConfig, Gemma4Error> {
        parse_internvl_vision_config(&v.to_string())
    }

    #[test]
    fn both_published_wrappers_parse_to_the_internvit_tower() {
        for (config, expect_image_token) in
            [(chat_config(), 151_667i64), (plain_config(), 151_671i64)]
        {
            let v = parse(&config).unwrap();
            assert_eq!(v.tower, VisionTowerKind::InternViT);
            assert_eq!(
                (v.depth, v.hidden_size, v.intermediate_size),
                (24, 1024, 4096)
            );
            assert_eq!((v.num_heads, v.patch_size), (16, 14));
            // 448 / 14 = 32, 32^2 + 1 class row.
            assert_eq!(v.num_position_embeddings, 1025);
            // downsample_ratio 0.5 shuffles 2x2 patches into one token.
            assert_eq!(v.spatial_merge_size, 2);
            assert_eq!(v.temporal_patch_size, 1);
            assert_eq!(v.in_channels, 3);
            assert_eq!(v.mrope_section, [0, 0, 0]);
            assert!(v.deepstack_visual_indexes.is_empty());
            assert_eq!(v.image_token_id, expect_image_token);
            // The img ends are tokenizer-resolved, never config-declared.
            assert_eq!((v.vision_start_token_id, v.vision_end_token_id), (0, 0));
            assert!(v.is_active());
        }
    }

    #[test]
    fn a_foreign_or_absent_model_type_is_refused_by_name() {
        let mut v = chat_config();
        v["model_type"] = "qwen2".into();
        let e = parse(&v).unwrap_err().to_string();
        assert!(e.contains("internvl_chat"), "{e}");
        let mut v = plain_config();
        v.as_object_mut().unwrap().remove("model_type");
        assert!(parse(&v).is_err());
    }

    #[test]
    fn the_two_refused_tower_variants_are_named() {
        let mut v = chat_config();
        v["vision_config"]["qk_normalization"] = true.into();
        let e = parse(&v).unwrap_err().to_string();
        assert!(e.contains("qk_normalization"), "{e}");
        let mut v = plain_config();
        v["vision_config"]["use_qk_norm"] = true.into();
        let e = parse(&v).unwrap_err().to_string();
        assert!(e.contains("use_qk_norm"), "{e}");
        let mut v = chat_config();
        v["vision_config"]["norm_type"] = "rms_norm".into();
        let e = parse(&v).unwrap_err().to_string();
        assert!(e.contains("rms_norm"), "{e}");
    }

    #[test]
    fn a_foreign_epsilon_or_activation_is_refused_by_name() {
        for (key, value) in [
            ("layer_norm_eps", serde_json::json!(1e-5)),
            ("hidden_act", serde_json::json!("gelu_pytorch_tanh")),
        ] {
            let mut v = chat_config();
            v["vision_config"][key] = value;
            let e = parse(&v).unwrap_err().to_string();
            assert!(e.contains(key), "{e}");
        }
    }

    #[test]
    fn a_downsample_ratio_without_a_whole_shuffle_factor_is_refused() {
        for ratio in [0.3f64, 0.25 * 3.0] {
            let mut v = chat_config();
            v["downsample_ratio"] = serde_json::json!(ratio);
            assert!(parse(&v).is_err(), "{ratio}");
        }
    }

    #[test]
    fn a_non_square_or_ragged_size_pair_is_refused() {
        let mut v = plain_config();
        v["vision_config"]["image_size"] = serde_json::json!([448, 224]);
        assert!(parse(&v).is_err());
        let mut v = plain_config();
        v["vision_config"]["image_size"] = serde_json::json!([448]);
        assert!(parse(&v).is_err());
        let mut v = chat_config();
        v["vision_config"]["image_size"] = 447.into();
        assert!(parse(&v).is_err());
    }

    #[test]
    fn a_config_with_no_vision_section_is_the_none_config() {
        let mut v = chat_config();
        v.as_object_mut().unwrap().remove("vision_config");
        assert!(!parse(&v).unwrap().is_active());
    }

    #[test]
    fn a_non_binary_fraction_stays_out_of_the_parsed_config() {
        // The parser must never copy a float field into the VisionConfig:
        // manifest floats round-trip only at binary fractions (Gotcha 24).
        // Every value it emits is an integer count or a token id, so the
        // float-bearing inputs are refused outright rather than rounded
        // into a field, and the shuffle factor arrives as the integer
        // square root of the ratio.
        let mut v = chat_config();
        v["downsample_ratio"] = serde_json::json!(0.3);
        assert!(parse(&v).is_err());
        let v = parse(&chat_config()).unwrap();
        assert_eq!(v.spatial_merge_size, 2);
        assert_eq!(v.depth, 24);
    }

    /// MUTATION GUARD: the layer-scale role table is what makes the runtime
    /// read `ls1`/`ls2` as the residual scales. Breaking the suffix table
    /// (lambda slots swapped onto the norm names) must fail the reader, not
    /// silently map one tensor onto another. Pin the two suffix tables by
    /// exact content so any edit here is a decision.
    #[test]
    fn the_two_block_suffix_tables_are_exact_and_the_layer_scale_slots_are_last() {
        assert_eq!(
            INTERNVL_BLOCK_ROLES.len(),
            INTERNVL_CHAT_BLOCK_SUFFIXES.len()
        );
        assert_eq!(INTERNVL_BLOCK_ROLES.len(), INTERNVL_BLOCK_SUFFIXES.len());
        assert_eq!(INTERNVL_BLOCK_ROLES[12], "ls1");
        assert_eq!(INTERNVL_BLOCK_ROLES[13], "ls2");
        assert_eq!(INTERNVL_CHAT_BLOCK_SUFFIXES[12], "ls1");
        assert_eq!(INTERNVL_CHAT_BLOCK_SUFFIXES[13], "ls2");
        assert_eq!(INTERNVL_BLOCK_SUFFIXES[12], "lambda_1");
        assert_eq!(INTERNVL_BLOCK_SUFFIXES[13], "lambda_2");
    }

    /// A tiny tower with every name the REAL published index uses for one
    /// flavor. The dimensions shrink (hidden 8, intermediate 12, out 16,
    /// depth 2) while the names, orders and shape RELATIONS stay the
    /// published ones, so a reader keyed on the wrong name or the wrong
    /// shape arithmetic refuses here rather than on a multi-GB stream.
    fn tiny_checkpoint(flavor: InternVlFlavor) -> (VisionConfig, Vec<String>, Vec<u8>) {
        let (hid, inter, text, depth) = (8usize, 12usize, 16usize, 2usize);
        let mut config = match flavor {
            InternVlFlavor::Chat => chat_config(),
            InternVlFlavor::Plain => plain_config(),
        };
        config["text_config"]["hidden_size"] = text.into();
        config["vision_config"]["hidden_size"] = hid.into();
        config["vision_config"]["intermediate_size"] = inter.into();
        config["vision_config"]["num_hidden_layers"] = depth.into();
        config["vision_config"]["num_attention_heads"] = 2.into();
        let vision = parse(&config).unwrap();
        let shaped = |name: String, shape: Vec<u64>, seed: u64| {
            let n: u64 = shape.iter().product();
            let mut t = crate::synthetic_tensors::bf16_vector(&name, n as usize, 0.5, seed);
            t.shape = shape;
            t
        };
        let (h, i) = (hid as u64, inter as u64);
        let pos_rows = vision.num_position_embeddings as u64; // 1025
        let mut ts: Vec<crate::synthetic_tensors::Tensor> = Vec::new();
        let block_shapes: Vec<(&str, Vec<u64>)> = match flavor {
            InternVlFlavor::Chat => vec![
                ("norm1.weight", vec![h]),
                ("norm1.bias", vec![h]),
                ("attn.qkv.weight", vec![3 * h, h]),
                ("attn.qkv.bias", vec![3 * h]),
                ("attn.proj.weight", vec![h, h]),
                ("attn.proj.bias", vec![h]),
                ("norm2.weight", vec![h]),
                ("norm2.bias", vec![h]),
                ("mlp.fc1.weight", vec![i, h]),
                ("mlp.fc1.bias", vec![i]),
                ("mlp.fc2.weight", vec![h, i]),
                ("mlp.fc2.bias", vec![h]),
                ("ls1", vec![h]),
                ("ls2", vec![h]),
            ],
            InternVlFlavor::Plain => vec![
                ("layernorm_before.weight", vec![h]),
                ("layernorm_before.bias", vec![h]),
                ("attention.q_proj.weight", vec![h, h]),
                ("attention.q_proj.bias", vec![h]),
                ("attention.k_proj.weight", vec![h, h]),
                ("attention.k_proj.bias", vec![h]),
                ("attention.v_proj.weight", vec![h, h]),
                ("attention.v_proj.bias", vec![h]),
                ("attention.projection_layer.weight", vec![h, h]),
                ("attention.projection_layer.bias", vec![h]),
                ("layernorm_after.weight", vec![h]),
                ("layernorm_after.bias", vec![h]),
                ("mlp.fc1.weight", vec![i, h]),
                ("mlp.fc1.bias", vec![i]),
                ("mlp.fc2.weight", vec![h, i]),
                ("mlp.fc2.bias", vec![h]),
                ("lambda_1", vec![h]),
                ("lambda_2", vec![h]),
            ],
        };
        for b in 0..depth {
            let base = format!(
                "{}encoder.{}{b}.",
                tower_prefix(flavor),
                match flavor {
                    InternVlFlavor::Chat => "layers.",
                    InternVlFlavor::Plain => "layer.",
                }
            );
            for (idx, (suffix, shape)) in block_shapes.iter().enumerate() {
                // A DISTINCT seed per tensor: same-shape siblings (the plain
                // flavor's q/k/v) must not be byte-identical, or a fusion-
                // order mutation would be invisible to the reader tests.
                ts.push(shaped(
                    format!("{base}{suffix}"),
                    shape.clone(),
                    10 * b as u64 + idx as u64,
                ));
            }
        }
        let (cls, patch, pos) = match flavor {
            InternVlFlavor::Chat => (
                "embeddings.class_embedding",
                "embeddings.patch_embedding",
                "embeddings.position_embedding",
            ),
            InternVlFlavor::Plain => (
                "embeddings.cls_token",
                "embeddings.patch_embeddings.projection",
                "embeddings.position_embeddings",
            ),
        };
        ts.push(shaped(
            format!("{}{cls}", tower_prefix(flavor)),
            vec![1, 1, h],
            70,
        ));
        ts.push(shaped(
            format!("{}{patch}.weight", tower_prefix(flavor)),
            vec![h, 14, 14, 3],
            71,
        ));
        ts.push(shaped(
            format!("{}{patch}.bias", tower_prefix(flavor)),
            vec![h],
            72,
        ));
        ts.push(shaped(
            format!("{}{pos}", tower_prefix(flavor)),
            vec![1, pos_rows, h],
            73,
        ));
        let (proj_norm, l1, l2) = match flavor {
            InternVlFlavor::Chat => ("0", "1", "3"),
            InternVlFlavor::Plain => ("layer_norm", "linear_1", "linear_2"),
        };
        let prefix = projector_prefix(flavor);
        let unfolded = (hid * 4) as u64;
        for (suffix, shape) in [
            (format!("{proj_norm}.weight"), vec![unfolded]),
            (format!("{proj_norm}.bias"), vec![unfolded]),
            (format!("{l1}.weight"), vec![text as u64, unfolded]),
            (format!("{l1}.bias"), vec![text as u64]),
            (format!("{l2}.weight"), vec![text as u64, text as u64]),
            (format!("{l2}.bias"), vec![text as u64]),
        ] {
            ts.push(shaped(format!("{prefix}{suffix}"), shape, 80));
        }
        let names = ts.iter().map(|t| t.name.clone()).collect();
        (
            vision,
            names,
            crate::synthetic_tensors::assemble_safetensors(&ts),
        )
    }

    fn tower_prefix(flavor: InternVlFlavor) -> &'static str {
        match flavor {
            InternVlFlavor::Chat => INTERNVL_CHAT_TOWER_PREFIX,
            InternVlFlavor::Plain => INTERNVL_TOWER_PREFIX,
        }
    }

    fn projector_prefix(flavor: InternVlFlavor) -> &'static str {
        match flavor {
            InternVlFlavor::Chat => INTERNVL_CHAT_PROJECTOR_PREFIX,
            InternVlFlavor::Plain => INTERNVL_PROJECTOR_PREFIX,
        }
    }

    fn read(flavor: InternVlFlavor) -> Result<VisionRead, Gemma4Error> {
        let (vision, names, blob) = tiny_checkpoint(flavor);
        read_from(&vision, &names, &blob)
    }

    fn read_from(
        vision: &VisionConfig,
        names: &[String],
        blob: &[u8],
    ) -> Result<VisionRead, Gemma4Error> {
        let header = crate::safetensors_header::parse_header(
            blob,
            crate::safetensors_header::DEFAULT_MAX_HEADER_BYTES,
        )
        .unwrap();
        let source = crate::ranged_download::MemoryRangeSource::new(blob);
        let shards = Gemma4Shards::single(&header, &source);
        let bases: Vec<&str> = names.iter().map(String::as_str).collect();
        // Through the shared entry point, so the tower-kind dispatch is what
        // is tested.
        super::super::vision::read_vision_entries(&shards, &bases, vision)
    }

    #[test]
    fn both_flavors_read_into_the_fourteen_role_blocks() {
        for flavor in [InternVlFlavor::Chat, InternVlFlavor::Plain] {
            let r = read(flavor).unwrap();
            assert_eq!(r.blocks.experts.len(), 2, "{flavor:?}");
            let roles = |i: usize| -> Vec<&str> {
                r.blocks.experts[i]
                    .sub_tensors
                    .iter()
                    .map(|t| t.role.as_str())
                    .collect()
            };
            assert_eq!(roles(0), INTERNVL_BLOCK_ROLES, "{flavor:?}");
            // Every fused qkv carries 3 * hidden rows at 2 bytes.
            let qkv = r.blocks.experts[0]
                .sub_tensors
                .iter()
                .find(|t| t.role == "qkv_w")
                .unwrap();
            assert_eq!(qkv.shape, vec![24, 8], "{flavor:?}");
            assert_eq!(qkv.bytes.len(), 24 * 8 * 2, "{flavor:?}");
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
                    "vision.cls_token",
                    "vision.patch_embed.weight",
                    "vision.patch_embed.bias",
                    "vision.pos_embed.weight",
                    "vision.projector.layer_norm.weight",
                    "vision.projector.layer_norm.bias",
                    "vision.projector.linear_1.weight",
                    "vision.projector.linear_1.bias",
                    "vision.projector.linear_2.weight",
                    "vision.projector.linear_2.bias",
                ],
                "{flavor:?}"
            );
            assert_eq!(r.block_stride % GTURBO_PAGE_BYTES, 0, "{flavor:?}");
        }
    }

    /// THE PLAIN FLAVOR'S FUSION: the separate q, k, v weights land in one
    /// blob in q, k, v row order, byte-for-byte the concatenation of the
    /// source tensors. A wrong order (k before q, or interleaved heads)
    /// would rotate the wrong rows into the attention kernel fluently.
    #[test]
    fn the_plain_flavors_separate_projections_fuse_in_qkv_order() {
        let (_vision, _names, blob) = tiny_checkpoint(InternVlFlavor::Plain);
        let header = crate::safetensors_header::parse_header(
            &blob,
            crate::safetensors_header::DEFAULT_MAX_HEADER_BYTES,
        )
        .unwrap();
        let source = crate::ranged_download::MemoryRangeSource::new(&blob);
        let shards = Gemma4Shards::single(&header, &source);
        let mut parts: Vec<Vec<u8>> = Vec::new();
        for part in ["q", "k", "v"] {
            let name =
                format!("{INTERNVL_TOWER_PREFIX}encoder.layer.0.attention.{part}_proj.weight");
            let t = shards.info(&name).unwrap();
            let converted = crate::gemma4_checkpoint::narrow::convert_raw_to_fp16(
                &name,
                &t.dtype,
                shards.read(&name).unwrap(),
            )
            .unwrap();
            parts.push(converted.bytes);
        }
        let r = read(InternVlFlavor::Plain).unwrap();
        let fused = &r.blocks.experts[0]
            .sub_tensors
            .iter()
            .find(|t| t.role == "qkv_w")
            .unwrap()
            .bytes;
        let total: usize = parts.iter().map(|p| p.len()).sum();
        assert_eq!(fused.len(), total);
        let mut at = 0usize;
        for (part, bytes) in ["q", "k", "v"].iter().zip(&parts) {
            assert_eq!(
                &fused[at..at + bytes.len()],
                &bytes[..],
                "{part} rows first in order"
            );
            at += bytes.len();
        }
    }

    #[test]
    fn a_missing_or_unknown_tensor_is_refused_by_name() {
        // Drop the plain flavor's lambda_2: the missing-slot report names it.
        let (vision, names, blob) = tiny_checkpoint(InternVlFlavor::Plain);
        let filtered: Vec<String> = names
            .iter()
            .filter(|n| !n.ends_with("layer.0.lambda_2"))
            .cloned()
            .collect();
        let e = read_from(&vision, &filtered, &blob)
            .err()
            .expect("missing lambda_2 refused");
        assert!(e.to_string().contains("lambda_2"), "{e}");
        // An unknown tower name is refused, not dropped.
        let (vision, mut names, blob) = tiny_checkpoint(InternVlFlavor::Chat);
        names.push(format!(
            "{INTERNVL_CHAT_TOWER_PREFIX}embeddings.something_new"
        ));
        let e = read_from(&vision, &names, &blob)
            .err()
            .expect("unknown tensor refused");
        assert!(e.to_string().contains("something_new"), "{e}");
    }
}
