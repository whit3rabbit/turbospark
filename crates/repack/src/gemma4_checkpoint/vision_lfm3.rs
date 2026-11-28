//! The LFM3-VL vision tower's config and ingest (`docs/LFM3_VL_VISION_PHASE0.md`).
//!
//! A sibling of [`super::vision`] and [`super::vision_gemma4`], not a variant
//! of either. The config half also serves the `lfm3_omni` checkpoints, whose
//! `vision_config` and projector fields are the VL ones verbatim and whose
//! audio half is refused by name (see [`LFM3_OMNI_AUDIO_TENSOR_PREFIXES`]).
//!
//! The tower alternates two block kinds, so a blob's role set
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

/// The root `model_type` of the LFM3-Omni checkpoints
/// (`LiquidAI/LFM3-Omni-2B-MLX-8bit` @ 5901d16c, unquantized
/// `LiquidAI/LFM3-Omni-2B` @ c01718a5). The plain-VL string sits beside it:
/// `lfm3_vl` for the VL wrappers, and a bare `lfm3` trunk config is refused
/// here because it belongs to the text-only parser.
pub const LFM3_VL_MODEL_TYPE: &str = "lfm3_vl";
pub const LFM3_OMNI_MODEL_TYPE: &str = "lfm3_omni";

/// The vision span ids an omni config does NOT declare. The VL config carries
/// `vision_start_token_id` / `vision_end_token_id` at the root; the omni
/// configs drop the two keys (checked against the 8-bit, 4-bit and bf16
/// builds) while keeping the same tokenizer: `LiquidAI/LFM3-Omni-2B`'s
/// `tokenizer.json` maps `<|vision_start|>` to 125009 and `<|vision_end|>` to
/// 125010, the exact ids the VL config declares, and the omni chat template
/// renders the same `<|vision_start|><|image|><|vision_end|>` span. So the
/// omni branch defaults to those two ids when absent, and a config that
/// declares OTHER values still wins; the mRoPE walk counts
/// start-then-placeholder pairs, so a wrong id here would find no images and
/// refuse, never mis-read one.
const LFM3_OMNI_DEFAULT_VISION_START_TOKEN_ID: i64 = 125_009;
const LFM3_OMNI_DEFAULT_VISION_END_TOKEN_ID: i64 = 125_010;

/// The audio sections an omni config carries beside its `vision_config`, each
/// with the `model_type` the published configs name inside it: the LFM3-Audio
/// encoder's own block and the MaskGIT depth decoder the talker reads. Read
/// off `LiquidAI/LFM3-Omni-2B-MLX-8bit` @ 5901d16c; the reference
/// `ModelConfig.__post_init__` refuses an omni config without both, and so
/// does this reader. The values are checked and then never ingested: no
/// field of an audio section reaches a [`VisionConfig`].
pub const LFM3_OMNI_AUDIO_CONFIG_SECTIONS: [(&str, &str); 2] = [
    ("audio_encoder_config", "lfm3_omni_encoder"),
    ("depth_config", "lfm3"),
];

/// The tensor prefixes the omni audio half carries, read off the published
/// `model.safetensors.index.json` of `LiquidAI/LFM3-Omni-2B-MLX-8bit`
/// @ 5901d16c (the audio tower's 170 tensors, the audio projector's 6, the
/// depth decoder's 64 and the codebook embedding: 241 in all). This port has
/// no LFM3-Audio path: classification drops every one of these by name and
/// the streamed writer reports the drop, so an omni install is image-only by
/// construction rather than by luck.
pub const LFM3_OMNI_AUDIO_TENSOR_PREFIXES: [&str; 4] = [
    "audio_tower.",
    "audio_projector.",
    "depth_decoder.",
    "codebook_embedding.",
];

/// Root keys a D1 DECISION model's config may carry. A decision model
/// (`LiquidAI/d1-Omni-2B`) declares the same root `model_type`, trunk fields
/// and `vision_config` as the chat checkpoint, so installed through the chat
/// flow it would decode fluently and wrongly. Every key is optional in the
/// reference (mlx-vlm `d1_lfm3_vl/config.py`, which also defines the two the
/// published 2B file does not set), so the class check in
/// [`refuse_lfm3_omni_d1`] is the primary discriminator and these are the
/// backstop.
pub const LFM3_OMNI_D1_KEYS: [&str; 7] = [
    "d1_attention",
    "d1_box_edges",
    "d1_codes",
    "d1_numbers",
    "d1_prompt",
    "d1_set_size",
    "d1_vision_max_pixels",
];

/// The model class of a D1 checkpoint, as `architectures` names it and as
/// `auto_map.AutoModel` ends. This is how the reference loader selects D1
/// (mlx-vlm `utils.py::get_model_type`), for `lfm3_vl` and `lfm3_omni` alike.
const D1_MODEL_CLASS: &str = "D1Model";

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

/// Refuse an `lfm3_omni` config whose audio half is not the published shape.
///
/// The audio sections are REQUIRED to be present and to name the published
/// models, for two reasons that point the same way. First, the reference
/// config's own `__post_init__` refuses an omni config without them, so a
/// config that lacks one is not a checkpoint this port has ever seen.
/// Second, their presence is what makes the audio REFUSAL explicit rather
/// than silent: this reader derives nothing from either section, the walk
/// drops the audio tensors by name ([`LFM3_OMNI_AUDIO_TENSOR_PREFIXES`]), and
/// the install that comes out is image-only because audio intake was refused
/// at every door, not because the sections went unread.
///
/// The error names the audio section, per the same rule every refusal here
/// follows: a publisher who changed the audio half has changed the
/// checkpoint, and a reader that ignored that would install a tower beside
/// an audio model it cannot describe.
fn refuse_foreign_lfm3_omni_audio(root: &serde_json::Value) -> Result<(), Gemma4Error> {
    let unsupported = |section: &str, detail: String| Gemma4Error::UnsupportedTensor {
        tensor: section.to_string(),
        detail,
    };
    for (section, model_type) in LFM3_OMNI_AUDIO_CONFIG_SECTIONS {
        let body = root.get(section).ok_or_else(|| {
            unsupported(
                section,
                "the published lfm3_omni checkpoints carry this audio section; a config \
                 without it is a shape this port has never seen"
                    .into(),
            )
        })?;
        if body.get("model_type").and_then(|v| v.as_str()) != Some(model_type) {
            return Err(unsupported(
                section,
                format!(
                    "model_type {:?} is not the published {model_type}",
                    body.get("model_type").and_then(|v| v.as_str())
                ),
            ));
        }
    }
    Ok(())
}

/// Refuse a D1 decision model wearing the `lfm3_vl` or `lfm3_omni` string.
/// `architectures` is read here only to REFUSE; family resolution still
/// ignores it.
///
/// Public because the trunk parser (`lfm3_config::parse_lfm3_config`) is the
/// FIRST gate every install flow passes through, and it must refuse a
/// decision model before its fields are ever read as a chat trunk's.
pub fn refuse_lfm3_omni_d1(root: &serde_json::Value) -> Result<(), Gemma4Error> {
    let by_class = root
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|a| a.iter().any(|v| v.as_str() == Some(D1_MODEL_CLASS)))
        || root
            .get("auto_map")
            .and_then(|m| m.get("AutoModel"))
            .and_then(serde_json::Value::as_str)
            .is_some_and(|m| m.ends_with(D1_MODEL_CLASS));
    let marker = if by_class {
        Some(D1_MODEL_CLASS)
    } else {
        LFM3_OMNI_D1_KEYS
            .into_iter()
            .find(|k| root.get(k).is_some_and(|v| !v.is_null()))
    };
    match marker {
        None => Ok(()),
        Some(marker) => Err(Gemma4Error::Config(format!(
            "config.json carries {marker}: this checkpoint is a D1 decision model \
             (LiquidAI/d1-Omni-2B), which answers through set-attention readout and has \
             no generation path on this port; refusing rather than installing it as a chat \
             lfm3 trunk"
        ))),
    }
}

/// Parse an LFM3-VL `config.json`'s vision half into a [`VisionConfig`].
///
/// Accepts BOTH wrappers the family resolves to: the `lfm3_vl` checkpoints
/// and the `lfm3_omni` ones, whose `vision_config` and projector fields are
/// the same nesting (the omni config is LFM3-VL's plus the audio sections
/// `refuse_foreign_lfm3_omni_audio` checks and refuses to ingest). Any OTHER
/// root `model_type` is refused by name: a plain `lfm3` trunk config has no
/// tower here, and an unknown wrapper is a checkpoint this reader has never
/// seen. A config that omits `model_type` entirely keeps the old behavior --
/// the trimmed fixtures in tests do, and the caller has already routed by
/// family.
///
/// Absent `vision_config` is [`VisionConfig::NONE`]. Every value the runtime
/// does not store is checked here and refused when it differs from the one
/// tower this port implements: the axial rope at theta 1e6, norm eps 1e-5,
/// the 3x3 conv, the projector's exact-GELU, bias and eps, and the layer
/// pattern's vocabulary.
pub fn parse_lfm3_vl_vision_config(json: &str) -> Result<VisionConfig, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    let bad = |m: String| Gemma4Error::Config(m);
    refuse_lfm3_omni_d1(&root)?;
    let is_omni = match root.get("model_type").and_then(|v| v.as_str()) {
        None | Some(LFM3_VL_MODEL_TYPE) => false,
        Some(LFM3_OMNI_MODEL_TYPE) => {
            refuse_foreign_lfm3_omni_audio(&root)?;
            true
        }
        Some(other) => {
            return Err(bad(format!(
                "config.json model_type {other:?} is neither {LFM3_VL_MODEL_TYPE} nor \
                 {LFM3_OMNI_MODEL_TYPE}; this reader accepts only the two wrappers the \
                 published LiquidAI checkpoints use"
            )))
        }
    };
    let vc = match root.get("vision_config") {
        None | Some(serde_json::Value::Null) => {
            if is_omni {
                return Err(bad(
                    "an lfm3_omni config without vision_config is not a checkpoint this port \
                     has seen; the published ones all declare the lfm3_vl_vision tower"
                        .into(),
                ));
            }
            return Ok(VisionConfig::NONE);
        }
        Some(v) => v,
    };
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

    // The span ids. The VL configs declare both at the root; the omni configs
    // declare neither and take the tokenizer's ids, which are the same two
    // numbers (`LFM3_OMNI_DEFAULT_VISION_*`'s doc). A config that declares a
    // different value still wins over the default.
    // Only an ABSENT (or null) key defaults: a declared non-integer refuses.
    let span_id = |key: &str, omni_default: i64| -> Result<i64, Gemma4Error> {
        if is_omni && root.get(key).is_none_or(serde_json::Value::is_null) {
            Ok(omni_default)
        } else {
            ri(key)
        }
    };
    let vision_start = span_id(
        "vision_start_token_id",
        LFM3_OMNI_DEFAULT_VISION_START_TOKEN_ID,
    )?;
    let vision_end = span_id("vision_end_token_id", LFM3_OMNI_DEFAULT_VISION_END_TOKEN_ID)?;
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
        vision_start_token_id: vision_start,
        vision_end_token_id: vision_end,
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

    /// The real omni checkpoint's config, trimmed to the keys this parser
    /// reads, with values from `LiquidAI/LFM3-Omni-2B-MLX-8bit@5901d16c`.
    /// The tower half is the VL config's; the audio sections are the omni
    /// additions this reader checks and refuses to ingest; and the two span
    /// ids the VL config declares at the root are ABSENT, as published (the
    /// 8-bit, 4-bit and bf16 builds all drop them).
    fn real_omni_config() -> serde_json::Value {
        let mut c = real_config();
        c["model_type"] = "lfm3_omni".into();
        let obj = c.as_object_mut().unwrap();
        obj.remove("vision_start_token_id");
        obj.remove("vision_end_token_id");
        c["audio_encoder_config"] = serde_json::json!({
            "model_type": "lfm3_omni_encoder",
            "hidden_size": 768,
            "num_mel_bins": 128,
        });
        c["depth_config"] = serde_json::json!({
            "model_type": "lfm3",
            "hidden_size": 1024,
            "num_hidden_layers": 3,
        });
        c
    }

    /// THE omni claim: the wrapper's tower config equals the VL one field for
    /// field, with the two span ids the omni config omits defaulting to the
    /// ids its own tokenizer carries (125009 / 125010, the same numbers the
    /// VL config declares).
    #[test]
    fn the_real_omni_config_parses_to_the_same_lfm3_vl_tower() {
        let omni = parse_lfm3_vl_vision_config(&real_omni_config().to_string()).unwrap();
        let vl = parse_lfm3_vl_vision_config(&real_config().to_string()).unwrap();
        assert_eq!(omni, vl);
        assert_eq!(omni.tower, VisionTowerKind::Lfm3Vl);
        assert_eq!(
            (
                omni.vision_start_token_id,
                omni.vision_end_token_id,
                omni.image_token_id
            ),
            (125_009, 125_010, 125_030)
        );
    }

    /// A config that declares DIFFERENT span ids wins over the defaults: the
    /// omni default exists because the published configs omit the keys, not
    /// to override a publisher.
    #[test]
    fn an_omni_config_that_declares_span_ids_keeps_them() {
        let mut c = real_omni_config();
        c["vision_start_token_id"] = 7.into();
        c["vision_end_token_id"] = 8.into();
        let v = parse_lfm3_vl_vision_config(&c.to_string()).unwrap();
        assert_eq!((v.vision_start_token_id, v.vision_end_token_id), (7, 8));
        // A declared value that is not an integer is not "absent".
        c["vision_end_token_id"] = "125010".into();
        let e = parse_lfm3_vl_vision_config(&c.to_string()).expect_err("string id");
        assert!(e.to_string().contains("vision_end_token_id"), "{e}");
    }

    #[test]
    fn an_omni_config_without_a_vision_config_is_refused() {
        let mut c = real_omni_config();
        c.as_object_mut().unwrap().remove("vision_config");
        let e = parse_lfm3_vl_vision_config(&c.to_string()).expect_err("no tower");
        assert!(e.to_string().contains("vision_config"), "{e}");
    }

    /// The published `d1-Omni-2B` and the reference loader both identify a
    /// decision model by its class, and every `d1_*` key is optional there.
    #[test]
    fn a_d1_decision_model_is_refused_by_its_class_or_any_d1_key() {
        for base in [real_config(), real_omni_config()] {
            let mut by_arch = base.clone();
            by_arch["architectures"] = serde_json::json!(["D1Model"]);
            let mut by_auto_map = base.clone();
            by_auto_map["auto_map"] =
                serde_json::json!({"AutoModel": "modeling_d1_lfm3_omni.D1Model"});
            // Spelled out, not read from the const, so dropping a key fails.
            let by_key = [
                "d1_attention",
                "d1_box_edges",
                "d1_codes",
                "d1_numbers",
                "d1_prompt",
                "d1_set_size",
                "d1_vision_max_pixels",
            ]
            .map(|key| {
                let mut c = base.clone();
                c[key] = 1.into();
                c
            });
            for c in [by_arch, by_auto_map].into_iter().chain(by_key) {
                let e = parse_lfm3_vl_vision_config(&c.to_string()).expect_err("d1");
                assert!(e.to_string().contains("D1 decision model"), "{e}");
            }
        }
    }

    /// The audio half is refused at the door, by section name: a config
    /// missing a section, or carrying one that names another audio model, is
    /// not a checkpoint this port has seen, and an audio half this reader
    /// cannot describe must never be ingested silently.
    #[test]
    fn an_omni_config_whose_audio_half_is_not_the_published_shape_is_refused() {
        for (drop, needle) in [
            ("audio_encoder_config", "audio_encoder_config"),
            ("depth_config", "depth_config"),
        ] {
            let mut c = real_omni_config();
            c.as_object_mut().unwrap().remove(drop);
            let e = parse_lfm3_vl_vision_config(&c.to_string()).expect_err(drop);
            assert!(e.to_string().contains(needle), "{e}");
        }
        let mut c = real_omni_config();
        c["audio_encoder_config"]["model_type"] = "whisper".into();
        let e = parse_lfm3_vl_vision_config(&c.to_string()).expect_err("wrong audio encoder");
        assert!(e.to_string().contains("lfm3_omni_encoder"), "{e}");
        let mut c = real_omni_config();
        c["depth_config"]["model_type"] = "lfm2".into();
        let e = parse_lfm3_vl_vision_config(&c.to_string()).expect_err("wrong depth decoder");
        assert!(e.to_string().contains("depth_config"), "{e}");
    }

    /// Mutation check for the audio refusal's OTHER half: the values inside
    /// the audio sections never reach the tower. Whatever the audio half
    /// says, the image path this reader returns is the same.
    #[test]
    fn audio_section_values_never_reach_the_tower() {
        let base = parse_lfm3_vl_vision_config(&real_omni_config().to_string()).unwrap();
        let mut c = real_omni_config();
        c["audio_encoder_config"]["hidden_size"] = 4_096.into();
        c["audio_encoder_config"]["num_mel_bins"] = 80.into();
        c["depth_config"]["num_hidden_layers"] = 9.into();
        c["audio_token_id"] = 42.into();
        c["audio_out_token_id"] = 43.into();
        assert_eq!(parse_lfm3_vl_vision_config(&c.to_string()).unwrap(), base);
    }

    /// The D1 decision model wears the same model_type, trunk fields and
    /// vision_config as the chat omni; its root `d1_*` keys are what refuses
    /// it, in the vision reader AND in the trunk parser (the latter is pinned
    /// in `tests/lfm3_config.rs`).
    #[test]
    fn a_d1_decision_model_is_refused_before_its_tower_is_read() {
        for key in LFM3_OMNI_D1_KEYS {
            let mut c = real_omni_config();
            c[key] = serde_json::json!({"probe": true});
            let e = parse_lfm3_vl_vision_config(&c.to_string()).expect_err(key);
            assert!(e.to_string().contains("D1 decision model"), "{e}");
        }
    }

    /// The reader accepts exactly the two wrappers the published LiquidAI
    /// checkpoints use. A plain `lfm3` trunk config has no tower here (its
    /// published configs carry either no `vision_config` or an empty one, and
    /// no vision path can reach this reader for one), and any other wrapper
    /// is a checkpoint this port has never seen.
    #[test]
    fn a_root_model_type_that_is_neither_wrapper_is_refused() {
        let mut c = real_config();
        c["model_type"] = "lfm3".into();
        let e = parse_lfm3_vl_vision_config(&c.to_string()).expect_err("plain trunk");
        assert!(e.to_string().contains("lfm3_vl"), "{e}");
        let mut c = real_config();
        c["model_type"] = "lfm2_vl".into();
        let e = parse_lfm3_vl_vision_config(&c.to_string()).expect_err("unknown wrapper");
        assert!(e.to_string().contains("lfm2_vl"), "{e}");
    }

    /// The audio tensors' classification: the four prefixes the published
    /// omni index carries are dropped (never ingested as trunk, tower, or
    /// unknown-name error), while the tower and projector prefixes beside
    /// them stay the tower. Read off the real index's own prefixes.
    #[test]
    fn the_omni_audio_prefixes_are_dropped_and_the_tower_is_ingested() {
        use crate::gemma4_checkpoint::{classify_for_family, Gemma4Bucket};
        let audio = [
            "audio_tower.lfm.layers.0.conv.conv.weight",
            "audio_projector.linear_1.weight",
            "depth_decoder.model.layers.0.self_attn.q_proj.weight",
            "codebook_embedding.weight",
        ];
        for name in audio {
            assert_eq!(
                classify_for_family(name, 24, model_io::ModelFamily::Lfm3),
                Gemma4Bucket::ExcludedMultimodal,
                "{name}"
            );
        }
        for (name, bucket) in [
            (
                "vision_tower.layers.0.operator_norm.weight",
                Gemma4Bucket::VisionTower,
            ),
            (
                "multi_modal_projector.linear_1.weight",
                Gemma4Bucket::VisionTower,
            ),
            (
                "language_model.model.layers.0.self_attn.q_proj.weight",
                Gemma4Bucket::LmResident,
            ),
        ] {
            assert_eq!(
                classify_for_family(name, 24, model_io::ModelFamily::Lfm3),
                bucket,
                "{name}"
            );
        }
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
