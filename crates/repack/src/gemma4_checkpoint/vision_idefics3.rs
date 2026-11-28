//! The variable-size SigLIP tower the Idefics3 and SmolVLM wrappers carry:
//! config and ingest (`docs/IDEFICS3_VISION_PHASE0.md`).
//!
//! **Two wrappers, one tower.** SmolVLM is an Idefics3 subclass that imports
//! the tower, the connector and the processing verbatim (`smolvlm.py`'s whole
//! body is `Model = Idefics3Model`), so ONE reader covers both and the
//! published MLX conversions spell every tensor the SAME way:
//!
//! - `idefics3` (root `model_type`, trunk `llama` per
//!   `mlx-community/Idefics3-8B-Llama3-4bit` @
//!   6b25ea35bf0ecd3122dba250f78e52167e2f8d0e and every published SmolVLM
//!   build read for this port): tower prefix `vision_model.` (separate
//!   `self_attn.{q,k,v,out}_proj`, fused here into one `qkv` blob in q, k, v
//!   row order), projector prefix `connector.modality_projection.proj.`.
//! - `smolvlm` (the same shape under its own root string; no published
//!   mlx-community conversion spells it, so the row admits the string the
//!   upstream config class registers and the reader accepts it beside
//!   `idefics3` rather than refusing a name the same tower answers to).
//!
//! The tower is the InternVL reader's twelve-role blob layout MINUS the two
//! layer-scale rows: the block is LayerNorm / biased attention / LayerNorm /
//! biased fc-fc MLP with NO layer scale and NO cls row. The connector is a
//! pixel shuffle (factor = the config's `scale_factor`, which is 2 on the
//! 8B and 3 and 4 on the published SmolVLM builds) plus ONE bias-free
//! Linear -- the mlx-vlm connector carries no LayerNorm, and neither
//! published index carries a connector one.

use model_io::{VisionConfig, VisionTowerKind};

use super::classify::VISION_INSTALL_PREFIX;
use super::config::{Gemma4Error, Gemma4Quant};
use super::shards::{shape4, Gemma4Shards, GTURBO_PAGE_BYTES};
use super::vision::{read_tower_tensor, VisionRead};
use crate::gturbo_writer::{ExpertBlob, LayerBlobs, SubTensor};
use crate::resident_writer::{RawTensorSpec, ResidentEntrySpec};

/// The root `model_type` of the Idefics3 wrapper. Every published
/// mlx-community SmolVLM conversion also declares this string at the root
/// (the three SmolVLM builds read for this port say `idefics3`, not
/// `smolvlm`), because the conversion ran through mlx-vlm's idefics3 path.
pub const IDEFICS3_MODEL_TYPE: &str = "idefics3";
/// The root `model_type` the upstream SmolVLM config class registers.
/// Accepted beside [`IDEFICS3_MODEL_TYPE`] for the same tower; no published
/// mlx-community build has been seen spelling it yet.
pub const SMOLVLM_MODEL_TYPE: &str = "smolvlm";

/// The tower's prefix. Both wrappers spell it this way on the published
/// indexes (`vision_tower.` was the InternVL plain flavor's spelling; no
/// Idefics3/SmolVLM index read for this port uses it).
pub const IDEFICS3_TOWER_PREFIX: &str = "vision_model.";
/// The projector's prefix: mlx-vlm's `Idefics3Connector.modality_projection`
/// holding one `proj` Linear. A `connector.` tensor that is not this proj is
/// refused by name.
pub const IDEFICS3_PROJECTOR_PREFIX: &str = "connector.modality_projection.proj.";

/// The connector's container prefix, what [`classify_for_family`] buckets as
/// the tower beside [`IDEFICS3_TOWER_PREFIX`]. Everything under it that is
/// not [`IDEFICS3_PROJECTOR_PREFIX`]weight is refused by name.
pub const IDEFICS3_CONNECTOR_PREFIX: &str = "connector.";

/// The block's twelve roles, in forward-pass order: the qwen tower's twelve
/// with this family's suffixes. There is NO layer scale on this tower, and
/// the runtime resolves exactly these.
pub const IDEFICS3_BLOCK_ROLES: [&str; 12] = [
    "ln1_w", "ln1_b", "qkv_w", "qkv_b", "proj_w", "proj_b", "ln2_w", "ln2_b", "fc1_w", "fc1_b",
    "fc2_w", "fc2_b",
];

/// The per-block source suffixes for the roles that are NOT the fused qkv,
/// in role order (q/k/v are separate tensors, fused in
/// [`read_idefics3_vision_entries`]).
pub const IDEFICS3_BLOCK_SUFFIXES: [&str; 10] = [
    "layer_norm1.weight",
    "layer_norm1.bias",
    "self_attn.out_proj.weight",
    "self_attn.out_proj.bias",
    "layer_norm2.weight",
    "layer_norm2.bias",
    "mlp.fc1.weight",
    "mlp.fc1.bias",
    "mlp.fc2.weight",
    "mlp.fc2.bias",
];

const QKV_WEIGHT_SOURCES: [&str; 3] = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
];
const QKV_BIAS_SOURCES: [&str; 3] = [
    "self_attn.q_proj.bias",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.bias",
];

/// The install-side resident names under [`VISION_INSTALL_PREFIX`], in the
/// order they are written. The runtime resolves exactly these.
pub const IDEFICS3_RESIDENT_TENSORS: [&str; 5] = [
    "patch_embed.weight",
    "patch_embed.bias",
    "pos_embed.weight",
    "post_norm.weight",
    "post_norm.bias",
];

/// The projector's install name under `vision.projector.`, beside the
/// InternVL projector's levels. One Linear, no bias: the mlx-vlm connector
/// carries none and no published index carries one.
pub const IDEFICS3_PROJECTOR_INSTALL_LEVEL: &str = "projector.proj.weight";

/// The `vision_config.model_type` spellings mlx-vlm's VisionModel accepts
/// (`vision.py`'s own allowlist), which is every spelling the published
/// conversions carry: `idefics3_vision` on the 8B and the SmolVLM 256M/500M
/// builds, `idefics3` on SmolVLM-Instruct.
const SUPPORTED_VISION_MODEL_TYPES: [&str; 4] = [
    "siglip_vision_model",
    "idefics3",
    "idefics3_vision",
    "smolvlm_vision",
];

/// The tower's LayerNorm epsilon. Every published config declares 1e-6; a
/// different value is refused rather than run, because no kernel here is
/// parameterized over it (the manifest's float-free rule).
const SUPPORTED_LAYER_NORM_EPS: f64 = 1e-6;

/// Parse an Idefics3 / SmolVLM `config.json`'s vision half into a
/// [`VisionConfig`].
///
/// Accepts BOTH wrapper spellings the tower answers to and refuses any other
/// root `model_type` BY NAME: a bare `llama` trunk config belongs to its own
/// parser, and an unknown wrapper is a checkpoint this reader has never seen.
/// A config that omits `model_type` entirely is refused too (the callers
/// route a wrapper by its root string).
///
/// Absent `vision_config` is [`VisionConfig::NONE`]. Every value the runtime
/// does not store is checked here and refused when it differs from the one
/// tower this port implements: a SigLIP-family vision model_type, LayerNorm
/// (eps 1e-6) blocks, three RGB channels, an image edge that is a whole
/// number of patches, a pixel-shuffle factor that divides the native patch
/// grid into whole blocks, and a `<image>` id to expand prompts around.
pub fn parse_idefics3_vision_config(json: &str) -> Result<VisionConfig, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    let bad = |m: String| Gemma4Error::Config(m);
    match root.get("model_type").and_then(|v| v.as_str()) {
        Some(IDEFICS3_MODEL_TYPE) | Some(SMOLVLM_MODEL_TYPE) => {}
        Some(other) => {
            return Err(bad(format!(
                "config.json model_type {other:?} is neither {IDEFICS3_MODEL_TYPE:?} nor \
                 {SMOLVLM_MODEL_TYPE:?}; this reader accepts only the two wrappers the \
                 variable-size SigLIP tower answers to"
            )))
        }
        None => {
            return Err(bad(
                "config.json declares no model_type; this reader accepts only idefics3 and \
                 smolvlm"
                    .into(),
            ))
        }
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
    let text_hidden = root
        .get("text_config")
        .and_then(|t| t.get("hidden_size"))
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| bad("missing text_config.hidden_size".into()))?;

    if let Some(kind) = vc.get("model_type").and_then(serde_json::Value::as_str) {
        if !SUPPORTED_VISION_MODEL_TYPES.contains(&kind) {
            return Err(bad(format!(
                "vision_config.model_type {kind:?} is not one this tower runs ({:?})",
                SUPPORTED_VISION_MODEL_TYPES
            )));
        }
    }
    if let Some(act) = vc.get("hidden_act").and_then(serde_json::Value::as_str) {
        if act != "gelu_pytorch_tanh" {
            return Err(bad(format!(
                "vision hidden_act {act:?}; the implemented tower runs gelu_pytorch_tanh"
            )));
        }
    }
    if root
        .get("text_config")
        .and_then(|t| t.get("use_resampler"))
        .and_then(serde_json::Value::as_bool)
        == Some(true)
    {
        return Err(bad(
            "use_resampler is true; this connector is the pixel shuffle and one Linear".into(),
        ));
    }
    let hidden = vi("hidden_size")?;
    let heads = vi("num_attention_heads")?;
    if heads <= 0 || hidden <= 0 || hidden % heads != 0 {
        return Err(bad(format!(
            "vision hidden {hidden} is not divisible by {heads} heads"
        )));
    }
    let depth = vi("num_hidden_layers")?;
    let image = vi("image_size")?;
    let patch = vi("patch_size")?;
    let intermediate = vi("intermediate_size")?;
    // NOT `image % patch == 0`: SmolVLM-Instruct publishes 384 over 14, and
    // the unpadded patch conv drops the last 6 pixels (27 patches a side,
    // a 729-row position table).
    if patch <= 0 || image < patch || depth <= 0 || intermediate <= 0 {
        return Err(bad(format!(
            "vision image_size {image}, patch_size {patch}, depth {depth}, intermediate \
             {intermediate} do not describe a tower"
        )));
    }
    let channels = vi("num_channels")?;
    if channels != 3 {
        return Err(bad(format!(
            "vision num_channels {channels}; the implemented preprocessing is RGB"
        )));
    }
    if let Some(eps) = vc.get("layer_norm_eps").and_then(serde_json::Value::as_f64) {
        if (eps - SUPPORTED_LAYER_NORM_EPS).abs() > 1e-12 {
            return Err(bad(format!(
                "vision layer_norm_eps {eps:e}; the implemented tower runs \
                 {SUPPORTED_LAYER_NORM_EPS:e}"
            )));
        }
    }
    // THE PIXEL SHUFFLE FACTOR. The config's own `scale_factor` (2 on the 8B,
    // 3 on SmolVLM-Instruct, 4 on the SmolVLM 256M/500M builds), stored as
    // `VisionConfig.spatial_merge_size` and checked against the native patch
    // grid: a grid that does not shuffle whole scale x scale blocks would
    // misplace every connector row.
    let scale = match root.get("scale_factor") {
        None | Some(serde_json::Value::Null) => 2,
        Some(v) => v.as_i64().ok_or_else(|| {
            bad(format!(
                "scale_factor {v} is not an integer pixel-shuffle factor"
            ))
        })?,
    };
    if !(1..=8).contains(&scale) {
        return Err(bad(format!(
            "scale_factor {scale} is outside the pixel-shuffle factors this port runs"
        )));
    }
    let side = image / patch;
    if side % scale != 0 {
        return Err(bad(format!(
            "scale_factor {scale}: a {side}x{side} patch grid does not shuffle whole blocks"
        )));
    }
    // The processor's token count per tile. The prompt expansion derives it
    // from the geometry, so a config that declares another number is refused.
    if let Some(declared) = root
        .get("image_seq_len")
        .and_then(serde_json::Value::as_i64)
    {
        let derived = (side / scale) * (side / scale);
        if declared != derived {
            return Err(bad(format!(
                "image_seq_len {declared} disagrees with the {derived} tokens the geometry gives"
            )));
        }
    }

    // THE `<image>` ID. Both published tokenizers spell it as ONE added
    // token and the prompt expansion runs on it; the reference's default
    // (49153) is WRONG for two of the three published vocabularies (it names
    // `<row_1_col_1>` on SmolVLM), so an absent id is refused rather than
    // defaulted.
    let image_token_id = root
        .get("image_token_id")
        .or_else(|| root.get("image_token_index"))
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| {
            bad(
                "missing image_token_id (and image_token_index): the prompt expansion needs \
                 the `<image>` id the checkpoint itself declares"
                    .into(),
            )
        })?;
    let vision = VisionConfig {
        depth,
        hidden_size: hidden,
        intermediate_size: intermediate,
        num_heads: heads,
        patch_size: patch,
        temporal_patch_size: 1,
        in_channels: channels,
        spatial_merge_size: scale,
        // NO class row on this tower: the table is the bare patch grid.
        num_position_embeddings: side * side,
        out_hidden_size: text_hidden,
        mrope_section: [0, 0, 0],
        deepstack_visual_indexes: Vec::new(),
        vision_start_token_id: 0,
        vision_end_token_id: 0,
        image_token_id,
        video_token_id: 0,
        tower: VisionTowerKind::SigLipNaViT,
    };
    if !vision.is_active() {
        return Err(bad("a parsed Idefics3 vision config must be active".into()));
    }
    Ok(vision)
}

/// Reads the variable-size SigLIP tower and its connector, converting every
/// tensor to FP16 (dequantizing the packed planes; the SmolVLM builds
/// quantize the tower bodies, the position table and the projector alike).
///
/// Every name under either prefix must be a known block role, a known
/// resident, or the connector's one proj weight; an unknown one is refused,
/// as every other reader refuses one, because a publisher who adds a tensor
/// has changed the tower. The separate `self_attn.{q,k,v}_proj` are
/// concatenated into one `qkv` blob in q, k, v row order, which is the
/// layout the runtime's attention dispatch reads at rows 0, hidden,
/// 2*hidden.
pub fn read_idefics3_vision_entries(
    shards: &Gemma4Shards<'_>,
    vision_bases: &[&str],
    vision: &VisionConfig,
    quant: Option<&Gemma4Quant>,
) -> Result<VisionRead, Gemma4Error> {
    if vision.tower != VisionTowerKind::SigLipNaViT || !vision.is_active() {
        return Err(Gemma4Error::Config(format!(
            "not a supported siglip_navit tower (kind {:?}, depth {})",
            vision.tower, vision.depth
        )));
    }
    let tower_prefix = IDEFICS3_TOWER_PREFIX;
    let depth = vision.depth as usize;
    let hidden = vision.hidden_size as usize;
    let inter = vision.intermediate_size as usize;
    let out_hidden = vision.out_hidden_size as usize;
    let patch = vision.patch_size as usize;
    let pos_rows = vision.num_position_embeddings as usize;
    let scale = vision.spatial_merge_size as usize;
    let unfolded = hidden * scale * scale;

    let mut block_seen = vec![[false; 12]; depth];
    // Per block, [weight, bias] x q/k/v: the parts belong to ONE block's
    // fused slot, never to the tower.
    let mut qkv_parts_seen = vec![[[false; 3]; 2]; depth];
    let mut resident_seen = [false; 5];
    let mut proj_seen = false;

    let resident_sources: [&str; 5] = [
        "embeddings.patch_embedding.weight",
        "embeddings.patch_embedding.bias",
        "embeddings.position_embedding.weight",
        "post_layernorm.weight",
        "post_layernorm.bias",
    ];

    for &name in vision_bases {
        if let Some(rest) = name.strip_prefix(tower_prefix) {
            if let Some((index_str, suffix)) = rest
                .strip_prefix("encoder.layers.")
                .and_then(|r| r.split_once('.'))
            {
                let index: usize = index_str.parse().map_err(|_| {
                    Gemma4Error::UnknownTensor(format!("{name}: block index is not a number"))
                })?;
                if index >= depth {
                    return Err(Gemma4Error::ShapeMismatch {
                        tensor: name.to_string(),
                        detail: format!("block {index} is past the declared depth of {depth}"),
                    });
                }
                // The ten direct suffixes map onto the twelve roles around
                // the two fused-qkv slots (suffix slot 2, out_proj.weight,
                // is role 4); q/k/v reach their slots through the per-part
                // tables below.
                let (role, part) =
                    if let Some(r) = IDEFICS3_BLOCK_SUFFIXES.iter().position(|s| *s == suffix) {
                        (if r < 2 { r } else { r + 2 }, None)
                    } else if let Some(i) = QKV_WEIGHT_SOURCES.iter().position(|s| *s == suffix) {
                        (2, Some(i))
                    } else if let Some(i) = QKV_BIAS_SOURCES.iter().position(|s| *s == suffix) {
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
            let slot = resident_sources
                .iter()
                .position(|r| *r == rest)
                .ok_or_else(|| {
                    Gemma4Error::UnknownTensor(format!(
                        "{name}: not a block tensor and not a variable-size SigLIP tower \
                         resident"
                    ))
                })?;
            if std::mem::replace(&mut resident_seen[slot], true) {
                return Err(Gemma4Error::UnknownTensor(format!("{name}: duplicate")));
            }
            continue;
        }
        if let Some(tail) = name.strip_prefix(IDEFICS3_CONNECTOR_PREFIX) {
            // The connector holds ONE Linear. Any other tensor under it is a
            // shape of the connector this port has never seen (the HF
            // original's LayerNorm lives here on some builds) and is refused
            // by name rather than dropped.
            if tail != "modality_projection.proj.weight" {
                return Err(Gemma4Error::UnknownTensor(format!(
                    "{name}: the connector this port runs carries only \
                     modality_projection.proj.weight (no bias, no LayerNorm)"
                )));
            }
            if std::mem::replace(&mut proj_seen, true) {
                return Err(Gemma4Error::UnknownTensor(format!("{name}: duplicate")));
            }
            continue;
        }
        return Err(Gemma4Error::UnknownTensor(format!(
            "{name}: not under the {IDEFICS3_TOWER_PREFIX} tower or the connector prefix"
        )));
    }

    // Every slot filled or refused by name.
    let container = |index: usize| format!("{tower_prefix}encoder.layers.{index}.");
    for (index, roles) in block_seen.iter().enumerate() {
        for (role, seen) in roles.iter().enumerate() {
            if *seen {
                continue;
            }
            let missing = if role == 2 || role == 3 {
                let sources = if role == 2 {
                    &QKV_WEIGHT_SOURCES
                } else {
                    &QKV_BIAS_SOURCES
                };
                let part = (0..3)
                    .find(|&i| !qkv_parts_seen[index][role - 2][i])
                    .unwrap_or(0);
                sources[part]
            } else {
                IDEFICS3_BLOCK_SUFFIXES[suffix_slot(role)]
            };
            return Err(Gemma4Error::MissingTensor(format!(
                "{}{missing}",
                container(index)
            )));
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
    if !proj_seen {
        return Err(Gemma4Error::MissingTensor(format!(
            "{IDEFICS3_PROJECTOR_PREFIX}weight"
        )));
    }

    // The closure and the fused-qkv loop both report lossy conversions, so
    // the list sits behind a RefCell rather than inside the closure's borrow.
    let lossy_cell = std::cell::RefCell::new(Vec::<(String, usize)>::new());
    let load = |name: &str, want: &[usize]| -> Result<Vec<u8>, Gemma4Error> {
        let (converted, shape) = read_tower_tensor(shards, name, quant)?;
        let got: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        // Checkpoints state tables with leading batch rows ([1, rows, width]
        // for the position table); only the element count and trailing dims
        // matter to a row-major reader, so EVERY leading 1 is squeezed
        // before the comparison.
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
    let want_shapes: [&[usize]; 5] = [
        &[hidden, patch, patch, 3],
        &[hidden],
        &[pos_rows, hidden],
        &[hidden],
        &[hidden],
    ];
    for (slot, name) in resident_sources.iter().enumerate() {
        let full = format!("{tower_prefix}{name}");
        let bytes = load(&full, want_shapes[slot])?;
        let shape = match slot {
            0 => (h32, patch_dim as u32, 0, 0),
            2 => (pos_rows as u32, h32, 0, 0),
            // The biases and the two post-norm planes are rank-1 [hidden];
            // shape4 records a single rank-1 extent.
            _ => shape4(&[hidden as u64]),
        };
        entries.push(ResidentEntrySpec::Raw(RawTensorSpec {
            name: format!("{VISION_INSTALL_PREFIX}{}", IDEFICS3_RESIDENT_TENSORS[slot]),
            dtype: crate::DTYPE_FP16,
            bytes,
            shape,
        }));
    }
    let proj_bytes = load(
        &format!("{IDEFICS3_PROJECTOR_PREFIX}weight"),
        &[out_hidden, unfolded],
    )?;
    entries.push(ResidentEntrySpec::Raw(RawTensorSpec {
        name: format!("{VISION_INSTALL_PREFIX}{IDEFICS3_PROJECTOR_INSTALL_LEVEL}"),
        dtype: crate::DTYPE_FP16,
        bytes: proj_bytes,
        shape: (out_hidden as u32, unfolded as u32, 0, 0),
    }));

    // The blocks. The separate q/k/v are concatenated in q, k, v row order
    // into one `qkv_w` / `qkv_b` blob, the layout the runtime reads at rows
    // 0, hidden, 2*hidden.
    let mut experts = Vec::with_capacity(depth);
    let mut widest = 0u64;
    for index in 0..depth {
        let mut sub_tensors = Vec::with_capacity(IDEFICS3_BLOCK_ROLES.len());
        let mut used = 0u64;
        for (role, role_name) in IDEFICS3_BLOCK_ROLES.iter().enumerate() {
            let (bytes, shape) = if role == 2 || role == 3 {
                // THE FUSED QKV. q, then k, then v, in row order. Each part is
                // checked at its LOGICAL shape: a packed plane's stored shape
                // is `[rows, words]`.
                let sources: &[&str; 3] = if role == 2 {
                    &QKV_WEIGHT_SOURCES
                } else {
                    &QKV_BIAS_SOURCES
                };
                let want: &[usize] = if role == 2 {
                    &[hidden, hidden]
                } else {
                    &[hidden]
                };
                let mut fused = Vec::new();
                for source in sources {
                    fused.extend(load(&format!("{}{source}", container(index)), want)?);
                }
                let shape = if role == 2 {
                    vec![3 * hidden as u64, hidden as u64]
                } else {
                    vec![3 * hidden as u64]
                };
                (fused, shape)
            } else {
                let suffix = IDEFICS3_BLOCK_SUFFIXES[suffix_slot(role)];
                let want: &[usize] = match role {
                    // proj (out_proj) is [hidden, hidden]; fc1 is
                    // [inter, hidden] and fc2 [hidden, inter]; every norm
                    // and bias is [hidden] except fc1's.
                    4 => &[hidden, hidden],
                    8 => &[inter, hidden],
                    10 => &[hidden, inter],
                    9 => &[inter],
                    _ => &[hidden],
                };
                let bytes = load(&format!("{}{suffix}", container(index)), want)?;
                (bytes, want.iter().map(|&d| d as u64).collect())
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

/// The [`IDEFICS3_BLOCK_SUFFIXES`] slot of a role that is not a fused qkv one:
/// the two qkv roles sit between the first norm and `out_proj`.
fn suffix_slot(role: usize) -> usize {
    if role < 2 {
        role
    } else {
        role - 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published `mlx-community/Idefics3-8B-Llama3-4bit` config, trimmed
    /// to the keys this parser reads (@ b9a23281).
    fn idefics3_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "idefics3",
            "scale_factor": 2,
            "image_token_id": 128257,
            "text_config": {"hidden_size": 4096, "model_type": "llama"},
            "vision_config": {
                "model_type": "idefics3_vision",
                "hidden_size": 1152, "intermediate_size": 4304,
                "num_hidden_layers": 27, "num_attention_heads": 16,
                "image_size": 364, "patch_size": 14, "num_channels": 3,
                "layer_norm_eps": 1e-6
            }
        })
    }

    /// The published `mlx-community/SmolVLM-256M-Instruct-4bit` config,
    /// trimmed (@ 69cb5195): the SAME wrapper spelling, scale factor 4.
    fn smolvlm_config() -> serde_json::Value {
        let mut v = idefics3_config();
        v["image_token_id"] = 49190.into();
        v["scale_factor"] = 4.into();
        v["text_config"] = serde_json::json!({"hidden_size": 576, "model_type": "llama"});
        v["vision_config"] = serde_json::json!({
            "model_type": "idefics3_vision",
            "hidden_size": 768, "intermediate_size": 3072,
            "num_hidden_layers": 12, "num_attention_heads": 12,
            "image_size": 512, "patch_size": 16, "num_channels": 3,
            "layer_norm_eps": 1e-6
        });
        v
    }

    fn parse(v: &serde_json::Value) -> Result<VisionConfig, Gemma4Error> {
        parse_idefics3_vision_config(&v.to_string())
    }

    #[test]
    fn both_wrappers_parse_to_the_siglip_navit_tower() {
        let v = parse(&idefics3_config()).unwrap();
        assert_eq!(v.tower, VisionTowerKind::SigLipNaViT);
        assert_eq!(
            (v.depth, v.hidden_size, v.intermediate_size),
            (27, 1152, 4304)
        );
        assert_eq!((v.num_heads, v.patch_size), (16, 14));
        // 364 / 14 = 26; the table is the bare grid, NO class row.
        assert_eq!(v.num_position_embeddings, 676);
        assert_eq!(v.spatial_merge_size, 2);
        assert_eq!(v.out_hidden_size, 4096);
        assert_eq!(v.image_token_id, 128257);
        assert_eq!(v.temporal_patch_size, 1);
        assert_eq!(v.in_channels, 3);
        assert_eq!(v.mrope_section, [0, 0, 0]);
        assert_eq!((v.vision_start_token_id, v.vision_end_token_id), (0, 0));
        assert!(v.is_active());

        let v = parse(&smolvlm_config()).unwrap();
        assert_eq!(v.tower, VisionTowerKind::SigLipNaViT);
        assert_eq!(
            (v.depth, v.hidden_size, v.intermediate_size),
            (12, 768, 3072)
        );
        // 512 / 16 = 32; scale 4 gives 64 tokens per tile.
        assert_eq!(v.num_position_embeddings, 1024);
        assert_eq!(v.spatial_merge_size, 4);
        assert_eq!(v.out_hidden_size, 576);
        assert_eq!(v.image_token_id, 49190);
        assert!(v.is_active());
    }

    #[test]
    fn the_smolvlm_root_string_is_accepted_and_a_foreign_one_refused() {
        let mut v = smolvlm_config();
        v["model_type"] = "smolvlm".into();
        assert!(parse(&v).is_ok());
        let mut v = smolvlm_config();
        v["model_type"] = "llava".into();
        let e = parse(&v).unwrap_err().to_string();
        assert!(e.contains("llava"), "{e}");
        let mut v = smolvlm_config();
        v.as_object_mut().unwrap().remove("model_type");
        assert!(parse(&v).is_err());
    }

    #[test]
    fn the_refused_tower_variants_are_named() {
        let mut v = idefics3_config();
        v["vision_config"]["model_type"] = "siglip_so400m".into();
        let e = parse(&v).unwrap_err().to_string();
        assert!(e.contains("siglip_so400m"), "{e}");
        let mut v = idefics3_config();
        v["vision_config"]["layer_norm_eps"] = serde_json::json!(1e-5);
        let e = parse(&v).unwrap_err().to_string();
        assert!(e.contains("layer_norm_eps"), "{e}");
        let mut v = idefics3_config();
        v["vision_config"]["num_channels"] = 1.into();
        let e = parse(&v).unwrap_err().to_string();
        assert!(e.contains("num_channels"), "{e}");
        let mut v = idefics3_config();
        v["scale_factor"] = 7.into();
        let e = parse(&v).unwrap_err().to_string();
        assert!(e.contains("scale_factor"), "{e}");
    }

    #[test]
    fn a_scale_that_raggeds_the_grid_or_a_missing_image_id_is_refused() {
        let mut v = idefics3_config();
        v["scale_factor"] = 8.into(); // 26 % 8 != 0
        assert!(parse(&v).is_err());
        let mut v = idefics3_config();
        v.as_object_mut().unwrap().remove("image_token_id");
        let e = parse(&v).unwrap_err().to_string();
        assert!(e.contains("image_token_id"), "{e}");
        // The reference's default of 2 applies when the key is absent.
        let mut v = idefics3_config();
        v.as_object_mut().unwrap().remove("scale_factor");
        assert_eq!(parse(&v).unwrap().spatial_merge_size, 2);
    }

    #[test]
    fn a_config_with_no_vision_section_is_the_none_config() {
        let mut v = idefics3_config();
        v.as_object_mut().unwrap().remove("vision_config");
        assert!(!parse(&v).unwrap().is_active());
    }

    /// MUTATION GUARD: the role table maps blob sub-tensors to weights by
    /// position, and the ten direct suffixes are laid out AROUND the two
    /// fused-qkv slots. Pin the exact table content so any edit here is a
    /// decision, and pin that the direct-suffix mapping keeps norms on the
    /// norm roles and the MLP on the MLP roles.
    #[test]
    fn the_role_table_and_suffix_mapping_are_pinned() {
        assert_eq!(IDEFICS3_BLOCK_ROLES.len(), 12);
        assert_eq!(IDEFICS3_BLOCK_SUFFIXES.len(), 10);
        assert_eq!(IDEFICS3_BLOCK_ROLES[0], "ln1_w");
        assert_eq!(IDEFICS3_BLOCK_ROLES[1], "ln1_b");
        assert_eq!(IDEFICS3_BLOCK_ROLES[2], "qkv_w");
        assert_eq!(IDEFICS3_BLOCK_ROLES[3], "qkv_b");
        assert_eq!(IDEFICS3_BLOCK_ROLES[4], "proj_w");
        assert_eq!(IDEFICS3_BLOCK_ROLES[5], "proj_b");
        assert_eq!(IDEFICS3_BLOCK_ROLES[8], "fc1_w");
        assert_eq!(IDEFICS3_BLOCK_ROLES[11], "fc2_b");
        // The direct suffixes skip the two fused slots: suffix slot 2
        // (out_proj.weight) maps onto ROLE 4, and suffix slot 9 (fc2.bias)
        // onto ROLE 11.
        assert_eq!(IDEFICS3_BLOCK_SUFFIXES[2], "self_attn.out_proj.weight");
        assert_eq!(IDEFICS3_BLOCK_SUFFIXES[9], "mlp.fc2.bias");
        // The qkv part sources are in q, k, v row order.
        assert_eq!(QKV_WEIGHT_SOURCES[0], "self_attn.q_proj.weight");
        assert_eq!(QKV_WEIGHT_SOURCES[1], "self_attn.k_proj.weight");
        assert_eq!(QKV_WEIGHT_SOURCES[2], "self_attn.v_proj.weight");
    }

    /// A tiny tower with every name the REAL published index uses. The
    /// dimensions shrink (hidden 8, intermediate 12, out 16, depth 2, patch
    /// 4 on a 8-pixel edge, scale 2) while the names, orders and shape
    /// RELATIONS stay the published ones, so a reader keyed on the wrong
    /// name or the wrong shape arithmetic refuses here rather than on a
    /// multi-GB stream.
    fn tiny_checkpoint() -> (VisionConfig, Vec<String>, Vec<u8>) {
        let (hid, inter, text, depth, patch, image, scale) =
            (8usize, 12usize, 16usize, 2usize, 4usize, 8usize, 2usize);
        let mut config = idefics3_config();
        config["text_config"]["hidden_size"] = text.into();
        config["scale_factor"] = scale.into();
        config["vision_config"]["hidden_size"] = hid.into();
        config["vision_config"]["intermediate_size"] = inter.into();
        config["vision_config"]["num_hidden_layers"] = depth.into();
        config["vision_config"]["num_attention_heads"] = 2.into();
        config["vision_config"]["image_size"] = image.into();
        config["vision_config"]["patch_size"] = patch.into();
        let vision = parse(&config).unwrap();
        let shaped = |name: String, shape: Vec<u64>, seed: u64| {
            let n: u64 = shape.iter().product();
            let mut t = crate::synthetic_tensors::bf16_vector(&name, n as usize, 0.5, seed);
            t.shape = shape;
            t
        };
        let (h, i) = (hid as u64, inter as u64);
        let pos_rows = vision.num_position_embeddings as u64; // (8/4)^2 = 4
        let mut ts: Vec<crate::synthetic_tensors::Tensor> = Vec::new();
        let block_shapes: Vec<(&str, Vec<u64>)> = vec![
            ("layer_norm1.weight", vec![h]),
            ("layer_norm1.bias", vec![h]),
            ("self_attn.q_proj.weight", vec![h, h]),
            ("self_attn.q_proj.bias", vec![h]),
            ("self_attn.k_proj.weight", vec![h, h]),
            ("self_attn.k_proj.bias", vec![h]),
            ("self_attn.v_proj.weight", vec![h, h]),
            ("self_attn.v_proj.bias", vec![h]),
            ("self_attn.out_proj.weight", vec![h, h]),
            ("self_attn.out_proj.bias", vec![h]),
            ("layer_norm2.weight", vec![h]),
            ("layer_norm2.bias", vec![h]),
            ("mlp.fc1.weight", vec![i, h]),
            ("mlp.fc1.bias", vec![i]),
            ("mlp.fc2.weight", vec![h, i]),
            ("mlp.fc2.bias", vec![h]),
        ];
        for b in 0..depth {
            let base = format!("{IDEFICS3_TOWER_PREFIX}encoder.layers.{b}.");
            for (idx, (suffix, shape)) in block_shapes.iter().enumerate() {
                // A DISTINCT seed per tensor: same-shape siblings (q/k/v)
                // must not be byte-identical, or a fusion-order mutation
                // would be invisible to the reader tests.
                ts.push(shaped(
                    format!("{base}{suffix}"),
                    shape.clone(),
                    10 * b as u64 + idx as u64,
                ));
            }
        }
        ts.push(shaped(
            format!("{IDEFICS3_TOWER_PREFIX}embeddings.patch_embedding.weight"),
            vec![h, patch as u64, patch as u64, 3],
            70,
        ));
        ts.push(shaped(
            format!("{IDEFICS3_TOWER_PREFIX}embeddings.patch_embedding.bias"),
            vec![h],
            71,
        ));
        ts.push(shaped(
            format!("{IDEFICS3_TOWER_PREFIX}embeddings.position_embedding.weight"),
            vec![1, pos_rows, h],
            72,
        ));
        ts.push(shaped(
            format!("{IDEFICS3_TOWER_PREFIX}post_layernorm.weight"),
            vec![h],
            73,
        ));
        ts.push(shaped(
            format!("{IDEFICS3_TOWER_PREFIX}post_layernorm.bias"),
            vec![h],
            74,
        ));
        let unfolded = (hid * scale * scale) as u64;
        ts.push(shaped(
            format!("{IDEFICS3_PROJECTOR_PREFIX}weight"),
            vec![text as u64, unfolded],
            80,
        ));
        let names = ts.iter().map(|t| t.name.clone()).collect();
        (
            vision,
            names,
            crate::synthetic_tensors::assemble_safetensors(&ts),
        )
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
    fn the_tower_reads_into_the_twelve_role_blocks() {
        let (vision, names, blob) = tiny_checkpoint();
        let r = read_from(&vision, &names, &blob).unwrap();
        assert_eq!(r.blocks.experts.len(), 2);
        let roles = |i: usize| -> Vec<&str> {
            r.blocks.experts[i]
                .sub_tensors
                .iter()
                .map(|t| t.role.as_str())
                .collect()
        };
        assert_eq!(roles(0), IDEFICS3_BLOCK_ROLES);
        let qkv_b = r.blocks.experts[0]
            .sub_tensors
            .iter()
            .find(|t| t.role == "qkv_b")
            .unwrap();
        assert_eq!(qkv_b.shape, vec![24]);
        // Every fused qkv carries 3 * hidden rows at 2 bytes.
        let qkv = r.blocks.experts[0]
            .sub_tensors
            .iter()
            .find(|t| t.role == "qkv_w")
            .unwrap();
        assert_eq!(qkv.shape, vec![24, 8]);
        assert_eq!(qkv.bytes.len(), 24 * 8 * 2);
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
                "vision.patch_embed.weight",
                "vision.patch_embed.bias",
                "vision.pos_embed.weight",
                "vision.post_norm.weight",
                "vision.post_norm.bias",
                "vision.projector.proj.weight",
            ]
        );
        assert_eq!(r.block_stride % GTURBO_PAGE_BYTES, 0);
    }

    /// SmolVLM-Instruct's geometry: 384 over 14 is not a whole number of
    /// patches, and the position table has the floored 27 x 27 rows.
    #[test]
    fn a_ragged_tile_edge_floors_the_patch_grid() {
        let mut v = idefics3_config();
        v["vision_config"]["image_size"] = 384.into();
        v["scale_factor"] = 3.into();
        v["image_seq_len"] = 81.into();
        assert_eq!(parse(&v).unwrap().num_position_embeddings, 729);
        v["image_seq_len"] = 169.into();
        let e = parse(&v).unwrap_err().to_string();
        assert!(e.contains("image_seq_len"), "{e}");
    }

    /// The published 4-bit builds pack q/k/v, out_proj, fc1, fc2 and the
    /// position table. Replacing one block's `q_proj.weight` and `fc1.weight`
    /// with packed planes must record the LOGICAL shapes and dequantize.
    #[test]
    fn packed_planes_record_their_logical_shapes() {
        let (vision, names, blob) = tiny_checkpoint();
        let header = crate::safetensors_header::parse_header(
            &blob,
            crate::safetensors_header::DEFAULT_MAX_HEADER_BYTES,
        )
        .unwrap();
        let source = crate::ranged_download::MemoryRangeSource::new(&blob);
        let shards = Gemma4Shards::single(&header, &source);
        let mut ts: Vec<crate::synthetic_tensors::Tensor> = names
            .iter()
            .map(|n| {
                let info = shards.info(n).unwrap();
                crate::synthetic_tensors::Tensor {
                    name: n.clone(),
                    dtype: "BF16",
                    shape: info.shape.clone(),
                    bytes: shards.read(n).unwrap(),
                }
            })
            .collect();
        // 8-bit at group 4: `cols` elements in `cols / 4` words, so a stored
        // shape never equals the logical one.
        let packed = |base: &str, rows: u64, cols: u64| {
            let words = cols / 4;
            let groups = cols / 4;
            let bf16 = |v: f32| compute::f32_to_bf16(v).to_le_bytes();
            vec![
                crate::synthetic_tensors::Tensor {
                    name: format!("{base}.weight"),
                    dtype: "U32",
                    shape: vec![rows, words],
                    bytes: vec![2u8; (rows * words * 4) as usize],
                },
                crate::synthetic_tensors::Tensor {
                    name: format!("{base}.scales"),
                    dtype: "BF16",
                    shape: vec![rows, groups],
                    bytes: (0..rows * groups).flat_map(|_| bf16(0.5)).collect(),
                },
                crate::synthetic_tensors::Tensor {
                    name: format!("{base}.biases"),
                    dtype: "BF16",
                    shape: vec![rows, groups],
                    bytes: (0..rows * groups).flat_map(|_| bf16(0.25)).collect(),
                },
            ]
        };
        for (suffix, rows) in [("self_attn.q_proj", 8u64), ("mlp.fc1", 12)] {
            let base = format!("{IDEFICS3_TOWER_PREFIX}encoder.layers.0.{suffix}");
            ts.retain(|t| t.name != format!("{base}.weight"));
            ts.extend(packed(&base, rows, 8));
        }
        let blob = crate::synthetic_tensors::assemble_safetensors(&ts);
        let header = crate::safetensors_header::parse_header(
            &blob,
            crate::safetensors_header::DEFAULT_MAX_HEADER_BYTES,
        )
        .unwrap();
        let source = crate::ranged_download::MemoryRangeSource::new(&blob);
        let shards = Gemma4Shards::single(&header, &source);
        let bases: Vec<&str> = names.iter().map(String::as_str).collect();
        let quant = Gemma4Quant {
            default_bits: 8,
            group_size: 4,
            ..Default::default()
        };
        let r = read_idefics3_vision_entries(&shards, &bases, &vision, Some(&quant)).unwrap();
        let sub = |role: &str| {
            r.blocks.experts[0]
                .sub_tensors
                .iter()
                .find(|t| t.role == role)
                .unwrap()
        };
        assert_eq!(sub("fc1_w").shape, vec![12, 8]);
        assert_eq!(sub("qkv_w").shape, vec![24, 8]);
        // q = 2, scale 0.5, bias 0.25: every dequantized element is 1.25.
        let first = u16::from_le_bytes([sub("fc1_w").bytes[0], sub("fc1_w").bytes[1]]);
        assert_eq!(compute::f16_to_f32(first), 1.25);
        let first = u16::from_le_bytes([sub("qkv_w").bytes[0], sub("qkv_w").bytes[1]]);
        assert_eq!(compute::f16_to_f32(first), 1.25);
    }

    /// THE FUSION: the separate q, k, v weights land in one blob in q, k, v
    /// row order, byte-for-byte the concatenation of the source tensors. A
    /// wrong order (k before q, or interleaved heads) would rotate the wrong
    /// rows into the attention kernel fluently.
    #[test]
    fn the_separate_projections_fuse_in_qkv_order() {
        let (_vision, _names, blob) = tiny_checkpoint();
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
                format!("{IDEFICS3_TOWER_PREFIX}encoder.layers.0.self_attn.{part}_proj.weight");
            let t = shards.info(&name).unwrap();
            let converted = crate::gemma4_checkpoint::narrow::convert_raw_to_fp16(
                &name,
                &t.dtype,
                shards.read(&name).unwrap(),
            )
            .unwrap();
            parts.push(converted.bytes);
        }
        let (vision, names, blob) = tiny_checkpoint();
        let r = read_from(&vision, &names, &blob).unwrap();
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
        // Drop block 0's v_proj bias: the missing-slot report names it.
        let (vision, names, blob) = tiny_checkpoint();
        let filtered: Vec<String> = names
            .iter()
            .filter(|n| !n.ends_with("layers.0.self_attn.v_proj.bias"))
            .cloned()
            .collect();
        let e = read_from(&vision, &filtered, &blob)
            .err()
            .expect("missing v_proj.bias refused");
        assert!(e.to_string().contains("v_proj.bias"), "{e}");
        // An unknown tower name is refused, not dropped.
        let (vision, mut names, blob) = tiny_checkpoint();
        names.push(format!("{IDEFICS3_TOWER_PREFIX}embeddings.something_new"));
        let e = read_from(&vision, &names, &blob)
            .err()
            .expect("unknown tensor refused");
        assert!(e.to_string().contains("something_new"), "{e}");
        // A connector tensor this port does not run (the HF original's
        // LayerNorm) is refused by name, not dropped.
        let (vision, mut names, blob) = tiny_checkpoint();
        names.push("connector.modality_projection.layer_norm.weight".to_string());
        let e = read_from(&vision, &names, &blob)
            .err()
            .expect("connector LayerNorm refused");
        assert!(e.to_string().contains("layer_norm"), "{e}");
    }
}
