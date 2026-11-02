//! The SigLIP tower DeepSeek-VL2 carries: config and ingest
//! (`docs/DEEPSEEK_VL2_VISION_PHASE0.md`).
//!
//! A sibling of [`super::vision_clip`], and it lands in the same place: blocks
//! pack into `packed_vision/` as the SAME twelve-role blob (`ln1_w ln1_b qkv_w
//! qkv_b proj_w proj_b ln2_w ln2_b fc1_w fc1_b fc2_w fc2_b`), so the runtime
//! reuses that role table and its streamer. The source differs in three ways:
//!
//! - q, k and v are ALREADY one fused `attn.qkv` (rows q, then k, then v), so
//!   nothing is concatenated;
//! - the MLX conversion quantizes most of the tower to 4-bit affine planes
//!   with F16 scales and biases. Those planes are DEQUANTIZED to FP16 here
//!   (the runtime tower is FP16 end to end), and `mlp.fc2`, whose 4304-wide
//!   input is not a multiple of any group size, ships F16 and passes through;
//! - the checkpoint carries the projector and two learned embeddings
//!   (`image_newline`, `view_separator`) beside the tower, all of which the
//!   runtime needs resident.
//!
//! `attn_pool` (the SigLIP attention-pool head) is recognized and DROPPED:
//! the model reads the tower's output before it (`ignore_head`), so nothing
//! ever uses those 5 MiB.

use model_io::{VisionConfig, VisionTowerKind};

use super::classify::VISION_INSTALL_PREFIX;
use super::config::{Gemma4Error, Gemma4Quant};
use super::shards::{shape4, Gemma4Shards, GTURBO_PAGE_BYTES};
use super::vision::{read_tower_tensor, VisionRead};
use crate::gturbo_writer::{ExpertBlob, LayerBlobs, SubTensor};
use crate::resident_writer::{RawTensorSpec, ResidentEntrySpec};

/// The HF `model_type` of a DeepSeek-VL2 checkpoint.
pub const DEEPSEEK_VL2_MODEL_TYPE: &str = "deepseek_vl_v2";

/// The tower's prefix in a DeepSeek-VL2 checkpoint.
pub const SIGLIP_MODEL_PREFIX: &str = "vision.vision_tower.";

/// The projector's prefix, beside the tower's.
pub const SIGLIP_PROJECTOR_PREFIX: &str = "projector.";

/// Every source prefix a tower fetch needs: the tower, the projector and the
/// two top-level learned embeddings.
pub const SIGLIP_SOURCE_PREFIXES: [&str; 4] = [
    SIGLIP_MODEL_PREFIX,
    SIGLIP_PROJECTOR_PREFIX,
    "image_newline",
    "view_separator",
];

/// `<image>` in the published tokenizer. The processor resolves it by name;
/// this is checked against the loaded tokenizer by every front end.
pub const DEEPSEEK_VL2_IMAGE_TOKEN_ID: i64 = 128_815;

/// The blob's roles, in the qwen tower's order (the runtime's role table).
const BLOCK_ROLE_NAMES: [&str; 12] = [
    "ln1_w", "ln1_b", "qkv_w", "qkv_b", "proj_w", "proj_b", "ln2_w", "ln2_b", "fc1_w", "fc1_b",
    "fc2_w", "fc2_b",
];

/// The source tensor for each role, under `<prefix>blocks.N.`.
const BLOCK_SOURCE_SUFFIXES: [&str; 12] = [
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
];

/// The install-side resident names, under [`VISION_INSTALL_PREFIX`], in the
/// order they are written. The runtime resolves exactly these.
pub const SIGLIP_RESIDENT_TENSORS: [&str; 11] = [
    "patch_embed.proj.weight",
    "patch_embed.proj.bias",
    "pos_embed.weight",
    "norm.weight",
    "norm.bias",
    "projector.layers.0.weight",
    "projector.layers.0.bias",
    "projector.layers.2.weight",
    "projector.layers.2.bias",
    "image_newline",
    "view_separator",
];

/// The source name of each resident, same order.
const RESIDENT_SOURCES: [&str; 11] = [
    "vision.vision_tower.patch_embed.proj.weight",
    "vision.vision_tower.patch_embed.proj.bias",
    "vision.vision_tower.pos_embed",
    "vision.vision_tower.norm.weight",
    "vision.vision_tower.norm.bias",
    "projector.layers.0.weight",
    "projector.layers.0.bias",
    "projector.layers.2.weight",
    "projector.layers.2.bias",
    "image_newline",
    "view_separator",
];

/// The reference defaults for a `vision_config` that omits them (the
/// published file states only `layers`, `width`, `patch_size`, `mlp_ratio`):
/// mlx-vlm's `VisionConfig` gives 16 heads and a 4304-wide MLP.
const DEFAULT_HEADS: i64 = 16;
const DEFAULT_INTERMEDIATE: i64 = 4304;
const DEFAULT_IMAGE_SIZE: i64 = 384;
const SUPPORTED_PATCH: i64 = 14;

/// Parse a DeepSeek-VL2 `config.json` into the tower's [`VisionConfig`].
///
/// Refused rather than ignored: a `tile_tag` other than `2D`, a
/// `global_view_pos` other than `head`, a `select_layer` other than the last,
/// a projector other than the 2x2 downsample MLP, a patch other than 14.
/// `out_hidden_size` is the projector's `n_embed`, which must be the trunk
/// width.
pub fn parse_deepseek_vl2_vision_config(json: &str) -> Result<VisionConfig, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    let bad = |m: String| Gemma4Error::Config(m);
    if root.get("model_type").and_then(serde_json::Value::as_str) != Some(DEEPSEEK_VL2_MODEL_TYPE) {
        return Err(bad(format!(
            "model_type is not {DEEPSEEK_VL2_MODEL_TYPE:?}; this reader is for DeepSeek-VL2"
        )));
    }
    let vc = match root.get("vision_config") {
        None | Some(serde_json::Value::Null) => return Ok(VisionConfig::NONE),
        Some(v) => v,
    };
    let get = |obj: &serde_json::Value, k: &str| obj.get(k).and_then(serde_json::Value::as_i64);
    if root.get("tile_tag").and_then(serde_json::Value::as_str) != Some("2D") {
        return Err(bad("tile_tag must be \"2D\" (the newline layout)".into()));
    }
    if root
        .get("global_view_pos")
        .and_then(serde_json::Value::as_str)
        != Some("head")
    {
        return Err(bad("global_view_pos must be \"head\"".into()));
    }
    if let Some(layer) = get(&root, "select_layer") {
        if layer != -1 {
            return Err(bad(format!(
                "select_layer {layer}; the implemented tower reads the last block"
            )));
        }
    }
    let projector = root
        .get("projector_config")
        .ok_or_else(|| bad("missing projector_config".into()))?;
    if let Some(kind) = projector
        .get("projector_type")
        .and_then(serde_json::Value::as_str)
    {
        if kind != "downsample_mlp_gelu" {
            return Err(bad(format!(
                "projector_type {kind:?}; need downsample_mlp_gelu"
            )));
        }
    }
    for (key, want) in [("depth", 2), ("mlp_ratio", 1), ("downsample_ratio", 2)] {
        if let Some(v) = get(projector, key) {
            if v != want {
                return Err(bad(format!(
                    "projector {key} {v}; the implemented one is {want}"
                )));
            }
        }
    }
    if projector
        .get("token_pooling")
        .and_then(serde_json::Value::as_bool)
        == Some(true)
    {
        return Err(bad("projector token_pooling is not implemented".into()));
    }
    let n_embed =
        get(projector, "n_embed").ok_or_else(|| bad("missing projector n_embed".into()))?;

    let layers = get(vc, "layers").ok_or_else(|| bad("missing vision_config.layers".into()))?;
    let width = get(vc, "width").ok_or_else(|| bad("missing vision_config.width".into()))?;
    let patch = get(vc, "patch_size").unwrap_or(SUPPORTED_PATCH);
    if patch != SUPPORTED_PATCH {
        return Err(bad(format!(
            "vision patch_size {patch}; the implemented preprocessing is patch {SUPPORTED_PATCH}"
        )));
    }
    let heads = get(vc, "num_attention_heads").unwrap_or(DEFAULT_HEADS);
    if heads <= 0 || width % heads != 0 {
        return Err(bad(format!(
            "vision width {width} is not divisible by {heads} heads"
        )));
    }
    let image = get(vc, "image_size").unwrap_or(DEFAULT_IMAGE_SIZE);
    let side = image / patch;
    if layers < 1 {
        return Err(bad(format!("vision layers {layers}")));
    }
    Ok(VisionConfig {
        depth: layers,
        hidden_size: width,
        intermediate_size: get(vc, "intermediate_size").unwrap_or(DEFAULT_INTERMEDIATE),
        num_heads: heads,
        patch_size: patch,
        temporal_patch_size: 1,
        in_channels: 3,
        spatial_merge_size: 2,
        num_position_embeddings: side * side,
        out_hidden_size: n_embed,
        mrope_section: [0, 0, 0],
        deepstack_visual_indexes: Vec::new(),
        vision_start_token_id: 0,
        vision_end_token_id: 0,
        image_token_id: DEEPSEEK_VL2_IMAGE_TOKEN_ID,
        video_token_id: 0,
        tower: VisionTowerKind::SigLip,
    })
}

/// Reads the SigLIP tower, projector and the two learned embeddings,
/// converting every tensor to FP16 (dequantizing the packed planes).
///
/// Every name under the tower prefix must be a known block tensor, a known
/// resident, or `attn_pool` (dropped). An unknown name is refused, as the
/// other readers refuse one.
pub fn read_siglip_vision_entries(
    shards: &Gemma4Shards<'_>,
    vision_bases: &[&str],
    vision: &VisionConfig,
    quant: Option<&Gemma4Quant>,
) -> Result<VisionRead, Gemma4Error> {
    if vision.tower != VisionTowerKind::SigLip || vision.depth < 1 {
        return Err(Gemma4Error::Config(format!(
            "not a SigLIP tower (kind {:?}, depth {})",
            vision.tower, vision.depth
        )));
    }
    let depth = vision.depth as usize;
    let hidden = vision.hidden_size as usize;
    let inter = vision.intermediate_size as usize;
    let out_hidden = vision.out_hidden_size as usize;
    let patch = vision.patch_size as usize;
    let pos_rows = vision.num_position_embeddings as usize;
    let merge = vision.spatial_merge_size as usize;

    // Classify every base name: a resident, a block tensor, or a drop.
    let mut block_seen = vec![[false; 12]; depth];
    let mut resident_seen = [false; RESIDENT_SOURCES.len()];
    for &name in vision_bases {
        if name.starts_with("vision.vision_tower.attn_pool.") {
            continue;
        }
        if let Some(slot) = RESIDENT_SOURCES.iter().position(|r| *r == name) {
            if std::mem::replace(&mut resident_seen[slot], true) {
                return Err(Gemma4Error::UnknownTensor(format!("{name}: duplicate")));
            }
            continue;
        }
        let rest = name
            .strip_prefix(SIGLIP_MODEL_PREFIX)
            .and_then(|t| t.strip_prefix("blocks."))
            .ok_or_else(|| {
                Gemma4Error::UnknownTensor(format!("{name}: not a SigLIP tower tensor"))
            })?;
        let (index, suffix) = rest.split_once('.').ok_or_else(|| {
            Gemma4Error::UnknownTensor(format!("{name}: no block index and suffix"))
        })?;
        let index: usize = index.parse().map_err(|_| {
            Gemma4Error::UnknownTensor(format!("{name}: block index is not a number"))
        })?;
        let role = BLOCK_SOURCE_SUFFIXES
            .iter()
            .position(|s| *s == suffix)
            .ok_or_else(|| {
                Gemma4Error::UnknownTensor(format!(
                    "{name}: {suffix} is not one of the twelve per-block tensors"
                ))
            })?;
        if index >= depth {
            return Err(Gemma4Error::UnknownTensor(format!(
                "{name}: block {index} is past the declared depth {depth}"
            )));
        }
        if std::mem::replace(&mut block_seen[index][role], true) {
            return Err(Gemma4Error::UnknownTensor(format!("{name}: duplicate")));
        }
    }
    // Every slot filled, or refused by name: a missing tensor would stream as
    // zeros and the tower would run with a sublayer silently off.
    for (index, roles) in block_seen.iter().enumerate() {
        for (role, seen) in roles.iter().enumerate() {
            if !seen {
                return Err(Gemma4Error::MissingTensor(format!(
                    "{SIGLIP_MODEL_PREFIX}blocks.{index}.{}",
                    BLOCK_SOURCE_SUFFIXES[role]
                )));
            }
        }
    }
    for (slot, seen) in resident_seen.iter().enumerate() {
        if !seen {
            return Err(Gemma4Error::MissingTensor(
                RESIDENT_SOURCES[slot].to_string(),
            ));
        }
    }

    let mut lossy_conversion = Vec::new();
    let mut load = |name: &str, want: &[usize]| -> Result<Vec<u8>, Gemma4Error> {
        let (converted, shape) = read_tower_tensor(shards, name, quant)?;
        let got: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        // Checkpoints state a position table as [1, rows, width]; only the
        // element count and trailing dims matter to a row-major reader.
        let squeezed: &[usize] = match got.as_slice() {
            [1, rest @ ..] if rest.len() == want.len() => rest,
            all => all,
        };
        if squeezed != want {
            return Err(Gemma4Error::ShapeMismatch {
                tensor: name.to_string(),
                detail: format!("shape {got:?}, expected {want:?}"),
            });
        }
        if converted.lossy > 0 {
            lossy_conversion.push((name.to_string(), converted.lossy));
        }
        Ok(converted.bytes)
    };

    let patch_dim = 3 * patch * patch;
    let unfolded = out_hidden_input(hidden, merge);
    let mut entries = Vec::with_capacity(RESIDENT_SOURCES.len());
    let mut push = |slot: usize, bytes: Vec<u8>, shape: (u32, u32, u32, u32)| {
        entries.push(ResidentEntrySpec::Raw(RawTensorSpec {
            name: format!("{VISION_INSTALL_PREFIX}{}", SIGLIP_RESIDENT_TENSORS[slot]),
            dtype: crate::DTYPE_FP16,
            bytes,
            shape,
        }));
    };
    let h32 = hidden as u32;
    // The conv weight is stored [out, kh, kw, C]: flattened row-major that is
    // already the (P_h, P_w, C) order vision-io emits a patch row in.
    push(
        0,
        load(RESIDENT_SOURCES[0], &[hidden, patch, patch, 3])?,
        (h32, patch_dim as u32, 0, 0),
    );
    push(
        1,
        load(RESIDENT_SOURCES[1], &[hidden])?,
        shape4(&[hidden as u64]),
    );
    push(
        2,
        load(RESIDENT_SOURCES[2], &[pos_rows, hidden])?,
        (pos_rows as u32, h32, 0, 0),
    );
    push(
        3,
        load(RESIDENT_SOURCES[3], &[hidden])?,
        shape4(&[hidden as u64]),
    );
    push(
        4,
        load(RESIDENT_SOURCES[4], &[hidden])?,
        shape4(&[hidden as u64]),
    );
    push(
        5,
        load(RESIDENT_SOURCES[5], &[out_hidden, unfolded])?,
        (out_hidden as u32, unfolded as u32, 0, 0),
    );
    push(
        6,
        load(RESIDENT_SOURCES[6], &[out_hidden])?,
        shape4(&[out_hidden as u64]),
    );
    push(
        7,
        load(RESIDENT_SOURCES[7], &[out_hidden, out_hidden])?,
        (out_hidden as u32, out_hidden as u32, 0, 0),
    );
    push(
        8,
        load(RESIDENT_SOURCES[8], &[out_hidden])?,
        shape4(&[out_hidden as u64]),
    );
    push(
        9,
        load(RESIDENT_SOURCES[9], &[out_hidden])?,
        shape4(&[out_hidden as u64]),
    );
    push(
        10,
        load(RESIDENT_SOURCES[10], &[out_hidden])?,
        shape4(&[out_hidden as u64]),
    );

    let mut experts = Vec::with_capacity(depth);
    let mut widest = 0u64;
    for index in 0..depth {
        let base = format!("{SIGLIP_MODEL_PREFIX}blocks.{index}.");
        let mut sub_tensors = Vec::with_capacity(BLOCK_ROLE_NAMES.len());
        let mut used = 0u64;
        let h = hidden;
        let shapes: [Vec<usize>; 12] = [
            vec![h],
            vec![h],
            vec![3 * h, h],
            vec![3 * h],
            vec![h, h],
            vec![h],
            vec![h],
            vec![h],
            vec![inter, h],
            vec![inter],
            vec![h, inter],
            vec![h],
        ];
        for (role, shape) in shapes.iter().enumerate() {
            let name = format!("{base}{}", BLOCK_SOURCE_SUFFIXES[role]);
            let bytes = load(&name, shape)?;
            used += bytes.len() as u64;
            sub_tensors.push(SubTensor {
                role: BLOCK_ROLE_NAMES[role].to_string(),
                bytes,
                dtype: "fp16".to_string(),
                shape: shape.iter().map(|&d| d as u64).collect(),
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

/// The first projector layer's input width: `merge^2` tower rows concatenated.
fn out_hidden_input(hidden: usize, merge: usize) -> usize {
    hidden * merge * merge
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published `mlx-community/deepseek-vl2-tiny-4bit` config, trimmed to
    /// the keys this parser reads.
    fn real_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "deepseek_vl_v2",
            "global_view_pos": "head",
            "tile_tag": "2D",
            "projector_config": {"model_type": "mlp_projector", "n_embed": 1280},
            "vision_config": {
                "layers": 27, "mlp_ratio": 3.7362, "model_name": "siglip_so400m_patch14_384",
                "model_type": "vision", "patch_size": 14, "width": 1152
            }
        })
    }

    fn parse(v: &serde_json::Value) -> Result<VisionConfig, Gemma4Error> {
        parse_deepseek_vl2_vision_config(&v.to_string())
    }

    #[test]
    fn the_published_config_parses_to_the_so400m_tower() {
        let v = parse(&real_config()).unwrap();
        assert_eq!(v.tower, VisionTowerKind::SigLip);
        assert_eq!(
            (v.depth, v.hidden_size, v.intermediate_size),
            (27, 1152, 4304)
        );
        assert_eq!(
            (v.num_heads, v.patch_size, v.num_position_embeddings),
            (16, 14, 729)
        );
        assert_eq!((v.spatial_merge_size, v.out_hidden_size), (2, 1280));
        assert_eq!(v.image_token_id, DEEPSEEK_VL2_IMAGE_TOKEN_ID);
        assert!(v.is_active());
    }

    #[test]
    fn a_tower_this_reader_does_not_implement_is_refused() {
        for (path, value) in [
            ("tile_tag", serde_json::json!("1D")),
            ("global_view_pos", serde_json::json!("tail")),
            ("select_layer", serde_json::json!(-2)),
        ] {
            let mut v = real_config();
            v[path] = value;
            assert!(parse(&v).is_err(), "{path}");
        }
        let mut v = real_config();
        v["vision_config"]["patch_size"] = 16.into();
        assert!(parse(&v).is_err());
        let mut v = real_config();
        v["projector_config"]["projector_type"] = "linear".into();
        assert!(parse(&v).is_err());
        let mut v = real_config();
        v["projector_config"]["downsample_ratio"] = 3.into();
        assert!(parse(&v).is_err());
        let mut v = real_config();
        v["model_type"] = "llava".into();
        assert!(parse(&v).is_err());
    }

    #[test]
    fn a_config_with_no_vision_section_is_the_none_config() {
        let mut v = real_config();
        v.as_object_mut().unwrap().remove("vision_config");
        assert!(!parse(&v).unwrap().is_active());
    }

    #[test]
    fn the_resident_list_and_its_sources_agree() {
        assert_eq!(RESIDENT_SOURCES.len(), SIGLIP_RESIDENT_TENSORS.len());
    }
}
