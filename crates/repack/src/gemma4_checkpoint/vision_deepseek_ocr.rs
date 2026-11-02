//! The DeepSeek-OCR encoder: config and ingest
//! (`docs/DEEPSEEK_OCR_VISION_PHASE0.md`).
//!
//! Two towers in one: a SAM-ViT-B whose output is the patch embedding of a
//! CLIP-L, concatenated and projected by one linear.
//!
//! How the bytes land, and why they split the way they do:
//!
//! - The CLIP-L's 24 blocks pack into `packed_vision/` as the SAME twelve-role
//!   blob the CLIP and SigLIP towers use (`ln1_w ln1_b qkv_w qkv_b proj_w
//!   proj_b ln2_w ln2_b fc1_w fc1_b fc2_w fc2_b`; its q/k/v are already one
//!   fused `qkv_proj`), so the runtime reuses the role table and the streamer.
//! - The SAM's twelve blocks are RESIDENT FP16, not streamed. They are 14
//!   tensors each (the blob has no slot for `rel_pos_h`/`rel_pos_w`), 768 wide
//!   where the streamed blocks are 1024 wide, and the whole half is 190 MiB,
//!   mapped lazily from the sidecar file: a model-wide stride cannot describe
//!   two widths, and a second packed layer would need the mixed-stride rules
//!   of `model-io` Gotcha 2 for a saving that does not matter at this size.
//! - Only the projector is quantized in the MLX conversion (4-bit affine,
//!   group 64). It is dequantized to FP16 here. Every other tensor is BF16
//!   and converts exactly.
//! - The CLIP's own `embeddings.patch_embedding` is DROPPED: this model feeds
//!   the SAM grid in as the patch embedding, so those 1.2 MiB are never read.

use model_io::{VisionConfig, VisionTowerKind};

use super::classify::VISION_INSTALL_PREFIX;
use super::config::{Gemma4Error, Gemma4Quant};
use super::shards::{shape4, Gemma4Shards, GTURBO_PAGE_BYTES};
use super::vision::{read_tower_tensor, VisionRead};
use crate::gturbo_writer::{ExpertBlob, LayerBlobs, SubTensor};
use crate::resident_writer::{RawTensorSpec, ResidentEntrySpec};

/// The HF `model_type` of a DeepSeek-OCR checkpoint, and the second spelling
/// `baidu/Unlimited-OCR-2` declares for the same trunk and tower shape.
pub const DEEPSEEK_OCR_MODEL_TYPE: &str = "deepseekocr";
pub const UNLIMITED_OCR_MODEL_TYPE: &str = "unlimited-ocr";

/// The SAM tower's prefix in the checkpoint.
pub const OCR_SAM_PREFIX: &str = "sam_model.";

/// The CLIP-L's prefix in the checkpoint.
pub const OCR_CLIP_PREFIX: &str = "vision_model.";

/// Every source prefix a tower fetch needs.
pub const OCR_SOURCE_PREFIXES: [&str; 5] = [
    OCR_SAM_PREFIX,
    OCR_CLIP_PREFIX,
    "projector.",
    "image_newline",
    "view_separator",
];

/// `<image>` in the published tokenizer (the same id DeepSeek-VL2 uses).
pub const DEEPSEEK_OCR_IMAGE_TOKEN_ID: i64 = 128_815;

/// SAM geometry: properties of the kind, not stored in the manifest.
pub const SAM_DEPTH: usize = 12;
pub const SAM_WIDTH: usize = 768;
pub const SAM_HEADS: usize = 12;
pub const SAM_MLP: usize = 3072;
pub const SAM_WINDOW: usize = 14;
pub const SAM_GLOBAL_BLOCKS: [usize; 4] = [2, 5, 8, 11];
/// The input view: 1024 px, 16 px patches, so a 64 x 64 grid.
pub const SAM_IMAGE: usize = 1024;
pub const SAM_PATCH: usize = 16;
pub const SAM_GRID: usize = SAM_IMAGE / SAM_PATCH;
pub const SAM_NECK: usize = 256;
pub const SAM_NET2: usize = 512;
pub const SAM_NET3: usize = 1024;

/// CLIP-L geometry, checked against the declared config.
const CLIP_DEPTH: usize = 24;
const CLIP_WIDTH: usize = 1024;
const CLIP_HEADS: usize = 16;
const CLIP_MLP: usize = 4096;
/// The SAM grid after the neck and two stride-2 convs: 16 x 16, plus the class row.
const CLIP_POSITIONS: usize = 257;

/// The blob's roles, in the qwen tower's order (the runtime's role table).
const BLOCK_ROLE_NAMES: [&str; 12] = [
    "ln1_w", "ln1_b", "qkv_w", "qkv_b", "proj_w", "proj_b", "ln2_w", "ln2_b", "fc1_w", "fc1_b",
    "fc2_w", "fc2_b",
];

/// The source tensor for each CLIP role, under
/// `vision_model.transformer.layers.N.`.
const CLIP_BLOCK_SOURCE_SUFFIXES: [&str; 12] = [
    "layer_norm1.weight",
    "layer_norm1.bias",
    "self_attn.qkv_proj.weight",
    "self_attn.qkv_proj.bias",
    "self_attn.out_proj.weight",
    "self_attn.out_proj.bias",
    "layer_norm2.weight",
    "layer_norm2.bias",
    "mlp.fc1.weight",
    "mlp.fc1.bias",
    "mlp.fc2.weight",
    "mlp.fc2.bias",
];

/// One SAM block's tensors, in the order they are written. The names are the
/// checkpoint's own tails under `sam_model.blocks.N.`; the install keeps them
/// verbatim under `vision.sam.blocks.N.`.
pub const SAM_BLOCK_TENSORS: [&str; 14] = [
    "norm1.weight",
    "norm1.bias",
    "attn.qkv.weight",
    "attn.qkv.bias",
    "attn.proj.weight",
    "attn.proj.bias",
    "attn.rel_pos_h",
    "attn.rel_pos_w",
    "norm2.weight",
    "norm2.bias",
    "mlp.lin1.weight",
    "mlp.lin1.bias",
    "mlp.lin2.weight",
    "mlp.lin2.bias",
];

/// Dropped on purpose (see the module doc).
const DROPPED: [&str; 1] = ["vision_model.embeddings.patch_embedding.weight"];

/// The install's resident name for a SAM tensor.
pub fn sam_resident_name(tail: &str) -> String {
    format!("{VISION_INSTALL_PREFIX}sam.{tail}")
}

/// The install's resident name for a SAM block tensor.
pub fn sam_block_resident_name(block: usize, tail: &str) -> String {
    format!("{VISION_INSTALL_PREFIX}sam.blocks.{block}.{tail}")
}

/// The install's resident name for a CLIP non-block tensor.
pub fn clip_resident_name(tail: &str) -> String {
    format!("{VISION_INSTALL_PREFIX}clip.{tail}")
}

/// The install's resident name for the projector and layout rows.
pub fn projector_resident_name(tail: &str) -> String {
    format!("{VISION_INSTALL_PREFIX}{tail}")
}

/// Parse a DeepSeek-OCR `config.json` into the tower's [`VisionConfig`].
///
/// Refused rather than ignored: a `tile_tag` other than `2D`, a
/// `global_view_pos` other than `head`, a projector other than the bare
/// linear, an `image_size` other than 1024, and any SAM or CLIP geometry that
/// is not the one the tower implements. The model's width and the projector's
/// input (CLIP + SAM channels, 2048) are cross-checked.
pub fn parse_deepseek_ocr_vision_config(json: &str) -> Result<VisionConfig, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    let bad = |m: String| Gemma4Error::Config(m);
    match root.get("model_type").and_then(serde_json::Value::as_str) {
        Some(DEEPSEEK_OCR_MODEL_TYPE) | Some(UNLIMITED_OCR_MODEL_TYPE) => {}
        other => {
            return Err(bad(format!(
                "model_type {other:?} is not a DeepSeek-OCR one ({DEEPSEEK_OCR_MODEL_TYPE:?} or \
                 {UNLIMITED_OCR_MODEL_TYPE:?})"
            )))
        }
    }
    let vc = match root.get("vision_config") {
        None | Some(serde_json::Value::Null) => return Ok(VisionConfig::NONE),
        Some(v) => v,
    };
    let int = |obj: &serde_json::Value, k: &str| obj.get(k).and_then(serde_json::Value::as_i64);
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
    let projector = root
        .get("projector_config")
        .ok_or_else(|| bad("missing projector_config".into()))?;
    if projector
        .get("projector_type")
        .and_then(serde_json::Value::as_str)
        != Some("linear")
    {
        return Err(bad("projector_type must be \"linear\"".into()));
    }
    let n_embed =
        int(projector, "n_embed").ok_or_else(|| bad("missing projector n_embed".into()))?;
    let input_dim = int(projector, "input_dim").unwrap_or((CLIP_WIDTH + SAM_NET3) as i64);
    if input_dim != (CLIP_WIDTH + SAM_NET3) as i64 {
        return Err(bad(format!(
            "projector input_dim {input_dim}; the tower produces {} (CLIP + SAM channels)",
            CLIP_WIDTH + SAM_NET3
        )));
    }
    if let Some(size) = int(vc, "image_size") {
        if size != SAM_IMAGE as i64 {
            return Err(bad(format!(
                "vision image_size {size}; only the 1024 px base view is implemented"
            )));
        }
    }
    let widths = vc
        .get("width")
        .ok_or_else(|| bad("missing vision_config.width".into()))?;
    let sam = widths
        .get("sam_vit_b")
        .ok_or_else(|| bad("missing vision_config.width.sam_vit_b".into()))?;
    let clip = widths
        .get("clip-l-14-224")
        .ok_or_else(|| bad("missing vision_config.width.clip-l-14-224".into()))?;
    for (obj, key, want, what) in [
        (sam, "layers", SAM_DEPTH, "SAM layers"),
        (sam, "width", SAM_WIDTH, "SAM width"),
        (sam, "heads", SAM_HEADS, "SAM heads"),
        (clip, "layers", CLIP_DEPTH, "CLIP layers"),
        (clip, "width", CLIP_WIDTH, "CLIP width"),
        (clip, "heads", CLIP_HEADS, "CLIP heads"),
    ] {
        if int(obj, key) != Some(want as i64) {
            return Err(bad(format!(
                "{what} is {:?}; only {want} is implemented",
                int(obj, key)
            )));
        }
    }
    let globals: Vec<i64> = sam
        .get("global_attn_indexes")
        .and_then(serde_json::Value::as_array)
        .map(|a| a.iter().filter_map(serde_json::Value::as_i64).collect())
        .unwrap_or_default();
    if globals != SAM_GLOBAL_BLOCKS.map(|b| b as i64) {
        return Err(bad(format!(
            "SAM global_attn_indexes {globals:?}; only {SAM_GLOBAL_BLOCKS:?} is implemented"
        )));
    }
    Ok(VisionConfig {
        depth: CLIP_DEPTH as i64,
        hidden_size: CLIP_WIDTH as i64,
        intermediate_size: CLIP_MLP as i64,
        num_heads: CLIP_HEADS as i64,
        patch_size: SAM_PATCH as i64,
        temporal_patch_size: 1,
        in_channels: 3,
        spatial_merge_size: 4,
        num_position_embeddings: CLIP_POSITIONS as i64,
        out_hidden_size: n_embed,
        mrope_section: [0, 0, 0],
        deepstack_visual_indexes: Vec::new(),
        vision_start_token_id: 0,
        vision_end_token_id: 0,
        image_token_id: DEEPSEEK_OCR_IMAGE_TOKEN_ID,
        video_token_id: 0,
        tower: VisionTowerKind::DeepseekOcr,
    })
}

/// Reads the SAM and CLIP-L towers, the projector and the two learned rows,
/// converting every tensor to FP16 (dequantizing the projector's packed
/// planes).
///
/// Every name under a tower prefix must be a known tensor or the dropped
/// patch embedding; an unknown name is refused, as the other readers refuse
/// one.
pub fn read_deepseek_ocr_vision_entries(
    shards: &Gemma4Shards<'_>,
    vision_bases: &[&str],
    vision: &VisionConfig,
    quant: Option<&Gemma4Quant>,
) -> Result<VisionRead, Gemma4Error> {
    if vision.tower != VisionTowerKind::DeepseekOcr || vision.depth != CLIP_DEPTH as i64 {
        return Err(Gemma4Error::Config(format!(
            "not a DeepSeek-OCR tower (kind {:?}, depth {})",
            vision.tower, vision.depth
        )));
    }
    let out_hidden = vision.out_hidden_size as usize;
    let proj_in = CLIP_WIDTH + SAM_NET3;

    // Every name the install knows, with where it comes from and what shape it
    // must have. Built once, so the classification pass and the read pass
    // cannot disagree about the set.
    struct Item {
        source: String,
        install: String,
        want: Vec<usize>,
        /// The shape the resident entry records (flattened past rank 4).
        recorded: (u32, u32),
    }
    let mut items: Vec<Item> = Vec::new();
    let mut push = |source: String, install: String, want: Vec<usize>, recorded: (usize, usize)| {
        items.push(Item {
            source,
            install,
            want,
            recorded: (recorded.0 as u32, recorded.1 as u32),
        });
    };
    let sam = |tail: &str| format!("{OCR_SAM_PREFIX}{tail}");
    // Convs are [out, kh, kw, in]; flattened row-major that is already the
    // (kh, kw, C) order the im2col kernel emits a patch row in.
    push(
        sam("patch_embed.proj.weight"),
        sam_resident_name("patch_embed.proj.weight"),
        vec![SAM_WIDTH, SAM_PATCH, SAM_PATCH, 3],
        (SAM_WIDTH, SAM_PATCH * SAM_PATCH * 3),
    );
    push(
        sam("patch_embed.proj.bias"),
        sam_resident_name("patch_embed.proj.bias"),
        vec![SAM_WIDTH],
        (SAM_WIDTH, 0),
    );
    push(
        sam("pos_embed"),
        sam_resident_name("pos_embed"),
        vec![SAM_GRID, SAM_GRID, SAM_WIDTH],
        (SAM_GRID * SAM_GRID, SAM_WIDTH),
    );
    for block in 0..SAM_DEPTH {
        let global = SAM_GLOBAL_BLOCKS.contains(&block);
        let rel = if global {
            2 * SAM_GRID - 1
        } else {
            2 * SAM_WINDOW - 1
        };
        let d = SAM_WIDTH / SAM_HEADS;
        let shapes: [(Vec<usize>, (usize, usize)); 14] = [
            (vec![SAM_WIDTH], (SAM_WIDTH, 0)),
            (vec![SAM_WIDTH], (SAM_WIDTH, 0)),
            (vec![3 * SAM_WIDTH, SAM_WIDTH], (3 * SAM_WIDTH, SAM_WIDTH)),
            (vec![3 * SAM_WIDTH], (3 * SAM_WIDTH, 0)),
            (vec![SAM_WIDTH, SAM_WIDTH], (SAM_WIDTH, SAM_WIDTH)),
            (vec![SAM_WIDTH], (SAM_WIDTH, 0)),
            (vec![rel, d], (rel, d)),
            (vec![rel, d], (rel, d)),
            (vec![SAM_WIDTH], (SAM_WIDTH, 0)),
            (vec![SAM_WIDTH], (SAM_WIDTH, 0)),
            (vec![SAM_MLP, SAM_WIDTH], (SAM_MLP, SAM_WIDTH)),
            (vec![SAM_MLP], (SAM_MLP, 0)),
            (vec![SAM_WIDTH, SAM_MLP], (SAM_WIDTH, SAM_MLP)),
            (vec![SAM_WIDTH], (SAM_WIDTH, 0)),
        ];
        for (tail, (want, recorded)) in SAM_BLOCK_TENSORS.iter().zip(shapes) {
            push(
                sam(&format!("blocks.{block}.{tail}")),
                sam_block_resident_name(block, tail),
                want,
                recorded,
            );
        }
    }
    push(
        sam("neck.0.weight"),
        sam_resident_name("neck.0.weight"),
        vec![SAM_NECK, 1, 1, SAM_WIDTH],
        (SAM_NECK, SAM_WIDTH),
    );
    for idx in ["neck.1", "neck.3"] {
        for leaf in ["weight", "bias"] {
            push(
                sam(&format!("{idx}.{leaf}")),
                sam_resident_name(&format!("{idx}.{leaf}")),
                vec![SAM_NECK],
                (SAM_NECK, 0),
            );
        }
    }
    push(
        sam("neck.2.weight"),
        sam_resident_name("neck.2.weight"),
        vec![SAM_NECK, 3, 3, SAM_NECK],
        (SAM_NECK, 9 * SAM_NECK),
    );
    push(
        sam("net_2.weight"),
        sam_resident_name("net_2.weight"),
        vec![SAM_NET2, 3, 3, SAM_NECK],
        (SAM_NET2, 9 * SAM_NECK),
    );
    push(
        sam("net_3.weight"),
        sam_resident_name("net_3.weight"),
        vec![SAM_NET3, 3, 3, SAM_NET2],
        (SAM_NET3, 9 * SAM_NET2),
    );
    let clip = |tail: &str| format!("{OCR_CLIP_PREFIX}{tail}");
    push(
        clip("embeddings.class_embedding"),
        clip_resident_name("class_embedding"),
        vec![CLIP_WIDTH],
        (CLIP_WIDTH, 0),
    );
    push(
        clip("embeddings.position_embedding.weight"),
        clip_resident_name("pos_embed"),
        vec![CLIP_POSITIONS, CLIP_WIDTH],
        (CLIP_POSITIONS, CLIP_WIDTH),
    );
    for leaf in ["weight", "bias"] {
        push(
            clip(&format!("pre_layrnorm.{leaf}")),
            clip_resident_name(&format!("pre_layrnorm.{leaf}")),
            vec![CLIP_WIDTH],
            (CLIP_WIDTH, 0),
        );
    }
    push(
        "projector.layers.weight".to_string(),
        projector_resident_name("projector.weight"),
        vec![out_hidden, proj_in],
        (out_hidden, proj_in),
    );
    push(
        "projector.layers.bias".to_string(),
        projector_resident_name("projector.bias"),
        vec![out_hidden],
        (out_hidden, 0),
    );
    push(
        "image_newline".to_string(),
        projector_resident_name("image_newline"),
        vec![out_hidden],
        (out_hidden, 0),
    );
    push(
        "view_separator".to_string(),
        projector_resident_name("view_separator"),
        vec![out_hidden],
        (out_hidden, 0),
    );

    // Classify every base name against the known set before any bytes move: a
    // CLIP block tensor, a known resident source, or a drop. Anything else is
    // refused by name.
    let mut item_seen = vec![false; items.len()];
    let mut clip_seen = vec![[false; 12]; CLIP_DEPTH];
    let clip_layers = format!("{OCR_CLIP_PREFIX}transformer.layers.");
    for &name in vision_bases {
        if DROPPED.contains(&name) {
            continue;
        }
        if let Some(slot) = items.iter().position(|i| i.source == name) {
            if std::mem::replace(&mut item_seen[slot], true) {
                return Err(Gemma4Error::UnknownTensor(format!("{name}: duplicate")));
            }
            continue;
        }
        let rest = name.strip_prefix(&clip_layers).ok_or_else(|| {
            Gemma4Error::UnknownTensor(format!("{name}: not a DeepSeek-OCR tower tensor"))
        })?;
        let (index, suffix) = rest.split_once('.').ok_or_else(|| {
            Gemma4Error::UnknownTensor(format!("{name}: no layer index and suffix"))
        })?;
        let index: usize = index.parse().map_err(|_| {
            Gemma4Error::UnknownTensor(format!("{name}: layer index is not a number"))
        })?;
        let role = CLIP_BLOCK_SOURCE_SUFFIXES
            .iter()
            .position(|s| *s == suffix)
            .ok_or_else(|| {
                Gemma4Error::UnknownTensor(format!(
                    "{name}: {suffix} is not one of the twelve per-layer tensors"
                ))
            })?;
        if index >= CLIP_DEPTH {
            return Err(Gemma4Error::UnknownTensor(format!(
                "{name}: layer {index} is past the declared depth {CLIP_DEPTH}"
            )));
        }
        if std::mem::replace(&mut clip_seen[index][role], true) {
            return Err(Gemma4Error::UnknownTensor(format!("{name}: duplicate")));
        }
    }
    // A missing tensor would stream as zeros and the tower would run with a
    // sublayer silently off: refuse it by name.
    if let Some(slot) = item_seen.iter().position(|seen| !seen) {
        return Err(Gemma4Error::MissingTensor(items[slot].source.clone()));
    }
    for (index, roles) in clip_seen.iter().enumerate() {
        if let Some(role) = roles.iter().position(|seen| !seen) {
            return Err(Gemma4Error::MissingTensor(format!(
                "{clip_layers}{index}.{}",
                CLIP_BLOCK_SOURCE_SUFFIXES[role]
            )));
        }
    }

    let mut lossy_conversion = Vec::new();
    let mut load = |name: &str, want: &[usize]| -> Result<Vec<u8>, Gemma4Error> {
        let (converted, shape) = read_tower_tensor(shards, name, quant)?;
        let got: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        // A position table is stored [1, h, w, c]: only the element count and
        // trailing dims matter to a row-major reader.
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

    let mut entries = Vec::with_capacity(items.len());
    for item in &items {
        let bytes = load(&item.source, &item.want)?;
        let shape = if item.recorded.1 == 0 {
            shape4(&[u64::from(item.recorded.0)])
        } else {
            (item.recorded.0, item.recorded.1, 0, 0)
        };
        entries.push(ResidentEntrySpec::Raw(RawTensorSpec {
            name: item.install.clone(),
            dtype: crate::DTYPE_FP16,
            bytes,
            shape,
        }));
    }

    let mut experts = Vec::with_capacity(CLIP_DEPTH);
    let mut widest = 0u64;
    for index in 0..CLIP_DEPTH {
        let base = format!("{clip_layers}{index}.");
        let h = CLIP_WIDTH;
        let shapes: [Vec<usize>; 12] = [
            vec![h],
            vec![h],
            vec![3 * h, h],
            vec![3 * h],
            vec![h, h],
            vec![h],
            vec![h],
            vec![h],
            vec![CLIP_MLP, h],
            vec![CLIP_MLP],
            vec![h, CLIP_MLP],
            vec![h],
        ];
        let mut sub_tensors = Vec::with_capacity(BLOCK_ROLE_NAMES.len());
        let mut used = 0u64;
        for (role, shape) in shapes.iter().enumerate() {
            let name = format!("{base}{}", CLIP_BLOCK_SOURCE_SUFFIXES[role]);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The published `mlx-community/DeepSeek-OCR-4bit` config, trimmed to the
    /// keys this parser reads.
    fn real_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "deepseekocr",
            "global_view_pos": "head",
            "tile_tag": "2D",
            "projector_config": {
                "input_dim": 2048, "model_type": "mlp_projector",
                "n_embed": 1280, "projector_type": "linear"
            },
            "vision_config": {
                "image_size": 1024, "mlp_ratio": 3.7362, "model_type": "vision",
                "width": {
                    "clip-l-14-224": {"heads": 16, "image_size": 224, "layers": 24,
                                      "patch_size": 14, "width": 1024},
                    "sam_vit_b": {"downsample_channels": [512, 1024],
                                  "global_attn_indexes": [2, 5, 8, 11],
                                  "heads": 12, "layers": 12, "width": 768}
                }
            }
        })
    }

    fn parse(v: &serde_json::Value) -> Result<VisionConfig, Gemma4Error> {
        parse_deepseek_ocr_vision_config(&v.to_string())
    }

    #[test]
    fn the_published_config_parses_to_the_sam_clip_tower() {
        let v = parse(&real_config()).unwrap();
        assert_eq!(v.tower, VisionTowerKind::DeepseekOcr);
        assert_eq!(
            (v.depth, v.hidden_size, v.intermediate_size),
            (24, 1024, 4096)
        );
        assert_eq!(
            (v.num_heads, v.patch_size, v.num_position_embeddings),
            (16, 16, 257)
        );
        assert_eq!((v.spatial_merge_size, v.out_hidden_size), (4, 1280));
        assert_eq!(v.image_token_id, DEEPSEEK_OCR_IMAGE_TOKEN_ID);
        assert!(v.is_active());
    }

    #[test]
    fn the_unlimited_ocr_spelling_is_accepted() {
        let mut v = real_config();
        v["model_type"] = "unlimited-ocr".into();
        assert!(parse(&v).is_ok());
    }

    #[test]
    fn a_tower_this_reader_does_not_implement_is_refused() {
        for (path, value) in [
            ("tile_tag", serde_json::json!("1D")),
            ("global_view_pos", serde_json::json!("tail")),
            ("model_type", serde_json::json!("deepseek_vl_v2")),
        ] {
            let mut v = real_config();
            v[path] = value;
            assert!(parse(&v).is_err(), "{path}");
        }
        let mut v = real_config();
        v["projector_config"]["projector_type"] = "downsample_mlp_gelu".into();
        assert!(parse(&v).is_err());
        let mut v = real_config();
        v["projector_config"]["input_dim"] = 1024.into();
        assert!(parse(&v).is_err());
        let mut v = real_config();
        v["vision_config"]["image_size"] = 640.into();
        assert!(parse(&v).is_err());
        let mut v = real_config();
        v["vision_config"]["width"]["sam_vit_b"]["global_attn_indexes"] = serde_json::json!([3, 7]);
        assert!(parse(&v).is_err());
        let mut v = real_config();
        v["vision_config"]["width"]["clip-l-14-224"]["layers"] = 12.into();
        assert!(parse(&v).is_err());
    }

    #[test]
    fn a_config_with_no_vision_section_is_the_none_config() {
        let mut v = real_config();
        v.as_object_mut().unwrap().remove("vision_config");
        assert!(!parse(&v).unwrap().is_active());
    }

    /// The names the runtime resolves, spelled out independently of the
    /// ingest's own table, so a rename on one side shows up here.
    #[test]
    fn the_install_names_are_unique_and_under_the_vision_prefix() {
        let mut names = std::collections::BTreeSet::new();
        for block in 0..SAM_DEPTH {
            for tail in SAM_BLOCK_TENSORS {
                assert!(names.insert(sam_block_resident_name(block, tail)));
            }
        }
        for tail in [
            "patch_embed.proj.weight",
            "patch_embed.proj.bias",
            "pos_embed",
            "neck.0.weight",
            "neck.1.weight",
            "neck.1.bias",
            "neck.2.weight",
            "neck.3.weight",
            "neck.3.bias",
            "net_2.weight",
            "net_3.weight",
        ] {
            assert!(names.insert(sam_resident_name(tail)));
        }
        for tail in [
            "class_embedding",
            "pos_embed",
            "pre_layrnorm.weight",
            "pre_layrnorm.bias",
        ] {
            assert!(names.insert(clip_resident_name(tail)));
        }
        for tail in [
            "projector.weight",
            "projector.bias",
            "image_newline",
            "view_separator",
        ] {
            assert!(names.insert(projector_resident_name(tail)));
        }
        assert_eq!(names.len(), 12 * 14 + 11 + 4 + 4);
        assert!(names.iter().all(|n| n.starts_with(VISION_INSTALL_PREFIX)));
    }
}
