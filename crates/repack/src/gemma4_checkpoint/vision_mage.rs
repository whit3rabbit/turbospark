//! The Mage-ViT tower of a `mage_vl` checkpoint: config and ingest, still
//! images only (`docs/MAGE_VL_VISION_PHASE0.md`).
//!
//! Read from the MLX conversions' names and shapes
//! (`mlx-community/Mage-VL-8bit` @ 40dc28d825dfaf0dfdea374f39539475d5ee35f2).
//! The block layout is the Qwen2-VL tower's, so the blob roles and the
//! resident names are that tower's, plus the LayerNorm ahead of block 0.

use model_io::{VisionConfig, VisionTowerKind};

use super::classify::{VISION_INSTALL_PREFIX, VISION_PREFIX};
use super::config::Gemma4Error;
use super::shards::{Gemma4Shards, GTURBO_PAGE_BYTES};
use super::vision::{read_tower_tensor, VisionRead, QWEN2_VL_BLOCK_ROLES};
use crate::gturbo_writer::{ExpertBlob, LayerBlobs, SubTensor};
use crate::resident_writer::{RawTensorSpec, ResidentEntrySpec};

/// `config.json`'s root `model_type`.
pub const MAGE_VL_MODEL_TYPE: &str = "mage_vl";

/// Per-block source suffixes, in [`QWEN2_VL_BLOCK_ROLES`] order.
const BLOCK_SUFFIXES: [&str; 12] = [
    "layer_norm1.weight",
    "layer_norm1.bias",
    "self_attn.qkv.weight",
    "self_attn.qkv.bias",
    "self_attn.proj.weight",
    "self_attn.proj.bias",
    "layer_norm2.weight",
    "layer_norm2.bias",
    "mlp.fc1.weight",
    "mlp.fc1.bias",
    "mlp.fc2.weight",
    "mlp.fc2.bias",
];

/// `(source tail, install tail)` for the non-block tensors.
const RESIDENT: [(&str, &str); 9] = [
    (
        "embeddings.patch_embedding.weight",
        "patch_embed.proj.weight",
    ),
    ("layernorm_pre.weight", "layernorm_pre.weight"),
    ("layernorm_pre.bias", "layernorm_pre.bias"),
    ("merger.ln_q.weight", "merger.ln_q.weight"),
    ("merger.ln_q.bias", "merger.ln_q.bias"),
    ("merger.mlp.0.weight", "merger.mlp.0.weight"),
    ("merger.mlp.0.bias", "merger.mlp.0.bias"),
    ("merger.mlp.2.weight", "merger.mlp.2.weight"),
    ("merger.mlp.2.bias", "merger.mlp.2.bias"),
];

fn bad(why: impl std::fmt::Display) -> Gemma4Error {
    Gemma4Error::Config(format!("mage_vl vision: {why}"))
}

/// Parses `vision_config`, refusing every value this port hardcodes when the
/// file states another.
pub fn parse_mage_vl_vision_config(json: &str) -> Result<VisionConfig, Gemma4Error> {
    let root: serde_json::Value = serde_json::from_str(json).map_err(bad)?;
    if root.get("model_type").and_then(|v| v.as_str()) != Some(MAGE_VL_MODEL_TYPE) {
        return Err(bad("root model_type is not mage_vl"));
    }
    let vc = root
        .get("vision_config")
        .ok_or_else(|| bad("missing vision_config"))?;
    let i = |key: &str| {
        vc.get(key)
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| bad(format!("missing vision_config.{key}")))
    };
    let root_id = |key: &str| {
        root.get(key)
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| bad(format!("missing {key}")))
    };
    for (key, want) in [("hidden_act", "gelu"), ("layer_norm_type", "layer_norm")] {
        if vc.get(key).and_then(|v| v.as_str()) != Some(want) {
            return Err(bad(format!("{key} must be {want:?}")));
        }
    }
    for (key, want) in [("layer_norm_eps", 1e-6), ("rope_theta", 10_000.0)] {
        if vc.get(key).and_then(serde_json::Value::as_f64) != Some(want) {
            return Err(bad(format!("{key} must be {want}")));
        }
    }
    for (key, want) in [
        ("patch_size", 16),
        ("temporal_patch_size", 1),
        ("spatial_merge_size", 2),
        ("num_channels", 3),
    ] {
        if i(key)? != want {
            return Err(bad(format!("{key} must be {want}")));
        }
    }
    for key in ["use_head", "use_patch_position_encoding"] {
        if vc.get(key).and_then(serde_json::Value::as_bool) == Some(true) {
            return Err(bad(format!("{key} true is not implemented")));
        }
    }
    let (depth, hidden, heads, intermediate, out) = (
        i("num_hidden_layers")?,
        i("hidden_size")?,
        i("num_attention_heads")?,
        i("intermediate_size")?,
        i("out_hidden_size")?,
    );
    // Bounded so no later size product can overflow.
    if !(1..=64).contains(&depth)
        || !(1..=8192).contains(&hidden)
        || !(1..=65536).contains(&intermediate)
        || !(1..=16384).contains(&out)
        || heads <= 0
        || hidden % heads != 0
        // The rope splits head_dim / 2 into 4:6:6 sixteenths.
        || (hidden / heads) % 32 != 0
    {
        return Err(bad(format!(
            "unsupported geometry: depth {depth}, hidden {hidden}, heads {heads}, \
             intermediate {intermediate}, out {out}"
        )));
    }
    Ok(VisionConfig {
        depth,
        hidden_size: hidden,
        intermediate_size: intermediate,
        num_heads: heads,
        patch_size: 16,
        temporal_patch_size: 1,
        in_channels: 3,
        spatial_merge_size: 2,
        num_position_embeddings: 0,
        out_hidden_size: out,
        mrope_section: [0, 0, 0],
        deepstack_visual_indexes: Vec::new(),
        vision_start_token_id: root_id("vision_start_token_id")?,
        vision_end_token_id: root_id("vision_end_token_id")?,
        image_token_id: root_id("image_token_id")?,
        video_token_id: root_id("video_token_id")?,
        tower: VisionTowerKind::MageVit,
    })
}

/// One tensor as FP16, refused unless its shape is exactly `want`. The patch
/// embedding is why shapes and not just sizes are compared: the PyTorch
/// `[out, C, P, P]` layout has the same byte count as `[out, P, P, C]`.
fn read(
    shards: &Gemma4Shards<'_>,
    name: &str,
    want: &[u64],
) -> Result<(Vec<u8>, usize), Gemma4Error> {
    let (converted, shape) = read_tower_tensor(shards, name, None)?;
    if shape != want {
        return Err(Gemma4Error::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!("shape {shape:?}, expected {want:?}"),
        });
    }
    Ok((converted.bytes, converted.lossy))
}

/// Reads the tower: nine resident tensors and `depth` blocks of twelve.
pub(super) fn read_mage_vl_vision_entries(
    shards: &Gemma4Shards<'_>,
    vision_bases: &[&str],
    vision: &VisionConfig,
) -> Result<VisionRead, Gemma4Error> {
    if vision.tower != VisionTowerKind::MageVit || vision.depth < 1 {
        return Err(bad("not an active mage_vit tower"));
    }
    let depth = vision.depth as usize;
    let h = vision.hidden_size as u64;
    let inter = vision.intermediate_size as u64;
    let out = vision.out_hidden_size as u64;
    let merged = h * 4;

    // Count first, so a config cannot make anything below proportional to a
    // depth the artifact does not carry. The listed names are the checks.
    let expected = depth * BLOCK_SUFFIXES.len() + RESIDENT.len();
    let known = |name: &str| {
        name.strip_prefix(VISION_PREFIX).is_some_and(|tail| {
            if let Some(rest) = tail.strip_prefix("encoder.layers.") {
                rest.split_once('.').is_some_and(|(n, suffix)| {
                    n.parse::<usize>()
                        .is_ok_and(|n| n < depth && !suffix.is_empty())
                        && BLOCK_SUFFIXES.contains(&suffix)
                })
            } else {
                RESIDENT.iter().any(|(source, _)| *source == tail)
            }
        })
    };
    if let Some(name) = vision_bases.iter().find(|n| !known(n)) {
        return Err(Gemma4Error::UnknownTensor(format!(
            "{name}: not a tensor of the {depth}-block mage_vit tower"
        )));
    }

    let mut lossy_conversion = Vec::new();
    let mut note = |name: &str, lossy: usize| {
        if lossy > 0 {
            lossy_conversion.push((name.to_string(), lossy));
        }
    };

    let mut entries = Vec::with_capacity(RESIDENT.len());
    for (source, install) in RESIDENT {
        let name = format!("{VISION_PREFIX}{source}");
        let (want, index): (Vec<u64>, (u32, u32)) = match install {
            "patch_embed.proj.weight" => (vec![h, 16, 16, 3], (h as u32, 768)),
            "layernorm_pre.weight"
            | "layernorm_pre.bias"
            | "merger.ln_q.weight"
            | "merger.ln_q.bias" => (vec![h], (h as u32, 0)),
            "merger.mlp.0.weight" => (vec![merged, merged], (merged as u32, merged as u32)),
            "merger.mlp.0.bias" => (vec![merged], (merged as u32, 0)),
            "merger.mlp.2.weight" => (vec![out, merged], (out as u32, merged as u32)),
            _ => (vec![out], (out as u32, 0)),
        };
        if !vision_bases.contains(&name.as_str()) {
            return Err(Gemma4Error::MissingTensor(name));
        }
        let (bytes, lossy) = read(shards, &name, &want)?;
        note(&name, lossy);
        entries.push(ResidentEntrySpec::Raw(RawTensorSpec {
            name: format!("{VISION_INSTALL_PREFIX}{install}"),
            dtype: crate::DTYPE_FP16,
            bytes,
            shape: (index.0, index.1, 0, 0),
        }));
    }
    if vision_bases.len() != expected {
        return Err(bad(format!(
            "{} tower tensors, expected {expected} for {depth} blocks",
            vision_bases.len()
        )));
    }

    let mut experts = Vec::with_capacity(depth);
    let mut widest = 0u64;
    for block in 0..depth {
        let mut sub_tensors = Vec::with_capacity(BLOCK_SUFFIXES.len());
        let mut used = 0u64;
        for (role, suffix) in BLOCK_SUFFIXES.iter().enumerate() {
            let name = format!("{VISION_PREFIX}encoder.layers.{block}.{suffix}");
            let want: Vec<u64> = match role {
                2 => vec![3 * h, h],
                3 => vec![3 * h],
                4 => vec![h, h],
                8 => vec![inter, h],
                9 => vec![inter],
                10 => vec![h, inter],
                _ => vec![h],
            };
            if !vision_bases.contains(&name.as_str()) {
                return Err(Gemma4Error::MissingTensor(name));
            }
            let (bytes, lossy) = read(shards, &name, &want)?;
            note(&name, lossy);
            used += bytes.len() as u64;
            sub_tensors.push(SubTensor {
                role: QWEN2_VL_BLOCK_ROLES[role].0.to_string(),
                bytes,
                dtype: "fp16".to_string(),
                shape: want,
            });
        }
        widest = widest.max(used);
        experts.push(ExpertBlob {
            expert: block,
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
