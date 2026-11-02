//! The MiniCPM-V 4.7 tower's ingest (`docs/MINICPMV_VISION_PHASE0.md`).
//!
//! Three source prefixes make one tower: `vision_tower.` (the SigLIP encoder),
//! `vit_merger.` (the 2x2 window attention-merger that runs after block 6) and
//! `merger.` (the final 2x2 MLP into the trunk). The 27 encoder blocks are
//! packed into `packed_vision/` and streamed; everything else is resident,
//! renamed under the install's `vision.` prefix with the source prefix kept as
//! the first path component of the tail (`vision.vit_merger.*`,
//! `vision.merger.*`), so no name can collide with another tower's.
//!
//! The checkpoint stores the tower in BF16; every tensor is narrowed to FP16
//! by the shared converter, which refuses a value FP16 cannot hold.

use model_io::VisionConfig;

use super::classify::{VISION_INSTALL_PREFIX, VISION_PREFIX};
use super::config::Gemma4Error;
use super::shards::{Gemma4Shards, GTURBO_PAGE_BYTES};
use super::vision::{read_tower_tensor, resident_shape, VisionRead};
use crate::gturbo_writer::{ExpertBlob, LayerBlobs, SubTensor};
use crate::resident_writer::{RawTensorSpec, ResidentEntrySpec};

/// `(role in the blob, suffix under vision_tower.encoder.layers.N.)`, in the
/// order the block executes. Sixteen per block: two LayerNorms with bias, four
/// attention projections with bias, and the two-layer MLP with bias.
pub const MINICPM_BLOCK_ROLES: [(&str, &str); 16] = [
    ("ln1_w", "layer_norm1.weight"),
    ("ln1_b", "layer_norm1.bias"),
    ("q_w", "self_attn.q_proj.weight"),
    ("q_b", "self_attn.q_proj.bias"),
    ("k_w", "self_attn.k_proj.weight"),
    ("k_b", "self_attn.k_proj.bias"),
    ("v_w", "self_attn.v_proj.weight"),
    ("v_b", "self_attn.v_proj.bias"),
    ("o_w", "self_attn.out_proj.weight"),
    ("o_b", "self_attn.out_proj.bias"),
    ("ln2_w", "layer_norm2.weight"),
    ("ln2_b", "layer_norm2.bias"),
    ("fc1_w", "mlp.fc1.weight"),
    ("fc1_b", "mlp.fc1.bias"),
    ("fc2_w", "mlp.fc2.weight"),
    ("fc2_b", "mlp.fc2.bias"),
];

/// `vision_tower.` tensors outside the blocks.
pub const MINICPM_TOWER_RESIDENT: [&str; 5] = [
    "embeddings.patch_embedding.weight",
    "embeddings.patch_embedding.bias",
    "embeddings.position_embedding.weight",
    "post_layernorm.weight",
    "post_layernorm.bias",
];

/// `vit_merger.` tensors: a 4-token window self-attention (`layer_norm1`,
/// q/k/v/out) and the window MLP (`pre_norm`, `linear_1`, `linear_2`).
pub const MINICPM_VIT_MERGER: [&str; 16] = [
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
    "pre_norm.weight",
    "pre_norm.bias",
    "linear_1.weight",
    "linear_1.bias",
    "linear_2.weight",
    "linear_2.bias",
];

/// `merger.` tensors: one `MergerBlock` (`mlp.0`, since `merger_times` is 1).
pub const MINICPM_MERGER: [&str; 6] = [
    "mlp.0.pre_norm.weight",
    "mlp.0.pre_norm.bias",
    "mlp.0.linear_1.weight",
    "mlp.0.linear_1.bias",
    "mlp.0.linear_2.weight",
    "mlp.0.linear_2.bias",
];

pub const VIT_MERGER_PREFIX: &str = "vit_merger.";
pub const MERGER_PREFIX: &str = "merger.";

/// Every source prefix of this tower, for the fetchers that select by prefix.
pub const MINICPM_SOURCE_PREFIXES: [&str; 3] = [VISION_PREFIX, VIT_MERGER_PREFIX, MERGER_PREFIX];

/// The element count each tensor must have, from the config, so a checkpoint
/// whose shapes disagree with its own `vision_config` is refused at intake
/// instead of binding the wrong number of bytes at a dispatch.
fn expected_elems(v: &VisionConfig, source_tail: &str, group: Group) -> Option<usize> {
    let h = v.hidden_size as usize;
    let i = v.intermediate_size as usize;
    let p = v.patch_size as usize;
    let c = v.in_channels as usize;
    let pos = v.num_position_embeddings as usize;
    let out = v.out_hidden_size as usize;
    let win = 4 * h;
    let win_inter = 4 * i;
    Some(match (group, source_tail) {
        (Group::Block, t) => match t {
            "layer_norm1.weight" | "layer_norm1.bias" | "layer_norm2.weight"
            | "layer_norm2.bias" => h,
            "self_attn.q_proj.bias"
            | "self_attn.k_proj.bias"
            | "self_attn.v_proj.bias"
            | "self_attn.out_proj.bias" => h,
            "self_attn.q_proj.weight"
            | "self_attn.k_proj.weight"
            | "self_attn.v_proj.weight"
            | "self_attn.out_proj.weight" => h * h,
            "mlp.fc1.weight" | "mlp.fc2.weight" => h * i,
            "mlp.fc1.bias" => i,
            "mlp.fc2.bias" => h,
            _ => return None,
        },
        (Group::Tower, t) => match t {
            "embeddings.patch_embedding.weight" => h * p * p * c,
            "embeddings.patch_embedding.bias" | "post_layernorm.weight" | "post_layernorm.bias" => {
                h
            }
            "embeddings.position_embedding.weight" => pos * h,
            _ => return None,
        },
        (Group::VitMerger, t) => match t {
            "layer_norm1.weight" | "layer_norm1.bias" => h,
            "self_attn.q_proj.weight"
            | "self_attn.k_proj.weight"
            | "self_attn.v_proj.weight"
            | "self_attn.out_proj.weight" => h * h,
            "self_attn.q_proj.bias"
            | "self_attn.k_proj.bias"
            | "self_attn.v_proj.bias"
            | "self_attn.out_proj.bias" => h,
            "pre_norm.weight" | "pre_norm.bias" => win,
            "linear_1.weight" => win * win_inter,
            "linear_1.bias" => win_inter,
            "linear_2.weight" => h * win_inter,
            "linear_2.bias" => h,
            _ => return None,
        },
        (Group::Merger, t) => match t {
            "mlp.0.pre_norm.weight" | "mlp.0.pre_norm.bias" => win,
            "mlp.0.linear_1.weight" => win * win,
            "mlp.0.linear_1.bias" => win,
            "mlp.0.linear_2.weight" => out * win,
            "mlp.0.linear_2.bias" => out,
            _ => return None,
        },
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Group {
    Block,
    Tower,
    VitMerger,
    Merger,
}

/// Reads the tower: 27 blocks into `packed_vision/` blobs and the residents,
/// all FP16. Anything under one of the three prefixes that is not in the role
/// tables is refused by name, and a missing tensor is refused rather than
/// zero-filled (a zero block decodes perfectly well).
pub fn read_minicpm_vision_entries(
    shards: &Gemma4Shards<'_>,
    vision_bases: &[&str],
    vision: &VisionConfig,
) -> Result<VisionRead, Gemma4Error> {
    let depth = usize::try_from(vision.depth)
        .ok()
        .filter(|d| *d == crate::qwen36_config::MINICPM_V_DEPTH as usize)
        .ok_or_else(|| {
            Gemma4Error::Config(format!(
                "minicpm_v vision depth {} is not the 27 blocks this port runs",
                vision.depth
            ))
        })?;

    let mut block_names: Vec<Vec<Option<&str>>> =
        vec![vec![None; MINICPM_BLOCK_ROLES.len()]; depth];
    let mut tower_names: Vec<Option<&str>> = vec![None; MINICPM_TOWER_RESIDENT.len()];
    let mut vit_names: Vec<Option<&str>> = vec![None; MINICPM_VIT_MERGER.len()];
    let mut merger_names: Vec<Option<&str>> = vec![None; MINICPM_MERGER.len()];

    for &name in vision_bases {
        if let Some(rest) = name
            .strip_prefix(VISION_PREFIX)
            .and_then(|t| t.strip_prefix("encoder.layers."))
        {
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
            let role = MINICPM_BLOCK_ROLES
                .iter()
                .position(|(_, s)| *s == suffix)
                .ok_or_else(|| {
                    Gemma4Error::UnknownTensor(format!(
                        "{name}: {suffix} is not one of this tower's sixteen per-block tensors"
                    ))
                })?;
            if block_names[index][role].replace(name).is_some() {
                return Err(Gemma4Error::ShapeMismatch {
                    tensor: name.to_string(),
                    detail: format!("two tensors for block {index} role {suffix}"),
                });
            }
        } else if let Some(tail) = name.strip_prefix(VISION_PREFIX) {
            let slot = MINICPM_TOWER_RESIDENT
                .iter()
                .position(|t| *t == tail)
                .ok_or_else(|| {
                    Gemma4Error::UnknownTensor(format!(
                        "{name}: not a block tensor and not one of the tower's five others"
                    ))
                })?;
            if tower_names[slot].replace(name).is_some() {
                return Err(Gemma4Error::ShapeMismatch {
                    tensor: name.to_string(),
                    detail: "duplicate non-block vision tensor".to_string(),
                });
            }
        } else if let Some(tail) = name.strip_prefix(VIT_MERGER_PREFIX) {
            let slot = MINICPM_VIT_MERGER
                .iter()
                .position(|t| *t == tail)
                .ok_or_else(|| {
                    Gemma4Error::UnknownTensor(format!("{name}: not a vit_merger tensor"))
                })?;
            if vit_names[slot].replace(name).is_some() {
                return Err(Gemma4Error::ShapeMismatch {
                    tensor: name.to_string(),
                    detail: "duplicate vit_merger tensor".to_string(),
                });
            }
        } else if let Some(tail) = name.strip_prefix(MERGER_PREFIX) {
            let slot = MINICPM_MERGER
                .iter()
                .position(|t| *t == tail)
                .ok_or_else(|| {
                    Gemma4Error::UnknownTensor(format!("{name}: not a merger tensor"))
                })?;
            if merger_names[slot].replace(name).is_some() {
                return Err(Gemma4Error::ShapeMismatch {
                    tensor: name.to_string(),
                    detail: "duplicate merger tensor".to_string(),
                });
            }
        } else {
            return Err(Gemma4Error::UnknownTensor(format!(
                "{name} was classified as a vision tensor but is under none of this tower's \
                 three prefixes"
            )));
        }
    }

    // Every slot must be filled.
    for (index, roles) in block_names.iter().enumerate() {
        for (role, name) in roles.iter().enumerate() {
            if name.is_none() {
                return Err(Gemma4Error::MissingTensor(format!(
                    "{VISION_PREFIX}encoder.layers.{index}.{}",
                    MINICPM_BLOCK_ROLES[role].1
                )));
            }
        }
    }
    for (names, table, prefix) in [
        (&tower_names, &MINICPM_TOWER_RESIDENT[..], VISION_PREFIX),
        (&vit_names, &MINICPM_VIT_MERGER[..], VIT_MERGER_PREFIX),
        (&merger_names, &MINICPM_MERGER[..], MERGER_PREFIX),
    ] {
        for (slot, name) in names.iter().enumerate() {
            if name.is_none() {
                return Err(Gemma4Error::MissingTensor(format!(
                    "{prefix}{}",
                    table[slot]
                )));
            }
        }
    }

    let mut lossy_conversion = Vec::new();
    let check = |name: &str, tail: &str, group: Group, got: &[u64]| -> Result<(), Gemma4Error> {
        let want = expected_elems(vision, tail, group).ok_or_else(|| {
            Gemma4Error::UnknownTensor(format!("{name}: no expected size for {tail}"))
        })?;
        let have: u64 = got.iter().product();
        if have as usize != want {
            return Err(Gemma4Error::ShapeMismatch {
                tensor: name.to_string(),
                detail: format!("{have} elements, the config implies {want}"),
            });
        }
        Ok(())
    };

    // Residents: tower, then vit_merger, then merger.
    let mut entries = Vec::new();
    for (names, table, group, install_prefix) in [
        (&tower_names, &MINICPM_TOWER_RESIDENT[..], Group::Tower, ""),
        (
            &vit_names,
            &MINICPM_VIT_MERGER[..],
            Group::VitMerger,
            VIT_MERGER_PREFIX,
        ),
        (
            &merger_names,
            &MINICPM_MERGER[..],
            Group::Merger,
            MERGER_PREFIX,
        ),
    ] {
        for (slot, name) in names.iter().enumerate() {
            let name = name.expect("checked above");
            let (converted, shape) = read_tower_tensor(shards, name, None)?;
            check(name, table[slot], group, &shape)?;
            if converted.lossy > 0 {
                lossy_conversion.push((name.to_string(), converted.lossy));
            }
            entries.push(ResidentEntrySpec::Raw(RawTensorSpec {
                name: format!("{VISION_INSTALL_PREFIX}{install_prefix}{}", table[slot]),
                dtype: converted.dtype,
                bytes: converted.bytes,
                shape: resident_shape(&shape),
            }));
        }
    }

    // The blocks.
    let mut experts = Vec::with_capacity(depth);
    let mut widest = 0u64;
    for (index, roles) in block_names.iter().enumerate() {
        let mut sub_tensors = Vec::with_capacity(MINICPM_BLOCK_ROLES.len());
        let mut used = 0u64;
        for (role, name) in roles.iter().enumerate() {
            let name = name.expect("checked above");
            let (converted, shape) = read_tower_tensor(shards, name, None)?;
            check(name, MINICPM_BLOCK_ROLES[role].1, Group::Block, &shape)?;
            if converted.lossy > 0 {
                lossy_conversion.push((name.to_string(), converted.lossy));
            }
            used += converted.bytes.len() as u64;
            sub_tensors.push(SubTensor {
                role: MINICPM_BLOCK_ROLES[role].0.to_string(),
                bytes: converted.bytes,
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

    #[test]
    fn the_role_tables_cover_the_published_tower() {
        // 16 per block x 27 + 5 tower + 16 vit_merger + 6 merger = the 459
        // vision tensors in the published index (437 + 16 + 6).
        assert_eq!(
            MINICPM_BLOCK_ROLES.len() * 27
                + MINICPM_TOWER_RESIDENT.len()
                + MINICPM_VIT_MERGER.len()
                + MINICPM_MERGER.len(),
            459
        );
        let mut suffixes: Vec<&str> = MINICPM_BLOCK_ROLES.iter().map(|(_, s)| *s).collect();
        suffixes.sort_unstable();
        suffixes.dedup();
        assert_eq!(suffixes.len(), 16);
    }
}
