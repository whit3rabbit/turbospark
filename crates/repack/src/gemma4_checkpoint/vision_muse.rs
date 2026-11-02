//! The Muse-Glimmer vision tower's ingest (`docs/MUSEGLIMMER_VISION_PHASE0.md`).
//!
//! The checkpoint splits its vision side across three prefixes and the reader
//! here treats them as ONE tower:
//!
//! | source prefix | what | dtype on `Muse-Glimmer-30B-4bit` |
//! |---|---|---|
//! | `vision_tower.` | patch embedding, position table, `ln_pre`/`ln_post`, 50 blocks | BF16 |
//! | `vision_adapter.` | `fc1` 6144 -> 4096, `fc2` 4096 -> 4096 | 4-bit affine, group 64 |
//! | `vision_projection.` | 4096 -> the trunk's 6656 | 4-bit affine, group 64 |
//!
//! Every tensor lands as FP16 (the tower is FP16 end to end), so the adapter
//! and projection are dequantized here and cost the runtime no kernel.
//!
//! **q, k and v ARE FUSED AT INGEST.** The reference keeps three bias-ful
//! `Linear`s per block; the runtime's fused-qkv matmul path reads one
//! `[3 * hidden, hidden]` weight at three row offsets, so the blob stores the
//! three weights back to back (q, then k, then v) and the three biases the
//! same way. A concatenation of rows is exactly the fused weight, with no
//! arithmetic, so the fused tower computes the reference's three projections.
//!
//! The per-block blob roles are the qwen3 tower's twelve (`BLOCK_ROLES`), in
//! the same order, because the block has the same twelve sub-tensors once qkv
//! is fused.

use std::collections::BTreeSet;

use model_io::VisionConfig;

use super::classify::VISION_INSTALL_PREFIX;
use super::config::{Gemma4Error, Gemma4Quant};
use super::shards::{Gemma4Shards, GTURBO_PAGE_BYTES};
use super::vision::{read_tower_tensor, VisionRead, BLOCK_ROLES};
use crate::gturbo_writer::{ExpertBlob, LayerBlobs, SubTensor};
use crate::resident_writer::{RawTensorSpec, ResidentEntrySpec};

/// Source tensor sub-names that make up each blob role, in `BLOCK_ROLES`
/// order. The qkv roles have three sources, concatenated in this order.
pub const MUSE_BLOCK_SOURCES: [&[&str]; 12] = [
    &["norm1.weight"],
    &["norm1.bias"],
    &[
        "attn.q_proj.weight",
        "attn.k_proj.weight",
        "attn.v_proj.weight",
    ],
    &["attn.q_proj.bias", "attn.k_proj.bias", "attn.v_proj.bias"],
    &["attn.proj.weight"],
    &["attn.proj.bias"],
    &["norm2.weight"],
    &["norm2.bias"],
    &["mlp.fc1.weight"],
    &["mlp.fc1.bias"],
    &["mlp.fc2.weight"],
    &["mlp.fc2.bias"],
];

/// The non-block tensors: `(source name, install name under `vision.`)`.
pub const MUSE_RESIDENT_TENSORS: [(&str, &str); 9] = [
    (
        "vision_tower.patch_embedder.patch_embedding.weight",
        "patch_embed.proj.weight",
    ),
    (
        "vision_tower.patch_embedder.position_embedding_table.weight",
        "pos_embed.weight",
    ),
    ("vision_tower.ln_pre.weight", "ln_pre.weight"),
    ("vision_tower.ln_pre.bias", "ln_pre.bias"),
    ("vision_tower.ln_post.weight", "ln_post.weight"),
    ("vision_tower.ln_post.bias", "ln_post.bias"),
    ("vision_adapter.fc1.weight", "adapter.fc1.weight"),
    ("vision_adapter.fc2.weight", "adapter.fc2.weight"),
    ("vision_projection.weight", "projection.weight"),
];

/// Source prefixes the sidecar fetch has to pull.
pub const MUSE_SOURCE_PREFIXES: [&str; 3] =
    ["vision_tower.", "vision_adapter.", "vision_projection."];

const BLOCK_PREFIX: &str = "vision_tower.layers.";

/// Reads the whole tower. `vision_bases` is every `.weight`/`.bias` name under
/// the three prefixes (the packed planes' `.scales`/`.biases` companions are
/// read beside their weight and are not bases).
pub fn read_muse_vision_entries(
    shards: &Gemma4Shards<'_>,
    vision_bases: &[&str],
    vision: &VisionConfig,
    quant: Option<&Gemma4Quant>,
) -> Result<VisionRead, Gemma4Error> {
    let depth = usize::try_from(vision.depth)
        .ok()
        .filter(|d| crate::qwen36_config::SUPPORTED_MUSE_VISION_DEPTHS.contains(&(*d as i64)))
        .ok_or_else(|| {
            Gemma4Error::Config(format!(
                "muse vision depth {} is unsupported; expected one of {:?}",
                vision.depth,
                crate::qwen36_config::SUPPORTED_MUSE_VISION_DEPTHS
            ))
        })?;
    let hidden = vision.hidden_size as u64;

    // The exact set of names this tower is made of. Anything else is refused
    // by name before any bytes move, and anything missing is too.
    let mut expected: BTreeSet<String> = BTreeSet::new();
    for layer in 0..depth {
        for sources in MUSE_BLOCK_SOURCES {
            for suffix in sources {
                expected.insert(format!("{BLOCK_PREFIX}{layer}.{suffix}"));
            }
        }
    }
    for (source, _) in MUSE_RESIDENT_TENSORS {
        expected.insert(source.to_string());
    }
    let present: BTreeSet<&str> = vision_bases.iter().copied().collect();
    if let Some(name) = vision_bases.iter().find(|n| !expected.contains(**n)) {
        return Err(Gemma4Error::UnknownTensor(format!(
            "{name}: not one of the muse vision tower's tensors"
        )));
    }
    if let Some(name) = expected.iter().find(|n| !present.contains(n.as_str())) {
        return Err(Gemma4Error::MissingTensor(name.clone()));
    }

    let mut lossy_conversion = Vec::new();
    let mut note_lossy = |name: &str, lossy: usize| {
        if lossy > 0 {
            lossy_conversion.push((name.to_string(), lossy));
        }
    };

    // The residents, shape-checked against the config so a tower this
    // config does not describe is refused here and not at a dispatch.
    let mut entries = Vec::with_capacity(MUSE_RESIDENT_TENSORS.len());
    let mut projector_hidden: Option<u64> = None;
    for (source, install) in MUSE_RESIDENT_TENSORS {
        let (converted, shape) = read_tower_tensor(shards, source, quant)?;
        note_lossy(source, converted.lossy);
        let bad = |why: String| Gemma4Error::ShapeMismatch {
            tensor: source.to_string(),
            detail: why,
        };
        match install {
            "patch_embed.proj.weight" => {
                let patch_dim = (vision.temporal_patch_size
                    * vision.in_channels
                    * vision.patch_size
                    * vision.patch_size) as u64;
                if shape != [hidden, patch_dim] {
                    return Err(bad(format!(
                        "expected [{hidden}, {patch_dim}], got {shape:?}"
                    )));
                }
            }
            "pos_embed.weight" => {
                if shape != [vision.num_position_embeddings as u64, hidden] {
                    return Err(bad(format!(
                        "expected [{}, {hidden}], got {shape:?}",
                        vision.num_position_embeddings
                    )));
                }
            }
            "adapter.fc1.weight" => {
                let merged =
                    hidden * (vision.spatial_merge_size * vision.spatial_merge_size) as u64;
                if shape.len() != 2 || shape[1] != merged {
                    return Err(bad(format!("expected [_, {merged}], got {shape:?}")));
                }
                projector_hidden = Some(shape[0]);
            }
            "adapter.fc2.weight" => {
                let p = projector_hidden.unwrap_or(0);
                if shape != [p, p] {
                    return Err(bad(format!("expected [{p}, {p}], got {shape:?}")));
                }
            }
            "projection.weight" => {
                let p = projector_hidden.unwrap_or(0);
                if shape != [vision.out_hidden_size as u64, p] {
                    return Err(bad(format!(
                        "expected [{}, {p}], got {shape:?}",
                        vision.out_hidden_size
                    )));
                }
            }
            _ => {
                if shape != [hidden] {
                    return Err(bad(format!("expected [{hidden}], got {shape:?}")));
                }
            }
        }
        entries.push(ResidentEntrySpec::Raw(RawTensorSpec {
            name: format!("{VISION_INSTALL_PREFIX}{install}"),
            dtype: converted.dtype,
            bytes: converted.bytes,
            shape: (
                shape[0] as u32,
                shape.get(1).copied().unwrap_or(0) as u32,
                0,
                0,
            ),
        }));
    }

    // The blocks: one blob each, the three projections fused.
    let mut experts = Vec::with_capacity(depth);
    let mut widest = 0u64;
    for layer in 0..depth {
        let mut sub_tensors = Vec::with_capacity(BLOCK_ROLES.len());
        let mut used = 0u64;
        for (role, sources) in BLOCK_ROLES.iter().zip(MUSE_BLOCK_SOURCES) {
            let mut bytes: Vec<u8> = Vec::new();
            let mut shape: Vec<u64> = Vec::new();
            for (k, suffix) in sources.iter().enumerate() {
                let name = format!("{BLOCK_PREFIX}{layer}.{suffix}");
                let (converted, part_shape) = read_tower_tensor(shards, &name, quant)?;
                note_lossy(&name, converted.lossy);
                if k == 0 {
                    shape = part_shape;
                } else if part_shape != shape[..] {
                    return Err(Gemma4Error::ShapeMismatch {
                        tensor: name,
                        detail: format!(
                            "fused projections must agree: {part_shape:?} against {shape:?}"
                        ),
                    });
                }
                bytes.extend_from_slice(&converted.bytes);
            }
            // The fused tensor stacks the parts along the leading axis.
            shape[0] *= sources.len() as u64;
            used += bytes.len() as u64;
            sub_tensors.push(SubTensor {
                role: role.0.to_string(),
                bytes,
                dtype: "fp16".to_string(),
                shape,
            });
        }
        widest = widest.max(used);
        experts.push(ExpertBlob {
            expert: layer,
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
    fn the_source_table_covers_the_published_tower() {
        // 16 source tensors per block (q/k/v counted separately) x 50 blocks
        // + the 6 tower residents = the 806 `vision_tower.` weights/biases
        // the published index carries, plus 3 adapter/projection weights.
        let per_block: usize = MUSE_BLOCK_SOURCES.iter().map(|s| s.len()).sum();
        assert_eq!(per_block, 16);
        assert_eq!(per_block * 50 + MUSE_RESIDENT_TENSORS.len(), 809);
        assert_eq!(MUSE_BLOCK_SOURCES.len(), BLOCK_ROLES.len());
    }

    #[test]
    fn qkv_is_the_only_fused_role_and_it_is_q_then_k_then_v() {
        for (role, sources) in BLOCK_ROLES.iter().zip(MUSE_BLOCK_SOURCES) {
            if role.0 == "qkv_w" || role.0 == "qkv_b" {
                assert_eq!(sources.len(), 3);
                assert!(sources[0].contains("q_proj"));
                assert!(sources[1].contains("k_proj"));
                assert!(sources[2].contains("v_proj"));
            } else {
                assert_eq!(sources.len(), 1, "{}", role.0);
            }
        }
    }
}
