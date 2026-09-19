//! The `qwen3_5` vision tower's ingest (ROADMAP M-V3).
//!
//! **This is the walk's first arm whose output goes to TWO destinations.**
//! Every other name group lands entirely in the resident index (`mtp.*`,
//! `dflash.*`) or entirely in packed blobs (the routed experts). The tower
//! splits: its 27 transformer blocks are packed into
//! `packed_vision/blobs.bin` and streamed at inference, while
//! `patch_embed.*`, `pos_embed` and `merger.*` become resident entries. The
//! split is the whole point of the design -- a block is ~32 MiB and only two
//! are live at once (double-buffered), so peak vision residency is bounded by
//! the SLOT COUNT rather than by the tower's 0.858 GiB.
//!
//! **The blocks reuse `PackedExpertsLayout` verbatim and that is not a pun.**
//! `StreamLayout` interprets nothing: a "layer" is a file and an "expert" is a
//! fixed-stride blob inside it, which is exactly what a block loop wants. So
//! the tower is ONE layer of `depth` experts, and it gets
//! `PreadExpertStreamer`, `ExpertCache` and the whole read pool with no new
//! I/O code.
//!
//! **It is also the walk's only FP16 destination**, against `narrow.rs`'s BF16
//! default for every text tensor. See `convert_raw_to_fp16` for why the rule
//! points the other way here and for the overflow refusal that comes with it.

use model_io::{ArchConfig, VisionConfig};

use super::classify::{VISION_INSTALL_PREFIX, VISION_PREFIX};
use super::config::Gemma4Error;
use super::narrow::convert_raw_to_fp16;
use super::shards::{shape4, Gemma4Shards, GTURBO_PAGE_BYTES};
use crate::gturbo_writer::{ExpertBlob, LayerBlobs, SubTensor};
use crate::qwen36_config::SUPPORTED_VISION_DEPTHS;
use crate::resident_writer::{RawTensorSpec, ResidentEntrySpec};

/// One block's twelve sub-tensors: `(role in the blob, suffix in the
/// checkpoint)`.
///
/// **The ORDER is the blob's byte order and the roles are what a kernel asks
/// for**, so the two halves of each pair answer different questions and
/// neither is derivable from the other. The order follows the forward pass --
/// norm1, qkv, proj, norm2, fc1, fc2 -- which costs nothing and means a
/// hexdump of a blob reads in the order the block executes.
///
/// Read off the real checkpoint header rather than from the reference's
/// module names (`docs/VISION_PHASE0.md` item 1). Twelve per block times 27
/// blocks is 324, plus the nine non-block tensors below, is the 333 the
/// published index declares -- an arithmetic check `the_role_table_covers_the_
/// published_tower` makes rather than leaves to this comment.
pub const BLOCK_ROLES: [(&str, &str); 12] = [
    ("ln1_w", "norm1.weight"),
    ("ln1_b", "norm1.bias"),
    ("qkv_w", "attn.qkv.weight"),
    ("qkv_b", "attn.qkv.bias"),
    ("proj_w", "attn.proj.weight"),
    ("proj_b", "attn.proj.bias"),
    ("ln2_w", "norm2.weight"),
    ("ln2_b", "norm2.bias"),
    ("fc1_w", "mlp.linear_fc1.weight"),
    ("fc1_b", "mlp.linear_fc1.bias"),
    ("fc2_w", "mlp.linear_fc2.weight"),
    ("fc2_b", "mlp.linear_fc2.bias"),
];

/// The tower's non-block tensors, which become RESIDENT entries.
///
/// Nine of them, and they are resident rather than streamed because each is
/// read exactly once per image and two of them are large: the merger's
/// `linear_fc1` is [4608, 4608] and its `linear_fc2` [5120, 4608], about
/// 90 MiB between them. Streaming a tensor used once buys nothing -- the whole
/// point of the block loop is that a block is read 27 times per tower run and
/// only two need to be live.
///
/// **`patch_embed.proj.weight` IS COPIED VERBATIM, WITH NO PERMUTATION**, and
/// that is a decision `crates/vision-io` already paid for. Its shape is
/// [1152, 2, 16, 16, 3] -- `(out, T, P_h, P_w, C)` -- and M-V1 chose to emit
/// patch rows in `(T, P_h, P_w, C)` order for exactly this reason, where the
/// mlx-vlm reference carries `(C, T, P_h, P_w)`. Permuting here would make the
/// preprocessing and the GEMM disagree, and nothing would crash: the patch
/// embedding would simply mix the wrong channels into the wrong positions and
/// the tower would produce fluent, wrong embeddings.
pub const RESIDENT_TENSORS: [&str; 9] = [
    "patch_embed.proj.weight",
    "patch_embed.proj.bias",
    "pos_embed.weight",
    "merger.norm.weight",
    "merger.norm.bias",
    "merger.linear_fc1.weight",
    "merger.linear_fc1.bias",
    "merger.linear_fc2.weight",
    "merger.linear_fc2.bias",
];

/// The `vision_tower.deepstack_merger_list.{k}.*` tensors, resident like the
/// main merger's nine. Six per merger (the main merger's structure minus
/// nothing: norm/fc1/fc2, each weight+bias), one merger per
/// `vision_config.deepstack_visual_indexes` entry -- 18 tensors on the
/// `qwen3_vl`-4B tower's three. Written ONLY when the config declares
/// indexes, per `vision_should_ingest`'s artifact-versus-config rule applied
/// one level down: a checkpoint that ships merger tensors without declaring
/// them falls into the unknown-name refusal, and one that declares indexes
/// without shipping the tensors is refused as MISSING rather than silently
/// injecting nothing.
///
/// The suffixes are shared with the main merger's names under
/// [`RESIDENT_TENSORS`], so the runtime resolves a deepstack merger with the
/// same six-role table at a different prefix.
pub const DEEPSTACK_MERGER_PREFIX: &str = "deepstack_merger_list.";
pub const DEEPSTACK_MERGER_SUFFIXES: [&str; 6] = [
    "norm.weight",
    "norm.bias",
    "linear_fc1.weight",
    "linear_fc1.bias",
    "linear_fc2.weight",
    "linear_fc2.bias",
];

/// The tower's ingest: streamed blocks, resident non-block tensors, and what
/// the FP16 conversion cost.
pub struct VisionRead {
    /// The resident entries: the nine main tensors plus, when the config
    /// declares deepstack indexes, six tensors per merger, all
    /// [`crate::DTYPE_FP16`].
    pub entries: Vec<ResidentEntrySpec>,
    /// The 27 blocks as one `LayerBlobs` at layer 0.
    pub blocks: LayerBlobs,
    /// Bytes per block blob, page-aligned. The model-wide stride and the
    /// layer's own are equal here and always will be, because there is
    /// exactly one layer -- the distinction `model-io` Gotcha 2 draws exists
    /// for a MIXED install and a tower is uniform by construction.
    pub block_stride: u64,
    /// One `(tensor, values that lost bits)` row per lossily-converted tensor.
    /// EMPTY for an F16 source. See `convert_raw_to_fp16`: a BF16 source is
    /// exact in FP16's normal range, so a nonzero row here means values in
    /// the subnormal range, not a wholesale precision loss.
    pub lossy_conversion: Vec<(String, usize)>,
}

/// Whether this walk should ingest a tower at all, given the arch the caller
/// asked for and the names the checkpoint actually carries.
///
/// **BOTH TERMS ARE LOAD-BEARING AND THEY FAIL IN OPPOSITE DIRECTIONS.**
///
/// `arch.vision.is_active()` is the CALLER's request. Every existing caller
/// passes `VisionConfig::NONE`, so every existing walk keeps writing the
/// text-only install it always did, even on a checkpoint whose 333 vision
/// tensors this file now knows how to read. That is what keeps M-V3 from
/// changing a single byte of any install anyone has already built.
///
/// `!vision_bases.is_empty()` is the ARTIFACT's answer, and it is the term
/// that stops a correct request from producing a broken install.
/// `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` declares a full `vision_config` --
/// the same depth 27, hidden 1152, intermediate 4304 tower the `qwen3_5`
/// checkpoints declare, verified against the published file -- and ships NO
/// `vision_tower.` tensors at all. Ingesting on the config alone would write
/// a manifest claiming a tower, and `validate_manifest` would then demand
/// `packed_vision/` files the walk never wrote, so a working install would
/// stop opening.
pub fn vision_should_ingest(arch: &ArchConfig, vision_bases: &[&str]) -> bool {
    arch.vision.is_active() && !vision_bases.is_empty()
}

/// The `ArchConfig` the manifest should DECLARE, given what was ingested.
///
/// Zeroes `vision` when no tower was written, so the manifest states what the
/// install HAS rather than what its architecture supports. That is the MTP
/// head's rule -- whether an install carries a drafter is answered by whether
/// `mtp.fc.weight` is in the resident index, with no flag that could disagree
/// with the bytes -- applied to a component that does have a manifest field.
///
/// Shared by both writers rather than written twice, because the two would
/// then have to agree on a condition, and `crates/repack`'s own history says
/// what happens when a vision-or-drafter arm exists in one writer and not the
/// other: the MTP head shipped a byte-identical HEADLESS install for a
/// release that way.
pub fn vision_arch_for_manifest<'a>(
    arch: &'a ArchConfig,
    ingested: bool,
) -> std::borrow::Cow<'a, ArchConfig> {
    if ingested || !arch.vision.is_active() {
        std::borrow::Cow::Borrowed(arch)
    } else {
        let mut zeroed = arch.clone();
        zeroed.vision = VisionConfig::NONE;
        std::borrow::Cow::Owned(zeroed)
    }
}

/// The shape a vision resident entry records, flattening anything past rank 4.
///
/// **`shape4` TRUNCATES, and `patch_embed.proj.weight` IS RANK 5.** Its shape
/// is `[1152, 2, 16, 16, 3]` -- `(out, T, P_h, P_w, C)` -- so `shape4` would
/// record `(1152, 2, 16, 16)` and drop the channel count, leaving an index
/// whose recorded shape's PRODUCT no longer equals the tensor's element count.
/// The bytes would be intact and every length check would pass; a later reader
/// computing `rows * cols` off the index would simply get a smaller number
/// than the tensor has.
///
/// Flattening the trailing dims is not a workaround for the truncation, it is
/// the shape the tensor is actually USED at. The patch embedding's conv has
/// kernel == stride (`docs/VISION_PHASE0.md`), so it is a plain
/// `[N, C*T*P*P] x [C*T*P*P, out]` GEMM and no convolution kernel exists or is
/// wanted. Recording `(1152, 1536)` states that, and it is the same collapse
/// `crates/vision-io` already performs when it emits patch rows in
/// `(T, P_h, P_w, C)` order -- which is why the tensor needs no permutation.
///
/// Every other vision tensor is rank 1 or 2 and passes through unchanged.
fn resident_shape(shape: &[u64]) -> (u32, u32, u32, u32) {
    if shape.len() <= 4 {
        return shape4(shape);
    }
    let rows = shape[0] as u32;
    let cols: u64 = shape[1..].iter().product();
    (rows, cols as u32, 0, 0)
}

/// Reads the tower, converting every tensor to FP16.
///
/// `vision_bases` comes from `ClassifiedNames::vision_bases` and is in plain
/// lexicographic order, which is NOT block order. That is deliberate and
/// costs nothing: this function addresses blocks by the index parsed out of
/// each name, so where a name sat in the input list cannot affect where its
/// bytes land. Anything under the prefix that is neither a known block role
/// nor a known non-block tensor is REFUSED by name -- the same rule
/// `classify_for_family`'s `Unknown` arm applies one level up, and for the
/// same reason: a publisher who adds a tensor has changed the tower, and
/// silently dropping it would produce an install that loads and computes
/// something else.
pub fn read_vision_entries(
    shards: &Gemma4Shards<'_>,
    vision_bases: &[&str],
    vision: &VisionConfig,
) -> Result<VisionRead, Gemma4Error> {
    let depth = usize::try_from(vision.depth).map_err(|_| {
        Gemma4Error::Config(format!(
            "vision depth {} cannot be represented as a block count",
            vision.depth
        ))
    })?;
    if !SUPPORTED_VISION_DEPTHS.contains(&vision.depth) {
        return Err(Gemma4Error::Config(format!(
            "vision depth {} is unsupported; expected one of {SUPPORTED_VISION_DEPTHS:?}",
            vision.depth
        )));
    }

    // Establish that the artifact actually contains every declared block
    // before allocating the per-block buckets. This pass is intentionally
    // independent of `depth`: malformed metadata cannot make its memory use
    // proportional to an attacker-controlled number.
    let mut present_blocks = vec![false; depth];
    for &name in vision_bases {
        let Some(rest) = name
            .strip_prefix(VISION_PREFIX)
            .and_then(|tail| tail.strip_prefix("blocks."))
        else {
            continue;
        };
        let (index, _) = rest.split_once('.').ok_or_else(|| {
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
        present_blocks[index] = true;
    }
    if let Some(index) = present_blocks.iter().position(|present| !present) {
        return Err(Gemma4Error::MissingTensor(format!(
            "{VISION_PREFIX}blocks.{index}.*"
        )));
    }

    // Bucket every name first, so an unknown one is refused before any bytes
    // move. On the real checkpoint that is 333 header lookups against a
    // multi-GB stream, which is the cheap half of Gotcha 8's rule applied
    // inside a single function.
    let mut block_names = Vec::new();
    block_names.try_reserve_exact(depth).map_err(|e| {
        Gemma4Error::Config(format!("cannot allocate {depth} vision block buckets: {e}"))
    })?;
    for _ in 0..depth {
        let mut roles = Vec::new();
        roles.try_reserve_exact(BLOCK_ROLES.len()).map_err(|e| {
            Gemma4Error::Config(format!("cannot allocate vision block role buckets: {e}"))
        })?;
        roles.resize(BLOCK_ROLES.len(), None);
        block_names.push(roles);
    }
    let mut resident_names: Vec<Option<&str>> = vec![None; RESIDENT_TENSORS.len()];

    // The deepstack mergers, when the config declares any. Bucketed with the
    // same all-slots-or-refuse discipline as the blocks, and the merger count
    // is the INDEX COUNT (`vision.deepstack_visual_indexes.len()`), so a
    // checkpoint shipping `deepstack_merger_list.3.*` against three declared
    // indexes is an unknown name, not a fourth merger.
    let deepstack_len = vision.deepstack_visual_indexes.len();
    let mut deepstack_names: Vec<Vec<Option<&str>>> = Vec::with_capacity(deepstack_len);
    for _ in 0..deepstack_len {
        let mut roles = Vec::new();
        roles
            .try_reserve_exact(DEEPSTACK_MERGER_SUFFIXES.len())
            .map_err(|e| {
                Gemma4Error::Config(format!(
                    "cannot allocate deepstack merger role buckets: {e}"
                ))
            })?;
        roles.resize(DEEPSTACK_MERGER_SUFFIXES.len(), None);
        deepstack_names.push(roles);
    }

    for &name in vision_bases {
        let tail = name.strip_prefix(VISION_PREFIX).ok_or_else(|| {
            Gemma4Error::UnknownTensor(format!("{name} was classified as a vision tensor but does not carry the {VISION_PREFIX} prefix"))
        })?;
        if let Some(rest) = tail.strip_prefix("blocks.") {
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
            let role = BLOCK_ROLES
                .iter()
                .position(|(_, s)| *s == suffix)
                .ok_or_else(|| {
                    Gemma4Error::UnknownTensor(format!(
                        "{name}: {suffix} is not one of this tower's twelve per-block tensors"
                    ))
                })?;
            if block_names[index][role].replace(name).is_some() {
                return Err(Gemma4Error::ShapeMismatch {
                    tensor: name.to_string(),
                    detail: format!("two tensors for block {index} role {suffix}"),
                });
            }
        } else if let Some(rest) = tail.strip_prefix(DEEPSTACK_MERGER_PREFIX) {
            let (index, suffix) = rest.split_once('.').ok_or_else(|| {
                Gemma4Error::UnknownTensor(format!("{name}: no deepstack merger index and suffix"))
            })?;
            let index: usize = index.parse().map_err(|_| {
                Gemma4Error::UnknownTensor(format!(
                    "{name}: deepstack merger index is not a number"
                ))
            })?;
            // An undeclared merger is an UNKNOWN NAME, not an extra one: the
            // runtime injects exactly `deepstack_visual_indexes.len()`
            // mergers, so bytes beyond that would be written, hashed, and
            // never read.
            if index >= deepstack_len {
                return Err(Gemma4Error::UnknownTensor(format!(
                    "{name}: deepstack merger {index} is past the {} the config declares",
                    if deepstack_len == 0 {
                        "zero mergers".to_string()
                    } else {
                        format!("{deepstack_len} mergers")
                    }
                )));
            }
            let role = DEEPSTACK_MERGER_SUFFIXES
                .iter()
                .position(|s| *s == suffix)
                .ok_or_else(|| {
                    Gemma4Error::UnknownTensor(format!(
                        "{name}: {suffix} is not one of a deepstack merger's six tensors"
                    ))
                })?;
            if deepstack_names[index][role].replace(name).is_some() {
                return Err(Gemma4Error::ShapeMismatch {
                    tensor: name.to_string(),
                    detail: format!("two tensors for deepstack merger {index} role {suffix}"),
                });
            }
        } else {
            let slot = RESIDENT_TENSORS
                .iter()
                .position(|t| *t == tail)
                .ok_or_else(|| {
                    Gemma4Error::UnknownTensor(format!(
                        "{name}: not a block tensor and not one of the tower's nine \
                         non-block tensors"
                    ))
                })?;
            if resident_names[slot].replace(name).is_some() {
                return Err(Gemma4Error::ShapeMismatch {
                    tensor: name.to_string(),
                    detail: "duplicate non-block vision tensor".to_string(),
                });
            }
        }
    }

    // Every slot must be filled. A MISSING tensor is refused here rather than
    // at a dispatch, because a zero-filled blob decodes perfectly well: the
    // tower would run and produce an embedding computed with one sublayer
    // silently disabled.
    for (index, roles) in block_names.iter().enumerate() {
        for (role, name) in roles.iter().enumerate() {
            if name.is_none() {
                return Err(Gemma4Error::MissingTensor(format!(
                    "{VISION_PREFIX}blocks.{index}.{}",
                    BLOCK_ROLES[role].1
                )));
            }
        }
    }
    for (slot, name) in resident_names.iter().enumerate() {
        if name.is_none() {
            return Err(Gemma4Error::MissingTensor(format!(
                "{VISION_PREFIX}{}",
                RESIDENT_TENSORS[slot]
            )));
        }
    }
    for (index, roles) in deepstack_names.iter().enumerate() {
        for (role, name) in roles.iter().enumerate() {
            if name.is_none() {
                return Err(Gemma4Error::MissingTensor(format!(
                    "{VISION_PREFIX}{DEEPSTACK_MERGER_PREFIX}{index}.{}",
                    DEEPSTACK_MERGER_SUFFIXES[role]
                )));
            }
        }
    }

    let mut lossy_conversion = Vec::new();

    // The resident nine, renamed onto the install's own prefix.
    let mut entries = Vec::with_capacity(RESIDENT_TENSORS.len());
    for (slot, name) in resident_names.iter().enumerate() {
        let name = name.expect("checked above");
        let t = shards.info(name)?;
        let converted = convert_raw_to_fp16(name, &t.dtype, shards.read(name)?)?;
        if converted.lossy > 0 {
            lossy_conversion.push((name.to_string(), converted.lossy));
        }
        entries.push(ResidentEntrySpec::Raw(RawTensorSpec {
            name: format!("{VISION_INSTALL_PREFIX}{}", RESIDENT_TENSORS[slot]),
            dtype: converted.dtype,
            bytes: converted.bytes,
            shape: resident_shape(&t.shape),
        }));
    }

    // The deepstack mergers, resident behind their own prefix level. Written
    // AFTER the main nine so a hexdump of the resident region reads tower
    // order: the main merger, then the three injections in index order.
    for (index, roles) in deepstack_names.iter().enumerate() {
        for (role, name) in roles.iter().enumerate() {
            let name = name.expect("checked above");
            let t = shards.info(name)?;
            let converted = convert_raw_to_fp16(name, &t.dtype, shards.read(name)?)?;
            if converted.lossy > 0 {
                lossy_conversion.push((name.to_string(), converted.lossy));
            }
            entries.push(ResidentEntrySpec::Raw(RawTensorSpec {
                name: format!(
                    "{VISION_INSTALL_PREFIX}{DEEPSTACK_MERGER_PREFIX}{index}.{}",
                    DEEPSTACK_MERGER_SUFFIXES[role]
                ),
                dtype: converted.dtype,
                bytes: converted.bytes,
                shape: resident_shape(&t.shape),
            }));
        }
    }

    // The blocks. One `ExpertBlob` each, sub-tensors in `BLOCK_ROLES` order.
    let mut experts = Vec::with_capacity(depth);
    let mut widest = 0u64;
    for (index, roles) in block_names.iter().enumerate() {
        let mut sub_tensors = Vec::with_capacity(BLOCK_ROLES.len());
        let mut used = 0u64;
        for (role, name) in roles.iter().enumerate() {
            let name = name.expect("checked above");
            let t = shards.info(name)?;
            let converted = convert_raw_to_fp16(name, &t.dtype, shards.read(name)?)?;
            if converted.lossy > 0 {
                lossy_conversion.push((name.to_string(), converted.lossy));
            }
            used += converted.bytes.len() as u64;
            sub_tensors.push(SubTensor {
                role: BLOCK_ROLES[role].0.to_string(),
                bytes: converted.bytes,
                dtype: "fp16".to_string(),
                shape: t.shape.clone(),
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
    use std::collections::BTreeMap;

    use crate::ranged_download::MemoryRangeSource;
    use crate::safetensors_header::SafetensorsHeader;

    use super::*;

    #[test]
    fn direct_reader_rejects_unbounded_depth_without_allocating() {
        let header = SafetensorsHeader {
            tensors: BTreeMap::new(),
            metadata: None,
            header_len: 0,
        };
        let source = MemoryRangeSource::new(&[]);
        let shards = Gemma4Shards::single(&header, &source);
        let mut vision = VisionConfig::NONE;
        vision.depth = i64::MAX;

        let error = read_vision_entries(&shards, &["vision_tower.blocks.0.norm1.weight"], &vision)
            .err()
            .expect("the public reader must defend against a manually built config");
        assert_eq!(
            error.to_string(),
            format!(
                "config.json invalid: vision depth {} is unsupported; expected one of [24, 27]",
                i64::MAX
            )
        );
    }
}
