//! Orchestration, classification planning, and resident reading.

use std::collections::BTreeMap;

use model_io::ArchConfig;

use super::classify::{classify_for_family, lm_order_key, Gemma4Bucket};
use super::config::{Gemma4Error, Gemma4Quant};
use super::expert_blobs::{expert_stride_from_headers, plan_one_expert_layer};
use super::narrow::narrow_raw_to_bf16;
use super::shards::{shape4, Gemma4Shards};
use super::vision::VisionRead;
use crate::gturbo_writer::LayerBlobs;
use crate::ranged_download::RangeSource;
use crate::resident_writer::{RawTensorSpec, ResidentEntrySpec};
use crate::safetensors_header::SafetensorsHeader;

/// Everything the writer needs for a full Gemma 4 `.gturbo` install:
/// ordered resident entries (pass-through quantized + raw), per-layer
/// routed-expert blobs, and the one model-wide page-rounded expert stride
/// (0 when the checkpoint has no routed experts).
pub struct Gemma4RepackOutput {
    pub resident: Vec<ResidentEntrySpec>,
    pub layers: Vec<LayerBlobs>,
    pub expert_stride: u64,
    pub excluded_multimodal: Vec<String>,
    /// One `(tensor, values that lost bits)` row per unquantized tensor this
    /// walk had to narrow to BF16. See `narrow_raw_to_bf16`.
    pub lossy_narrowing: Vec<(String, usize)>,
    /// The vision tower's packed blocks, when this walk ingested one
    /// (ROADMAP M-V3). `None` on every text-only walk, which is every caller
    /// that passes `VisionConfig::NONE`.
    ///
    /// Its RESIDENT entries are not here: they are already in `resident`,
    /// appended where the MTP head's are. Carrying them twice would invite a
    /// writer to append them a second time, and a duplicated name in the
    /// resident index is not something the reader would notice.
    pub vision: Option<VisionRead>,
}

/// Walks a Gemma 4 checkpoint's tensors: classifies every name, orders and
/// reads the resident LM set (pass-through for `U32` `.weight` tensors with
/// their BF16 companions, raw bytes for everything else), and slices each
/// layer's routed-expert `gate/up/down` bundles into per-expert blobs.
pub fn orchestrate_gemma4_checkpoint(
    header: &SafetensorsHeader,
    source: &dyn RangeSource,
    arch: &ArchConfig,
    quant: &Gemma4Quant,
) -> Result<Gemma4RepackOutput, Gemma4Error> {
    orchestrate_gemma4_checkpoint_sharded(&Gemma4Shards::single(header, source), arch, quant)
}

/// Multi-shard variant of [`orchestrate_gemma4_checkpoint`] -- what a real
/// (three-shard) checkpoint goes through.
pub fn orchestrate_gemma4_checkpoint_sharded(
    shards: &Gemma4Shards<'_>,
    arch: &ArchConfig,
    quant: &Gemma4Quant,
) -> Result<Gemma4RepackOutput, Gemma4Error> {
    let plan = classify_all(shards, arch)?;
    let mut resident = read_resident_entries(shards, &plan.resident_bases, quant)?;
    // The head is APPENDED, after `lm_head` and after the trunk's own
    // ordering has been settled. Absent from the checkpoint means absent from
    // the install, with no flag and no manifest field to disagree with the
    // bytes: whether an install has a drafter is answered by whether
    // `mtp.fc.weight` is in the resident index.
    if !plan.mtp_bases.is_empty() {
        let head = super::mtp::read_mtp_entries(shards, &plan.mtp_bases)?;
        resident.entries.extend(head.entries);
        resident.lossy_narrowing.extend(head.lossy_narrowing);
    }
    // The DFlash2 drafter, APPENDED after the head for the same reason the
    // head is appended after the trunk: it is a separate model sharing one
    // resident region, and its names keep to their own `dflash.` group
    // rather than interleaving with any layer's.
    if !plan.dflash_bases.is_empty() {
        let drafter = super::dflash::read_dflash_entries(shards, &plan.dflash_bases)?;
        resident.entries.extend(drafter.entries);
        resident.lossy_narrowing.extend(drafter.lossy_narrowing);
    }
    // THE VISION TOWER (ROADMAP M-V3). This is the NON-streamed writer's arm
    // and `write_gemma4_install_streamed` carries its own; neither is a
    // fallback for the other. See that one's comment for why an arm in one
    // writer alone is a hole the fixtures cannot see.
    //
    // Appended after the drafters for their reason: it is a separate model
    // sharing one resident region, and its names keep to their own `vision.`
    // group rather than interleaving with any trunk layer's.
    let vision = if super::vision::vision_should_ingest(arch, &plan.vision_bases) {
        let mut tower =
            super::vision::read_vision_entries(shards, &plan.vision_bases, &arch.vision)?;
        resident.entries.extend(std::mem::take(&mut tower.entries));
        Some(tower)
    } else {
        None
    };
    let expert_stride = expert_stride_from_headers(shards, arch, quant, &plan.routed)?;
    let mut layers = Vec::new();
    if !plan.routed.is_empty() {
        for layer in 0..arch.num_layers as usize {
            let (blobs, _used) = plan_one_expert_layer(shards, arch, quant, &plan.routed, layer)?;
            layers.push(blobs);
        }
    }
    Ok(Gemma4RepackOutput {
        resident: resident.entries,
        layers,
        expert_stride,
        excluded_multimodal: plan.excluded,
        lossy_narrowing: resident.lossy_narrowing,
        vision,
    })
}

/// Classified tensor names partitioned by structural role.
pub struct ClassifiedNames<'a> {
    /// Base names for resident language-model weights.
    pub resident_bases: Vec<&'a str>,
    /// Routed expert tensors grouped by layer index and role.
    pub routed: BTreeMap<usize, BTreeMap<&'static str, &'a str>>,
    /// Multimodal or non-LM tensor names excluded from the repack.
    pub excluded: Vec<String>,
    /// The multi-token-prediction head's tensors, kept OUT of
    /// `resident_bases` rather than merged into it.
    ///
    /// Two reasons, and the second is the one that bites. The head takes a
    /// quantizing arm no trunk tensor takes (`mtp::read_mtp_entries`), so
    /// merging would mean `read_resident_entries` deciding per tensor which
    /// of three paths a name wants. And `lm_order_key` sorts on
    /// `layer_index`, which finds `.layers.` inside `mtp.layers.0.*` and
    /// would interleave the head's block with TRUNK LAYER 0's tensors --
    /// harmless for a name-keyed index, but it puts an unrelated model's
    /// weights in the middle of a layer group for every future reader.
    pub mtp_bases: Vec<&'a str>,
    /// The DFlash2 drafter's tensors, kept out of `resident_bases` for the
    /// head's two reasons plus a third of its own: the drafter also has
    /// rank-3 tensors (`base_kernel`), which no trunk arm accepts at all.
    pub dflash_bases: Vec<&'a str>,
    /// The vision tower's tensors (ROADMAP M-V3), kept out of
    /// `resident_bases` for the head's two reasons and a third that is
    /// sharper here than for either drafter.
    ///
    /// The head's first reason: the tower takes a dtype arm no trunk tensor
    /// takes. It stays FP16 verbatim where every unquantized trunk tensor is
    /// narrowed to BF16 (`narrow_raw_to_bf16`, AGENTS.md Gotcha 45), so
    /// merging would put a third path inside `read_resident_entries`.
    ///
    /// The head's second reason, and the one that would actually corrupt the
    /// index: `lm_order_key` sorts on `layer_index`, which finds `.layers.`
    /// inside a name -- and `vision_tower.blocks.N.*` would sort into the
    /// TRUNK's layer groups. Twenty-seven blocks interleaved through sixty-four
    /// trunk layers is not a wrong byte anywhere, just an index no later reader
    /// can make sense of.
    ///
    /// The third is this bucket's own: most of these tensors do not become
    /// resident entries at all. The 27 blocks are packed into
    /// `packed_vision/blobs.bin` and streamed; only `patch_embed.*`,
    /// `pos_embed` and `merger.*` land in the index. So the split is not a
    /// tidiness preference here, it is two different destinations.
    pub vision_bases: Vec<&'a str>,
}

/// Classifies all tensor names in the shards into their respective roles.
pub fn classify_all<'a>(
    shards: &Gemma4Shards<'a>,
    arch: &ArchConfig,
) -> Result<ClassifiedNames<'a>, Gemma4Error> {
    let num_layers = arch.num_layers as usize;
    let mut resident_bases: Vec<&str> = Vec::new();
    let mut excluded: Vec<String> = Vec::new();
    let mut mtp_bases: Vec<&str> = Vec::new();
    let mut dflash_bases: Vec<&str> = Vec::new();
    let mut vision_bases: Vec<&str> = Vec::new();
    let mut routed: BTreeMap<usize, BTreeMap<&'static str, &str>> = BTreeMap::new();

    for name in shards.names() {
        if name.ends_with(".scales") || name.ends_with(".biases") {
            continue;
        }
        match classify_for_family(name, num_layers, arch.family) {
            Gemma4Bucket::LmResident => resident_bases.push(name),
            Gemma4Bucket::RoutedExpert { role, layer } => {
                if routed
                    .entry(layer)
                    .or_default()
                    .insert(role, name.as_str())
                    .is_some()
                {
                    return Err(Gemma4Error::ShapeMismatch {
                        tensor: name.to_string(),
                        detail: format!("two routed-expert tensors for layer {layer} role {role}"),
                    });
                }
            }
            Gemma4Bucket::ExcludedMultimodal => excluded.push(name.clone()),
            Gemma4Bucket::MtpHead => mtp_bases.push(name),
            Gemma4Bucket::DflashDrafter => dflash_bases.push(name),
            Gemma4Bucket::VisionTower => vision_bases.push(name),
            // **REFUSED BY NAME, NOT DROPPED AND NOT FILED RESIDENT.** The
            // n-gram store's writer does not exist yet, and the two wrong
            // answers are both silent: dropping these writes an install whose
            // layer 1 has no table, and letting them fall to `LmResident`
            // puts 32 GB into `model_weights.bin`. Refusing means the walk
            // stops at classification, seconds in, rather than after an hour
            // of streaming (`crates/repack` Gotcha 8's fixture-first rule
            // applied to a component that has a classifier and no
            // destination).
            Gemma4Bucket::NgramShard { .. } | Gemma4Bucket::NgramMeta { .. } => {
                return Err(Gemma4Error::UnsupportedTensor {
                    tensor: name.clone(),
                    detail: "qwen4_exp's hashed n-gram PLE table has no store writer yet; it \
                             is 30.8% of the checkpoint and cannot go in the resident index"
                        .to_string(),
                })
            }
            Gemma4Bucket::Unknown => return Err(Gemma4Error::UnknownTensor(name.clone())),
        }
    }
    resident_bases.sort_by(|a, b| lm_order_key(a).cmp(&lm_order_key(b)));
    excluded.sort();
    // Plain lexicographic, which puts `mtp.fc` and the three structural norms
    // ahead of `mtp.layers.0.*`. There is exactly one block, so no
    // layer-aware ordering is owed; `lm_order_key` is deliberately not reused
    // here (see `ClassifiedNames::mtp_bases`).
    mtp_bases.sort_unstable();
    // Plain lexicographic again: one name group, and the drafter's own
    // `layers.N` ordering inside it is lexicographic's natural order.
    dflash_bases.sort_unstable();
    // Plain lexicographic, deliberately. It is NOT the tower's block order --
    // `blocks.10.*` sorts before `blocks.2.*` -- and that is fine because
    // nothing downstream reads this order: `vision::read_vision_entries`
    // indexes blocks by parsing the number out of each name, so a block's
    // position in the packed file is a function of its INDEX rather than of
    // where it landed here. Reusing `lm_order_key` would be worse than
    // useless, since it is the function that would misfile these names in the
    // first place (see `ClassifiedNames::vision_bases`).
    vision_bases.sort_unstable();
    Ok(ClassifiedNames {
        resident_bases,
        routed,
        excluded,
        mtp_bases,
        dflash_bases,
        vision_bases,
    })
}

/// The resident set plus what narrowing it to BF16 cost.
pub struct ResidentRead {
    pub entries: Vec<ResidentEntrySpec>,
    /// One `(tensor, values that lost bits)` row per lossily-narrowed tensor,
    /// mirroring `GgufRepackOutput::lossy_narrowing`. Empty for every
    /// BF16-source checkpoint, which is every one but Bonsai-27B.
    pub lossy_narrowing: Vec<(String, usize)>,
}

/// Reads resident weight and norm entries for the classified base names.
pub fn read_resident_entries(
    shards: &Gemma4Shards<'_>,
    resident_bases: &[&str],
    quant: &Gemma4Quant,
) -> Result<ResidentRead, Gemma4Error> {
    let mut entries = Vec::with_capacity(resident_bases.len());
    let mut lossy_narrowing = Vec::new();
    for &name in resident_bases {
        let t = shards.info(name)?;
        if t.dtype == "U32" && name.ends_with(".weight") {
            entries.push(super::narrow::pass_through_packed(shards, name, quant)?);
        } else {
            // NARROWED, not tagged. The deleted `raw_dtype_tag` recorded F16
            // or F32 and nothing downstream reads either tag: every consumer
            // of an unquantized tensor decodes it as BF16 off its byte size,
            // so an F16 norm written verbatim is misread rather than
            // refused. See `narrow_raw_to_bf16` for the measurement behind
            // accepting the loss.
            let narrowed = narrow_raw_to_bf16(name, &t.dtype, shards.read(name)?)?;
            if narrowed.lossy > 0 {
                lossy_narrowing.push((name.to_string(), narrowed.lossy));
            }
            entries.push(ResidentEntrySpec::Raw(RawTensorSpec {
                name: name.to_string(),
                dtype: narrowed.dtype,
                bytes: narrowed.bytes,
                shape: shape4(&t.shape),
            }));
        }
    }
    Ok(ResidentRead {
        entries,
        lossy_narrowing,
    })
}
