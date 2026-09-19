//! The vision-tower SIDECAR install format (vision memory sidecar, part A1):
//! a tower installed once as its own small `<alias>.gturbo-vision/`
//! directory, distinct from a full trunk install and attachable to any
//! compatible text-only trunk at runtime (a later part -- this module is
//! format only, no binding).
//!
//! A sidecar directory carries the same `manifest.json` / `model_weights.bin`
//! / `packed_experts/layout.json` / `packed_vision/` shape a full install
//! does, at `numLayers: 0` and `hiddenSize` set to the tower's OWN output
//! width (`VisionConfig::out_hidden_size`) rather than to any text trunk's.
//! That is what lets it load through the same [`crate::manifest::load`]
//! every other install goes through, with no format fork.
//!
//! **The manifest cannot say "this is a tower, not a model", and that is
//! why [`SidecarRecord`] exists as a separate file.** `load_manifest` refuses
//! an unknown `manifest.flags` key (`crate::manifest::known_flags`), and
//! `is_production_arch` keys off `(num_layers, hidden_size)` alone, so there
//! is no manifest-native place to stamp a "kind" marker without inventing a
//! new flag every OTHER loader would then have to ignore. `vision_sidecar.json`
//! carries what the manifest structurally cannot: the marker itself
//! (`kind`), which text family and hidden size this tower is meant to pair
//! with (`pairs_with`, cross-checked against the manifest's own declared
//! `visionOutHiddenSize` on every [`load`]), and where the tower's bytes came
//! from (`source`).

use std::io::Read;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::arch_config::{ArchConfig, ModelFamily, VisionConfig};
use crate::error::ModelError;

/// The record file's basename, sitting beside `manifest.json` in a sidecar
/// directory.
pub const SIDECAR_RECORD_FILE: &str = "vision_sidecar.json";

/// [`SidecarRecord::kind`]'s only valid value today. A distinct constant
/// rather than a bare string literal at each call site, so a typo in one
/// arm cannot silently stop matching the other.
pub const SIDECAR_KIND: &str = "vision-tower";

/// Maximum size accepted for `vision_sidecar.json`.
const SIDECAR_RECORD_MAX_BYTES: u64 = 64 * 1024;

fn read_bounded(path: &Path, name: &str, max_bytes: u64) -> Result<Vec<u8>, ModelError> {
    let file = std::fs::File::open(path).map_err(|e| ModelError::IoFailed {
        call: "read".to_string(),
        detail: format!("{}: {e}", path.display()),
    })?;
    let mut data = Vec::new();
    file.take(max_bytes + 1)
        .read_to_end(&mut data)
        .map_err(|e| ModelError::IoFailed {
            call: "read".to_string(),
            detail: format!("{}: {e}", path.display()),
        })?;
    if data.len() as u64 > max_bytes {
        return Err(ModelError::IndexCorrupt {
            detail: format!("{name} size exceeds metadata cap {max_bytes}"),
        });
    }
    Ok(data)
}

/// Which text family and hidden size this tower's merger was built to feed.
///
/// Not validated against anything at write time -- a sidecar is written once
/// per source checkpoint's tower and can outlive any particular trunk
/// install -- but checked at [`load`] against the manifest's OWN
/// `visionOutHiddenSize`, because a record and a manifest that disagree
/// about the tower's own output width is a corrupted or hand-edited
/// directory, not a compatibility question for a later part to answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PairsWith {
    /// The text family's wire string (`ModelFamily::as_str`), e.g. `"qwen35"`.
    pub family: String,
    /// The trunk's `hidden_size`, which the merger's output width must equal
    /// for the tower's rows to land in the right place in the residual
    /// stream.
    pub hidden_size: i64,
}

/// Where the tower's bytes came from, for provenance and for re-deriving the
/// sidecar without re-reading the whole trunk checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SidecarSource {
    /// The Hugging Face repository the tower was read from.
    pub repo: String,
    /// The pinned revision (commit hash), never `"main"` -- see
    /// `crates/repack` Gotcha 19 for why an unpinned tower fetch is a silent
    /// hazard the moment two checkpoints' towers diverge.
    pub revision: String,
    /// The tensor-name prefix the tower carried in the SOURCE checkpoint
    /// before canonicalization (`"vision_tower."` or `"model.visual."`).
    pub prefix: String,
    /// The specific file (or shard) within the repository the tower's
    /// tensors were read from.
    pub file: String,
}

/// The sidecar's compatibility record: `vision_sidecar.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SidecarRecord {
    /// Always [`SIDECAR_KIND`] for a directory this module wrote. Read back
    /// as plain data by [`is_sidecar_dir`] and [`load`] rather than as an
    /// enum, so a FUTURE second kind (should one ever exist) does not need
    /// this crate's cooperation to be rejected by name.
    pub kind: String,
    /// The text family and hidden size this tower is meant to pair with.
    pub pairs_with: PairsWith,
    /// Where the tower's bytes came from.
    pub source: SidecarSource,
    /// Transformer blocks in the tower (`VisionConfig::depth`), restated here
    /// as plain metadata a reader can see without decoding
    /// `packed_vision/layout.json`.
    pub tower_blocks: i64,
    /// Bytes per block blob, page-aligned -- `VisionRead::block_stride` at
    /// write time, restated for the same reason as `tower_blocks`.
    pub block_stride: u64,
}

impl SidecarRecord {
    /// Writes `vision_sidecar.json` into `dir`.
    pub fn write(&self, dir: &Path) -> Result<(), ModelError> {
        let bytes = serde_json::to_vec_pretty(self).map_err(|e| ModelError::IndexCorrupt {
            detail: format!("vision_sidecar.json: {e}"),
        })?;
        let path = dir.join(SIDECAR_RECORD_FILE);
        std::fs::write(&path, bytes).map_err(|e| ModelError::IoFailed {
            call: "write".to_string(),
            detail: format!("{}: {e}", path.display()),
        })
    }

    /// Reads `vision_sidecar.json` back out of `dir`, with no manifest
    /// cross-check -- see [`load`] for the full, validated read.
    pub fn read(dir: &Path) -> Result<Self, ModelError> {
        let path = dir.join(SIDECAR_RECORD_FILE);
        let data = read_bounded(&path, SIDECAR_RECORD_FILE, SIDECAR_RECORD_MAX_BYTES)?;
        serde_json::from_slice(&data).map_err(|e| ModelError::IndexCorrupt {
            detail: format!("vision_sidecar.json: {e}"),
        })
    }
}

/// The `ArchConfig` a sidecar's manifest declares and validates against:
/// `known_architecture(family)` with `num_layers` zeroed, `hidden_size` set
/// to the tower's own output width, `full_attention_layer_mask` emptied to
/// match (a non-empty mask paired with zero layers is a manifest describing
/// two different layer counts to two different readers), and every vision
/// field taken from `vision` rather than from the baseline's `NONE`.
///
/// **This is the ONE place both the writer (`crates/repack`'s
/// `write_vision_sidecar`) and the reader ([`load`]) construct this arch**,
/// so a future change to either cannot drift the other out of agreement --
/// the same reason `vision_arch_for_manifest` is shared by both of
/// `crates/repack`'s full-install writers rather than written twice.
pub fn sidecar_arch(family: ModelFamily, hidden_size: i64, vision: &VisionConfig) -> ArchConfig {
    let mut arch = crate::arch_baselines::known_architecture(family);
    arch.num_layers = 0;
    arch.hidden_size = hidden_size;
    arch.full_attention_layer_mask = Vec::new();
    arch.vision = vision.clone();
    arch
}

/// True iff `dir` carries a `vision_sidecar.json` naming [`SIDECAR_KIND`].
///
/// This is the whole of A1's "is this a sidecar" answer. Teaching whatever
/// normally opens a full model to REFUSE a sidecar directory (so a caller
/// cannot accidentally try to run one as a trunk) is runtime wiring for a
/// later part, not a format question this function can settle alone.
pub fn is_sidecar_dir(dir: &Path) -> bool {
    match SidecarRecord::read(dir) {
        Ok(record) => record.kind == SIDECAR_KIND,
        Err(_) => false,
    }
}

/// Reads and fully validates a sidecar directory: `vision_sidecar.json`
/// parses and declares [`SIDECAR_KIND`] under a known family, the arch it
/// implies rebuilds through [`sidecar_arch`] and passes the real manifest
/// loader/validator (`crate::manifest::load`) against `manifest.json` in
/// `dir`, and the record's declared pairing hidden size agrees with what the
/// manifest itself declares for the tower's output width.
///
/// The record is read before the manifest's vision fields are known, so the
/// two are checked in an order that has to peek the manifest's `arch` object
/// once before `sidecar_arch` can be built (the same reason
/// `crate::manifest::peek_family` exists) and then run the whole thing
/// through the real loader for everything else `manifest.json` needs to get
/// right (magic, version, quant, file declarations, ...).
pub fn load(dir: &Path) -> Result<(SidecarRecord, VisionConfig), ModelError> {
    let record = SidecarRecord::read(dir)?;
    if record.kind != SIDECAR_KIND {
        return Err(ModelError::IndexCorrupt {
            detail: format!(
                "vision_sidecar.json kind {:?} is not {SIDECAR_KIND:?}",
                record.kind
            ),
        });
    }
    let family =
        ModelFamily::parse(&record.pairs_with.family).ok_or_else(|| ModelError::IndexCorrupt {
            detail: format!(
                "vision_sidecar.json pairsWith.family {:?} is not a known family",
                record.pairs_with.family
            ),
        })?;

    // Peek `manifest.json`'s `arch` object directly, the same way
    // `crate::manifest::peek_family` does, because the vision fields it
    // carries are what `sidecar_arch` needs BEFORE the full validated load
    // below has anything to validate against.
    let manifest_path = dir.join("manifest.json");
    let data = read_bounded(
        &manifest_path,
        "manifest.json",
        crate::manifest::DEFAULT_MAX_BYTES,
    )?;
    let raw: serde_json::Value =
        serde_json::from_slice(&data).map_err(|e| ModelError::IndexCorrupt {
            detail: format!("manifest.json: {e}"),
        })?;
    let manifest_arch: crate::manifest::ManifestArch = serde_json::from_value(raw["arch"].clone())
        .map_err(|e| ModelError::IndexCorrupt {
            detail: format!("manifest.json arch: {e}"),
        })?;
    let vision = manifest_arch.vision_config();

    if record.pairs_with.hidden_size != vision.out_hidden_size {
        return Err(ModelError::IndexCorrupt {
            detail: format!(
                "vision_sidecar.json pairsWith.hiddenSize {} does not match manifest.json's \
                 visionOutHiddenSize {}",
                record.pairs_with.hidden_size, vision.out_hidden_size
            ),
        });
    }

    let expected = sidecar_arch(family, vision.out_hidden_size, &vision);
    crate::manifest::load(dir, &expected, crate::manifest::DEFAULT_MAX_BYTES)?;

    Ok((record, vision))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_vision() -> VisionConfig {
        VisionConfig {
            depth: 2,
            hidden_size: 64,
            intermediate_size: 96,
            num_heads: 4,
            patch_size: 4,
            temporal_patch_size: 2,
            in_channels: 3,
            spatial_merge_size: 2,
            num_position_embeddings: 16,
            out_hidden_size: 5120,
            mrope_section: [11, 11, 10],
            vision_start_token_id: 248_053,
            vision_end_token_id: 248_054,
            image_token_id: 248_056,
            video_token_id: 248_057,
            deepstack_visual_indexes: Vec::new(),
        }
    }

    #[test]
    fn sidecar_arch_zeroes_layers_and_carries_the_tower() {
        let vision = tiny_vision();
        let arch = sidecar_arch(ModelFamily::QwenGdnDense, vision.out_hidden_size, &vision);
        assert_eq!(arch.num_layers, 0);
        assert_eq!(arch.hidden_size, vision.out_hidden_size);
        assert!(arch.full_attention_layer_mask.is_empty());
        assert_eq!(arch.vision, vision);
        assert_eq!(arch.family, ModelFamily::QwenGdnDense);
    }

    /// The one place both the writer and the reader build this arch: calling
    /// it twice with the same inputs must agree, or a later drift between
    /// the two call sites would be invisible to anything but a full
    /// round-trip test.
    #[test]
    fn sidecar_arch_is_deterministic() {
        let vision = tiny_vision();
        let a = sidecar_arch(ModelFamily::QwenGdnMoe, vision.out_hidden_size, &vision);
        let b = sidecar_arch(ModelFamily::QwenGdnMoe, vision.out_hidden_size, &vision);
        assert_eq!(a, b);
    }

    #[test]
    fn is_sidecar_dir_is_false_for_a_missing_record() {
        let dir = std::env::temp_dir().join(format!(
            "turbospark-model-io-vision-sidecar-missing-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!is_sidecar_dir(&dir));
    }
}
