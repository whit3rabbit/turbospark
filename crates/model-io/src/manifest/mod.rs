//! `manifest.json` decode and validation against a resolved [`ArchConfig`].
//! Ported from `Infrastructure/ModelIO/ManifestReader.swift`.

mod quant;
mod types;

use std::collections::HashSet;
use std::path::Path;

pub(crate) use quant::validate_quant;
pub use quant::EXECUTABLE_GGUF_TYPES;
pub use types::{
    Manifest, ManifestArch, ManifestFileEntry, ManifestHadamard, ManifestHadamardSigns,
    ManifestProducer, ManifestQuant, ManifestQuantSlot,
};

use crate::arch_baselines::all_known_architectures;
use crate::arch_config::{ArchConfig, ModelFamily};
use crate::bounded_read::read_capped;
use crate::error::ModelError;

/// Default maximum byte limit for reading `manifest.json`.
pub const DEFAULT_MAX_BYTES: u64 = 4 * 1024 * 1024;

/// The manifest format the writer emits and this loader reads. The major
/// changes when an older reader would misparse the install; the minor
/// changes when the layout gains something an older reader can skip. The
/// loader accepts any minor up to [`MANIFEST_VERSION_MINOR`] and refuses a
/// newer one, so an install written by a later turbospark is refused by
/// name instead of being read with the fields it does not know ignored.
pub const MANIFEST_VERSION_MAJOR: i64 = 1;
/// Minor 1: `model_weights.bin`'s index region ends with the format
/// trailer (`resident_index::INDEX_MAGIC`). Minor 0 installs have none and
/// still load.
pub const MANIFEST_VERSION_MINOR: i64 = 1;

/// Required file entries (relative to `model.gturbo/`).
pub const REQUIRED_FILES: [&str; 2] = ["model_weights.bin", "packed_experts/layout.json"];

/// Recognized flag keys. Anything else in `manifest.flags` is an error.
pub fn known_flags() -> HashSet<&'static str> {
    ["streamingPresent", "turboQuantKV", "aneSharedExpert"]
        .into_iter()
        .collect()
}

pub fn load(dir: &Path, expecting: &ArchConfig, max_bytes: u64) -> Result<Manifest, ModelError> {
    let manifest_path = dir.join("manifest.json");
    if !manifest_path.exists() {
        return Err(ModelError::PartialInstall {
            path: dir.display().to_string(),
        });
    }
    let size = file_size(&manifest_path)?;
    if size > max_bytes {
        return Err(ModelError::IndexCorrupt {
            detail: format!("manifest.json size {size} exceeds metadata cap {max_bytes}"),
        });
    }
    let data = read_capped(&manifest_path, max_bytes)
        .map_err(|e| ModelError::IoFailed {
            call: "read".to_string(),
            detail: e.to_string(),
        })?
        .ok_or_else(|| ModelError::IndexCorrupt {
            detail: format!("manifest.json size exceeds metadata cap {max_bytes}"),
        })?;
    let raw: serde_json::Value =
        serde_json::from_slice(&data).map_err(|e| ModelError::IndexCorrupt {
            detail: format!("manifest.json: {e}"),
        })?;
    if raw.get("capability").and_then(serde_json::Value::as_str) == Some("image-generation") {
        return Err(ModelError::UnsupportedCapability {
            capability: "image-generation".to_string(),
        });
    }
    let manifest: Manifest = serde_json::from_value(raw).map_err(|e| ModelError::IndexCorrupt {
        detail: format!("manifest.json: {e}"),
    })?;
    validate(&manifest, expecting)?;
    Ok(manifest)
}

fn file_size(path: &Path) -> Result<u64, ModelError> {
    std::fs::metadata(path)
        .map(|m| m.len())
        .map_err(|e| ModelError::IoFailed {
            call: "stat".to_string(),
            detail: e.to_string(),
        })
}

pub fn validate(m: &Manifest, expected: &ArchConfig) -> Result<(), ModelError> {
    if m.magic != "GTURBO" {
        return Err(ModelError::NotAGTurboDirectory);
    }
    if m.version_major != MANIFEST_VERSION_MAJOR || m.version_minor > MANIFEST_VERSION_MINOR {
        return Err(ModelError::UnsupportedVersion {
            major: m.version_major,
            minor: m.version_minor,
        });
    }
    let known = known_flags();
    for key in m.flags.keys() {
        if !known.contains(key.as_str()) {
            return Err(ModelError::UnknownFlag { name: key.clone() });
        }
    }
    if m.flags.get("turboQuantKV") == Some(&true) {
        return Err(ModelError::IndexCorrupt {
            detail: "manifest.flags.turboQuantKV is not a real flag this port reads; TurboQuant \
                     KV-cache quantization is controlled by --kv-bits at open, not by a \
                     manifest flag"
                .to_string(),
        });
    }
    crate::arch_validation::validate_arch(&m.arch, expected)?;
    if let Some(quant) = &m.quant {
        validate_quant(quant)?;
    } else if is_production_arch(expected) {
        return Err(ModelError::IndexCorrupt {
            detail: "manifest.quant is required for the production architecture".to_string(),
        });
    }
    let page_size = page_size_bytes();
    if m.expert_stride % page_size != 0 {
        return Err(ModelError::ExpertStrideNotPageAligned {
            stride: m.expert_stride,
            page_size,
        });
    }
    for f in REQUIRED_FILES {
        if !m.files.contains_key(f) {
            return Err(ModelError::MissingFile {
                name: f.to_string(),
            });
        }
    }
    for layer in 0..m.num_layers {
        let padded = format!("packed_experts/layer_{layer:02}.bin");
        let plain = format!("packed_experts/layer_{layer}.bin");
        if !m.files.contains_key(&padded) && !m.files.contains_key(&plain) {
            return Err(ModelError::MissingFile { name: padded });
        }
    }
    // The vision tower's two files, required only when the resolved arch says
    // there IS a tower (ROADMAP M-V3). Gated on `expected` rather than added
    // to `REQUIRED_FILES`, because that list applies to every install ever
    // written and none before M-V3 carries these -- the same reason the
    // per-layer expert files above are generated from `num_layers` instead of
    // being listed.
    if expected.vision.is_active() {
        for f in VISION_FILES {
            if !m.files.contains_key(f) {
                return Err(ModelError::MissingFile {
                    name: f.to_string(),
                });
            }
        }
    }
    if let Some(h) = &m.hadamard {
        let file = m
            .files
            .get("hadamard.bin")
            .ok_or_else(|| ModelError::MissingFile {
                name: "hadamard.bin".to_string(),
            })?;
        validate_hadamard(h, expected, file.size)?;
    }
    Ok(())
}

/// Light structural checks on the Hadamard section. The runtime re-checks
/// what it consumes (every folded width resolves to a sign vector, the
/// butterfly fits the threadgroup); this catches a hand-edited or truncated
/// section before the weights are mapped.
fn validate_hadamard(
    h: &ManifestHadamard,
    expected: &ArchConfig,
    file_bytes: u64,
) -> Result<(), ModelError> {
    let block = h.block;
    if block <= 0 || (block & (block - 1)) != 0 || block > 4096 {
        return Err(ModelError::IndexCorrupt {
            detail: format!(
                "manifest.hadamard.block {block} is not a butterfly width this port compiles \
                 (power of two, 512..=4096 on the real contract)"
            ),
        });
    }
    if h.signs.is_empty() {
        return Err(ModelError::IndexCorrupt {
            detail: "manifest.hadamard.signs is empty".to_string(),
        });
    }
    let mut allowed_widths = HashSet::new();
    for width in [
        expected.hidden_size,
        expected.intermediate_size,
        expected
            .num_heads
            .checked_mul(expected.full_head_dim)
            .unwrap_or(0),
        expected
            .linear_attention
            .num_v_heads
            .checked_mul(expected.linear_attention.value_head_dim)
            .unwrap_or(0),
    ] {
        if width > 0 {
            allowed_widths.insert(width);
        }
    }
    let mut seen_widths = HashSet::new();
    let mut ranges = Vec::with_capacity(h.signs.len());
    let mut aggregate_bytes = 0_u64;
    for s in &h.signs {
        if s.width <= 0 || s.width % block != 0 {
            return Err(ModelError::IndexCorrupt {
                detail: format!(
                    "manifest.hadamard sign width {} is not a positive multiple of block {block}",
                    s.width
                ),
            });
        }
        let expected_bytes = u64::try_from(s.width)
            .ok()
            .and_then(|width| width.checked_mul(4));
        if expected_bytes != Some(s.bytes) {
            return Err(ModelError::IndexCorrupt {
                detail: format!(
                    "manifest.hadamard sign width {} declares {} bytes; F32 signs are width*4",
                    s.width, s.bytes
                ),
            });
        }
        if !allowed_widths.contains(&s.width) {
            return Err(ModelError::IndexCorrupt {
                detail: format!(
                    "manifest.hadamard sign width {} is not an activation width in this architecture",
                    s.width
                ),
            });
        }
        if !seen_widths.insert(s.width) {
            return Err(ModelError::IndexCorrupt {
                detail: format!("manifest.hadamard repeats sign width {}", s.width),
            });
        }
        let end = s
            .offset
            .checked_add(s.bytes)
            .ok_or_else(|| ModelError::IndexCorrupt {
                detail: "manifest.hadamard sign range overflows u64".to_string(),
            })?;
        if end > file_bytes {
            return Err(ModelError::IndexCorrupt {
                detail: format!(
                    "manifest.hadamard sign range [{}..{end}] exceeds hadamard.bin size {file_bytes}",
                    s.offset
                ),
            });
        }
        aggregate_bytes =
            aggregate_bytes
                .checked_add(s.bytes)
                .ok_or_else(|| ModelError::IndexCorrupt {
                    detail: "manifest.hadamard aggregate sign bytes overflow u64".to_string(),
                })?;
        ranges.push((s.offset, end));
    }
    ranges.sort_unstable();
    if ranges.windows(2).any(|pair| pair[1].0 < pair[0].1) {
        return Err(ModelError::IndexCorrupt {
            detail: "manifest.hadamard sign ranges overlap".to_string(),
        });
    }
    let max_upload_bytes = allowed_widths.iter().try_fold(0_u64, |total, width| {
        u64::try_from(*width)
            .ok()
            .and_then(|width| width.checked_mul(4))
            .and_then(|bytes| total.checked_add(bytes))
    });
    if max_upload_bytes.is_none_or(|cap| aggregate_bytes > cap || file_bytes > cap) {
        return Err(ModelError::IndexCorrupt {
            detail: format!(
                "manifest.hadamard data exceeds the architecture-derived upload cap (file {file_bytes}, upload {aggregate_bytes})"
            ),
        });
    }
    if h.folded.is_empty() && h.inverse.is_empty() {
        return Err(ModelError::IndexCorrupt {
            detail: "manifest.hadamard carries no folded and no inverse entries".to_string(),
        });
    }
    Ok(())
}

/// The files an install carrying a vision tower must declare. ONE blob rather
/// than one per block: the tower streams by BLOCK the way an MoE layer streams
/// by expert, and `PackedExpertsLayout`'s schema puts all of one layer's
/// experts in one file.
const VISION_FILES: [&str; 2] = ["packed_vision/layout.json", "packed_vision/blobs.bin"];

/// A FLOOR on the alignment `expertStride` must respect, not the real page
/// size. Apple Silicon macOS actually uses 16 KiB pages (see
/// `resident_buffer.rs`'s `page_size_bytes`, which DOES query the OS for the
/// mmap alignment that genuinely has to match it) and the writer aligns to
/// that exact figure (`GTURBO_PAGE_BYTES = 16_384`), so every real install's
/// stride is already a multiple of 16 KiB and therefore of the 4 KiB
/// checked here. Left at 4096 rather than raised to 16_384 without having
/// checked every install already on disk against the stricter bound; the
/// manifest format itself has no per-install page size field to validate
/// against in either case.
fn page_size_bytes() -> u64 {
    4096
}

/// A manifest matching one of the shipped production baselines must carry
/// quantization metadata; toy/synthetic manifests may omit it.
///
/// Matches on family plus every shape field a real checkpoint of that
/// baseline always carries unchanged. `(num_layers, hidden_size)` alone let
/// a toy arch that happened to share both be forced to carry `quant`.
fn is_production_arch(expected: &ArchConfig) -> bool {
    all_known_architectures().iter().any(|b| {
        b.family == expected.family
            && b.num_layers == expected.num_layers
            && b.hidden_size == expected.hidden_size
            && b.num_heads == expected.num_heads
            && b.num_kv_heads == expected.num_kv_heads
            && b.head_dim == expected.head_dim
            && b.vocab_size == expected.vocab_size
            && b.intermediate_size == expected.intermediate_size
            && b.moe_intermediate_size == expected.moe_intermediate_size
            && b.num_experts == expected.num_experts
    })
}

/// Decode just enough of `manifest.json` to identify the model family,
/// without arch validation.
pub fn peek_family(dir: &Path, max_bytes: u64) -> Result<ModelFamily, ModelError> {
    let manifest_path = dir.join("manifest.json");
    if !manifest_path.exists() {
        return Err(ModelError::PartialInstall {
            path: dir.display().to_string(),
        });
    }
    let size = file_size(&manifest_path)?;
    if size > max_bytes {
        return Err(ModelError::IndexCorrupt {
            detail: format!("manifest.json size {size} exceeds metadata cap {max_bytes}"),
        });
    }
    let data = read_capped(&manifest_path, max_bytes)
        .map_err(|e| ModelError::IoFailed {
            call: "read".to_string(),
            detail: e.to_string(),
        })?
        .ok_or_else(|| ModelError::IndexCorrupt {
            detail: format!("manifest.json size exceeds metadata cap {max_bytes}"),
        })?;
    let manifest: Manifest =
        serde_json::from_slice(&data).map_err(|e| ModelError::IndexCorrupt {
            detail: format!("manifest.json: {e}"),
        })?;
    let Some(raw) = manifest.arch.family else {
        return Ok(ModelFamily::Gemma4);
    };
    ModelFamily::parse(&raw).ok_or_else(|| ModelError::IndexCorrupt {
        detail: format!("unknown arch.family \"{raw}\""),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every shipped baseline must still require `quant`.
    #[test]
    fn every_shipped_baseline_is_production() {
        for b in all_known_architectures() {
            assert!(is_production_arch(&b), "{:?} lost production", b.family);
        }
    }

    /// A toy arch that shares only `num_layers` and `hidden_size` with a
    /// baseline is not production: each identifying field on its own must
    /// break the match, or a toy manifest is forced to carry `quant`.
    #[test]
    fn a_toy_sharing_layers_and_hidden_is_not_production() {
        let base = crate::arch_baselines::qwen3_4b();
        type Mutation = (&'static str, fn(&mut ArchConfig));
        let mutations: [Mutation; 8] = [
            ("family", |a| a.family = ModelFamily::Gemma4),
            ("num_heads", |a| a.num_heads += 1),
            ("num_kv_heads", |a| a.num_kv_heads += 1),
            ("head_dim", |a| a.head_dim += 1),
            ("vocab_size", |a| a.vocab_size += 1),
            ("intermediate_size", |a| a.intermediate_size += 1),
            ("moe_intermediate_size", |a| a.moe_intermediate_size += 1),
            ("num_experts", |a| a.num_experts += 1),
        ];
        for (field, mutate) in mutations {
            let mut toy = base.clone();
            mutate(&mut toy);
            assert!(!is_production_arch(&toy), "{field} alone kept the match");
        }
    }
}
