//! `manifest.json` decode and validation against a resolved [`ArchConfig`].
//! Ported from `Infrastructure/ModelIO/ManifestReader.swift`.

mod quant;
mod types;

use std::collections::HashSet;
use std::path::Path;

pub(crate) use quant::validate_quant;
pub use quant::EXECUTABLE_GGUF_TYPES;
pub use types::{Manifest, ManifestArch, ManifestFileEntry, ManifestQuant, ManifestQuantSlot};

use crate::arch_baselines::all_known_architectures;
use crate::arch_config::{ArchConfig, ModelFamily};
use crate::error::ModelError;

/// Default maximum byte limit for reading `manifest.json`.
pub const DEFAULT_MAX_BYTES: u64 = 4 * 1024 * 1024;

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
    let data = std::fs::read(&manifest_path).map_err(|e| ModelError::IoFailed {
        call: "read".to_string(),
        detail: e.to_string(),
    })?;
    let manifest: Manifest =
        serde_json::from_slice(&data).map_err(|e| ModelError::IndexCorrupt {
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
    if m.version_major != 1 {
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
fn is_production_arch(expected: &ArchConfig) -> bool {
    all_known_architectures()
        .iter()
        .any(|b| b.num_layers == expected.num_layers && b.hidden_size == expected.hidden_size)
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
    let data = std::fs::read(&manifest_path).map_err(|e| ModelError::IoFailed {
        call: "read".to_string(),
        detail: e.to_string(),
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
