//! Full-SHA256 install verification: composes `turbospark_model_io`'s
//! manifest loader, resident-index reader, and SHA-256 hasher into the
//! `ModelIntegrityPolicy::FullSha256` path from
//! `Infrastructure/ModelIO/VerifiedInstallReceipt.swift` — re-hashing
//! every file in `manifest.files` rather than trusting a prior receipt.

use std::path::{Component, Path};

use model_io::{hash_file, load_manifest, ArchConfig, ModelError};

/// Re-hashes every file `manifest.json` lists and compares against its
/// declared SHA-256, failing on the first mismatch. Does not touch a
/// [`model_io::VerifiedInstallReceipt`] — that's the faster,
/// trust-a-prior-verification path; this is the one that reads every byte.
pub fn verify_install_full_sha256(dir: &Path, expecting: &ArchConfig) -> Result<(), ModelError> {
    let manifest = load_manifest(dir, expecting, model_io::DEFAULT_MAX_BYTES)?;
    for (relative_path, entry) in &manifest.files {
        // Manifests come from third-party installs. Without this, `..` or an
        // absolute key makes the hash compare an oracle for files outside
        // the install directory.
        let relative = Path::new(relative_path);
        if relative_path.is_empty()
            || !relative
                .components()
                .all(|c| matches!(c, Component::Normal(_)))
        {
            return Err(ModelError::IndexCorrupt {
                detail: format!(
                    "manifest.files key {relative_path:?} is not a path inside the install"
                ),
            });
        }
        let path = dir.join(relative);
        let actual = hash_file(&path, 1 << 20)?;
        if actual.to_lowercase() != entry.sha256.to_lowercase() {
            return Err(ModelError::ChecksumMismatch {
                file: relative_path.clone(),
            });
        }
    }
    Ok(())
}
