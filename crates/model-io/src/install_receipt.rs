//! Trusted install receipt: an offline-verified record of an install's
//! per-file size/SHA-256, checked instead of re-hashing large files on every
//! load. Ported from `Infrastructure/ModelIO/VerifiedInstallReceipt.swift`.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::ModelError;
use crate::manifest::Manifest;

/// Policy for verifying model installation file integrity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelIntegrityPolicy {
    /// Fully hash all model files using SHA-256 on load.
    FullSha256,
    /// Fast validation checking file sizes against a trusted receipt.
    SizeCheckTrustedReceipt,
}

/// Recorded file size and SHA-256 digest entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileEntry {
    /// Expected file size in bytes.
    pub size: u64,
    /// Expected SHA-256 hash string.
    pub sha256: String,
}

/// Trusted install receipt record saved in `verified-install.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VerifiedInstallReceipt {
    /// Receipt schema version number.
    #[serde(default = "default_schema_version")]
    pub schema_version: i64,
    /// SHA-256 digest of `manifest.json`.
    pub manifest_sha256: String,
    /// Canonical model directory path.
    pub model_directory_path: String,
    /// Hugging Face source repository ID if present.
    #[serde(default)]
    pub source_repo_id: Option<String>,
    /// Git commit hash or revision string if present.
    #[serde(default)]
    pub source_revision: Option<String>,
    /// Verification timestamp string.
    pub verification_timestamp: String,
    /// Repack/verification tool version string.
    pub tool_version: String,
    /// File size and hash map for installed files.
    pub files: BTreeMap<String, FileEntry>,
}

fn default_schema_version() -> i64 {
    1
}

/// Default filename for trusted install receipt.
pub const FILE_NAME: &str = "verified-install.json";
/// Maximum allowed file size for trusted install receipt metadata.
pub const DEFAULT_MAX_BYTES: u64 = 4 * 1024 * 1024;

fn invalid(detail: impl Into<String>) -> ModelError {
    ModelError::TrustedReceiptInvalid {
        detail: detail.into(),
    }
}

/// Loads and parses trusted install receipt from `dir`.
pub fn load(dir: &Path, max_bytes: u64) -> Result<VerifiedInstallReceipt, ModelError> {
    let path = dir.join(FILE_NAME);
    if !path.exists() {
        return Err(invalid(format!("{FILE_NAME} is missing")));
    }
    let size = std::fs::metadata(&path)
        .map_err(|e| invalid(format!("{FILE_NAME} size unavailable: {e}")))?
        .len();
    if size > max_bytes {
        return Err(invalid(format!(
            "{FILE_NAME} size {size} exceeds metadata cap {max_bytes}"
        )));
    }
    let data = std::fs::read(&path).map_err(|e| invalid(format!("{FILE_NAME}: {e}")))?;
    serde_json::from_slice(&data).map_err(|e| invalid(format!("{FILE_NAME}: {e}")))
}

/// Validates receipt manifest SHA-256 binding and canonical directory path.
pub fn validate_manifest_binding(
    receipt: &VerifiedInstallReceipt,
    dir: &Path,
    manifest_sha256: &str,
) -> Result<(), ModelError> {
    if receipt.schema_version != 1 {
        return Err(invalid(format!(
            "unsupported schemaVersion {}",
            receipt.schema_version
        )));
    }
    if receipt.manifest_sha256.to_lowercase() != manifest_sha256.to_lowercase() {
        return Err(invalid("manifest SHA mismatch"));
    }
    let actual_path = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let actual_path = actual_path.display().to_string();
    if receipt.model_directory_path != actual_path {
        return Err(invalid("model directory mismatch"));
    }
    Ok(())
}

/// Validates receipt contents against manifest and disk files.
pub fn validate(
    receipt: &VerifiedInstallReceipt,
    dir: &Path,
    manifest: &Manifest,
    manifest_sha256: &str,
    manifest_size: u64,
) -> Result<(), ModelError> {
    validate_manifest_binding(receipt, dir, manifest_sha256)?;

    let mut expected_files: HashSet<&str> = manifest.files.keys().map(String::as_str).collect();
    expected_files.insert("manifest.json");
    let receipt_files: HashSet<&str> = receipt.files.keys().map(String::as_str).collect();
    if receipt_files != expected_files {
        let mut missing: Vec<&str> = expected_files.difference(&receipt_files).copied().collect();
        missing.sort_unstable();
        let mut extra: Vec<&str> = receipt_files.difference(&expected_files).copied().collect();
        extra.sort_unstable();
        return Err(invalid(format!(
            "receipt file set mismatch missing={missing:?} extra={extra:?}"
        )));
    }

    let manifest_entry = receipt
        .files
        .get("manifest.json")
        .ok_or_else(|| invalid("receipt missing manifest.json"))?;
    if manifest_entry.size != manifest_size {
        return Err(invalid("manifest.json size mismatch"));
    }
    if manifest_entry.sha256.to_lowercase() != manifest_sha256.to_lowercase() {
        return Err(invalid("manifest.json SHA mismatch"));
    }

    for (rel, manifest_entry) in &manifest.files {
        let receipt_entry = receipt
            .files
            .get(rel)
            .ok_or_else(|| invalid(format!("receipt missing {rel}")))?;
        if receipt_entry.size != manifest_entry.size {
            return Err(invalid(format!("receipt size mismatch for {rel}")));
        }
        if receipt_entry.sha256.to_lowercase() != manifest_entry.sha256.to_lowercase() {
            return Err(invalid(format!("receipt SHA mismatch for {rel}")));
        }
    }
    Ok(())
}
