//! Tests for the trusted install receipt binding/validation logic.

use std::collections::BTreeMap;

use turbospark_model_io::{
    validate_install_receipt_manifest_binding, InstallReceiptFileEntry, ModelError,
    VerifiedInstallReceipt,
};

fn receipt(manifest_sha256: &str, dir_path: &str) -> VerifiedInstallReceipt {
    let mut files = BTreeMap::new();
    files.insert(
        "manifest.json".to_string(),
        InstallReceiptFileEntry {
            size: 10,
            sha256: manifest_sha256.to_string(),
        },
    );
    VerifiedInstallReceipt {
        schema_version: 1,
        manifest_sha256: manifest_sha256.to_string(),
        model_directory_path: dir_path.to_string(),
        source_repo_id: None,
        source_revision: None,
        verification_timestamp: "2024-01-01T00:00:00Z".to_string(),
        tool_version: "test".to_string(),
        files,
    }
}

#[test]
fn binding_accepts_matching_sha_and_canonical_directory() {
    let dir = std::env::temp_dir().join(format!("turbospark-receipt-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let canonical = dir.canonicalize().unwrap();
    let r = receipt("abc123", &canonical.display().to_string());
    validate_install_receipt_manifest_binding(&r, &dir, "ABC123").unwrap();
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn binding_rejects_manifest_sha_mismatch() {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-receipt-mismatch-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let canonical = dir.canonicalize().unwrap();
    let r = receipt("abc123", &canonical.display().to_string());
    let err = validate_install_receipt_manifest_binding(&r, &dir, "different").unwrap_err();
    assert!(matches!(err, ModelError::TrustedReceiptInvalid { .. }));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn binding_rejects_unsupported_schema_version() {
    let dir =
        std::env::temp_dir().join(format!("turbospark-receipt-schema-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let canonical = dir.canonicalize().unwrap();
    let mut r = receipt("abc123", &canonical.display().to_string());
    r.schema_version = 2;
    let err = validate_install_receipt_manifest_binding(&r, &dir, "abc123").unwrap_err();
    assert!(matches!(err, ModelError::TrustedReceiptInvalid { .. }));
    std::fs::remove_dir_all(&dir).ok();
}
