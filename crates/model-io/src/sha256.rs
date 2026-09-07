//! Streaming SHA-256 verification of install files. Ported from
//! `Infrastructure/ModelIO/Sha256Verifier.swift`, backed by the maintained
//! `sha2` crate (RustCrypto) instead of `CommonCrypto`.

use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::error::ModelError;

/// Compute the lowercase-hex SHA-256 of the entire file at `path` by
/// streaming through a fixed-size scratch read. Does not allocate the whole
/// file.
pub fn hash_file(path: &Path, chunk_bytes: usize) -> Result<String, ModelError> {
    let mut file = std::fs::File::open(path).map_err(|e| ModelError::IoFailed {
        call: "open".to_string(),
        detail: e.to_string(),
    })?;
    let mut hasher = Sha256::new();
    // A zero-length buffer's `read` always returns `Ok(0)` regardless of
    // what the file actually holds (the `Read` trait's own contract), so
    // `chunk_bytes: 0` would otherwise report the EMPTY-file hash for any
    // file, silently.
    let mut buf = vec![0u8; chunk_bytes.max(1)];
    loop {
        let got = file.read(&mut buf).map_err(|e| ModelError::IoFailed {
            call: "read".to_string(),
            detail: e.to_string(),
        })?;
        if got == 0 {
            break;
        }
        hasher.update(&buf[..got]);
    }
    Ok(hex_encode(&hasher.finalize()))
}

pub fn hash_data(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex_encode(&hasher.finalize())
}

/// Returns [`ModelError::ChecksumMismatch`] if the on-disk file's SHA-256
/// does not match `expected_hex` (case-insensitive).
pub fn verify_file(path: &Path, name: &str, expected_hex: &str) -> Result<(), ModelError> {
    let actual = hash_file(path, 1 << 20)?;
    if actual.to_lowercase() != expected_hex.to_lowercase() {
        return Err(ModelError::ChecksumMismatch {
            file: name.to_string(),
        });
    }
    Ok(())
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
