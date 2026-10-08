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
    Ok(hash_file_observed(path, chunk_bytes, |_| true)?
        .expect("an observer that never stops cannot cancel the hash"))
}

/// [`hash_file`] with a hook after every chunk. `observe` receives the bytes
/// hashed so far and returns whether to continue; returning `false` abandons
/// the hash and yields `Ok(None)`. The granularity is one chunk, so a
/// multi-gigabyte file can be stopped within `chunk_bytes` of the request
/// rather than only between files.
pub fn hash_file_observed(
    path: &Path,
    chunk_bytes: usize,
    mut observe: impl FnMut(u64) -> bool,
) -> Result<Option<String>, ModelError> {
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
    let mut done = 0u64;
    loop {
        let got = file.read(&mut buf).map_err(|e| ModelError::IoFailed {
            call: "read".to_string(),
            detail: e.to_string(),
        })?;
        if got == 0 {
            break;
        }
        hasher.update(&buf[..got]);
        done += got as u64;
        if !observe(done) {
            return Ok(None);
        }
    }
    Ok(Some(hex_encode(&hasher.finalize())))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str, len: usize) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("sha256-{}-{name}", std::process::id()));
        // Not a repeating pattern, so a hash over the wrong span differs.
        let bytes: Vec<u8> = (0..len).map(|i| (i.wrapping_mul(2654435761) >> 7) as u8).collect();
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn an_observer_that_never_stops_yields_the_plain_digest() {
        let path = scratch("plain", 5 * 1024 * 1024 + 123);
        let plain = hash_file(&path, 1 << 20).unwrap();
        let observed = hash_file_observed(&path, 1 << 20, |_| true).unwrap();
        assert_eq!(observed.as_deref(), Some(plain.as_str()));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn the_observer_sees_cumulative_bytes_and_can_stop_inside_a_file() {
        let chunk = 1 << 20;
        let path = scratch("stop", 5 * chunk + 123);
        let mut seen = Vec::new();
        let result = hash_file_observed(&path, chunk, |done| {
            seen.push(done);
            done < 2 * chunk as u64 // stop after the second chunk
        })
        .unwrap();
        assert_eq!(result, None, "a stopped hash yields no digest");
        assert_eq!(seen, vec![chunk as u64, 2 * chunk as u64], "stopped within a chunk of the request");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn an_empty_file_still_hashes() {
        let path = scratch("empty", 0);
        assert_eq!(
            hash_file_observed(&path, 1 << 20, |_| true).unwrap().unwrap(),
            hash_data(&[])
        );
        std::fs::remove_file(path).unwrap();
    }
}
