//! Tests for streaming SHA-256 file hashing and verification.

use std::io::Write;

use turbospark_model_io::{hash_data, hash_file, verify_file, ModelError};

#[test]
fn hash_data_matches_known_vector() {
    // SHA-256("abc")
    assert_eq!(
        hash_data(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn hash_file_matches_hash_data_and_is_chunk_size_independent() {
    let dir = std::env::temp_dir().join(format!("turbospark-sha256-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("data.bin");
    let content = vec![0x5au8; 10_000];
    std::fs::File::create(&path)
        .unwrap()
        .write_all(&content)
        .unwrap();

    let expected = hash_data(&content);
    assert_eq!(hash_file(&path, 1 << 20).unwrap(), expected);
    assert_eq!(hash_file(&path, 37).unwrap(), expected);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn verify_file_rejects_checksum_mismatch() {
    let dir = std::env::temp_dir().join(format!("turbospark-sha256-verify-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("data.bin");
    std::fs::File::create(&path)
        .unwrap()
        .write_all(b"hello")
        .unwrap();

    let err = verify_file(&path, "data.bin", "0000").unwrap_err();
    assert!(matches!(err, ModelError::ChecksumMismatch { .. }));

    let good = hash_data(b"hello");
    verify_file(&path, "data.bin", &good.to_uppercase()).unwrap();

    std::fs::remove_dir_all(&dir).ok();
}
