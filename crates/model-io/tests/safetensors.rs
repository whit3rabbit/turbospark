//! Tests for `SafetensorsFile`, using hand-assembled files rather than a
//! real checkpoint: an 8-byte little-endian header length, a JSON header,
//! then the tensor data region.

use std::io::Write;

use turbospark_model_io::{ModelError, SafetensorsFile};

/// One F32 tensor named "w", shape [2, 2] (4 elements, 16 bytes), at
/// `data_offsets: (0, 16)`.
fn valid_file_bytes() -> Vec<u8> {
    let header = br#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
    let mut buf = Vec::new();
    buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
    buf.extend_from_slice(header);
    buf.extend_from_slice(&[0u8; 16]);
    buf
}

fn write_file(bytes: &[u8]) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-safetensors-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    std::fs::File::create(&path)
        .unwrap()
        .write_all(bytes)
        .unwrap();
    path
}

#[test]
fn a_well_formed_file_loads_and_decodes() {
    let path = write_file(&valid_file_bytes());
    let file = SafetensorsFile::open(&path).unwrap();
    assert!(file.contains_tensor("w"));
    let values = file.load_as_f32("w").unwrap();
    assert_eq!(values.len(), 4);
}

#[test]
fn a_backwards_data_offsets_range_is_refused() {
    let header = br#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[16,0]}}"#;
    let mut buf = Vec::new();
    buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
    buf.extend_from_slice(header);
    buf.extend_from_slice(&[0u8; 16]);
    let path = write_file(&buf);
    let file = SafetensorsFile::open(&path).unwrap();
    let err = file.raw_bytes("w").unwrap_err();
    assert!(matches!(err, ModelError::IndexCorrupt { .. }), "{err:?}");
}

#[test]
fn data_offsets_past_the_file_are_refused() {
    // The file itself is only 16 bytes of data, but the descriptor claims
    // 32.
    let header = br#"{"w":{"dtype":"F32","shape":[8,1],"data_offsets":[0,32]}}"#;
    let mut buf = Vec::new();
    buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
    buf.extend_from_slice(header);
    buf.extend_from_slice(&[0u8; 16]);
    let path = write_file(&buf);
    let file = SafetensorsFile::open(&path).unwrap();
    let err = file.raw_bytes("w").unwrap_err();
    assert!(matches!(err, ModelError::IndexCorrupt { .. }), "{err:?}");
}

/// A shape that disagrees with the byte range it decodes -- here declaring
/// 9 elements (36 bytes at F32) over a 16-byte range -- must fail rather
/// than silently return a 4-element vector under a caller's belief that it
/// holds 9.
#[test]
fn a_shape_disagreeing_with_the_byte_range_is_refused() {
    let header = br#"{"w":{"dtype":"F32","shape":[3,3],"data_offsets":[0,16]}}"#;
    let mut buf = Vec::new();
    buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
    buf.extend_from_slice(header);
    buf.extend_from_slice(&[0u8; 16]);
    let path = write_file(&buf);
    let file = SafetensorsFile::open(&path).unwrap();
    let err = file.load_as_f32("w").unwrap_err();
    assert!(matches!(err, ModelError::IndexCorrupt { .. }), "{err:?}");
}

/// A tensor descriptor missing a required field (`shape`) used to be
/// silently dropped by an `if let Ok`; it must now fail to open, naming the
/// key.
#[test]
fn a_malformed_tensor_descriptor_fails_to_open_rather_than_being_dropped() {
    let header = br#"{"w":{"dtype":"F32","data_offsets":[0,16]}}"#;
    let mut buf = Vec::new();
    buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
    buf.extend_from_slice(header);
    buf.extend_from_slice(&[0u8; 16]);
    let path = write_file(&buf);
    // `SafetensorsFile` has no `Debug` impl, so `unwrap_err` cannot be used
    // here (AGENTS.md's `expect_err` gotcha, one type over).
    let Err(err) = SafetensorsFile::open(&path) else {
        panic!("expected the malformed descriptor to be refused at open");
    };
    let ModelError::IndexCorrupt { detail } = err else {
        panic!("expected IndexCorrupt");
    };
    assert!(detail.contains('w'), "{detail}");
}

/// A header length so large that `8 + header_len` overflows `usize` must be
/// refused rather than panicking or wrapping into a small, wrong offset.
#[test]
fn a_header_length_overflow_is_refused() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&u64::MAX.to_le_bytes());
    buf.extend_from_slice(&[0u8; 8]);
    let path = write_file(&buf);
    let Err(err) = SafetensorsFile::open(&path) else {
        panic!("expected the header-length overflow to be refused at open");
    };
    assert!(matches!(err, ModelError::IndexCorrupt { .. }), "{err:?}");
}
