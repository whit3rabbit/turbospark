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
    let err = SafetensorsFile::open(&path)
        .err()
        .expect("invalid range must fail at open");
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
    let err = SafetensorsFile::open(&path)
        .err()
        .expect("invalid range must fail at open");
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

fn framed(header: &serde_json::Value, data: &[u8]) -> Vec<u8> {
    let json = serde_json::to_vec(header).unwrap();
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend(json);
    bytes.extend(data);
    bytes
}

#[test]
fn globally_overlapping_ranges_refuse_before_owned_loading() {
    // Identical, nested, and partially overlapping ranges are independently
    // in bounds. None may authorize another owned copy of the same bytes.
    for offsets in [[0, 16], [4, 12], [8, 24]] {
        let header = serde_json::json!({
            "z": {"dtype":"F32", "shape":[4], "data_offsets":[0,16]},
            "a": {"dtype":"F32", "shape":[(offsets[1]-offsets[0])/4], "data_offsets":offsets}
        });
        let path = write_file(&framed(&header, &[0; 24]));
        let err = SafetensorsFile::open(&path)
            .err()
            .expect("overlap must fail at open");
        assert!(err.to_string().contains("overlap"), "{err}");
    }
}

#[test]
fn many_aliases_cannot_amplify_a_small_checkpoint() {
    let mut header = serde_json::Map::new();
    for layer in 0..512 {
        header.insert(
            format!("layer.{layer}"),
            serde_json::json!({
                "dtype":"F32", "shape":[4], "data_offsets":[0,16]
            }),
        );
    }
    let path = write_file(&framed(&header.into(), &[0; 16]));
    let err = SafetensorsFile::open(&path)
        .err()
        .expect("aliases must fail at open");
    assert!(err.to_string().contains("overlap"), "{err}");
}

#[test]
fn ordered_by_offsets_adjacent_and_empty_tensors_remain_valid() {
    let header = serde_json::json!({
        "z": {"dtype":"F32", "shape":[1], "data_offsets":[0,4]},
        "a": {"dtype":"F32", "shape":[1], "data_offsets":[4,8]},
        "empty": {"dtype":"F32", "shape":[0], "data_offsets":[4,4]}
    });
    let bytes = framed(&header, &[0; 8]);
    let path = write_file(&bytes);
    let file = SafetensorsFile::open(&path).unwrap();
    assert_eq!(file.load_as_f32("a").unwrap(), vec![0.0]);
    assert!(file.load_as_f32("empty").unwrap().is_empty());
    file.verify_sha256(&turbospark_model_io::hash_data(&bytes))
        .unwrap();
    assert!(file.verify_sha256(&"0".repeat(64)).is_err());
}

#[test]
fn gaps_and_unclaimed_trailing_data_refuse_at_open() {
    for offsets in [[4, 8], [0, 4]] {
        let header = serde_json::json!({
            "w": {"dtype":"F32", "shape":[1], "data_offsets":offsets}
        });
        let path = write_file(&framed(&header, &[0; 8]));
        assert!(SafetensorsFile::open(&path).is_err());
    }
}

#[test]
fn model_specific_header_limit_is_checked_before_json_expansion() {
    let bytes = valid_file_bytes();
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    let path = write_file(&bytes);
    assert!(SafetensorsFile::open_with_header_limit(&path, header_len).is_ok());
    let err = SafetensorsFile::open_with_header_limit(&path, header_len - 1)
        .err()
        .unwrap();
    assert!(err.to_string().contains("header exceeds"), "{err}");
}
