//! Tests for safetensors header parsing and ranged-download planning,
//! using synthetic byte fixtures (no network needed).

use turbospark_repack::{
    fetch_safetensors_header, parse_header, required_prefix_len, MemoryRangeSource,
    SafetensorsHeaderError,
};

/// Builds a synthetic safetensors file: 8-byte LE header length, then the
/// header JSON, then `data_len` zero bytes standing in for tensor data.
fn build_file(header_json: &str, data_len: usize) -> Vec<u8> {
    let header_bytes = header_json.as_bytes();
    let mut file = Vec::new();
    file.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    file.extend_from_slice(header_bytes);
    file.extend(std::iter::repeat_n(0u8, data_len));
    file
}

fn sample_header_json() -> &'static str {
    r#"{
        "__metadata__": {"format": "pt"},
        "weight.0": {"dtype": "F32", "shape": [2, 64], "data_offsets": [0, 512]},
        "weight.1": {"dtype": "F32", "shape": [2, 64], "data_offsets": [512, 1024]}
    }"#
}

#[test]
fn parse_header_reads_tensors_and_metadata() {
    let file = build_file(sample_header_json(), 1024);
    let header = parse_header(&file, 1 << 20).unwrap();

    assert_eq!(header.tensors.len(), 2);
    let t0 = &header.tensors["weight.0"];
    assert_eq!(t0.dtype, "F32");
    assert_eq!(t0.shape, vec![2, 64]);
    assert_eq!(t0.data_offsets, (0, 512));
    assert_eq!(header.metadata.as_ref().unwrap()["format"], "pt");
}

#[test]
fn absolute_range_offsets_by_the_data_region_start() {
    let file = build_file(sample_header_json(), 1024);
    let header = parse_header(&file, 1 << 20).unwrap();
    let (start, end) = header.absolute_range("weight.1").unwrap();
    assert_eq!(start, header.data_region_start() + 512);
    assert_eq!(end, header.data_region_start() + 1024);
    assert!(header.absolute_range("missing").is_none());
}

#[test]
fn parse_header_rejects_a_length_prefix_exceeding_the_cap() {
    let mut bytes = vec![0u8; 8];
    bytes[0..8].copy_from_slice(&(1u64 << 40).to_le_bytes());
    let err = parse_header(&bytes, 1 << 20).unwrap_err();
    assert!(matches!(err, SafetensorsHeaderError::HeaderTooLarge { .. }));
}

#[test]
fn parse_header_rejects_truncated_input() {
    let file = build_file(sample_header_json(), 0);
    let truncated = &file[..file.len() - 5];
    let err = parse_header(truncated, 1 << 20).unwrap_err();
    assert_eq!(err, SafetensorsHeaderError::TooShort);
}

#[test]
fn parse_header_rejects_malformed_json() {
    let mut file = vec![0u8; 8];
    let bad = b"not json";
    file[0..8].copy_from_slice(&(bad.len() as u64).to_le_bytes());
    file.extend_from_slice(bad);
    let err = parse_header(&file, 1 << 20).unwrap_err();
    assert!(matches!(err, SafetensorsHeaderError::InvalidJson(_)));
}

#[test]
fn required_prefix_len_matches_header_len_plus_eight() {
    let file = build_file(sample_header_json(), 1024);
    let mut first_8 = [0u8; 8];
    first_8.copy_from_slice(&file[0..8]);
    assert_eq!(
        required_prefix_len(&first_8),
        8 + sample_header_json().len() as u64
    );
}

#[test]
fn fetch_safetensors_header_over_memory_source_matches_direct_parse() {
    let file = build_file(sample_header_json(), 1024);
    let source = MemoryRangeSource::new(&file);
    let via_fetch = fetch_safetensors_header(&source).unwrap();
    let via_parse = parse_header(&file, turbospark_repack::DEFAULT_MAX_HEADER_BYTES).unwrap();
    assert_eq!(via_fetch, via_parse);
}
