//! Tests for safetensors header parsing and ranged-download planning,
//! using synthetic byte fixtures (no network needed).

use std::cell::Cell;

use turbospark_repack::{
    fetch_safetensors_header, parse_header, required_prefix_len, DownloadError, MemoryRangeSource,
    RangeSource, SafetensorsHeaderError,
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

/// `"__metadata__": null`, the exact shape a real converter writes
/// (`sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit`'s shard headers), which an
/// untagged `Tensor | Metadata(BTreeMap<..>)` enum cannot deserialize from
/// `null` at all -- it failed the WHOLE header with a message naming neither
/// the key nor the reason. `null` means the same as an absent key: no
/// metadata, never an error.
#[test]
fn parse_header_accepts_a_null_metadata_value() {
    let json = r#"{
        "__metadata__": null,
        "weight.0": {"dtype": "F32", "shape": [2, 64], "data_offsets": [0, 512]}
    }"#;
    let file = build_file(json, 512);
    let header = parse_header(&file, 1 << 20).unwrap();
    assert_eq!(header.tensors.len(), 1);
    assert_eq!(header.metadata, None);
}

/// A `__metadata__` value that is neither `null` nor an object is still
/// refused, by name -- the fix widens what a REAL converter's `null` means,
/// not what any other shape is allowed to mean.
#[test]
fn parse_header_rejects_a_non_null_non_object_metadata_value() {
    let json = r#"{
        "__metadata__": "not an object",
        "weight.0": {"dtype": "F32", "shape": [2, 64], "data_offsets": [0, 512]}
    }"#;
    let file = build_file(json, 512);
    let err = parse_header(&file, 1 << 20).unwrap_err();
    let SafetensorsHeaderError::InvalidJson(detail) = err else {
        panic!("expected InvalidJson, got {err:?}");
    };
    assert!(
        detail.contains("__metadata__"),
        "message should name __metadata__, got: {detail}"
    );
}

#[test]
fn parse_header_rejects_unknown_tensor_fields() {
    let json = r#"{
        "weight.0": {
            "dtype": "F32", "shape": [2, 64], "data_offsets": [0, 512],
            "ignored": [0, 0, 0, 0]
        }
    }"#;
    let file = build_file(json, 512);
    let err = parse_header(&file, 1 << 20).unwrap_err();
    let SafetensorsHeaderError::InvalidJson(detail) = err else {
        panic!("expected InvalidJson, got {err:?}");
    };
    assert!(detail.contains("unknown field `ignored`"), "got: {detail}");
}

#[test]
fn parse_header_rejects_excessive_tensor_dimensions() {
    let dimensions = std::iter::repeat_n("1", 33).collect::<Vec<_>>().join(",");
    let json = format!(
        r#"{{"weight.0": {{"dtype": "F32", "shape": [{dimensions}], "data_offsets": [0, 4]}}}}"#
    );
    let file = build_file(&json, 4);
    let err = parse_header(&file, 1 << 20).unwrap_err();
    let SafetensorsHeaderError::InvalidJson(detail) = err else {
        panic!("expected InvalidJson, got {err:?}");
    };
    assert!(detail.contains("exceeds 32 dimensions"), "got: {detail}");
}

/// `data_offsets` end before start reaches `absolute_range` as a
/// plausible-looking pair and then every unchecked `end - start` downstream
/// (ranged_download's chunking, the expert-blob planner) either panics or
/// wraps to a huge allocation. Refused at parse time instead.
#[test]
fn parse_header_rejects_an_inverted_data_offsets_range() {
    let json = r#"{
        "weight.0": {"dtype": "F32", "shape": [2, 64], "data_offsets": [512, 0]}
    }"#;
    let file = build_file(json, 1024);
    let err = parse_header(&file, 1 << 20).unwrap_err();
    let SafetensorsHeaderError::InvalidJson(detail) = err else {
        panic!("expected InvalidJson, got {err:?}");
    };
    assert!(
        detail.contains("weight.0"),
        "message should name the tensor, got: {detail}"
    );
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

/// `MemoryRangeSource` is test-only, but every fixture test funnels through
/// it, so a malformed range (start past end, or past the fixture's own
/// length) must report `ShortRead` rather than panic on the slice.
#[test]
fn memory_range_source_reports_short_read_instead_of_panicking() {
    let data = vec![1u8, 2, 3, 4];
    let source = MemoryRangeSource::new(&data);

    assert!(matches!(
        source.read_range(3, 1),
        Err(DownloadError::ShortRead { .. })
    ));
    assert!(matches!(
        source.read_range(0, 100),
        Err(DownloadError::ShortRead { .. })
    ));
    assert!(matches!(
        source.read_range(100, 200),
        Err(DownloadError::ShortRead { .. })
    ));
    // The ordinary case must still work.
    assert_eq!(source.read_range(1, 3).unwrap(), vec![2u8, 3]);
}

/// Counts `read_range` calls, so the "reject before the second fetch" claim
/// is asserted rather than assumed.
struct CountingSource {
    data: Vec<u8>,
    reads: Cell<usize>,
}

impl RangeSource for CountingSource {
    fn read_range(&self, start: u64, end_exclusive: u64) -> Result<Vec<u8>, DownloadError> {
        self.reads.set(self.reads.get() + 1);
        let start = start as usize;
        let end = end_exclusive as usize;
        if end > self.data.len() {
            return Err(DownloadError::ShortRead {
                expected: end_exclusive - start as u64,
                actual: self.data.len().saturating_sub(start) as u64,
            });
        }
        Ok(self.data[start..end].to_vec())
    }
}

/// A length prefix declaring 2^40 must be rejected off the FIRST 8-byte
/// read: `HttpRangeSource::read_range` allocates its whole destination
/// buffer before any parse runs, so fetching that much before checking it
/// against the cap would abort a real process (or, at a merely large value,
/// issue a multi-GB GET).
#[test]
fn fetch_safetensors_header_rejects_a_hostile_length_prefix_before_the_second_read() {
    let mut bytes = vec![0u8; 8];
    bytes[0..8].copy_from_slice(&(1u64 << 40).to_le_bytes());
    let source = CountingSource {
        data: bytes,
        reads: Cell::new(0),
    };
    let err = fetch_safetensors_header(&source).unwrap_err();
    assert!(
        matches!(
            err,
            DownloadError::Header(SafetensorsHeaderError::HeaderTooLarge { .. })
        ),
        "expected HeaderTooLarge, got {err:?}"
    );
    assert_eq!(
        source.reads.get(),
        1,
        "must reject off the length prefix alone, never issuing the (multi-GB) second read"
    );
}
