//! Tests for the `model_weights.bin` leading index region decode, using a
//! hand-assembled binary fixture matching the documented header/entry/name
//! layout.

use std::io::Write;

use turbospark_model_io::{load_resident_index, ModelError, ENTRY_BYTES, HEADER_BYTES};

fn build_index_bytes(name: &str) -> Vec<u8> {
    let name_bytes = name.as_bytes();
    let name_offset = HEADER_BYTES + ENTRY_BYTES; // right after the single entry
    let string_table_len = name_bytes.len();
    let index_size = name_offset + string_table_len;

    let mut buf = Vec::new();
    // Header: indexSize, residentSize, entryCount.
    buf.extend_from_slice(&(index_size as u64).to_le_bytes());
    buf.extend_from_slice(&4096u64.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes());
    assert_eq!(buf.len(), HEADER_BYTES);

    // Entry 0.
    buf.extend_from_slice(&(name_offset as u32).to_le_bytes()); // nameOffset
    buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes()); // nameLength
    buf.push(1); // dtype = BF16
    buf.push(0); // reserved
    buf.extend_from_slice(&(index_size as u64).to_le_bytes()); // fileOffset
    buf.extend_from_slice(&4096u64.to_le_bytes()); // sizeBytes
    buf.extend_from_slice(&64u32.to_le_bytes()); // shape.0
    buf.extend_from_slice(&64u32.to_le_bytes()); // shape.1
    buf.extend_from_slice(&1u32.to_le_bytes()); // shape.2
    buf.extend_from_slice(&1u32.to_le_bytes()); // shape.3
    buf.extend_from_slice(&0u64.to_le_bytes()); // scaleOffset
    buf.extend_from_slice(&0u64.to_le_bytes()); // scaleSize
    buf.extend_from_slice(&0u64.to_le_bytes()); // biasOffset
    buf.extend_from_slice(&0u64.to_le_bytes()); // biasSize
    assert_eq!(buf.len(), HEADER_BYTES + ENTRY_BYTES);

    // String table.
    buf.extend_from_slice(name_bytes);
    assert_eq!(buf.len(), index_size);

    // Trailing tensor data region (arbitrary, just enough to not be empty).
    buf.extend_from_slice(&[0u8; 4096]);
    buf
}

fn write_file(bytes: &[u8]) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-resident-index-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model_weights.bin");
    std::fs::File::create(&path)
        .unwrap()
        .write_all(bytes)
        .unwrap();
    path
}

#[test]
fn load_parses_header_and_one_entry() {
    let bytes = build_index_bytes("blk.0.attn.q_proj");
    let path = write_file(&bytes);
    let index = load_resident_index(&path).unwrap();

    assert_eq!(index.header.entry_count, 1);
    let entry = &index.entries["blk.0.attn.q_proj"];
    assert_eq!(entry.dtype, 1);
    assert_eq!(entry.shape, (64, 64, 1, 1));
    assert_eq!(entry.size_bytes, 4096);
}

#[test]
fn load_rejects_short_header() {
    let path = write_file(&[0u8; 4]);
    let err = load_resident_index(&path).unwrap_err();
    assert!(matches!(err, ModelError::IndexCorrupt { .. }));
}

#[test]
fn load_rejects_name_range_out_of_bounds() {
    let mut bytes = build_index_bytes("x");
    // Corrupt the name length field (byte offset 4..6 of the single entry)
    // to point past the index region.
    let name_len_offset = HEADER_BYTES + 4;
    bytes[name_len_offset..name_len_offset + 2].copy_from_slice(&9999u16.to_le_bytes());
    let path = write_file(&bytes);
    let err = load_resident_index(&path).unwrap_err();
    assert!(matches!(err, ModelError::IndexCorrupt { .. }));
}
