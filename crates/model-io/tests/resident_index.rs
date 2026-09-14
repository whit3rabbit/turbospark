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

fn set_u64(bytes: &mut [u8], field_offset: usize, value: u64) {
    let start = HEADER_BYTES + field_offset;
    bytes[start..start + 8].copy_from_slice(&value.to_le_bytes());
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

/// `entry_count` near `u64::MAX` must be refused via `checked_mul`/
/// `checked_add`, never wrap into a small, plausible-looking table size.
#[test]
fn load_rejects_an_entry_count_that_overflows_the_table_size() {
    let mut bytes = build_index_bytes("x");
    let entry_count_offset = 16;
    bytes[entry_count_offset..entry_count_offset + 8]
        .copy_from_slice(&(u64::MAX - 1).to_le_bytes());
    let path = write_file(&bytes);
    let err = load_resident_index(&path).unwrap_err();
    assert!(matches!(err, ModelError::IndexCorrupt { .. }));
}

/// `indexSize` (and so `indexSize + residentSize`) past the file's actual
/// length must be refused before it sizes the index-region allocation, not
/// discovered as a short read partway through.
#[test]
fn load_rejects_an_index_size_past_end_of_file() {
    let mut bytes = build_index_bytes("x");
    // Declare an indexSize far larger than this (short) file actually is.
    bytes[0..8].copy_from_slice(&(1u64 << 40).to_le_bytes());
    let path = write_file(&bytes);
    let err = load_resident_index(&path).unwrap_err();
    assert!(matches!(err, ModelError::IndexCorrupt { .. }));
}

/// An entry whose payload extends past `indexSize + residentSize` is
/// refused, whatever the entry table itself looks like.
#[test]
fn load_rejects_a_payload_past_the_resident_region() {
    let mut bytes = build_index_bytes("x");
    // The entry's sizeBytes field sits right after fileOffset (8 bytes in).
    let size_bytes_offset = HEADER_BYTES + 16;
    bytes[size_bytes_offset..size_bytes_offset + 8].copy_from_slice(&(4096u64 * 10).to_le_bytes());
    let path = write_file(&bytes);
    let err = load_resident_index(&path).unwrap_err();
    let ModelError::IndexCorrupt { detail } = err else {
        panic!("expected IndexCorrupt, got {err:?}");
    };
    assert!(detail.contains("payload"), "{detail}");
}

/// A scale plane past the resident region is refused the same way, even
/// though the payload itself is fine -- each companion is checked
/// independently.
#[test]
fn load_rejects_a_scale_plane_past_the_resident_region() {
    let mut bytes = build_index_bytes("x");
    // scaleOffset (8 bytes) then scaleSize (8 bytes), following the four
    // shape u32s at HEADER_BYTES+24.
    let scale_offset_offset = HEADER_BYTES + 40;
    let scale_size_offset = HEADER_BYTES + 48;
    bytes[scale_offset_offset..scale_offset_offset + 8]
        .copy_from_slice(&(1u64 << 20).to_le_bytes());
    bytes[scale_size_offset..scale_size_offset + 8].copy_from_slice(&64u64.to_le_bytes());
    let path = write_file(&bytes);
    let err = load_resident_index(&path).unwrap_err();
    let ModelError::IndexCorrupt { detail } = err else {
        panic!("expected IndexCorrupt, got {err:?}");
    };
    assert!(detail.contains("scale"), "{detail}");
}

/// `file_offset` below `index_size` -- inside the index region itself,
/// rather than the resident data past it -- is refused even when the
/// declared size would otherwise fit inside the file.
#[test]
fn load_rejects_a_payload_offset_inside_the_index_region() {
    let mut bytes = build_index_bytes("x");
    let file_offset_offset = HEADER_BYTES + 8;
    bytes[file_offset_offset..file_offset_offset + 8].copy_from_slice(&0u64.to_le_bytes());
    let path = write_file(&bytes);
    let err = load_resident_index(&path).unwrap_err();
    let ModelError::IndexCorrupt { detail } = err else {
        panic!("expected IndexCorrupt, got {err:?}");
    };
    assert!(detail.contains("payload"), "{detail}");
}

#[test]
fn load_rejects_affine_planes_shorter_than_the_declared_shape_requires() {
    let mut bytes = build_index_bytes("x");
    bytes[HEADER_BYTES + 6] = 4;
    set_u64(&mut bytes, 16, 2048);
    set_u64(&mut bytes, 40, 96 + 2048);
    set_u64(&mut bytes, 48, 2);
    set_u64(&mut bytes, 56, 96 + 2050);
    set_u64(&mut bytes, 64, 2);

    let err = load_resident_index(&write_file(&bytes)).unwrap_err();
    let ModelError::IndexCorrupt { detail } = err else {
        panic!("expected IndexCorrupt, got {err:?}");
    };
    assert!(detail.contains("affine planes"), "{detail}");
}

#[test]
fn load_rejects_affine_metadata_with_a_false_shape_or_unaligned_plane() {
    let mut false_shape = build_index_bytes("x");
    false_shape[HEADER_BYTES + 6] = 4;
    false_shape[HEADER_BYTES + 24..HEADER_BYTES + 28].copy_from_slice(&999u32.to_le_bytes());
    assert!(load_resident_index(&write_file(&false_shape)).is_err());

    let mut unaligned = build_index_bytes("x");
    unaligned[HEADER_BYTES + 6] = 4;
    set_u64(&mut unaligned, 16, 2048);
    set_u64(&mut unaligned, 40, 97);
    set_u64(&mut unaligned, 48, 128);
    set_u64(&mut unaligned, 56, 224);
    set_u64(&mut unaligned, 64, 128);
    assert!(load_resident_index(&write_file(&unaligned)).is_err());
}
