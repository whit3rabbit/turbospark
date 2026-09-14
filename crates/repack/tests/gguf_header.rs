//! GGUF v3 header parsing: round trip against the synthetic writer, and the
//! rejection cases that keep a corrupt or hostile file from being read as a
//! plausible one.

use std::cell::Cell;

use std::collections::BTreeMap;

use turbospark_repack::{
    fetch_gguf_header, ggml_type_block, ggml_type_name, parse_gguf_header, DownloadError,
    GgufBuilder, GgufHeader, GgufHeaderError, GgufTensorInfo, GgufValue, RangeSource,
    GGUF_DEFAULT_MAX_HEADER_BYTES, GGUF_INITIAL_FETCH_BYTES,
};

/// A fixture exercising every value kind the reader implements, plus three
/// tensors of two different element types.
fn fixture() -> turbospark_repack::GgufFileAndRanges {
    GgufBuilder::new()
        .metadata_str("general.architecture", "gemma4")
        .metadata_u32("gemma4.block_count", 2)
        .metadata_f32("gemma4.attention.layer_norm_rms_epsilon", 1e-6)
        .metadata("general.file_type", GgufValue::U8(7))
        .metadata("truncated", GgufValue::Bool(true))
        .metadata("signed", GgufValue::I32(-3))
        .metadata("wide", GgufValue::U64(1 << 40))
        .metadata(
            "tokenizer.ggml.tokens",
            GgufValue::Array(vec![
                GgufValue::String("<pad>".to_string()),
                GgufValue::String("hi".to_string()),
            ]),
        )
        .metadata(
            "nested",
            GgufValue::Array(vec![GgufValue::Array(vec![
                GgufValue::U32(1),
                GgufValue::U32(2),
            ])]),
        )
        .q8_0_tensor("blk.0.ffn_gate_exps.weight", &[64, 2, 4], 1)
        .q8_0_tensor("blk.0.attn_q.weight", &[64, 32], 9)
        .tensor("output_norm.weight", 0, &[8], vec![0u8; 32])
        .build()
}

#[test]
fn round_trips_metadata_tensors_and_layout() {
    let (bytes, ranges) = fixture();
    let h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).expect("parse");

    assert_eq!(h.version, 3);
    assert_eq!(h.architecture(), Some("gemma4"));
    assert_eq!(h.metadata_u64("gemma4.block_count"), Some(2));
    assert_eq!(h.metadata_u64("wide"), Some(1 << 40));
    assert_eq!(h.metadata_u64("signed"), None, "negative must not wrap");
    assert_eq!(
        h.metadata
            .get("gemma4.attention.layer_norm_rms_epsilon")
            .and_then(GgufValue::as_f64),
        Some(f64::from(1e-6f32))
    );
    assert_eq!(
        h.metadata.get("truncated").and_then(GgufValue::as_bool),
        Some(true)
    );

    let tokens = h
        .metadata
        .get("tokenizer.ggml.tokens")
        .and_then(GgufValue::as_array)
        .expect("token array");
    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[1].as_str(), Some("hi"));

    let nested = h
        .metadata
        .get("nested")
        .and_then(GgufValue::as_array)
        .expect("nested array");
    assert_eq!(nested[0].as_array().map(<[GgufValue]>::len), Some(2));

    // No general.alignment was written, so the spec default applies.
    assert_eq!(h.alignment, 32);
    assert_eq!(h.data_region_start % 32, 0);

    assert_eq!(h.tensors.len(), 3);
    let gate = &h.tensors["blk.0.ffn_gate_exps.weight"];
    assert_eq!(gate.ggml_type, 8, "Q8_0");
    assert_eq!(gate.dims, vec![64, 2, 4]);
    assert_eq!(gate.element_count(), 512);
    assert_eq!(
        gate.byte_size("blk.0.ffn_gate_exps.weight").unwrap(),
        512 / 32 * 34
    );

    // The ranges the reader derives must match the ones the writer laid
    // out, and the bytes there must be the tensor's own. This is the
    // property the repack walk depends on for losslessness.
    for (name, (start, end)) in ranges {
        let derived = h
            .absolute_range(&name)
            .expect("known tensor")
            .expect("size");
        assert_eq!(derived, (start, end), "{name} range");
        assert!(end as usize <= bytes.len());
    }
}

#[test]
fn honours_a_non_default_alignment() {
    let (bytes, ranges) = GgufBuilder::new()
        .with_alignment(4096)
        .metadata_str("general.architecture", "qwen36")
        .q8_0_tensor("token_embd.weight", &[64, 4], 2)
        .q8_0_tensor("output.weight", &[64, 4], 3)
        .build();
    let h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).expect("parse");

    assert_eq!(h.alignment, 4096);
    assert_eq!(h.data_region_start % 4096, 0);
    for (name, expected) in ranges {
        let derived = h.absolute_range(&name).unwrap().unwrap();
        assert_eq!(derived, expected, "{name} range under 4096-byte alignment");
    }
}

/// The contract the ranged fetcher is built on: a short buffer reports the
/// absolute offset it wanted, and feeding exactly that back makes progress
/// until the parse completes.
#[test]
fn short_buffers_report_the_offset_they_need() {
    let (bytes, _) = fixture();

    let mut have = 8usize;
    let mut rounds = 0;
    loop {
        match parse_gguf_header(&bytes[..have], GGUF_DEFAULT_MAX_HEADER_BYTES) {
            Ok(h) => {
                assert_eq!(h.tensors.len(), 3);
                break;
            }
            Err(GgufHeaderError::TooShort { needed }) => {
                assert!(
                    needed as usize > have,
                    "needed {needed} must advance past the {have} bytes already held"
                );
                have = needed as usize;
                rounds += 1;
                assert!(rounds < 5000, "TooShort loop did not converge");
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }
    assert!(rounds > 0, "the fixture must actually exercise the path");
}

#[test]
fn rejects_a_non_gguf_file() {
    let mut bytes = fixture().0;
    bytes[0] = b'X';
    assert!(matches!(
        parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES),
        Err(GgufHeaderError::BadMagic { .. })
    ));
}

#[test]
fn rejects_other_versions_rather_than_misparsing_them() {
    let mut bytes = fixture().0;
    bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
    assert!(matches!(
        parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES),
        Err(GgufHeaderError::UnsupportedVersion { version: 2 })
    ));
}

#[test]
fn rejects_counts_that_cannot_fit_the_cap() {
    let mut bytes = fixture().0;
    bytes[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(matches!(
        parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES),
        Err(GgufHeaderError::HeaderTooLarge { .. })
    ));
}

#[test]
fn rejects_an_alignment_that_is_not_a_power_of_two() {
    let (mut bytes, _) = GgufBuilder::new()
        .with_alignment(64)
        .q8_0_tensor("token_embd.weight", &[64], 1)
        .build();

    // Patch the written general.alignment value in place: 8-byte key
    // length, the key, a 4-byte type id, then the u32 payload.
    let key = b"general.alignment";
    let at = bytes
        .windows(key.len())
        .position(|w| w == key)
        .expect("key present");
    let value_at = at + key.len() + 4;
    bytes[value_at..value_at + 4].copy_from_slice(&48u32.to_le_bytes());

    assert!(matches!(
        parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES),
        Err(GgufHeaderError::BadAlignment { alignment: 48 })
    ));
}

#[test]
fn rejects_a_duplicate_tensor_name() {
    let (bytes, _) = GgufBuilder::new()
        .q8_0_tensor("blk.0.attn_q.weight", &[64], 1)
        .q8_0_tensor("blk.0.attn_q.weight", &[64], 2)
        .build();
    assert!(matches!(
        parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES),
        Err(GgufHeaderError::DuplicateTensor { .. })
    ));
}

/// A type whose block size this port has not verified must be named, not
/// guessed at. TQ1_0 is a real ggml type with a real id; what is missing
/// is only its byte size.
///
/// THE EXEMPLAR HAS TO BE RE-PICKED whenever a type gains a
/// `ggml_type_block` row, and the failure is this test rather than anything
/// subtle: it read Q5_K until Phase S's header probes added that row. Pick a
/// replacement that is real, unhandled, and unlikely to land soon. TQ1_0
/// qualifies: it is a distinct ternary layout and no current candidate uses it.
#[test]
fn names_an_unsupported_but_real_ggml_type() {
    let (bytes, _) = GgufBuilder::new()
        .tensor("blk.0.attn_q.weight", 34, &[256], vec![0u8; 1])
        .build();
    let h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).expect("header parses");
    match h.absolute_range("blk.0.attn_q.weight").unwrap() {
        Err(GgufHeaderError::UnsupportedType { name, id }) => {
            assert_eq!((name.as_str(), id), ("TQ1_0", 34));
        }
        other => panic!("expected UnsupportedType, got {other:?}"),
    }
}

#[test]
fn rejects_an_element_count_that_is_not_whole_blocks() {
    let (bytes, _) = GgufBuilder::new()
        .tensor("blk.0.attn_q.weight", 8, &[40], vec![0u8; 34])
        .build();
    let h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).expect("header parses");
    assert!(matches!(
        h.absolute_range("blk.0.attn_q.weight").unwrap(),
        Err(GgufHeaderError::RaggedTensor { block: 32, .. })
    ));
}

/// Two dimensions whose product overflows `u64` must be reported as
/// `Overflow`, not wrapped into a small, plausible-looking element count
/// that then under-reads the file.
#[test]
fn byte_size_rejects_a_dimension_product_that_overflows() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "blk.0.huge.weight".to_string(),
        GgufTensorInfo {
            ggml_type: 0, // F32
            dims: vec![u64::MAX, 2],
            offset: 0,
        },
    );
    let header = GgufHeader {
        version: 3,
        metadata: BTreeMap::new(),
        tensors,
        alignment: 32,
        data_region_start: 32,
    };
    assert!(matches!(
        header.absolute_range("blk.0.huge.weight").unwrap(),
        Err(GgufHeaderError::Overflow { .. })
    ));
}

/// An offset near `u64::MAX` plus a small `data_region_start` must not wrap
/// into a small, in-bounds-looking range; `absolute_range` must report the
/// overflow rather than silently address the wrong bytes.
#[test]
fn absolute_range_rejects_an_offset_that_overflows_the_data_region_start() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "blk.0.attn_q.weight".to_string(),
        GgufTensorInfo {
            ggml_type: 0, // F32, block size (1, 4)
            dims: vec![1],
            offset: u64::MAX - 1,
        },
    );
    let header = GgufHeader {
        version: 3,
        metadata: BTreeMap::new(),
        tensors,
        alignment: 32,
        data_region_start: 32,
    };
    assert!(matches!(
        header.absolute_range("blk.0.attn_q.weight").unwrap(),
        Err(GgufHeaderError::Overflow { .. })
    ));
}

/// The spec requires every tensor's offset to be a multiple of the header's
/// own alignment; a corrupt or hostile file that violates it must be
/// refused at parse time rather than accepted with a data region that
/// straddles a tensor's own bytes.
#[test]
fn rejects_a_tensor_offset_that_is_not_aligned() {
    let (mut bytes, _) = GgufBuilder::new()
        .tensor("blk.0.a.weight", 0, &[1], vec![0u8; 4])
        .tensor("blk.0.b.weight", 0, &[1], vec![0u8; 4])
        .build();

    // Patch the second tensor's 8-byte offset field (the last 8 bytes of its
    // entry, written last) to something not a multiple of the default
    // 32-byte alignment. It was 32 (the first tensor rounds up to one
    // alignment unit); shifting it by one keeps it in range but misaligned.
    let key = b"blk.0.b.weight";
    let at = bytes
        .windows(key.len())
        .position(|w| w == key)
        .expect("name present");
    // 8 (name len) + name + 4 (n_dims) + 8 (one dim) + 4 (ggml_type) = start
    // of the 8-byte offset field.
    let offset_at = at + key.len() + 4 + 8 + 4;
    let original = u64::from_le_bytes(bytes[offset_at..offset_at + 8].try_into().unwrap());
    assert_ne!(original, 0, "fixture must not already sit at offset 0");
    bytes[offset_at..offset_at + 8].copy_from_slice(&(original + 1).to_le_bytes());

    assert!(matches!(
        parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES),
        Err(GgufHeaderError::MisalignedTensor { .. })
    ));
}

#[test]
fn rejects_a_zero_tensor_dimension() {
    let (bytes, _) = GgufBuilder::new()
        .tensor("blk.0.attn_gate.weight", 0, &[0], Vec::new())
        .build();

    assert!(matches!(
        parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES),
        Err(GgufHeaderError::BadDimensions { .. })
    ));
}

/// Counts reads, so the growth policy can be asserted rather than assumed:
/// the point of [`fetch_gguf_header`] is that it does NOT make one request
/// per header field.
struct CountingSource {
    data: Vec<u8>,
    reads: Cell<usize>,
    bytes: Cell<u64>,
}

impl CountingSource {
    fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            reads: Cell::new(0),
            bytes: Cell::new(0),
        }
    }
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
        self.bytes.set(self.bytes.get() + (end - start) as u64);
        Ok(self.data[start..end].to_vec())
    }
}

/// A header bigger than the first speculative read, so the growth loop runs
/// for real. Built from a token list, which is what makes real headers big.
fn oversized_fixture() -> Vec<u8> {
    let tokens: Vec<GgufValue> = (0..60_000)
        .map(|i| GgufValue::String(format!("token_{i:06}")))
        .collect();
    GgufBuilder::new()
        .metadata_str("general.architecture", "gemma4")
        .metadata("tokenizer.ggml.tokens", GgufValue::Array(tokens))
        .q8_0_tensor("token_embd.weight", &[64, 4], 1)
        .build()
        .0
}

#[test]
fn fetches_a_small_header_without_overreading() {
    let source = CountingSource::new(fixture().0);
    let h = fetch_gguf_header(&source).expect("fetch");
    assert_eq!(h.tensors.len(), 3);
    // One speculative read past EOF, then one bounded by the real length.
    assert!(source.reads.get() <= 2, "{} reads", source.reads.get());
}

#[test]
fn grows_geometrically_for_a_large_header() {
    let bytes = oversized_fixture();
    assert!(
        bytes.len() as u64 > GGUF_INITIAL_FETCH_BYTES,
        "fixture must exceed the first read to exercise growth"
    );
    let source = CountingSource::new(bytes);
    let h = fetch_gguf_header(&source).expect("fetch");
    assert_eq!(h.architecture(), Some("gemma4"));
    // Geometric growth, not one request per field: a header a few MB long
    // must not cost thousands of reads.
    assert!(
        (2..=6).contains(&source.reads.get()),
        "{} reads is outside the geometric-growth band",
        source.reads.get()
    );
}

#[test]
fn reports_a_file_that_ends_inside_its_own_header() {
    let mut bytes = fixture().0;
    bytes.truncate(40);
    let source = CountingSource::new(bytes);
    assert!(matches!(
        fetch_gguf_header(&source),
        Err(DownloadError::TruncatedGguf { .. })
    ));
}

/// The block table is deliberately partial, but every id it answers for must
/// agree with the ggml spec, and every id it declines must still have a name
/// so the error can say which type it was.
#[test]
fn block_table_matches_the_spec_where_it_answers() {
    assert_eq!(ggml_type_block(0), Some((1, 4)), "F32");
    assert_eq!(ggml_type_block(1), Some((1, 2)), "F16");
    assert_eq!(ggml_type_block(2), Some((32, 18)), "Q4_0");
    assert_eq!(ggml_type_block(8), Some((32, 34)), "Q8_0");
    assert_eq!(ggml_type_block(12), Some((256, 144)), "Q4_K");
    assert_eq!(ggml_type_block(14), Some((256, 210)), "Q6_K");
    assert_eq!(ggml_type_block(10), Some((256, 84)), "Q2_K");
    assert_eq!(ggml_type_block(16), Some((256, 66)), "IQ2_XXS");
    assert_eq!(ggml_type_block(17), Some((256, 74)), "IQ2_XS");
    assert_eq!(ggml_type_block(19), Some((256, 50)), "IQ1_S");
    assert_eq!(ggml_type_block(21), Some((256, 110)), "IQ3_S");
    assert_eq!(ggml_type_block(22), Some((256, 82)), "IQ2_S");
    assert_eq!(ggml_type_block(29), Some((256, 56)), "IQ1_M");
    assert_eq!(ggml_type_block(30), Some((1, 2)), "BF16");

    assert_eq!(ggml_type_name(8), Some("Q8_0"));
    assert_eq!(ggml_type_name(12), Some("Q4_K"));
    assert_eq!(ggml_type_name(30), Some("BF16"));
    // 4 and 5 are removed types and must stay unnamed.
    assert_eq!(ggml_type_name(4), None);
    assert_eq!(ggml_type_name(5), None);
    assert_eq!(ggml_type_name(99), None);
}

/// The parser's idea of a Q8_0 block and the CPU reference's must not drift
/// apart. `turbospark_compute` cannot depend on this crate (the dependency runs
/// the other way), so the two declare the constants independently and this is
/// where they are held to each other. A disagreement would show up as every
/// tensor after the first being read from the wrong offset.
#[test]
fn the_q8_0_block_matches_the_cpu_reference() {
    assert_eq!(
        ggml_type_block(8),
        Some((
            compute::Q8_0_BLOCK_ELEMS as u64,
            compute::Q8_0_BLOCK_BYTES as u64
        ))
    );
}

/// Same standing agreement for Q4_K, and it matters more here: a superblock
/// is 144 bytes over 256 elements, so a disagreement of one byte shifts every
/// later tensor by thousands.
#[test]
fn the_q4_k_block_matches_the_cpu_reference() {
    assert_eq!(
        ggml_type_block(12),
        Some((
            compute::Q4_K_BLOCK_ELEMS as u64,
            compute::Q4_K_BLOCK_BYTES as u64
        ))
    );
}

/// The same agreement for the three IQ types (ROADMAP Phase S). These rows
/// predate their CPU reference by a session -- they were added parse-only so
/// candidate checkpoints could be header-probed -- so this is the first thing
/// that holds the parser's numbers and the decoder's to each other.
#[test]
fn the_iq_blocks_match_the_cpu_reference() {
    for (id, elems, bytes, name) in [
        (
            16,
            compute::IQ_LOWBIT_BLOCK_ELEMS,
            compute::IQ2_XXS_BLOCK_BYTES,
            "IQ2_XXS",
        ),
        (
            17,
            compute::IQ_LOWBIT_BLOCK_ELEMS,
            compute::IQ2_XS_BLOCK_BYTES,
            "IQ2_XS",
        ),
        (
            19,
            compute::IQ_LOWBIT_BLOCK_ELEMS,
            compute::IQ1_S_BLOCK_BYTES,
            "IQ1_S",
        ),
        (
            21,
            compute::IQ_LOWBIT_BLOCK_ELEMS,
            compute::IQ3_S_BLOCK_BYTES,
            "IQ3_S",
        ),
        (
            22,
            compute::IQ_LOWBIT_BLOCK_ELEMS,
            compute::IQ2_S_BLOCK_BYTES,
            "IQ2_S",
        ),
        (
            29,
            compute::IQ_LOWBIT_BLOCK_ELEMS,
            compute::IQ1_M_BLOCK_BYTES,
            "IQ1_M",
        ),
    ] {
        assert_eq!(
            ggml_type_block(id),
            Some((elems as u64, bytes as u64)),
            "{name}"
        );
    }
    assert_eq!(
        ggml_type_block(18),
        Some((
            compute::IQ3_XXS_BLOCK_ELEMS as u64,
            compute::IQ3_XXS_BLOCK_BYTES as u64
        )),
        "IQ3_XXS"
    );
    assert_eq!(
        ggml_type_block(20),
        Some((
            compute::IQ4_NL_BLOCK_ELEMS as u64,
            compute::IQ4_NL_BLOCK_BYTES as u64
        )),
        "IQ4_NL"
    );
    assert_eq!(
        ggml_type_block(23),
        Some((
            compute::IQ4_XS_BLOCK_ELEMS as u64,
            compute::IQ4_XS_BLOCK_BYTES as u64
        )),
        "IQ4_XS"
    );
}
