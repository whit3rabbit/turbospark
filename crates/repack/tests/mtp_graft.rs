//! `resident_reader::read_resident_entries` and `graft_qwen_gdn_dense_mtp_head`
//! (`docs/MTP_SPECULATIVE.md`'s graft step): adding a multi-token-prediction
//! head to an install already on disk by reading its resident entries back
//! byte for byte, rather than re-streaming its trunk over the network.
//!
//! No network here -- both fixtures are synthetic, matching this crate's
//! fixture-before-download rule (`crates/repack/CLAUDE.md` Gotcha 8). The
//! real-network case (a real `qwen38-27b` trunk plus the real official-repo
//! head) was verified end to end while building the catalog's own
//! cross-repo pull; what this file covers is the NEW code the catalog path
//! does not exercise -- reading raw bytes back off an on-disk resident index.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use turbospark_repack::{
    build_resident_weights_bin_mixed, build_synthetic_qwen_gdn_dense_install,
    build_synthetic_qwen_gdn_dense_install_at_bits, graft_qwen_gdn_dense_mtp_head,
    read_resident_entries, Gemma4Quant, Gemma4Shards, MemoryRangeSource, RangeSource,
    SafetensorsHeader,
};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("turbospark-mtp-graft-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

const VOCAB: i64 = 256;
const LAYERS: i64 = 4;

fn build_headless() -> (PathBuf, model_io::ArchConfig) {
    let dir = temp_dir();
    let arch = build_synthetic_qwen_gdn_dense_install(&dir, VOCAB, LAYERS, "qwen35-toy")
        .expect("the headless dense install writes");
    (dir, arch)
}

/// The exact inverse property `build_resident_weights_bin_mixed` needs to be
/// true for grafting to be safe at all: read a real install's resident
/// region back into specs, rebuild it, and get the SAME bytes -- not merely
/// an install of the same size.
#[test]
fn read_resident_entries_round_trips_a_synthetic_install_byte_for_byte() {
    let (dir, _arch) = build_headless();
    let original = std::fs::read(dir.join("model_weights.bin")).expect("read original");

    let entries = read_resident_entries(&dir).expect("read the resident entries back");
    assert!(!entries.is_empty(), "a dense install has resident tensors");

    let rebuilt = build_resident_weights_bin_mixed(&entries);
    assert_eq!(
        rebuilt, original,
        "reading an install's resident entries back and rebuilding them must reproduce \
         the exact bytes on disk, or grafting a head onto them would silently corrupt \
         the trunk"
    );
}

/// The 1-bit dense fixture above exercises only `DTYPE_BF16` and
/// `DTYPE_INT1_AFFINE` (its every routed-style tensor is 1-bit and its
/// norms are BF16). The real production caller
/// (`graft_qwen_gdn_dense_mtp_head` grafting onto a real `qwen38-27b`
/// trunk) runs the INT4 arm, which had no coverage here at all.
#[test]
fn read_resident_entries_round_trips_an_int4_dense_install_byte_for_byte() {
    let dir = temp_dir();
    build_synthetic_qwen_gdn_dense_install_at_bits(&dir, VOCAB, LAYERS, "qwen35-toy-int4", 4)
        .expect("the int4 dense install writes");
    let original = std::fs::read(dir.join("model_weights.bin")).expect("read original");

    let entries = read_resident_entries(&dir).expect("read the resident entries back");
    assert!(!entries.is_empty());
    assert!(
        entries
            .iter()
            .any(|e| matches!(e, turbospark_repack::ResidentEntrySpec::Int4(_))),
        "an int4 install must exercise the DTYPE_INT4_AFFINE arm"
    );

    let rebuilt = build_resident_weights_bin_mixed(&entries);
    assert_eq!(rebuilt, original, "int4 round trip must be byte-exact");
}

/// Same property again for `DTYPE_INT8_AFFINE`, which the dense fixtures
/// above never reach at all -- only a MoE install's router and
/// sigmoid-gated shared-expert gate are INT8.
#[test]
fn read_resident_entries_round_trips_a_moe_installs_int8_router_byte_for_byte() {
    let dir = temp_dir();
    turbospark_repack::build_synthetic_qwen_gdn_moe_install(&dir, VOCAB, LAYERS, 4, "qwen36-toy")
        .expect("the moe install writes");
    let original = std::fs::read(dir.join("model_weights.bin")).expect("read original");

    let entries = read_resident_entries(&dir).expect("read the resident entries back");
    assert!(
        entries
            .iter()
            .any(|e| matches!(e, turbospark_repack::ResidentEntrySpec::Int8(_))),
        "a MoE install's router must exercise the DTYPE_INT8_AFFINE arm"
    );

    let rebuilt = build_resident_weights_bin_mixed(&entries);
    assert_eq!(rebuilt, original, "int8 round trip must be byte-exact");
}

/// A GGUF-block dtype (here Q8_0) has no `ResidentEntrySpec` variant to
/// round-trip through, and this must fail loudly by name rather than be
/// silently skipped or misread as one of the affine layouts. Patches a real
/// install's first resident entry in place, which is enough: `load` reads
/// `dtype` as a raw byte with no validation, so the reader's own dispatch is
/// what has to refuse it.
#[test]
fn refuses_a_gguf_block_dtype_by_name() {
    let dir = temp_dir();
    build_synthetic_qwen_gdn_dense_install_at_bits(&dir, VOCAB, LAYERS, "qwen35-toy-q8", 4)
        .expect("the int4 dense install writes");

    let weights_path = dir.join("model_weights.bin");
    let mut bytes = std::fs::read(&weights_path).expect("read model_weights.bin");
    // Header is 24 bytes, entry table starts there; each entry is 72 bytes
    // with `dtype` at offset 6 (`resident_writer.rs`'s `out[base + 6]`).
    const HEADER_BYTES: usize = 24;
    const DTYPE_GGUF_Q8_0: u8 = 6;
    bytes[HEADER_BYTES + 6] = DTYPE_GGUF_Q8_0;
    std::fs::write(&weights_path, &bytes).expect("patch dtype");

    let err = read_resident_entries(&dir).unwrap_err();
    assert!(
        err.contains("GGUF block type"),
        "expected the GGUF-block refusal, got: {err}"
    );
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for v in values {
        let bits = (v.to_bits() >> 16) as u16;
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

/// A minimal two-tensor synthetic "head": one rank-2 projection (quantized)
/// and one rank-1 norm (narrowed), which is the whole split
/// `mtp::read_mtp_entries` keys on. Real published heads have fifteen
/// tensors; this fixture exists to prove the GRAFT WIRING rather than to
/// restate the head's real shape, which `synthetic_qwen35.rs`'s
/// `build_with_mtp` already covers through the ordinary (non-graft) path.
fn synthetic_head() -> (SafetensorsHeader, Vec<u8>) {
    // `quantize_matrix_int4` needs every row to be a whole number of GROUPS
    // (64 elements), so the projection is 2 rows of 64 rather than something
    // smaller and more readable.
    const COLS: usize = 64;
    let proj_values: Vec<f32> = (0..2 * COLS)
        .map(|i| (i as f32 - COLS as f32) / COLS as f32)
        .collect();
    let norm_values: Vec<f32> = vec![1.0; COLS];
    let proj_bytes = bf16_bytes(&proj_values);
    let norm_bytes = bf16_bytes(&norm_values);

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "mtp.fake_proj.weight".to_string(),
        turbospark_repack::TensorInfo {
            dtype: "BF16".to_string(),
            shape: vec![2, COLS as u64],
            data_offsets: (0, proj_bytes.len() as u64),
        },
    );
    tensors.insert(
        "mtp.fake_norm.weight".to_string(),
        turbospark_repack::TensorInfo {
            dtype: "BF16".to_string(),
            shape: vec![COLS as u64],
            data_offsets: (
                proj_bytes.len() as u64,
                (proj_bytes.len() + norm_bytes.len()) as u64,
            ),
        },
    );
    let header = SafetensorsHeader {
        tensors,
        metadata: None,
        header_len: 0,
    };
    // `data_region_start()` is `8 + header_len`: every real safetensors file
    // has an 8-byte length prefix ahead of its JSON header, and tensor
    // offsets are relative to the END of that header. `header_len: 0` still
    // needs the 8-byte prefix accounted for in the underlying source, or
    // `absolute_range` points 8 bytes past this buffer's actual data.
    let mut data = vec![0u8; 8];
    data.extend_from_slice(&proj_bytes);
    data.extend_from_slice(&norm_bytes);
    (header, data)
}

/// The graft itself: an existing headless install plus a synthetic head,
/// combined WITHOUT touching the trunk's own bytes.
#[test]
fn graft_adds_a_head_and_leaves_the_trunk_untouched() {
    let (existing_dir, arch) = build_headless();
    let original_entries = read_resident_entries(&existing_dir).expect("read trunk entries");

    let (header, data) = synthetic_head();
    let source = MemoryRangeSource::new(&data);
    let shards = Gemma4Shards::single(&header, &source as &dyn RangeSource);
    let mtp_bases = ["mtp.fake_norm.weight", "mtp.fake_proj.weight"];

    let out_dir = temp_dir();
    let quant = Gemma4Quant::default();
    graft_qwen_gdn_dense_mtp_head(
        &existing_dir,
        &out_dir,
        &arch,
        "qwen35-toy-mtp",
        &shards,
        &mtp_bases,
        &quant,
        |_stage| {},
    )
    .expect("the graft writes");

    model_io::load_manifest(&out_dir, &arch, model_io::DEFAULT_MAX_BYTES)
        .expect("the grafted install's manifest validates");
    let grafted = model_io::load_resident_index(&out_dir.join("model_weights.bin"))
        .expect("the grafted resident index loads");

    // The two new tensors are there, and nothing else `mtp.`-prefixed is.
    for name in mtp_bases {
        assert!(
            grafted.entries.contains_key(name),
            "grafted install is missing {name}"
        );
    }
    assert_eq!(
        grafted
            .entries
            .keys()
            .filter(|k| k.starts_with("mtp."))
            .count(),
        mtp_bases.len(),
        "the grafted install carries an unexpected mtp.-prefixed tensor"
    );

    // Every ORIGINAL trunk tensor survived, unchanged, at the SAME dtype and
    // the SAME byte length -- the whole point of reusing rather than
    // re-streaming it.
    assert_eq!(
        grafted.entries.len(),
        original_entries.len() + mtp_bases.len(),
        "grafted entry count should be the trunk's plus the head's, no more and no less"
    );
    let existing_index = model_io::load_resident_index(&existing_dir.join("model_weights.bin"))
        .expect("original resident index loads");
    for (name, entry) in &existing_index.entries {
        let grafted_entry = grafted
            .entries
            .get(name)
            .unwrap_or_else(|| panic!("grafted install lost trunk tensor {name}"));
        assert_eq!(grafted_entry.dtype, entry.dtype, "{name} changed dtype");
        assert_eq!(
            grafted_entry.size_bytes, entry.size_bytes,
            "{name} changed size"
        );
    }
}

/// An install that already carries a head is refused rather than silently
/// growing a second one under the same names.
#[test]
fn grafting_onto_an_already_headed_install_is_refused() {
    let (existing_dir, arch) = build_headless();
    let (header, data) = synthetic_head();
    let source = MemoryRangeSource::new(&data);
    let shards = Gemma4Shards::single(&header, &source as &dyn RangeSource);
    let mtp_bases = ["mtp.fake_norm.weight", "mtp.fake_proj.weight"];
    let quant = Gemma4Quant::default();

    let out_dir = temp_dir();
    graft_qwen_gdn_dense_mtp_head(
        &existing_dir,
        &out_dir,
        &arch,
        "qwen35-toy-mtp",
        &shards,
        &mtp_bases,
        &quant,
        |_| {},
    )
    .expect("the first graft writes");

    let second_out = temp_dir();
    let err = graft_qwen_gdn_dense_mtp_head(
        &out_dir,
        &second_out,
        &arch,
        "qwen35-toy-mtp-2",
        &shards,
        &mtp_bases,
        &quant,
        |_| {},
    )
    .expect_err("grafting onto an already-headed install must be refused");
    assert!(
        err.to_string().contains("already carries an MTP head"),
        "unexpected refusal message: {err}"
    );
}
