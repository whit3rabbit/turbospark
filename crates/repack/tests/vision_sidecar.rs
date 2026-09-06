//! The vision-tower SIDECAR format (vision memory sidecar, part A1): a tower
//! installed once as its own small `<alias>.gturbo-vision/` directory rather
//! than bundled inside a full trunk install.
//!
//! What this file can and cannot see, following `synthetic_qwen35_vision.rs`'s
//! own framing one level over: it CAN see that the synthetic sidecar loads
//! through the real manifest validator, that a plain trunk install and a
//! sidecar install are distinguishable by `is_sidecar_dir`, that the
//! sidecar's bytes are IDENTICAL to the combined install's tower bytes (the
//! load-bearing property of the whole design), and that a `model.visual.`
//! -prefixed source canonicalizes onto the same bytes a `vision_tower.`
//! -prefixed one produces. It CANNOT see anything about attaching a sidecar
//! to a trunk at runtime -- that binding is a later part (A2) of this
//! feature and is out of scope here.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use turbospark_repack::{
    build_synthetic_qwen_gdn_dense_install, build_synthetic_qwen_gdn_dense_install_with_vision,
    build_synthetic_vision_sidecar, canonicalize_vision_header, read_vision_entries,
    write_packed_vision, Gemma4Shards, MemoryRangeSource, ResidentEntrySpec, SafetensorsHeader,
    TensorInfo, VISION_BLOCK_ROLES, VISION_PREFIX, VISION_RESIDENT_TENSORS,
};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-vision-sidecar-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

const VOCAB: i64 = 256;
const LAYERS: i64 = 4;
const BITS: u32 = 1;

/// The synthetic sidecar loads end to end: `vision_sidecar.json` parses,
/// declares `vision-tower`, and the arch it implies passes the real
/// `manifest.json` loader/validator.
#[test]
fn a_synthetic_sidecar_loads_through_the_real_validator() {
    let dir = temp_dir();
    let vision = build_synthetic_vision_sidecar(&dir, "vision-sidecar-toy")
        .expect("the sidecar-only fixture writes");

    let (record, loaded) =
        model_io::load_vision_sidecar(&dir).expect("a synthetic sidecar loads end to end");
    assert_eq!(record.kind, model_io::SIDECAR_KIND);
    assert_eq!(record.pairs_with.hidden_size, vision.out_hidden_size);
    assert_eq!(record.tower_blocks, vision.depth);
    assert_eq!(
        loaded, vision,
        "load() must return the same tower the writer wrote"
    );

    assert!(
        model_io::is_sidecar_dir(&dir),
        "a directory load() accepts must also satisfy is_sidecar_dir"
    );
}

/// A plain trunk install (no `vision_sidecar.json` at all) is not a sidecar.
#[test]
fn a_plain_trunk_install_is_not_a_sidecar() {
    let dir = temp_dir();
    build_synthetic_qwen_gdn_dense_install(&dir, VOCAB, LAYERS, "trunk-toy")
        .expect("a text-only trunk install writes");
    assert!(
        !model_io::is_sidecar_dir(&dir),
        "a plain trunk install must not read as a sidecar"
    );
}

/// A sidecar directory's manifest is a valid (if degenerate) `manifest.json`,
/// but it cannot pass today's real-model validator AS a full trunk install of
/// its own paired family: `sidecar_arch` zeroes `numLayers` and the empty
/// `fullAttentionLayerMask`, both of which a real per-family baseline
/// declares non-trivially, so the field-by-field comparison refuses it.
///
/// This is the extent to which "refuse opening a sidecar as a model" is
/// covered here -- a DEDICATED "is this dir a real model" check (as opposed
/// to arch validation naturally disagreeing) is runtime wiring for a later
/// part (A2), not a format question this crate settles.
#[test]
fn opening_a_sidecar_as_a_full_trunk_install_fails_arch_validation() {
    let dir = temp_dir();
    build_synthetic_vision_sidecar(&dir, "vision-sidecar-toy").expect("sidecar writes");

    let trunk_arch = model_io::known_architecture(model_io::ModelFamily::QwenGdnDense);
    let err = model_io::load_manifest(&dir, &trunk_arch, 4 * 1024 * 1024)
        .expect_err("a sidecar's degenerate manifest must not validate as a full trunk install");
    assert!(
        matches!(err, model_io::ModelError::ArchMismatch { .. }),
        "expected an ArchMismatch opening a sidecar as a full trunk install, got {err:?}"
    );
}

/// **THE LOAD-BEARING ASSERTION FOR THE WHOLE DESIGN.** `packed_vision/blobs.bin`
/// and the nine resident tensors' bytes must be IDENTICAL between "tower
/// baked into a full install" and "tower alone in a sidecar", because both
/// are built from the exact same `vision_tower_tensors()` fixture through the
/// exact same `read_vision_entries` / `write_packed_vision` machinery. If a
/// caller ever streams a tower once and attaches it to two different trunks
/// (a later part's whole point), the sidecar's bytes have to be the same
/// bytes a from-scratch combined install of either trunk would have written.
#[test]
fn the_sidecar_tower_bytes_match_the_combined_installs_tower_bytes() {
    let combined_dir = temp_dir();
    build_synthetic_qwen_gdn_dense_install_with_vision(
        &combined_dir,
        VOCAB,
        LAYERS,
        "combined-toy",
        BITS,
    )
    .expect("a combined install with a vision tower writes");

    let sidecar_dir = temp_dir();
    build_synthetic_vision_sidecar(&sidecar_dir, "sidecar-toy").expect("the sidecar writes");

    let combined_blobs = std::fs::read(combined_dir.join("packed_vision").join("blobs.bin"))
        .expect("combined packed_vision/blobs.bin");
    let sidecar_blobs = std::fs::read(sidecar_dir.join("packed_vision").join("blobs.bin"))
        .expect("sidecar packed_vision/blobs.bin");
    assert_eq!(
        combined_blobs, sidecar_blobs,
        "packed_vision/blobs.bin differs between the combined install and the sidecar-only one"
    );

    let combined_index = model_io::load_resident_index(&combined_dir.join("model_weights.bin"))
        .expect("combined resident index");
    let sidecar_index = model_io::load_resident_index(&sidecar_dir.join("model_weights.bin"))
        .expect("sidecar resident index");

    for suffix in VISION_RESIDENT_TENSORS {
        let name = format!("vision.{suffix}");
        let c = combined_index
            .entries
            .get(&name)
            .unwrap_or_else(|| panic!("combined install is missing {name}"));
        let s = sidecar_index
            .entries
            .get(&name)
            .unwrap_or_else(|| panic!("sidecar is missing {name}"));
        let c_bytes = read_range(
            &combined_dir.join("model_weights.bin"),
            c.file_offset,
            c.size_bytes,
        );
        let s_bytes = read_range(
            &sidecar_dir.join("model_weights.bin"),
            s.file_offset,
            s.size_bytes,
        );
        assert_eq!(
            c_bytes, s_bytes,
            "{name} bytes differ between the combined install and the sidecar"
        );
    }
}

fn read_range(path: &std::path::Path, offset: u64, size: u64) -> Vec<u8> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f =
        std::fs::File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    f.seek(SeekFrom::Start(offset)).unwrap();
    let mut buf = vec![0u8; size as usize];
    f.read_exact(&mut buf).unwrap();
    buf
}

/// A minimal one-block tower's tensors, all F16, at a fixed 4-element shape
/// (the format has no shape validation `read_vision_entries` applies against
/// `VisionConfig`, so a uniform toy shape is enough), named under `prefix`.
/// Content is deterministic per NAME (not per source prefix), so the same
/// logical tensor under two different prefixes carries the same bytes -- the
/// property this file's canonicalization tests need to isolate the renaming
/// from the content.
fn tiny_tower_bytes(name: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(8);
    for (i, b) in name.bytes().cycle().take(4).enumerate() {
        let v = (b as u16).wrapping_add(i as u16 * 7);
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Builds a hand-assembled one-block tower under `prefix` (every one of the
/// twelve block roles plus the nine resident tensors), with an 8-byte dummy
/// safetensors preamble so `SafetensorsHeader::data_region_start()` (`8 +
/// header_len`, with `header_len` left at 0 here) lines up with the raw
/// buffer's actual layout.
fn build_tower_header_and_blob(prefix: &str) -> (SafetensorsHeader, Vec<u8>) {
    let mut names: Vec<String> = VISION_BLOCK_ROLES
        .iter()
        .map(|(_, suffix)| format!("{prefix}blocks.0.{suffix}"))
        .collect();
    names.extend(
        VISION_RESIDENT_TENSORS
            .iter()
            .map(|s| format!("{prefix}{s}")),
    );

    let mut tensors = BTreeMap::new();
    let mut data = Vec::new();
    for name in &names {
        // Content is derived from the SUFFIX (post-prefix name), not the
        // full name, so the same logical tensor carries the same bytes
        // whichever prefix it is filed under -- exactly the property the
        // canonicalization tests below need to isolate the rename from the
        // content.
        let suffix = name.strip_prefix(prefix).expect("built with this prefix");
        let bytes = tiny_tower_bytes(suffix);
        let start = data.len() as u64;
        let end = start + bytes.len() as u64;
        tensors.insert(
            name.clone(),
            TensorInfo {
                dtype: "F16".to_string(),
                shape: vec![bytes.len() as u64 / 2],
                data_offsets: (start, end),
            },
        );
        data.extend_from_slice(&bytes);
    }

    let header = SafetensorsHeader {
        tensors,
        metadata: None,
        header_len: 0,
    };
    let mut blob = vec![0u8; 8];
    blob.extend_from_slice(&data);
    (header, blob)
}

fn tiny_test_vision() -> model_io::VisionConfig {
    model_io::VisionConfig {
        depth: 1,
        hidden_size: 4,
        intermediate_size: 4,
        num_heads: 1,
        patch_size: 1,
        temporal_patch_size: 1,
        in_channels: 1,
        spatial_merge_size: 1,
        num_position_embeddings: 1,
        out_hidden_size: 4,
        mrope_section: [0, 0, 0],
        vision_start_token_id: 0,
        vision_end_token_id: 0,
        image_token_id: 0,
        video_token_id: 0,
    }
}

fn raw_entry(spec: &ResidentEntrySpec) -> (&str, &[u8]) {
    match spec {
        ResidentEntrySpec::Raw(r) => (r.name.as_str(), r.bytes.as_slice()),
        _ => panic!("every vision resident entry is Raw"),
    }
}

/// A `model.visual.*`-prefixed header canonicalizes onto exactly the bytes a
/// `vision_tower.*`-prefixed one produces for the same synthetic tensors --
/// both through `read_vision_entries`'s in-memory result and through the
/// bytes `write_packed_vision` actually puts on disk.
#[test]
fn a_model_visual_header_canonicalizes_to_the_same_bytes_as_vision_tower() {
    let vision = tiny_test_vision();

    let (canon_header, canon_blob) = build_tower_header_and_blob(VISION_PREFIX);
    let canon_source = MemoryRangeSource::new(&canon_blob);
    let canon_shards = Gemma4Shards::single(&canon_header, &canon_source);
    let canon_names: Vec<&str> = canon_header.tensors.keys().map(String::as_str).collect();
    let canon_read = read_vision_entries(&canon_shards, &canon_names, &vision)
        .expect("the canonical-prefix header reads");

    let (mut hf_header, hf_blob) = build_tower_header_and_blob("model.visual.");
    canonicalize_vision_header(&mut hf_header).expect("a purely HF-prefixed header canonicalizes");
    assert!(
        hf_header
            .tensors
            .keys()
            .all(|k| k.starts_with(VISION_PREFIX)),
        "every tensor should now carry the canonical prefix: {:?}",
        hf_header.tensors.keys().collect::<Vec<_>>()
    );
    let hf_source = MemoryRangeSource::new(&hf_blob);
    let hf_shards = Gemma4Shards::single(&hf_header, &hf_source);
    let hf_names: Vec<&str> = hf_header.tensors.keys().map(String::as_str).collect();
    let hf_read = read_vision_entries(&hf_shards, &hf_names, &vision)
        .expect("the canonicalized HF header reads");

    // The in-memory read result: block sub-tensors, role by role.
    assert_eq!(
        canon_read.blocks.experts.len(),
        hf_read.blocks.experts.len()
    );
    for (c, h) in canon_read
        .blocks
        .experts
        .iter()
        .zip(hf_read.blocks.experts.iter())
    {
        assert_eq!(c.sub_tensors.len(), h.sub_tensors.len());
        for (cs, hs) in c.sub_tensors.iter().zip(h.sub_tensors.iter()) {
            assert_eq!(cs.role, hs.role);
            assert_eq!(cs.bytes, hs.bytes, "block role {} bytes differ", cs.role);
        }
    }

    // The resident nine, by name (both sides' INSTALL-side names are always
    // `vision.<suffix>` regardless of the source prefix, so this compares
    // like for like).
    let canon_map: HashMap<&str, &[u8]> = canon_read.entries.iter().map(raw_entry).collect();
    let hf_map: HashMap<&str, &[u8]> = hf_read.entries.iter().map(raw_entry).collect();
    assert_eq!(canon_map.len(), hf_map.len());
    for (name, bytes) in &canon_map {
        let other = hf_map
            .get(name)
            .unwrap_or_else(|| panic!("the HF-prefixed read is missing {name}"));
        assert_eq!(bytes, other, "resident tensor {name} differs");
    }

    // And the bytes actually written to disk agree too.
    let canon_dir = temp_dir();
    write_packed_vision(&canon_dir, &canon_read.blocks, canon_read.block_stride)
        .expect("writing the canonical-prefix blocks");
    let hf_dir = temp_dir();
    write_packed_vision(&hf_dir, &hf_read.blocks, hf_read.block_stride)
        .expect("writing the canonicalized-HF blocks");
    let canon_blob_bytes =
        std::fs::read(canon_dir.join("packed_vision").join("blobs.bin")).unwrap();
    let hf_blob_bytes = std::fs::read(hf_dir.join("packed_vision").join("blobs.bin")).unwrap();
    assert_eq!(
        canon_blob_bytes, hf_blob_bytes,
        "packed_vision/blobs.bin differs between the canonical and HF-native source prefixes"
    );
}

/// A header carrying tensors under BOTH prefixes at once is refused, and the
/// error names both prefixes rather than silently preferring one -- an
/// install ingesting the wrong choice would either duplicate the tower under
/// two names or drop half of it, and neither fails until a dispatch four
/// layers in.
#[test]
fn a_header_carrying_both_vision_prefixes_is_refused() {
    let (mut header, _blob) = build_tower_header_and_blob(VISION_PREFIX);
    let (hf_header, _hf_blob) = build_tower_header_and_blob("model.visual.");
    // Fold in just one HF-prefixed tensor, so the header now carries both
    // spellings at once.
    let (name, info) = hf_header.tensors.into_iter().next().unwrap();
    header.tensors.insert(name, info);

    let err = canonicalize_vision_header(&mut header)
        .expect_err("a header mixing both prefixes must be refused");
    let text = err.to_string();
    assert!(
        text.contains(VISION_PREFIX),
        "the refusal should name {VISION_PREFIX:?}: {text}"
    );
    assert!(
        text.contains("model.visual."),
        "the refusal should name \"model.visual.\": {text}"
    );
}
