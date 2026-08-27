//! The synthetic `qwen3_5` VISION TOWER, end to end through the real repack
//! walk and back out through every loader it has to satisfy (ROADMAP M-V3,
//! stage 1).
//!
//! **This file exists so the 16 GB stream is not the thing that finds the
//! holes** -- `crates/repack` Gotcha 8's rule, which the MTP head paid a
//! 15-minute re-stream for and M4's dense `llama` half paid three. Every
//! assertion here is one the real `qwen38_checkpoint_network` walk would
//! otherwise have made, minutes at a time.
//!
//! What it can and cannot see is worth stating. It CAN see: that both writers
//! carry the tower, that `packed_vision/` decodes through the same loader
//! `packed_experts/` does, that every block resolves to a distinct span
//! inside the blob file, that the tower's resident tensors are tagged FP16 and its text
//! neighbours BF16, that a text-only walk still drops the tower, and that the
//! declared-but-unshipped case writes no tower at all. It CANNOT see anything
//! about the NUMBERS: the weights are untrained and no tower forward pass
//! exists yet (that is M-V4).

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use turbospark_repack::{
    build_synthetic_qwen_gdn_dense_install, build_synthetic_qwen_gdn_dense_install_with_vision,
    build_synthetic_qwen_gdn_dense_install_with_vision_streamed, tiny_qwen_gdn_dense_arch,
    tiny_vision_config, vision_arch_for_manifest, vision_should_ingest, VISION_BLOCK_ROLES,
    VISION_RESIDENT_TENSORS,
};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-qwen35-vision-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

const VOCAB: i64 = 256;
const LAYERS: i64 = 4;
const BITS: u32 = 1;

fn build_streamed() -> (PathBuf, model_io::ArchConfig) {
    let dir = temp_dir();
    let arch = build_synthetic_qwen_gdn_dense_install_with_vision_streamed(
        &dir,
        VOCAB,
        LAYERS,
        "qwen35-vision-toy",
        BITS,
    )
    .expect("the streamed writer builds a dense install with a vision tower");
    (dir, arch)
}

fn load_vision_layout(dir: &std::path::Path) -> model_io::PackedExpertsLayout {
    model_io::load_packed_layout_from(
        dir,
        model_io::PACKED_VISION_DIR,
        model_io::PACKED_EXPERTS_LAYOUT_DEFAULT_MAX_BYTES,
    )
    .expect("packed_vision/layout.json decodes through the packed-experts loader")
}

/// The manifest the walk writes is one `load_manifest` accepts, with the
/// tower declared.
///
/// The single most valuable assertion in the file, for the reason its
/// text-only sibling gives: the manifest gate is where an install stops being
/// openable, and it is exactly the thing a fixture can check in milliseconds
/// and a download discovers after twenty-five minutes.
#[test]
fn a_vision_install_writes_a_manifest_the_loader_accepts() {
    let (dir, arch) = build_streamed();
    let manifest = model_io::load_manifest(&dir, &arch, 4 * 1024 * 1024)
        .expect("the manifest a vision walk writes is one load_manifest accepts");

    // Both files are DECLARED, which is what `validate_manifest`'s
    // vision-gated required-file check reads.
    for f in ["packed_vision/layout.json", "packed_vision/blobs.bin"] {
        assert!(
            manifest.files.contains_key(f),
            "the manifest does not declare {f}: {:?}",
            manifest.files.keys().collect::<Vec<_>>()
        );
    }
    assert!(arch.vision.is_active(), "the fixture asked for a tower");
}

/// **THE TEST THIS MILESTONE MOST NEEDED, and the one whose absence shipped a
/// bug for the MTP head.**
///
/// The head's ingest landed in `orchestrate_gemma4_checkpoint_sharded` alone.
/// Every fixture took that non-streamed path and every REAL install takes
/// `write_gemma4_install_streamed`, which classified `mtp.*` correctly and
/// then never read it -- so the first real stream that asked for a head wrote
/// a byte-identical HEADLESS install, with no error and nothing in the
/// progress log to say so.
///
/// Removing EITHER writer's vision arm reddens this, which is the property
/// that makes it worth its runtime.
#[test]
fn both_writers_carry_the_vision_tower() {
    let (streamed_dir, _) = build_streamed();
    let plain_dir = temp_dir();
    build_synthetic_qwen_gdn_dense_install_with_vision(
        &plain_dir,
        VOCAB,
        LAYERS,
        "qwen35-vision-toy",
        BITS,
    )
    .expect("the non-streamed writer builds a dense install with a vision tower");

    for dir in [&streamed_dir, &plain_dir] {
        let layout = load_vision_layout(dir);
        assert_eq!(
            layout.experts_per_layer,
            tiny_vision_config().depth as usize,
            "{} dropped blocks",
            dir.display()
        );
    }

    // And the two writers agree BYTE FOR BYTE, not merely in block count. A
    // pair of arms that both run but pack differently is a subtler version of
    // the same bug and no count can see it.
    let streamed_blobs =
        std::fs::read(streamed_dir.join("packed_vision").join("blobs.bin")).expect("streamed");
    let plain_blobs =
        std::fs::read(plain_dir.join("packed_vision").join("blobs.bin")).expect("plain");
    assert_eq!(
        streamed_blobs, plain_blobs,
        "the two writers pack the tower differently"
    );

    // The resident side too: the tower's nine non-block tensors have to reach
    // BOTH indexes. Appending them in one writer only is the same hole one
    // destination over, and the blob comparison above cannot see it.
    for dir in [&streamed_dir, &plain_dir] {
        let index =
            model_io::load_resident_index(&dir.join("model_weights.bin")).expect("resident index");
        let vision: Vec<&String> = index
            .entries
            .keys()
            .filter(|k| k.starts_with("vision."))
            .collect();
        assert_eq!(
            vision.len(),
            VISION_RESIDENT_TENSORS.len(),
            "{} carries {vision:?}",
            dir.display()
        );
    }
}

/// `packed_vision/` decodes through the loader `packed_experts/` uses, and
/// every block resolves to its own span inside the blob file.
///
/// This is the round trip that says the schema reuse is real rather than
/// asserted: the tower is written by the same `build_layer_file` and decoded
/// by the same `load_from`, so the addressing M-V4 needs at inference is
/// already proven to resolve.
#[test]
fn the_tower_round_trips_through_the_packed_experts_schema() {
    let (dir, _) = build_streamed();
    let vision = tiny_vision_config();
    let layout = load_vision_layout(&dir);

    assert_eq!(layout.num_layers, 1, "the tower is ONE layer of blocks");
    assert_eq!(layout.experts_per_layer, vision.depth as usize);
    assert_eq!(layout.layers.len(), 1);
    let layer = &layout.layers[0];
    assert_eq!(layer.file, "blobs.bin");
    assert_eq!(layer.experts.len(), vision.depth as usize);

    // Every block carries every role, and the roles are the twelve the table
    // declares rather than whatever happened to be written.
    for block in &layer.experts {
        let mut roles: Vec<&str> = block.sub_tensors.keys().map(String::as_str).collect();
        roles.sort_unstable();
        let mut expected: Vec<&str> = VISION_BLOCK_ROLES.iter().map(|(r, _)| *r).collect();
        expected.sort_unstable();
        assert_eq!(roles, expected, "block {} roles", block.expert);
        for (role, sub) in &block.sub_tensors {
            assert_eq!(sub.dtype, "fp16", "block {} role {role}", block.expert);
        }
    }

    // Every block resolves to a span inside the file, which is the bounds
    // check `PreadExpertStreamer` applies per load. Asserted from the layout's
    // own numbers rather than through `StreamLayout`, because that type lives
    // in `crates/streaming` and this crate does not depend on it -- the
    // `from_packed_layer_in` binding is covered where it lives, in
    // `crates/streaming/tests/stream_layout.rs`.
    let on_disk = std::fs::metadata(dir.join("packed_vision").join("blobs.bin"))
        .expect("blobs.bin")
        .len();
    for block in &layer.experts {
        assert_eq!(
            block.size, layer.expert_stride,
            "block {} is not one stride",
            block.expert
        );
        assert!(
            block.offset + layer.expert_stride <= on_disk,
            "block {} at {} runs past the {on_disk}-byte file",
            block.expert,
            block.offset
        );
    }
    // And the blocks do not OVERLAP, which the per-block bound above cannot
    // see: every offset inside the file is satisfied by writing them all at 0.
    let mut offsets: Vec<u64> = layer.experts.iter().map(|b| b.offset).collect();
    offsets.sort_unstable();
    offsets.dedup();
    assert_eq!(
        offsets.len(),
        vision.depth as usize,
        "two blocks share an offset"
    );
}

/// The tower is FP16 and its text neighbours are BF16, in one index.
///
/// **This is the assertion the whole dtype decision rests on.** Every
/// unquantized TEXT tensor is narrowed to BF16 because that is the only
/// unquantized width the text kernels dispatch (AGENTS.md Gotcha 45); the
/// tower is kept at FP16 because the M-V2 kernels bind `half`. Both rules
/// applying to one `model_weights.bin` is the thing that could quietly stop
/// being true, and a tag is what says which rule ran.
#[test]
fn the_tower_is_fp16_where_the_trunk_is_bf16() {
    let (dir, _) = build_streamed();
    let index =
        model_io::load_resident_index(&dir.join("model_weights.bin")).expect("resident index");

    const DTYPE_BF16: u8 = 1;
    const DTYPE_FP16: u8 = 2;

    for suffix in VISION_RESIDENT_TENSORS {
        let name = format!("vision.{suffix}");
        let entry = index
            .entries
            .get(&name)
            .unwrap_or_else(|| panic!("{name} is not in the resident index"));
        assert_eq!(entry.dtype, DTYPE_FP16, "{name} should be FP16");
    }

    // A text norm, through the same index, still BF16. Named explicitly
    // rather than "any non-vision tensor", because the quantized trunk
    // tensors carry their own affine tags and would satisfy a loose check
    // without saying anything about the narrowing.
    let trunk = index
        .entries
        .get("language_model.model.norm.weight")
        .expect("the trunk's final norm");
    assert_eq!(
        trunk.dtype, DTYPE_BF16,
        "the trunk's norms must still narrow to BF16"
    );
}

/// **THE ORNITH CASE: a checkpoint that DECLARES a tower and SHIPS none.**
///
/// `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` publishes a full `vision_config`
/// -- the same depth 27, hidden 1152, intermediate 4304 tower the three
/// `qwen3_5` checkpoints declare, read off the published file rather than
/// assumed -- and its MLX conversion carries no `vision_tower.` tensor at all.
/// It is `ornith_config.rs`'s own "DECLARED AND UNSHIPPED" note, one component
/// over from `mtp.*`.
///
/// Both halves matter and they fail differently. Ingesting on the config alone
/// would write a manifest claiming a tower and `validate_manifest` would then
/// demand `packed_vision/` files that do not exist, so a WORKING install would
/// stop opening. Leaving the arch untouched while writing nothing would do the
/// same thing one step later.
///
/// Asserted against the two pure functions rather than through a build,
/// because the fixture builder ties the arch and the tensors together by
/// construction and so cannot produce this input at all.
#[test]
fn a_declared_but_unshipped_tower_is_not_ingested_and_not_declared() {
    let mut arch = tiny_qwen_gdn_dense_arch(VOCAB, LAYERS);
    arch.vision = tiny_vision_config();
    assert!(arch.vision.is_active(), "the arch declares a tower");

    // The artifact ships nothing: no ingest.
    assert!(
        !vision_should_ingest(&arch, &[]),
        "a declared-but-unshipped tower must not be ingested"
    );
    // And the manifest must not claim one either.
    let declared = vision_arch_for_manifest(&arch, false);
    assert!(
        !declared.vision.is_active(),
        "the manifest would claim a tower the install does not have"
    );

    // The two other corners, so the conjunction is pinned in both directions
    // rather than only where it happens to be false.
    let names = ["vision_tower.blocks.0.norm1.weight"];
    assert!(
        vision_should_ingest(&arch, &names),
        "a declared AND shipped tower must be ingested"
    );
    assert!(
        !vision_should_ingest(&tiny_qwen_gdn_dense_arch(VOCAB, LAYERS), &names),
        "a text-only request must not ingest a tower the checkpoint happens to carry"
    );
    // An ingested tower keeps its declaration.
    assert!(vision_arch_for_manifest(&arch, true).vision.is_active());
}

/// A text-only walk on the SAME checkpoint still drops the tower.
///
/// The guarantee that M-V3 changed no install anyone has already built: every
/// existing caller passes `VisionConfig::NONE`, and `should_ingest` requires
/// it to be active. Without this, the new classify arm would silently start
/// ingesting towers into installs that never had them.
#[test]
fn a_text_only_walk_writes_no_tower() {
    let dir = temp_dir();
    let arch = build_synthetic_qwen_gdn_dense_install(&dir, VOCAB, LAYERS, "qwen35-toy")
        .expect("the text-only install writes");

    assert!(!arch.vision.is_active());
    assert!(
        !dir.join("packed_vision").exists(),
        "a text-only walk wrote a packed_vision directory"
    );
    let manifest = model_io::load_manifest(&dir, &arch, 4 * 1024 * 1024)
        .expect("the text-only manifest still loads");
    assert!(
        !manifest
            .files
            .keys()
            .any(|f| f.starts_with("packed_vision/")),
        "a text-only manifest declares vision files"
    );
}

/// **THE CONVERSION ARM, which the fixture above cannot reach.**
///
/// The tower ships F16 in `prism-ml/Bonsai-27B-mlx-1bit` and **BF16** in
/// `mlx-community/Qwen3.8-27B-4bit` -- and the second is the artifact the real
/// M-V3 gate streams. So the fixture, which is F16 like the rest of the dense
/// 1-bit install it hangs off, exercises only the verbatim arm. A walk that
/// handled F16 alone would pass every test above and refuse the very
/// checkpoint the milestone exists to ingest.
///
/// Three properties, and the middle one is why FP16 is affordable at all.
#[test]
fn a_bf16_tower_converts_to_fp16_exactly_in_the_normal_range() {
    // 1. BF16 -> FP16 is EXACT wherever FP16 has the range. The mantissa
    //    WIDENS (7 bits to 10), so it cannot round; only the exponent can
    //    fail. Checked over the whole BF16 grid in a realistic weight band
    //    rather than on a few hand-picked values.
    // The band is FP16's own NORMAL range -- [2^-14, 65504] -- because that is
    // exactly the range the claim is about. BF16 lays ~128 values per octave
    // per sign, and this spans 30 octaves, so the count below is arithmetic
    // rather than a threshold someone picked.
    const FP16_MIN_NORMAL: f32 = 6.103_515_6e-5;
    const FP16_MAX: f32 = 65504.0;
    let mut checked = 0usize;
    for raw in 0u16..=u16::MAX {
        let value = compute::bf16_to_f32(raw);
        if !value.is_finite() || value.abs() < FP16_MIN_NORMAL || value.abs() > FP16_MAX {
            continue;
        }
        let bytes = raw.to_le_bytes().to_vec();
        let out = turbospark_repack::convert_raw_to_fp16("w", "BF16", bytes)
            .expect("a weight-sized BF16 value converts");
        assert_eq!(out.lossy, 0, "BF16 {value} lost bits reaching FP16");
        let back = compute::f16_to_f32(u16::from_le_bytes([out.bytes[0], out.bytes[1]]));
        assert_eq!(back, value, "BF16 {value} did not round-trip");
        checked += 1;
    }
    assert!(
        checked > 7_000,
        "only {checked} values exercised; ~30 octaves x 128 x 2 signs was expected"
    );

    // 2. An F16 source is passed through VERBATIM, not re-encoded. Same bytes
    //    out as in, which is what makes the Bonsai arm free.
    let f16_bytes = vec![0x00, 0x3c, 0x00, 0xc0];
    let out = turbospark_repack::convert_raw_to_fp16("w", "F16", f16_bytes.clone())
        .expect("an F16 tower is carried verbatim");
    assert_eq!(out.bytes, f16_bytes);
    assert_eq!(out.lossy, 0);

    // 3. An OVERFLOW is refused by name, never clamped to inf. BF16 reaches
    //    3.4e38 where FP16 stops at 65504, and an `inf` weight is AGENTS.md
    //    Gotcha 59's failure mode: it does not crash, it becomes NaN in the
    //    activations, and NaN then reads as a PERFECT score on every rank and
    //    top-k instrument downstream.
    let big = compute::f32_to_bf16(1.0e30);
    // `expect_err` does not work here: it requires the OK type to be `Debug`,
    // and deriving that on a struct holding the tensor's bytes would dump
    // megabytes on a failure. Same shape as `crates/runtime` Gotcha 17.
    let Err(err) = turbospark_repack::convert_raw_to_fp16("w", "BF16", big.to_le_bytes().to_vec())
    else {
        panic!("a value past FP16's ceiling must be refused");
    };
    let text = err.to_string();
    assert!(
        text.contains("65504"),
        "the refusal should name FP16's ceiling: {text}"
    );

    // 4. A source that was ALREADY non-finite is passed through rather than
    //    blamed on this step. The checkpoint said so; inventing a refusal for
    //    it would report the publisher's bytes as this conversion's fault.
    let inf = compute::f32_to_bf16(f32::INFINITY);
    let out = turbospark_repack::convert_raw_to_fp16("w", "BF16", inf.to_le_bytes().to_vec())
        .expect("an already-infinite source is the checkpoint's statement, not an overflow");
    assert!(!compute::f16_to_f32(u16::from_le_bytes([out.bytes[0], out.bytes[1]])).is_finite());
}

/// The rank-5 patch embedding is recorded at the 2-D shape it is USED at, and
/// its recorded product equals its element count.
///
/// `shape4` truncates to four dims, and `patch_embed.proj.weight` is rank 5
/// (`(out, T, P_h, P_w, C)`), so recording it naively drops the channel count:
/// the bytes stay intact, every length check passes, and a reader computing
/// `rows * cols` off the index gets a number smaller than the tensor has. The
/// flattened form is also the truthful one -- the conv has kernel == stride,
/// so this tensor is a plain GEMM matrix and no conv kernel exists.
#[test]
fn the_rank_five_patch_embedding_records_a_shape_that_matches_its_bytes() {
    let (dir, _) = build_streamed();
    let vision = tiny_vision_config();
    let index =
        model_io::load_resident_index(&dir.join("model_weights.bin")).expect("resident index");

    let entry = index
        .entries
        .get("vision.patch_embed.proj.weight")
        .expect("the patch embedding");
    let expected_cols =
        vision.temporal_patch_size * vision.patch_size * vision.patch_size * vision.in_channels;
    assert_eq!(
        (entry.shape.0 as i64, entry.shape.1 as i64),
        (vision.hidden_size, expected_cols),
        "the patch embedding is not recorded at its GEMM shape"
    );
    assert_eq!(
        (entry.shape.2, entry.shape.3),
        (0, 0),
        "the trailing dims should be collapsed, not carried"
    );
    // The product matches the bytes, which is the property truncation breaks.
    assert_eq!(
        entry.shape.0 as u64 * entry.shape.1 as u64 * 2,
        entry.size_bytes,
        "recorded shape does not account for the tensor's FP16 bytes"
    );

    // A rank-2 neighbour is unchanged, so the flattening did not become a
    // blanket reshape of every vision tensor.
    let pos = index
        .entries
        .get("vision.pos_embed.weight")
        .expect("the position table");
    assert_eq!(
        (pos.shape.0 as i64, pos.shape.1 as i64),
        (vision.num_position_embeddings, vision.hidden_size)
    );
}
