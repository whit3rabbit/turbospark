//! `NgramTableWriter` end to end: the interleave, the reader that addresses it,
//! and the refusals that stop a table nothing could detect as wrong.
//!
//! **THE HAZARD THIS FILE EXISTS FOR IS A TABLE WHOSE EVERY ROW IS REAL AND AT
//! THE WRONG ID.** The store carries no per-row name, checksum or tag: a row is
//! located by arithmetic alone, so a shard written out of order, a plane
//! interleaved in the wrong order, or a stride off by ten bytes all produce a
//! file of exactly the right size, full of the checkpoint's own bytes, that
//! validates structurally and returns the wrong 160 values for every token.
//! Nothing downstream can see it -- the model simply becomes fluent nonsense.
//!
//! So the cases below check PLACEMENT rather than presence, against a fixture
//! whose bytes identify the row and plane they came from.

use turbospark_repack::{NgramTableSpec, NgramTableWriter};

/// A small table with the REAL table's proportions: 4-bit at group 32, three
/// planes, several shards.
///
/// `ROWS_PER_SHARD` is deliberately not a power of two and not equal to
/// `SHARDS`, so a writer that confused the two axes, or that indexed rows by
/// shard, disagrees with this fixture rather than coinciding with it.
const ROWS_PER_SHARD: u64 = 7;
const SHARDS: u64 = 5;
const HEAD_DIM: u64 = 160;
const GROUP: u64 = 32;

fn spec() -> NgramTableSpec {
    NgramTableSpec {
        rows_per_shard: ROWS_PER_SHARD,
        shards: SHARDS,
        head_dim: HEAD_DIM,
        group_size: GROUP,
        bits: 4,
        layer_index: 1,
    }
}

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-ngram-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

/// Bytes that say which shard, row and plane they are.
///
/// Every byte of a row's weight plane is `(gid * 3 + 0) as u8`, its scales
/// `+ 1` and its biases `+ 2`. So a record read back at the wrong id, or with
/// its planes in the wrong order, differs in EVERY byte rather than in one --
/// which is what makes a mis-placement visible instead of subtle.
fn plane_byte(gid: u64, plane: u8) -> u8 {
    (gid.wrapping_mul(3).wrapping_add(plane as u64) & 0xFF) as u8
}

fn shard_planes(shard: u64, spec: &NgramTableSpec) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = spec.weight_bytes().expect("valid spec") as usize;
    let c = spec.companion_bytes().expect("valid spec") as usize;
    let mut weight = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    for r in 0..spec.rows_per_shard {
        let gid = shard * spec.rows_per_shard + r;
        weight.extend(std::iter::repeat_n(plane_byte(gid, 0), w));
        scales.extend(std::iter::repeat_n(plane_byte(gid, 1), c));
        biases.extend(std::iter::repeat_n(plane_byte(gid, 2), c));
    }
    (weight, scales, biases)
}

fn write_table(dir: &std::path::Path) -> NgramTableSpec {
    let s = spec();
    let mut w = NgramTableWriter::create(dir, s).expect("create");
    for shard in 0..SHARDS {
        let (weight, scales, biases) = shard_planes(shard, &s);
        w.write_shard(shard, &weight, &scales, &biases)
            .expect("write shard");
    }
    // One multiplier per n-gram order; the other two are one per hash head.
    //
    // The heads must fit INSIDE the table, and at the real table's proportions
    // they underfill it: 16 heads of 2 rows is 32 against 35, leaving 3
    // padding rows exactly as the real one leaves them (the concatenated head
    // sizes are rounded up to `make_ngram_vocab_size_divisible_by` and then
    // split into equal shards).
    //
    // The first draft of this helper gave the heads 168 rows in a 35-row
    // table, and `finish`'s self-check through the READER's validator is what
    // caught it -- before any assertion in this file ran. That is the check
    // earning its place on a fixture rather than on a 68 GiB stream.
    let heads = 16i64;
    let sizes: Vec<i64> = std::iter::repeat_n(2, heads as usize).collect();
    let mut offsets = Vec::new();
    let mut total = 0i64;
    for s in &sizes {
        offsets.push(total);
        total += s;
    }
    assert!(
        (total as u64) < s.rows().expect("valid spec"),
        "the fixture's heads must underfill its table, as the real one's do"
    );
    w.finish(vec![1, 3, 5], sizes, offsets).expect("finish");
    s
}

#[test]
fn the_written_table_reads_back_at_every_row() {
    let dir = temp_dir("roundtrip");
    let s = write_table(&dir);

    let layout = model_io::load_ngram_table_layout(&dir)
        .expect("header loads")
        .expect("a table was written");
    assert_eq!(layout.rows, ROWS_PER_SHARD * SHARDS);
    assert_eq!(layout.record_bytes, s.record_bytes().expect("valid spec"));

    let blob = std::fs::read(
        dir.join(model_io::NGRAM_TABLE_DIR)
            .join(model_io::NGRAM_TABLE_BLOB),
    )
    .expect("rows.bin");
    assert_eq!(
        blob.len() as u64,
        layout.blob_bytes().expect("sized"),
        "the blob is exactly rows x record_bytes, with no padding or slack"
    );

    let w = layout.weight_bytes as usize;
    let c = layout.scale_bytes as usize;
    // EVERY row, not a sample: the failure this guards against is placement,
    // and a sample that happened to hit aligned ids would miss a stride error.
    for gid in 0..layout.rows {
        let off = layout.row_offset(gid).expect("in range") as usize;
        let rec = &blob[off..off + layout.record_bytes as usize];
        assert!(
            rec[..w].iter().all(|b| *b == plane_byte(gid, 0)),
            "row {gid}'s weight plane is not its own"
        );
        assert!(
            rec[w..w + c].iter().all(|b| *b == plane_byte(gid, 1)),
            "row {gid}'s scale plane is not its own"
        );
        assert!(
            rec[w + c..].iter().all(|b| *b == plane_byte(gid, 2)),
            "row {gid}'s bias plane is not its own"
        );
    }
}

/// **THE PLANE ORDER INSIDE A RECORD IS WEIGHT, SCALES, BIASES**, and the two
/// companions are the same width so swapping them is invisible to every length
/// check.
///
/// Asserted as a distinct case because the round-trip above would still pass
/// if the writer and this test agreed on a wrong order. Here the expected byte
/// is derived from the PLANE's identity rather than from the writer.
#[test]
fn the_two_companion_planes_are_not_interchangeable() {
    let dir = temp_dir("planes");
    let layout_spec = write_table(&dir);
    let layout = model_io::load_ngram_table_layout(&dir)
        .expect("loads")
        .expect("present");
    let blob = std::fs::read(
        dir.join(model_io::NGRAM_TABLE_DIR)
            .join(model_io::NGRAM_TABLE_BLOB),
    )
    .expect("rows.bin");

    assert_eq!(
        layout.scale_bytes, layout.bias_bytes,
        "the two companions are the same width, which is why order is silent"
    );

    let gid = 11u64;
    let off = layout.row_offset(gid).expect("in range") as usize;
    let w = layout.weight_bytes as usize;
    let c = layout.scale_bytes as usize;
    let scales = blob[off + w];
    let biases = blob[off + w + c];
    assert_eq!(scales, plane_byte(gid, 1), "scales come before biases");
    assert_eq!(biases, plane_byte(gid, 2));
    assert_ne!(
        scales, biases,
        "the fixture must distinguish the two planes or this case proves nothing"
    );
    let _ = layout_spec;
    let _ = std::fs::remove_dir_all(&dir);
}

/// **SHARDS OUT OF ORDER ARE REFUSED, BECAUSE NOTHING DOWNSTREAM COULD SEE
/// IT.** The addressing is linear in the global row id only because the shards
/// were concatenated in that order; a walk emitting them from a HashMap would
/// write a file of the right size, full of the checkpoint's own bytes, whose
/// every row is at the wrong id.
#[test]
fn a_shard_written_out_of_order_is_refused() {
    let dir = temp_dir("order");
    let s = spec();
    let mut w = NgramTableWriter::create(&dir, s).expect("create");
    let (weight, scales, biases) = shard_planes(1, &s);
    let err = w
        .write_shard(1, &weight, &scales, &biases)
        .expect_err("shard 1 before shard 0 is refused");
    let msg = format!("{err}");
    assert!(msg.contains("id order"), "{msg}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A short table is refused at `finish`, not written and left for the reader.
///
/// The header would otherwise declare `rows_per_shard * shards` while the blob
/// held fewer, so every id past the written region indexes off the end -- and
/// the reader's own `blob_bytes` check is against the HEADER, which would
/// agree with itself.
#[test]
fn a_table_missing_a_shard_is_refused_at_finish() {
    let dir = temp_dir("short");
    let s = spec();
    let mut w = NgramTableWriter::create(&dir, s).expect("create");
    for shard in 0..SHARDS - 1 {
        let (weight, scales, biases) = shard_planes(shard, &s);
        w.write_shard(shard, &weight, &scales, &biases).expect("ok");
    }
    let err = w
        .finish(vec![1, 3, 5], vec![3], vec![0])
        .expect_err("a short table is refused");
    let msg = format!("{err}");
    assert!(msg.contains("shards written"), "{msg}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A plane whose length disagrees with the declared shape is refused BEFORE any
/// of it is written.
#[test]
fn a_plane_of_the_wrong_length_is_refused() {
    let dir = temp_dir("length");
    let s = spec();
    let mut w = NgramTableWriter::create(&dir, s).expect("create");
    let (weight, scales, biases) = shard_planes(0, &s);

    let err = w
        .write_shard(0, &weight[..weight.len() - 1], &scales, &biases)
        .expect_err("a short weight plane is refused");
    assert!(format!("{err}").contains("weight"), "{err}");

    let mut long = scales.clone();
    long.push(0);
    let err = w
        .write_shard(0, &weight, &long, &biases)
        .expect_err("a long scale plane is refused");
    assert!(format!("{err}").contains("scales"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The writer refuses a shape it cannot express before opening the file.
///
/// `let Err(..) else { panic!() }` rather than `expect_err`, which needs the
/// OK type to be `Debug` and `NgramTableWriter` is not -- `crates/runtime`
/// Gotcha 17's exact trap, one crate over, and clippy suggests the form that
/// does not compile.
#[test]
fn an_unwritable_shape_is_refused_before_anything_is_created() {
    let dir = temp_dir("shape");

    let mut s = spec();
    s.head_dim = 161;
    let Err(err) = NgramTableWriter::create(&dir, s) else {
        panic!("161 values is not a whole number of 32-value groups");
    };
    assert!(format!("{err}").contains("groups"), "{err}");

    let mut s = spec();
    s.bits = 3;
    let Err(err) = NgramTableWriter::create(&dir, s) else {
        panic!("3-bit rows have no dequantizer here");
    };
    assert!(format!("{err}").contains("dequantizer"), "{err}");

    let mut s = spec();
    s.head_dim = 1 << 62;
    let Err(err) = NgramTableWriter::create(&dir, s) else {
        panic!("overflowing row dimensions must be refused");
    };
    assert!(format!("{err}").contains("whole number of bytes"), "{err}");

    let mut s = spec();
    s.rows_per_shard = u64::MAX;
    s.shards = 2;
    let Err(err) = NgramTableWriter::create(&dir, s) else {
        panic!("overflowing table row count must be refused");
    };
    assert!(format!("{err}").contains("overflow"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// **THE SELF-CHECK IS LOAD-BEARING, AND ONLY AN INCONSISTENT HEADER CAN SHOW
/// IT.**
///
/// `finish` runs the READER's validator over the header it just wrote. Removing
/// that call leaves every other case in this file green, because on the happy
/// path the header is correct either way -- so a content assertion cannot see
/// the check at all. Mutation-checked: deleting it was the one survivor of six
/// until this case existed.
///
/// What makes it visible is a header the reader refuses. Hash heads that
/// address more rows than the table holds is the natural one, and it is not
/// hypothetical: the first draft of this file's own `write_table` helper gave
/// 16 heads 168 rows in a 35-row table, and this check is what caught it,
/// before any assertion ran.
#[test]
fn a_header_the_reader_would_refuse_is_refused_at_finish() {
    let dir = temp_dir("selfcheck-neg");
    let s = spec();
    let mut w = NgramTableWriter::create(&dir, s).expect("create");
    for shard in 0..SHARDS {
        let (weight, scales, biases) = shard_planes(shard, &s);
        w.write_shard(shard, &weight, &scales, &biases).expect("ok");
    }
    // 16 heads of 20 rows is 320, against a table of 35. Every plane is
    // correct and every byte is in place; only the ADDRESSING is impossible.
    let sizes: Vec<i64> = std::iter::repeat_n(20, 16).collect();
    let offsets: Vec<i64> = (0..16).map(|h| h * 20).collect();
    let err = w
        .finish(vec![1, 3, 5], sizes, offsets)
        .expect_err("a header whose heads overrun the table is refused");
    let msg = format!("{err}");
    assert!(
        msg.contains("the reader refuses"),
        "the writer should report that its own output failed the reader's check: {msg}"
    );
    assert!(msg.contains("address"), "{msg}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The header's CONTENT, on the happy path.
///
/// Separate from the case above because the two ask different questions: this
/// one is "did the fields reach the file", that one is "would the reader take
/// it". A writer that dropped the hashing buffers passes that one and fails
/// this.
#[test]
fn the_writer_checks_its_own_header_against_the_readers_rules() {
    let dir = temp_dir("selfcheck");
    write_table(&dir);
    // The proof is that `finish` returned Ok above, which it only does when
    // `load_ngram_table_layout` accepted what it wrote. Assert the artifact
    // that makes it so, rather than restating the call.
    let header = std::fs::read_to_string(
        dir.join(model_io::NGRAM_TABLE_DIR)
            .join(model_io::NGRAM_TABLE_HEADER),
    )
    .expect("header.json");
    let parsed: serde_json::Value = serde_json::from_str(&header).expect("valid json");
    assert_eq!(parsed["companionDtype"], "bf16");
    assert_eq!(parsed["version"], 1);
    assert_eq!(parsed["layerIndex"], 1);
    // The hashing buffers reached the header rather than being dropped.
    assert_eq!(parsed["multipliers"].as_array().expect("array").len(), 3);
    assert_eq!(
        parsed["headVocabSizes"].as_array().expect("array").len(),
        16
    );
    assert_eq!(parsed["headOffsets"].as_array().expect("array").len(), 16);
    let _ = std::fs::remove_dir_all(&dir);
}
