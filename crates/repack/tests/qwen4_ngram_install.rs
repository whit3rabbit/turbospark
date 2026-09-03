//! `qwen4_exp`'s n-gram table wired into the real repack walk, end to end.
//!
//! **THE THING THIS FILE HAS TO CATCH IS THE MTP HEAD'S BUG, ONE COMPONENT
//! OVER.** `crates/repack` Gotcha 8 (`gemma4_checkpoint/mtp.rs`'s doc)
//! records an ingest that landed in the non-streamed writer alone: every
//! fixture took that path, every real install takes
//! `write_gemma4_install_streamed`, and the first real stream that asked for
//! a head wrote a byte-identical HEADLESS install with no error. The n-gram
//! table cannot even be carried the way the head is (32 GB will not fit in
//! `Gemma4RepackOutput`), so its write happens at BOTH writer entry points
//! independently (`gemma4_checkpoint::ngram::write_ngram_table`'s doc) --
//! which is exactly the shape that bug needs to recur in. `both_writers_
//! carry_the_ngram_table` is what keeps the two from drifting apart again.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use turbospark_repack::{
    build_synthetic_qwen4_exp_install, build_synthetic_qwen4_exp_install_streamed,
    verify_install_full_sha256,
};

const VOCAB: i64 = 64;
const LAYERS: i64 = 2;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-qwen4-ngram-install-{tag}-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// **BOTH WRITER ENTRY POINTS ACTUALLY WRITE THE TABLE, AND THEY AGREE.**
///
/// The non-streamed writer (`write_gemma4_install`, what every OTHER
/// family's fixture exercises) and the streamed one (`write_gemma4_install_
/// streamed`, what every REAL checkpoint takes) each call `write_ngram_table`
/// independently rather than sharing one in-memory read the way the MTP
/// head's ingest does. Asserting byte-for-byte agreement is what would have
/// caught the head's bug in this shape: a table written by only one of the
/// two would still pass every OTHER assertion in this file.
#[test]
fn both_writers_carry_the_ngram_table() {
    let streamed_dir = temp_dir("streamed");
    build_synthetic_qwen4_exp_install_streamed(&streamed_dir, VOCAB, LAYERS, "qwen4-ngram-toy")
        .expect("the streamed writer builds an install with an n-gram table");

    let plain_dir = temp_dir("plain");
    build_synthetic_qwen4_exp_install(&plain_dir, VOCAB, LAYERS, "qwen4-ngram-toy")
        .expect("the non-streamed writer builds an install with an n-gram table");

    let streamed_layout = model_io::load_ngram_table_layout(&streamed_dir)
        .expect("streamed header loads")
        .expect("the streamed writer wrote a table");
    let plain_layout = model_io::load_ngram_table_layout(&plain_dir)
        .expect("plain header loads")
        .expect("the non-streamed writer wrote a table");

    assert_eq!(
        streamed_layout, plain_layout,
        "the two writers disagree about the table's header"
    );
    // Six rows: three shards of two, per `synthetic_qwen::qwen4`'s constants.
    assert_eq!(streamed_layout.rows, 6);
    assert_eq!(streamed_layout.layer_index, 1);

    let streamed_blob = std::fs::read(
        streamed_dir
            .join(model_io::NGRAM_TABLE_DIR)
            .join(model_io::NGRAM_TABLE_BLOB),
    )
    .expect("streamed rows.bin");
    let plain_blob = std::fs::read(
        plain_dir
            .join(model_io::NGRAM_TABLE_DIR)
            .join(model_io::NGRAM_TABLE_BLOB),
    )
    .expect("plain rows.bin");
    assert_eq!(
        streamed_blob, plain_blob,
        "the two writers disagree about the table's bytes"
    );
    assert_eq!(
        streamed_blob.len() as u64,
        streamed_layout.blob_bytes().expect("sized"),
        "the blob is exactly rows x record_bytes"
    );
}

/// **PLACEMENT, NOT JUST PRESENCE.** `synthetic_qwen::qwen4::ngram_tensors`
/// fills each shard's weight plane with its own shard index, so a shard
/// written at the wrong global row id -- every byte correct, wrong place --
/// is visible here even though it would pass every check above (right
/// header, right total length, right per-writer agreement, since a
/// consistently-wrong placement agrees with itself).
#[test]
fn the_installed_table_places_each_shard_at_its_own_rows() {
    let dir = temp_dir("placement");
    build_synthetic_qwen4_exp_install_streamed(&dir, VOCAB, LAYERS, "qwen4-ngram-placement")
        .expect("install");

    let layout = model_io::load_ngram_table_layout(&dir)
        .expect("header loads")
        .expect("a table was written");
    let blob = std::fs::read(
        dir.join(model_io::NGRAM_TABLE_DIR)
            .join(model_io::NGRAM_TABLE_BLOB),
    )
    .expect("rows.bin");

    let rows_per_shard = layout.rows_per_shard;
    for gid in 0..layout.rows {
        let shard = gid / rows_per_shard;
        let off = layout.row_offset(gid).expect("in range") as usize;
        let weight_byte = blob[off];
        assert_eq!(
            weight_byte, shard as u8,
            "row {gid} (shard {shard}) carries another shard's weight byte"
        );
    }
}

/// Phase 1's own stated gate: `install_verifier` accepts a `qwen4_exp`
/// install with an n-gram table. `verify_install_full_sha256` re-hashes
/// every file `manifest.json` lists and nothing else -- `ngram_table/rows.bin`
/// is deliberately NOT one of them (`build_manifest_json` would have to
/// `std::fs::read` a file this whole module exists to never hold in memory),
/// so this test is also a check that omission does not trip the verifier by
/// naming a file it cannot find.
#[test]
fn the_install_verifier_accepts_a_qwen4_exp_install() {
    let dir = temp_dir("verify");
    let arch = build_synthetic_qwen4_exp_install_streamed(&dir, VOCAB, LAYERS, "qwen4-verify")
        .expect("install");
    verify_install_full_sha256(&dir, &arch).expect("the verifier accepts its own install");
}

/// The NON-streamed writer's install passes the same gate. Not redundant
/// with the streamed case above: `verify_install_full_sha256` reads
/// `manifest.json`'s `files` map, which is built once per writer
/// (`build_manifest_json`) and could in principle diverge between the two
/// entry points the way the table's own bytes almost did.
#[test]
fn the_install_verifier_accepts_the_non_streamed_writers_install_too() {
    let dir = temp_dir("verify-plain");
    let arch = build_synthetic_qwen4_exp_install(&dir, VOCAB, LAYERS, "qwen4-verify-plain")
        .expect("install");
    verify_install_full_sha256(&dir, &arch).expect("the verifier accepts its own install");
}
