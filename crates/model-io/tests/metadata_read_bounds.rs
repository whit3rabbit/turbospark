//! Metadata size caps must bound the bytes actually read, not only the size
//! `stat` reported before the read.
//!
//! A FIFO (or a symlink to `/dev/zero`, or a file that grows after the stat)
//! reports a length of 0, passes a stat-only cap, and then `fs::read` takes
//! whatever arrives. A FIFO is the deterministic stand-in for the race: the
//! writer below delivers far more than the cap after the loader has stat-ed.
#![cfg(unix)]

use std::io::Write;
use std::path::{Path, PathBuf};

use turbospark_model_io::{
    load_install_receipt, load_manifest, load_ngram_table_layout, load_packed_layout_from,
    peek_family, qwen3_4b, ModelError, NGRAM_HEADER_MAX_BYTES, NGRAM_TABLE_DIR, NGRAM_TABLE_HEADER,
};

const CAP: u64 = 4096;

fn tempdir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-meta-bounds-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Makes `path` a FIFO and feeds it `len` bytes of JSON whitespace from a
/// detached thread. Write errors are ignored: a bounded reader closes early,
/// which is the behavior under test.
fn fifo_with(path: &Path, len: usize) {
    let status = std::process::Command::new("mkfifo")
        .arg(path)
        .status()
        .expect("run mkfifo");
    assert!(status.success(), "mkfifo failed");
    let path = path.to_path_buf();
    std::thread::spawn(move || {
        if let Ok(mut f) = std::fs::OpenOptions::new().write(true).open(&path) {
            let chunk = vec![b' '; 64 * 1024];
            let mut left = len;
            while left > 0 {
                let n = left.min(chunk.len());
                if f.write_all(&chunk[..n]).is_err() {
                    return;
                }
                left -= n;
            }
        }
    });
}

fn corrupt_detail(err: ModelError) -> String {
    match err {
        ModelError::IndexCorrupt { detail } => detail,
        other => panic!("expected IndexCorrupt, got {other:?}"),
    }
}

#[test]
fn manifest_load_caps_bytes_read_not_just_stat_size() {
    let dir = tempdir("manifest");
    fifo_with(&dir.join("manifest.json"), 64 * CAP as usize);
    let err = load_manifest(&dir, &qwen3_4b(), CAP).expect_err("over-cap read must fail");
    let detail = corrupt_detail(err);
    assert!(detail.contains("exceeds metadata cap"), "{detail}");
}

#[test]
fn manifest_peek_family_caps_bytes_read_not_just_stat_size() {
    let dir = tempdir("peek");
    fifo_with(&dir.join("manifest.json"), 64 * CAP as usize);
    let err = peek_family(&dir, CAP).expect_err("over-cap read must fail");
    let detail = corrupt_detail(err);
    assert!(detail.contains("exceeds metadata cap"), "{detail}");
}

#[test]
fn packed_layout_caps_bytes_read_not_just_stat_size() {
    let dir = tempdir("packed");
    std::fs::create_dir_all(dir.join("packed_experts")).unwrap();
    fifo_with(&dir.join("packed_experts/layout.json"), 64 * CAP as usize);
    let err =
        load_packed_layout_from(&dir, "packed_experts", CAP).expect_err("over-cap read must fail");
    let detail = corrupt_detail(err);
    assert!(detail.contains("exceeds metadata cap"), "{detail}");
}

#[test]
fn install_receipt_caps_bytes_read_not_just_stat_size() {
    let dir = tempdir("receipt");
    fifo_with(&dir.join("verified-install.json"), 64 * CAP as usize);
    let err = load_install_receipt(&dir, CAP).expect_err("over-cap read must fail");
    let ModelError::TrustedReceiptInvalid { detail } = err else {
        panic!("expected TrustedReceiptInvalid, got {err:?}");
    };
    assert!(detail.contains("exceeds metadata cap"), "{detail}");
}

#[test]
fn ngram_header_caps_bytes_read_not_just_stat_size() {
    let dir = tempdir("ngram");
    std::fs::create_dir_all(dir.join(NGRAM_TABLE_DIR)).unwrap();
    fifo_with(
        &dir.join(NGRAM_TABLE_DIR).join(NGRAM_TABLE_HEADER),
        2 * NGRAM_HEADER_MAX_BYTES as usize,
    );
    let err = load_ngram_table_layout(&dir).expect_err("over-cap read must fail");
    let detail = corrupt_detail(err);
    assert!(detail.contains("ceiling"), "{detail}");
}
