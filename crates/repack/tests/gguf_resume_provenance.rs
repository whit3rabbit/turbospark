//! Leftover expert layers are adopted only under a matching source identity.
//!
//! A re-upload or another fine-tune at the same quant has the same shapes and
//! so the same file sizes; size alone must not be believed.

use std::cell::Cell;
use std::path::PathBuf;

use turbospark_repack::{
    build_synthetic_gpt_oss_gguf, parse_gguf_header, write_gguf_install_streamed,
    write_gguf_install_streamed_resumable, DownloadError, MemoryRangeSource, RangeSource,
    ResumeProvenance, SyntheticGptOssShape, GGUF_DEFAULT_MAX_HEADER_BYTES,
};

/// Counts reads and fails every read from index `fail_from` on, standing in
/// for a network failure part way through the expert layers.
struct Flaky<'a> {
    inner: MemoryRangeSource<'a>,
    reads: Cell<usize>,
    fail_from: usize,
}

impl RangeSource for Flaky<'_> {
    fn read_range(&self, start: u64, end: u64) -> Result<Vec<u8>, DownloadError> {
        let n = self.reads.get();
        self.reads.set(n + 1);
        if n >= self.fail_from {
            return Err(DownloadError::ShortRead {
                expected: end - start,
                actual: 0,
            });
        }
        self.inner.read_range(start, end)
    }
}

fn temp(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ts-resume-prov-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn prov(c: char) -> ResumeProvenance {
    ResumeProvenance::new("o/r", &c.to_string().repeat(40), "m.gguf").unwrap()
}

/// Runs a walk that dies after layer 0 is on disk, returning the file bytes.
fn interrupted(dir: &std::path::Path, bytes: &[u8], p: Option<&ResumeProvenance>) {
    let header = parse_gguf_header(bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    // Learn the read count of a clean walk, then fail on the final read,
    // which belongs to the last layer, so layer 0 is complete.
    let probe = Flaky {
        inner: MemoryRangeSource::new(bytes),
        reads: Cell::new(0),
        fail_from: usize::MAX,
    };
    let scratch = temp("probe");
    write_gguf_install_streamed(&scratch, &header, &probe, "m", |_| {}).unwrap();
    let total = probe.reads.get();
    let _ = std::fs::remove_dir_all(&scratch);

    let flaky = Flaky {
        inner: MemoryRangeSource::new(bytes),
        reads: Cell::new(0),
        fail_from: total - 1,
    };
    let result = write_gguf_install_streamed_resumable(dir, &header, &flaky, "m", p, |_| {});
    assert!(result.is_err(), "the walk must be interrupted");
    assert!(dir.join("packed_experts/layer_00.bin").exists());
}

fn resume(dir: &std::path::Path, bytes: &[u8], p: Option<&ResumeProvenance>) -> Vec<String> {
    let header = parse_gguf_header(bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    let mut lines = Vec::new();
    write_gguf_install_streamed_resumable(
        dir,
        &header,
        &MemoryRangeSource::new(bytes),
        "m",
        p,
        |l| lines.push(l.to_string()),
    )
    .unwrap();
    lines
}

fn adopted(lines: &[String]) -> bool {
    lines.iter().any(|l| l.contains("layer 0 adopted"))
}

#[test]
fn same_commit_adopts_but_same_size_different_commit_or_absent_does_not() {
    let (bytes, _) = build_synthetic_gpt_oss_gguf(SyntheticGptOssShape::default());

    // Control: identical provenance resumes.
    let dir = temp("same");
    interrupted(&dir, &bytes, Some(&prov('a')));
    assert!(adopted(&resume(&dir, &bytes, Some(&prov('a')))));

    // Same size, different commit: the leftover must be refused and rewritten
    // from the new bytes (a flipped weight byte keeps the size identical).
    let mut other = bytes.clone();
    let last = other.len() - 1;
    other[last - 8] ^= 0xFF;
    let dir = temp("diff");
    interrupted(&dir, &bytes, Some(&prov('a')));
    let lines = resume(&dir, &other, Some(&prov('b')));
    assert!(!adopted(&lines), "a different commit must not be adopted");
    assert!(lines.iter().any(|l| l.contains("layer 0 written")));

    // Leftovers from a walk with no provenance are never trusted later.
    let dir = temp("absent");
    interrupted(&dir, &bytes, None);
    assert!(!adopted(&resume(&dir, &bytes, Some(&prov('a')))));

    // And a walk with no provenance never adopts a recorded leftover.
    let dir = temp("none");
    interrupted(&dir, &bytes, Some(&prov('a')));
    assert!(!adopted(&resume(&dir, &bytes, None)));
}
