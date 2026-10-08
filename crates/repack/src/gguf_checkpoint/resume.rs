//! Provenance for resuming a GGUF expert-layer walk.
//!
//! A leftover `layer_NN.bin` is only believable when it came from the exact
//! bytes this walk would read. Size alone cannot prove that: an upstream
//! re-upload of the same file name, or another fine-tune of the same base at
//! the same quant, produces identical shapes and therefore identical file
//! sizes. So a walk records WHERE it is reading from before it writes its
//! first layer, and a later walk adopts leftovers only when that record
//! matches its own source exactly. Absent, unreadable, or different means
//! the leftovers are deleted, never adopted.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const RECORD_NAME: &str = ".resume-provenance.json";

/// The immutable identity of a GGUF source: repository, resolved 40-hex
/// commit, and file. A floating branch is not an identity, so a caller with
/// only a branch name passes `None` and no resume is attempted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResumeProvenance {
    pub repo: String,
    pub commit: String,
    pub file: String,
}

impl ResumeProvenance {
    /// `None` unless `commit` is a 40-hex string (case-folded), the same
    /// test the network range cache applies before trusting retained bytes.
    pub fn new(repo: &str, commit: &str, file: &str) -> Option<Self> {
        (commit.len() == 40 && commit.bytes().all(|b| b.is_ascii_hexdigit())).then(|| Self {
            repo: repo.to_string(),
            commit: commit.to_ascii_lowercase(),
            file: file.to_string(),
        })
    }
}

fn record_path(dir: &Path) -> PathBuf {
    dir.join(RECORD_NAME)
}

/// Decides whether leftover layer files may be adopted, and makes the
/// directory consistent with the answer: on a mismatch every leftover
/// `layer_*.bin` is deleted (so a later resume cannot adopt a stale tail
/// under a freshly written record) and, when `current` is known, the new
/// record is written before any layer is.
///
/// Returns `true` only when the stored record equals `current`.
pub(crate) fn prepare(dir: &Path, current: Option<&ResumeProvenance>) -> std::io::Result<bool> {
    let path = record_path(dir);
    let stored: Option<ResumeProvenance> = std::fs::read(&path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok());
    if let (Some(stored), Some(current)) = (&stored, current) {
        if stored == current {
            return Ok(true);
        }
    }
    let experts = dir.join("packed_experts");
    if let Ok(entries) = std::fs::read_dir(&experts) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("layer_") && name.ends_with(".bin") {
                std::fs::remove_file(entry.path())?;
            }
        }
    }
    match current {
        Some(current) => {
            std::fs::create_dir_all(dir)?;
            let tmp = dir.join(format!("{RECORD_NAME}.tmp"));
            std::fs::write(&tmp, serde_json::to_vec(current)?)?;
            std::fs::rename(&tmp, &path)?;
        }
        None => match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        },
    }
    Ok(false)
}

/// Drops the record once the install is complete; it only matters while
/// layers are being written. Best effort, the install is already valid.
pub(crate) fn finish(dir: &Path) {
    let _ = std::fs::remove_file(record_path(dir));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ts-resume-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("packed_experts")).unwrap();
        dir
    }

    fn prov(commit_char: char) -> ResumeProvenance {
        ResumeProvenance::new("o/r", &commit_char.to_string().repeat(40), "m.gguf").unwrap()
    }

    #[test]
    fn only_a_full_hex_commit_is_an_identity() {
        assert!(ResumeProvenance::new("o/r", "main", "f").is_none());
        assert!(ResumeProvenance::new("o/r", &"g".repeat(40), "f").is_none());
        assert!(ResumeProvenance::new("o/r", &"A".repeat(40), "f").is_some());
    }

    #[test]
    fn same_record_adopts_and_different_or_absent_deletes_leftovers() {
        let dir = temp();
        let layer = dir.join("packed_experts/layer_00.bin");

        // First walk: nothing stored, so nothing adopted; record written.
        assert!(!prepare(&dir, Some(&prov('a'))).unwrap());
        std::fs::write(&layer, b"x").unwrap();
        // Same commit: adopt, leftover kept.
        assert!(prepare(&dir, Some(&prov('a'))).unwrap());
        assert!(layer.exists());
        // Same size, different commit: refuse and delete.
        assert!(!prepare(&dir, Some(&prov('b'))).unwrap());
        assert!(!layer.exists());

        // A walk with no provenance clears the record, so its layers are
        // never trusted by a later provenanced walk.
        std::fs::write(&layer, b"x").unwrap();
        assert!(!prepare(&dir, None).unwrap());
        assert!(!layer.exists());
        std::fs::write(&layer, b"x").unwrap();
        assert!(!prepare(&dir, Some(&prov('b'))).unwrap());
        assert!(!layer.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
