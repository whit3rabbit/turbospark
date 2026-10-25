//! One measuring session per machine.
//!
//! Two sessions timing models on one GPU each read the other's load as their
//! own result. The host guard can only notice that after the fact; this lock
//! prevents it among tslab users. The path is documented in
//! `docs/BENCHMARKING.md` so the Python drivers can take the same lock
//! (`fcntl.flock` on the same file is the same kernel lock).

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

pub struct MachineLock {
    _file: File,
    path: PathBuf,
}

impl MachineLock {
    /// Take `<lab>/.machine.lock` without blocking. On contention the error
    /// names the holder (pid and purpose), which the holder wrote on entry.
    pub fn acquire(lab: &Path, purpose: &str) -> Result<Self, String> {
        std::fs::create_dir_all(lab).map_err(|e| format!("{}: {e}", lab.display()))?;
        let path = lab.join(".machine.lock");
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        // SAFETY: the descriptor belongs to `file`, which outlives the lock.
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            let mut holder = String::new();
            let _ = file.read_to_string(&mut holder);
            return Err(format!(
                "another measuring session holds {} ({}); wait for it, or check that it is alive",
                path.display(),
                holder.trim()
            ));
        }
        let note = format!(
            "pid {} since {} ms: {purpose}",
            std::process::id(),
            crate::ledger::unix_ms()
        );
        let _ = file.set_len(0);
        let _ = file.seek(SeekFrom::Start(0));
        let _ = file.write_all(note.as_bytes());
        let _ = file.sync_all();
        Ok(Self { _file: file, path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_holder_is_refused_with_the_first_named() {
        let dir = std::env::temp_dir().join(format!("tslab-lock-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let first = MachineLock::acquire(&dir, "first purpose").unwrap();
        let err = MachineLock::acquire(&dir, "second").err().unwrap();
        assert!(err.contains("first purpose"), "{err}");
        drop(first);
        MachineLock::acquire(&dir, "third").unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
