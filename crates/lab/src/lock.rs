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
use std::sync::Mutex;

pub struct MachineLock {
    /// `None` for a nested guard under a workflow's hold (`hold`).
    _file: Option<File>,
    path: PathBuf,
    held: bool,
}

/// The lock path a workflow holds for this process, if any. While it is
/// set, `acquire` on that path nests instead of taking a second flock,
/// which the kernel would refuse: a workflow's steps are commands that
/// each take the lock, run inside one hold.
static HELD: Mutex<Option<PathBuf>> = Mutex::new(None);

impl MachineLock {
    /// Take the lock for a workflow: like `acquire`, and every `acquire` of
    /// the same path in this process nests under it until it drops.
    pub fn hold(lab: &Path, purpose: &str) -> Result<Self, String> {
        let mut lock = Self::acquire(lab, purpose)?;
        if lock._file.is_some() {
            *HELD.lock().unwrap_or_else(|p| p.into_inner()) = Some(lock.path.clone());
            lock.held = true;
        }
        Ok(lock)
    }

    /// Take `<lab>/.machine.lock` without blocking. On contention the error
    /// names the holder (pid and purpose), which the holder wrote on entry.
    pub fn acquire(lab: &Path, purpose: &str) -> Result<Self, String> {
        std::fs::create_dir_all(lab).map_err(|e| format!("{}: {e}", lab.display()))?;
        let path = lab.join(".machine.lock");
        if HELD.lock().unwrap_or_else(|p| p.into_inner()).as_deref() == Some(path.as_path()) {
            return Ok(Self {
                _file: None,
                path,
                held: false,
            });
        }
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
        Ok(Self {
            _file: Some(file),
            path,
            held: false,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for MachineLock {
    fn drop(&mut self) {
        if self.held {
            *HELD.lock().unwrap_or_else(|p| p.into_inner()) = None;
        }
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

    #[test]
    fn a_workflow_hold_nests_its_steps_and_releases() {
        let dir = std::env::temp_dir().join(format!("tslab-lock-hold-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let hold = MachineLock::hold(&dir, "workflow").unwrap();
        let step = MachineLock::acquire(&dir, "step").unwrap();
        drop(step);
        MachineLock::acquire(&dir, "another step").unwrap();
        drop(hold);
        // After the hold, a second taker in this process is refused again.
        let a = MachineLock::acquire(&dir, "a").unwrap();
        assert!(MachineLock::acquire(&dir, "b").is_err());
        drop(a);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
