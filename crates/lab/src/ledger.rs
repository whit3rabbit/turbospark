//! The append-only results ledger, one JSON object per line.
//!
//! Every run, refusal, verdict, calibration and kernel curve is a line here,
//! including the runs the host guard rejected (`eligible: false`): dropping
//! them would make a noisy host look quiet. Lines are never rewritten. A
//! correction is a new line that names the one it supersedes.

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use serde_json::{json, Value};

use crate::schema::{RecordRef, SCHEMA};

pub struct Ledger {
    path: PathBuf,
}

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Every line this process appended, in order, with its full record, for
/// the `--json` envelope.
static APPENDED: Mutex<Vec<(RecordRef, Value)>> = Mutex::new(Vec::new());

/// Return lines to the front of the collected list: a caller that took
/// them to look (a workflow step) hands them back for the envelope.
pub fn put_back(mut lines: Vec<(RecordRef, Value)>) {
    let mut g = APPENDED.lock().unwrap_or_else(|p| p.into_inner());
    lines.append(&mut g);
    *g = lines;
}

/// The lines appended since the last call, oldest first.
pub fn take_appended() -> Vec<(RecordRef, Value)> {
    std::mem::take(&mut *APPENDED.lock().unwrap_or_else(|p| p.into_inner()))
}

/// Milliseconds since the Unix epoch.
pub fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// A process-unique, time-ordered id: `<unix_ms>-<pid>-<n>`.
pub fn new_id() -> String {
    format!(
        "{}-{}-{}",
        unix_ms(),
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

impl Ledger {
    /// The ledger at `<dir>/ledger.jsonl`; the directory is created.
    pub fn open(dir: &Path) -> Result<Self, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        Ok(Self {
            path: dir.join("ledger.jsonl"),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one record of `kind` and return its id. `payload` must be an
    /// object; `id`, `kind`, `ts_ms` and `schema` are added and win over any
    /// payload keys of the same name.
    ///
    /// One `write` of the whole line under an exclusive `flock`, then fsync:
    /// two processes appending at once (an orchestrator and a `tslab report`
    /// in another shell) cannot interleave bytes within a line.
    pub fn append(&self, kind: &str, payload: Value) -> Result<String, String> {
        let Value::Object(mut map) = payload else {
            return Err("ledger payload must be a JSON object".into());
        };
        let id = new_id();
        map.insert("id".into(), json!(id));
        map.insert("kind".into(), json!(kind));
        map.insert("ts_ms".into(), json!(unix_ms()));
        map.insert("schema".into(), json!(SCHEMA));
        let record = Value::Object(map);
        let mut line = serde_json::to_string(&record).map_err(|e| e.to_string())?;
        line.push('\n');
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| format!("{}: {e}", self.path.display()))?;
        let fd = f.as_raw_fd();
        // SAFETY: `fd` is a valid descriptor owned by `f` for this scope.
        if unsafe { libc::flock(fd, libc::LOCK_EX) } != 0 {
            return Err(format!(
                "{}: flock: {}",
                self.path.display(),
                std::io::Error::last_os_error()
            ));
        }
        let result = f
            .write_all(line.as_bytes())
            .and_then(|_| f.sync_all())
            .map_err(|e| format!("{}: {e}", self.path.display()));
        // SAFETY: as above; unlocking a lock this descriptor holds.
        unsafe { libc::flock(fd, libc::LOCK_UN) };
        result?;
        APPENDED.lock().unwrap_or_else(|p| p.into_inner()).push((
            RecordRef {
                id: id.clone(),
                kind: kind.to_string(),
                ledger: self.path.display().to_string(),
            },
            record,
        ));
        Ok(id)
    }

    /// Every record, in append order. A line that does not parse is an error
    /// rather than skipped: a ledger that silently loses lines is not one.
    pub fn read_all(&self) -> Result<Vec<Value>, String> {
        let f = match std::fs::File::open(&self.path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(format!("{}: {e}", self.path.display())),
        };
        let mut out = Vec::new();
        for (n, line) in BufReader::new(f).lines().enumerate() {
            let line = line.map_err(|e| format!("{}: {e}", self.path.display()))?;
            if line.trim().is_empty() {
                continue;
            }
            out.push(
                serde_json::from_str(&line)
                    .map_err(|e| format!("{} line {}: {e}", self.path.display(), n + 1))?,
            );
        }
        Ok(out)
    }

    /// Records of one kind, in append order.
    pub fn of_kind(&self, kind: &str) -> Result<Vec<Value>, String> {
        Ok(self
            .read_all()?
            .into_iter()
            .filter(|v| v["kind"] == kind)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("tslab-ledger-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn concurrent_appends_keep_every_line_whole() {
        let dir = scratch("conc");
        let threads: Vec<_> = (0..8)
            .map(|t| {
                let dir = dir.clone();
                std::thread::spawn(move || {
                    let ledger = Ledger::open(&dir).unwrap();
                    for i in 0..50 {
                        // A long payload makes a torn write likely if the
                        // lock or the single write were missing.
                        ledger
                            .append("run", json!({"t": t, "i": i, "pad": "x".repeat(4000)}))
                            .unwrap();
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        let all = Ledger::open(&dir).unwrap().read_all().unwrap();
        assert_eq!(all.len(), 400);
        let mut ids: Vec<_> = all.iter().map(|v| v["id"].as_str().unwrap()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 400, "ids not unique");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn reserved_keys_win_and_non_objects_refused() {
        let dir = scratch("keys");
        let ledger = Ledger::open(&dir).unwrap();
        let id = ledger.append("verdict", json!({"kind": "spoof"})).unwrap();
        let rec = &ledger.read_all().unwrap()[0];
        assert_eq!(rec["kind"], "verdict");
        assert_eq!(rec["id"], id.as_str());
        assert_eq!(rec["schema"], SCHEMA);
        assert!(take_appended()
            .iter()
            .any(|(r, v)| r.id == id && r.kind == "verdict" && v["id"] == id.as_str()));
        assert!(ledger.append("run", json!([1, 2])).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
