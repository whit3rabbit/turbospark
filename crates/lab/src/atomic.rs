//! Write-once files: a result that already exists is evidence, and a rerun
//! that replaced it would erase the run it is being compared against.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

/// Write `bytes` to `path` atomically and refuse if `path` exists.
///
/// The bytes land in a sibling temp file first, are fsynced, then hard-linked
/// into place. `hard_link` fails on an existing target, which makes the
/// refusal and the publish one step: there is no window where a reader sees a
/// half-written file or where two writers both succeed.
pub fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let tmp = path.with_extension(format!(
        "{}.tmp-{}",
        path.extension().and_then(|e| e.to_str()).unwrap_or(""),
        std::process::id()
    ));
    let result = (|| {
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .map_err(|e| format!("{}: {e}", tmp.display()))?;
        f.write_all(bytes)
            .and_then(|_| f.sync_all())
            .map_err(|e| format!("{}: {e}", tmp.display()))?;
        fs::hard_link(&tmp, path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                format!("{} already exists; refusing to overwrite", path.display())
            } else {
                format!("{}: {e}", path.display())
            }
        })
    })();
    let _ = fs::remove_file(&tmp);
    result
}

/// [`write_new`] for a JSON value, pretty-printed with a trailing newline.
pub fn write_new_json(path: &Path, value: &serde_json::Value) -> Result<(), String> {
    let mut text = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    text.push('\n');
    write_new(path, text.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_to_overwrite_and_leaves_no_temp() {
        let dir = std::env::temp_dir().join(format!("tslab-atomic-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("run.json");
        write_new(&path, b"first").unwrap();
        let err = write_new(&path, b"second").unwrap_err();
        assert!(err.contains("refusing to overwrite"), "{err}");
        assert_eq!(fs::read(&path).unwrap(), b"first");
        let leftovers: Vec<_> = fs::read_dir(&dir).unwrap().collect();
        assert_eq!(leftovers.len(), 1, "temp file left behind");
        fs::remove_dir_all(&dir).unwrap();
    }
}
