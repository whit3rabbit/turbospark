//! Refusal policy when another process holds the model.
//!
//! Two model loads of a multi-gigabyte install on one machine is the failure
//! this prevents: the second one swaps or fails with an allocation error that
//! names nothing. The simplest safe behavior is to REFUSE, never to wait or
//! to share, and to say who holds it:
//!
//! 1. Another `turbospark-acp-agent` on the same model directory holds an
//!    advisory `flock` on `<root>/run/acp-<fnv64(model dir)>.lock`. The lock
//!    is tied to the open file description, so a crash or kill releases it.
//! 2. A running `turbospark serve` daemon whose recorded `--model` resolves to
//!    the same directory (read from `<root>/run/server.meta`, pid checked live).
//!
//! NOT covered, stated so nobody assumes it: the macOS app takes no lock and
//! writes no record a CLI can read, so an ACP process and the app can both
//! hold the same model. That needs the app to take the same lock and is
//! deferred (docs/ACP.md). The check is advisory between cooperating
//! TurboSpark processes only.
//!
//! Unsafe: two libc calls (`flock`, `kill` with signal 0) on a valid fd and a
//! pid, the same pattern as `daemon.rs`.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

/// Held for the life of the loaded model. Dropping it closes the descriptor,
/// which releases the lock.
#[derive(Debug)]
pub struct ModelLock {
    _file: File,
}

/// FNV-1a, spelled out so the lock file name is stable across Rust versions
/// (`DefaultHasher` makes no such promise).
fn fnv64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub fn lock_path(run_dir: &Path, model_dir: &Path) -> PathBuf {
    let key = model_dir.to_string_lossy();
    run_dir.join(format!("acp-{:016x}.lock", fnv64(key.as_bytes())))
}

#[cfg(unix)]
pub fn acquire(run_dir: &Path, model_dir: &Path) -> Result<ModelLock, String> {
    use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
    use std::os::unix::io::AsRawFd;

    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(run_dir)
        .map_err(|e| format!("creating {}: {e}", run_dir.display()))?;
    let path = lock_path(run_dir, model_dir);
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .mode(0o600)
        .open(&path)
        .map_err(|e| format!("opening model lock {}: {e}", path.display()))?;
    // SAFETY: `flock` on a valid, open descriptor owned by `file`.
    let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if rc != 0 {
        let err = std::io::Error::last_os_error();
        if err.kind() == std::io::ErrorKind::WouldBlock {
            return Err(format!(
                "the model {} is already loaded by another `turbospark acp` process \
                 (lock {}); close that editor session or point this one at a different model",
                model_dir.display(),
                path.display()
            ));
        }
        return Err(format!("locking {}: {err}", path.display()));
    }
    Ok(ModelLock { _file: file })
}

#[cfg(not(unix))]
pub fn acquire(_run_dir: &Path, _model_dir: &Path) -> Result<ModelLock, String> {
    Err("model locking needs a unix host".to_string())
}

#[cfg(unix)]
pub fn pid_alive(pid: i32) -> bool {
    if pid <= 0 {
        return false;
    }
    // SAFETY: signal 0 only checks that the pid exists.
    let rc = unsafe { libc::kill(pid, 0) };
    rc == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(not(unix))]
pub fn pid_alive(_pid: i32) -> bool {
    false
}

/// The `--model` value a `server.meta` record says its daemon was started
/// with, when that daemon's pid is alive per `alive`.
pub fn daemon_model(meta_json: &str, alive: impl Fn(i32) -> bool) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(meta_json).ok()?;
    let pid = v.get("pid")?.as_i64()? as i32;
    if !alive(pid) {
        return None;
    }
    let args = v.get("args")?.as_array()?;
    let mut it = args.iter().filter_map(|a| a.as_str());
    while let Some(a) = it.next() {
        if a == "--model" {
            return it.next().map(str::to_string);
        }
        if let Some(rest) = a.strip_prefix("--model=") {
            return Some(rest.to_string());
        }
    }
    None
}

/// Refuses when a live daemon serves `model_dir`. `resolve` maps the daemon's
/// recorded alias or path to a directory, the way `--model` is resolved.
pub fn refuse_if_daemon_holds(
    run_dir: &Path,
    model_dir: &Path,
    resolve: impl Fn(&str) -> PathBuf,
    alive: impl Fn(i32) -> bool,
) -> Result<(), String> {
    let Ok(meta) = std::fs::read_to_string(run_dir.join("server.meta")) else {
        return Ok(());
    };
    let Some(model) = daemon_model(&meta, alive) else {
        return Ok(());
    };
    let theirs = resolve(&model);
    let same = match (theirs.canonicalize(), model_dir.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => theirs == model_dir,
    };
    if same {
        return Err(format!(
            "the model {} is being served by the `turbospark serve` daemon; stop the \
             server first, or point `turbospark acp` at a different model",
            model_dir.display()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("acp-lock-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[cfg(unix)]
    #[test]
    fn second_holder_is_refused_and_release_frees_it() {
        let run = tmp("flock");
        let model = run.join("m.gturbo");
        let a = acquire(&run, &model).expect("first holder");
        let err = acquire(&run, &model).unwrap_err();
        assert!(err.contains("already loaded"), "{err}");
        // A different model is independent.
        let other = acquire(&run, &run.join("other.gturbo")).expect("other model");
        drop(a);
        acquire(&run, &model).expect("released on drop");
        drop(other);
        let _ = std::fs::remove_dir_all(&run);
    }

    #[test]
    fn lock_name_is_stable() {
        let p = lock_path(Path::new("/r"), Path::new("/m"));
        assert_eq!(p, PathBuf::from("/r/acp-07d65b07b49cc4c9.lock"));
    }

    #[test]
    fn daemon_meta_parsing() {
        let meta = r#"{"pid": 42, "port": 8080, "args": ["--port","8080","--model","qwen"]}"#;
        assert_eq!(daemon_model(meta, |_| true), Some("qwen".into()));
        assert_eq!(
            daemon_model(meta, |_| false),
            None,
            "dead pid does not hold"
        );
        let eq = r#"{"pid": 1, "args": ["--model=/x/y"]}"#;
        assert_eq!(daemon_model(eq, |_| true), Some("/x/y".into()));
        assert_eq!(daemon_model("not json", |_| true), None);
    }

    #[test]
    fn live_daemon_on_same_model_refuses() {
        let run = tmp("daemon");
        let model = run.join("m.gturbo");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(
            run.join("server.meta"),
            r#"{"pid": 7, "args": ["--model","alias"]}"#,
        )
        .unwrap();
        let same = |_: &str| model.clone();
        let err = refuse_if_daemon_holds(&run, &model, same, |_| true).unwrap_err();
        assert!(err.contains("daemon"), "{err}");
        // Dead daemon, or a daemon on another model: allowed.
        assert!(refuse_if_daemon_holds(&run, &model, same, |_| false).is_ok());
        let other = |_: &str| run.join("elsewhere");
        assert!(refuse_if_daemon_holds(&run, &model, other, |_| true).is_ok());
        // No record at all: allowed.
        assert!(refuse_if_daemon_holds(&run.join("none"), &model, same, |_| true).is_ok());
        let _ = std::fs::remove_dir_all(&run);
    }
}
