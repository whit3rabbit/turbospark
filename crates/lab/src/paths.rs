//! Where results live. Durable by default: a ledger in `/tmp` is a ledger
//! that a reboot or a scratchpad cleanup deletes.

use std::path::{Path, PathBuf};

/// `$TURBOSPARK_LAB_DIR`, else `$TURBOSPARK_HOME/lab`, else `~/.turbospark/lab`.
pub fn lab_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("TURBOSPARK_LAB_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(home) = std::env::var_os("TURBOSPARK_HOME") {
        return PathBuf::from(home).join("lab");
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default();
    home.join(".turbospark").join("lab")
}

/// An experiment's directory under `lab`. The name becomes a path component,
/// so it is restricted to a safe alphabet rather than sanitized: a name that
/// silently changed would file results under a different experiment.
pub fn experiment_dir(lab: &Path, name: &str) -> Result<PathBuf, String> {
    validate_name(name)?;
    Ok(lab.join(name))
}

/// Names for experiments and arms: `[A-Za-z0-9._-]`, 1 to 64 bytes, not
/// starting with a dot.
pub fn validate_name(name: &str) -> Result<(), String> {
    let ok = !name.is_empty()
        && name.len() <= 64
        && !name.starts_with('.')
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'));
    if ok {
        Ok(())
    } else {
        Err(format!(
            "invalid name {name:?}: use 1-64 of [A-Za-z0-9._-], not starting with '.'"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_restricted_not_sanitized() {
        assert!(validate_name("qwen36-slots_16.a").is_ok());
        for bad in ["", ".hidden", "a/b", "../x", "with space", &"x".repeat(65)] {
            assert!(validate_name(bad).is_err(), "{bad:?} accepted");
        }
    }
}
