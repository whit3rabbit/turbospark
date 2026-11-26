//! What a number was measured on. Every worker record carries this, so a
//! ledger line can be traced to the exact binary, model and machine state
//! without trusting the label an arm was given.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    /// HEAD of the source tree this binary was built from, read at run time.
    pub git_rev: Option<String>,
    /// Tracked files modified in that tree at run time.
    pub git_dirty: Option<bool>,
    /// SHA-256 of the running executable: the identity that matters when the
    /// tree has moved since the build.
    pub exe_sha256: Option<String>,
    pub exe_path: Option<String>,
    pub chip: Option<String>,
    pub macos: Option<String>,
    pub macos_build: Option<String>,
    /// Xcode and Metal compiler versions (LAB-30): a compiler bump can move
    /// output bits with no source change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub toolchain_id: Option<String>,
    pub features: Vec<String>,
    /// Every `TURBOSPARK_*` variable this process saw.
    pub turbospark_env: BTreeMap<String, String>,
}

fn capture(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The source tree this crate was compiled from.
pub fn source_root() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

pub fn git_rev(root: &Path) -> Option<String> {
    capture("git", &["-C", &root.to_string_lossy(), "rev-parse", "HEAD"])
}

pub fn git_dirty(root: &Path) -> Option<bool> {
    capture(
        "git",
        &[
            "-C",
            &root.to_string_lossy(),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
    )
    .map(|s| !s.is_empty())
}

pub fn sha256_file(path: &Path) -> Option<String> {
    model_io::hash_file(path, 8 << 20).ok()
}

pub fn turbospark_env() -> BTreeMap<String, String> {
    std::env::vars()
        .filter(|(k, _)| k.starts_with("TURBOSPARK_"))
        .collect()
}

/// One line from `xcodebuild -version` (two lines) and the first line of
/// `xcrun metal --version`. `None` when neither tool answers.
fn toolchain_id(xcode: Option<String>, metal: Option<String>) -> Option<String> {
    let line = |s: &str, n: usize| {
        s.lines()
            .take(n)
            .map(str::trim)
            .collect::<Vec<_>>()
            .join(" ")
    };
    let parts: Vec<String> = [xcode.map(|x| line(&x, 2)), metal.map(|m| line(&m, 1))]
        .into_iter()
        .flatten()
        .filter(|p| !p.is_empty())
        .collect();
    (!parts.is_empty()).then(|| parts.join("; "))
}

pub fn collect() -> Provenance {
    let root = source_root();
    let exe = std::env::current_exe().ok();
    let mut features = Vec::new();
    if cfg!(feature = "ane-prefill") {
        features.push("ane-prefill".to_string());
    }
    Provenance {
        git_rev: git_rev(root),
        git_dirty: git_dirty(root),
        exe_sha256: exe.as_deref().and_then(sha256_file),
        exe_path: exe.map(|p| p.display().to_string()),
        chip: turbospark_bench::memory::chip_brand_string(),
        macos: capture("sw_vers", &["-productVersion"]),
        macos_build: capture("sw_vers", &["-buildVersion"]),
        toolchain_id: toolchain_id(
            capture("xcodebuild", &["-version"]),
            capture("xcrun", &["metal", "--version"]),
        ),
        features,
        turbospark_env: turbospark_env(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toolchain_lines_join_and_absence_stays_absent() {
        assert_eq!(
            toolchain_id(
                Some("Xcode 16.4\nBuild version 16F6\n".into()),
                Some("Apple metal version 32023\nTarget: air64".into())
            )
            .as_deref(),
            Some("Xcode 16.4 Build version 16F6; Apple metal version 32023")
        );
        assert_eq!(toolchain_id(None, None), None);
        assert_eq!(toolchain_id(Some(String::new()), None), None);
    }
}
