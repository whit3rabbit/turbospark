//! What an experiment may not touch, and whether the evaluator running it
//! is the trusted build.
//!
//! An agent iterating against tslab is rewarded by tslab's verdict, so the
//! verdict's inputs are off limits to it: this crate, the shared quality and
//! memory math, the frozen prompts, frozen benchmark rows, the agent rules,
//! manifests and the lockfile, and every EXISTING test file (their
//! tolerances are part of the contract). New tests are allowed and expected.
//! `run-experiment` refuses before measuring if the experiment's diff from
//! its base breaks any of this.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Paths no experiment may add, modify or delete (prefix match).
pub const FORBIDDEN_PREFIXES: &[&str] = &[
    "crates/lab/",
    "crates/bench/src/logit_compare.rs",
    "crates/bench/src/quality_corpus.rs",
    "crates/bench/src/memory.rs",
    "crates/bench/prompts/",
    "docs/BENCHMARKS.md",
    "Cargo.lock",
    ".github/",
];

/// File names no experiment may touch anywhere in the tree.
pub const FORBIDDEN_NAMES: &[&str] = &["AGENTS.md", "CLAUDE.md", "Cargo.toml"];

/// One changed path with its git status letter (A, M, D, R, ...).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Change {
    pub status: char,
    pub path: String,
}

fn is_test_file(path: &str) -> bool {
    let parts: Vec<&str> = path.split('/').collect();
    parts.len() >= 4 && parts[0] == "crates" && parts[2] == "tests"
}

/// Every rule an experiment's changes break, empty when clean.
pub fn violations(changes: &[Change]) -> Vec<String> {
    let mut out = Vec::new();
    for c in changes {
        let name = c.path.rsplit('/').next().unwrap_or(&c.path);
        if let Some(p) = FORBIDDEN_PREFIXES.iter().find(|p| c.path.starts_with(*p)) {
            out.push(format!(
                "{} {}: under {p}, which the evaluator owns",
                c.status, c.path
            ));
        } else if FORBIDDEN_NAMES.contains(&name) {
            out.push(format!(
                "{} {}: {name} files are off limits to experiments",
                c.status, c.path
            ));
        } else if is_test_file(&c.path) && c.status != 'A' {
            out.push(format!(
                "{} {}: existing tests (and their tolerances) may not change; add a new test file instead",
                c.status, c.path
            ));
        }
    }
    out
}

fn git(repo: &Path, args: &[&str]) -> Result<String, String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .map_err(|e| format!("git: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// The top of the repository containing `dir`.
pub fn repo_root(dir: &Path) -> Result<PathBuf, String> {
    Ok(PathBuf::from(
        git(dir, &["rev-parse", "--show-toplevel"])?.trim(),
    ))
}

/// Changes from `base` to the working tree, untracked files included.
pub fn changes_since(repo: &Path, base: &str) -> Result<Vec<Change>, String> {
    let mut out = Vec::new();
    for line in git(repo, &["diff", "--name-status", "--no-renames", base])?.lines() {
        let mut it = line.split('\t');
        let (Some(status), Some(path)) = (it.next(), it.next()) else {
            continue;
        };
        out.push(Change {
            status: status.chars().next().unwrap_or('?'),
            path: path.to_string(),
        });
    }
    for path in git(repo, &["ls-files", "--others", "--exclude-standard"])?.lines() {
        out.push(Change {
            status: 'A',
            path: path.to_string(),
        });
    }
    Ok(out)
}

/// The default base: where HEAD left `main`.
pub fn default_base(repo: &Path) -> Result<String, String> {
    Ok(git(repo, &["merge-base", "HEAD", "main"])?
        .trim()
        .to_string())
}

/// The running `tslab` must be a copy installed under `<lab>/bin/`, built
/// from a clean tree before the experiment started. A `tslab` built inside
/// the experiment's own tree would be grading itself.
pub fn trusted_exe(lab: &Path) -> Result<PathBuf, String> {
    let exe = std::env::current_exe()
        .and_then(|p| p.canonicalize())
        .map_err(|e| e.to_string())?;
    let bin = lab.join("bin");
    let bin = bin.canonicalize().unwrap_or(bin);
    if exe.starts_with(&bin) {
        Ok(exe)
    } else {
        Err(format!(
            "this tslab ({}) is not a trusted install under {}; build it from a clean tree and copy it there \
             (crates/lab/program.md, Setup), or pass --dev-untrusted for a run that can never be kept",
            exe.display(),
            bin.display()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ch(status: char, path: &str) -> Change {
        Change {
            status,
            path: path.into(),
        }
    }

    #[test]
    fn kernel_and_runtime_edits_and_new_tests_are_allowed() {
        let ok = [
            ch('M', "crates/gpu/src/shaders/attention.metal"),
            ch('M', "crates/runtime/src/families/qwen/prefill.rs"),
            ch('A', "crates/gpu/tests/my_kernel_parity.rs"),
            ch('A', "docs/MY_EXPERIMENT.md"),
        ];
        assert!(violations(&ok).is_empty(), "{:?}", violations(&ok));
    }

    #[test]
    fn evaluator_inputs_are_refused() {
        for c in [
            ch('M', "crates/lab/src/stats.rs"),
            ch('A', "crates/lab/src/new.rs"),
            ch('M', "crates/bench/src/logit_compare.rs"),
            ch(
                'M',
                "crates/bench/prompts/real-generation-v1/long-synthesis.txt",
            ),
            ch('M', "docs/BENCHMARKS.md"),
            ch('M', "Cargo.lock"),
            ch('M', "crates/gpu/Cargo.toml"),
            ch('M', "crates/runtime/AGENTS.md"),
            ch('M', "crates/bench/tests/qwen38_prefill_arm_probe.rs"),
            ch('D', "crates/gpu/tests/attention_decode_gqa_rows_parity.rs"),
        ] {
            assert_eq!(
                violations(std::slice::from_ref(&c)).len(),
                1,
                "{c:?} allowed"
            );
        }
    }
}
