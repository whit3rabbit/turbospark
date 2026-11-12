//! Holds the output contract to the source. Offline, not ignored: a new
//! ledger kind that nobody added to `schema.rs` and the schema document, or a
//! command whose `--json` stdout is not exactly one envelope, fails here
//! instead of breaking a dashboard or an agent later.

use std::path::{Path, PathBuf};

use turbospark_lab::schema::{is_known_kind, SCHEMA};

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let p = e.path();
        if p.is_dir() {
            rust_sources(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// The kind literal of every `.append(` call: the first string literal
/// after it, on the same line or the next (rustfmt splits long calls).
fn appended_kinds() -> Vec<(String, String)> {
    let mut files = Vec::new();
    rust_sources(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut files,
    );
    let mut out = Vec::new();
    for f in files {
        let text = std::fs::read_to_string(&f).unwrap();
        // The ledger's own unit tests append arbitrary kinds on purpose.
        if f.ends_with("ledger.rs") {
            continue;
        }
        for (i, _) in text.match_indices(".append(") {
            let rest = &text[i + ".append(".len()..];
            let rest = rest.trim_start();
            // `.append(true)` on OpenOptions and similar: not a ledger call.
            let Some(lit) = rest.strip_prefix('"') else {
                continue;
            };
            let kind = &lit[..lit.find('"').unwrap()];
            out.push((f.display().to_string(), kind.to_string()));
        }
    }
    out
}

#[test]
fn every_appended_kind_is_in_the_contract() {
    let kinds = appended_kinds();
    assert!(
        kinds.iter().any(|(_, k)| k == "run") && kinds.iter().any(|(_, k)| k == "kernel_curve"),
        "the scan found no appends; the pattern drifted: {kinds:?}"
    );
    for (file, kind) in &kinds {
        assert!(
            is_known_kind(kind),
            "{file} appends {kind:?}, which schema.rs KINDS lacks"
        );
    }
}

#[cfg(target_os = "macos")]
fn tslab(args: &[&str]) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_tslab"))
        .args(args)
        // Commands here never open a ledger, but keep any write away from
        // the machine's real lab directory.
        .env(
            "TURBOSPARK_LAB_DIR",
            std::env::temp_dir().join("tslab-schema-contract"),
        )
        .output()
        .expect("run tslab")
}

#[cfg(target_os = "macos")]
#[test]
fn json_stdout_is_exactly_one_envelope() {
    for args in [
        &["knobs", "--json"][..],
        &["--json", "schema"][..],
        &["kernel", "--list", "--json"][..],
    ] {
        let out = tslab(args);
        assert!(
            out.status.success(),
            "{args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout = String::from_utf8(out.stdout).unwrap();
        assert_eq!(
            stdout.lines().count(),
            1,
            "{args:?} printed more than one line: {stdout}"
        );
        let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
        assert_eq!(v["schema"], SCHEMA, "{args:?}");
        assert_eq!(v["ok"], true);
        assert_eq!(v["exit_code"], 0);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn a_failed_command_still_prints_an_envelope() {
    let out = tslab(&["kernel", "--json"]);
    assert_eq!(out.status.code(), Some(1));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["ok"], false);
    assert!(v["error"].as_str().unwrap().contains("--model"), "{v}");
}

#[cfg(target_os = "macos")]
#[test]
fn without_json_stdout_has_the_human_table() {
    let out = tslab(&["knobs"]);
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("TURBOSPARK_INT4_MMA8"), "{stdout}");
    assert!(serde_json::from_str::<serde_json::Value>(&stdout).is_err());
}
