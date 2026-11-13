//! `skill/scripts/preflight.sh` must flag the same processes the host guard
//! does. The guard matches `MODEL_MARKERS` fragments against the executable
//! path; an exact-basename list let test binaries such as `x_gate-<hash>`
//! through. These tests hold the script's copy of the list to
//! `src/host.rs` and run the script against a fake `ps`.

use std::path::PathBuf;
use std::process::Command;

const SCRIPT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/skill/scripts/preflight.sh");

fn host_markers() -> Vec<String> {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/host.rs"))
        .expect("read host.rs");
    let start = src.find("const MODEL_MARKERS").expect("MODEL_MARKERS in host.rs");
    let body = &src[start..];
    let open = body.find('[').expect("list open");
    let close = body.find("];").expect("list close");
    body[open..close]
        .split('"')
        .skip(1)
        .step_by(2)
        .map(str::to_string)
        .collect()
}

fn script_markers() -> Vec<String> {
    let src = std::fs::read_to_string(SCRIPT).expect("read preflight.sh");
    let line = src
        .lines()
        .find(|l| l.starts_with("MODEL_MARKERS='"))
        .expect("MODEL_MARKERS line in preflight.sh");
    line.trim_start_matches("MODEL_MARKERS='")
        .trim_end_matches('\'')
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

#[test]
fn script_marker_list_matches_the_guard() {
    let host = host_markers();
    assert!(host.len() > 10, "parsed too few markers from host.rs: {host:?}");
    assert_eq!(script_markers(), host);
}

/// Run the script with `ps` returning `comms` and `pmset` reporting AC power,
/// and return its stdout.
fn run_with_fake_ps(tag: &str, comms: &[&str]) -> String {
    let dir: PathBuf = std::env::temp_dir().join(format!(
        "tslab-preflight-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let ps = dir.join("ps");
    std::fs::write(&ps, format!("#!/bin/sh\ncat <<'EOF'\n{}\nEOF\n", comms.join("\n"))).unwrap();
    let pmset = dir.join("pmset");
    std::fs::write(&pmset, "#!/bin/sh\necho 'Now drawing from AC Power'\n").unwrap();
    for f in [&ps, &pmset] {
        let status = Command::new("chmod").arg("+x").arg(f).status().unwrap();
        assert!(status.success());
    }
    let path = format!("{}:{}", dir.display(), std::env::var("PATH").unwrap_or_default());
    let out = Command::new("sh")
        .arg(SCRIPT)
        .env("PATH", path)
        .output()
        .expect("run preflight.sh");
    let _ = std::fs::remove_dir_all(&dir);
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn fragment_named_test_binaries_are_busy() {
    for (tag, comm) in [
        ("gate", "/x/target/debug/deps/qwen36_quality_gate-abc123"),
        ("probe", "/x/target/debug/deps/kflash_tree_replay_probe-abc123"),
        ("oracle", "/x/target/debug/deps/vision_memory_oracle-abc123"),
        ("bench", "/x/target/release/deps/decode_bench-abc123"),
        ("upper", "/Users/me/Dev/OLLAMA/serve"),
    ] {
        let out = run_with_fake_ps(tag, &["/sbin/launchd", comm]);
        assert!(out.contains("busy processes:"), "{comm} not flagged:\n{out}");
    }
}

#[test]
fn unrelated_processes_stay_quiet() {
    let out = run_with_fake_ps(
        "quiet",
        &["/sbin/launchd", "/usr/libexec/logd", "/bin/zsh", "/usr/bin/ssh"],
    );
    assert!(
        out.contains("no build, trace or other model process running"),
        "false positive:\n{out}"
    );
}

#[test]
fn basename_only_names_still_match_but_not_as_a_path_component() {
    let out = run_with_fake_ps("cargo", &["/Users/me/.cargo/bin/cargo"]);
    assert!(out.contains("busy processes:"), "cargo not flagged:\n{out}");
    // `.cargo` as a directory is not the cargo binary.
    let out = run_with_fake_ps("cargodir", &["/Users/me/.cargo/registry/some-tool"]);
    assert!(out.contains("no build, trace or other model process running"), "{out}");
}
