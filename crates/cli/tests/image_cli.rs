//! Offline contract tests for `turbospark-image`.
//!
//! These tests stop before opening a model. The packed end-to-end gates live
//! under `crates/image/tests` and are ignored until a pinned image install is
//! supplied, while malformed requests and output safety must stay cheap.

use std::process::Command;

fn bin() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_turbospark-image"));
    command.env(
        "TURBOSPARK_HOME",
        std::env::temp_dir().join(format!("turbospark-image-cli-{}", std::process::id())),
    );
    command
}

fn run(args: &[&str]) -> (i32, String, String) {
    let output = bin().args(args).output().expect("image binary runs");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
    )
}

#[test]
fn help_identifies_native_default_and_reference_escape_hatch() {
    let (code, stdout, _) = run(&["--help"]);
    assert_eq!(code, 0);
    assert!(stdout.contains("native Metal"), "{stdout}");
    assert!(stdout.contains("--backend native|reference"), "{stdout}");
    assert!(stdout.contains("never overwritten"), "{stdout}");
}

#[test]
fn invalid_request_is_rejected_before_model_io() {
    for (flag, value, expected) in [
        ("--width", "512", "outside the IG2 envelope"),
        ("--steps", "8", "unsupported image request envelope"),
    ] {
        let (code, _, stderr) = run(&[
            "generate",
            "--model",
            "/does/not/exist",
            "--prompt",
            "a lighthouse",
            "--output",
            "/tmp/image-cli-invalid.png",
            flag,
            value,
            "--backend",
            "reference",
        ]);
        assert_eq!(code, 1, "{flag}: {stderr}");
        assert!(stderr.contains(expected), "{flag}: {stderr}");
        assert!(
            !stderr.contains("failed to read image manifest"),
            "{stderr}"
        );
    }
}

#[test]
fn overwrite_is_refused_before_model_io() {
    let path = std::env::temp_dir().join(format!(
        "turbospark-image-existing-{}.png",
        std::process::id()
    ));
    std::fs::write(&path, b"keep this file").expect("write sentinel");
    let (code, _, stderr) = run(&[
        "generate",
        "--model",
        "/does/not/exist",
        "--prompt",
        "a lighthouse",
        "--output",
        path.to_str().expect("temporary path is UTF-8"),
        "--backend",
        "reference",
    ]);
    assert_eq!(code, 1);
    assert!(
        stderr.contains("refusing to overwrite existing output"),
        "{stderr}"
    );
    assert_eq!(
        std::fs::read(&path).expect("read sentinel"),
        b"keep this file"
    );
    std::fs::remove_file(path).expect("remove sentinel");
}

#[cfg(not(target_os = "macos"))]
#[test]
fn native_backend_is_refused_without_macOS_before_model_io() {
    let (code, _, stderr) = run(&[
        "generate",
        "--model",
        "/does/not/exist",
        "--prompt",
        "a lighthouse",
        "--output",
        "/tmp/image-cli-no-device.png",
    ]);
    assert_eq!(code, 1);
    assert!(stderr.contains("requires macOS Metal"), "{stderr}");
    assert!(
        !stderr.contains("failed to read image manifest"),
        "{stderr}"
    );
}
