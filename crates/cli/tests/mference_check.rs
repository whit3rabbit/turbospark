//! Black-box tests of the `mference-check` binary: exit codes and stream
//! routing for a help request, a validated invocation, and a parse failure.

use std::process::Command;

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_mference-check"))
        .args(args)
        .output()
        .expect("binary should run")
}

#[test]
fn help_prints_usage_to_stdout_and_exits_zero() {
    let output = run(&["--help"]);
    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--model"));
    assert!(output.stderr.is_empty());
}

#[test]
fn missing_required_model_exits_two_and_writes_stderr() {
    let output = run(&["--prompt", "hi"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--model") || stderr.contains("MissingRequired"));
}

#[test]
fn valid_invocation_exits_zero_and_prints_resolved_request() {
    let output = run(&[
        "--model",
        "/tmp/does-not-need-to-exist.gturbo",
        "--prompt",
        "hi",
    ]);
    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("resolved invocation:"));
    assert!(stdout.contains("model: /tmp/does-not-need-to-exist.gturbo"));
    assert!(stdout.contains("max_new: 1024"));
    // On macOS, real generation is attempted next (see generate.rs) and
    // fails cleanly since the model path does not exist; on other
    // platforms nothing is attempted and stderr stays empty.
    #[cfg(target_os = "macos")]
    assert!(String::from_utf8_lossy(&output.stderr).contains("not attempting real generation"));
    #[cfg(not(target_os = "macos"))]
    assert!(output.stderr.is_empty());
}

#[test]
fn no_arguments_fails_on_missing_required_model() {
    let output = run(&[]);
    assert_eq!(output.status.code(), Some(2));
}
