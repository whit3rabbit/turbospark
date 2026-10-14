//! Black-box tests of the `turbospark-check` binary: exit codes and stream
//! routing for a help request, a validated invocation, and a parse failure.

use std::process::Command;

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_turbospark-check"))
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
fn valid_invocation_prints_resolved_request_and_reports_a_failed_generation() {
    let output = run(&[
        "--model",
        "/tmp/does-not-need-to-exist.gturbo",
        "--prompt",
        "hi",
    ]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("resolved invocation:"));
    assert!(stdout.contains("model: /tmp/does-not-need-to-exist.gturbo"));
    assert!(stdout.contains("max_new: 1024"));
    // On macOS, real generation is attempted next (see generate.rs) and
    // fails cleanly since the model path does not exist, which is exit 1: the
    // command line was valid (not 2) but the run it asked for did not happen
    // (not 0). On other platforms nothing is attempted, stderr stays empty
    // and the exit status is 0.
    #[cfg(target_os = "macos")]
    {
        assert_eq!(output.status.code(), Some(1));
        assert!(String::from_utf8_lossy(&output.stderr).contains("not attempting real generation"));
    }
    #[cfg(not(target_os = "macos"))]
    {
        assert_eq!(output.status.code(), Some(0));
        assert!(output.stderr.is_empty());
    }
}

/// `--chat` builds text-only turns, so an `--image` would be accepted by the
/// parser and then never reach the model. It is refused up front (before any
/// model is opened) and the run fails, rather than answering as if the
/// picture had been seen.
#[cfg(target_os = "macos")]
#[test]
fn chat_with_image_is_refused_and_fails_before_opening_a_model() {
    let output = run(&[
        "--model",
        "/tmp/does-not-need-to-exist.gturbo",
        "--chat",
        "--image",
        "/tmp/page.png",
    ]);
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--image is not supported with --chat"),
        "unexpected stderr: {stderr}"
    );
}

#[test]
fn no_arguments_fails_on_missing_required_model() {
    let output = run(&[]);
    assert_eq!(output.status.code(), Some(2));
}
