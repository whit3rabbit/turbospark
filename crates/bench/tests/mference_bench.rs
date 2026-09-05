//! Black-box test of the `turbospark-bench` binary: runs against the real
//! vendored ChatML tokenizer fixture and checks the report shape.

use std::process::Command;

fn fixture_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tokenizer/tests/fixtures/ChatMLTokenizer")
}

#[test]
fn reports_three_prompts_and_an_aggregate_line() {
    let output = Command::new(env!("CARGO_BIN_EXE_turbospark-bench"))
        .arg(fixture_dir())
        .output()
        .expect("binary should run");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("3 fixed prompts"));
    assert!(stdout.contains("aggregate:"));
    let data_lines = stdout
        .lines()
        .filter(|l| l.starts_with(char::is_numeric))
        .count();
    assert_eq!(data_lines, 3);
}

#[test]
fn missing_argument_exits_with_usage_error() {
    let output = Command::new(env!("CARGO_BIN_EXE_turbospark-bench"))
        .output()
        .expect("binary should run");
    assert_eq!(output.status.code(), Some(2));
}

// --- `--prefill-chunk`, the seam that makes chunked prefill measurable ---
//
// Every case below runs with NO INSTALL and on any platform: the parse loop
// and both `main.rs` guards execute before `--model`'s directory is ever
// touched, which is why those two guards live in `main.rs` rather than in
// `model_mode.rs`. The family refusal cannot be reached this way (it needs an
// opened runner to know the family), and is exercised by hand against a real
// install of a family with no chunked driver.

/// Helper: run the binary against a deliberately absent install directory.
fn run_bench(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_turbospark-bench"))
        .args(["--model", "/nonexistent-install-for-arg-parsing"])
        .args(args)
        .env_remove("TURBOSPARK_PREFILL_CHUNK")
        .output()
        .expect("binary should run")
}

/// A chunk size outside `ALLOWED_CHUNK_SIZES` is refused with the set named,
/// matching how `--expert-cache-slots` refuses a slot count outside its own.
#[test]
fn prefill_chunk_refuses_a_size_outside_the_allowed_set() {
    let output = run_bench(&["--prefill-chunk", "7"]);
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--prefill-chunk needs off, auto or one of"),
        "the refusal must name the allowed set; got {stderr}"
    );
}

/// A trailing `--prefill-chunk` with no value is refused rather than silently
/// defaulting, which for THIS flag would mean turning on the very path every
/// frozen row in this crate was measured without.
#[test]
fn prefill_chunk_needs_a_value() {
    let output = run_bench(&["--prefill-chunk"]);
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--prefill-chunk needs"), "got {stderr}");
}

/// Speculation and chunked prefill are not composable, and the refusal must
/// land BEFORE the install is opened.
///
/// The negative assertion is the load-bearing half: it pins the guard's
/// POSITION, not just its existence. Moving the check into `model_mode.rs`
/// (after the open) leaves the positive assertion green while making the
/// guard unreachable on any machine without that install.
#[test]
fn speculation_and_chunked_prefill_are_refused_before_the_open() {
    let output = run_bench(&[
        "--shaping",
        "greedy",
        "--speculative",
        "auto",
        "--prefill-chunk",
        "auto",
    ]);
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not composable"),
        "the refusal must say why; got {stderr}"
    );
    assert!(
        !stderr.contains("failed to open"),
        "the guard must fire before the install is opened; got {stderr}"
    );
}

/// `TURBOSPARK_PREFILL_CHUNK` is REFUSED here, neither inherited nor ignored,
/// and this binary is the one place that deliberately diverges from
/// `crates/cli` (where the env var correctly wins, because the CLI serves a
/// user rather than labelling a measurement).
///
/// Inheriting it would hand this harness an arm nobody typed (AGENTS.md
/// Gotcha 35). Ignoring it is the silent-ignore class. Only refusing fails
/// loudly, and NOTHING ELSE IN THIS FILE CAN SEE THE DIFFERENCE -- flipping
/// the guard to either alternative reddens this case alone.
#[test]
fn the_prefill_chunk_env_seam_is_refused_rather_than_inherited_or_ignored() {
    let output = Command::new(env!("CARGO_BIN_EXE_turbospark-bench"))
        .args(["--model", "/nonexistent-install-for-arg-parsing"])
        .env("TURBOSPARK_PREFILL_CHUNK", "128")
        .output()
        .expect("binary should run");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("TURBOSPARK_PREFILL_CHUNK") && stderr.contains("--prefill-chunk"),
        "the refusal must name both the seam and the flag that replaces it; got {stderr}"
    );
}

/// The usage line names the flag, so `--help`-by-way-of-error stays complete.
#[test]
fn the_usage_line_names_prefill_chunk() {
    let output = Command::new(env!("CARGO_BIN_EXE_turbospark-bench"))
        .args(["--model"])
        .output()
        .expect("binary should run");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--prefill-chunk off|auto|N"),
        "the usage line must name the flag and its values; got {stderr}"
    );
}
