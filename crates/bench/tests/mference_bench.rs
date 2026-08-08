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
