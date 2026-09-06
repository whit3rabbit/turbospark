use std::process::Command;

fn turbospark() -> Command {
    Command::new(env!("CARGO_BIN_EXE_turbospark"))
}

#[test]
fn turbospark_help_prints_usage() {
    let output = turbospark().arg("--help").output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("turbospark -- unified"));
    assert!(stdout.contains("run <MODEL>"));
    assert!(stdout.contains("serve [OPTIONS]"));
    assert!(stdout.contains("start [OPTIONS]"));
    assert!(stdout.contains("stop"));
    assert!(stdout.contains("status"));
    assert!(stdout.contains("list"));
    assert!(stdout.contains("pull"));
}

#[test]
fn turbospark_version_prints_version() {
    let output = turbospark().arg("--version").output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.starts_with("turbospark "));
}

#[test]
fn turbospark_run_help() {
    let output = turbospark().args(["run", "--help"]).output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("turbospark run <MODEL>"));
}

#[test]
fn turbospark_status_when_stopped() {
    let output = turbospark().arg("status").output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Status:"));
}

#[test]
fn turbospark_unknown_command_exits_two() {
    let output = turbospark().arg("nonexistent_subcmd").output().unwrap();
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("unknown command: nonexistent_subcmd"));
}

#[test]
fn turbospark_start_agent_dry_run_and_instructions() {
    let output = turbospark()
        .args(["start", "claude", "--dry-run"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("ANTHROPIC_BASE_URL"));

    let output_dsh = turbospark().args(["start", "dsh"]).output().unwrap();
    assert!(output_dsh.status.success());
    let stdout_dsh = String::from_utf8_lossy(&output_dsh.stdout);
    assert!(stdout_dsh.contains("OPENAI_BASE_URL"));
}
