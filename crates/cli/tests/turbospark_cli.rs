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

/// A hermetic store root: no daemon can be running there, and no model
/// resolves, whatever the invoking machine has under its real HOME.
fn isolated_home() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-agent-cli-{}-{}",
        std::process::id(),
        std::thread::current()
            .name()
            .unwrap_or("t")
            .replace("::", "-")
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn fake_install(home: &std::path::Path, name: &str) -> std::path::PathBuf {
    let dir = home.join(name);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("manifest.json"), "{}").unwrap();
    dir
}

#[test]
fn turbospark_start_agent_dry_run_and_instructions() {
    let home = isolated_home();
    let output = turbospark()
        .env("TURBOSPARK_HOME", &home)
        .args(["start", "claude", "--dry-run"])
        .output()
        .unwrap();
    let _ = std::fs::remove_dir_all(&home);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("ANTHROPIC_BASE_URL"));
    assert!(stdout.contains("\"ANTHROPIC_BASE_URL\":\"http://127.0.0.1:8080\""));
    assert!(!stdout.contains("\"ANTHROPIC_BASE_URL\":\"http://127.0.0.1:8080/v1\""));
    assert!(stdout.contains("\"CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY\":\"true\""));
    assert!(stdout.contains("\"CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT\":\"1\""));
    assert!(stdout.contains("claude --settings"));
    // No model was named and no server is running, so no slot is set.
    assert!(!stdout.contains("ANTHROPIC_MODEL"));
}

#[test]
fn turbospark_start_agent_dry_run_with_model_prints_concrete_plan() {
    let home = isolated_home();
    let install = fake_install(&home, "gemma4.gturbo");
    let output = turbospark()
        .env("TURBOSPARK_HOME", &home)
        .args([
            "start",
            "claude",
            "--model",
            install.to_str().unwrap(),
            "--dry-run",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Every tier slot carries the advertised alias of the named install.
    assert!(stdout.contains("\"ANTHROPIC_MODEL\":\"claude-turbospark-gemma4.gturbo\""));
    assert!(stdout.contains("\"ANTHROPIC_SMALL_FAST_MODEL\":\"claude-turbospark-gemma4.gturbo\""));

    let codex = turbospark()
        .env("TURBOSPARK_HOME", &home)
        .args([
            "start",
            "codex",
            "--model",
            install.to_str().unwrap(),
            "--dry-run",
        ])
        .output()
        .unwrap();
    let _ = std::fs::remove_dir_all(&home);
    assert!(codex.status.success());
    let stdout_codex = String::from_utf8_lossy(&codex.stdout);
    assert!(
        stdout_codex.contains(r#"model_provider="turbospark""#),
        "{stdout_codex}"
    );
    assert!(stdout_codex.contains(r#"wire_api="chat""#));
    assert!(stdout_codex.contains("-m x.gturbo") || stdout_codex.contains("-m gemma4.gturbo"));
}

#[test]
fn turbospark_model_then_agent_launches_the_same_way() {
    let home = isolated_home();
    let install = fake_install(&home, "x.gturbo");
    let model = install.to_str().unwrap().to_string();
    let direct = turbospark()
        .env("TURBOSPARK_HOME", &home)
        .args(["start", "codex", "--model", &model, "--dry-run"])
        .output()
        .unwrap();
    let sugar = turbospark()
        .env("TURBOSPARK_HOME", &home)
        .args([&model, "codex", "--dry-run"])
        .output()
        .unwrap();
    let _ = std::fs::remove_dir_all(&home);
    assert!(direct.status.success());
    assert!(sugar.status.success());
    let stdout = String::from_utf8_lossy(&sugar.stdout);
    assert!(stdout.contains("-m x.gturbo"), "{stdout}");
    // The two spellings produce the same plan.
    assert_eq!(
        String::from_utf8_lossy(&direct.stdout).replace(&model, "M"),
        stdout.replace(&model, "M")
    );
}

#[test]
fn turbospark_start_agent_without_model_or_server_refuses() {
    let home = isolated_home();
    let output = turbospark()
        .env("TURBOSPARK_HOME", &home)
        .args(["start", "dsh"])
        .output()
        .unwrap();
    let _ = std::fs::remove_dir_all(&home);
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("no turbospark server is running"),
        "{stderr}"
    );
    assert!(stderr.contains("--model"));
}

#[test]
fn turbospark_start_agent_with_unknown_model_refuses() {
    let home = isolated_home();
    let output = turbospark()
        .env("TURBOSPARK_HOME", &home)
        .args(["start", "claude", "--model", "no-such-install", "--dry-run"])
        .output()
        .unwrap();
    let _ = std::fs::remove_dir_all(&home);
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("no such install: no-such-install"),
        "{stderr}"
    );
}
