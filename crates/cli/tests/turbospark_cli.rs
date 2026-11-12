use std::process::Command;

fn turbospark() -> Command {
    Command::new(env!("CARGO_BIN_EXE_turbospark"))
}

#[test]
fn turbospark_help_prints_usage() {
    let output = turbospark().arg("--help").output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    for command in [
        "run MODEL [PROMPT]",
        "pull NAME...",
        "ls",
        "rm NAME...",
        "serve",
        "start | stop | restart",
        "status | ps",
        "launch AGENT",
        "doctor",
    ] {
        assert!(
            stdout.contains(command),
            "help lacks {command:?}:\n{stdout}"
        );
    }
    // Piped, so plain: color is for terminals only.
    assert!(!stdout.contains('\u{1b}'), "escape codes in piped help");
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
    assert!(stdout.contains("turbospark run <MODEL>"), "{stdout}");
    assert!(
        stdout.contains("--raw"),
        "run --help must explain --raw: {stdout}"
    );
}

#[test]
fn turbospark_status_when_stopped() {
    let home = isolated_home();
    let output = turbospark()
        .env("TURBOSPARK_HOME", &home)
        .arg("status")
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("turbospark server") && stdout.contains("stopped"),
        "{stdout}"
    );

    let json = turbospark()
        .env("TURBOSPARK_HOME", &home)
        .args(["status", "--json"])
        .output()
        .unwrap();
    let _ = std::fs::remove_dir_all(&home);
    let value: serde_json::Value = serde_json::from_slice(&json.stdout).unwrap();
    assert_eq!(value["running"], false);
}

#[test]
fn turbospark_unknown_command_exits_two() {
    let output = turbospark().arg("nonexistent_subcmd").output().unwrap();
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("unknown command: nonexistent_subcmd"));

    let typo = turbospark().arg("pul").output().unwrap();
    assert_eq!(typo.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&typo.stderr).contains("turbospark pull"));
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

/// Dry-run stdout lands in transcripts and CI logs, so a real key must not
/// appear there on any agent's branch. One agent per `print_plan` arm.
#[test]
fn turbospark_start_agent_dry_run_masks_an_exported_key() {
    const SECRET: &str = "sentinel-secret-value";
    let home = isolated_home();
    let plan = |agent: &str, key: Option<&str>| {
        // A user's own Anthropic credential would divert the claude arm away
        // from the line under test.
        let mut cmd = turbospark();
        cmd.env("TURBOSPARK_HOME", &home)
            .env_remove("ANTHROPIC_API_KEY")
            .env_remove("ANTHROPIC_AUTH_TOKEN")
            .env_remove("TURBOSPARK_API_KEY")
            .args(["start", agent, "--dry-run"]);
        if let Some(key) = key {
            cmd.env("TURBOSPARK_API_KEY", key);
        }
        cmd.output().unwrap()
    };
    let cases = [
        ("claude", "ANTHROPIC_API_KEY"),
        ("codex", "TURBOSPARK_API_KEY"),
        ("opencode", "OPENAI_API_KEY"),
    ];
    for (agent, var) in cases {
        let masked = plan(agent, Some(SECRET));
        assert!(masked.status.success(), "{agent}");
        let stdout = String::from_utf8_lossy(&masked.stdout);
        assert!(
            !stdout.contains(SECRET),
            "{agent} leaked the key:\n{stdout}"
        );
        assert!(
            stdout.contains(&format!("{var}=<set from env>")),
            "{agent}:\n{stdout}"
        );
        assert!(
            !String::from_utf8_lossy(&masked.stderr).contains(SECRET),
            "{agent} leaked the key on stderr"
        );

        // The placeholder is not a secret and tells the user what the agent
        // will really receive, so it stays visible.
        let placeholder = plan(agent, None);
        assert!(placeholder.status.success(), "{agent}");
        let stdout = String::from_utf8_lossy(&placeholder.stdout);
        assert!(
            stdout.contains(&format!("{var}=local")),
            "{agent}:\n{stdout}"
        );
    }
    let _ = std::fs::remove_dir_all(&home);
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

/// A listener this launcher did not start can answer /health on the port
/// (another user's process on a shared machine). Handing the agent's
/// prompts and code context to it is refused, not done "as-is".
#[test]
fn turbospark_start_agent_refuses_an_unverified_listener_on_the_port() {
    use std::io::{Read, Write};
    let home = isolated_home();
    let model = home.join("fake-install");
    std::fs::create_dir_all(&model).unwrap();
    // An empty PATH keeps a real `dsh` (or a PATH turbospark-server) out of
    // the run; the squatter is the only thing that can make this succeed.
    let empty_path = home.join("empty-path");
    std::fs::create_dir_all(&empty_path).unwrap();

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let served = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let served_in = served.clone();
    std::thread::spawn(move || {
        for mut stream in listener.incoming().flatten() {
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf);
            served_in.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let _ = stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok");
        }
    });

    let output = turbospark()
        .env("TURBOSPARK_HOME", &home)
        .env("PATH", &empty_path)
        .env_remove("TURBOSPARK_API_KEY")
        .args([
            "start",
            "dsh",
            "--model",
            model.to_str().unwrap(),
            "--port",
            &port.to_string(),
        ])
        .output()
        .unwrap();
    let _ = std::fs::remove_dir_all(&home);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // The squatter really was probed, so a pass is not a missed listener.
    assert!(
        served.load(std::sync::atomic::Ordering::SeqCst) > 0,
        "{stderr}"
    );
    assert_eq!(output.status.code(), Some(1), "{stdout}\n{stderr}");
    assert!(
        !stdout.contains("Connecting dsh"),
        "launched against the squatter:\n{stdout}"
    );
    assert!(stderr.contains(&format!("port {port}")), "{stderr}");
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

/// `turbospark` with an isolated store and the given args, plain output.
fn isolated(home: &std::path::Path, args: &[&str]) -> std::process::Output {
    turbospark()
        .env("TURBOSPARK_HOME", home)
        .env("TURBOSPARK_TEST_CHIP", "Apple M4 Max")
        .env_remove("CLICOLOR_FORCE")
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn ls_on_an_empty_store_says_so_and_json_is_an_empty_array() {
    let home = isolated_home();
    let text = isolated(&home, &["ls"]);
    let json = isolated(&home, &["ls", "--json"]);
    let _ = std::fs::remove_dir_all(&home);
    assert!(text.status.success());
    assert!(String::from_utf8_lossy(&text.stdout).contains("No models installed yet"));
    let value: serde_json::Value = serde_json::from_slice(&json.stdout).unwrap();
    assert_eq!(value, serde_json::json!([]));
}

#[test]
fn search_and_show_take_tags_and_speak_user_words() {
    let home = isolated_home();
    let search = isolated(&home, &["search", "qwen3.6"]);
    let show = isolated(&home, &["show", "qwen3.6:35b", "--json"]);
    let miss = isolated(&home, &["show", "qwen3.6:99b"]);
    let _ = std::fs::remove_dir_all(&home);

    let stdout = String::from_utf8_lossy(&search.stdout);
    assert!(
        stdout.contains("qwen36") && stdout.contains("tested"),
        "{stdout}"
    );
    assert!(
        !stdout.contains("frozen gate"),
        "maintainer wording leaked: {stdout}"
    );

    let value: serde_json::Value = serde_json::from_slice(&show.stdout).unwrap();
    assert_eq!(value["alias"], "qwen36");
    assert!(value["tags"]
        .as_array()
        .unwrap()
        .contains(&"qwen3.6:35b".into()));
    assert_eq!(value["installed"], serde_json::Value::Null);

    assert_eq!(miss.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&miss.stderr).contains("qwen36"));
}

#[test]
fn pull_refuses_an_unknown_name_before_any_download() {
    let home = isolated_home();
    let output = isolated(&home, &["pull", "qwen36", "definitely-not-a-model"]);
    let installed = home.join("models").exists();
    let _ = std::fs::remove_dir_all(&home);
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("definitely-not-a-model"));
    assert!(
        !installed,
        "a valid first name must not download before the typo is caught"
    );
}

#[test]
fn run_names_close_models_for_an_unknown_one() {
    let home = isolated_home();
    let output = isolated(&home, &["run", "qwen3.66", "hi"]);
    let _ = std::fs::remove_dir_all(&home);
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("no model named \"qwen3.66\""), "{stderr}");
}

#[test]
fn completions_render_for_zsh() {
    let output = turbospark().args(["completions", "zsh"]).output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("#compdef turbospark"));
    assert!(stdout.contains("doctor"));
}

#[test]
fn run_hides_the_resolved_request_and_direct_check_still_prints_it() {
    // A path model is passed through untouched, so this reaches
    // turbospark-check without an install; generation then fails (exit 1)
    // after the point where the printout would have been.
    let missing = std::env::temp_dir().join("turbospark-front-door-no-such.gturbo");
    let missing = missing.to_str().unwrap();

    let front = turbospark()
        .args(["run", missing, "hi", "--max-new", "1"])
        .output()
        .unwrap();
    let front_out = String::from_utf8_lossy(&front.stdout);
    assert!(
        !front_out.contains("resolved invocation:"),
        "turbospark run printed the request dump:\n{front_out}"
    );

    let direct = Command::new(env!("CARGO_BIN_EXE_turbospark-check"))
        .args(["--model", missing, "--prompt", "hi", "--max-new", "1"])
        .env_remove("TURBOSPARK_HIDE_RESOLVED_REQUEST")
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&direct.stdout).contains("resolved invocation:"));
}
