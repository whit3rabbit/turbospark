//! Agent connectors: launch external coding-agent CLIs (Claude Code, Codex,
//! OpenCode, Grok, Gemini, Hermes, OpenClaw, DSH) against the local
//! TurboSpark server, the way opencodex points Codex and Claude Code at its
//! own proxy: ensure the server is serving the requested model, wire the
//! endpoint into each agent's own configuration mechanism (Claude Code reads
//! env plus a `--settings` overlay; Codex takes `-c` config overrides; the
//! rest get the generic `OPENAI_BASE_URL`/`OPENAI_API_KEY` pair), then exec
//! the agent with stdio inherited and propagate its exit status.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use super::daemon;

const KNOWN_AGENTS: &[&str] = &[
    "claude", "codex", "opencode", "grok", "gemini", "hermes", "openclaw", "dsh",
];

/// Placeholder credential handed to agents when the server runs without
/// auth. A loopback bind with neither `--api-key` nor `TURBOSPARK_API_KEY`
/// accepts any key, so the agents (which insist that one be set) stay
/// satisfied without a secret on a command line.
const PLACEHOLDER_KEY: &str = "local";

/// How long to wait for a started daemon to answer `/health`. The server
/// binds only AFTER the model opens, and a large install can take minutes
/// to map and compile; the poll reports progress rather than appearing hung.
const HEALTH_TIMEOUT_SECS: u64 = 600;

/// Check if the command name corresponds to a supported coding or tool agent.
pub fn is_agent(name: &str) -> bool {
    KNOWN_AGENTS.contains(&name)
}

fn which(binary_name: &str) -> Option<std::path::PathBuf> {
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let candidate = dir.join(binary_name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Launch plans (pure, unit-tested)
// ---------------------------------------------------------------------------

/// What would be exec'd: the program, its arguments, and the environment
/// variables to SET on the child (inherited env passes through untouched,
/// so every set below is a default the caller's own export outranks).
struct LaunchPlan {
    program: String,
    args: Vec<String>,
    env: Vec<(String, String)>,
}

fn set_default(
    env: &mut Vec<(String, String)>,
    inherited: &HashMap<String, String>,
    name: &str,
    value: String,
) {
    if inherited.get(name).is_some_and(|v| !v.is_empty()) {
        return;
    }
    env.push((name.to_string(), value));
}

/// The `--settings` overlay: endpoint plus the non-secret wiring. The API
/// key never goes in here -- `--settings` is a process argument, and process
/// arguments are visible through inspection and shell history.
fn claude_settings_json(base_url: &str, model_id: Option<&str>) -> String {
    let mut env_block = serde_json::json!({
        "ANTHROPIC_BASE_URL": base_url,
        "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "true",
        "CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT": "1",
    });
    if let Some(id) = model_id {
        // The advertised alias keeps the "claude" substring Claude Code's
        // gateway discovery filters for, and resolves server-side like the
        // bare id does. Every tier slot routes to the one model this server
        // serves, so background and subagent traffic stays local too.
        let alias = format!("claude-turbospark-{id}");
        for slot in [
            "ANTHROPIC_MODEL",
            "ANTHROPIC_DEFAULT_OPUS_MODEL",
            "ANTHROPIC_DEFAULT_SONNET_MODEL",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL",
            "ANTHROPIC_SMALL_FAST_MODEL",
        ] {
            env_block[slot] = serde_json::Value::String(alias.clone());
        }
    }
    serde_json::json!({ "env": env_block }).to_string()
}

fn build_claude_launch(
    base_url: &str,
    model_id: Option<&str>,
    api_key: &str,
    inherited: &HashMap<String, String>,
) -> LaunchPlan {
    let mut env = Vec::new();
    // Never hand Claude Code both an API key and an auth token: setting the
    // pair triggers its auth-conflict warning, and a non-empty inherited
    // value is the user's own credential, which outranks ours.
    let has_user_auth = ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"]
        .iter()
        .any(|name| inherited.get(*name).is_some_and(|v| !v.trim().is_empty()));
    if !has_user_auth {
        env.push(("ANTHROPIC_API_KEY".to_string(), api_key.to_string()));
    }
    LaunchPlan {
        program: "claude".to_string(),
        args: vec![
            "--settings".to_string(),
            claude_settings_json(base_url, model_id),
        ],
        env,
    }
}

fn build_codex_launch(
    base_url: &str,
    model_id: Option<&str>,
    api_key: &str,
    inherited: &HashMap<String, String>,
) -> LaunchPlan {
    // Codex does not read OPENAI_BASE_URL; its custom-provider mechanism is
    // a `[model_providers.<id>]` config table. The `-c` overrides below are
    // that table spelled on the command line, so `~/.codex/config.toml` is
    // never touched and there is nothing to restore afterwards. Values are
    // parsed as TOML, so string values carry their own quotes. `wire_api`
    // is "chat" because `/v1/chat/completions` is the wire this server
    // exercises end to end.
    let mut args = vec![
        "-c".to_string(),
        "model_provider=\"turbospark\"".to_string(),
        "-c".to_string(),
        "model_providers.turbospark.name=\"TurboSpark\"".to_string(),
        "-c".to_string(),
        format!("model_providers.turbospark.base_url=\"{base_url}/v1\""),
        "-c".to_string(),
        "model_providers.turbospark.env_key=\"TURBOSPARK_API_KEY\"".to_string(),
        "-c".to_string(),
        "model_providers.turbospark.wire_api=\"chat\"".to_string(),
    ];
    if let Some(id) = model_id {
        // Ours first, so an agent-side `-m` in the passthrough args wins.
        args.push("-m".to_string());
        args.push(id.to_string());
    }
    let mut env = Vec::new();
    set_default(
        &mut env,
        inherited,
        "TURBOSPARK_API_KEY",
        api_key.to_string(),
    );
    LaunchPlan {
        program: "codex".to_string(),
        args,
        env,
    }
}

fn build_openai_launch(
    agent: &str,
    base_url: &str,
    api_key: &str,
    inherited: &HashMap<String, String>,
) -> LaunchPlan {
    let mut env = Vec::new();
    set_default(
        &mut env,
        inherited,
        "OPENAI_BASE_URL",
        format!("{base_url}/v1"),
    );
    set_default(&mut env, inherited, "OPENAI_API_KEY", api_key.to_string());
    LaunchPlan {
        program: agent.to_string(),
        args: Vec::new(),
        env,
    }
}

fn build_launch(
    agent: &str,
    base_url: &str,
    model_id: Option<&str>,
    api_key: &str,
    inherited: &HashMap<String, String>,
) -> LaunchPlan {
    match agent {
        "claude" => build_claude_launch(base_url, model_id, api_key, inherited),
        "codex" => build_codex_launch(base_url, model_id, api_key, inherited),
        other => build_openai_launch(other, base_url, api_key, inherited),
    }
}

// ---------------------------------------------------------------------------
// Flag interception
// ---------------------------------------------------------------------------

struct LaunchFlags {
    model: Option<String>,
    port: Option<u16>,
    dry_run: bool,
    agent_args: Vec<String>,
}

/// Split the arguments after the agent name into the launcher's own flags
/// and the agent's passthrough arguments. `--` ends interception: everything
/// after it reaches the agent verbatim, so an agent's own `--model` stays
/// reachable that way.
fn parse_launch_flags(args: &[String]) -> Result<LaunchFlags, String> {
    let mut flags = LaunchFlags {
        model: None,
        port: None,
        dry_run: false,
        agent_args: Vec::new(),
    };
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--" {
            flags.agent_args.extend_from_slice(&args[i + 1..]);
            break;
        }
        if let Some(value) = arg.strip_prefix("--model=") {
            flags.model = Some(value.to_string());
        } else if arg == "--model" {
            i += 1;
            flags.model = Some(args.get(i).cloned().ok_or("--model requires a value")?);
        } else if let Some(value) = arg.strip_prefix("--port=") {
            flags.port = Some(
                value
                    .parse()
                    .map_err(|_| format!("invalid --port value: {value}"))?,
            );
        } else if arg == "--port" {
            i += 1;
            let value = args.get(i).cloned().ok_or("--port requires a value")?;
            flags.port = Some(
                value
                    .parse()
                    .map_err(|_| format!("invalid --port value: {value}"))?,
            );
        } else if arg == "--dry-run" {
            flags.dry_run = true;
        } else {
            flags.agent_args.push(arg.clone());
        }
        i += 1;
    }
    Ok(flags)
}

// ---------------------------------------------------------------------------
// Server liveness (std-only; no HTTP client in this crate)
// ---------------------------------------------------------------------------

/// GET /health over a hand-rolled HTTP/1.1 request. The server routes
/// /health outside its auth layer exactly so an unauthenticated liveness
/// probe can never read as the process being down.
fn http_health_ok(port: u16) -> bool {
    let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
    let request =
        format!("GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = String::new();
    stream.read_to_string(&mut response).is_ok() && response.starts_with("HTTP/1.1 200")
}

/// The advertised model id for an install path or alias: the resolved
/// directory's own name, which is exactly what `GET /v1/models` lists
/// (`crates/server/src/real_model.rs`).
fn model_id_of(model_arg: &str) -> Option<String> {
    catalog::resolve_model_arg(model_arg)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
}

struct ServerTarget {
    port: u16,
    model_id: Option<String>,
}

/// Ensure a daemon is serving the requested model, starting or restarting
/// one when needed, and wait for it to answer /health.
///
/// The daemon's own metadata (`run/server.meta`) is the source of truth for
/// what is already running, so a server this process did not start is still
/// recognized -- and a model alias compares RESOLVED, so `gemma4` and the
/// absolute path to the same install are the same server.
fn ensure_server(
    model_path: Option<&Path>,
    port_override: Option<u16>,
) -> Result<ServerTarget, String> {
    if let Some(running) = daemon::running_server() {
        let same_model = match (model_path, &running.model) {
            (None, _) => true,
            (Some(want), Some(have)) => catalog::resolve_model_arg(have) == want,
            (Some(_), None) => false,
        };
        if same_model {
            if !http_health_ok(running.port) {
                // Alive per its pid file but not answering yet: still
                // opening. The wait loop below watches the same pid.
                wait_for_server(running.port)?;
            }
            return Ok(ServerTarget {
                port: running.port,
                model_id: running.model.as_deref().and_then(model_id_of),
            });
        }
        // Serving something else. A server that silently kept the old model
        // would launch the agent against the wrong weights, so restart on
        // the same port and say why.
        let have = running
            .model
            .clone()
            .unwrap_or_else(|| "(no model)".to_string());
        let want = model_path
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "(none)".to_string());
        println!("restarting turbospark server to serve {want} (was serving {have}) ...");
        daemon::stop().map_err(|e| format!("stopping the previous server: {e}"))?;
    }

    let want = model_path.ok_or_else(|| {
        "no turbospark server is running; pass --model <install-or-alias>, \
         e.g. turbospark start claude --model gemma4"
            .to_string()
    })?;
    let mut server_args = vec!["--model".to_string(), want.display().to_string()];
    if let Some(port) = port_override {
        server_args.push("--port".to_string());
        server_args.push(port.to_string());
    }
    if let Err(e) = daemon::start(&server_args) {
        // The daemon died before its 400ms liveness check only for an
        // immediate abort; a bind conflict surfaces LATER (the server opens
        // the model first and binds after), which wait_for_server reports.
        // But a server we did not start may already hold the port and answer
        // healthily; using it is better than refusing, said out loud.
        let port = port_override.unwrap_or(8080);
        if http_health_ok(port) {
            eprintln!("note: {e}");
            eprintln!(
                "a server this launcher did not start is answering on port {port}; \
                 using it as-is (its served model could not be verified)."
            );
            return Ok(ServerTarget {
                port,
                model_id: None,
            });
        }
        return Err(e);
    }
    let port = daemon::running_server()
        .map(|s| s.port)
        .unwrap_or(port_override.unwrap_or(8080));
    wait_for_server(port)?;
    Ok(ServerTarget {
        port,
        model_id: want.file_name().map(|n| n.to_string_lossy().into_owned()),
    })
}

fn wait_for_server(port: u16) -> Result<(), String> {
    let pid = daemon::get_running_pid();
    let started = Instant::now();
    let mut last_report = Instant::now();
    loop {
        if http_health_ok(port) {
            return Ok(());
        }
        if let Some(pid) = pid {
            if !daemon::is_pid_alive(pid) {
                return Err(format!(
                    "turbospark-server exited during startup (port {port}).\n\
                     Last log output:\n{}",
                    daemon::log_tail()
                ));
            }
        }
        if started.elapsed() > Duration::from_secs(HEALTH_TIMEOUT_SECS) {
            return Err(format!(
                "turbospark-server on port {port} did not answer /health within \
                 {HEALTH_TIMEOUT_SECS}s; check {}",
                daemon::log_file().display()
            ));
        }
        if last_report.elapsed() >= Duration::from_secs(5) {
            eprintln!(
                "waiting for turbospark-server on port {port} to finish opening ({}s) ...",
                started.elapsed().as_secs()
            );
            last_report = Instant::now();
        }
        thread::sleep(Duration::from_millis(500));
    }
}

// ---------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------

fn shell_single_quoted(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

/// Render one argument for a copy-pastable printed command line: bare when
/// it is made of shell-safe characters, single-quoted otherwise.
fn shell_arg(value: &str) -> String {
    let safe = !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_./=:@+".contains(c));
    if safe {
        value.to_string()
    } else {
        shell_single_quoted(value)
    }
}

fn install_hint(agent: &str) -> &str {
    match agent {
        "claude" => "npm install -g @anthropic-ai/claude-code",
        "codex" => "npm install -g @openai/codex",
        _ => "install the agent CLI and make sure it is on PATH",
    }
}

/// Print what would be launched without touching the daemon or spawning the
/// agent. The port is the one that WOULD be used (an explicit `--port`, else
/// the running daemon's, else the default), not a probe result.
fn print_plan(agent: &str, flags: &LaunchFlags, model_path: Option<&Path>) -> Result<(), String> {
    let port = flags
        .port
        .or_else(|| daemon::running_server().map(|s| s.port))
        .unwrap_or(8080);
    let base_url = format!("http://127.0.0.1:{port}");
    let model_id = model_path
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .or_else(|| {
            daemon::running_server()
                .and_then(|s| s.model)
                .and_then(|m| model_id_of(&m))
        });
    let api_key = std::env::var("TURBOSPARK_API_KEY")
        .ok()
        .filter(|k| !k.is_empty())
        .unwrap_or_else(|| PLACEHOLDER_KEY.to_string());
    let inherited: HashMap<String, String> = std::env::vars().collect();
    let plan = build_launch(agent, &base_url, model_id.as_deref(), &api_key, &inherited);

    println!("Dry run: no server started, no agent launched.");
    println!("  Endpoint: {base_url}");
    match &model_id {
        Some(id) => println!("  Model:    {id}"),
        None => println!("  Model:    (the running server's own)"),
    }
    match agent {
        "claude" => {
            let rendered = plan
                .args
                .iter()
                .map(|a| shell_arg(a))
                .collect::<Vec<_>>()
                .join(" ");
            println!("  Command:  {} {rendered} [agent args...]", plan.program);
            let key_line = if plan.env.iter().any(|(k, _)| k == "ANTHROPIC_API_KEY") {
                format!("ANTHROPIC_API_KEY={api_key} (env; skipped when your own credential is exported)")
            } else {
                "using your exported ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN".to_string()
            };
            println!("  Env:      {key_line}");
        }
        "codex" => {
            println!(
                "  Command:  {} {} [agent args...]",
                plan.program,
                plan.args.join(" ")
            );
            println!("  Env:      TURBOSPARK_API_KEY={api_key}");
        }
        other => {
            println!("  Command:  {other} [agent args...]");
            println!("  Env:      OPENAI_BASE_URL={base_url}/v1 OPENAI_API_KEY={api_key}");
        }
    }
    if which(&plan.program).is_none() {
        println!();
        println!("'{}' was not found in PATH.", plan.program);
        println!("Install it first:");
        println!("    {}", install_hint(agent));
    }
    Ok(())
}

/// Connect and launch an agent against the local TurboSpark server.
///
/// `extra_args` are the arguments after the agent name: the launcher's own
/// `--model`/`--port`/`--dry-run` are intercepted, `--` ends interception,
/// and everything else passes through to the agent.
pub fn start_agent(agent: &str, extra_args: &[String]) -> Result<(), String> {
    if !is_agent(agent) {
        return Err(format!(
            "unknown agent '{agent}'. Supported agents: {}",
            KNOWN_AGENTS.join(", ")
        ));
    }
    let flags = parse_launch_flags(extra_args)?;
    let model_path = flags.model.as_deref().map(catalog::resolve_model_arg);
    if let (Some(alias), Some(path)) = (flags.model.as_deref(), model_path.as_deref()) {
        if !path.is_dir() {
            return Err(format!(
                "no such install: {alias} (resolved to {})",
                path.display()
            ));
        }
    }
    if flags.dry_run {
        return print_plan(agent, &flags, model_path.as_deref());
    }

    let target = ensure_server(model_path.as_deref(), flags.port)?;
    let base_url = format!("http://127.0.0.1:{}", target.port);
    let api_key = std::env::var("TURBOSPARK_API_KEY")
        .ok()
        .filter(|k| !k.is_empty())
        .unwrap_or_else(|| PLACEHOLDER_KEY.to_string());
    let inherited: HashMap<String, String> = std::env::vars().collect();
    let plan = build_launch(
        agent,
        &base_url,
        target.model_id.as_deref(),
        &api_key,
        &inherited,
    );

    println!("Connecting {agent} to TurboSpark at {base_url} ...");
    if let Some(id) = &target.model_id {
        println!("  model: {id}");
    }
    if which(&plan.program).is_none() {
        println!();
        println!("'{}' command was not found in PATH.", plan.program);
        println!("Install it first:");
        println!("    {}", install_hint(agent));
        return Ok(());
    }

    let mut cmd = Command::new(&plan.program);
    cmd.args(&plan.args);
    for (name, value) in &plan.env {
        cmd.env(name, value);
    }
    cmd.args(&flags.agent_args);
    let status = cmd
        .status()
        .map_err(|e| format!("failed to launch {}: {e}", plan.program))?;
    if !status.success() {
        return Err(format!("{agent} exited with status {status}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inherited(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn claude_plan_sets_endpoint_discovery_and_model_slots() {
        let plan = build_launch(
            "claude",
            "http://127.0.0.1:8080",
            Some("gemma4.gturbo"),
            "local",
            &inherited(&[]),
        );
        assert_eq!(plan.program, "claude");
        assert_eq!(plan.args[0], "--settings");
        let settings = &plan.args[1];
        assert!(
            settings.contains("\"ANTHROPIC_BASE_URL\":\"http://127.0.0.1:8080\""),
            "{settings}"
        );
        assert!(settings.contains("\"CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY\":\"true\""));
        assert!(settings.contains("\"CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT\":\"1\""));
        // Every tier slot routes to the advertised alias.
        assert!(settings.contains("\"ANTHROPIC_MODEL\":\"claude-turbospark-gemma4.gturbo\""));
        assert!(settings
            .contains("\"ANTHROPIC_DEFAULT_OPUS_MODEL\":\"claude-turbospark-gemma4.gturbo\""));
        assert!(
            settings.contains("\"ANTHROPIC_SMALL_FAST_MODEL\":\"claude-turbospark-gemma4.gturbo\"")
        );
        // The credential travels in the child env, never in a process argument.
        assert!(!settings.contains("ANTHROPIC_API_KEY"));
        assert!(plan
            .env
            .contains(&("ANTHROPIC_API_KEY".to_string(), "local".to_string())));
    }

    #[test]
    fn claude_plan_without_a_model_sets_no_slots() {
        let plan = build_launch(
            "claude",
            "http://127.0.0.1:9000",
            None,
            "local",
            &inherited(&[]),
        );
        let settings = &plan.args[1];
        assert!(settings.contains("\"ANTHROPIC_BASE_URL\":\"http://127.0.0.1:9000\""));
        assert!(!settings.contains("ANTHROPIC_MODEL"));
    }

    #[test]
    fn claude_plan_defers_to_an_exported_credential() {
        for name in ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"] {
            let plan = build_launch(
                "claude",
                "http://127.0.0.1:8080",
                None,
                "local",
                &inherited(&[(name, "user-own-key")]),
            );
            // Both spellings trigger Claude Code's auth-conflict warning, so
            // ours stays out entirely rather than pairing with theirs.
            assert!(plan.env.is_empty(), "{name}: {:?}", plan.env);
        }
    }

    #[test]
    fn codex_plan_overrides_config_on_the_command_line() {
        let plan = build_launch(
            "codex",
            "http://127.0.0.1:8080",
            Some("gemma4.gturbo"),
            "local",
            &inherited(&[]),
        );
        assert_eq!(plan.program, "codex");
        let joined = plan.args.join(" ");
        assert!(joined.contains("model_provider=\"turbospark\""), "{joined}");
        assert!(joined.contains("model_providers.turbospark.base_url=\"http://127.0.0.1:8080/v1\""));
        assert!(joined.contains("model_providers.turbospark.env_key=\"TURBOSPARK_API_KEY\""));
        assert!(joined.contains("model_providers.turbospark.wire_api=\"chat\""));
        // -m precedes the passthrough so an agent-side -m can override it.
        assert_eq!(plan.args[plan.args.len() - 2], "-m");
        assert_eq!(plan.args[plan.args.len() - 1], "gemma4.gturbo");
        assert!(plan
            .env
            .contains(&("TURBOSPARK_API_KEY".to_string(), "local".to_string())));
    }

    #[test]
    fn codex_plan_keeps_an_exported_key() {
        let plan = build_launch(
            "codex",
            "http://127.0.0.1:8080",
            None,
            "local",
            &inherited(&[("TURBOSPARK_API_KEY", "real-secret")]),
        );
        assert!(plan.env.is_empty());
    }

    #[test]
    fn generic_agents_get_the_openai_pair_only() {
        for agent in ["opencode", "grok", "gemini", "hermes", "openclaw", "dsh"] {
            let plan = build_launch(
                agent,
                "http://127.0.0.1:8080",
                Some("x.gturbo"),
                "local",
                &inherited(&[]),
            );
            assert_eq!(plan.program, agent);
            assert!(plan.args.is_empty());
            assert!(plan.env.contains(&(
                "OPENAI_BASE_URL".to_string(),
                "http://127.0.0.1:8080/v1".to_string()
            )));
            assert!(plan
                .env
                .contains(&("OPENAI_API_KEY".to_string(), "local".to_string())));
        }
    }

    #[test]
    fn launch_flags_are_intercepted_and_dash_dash_ends_interception() {
        let args: Vec<String> = [
            "--model",
            "gemma4",
            "--port",
            "9000",
            "extra",
            "--",
            "--model",
            "agents-own",
            "--dry-run",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let flags = parse_launch_flags(&args).unwrap();
        assert_eq!(flags.model.as_deref(), Some("gemma4"));
        assert_eq!(flags.port, Some(9000));
        assert!(!flags.dry_run);
        assert_eq!(
            flags.agent_args,
            ["extra", "--model", "agents-own", "--dry-run"].map(String::from)
        );
    }

    #[test]
    fn launch_flags_accept_equals_forms_and_dry_run() {
        let args: Vec<String> = ["--model=x.gturbo", "--port=1234", "--dry-run"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let flags = parse_launch_flags(&args).unwrap();
        assert_eq!(flags.model.as_deref(), Some("x.gturbo"));
        assert_eq!(flags.port, Some(1234));
        assert!(flags.dry_run);
        assert!(flags.agent_args.is_empty());
    }

    #[test]
    fn missing_flag_values_and_bad_ports_are_refused() {
        let missing: Vec<String> = ["--model"].iter().map(|s| s.to_string()).collect();
        assert!(parse_launch_flags(&missing).is_err());
        let bad: Vec<String> = ["--port", "not-a-port"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(parse_launch_flags(&bad).is_err());
    }

    #[test]
    fn health_probe_reads_a_200_and_refuses_a_refused_port() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            // Each connection is dropped before the next accept so the
            // probe's read-to-EOF sees the close.
            {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buf = [0u8; 1024];
                let _ = std::io::Read::read(&mut stream, &mut buf);
                std::io::Write::write_all(
                    &mut stream,
                    b"HTTP/1.1 200 OK\r\ncontent-type: text/plain\r\n\r\nok",
                )
                .unwrap();
            }
            // A second connection that answers NON-200: a status-line-blind
            // probe would accept it.
            {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buf = [0u8; 1024];
                let _ = std::io::Read::read(&mut stream, &mut buf);
                std::io::Write::write_all(&mut stream, b"HTTP/1.1 404 Not Found\r\n\r\nno")
                    .unwrap();
            }
        });
        assert!(http_health_ok(port));
        assert!(!http_health_ok(port), "a 404 must not read as healthy");
        server.join().unwrap();
        // Nothing listens here.
        assert!(!http_health_ok(1));
    }

    #[test]
    fn known_agents_cover_the_documented_set() {
        for name in [
            "claude", "codex", "opencode", "grok", "gemini", "hermes", "openclaw", "dsh",
        ] {
            assert!(is_agent(name), "{name}");
        }
        assert!(!is_agent("serve"));
        assert!(!is_agent("--model"));
    }
}
