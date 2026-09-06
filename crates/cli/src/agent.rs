//! Agent connectors inspired by Unsloth Start: connects Claude Code, Codex,
//! OpenCode, Hermes, OpenClaw, and DSH to a local TurboSpark server.

use std::path::PathBuf;
use std::process::{Command, ExitStatus};

const KNOWN_AGENTS: &[&str] = &["claude", "codex", "opencode", "hermes", "openclaw", "dsh"];

/// Check if the command name corresponds to a supported coding or tool agent.
pub fn is_agent(name: &str) -> bool {
    KNOWN_AGENTS.contains(&name)
}

fn read_server_port() -> u16 {
    let meta_path = catalog::default_root()
        .unwrap_or_else(|| PathBuf::from(".turbospark"))
        .join("run")
        .join("server.meta");

    if let Ok(text) = std::fs::read_to_string(meta_path) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
            if let Some(p) = v["port"].as_u64() {
                return p as u16;
            }
        }
    }
    8080
}

fn which(binary_name: &str) -> Option<PathBuf> {
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

/// Connect and launch an agent against the local TurboSpark server endpoint.
pub fn start_agent(agent: &str, extra_args: &[String]) -> Result<(), String> {
    let port = read_server_port();
    let base_url = format!("http://127.0.0.1:{port}/v1");

    match agent {
        "claude" => start_claude(&base_url, port, extra_args),
        "codex" | "opencode" | "hermes" | "openclaw" | "dsh" => {
            start_openai_agent(agent, &base_url, port, extra_args)
        }
        other => Err(format!(
            "unknown agent '{other}'. Supported agents: {}",
            KNOWN_AGENTS.join(", ")
        )),
    }
}

fn start_claude(base_url: &str, port: u16, extra_args: &[String]) -> Result<(), String> {
    let dry_run = extra_args
        .iter()
        .any(|a| a == "--dry-run" || a == "--print");
    if dry_run || which("claude").is_none() {
        if !dry_run {
            println!("\n'claude' command was not found in PATH.");
            println!("Install Claude Code:");
            println!("    npm install -g @anthropic-ai/claude-code\n");
        }
        println!("To connect Claude Code to TurboSpark:");
        println!("    export ANTHROPIC_BASE_URL=\"http://127.0.0.1:{port}/v1\"");
        println!("    export ANTHROPIC_API_KEY=\"local\"");
        println!("    claude");
        return Ok(());
    }

    println!("Connecting Claude Code to TurboSpark at {base_url} ...");
    let status: ExitStatus = Command::new("claude")
        .env("ANTHROPIC_BASE_URL", base_url)
        .env("ANTHROPIC_API_KEY", "local")
        .args(extra_args)
        .status()
        .map_err(|e| format!("failed to launch claude: {e}"))?;

    if !status.success() {
        return Err(format!("claude exited with status {status}"));
    }
    Ok(())
}

fn start_openai_agent(
    agent: &str,
    base_url: &str,
    port: u16,
    extra_args: &[String],
) -> Result<(), String> {
    let dry_run = extra_args
        .iter()
        .any(|a| a == "--dry-run" || a == "--print");
    if dry_run || which(agent).is_none() {
        if !dry_run {
            println!("\n'{agent}' command was not found in PATH.");
        }
        println!("To connect {agent} to TurboSpark:");
        println!("    export OPENAI_BASE_URL=\"http://127.0.0.1:{port}/v1\"");
        println!("    export OPENAI_API_KEY=\"local\"");
        println!("    {agent}");
        return Ok(());
    }

    println!("Connecting {agent} to TurboSpark at {base_url} ...");
    let status: ExitStatus = Command::new(agent)
        .env("OPENAI_BASE_URL", base_url)
        .env("OPENAI_API_KEY", "local")
        .args(extra_args)
        .status()
        .map_err(|e| format!("failed to launch {agent}: {e}"))?;

    if !status.success() {
        return Err(format!("{agent} exited with status {status}"));
    }
    Ok(())
}
