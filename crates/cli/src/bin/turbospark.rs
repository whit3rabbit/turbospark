//! `turbospark`: Unified command-line interface for TurboSpark.
//!
//! Provides an intuitive and cohesive CLI compatible with oMLX and Unsloth workflows:
//! - Interactive chat and quick prompt runner (`run`)
//! - Foreground and background managed server (`serve`, `start`, `stop`, `restart`, `status`)
//! - Coding agent connectors (`start claude`, `start codex`, etc.)
//! - Model catalog and management (`list`, `pull`, `info`, `rm`, `probe`, `recommend`, `auth`)
//! - Benchmark harness (`bench`)
//! - Image generation (`image generate`)

use std::path::PathBuf;
use std::process::{Command, ExitCode};

#[path = "../agent.rs"]
mod agent;
#[path = "../daemon.rs"]
mod daemon;

const USAGE: &str = "\
turbospark -- unified inference, model management, and server CLI

USAGE:
    turbospark <COMMAND> [OPTIONS]

RUN & CHAT:
    run <MODEL> [PROMPT] [FLAGS]   run interactive chat REPL or a quick prompt
                                   (e.g. 'turbospark run gemma4' or
                                    'turbospark run gemma4 \"explain quantum computing\"')

SERVER LIFECYCLE (oMLX style):
    serve [OPTIONS]                run OpenAI/Anthropic HTTP server in foreground
    start [OPTIONS]                start managed background server daemon
    stop                           stop background server daemon
    restart [OPTIONS]              restart background server daemon
    status                         check server daemon status and health

AGENTS (Unsloth Start style):
    start <AGENT> [OPTIONS]        connect coding agents to local server:
                                   claude, codex, opencode, hermes, openclaw, dsh

MODEL MANAGEMENT (Unsloth & oMLX style):
    list [--filter TEXT]           list curated models, marking installed ones
    pull <ALIAS|--repo REPO>       install a model into the local store
    info <ALIAS>                   inspect model details and requirements
    rm <ALIAS>                     delete an installed model
    probe <REPO>[@REV]             inspect Hugging Face repo headers without download
    recommend                      rank models by fit for this machine's memory
    path <ALIAS>                   print install path for an alias
    auth [TOKEN]                   manage Hugging Face credentials (--set, --status, --clear)

BENCHMARK:
    bench [OPTIONS]                run throughput and memory benchmarks

IMAGE:
    image generate [OPTIONS]       generate one 1024x1024 PNG from a prompt
    image pack [OPTIONS]           pack a Diffusers image export into an install

GLOBAL OPTIONS:
    --help, -h, help               print this help message
    --version, -V, version         print version

For command-specific options, pass --help to that command:
    turbospark run --help
    turbospark serve --help
    turbospark pull --help
    turbospark image generate --help
    turbospark image pack --help
";

fn find_peer_binary(name: &str) -> PathBuf {
    if let Ok(mut exe) = std::env::current_exe() {
        exe.pop();
        let candidate = exe.join(name);
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from(name)
}

fn execute_peer(binary_name: &str, args: &[String]) -> ExitCode {
    let bin = find_peer_binary(binary_name);
    match Command::new(&bin).args(args).status() {
        Ok(status) => match status.code() {
            Some(code) => ExitCode::from(code as u8),
            None => ExitCode::FAILURE,
        },
        Err(e) => {
            eprintln!("error executing {}: {e}", bin.display());
            ExitCode::FAILURE
        }
    }
}

fn handle_run(args: &[String]) -> ExitCode {
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
        println!("usage: turbospark run <MODEL> [PROMPT] [flags...]\n");
        println!("Examples:");
        println!("  turbospark run gemma4");
        println!("  turbospark run gemma4 \"Explain quantum computing\"");
        println!("  turbospark run gemma4 --temperature 0.7 --max-new 256");
        return ExitCode::SUCCESS;
    }

    let model = &args[0];
    let mut check_args = vec!["--model".to_string(), model.clone()];

    let mut has_mode = false;
    let mut positional_prompt = None;
    let mut extra_flags = Vec::new();

    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--chat" || arg == "--prompt" || arg == "--messages-file" {
            has_mode = true;
            check_args.push(arg.clone());
            if (arg == "--prompt" || arg == "--messages-file") && i + 1 < args.len() {
                i += 1;
                check_args.push(args[i].clone());
            }
        } else if !arg.starts_with('-') && positional_prompt.is_none() && !has_mode {
            positional_prompt = Some(arg.clone());
        } else {
            extra_flags.push(arg.clone());
        }
        i += 1;
    }

    if !has_mode {
        if let Some(prompt) = positional_prompt {
            check_args.push("--prompt".to_string());
            check_args.push(prompt);
        } else {
            check_args.push("--chat".to_string());
        }
    }

    check_args.extend(extra_flags);
    execute_peer("turbospark-check", &check_args)
}

fn handle_image(args: &[String]) -> ExitCode {
    let mut image_args = Vec::with_capacity(args.len() + 2);
    if args.first().map(String::as_str) != Some("generate") {
        image_args.push("generate".to_string());
    }
    image_args.extend_from_slice(args);
    if args.is_empty() {
        image_args.push("--help".to_string());
    }
    execute_peer("turbospark-image", &image_args)
}

fn handle_start(args: &[String]) -> ExitCode {
    if let Some(first) = args.first() {
        if agent::is_agent(first) {
            match agent::start_agent(first, &args[1..]) {
                Ok(()) => return ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
    }

    match daemon::start(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn handle_stop() -> ExitCode {
    match daemon::stop() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn handle_restart(args: &[String]) -> ExitCode {
    match daemon::restart(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn handle_status() -> ExitCode {
    match daemon::status() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }

    let command = args[0].as_str();
    let rest = &args[1..];

    match command {
        "--help" | "-h" | "help" => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        "--version" | "-V" | "version" => {
            println!("turbospark {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        "run" => handle_run(rest),
        "image" => handle_image(rest),
        "serve" => execute_peer("turbospark-server", rest),
        "start" => handle_start(rest),
        "stop" => handle_stop(),
        "restart" => handle_restart(rest),
        "status" => handle_status(),
        "list" | "pull" | "info" | "rm" | "probe" | "recommend" | "path" | "auth" => {
            let mut model_args = vec![command.to_string()];
            model_args.extend_from_slice(rest);
            execute_peer("turbospark-model", &model_args)
        }
        "bench" => execute_peer("turbospark-bench", rest),
        other => {
            eprintln!("unknown command: {other}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}
