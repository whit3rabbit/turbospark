//! `turbospark-acp-agent`: the real local engine behind `turbospark acp`.
//!
//! Serves the Agent Client Protocol on this process's stdin and stdout
//! (`turbospark-acp`'s `serve`) and answers prompts with a local model loaded
//! on the first prompt. stdout is the protocol channel and carries nothing
//! else. A SEPARATE binary, like `turbospark-model` and `turbospark-steer`,
//! because `turbospark-check`'s parser is a pure flat grammar. It is not named
//! `turbospark-acp`: that is `crates/acp`'s protocol-only binary.
//!
//! Flags:
//!   --model M            install path or catalog alias (required)
//!   --allow-root DIR     restrict session workspaces to DIR (repeatable)
//!   --state-dir DIR      where session records go
//!   --no-persist         keep sessions in memory only
//!
//! Permissions stay default-deny: nothing in this binary approves a tool.
//! Exit status: 0 clean EOF, 1 input error, 2 invalid command line.

// The generation modules are shared with `turbospark-check` by path; this
// binary uses only part of them.
#![allow(dead_code)]

use std::io::{BufReader, Write};
use std::path::PathBuf;
use turbospark_acp::{serve, AcpConfig};

#[path = "../acp_engine.rs"]
mod acp_engine;
#[path = "../acp_lock.rs"]
mod acp_lock;
#[cfg(target_os = "macos")]
#[path = "../acp_real.rs"]
mod acp_real;
#[cfg(target_os = "macos")]
#[path = "../chat.rs"]
mod chat;
#[cfg(target_os = "macos")]
#[path = "../generate/mod.rs"]
mod generate;

fn default_state_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("TURBOSPARK_STATE_DIR") {
        return Some(PathBuf::from(d).join("acp-sessions"));
    }
    if let Some(h) = std::env::var_os("TURBOSPARK_HOME") {
        return Some(PathBuf::from(h).join("acp-sessions"));
    }
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".turbospark").join("acp-sessions"))
}

fn usage(msg: &str) -> ! {
    eprintln!("turbospark-acp-agent: {msg}");
    eprintln!(
        "usage: turbospark acp --model MODEL [--allow-root DIR]... [--state-dir DIR | --no-persist]"
    );
    std::process::exit(2);
}

#[cfg(target_os = "macos")]
fn make_engine(model: String) -> acp_engine::LocalEngine {
    let id = model.clone();
    let factory: acp_engine::BackendFactory = Box::new(move || {
        acp_real::RealBackend::open(&model).map(|b| Box::new(b) as Box<dyn acp_engine::ChatBackend>)
    });
    acp_engine::LocalEngine::new(id, factory)
}

#[cfg(not(target_os = "macos"))]
fn make_engine(model: String) -> acp_engine::LocalEngine {
    let factory: acp_engine::BackendFactory =
        Box::new(|| Err("local generation needs macOS with a Metal device".to_string()));
    acp_engine::LocalEngine::new(model, factory)
}

fn main() {
    let mut cfg = AcpConfig {
        state_dir: default_state_dir(),
        ..AcpConfig::default()
    };
    let mut model: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => match args.next() {
                Some(m) if !m.is_empty() => model = Some(m),
                _ => usage("--model needs a value"),
            },
            "--allow-root" => match args.next() {
                Some(d) => cfg.allowed_roots.push(PathBuf::from(d)),
                None => usage("--allow-root needs a directory"),
            },
            "--state-dir" => match args.next() {
                Some(d) => cfg.state_dir = Some(PathBuf::from(d)),
                None => usage("--state-dir needs a directory"),
            },
            "--no-persist" => cfg.state_dir = None,
            "--version" => {
                println!("turbospark-acp-agent {}", env!("CARGO_PKG_VERSION"));
                return;
            }
            "--help" | "-h" => {
                println!(
                    "turbospark acp: serve the Agent Client Protocol on stdio with a local model\n\n\
                     usage: turbospark acp --model MODEL [--allow-root DIR]... \
                     [--state-dir DIR | --no-persist]"
                );
                return;
            }
            _ => usage("unknown argument"),
        }
    }
    let Some(model) = model else {
        usage("--model is required");
    };

    let engine = make_engine(model);
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let code = match serve(&engine, &cfg, BufReader::new(stdin.lock()), stdout) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("turbospark-acp-agent: input error: {}", e.kind());
            1
        }
    };
    // Dropped explicitly: `process::exit` skips destructors, and the drop is
    // what joins the model thread, unmaps the model and frees the lock.
    drop(engine);
    let _ = std::io::stderr().flush();
    std::process::exit(code);
}
