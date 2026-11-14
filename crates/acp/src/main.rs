//! `turbospark-acp`: serve the adapter on this process's stdin and stdout.
//!
//! Flags (no others):
//!   --allow-root <DIR>   restrict session workspaces to DIR (repeatable)
//!   --state-dir <DIR>    where session records go (default: see docs/ACP.md)
//!   --no-persist         keep sessions in memory only
//!
//! stdout is the protocol channel and carries nothing else.

use std::io::{BufReader, Write};
use std::path::PathBuf;
use turbospark_acp::{serve, AcpConfig, UnwiredEngine};

fn default_state_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("TURBOSPARK_STATE_DIR") {
        return Some(PathBuf::from(d).join("acp-sessions"));
    }
    if let Some(h) = std::env::var_os("TURBOSPARK_HOME") {
        return Some(PathBuf::from(h).join("acp-sessions"));
    }
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".turbospark").join("acp-sessions"))
}

fn main() {
    let mut cfg = AcpConfig {
        state_dir: default_state_dir(),
        ..AcpConfig::default()
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
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
                println!("turbospark-acp {}", env!("CARGO_PKG_VERSION"));
                return;
            }
            _ => usage("unknown argument"),
        }
    }
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let engine = UnwiredEngine;
    let code = match serve(&engine, &cfg, BufReader::new(stdin.lock()), stdout) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("turbospark-acp: input error: {}", e.kind());
            1
        }
    };
    let _ = std::io::stderr().flush();
    std::process::exit(code);
}

fn usage(msg: &str) -> ! {
    eprintln!("turbospark-acp: {msg}");
    eprintln!("usage: turbospark-acp [--allow-root DIR]... [--state-dir DIR | --no-persist]");
    std::process::exit(2);
}
