//! `turbospark-server`: binds the generation router (OpenAI
//! `/v1/chat/completions`, Anthropic `/v1/messages`, and `/v1/models`) to
//! loopback, or to this machine's Tailscale IPv4 address. Two modes:
//!
//!   turbospark-server --model <install-dir|alias> [--port N]
//!                   [--max-context N|auto] [--expert-cache-slots auto|N]
//!                   [--bind loopback|tailnet]
//!                   [--power-profile performance|balanced|efficiency]
//!                   [--max-tokens-per-sec R]
//!   turbospark-server <tokenizer-dir> [port]
//!   turbospark-server --help | --version
//!
//! The first serves real generation from a `.gturbo` install through
//! `RealForwardRunner` (macOS only; one runner per process, requests
//! serialized). Its `--model` takes a directory or a `turbospark-model`
//! alias, resolved through the same `catalog::resolve_model_arg` that backs
//! `turbospark-check --model`. The second is the portable scripted mode: it
//! takes only a tokenizer and every response comes from `ScriptedChatModel`,
//! a fixed placeholder sequence, not a forward pass; it always binds
//! loopback.
//!
//! Default port 8080, default bind loopback. `--bind tailnet` is NOT
//! authentication: the server has no auth and no TLS, so access is governed
//! entirely by the Tailnet ACL.

mod args;
mod bind;

use std::path::PathBuf;
use std::sync::Arc;

use args::{parse_model_args, short_circuit, ModelArgs, USAGE};
use tokenizer::MfTokenizer;

#[cfg(target_os = "macos")]
fn open_real_model(args: &ModelArgs) -> Result<Arc<dyn turbospark_server::ChatModel>, String> {
    // `--model` takes a path OR a `turbospark-model` alias, resolved the same
    // way `turbospark-check` resolves it, so one install serves both binaries
    // under one name. An existing directory always wins over an alias: a bare
    // name that silently preferred an alias would serve a DIFFERENT model
    // than the one on the command line, and a server does that unattended.
    let dir = catalog::resolve_model_arg(&args.model);
    // A 13 GB install takes a noticeable while to map and compile pipelines
    // for; without this line the startup reads as hung. Print what the
    // argument RESOLVED to when the two differ, since an alias says nothing
    // about which directory is being served.
    if dir.as_os_str() == args.model.as_str() {
        eprintln!("opening {} ...", args.model);
    } else {
        eprintln!("opening {} ({}) ...", args.model, dir.display());
    }
    // Resolved once here, which is also the one place this process asks the
    // OS about Low Power Mode.
    let profile = runtime::resolve_profile(args.power_profile);
    let rate = runtime::rate_control_for(profile, args.max_tokens_per_sec);
    let model = turbospark_server::RealChatModel::open(
        &dir,
        args.max_context,
        args.expert_cache_slots,
        rate,
        args.speculation,
        args.drafter,
        args.guardrails,
        args.steering.clone(),
    )?;
    // Both sized figures are the RESOLVED ones, never `args`: under `auto`
    // the request carries no number, and each has to be readable beside any
    // throughput or footprint the operator goes on to measure. The context
    // line additionally names the checkpoint's own trained window, which is
    // what explains an `auto` of 4,096 on a machine with room for more.
    let context = model.context_plan();
    eprintln!(
        "model open (max_context {}{} [{}, {:.0} MiB of KV, suggested {}], \
         {} expert cache slots{}, {} profile, rate cap {})",
        context.resolved,
        if args.max_context.is_none() {
            " (auto)"
        } else {
            ""
        },
        match context.trained {
            Some(t) => format!("model {t}"),
            None => "model declares none".to_string(),
        },
        context.kv_bytes as f64 / (1024.0 * 1024.0),
        context.suggested,
        model.expert_cache_slots(),
        if args.expert_cache_slots.is_none() {
            " (auto)"
        } else {
            ""
        },
        profile.as_str(),
        match rate.max_tokens_per_sec {
            Some(r) => format!("{r} tok/s"),
            None => "none".to_string(),
        }
    );
    // A WARNING AND NOT SILENCE, for the reason `open_session` prints the
    // same line: an install carrying a drafter and decoding one token at a
    // time with nothing said is the failure the feature was built to end. A
    // server says it once, at startup, where an operator sees it.
    eprintln!("{}", model.speculation_line());
    eprintln!(
        "  guardrails: {}",
        if args.guardrails.active() {
            "on (tool calls rescued and argument-checked; a request with tools is buffered, not streamed)"
        } else {
            "off"
        }
    );
    Ok(Arc::new(model))
}

#[cfg(not(target_os = "macos"))]
fn open_real_model(_args: &ModelArgs) -> Result<Arc<dyn turbospark_server::ChatModel>, String> {
    Err("--model needs macOS and a Metal device; only the scripted \
         <tokenizer-dir> mode is available on this platform"
        .to_string())
}

fn open_scripted(tokenizer_dir: &str) -> Result<Arc<dyn turbospark_server::ChatModel>, String> {
    let tok = MfTokenizer::load_from_dir(&PathBuf::from(tokenizer_dir))
        .map_err(|e| format!("failed to load tokenizer: {e}"))?;
    Ok(Arc::new(turbospark_server::ScriptedChatModel::new(
        tok,
        4096,
        Vec::new(),
    )))
}

#[tokio::main]
async fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("{USAGE}");
        return std::process::ExitCode::from(2);
    }
    if let Some(text) = short_circuit(&args) {
        print!("{text}");
        return std::process::ExitCode::SUCCESS;
    }

    let (model, port, bind) = match parse_model_args(&args) {
        Err(e) => {
            eprintln!("{e}");
            return std::process::ExitCode::from(2);
        }
        Ok(Some(parsed)) => {
            // Resolve the host BEFORE opening the model: a missing Tailscale
            // should not cost a multi-gigabyte map and a pipeline compile
            // first.
            let host = match parsed.bind.host() {
                Ok(h) => h,
                Err(e) => {
                    eprintln!("{e}");
                    return std::process::ExitCode::from(2);
                }
            };
            match open_real_model(&parsed) {
                Ok(m) => (m, parsed.port, host),
                Err(e) => {
                    eprintln!("{e}");
                    return std::process::ExitCode::from(2);
                }
            }
        }
        Ok(None) => {
            let port: u16 = args.get(1).and_then(|p| p.parse().ok()).unwrap_or(8080);
            match open_scripted(&args[0]) {
                Ok(m) => (m, port, "127.0.0.1".to_string()),
                Err(e) => {
                    eprintln!("{e}");
                    return std::process::ExitCode::from(1);
                }
            }
        }
    };

    let router = turbospark_server::build_router(model);
    let addr = format!("{bind}:{port}");
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("failed to bind {addr}: {e}");
            return std::process::ExitCode::from(1);
        }
    };
    eprintln!("turbospark-server listening on http://{addr}");
    eprintln!("  POST /v1/chat/completions   (OpenAI)");
    eprintln!("  POST /v1/messages           (Anthropic)");
    eprintln!("  GET  /v1/models");
    if let Err(e) = axum::serve(listener, router).await {
        eprintln!("server error: {e}");
        return std::process::ExitCode::from(1);
    }
    std::process::ExitCode::SUCCESS
}

#[cfg(test)]
#[path = "main_tests.rs"]
mod tests;
