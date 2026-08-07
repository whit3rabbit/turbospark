//! `mference-server`: binds the OpenAI-compatible Chat Completions router to
//! loopback. Two modes:
//!
//!   mference-server --model <install-dir> [--port N] [--max-context N]
//!                   [--expert-cache-slots N]
//!   mference-server <tokenizer-dir> [port]
//!
//! The first serves real generation from a `.gturbo` install through
//! `RealForwardRunner` (macOS only; one runner per process, requests
//! serialized). The second is the portable scripted mode: it takes only a
//! tokenizer and every response comes from `ScriptedChatModel`, a fixed
//! placeholder sequence, not a forward pass.
//!
//! Default port 8080. Optional tailnet bind (vs. loopback-only) is not
//! implemented.

use std::path::PathBuf;
use std::sync::Arc;

use tokenizer::MfTokenizer;

const USAGE: &str = "usage: mference-server --model <install-dir> [--port N] [--max-context N] [--expert-cache-slots N]\n       mference-server <tokenizer-dir> [port]";

struct ModelArgs {
    model: String,
    port: u16,
    max_context: u32,
    expert_cache_slots: u32,
}

/// Parses the `--model` mode's flags. Returns `Ok(None)` when the first
/// argument is not `--model`, leaving the caller on the legacy positional
/// path.
fn parse_model_args(args: &[String]) -> Result<Option<ModelArgs>, String> {
    if args.first().map(String::as_str) != Some("--model") {
        return Ok(None);
    }
    let mut parsed = ModelArgs {
        model: String::new(),
        port: 8080,
        max_context: 4096,
        expert_cache_slots: 16,
    };
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        let value = args
            .get(i + 1)
            .ok_or_else(|| format!("{flag} needs a value"))?;
        let number = || value.parse::<u32>().map_err(|e| format!("{flag}: {e}"));
        match flag {
            "--model" => parsed.model = value.clone(),
            "--port" => parsed.port = value.parse::<u16>().map_err(|e| format!("--port: {e}"))?,
            "--max-context" => parsed.max_context = number()?,
            "--expert-cache-slots" => parsed.expert_cache_slots = number()?,
            other => return Err(format!("unknown option {other}\n{USAGE}")),
        }
        i += 2;
    }
    if parsed.model.is_empty() {
        return Err(format!("--model needs a value\n{USAGE}"));
    }
    // Read the allowed set rather than re-hardcoding it (AGENTS.md Gotcha 2);
    // the runtime setter would panic on a value outside it.
    if !foundation::ALLOWED_CACHE_SLOTS.contains(&parsed.expert_cache_slots) {
        return Err(format!(
            "--expert-cache-slots must be one of {:?}",
            foundation::ALLOWED_CACHE_SLOTS
        ));
    }
    Ok(Some(parsed))
}

#[cfg(target_os = "macos")]
fn open_real_model(args: &ModelArgs) -> Result<Arc<dyn mrefrust_server::ChatModel>, String> {
    // A 13 GB install takes a noticeable while to map and compile pipelines
    // for; without this line the startup reads as hung.
    eprintln!("opening {} ...", args.model);
    let model = mrefrust_server::RealChatModel::open(
        &PathBuf::from(&args.model),
        args.max_context,
        args.expert_cache_slots,
    )?;
    eprintln!(
        "model open (max_context {}, {} expert cache slots)",
        args.max_context, args.expert_cache_slots
    );
    Ok(Arc::new(model))
}

#[cfg(not(target_os = "macos"))]
fn open_real_model(_args: &ModelArgs) -> Result<Arc<dyn mrefrust_server::ChatModel>, String> {
    Err("--model needs macOS and a Metal device; only the scripted \
         <tokenizer-dir> mode is available on this platform"
        .to_string())
}

fn open_scripted(tokenizer_dir: &str) -> Result<Arc<dyn mrefrust_server::ChatModel>, String> {
    let tok = MfTokenizer::load_from_dir(&PathBuf::from(tokenizer_dir))
        .map_err(|e| format!("failed to load tokenizer: {e}"))?;
    Ok(Arc::new(mrefrust_server::ScriptedChatModel::new(
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

    let (model, port) = match parse_model_args(&args) {
        Err(e) => {
            eprintln!("{e}");
            return std::process::ExitCode::from(2);
        }
        Ok(Some(parsed)) => match open_real_model(&parsed) {
            Ok(m) => (m, parsed.port),
            Err(e) => {
                eprintln!("{e}");
                return std::process::ExitCode::from(2);
            }
        },
        Ok(None) => {
            let port: u16 = args.get(1).and_then(|p| p.parse().ok()).unwrap_or(8080);
            match open_scripted(&args[0]) {
                Ok(m) => (m, port),
                Err(e) => {
                    eprintln!("{e}");
                    return std::process::ExitCode::from(1);
                }
            }
        }
    };

    let router = mrefrust_server::build_router(model);
    let addr = format!("127.0.0.1:{port}");
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("failed to bind {addr}: {e}");
            return std::process::ExitCode::from(1);
        }
    };
    eprintln!("mference-server listening on http://{addr}/v1/chat/completions");
    if let Err(e) = axum::serve(listener, router).await {
        eprintln!("server error: {e}");
        return std::process::ExitCode::from(1);
    }
    std::process::ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(argv: &[&str]) -> Result<Option<ModelArgs>, String> {
        let owned: Vec<String> = argv.iter().map(|s| s.to_string()).collect();
        parse_model_args(&owned)
    }

    #[test]
    fn legacy_positional_mode_is_left_alone() {
        assert!(parse(&["/tmp/tok", "9000"]).unwrap().is_none());
    }

    #[test]
    fn model_mode_defaults_and_overrides() {
        let d = parse(&["--model", "/tmp/m"]).unwrap().unwrap();
        assert_eq!(
            (d.port, d.max_context, d.expert_cache_slots),
            (8080, 4096, 16)
        );
        let o = parse(&[
            "--model",
            "/tmp/m",
            "--port",
            "9",
            "--max-context",
            "1024",
            "--expert-cache-slots",
            "32",
        ])
        .unwrap()
        .unwrap();
        assert_eq!((o.port, o.max_context, o.expert_cache_slots), (9, 1024, 32));
    }

    #[test]
    fn bad_model_mode_arguments_are_rejected() {
        // A slot count outside the allowed set would panic the runtime
        // config setter, so it has to fail here instead.
        assert!(parse(&["--model", "/tmp/m", "--expert-cache-slots", "12"]).is_err());
        assert!(parse(&["--model", "/tmp/m", "--port", "70000"]).is_err());
        assert!(parse(&["--model"]).is_err());
        assert!(parse(&["--model", "/tmp/m", "--nope", "1"]).is_err());
    }
}
