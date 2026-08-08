//! `turbospark-server`: binds the generation router (OpenAI
//! `/v1/chat/completions`, Anthropic `/v1/messages`, and `/v1/models`) to
//! loopback, or to this machine's Tailscale IPv4 address. Two modes:
//!
//!   turbospark-server --model <install-dir> [--port N] [--max-context N]
//!                   [--expert-cache-slots N] [--bind loopback|tailnet]
//!   turbospark-server <tokenizer-dir> [port]
//!
//! The first serves real generation from a `.gturbo` install through
//! `RealForwardRunner` (macOS only; one runner per process, requests
//! serialized). The second is the portable scripted mode: it takes only a
//! tokenizer and every response comes from `ScriptedChatModel`, a fixed
//! placeholder sequence, not a forward pass; it always binds loopback.
//!
//! Default port 8080, default bind loopback. `--bind tailnet` is NOT
//! authentication: the server has no auth and no TLS, so access is governed
//! entirely by the Tailnet ACL.

use std::path::PathBuf;
use std::sync::Arc;

use tokenizer::MfTokenizer;

const USAGE: &str = "usage: turbospark-server --model <install-dir> [--port N] [--max-context N] [--expert-cache-slots N] [--bind loopback|tailnet]\n       turbospark-server <tokenizer-dir> [port]";

/// Interface the server listens on. Resolution fails rather than widening:
/// there is no path from `Tailnet` to a wildcard or LAN address.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BindMode {
    Loopback,
    Tailnet,
}

impl BindMode {
    fn host(self) -> Result<String, String> {
        match self {
            BindMode::Loopback => Ok("127.0.0.1".to_string()),
            BindMode::Tailnet => tailnet_host(&tailscale_ipv4_output()?),
        }
    }
}

/// Accepts exactly one Tailscale IPv4 address. Empty, ambiguous, IPv6-only,
/// malformed, and off-range output all fail; none of them fall back.
fn tailnet_host(output: &str) -> Result<String, String> {
    let fields: Vec<&str> = output.split_whitespace().collect();
    match fields.as_slice() {
        [] => Err(
            "tailscale reported no IPv4 address; ensure Tailscale is running and connected"
                .to_string(),
        ),
        [only] if is_tailscale_ipv4(only) => Ok((*only).to_string()),
        [only] => Err(format!(
            "tailscale reported \"{}\", which is not a Tailnet IPv4 address",
            &only[..only.len().min(64)]
        )),
        many => Err(format!(
            "tailscale reported {} IPv4 addresses; refusing to guess which to bind",
            many.len()
        )),
    }
}

/// True for a dotted-quad IPv4 inside 100.64.0.0/10, the range Tailscale
/// allocates from. Restricting to that range keeps a wildcard, loopback, or
/// LAN address from ever being bound.
fn is_tailscale_ipv4(text: &str) -> bool {
    let parts: Vec<&str> = text.split('.').collect();
    if parts.len() != 4 {
        return false;
    }
    let mut octets = [0u8; 4];
    for (slot, part) in octets.iter_mut().zip(parts) {
        // Reject leading zeros: "100.064.0.1" would otherwise pass here and
        // then be read as octal by some resolvers.
        if part.len() > 1 && part.starts_with('0') {
            return false;
        }
        match part.parse::<u8>() {
            Ok(octet) => *slot = octet,
            Err(_) => return false,
        }
    }
    octets[0] == 100 && (64..=127).contains(&octets[1])
}

/// Raw stdout of `tailscale ip -4`. Spawned directly with no shell, so
/// nothing is interpolated into a command line.
fn tailscale_ipv4_output() -> Result<String, String> {
    let out = std::process::Command::new("tailscale")
        .args(["ip", "-4"])
        .stderr(std::process::Stdio::null())
        .output()
        .map_err(|e| {
            format!("could not run tailscale ({e}); install its CLI and keep it on PATH")
        })?;
    if !out.status.success() {
        return Err(format!(
            "tailscale ip -4 exited with {}; ensure Tailscale is running and connected",
            out.status
        ));
    }
    String::from_utf8(out.stdout)
        .map_err(|_| "tailscale ip -4 returned non-UTF-8 output".to_string())
}

struct ModelArgs {
    model: String,
    port: u16,
    max_context: u32,
    expert_cache_slots: u32,
    bind: BindMode,
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
        bind: BindMode::Loopback,
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
            "--bind" => {
                parsed.bind = match value.as_str() {
                    "loopback" => BindMode::Loopback,
                    "tailnet" => BindMode::Tailnet,
                    other => {
                        return Err(format!("--bind must be loopback or tailnet, not {other}"))
                    }
                }
            }
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
fn open_real_model(args: &ModelArgs) -> Result<Arc<dyn turbospark_server::ChatModel>, String> {
    // A 13 GB install takes a noticeable while to map and compile pipelines
    // for; without this line the startup reads as hung.
    eprintln!("opening {} ...", args.model);
    let model = turbospark_server::RealChatModel::open(
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
        assert!(parse(&["--model", "/tmp/m", "--bind", "lan"]).is_err());
    }

    #[test]
    fn bind_mode_defaults_to_loopback() {
        assert_eq!(
            parse(&["--model", "/tmp/m"]).unwrap().unwrap().bind,
            BindMode::Loopback
        );
        let t = parse(&["--model", "/tmp/m", "--bind", "tailnet"])
            .unwrap()
            .unwrap();
        assert_eq!(t.bind, BindMode::Tailnet);
        assert_eq!(BindMode::Loopback.host().unwrap(), "127.0.0.1");
    }

    #[test]
    fn tailnet_host_accepts_exactly_one_in_range_address() {
        assert_eq!(tailnet_host("100.64.0.1\n").unwrap(), "100.64.0.1");
        assert_eq!(
            tailnet_host("100.127.255.254\n").unwrap(),
            "100.127.255.254"
        );
    }

    #[test]
    fn tailnet_host_never_falls_back() {
        // Empty, ambiguous, out-of-range, IPv6, and malformed all fail rather
        // than widening to a loopback/LAN/wildcard bind.
        assert!(tailnet_host("").is_err());
        assert!(tailnet_host("  \n").is_err());
        assert!(tailnet_host("100.64.0.1 100.64.0.2\n").is_err());
        assert!(tailnet_host("100.63.0.1").is_err());
        assert!(tailnet_host("100.128.0.1").is_err());
        assert!(tailnet_host("192.168.1.5").is_err());
        assert!(tailnet_host("127.0.0.1").is_err());
        assert!(tailnet_host("0.0.0.0").is_err());
        assert!(tailnet_host("fd7a:115c:a1e0::1").is_err());
        assert!(tailnet_host("100.64.0").is_err());
        assert!(tailnet_host("100.64.0.256").is_err());
        assert!(tailnet_host("100.064.0.1").is_err());
        assert!(tailnet_host("100.64.0.1;rm -rf /").is_err());
    }
}
