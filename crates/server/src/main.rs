//! `turbospark-server`: binds the generation router (OpenAI
//! `/v1/chat/completions`, Anthropic `/v1/messages`, and `/v1/models`) to
//! loopback, or to this machine's Tailscale IPv4 address. Two modes:
//!
//!   turbospark-server --model <install-dir|alias> [--port N] [--max-context N]
//!                   [--expert-cache-slots auto|N] [--bind loopback|tailnet]
//!                   [--power-profile performance|balanced|efficiency]
//!                   [--max-tokens-per-sec R]
//!   turbospark-server <tokenizer-dir> [port]
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

use std::path::PathBuf;
use std::sync::Arc;

use tokenizer::MfTokenizer;

const USAGE: &str = "usage: turbospark-server --model <install-dir|alias> [--port N] [--max-context N] [--expert-cache-slots auto|N] [--bind loopback|tailnet] [--power-profile performance|balanced|efficiency] [--max-tokens-per-sec R]\n       turbospark-server <tokenizer-dir> [port]\n\n`--model` takes a .gturbo directory or a turbospark-model alias (`turbospark-model list`).";

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
    /// `None` is `auto`, which is also the default -- the slot count is
    /// sized against this machine and this install at open. Spelled as an
    /// `Option` rather than reusing `invocation`'s enum because this binary
    /// has its own flat parser and does not depend on that crate.
    expert_cache_slots: Option<u32>,
    bind: BindMode,
    /// ROADMAP Phase P2. Process-level, like every other flag here: there
    /// is one runner per process, so there is nothing per-request to vary.
    power_profile: Option<runtime::PowerProfile>,
    max_tokens_per_sec: Option<f64>,
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
        expert_cache_slots: None,
        bind: BindMode::Loopback,
        power_profile: None,
        max_tokens_per_sec: None,
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
            "--expert-cache-slots" => {
                parsed.expert_cache_slots = if value == "auto" {
                    None
                } else {
                    Some(number()?)
                }
            }
            "--bind" => {
                parsed.bind = match value.as_str() {
                    "loopback" => BindMode::Loopback,
                    "tailnet" => BindMode::Tailnet,
                    other => {
                        return Err(format!("--bind must be loopback or tailnet, not {other}"))
                    }
                }
            }
            "--power-profile" => {
                let profile = runtime::PowerProfile::parse(value).ok_or_else(|| {
                    format!("--power-profile must be a profile name, not {value}")
                })?;
                parsed.power_profile = Some(profile);
            }
            "--max-tokens-per-sec" => {
                let rate = value
                    .parse::<f64>()
                    .map_err(|e| format!("--max-tokens-per-sec: {e}"))?;
                if !rate.is_finite() || rate <= 0.0 {
                    return Err(format!(
                        "--max-tokens-per-sec must be greater than 0, not {value}"
                    ));
                }
                parsed.max_tokens_per_sec = Some(rate);
            }
            other => return Err(format!("unknown option {other}\n{USAGE}")),
        }
        i += 2;
    }
    if parsed.model.is_empty() {
        return Err(format!("--model needs a value\n{USAGE}"));
    }
    // Read the allowed set rather than re-hardcoding it (AGENTS.md Gotcha 2);
    // the runtime setter would panic on a value outside it. `auto` is
    // unchecked because the resolver only ever returns a member of that same
    // set, which `every_resolved_value_is_in_the_allowed_set` asserts.
    if let Some(n) = parsed.expert_cache_slots {
        if !foundation::ALLOWED_CACHE_SLOTS.contains(&n) {
            return Err(format!(
                "--expert-cache-slots must be auto or one of {:?}",
                foundation::ALLOWED_CACHE_SLOTS
            ));
        }
    }
    Ok(Some(parsed))
}

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
    )?;
    // The slot count is the RESOLVED one, never `args`: under `auto` the
    // request carries no number, and the figure has to be readable beside
    // any throughput or footprint the operator goes on to measure.
    eprintln!(
        "model open (max_context {}, {} expert cache slots{}, {} profile, rate cap {})",
        args.max_context,
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

    /// `--model` reaches the catalog store, so one install serves this
    /// binary and `turbospark-check` under one alias.
    ///
    /// The assertion is deliberately on a name that is NOT a directory:
    /// swapping `resolve_model_arg` back for `PathBuf::from` leaves every
    /// path case passing, because for a real path the two agree. Only the
    /// alias arm can tell them apart, and this is the cheapest form of it
    /// -- the "default install location that happens to exist" arm, which
    /// needs no `installed.json` and no model.
    ///
    /// It is the only test in this binary that touches `TURBOSPARK_HOME`,
    /// which is what keeps it safe under the default parallel test threads.
    #[cfg(target_os = "macos")]
    #[test]
    fn an_alias_resolves_to_its_install_directory() {
        let root = std::env::temp_dir().join(format!(
            "turbospark-server-alias-{}-{}",
            std::process::id(),
            line!()
        ));
        let installed = root.join("models").join("some-alias.gturbo");
        std::fs::create_dir_all(&installed).unwrap();
        std::env::set_var("TURBOSPARK_HOME", &root);

        assert_eq!(
            catalog::resolve_model_arg("some-alias"),
            installed,
            "an alias should resolve to its install directory"
        );
        // The property the resolution order exists to protect: a bare name
        // that is also a real directory is that directory, never the alias.
        assert_ne!(
            catalog::resolve_model_arg("some-alias"),
            PathBuf::from("some-alias"),
            "resolution must not be a pass-through for a known alias"
        );

        std::env::remove_var("TURBOSPARK_HOME");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn power_flags_default_to_unset_and_parse_their_documented_values() {
        // Unset rather than `performance`: the Low Power Mode default is
        // resolved at model open, where the OS can be asked.
        let d = parse(&["--model", "/tmp/m"]).unwrap().unwrap();
        assert_eq!(d.power_profile, None);
        assert_eq!(d.max_tokens_per_sec, None);

        let o = parse(&[
            "--model",
            "/tmp/m",
            "--power-profile",
            "efficiency",
            "--max-tokens-per-sec",
            "7.5",
        ])
        .unwrap()
        .unwrap();
        assert_eq!(o.power_profile, Some(runtime::PowerProfile::Efficiency));
        assert_eq!(o.max_tokens_per_sec, Some(7.5));
    }

    #[test]
    fn bad_power_flag_values_are_rejected() {
        assert!(parse(&["--model", "/tmp/m", "--power-profile", "turbo"]).is_err());
        for bad in ["0", "-1", "abc", "inf"] {
            assert!(
                parse(&["--model", "/tmp/m", "--max-tokens-per-sec", bad]).is_err(),
                "expected {bad} to be rejected"
            );
        }
    }

    #[test]
    fn model_mode_defaults_and_overrides() {
        let d = parse(&["--model", "/tmp/m"]).unwrap().unwrap();
        // Slots default to `None`, i.e. `auto`: sized at open against this
        // machine and this install, never below the shipped 16.
        assert_eq!(
            (d.port, d.max_context, d.expert_cache_slots),
            (8080, 4096, None)
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
        assert_eq!(
            (o.port, o.max_context, o.expert_cache_slots),
            (9, 1024, Some(32))
        );
        // `auto` is accepted by name as well as by omission, and is the one
        // value the allowed-set check must not reject.
        let a = parse(&["--model", "/tmp/m", "--expert-cache-slots", "auto"])
            .unwrap()
            .unwrap();
        assert_eq!(a.expert_cache_slots, None);
        assert!(parse(&["--model", "/tmp/m", "--expert-cache-slots", "20"]).is_err());
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
