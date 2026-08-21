pub const USAGE: &str = "usage: turbospark-server --model <install-dir|alias> [--port N] [--max-context N|auto] [--expert-cache-slots auto|N] [--bind loopback|tailnet] [--power-profile performance|balanced|efficiency] [--max-tokens-per-sec R] [--speculative off|auto|N] [--speculative-drafter auto|mtp|dflash]\n       turbospark-server <tokenizer-dir> [port]\n       turbospark-server --help | --version\n\noptions:\n  --model              a .gturbo directory or a turbospark-model alias (`turbospark-model list`)\n  --port               listen port (default 8080)\n  --max-context        context window in tokens, or auto (default auto: the\n                       checkpoint's trained context, capped by what memory\n                       holds, and 4096 when the install declares none)\n  --expert-cache-slots routed-cache slots per layer: auto or 8/16/24/32 (default auto)\n  --bind               loopback or tailnet (default loopback; tailnet is NOT auth)\n  --power-profile      performance, balanced or efficiency\n  --max-tokens-per-sec decode rate cap, greater than 0\n  --speculative        off, auto, or a block size 1-15 (default auto). Speculation\n                       applies to temperature-0 requests only; others decode\n                       sequentially\n  --speculative-drafter auto, mtp or dflash (default auto; auto reports a DFlash2\n                       drafter but does not enable it -- see docs/DFLASH2.md)\n  --help               print this text and exit\n  --version            print the version and exit";

/// Interface the server listens on. Resolution fails rather than widening:
/// there is no path from `Tailnet` to a wildcard or LAN address.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BindMode {
    Loopback,
    Tailnet,
}

impl BindMode {
    pub fn host(self) -> Result<String, String> {
        match self {
            BindMode::Loopback => Ok("127.0.0.1".to_string()),
            BindMode::Tailnet => tailnet_host(&tailscale_ipv4_output()?),
        }
    }
}

/// Accepts exactly one Tailscale IPv4 address. Empty, ambiguous, IPv6-only,
/// malformed, and off-range output all fail; none of them fall back.
pub fn tailnet_host(output: &str) -> Result<String, String> {
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
pub fn is_tailscale_ipv4(text: &str) -> bool {
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
pub fn tailscale_ipv4_output() -> Result<String, String> {
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

#[derive(Debug)]
pub struct ModelArgs {
    pub model: String,
    pub port: u16,
    /// `None` is `auto`, which is also the default -- resolved against the
    /// checkpoint's trained context and this machine's memory at open.
    /// Spelled as an `Option` rather than reusing `invocation`'s enum for the
    /// reason `expert_cache_slots` is: this binary has its own flat parser
    /// and does not depend on that crate.
    pub max_context: Option<u32>,
    /// `None` is `auto`, which is also the default -- the slot count is
    /// sized against this machine and this install at open. Spelled as an
    /// `Option` rather than reusing `invocation`'s enum because this binary
    /// has its own flat parser and does not depend on that crate.
    pub expert_cache_slots: Option<u32>,
    pub bind: BindMode,
    /// ROADMAP Phase P2. Process-level, like every other flag here: there
    /// is one runner per process, so there is nothing per-request to vary.
    pub power_profile: Option<runtime::PowerProfile>,
    pub max_tokens_per_sec: Option<f64>,
    /// Process-level for the same reason the two above are, and one more:
    /// the drafter's state is allocated at OPEN, so there is nothing a
    /// request could switch. The per-request half is whether the request is
    /// deterministic, which `RealChatModel::run_completion` applies.
    pub speculation: runtime::Speculation,
    pub drafter: runtime::SpeculativeDrafter,
}

/// Parses the `--model` mode's flags. Returns `Ok(None)` when the first
/// argument is not an OPTION at all, leaving the caller on the legacy
/// positional path.
///
/// **The test used to be `args[0] == "--model"`, and that is why
/// `turbospark-server --port 8080 --model X` died with "failed to load
/// tokenizer".** Any flag-led invocation whose first token was not exactly
/// `--model` fell through to the scripted mode, which read that flag as a
/// tokenizer DIRECTORY and reported a filesystem error about a path nobody
/// had typed. Anything starting with `-` is now handled here, so a
/// mis-ordered or misspelled flag gets the usage text.
pub fn parse_model_args(args: &[String]) -> Result<Option<ModelArgs>, String> {
    if !args.first().is_some_and(|a| a.starts_with('-')) {
        return Ok(None);
    }
    let mut parsed = ModelArgs {
        model: String::new(),
        port: 8080,
        max_context: None,
        expert_cache_slots: None,
        bind: BindMode::Loopback,
        power_profile: None,
        max_tokens_per_sec: None,
        speculation: runtime::Speculation::Auto,
        drafter: runtime::SpeculativeDrafter::Auto,
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
            "--max-context" => {
                parsed.max_context = if value == "auto" {
                    None
                } else {
                    let n = number()?;
                    if n == 0 {
                        return Err("--max-context must be auto or greater than 0".to_string());
                    }
                    Some(n)
                }
            }
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
            "--speculative" => {
                parsed.speculation = match value.as_str() {
                    "off" => runtime::Speculation::Off,
                    "auto" => runtime::Speculation::Auto,
                    // Read the allowed range rather than re-hardcoding it,
                    // exactly as `--expert-cache-slots` below reads
                    // `ALLOWED_CACHE_SLOTS` (AGENTS.md Gotcha 2). It lives in
                    // `foundation` so this parser and `crates/invocation`'s
                    // share one range.
                    _ => match value.parse::<u32>() {
                        Ok(n)
                            if foundation::runtime_config::ALLOWED_SPECULATION_BLOCKS
                                .contains(&n) =>
                        {
                            runtime::Speculation::Block(n)
                        }
                        _ => {
                            return Err(format!(
                                "--speculative must be off, auto, or a block in {:?}, not {value}",
                                foundation::runtime_config::ALLOWED_SPECULATION_BLOCKS
                            ))
                        }
                    },
                }
            }
            "--speculative-drafter" => {
                parsed.drafter = match value.as_str() {
                    "auto" => runtime::SpeculativeDrafter::Auto,
                    "mtp" => runtime::SpeculativeDrafter::Mtp,
                    "dflash" => runtime::SpeculativeDrafter::Dflash,
                    other => {
                        return Err(format!(
                            "--speculative-drafter must be auto, mtp or dflash, not {other}"
                        ))
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

/// Text a `--help` or `--version` token short-circuits to, whichever is
/// reached FIRST in a left-to-right scan.
///
/// Handled ahead of [`parse_model_args`] because both take no value, and
/// that loop advances two tokens per flag: reaching them there would consume
/// whatever followed as a value. Mirrors `turbospark-check`, where
/// `crates/invocation`'s scan returns at the token for the same reason.
pub fn short_circuit(args: &[String]) -> Option<String> {
    args.iter().find_map(|arg| match arg.as_str() {
        "--help" | "-h" => Some(format!("{USAGE}\n")),
        // From cargo, not a literal, and deliberately NOT by depending on
        // `turbospark-invocation` for its `render_version`: every crate here
        // inherits `version.workspace = true`, so this env var is the same
        // string that crate would return, and a dependency edge added for
        // one format string is the wrong trade. The two spellings are
        // pinned against each other by `the_version_line_matches_the_clis`.
        "--version" | "-V" => Some(format!("turbospark {}\n", env!("CARGO_PKG_VERSION"))),
        _ => None,
    })
}
