/// Command line usage and flag description text for `turbospark-server`.
pub const USAGE: &str = "usage: turbospark-server [--model <install-dir|alias>]... [--embedding-model <install-dir|alias>] [--model-dir PATH] [--port N] [--max-context N|auto] [--load-guard TIER|BYTES] [--memory-guard TIER] [--memory-guard-gb N] [--min-auto-context N] [--expert-cache-slots auto|N] [--bind loopback|tailnet] [--power-profile performance|balanced|efficiency] [--max-tokens-per-sec R] [--speculative off|auto|N] [--speculative-drafter auto|mtp|dflash] [--guardrails on|off] [--prefix-reuse on|off] [--session-slots N] [--max-concurrent-requests N] [--pool-size N] [--reasoning off|low|medium|high|xhigh] [--system TEXT] [--system-file PATH] [--api-key KEY] [--hf-endpoint URL] [--paged-ssd-cache-dir PATH] [--hot-cache-max-size SIZE] [--mcp-config PATH] [--steering PATH] [--steering-mode ablate|add|clamp|renorm] [--steering-scale F] [--steering-layers S:E] [--steering-target F] [--steering-gate F] [--vision-sidecar PATH|auto] [--kv-bits off|2|3|3.5|4]\n       turbospark-server <tokenizer-dir> [port]\n       turbospark-server --help | --version\n\noptions:\n  --model              a .gturbo directory or a turbospark-model alias (`turbospark-model list`); repeat to attach distinct generation models\n  --embedding-model    path to an embedding model (.safetensors directory) or alias\n                       to serve for /v1/embeddings alongside generation\n  --model-dir          directory containing .gturbo models (omlx compatibility)\n  --port               listen port (default 8080)\n  --max-context        context window in tokens, or auto (default auto: the\n                       checkpoint's trained context, capped by what memory\n                       holds, and 4096 when the install declares none)\n  --load-guard         how much of the machine a session may commit: off,\n                       relaxed (default), balanced, strict, or a byte ceiling on\n                       what the engine ALLOCATES. relaxed is what shipped before\n                       this flag and what every published memory figure was\n                       measured under; see docs/LOAD_GUARD.md\n  --memory-guard       alias for --load-guard: safe (balanced), balanced, strict,\n                       relaxed, off (omlx compatibility)\n  --memory-guard-gb    set custom memory guard ceiling in gigabytes (omlx compatibility)\n  --min-auto-context   refuse to open when --max-context auto resolves below this\n                       many tokens (default 0, no floor). Says nothing about an\n                       explicit --max-context\n  --expert-cache-slots routed-cache slots per layer: auto or 8/16/24/32/48/64/96/128 (default auto)\n  --bind               loopback or tailnet (default loopback; tailnet requires\n                       --api-key or $TURBOSPARK_API_KEY)\n  --power-profile      performance, balanced or efficiency\n  --max-tokens-per-sec decode rate cap, greater than 0\n  --speculative        off, auto, or a block size 1-15 (default auto). Speculation\n                       applies to temperature-0 requests only; others decode\n                       sequentially\n  --speculative-drafter auto, mtp or dflash (default auto; auto reports a DFlash2\n                       drafter but does not enable it -- see docs/DFLASH2.md)\n  --guardrails         on or off (default on). Rescues a tool call the decoder\n                       could not parse, checks arguments against the request's\n                       own schema, and re-asks once. A request carrying TOOLS is\n                       buffered rather than streamed while this is on, because a\n                       verdict needs the whole turn; requests without tools are\n                       unaffected\n  --prefix-reuse       on or off (default off). When explicitly enabled, a request\n                       continues from the previous request's KV cache wherever\n                       the prompts agree, instead of re-prefilling the whole\n                       transcript. Enable only when every request belongs to one\n                       trusted client: the shared cache is not partitioned by API\n                       key or client, so reuse can expose prefix matches through\n                       response timing. It also raises the\n                       idle-memory floor between requests, not the peak, since\n                       pages that would normally be released stay resident.\n                       See crates/runtime/CLAUDE.md Gotcha 30\n  --session-slots      how many DISTINCT conversations this runner may keep\n                       reusable KV/recurrent state for at once (default 1, i.e.\n                       no pool). Real committed memory per extra slot, unlike\n                       --prefix-reuse's floor-only cost; needs --prefix-reuse on,\n                       since a parked session is never reused\n                       without it. See crates/server/CLAUDE.md's --session-slots\n                       Gotcha\n  --pool-size          how many independent runners to open of --model's ONE install
                       (default 1, i.e. one runner per process as always). N > 1
                       serves N CONCURRENT generations of one model: each member
                       has its own KV, session pool and admission gate, and
                       requests route to the least-busy member. Every member pays
                       real memory and the load guard on its own, so a member
                       that does not fit refuses at startup. See ROADMAP P3.6
  --max-concurrent-requests alias for --session-slots (omlx compatibility). This runner\n                       is serial (one generation at a time, crates/server/CLAUDE.md\n                       Gotcha 1), so this does NOT raise how many requests run at\n                       once -- it commits real KV/recurrent memory per extra slot\n                       so a request from a DIFFERENT conversation can be served\n                       without discarding this one's reusable state\n  --reasoning          default reasoning effort for requests that do not specify\n                       reasoning_effort: off, low, medium, high or xhigh\n                       (default off)\n  --system             default system prompt for requests that carry no system\n                       or developer message of their own. Repeatable; repeats\n                       join with a newline. A request that sends its own system\n                       message is left exactly as it arrived\n  --system-file        read the same default system prompt from a file, for a\n                       prompt too long to sit on a command line. Mutually\n                       exclusive with --system\n  --api-key            require this key on every request except GET /health,\n                       as `Authorization: Bearer <key>` or `x-api-key: <key>`.\n                       Falls back to $TURBOSPARK_API_KEY when absent (keeps\n                       the key out of `ps`); with neither, the server has no\n                       auth at all, same as before this flag existed\n  --hf-endpoint        Hugging Face mirror endpoint (e.g. https://hf-mirror.com)\n  --paged-ssd-cache-dir tiered KV SSD cache directory (omlx compatibility; accepted, not honoured)\n  --hot-cache-max-size in-memory hot cache size (e.g. 20%) (omlx compatibility; accepted, not honoured)\n  --mcp-config         path to MCP tools configuration file (omlx compatibility; accepted, not honoured)\n  --steering           path to a control vector (.gguf, llama.cpp layout). Applies a\n                       directional edit to the residual stream of EVERY request this\n                       process serves; no weight byte is modified. Repeatable to\n                       apply several vectors in order. See docs/OBLITERATION.md\n  --steering-mode      ablate, add, clamp or renorm (default: each vector file's\n                       declared mode, or ablate). Repeatable, paired positionally\n                       with --steering\n  --steering-scale     strength (default 1.0 when --steering is given; 0.0 is the exact\n                       identity). Repeatable, paired positionally with --steering\n  --steering-layers    START:END, inclusive and 0-based (default every layer each\n                       vector covers). Repeatable, paired positionally with --steering\n  --steering-target    coefficient --steering-mode clamp pins the stream to (default 0)\n  --steering-gate      only steer where the coefficient reaches this magnitude\n                       (default 0, meaning always)\n  --vision-sidecar     path to a standalone vision-tower sidecar install to attach to a\n                       text-only trunk, or auto to resolve one from the installed store\n                       by the trunk's own family and hidden size (default: none; the\n                       trunk's own tower, if any, is used)\n  --kv-bits            TurboQuant KV-cache quantization: off, 2, 3, 3.5, or 4\n                       (default off). An unsupported head_dim or family\n                       REFUSES the flag at open rather than falling back to\n                       FP16. See docs/TRUBOQUANT.md\n  --help               print this text and exit\n  --version            print the version and exit";

pub use crate::bind::BindMode;

/// Add flags introduced after the legacy usage string was frozen. Keeping
/// this small compatibility shim avoids duplicating the long help text while
/// ensuring `--help` advertises every accepted residency mode.
pub fn usage() -> String {
    USAGE.replace(
        "[--expert-cache-slots auto|N]",
        "[--expert-cache-slots auto|N] [--expert-residency auto|streamed|mapped]",
    )
}

/// Parsed command line arguments for running `turbospark-server` in `--model` mode.
#[derive(Debug)]
pub struct ModelArgs {
    /// Every explicitly attached generation install, in command-line order.
    /// `model` remains the first entry for compatibility with existing
    /// callers and diagnostics; multi-install startup uses this complete list.
    pub models: Vec<String>,
    pub model: String,
    /// How many independent runners to open of `--model`'s ONE install
    /// (ROADMAP P3.6). 1 is every pre-flag behaviour; N > 1 opens N
    /// `RealChatModel`s behind one public id, each with its own KV, session
    /// pool and generation gate, so N requests generate concurrently. Each
    /// open pays the load guard on its own, so the members that do not fit
    /// are refused at startup with the subtraction rather than swapped
    /// later.
    pub pool_size: u32,
    pub embedding_model: Option<String>,
    pub model_dir: Option<std::path::PathBuf>,
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
    /// Routed-expert storage policy, resolved against the install and machine
    /// at open. Auto remains streamed when the minimum cache fits.
    pub expert_residency: runtime::ExpertResidency,
    pub bind: BindMode,
    /// ROADMAP Phase P2. Process-level, like every other flag here: there
    /// is one runner per process, so there is nothing per-request to vary.
    pub power_profile: Option<runtime::PowerProfile>,
    /// How much of the machine may be committed, and the floor under an
    /// automatic window. `Default` is `relaxed` with no floor, which is what
    /// this binary did before either flag existed.
    pub load_policy: runtime::LoadPolicy,
    pub max_tokens_per_sec: Option<f64>,
    /// Process-level for the same reason the two above are, and one more:
    /// the drafter's state is allocated at OPEN, so there is nothing a
    /// request could switch. The per-request half is whether the request is
    /// deterministic, which `RealChatModel::run_completion` applies.
    pub speculation: runtime::Speculation,
    pub drafter: runtime::SpeculativeDrafter,
    /// Tool-call guardrails. Process-level for the reason the three above
    /// are, plus one of its own: a per-request field would let any client
    /// opt its own traffic out of the repair this deployment chose.
    /// A directional-steering policy, resolved once at startup
    /// (`docs/OBLITERATION.md`). PROCESS-level like every other flag here,
    /// and for a stronger reason than the drafter's: the direction buffers
    /// are allocated at open, AND the edit changes the tokens, so there is
    /// nothing a request could safely switch mid-flight.
    ///
    /// Unlike speculation this has NO per-request half. Speculation falls
    /// back silently for a sampled request because acceptance is exact only
    /// at temperature 0; steering has no such precondition, so a server
    /// started with it steers every request it serves.
    pub steering: runtime::SteeringPolicy,
    pub guardrails: turbospark_server::GuardrailConfig,
    /// Continue a request from the previous request's KV cache wherever the
    /// prompts agree, instead of re-prefilling the whole transcript.
    pub prefix_reuse: bool,
    /// How many distinct conversations one runner may keep reusable KV/
    /// recurrent state for at once (`crate::session_pool` /
    /// `--session-slots`). Default 1.
    pub session_slots: u32,
    /// Default reasoning effort for requests that do not specify reasoning_effort.
    pub reasoning: tokenizer::ReasoningEffort,
    /// Default system prompt for requests that carry no system or developer
    /// message of their own.
    pub default_system: Option<String>,
    /// Path to a standalone vision-tower sidecar install to attach to a
    /// text-only trunk (vision memory sidecar, Part A4), the literal
    /// `auto` to resolve one from the installed store by the trunk's own
    /// family and hidden size, or `None` to use the trunk's own tower (if
    /// any). An opaque string, like `steering` above: this binary reads no
    /// directory itself, so resolving and attaching it is
    /// `RealChatModel::open`'s job.
    pub vision_sidecar: Option<String>,
    /// TurboQuant KV-cache quantization. Process-level for the reason
    /// `steering` is: the packed quant tables are allocated at OPEN and the
    /// edit changes the KV BYTES a decode reads, so there is nothing a
    /// request could safely switch mid-flight. `Off` (the default)
    /// reproduces every release before this flag existed byte for byte
    /// (`docs/TRUBOQUANT.md`).
    pub kv_bits: runtime::KvQuant,
    /// The `--api-key` flag's OWN value, or `None` if absent. Deliberately
    /// NOT resolved against `$TURBOSPARK_API_KEY` here: this parser reads
    /// only `args`, matching `power_profile`'s split (`None` here,
    /// resolved against the OS at `open_real_model`) so a test asserting
    /// what a given argv parses to is not at the mercy of whatever the test
    /// process's environment happens to carry. `main` applies the env
    /// fallback once, after parsing.
    pub api_key: Option<String>,
    /// Hugging Face mirror endpoint override, if set.
    pub hf_endpoint: Option<String>,
    /// Tiered KV cache SSD directory (omlx compatibility).
    pub paged_ssd_cache_dir: Option<std::path::PathBuf>,
    /// In-memory hot cache size hint (omlx compatibility).
    pub hot_cache_max_size: Option<String>,
    /// MCP configuration path (omlx compatibility).
    pub mcp_config: Option<std::path::PathBuf>,
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
        models: Vec::new(),
        model: String::new(),
        embedding_model: None,
        port: 8080,
        max_context: None,
        expert_cache_slots: None,
        expert_residency: runtime::ExpertResidency::Auto,
        bind: BindMode::Loopback,
        power_profile: None,
        load_policy: runtime::LoadPolicy::default(),
        max_tokens_per_sec: None,
        speculation: runtime::Speculation::Auto,
        drafter: runtime::SpeculativeDrafter::Auto,
        guardrails: turbospark_server::GuardrailConfig::default(),
        prefix_reuse: false,
        session_slots: 1,
        pool_size: 1,
        steering: runtime::SteeringPolicy::off(),
        reasoning: tokenizer::ReasoningEffort::Off,
        default_system: None,
        vision_sidecar: None,
        kv_bits: runtime::KvQuant::Off,
        api_key: None,
        model_dir: None,
        hf_endpoint: None,
        paged_ssd_cache_dir: None,
        hot_cache_max_size: None,
        mcp_config: None,
    };
    // Held aside so the per-vector resolution after the loop sees every path
    // and every knob regardless of flag order: `--steering-layers` may come
    // before or after its `--steering`, and the i-th knob value pairs with
    // the i-th path -- index it, else the LAST supplied value, else the
    // default. `invocation::steering_knob` is that rule's canonical
    // definition; this crate does not depend on that one, so the rule is
    // restated here and pinned by `main_tests`' parse cases.
    let mut steering_paths: Vec<String> = Vec::new();
    let mut steering_layers: Vec<(usize, usize)> = Vec::new();
    // Repeatable, matching `turbospark-check`'s `--system` grammar: repeats
    // join with a newline. Held aside rather than written straight into
    // `parsed` so the `--system-file` conflict check below can tell "flag
    // absent" from "flag given an empty value".
    let mut system_parts: Vec<String> = Vec::new();
    let mut system_file: Option<String> = None;
    // Tracked separately so each FILE's declared mode can win where the flag
    // is absent for ITS index, and the flag where it is present -- the
    // precedence `crates/cli`'s `resolve_steering` applies, stated the same
    // way.
    let mut steering_mode: Vec<foundation::SteeringMode> = Vec::new();
    let mut steering_scale: Vec<f32> = Vec::new();
    // Both of these land DIRECTLY in `parsed.steering` and both are legal at
    // the 0.0 the `off()` default already carries, so the value cannot say
    // whether a caller supplied one. Tracked for the orphan check below.
    let mut steering_target_explicit = false;
    let mut steering_gate_explicit = false;
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        let value = args
            .get(i + 1)
            .ok_or_else(|| format!("{flag} needs a value"))?;
        let number = || value.parse::<u32>().map_err(|e| format!("{flag}: {e}"));
        match flag {
            "--model" => {
                if parsed.model.is_empty() {
                    parsed.model = value.clone();
                }
                parsed.models.push(value.clone());
            }
            "--embedding-model" => parsed.embedding_model = Some(value.clone()),
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
            "--pool-size" => {
                let n = number()?;
                if n == 0 {
                    return Err("--pool-size must be at least 1, not 0".to_string());
                }
                parsed.pool_size = n;
            }
            "--expert-cache-slots" => {
                parsed.expert_cache_slots = if value == "auto" {
                    None
                } else {
                    Some(number()?)
                }
            }
            "--expert-residency" => {
                parsed.expert_residency =
                    runtime::ExpertResidency::parse(value).ok_or_else(|| {
                        format!("--expert-residency must be auto, streamed or mapped, not {value}")
                    })?;
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
            // A tier word or a byte ceiling, one flag, for the reason
            // `crates/invocation`'s arm gives: a separate bytes flag would let
            // a caller name a tier and a ceiling that disagree.
            "--load-guard" => {
                parsed.load_policy.guard = match runtime::LoadGuard::parse(value) {
                    Some(g) => g,
                    None => match value.parse::<u64>() {
                        Ok(n) if n > 0 => runtime::LoadGuard::Custom {
                            max_counted_bytes: n,
                        },
                        _ => {
                            return Err(format!(
                                "--load-guard must be off, relaxed, balanced, strict, \
                                 or a positive byte ceiling, not {value}"
                            ))
                        }
                    },
                };
            }
            "--min-auto-context" => parsed.load_policy.min_auto_context = number()?,
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
            "--steering" => {
                steering_paths.push(value.clone());
            }
            // The accepted set is `SteeringMode::parse`'s and the message has
            // to be spelled from it rather than recalled: this read "ablate,
            // add or clamp" for a release after `renorm` landed, so a caller
            // who misspelled the fourth mode was told there were three.
            "--steering-mode" => {
                steering_mode.push(foundation::SteeringMode::parse(value.as_str()).ok_or_else(
                    || {
                        format!(
                            "--steering-mode must be one of {}, not {value}",
                            foundation::STEERING_MODE_NAMES.join(", ")
                        )
                    },
                )?);
            }
            "--steering-scale" => {
                steering_scale.push(match value.parse::<f32>() {
                    Ok(v) if v.is_finite() => v,
                    _ => {
                        return Err(format!(
                            "--steering-scale must be a finite number, not {value}"
                        ))
                    }
                });
            }
            "--steering-target" => {
                parsed.steering.target = match value.parse::<f32>() {
                    Ok(v) if v.is_finite() => v,
                    _ => {
                        return Err(format!(
                            "--steering-target must be a finite number, not {value}"
                        ))
                    }
                };
                steering_target_explicit = true;
            }
            "--steering-gate" => {
                parsed.steering.gate_threshold = match value.parse::<f32>() {
                    Ok(v) if v.is_finite() && v >= 0.0 => v,
                    _ => {
                        return Err(format!(
                            "--steering-gate must be a finite number >= 0, not {value}"
                        ))
                    }
                };
                steering_gate_explicit = true;
            }
            "--steering-layers" => {
                let bad = || {
                    format!(
                        "--steering-layers must be START:END, inclusive and 0-based, not {value}"
                    )
                };
                let (a, b) = value.split_once(':').ok_or_else(bad)?;
                let start: usize = a.trim().parse().map_err(|_| bad())?;
                let end: usize = b.trim().parse().map_err(|_| bad())?;
                if end < start {
                    return Err(bad());
                }
                steering_layers.push((start, end));
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
            "--guardrails" => {
                parsed.guardrails = match value.as_str() {
                    "on" => turbospark_server::GuardrailConfig::default(),
                    "off" => turbospark_server::GuardrailConfig::OFF,
                    other => return Err(format!("--guardrails must be on or off, not {other}")),
                }
            }
            "--prefix-reuse" => {
                parsed.prefix_reuse = match value.as_str() {
                    "on" => true,
                    "off" => false,
                    other => return Err(format!("--prefix-reuse must be on or off, not {other}")),
                }
            }
            "--session-slots" => {
                let n = number()?;
                if n == 0 {
                    return Err("--session-slots must be at least 1, not 0".to_string());
                }
                parsed.session_slots = n;
            }
            "--reasoning" => {
                parsed.reasoning = tokenizer::ReasoningEffort::parse(value).ok_or_else(|| {
                    format!("--reasoning must be off, low, medium, high or xhigh, not {value}")
                })?;
            }
            "--system" => system_parts.push(value.clone()),
            "--system-file" => {
                if system_file.is_some() {
                    return Err("--system-file may only be given once".to_string());
                }
                system_file = Some(value.clone());
            }
            // The path stays an OPAQUE string, exactly as `--steering`'s
            // does: this binary reads no directory itself, so resolving and
            // attaching it is `RealChatModel::open`'s job.
            "--vision-sidecar" => parsed.vision_sidecar = Some(value.clone()),
            "--kv-bits" => {
                parsed.kv_bits = runtime::KvQuant::parse(value)?;
            }
            "--api-key" => {
                if value.is_empty() {
                    return Err(
                        "--api-key must not be empty; an empty key authenticates every request"
                            .to_string(),
                    );
                }
                parsed.api_key = Some(value.clone());
            }
            "--model-dir" => {
                parsed.model_dir = Some(std::path::PathBuf::from(value));
            }
            "--memory-guard" => {
                let tier = if value == "safe" {
                    "balanced"
                } else {
                    value.as_str()
                };
                parsed.load_policy.guard = match runtime::LoadGuard::parse(tier) {
                    Some(g) => g,
                    None => {
                        return Err(format!(
                            "--memory-guard must be safe, balanced, relaxed, strict, or off, not {value}"
                        ))
                    }
                };
            }
            "--memory-guard-gb" => {
                let gb = value
                    .parse::<u64>()
                    .map_err(|e| format!("--memory-guard-gb: {e}"))?;
                if gb == 0 {
                    return Err("--memory-guard-gb must be greater than 0".to_string());
                }
                let max_counted_bytes = gb.checked_mul(1024 * 1024 * 1024).ok_or_else(|| {
                    format!("--memory-guard-gb {gb} is too large (overflows a byte count)")
                })?;
                parsed.load_policy.guard = runtime::LoadGuard::Custom { max_counted_bytes };
            }
            "--max-concurrent-requests" => {
                let n = number()?;
                if n == 0 {
                    return Err("--max-concurrent-requests must be at least 1, not 0".to_string());
                }
                parsed.session_slots = n;
            }
            // NOT applied here: this parser reads only `args` (the same
            // split `power_profile` and `api_key` document), and
            // `std::env::set_var` is not thread-safe on Unix -- `main_tests.rs`
            // runs this parser from a shared test binary's threads. `main`
            // applies it once, right before opening the model, the one call
            // site that can act on it.
            "--hf-endpoint" => parsed.hf_endpoint = Some(value.clone()),
            "--paged-ssd-cache-dir" => {
                parsed.paged_ssd_cache_dir = Some(std::path::PathBuf::from(value));
            }
            "--hot-cache-max-size" => {
                parsed.hot_cache_max_size = Some(value.clone());
            }
            "--mcp-config" => {
                parsed.mcp_config = Some(std::path::PathBuf::from(value));
            }
            other => return Err(format!("unknown option {other}\n{USAGE}")),
        }
        i += 2;
    }
    if parsed.models.is_empty() {
        if let Some(ref dir) = parsed.model_dir {
            if dir.join("manifest.json").exists() {
                parsed.model = dir.display().to_string();
                parsed.models.push(parsed.model.clone());
            } else if dir.is_dir() {
                if let Ok(entries) = std::fs::read_dir(dir) {
                    let mut found = Vec::new();
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.is_dir()
                            && (path.extension().is_some_and(|e| e == "gturbo")
                                || path.join("manifest.json").exists())
                        {
                            found.push(path);
                        }
                    }
                    found.sort();
                    if let Some(first) = found.into_iter().next() {
                        parsed.model = first.display().to_string();
                        parsed.models.push(parsed.model.clone());
                    }
                }
            }
            if parsed.model.is_empty() && parsed.embedding_model.is_none() {
                return Err(format!(
                    "no .gturbo model found in --model-dir {}\n{USAGE}",
                    dir.display()
                ));
            }
        } else if parsed.embedding_model.is_none() {
            // `--model` alone is required, UNLESS `--embedding-model` was
            // given instead: `main.rs::open_models_registry`'s
            // `(false, Some(emb_arg))` arm serves an embedding-only server
            // with no chat model at all, and this check used to refuse that
            // request before it ever reached there -- the arm existed and
            // was unreachable through this parser.
            return Err(format!(
                "--model or --embedding-model needs a value\n{USAGE}"
            ));
        }
    } else if let Some(ref dir) = parsed.model_dir {
        for model in &mut parsed.models {
            let direct = dir.join(&*model);
            let with_ext = dir.join(format!("{}.gturbo", model));
            if direct.exists() {
                *model = direct.display().to_string();
            } else if with_ext.exists() {
                *model = with_ext.display().to_string();
            }
        }
        parsed.model = parsed.models[0].clone();
    }
    if parsed.models.len() > 1 && parsed.pool_size > 1 {
        return Err(
            "--pool-size cannot be combined with multiple --model flags; repeat --model for distinct installs, or use --pool-size for one install"
                .to_string(),
        );
    }
    // NAMING BOTH IS REFUSED RATHER THAN RESOLVED. They are two spellings of
    // one setting, so a command line carrying both says two things, and
    // picking either silently is the shape this parser already refuses for an
    // orphan steering parameter below.
    if !system_parts.is_empty() && system_file.is_some() {
        return Err(
            "--system and --system-file are two spellings of one setting; give one".to_string(),
        );
    }
    if let Some(path) = &system_file {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("--system-file {path}: {e}"))?;
        // An EMPTY file is refused rather than treated as "no default". A
        // caller who named a file meant to set a prompt, and silently serving
        // every request unprompted is the failure they would not notice.
        if text.trim().is_empty() {
            return Err(format!("--system-file {path} is empty"));
        }
        parsed.default_system = Some(text.trim_end().to_string());
    } else if !system_parts.is_empty() {
        let text = system_parts.join("\n");
        if text.trim().is_empty() {
            return Err("--system must not be empty or whitespace".to_string());
        }
        parsed.default_system = Some(text);
    }
    // A pool with nothing to reuse a parked session FOR is not a smaller
    // version of the feature, it is a memory commitment that does nothing:
    // `1` is the only value reachable without naming this flag, so `> 1`
    // here always means the caller asked for a pool explicitly.
    if parsed.session_slots > 1 && !parsed.prefix_reuse {
        return Err(
            "--session-slots needs --prefix-reuse on; without prefix reuse \
             there is nothing for a parked session to be reused for"
                .to_string(),
        );
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
    // Resolved AFTER the loop so flag order does not matter, one vector per
    // `--steering` path with the knobs at their own indexes (last value
    // extends -- see the rule comment at the top of the parse loop). The
    // flag wins over the file's declared mode, the file's over the default,
    // and a vector with no scale means full strength rather than the zero
    // the `off()` default carries.
    if !steering_paths.is_empty() {
        // MORE KNOB VALUES THAN PATHS is refused, not ignored: with the
        // positional pairing, an extra value can never reach a vector, so it
        // is a typo the command line carries. (A SHORTER list is legal and
        // extends by its last value.) First offender in a fixed order,
        // matching `crates/invocation`'s arm.
        let surplus = if steering_mode.len() > steering_paths.len() {
            Some("--steering-mode")
        } else if steering_scale.len() > steering_paths.len() {
            Some("--steering-scale")
        } else if steering_layers.len() > steering_paths.len() {
            Some("--steering-layers")
        } else {
            None
        };
        if let Some(flag) = surplus {
            return Err(format!(
                "more {flag} values than --steering paths; the extras can never \
                 reach a vector"
            ));
        }
        let mut vectors = Vec::with_capacity(steering_paths.len());
        for (i, path) in steering_paths.iter().enumerate() {
            let mut set = repack::control_vector::load_control_vector(std::path::Path::new(path))
                .map_err(|e| format!("--steering {path}: {e}"))?;
            let range = steering_layers
                .get(i)
                .or_else(|| steering_layers.last())
                .copied();
            if let Some((start, end)) = range {
                set.restrict_to_range(start, end);
            }
            // The flag wins over the file's declared mode, the file's over
            // the default. Read the declared mode before `set` moves into
            // the vector.
            let declared = set.declared_mode;
            vectors.push(runtime::SteeringVector {
                set,
                mode: steering_mode
                    .get(i)
                    .or_else(|| steering_mode.last())
                    .copied()
                    .or(declared)
                    .unwrap_or_default(),
                alpha: steering_scale
                    .get(i)
                    .or_else(|| steering_scale.last())
                    .copied()
                    .unwrap_or(1.0),
            });
        }
        parsed.steering.vectors = vectors;
    } else {
        // A STEERING PARAMETER WITHOUT `--steering` IS REFUSED, NOT IGNORED.
        // Without a direction set the process serves every request unsteered,
        // so a parameter here is not a weaker request for the edit -- it is a
        // command line that says one thing while the server does another, on
        // an axis that changes the TOKENS. Every neighbouring silent-no-op is
        // already closed (an empty layer range, a family that does not
        // dispatch the edit, a direction file that will not parse); this was
        // the one left open, and it is worse on a server than on the CLI
        // because nobody watches a daemon start.
        //
        // First offender in a fixed order, matching `crates/invocation`'s
        // arm, which is the same check on the same six flags.
        let orphan = if !steering_mode.is_empty() {
            Some("--steering-mode")
        } else if !steering_scale.is_empty() {
            Some("--steering-scale")
        } else if !steering_layers.is_empty() {
            Some("--steering-layers")
        } else if steering_target_explicit {
            Some("--steering-target")
        } else if steering_gate_explicit {
            Some("--steering-gate")
        } else {
            None
        };
        if let Some(flag) = orphan {
            return Err(format!(
                "{flag} needs --steering; without a direction set this server would decode \
                 unsteered and the flag would do nothing"
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
        "--help" | "-h" => Some(format!("{}\n", usage())),
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
