/// Command line usage and flag description text for `turbospark-server`.
pub const USAGE: &str = "usage: turbospark-server [--model <install-dir|alias>] [--embedding-model <install-dir|alias>] [--model-dir PATH] [--port N] [--max-context N|auto] [--load-guard TIER|BYTES] [--memory-guard TIER] [--memory-guard-gb N] [--min-auto-context N] [--expert-cache-slots auto|N] [--bind loopback|tailnet] [--power-profile performance|balanced|efficiency] [--max-tokens-per-sec R] [--speculative off|auto|N] [--speculative-drafter auto|mtp|dflash] [--guardrails on|off] [--prefix-reuse on|off] [--session-slots N] [--max-concurrent-requests N] [--reasoning off|low|medium|high|xhigh] [--system TEXT] [--system-file PATH] [--api-key KEY] [--hf-endpoint URL] [--paged-ssd-cache-dir PATH] [--hot-cache-max-size SIZE] [--mcp-config PATH] [--steering PATH] [--steering-mode ablate|add|clamp|renorm] [--steering-scale F] [--steering-layers S:E] [--steering-target F] [--steering-gate F]\n       turbospark-server <tokenizer-dir> [port]\n       turbospark-server --help | --version\n\noptions:\n  --model              a .gturbo directory or a turbospark-model alias (`turbospark-model list`)\n  --embedding-model    path to an embedding model (.safetensors directory) or alias\n                       to serve for /v1/embeddings alongside generation\n  --model-dir          directory containing .gturbo models (omlx compatibility)\n  --port               listen port (default 8080)\n  --max-context        context window in tokens, or auto (default auto: the\n                       checkpoint's trained context, capped by what memory\n                       holds, and 4096 when the install declares none)\n  --load-guard         how much of the machine a session may commit: off,\n                       relaxed (default), balanced, strict, or a byte ceiling on\n                       what the engine ALLOCATES. relaxed is what shipped before\n                       this flag and what every published memory figure was\n                       measured under; see docs/LOAD_GUARD.md\n  --memory-guard       alias for --load-guard: safe (balanced), balanced, strict,\n                       relaxed, off (omlx compatibility)\n  --memory-guard-gb    set custom memory guard ceiling in gigabytes (omlx compatibility)\n  --min-auto-context   refuse to open when --max-context auto resolves below this\n                       many tokens (default 0, no floor). Says nothing about an\n                       explicit --max-context\n  --expert-cache-slots routed-cache slots per layer: auto or 8/16/24/32/48/64/96/128 (default auto)\n  --bind               loopback or tailnet (default loopback; tailnet is NOT auth)\n  --power-profile      performance, balanced or efficiency\n  --max-tokens-per-sec decode rate cap, greater than 0\n  --speculative        off, auto, or a block size 1-15 (default auto). Speculation\n                       applies to temperature-0 requests only; others decode\n                       sequentially\n  --speculative-drafter auto, mtp or dflash (default auto; auto reports a DFlash2\n                       drafter but does not enable it -- see docs/DFLASH2.md)\n  --guardrails         on or off (default on). Rescues a tool call the decoder\n                       could not parse, checks arguments against the request's\n                       own schema, and re-asks once. A request carrying TOOLS is\n                       buffered rather than streamed while this is on, because a\n                       verdict needs the whole turn; requests without tools are\n                       unaffected\n  --prefix-reuse       on or off (default on). A request continues from the\n                       previous request's KV cache wherever the prompts agree,\n                       instead of re-prefilling the whole transcript. Helps\n                       only when consecutive requests are the same\n                       conversation -- unrelated interleaved requests each\n                       discard the other's reusable prefix -- and raises the\n                       idle-memory floor between requests, not the peak, since\n                       pages that would normally be released stay resident.\n                       See crates/runtime/CLAUDE.md Gotcha 30\n  --session-slots      how many DISTINCT conversations this runner may keep\n                       reusable KV/recurrent state for at once (default 1, i.e.\n                       no pool). Real committed memory per extra slot, unlike\n                       --prefix-reuse's floor-only cost; needs --prefix-reuse on\n                       (the default), since a parked session is never reused\n                       without it. See crates/server/CLAUDE.md's --session-slots\n                       Gotcha\n  --max-concurrent-requests alias for --session-slots (omlx compatibility)\n  --reasoning          default reasoning effort for requests that do not specify\n                       reasoning_effort: off, low, medium, high or xhigh\n                       (default off)\n  --system             default system prompt for requests that carry no system\n                       or developer message of their own. Repeatable; repeats\n                       join with a newline. A request that sends its own system\n                       message is left exactly as it arrived\n  --system-file        read the same default system prompt from a file, for a\n                       prompt too long to sit on a command line. Mutually\n                       exclusive with --system\n  --api-key            require this key on every request except GET /health,\n                       as `Authorization: Bearer <key>` or `x-api-key: <key>`.\n                       Falls back to $TURBOSPARK_API_KEY when absent (keeps\n                       the key out of `ps`); with neither, the server has no\n                       auth at all, same as before this flag existed\n  --hf-endpoint        Hugging Face mirror endpoint (e.g. https://hf-mirror.com)\n  --paged-ssd-cache-dir tiered KV SSD cache directory (omlx compatibility)\n  --hot-cache-max-size in-memory hot cache size (e.g. 20%) (omlx compatibility)\n  --mcp-config         path to MCP tools configuration file (omlx compatibility)\n  --steering           path to a control vector (.gguf, llama.cpp layout). Applies a\n                       directional edit to the residual stream of EVERY request this\n                       process serves; no weight byte is modified. See\n                       docs/OBLITERATION.md\n  --steering-mode      ablate, add, clamp or renorm (default: the vector file's declared mode,\n                       or ablate)\n  --steering-scale     strength (default 1.0 when --steering is given; 0.0 is the exact\n                       identity)\n  --steering-layers    START:END, inclusive and 0-based (default every layer the\n                       vector covers)\n  --steering-target    coefficient --steering-mode clamp pins the stream to (default 0)\n  --steering-gate      only steer where the coefficient reaches this magnitude\n                       (default 0, meaning always)\n  --help               print this text and exit\n  --version            print the version and exit";

pub use crate::bind::BindMode;

/// Parsed command line arguments for running `turbospark-server` in `--model` mode.
#[derive(Debug)]
pub struct ModelArgs {
    pub model: String,
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
    /// The `--api-key` flag's OWN value, or `None` if absent.
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
        model: String::new(),
        embedding_model: None,
        port: 8080,
        max_context: None,
        expert_cache_slots: None,
        bind: BindMode::Loopback,
        power_profile: None,
        load_policy: runtime::LoadPolicy::default(),
        max_tokens_per_sec: None,
        speculation: runtime::Speculation::Auto,
        drafter: runtime::SpeculativeDrafter::Auto,
        guardrails: turbospark_server::GuardrailConfig::default(),
        prefix_reuse: true,
        session_slots: 1,
        steering: runtime::SteeringPolicy::off(),
        reasoning: tokenizer::ReasoningEffort::Off,
        default_system: None,
        api_key: None,
        model_dir: None,
        hf_endpoint: None,
        paged_ssd_cache_dir: None,
        hot_cache_max_size: None,
        mcp_config: None,
    };
    // Held aside because `--steering-layers` may be given BEFORE or AFTER
    // `--steering`, and the restriction has to survive either order: the
    // range is applied when the set arrives and again here if it already has.
    let mut steering_layers: Option<(usize, usize)> = None;
    // Repeatable, matching `turbospark-check`'s `--system` grammar: repeats
    // join with a newline. Held aside rather than written straight into
    // `parsed` so the `--system-file` conflict check below can tell "flag
    // absent" from "flag given an empty value".
    let mut system_parts: Vec<String> = Vec::new();
    let mut system_file: Option<String> = None;
    // Tracked separately so the FILE's declared mode can win where the flag
    // is absent, and the flag where it is present -- the precedence
    // `crates/cli`'s `resolve_steering` applies, stated the same way.
    let mut steering_mode: Option<foundation::SteeringMode> = None;
    let mut steering_scale: Option<f32> = None;
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
            "--model" => parsed.model = value.clone(),
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
                let mut set = repack::control_vector::load_control_vector(std::path::Path::new(
                    value.as_str(),
                ))
                .map_err(|e| format!("--steering {value}: {e}"))?;
                if let Some((start, end)) = steering_layers {
                    set.restrict_to_range(start, end);
                }
                parsed.steering.set = Some(set);
            }
            // The accepted set is `SteeringMode::parse`'s and the message has
            // to be spelled from it rather than recalled: this read "ablate,
            // add or clamp" for a release after `renorm` landed, so a caller
            // who misspelled the fourth mode was told there were three.
            "--steering-mode" => {
                steering_mode = Some(foundation::SteeringMode::parse(value.as_str()).ok_or_else(
                    || {
                        format!(
                            "--steering-mode must be one of {}, not {value}",
                            foundation::STEERING_MODE_NAMES.join(", ")
                        )
                    },
                )?);
            }
            "--steering-scale" => {
                steering_scale = Some(match value.parse::<f32>() {
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
                steering_layers = Some((start, end));
                if let Some(set) = parsed.steering.set.as_mut() {
                    set.restrict_to_range(start, end);
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
                parsed.load_policy.guard = runtime::LoadGuard::Custom {
                    max_counted_bytes: gb * 1024 * 1024 * 1024,
                };
            }
            "--max-concurrent-requests" => {
                let n = number()?;
                if n == 0 {
                    return Err("--max-concurrent-requests must be at least 1, not 0".to_string());
                }
                parsed.session_slots = n;
            }
            "--hf-endpoint" => {
                std::env::set_var("HF_ENDPOINT", value);
                parsed.hf_endpoint = Some(value.clone());
            }
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
    if parsed.model.is_empty() {
        if let Some(ref dir) = parsed.model_dir {
            if dir.join("manifest.json").exists() {
                parsed.model = dir.display().to_string();
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
                    }
                }
            }
            if parsed.model.is_empty() {
                return Err(format!(
                    "no .gturbo model found in --model-dir {}\n{USAGE}",
                    dir.display()
                ));
            }
        } else {
            return Err(format!("--model needs a value\n{USAGE}"));
        }
    } else if let Some(ref dir) = parsed.model_dir {
        let direct = dir.join(&parsed.model);
        let with_ext = dir.join(format!("{}.gturbo", parsed.model));
        if direct.exists() {
            parsed.model = direct.display().to_string();
        } else if with_ext.exists() {
            parsed.model = with_ext.display().to_string();
        }
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
            "--session-slots needs --prefix-reuse on (the default); without prefix reuse \
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
    // Resolved AFTER the loop so flag order does not matter: the flag wins
    // over the file's declared mode, the file's over the default, and a set
    // present with no scale means full strength rather than the zero the
    // `off()` default carries.
    if parsed.steering.set.is_some() {
        parsed.steering.mode = steering_mode
            .or_else(|| {
                parsed
                    .steering
                    .set
                    .as_ref()
                    .and_then(|set| set.declared_mode)
            })
            .unwrap_or_default();
        parsed.steering.alpha = steering_scale.unwrap_or(1.0);
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
        let orphan = if steering_mode.is_some() {
            Some("--steering-mode")
        } else if steering_scale.is_some() {
            Some("--steering-scale")
        } else if steering_layers.is_some() {
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
