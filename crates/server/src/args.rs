/// Command line usage and flag description text for `turbospark-server`.
pub const USAGE: &str = "usage: turbospark-server --model <install-dir|alias> [--port N] [--max-context N|auto] [--expert-cache-slots auto|N] [--bind loopback|tailnet] [--power-profile performance|balanced|efficiency] [--max-tokens-per-sec R] [--speculative off|auto|N] [--speculative-drafter auto|mtp|dflash] [--guardrails on|off] [--steering PATH] [--steering-mode ablate|add|clamp|renorm] [--steering-scale F] [--steering-layers S:E] [--steering-target F] [--steering-gate F]\n       turbospark-server <tokenizer-dir> [port]\n       turbospark-server --help | --version\n\noptions:\n  --model              a .gturbo directory or a turbospark-model alias (`turbospark-model list`)\n  --port               listen port (default 8080)\n  --max-context        context window in tokens, or auto (default auto: the\n                       checkpoint's trained context, capped by what memory\n                       holds, and 4096 when the install declares none)\n  --expert-cache-slots routed-cache slots per layer: auto or 8/16/24/32 (default auto)\n  --bind               loopback or tailnet (default loopback; tailnet is NOT auth)\n  --power-profile      performance, balanced or efficiency\n  --max-tokens-per-sec decode rate cap, greater than 0\n  --speculative        off, auto, or a block size 1-15 (default auto). Speculation\n                       applies to temperature-0 requests only; others decode\n                       sequentially\n  --speculative-drafter auto, mtp or dflash (default auto; auto reports a DFlash2\n                       drafter but does not enable it -- see docs/DFLASH2.md)\n  --guardrails         on or off (default on). Rescues a tool call the decoder\n                       could not parse, checks arguments against the request's\n                       own schema, and re-asks once. A request carrying TOOLS is\n                       buffered rather than streamed while this is on, because a\n                       verdict needs the whole turn; requests without tools are\n                       unaffected\n  --steering           path to a control vector (.gguf, llama.cpp layout). Applies a\n                       directional edit to the residual stream of EVERY request this\n                       process serves; no weight byte is modified. See\n                       docs/OBLITERATION.md\n  --steering-mode      ablate, add, clamp or renorm (default: the vector file's declared mode,\n                       or ablate)\n  --steering-scale     strength (default 1.0 when --steering is given; 0.0 is the exact\n                       identity)\n  --steering-layers    START:END, inclusive and 0-based (default every layer the\n                       vector covers)\n  --steering-target    coefficient --steering-mode clamp pins the stream to (default 0)\n  --steering-gate      only steer where the coefficient reaches this magnitude\n                       (default 0, meaning always)\n  --help               print this text and exit\n  --version            print the version and exit";

pub use crate::bind::BindMode;

/// Parsed command line arguments for running `turbospark-server` in `--model` mode.
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
        guardrails: turbospark_server::GuardrailConfig::default(),
        steering: runtime::SteeringPolicy::off(),
    };
    // Held aside because `--steering-layers` may be given BEFORE or AFTER
    // `--steering`, and the restriction has to survive either order: the
    // range is applied when the set arrives and again here if it already has.
    let mut steering_layers: Option<(usize, usize)> = None;
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
