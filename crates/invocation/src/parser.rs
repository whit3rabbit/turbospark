//! The left-to-right token scan and the fixed-order whole-invocation checks
//! that turn a token list into exactly one outcome.
//!
//! Parsing is a pure function of the supplied token list: no filesystem,
//! environment, or other I/O is performed, and the same token list always
//! produces the same outcome.

use crate::failure::ParseFailure;
use crate::options::OPTIONS;
use crate::request::{
    ExpertCacheSlots, InvocationRequest, KvBits, LoadGuard, MaxContext, Mode, PowerProfile,
    PrefillChunk, ReadAheadMode, ReasoningEffort, Speculation, SpeculativeDrafter, SteeringMode,
    ALLOWED_SPECULATION_BLOCKS, DEFAULT_MAX_NEW, DEFAULT_REPETITION_PENALTY, DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K, DEFAULT_TOP_P, MAX_TOP_K,
};
use foundation::runtime_config::{ALLOWED_CACHE_SLOTS, ALLOWED_CHUNK_SIZES};

/// One of the three possible outcomes of parsing a token list. There is no
/// fourth outcome and no partially populated result.
#[derive(Debug, Clone, PartialEq)]
// `Success` carries the whole request and the other three carry nothing, so
// this is inherently lopsided; `--steering`'s six fields pushed it past
// clippy's threshold. Boxing the payload would change a type every front end
// matches on, to save a move that happens once per process.
#[allow(clippy::large_enum_variant)]
pub enum ParseOutcome {
    /// A fully populated, validated invocation request.
    Success(InvocationRequest),
    /// A help short-circuit. Carries no invocation value.
    Help,
    /// A version short-circuit. Carries no invocation value.
    ///
    /// A sibling of [`Self::Help`] rather than a variant of it: both exit 0
    /// on stdout, but a caller printing usage where a version was asked for
    /// is a different wrong answer than either.
    Version,
    /// One of six distinguishable typed parsing failures.
    Failure(ParseFailure),
}

/// Parse an ordered list of command-line tokens into exactly one outcome.
///
/// Tokens are scanned strictly left to right. A value-taking option always
/// consumes the very next token as its value, whatever that token's text
/// looks like. A help token short-circuits the instant it is reached, ahead
/// of every whole-invocation check; the whole-invocation checks themselves
/// run only after a full scan completes with no per-token failure and no
/// help request, in one fixed order, and only the first failing check is
/// reported.
pub fn parse(tokens: &[String]) -> ParseOutcome {
    let mut model: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut messages_file: Option<String> = None;
    let mut chat = false;
    let mut system_parts: Vec<String> = Vec::new();
    let mut max_new = DEFAULT_MAX_NEW;
    let mut max_context = MaxContext::default();
    let mut load_guard = LoadGuard::default();
    let mut min_auto_context = 0u32;
    let mut temperature = DEFAULT_TEMPERATURE;
    let mut top_k = DEFAULT_TOP_K;
    let mut top_p = DEFAULT_TOP_P;
    let mut top_p_explicit = false;
    let mut repetition_penalty = DEFAULT_REPETITION_PENALTY;
    let mut seed: Option<u64> = None;
    let mut stop: Vec<String> = Vec::new();
    let mut images: Vec<String> = Vec::new();
    let mut image_batch = false;
    let mut rdadvise = ReadAheadMode::default();
    let mut expert_cache_slots = ExpertCacheSlots::default();
    let mut speculation = Speculation::default();
    let mut speculative_drafter = SpeculativeDrafter::default();
    // Repeatable and ORDER-PRESERVING: the i-th occurrence of each knob
    // configures the i-th `--steering` path (`steering_knob` is the rule's
    // one definition). `--steering` itself is the switch; the knobs are
    // collected alongside it and the whole-invocation checks below refuse
    // both orphans (a knob with no path) and surplus (more knob values than
    // paths).
    let mut steering: Vec<String> = Vec::new();
    let mut steering_mode: Vec<SteeringMode> = Vec::new();
    let mut steering_scale: Vec<f32> = Vec::new();
    let mut steering_layers: Vec<(u32, u32)> = Vec::new();
    let mut steering_target: f32 = 0.0;
    let mut steering_gate: f32 = 0.0;
    // Tracked because both default to 0.0 and both are legal AT 0.0, so the
    // value cannot say whether a caller supplied it. Same shape as
    // `top_p_explicit` below, and needed for the same kind of whole-invocation
    // check.
    let mut steering_target_explicit = false;
    let mut steering_gate_explicit = false;
    let mut vision_sidecar: Option<String> = None;
    let mut prefill_chunk = PrefillChunk::default();
    let mut power_profile: Option<PowerProfile> = None;
    let mut max_tokens_per_sec: Option<f64> = None;
    let mut reasoning = ReasoningEffort::default();
    let mut kv_bits = KvBits::default();
    let mut quiet = false;

    let mut i = 0;
    while i < tokens.len() {
        let token = tokens[i].as_str();

        // Honored at the position its token is reached: nothing scanned
        // after this point is ever inspected, and no whole-invocation check
        // runs at all.
        if token == "--help" {
            return ParseOutcome::Help;
        }
        // Same rule as `--help`, one line down so help wins when both are
        // present: whichever is reached FIRST short-circuits, which is the
        // documented left-to-right scan rather than a precedence.
        if token == "--version" {
            return ParseOutcome::Version;
        }

        let Some(decl) = OPTIONS.iter().find(|o| o.flag == token) else {
            return ParseOutcome::Failure(ParseFailure::UnknownOption {
                token: token.to_string(),
            });
        };

        if !decl.takes_value {
            match token {
                "--chat" => chat = true,
                "--quiet" => quiet = true,
                "--image-batch" => image_batch = true,
                other => unreachable!("no-value option {other} not handled"),
            }
            i += 1;
            continue;
        }

        let Some(value) = tokens.get(i + 1) else {
            return ParseOutcome::Failure(ParseFailure::MissingValue { option: decl.flag });
        };

        match token {
            "--model" => model = Some(value.clone()),
            "--prompt" => prompt = Some(value.clone()),
            "--messages-file" => messages_file = Some(value.clone()),
            "--system" => system_parts.push(value.clone()),
            "--max-new" => match value.parse::<u32>() {
                Ok(n) if n > 0 => max_new = n,
                _ => return invalid("--max-new", value),
            },
            "--max-context" => {
                if value == "auto" {
                    max_context = MaxContext::Auto;
                } else {
                    match value.parse::<u32>() {
                        Ok(n) if n > 0 => max_context = MaxContext::Fixed(n),
                        _ => return invalid("--max-context", value),
                    }
                }
            }
            // A bare integer is `Custom`'s byte ceiling. The tier words and
            // a number are the same flag because they answer one question --
            // how much may be committed -- and a separate `--load-guard-bytes`
            // would let a caller name a tier and a ceiling that disagree.
            "--load-guard" => match LoadGuard::parse(value) {
                Some(g) => load_guard = g,
                None => match value.parse::<u64>() {
                    Ok(n) if n > 0 => load_guard = LoadGuard::Custom(n),
                    _ => return invalid("--load-guard", value),
                },
            },
            "--min-auto-context" => match value.parse::<u32>() {
                Ok(n) => min_auto_context = n,
                _ => return invalid("--min-auto-context", value),
            },
            "--temperature" => match value.parse::<f64>() {
                Ok(t) if t.is_finite() && t >= 0.0 => temperature = t,
                _ => return invalid("--temperature", value),
            },
            "--top-k" => match value.parse::<u32>() {
                Ok(k) if k <= MAX_TOP_K => top_k = k,
                _ => return invalid("--top-k", value),
            },
            "--top-p" => match value.parse::<f64>() {
                Ok(p) if p.is_finite() && p > 0.0 && p <= 1.0 => {
                    top_p = p;
                    top_p_explicit = true;
                }
                _ => return invalid("--top-p", value),
            },
            "--repetition-penalty" => match value.parse::<f64>() {
                Ok(r) if r.is_finite() && r > 0.0 => repetition_penalty = r,
                _ => return invalid("--repetition-penalty", value),
            },
            "--seed" => match value.parse::<u64>() {
                Ok(s) => seed = Some(s),
                Err(_) => return invalid("--seed", value),
            },
            "--stop" => stop.push(value.clone()),
            // Repeatable and ORDER-PRESERVING, like `--stop` above it: the
            // order is what pairs each path with its marker run in the
            // rendered prompt, so a set would be the wrong container.
            "--image" => images.push(value.clone()),
            "--rdadvise" => match ReadAheadMode::parse(value) {
                Some(mode) => rdadvise = mode,
                None => return invalid("--rdadvise", value),
            },
            "--expert-cache-slots" => {
                if value == "auto" {
                    expert_cache_slots = ExpertCacheSlots::Auto;
                } else {
                    match value.parse::<u32>() {
                        Ok(n) if ALLOWED_CACHE_SLOTS.contains(&n) => {
                            expert_cache_slots = ExpertCacheSlots::Fixed(n);
                        }
                        _ => return invalid("--expert-cache-slots", value),
                    }
                }
            }
            "--speculative" => match value.as_str() {
                "auto" => speculation = Speculation::Auto,
                "off" => speculation = Speculation::Off,
                _ => match value.parse::<u32>() {
                    Ok(n) if ALLOWED_SPECULATION_BLOCKS.contains(&n) => {
                        speculation = Speculation::Block(n)
                    }
                    _ => return invalid("--speculative", value),
                },
            },
            "--speculative-drafter" => match value.as_str() {
                "auto" => speculative_drafter = SpeculativeDrafter::Auto,
                "mtp" => speculative_drafter = SpeculativeDrafter::Mtp,
                "dflash" => speculative_drafter = SpeculativeDrafter::Dflash,
                _ => return invalid("--speculative-drafter", value),
            },
            "--prefill-chunk" => {
                if value == "auto" {
                    prefill_chunk = PrefillChunk::Auto;
                } else {
                    match value.parse::<u32>() {
                        Ok(n) if ALLOWED_CHUNK_SIZES.contains(&n) => {
                            prefill_chunk = PrefillChunk::Fixed(n)
                        }
                        _ => return invalid("--prefill-chunk", value),
                    }
                }
            }
            "--steering" => steering.push(value.to_string()),
            "--steering-mode" => match SteeringMode::parse(value) {
                Some(m) => steering_mode.push(m),
                None => return invalid("--steering-mode", value),
            },
            // Rejected rather than clamped, and NON-FINITE is rejected too: a
            // NaN alpha puts a NaN into the residual stream, and NaN reads as
            // a PERFECT score on every rank instrument downstream.
            "--steering-scale" => match value.parse::<f32>() {
                Ok(v) if v.is_finite() => steering_scale.push(v),
                _ => return invalid("--steering-scale", value),
            },
            "--steering-target" => match value.parse::<f32>() {
                Ok(v) if v.is_finite() => {
                    steering_target = v;
                    steering_target_explicit = true;
                }
                _ => return invalid("--steering-target", value),
            },
            "--steering-gate" => match value.parse::<f32>() {
                Ok(v) if v.is_finite() && v >= 0.0 => {
                    steering_gate = v;
                    steering_gate_explicit = true;
                }
                _ => return invalid("--steering-gate", value),
            },
            // START:END, inclusive and 0-based. An inverted range is refused
            // here rather than silently steering nothing: a run that asked to
            // steer and quietly did not would measure the unsteered engine.
            "--steering-layers" => match parse_layer_range(value) {
                Some(r) => steering_layers.push(r),
                None => return invalid("--steering-layers", value),
            },
            "--power-profile" => match PowerProfile::parse(value) {
                Some(profile) => power_profile = Some(profile),
                None => return invalid("--power-profile", value),
            },
            "--max-tokens-per-sec" => match value.parse::<f64>() {
                Ok(r) if r.is_finite() && r > 0.0 => max_tokens_per_sec = Some(r),
                _ => return invalid("--max-tokens-per-sec", value),
            },
            "--reasoning" => match ReasoningEffort::parse(value) {
                Some(level) => reasoning = level,
                None => return invalid("--reasoning", value),
            },
            // The path stays an OPAQUE string, like `--steering`'s: this
            // crate is pure and reads no directory, so resolving and
            // attaching it is the front end's job. `"auto"` is read as a
            // literal path rather than a keyword -- catalog-based
            // resolution is a later part, not yet built here.
            "--vision-sidecar" => vision_sidecar = Some(value.to_string()),
            "--kv-bits" => match KvBits::parse(value) {
                Some(bits) => kv_bits = bits,
                None => return invalid("--kv-bits", value),
            },
            other => unreachable!("value-taking option {other} not handled"),
        }

        i += 2;
    }

    // Whole-invocation checks, in fixed order. Only the first failure found
    // is reported.
    let Some(model) = model else {
        return ParseOutcome::Failure(ParseFailure::MissingRequired { option: "--model" });
    };

    if prompt.is_some() && messages_file.is_some() {
        return ParseOutcome::Failure(ParseFailure::MutuallyExclusive {
            first: "--prompt",
            second: "--messages-file",
        });
    }
    if prompt.is_some() && chat {
        return ParseOutcome::Failure(ParseFailure::MutuallyExclusive {
            first: "--prompt",
            second: "--chat",
        });
    }
    if messages_file.is_some() && chat {
        return ParseOutcome::Failure(ParseFailure::MutuallyExclusive {
            first: "--messages-file",
            second: "--chat",
        });
    }

    let mode = if let Some(p) = prompt {
        Mode::Prompt(p)
    } else if let Some(m) = messages_file {
        Mode::MessagesFile(m)
    } else if chat {
        Mode::Chat
    } else {
        return ParseOutcome::Failure(ParseFailure::NoModeSelected);
    };

    let system = if system_parts.is_empty() {
        None
    } else {
        // Repeated leading-message occurrences fold into one combined
        // value, preserving supplied order. The joiner is destination
        // selected and is not contractual.
        Some(system_parts.join("\n"))
    };

    if system.is_some() && !matches!(mode, Mode::Chat) {
        return ParseOutcome::Failure(ParseFailure::InvalidValue {
            option: "--system",
            value: "system message requires interactive chat mode".to_string(),
        });
    }

    // Only fires when the caller explicitly supplied a sub-one threshold;
    // the documented default threshold paired with rank-based shaping
    // turned off is not itself a violation.
    if top_p_explicit && top_p < 1.0 && top_k == 0 {
        return ParseOutcome::Failure(ParseFailure::InvalidValue {
            option: "--top-p",
            value: "cumulative-probability truncation below 1.0 requires rank-based truncation to be enabled"
                .to_string(),
        });
    }

    // A STEERING PARAMETER WITHOUT `--steering` IS REFUSED, NOT IGNORED, and
    // it is the same argument the inverted layer range is refused on: the run
    // would decode unsteered while the command line says otherwise, so a
    // caller measuring an edit would measure the engine without one. Every
    // other silent-no-op on this path is already closed -- an empty layer
    // range, a family that does not dispatch the edit, a direction file that
    // will not parse -- and this was the one left open, because five of the
    // six flags mean nothing on their own and the sixth is what turns the
    // feature on.
    //
    // MORE KNOB VALUES THAN PATHS is refused on the same argument, extended:
    // with the positional pairing rule, a third `--steering-scale` against
    // two paths can never reach a vector, so it is a typo the command line
    // carries rather than a configuration. (A SHORTER knob list is legal and
    // extends by its last value -- `steering_knob`'s documented rule.)
    //
    // Reported against the first offending flag in a FIXED order rather than
    // all of them, matching the mutually-exclusive checks above: a caller
    // fixes one flag per run either way, and one name keeps the payload a
    // sentence.
    let orphan = if steering.is_empty() {
        if !steering_mode.is_empty() {
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
        }
    } else if steering_mode.len() > steering.len() {
        Some("--steering-mode")
    } else if steering_scale.len() > steering.len() {
        Some("--steering-scale")
    } else if steering_layers.len() > steering.len() {
        Some("--steering-layers")
    } else {
        None
    };
    if let Some(option) = orphan {
        let value = if steering.is_empty() {
            "a steering parameter needs --steering; without a direction set the \
             run decodes unsteered and the parameter does nothing"
        } else {
            "more per-vector steering values than --steering paths; the extras can \
             never reach a vector"
        };
        return ParseOutcome::Failure(ParseFailure::InvalidValue {
            option,
            value: value.to_string(),
        });
    }

    ParseOutcome::Success(InvocationRequest {
        model,
        mode,
        system,
        max_new,
        max_context,
        load_guard,
        min_auto_context,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        seed,
        stop,
        images,
        image_batch,
        rdadvise,
        expert_cache_slots,
        speculation,
        speculative_drafter,
        steering,
        steering_mode,
        steering_scale,
        steering_layers,
        steering_target,
        steering_gate,
        prefill_chunk,
        power_profile,
        max_tokens_per_sec,
        vision_sidecar,
        reasoning,
        kv_bits,
        quiet,
        // PLACEHOLDER so the tree compiles while the --expert-residency parse
        // arm is in flight in another session: flipped to the shorthand once
        // the local binding exists. `Auto` is the request's documented
        // default, identical to the pre-flag behaviour.
        expert_residency: Default::default(),
    })
}

/// Parses `START:END`, inclusive and 0-based, for `--steering-layers`.
///
/// Inclusive because that is what llama.cpp's `--control-vector-layer-range`
/// means and a caller moving between the two should not have to know they
/// differ. An inverted range (`END < START`) is `None` rather than an empty
/// selection: it would steer nothing, and a run that asked to steer and
/// silently did not would measure the unsteered engine and report it as the
/// steered one.
fn parse_layer_range(value: &str) -> Option<(u32, u32)> {
    let (a, b) = value.split_once(':')?;
    let start: u32 = a.trim().parse().ok()?;
    let end: u32 = b.trim().parse().ok()?;
    if end < start {
        return None;
    }
    Some((start, end))
}

fn invalid(option: &'static str, value: &str) -> ParseOutcome {
    ParseOutcome::Failure(ParseFailure::InvalidValue {
        option,
        value: value.to_string(),
    })
}
