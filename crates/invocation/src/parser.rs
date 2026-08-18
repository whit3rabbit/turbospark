//! The left-to-right token scan and the fixed-order whole-invocation checks
//! that turn a token list into exactly one outcome.
//!
//! Parsing is a pure function of the supplied token list: no filesystem,
//! environment, or other I/O is performed, and the same token list always
//! produces the same outcome.

use crate::failure::ParseFailure;
use crate::options::OPTIONS;
use crate::request::{
    ExpertCacheSlots, InvocationRequest, MaxContext, Mode, PowerProfile, PrefillChunk,
    ReadAheadMode, ReasoningEffort, DEFAULT_MAX_NEW, DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P, MAX_TOP_K,
};
use foundation::runtime_config::{ALLOWED_CACHE_SLOTS, ALLOWED_CHUNK_SIZES};

/// One of the three possible outcomes of parsing a token list. There is no
/// fourth outcome and no partially populated result.
#[derive(Debug, Clone, PartialEq)]
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
    let mut temperature = DEFAULT_TEMPERATURE;
    let mut top_k = DEFAULT_TOP_K;
    let mut top_p = DEFAULT_TOP_P;
    let mut top_p_explicit = false;
    let mut repetition_penalty = DEFAULT_REPETITION_PENALTY;
    let mut seed: Option<u64> = None;
    let mut stop: Vec<String> = Vec::new();
    let mut rdadvise = ReadAheadMode::default();
    let mut expert_cache_slots = ExpertCacheSlots::default();
    let mut prefill_chunk = PrefillChunk::default();
    let mut power_profile: Option<PowerProfile> = None;
    let mut max_tokens_per_sec: Option<f64> = None;
    let mut reasoning = ReasoningEffort::default();
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

    ParseOutcome::Success(InvocationRequest {
        model,
        mode,
        system,
        max_new,
        max_context,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        seed,
        stop,
        rdadvise,
        expert_cache_slots,
        prefill_chunk,
        power_profile,
        max_tokens_per_sec,
        reasoning,
        quiet,
    })
}

fn invalid(option: &'static str, value: &str) -> ParseOutcome {
    ParseOutcome::Failure(ParseFailure::InvalidValue {
        option,
        value: value.to_string(),
    })
}
