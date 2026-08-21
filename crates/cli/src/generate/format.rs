//! Output channel parsing, message formatting, and diagnostic printouts.

use std::collections::HashSet;

use runtime::RawDecodeResult;
use tokenizer::{
    ChatDialect, Message, MfTokenizer, ReasoningEffort, Role, StructuredAssistantDecoder,
    StructuredAssistantEvent,
};

use super::session::Session;

/// Splits a generated token's text into the answer and the reasoning that
/// preceded it.
///
/// THREE dialects reach this, for two different reasons, and the difference
/// decides when a decoder is built at all.
///
/// `gpt-oss` ALWAYS needs one: Harmony puts the model's reasoning in an
/// `analysis` channel before its answer whatever the caller asked for, so
/// without this the reasoning and the frame markup print as the reply.
///
/// ChatML (Qwen) and Gemma need one only when `--reasoning` asked for
/// thinking. Their thought channels are unreachable otherwise -- ChatML's
/// rendered generation prompt closes its `<think>` block immediately
/// (`<think>\n\n</think>`) and Gemma's template never opens one -- so building
/// a decoder unconditionally would route shipped families through a state
/// machine with nothing to do, and route a spontaneous `<tool_call>` into a
/// parser this binary has no allowlist for. Keyed on the REQUEST, so every
/// existing invocation takes the identity path it always took.
///
/// **SKIPPING EITHER IS NOT A COSMETIC LOSS**, which is why this list is not
/// just Harmony's. Measured on the real Gemma 4 install the first time a
/// level was asked for: the reply began with a bare `thought`, then the
/// model's scratch work, then its answer, all as one run of content. The
/// frame tokens render to the empty string, so a caller cannot tell where
/// one ends and the other starts.
///
/// **The ANSWER is what a caller accumulates as the assistant turn**, not the
/// pair. Harmony's own convention drops the analysis channel from prior turns,
/// and Qwen's template drops `<think>` blocks from prior turns for the same
/// reason, so feeding either back into `--chat` history would send the model
/// something it was never trained to read.
pub(crate) struct ChannelSplit<'a> {
    decoder: Option<StructuredAssistantDecoder<'a>>,
}

impl<'a> ChannelSplit<'a> {
    pub(crate) fn new(tokenizer: &'a MfTokenizer, reasoning: ReasoningEffort) -> Self {
        let wanted = tokenizer.dialect == ChatDialect::Harmony
            || (reasoning != ReasoningEffort::Off
                && matches!(tokenizer.dialect, ChatDialect::ChatMl | ChatDialect::Gemma));
        Self {
            decoder: wanted.then(|| {
                // An EMPTY allowlist, and that is what keeps this binary out
                // of the tool business rather than an accident: the decoder
                // parses a Harmony call only when the caller offered the tool
                // by name, so with no tools on offer a `commentary` body
                // stays reasoning and prints to stderr with the rest of it.
                // `turbospark-check` has no way to run a tool and no shape to
                // render one in; the server is where that lives.
                StructuredAssistantDecoder::new(tokenizer, HashSet::new(), String::new)
            }),
        }
    }

    /// One token's `(answer, reasoning)`. Either may be empty.
    pub(crate) fn push(&mut self, id: i32, text: &str) -> (String, String) {
        let Some(decoder) = self.decoder.as_mut() else {
            return (text.to_string(), String::new());
        };
        let (mut answer, mut reasoning) = (String::new(), String::new());
        match decoder.consume(id, text) {
            Ok(events) => {
                for event in events {
                    match event {
                        StructuredAssistantEvent::Content(c) => answer.push_str(&c),
                        StructuredAssistantEvent::Reasoning(r) => reasoning.push_str(&r),
                        // Unreachable on this dialect (see `new`), and
                        // dropping beats inventing a rendering for it.
                        StructuredAssistantEvent::ToolCall(_) => {}
                    }
                }
            }
            // The Harmony arm has no failure mode, but a caller losing its
            // output to one would be the worst outcome: pass the text through.
            Err(_) => answer.push_str(text),
        }
        (answer, reasoning)
    }
}

/// The Swift original's `MFERENCE_PHASES=1` breakdown: where the wall
/// clock of a forward pass actually goes. Counters are cumulative over
/// every `produce` call the runner served, prefill included, so read this
/// with a short prompt and a long generation if you want a decode number.
pub(crate) fn print_phases(session: &Session) {
    if std::env::var("MFERENCE_PHASES").as_deref() != Ok("1") {
        return;
    }
    let p = session.runner.phase_counters();
    if p.calls == 0 {
        return;
    }
    let ms = |nanos: u64| nanos as f64 / 1e6;
    let per_call = |nanos: u64| nanos as f64 / 1e6 / p.calls as f64;
    // Everything not in a named bucket: CPU dispatch encoding plus the
    // final full-vocab logits readback.
    let accounted = p.gpu_wait_nanos
        + p.final_wait_nanos
        + p.router_nanos
        + p.expert_io_nanos
        + p.bind_nanos
        + p.pipeline_wait_nanos;
    let other = p.total_nanos.saturating_sub(accounted);
    eprintln!(
        "[phases over {} forward passes, {:.0} ms total]",
        p.calls,
        ms(p.total_nanos)
    );
    for (label, nanos) in [
        ("gpu wait (layer cb1)  ", p.gpu_wait_nanos),
        ("final wait (end token)", p.final_wait_nanos),
        ("router readback+topk  ", p.router_nanos),
        ("expert io (pread)     ", p.expert_io_nanos),
        ("routed bind+upload    ", p.bind_nanos),
        ("routed cb retire      ", p.pipeline_wait_nanos),
        ("encode + logit readbk ", other),
    ] {
        eprintln!(
            "  {label}: {:>8.1} ms  {:>5.1}%  {:>6.2} ms/token",
            ms(nanos),
            100.0 * nanos as f64 / p.total_nanos.max(1) as f64,
            per_call(nanos)
        );
    }
    // GPU-side busy time per command-buffer class: a separate axis from
    // the wall-clock buckets above (never part of their sum). Shared and
    // hit buffers are dropped unwaited and stay unattributed.
    if p.cb1_gpu_nanos > 0 {
        eprintln!(
            "  gpu busy: cb1 (attn+router{}) {:.2} ms/token, routed cb {:.2}, final {:.2}",
            if p.routed_cb_gpu_nanos > 0 {
                ""
            } else {
                "+routed"
            },
            per_call(p.cb1_gpu_nanos),
            per_call(p.routed_cb_gpu_nanos),
            per_call(p.final_cb_gpu_nanos)
        );
    }
    if p.expert_requests > 0 {
        eprintln!(
            "  expert cache: {} requests, {} hits ({:.1}%), {} misses",
            p.expert_requests,
            p.expert_hits,
            100.0 * p.expert_hits as f64 / p.expert_requests as f64,
            p.expert_requests - p.expert_hits
        );
    }
    // One level below the buffer buckets above: which dispatch inside a
    // buffer owns its time. Off unless MFERENCE_DISPATCH_PROFILE=1, which
    // perturbs the run it measures -- read the module doc before quoting
    // a number from it.
    if let Some(report) = runtime::dispatch_profile_report(p.calls) {
        eprint!("{report}");
    }
}

/// The Swift original's one-line run summary, on stderr, silenced by
/// `--quiet`.
pub(crate) fn print_footer(result: &RawDecodeResult, quiet: bool) {
    if quiet {
        return;
    }
    let tokens_per_second = if result.decode_seconds > 0.0 {
        result.new_tokens as f64 / result.decode_seconds
    } else {
        0.0
    };
    eprintln!(
        "[stop={:?} prefill={}tok/{:.2}s new={}tok decode={:.2}s tok/s={:.3}]",
        result.reason,
        result.prompt_tokens,
        result.prefill_seconds,
        result.new_tokens,
        result.decode_seconds,
        tokens_per_second
    );
}

/// The third of these mappings, for the reason [`map_power_profile`] is the
/// first: the parser crate may not depend on `tokenizer` either.
pub(crate) fn map_reasoning_effort(effort: invocation::ReasoningEffort) -> ReasoningEffort {
    match effort {
        invocation::ReasoningEffort::Off => ReasoningEffort::Off,
        invocation::ReasoningEffort::Low => ReasoningEffort::Low,
        invocation::ReasoningEffort::Medium => ReasoningEffort::Medium,
        invocation::ReasoningEffort::High => ReasoningEffort::High,
        invocation::ReasoningEffort::XHigh => ReasoningEffort::XHigh,
    }
}

/// Decode the `--messages-file` JSON: an array of `{"role", "content"}`
/// objects, exactly the Swift original's shape. An unknown role is an
/// error, not a silent fallback to `user`.
pub(crate) fn parse_messages_file(path: &str) -> Result<Vec<Message>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("{path}: {e}"))?;
    let rows = value
        .as_array()
        .ok_or_else(|| format!("{path}: expected a JSON array of messages"))?;
    let mut messages = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        let role = row["role"]
            .as_str()
            .ok_or_else(|| format!("{path}: message {index} has no string \"role\""))?;
        let content = row["content"]
            .as_str()
            .ok_or_else(|| format!("{path}: message {index} has no string \"content\""))?;
        let role = parse_role(role)
            .ok_or_else(|| format!("{path}: message {index} has unsupported role {role:?}"))?;
        messages.push(Message::new(role, content));
    }
    if messages.is_empty() {
        return Err(format!("{path}: no messages"));
    }
    Ok(messages)
}

/// `tokenizer::Role`'s own `as_str` is crate-private, so the CLI owns this
/// mapping. The spellings match the Swift original's raw values.
pub(crate) fn parse_role(role: &str) -> Option<Role> {
    match role {
        "system" => Some(Role::System),
        "developer" => Some(Role::Developer),
        "user" => Some(Role::User),
        "assistant" => Some(Role::Assistant),
        "tool" => Some(Role::Tool),
        _ => None,
    }
}

/// The inverse of [`parse_role`], for `/history` output.
pub(crate) fn role_name(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::Developer => "developer",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}
