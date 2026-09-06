//! Message formatting and diagnostic printouts.

use runtime::RawDecodeResult;
use tokenizer::{ContentPart, Message, ReasoningEffort, Role};

use super::session::Session;

/// The forward-pass phase timing breakdown (`TURBOSPARK_PHASES=1`): where
/// the wall clock of a forward pass actually goes. Counters are cumulative over
/// every `produce` call the runner served, prefill included, so read this
/// with a short prompt and a long generation if you want a decode number.
pub(crate) fn print_phases(session: &Session) {
    let phases_enabled = std::env::var("TURBOSPARK_PHASES").as_deref() == Ok("1");
    if !phases_enabled {
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
    // Bytes, not wall clock: a THIRD axis alongside the `gpu busy` rows,
    // deliberately outside the named-bucket sum above. This is what says
    // whether `expert io` above was a page-cache memcpy or real disk I/O,
    // which is a question the ms/token figure cannot answer on its own.
    if p.expert_io_bytes_requested > 0 {
        let mib_per_token = |bytes: u64| bytes as f64 / (1024.0 * 1024.0) / p.calls.max(1) as f64;
        // An unmeasured run must not read as a proven-warm one: without a
        // sample there is no physical number to print at all.
        let physical = if p.expert_io_samples > 0 {
            format!(
                "{:.1} MiB/token, {:.2}x amplification",
                mib_per_token(p.expert_io_bytes_physical),
                p.expert_io_bytes_physical as f64 / p.expert_io_bytes_requested as f64
            )
        } else {
            "n/a (set TURBOSPARK_EXPERT_DISK_IO=1)".to_string()
        };
        eprintln!(
            "  expert bytes: requested {:.1} MiB/token, physical {}",
            mib_per_token(p.expert_io_bytes_requested),
            physical
        );
    }
    // One level below the buffer buckets above: which dispatch inside a
    // buffer owns its time. Off unless TURBOSPARK_DISPATCH_PROFILE=1, which
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
///
/// **`content` MAY ALSO BE A PART LIST** (ROADMAP M-V7), which is how a
/// conversation carries images:
///
/// ```json
/// {"role": "user", "content": [
///   {"type": "image", "path": "page.png"},
///   {"type": "text",  "text": "Transcribe this."}
/// ]}
/// ```
///
/// The shape is HF's, matching what the checkpoint's own template branches
/// on, with one addition: `path` names the file, because nothing else in this
/// JSON could. Image paths come back beside the messages IN ORDER, since the
/// nth path pairs with the nth marker run the template renders.
///
/// A bare string keeps its exact previous meaning, which is what leaves every
/// text conversation on the path it has always taken.
pub(crate) fn parse_messages_file(path: &str) -> Result<(Vec<Message>, Vec<String>), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("{path}: {e}"))?;
    let rows = value
        .as_array()
        .ok_or_else(|| format!("{path}: expected a JSON array of messages"))?;
    let mut messages = Vec::with_capacity(rows.len());
    let mut images = Vec::new();
    for (index, row) in rows.iter().enumerate() {
        let role = row["role"]
            .as_str()
            .ok_or_else(|| format!("{path}: message {index} has no string \"role\""))?;
        let role = parse_role(role)
            .ok_or_else(|| format!("{path}: message {index} has unsupported role {role:?}"))?;

        match &row["content"] {
            serde_json::Value::String(text) => messages.push(Message::new(role, text)),
            serde_json::Value::Array(parts) => {
                let parsed = parse_content_parts(path, index, parts, &mut images)?;
                messages.push(Message::with_parts(role, parsed));
            }
            _ => {
                return Err(format!(
                    "{path}: message {index} has no \"content\" string or part list"
                ))
            }
        }
    }
    if messages.is_empty() {
        return Err(format!("{path}: no messages"));
    }
    Ok((messages, images))
}

/// One message's ordered content parts, appending any image paths to
/// `images`.
///
/// An unknown `type` is an ERROR rather than a skip: silently dropping a part
/// builds a prompt missing something the caller asked for, and on an image
/// part specifically it would desynchronise every later path from its span.
fn parse_content_parts(
    path: &str,
    index: usize,
    parts: &[serde_json::Value],
    images: &mut Vec<String>,
) -> Result<Vec<ContentPart>, String> {
    let mut out = Vec::with_capacity(parts.len());
    for (p, part) in parts.iter().enumerate() {
        let where_ = || format!("{path}: message {index} part {p}");
        let kind = part["type"]
            .as_str()
            .ok_or_else(|| format!("{}: has no string \"type\"", where_()))?;
        match kind {
            "text" => {
                let text = part["text"]
                    .as_str()
                    .ok_or_else(|| format!("{}: a text part needs \"text\"", where_()))?;
                out.push(ContentPart::Text(text.to_string()));
            }
            "image" => {
                let image = part["path"].as_str().ok_or_else(|| {
                    format!(
                        "{}: an image part needs \"path\" naming a file on disk",
                        where_()
                    )
                })?;
                images.push(image.to_string());
                out.push(ContentPart::Image);
            }
            other => {
                return Err(format!(
                    "{}: unsupported content part type {other:?} (expected \"text\" or \"image\")",
                    where_()
                ))
            }
        }
    }
    Ok(out)
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
