//! Real generation, wired to `RealForwardRunner` (macOS/GPU only). Loads a
//! `.gturbo` install from `request.model`, reconstructs its full
//! `ArchConfig` from `manifest.json`'s own `arch` object (shape fields
//! read as written; family-extension fields fall back to Gemma 4's
//! baseline values, the same fallback rule `arch_validation` applies to
//! omitted manifest fields), and opens it with `RealForwardRunner`. This
//! covers both the synthetic short-name installs and real Gemma 4
//! installs repacked by `repack::write_gemma4_install` (verbatim
//! checkpoint tensor naming, MoE, mixed SWA/full attention); anything the
//! runner does not support (linear/compressed layers, non-Gemma families)
//! is rejected with a clear error, not a crash. The tokenizer is expected
//! to live alongside the `.gturbo` files in the same directory (the usual
//! HF checkpoint bundling convention).
//!
//! All three invocation modes generate: `--prompt` encodes its text
//! verbatim (no templating, matching the Swift original), `--messages-file`
//! renders a JSON conversation through the tokenizer's own chat template,
//! and `--chat` runs the interactive REPL in [`crate::chat`].

use std::collections::HashSet;
use std::io::Write;
use std::path::Path;

use invocation::{InvocationRequest, Mode};
use runtime::{
    run_raw_completion, GenerationConfig, RateControl, RawDecodeProgress, RawDecodeResult,
    RealForwardRunner,
};
use selection::ShapingConfig;
use tokenizer::{
    ChatDialect, Message, MfTokenizer, Role, StructuredAssistantDecoder, StructuredAssistantEvent,
};

/// Everything a generating mode needs: the loaded model, its tokenizer, and
/// the validated sampling configuration. Opened once per process, reused by
/// every turn of an interactive chat.
pub(crate) struct Session {
    pub(crate) tokenizer: MfTokenizer,
    pub(crate) runner: RealForwardRunner,
    pub(crate) shaping: ShapingConfig,
    /// Resolved once at open, because resolving it per turn would let Low
    /// Power Mode toggling mid-chat change the pace for reasons the caller
    /// never asked about.
    pub(crate) rate: RateControl,
}

/// Splits a generated token's text into the answer and the reasoning that
/// preceded it.
///
/// `gpt-oss` is the only family here that separates the two: Harmony puts the
/// model's reasoning in an `analysis` channel BEFORE its answer, so without
/// this the reasoning and the frame markup print as the reply. For every other
/// dialect [`Self::push`] is the identity and no decoder is built at all, so
/// five families' output is byte-identical to what it was.
///
/// **The ANSWER is what a caller accumulates as the assistant turn**, not the
/// pair. Harmony's own convention drops the analysis channel from prior turns,
/// so feeding it back into `--chat` history would send the model something it
/// was never trained to read.
struct ChannelSplit<'a> {
    decoder: Option<StructuredAssistantDecoder<'a>>,
}

impl<'a> ChannelSplit<'a> {
    fn new(tokenizer: &'a MfTokenizer) -> Self {
        Self {
            decoder: (tokenizer.dialect == ChatDialect::Harmony).then(|| {
                // No tool names: Harmony frames a call as a channel header
                // rather than as the bracketing token pair this decoder's
                // tool contract describes, so it parses none (ROADMAP's
                // Harmony tool-calling item).
                StructuredAssistantDecoder::new(tokenizer, HashSet::new(), String::new)
            }),
        }
    }

    /// One token's `(answer, reasoning)`. Either may be empty.
    fn push(&mut self, id: i32, text: &str) -> (String, String) {
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

pub fn try_generate(request: &InvocationRequest) {
    match &request.mode {
        Mode::Prompt(text) => run_prompt(request, text),
        Mode::MessagesFile(path) => run_messages_file(request, path),
        Mode::Chat => crate::chat::run(request),
    }
}

fn run_prompt(request: &InvocationRequest, prompt: &str) {
    let mut session = match open_session(request) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("note: not attempting real generation: {e}");
            return;
        }
    };

    // Verbatim, with a BOS prefix and no chat template: this mode's contract,
    // matching the Swift original. An instruction-tuned model babbles here;
    // that is the missing markup, not a decode bug.
    let prompt_ids = session.tokenizer.encode(prompt, true);
    // Clamp rather than let `check_admission` refuse the whole run: a long
    // raw prompt generates into whatever room is left, the same as the other
    // two modes.
    let max_new = match clamp_max_new(request, prompt_ids.len()) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("note: not attempting real generation: {e}");
            return;
        }
    };

    println!("generating (real forward pass):");
    // Shared with the other two modes rather than reimplemented: this is
    // where the withheld `Tail` is printed (dropping it truncates the reply)
    // and where Harmony's channels are split.
    match stream_turn(&mut session, request, &prompt_ids, max_new) {
        Ok((_, result)) => print_footer(&result, request.quiet),
        Err(e) => eprintln!("generation failed: {e}"),
    }
    print_phases(&session);
}

fn run_messages_file(request: &InvocationRequest, path: &str) {
    // Decode the conversation before loading the model: a typo in the JSON
    // should not cost a multi-gigabyte install open first.
    let messages = match parse_messages_file(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("note: not attempting real generation: {e}");
            return;
        }
    };
    let mut session = match open_session(request) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("note: not attempting real generation: {e}");
            return;
        }
    };

    let prompt_ids = match render_prompt(&session.tokenizer, &messages) {
        Ok(ids) => ids,
        Err(e) => {
            eprintln!("note: not attempting real generation: {e}");
            return;
        }
    };
    let max_new = match clamp_max_new(request, prompt_ids.len()) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("note: not attempting real generation: {e}");
            return;
        }
    };

    println!("generating (real forward pass, chat template applied):");
    match stream_turn(&mut session, request, &prompt_ids, max_new) {
        Ok((_, result)) => print_footer(&result, request.quiet),
        Err(e) => eprintln!("generation failed: {e}"),
    }
    print_phases(&session);
}

/// Render `messages` through the loaded tokenizer's own dialect template.
///
/// `add_bos` is false on purpose: the Gemma template emits the literal
/// `<bos>` mark itself, so encoding with a BOS prefix would double it.
pub(crate) fn render_prompt(
    tokenizer: &MfTokenizer,
    messages: &[Message],
) -> Result<Vec<i32>, String> {
    let rendered = tokenizer
        .apply_chat_template(messages)
        .map_err(|e| format!("chat template: {e}"))?;
    Ok(tokenizer.encode(&rendered, false))
}

/// The per-turn generation budget: never more than `--max-new`, never more
/// than the context leaves room for.
pub(crate) fn clamp_max_new(request: &InvocationRequest, prompt_len: usize) -> Result<u32, String> {
    if prompt_len >= request.max_context as usize {
        return Err(format!(
            "context overflow: prompt {prompt_len} reaches max_context {}",
            request.max_context
        ));
    }
    let room = request.max_context - prompt_len as u32;
    Ok(request.max_new.min(room))
}

/// Generate one turn, streaming deltas to stdout, and return the text that
/// was streamed so an interactive caller can append it to its history.
pub(crate) fn stream_turn(
    session: &mut Session,
    request: &InvocationRequest,
    prompt_ids: &[i32],
    max_new: u32,
) -> Result<(String, RawDecodeResult), runtime::RuntimeError> {
    let config = GenerationConfig {
        shaping: session.shaping,
        max_new_tokens: max_new,
        stop_strings: request.stop.clone(),
        extra_stop_tokens: Vec::new(),
        rate: session.rate,
    };
    let vocab_size = session.runner.vocab_size();

    let mut reply = String::new();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut split = ChannelSplit::new(&session.tokenizer);
    let result = run_raw_completion(
        &mut session.runner,
        &session.tokenizer,
        prompt_ids,
        &config,
        request.max_context,
        vocab_size,
        |event| {
            // Both variants carry visible text: `Tail` is what the stop
            // matcher withheld, so dropping it truncates the reply.
            let (id, text) = match event {
                RawDecodeProgress::Token { id, delta, .. } => (id, delta),
                // A withheld tail has no token id behind it, which is what
                // the tokenizer's "no such token" sentinel means.
                RawDecodeProgress::Tail(tail) => (tokenizer::NO_SUCH_TOKEN_ID, tail),
                RawDecodeProgress::Prefill { .. } => return,
            };
            // NO EARLY RETURN ON EMPTY TEXT. Harmony's frame tokens decode to
            // nothing at all -- the detokenizer skips special tokens -- so
            // skipping them here means the state machine never sees a single
            // `<|channel|>` and the whole turn reads as one run of content.
            // Every transition this split makes arrives as `(id, "")`.
            //
            // Reasoning goes to stderr so redirecting stdout captures the
            // ANSWER alone, and only the answer becomes the assistant turn.
            let (answer, reasoning) = split.push(id, &text);
            if !reasoning.is_empty() {
                eprint!("{reasoning}");
            }
            if answer.is_empty() {
                return;
            }
            let _ = write!(out, "{answer}");
            let _ = out.flush();
            reply.push_str(&answer);
        },
    )?;
    let _ = writeln!(out);
    Ok((reply, result))
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

pub(crate) fn open_session(request: &InvocationRequest) -> Result<Session, String> {
    let model_dir = Path::new(&request.model);
    let arch = repack::peek_manifest_arch(model_dir)?;

    let tokenizer = MfTokenizer::load_from_dir(model_dir).map_err(|e| {
        format!(
            "failed to load a tokenizer from {}: {e}",
            model_dir.display()
        )
    })?;

    // Size the KV cache to the same bound the completion loop admits
    // against, rather than the runner's own 4096-token default, and honor
    // --expert-cache-slots instead of the runner's default 16.
    let runner = RealForwardRunner::open_with_options(
        model_dir,
        arch,
        request.max_context as usize,
        request.expert_cache_slots as usize,
    )
    .map_err(|e| e.to_string())?;

    let shaping = ShapingConfig::new(
        request.temperature,
        request.top_k,
        Some(request.top_p),
        request.repetition_penalty,
        request.seed,
    )
    .map_err(|e| e.to_string())?;

    // ROADMAP Phase P2. `resolve_profile` is where the OS gets asked about
    // Low Power Mode, and it is asked exactly once per process.
    let profile = runtime::resolve_profile(request.power_profile.map(map_power_profile));
    let rate = runtime::rate_control_for(profile, request.max_tokens_per_sec);

    Ok(Session {
        tokenizer,
        runner,
        shaping,
        rate,
    })
}

/// The two crates declare their own profile enums on purpose: `invocation`
/// is pure and depends only on `foundation`. This is the one place the two
/// spellings meet.
fn map_power_profile(profile: invocation::PowerProfile) -> runtime::PowerProfile {
    match profile {
        invocation::PowerProfile::Performance => runtime::PowerProfile::Performance,
        invocation::PowerProfile::Balanced => runtime::PowerProfile::Balanced,
        invocation::PowerProfile::Efficiency => runtime::PowerProfile::Efficiency,
    }
}

/// Decode the `--messages-file` JSON: an array of `{"role", "content"}`
/// objects, exactly the Swift original's shape. An unknown role is an
/// error, not a silent fallback to `user`.
fn parse_messages_file(path: &str) -> Result<Vec<Message>, String> {
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
