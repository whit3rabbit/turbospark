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

use std::io::Write;
use std::path::Path;

use invocation::{InvocationRequest, Mode};
use runtime::{
    run_raw_completion, GenerationConfig, RawDecodeProgress, RawDecodeResult, RealForwardRunner,
};
use selection::ShapingConfig;
use tokenizer::{Message, MfTokenizer, Role};

/// Everything a generating mode needs: the loaded model, its tokenizer, and
/// the validated sampling configuration. Opened once per process, reused by
/// every turn of an interactive chat.
pub(crate) struct Session {
    pub(crate) tokenizer: MfTokenizer,
    pub(crate) runner: RealForwardRunner,
    pub(crate) shaping: ShapingConfig,
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

    let config = GenerationConfig {
        shaping: session.shaping,
        max_new_tokens: request.max_new,
        stop_strings: request.stop.clone(),
        extra_stop_tokens: Vec::new(),
    };

    let prompt_ids = session.tokenizer.encode(prompt, true);
    let vocab_size = session.tokenizer.vocab_size;

    println!("generating (real forward pass, synthetic/untrained weights):");
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let result = run_raw_completion(
        &mut session.runner,
        &session.tokenizer,
        &prompt_ids,
        &config,
        request.max_context,
        vocab_size,
        |event| {
            if let RawDecodeProgress::Token { delta, .. } = event {
                let _ = write!(out, "{delta}");
                let _ = out.flush();
            }
        },
    );
    println!();

    match result {
        Ok(r) => println!(
            "note: {} prompt tokens, {} generated, stop reason {:?}",
            r.prompt_tokens, r.new_tokens, r.reason
        ),
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
    };
    let vocab_size = session.tokenizer.vocab_size;

    let mut reply = String::new();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
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
            let text = match event {
                RawDecodeProgress::Token { delta, .. } => delta,
                RawDecodeProgress::Tail(tail) => tail,
                RawDecodeProgress::Prefill { .. } => return,
            };
            if text.is_empty() {
                return;
            }
            let _ = write!(out, "{text}");
            let _ = out.flush();
            reply.push_str(&text);
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
        + p.router_nanos
        + p.hit_cb_nanos
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
        ("gpu wait (commit+wait)", p.gpu_wait_nanos),
        ("router readback+topk  ", p.router_nanos),
        ("hit-expert phase1 cb  ", p.hit_cb_nanos),
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
    if p.expert_requests > 0 {
        eprintln!(
            "  expert cache: {} requests, {} hits ({:.1}%), {} misses",
            p.expert_requests,
            p.expert_hits,
            100.0 * p.expert_hits as f64 / p.expert_requests as f64,
            p.expert_requests - p.expert_hits
        );
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

    Ok(Session {
        tokenizer,
        runner,
        shaping,
    })
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
