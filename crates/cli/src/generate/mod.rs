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

use invocation::{InvocationRequest, Mode};
use runtime::{
    run_raw_completion, run_raw_completion_chunked, run_raw_completion_speculative,
    GenerationConfig, RawDecodeProgress, RawDecodeResult,
};
use tokenizer::{Message, MfTokenizer, ReasoningEffort, ReasoningSupport};

pub(crate) mod format;
pub(crate) mod session;

pub(crate) use format::{
    map_reasoning_effort, parse_messages_file, print_footer, print_phases, role_name, ChannelSplit,
};
pub(crate) use runtime::SpeculationPlan;
pub(crate) use session::{open_session, Session};

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
    let max_new = match clamp_max_new(&session, request, prompt_ids.len()) {
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

    let prompt_ids = match render_prompt(
        &session.tokenizer,
        &messages,
        map_reasoning_effort(request.reasoning),
    ) {
        Ok(ids) => ids,
        Err(e) => {
            eprintln!("note: not attempting real generation: {e}");
            return;
        }
    };
    let max_new = match clamp_max_new(&session, request, prompt_ids.len()) {
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
///
/// **A LEVEL THIS CHECKPOINT CANNOT SPELL IS REPORTED, NOT SWALLOWED.** A
/// template with no effort key still honours `enable_thinking`, so the
/// request is half-served and the half that was dropped is invisible in the
/// output -- exactly the silent no-op `--reasoning` exists to avoid being.
pub(crate) fn render_prompt(
    tokenizer: &MfTokenizer,
    messages: &[Message],
    reasoning: ReasoningEffort,
) -> Result<Vec<i32>, String> {
    if reasoning != ReasoningEffort::Off
        && tokenizer.reasoning_support() == ReasoningSupport::ToggleOnly
    {
        eprintln!(
            "note: this checkpoint's chat template has no reasoning-effort knob, so \
             --reasoning {} turns thinking ON but sets no level",
            reasoning.as_str()
        );
    }
    let rendered = tokenizer
        .apply_chat_template_with_reasoning(messages, reasoning)
        .map_err(|e| format!("chat template: {e}"))?;
    Ok(tokenizer.encode(&rendered, false))
}

/// The per-turn generation budget: never more than `--max-new`, never more
/// than the context leaves room for.
///
/// Takes the window from the SESSION rather than the request, because under
/// `--max-context auto` the request carries no number and the KV cache has
/// already been allocated at the resolved one.
pub(crate) fn clamp_max_new(
    session: &Session,
    request: &InvocationRequest,
    prompt_len: usize,
) -> Result<u32, String> {
    if prompt_len >= session.max_context as usize {
        return Err(format!(
            "context overflow: prompt {prompt_len} reaches max_context {}",
            session.max_context
        ));
    }
    let room = session.max_context - prompt_len as u32;
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
    let mut split = ChannelSplit::new(&session.tokenizer, map_reasoning_effort(request.reasoning));
    let on_progress = |event| {
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
    };
    // `MFERENCE_PREFILL_CHUNK=<tokens>` routes prefill through
    // `run_raw_completion_chunked` and the runner's chunk driver
    // (`docs/BATCHED_PREFILL.md` step 1). An A/B SEAM, spelled like the two
    // decode-path ones beside it (`MFERENCE_SHARED_CB`,
    // `MFERENCE_ROUTED_PIPELINE`) rather than a flag, for the same reason:
    // both arms must produce identical tokens, so what it varies is
    // throughput and nothing a user needs to reach for. Unset, unparsable
    // or 0 is the sequential path, byte for byte what it always was.
    let chunk_tokens = std::env::var("MFERENCE_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0);
    let result = match (&session.speculation, chunk_tokens) {
        // Speculation wins over the chunked-prefill seam when both are on:
        // the speculative loop has its own prefill (it primes the drafter as
        // it walks) and no chunked variant. They are not composable today and
        // pretending otherwise would silently run one of them.
        (SpeculationPlan::Enabled { block }, _) => run_raw_completion_speculative(
            &mut session.runner,
            &session.tokenizer,
            prompt_ids,
            &config,
            session.max_context,
            vocab_size,
            *block,
            on_progress,
        )?,
        (SpeculationPlan::Disabled { .. }, chunk_tokens) => match chunk_tokens {
            Some(chunk) => run_raw_completion_chunked(
                &mut session.runner,
                &session.tokenizer,
                prompt_ids,
                &config,
                session.max_context,
                vocab_size,
                chunk,
                on_progress,
            )?,
            None => run_raw_completion(
                &mut session.runner,
                &session.tokenizer,
                prompt_ids,
                &config,
                session.max_context,
                vocab_size,
                on_progress,
            )?,
        },
    };
    let _ = writeln!(out);
    Ok((reply, result))
}
