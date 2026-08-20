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

use invocation::{InvocationRequest, Mode};
use runtime::{
    run_raw_completion, run_raw_completion_chunked, run_raw_completion_speculative,
    GenerationConfig, RateControl, RawDecodeProgress, RawDecodeResult, RealForwardRunner,
};
use selection::ShapingConfig;
use tokenizer::{
    ChatDialect, Message, MfTokenizer, ReasoningEffort, ReasoningSupport, Role,
    StructuredAssistantDecoder, StructuredAssistantEvent,
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
    /// The RESOLVED context window, never `request.max_context`.
    ///
    /// Under `auto` the request carries no number, and every consumer of
    /// this value -- the KV cache the runner already allocated, the
    /// admission check, the per-turn budget -- has to agree with what was
    /// allocated rather than with what was asked for. Reading the request
    /// downstream of `open_session` is how those two come apart.
    pub(crate) max_context: u32,
    /// Whether this session drafts ahead, resolved once at open against the
    /// install and the sampling settings. See [`resolve_speculation`].
    pub(crate) speculation: SpeculationPlan,
}

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
struct ChannelSplit<'a> {
    decoder: Option<StructuredAssistantDecoder<'a>>,
}

impl<'a> ChannelSplit<'a> {
    fn new(tokenizer: &'a MfTokenizer, reasoning: ReasoningEffort) -> Self {
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

/// Which drafter an install actually carries, for
/// [`invocation::SpeculativeDrafter::Auto`].
///
/// Reads the resident INDEX -- the leading index region of
/// `model_weights.bin`, kilobytes, not the weights -- and asks the same two
/// presence predicates `RealForwardRunner::open` asks. It is deliberately a
/// presence check and NOT a can-it-run check: whether the verify pass can
/// serve this install is `speculation_blocker`'s question, asked after the
/// open with the full architecture in hand, and answering it twice in two
/// places is how the two answers drift apart.
///
/// An install with BOTH resolves to `Mtp`, which is what the default was
/// before `Auto` existed, so no run that worked changes. An install with
/// NEITHER also resolves to `Mtp`, so the "no drafter" message a user sees
/// is the one they have always seen; the DFlash2 message would name a
/// tensor they have never heard of.
///
/// An unreadable index resolves to `Mtp` rather than failing: this function
/// picks a drafter, and `open` is entitled to be the one that refuses a
/// broken install.
fn resolve_drafter(
    requested: invocation::SpeculativeDrafter,
    model_dir: &std::path::Path,
) -> invocation::SpeculativeDrafter {
    if requested != invocation::SpeculativeDrafter::Auto {
        return requested;
    }
    let Ok(index) = model_io::load_resident_index(&model_dir.join("model_weights.bin")) else {
        return invocation::SpeculativeDrafter::Mtp;
    };
    if !runtime::install_has_mtp_head(&index) && runtime::install_has_dflash(&index) {
        invocation::SpeculativeDrafter::Dflash
    } else {
        invocation::SpeculativeDrafter::Mtp
    }
}

/// What `open_session` decided about speculative decoding, resolved once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SpeculationPlan {
    Enabled {
        block: usize,
    },
    /// `reason` is `Some` when the user might have expected otherwise, and
    /// `None` when they asked for `off` and got it.
    Disabled {
        reason: Option<String>,
    },
}

/// Decides speculation from the request and what the install turned out to
/// be. Pure, so the hard-fail/warn split is testable without a 14 GB model.
///
/// **THE TWO OUTCOMES DIFFER BY WHO ASKED.** A named block is a promise the
/// caller made to themselves -- they are measuring, or they have read
/// `docs/MTP.md` and chosen 2 -- so failing to keep it is an ERROR carrying
/// the reason. `auto` asked for "speculate if you can", so the same condition
/// is a warning and the run continues. Silently doing neither is what the
/// engine did before the head was detected at all.
///
/// The sampled case is a refusal on BOTH paths rather than a quiet downgrade
/// to greedy: acceptance is `argmax(target) == proposal`, which is exact only
/// at temperature 0. Approximating it would change what the model writes
/// while reporting success.
pub(crate) fn resolve_speculation(
    requested: invocation::Speculation,
    drafter: invocation::SpeculativeDrafter,
    engine_blocker: Option<String>,
    deterministic: bool,
) -> Result<SpeculationPlan, String> {
    let unavailable = if let Some(blocker) = engine_blocker {
        Some(blocker)
    } else if !deterministic {
        Some(
            "acceptance is exact only at --temperature 0, and this run samples; \
             sampled speculation needs rejection sampling with residual correction, \
             which is not implemented"
                .to_string(),
        )
    } else {
        None
    };

    // `auto` resolves to the drafter's own default block, which is not one
    // number: the MTP head's measured optimum is 2 (`docs/MTP.md`) and the
    // DFlash2 drafter's trained block is 8 (`docs/DFLASH2.md`).
    let auto_block = match drafter {
        invocation::SpeculativeDrafter::Dflash => runtime::DFLASH_SERVING_BLOCK,
        // `Auto` is resolved to a concrete drafter by `resolve_drafter`
        // before this is called; it shares MTP's block if one ever arrives
        // here, which is the block this engine defaulted to before DFlash2.
        _ => runtime::DEFAULT_SPECULATION_BLOCK,
    };

    match (requested, unavailable) {
        (invocation::Speculation::Off, _) => Ok(SpeculationPlan::Disabled { reason: None }),
        (invocation::Speculation::Auto, Some(reason)) => Ok(SpeculationPlan::Disabled {
            reason: Some(reason),
        }),
        (invocation::Speculation::Auto, None) => Ok(SpeculationPlan::Enabled { block: auto_block }),
        (invocation::Speculation::Block(_), Some(reason)) => Err(format!(
            "--speculative was asked for but cannot be served: {reason}"
        )),
        (invocation::Speculation::Block(n), None) => {
            Ok(SpeculationPlan::Enabled { block: n as usize })
        }
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

pub(crate) fn open_session(request: &InvocationRequest) -> Result<Session, String> {
    // `--model` takes a path OR a `turbospark-model` alias, resolved HERE
    // rather than in `invocation`, which is pure and whose contract keeps the
    // value an opaque string. A path that exists always wins: a bare name
    // that silently preferred an alias would run a DIFFERENT model than the
    // one on the command line, fluently and with no error.
    let resolved = catalog::resolve_model_arg(&request.model);
    let model_dir = resolved.as_path();
    let arch = repack::peek_manifest_arch(model_dir)?;

    let tokenizer = MfTokenizer::load_from_dir(model_dir).map_err(|e| {
        format!(
            "failed to load a tokenizer from {}: {e}",
            model_dir.display()
        )
    })?;

    // Resolve the context window BEFORE opening, because the failure this
    // catches is an allocation: `KvCacheManager::new` sizes every layer's K
    // and V buffers up front, so a window that does not fit is a Metal
    // allocation error or a swapping machine, with no number in either
    // pointing back at `--max-context`.
    //
    // The three inputs are the ones neither `crates/invocation` (pure) nor
    // the policy module (portable) may read for itself: the checkpoint's own
    // trained context out of the install, the mapped weight region, and this
    // machine's memory.
    let trained = repack::trained_context_meta::peek(model_dir);
    let plan = runtime::resolve_max_context(
        match request.max_context {
            invocation::MaxContext::Auto => runtime::MaxContext::Auto,
            invocation::MaxContext::Fixed(n) => runtime::MaxContext::Fixed(n),
        },
        &arch,
        trained,
        invocation::request::DEFAULT_MAX_CONTEXT,
        runtime::physical_memory(),
        runtime::committed_bytes(model_dir),
    )
    .map_err(|e| e.to_string())?;

    if !request.quiet {
        report_context(&plan, request.max_context);
    }
    // Past the checkpoint's trained context is a QUALITY warning and never an
    // error: RoPE extrapolates rather than failing, and an install written
    // before the trained context was recorded declares none at all, so
    // refusing would be enforced on some installs and not others.
    if plan.past_trained {
        eprintln!(
            "warning: --max-context {} exceeds the checkpoint's trained context of {}; \
             output quality degrades past that point",
            plan.resolved,
            plan.trained.unwrap_or(0)
        );
    }

    // Size the KV cache to the same bound the completion loop admits
    // against, rather than the runner's own 4096-token default, and honor
    // --expert-cache-slots instead of the runner's own fixed default.
    //
    // The slot POLICY crosses the boundary unresolved, exactly as
    // `--power-profile` does: `crates/invocation` is pure and may not read
    // the machine's memory or the install's expert stride, and both are
    // needed to size the cache.
    //
    // THE DRAFTER IS RESOLVED FIRST, because the policies below name exactly
    // one and the wrong one is indistinguishable from an install with no
    // drafter at all: pinned at `Mtp`, a DFlash2 install reported "carries no
    // multi-token-prediction head" and decoded sequentially with a working
    // drafter on disk. `Auto` reads the resident index, which is the index
    // region alone and not the weights.
    let drafter = resolve_drafter(request.speculative_drafter, model_dir);
    let runner = RealForwardRunner::open_with_slot_policy_and_speculation(
        model_dir,
        arch,
        plan.resolved as usize,
        match request.expert_cache_slots {
            invocation::ExpertCacheSlots::Auto => runtime::ExpertCacheSlots::Auto,
            invocation::ExpertCacheSlots::Fixed(n) => runtime::ExpertCacheSlots::Fixed(n as usize),
        },
        // The drafter the flag named owns the request; the other is pinned
        // OFF so a dflash-carrying install never silently opens BOTH
        // drafters' state.
        match drafter {
            invocation::SpeculativeDrafter::Mtp => {
                runtime::DraftPolicies::mtp(match request.speculation {
                    invocation::Speculation::Off => runtime::MtpDraftPolicy::Off,
                    invocation::Speculation::Auto => runtime::MtpDraftPolicy::Auto,
                    invocation::Speculation::Block(n) => runtime::MtpDraftPolicy::Fixed(n as usize),
                })
            }
            // `Auto` is resolved above and cannot reach here.
            invocation::SpeculativeDrafter::Auto | invocation::SpeculativeDrafter::Dflash => {
                runtime::DraftPolicies {
                    mtp: runtime::MtpDraftPolicy::Off,
                    dflash: match request.speculation {
                        invocation::Speculation::Off => runtime::DflashDraftPolicy::Off,
                        invocation::Speculation::Auto => runtime::DflashDraftPolicy::Auto,
                        invocation::Speculation::Block(n) => {
                            runtime::DflashDraftPolicy::Fixed(n as usize)
                        }
                    },
                }
            }
        },
    )
    .map_err(|e| e.to_string())?;

    // Report the RESOLVED slot count, not the request. Under `auto` the
    // request carries no number, and this one is a property of the machine
    // and the install -- 44.2 tok/s at 16 slots against 51.2 at 32 on the
    // same Gemma 4 install (`docs/DECODE_BUDGET.md`), so no throughput or
    // footprint figure from this run is readable without it.
    if !request.quiet {
        eprintln!(
            "expert cache: {} slots per layer{}",
            runner.expert_cache_slots(),
            match request.expert_cache_slots {
                invocation::ExpertCacheSlots::Auto => " (auto)",
                invocation::ExpertCacheSlots::Fixed(_) => "",
            }
        );
    }

    let shaping = ShapingConfig::new(
        request.temperature,
        request.top_k,
        Some(request.top_p),
        request.repetition_penalty,
        request.seed,
    )
    .map_err(|e| e.to_string())?;

    // Decided ONCE, here, and not per turn: both inputs are fixed for the
    // process, and a `--chat` session that started speculating must not stop
    // silently three turns in.
    // `speculation_blocker` and not `mtp_draft_depth() > 0`: a head is
    // NECESSARY and not sufficient. The batched verify is dense-only and
    // INT4-only, so a 1-bit, 2-bit or MoE install carrying a head would pass
    // a head-presence check and then fail at the first verify with the
    // generation already under way. The runner owns that list because the
    // runner owns the refusals it mirrors.
    let speculation = resolve_speculation(
        request.speculation,
        // The RESOLVED drafter, because `auto` takes its block from the
        // drafter's own default and the two differ: 2 for the MTP head, 8
        // for DFlash2.
        drafter,
        match drafter {
            invocation::SpeculativeDrafter::Dflash => runner.dflash_speculation_blocker(),
            _ => runner.speculation_blocker(),
        },
        shaping.is_deterministic(),
    )?;
    if !request.quiet {
        match &speculation {
            SpeculationPlan::Enabled { block } => {
                // The DRAFTER is named, not just the block. Two drafters
                // serve this family and they have different shapes and
                // different measured optima, so a throughput number from
                // this run is unreadable without knowing which one ran.
                let which = match drafter {
                    invocation::SpeculativeDrafter::Dflash => "dflash2 (block drafter)",
                    _ => "mtp head (step drafter)",
                };
                eprintln!("speculative decoding: on, {which}, block {block}");
            }
            // A WARNING and not silence. An install carrying a drafter and
            // decoding one token at a time with nothing said is the exact
            // failure this feature was built to end.
            SpeculationPlan::Disabled { reason: Some(why) } => {
                eprintln!("speculative decoding: off ({why})");
            }
            SpeculationPlan::Disabled { reason: None } => {}
        }
    }

    // ROADMAP Phase P2. `resolve_profile` is where the OS gets asked about
    // Low Power Mode, and it is asked exactly once per process.
    let profile = runtime::resolve_profile(request.power_profile.map(map_power_profile));
    let rate = runtime::rate_control_for(profile, request.max_tokens_per_sec);

    Ok(Session {
        tokenizer,
        runner,
        shaping,
        speculation,
        rate,
        max_context: plan.resolved,
    })
}

/// The resolved context window and the arithmetic behind it.
///
/// Reports the SUGGESTION even when the caller named a number, for the same
/// reason the expert-cache line reports the resolved slot count: a window is
/// most of the KV footprint, and a reader comparing a peak or a prompt
/// refusal against another run needs to see both what was asked for and what
/// the machine and the checkpoint would have allowed.
fn report_context(plan: &runtime::ContextPlan, requested: invocation::MaxContext) {
    let trained = match plan.trained {
        Some(t) => format!("model {t}"),
        // Worth naming rather than omitting: it is why an old install's
        // `auto` reads 4,096 on a machine with room for far more.
        None => "model declares none".to_string(),
    };
    eprintln!(
        "context: {} tokens{} ({}, {:.0} MiB of KV; suggested {})",
        plan.resolved,
        match requested {
            invocation::MaxContext::Auto => " (auto)",
            invocation::MaxContext::Fixed(_) => "",
        },
        trained,
        plan.kv_bytes as f64 / (1024.0 * 1024.0),
        plan.suggested,
    );
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

/// The hard-fail / warn split, which is the whole contract of
/// `--speculative`. Pure, so it needs no install: the two inputs are "does
/// this install carry a head" and "is this run deterministic", and every
/// interesting combination is reachable as a pair of booleans.
#[cfg(test)]
mod speculation_policy {
    use super::{resolve_drafter, resolve_speculation, SpeculationPlan};
    use invocation::{Speculation, SpeculativeDrafter};

    /// `auto` takes its block from the drafter's OWN default, and the two
    /// differ: 2 for the MTP head (`docs/MTP.md`'s measured optimum) and 8
    /// for DFlash2 (its trained block).
    ///
    /// What this pins is the MAPPING, and it asserts the two blocks differ so
    /// the fixture can see a drafter that was never resolved. It does NOT
    /// cover the CALL SITE, which needs a runner -- and the call site is
    /// exactly where this went wrong for one build: `open_session` passed the
    /// unresolved `Auto` and the CLI printed "dflash2 (block drafter), block
    /// 2", a shape neither drafter was measured at. Only a real-model run
    /// catches that one.
    #[test]
    fn auto_takes_each_drafters_own_block() {
        let block_of = |d| match resolve_speculation(Speculation::Auto, d, None, true) {
            Ok(SpeculationPlan::Enabled { block }) => block,
            other => panic!("expected an enabled plan, got {other:?}"),
        };
        assert_eq!(
            block_of(SpeculativeDrafter::Mtp),
            runtime::DEFAULT_SPECULATION_BLOCK
        );
        assert_eq!(
            block_of(SpeculativeDrafter::Dflash),
            runtime::DFLASH_SERVING_BLOCK
        );
        // THE TWO SERVING DEFAULTS NOW AGREE AT 2, independently measured:
        // the MTP head's optimum (`docs/MTP.md`) and DFlash2's two-workload
        // sweep (`DFLASH_SERVING_BLOCK`) landed on the same number, because
        // what decides both is this engine's rollback cost rather than
        // either drafter. So this test pins the MAPPING and can NO LONGER
        // see an unresolved drafter by its block alone; `resolve_drafter`
        // below covers that, and only a real-model run covers the call
        // site. Asserted as an equality so the day they diverge is the day
        // the discriminating check can come back, rather than a day nobody
        // notices.
        assert_eq!(
            runtime::DEFAULT_SPECULATION_BLOCK,
            runtime::DFLASH_SERVING_BLOCK,
            "the serving defaults diverged; restore a discriminating assertion here"
        );
    }

    /// An EXPLICIT drafter is passed through untouched, because it is a
    /// promise the caller made to themselves: `--speculative-drafter mtp` on
    /// a DFlash2 install must reach the MTP blocker and hard-fail there,
    /// never be silently rerouted to the drafter that happens to be present.
    #[test]
    fn an_explicit_drafter_is_never_re_resolved() {
        let missing = std::path::Path::new("/nonexistent/turbospark/install");
        for named in [SpeculativeDrafter::Mtp, SpeculativeDrafter::Dflash] {
            assert_eq!(resolve_drafter(named, missing), named);
        }
    }

    /// An install whose index cannot be read resolves to `Mtp` rather than
    /// failing: this function picks a drafter, and `open` is entitled to be
    /// the one that refuses a broken install.
    #[test]
    fn an_unreadable_index_resolves_to_the_pre_existing_default() {
        assert_eq!(
            resolve_drafter(
                SpeculativeDrafter::Auto,
                std::path::Path::new("/nonexistent/turbospark/install")
            ),
            SpeculativeDrafter::Mtp
        );
    }

    // Verbatim shapes of what `RealForwardRunner::speculation_blocker`
    // returns. The CLI does not construct these -- it forwards whatever the
    // engine says -- so what these fixtures pin is the ROUTING, not the text.
    const NO_HEAD: &str = "this install carries no multi-token-prediction head \
                           (mtp.fc.weight is not in the resident index)";
    const NOT_INT4: &str = "the batched verify is INT4-only and this install's \
                            ...q_proj.weight is dtype 16";
    const MOE: &str = "the batched verify is dense-only and this install routes to 128 experts";

    #[test]
    fn a_named_block_fails_hard_when_it_cannot_be_served() {
        // No head. The caller named a block, so this is an ERROR: they are
        // measuring or have chosen deliberately, and a run that quietly did
        // not speculate would be recorded as the speculative number.
        let err = resolve_speculation(
            Speculation::Block(2),
            SpeculativeDrafter::Mtp,
            Some(NO_HEAD.to_string()),
            true,
        )
        .expect_err("a named block on a headless install must fail");
        assert!(err.contains("no multi-token-prediction head"), "got: {err}");

        // Sampled. Refused rather than downgraded to greedy, which would
        // change what the model writes while reporting success.
        let err = resolve_speculation(Speculation::Block(2), SpeculativeDrafter::Mtp, None, false)
            .expect_err("a named block on a sampled run must fail");
        assert!(err.contains("temperature 0"), "got: {err}");
    }

    #[test]
    fn auto_warns_and_continues_where_a_named_block_fails() {
        let cases = [
            (Some(NO_HEAD.to_string()), true),
            (None, false),
            (Some(NOT_INT4.to_string()), true),
            (Some(MOE.to_string()), false),
        ];
        for (blocker, deterministic) in cases {
            let plan = resolve_speculation(
                Speculation::Auto,
                SpeculativeDrafter::Mtp,
                blocker,
                deterministic,
            )
            .expect("auto never fails; it declines");
            match plan {
                SpeculationPlan::Disabled { reason: Some(_) } => {}
                other => panic!("auto must warn and continue, got {other:?}"),
            }
        }
    }

    #[test]
    fn auto_speculates_when_the_install_and_the_settings_allow_it() {
        let plan = resolve_speculation(Speculation::Auto, SpeculativeDrafter::Mtp, None, true)
            .expect("serviceable");
        assert_eq!(
            plan,
            SpeculationPlan::Enabled {
                block: runtime::DEFAULT_SPECULATION_BLOCK
            }
        );
    }

    #[test]
    fn off_is_silent_even_where_speculation_would_have_worked() {
        // No warning: the caller asked for off and got off. A warning here
        // would train people to ignore the one that matters.
        //
        // THE UNSERVICEABLE CASES ARE THE DISCRIMINATING ONES. With a head
        // present and a deterministic run there is no reason to leak in the
        // first place, so a fixture built only from that combination passes
        // against an `Off` arm that forwards whatever reason it was handed.
        for (blocker, deterministic) in [
            (None, true),
            (Some(NO_HEAD.to_string()), true),
            (None, false),
            (Some(NOT_INT4.to_string()), false),
        ] {
            let plan = resolve_speculation(
                Speculation::Off,
                SpeculativeDrafter::Mtp,
                blocker.clone(),
                deterministic,
            )
            .expect("off never fails");
            assert_eq!(
                plan,
                SpeculationPlan::Disabled { reason: None },
                "off must stay silent at blocker={blocker:?} deterministic={deterministic}"
            );
        }
    }

    #[test]
    fn a_named_block_is_honoured_rather_than_replaced_by_the_default() {
        let plan = resolve_speculation(Speculation::Block(7), SpeculativeDrafter::Mtp, None, true)
            .expect("serviceable");
        assert_eq!(plan, SpeculationPlan::Enabled { block: 7 });
    }
}
