//! The raw-completion prefill + decode loop. Ported from
//! `Runtime/Generation/RawCompletion.swift`. [`run_raw_completion`] feeds
//! every prefill token to the producer one at a time ("off" prefill mode in
//! the Swift original); [`run_raw_completion_chunked`] instead splits the
//! prompt into fixed-size chunks (`foundation::prefill_chunk_spans`) and
//! hands each whole chunk to a [`ChunkedPrefillRunner`] in one call,
//! mirroring the Swift original's `chunked` prefill mode. BOTH loops offer
//! cached-prompt continuation (`crate::kv_prefix`): a turn continues from
//! the previous turn's KV wherever the two prompts agree, instead of
//! resetting. It is off unless the producer opts in, so the default path is
//! the reset-and-prefill-from-zero one this loop has always run.

use std::time::Instant;

use foundation::{LogitValue, LogitsView, TokenId};
use selection::select;
use tokenizer::MfTokenizer;

use crate::config::GenerationConfig;
use crate::error::RuntimeError;
use crate::pacing::{Pacer, THERMAL_POLL_TOKENS};
use crate::power::{stepped_cap, MemoryPressure, ThermalLevel};
use crate::producer::LogitProducer;
pub use crate::raw_completion_chunked::{
    run_raw_completion_chunked, run_raw_completion_chunked_cancellable,
};
pub(crate) use crate::token_sink::TokenSink;

/// Reason why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Natural end-of-turn token encountered.
    EndOfTurn,
    /// Tool call invocation token encountered.
    ToolCalls,
    /// End-of-sequence token encountered.
    Eos,
    /// Configured stop string matched.
    StopString,
    /// Generation hit the configured max_new_tokens limit.
    MaxTokens,
    /// The caller's `cancel` predicate returned true. Only reachable from
    /// [`run_raw_completion_cancellable`] and its chunked sibling; the two
    /// original entry points pass a predicate that is always false, so this
    /// variant cannot appear on any pre-existing path.
    ///
    /// The [`RawDecodeResult`] carrying it is otherwise ORDINARY: the stop
    /// matcher's withheld tail has been flushed and `kv_position` /
    /// `kv_backed_token_ids` describe the KV cache honestly, so a caller may
    /// keep the partial turn and continue the conversation from it.
    Cancelled,
}

/// Progress event emitted during prefill and decoding.
#[derive(Debug, Clone, PartialEq)]
pub enum RawDecodeProgress {
    /// Prefill progress showing processed vs total prompt tokens.
    Prefill {
        /// Number of prompt tokens processed so far.
        done: usize,
        /// Total number of prompt tokens to prefill.
        total: usize,
    },
    /// A single decoded token.
    Token {
        /// 0-indexed generation step.
        index: usize,
        /// Sampled token ID.
        id: TokenId,
        /// Detokenized text delta.
        delta: String,
    },
    /// Flushed trailing text from the stop matcher.
    Tail(String),
}

/// Summary result of a completed raw generation run.
#[derive(Debug, Clone, PartialEq)]
pub struct RawDecodeResult {
    /// Number of prompt tokens processed during prefill.
    pub prompt_tokens: usize,
    /// Number of newly generated tokens decoded.
    pub new_tokens: usize,
    /// Wall-clock prefill duration in seconds.
    pub prefill_seconds: f64,
    /// Wall-clock decode duration in seconds.
    pub decode_seconds: f64,
    /// The condition that caused generation to terminate.
    pub reason: StopReason,
    /// Final KV cache write cursor position.
    pub kv_position: usize,
    /// List of token IDs currently resident in the KV cache.
    pub kv_backed_token_ids: Vec<TokenId>,
    /// How many leading prompt tokens this turn CONTINUED from the previous
    /// one instead of prefilling (`crate::kv_prefix`). Zero unless the
    /// producer opted in and found a match, which is every caller by default.
    ///
    /// Reported because the match RATE is an empirical property of the
    /// checkpoint's template and tokenizer rather than something the
    /// mechanism can promise, and because a reuse that silently never fires
    /// differs from a working one only in wall-clock -- which thermal drift
    /// alone can cover (AGENTS.md Gotcha 28).
    pub reused_prefix_tokens: usize,
    /// Whether serving this turn evicted a DIFFERENT session's still-usable
    /// KV/recurrent state from the pool to make room (`--session-slots`,
    /// `crate::session_pool`). Always `false` unless the producer opted in
    /// AND is actually being churned under, which is the operational
    /// question an operator sizing the flag needs answered -- "is the
    /// feature on" is not it. See `LogitProducer::session_slot_evicted`.
    pub session_slot_evicted: bool,
    /// The WORST memory pressure observed while this turn decoded.
    ///
    /// **Always `Normal` when the profile does no stepping**, which is the
    /// default: no probe runs, so this is the absence of a reading rather
    /// than a reading of "fine". A host deciding whether to unload something
    /// should read `ts_system_info_json` as well, which polls unconditionally.
    ///
    /// Reported rather than acted on. The engine caps its own decode rate
    /// (that is [`crate::stepped_cap`]); it does not close sessions, because
    /// it does not own them -- the FFI's handle belongs to the caller and a
    /// session that destroyed itself would leave every host holding a dead
    /// pointer it never asked to be given.
    pub peak_memory_pressure: MemoryPressure,
}

pub(crate) fn check_admission(
    prompt_ids: &[TokenId],
    config: &GenerationConfig,
    max_context: u32,
) -> Result<(), RuntimeError> {
    if prompt_ids.is_empty() {
        return Err(RuntimeError::EmptyPrompt);
    }
    let required = prompt_ids.len() as u64 + config.max_new_tokens as u64;
    if required > max_context as u64 {
        return Err(RuntimeError::ContextOverflow {
            prompt: prompt_ids.len(),
            max_new: config.max_new_tokens,
            max_context,
        });
    }
    Ok(())
}

/// The result of a run cancelled before decoding began.
///
/// Shared by both prefill modes so the two cannot drift. There is no stop
/// matcher or detokenizer to flush here: neither is constructed until
/// `decode`, and no generated token has been seen, so the withheld-tail
/// problem the decode arm has does not arise.
pub(crate) fn cancelled_during_prefill(
    history: Vec<TokenId>,
    position: usize,
    prompt_tokens: usize,
    prefill_start: Instant,
    reused: usize,
    session_slot_evicted: bool,
) -> RawDecodeResult {
    RawDecodeResult {
        prompt_tokens,
        new_tokens: 0,
        prefill_seconds: prefill_start.elapsed().as_secs_f64(),
        // Zero rather than unmeasured: decoding never started, and a
        // caller dividing tokens by seconds must not see a rate.
        decode_seconds: 0.0,
        reason: StopReason::Cancelled,
        kv_position: position,
        kv_backed_token_ids: history,
        reused_prefix_tokens: reused,
        session_slot_evicted,
        // Decoding never started, so nothing was ever polled.
        peak_memory_pressure: MemoryPressure::Normal,
    }
}

/// The predicate a cancellable run polls. `true` means stop.
///
/// A `&dyn Fn` rather than a field on [`GenerationConfig`] deliberately: that
/// struct has no `Default` and is built as a literal in over twenty places,
/// so a new field would touch every one of them to buy nothing. An extra
/// parameter on two new functions costs the existing call sites nothing at
/// all.
pub type CancelFlag<'a> = &'a dyn Fn() -> bool;

/// The predicate the two non-cancellable entry points pass. Always false, so
/// they execute the same statement sequence they did before cancellation
/// existed.
pub(crate) const NEVER: &dyn Fn() -> bool = &|| false;

/// Feeds every prompt token to `producer` one at a time (`off` prefill
/// mode), then decodes.
#[allow(clippy::too_many_arguments)]
pub fn run_raw_completion(
    producer: &mut dyn LogitProducer,
    tokenizer: &MfTokenizer,
    prompt_ids: &[TokenId],
    config: &GenerationConfig,
    max_context: u32,
    vocab_size: usize,
    on_progress: impl FnMut(RawDecodeProgress),
) -> Result<RawDecodeResult, RuntimeError> {
    run_raw_completion_cancellable(
        producer,
        tokenizer,
        prompt_ids,
        config,
        max_context,
        vocab_size,
        NEVER,
        on_progress,
    )
}

/// [`run_raw_completion`], polling `cancel` once per prefill token and once
/// per decoded token.
///
/// Cancelling is not an error: the run returns a normal [`RawDecodeResult`]
/// with [`StopReason::Cancelled`] and whatever it had generated. A cancel
/// observed during PREFILL yields zero new tokens, which is the honest answer
/// rather than a failure.
#[allow(clippy::too_many_arguments)]
pub fn run_raw_completion_cancellable(
    producer: &mut dyn LogitProducer,
    tokenizer: &MfTokenizer,
    prompt_ids: &[TokenId],
    config: &GenerationConfig,
    max_context: u32,
    vocab_size: usize,
    cancel: CancelFlag<'_>,
    mut on_progress: impl FnMut(RawDecodeProgress),
) -> Result<RawDecodeResult, RuntimeError> {
    check_admission(prompt_ids, config, max_context)?;

    // How much of this prompt the producer's state already covers. Zero
    // unless the producer both opted in and can prove a prefix match, so the
    // default path resets and prefills from 0 exactly as it always did.
    // Clamped to `len() - 1` rather than `len()`: reuse must leave at least
    // one token to feed, because `decode` starts by sampling from `logits`
    // and a zero-token prefill would hand it the PREVIOUS turn's.
    let reused = producer
        .try_reuse_prefix(prompt_ids)
        .min(prompt_ids.len() - 1);
    if reused == 0 {
        producer.reset();
    }
    let mut history: Vec<TokenId> =
        Vec::with_capacity(prompt_ids.len() + config.max_new_tokens as usize);
    let mut logits = vec![LogitValue::from_f32(0.0); vocab_size];

    let prefill_start = Instant::now();
    // The reused positions are already in the producer's KV, so the cursor
    // starts past them and `history` is seeded with the ids that built them.
    // Both have to agree with the producer or the returned `kv_position`
    // describes a cache that does not exist.
    let mut position = reused;
    history.extend_from_slice(&prompt_ids[..reused]);
    // Only the last prompt token's logits are ever read (`decode` starts by
    // sampling from `logits`), so every earlier one goes through
    // `produce_prefill` and lets the producer skip its output head.
    // `check_admission` rejected an empty prompt, so this cannot underflow.
    let last = prompt_ids.len() - 1;
    for (i, &token) in prompt_ids.iter().enumerate().skip(reused) {
        if i == last {
            producer.produce(token, position, &mut logits)
        } else {
            producer.produce_prefill(token, position, &mut logits)
        }
        .map_err(RuntimeError::Producer)?;
        position += 1;
        history.push(token);
        on_progress(RawDecodeProgress::Prefill {
            done: position,
            total: prompt_ids.len(),
        });
        // Polled AFTER the token is committed, never between `produce` and
        // the bookkeeping below it: the producer has already advanced its KV
        // cursor by the time this is reached, so `position` and `history`
        // have to agree with it or the returned `kv_position` describes a
        // cache that does not exist.
        if cancel() {
            return Ok(cancelled_during_prefill(
                history,
                position,
                prompt_ids.len(),
                prefill_start,
                reused,
                producer.session_slot_evicted(),
            ));
        }
    }
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();

    let mut result = decode(
        producer,
        tokenizer,
        config,
        &mut logits,
        history,
        position,
        prompt_ids.len(),
        prefill_seconds,
        cancel,
        on_progress,
    )?;
    result.reused_prefix_tokens = reused;
    result.session_slot_evicted = producer.session_slot_evicted();
    Ok(result)
}

/// The decode loop shared by both prefill modes: sample, stop-check,
/// detokenize, and (if continuing) produce the next position's logits.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode<P: LogitProducer + ?Sized>(
    producer: &mut P,
    tokenizer: &MfTokenizer,
    config: &GenerationConfig,
    logits: &mut [LogitValue],
    history: Vec<TokenId>,
    mut position: usize,
    prompt_tokens: usize,
    prefill_seconds: f64,
    cancel: CancelFlag<'_>,
    mut on_progress: impl FnMut(RawDecodeProgress),
) -> Result<RawDecodeResult, RuntimeError> {
    let decode_start = Instant::now();
    let mut sink = TokenSink::new(tokenizer, config, history);
    let reason;

    // ROADMAP Phase P2. `None` for the default uncapped config, which
    // leaves the loop below executing exactly the statement sequence it
    // did before rate control existed.
    //
    // The WORST memory pressure seen across the run, reported on the result.
    // A watcher that only exposed the level at the moment a caller happened
    // to ask would miss a spike entirely, and a spike is the whole event
    // worth telling a host about -- it is what makes unloading something
    // else the right response.
    let mut peak_memory_pressure = MemoryPressure::Normal;
    let mut pacer = config.rate.is_active().then(|| {
        let level = config
            .rate
            .thermal_probe
            .map_or(ThermalLevel::Nominal, |probe| probe());
        let memory = config
            .rate
            .memory_probe
            .map_or(MemoryPressure::Normal, |probe| probe());
        peak_memory_pressure = peak_memory_pressure.max(memory);
        Pacer::new(
            stepped_cap(config.rate.max_tokens_per_sec, level, memory),
            decode_start,
        )
    });

    loop {
        let token_id = select(
            LogitsView::new(logits),
            &config.shaping,
            &sink.history,
            sink.generated as u64,
        )?;

        if let Some(stop) = sink.commit(token_id, cancel, &mut on_progress) {
            reason = stop;
            break;
        }

        // Pace AFTER the decision to continue, so the last token of a
        // generation never pays a sleep nobody waits through, and BEFORE
        // `produce`, so the idle window falls between forward passes
        // rather than inside one.
        if let Some(pacer) = pacer.as_mut() {
            pacer.note_token();
            // ONE poll block for both signals, on the boundary that already
            // existed. Giving memory its own cadence would add a second
            // syscall schedule for a reading whose ladder shares these
            // ceilings, and would make the two able to disagree about which
            // token they describe.
            let watching =
                config.rate.thermal_probe.is_some() || config.rate.memory_probe.is_some();
            if watching && sink.generated % THERMAL_POLL_TOKENS == 0 {
                let level = config
                    .rate
                    .thermal_probe
                    .map_or(ThermalLevel::Nominal, |probe| probe());
                let memory = config
                    .rate
                    .memory_probe
                    .map_or(MemoryPressure::Normal, |probe| probe());
                peak_memory_pressure = peak_memory_pressure.max(memory);
                let cap = stepped_cap(config.rate.max_tokens_per_sec, level, memory);
                pacer.apply_cap(cap, Instant::now());
            }
            if let Some(wait) = pacer.due_in(Instant::now()) {
                std::thread::sleep(wait);
            }
        }

        producer
            .produce(token_id, position, logits)
            .map_err(RuntimeError::Producer)?;
        position += 1;
    }

    Ok(RawDecodeResult {
        // `decode` is shared by all three prefill loops and knows nothing
        // about reuse; the one loop that reuses patches this on the way out.
        reused_prefix_tokens: 0,
        // Same reasoning: patched by the caller that actually reset (or
        // didn't) the producer.
        session_slot_evicted: false,
        prompt_tokens,
        new_tokens: sink.generated,
        prefill_seconds,
        decode_seconds: decode_start.elapsed().as_secs_f64(),
        reason,
        kv_position: position,
        kv_backed_token_ids: sink.history,
        peak_memory_pressure,
    })
}
