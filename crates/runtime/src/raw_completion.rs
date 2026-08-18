//! The raw-completion prefill + decode loop. Ported from
//! `Runtime/Generation/RawCompletion.swift`. [`run_raw_completion`] feeds
//! every prefill token to the producer one at a time ("off" prefill mode in
//! the Swift original); [`run_raw_completion_chunked`] instead splits the
//! prompt into fixed-size chunks (`foundation::prefill_chunk_spans`) and
//! hands each whole chunk to a [`ChunkedPrefillRunner`] in one call,
//! mirroring the Swift original's `chunked` prefill mode. Cached-prompt
//! continuation (`ContinuableLogitProducer`) is not ported — see
//! `DEVIATIONS.md`.

use std::time::Instant;

use foundation::{prefill_chunk_spans, LogitValue, LogitsView, PrefillChunkCommitState, TokenId};
use selection::select;
use tokenizer::{MfDetokenizer, MfTokenizer, StreamingStopMatcher};

use crate::config::GenerationConfig;
use crate::error::RuntimeError;
use crate::pacing::{Pacer, THERMAL_POLL_TOKENS};
use crate::power::{stepped_cap, ThermalLevel};
use crate::producer::{ChunkedPrefillRunner, LogitProducer};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    EndOfTurn,
    ToolCalls,
    Eos,
    StopString,
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

#[derive(Debug, Clone, PartialEq)]
pub enum RawDecodeProgress {
    Prefill {
        done: usize,
        total: usize,
    },
    Token {
        index: usize,
        id: TokenId,
        delta: String,
    },
    Tail(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct RawDecodeResult {
    pub prompt_tokens: usize,
    pub new_tokens: usize,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    pub reason: StopReason,
    pub kv_position: usize,
    pub kv_backed_token_ids: Vec<TokenId>,
}

fn check_admission(
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
fn cancelled_during_prefill(
    history: Vec<TokenId>,
    position: usize,
    prompt_tokens: usize,
    prefill_start: Instant,
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
const NEVER: &dyn Fn() -> bool = &|| false;

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

    producer.reset();
    let mut history: Vec<TokenId> =
        Vec::with_capacity(prompt_ids.len() + config.max_new_tokens as usize);
    let mut logits = vec![LogitValue::from_f32(0.0); vocab_size];

    let prefill_start = Instant::now();
    let mut position = 0usize;
    // Only the last prompt token's logits are ever read (`decode` starts by
    // sampling from `logits`), so every earlier one goes through
    // `produce_prefill` and lets the producer skip its output head.
    // `check_admission` rejected an empty prompt, so this cannot underflow.
    let last = prompt_ids.len() - 1;
    for (i, &token) in prompt_ids.iter().enumerate() {
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
            ));
        }
    }
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();

    decode(
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
    )
}

/// Splits `prompt_ids` into fixed-size chunks and hands each whole chunk to
/// `producer` in one call, then decodes. `chunk_tokens` should come from a
/// validated [`foundation::PrefillRuntimeConfig`].
#[allow(clippy::too_many_arguments)]
pub fn run_raw_completion_chunked(
    producer: &mut dyn ChunkedPrefillRunner,
    tokenizer: &MfTokenizer,
    prompt_ids: &[TokenId],
    config: &GenerationConfig,
    max_context: u32,
    vocab_size: usize,
    chunk_tokens: usize,
    on_progress: impl FnMut(RawDecodeProgress),
) -> Result<RawDecodeResult, RuntimeError> {
    run_raw_completion_chunked_cancellable(
        producer,
        tokenizer,
        prompt_ids,
        config,
        max_context,
        vocab_size,
        chunk_tokens,
        NEVER,
        on_progress,
    )
}

/// [`run_raw_completion_chunked`], polling `cancel` once per prefill CHUNK
/// and once per decoded token.
///
/// The prefill granularity is coarser than the sequential path's by
/// construction: a chunk is one indivisible `prefill_chunk` call, and
/// `PrefillChunkCommitState` refuses to begin the next one while the last is
/// uncommitted. So a cancel mid-chunk is observed at the chunk boundary, and
/// on a 128-token chunk that is a longer wait than a caller might assume.
#[allow(clippy::too_many_arguments)]
pub fn run_raw_completion_chunked_cancellable(
    producer: &mut dyn ChunkedPrefillRunner,
    tokenizer: &MfTokenizer,
    prompt_ids: &[TokenId],
    config: &GenerationConfig,
    max_context: u32,
    vocab_size: usize,
    chunk_tokens: usize,
    cancel: CancelFlag<'_>,
    mut on_progress: impl FnMut(RawDecodeProgress),
) -> Result<RawDecodeResult, RuntimeError> {
    check_admission(prompt_ids, config, max_context)?;

    producer.reset();
    let mut history: Vec<TokenId> =
        Vec::with_capacity(prompt_ids.len() + config.max_new_tokens as usize);
    let mut logits = vec![LogitValue::from_f32(0.0); vocab_size];
    let mut commit_state = PrefillChunkCommitState::new();

    let prefill_start = Instant::now();
    let spans = prefill_chunk_spans(prompt_ids.len(), 0, chunk_tokens);
    let mut position = 0usize;
    for span in &spans {
        commit_state
            .require_clean("chunked prefill")
            .map_err(|e| RuntimeError::Producer(e.to_string()))?;
        let chunk = &prompt_ids[span.token_offset..span.token_offset + span.token_count];
        commit_state.mark_dirty(span.start_position, span.token_count);
        producer
            .prefill_chunk(chunk, span.start_position, &mut logits)
            .map_err(RuntimeError::Producer)?;
        commit_state.mark_committed();

        history.extend_from_slice(chunk);
        position = span.completed_count;
        on_progress(RawDecodeProgress::Prefill {
            done: position,
            total: prompt_ids.len(),
        });
        // At the chunk boundary, where `commit_state` is clean: bailing with
        // a chunk marked dirty would leave the producer's KV cursor ahead of
        // `position`, and `require_clean` exists to make exactly that state
        // unreachable.
        if cancel() {
            return Ok(cancelled_during_prefill(
                history,
                position,
                prompt_ids.len(),
                prefill_start,
            ));
        }
    }
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();

    decode(
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
    )
}

/// The decode loop shared by both prefill modes: sample, stop-check,
/// detokenize, and (if continuing) produce the next position's logits.
#[allow(clippy::too_many_arguments)]
fn decode<P: LogitProducer + ?Sized>(
    producer: &mut P,
    tokenizer: &MfTokenizer,
    config: &GenerationConfig,
    logits: &mut [LogitValue],
    mut history: Vec<TokenId>,
    mut position: usize,
    prompt_tokens: usize,
    prefill_seconds: f64,
    cancel: CancelFlag<'_>,
    mut on_progress: impl FnMut(RawDecodeProgress),
) -> Result<RawDecodeResult, RuntimeError> {
    let decode_start = Instant::now();
    let mut stop_matcher = StreamingStopMatcher::new(config.stop_strings.clone());
    let mut detok = MfDetokenizer::new(tokenizer);
    let mut generated = 0usize;
    let reason;

    // ROADMAP Phase P2. `None` for the default uncapped config, which
    // leaves the loop below executing exactly the statement sequence it
    // did before rate control existed.
    let mut pacer = config.rate.is_active().then(|| {
        let level = config
            .rate
            .thermal_probe
            .map_or(ThermalLevel::Nominal, |probe| probe());
        Pacer::new(
            stepped_cap(config.rate.max_tokens_per_sec, level),
            decode_start,
        )
    });

    loop {
        let token_id = select(
            LogitsView::new(logits),
            &config.shaping,
            &history,
            generated as u64,
        )?;
        generated += 1;

        let is_stop_token = tokenizer.stop_token_ids.contains(&token_id)
            || config.extra_stop_tokens.contains(&token_id);
        if is_stop_token {
            // `tool_call_stop_id` rather than `tool_response_id`: the two are
            // the same token on Gemma and are NOT on Harmony, whose
            // `tool_response_id` is `NO_SUCH_TOKEN_ID` and whose tool stop is
            // `<|call|>`. Reading the response marker here sent every gpt-oss
            // tool call to a client as `finish_reason: "stop"`.
            reason = if token_id == tokenizer.end_of_turn_id {
                StopReason::EndOfTurn
            } else if token_id == tokenizer.tool_call_stop_id {
                StopReason::ToolCalls
            } else {
                StopReason::Eos
            };
            let mut tail = stop_matcher.push(&detok.flush());
            tail += &stop_matcher.finish();
            if !tail.is_empty() {
                on_progress(RawDecodeProgress::Tail(tail));
            }
            break;
        }

        let delta = detok.push(token_id);
        let visible = stop_matcher.push(&delta);
        on_progress(RawDecodeProgress::Token {
            index: generated - 1,
            id: token_id,
            delta: visible,
        });

        let hit_stop_string = stop_matcher.is_stopped();
        let hit_max = generated as u32 >= config.max_new_tokens;
        // Polled here rather than at the top of the loop so a cancelled run
        // takes the SAME exit path the other two do, flushing the stop
        // matcher's withheld tail. Breaking early instead would silently drop
        // whatever the matcher was holding back, which is a truncated reply
        // rather than a cancelled one.
        let hit_cancel = cancel();
        if hit_stop_string || hit_max || hit_cancel {
            let mut tail = stop_matcher.push(&detok.flush());
            tail += &stop_matcher.finish();
            if !tail.is_empty() {
                on_progress(RawDecodeProgress::Tail(tail));
            }
            // Cancellation is LAST in precedence: a run that would have
            // stopped on its own terms this token reports why it really
            // stopped, so a Stop button pressed as the model finishes does
            // not relabel a complete turn as a truncated one.
            reason = if hit_stop_string {
                StopReason::StopString
            } else if hit_max {
                StopReason::MaxTokens
            } else {
                StopReason::Cancelled
            };
            break;
        }

        history.push(token_id);

        // Pace AFTER the decision to continue, so the last token of a
        // generation never pays a sleep nobody waits through, and BEFORE
        // `produce`, so the idle window falls between forward passes
        // rather than inside one.
        if let Some(pacer) = pacer.as_mut() {
            pacer.note_token();
            if let Some(probe) = config.rate.thermal_probe {
                if generated % THERMAL_POLL_TOKENS == 0 {
                    let cap = stepped_cap(config.rate.max_tokens_per_sec, probe());
                    pacer.apply_cap(cap, Instant::now());
                }
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
        prompt_tokens,
        new_tokens: generated,
        prefill_seconds,
        decode_seconds: decode_start.elapsed().as_secs_f64(),
        reason,
        kv_position: position,
        kv_backed_token_ids: history,
    })
}
