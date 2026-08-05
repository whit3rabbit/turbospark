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
use crate::producer::{ChunkedPrefillRunner, LogitProducer};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    EndOfTurn,
    ToolCalls,
    Eos,
    StopString,
    MaxTokens,
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
    mut on_progress: impl FnMut(RawDecodeProgress),
) -> Result<RawDecodeResult, RuntimeError> {
    check_admission(prompt_ids, config, max_context)?;

    producer.reset();
    let mut history: Vec<TokenId> =
        Vec::with_capacity(prompt_ids.len() + config.max_new_tokens as usize);
    let mut logits = vec![LogitValue::from_f32(0.0); vocab_size];

    let prefill_start = Instant::now();
    let mut position = 0usize;
    for &token in prompt_ids {
        producer
            .produce(token, position, &mut logits)
            .map_err(RuntimeError::Producer)?;
        position += 1;
        history.push(token);
        on_progress(RawDecodeProgress::Prefill {
            done: position,
            total: prompt_ids.len(),
        });
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
    mut on_progress: impl FnMut(RawDecodeProgress),
) -> Result<RawDecodeResult, RuntimeError> {
    let decode_start = Instant::now();
    let mut stop_matcher = StreamingStopMatcher::new(config.stop_strings.clone());
    let mut detok = MfDetokenizer::new(tokenizer);
    let mut generated = 0usize;
    let reason;

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
            reason = if token_id == tokenizer.end_of_turn_id {
                StopReason::EndOfTurn
            } else if token_id == tokenizer.tool_response_id {
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
        if hit_stop_string || hit_max {
            let mut tail = stop_matcher.push(&detok.flush());
            tail += &stop_matcher.finish();
            if !tail.is_empty() {
                on_progress(RawDecodeProgress::Tail(tail));
            }
            reason = if hit_stop_string {
                StopReason::StopString
            } else {
                StopReason::MaxTokens
            };
            break;
        }

        history.push(token_id);
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
