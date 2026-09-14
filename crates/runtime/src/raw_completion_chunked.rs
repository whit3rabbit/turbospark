use std::time::Instant;

use foundation::{prefill_chunk_spans, LogitValue, PrefillChunkCommitState, TokenId};
use tokenizer::MfTokenizer;

use crate::config::GenerationConfig;
use crate::error::RuntimeError;
use crate::producer::ChunkedPrefillRunner;
use crate::raw_completion::{
    cancelled_during_prefill, check_admission, decode, CancelFlag, RawDecodeProgress,
    RawDecodeResult, NEVER,
};

/// Splits `prompt_ids` into fixed-size chunks and hands each whole chunk to
/// `producer` in one call, then decodes. `chunk_tokens` should come from one
/// of [`foundation::runtime_config::ALLOWED_CHUNK_SIZES`], clamped to
/// [`foundation::MAX_CHUNK_TOKENS`] by [`foundation::prefill_chunk_spans`].
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

    // Same contract as the sequential loop's: 0 unless the producer opted in
    // and found a match, and the reset is skipped exactly when it did.
    //
    // THIS loop is the one that matters for reuse in practice. Every family
    // that supports chunked prefill routes here, so the CLI and the server
    // reach it rather than `run_raw_completion` -- wiring reuse only there
    // left the measured match rate at 0/33 and 0/49 in the real chat REPL,
    // an optimisation of the path nobody takes.
    let reused = producer
        .try_reuse_prefix(prompt_ids)
        .min(prompt_ids.len() - 1);
    if reused == 0 {
        producer.reset();
    }
    let mut history: Vec<TokenId> =
        Vec::with_capacity(prompt_ids.len() + config.max_new_tokens as usize);
    let mut logits = vec![LogitValue::from_f32(0.0); vocab_size];
    let mut commit_state = PrefillChunkCommitState::new();

    let prefill_start = Instant::now();
    // Spans start at the reused offset, so the first chunk begins where the
    // previous turn's state ended rather than at 0.
    let spans = prefill_chunk_spans(prompt_ids.len() - reused, reused, chunk_tokens);
    history.extend_from_slice(&prompt_ids[..reused]);
    let mut position = reused;
    for (span_index, span) in spans.iter().enumerate() {
        commit_state
            .require_clean("chunked prefill")
            .map_err(|e| RuntimeError::Producer(e.to_string()))?;
        // `token_offset` is relative to the span walk's own start, which is
        // the reused offset rather than 0.
        let base = reused + span.token_offset;
        let chunk = &prompt_ids[base..base + span.token_count];
        commit_state.mark_dirty(span.start_position, span.token_count);
        producer
            .prefill_chunk_with_status(
                chunk,
                span.start_position,
                &mut logits,
                span_index + 1 == spans.len(),
            )
            .map_err(RuntimeError::Producer)?;
        commit_state.mark_committed();

        history.extend_from_slice(chunk);
        // `completed_count` counts from the WALK's start, which is `reused`,
        // while `position` is the absolute KV cursor. `start_position` above
        // is already absolute (the walk was given `reused` as its base), so
        // only this one needs the offset added.
        position = reused + span.completed_count;
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
