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
