//! Produces next-token logits for [`crate::run_raw_completion`]. Ported
//! from the `LogitProducer` protocol in
//! `Runtime/Generation/LogitProducer.swift`.
//!
//! The production implementation (`RealForwardRunner`, wiring the GPU
//! kernel stack in `turbospark_gpu` to real model weights) is out of scope
//! for this port: no trained `.gturbo` weights or full kernel set are
//! available to validate one against. [`ScriptedLogitProducer`] plays the
//! same role Swift's `ScriptedLogitProducer` test fixture does — it lets
//! the loop's control flow (stop handling, detokenizer/stop-matcher
//! ordering, history bookkeeping) be exercised and tested independently of
//! the kernel stack, exactly as the upstream design intends.

use foundation::{LogitValue, TokenId};

/// Produces next-token logits for the generation loop.
pub trait LogitProducer {
    /// Clear any per-generation state (e.g. a KV cache).
    fn reset(&mut self);

    /// Run one token at `position`, writing FP16 logits into `logits`
    /// (length == vocab size).
    fn produce(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String>;

    /// Run one token whose logits the caller will discard: every prompt
    /// token but the last. `scratch` is the caller's logits buffer and its
    /// contents after the call are unspecified. A producer with an output
    /// head may skip that head here, but must still advance every other
    /// per-token side effect (KV cache, position) exactly as [`Self::produce`]
    /// does. Unrelated to [`ChunkedPrefillRunner::prefill_chunk`], which
    /// runs a whole chunk and does produce usable logits.
    fn produce_prefill(
        &mut self,
        token: i32,
        position: usize,
        scratch: &mut [LogitValue],
    ) -> Result<(), String> {
        self.produce(token, position, scratch)
    }
}

/// A [`LogitProducer`] that can also process a whole prefill chunk in one
/// call, writing the logits state for the position immediately after the
/// chunk. Ported from the `ChunkedPrefillRunner` protocol in
/// `Runtime/Generation/LogitProducer.swift`; only the logits-output mode is
/// ported (the fused-greedy-head shortcut needs a real forward pass to be
/// meaningful — see `DEVIATIONS.md`).
///
/// **The bar an implementor is held to is byte-identity with the sequential
/// path, not coherence.** A chunk must leave the same logits and the same
/// engine state (KV rows, position cursor) that `produce_prefill` over the
/// same tokens would; `crates/runtime/tests/real_forward_gemma4_chunked.rs`
/// asserts that against a NON-chunked reference, never against a second
/// chunked run, which agrees with the first whenever both are wrong the
/// same way. Two implementors today: [`ScriptedLogitProducer`] below and
/// `RealForwardRunner` on the real Gemma 4 flow (`docs/BATCHED_PREFILL.md`
/// step 1), which refuses every other family by name.
pub trait ChunkedPrefillRunner: LogitProducer {
    /// Runs `tokens` (a prefill chunk) starting at `start_position`,
    /// writing the resulting logits into `logits`.
    fn prefill_chunk(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String>;
}

/// A [`LogitProducer`] that can also DRAFT tokens ahead of itself and VERIFY
/// a block of them in one pass, which is what [`crate::run_raw_completion_speculative`]
/// drives (`docs/MTP.md`; measured 1.44x at block 2 on the `qwen3_5` MTP head).
///
/// Deliberately NOT object-safe: `Checkpoint` is an associated type because
/// what a rollback has to restore is the producer's business. On the one real
/// implementor it is a whole gated-DeltaNet snapshot, which is also why
/// [`Self::rollback`] is the expensive call in the round and why the block
/// size that pays is small.
///
/// **Every method here is about the DRAFTER except [`Self::verify`], which is
/// the target model.** Conflating the two is the mistake the shape guards
/// against: the drafter and the target keep separate caches at separate
/// cursors, and a round advances them to DIFFERENT positions -- the target
/// goes back to where the block started and replays, while the drafter goes
/// to where the accepted prefix ended and continues, because it cannot
/// recompute rows whose hidden states the replay has overwritten.
pub trait SpeculativeProducer: LogitProducer {
    /// What [`Self::rollback`] restores.
    type Checkpoint;

    /// Advance the drafter over a known pair without producing logits: the
    /// prompt walk. `next` is the token at `position + 1`, which during a
    /// prompt is known rather than guessed.
    ///
    /// **Skipping this is not a soft failure.** A drafter whose cache was
    /// never primed attends over rows nobody wrote: no error, finite logits,
    /// plausible tokens, and a depressed accept length that reads as a
    /// verdict about speculation rather than as a bug.
    fn prime_drafter(&mut self, next: TokenId, position: usize) -> Result<(), String>;

    /// One drafting step. Writes the DRAFTER's logits, so the caller samples
    /// with the same shaping the target is sampled with and, once rejection
    /// sampling lands, still has the proposal distribution it needs.
    fn draft_step(
        &mut self,
        token: TokenId,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String>;

    /// Move the drafter's cursor to `position`, where the accepted prefix
    /// ended.
    fn rewind_drafter(&mut self, position: usize) -> Result<(), String>;

    /// Capture enough target state to undo an over-long verify.
    fn checkpoint(&mut self) -> Self::Checkpoint;

    /// Restore the target to `point`. The drafter is NOT rewound by this;
    /// see the trait note.
    fn rollback(&mut self, point: &Self::Checkpoint);

    /// Run `feed.len()` positions through the TARGET starting at `base`,
    /// writing one full-vocab row per fed token. Row `i` predicts the token
    /// after `feed[i]`.
    fn verify(
        &mut self,
        feed: &[TokenId],
        base: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String>;
}

/// A fixed sequence of pre-scripted logit vectors, replayed in order. Errors
/// once exhausted. Mirrors the Swift validation suite's
/// `ScriptedLogitProducer` fixture: it decouples decode-loop control flow
/// from the kernel stack.
pub struct ScriptedLogitProducer {
    steps: Vec<Vec<LogitValue>>,
    cursor: usize,
}

impl ScriptedLogitProducer {
    pub fn new(steps: Vec<Vec<LogitValue>>) -> Self {
        Self { steps, cursor: 0 }
    }
}

impl LogitProducer for ScriptedLogitProducer {
    fn reset(&mut self) {
        self.cursor = 0;
    }

    fn produce(
        &mut self,
        _token: i32,
        _position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String> {
        let step = self
            .steps
            .get(self.cursor)
            .ok_or_else(|| "ScriptedLogitProducer exhausted its scripted steps".to_string())?;
        if step.len() != logits.len() {
            return Err(format!(
                "scripted step {} has {} logits, expected {}",
                self.cursor,
                step.len(),
                logits.len()
            ));
        }
        logits.copy_from_slice(step);
        self.cursor += 1;
        Ok(())
    }
}

impl ChunkedPrefillRunner for ScriptedLogitProducer {
    /// Consumes exactly one scripted step per call, regardless of
    /// `tokens.len()` — a real chunked kernel processes the whole chunk in
    /// one dispatch and writes one logits state, so one script step per
    /// `prefill_chunk` call is the scripted analog.
    fn prefill_chunk(
        &mut self,
        tokens: &[i32],
        _start_position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String> {
        if tokens.is_empty() {
            return Err("prefill_chunk called with an empty chunk".to_string());
        }
        self.produce(*tokens.last().unwrap(), 0, logits)
    }
}
