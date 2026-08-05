//! Produces next-token logits for [`crate::run_raw_completion`]. Ported
//! from the `LogitProducer` protocol in
//! `Runtime/Generation/LogitProducer.swift`.
//!
//! The production implementation (`RealForwardRunner`, wiring the GPU
//! kernel stack in `mrefrust_gpu` to real model weights) is out of scope
//! for this port: no trained `.gturbo` weights or full kernel set are
//! available to validate one against. [`ScriptedLogitProducer`] plays the
//! same role Swift's `ScriptedLogitProducer` test fixture does — it lets
//! the loop's control flow (stop handling, detokenizer/stop-matcher
//! ordering, history bookkeeping) be exercised and tested independently of
//! the kernel stack, exactly as the upstream design intends.

use foundation::LogitValue;

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
}

/// A [`LogitProducer`] that can also process a whole prefill chunk in one
/// call, writing the logits state for the position immediately after the
/// chunk. Ported from the `ChunkedPrefillRunner` protocol in
/// `Runtime/Generation/LogitProducer.swift`; only the logits-output mode is
/// ported (the fused-greedy-head shortcut needs a real forward pass to be
/// meaningful — see `DEVIATIONS.md`).
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
