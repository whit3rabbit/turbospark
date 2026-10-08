//! Wall-clock breakdown of one Music 3 generation request.
//!
//! The accumulator costs a handful of `Instant::now()` calls per AR frame
//! and is always on, so the ordinary and timed entry points run identical
//! code. Backend-dispatched work is attributed to the stage that issued it.

use std::time::Duration;

/// Per-stage wall time for one request, summed over frames and chunks.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StageTimings {
    /// Prompt cleanup and tokenizer encode.
    pub tokenize: Duration,
    /// Text-pair embedding and the prefill forward.
    pub prefill: Duration,
    /// Per-frame LM head projection (both CFG rows).
    pub lm_head: Duration,
    /// Per-frame masking, CFG combine, top-k selection, and sampling.
    pub sampling: Duration,
    /// Per-frame depth decoder expansion (all residual codebooks).
    pub depth: Duration,
    /// Per-frame feedback embedding and cached LM decode step.
    pub lm_decode: Duration,
    /// Condition encoder, noise generation, and chunk bookkeeping.
    pub condition: Duration,
    /// Euler steps through the DiT, both CFG branches.
    pub dit: Duration,
    /// Vocoder decode of each chunk.
    pub vocoder: Duration,
    /// Crop-stitch and interleave to `[S, 2]`.
    pub stitch: Duration,
    /// Whole `generate_text_timed` call.
    pub total: Duration,
    /// AR frames emitted.
    pub frames: usize,
    /// Flow chunks decoded.
    pub chunks: usize,
}

impl StageTimings {
    /// Sum of the AR stage buckets (prefill through LM decode).
    pub fn autoregressive(&self) -> Duration {
        self.prefill + self.lm_head + self.sampling + self.depth + self.lm_decode
    }

    /// Sum of the flow stage buckets (condition through vocoder).
    pub fn flow(&self) -> Duration {
        self.condition + self.dit + self.vocoder
    }
}
