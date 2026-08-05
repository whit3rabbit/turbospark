//! Chunked-prefill planning: splitting a prompt into fixed-size spans, and
//! tracking whether a chunked prefill runner has left uncommitted KV rows
//! behind. Ported from `Runtime/Prefill/PrefillRuntimeConfig.swift`.
//!
//! This is pure span/state bookkeeping; the GPU-side scratch buffer sizing
//! (`PrefillChunkScratchLayout`/`Buffers` in the Swift original, sized for
//! the full attention+MoE tile pipeline) is not ported — it needs a real
//! forward pass's attention and MoE kernels to size scratch for, neither of
//! which exists in this port yet (see `DEVIATIONS.md`).

use std::fmt;

use crate::runtime_config::ALLOWED_CHUNK_SIZES;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefillError {
    ChunkedUnsupported(String),
    ChunkedRunnerDirty(String),
    PrefillCursorMismatch(String),
    UnsupportedPrefillSeed(String),
}

impl PrefillError {
    /// Shared reason text for "chunked mode was requested but the producer
    /// does not implement chunked prefill".
    pub const CHUNKED_REQUIRES_CHUNKED_RUNNER_REASON: &'static str =
        "chunked prefill requires a ChunkedPrefillRunner-backed runtime";
}

impl fmt::Display for PrefillError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrefillError::ChunkedUnsupported(reason)
            | PrefillError::ChunkedRunnerDirty(reason)
            | PrefillError::PrefillCursorMismatch(reason)
            | PrefillError::UnsupportedPrefillSeed(reason) => write!(f, "{reason}"),
        }
    }
}

impl std::error::Error for PrefillError {}

/// One chunk of a chunked prefill plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillChunkSpan {
    pub token_offset: usize,
    pub token_count: usize,
    pub start_position: usize,
    pub completed_count: usize,
}

/// Splits `token_count` tokens starting at `start_position` into spans of at
/// most `chunk_tokens` tokens each, in order. Returns an empty plan for zero
/// tokens. `chunk_tokens` is clamped to `[1, PrefillRuntimeConfig::MAX_CHUNK_TOKENS]`.
pub fn prefill_chunk_spans(
    token_count: usize,
    start_position: usize,
    chunk_tokens: usize,
) -> Vec<PrefillChunkSpan> {
    let chunk = chunk_tokens.clamp(1, PrefillRuntimeConfig::MAX_CHUNK_TOKENS);
    if token_count == 0 {
        return Vec::new();
    }

    let mut spans = Vec::with_capacity(token_count.div_ceil(chunk));
    let mut offset = 0usize;
    while offset < token_count {
        let count = chunk.min(token_count - offset);
        let completed = offset + count;
        spans.push(PrefillChunkSpan {
            token_offset: offset,
            token_count: count,
            start_position: start_position + offset,
            completed_count: completed,
        });
        offset = completed;
    }
    spans
}

/// Dirty/in-flight tracking for a chunked-prefill runner: whether it has
/// written KV rows for a chunk without yet committing them. A runner that
/// panics or errors mid-chunk must not be reused until `reset()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PrefillChunkCommitState {
    is_dirty: bool,
    in_flight_start_position: Option<usize>,
    in_flight_token_count: Option<usize>,
}

impl PrefillChunkCommitState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_dirty(&self) -> bool {
        self.is_dirty
    }

    pub fn in_flight_end_position(&self) -> Option<usize> {
        Some(self.in_flight_start_position? + self.in_flight_token_count?)
    }

    pub fn mark_dirty(&mut self, start_position: usize, token_count: usize) {
        assert!(
            token_count > 0,
            "prefill dirty token_count must be positive"
        );
        self.is_dirty = true;
        self.in_flight_start_position = Some(start_position);
        self.in_flight_token_count = Some(token_count);
    }

    pub fn mark_committed(&mut self) {
        self.is_dirty = false;
        self.in_flight_start_position = None;
        self.in_flight_token_count = None;
    }

    pub fn reset(&mut self) {
        self.mark_committed();
    }

    /// Errors with [`PrefillError::ChunkedRunnerDirty`] if a previous chunk
    /// was left uncommitted, naming `operation` and the in-flight range in
    /// the message.
    pub fn require_clean(&self, operation: &str) -> Result<(), PrefillError> {
        if !self.is_dirty {
            return Ok(());
        }
        let range = match (self.in_flight_start_position, self.in_flight_end_position()) {
            (Some(start), Some(end)) => format!(" for in-flight chunk [{start}, {end})"),
            _ => String::new(),
        };
        Err(PrefillError::ChunkedRunnerDirty(format!(
            "{operation} rejected because a previous chunked prefill wrote KV rows{range} but did not commit; call reset() before reusing the runner"
        )))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillMode {
    Off,
    Chunked,
}

/// The prompt-processing mode and chunk size a chunked prefill runner uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillRuntimeConfig {
    pub mode: PrefillMode,
    pub chunk_tokens: usize,
}

impl PrefillRuntimeConfig {
    /// Largest supported prefill chunk. Chunked prefill re-reads each
    /// layer's routed experts once per chunk, so expert I/O scales with
    /// `prompt_tokens / chunk_tokens`; this is the practical ceiling before
    /// that re-read cost dominates.
    pub const MAX_CHUNK_TOKENS: usize = 4096;

    pub fn off() -> Self {
        Self {
            mode: PrefillMode::Off,
            chunk_tokens: 128,
        }
    }

    pub fn default_chunked() -> Self {
        Self::production(128).expect("128 is in ALLOWED_CHUNK_SIZES")
    }

    /// Builds a chunked config, rejecting a `chunk_tokens` outside the
    /// runtime-configuration allowed set.
    pub fn production(chunk_tokens: usize) -> Result<Self, PrefillError> {
        if !ALLOWED_CHUNK_SIZES.contains(&(chunk_tokens as u32)) {
            return Err(PrefillError::ChunkedUnsupported(
                "unsupported prefill chunk size".to_string(),
            ));
        }
        Ok(Self {
            mode: PrefillMode::Chunked,
            chunk_tokens,
        })
    }

    pub fn enabled(&self) -> bool {
        self.mode == PrefillMode::Chunked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_cover_the_whole_token_count_in_order() {
        let spans = prefill_chunk_spans(10, 0, 4);
        assert_eq!(spans.len(), 3);
        assert_eq!(
            spans[0],
            PrefillChunkSpan {
                token_offset: 0,
                token_count: 4,
                start_position: 0,
                completed_count: 4
            }
        );
        assert_eq!(
            spans[1],
            PrefillChunkSpan {
                token_offset: 4,
                token_count: 4,
                start_position: 4,
                completed_count: 8
            }
        );
        assert_eq!(
            spans[2],
            PrefillChunkSpan {
                token_offset: 8,
                token_count: 2,
                start_position: 8,
                completed_count: 10
            }
        );
    }

    #[test]
    fn spans_are_empty_for_zero_tokens() {
        assert!(prefill_chunk_spans(0, 0, 4).is_empty());
    }

    #[test]
    fn spans_at_a_nonzero_start_position_offset_start_position_only() {
        let spans = prefill_chunk_spans(3, 100, 8);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].start_position, 100);
        assert_eq!(spans[0].token_offset, 0);
    }

    #[test]
    fn exact_multiple_produces_no_remainder_chunk() {
        let spans = prefill_chunk_spans(8, 0, 4);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[1].token_count, 4);
    }

    #[test]
    fn commit_state_starts_clean() {
        let state = PrefillChunkCommitState::new();
        assert!(!state.is_dirty());
        assert!(state.require_clean("op").is_ok());
    }

    #[test]
    fn commit_state_reports_dirty_range_in_the_error_message() {
        let mut state = PrefillChunkCommitState::new();
        state.mark_dirty(10, 5);
        let err = state.require_clean("prefill").unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("wrote KV rows for in-flight chunk [10, 15)"),
            "message = {message}"
        );
        assert!(message.contains("prefill rejected"));
    }

    #[test]
    fn mark_committed_clears_dirty_state() {
        let mut state = PrefillChunkCommitState::new();
        state.mark_dirty(0, 4);
        state.mark_committed();
        assert!(!state.is_dirty());
        assert!(state.require_clean("op").is_ok());
    }

    #[test]
    fn production_rejects_a_disallowed_chunk_size() {
        assert!(PrefillRuntimeConfig::production(129).is_err());
    }

    #[test]
    fn production_accepts_an_allowed_chunk_size() {
        let config = PrefillRuntimeConfig::production(256).unwrap();
        assert!(config.enabled());
        assert_eq!(config.chunk_tokens, 256);
    }
}
