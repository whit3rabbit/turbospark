//! Progress reporting and cancellation for a generation request.

/// A milestone reported to the callback of
/// [`Model::generate_text_with_progress`](super::Model::generate_text_with_progress).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Progress {
    /// The prompt was tokenized; the AR stage is about to start.
    Tokenized { prompt_tokens: usize },
    /// One AR frame was emitted. `target` is the requested frame count; an
    /// early end token can stop the stage before it is reached.
    ArFrame { emitted: usize, target: usize },
    /// A flow chunk is about to be denoised and decoded.
    FlowChunk { index: usize, total: usize },
    /// One residual codebook is about to run (frame zero is AR warmup).
    DepthStep {
        frame: usize,
        codebook: usize,
        total: usize,
    },
    /// One Euler update is about to run inside a flow chunk.
    FlowStep {
        chunk: usize,
        step: usize,
        total: usize,
    },
    /// A stereo channel's projection, upsampling block, or output is about to run.
    VocoderStage {
        chunk: usize,
        channel: usize,
        stage: usize,
        total: usize,
    },
}

/// What the progress callback asks the pipeline to do next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Control {
    Continue,
    /// Stop at the next safe point and return `Ok(None)`.
    Cancel,
}

pub(crate) fn ignore(_: Progress) -> Control {
    Control::Continue
}
