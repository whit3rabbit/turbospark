//! The pluggable generation backend a request is served from.
//!
//! Two implementations: [`ScriptedChatModel`], which drives the
//! raw-completion loop with a fixed, pre-scripted logit sequence (portable,
//! and what the integration tests use to prove the HTTP envelopes, chat
//! templating, and SSE framing), and [`crate::RealChatModel`] (macOS only),
//! which drives a real `RealForwardRunner` forward pass.
//!
//! [`ChatModel::with_producer`] is a pass-through rather than a
//! `new_producer() -> Box<dyn LogitProducer>` factory on purpose: a real
//! runner costs a multi-gigabyte mmap plus a full Metal pipeline compile to
//! open, so there is exactly one per process and it is borrowed mutably for
//! the duration of a request, not handed out by value.

use runtime::{LogitProducer, RawDecodeResult, RuntimeError};
use tokenizer::MfTokenizer;

pub trait ChatModel: Send + Sync {
    fn tokenizer(&self) -> &MfTokenizer;
    fn vocab_size(&self) -> usize;
    fn max_context(&self) -> u32;

    /// The id `GET /v1/models` advertises. Requests do not have to match it:
    /// there is one backend per process, so whatever `model` a request names
    /// is echoed back rather than routed on.
    fn model_id(&self) -> &str;

    /// Lends a producer to `f` for one generation. Implementations may
    /// serialize concurrent calls; `run_raw_completion` resets the producer
    /// on entry, so a producer reused across calls carries no state over.
    fn with_producer(
        &self,
        f: &mut dyn FnMut(&mut dyn LogitProducer) -> Result<RawDecodeResult, RuntimeError>,
    ) -> Result<RawDecodeResult, RuntimeError>;
}

/// Always replays the same scripted logit sequence, regardless of the
/// prompt. `steps` must include one entry per prefill token plus one per
/// decode step the caller expects to reach before a stop condition.
pub struct ScriptedChatModel {
    tokenizer: MfTokenizer,
    vocab_size: usize,
    max_context: u32,
    steps: Vec<Vec<foundation::LogitValue>>,
}

impl ScriptedChatModel {
    pub fn new(
        tokenizer: MfTokenizer,
        max_context: u32,
        steps: Vec<Vec<foundation::LogitValue>>,
    ) -> Self {
        let vocab_size = tokenizer.vocab_size;
        Self {
            tokenizer,
            vocab_size,
            max_context,
            steps,
        }
    }
}

impl ChatModel for ScriptedChatModel {
    fn tokenizer(&self) -> &MfTokenizer {
        &self.tokenizer
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn max_context(&self) -> u32 {
        self.max_context
    }

    fn model_id(&self) -> &str {
        "scripted"
    }

    fn with_producer(
        &self,
        f: &mut dyn FnMut(&mut dyn LogitProducer) -> Result<RawDecodeResult, RuntimeError>,
    ) -> Result<RawDecodeResult, RuntimeError> {
        let mut producer = runtime::ScriptedLogitProducer::new(self.steps.clone());
        f(&mut producer)
    }
}
