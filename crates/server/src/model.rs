//! The pluggable generation backend a request is served from.
//!
//! No trained `.gturbo` weights are available to this port (see
//! `mrefrust-runtime`'s module docs), so [`ScriptedChatModel`] is the only
//! implementation today: it always drives the raw-completion loop with a
//! fixed, pre-scripted logit sequence. It exists to prove the HTTP
//! request/response envelopes, chat templating, and SSE streaming framing
//! all work end to end against a real (if not real-weights-backed) server.
//! A real backend implements the same [`ChatModel`] trait once a forward
//! pass exists to back it.

use runtime::LogitProducer;
use tokenizer::MfTokenizer;

pub trait ChatModel: Send + Sync {
    fn tokenizer(&self) -> &MfTokenizer;
    fn vocab_size(&self) -> usize;
    fn max_context(&self) -> u32;
    fn new_producer(&self) -> Box<dyn LogitProducer + Send>;
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

    fn new_producer(&self) -> Box<dyn LogitProducer + Send> {
        Box::new(runtime::ScriptedLogitProducer::new(self.steps.clone()))
    }
}
