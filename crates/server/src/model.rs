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

use runtime::{
    run_raw_completion, GenerationConfig, LogitProducer, RawDecodeProgress, RawDecodeResult,
    RuntimeError,
};
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

    /// Runs one generation, and owns the choice of WHICH decode loop.
    ///
    /// **A method rather than a call in `exec.rs`, because the speculative
    /// loop cannot be reached through [`Self::with_producer`] at all.**
    /// `run_raw_completion_speculative` is generic over
    /// `runtime::SpeculativeProducer`, which carries an associated
    /// `Checkpoint` type and so is not object-safe; `with_producer` hands out
    /// a `&mut dyn LogitProducer`. Only a backend holding the CONCRETE runner
    /// can call it, so the decision belongs to the backend.
    ///
    /// The default is the sequential loop through `with_producer`, which is
    /// byte for byte what every caller did before this method existed --
    /// `ScriptedChatModel` therefore needs no implementation and every
    /// integration test keeps the exact path it had.
    fn run_completion(
        &self,
        prompt_ids: &[foundation::TokenId],
        config: &GenerationConfig,
        on_progress: &mut dyn FnMut(RawDecodeProgress),
    ) -> Result<RawDecodeResult, RuntimeError> {
        self.with_producer(&mut |producer| {
            run_raw_completion(
                producer,
                self.tokenizer(),
                prompt_ids,
                config,
                self.max_context(),
                self.vocab_size(),
                &mut *on_progress,
            )
        })
    }

    /// The decode rate cap and thermal stepping this backend generates
    /// under (ROADMAP Phase P2). Process-level, not per request: there is
    /// one runner per process and a power setting is a property of the
    /// machine, not of a caller's prompt. The default is uncapped, which
    /// is what the scripted backend wants.
    fn rate_control(&self) -> runtime::RateControl {
        runtime::RateControl::default()
    }

    /// Which tool-call guardrails this backend generates under.
    ///
    /// Process-level for exactly the reason [`ChatModel::rate_control`] is,
    /// and NOT for a symmetry: whether a deployment repairs a malformed tool
    /// call is a property of the deployment, and a per-request field would let
    /// any client opt its own traffic out of it.
    ///
    /// The default is ON, which costs the scripted backend nothing: every
    /// guardrail is keyed on the request carrying tools, and a request with
    /// none takes the path it always took.
    fn guardrails(&self) -> crate::guardrails::GuardrailConfig {
        crate::guardrails::GuardrailConfig::default()
    }
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
