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
    run_raw_completion_cancellable, CancelFlag, GenerationConfig, LogitProducer, RawDecodeProgress,
    RawDecodeResult, RuntimeError,
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
    /// This install's vision configuration, or `None` where it has no tower.
    ///
    /// Read off the INSTALL by the backend: the pixel budget and the special
    /// ids are per-checkpoint, and a constant in the handler would be
    /// AGENTS.md Gotcha 38's shape. `None` is what makes an image request a
    /// reported DROP rather than a silent one.
    fn vision(&self) -> Option<crate::vision::VisionInfo> {
        None
    }

    /// Runs one generation, and owns the choice of WHICH decode loop.
    ///
    /// **`images` CANNOT BE A SEPARATE CALL, and that is a concurrency
    /// property rather than a style choice.** A backend serializes on its one
    /// runner per call (Gotcha 1), so `set_prompt_vision` followed by
    /// `run_completion` would be two locks with a gap: a second request
    /// arriving in that gap overwrites the map, and the first generation then
    /// prefills the second's picture. Both are fluent. Passing the images
    /// here is what puts the encode, the injection and the decode inside ONE
    /// lock.
    ///
    /// `cancel` is polled once per prefill and decoded token by the runtime
    /// loop underneath; a caller with nothing to cancel on passes `&|| false`,
    /// which is the same statement sequence this ran before cancellation
    /// existed (`runtime::run_raw_completion`'s own `NEVER` constant, not
    /// exported past that crate, is exactly this).
    fn run_completion(
        &self,
        prompt_ids: &[foundation::TokenId],
        config: &GenerationConfig,
        images: Option<&crate::vision::RequestImages>,
        cancel: CancelFlag<'_>,
        on_progress: &mut dyn FnMut(RawDecodeProgress),
    ) -> Result<RawDecodeResult, RuntimeError> {
        if images.is_some() {
            // The DEFAULT backend is the scripted one, which has no tower and
            // no runner to encode with. Refused BY NAME rather than dropped:
            // a caller who sent a picture and got a text answer would read it
            // as the model ignoring the image.
            return Err(RuntimeError::Producer(
                "this backend cannot encode images; run the server against a .gturbo install \
                 whose checkpoint carries a vision tower"
                    .to_string(),
            ));
        }
        self.with_producer(&mut |producer| {
            run_raw_completion_cancellable(
                producer,
                self.tokenizer(),
                prompt_ids,
                config,
                self.max_context(),
                self.vocab_size(),
                cancel,
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

    /// Default reasoning effort level when a request omits `reasoning_effort`.
    fn default_reasoning(&self) -> tokenizer::ReasoningEffort {
        tokenizer::ReasoningEffort::Off
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
    default_reasoning: tokenizer::ReasoningEffort,
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
            default_reasoning: tokenizer::ReasoningEffort::Off,
        }
    }

    pub fn with_default_reasoning(mut self, reasoning: tokenizer::ReasoningEffort) -> Self {
        self.default_reasoning = reasoning;
        self
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

    fn default_reasoning(&self) -> tokenizer::ReasoningEffort {
        self.default_reasoning
    }

    fn with_producer(
        &self,
        f: &mut dyn FnMut(&mut dyn LogitProducer) -> Result<RawDecodeResult, RuntimeError>,
    ) -> Result<RawDecodeResult, RuntimeError> {
        let mut producer = runtime::ScriptedLogitProducer::new(self.steps.clone());
        f(&mut producer)
    }
}
