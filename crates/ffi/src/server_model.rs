//! `turbospark_server::ChatModel` over this crate's own `SessionCore`, so
//! `ts_server_start` can serve one already-open model over HTTP without
//! opening a second `RealForwardRunner` for it (`session.rs`'s module doc).
//!
//! **THIS IS A SECOND COPY OF `RealChatModel::run_completion`'s DISPATCH,
//! AND THAT IS A DELIBERATE, SCOPED-DOWN CHOICE, NOT AN OVERSIGHT.** The
//! server's `RealChatModel` (`crates/server/src/real_model.rs`) additionally
//! carries vision (image encode + injection under one lock) and tool-call
//! guardrails; this adapter carries neither. A GUI session opened through
//! `ts_session_open` has no vision wiring reachable from this crate's own
//! `generate.rs` either (images are a server-only feature, ROADMAP M-V8, and
//! the FFI surface was never extended to take one), so an in-process server
//! built on it would be lying about a capability it cannot actually serve --
//! `run_completion` refuses an image request BY NAME for that reason, the
//! same shape `ChatModel`'s own default impl uses. Guardrails default to
//! whatever `ChatModel::guardrails()`'s trait default is (on); nothing here
//! overrides it, matching `ScriptedChatModel`.
//!
//! Speculation and chunked prefill ARE replicated, because both are already
//! live on this crate's own `generate.rs` dispatch (`Engine::Real` there) and
//! skipping either here would make the in-process server slower than the
//! same session's direct `ts_generate` calls for no reason a caller could
//! see.

use std::sync::Arc;

use runtime::{
    run_raw_completion_cancellable, CancelFlag, GenerationConfig, LogitProducer, RawDecodeProgress,
    RawDecodeResult, RuntimeError,
};
#[cfg(target_os = "macos")]
use runtime::{run_raw_completion_chunked_cancellable, run_raw_completion_speculative_cancellable};
use tokenizer::MfTokenizer;
use turbospark_server::ChatModel;

use crate::session::{Engine, SessionCore};

pub(crate) struct FfiChatModel {
    core: Arc<SessionCore>,
    model_id: String,
}

impl FfiChatModel {
    pub(crate) fn new(core: Arc<SessionCore>, model_id: String) -> Self {
        Self { core, model_id }
    }

    /// The MODEL's padded head width, never the tokenizer dialect's
    /// constant (AGENTS.md Gotcha 37): two checkpoints can share a dialect
    /// and pad differently. Matches `generate.rs`'s own read of this.
    fn vocab_size(&self) -> usize {
        let engine = self
            .core
            .engine
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match &*engine {
            #[cfg(target_os = "macos")]
            Engine::Real(runner) => runner.vocab_size(),
            Engine::Scripted(_) => self.core.info.vocab_size,
        }
    }
}

impl ChatModel for FfiChatModel {
    fn tokenizer(&self) -> &MfTokenizer {
        &self.core.tokenizer
    }

    fn vocab_size(&self) -> usize {
        FfiChatModel::vocab_size(self)
    }

    fn max_context(&self) -> u32 {
        self.core.max_context
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn rate_control(&self) -> runtime::RateControl {
        self.core.rate
    }

    fn with_producer(
        &self,
        f: &mut dyn FnMut(&mut dyn LogitProducer) -> Result<RawDecodeResult, RuntimeError>,
    ) -> Result<RawDecodeResult, RuntimeError> {
        let mut engine = self
            .core
            .engine
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match &mut *engine {
            #[cfg(target_os = "macos")]
            Engine::Real(runner) => f(runner.as_mut()),
            Engine::Scripted(producer) => f(producer.as_mut()),
        }
    }

    /// The speculative loop when this session resolved a block AND the
    /// request is deterministic (acceptance is exact only at temperature 0,
    /// same per-turn gate as `generate.rs::turn_block`); chunked prefill
    /// when the install's family supports it and neither of the above
    /// applies; the sequential loop otherwise.
    fn run_completion(
        &self,
        prompt_ids: &[foundation::TokenId],
        config: &GenerationConfig,
        images: Option<&turbospark_server::vision::RequestImages>,
        cancel: CancelFlag<'_>,
        on_progress: &mut dyn FnMut(RawDecodeProgress),
    ) -> Result<RawDecodeResult, RuntimeError> {
        if images.is_some() {
            return Err(RuntimeError::Producer(
                "this in-process server was started from an FFI session, which carries no \
                 vision wiring; send text-only requests, or run the standalone \
                 turbospark-server binary against an install with a vision tower"
                    .to_string(),
            ));
        }
        let vocab_size = self.vocab_size();
        let block = self
            .core
            .speculation_block
            .filter(|_| config.shaping.is_deterministic());
        let mut engine = self
            .core
            .engine
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match (&mut *engine, block) {
            #[cfg(target_os = "macos")]
            (Engine::Real(runner), Some(block)) => run_raw_completion_speculative_cancellable(
                runner.as_mut(),
                &self.core.tokenizer,
                prompt_ids,
                config,
                self.core.max_context,
                vocab_size,
                block,
                cancel,
                on_progress,
            ),
            #[cfg(target_os = "macos")]
            (Engine::Real(runner), None) => {
                if runner.supports_chunked_prefill() {
                    run_raw_completion_chunked_cancellable(
                        runner.as_mut(),
                        &self.core.tokenizer,
                        prompt_ids,
                        config,
                        self.core.max_context,
                        vocab_size,
                        foundation::DEFAULT_CHUNK_SIZE as usize,
                        cancel,
                        on_progress,
                    )
                } else {
                    run_raw_completion_cancellable(
                        runner.as_mut(),
                        &self.core.tokenizer,
                        prompt_ids,
                        config,
                        self.core.max_context,
                        vocab_size,
                        cancel,
                        on_progress,
                    )
                }
            }
            // A scripted producer implements no drafter and no chunked
            // prefill, so `block` is always `None` and this is a plain
            // wildcard rather than a case this could get wrong.
            (Engine::Scripted(producer), _) => run_raw_completion_cancellable(
                producer.as_mut(),
                &self.core.tokenizer,
                prompt_ids,
                config,
                self.core.max_context,
                vocab_size,
                cancel,
                on_progress,
            ),
        }
    }
}
