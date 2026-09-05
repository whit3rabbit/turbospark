//! `turbospark_server::ChatModel` over this crate's own `SessionCore`, so
//! `ts_server_start` can serve one already-open model over HTTP without
//! opening a second `RealForwardRunner` for it (`session.rs`'s module doc).
//!
//! **THIS IS A SECOND COPY OF `RealChatModel::run_completion`'s DISPATCH,
//! AND THAT IS A DELIBERATE, SCOPED-DOWN CHOICE, NOT AN OVERSIGHT.** The
//! server's `RealChatModel` (`crates/server/src/real_model.rs`) is the
//! reference for every arm here.
//!
//! **VISION IS SERVED SINCE `ts_generate` GREW IMAGE CONTENT PARTS.** This
//! paragraph used to say the opposite, and it was true when written: images
//! were a server-only feature and the FFI surface took none, so an
//! in-process server over an FFI session would have been claiming a
//! capability nothing behind it could reach. `crate::vision` is that wiring
//! now, `run_with_images` mirrors `RealChatModel`'s one-lock encode, and
//! `vision()` reports what the install would accept. The refusals that
//! remain are honest ones: a SCRIPTED session has no tower to run, and off
//! macOS there is no runner at all.
//!
//! **GUARDRAILS ARE RESOLVED AT `ts_server_start` AND CARRIED PER MODEL.**
//! This paragraph used to say "nothing here overrides" the
//! `ChatModel::guardrails()` trait default, which was true and was the reason
//! a host with its own guardrails setting could not make the SERVED path
//! agree with it: the setting looked global and reached only whatever the
//! host did with a reply itself. `ts_server_start`'s `guardrails` option is
//! the override, taking the same `on` | `off` grammar and the same
//! process-level scope `turbospark-server --guardrails` has, and absent it
//! the trait default (on) still applies.
//!
//! Speculation and chunked prefill ARE replicated, because both are already
//! live on this crate's own `generate` dispatch (`Engine::Real` there) and
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
    guardrails: turbospark_server::GuardrailConfig,
}

impl FfiChatModel {
    pub(crate) fn new(
        core: Arc<SessionCore>,
        model_id: String,
        guardrails: turbospark_server::GuardrailConfig,
    ) -> Self {
        Self {
            core,
            model_id,
            guardrails,
        }
    }

    /// The MODEL's padded head width, never the tokenizer dialect's
    /// constant (AGENTS.md Gotcha 37): two checkpoints can share a dialect
    /// and pad differently. Matches `generate`'s own read of this.
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

#[cfg(target_os = "macos")]
impl FfiChatModel {
    /// Encode the images, inject them, and generate -- all under ONE lock.
    ///
    /// **The single lock is the whole point**, and it is `RealChatModel`'s
    /// reasoning rather than a copy of its code: `set_prompt_vision` and the
    /// generation are two mutations of one runner, so a concurrent request
    /// landing between them overwrites the map and the first generation
    /// prefills the second's picture. Both answer fluently, and no test of
    /// either request alone can see it.
    ///
    /// The map is CLEARED at the END whether the generation succeeded or not,
    /// which is the contract M-V7 established the hard way: `reset()` at the
    /// loop's entry would land on the map for the prompt about to be
    /// prefilled (`crates/cli` Gotcha 13).
    ///
    /// Speculation and chunked prefill are BOTH skipped here, matching the
    /// standalone server: the qwen family this tower belongs to serves
    /// neither, so composing them would be untested code on an unreachable
    /// path.
    fn run_with_images(
        &self,
        prompt_ids: &[foundation::TokenId],
        config: &GenerationConfig,
        images: &turbospark_server::vision::RequestImages,
        cancel: CancelFlag<'_>,
        on_progress: &mut dyn FnMut(RawDecodeProgress),
    ) -> Result<RawDecodeResult, RuntimeError> {
        let vocab_size = self.vocab_size();
        let mut engine = self
            .core
            .engine
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Engine::Real(runner) = &mut *engine else {
            return Err(RuntimeError::Producer(
                crate::generate::SCRIPTED_HAS_NO_TOWER.to_string(),
            ));
        };
        let params = crate::vision::preprocess_params(runner, &self.core.model_dir)
            .map_err(RuntimeError::Producer)?;

        let mut embeddings = Vec::with_capacity(images.images.len());
        for (i, image) in images.images.iter().enumerate() {
            embeddings.push(
                runner
                    .encode_image(image, &params)
                    .map_err(|e| RuntimeError::Producer(format!("image {i}: {e}")))?,
            );
        }
        runner
            .set_prompt_vision(&embeddings, &images.positions, prompt_ids.len())
            .map_err(|e| RuntimeError::Producer(e.to_string()))?;

        let result = run_raw_completion_cancellable(
            runner.as_mut(),
            &self.core.tokenizer,
            prompt_ids,
            config,
            self.core.max_context,
            vocab_size,
            cancel,
            on_progress,
        );
        runner.clear_prompt_vision();
        result
    }
}

impl ChatModel for FfiChatModel {
    fn tokenizer(&self) -> &MfTokenizer {
        &self.core.tokenizer
    }

    /// What this install would accept, or `None` when it would accept
    /// nothing.
    ///
    /// Gated on `info.vision.active`, which is resolved at open and already
    /// means "an image would be SERVED" rather than "a tower exists" -- so
    /// this method and `run_with_images` cannot disagree about whether the
    /// preprocessing config is readable.
    fn vision(&self) -> Option<turbospark_server::vision::VisionInfo> {
        #[cfg(target_os = "macos")]
        {
            if !self.core.info.vision.active {
                return None;
            }
            let engine = self
                .core
                .engine
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let Engine::Real(runner) = &*engine else {
                return None;
            };
            let params = crate::vision::preprocess_params(runner, &self.core.model_dir).ok()?;
            Some(turbospark_server::vision::VisionInfo {
                params,
                specials: crate::vision::special_ids(runner),
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            None
        }
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

    /// What `ts_server_start` resolved, not the trait default.
    fn guardrails(&self) -> turbospark_server::GuardrailConfig {
        self.guardrails
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
    /// same per-turn gate as `generate::turn_block`); chunked prefill
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
        #[cfg(target_os = "macos")]
        if let Some(images) = images {
            return self.run_with_images(prompt_ids, config, images, cancel, on_progress);
        }
        #[cfg(not(target_os = "macos"))]
        if images.is_some() {
            return Err(RuntimeError::Producer(
                "the engine is macOS-only, so no vision tower can run here".to_string(),
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
