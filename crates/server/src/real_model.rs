//! The real generation backend: a live `RealForwardRunner` behind the
//! [`ChatModel`] trait. macOS only, for the same reason `crates/gpu` is.
//!
//! Concurrency contract: ONE runner per process. It owns a multi-gigabyte
//! resident mapping, the Metal pipelines, and a KV cache, so it is neither
//! cheap to open nor safe to share; a mutex serializes generation and
//! concurrent requests queue on it (each waiter holding a tokio blocking
//! thread). That is the right shape for a loopback single-user server, not
//! for a fleet one. Two known limitations follow from it: throughput is one
//! request at a time, and a client that disconnects mid-stream does not
//! abort generation -- the run finishes and only then releases the lock.
//! A `--max-tokens-per-sec` cap lengthens the lock hold in proportion,
//! which is acceptable for the same reason: the queue is already serial.

use std::path::Path;
use std::sync::Mutex;

use runtime::{LogitProducer, RateControl, RawDecodeResult, RealForwardRunner, RuntimeError};
use tokenizer::MfTokenizer;

use crate::model::ChatModel;

pub struct RealChatModel {
    tokenizer: MfTokenizer,
    runner: Mutex<RealForwardRunner>,
    max_context: u32,
    vocab_size: usize,
    expert_cache_slots: usize,
    model_id: String,
    rate: RateControl,
}

impl RealChatModel {
    /// Opens a `.gturbo` install, mirroring the CLI's `open_session`: the
    /// architecture comes from the install's own `manifest.json` and the
    /// tokenizer is expected to be bundled in the same directory.
    ///
    /// `expert_cache_slots` is a POLICY rather than a count: `None` means
    /// `auto`, sized against this machine and this install at open. Read the
    /// count back with [`Self::expert_cache_slots`] -- under `auto` the
    /// request says nothing about what was allocated.
    pub fn open(
        model_dir: &Path,
        max_context: u32,
        expert_cache_slots: Option<u32>,
        rate: RateControl,
    ) -> Result<Self, String> {
        let arch = repack::peek_manifest_arch(model_dir)?;
        let tokenizer = MfTokenizer::load_from_dir(model_dir).map_err(|e| {
            format!(
                "failed to load a tokenizer from {}: {e}",
                model_dir.display()
            )
        })?;
        // The install directory's own name is the advertised model id (e.g.
        // `gemma4.gturbo`). `manifest.json` carries no model name field to
        // read instead, and the full path is not something to publish.
        let model_id = model_dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "model".to_string());
        let runner = RealForwardRunner::open_with_slot_policy(
            model_dir,
            arch,
            max_context as usize,
            match expert_cache_slots {
                Some(n) => runtime::ExpertCacheSlots::Fixed(n as usize),
                None => runtime::ExpertCacheSlots::Auto,
            },
        )
        .map_err(|e| e.to_string())?;
        // The MODEL's padded head width, not the tokenizer dialect's
        // constant: two checkpoints can share a dialect and pad differently.
        let vocab_size = runner.vocab_size();
        let expert_cache_slots = runner.expert_cache_slots();
        Ok(Self {
            tokenizer,
            runner: Mutex::new(runner),
            max_context,
            vocab_size,
            expert_cache_slots,
            model_id,
            rate,
        })
    }

    /// The per-layer routed-expert slot count the runner actually opened
    /// with, for the startup line to report.
    pub fn expert_cache_slots(&self) -> usize {
        self.expert_cache_slots
    }
}

impl ChatModel for RealChatModel {
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
        &self.model_id
    }

    fn with_producer(
        &self,
        f: &mut dyn FnMut(&mut dyn LogitProducer) -> Result<RawDecodeResult, RuntimeError>,
    ) -> Result<RawDecodeResult, RuntimeError> {
        // Poison is recoverable here: a panicking request leaves the runner
        // with a stale KV cache at worst, and `run_raw_completion` resets the
        // producer before its first token, so the next request starts clean.
        let mut runner = self
            .runner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        f(&mut *runner)
    }

    fn rate_control(&self) -> RateControl {
        self.rate
    }
}
