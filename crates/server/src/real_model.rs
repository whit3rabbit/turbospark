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

use runtime::{
    run_raw_completion, run_raw_completion_speculative, GenerationConfig, LogitProducer,
    RateControl, RawDecodeProgress, RawDecodeResult, RealForwardRunner, RuntimeError,
};
use tokenizer::MfTokenizer;

use crate::model::ChatModel;

pub struct RealChatModel {
    tokenizer: MfTokenizer,
    runner: Mutex<RealForwardRunner>,
    /// The RESOLVED window and the arithmetic behind it, never the request.
    context: runtime::ContextPlan,
    vocab_size: usize,
    expert_cache_slots: usize,
    model_id: String,
    rate: RateControl,
    /// Whether this process may draft ahead, resolved ONCE at open against
    /// the install -- the drafter's state is allocated there and there is one
    /// runner per process, so it is no more per-request than the rate cap is.
    ///
    /// Resolved as though the request were deterministic, because the second
    /// input is not knowable at open: see [`Self::run_completion`].
    speculation: runtime::SpeculationPlan,
    /// Which drafter [`Self::speculation`] would drive, for the startup line.
    drafter: runtime::SpeculativeDrafter,
    /// Tool-call guardrails, resolved once at open like the rate cap.
    guardrails: crate::GuardrailConfig,
}

impl RealChatModel {
    /// Opens a `.gturbo` install, mirroring the CLI's `open_session`: the
    /// architecture comes from the install's own `manifest.json` and the
    /// tokenizer is expected to be bundled in the same directory.
    ///
    /// `expert_cache_slots` and `max_context` are both POLICIES rather than
    /// counts: `None` means `auto`, sized against this machine and this
    /// install at open. Read each back with [`Self::expert_cache_slots`] and
    /// [`Self::context_plan`] -- under `auto` the request says nothing about
    /// what was allocated.
    ///
    /// A context window too large for the machine is refused HERE, before
    /// the KV buffers are allocated, because `KvCacheManager::new` sizes
    /// every layer up front and its failure is a Metal allocation error with
    /// no number in it pointing back at the flag.
    pub fn open(
        model_dir: &Path,
        max_context: Option<u32>,
        expert_cache_slots: Option<u32>,
        rate: RateControl,
        speculation: runtime::Speculation,
        drafter: runtime::SpeculativeDrafter,
        guardrails: crate::GuardrailConfig,
    ) -> Result<Self, String> {
        let arch = repack::peek_manifest_arch(model_dir)?;
        let context = runtime::resolve_max_context(
            match max_context {
                Some(n) => runtime::MaxContext::Fixed(n),
                None => runtime::MaxContext::Auto,
            },
            &arch,
            repack::trained_context_meta::peek(model_dir),
            foundation::runtime_config::DEFAULT_MAX_CONTEXT,
            runtime::physical_memory(),
            runtime::committed_bytes(model_dir),
        )
        .map_err(|e| e.to_string())?;
        // A quality warning and never an error: RoPE extrapolates rather
        // than failing, and an install written before the trained context
        // was recorded declares none, so refusing would apply to some
        // installs and not others. A server runs unattended, so this goes
        // out at startup where an operator sees it once.
        if context.past_trained {
            eprintln!(
                "warning: max_context {} exceeds the checkpoint's trained context of {}; \
                 output quality degrades past that point",
                context.resolved,
                context.trained.unwrap_or(0)
            );
        }
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
        // THE DRAFTER IS RESOLVED BEFORE THE OPEN, exactly as `open_session`
        // does it: the policies name one drafter and the wrong one is
        // indistinguishable from an install with no drafter at all. Shared
        // with the CLI rather than reimplemented (`runtime::speculation_policy`)
        // -- two copies would name different causes the first time they
        // disagreed.
        let choice = runtime::resolve_drafter(drafter, model_dir);
        let runner = RealForwardRunner::open_with_slot_policy_and_speculation(
            model_dir,
            arch,
            context.resolved as usize,
            match expert_cache_slots {
                Some(n) => runtime::ExpertCacheSlots::Fixed(n as usize),
                None => runtime::ExpertCacheSlots::Auto,
            },
            runtime::draft_policies(&choice, speculation),
        )
        .map_err(|e| e.to_string())?;
        // The MODEL's padded head width, not the tokenizer dialect's
        // constant: two checkpoints can share a dialect and pad differently.
        let vocab_size = runner.vocab_size();
        let expert_cache_slots = runner.expert_cache_slots();
        // **RESOLVED AS THOUGH THE REQUEST WERE DETERMINISTIC, which is the
        // one place this server cannot follow the CLI's shape.** On the CLI
        // the whole process has one shaping, so `open_session` knows at open
        // whether acceptance can be exact. Here every request carries its own
        // temperature. What is fixed at open is the INSTALL half -- does it
        // carry a usable drafter -- so that is what is resolved here, and the
        // per-request half is applied in `run_completion`.
        let plan = runtime::resolve_speculation(
            speculation,
            choice.drafter,
            match choice.drafter {
                runtime::SpeculativeDrafter::Dflash => runner.dflash_speculation_blocker(),
                // The note outranks the engine's blocker where there is one,
                // for `crates/cli/CLAUDE.md` Gotcha 10's reason: both are true
                // of a DFlash2-only install and only one names a flag.
                _ => choice.note.clone().or_else(|| runner.speculation_blocker()),
            },
            true,
        )?;
        Ok(Self {
            tokenizer,
            runner: Mutex::new(runner),
            context,
            vocab_size,
            expert_cache_slots,
            model_id,
            rate,
            speculation: plan,
            drafter: choice.drafter,
            guardrails,
        })
    }

    /// What speculation this process resolved to, for the startup line.
    ///
    /// Reported rather than silent for the reason the CLI reports it: an
    /// install carrying a drafter and decoding one token at a time with
    /// nothing said is the failure the feature was built to end. A server
    /// says it once, at startup, where an operator sees it.
    pub fn speculation_line(&self) -> String {
        match &self.speculation {
            runtime::SpeculationPlan::Enabled { block } => {
                let which = match self.drafter {
                    runtime::SpeculativeDrafter::Dflash => "dflash2 (block drafter)",
                    _ => "mtp head (step drafter)",
                };
                format!(
                    "speculative decoding: on, {which}, block {block} \
                     (temperature-0 requests only)"
                )
            }
            runtime::SpeculationPlan::Disabled { reason: Some(why) } => {
                format!("speculative decoding: off ({why})")
            }
            runtime::SpeculationPlan::Disabled { reason: None } => {
                "speculative decoding: off".to_string()
            }
        }
    }

    /// The per-layer routed-expert slot count the runner actually opened
    /// with, for the startup line to report.
    pub fn expert_cache_slots(&self) -> usize {
        self.expert_cache_slots
    }

    /// The resolved context window and the arithmetic behind it, for the
    /// startup line. Under `auto` the request carries no number, so a line
    /// echoing the argument would describe nothing.
    pub fn context_plan(&self) -> &runtime::ContextPlan {
        &self.context
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
        self.context.resolved
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

    fn guardrails(&self) -> crate::GuardrailConfig {
        self.guardrails
    }

    /// The speculative loop when this process resolved one AND this request
    /// can be served by it; the sequential loop otherwise.
    ///
    /// **THE SECOND CONDITION IS PER REQUEST, AND THAT IS THE ONE THING THIS
    /// SERVER CANNOT INHERIT FROM THE CLI.** Acceptance is
    /// `argmax(target) == proposal`, exact only at temperature 0. On the CLI
    /// that is a property of the process, so `open_session` can refuse once
    /// and be done. Here it is a property of the REQUEST, so the check has to
    /// be made per call and the answer differs between two requests to one
    /// server.
    ///
    /// A sampled request falls back SILENTLY rather than failing. It is the
    /// normal case -- OpenAI and Anthropic clients send a non-zero temperature
    /// by default, so a server started with `--speculative` speculates on a
    /// minority of its traffic -- and a per-request warning for the normal
    /// case is noise that trains an operator to ignore the startup line that
    /// matters. Refusing would be worse still: it turns a valid request into
    /// an error for a setting the caller never sent.
    fn run_completion(
        &self,
        prompt_ids: &[foundation::TokenId],
        config: &GenerationConfig,
        on_progress: &mut dyn FnMut(RawDecodeProgress),
    ) -> Result<RawDecodeResult, RuntimeError> {
        let block = match &self.speculation {
            runtime::SpeculationPlan::Enabled { block } if config.shaping.is_deterministic() => {
                *block
            }
            _ => {
                return self.with_producer(&mut |producer| {
                    run_raw_completion(
                        producer,
                        &self.tokenizer,
                        prompt_ids,
                        config,
                        self.context.resolved,
                        self.vocab_size,
                        &mut *on_progress,
                    )
                })
            }
        };
        // The CONCRETE runner, which is the whole reason this override exists:
        // `SpeculativeProducer` has an associated type and cannot be reached
        // through the `&mut dyn LogitProducer` `with_producer` lends.
        let mut runner = self
            .runner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        run_raw_completion_speculative(
            &mut *runner,
            &self.tokenizer,
            prompt_ids,
            config,
            self.context.resolved,
            self.vocab_size,
            block,
            &mut *on_progress,
        )
    }
}
