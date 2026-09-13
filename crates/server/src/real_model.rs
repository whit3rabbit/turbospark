//! The real generation backend: a live `RealForwardRunner` behind the
//! [`ChatModel`] trait. macOS only, for the same reason `crates/gpu` is.
//!
//! Concurrency contract: ONE runner per process. It owns a multi-gigabyte
//! resident mapping, the Metal pipelines, and a KV cache, so it is neither
//! cheap to open nor safe to share; generation is serialized on the runner
//! mutex, and ADMISSION to that mutex is ordered by the FIFO gate this
//! backend exposes through [`ChatModel::generation_queue`]
//! (`crate::queue`, ROADMAP P1 item 5): permits are granted in arrival
//! order, a queued request holds no blocking thread while it waits (the
//! wait is async and only the request that actually generates enters
//! `spawn_blocking`), and a request whose client disconnects while queued
//! releases without generating at all. So throughput is one request at a
//! time regardless, and a `--max-tokens-per-sec` cap lengthens the permit
//! hold in proportion, which is acceptable for the same reason: the queue
//! is already serial. A client that disconnects mid-stream DOES abort the
//! queued or in-flight generation it was waiting on --
//! `run_completion`'s `cancel` predicate, polled once per prefill and
//! decoded token, is what a waiter's dropped request sets
//! (`crates/server/CLAUDE.md` Gotcha 25) -- but the permit itself is still
//! held until that generation actually stops, so a cancel shortens the
//! wait rather than skipping the queue.

use std::path::Path;
use std::sync::Mutex;

use runtime::{
    run_raw_completion_cancellable, run_raw_completion_chunked_cancellable,
    run_raw_completion_speculative_cancellable, CancelFlag, GenerationConfig, LogitProducer,
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
    /// Default reasoning effort level for requests that do not specify one.
    default_reasoning: tokenizer::ReasoningEffort,
    /// Deployment-wide system prompt for requests that send none of their own.
    default_system: Option<String>,
    /// The checkpoint's own image preprocessing parameters, read from the
    /// install's `preprocessor_config.json` at open (ROADMAP M-V8).
    ///
    /// `None` on an install with no tower OR one whose sidecar is missing,
    /// and the two are deliberately the same answer HERE: both mean this
    /// server cannot serve an image, and the handler reports the drop either
    /// way. The startup line distinguishes them.
    preprocess_params: Option<turbospark_vision_io::PreprocessParams>,
    /// [`ChatModel::vision`]'s answer, resolved ONCE here rather than on
    /// every call. `has_vision_tower` and `vision_config` are both pure
    /// reads of `arch.vision` (`runtime::real_forward_api`), so the answer
    /// cannot change after open -- taking the runner mutex on every request
    /// to re-derive it bought nothing but a lock hold on the async executor.
    /// `plan` calls this for EVERY request, including a text-only one on a
    /// vision install, so before this cached, an image-carrying request
    /// (`count_tokens` included) parked a tokio worker for the whole
    /// generation holding the lock: enough of them pin every worker and
    /// `/health` stops answering, which Gotcha 22 says it must not.
    vision_info: Option<crate::vision::VisionInfo>,
    /// The FIFO admission gate this runner's generations queue on
    /// (ROADMAP P1 item 5). One per RUNNER: the registry could hold two
    /// real backends someday and each serializes on its own gate, not on a
    /// process-global one.
    queue: std::sync::Arc<crate::queue::GenerationQueue>,
}

impl RealChatModel {
    /// Locks the runner, recovering a poisoned mutex rather than propagating
    /// it -- every call site in this file used this same
    /// `unwrap_or_else(|poisoned| poisoned.into_inner())` line until now.
    ///
    /// **A panicking request can no longer be assumed to leave behind
    /// something the next request may safely inherit.** The one-line
    /// recovery was correct back when `run_raw_completion` unconditionally
    /// reset the producer at entry; since prefix reuse (`--prefix-reuse`,
    /// on by default) landed, `try_reuse_prefix` only resets when it finds
    /// no reusable prefix (`raw_completion.rs`), and a panic unwinds past
    /// both the success path that RECORDS a prefix and the failure path
    /// that TAINTS one. So a poisoned runner can carry a `kv_prefix` that
    /// looks valid but describes a generation that never finished, and the
    /// next request would try to extend it. Recovery therefore resets the
    /// runner explicitly rather than trusting whatever state the panic left
    /// behind. It also clears any pending vision map:
    /// `run_with_images`'s own clear is a plain statement after the
    /// generation call, so a panic between `set_prompt_vision` and that
    /// clear would otherwise leave the map installed for the next TEXT
    /// request (Gotcha 21's failure through a different door).
    fn locked_runner(&self) -> std::sync::MutexGuard<'_, RealForwardRunner> {
        match self.runner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                let mut guard = poisoned.into_inner();
                guard.reset();
                guard.clear_prompt_vision();
                guard
            }
        }
    }

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
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        model_dir: &Path,
        max_context: Option<u32>,
        expert_cache_slots: Option<u32>,
        rate: RateControl,
        speculation: runtime::Speculation,
        drafter: runtime::SpeculativeDrafter,
        guardrails: crate::GuardrailConfig,
        steering: runtime::SteeringPolicy,
        load_policy: runtime::LoadPolicy,
        default_reasoning: tokenizer::ReasoningEffort,
        default_system: Option<String>,
        prefix_reuse: bool,
        session_slots: u32,
        vision_sidecar: Option<&Path>,
        kv_bits: runtime::KvQuant,
    ) -> Result<Self, String> {
        let arch = repack::peek_manifest_arch(model_dir)?;
        // Captured this early because the open below consumes `arch`: the
        // `--vision-sidecar auto` resolution after the open needs the
        // trunk's own family and hidden size, and both are Copy.
        let trunk_family = arch.family;
        let trunk_hidden_size = arch.hidden_size;
        // Resolved here, ahead of `committed_breakdown` below, so the same
        // policy sizes both the context budget and the actual open.
        let expert_cache_slots_policy = match expert_cache_slots {
            Some(n) => runtime::ExpertCacheSlots::Fixed(n as usize),
            None => runtime::ExpertCacheSlots::Auto,
        };
        // Bound rather than computed inline: the vision pixel budget (Part
        // B3) needs the SAME committed-bytes figure `max_context` resolved
        // against, and the `--session-slots` refusal below already reads
        // this same install a second time -- one read, three consumers.
        // `committed_breakdown` resolves the slot cache to what THIS open
        // will actually request, not `committed_bytes`'s worst case, so a
        // `--load-guard custom` ceiling is checked against a real
        // allocation.
        let committed = runtime::committed_breakdown(
            model_dir,
            runtime::physical_memory(),
            expert_cache_slots_policy,
        );
        let context = runtime::resolve_max_context_with(
            match max_context {
                Some(n) => runtime::MaxContext::Fixed(n),
                None => runtime::MaxContext::Auto,
            },
            &arch,
            repack::trained_context_meta::peek(model_dir),
            foundation::runtime_config::DEFAULT_MAX_CONTEXT,
            runtime::physical_memory(),
            committed,
            &load_policy,
            kv_bits,
        )
        .map_err(|e| e.to_string())?;
        // Parked session caches are allocated by the runner at open, so they
        // are spoken for both by the admission check below and by the vision
        // scratch budget resolved after the runner opens.
        let pool_extra =
            runtime::session_pool_bytes_with(&arch, context.resolved, session_slots, kv_bits);
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
        // A `--session-slots` this machine cannot afford is refused HERE,
        // before `open_inner` allocates a single parked slot, for exactly
        // the reason `resolve_max_context` above already refuses an
        // oversized `--max-context`: the alternative is an unattributed
        // Metal allocation failure with no number in it pointing back at
        // the flag. Skipped under the same two conditions that check does
        // (`LoadGuard::Off`, or a machine `physical_memory()` cannot read),
        // since neither of those refuses `--max-context` either.
        if session_slots > 1 {
            let physical = runtime::physical_memory();
            let budget = load_policy.guard.budget();
            if physical > 0 && budget.refuses {
                let available = load_policy.guard.available(physical, committed.total());
                let needs = context.kv_bytes.saturating_add(pool_extra);
                if needs > available {
                    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
                    return Err(format!(
                        "--session-slots {session_slots} needs {:.1} GiB total ({:.1} GiB for \
                         the live session plus {:.1} GiB for {} parked slot{}); {:.1} GiB \
                         available ({:.1} GiB physical - weights, expert cache and reserve). \
                         Fewer slots or a smaller --max-context would fit.",
                        gib(needs),
                        gib(context.kv_bytes),
                        gib(pool_extra),
                        session_slots - 1,
                        if session_slots == 2 { "" } else { "s" },
                        gib(available),
                        gib(physical),
                    ));
                }
            }
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
        let mut runner = RealForwardRunner::open_with_kv_quant(
            model_dir,
            arch,
            context.resolved as usize,
            expert_cache_slots_policy,
            runtime::draft_policies(&choice, speculation),
            steering,
            session_slots as usize,
            kv_bits,
        )
        .map_err(|e| e.to_string())?;
        // A request continues from the previous request's KV wherever the
        // prompts agree, instead of re-prefilling the whole transcript
        // (`crates/runtime/CLAUDE.md` Gotcha 30). Set once at open, like every
        // other process-level policy here: there is one runner per process,
        // and this call is the ONLY place that runner is ever mutably touched
        // outside a request.
        runner.set_prefix_reuse(prefix_reuse);
        // Echoed on the startup line beside the speculation one. Steering
        // changes the TOKENS, so a server serving an edited model has to say
        // so where an operator reading the log will see it.
        if let Some(line) = runner.steering_line() {
            eprintln!("{line}");
        }
        // The MODEL's padded head width, not the tokenizer dialect's
        // constant: two checkpoints can share a dialect and pad differently.
        let vocab_size = runner.vocab_size();
        let expert_cache_slots = runner.expert_cache_slots();
        // Vision memory sidecar (Part A4): attach BEFORE computing
        // `preprocess_params` below -- Part A3 already routes that read
        // through `runner.vision_dir()` for exactly this sequence, so
        // attaching here is enough to make the rest of the pipeline pick up
        // the sidecar with no further change.
        //
        // `--vision-sidecar auto` resolves against the default store AFTER
        // the trunk is open, by the trunk's own family and hidden size. A
        // no-op (own tower already present) or an absent tower serves
        // text-only with the reason on the startup line; an ambiguity is
        // still a refused open, because the revision pin is load-bearing
        // and `auto` never picks between towers.
        //
        // `verify_image_markers` catches a mismatched sidecar/tokenizer
        // pairing HERE, naming the actual ids, rather than letting it
        // surface deep inside `turbospark_vision_io::splice_and_walk` at
        // the first image a client sends.
        let requested_auto = vision_sidecar == Some(std::path::Path::new("auto"));
        let sidecar_dir: Option<std::path::PathBuf> = if requested_auto {
            let store = catalog::Store::default_store()
                .map_err(|e| format!("--vision-sidecar auto: {e}"))?;
            match catalog::resolve_vision_sidecar_auto(
                &store,
                runner.has_vision_tower(),
                trunk_family,
                trunk_hidden_size,
            )
            .map_err(|e| format!("--vision-sidecar auto: {e}"))?
            {
                catalog::AutoVisionSidecar::Attached(dir) => Some(dir),
                catalog::AutoVisionSidecar::TextOnly(reason) => {
                    eprintln!("vision: auto -- {reason}");
                    None
                }
            }
        } else {
            vision_sidecar.map(std::path::Path::to_path_buf)
        };
        if let Some(dir) = &sidecar_dir {
            runner
                .attach_vision_sidecar(dir)
                .map_err(|e| format!("--vision-sidecar {}: {e}", dir.display()))?;
            let vision = runner.vision_config();
            tokenizer
                .verify_image_markers(
                    vision.vision_start_token_id as i32,
                    vision.image_token_id as i32,
                )
                .map_err(|e| format!("--vision-sidecar {}: {e}", dir.display()))?;
            // Print only when a sidecar was actually attached: most servers
            // run with none, and the guardrails/prefix-reuse lines' "always
            // print" shape exists for a fixed toggle every deployment sets
            // one way or the other, not for a flag whose default has
            // nothing to report.
            eprintln!(
                "vision: sidecar {}{}",
                dir.display(),
                if requested_auto { " (auto)" } else { "" }
            );
        }
        // Read from `runner.vision_dir()` rather than `model_dir` directly,
        // so a later `--vision-sidecar` attach (vision memory sidecar, Part
        // A4) finds `preprocessor_config.json` beside the sidecar's own
        // `manifest.json` -- reading it here, before `runner` moves into the
        // `Mutex` below, is what lets that method see whichever directory is
        // actually authoritative once one is attached.
        let mut preprocess_params =
            std::fs::read_to_string(runner.vision_dir().join("preprocessor_config.json"))
                .ok()
                .and_then(|json| {
                    turbospark_vision_io::PreprocessParams::from_preprocessor_config_json(&json)
                        .ok()
                });
        // Vision memory sidecar (Part B3): clamp the checkpoint's own
        // declared `max_pixels` ceiling to what THIS process's own
        // `--load-guard` tier can afford on top of the KV cache and
        // resident weights already committed above -- the SAME tier
        // `--max-context` resolved against, never a second one
        // (`crates/ffi/CLAUDE.md` Gotcha 12's rule). Resolved ONCE here,
        // like `preprocess_params` itself, and printed only when it
        // actually moved something -- matching this crate's other
        // resolved-at-open lines (the sidecar-attach line just above,
        // the guardrails/prefix-reuse lines) that stay silent on the
        // common case.
        // A budget refusal (not even `min_pixels` fits) degrades to "this
        // server cannot serve an image" rather than failing the whole open,
        // matching the `.ok()` chain just above: a text-only request must
        // still work on an install whose vision config the machine cannot
        // afford. Reported on stderr either way, since a silently-dropped
        // capability is worse here than the sidecar/parse failures the
        // `.ok()` chain already swallows without a line of its own.
        if let Some(params) = preprocess_params.as_mut() {
            let declared_max_pixels = params.max_pixels;
            match runner.resolve_vision_pixel_budget(
                declared_max_pixels,
                params.min_pixels,
                load_policy.guard,
                runtime::physical_memory(),
                committed.total().saturating_add(pool_extra),
                context.kv_bytes,
            ) {
                Ok(budget) if budget.clamped => {
                    params.max_pixels = budget.resolved_max_pixels;
                    eprintln!(
                        "vision: max_pixels {declared_max_pixels} -> {} (memory budget, {})",
                        budget.resolved_max_pixels,
                        load_policy.guard.as_str(),
                    );
                }
                Ok(_) => {}
                Err(e) => {
                    eprintln!("vision: disabled ({e})");
                    preprocess_params = None;
                }
            }
        }
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
        // Resolved HERE, before `runner` moves into the `Mutex` below, so
        // `ChatModel::vision` never has to lock for it: `has_vision_tower`
        // and `vision_config` are pure reads of a field this checkpoint
        // fixed at load, and `preprocess_params` is already final by this
        // point (the budget clamp above is the last thing that can change
        // it).
        let vision_info = if runner.has_vision_tower() {
            let v = runner.vision_config();
            preprocess_params
                .clone()
                .map(|params| crate::vision::VisionInfo {
                    params,
                    specials: turbospark_vision_io::VisionSpecialIds {
                        vision_start: v.vision_start_token_id as i32,
                        image_pad: v.image_token_id as i32,
                    },
                })
        } else {
            None
        };
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
            default_reasoning,
            default_system,
            preprocess_params,
            vision_info,
            queue: crate::queue::GenerationQueue::shared(),
        })
    }

    /// Encode the images, inject them, and generate -- all under ONE lock.
    ///
    /// **The single lock is the whole point** (see `run_completion`'s doc).
    /// `set_prompt_vision` and the generation are two mutations of one
    /// runner, and a concurrent request landing between them would overwrite
    /// the map: the first generation then prefills the second's picture, with
    /// both requests answering fluently about the wrong thing.
    ///
    /// The map is CLEARED at the end rather than at the start, which is the
    /// contract M-V7 established the hard way: `run_raw_completion` resets the
    /// producer at ENTRY, so a clear there lands on the map for the prompt
    /// about to be prefilled (`crates/runtime/src/real_forward_traits.rs`).
    ///
    /// Speculation is skipped here, deliberately: no published conversion of
    /// a vision family carries an ingestible drafter, so composing the two
    /// would be untested code on an unreachable path. Chunked prefill is NOT
    /// skipped: the chunked driver handles vision rows (its dense-qwen
    /// prefill mirrors this arm's tower blit and mRoPE dispatch), so an
    /// image prompt takes the same `supports_chunked_prefill` check and the
    /// same `DEFAULT_CHUNK_SIZE` the text arm uses. The seam difference is
    /// granularity only -- prefill cancellation lands on a chunk boundary
    /// (still honored) and `Prefill` progress coarsens to one event per
    /// chunk -- and the exec loop ignores `Prefill` events anyway.
    fn run_with_images(
        &self,
        prompt_ids: &[foundation::TokenId],
        config: &GenerationConfig,
        images: &crate::vision::RequestImages,
        cancel: CancelFlag<'_>,
        on_progress: &mut dyn FnMut(RawDecodeProgress),
    ) -> Result<RawDecodeResult, RuntimeError> {
        let Some(params) = self.preprocess_params.clone() else {
            return Err(RuntimeError::Producer(
                "this install declares no image preprocessing config; re-stream it with its \
                 sidecars"
                    .to_string(),
            ));
        };
        let mut runner = self.locked_runner();

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

        let result = if runner.supports_chunked_prefill() {
            run_raw_completion_chunked_cancellable(
                &mut *runner,
                &self.tokenizer,
                prompt_ids,
                config,
                self.context.resolved,
                self.vocab_size,
                foundation::DEFAULT_CHUNK_SIZE as usize,
                cancel,
                &mut *on_progress,
            )
        } else {
            run_raw_completion_cancellable(
                &mut *runner,
                &self.tokenizer,
                prompt_ids,
                config,
                self.context.resolved,
                self.vocab_size,
                cancel,
                &mut *on_progress,
            )
        };
        // CONSUMED, whether the generation succeeded or not: a map left
        // behind would apply to whatever text request arrives next.
        runner.clear_prompt_vision();
        result
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
                    runtime::SpeculativeDrafter::Dflash => {
                        "dflash2 (block drafter, temperature-0 requests only)"
                    }
                    _ => "mtp head (step drafter, any temperature)",
                };
                format!("speculative decoding: on, {which}, block {block}")
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

    /// How many distinct sessions this runner's pool holds reusable KV/
    /// recurrent state for at once, for the startup line. `1` unless
    /// `--session-slots` asked for more.
    pub fn session_pool_size(&self) -> usize {
        self.locked_runner().session_pool_size()
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
        // Poison recovery lives in `locked_runner`: it resets the runner
        // rather than trusting whatever a panicking request left behind (see
        // that method's doc for why the old "the next request starts clean"
        // reasoning stopped holding once prefix reuse landed).
        let mut runner = self.locked_runner();
        f(&mut *runner)
    }

    fn rate_control(&self) -> RateControl {
        self.rate
    }

    fn generation_queue(&self) -> Option<std::sync::Arc<crate::queue::GenerationQueue>> {
        Some(self.queue.clone())
    }

    fn guardrails(&self) -> crate::GuardrailConfig {
        self.guardrails
    }

    fn default_reasoning(&self) -> tokenizer::ReasoningEffort {
        self.default_reasoning
    }

    fn default_system(&self) -> Option<&str> {
        self.default_system.as_deref()
    }

    fn vision(&self) -> Option<crate::vision::VisionInfo> {
        // Resolved once at `open`, never re-derived: see `vision_info`'s own
        // doc for why taking the runner lock here was both unnecessary and
        // a liveness hazard (`/health` must answer while this process is
        // mid-generation, per Gotcha 22).
        self.vision_info.clone()
    }

    /// The speculative loop when this process resolved one AND this request
    /// can be served by it; the sequential loop otherwise.
    ///
    /// **THE SECOND CONDITION IS PER REQUEST, AND THAT IS THE ONE THING THIS
    /// SERVER CANNOT INHERIT FROM THE CLI.** Whether the speculative loop
    /// can serve a request exactly is a property of the request's
    /// temperature AND of the drafter: the MTP step drafter serves any
    /// temperature through exact rejection sampling (ROADMAP P1 item 4),
    /// while the DFlash2 block drafter has no distribution to ratio against
    /// and remains temperature-0 only. On the CLI that split is a property
    /// of the process, so `open_session` can resolve it once. Here the
    /// temperature arrives per request, so the check is made per call.
    ///
    /// A sampled request on the BLOCK drafter falls back SILENTLY rather
    /// than failing, and a sampled request on the STEP drafter speculates.
    /// The silent half is the normal case for a dflash server -- OpenAI and
    /// Anthropic clients send a non-zero temperature by default -- and a
    /// per-request warning for the normal case is noise that trains an
    /// operator to ignore the startup line that matters. Refusing would be
    /// worse still: it turns a valid request into an error for a setting
    /// the caller never sent.
    fn run_completion(
        &self,
        prompt_ids: &[foundation::TokenId],
        config: &GenerationConfig,
        images: Option<&crate::vision::RequestImages>,
        cancel: CancelFlag<'_>,
        on_progress: &mut dyn FnMut(RawDecodeProgress),
    ) -> Result<RawDecodeResult, RuntimeError> {
        // IMAGES FIRST, AND UNDER THE SAME LOCK AS THE GENERATION. Encoding
        // through the tower and then generating in two separately-locked calls
        // leaves a gap a concurrent request can land in, overwriting the map
        // between them -- so the first generation would prefill the second
        // request's picture, fluently. `run_with_images` takes the lock once
        // and holds it across the encode, the injection and the decode.
        if let Some(images) = images {
            return self.run_with_images(prompt_ids, config, images, cancel, on_progress);
        }
        let block = match &self.speculation {
            // Greedy requests speculate under either drafter; sampled ones
            // only under the STEP drafter (see the doc above for why the
            // block drafter cannot serve them exactly).
            runtime::SpeculationPlan::Enabled { block }
                if config.shaping.is_deterministic()
                    || self.drafter == runtime::SpeculativeDrafter::Mtp =>
            {
                *block
            }
            // Chunked prefill wins over the sequential loop whenever this
            // install's family can serve it (`supports_chunked_prefill`,
            // the SAME predicate the CLI's default `--prefill-chunk` wiring
            // checks). No per-request flag: prefill shape is a property of
            // the install rather than of a caller's prompt, matching the
            // rate cap, speculation and the guardrails toggle (Gotchas 10,
            // 17, 18). Decode's per-token progress callback is unaffected
            // either way, since only the PREFILL portion routes
            // differently, and speculation is checked first above -- the
            // two seams are not composable today, same as the CLI.
            _ => {
                let mut runner = self.locked_runner();
                if runner.supports_chunked_prefill() {
                    return run_raw_completion_chunked_cancellable(
                        &mut *runner,
                        &self.tokenizer,
                        prompt_ids,
                        config,
                        self.context.resolved,
                        self.vocab_size,
                        foundation::DEFAULT_CHUNK_SIZE as usize,
                        cancel,
                        &mut *on_progress,
                    );
                }
                drop(runner);
                return self.with_producer(&mut |producer| {
                    run_raw_completion_cancellable(
                        producer,
                        &self.tokenizer,
                        prompt_ids,
                        config,
                        self.context.resolved,
                        self.vocab_size,
                        cancel,
                        &mut *on_progress,
                    )
                });
            }
        };
        // The CONCRETE runner, which is the whole reason this override exists:
        // `SpeculativeProducer` has an associated type and cannot be reached
        // through the `&mut dyn LogitProducer` `with_producer` lends.
        let mut runner = self.locked_runner();
        run_raw_completion_speculative_cancellable(
            &mut *runner,
            &self.tokenizer,
            prompt_ids,
            config,
            self.context.resolved,
            self.vocab_size,
            block,
            cancel,
            &mut *on_progress,
        )
    }
}
