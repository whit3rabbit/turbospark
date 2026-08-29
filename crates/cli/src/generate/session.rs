//! Generation session lifecycle and hardware setup.

use invocation::InvocationRequest;
use runtime::{
    resolve_drafter, resolve_speculation, DrafterChoice, RateControl, RealForwardRunner,
    SpeculationPlan,
};
use selection::ShapingConfig;
use tokenizer::MfTokenizer;

/// Everything a generating mode needs: the loaded model, its tokenizer, and
/// the validated sampling configuration. Opened once per process, reused by
/// every turn of an interactive chat.
pub(crate) struct Session {
    /// The RESOLVED install directory, kept because `--image` reads the
    /// checkpoint's own `preprocessor_config.json` out of it.
    ///
    /// Resolved rather than `request.model`: that may be a catalog ALIAS, and
    /// re-resolving it downstream is how the two come apart (Gotcha 5).
    pub(crate) model_dir: std::path::PathBuf,
    pub(crate) tokenizer: MfTokenizer,
    pub(crate) runner: RealForwardRunner,
    pub(crate) shaping: ShapingConfig,
    /// Resolved once at open, because resolving it per turn would let Low
    /// Power Mode toggling mid-chat change the pace for reasons the caller
    /// never asked about.
    pub(crate) rate: RateControl,
    /// The RESOLVED context window, never `request.max_context`.
    ///
    /// Under `auto` the request carries no number, and every consumer of
    /// this value -- the KV cache the runner already allocated, the
    /// admission check, the per-turn budget -- has to agree with what was
    /// allocated rather than with what was asked for. Reading the request
    /// downstream of `open_session` is how those two come apart.
    pub(crate) max_context: u32,
    /// Whether this session drafts ahead, resolved once at open against the
    /// install and the sampling settings. See [`resolve_speculation`].
    pub(crate) speculation: SpeculationPlan,
}

pub(crate) fn open_session(request: &InvocationRequest) -> Result<Session, String> {
    // `--model` takes a path OR a `turbospark-model` alias, resolved HERE
    // rather than in `invocation`, which is pure and whose contract keeps the
    // value an opaque string. A path that exists always wins: a bare name
    // that silently preferred an alias would run a DIFFERENT model than the
    // one on the command line, fluently and with no error.
    let resolved = catalog::resolve_model_arg(&request.model);
    let model_dir = resolved.as_path();
    let arch = repack::peek_manifest_arch(model_dir)?;

    let tokenizer = MfTokenizer::load_from_dir(model_dir).map_err(|e| {
        format!(
            "failed to load a tokenizer from {}: {e}",
            model_dir.display()
        )
    })?;

    // Resolve the context window BEFORE opening, because the failure this
    // catches is an allocation: `KvCacheManager::new` sizes every layer's K
    // and V buffers up front, so a window that does not fit is a Metal
    // allocation error or a swapping machine, with no number in either
    // pointing back at `--max-context`.
    //
    // The three inputs are the ones neither `crates/invocation` (pure) nor
    // the policy module (portable) may read for itself: the checkpoint's own
    // trained context out of the install, the mapped weight region, and this
    // machine's memory.
    let trained = repack::trained_context_meta::peek(model_dir);
    let plan = runtime::resolve_max_context(
        match request.max_context {
            invocation::MaxContext::Auto => runtime::MaxContext::Auto,
            invocation::MaxContext::Fixed(n) => runtime::MaxContext::Fixed(n),
        },
        &arch,
        trained,
        invocation::request::DEFAULT_MAX_CONTEXT,
        runtime::physical_memory(),
        runtime::committed_bytes(model_dir),
    )
    .map_err(|e| e.to_string())?;

    if !request.quiet {
        report_context(&plan, request.max_context);
    }
    // Past the checkpoint's trained context is a QUALITY warning and never an
    // error: RoPE extrapolates rather than failing, and an install written
    // before the trained context was recorded declares none at all, so
    // refusing would be enforced on some installs and not others.
    if plan.past_trained {
        eprintln!(
            "warning: --max-context {} exceeds the checkpoint's trained context of {}; \
             output quality degrades past that point",
            plan.resolved,
            plan.trained.unwrap_or(0)
        );
    }

    // Size the KV cache to the same bound the completion loop admits
    // against, rather than the runner's own 4096-token default, and honor
    // --expert-cache-slots instead of the runner's own fixed default.
    //
    // The slot POLICY crosses the boundary unresolved, exactly as
    // `--power-profile` does: `crates/invocation` is pure and may not read
    // the machine's memory or the install's expert stride, and both are
    // needed to size the cache.
    //
    // THE DRAFTER IS RESOLVED FIRST, because the policies below name exactly
    // one and the wrong one is indistinguishable from an install with no
    // drafter at all: pinned at `Mtp`, a DFlash2 install reported "carries no
    // multi-token-prediction head" and decoded sequentially with a working
    // drafter on disk. `Auto` reads the resident index, which is the index
    // region alone and not the weights.
    let asked = map_speculation(request.speculation);
    let choice = resolve_drafter(map_drafter(request.speculative_drafter), model_dir);
    let steering = resolve_steering(request)?;
    let runner = RealForwardRunner::open_with_slot_policy_speculation_and_steering(
        model_dir,
        arch,
        plan.resolved as usize,
        match request.expert_cache_slots {
            invocation::ExpertCacheSlots::Auto => runtime::ExpertCacheSlots::Auto,
            invocation::ExpertCacheSlots::Fixed(n) => runtime::ExpertCacheSlots::Fixed(n as usize),
        },
        runtime::draft_policies(&choice, asked),
        steering,
    )
    .map_err(|e| e.to_string())?;
    // Reported beside the resolved slot count, for that field's reason: an
    // edit applied to every token has to be readable next to any number
    // taken from the run.
    if let Some(line) = runner.steering_line() {
        eprintln!("{line}");
    }
    // The two presence flags are `draft_policies`' input above and nothing
    // this binary prints, so they are dropped by name rather than with a `..`
    // -- a wildcard here would silently swallow the next field somebody adds,
    // which is how `install_has_dflash` would have arrived unnoticed.
    let DrafterChoice {
        drafter,
        install_has_mtp_head: _,
        install_has_dflash: _,
        note: drafter_note,
    } = choice;

    // Report the RESOLVED slot count, not the request. Under `auto` the
    // request carries no number, and this one is a property of the machine
    // and the install -- 44.2 tok/s at 16 slots against 51.2 at 32 on the
    // same Gemma 4 install (`docs/DECODE_BUDGET.md`), so no throughput or
    // footprint figure from this run is readable without it.
    if !request.quiet {
        eprintln!(
            "expert cache: {} slots per layer{}",
            runner.expert_cache_slots(),
            match request.expert_cache_slots {
                invocation::ExpertCacheSlots::Auto => " (auto)",
                invocation::ExpertCacheSlots::Fixed(_) => "",
            }
        );
    }

    let shaping = ShapingConfig::new(
        request.temperature,
        request.top_k,
        Some(request.top_p),
        request.repetition_penalty,
        request.seed,
    )
    .map_err(|e| e.to_string())?;

    // Decided ONCE, here, and not per turn: both inputs are fixed for the
    // process, and a `--chat` session that started speculating must not stop
    // silently three turns in.
    // `speculation_blocker` and not `mtp_draft_depth() > 0`: a head is
    // NECESSARY and not sufficient. The batched verify is INT4-only, so a
    // 1-bit or 2-bit install carrying a head would pass a head-presence check
    // and then fail at the first verify with the generation already under way.
    // A MoE install is refused too, but as a POLICY rather than a capability
    // since ROADMAP Phase 3 landed the routed verify: no published MoE
    // conversion of this architecture ships a drafter to drive it. The runner
    // owns that list because the runner owns the refusals it mirrors.
    let speculation = resolve_speculation(
        asked,
        // The RESOLVED drafter, because `auto` takes its block from the
        // drafter's own default and the two differ: 2 for the MTP head, 8
        // for DFlash2.
        drafter,
        match drafter {
            runtime::SpeculativeDrafter::Dflash => runner.dflash_speculation_blocker(),
            // THE NOTE WINS WHERE THERE IS ONE. Both reasons are true of a
            // DFlash2-only install under `auto` -- it has no MTP head, and
            // its DFlash2 drafter was deliberately not enabled -- and only
            // one of them names something the caller can act on. Under a
            // NAMED block this is what turns into the hard error, which is
            // right: `--speculative 2` alone does not say which drafter, and
            // the message says which flag would.
            _ => drafter_note.or_else(|| runner.speculation_blocker()),
        },
        shaping.is_deterministic(),
    )?;
    if !request.quiet {
        match &speculation {
            SpeculationPlan::Enabled { block } => {
                // The DRAFTER is named, not just the block. Two drafters
                // serve this family and they have different shapes and
                // different measured optima, so a throughput number from
                // this run is unreadable without knowing which one ran.
                let which = match drafter {
                    runtime::SpeculativeDrafter::Dflash => "dflash2 (block drafter)",
                    _ => "mtp head (step drafter)",
                };
                eprintln!("speculative decoding: on, {which}, block {block}");
            }
            // A WARNING and not silence. An install carrying a drafter and
            // decoding one token at a time with nothing said is the exact
            // failure this feature was built to end.
            SpeculationPlan::Disabled { reason: Some(why) } => {
                eprintln!("speculative decoding: off ({why})");
            }
            SpeculationPlan::Disabled { reason: None } => {}
        }
    }

    // ROADMAP Phase P2. `resolve_profile` is where the OS gets asked about
    // Low Power Mode, and it is asked exactly once per process.
    let profile = runtime::resolve_profile(request.power_profile.map(map_power_profile));
    let rate = runtime::rate_control_for(profile, request.max_tokens_per_sec);

    Ok(Session {
        model_dir: model_dir.to_path_buf(),
        tokenizer,
        runner,
        shaping,
        speculation,
        rate,
        max_context: plan.resolved,
    })
}

/// The resolved context window and the arithmetic behind it.
///
/// Reports the SUGGESTION even when the caller named a number, for the same
/// reason the expert-cache line reports the resolved slot count: a window is
/// most of the KV footprint, and a reader comparing a peak or a prompt
/// refusal against another run needs to see both what was asked for and what
/// the machine and the checkpoint would have allowed.
fn report_context(plan: &runtime::ContextPlan, requested: invocation::MaxContext) {
    let trained = match plan.trained {
        Some(t) => format!("model {t}"),
        // Worth naming rather than omitting: it is why an old install's
        // `auto` reads 4,096 on a machine with room for far more.
        None => "model declares none".to_string(),
    };
    eprintln!(
        "context: {} tokens{} ({}, {:.0} MiB of KV; suggested {})",
        plan.resolved,
        match requested {
            invocation::MaxContext::Auto => " (auto)",
            invocation::MaxContext::Fixed(_) => "",
        },
        trained,
        plan.kv_bytes as f64 / (1024.0 * 1024.0),
        plan.suggested,
    );
}

/// The two crates declare their own profile enums on purpose: `invocation`
/// is pure and depends only on `foundation`. This is the one place the two
/// spellings meet.
/// The parser's speculation enums onto the runtime's, in the one place they
/// meet.
///
/// Two enums rather than one, exactly as [`map_power_profile`] below has two:
/// `crates/invocation` is pure and depends only on `foundation`, while every
/// decision the runtime side makes reads an install or a machine. Both
/// matches are exhaustive with no wildcard arm, so a fourth spelling of
/// either is a compile error here rather than a silent default somewhere
/// downstream (AGENTS.md Gotchas 24/37/39).
fn map_speculation(speculation: invocation::Speculation) -> runtime::Speculation {
    match speculation {
        invocation::Speculation::Auto => runtime::Speculation::Auto,
        invocation::Speculation::Off => runtime::Speculation::Off,
        invocation::Speculation::Block(n) => runtime::Speculation::Block(n),
    }
}

fn map_drafter(drafter: invocation::SpeculativeDrafter) -> runtime::SpeculativeDrafter {
    match drafter {
        invocation::SpeculativeDrafter::Auto => runtime::SpeculativeDrafter::Auto,
        invocation::SpeculativeDrafter::Mtp => runtime::SpeculativeDrafter::Mtp,
        invocation::SpeculativeDrafter::Dflash => runtime::SpeculativeDrafter::Dflash,
    }
}

/// Loads and shapes the direction set a run asked for.
///
/// The FILE is parsed here, before open, for the reason `resolve_drafter`
/// reads a resident index here: `crates/runtime` cannot reach
/// `crates/repack`, which owns the GGUF parser (AGENTS.md Gotcha 8), so the
/// front end hands the runner a plain `model_io::SteeringSet`.
///
/// Every failure is an error rather than a fallback to steering off. A caller
/// who named a vector and silently got none would measure the unsteered
/// engine and report it as the steered one -- the argument `MtpState::build`
/// makes for an explicitly-requested drafter, on an axis that changes the
/// TOKENS rather than the throughput.
fn resolve_steering(request: &InvocationRequest) -> Result<runtime::SteeringPolicy, String> {
    let Some(path) = request.steering.as_deref() else {
        return Ok(runtime::SteeringPolicy::off());
    };
    let mut set = repack::control_vector::load_control_vector(std::path::Path::new(path))
        .map_err(|e| format!("--steering {path}: {e}"))?;
    if let Some((start, end)) = request.steering_layers {
        set.restrict_to_range(start as usize, end as usize);
    }
    Ok(runtime::SteeringPolicy {
        // The flag wins over the file's declared mode, and the file's wins
        // over the default: a vector built for one edit should apply that
        // edit unless someone says otherwise.
        mode: request
            .steering_mode
            .or(set.declared_mode)
            .unwrap_or_default(),
        alpha: request.steering_scale.unwrap_or(1.0),
        target: request.steering_target,
        gate_threshold: request.steering_gate,
        set: Some(set),
    })
}

fn map_power_profile(profile: invocation::PowerProfile) -> runtime::PowerProfile {
    match profile {
        invocation::PowerProfile::Performance => runtime::PowerProfile::Performance,
        invocation::PowerProfile::Balanced => runtime::PowerProfile::Balanced,
        invocation::PowerProfile::Efficiency => runtime::PowerProfile::Efficiency,
    }
}
