//! Opening an install, macOS only.
//!
//! This mirrors `crates/cli/src/generate.rs::open_session` step for step, and
//! it was written by reading that function rather than by reinventing the
//! sequence. Two traps it documents are live here too:
//!
//! - The resolved context window is carried on the [`Session`] and never
//!   re-read from the caller's request. Under `auto` the request carries no
//!   number at all, and the KV cache has already been allocated at the
//!   resolved one, so every downstream consumer has to agree with what was
//!   allocated rather than with what was asked for.
//! - The resolved SLOT COUNT is reported, not the requested one, for the same
//!   reason: under `auto` there is no requested one, and the count is worth
//!   44.2 tok/s against 51.2 on the same install.
//!
//! The window is resolved BEFORE opening because the failure it catches is an
//! allocation: `KvCacheManager::new` sizes every layer up front, so a window
//! that does not fit surfaces as a Metal allocation error with no number in
//! it pointing back at the caller's setting.

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use tokenizer::MfTokenizer;

use crate::session::{Engine, Session};
use crate::wire::{sized, OpenOptions, SessionInfo};

/// Maps the wire spelling of a power profile.
fn power_profile(name: &str) -> Result<runtime::PowerProfile, String> {
    match name {
        "performance" => Ok(runtime::PowerProfile::Performance),
        "balanced" => Ok(runtime::PowerProfile::Balanced),
        "efficiency" => Ok(runtime::PowerProfile::Efficiency),
        other => Err(format!(
            "powerProfile must be performance, balanced or efficiency, got {other:?}"
        )),
    }
}

pub(crate) fn open(model: &str, options: &OpenOptions) -> Result<Session, String> {
    // `model` takes a path OR a `turbospark-model` alias, and an existing
    // directory always wins: a bare name that silently preferred an alias
    // would run a DIFFERENT model than the caller named, fluently and with
    // no error.
    let resolved = catalog::resolve_model_arg(model);
    let dir: &Path = resolved.as_path();
    if !dir.is_dir() {
        return Err(format!(
            "{} is not a directory (and matched no installed alias)",
            dir.display()
        ));
    }
    let arch = repack::peek_manifest_arch(dir)?;
    // Captured before `arch` is moved into the runner. The manifest's own
    // spelling, which is what every other surface in this repo prints.
    let family = arch.family.as_str().to_string();

    let tokenizer = MfTokenizer::load_from_dir(dir)
        .map_err(|e| format!("failed to load a tokenizer from {}: {e}", dir.display()))?;

    let trained = repack::trained_context_meta::peek(dir);
    let plan = runtime::resolve_max_context(
        match sized(&options.max_context, "maxContext")? {
            Some(n) => runtime::MaxContext::Fixed(n),
            None => runtime::MaxContext::Auto,
        },
        &arch,
        trained,
        // The window an install declaring NO trained context resolves to.
        // Deliberately not "whatever memory allows": that is every install
        // written before the field existed, and sizing those from free RAM
        // would take a 13 GB install from its documented 4,096 to a
        // six-figure window the first time anyone re-ran the same call.
        // Taken from `foundation` rather than restated, so it cannot drift
        // from what `turbospark-check` resolves for the same install.
        foundation::runtime_config::DEFAULT_MAX_CONTEXT,
        runtime::physical_memory(),
        runtime::committed_bytes(dir),
    )
    .map_err(|e| e.to_string())?;

    let runner = runtime::RealForwardRunner::open_with_slot_policy(
        dir,
        arch,
        plan.resolved as usize,
        match sized(&options.expert_cache_slots, "expertCacheSlots")? {
            Some(n) => runtime::ExpertCacheSlots::Fixed(n as usize),
            None => runtime::ExpertCacheSlots::Auto,
        },
    )
    .map_err(|e| e.to_string())?;

    let profile = runtime::resolve_profile(
        options
            .power_profile
            .as_deref()
            .map(power_profile)
            .transpose()?,
    );
    let rate = runtime::rate_control_for(profile, options.max_tokens_per_sec);

    let info = SessionInfo {
        model_path: dir.display().to_string(),
        family,
        max_context: plan.resolved,
        trained_context: plan.trained,
        // Reported rather than refused: RoPE extrapolates past the trained
        // context rather than failing, some checkpoints carry YaRN scaling
        // meant to exceed it, and an install written before the field
        // existed declares none -- so refusing would be enforced on some
        // installs and not others.
        past_trained_context: plan.past_trained,
        expert_cache_slots: runner.expert_cache_slots(),
        vocab_size: runner.vocab_size(),
        dialect: format!("{:?}", tokenizer.dialect),
        reasoning_support: match tokenizer.reasoning_support() {
            tokenizer::ReasoningSupport::Level => "level",
            tokenizer::ReasoningSupport::ToggleOnly => "toggleOnly",
            tokenizer::ReasoningSupport::None => "none",
        }
        .to_string(),
    };

    Ok(Session {
        engine: Mutex::new(Engine::Real(Box::new(runner))),
        tokenizer,
        cancel: Arc::new(AtomicBool::new(false)),
        max_context: plan.resolved,
        rate,
        info,
    })
}
