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

use crate::session::{Engine, Session, SessionCore};
use crate::wire::{load_guard, sized, OpenOptions, SessionInfo, SpeculationInfo};

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

/// Maps the wire spelling of `speculation`, which is `"off"`, `"auto"`, or a
/// block size written either as a number or as a string.
///
/// Both spellings of a block are accepted for the reason [`sized`] accepts
/// three spellings of automatic: a Swift enum encoding a mixed number/string
/// value may reasonably choose either, and making them mean different things
/// would be a trap invisible from the header. Any other string is an error
/// rather than a fallback to `auto`, so `"of"` is heard about.
///
/// The allowed range is READ from `foundation` rather than restated, exactly
/// as `turbospark-server`'s parser reads it, so the three front ends cannot
/// come to accept different blocks (AGENTS.md Gotcha 2).
fn speculation(value: &Option<serde_json::Value>) -> Result<runtime::Speculation, String> {
    let allowed = foundation::runtime_config::ALLOWED_SPECULATION_BLOCKS;
    let block = |n: u64| {
        u32::try_from(n)
            .ok()
            .filter(|v| allowed.contains(v))
            .map(runtime::Speculation::Block)
            .ok_or_else(|| format!("speculation block must be in {allowed:?}, got {n}"))
    };
    match value {
        None | Some(serde_json::Value::Null) => Ok(runtime::Speculation::Auto),
        Some(serde_json::Value::String(s)) if s.eq_ignore_ascii_case("auto") => {
            Ok(runtime::Speculation::Auto)
        }
        Some(serde_json::Value::String(s)) if s.eq_ignore_ascii_case("off") => {
            Ok(runtime::Speculation::Off)
        }
        Some(serde_json::Value::String(s)) => match s.parse::<u64>() {
            Ok(n) => block(n),
            Err(_) => Err(format!(
                "speculation must be \"off\", \"auto\" or a block in {allowed:?}, got {s:?}"
            )),
        },
        Some(serde_json::Value::Number(n)) => match n.as_u64() {
            Some(n) => block(n),
            None => Err(format!("speculation block must be in {allowed:?}, got {n}")),
        },
        Some(other) => Err(format!(
            "speculation must be \"off\", \"auto\" or a block in {allowed:?}, got {other}"
        )),
    }
}

/// Maps the wire spelling of a drafter.
fn drafter(name: Option<&str>) -> Result<runtime::SpeculativeDrafter, String> {
    match name {
        None | Some("auto") => Ok(runtime::SpeculativeDrafter::Auto),
        Some("mtp") => Ok(runtime::SpeculativeDrafter::Mtp),
        Some("dflash") => Ok(runtime::SpeculativeDrafter::Dflash),
        Some(other) => Err(format!(
            "speculativeDrafter must be auto, mtp or dflash, got {other:?}"
        )),
    }
}

/// The wire name of a resolved drafter. Only ever reported beside an ENABLED
/// block, so `Auto` -- which `resolve_drafter` has already replaced by the
/// time this is called -- has no spelling of its own to invent.
fn drafter_name(drafter: runtime::SpeculativeDrafter) -> &'static str {
    match drafter {
        runtime::SpeculativeDrafter::Dflash => "dflash",
        _ => "mtp",
    }
}

/// Maps the wire spelling of a steering mode (`ablate`, `add`, `clamp`, `renorm`).
fn steering_mode(name: &str) -> Result<foundation::SteeringMode, String> {
    foundation::SteeringMode::parse(name).ok_or_else(|| {
        format!(
            "steeringMode must be one of {:?}, got {name:?}",
            foundation::STEERING_MODE_NAMES
        )
    })
}

/// Maps a layer range string (`START:END`, inclusive and 0-based).
fn layer_range(range_str: &str) -> Result<(usize, usize), String> {
    let bad =
        || format!("steeringLayers must be START:END, inclusive and 0-based, got {range_str:?}");
    let (a, b) = range_str.split_once(':').ok_or_else(bad)?;
    let start: usize = a.trim().parse().map_err(|_| bad())?;
    let end: usize = b.trim().parse().map_err(|_| bad())?;
    if end < start {
        return Err(bad());
    }
    Ok((start, end))
}

/// Maps a steering scale multiplier (finite number, default 1.0).
fn steering_scale(scale: Option<f64>) -> Result<f32, String> {
    match scale {
        None => Ok(1.0),
        Some(s) if s.is_finite() => Ok(s as f32),
        Some(s) => Err(format!("steeringScale must be a finite number, got {s}")),
    }
}

/// Maps a steering clamp target (finite number, default 0.0).
fn steering_target(target: Option<f64>) -> Result<f32, String> {
    match target {
        None => Ok(0.0),
        Some(t) if t.is_finite() => Ok(t as f32),
        Some(t) => Err(format!("steeringTarget must be a finite number, got {t}")),
    }
}

/// Maps a steering gate threshold (finite number >= 0.0, default 0.0).
fn steering_gate(gate: Option<f64>) -> Result<f32, String> {
    match gate {
        None => Ok(0.0),
        Some(g) if g.is_finite() && g >= 0.0 => Ok(g as f32),
        Some(g) => Err(format!(
            "steeringGate must be a finite number >= 0, got {g}"
        )),
    }
}

pub(crate) fn open(model: &str, options: &OpenOptions) -> Result<Session, String> {
    // EVERY OPTION IS MAPPED BEFORE ANYTHING IS READ FROM DISK, and that
    // ordering is worth keeping. A misspelled key is the caller's own
    // mistake and is answerable in microseconds, so answering it first
    // means a caller who sent both a bad path and a bad option hears about
    // the one they can fix from the header alone -- and it is what lets the
    // SwiftPM target, the only thing that can check `turbospark.h`
    // (`CLAUDE.md` Gotcha 2), reach these spellings with no install on the
    // machine.
    let max_context = sized(&options.max_context, "maxContext")?;
    let expert_cache_slots = sized(&options.expert_cache_slots, "expertCacheSlots")?;
    // Mapped here with the rest, BEFORE anything is read from disk, so a
    // misspelled tier outranks a bad path in the error -- the rule this file
    // already follows, and what lets the SwiftPM target reach these spellings
    // with no install on the machine.
    let load_policy = runtime::LoadPolicy {
        guard: load_guard(&options.load_guard)?,
        min_auto_context: options.min_auto_context.unwrap_or(0),
    };
    let asked = speculation(&options.speculation)?;
    let requested_drafter = drafter(options.speculative_drafter.as_deref())?;
    let requested_steering_mode = options
        .steering_mode
        .as_deref()
        .map(steering_mode)
        .transpose()?;
    let requested_steering_scale = steering_scale(options.steering_scale)?;
    let requested_steering_target = steering_target(options.steering_target)?;
    let requested_steering_gate = steering_gate(options.steering_gate)?;
    let requested_steering_layers = options
        .steering_layers
        .as_deref()
        .map(layer_range)
        .transpose()?;
    // The SPELLING only. `resolve_profile` is where the OS is asked about
    // Low Power Mode and it stays below, next to the rate control it feeds.
    let requested_profile = options
        .power_profile
        .as_deref()
        .map(power_profile)
        .transpose()?;

    if options.steering.is_none()
        && (requested_steering_mode.is_some()
            || options.steering_scale.is_some()
            || requested_steering_layers.is_some()
            || options.steering_target.is_some()
            || options.steering_gate.is_some())
    {
        return Err("steering options given without a steering vector path".to_string());
    }

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
        match max_context {
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
        &load_policy,
    )
    .map_err(|e| e.to_string())?;

    // Steering policy is loaded before open, following CLI and server pattern.
    let steering_policy = if let Some(path_str) = options.steering.as_deref() {
        let path = Path::new(path_str);
        let mut set = repack::control_vector::load_control_vector(path)
            .map_err(|e| format!("steering {path_str}: {e}"))?;
        if let Some((start, end)) = requested_steering_layers {
            set.restrict_to_range(start, end);
        }
        let mode = requested_steering_mode
            .or(set.declared_mode)
            .unwrap_or_default();
        runtime::SteeringPolicy {
            set: Some(set),
            mode,
            alpha: requested_steering_scale,
            target: requested_steering_target,
            gate_threshold: requested_steering_gate,
        }
    } else {
        runtime::SteeringPolicy::off()
    };

    // THE DRAFTER IS RESOLVED BEFORE THE OPEN, exactly as the CLI's
    // `open_session` and the server's `RealChatModel::open` do it: the
    // policies below name exactly one drafter, and the wrong one is
    // indistinguishable from an install carrying none at all -- pinned at
    // `Mtp`, a DFlash2 install reports "carries no multi-token-prediction
    // head" and decodes sequentially with a working drafter on disk. `Auto`
    // reads the resident INDEX, which is kilobytes and not the weights.
    //
    // The three decisions are `runtime::speculation_policy`'s and are not
    // restated here: a second copy names the wrong cause the first time two
    // front ends disagree, which is why that module left `crates/cli` when
    // the server needed it.
    let choice = runtime::resolve_drafter(requested_drafter, dir);
    let runner = runtime::RealForwardRunner::open_with_slot_policy_speculation_and_steering(
        dir,
        arch,
        plan.resolved as usize,
        match expert_cache_slots {
            Some(n) => runtime::ExpertCacheSlots::Fixed(n as usize),
            None => runtime::ExpertCacheSlots::Auto,
        },
        runtime::draft_policies(&choice, asked),
        steering_policy.clone(),
    )
    .map_err(|e| e.to_string())?;

    // **RESOLVED AS THOUGH EVERY TURN WERE DETERMINISTIC, which is the one
    // place this binding follows the SERVER rather than the CLI.** On the
    // CLI a process has one shaping, so `open_session` can settle both
    // halves at once. Here, as on a server, the temperature belongs to the
    // request: what is fixed at open is the INSTALL half -- does it carry a
    // drafter this engine can verify with -- and the per-turn half is
    // applied in `generate`.
    //
    // A NAMED block that cannot be served fails HERE, which is the whole
    // hard-fail/warn split: a caller who named a block is measuring, and a
    // session that quietly did not speculate is the number that ends up in
    // a table. `auto` opens and reports the reason instead.
    let speculation_plan = runtime::resolve_speculation(
        asked,
        choice.drafter,
        match choice.drafter {
            runtime::SpeculativeDrafter::Dflash => runner.dflash_speculation_blocker(),
            // The note outranks the engine's own blocker where there is
            // one: both are true of a DFlash2-only install under `auto`,
            // and only one names something the caller can act on.
            _ => choice.note.clone().or_else(|| runner.speculation_blocker()),
        },
        true,
    )?;
    let speculation_block = match &speculation_plan {
        runtime::SpeculationPlan::Enabled { block } => Some(*block),
        runtime::SpeculationPlan::Disabled { .. } => None,
    };

    let profile = runtime::resolve_profile(requested_profile);
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
        // Five renders of a two-line conversation, at open only. The
        // alternative a GUI is otherwise forced into is a per-family table,
        // which is a second and staler copy of what the template states by
        // name (root Gotcha 56).
        reasoning_levels: tokenizer
            .accepted_reasoning_levels()
            .into_iter()
            .map(|level| level.as_str().to_string())
            .collect(),
        steering: if runner.steering_line().is_some() {
            crate::wire::SteeringInfo {
                active: true,
                mode: Some(steering_policy.mode.as_str().to_string()),
                scale: Some(steering_policy.alpha as f64),
                summary: runner.steering_line(),
            }
        } else {
            crate::wire::SteeringInfo::default()
        },
        speculation: SpeculationInfo {
            block: speculation_block,
            drafter: speculation_block.map(|_| drafter_name(choice.drafter).to_string()),
            reason: match speculation_plan {
                runtime::SpeculationPlan::Disabled { reason } => reason,
                runtime::SpeculationPlan::Enabled { .. } => None,
            },
        },
        vision: vision_info(&runner, dir),
        special_tokens: crate::wire::SpecialTokensInfo {
            bos_id: (tokenizer.bos_id >= 0).then_some(tokenizer.bos_id),
            eos_id: (tokenizer.eos_id >= 0).then_some(tokenizer.eos_id),
            pad_id: (tokenizer.pad_id >= 0).then_some(tokenizer.pad_id),
            end_of_turn_id: (tokenizer.end_of_turn_id >= 0).then_some(tokenizer.end_of_turn_id),
            stop_token_ids: tokenizer.stop_token_ids.iter().copied().collect(),
            think_start_id: tokenizer.think_start_id,
            think_end_id: tokenizer.think_end_id,
        },
    };

    Ok(Session::new(SessionCore {
        engine: Mutex::new(Engine::Real(Box::new(runner))),
        tokenizer,
        cancel: Arc::new(AtomicBool::new(false)),
        model_dir: dir.to_path_buf(),
        max_context: plan.resolved,
        rate,
        speculation_block,
        info,
    }))
}

/// Whether this install would actually SERVE an image, and why not when it
/// carries a tower and would not.
///
/// The config read is a `stat` and a parse of a few KB, done once at open so a
/// host can gate a control on the answer rather than discovering it on the
/// first attachment. Deliberately the same two conditions `attach_images`
/// checks, in the same order, so the gate and the refusal cannot disagree --
/// a detection probe must not be able to pass where its own resolver fails
/// (AGENTS.md Gotcha 52).
fn vision_info(runner: &runtime::RealForwardRunner, dir: &Path) -> crate::wire::VisionInfo {
    if !runner.has_vision_tower() {
        return crate::wire::VisionInfo::default();
    }
    let image_token_id = runner.vision_config().image_token_id as i32;
    match crate::vision::preprocess_params(runner, dir) {
        Ok(_) => crate::wire::VisionInfo {
            active: true,
            image_token_id: Some(image_token_id),
            reason: None,
        },
        Err(reason) => crate::wire::VisionInfo {
            active: false,
            image_token_id: None,
            reason: Some(reason),
        },
    }
}

/// The option MAPPERS, which are the half of this module reachable without a
/// multi-gigabyte install.
///
/// `tests/c_surface.rs` drives everything else here through a scripted
/// session; `open` itself needs a real one, so the wire spellings would
/// otherwise be covered by nothing at all.
#[cfg(test)]
#[path = "open_tests.rs"]
mod open_tests;
