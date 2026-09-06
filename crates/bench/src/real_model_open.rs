//! Real model opening and runner initialization for benchmark protocol runs.

use std::path::Path;

use model_io::ArchConfig;
use runtime::RealForwardRunner;
use tokenizer::MfTokenizer;

use crate::protocol::PROTOCOL_MAX_CONTEXT;
use crate::real_model_params::{protocol_parameters, ProtocolParameters};

/// Open a real `.gturbo` install for the protocol: arch reconstructed from
/// its own `manifest.json`, tokenizer loaded from the same directory (the
/// usual checkpoint bundling convention), KV sized to the protocol's 4K.
///
/// `slots` is the per-layer routed-expert cache size (allowed
/// 8/16/24/32/48/64/96/128, same set the CLI's `--expert-cache-slots`
/// takes). Output is NOT
/// md5-identical across slot counts: the hit/miss split permutes the
/// phase-2 reduce order and FP addition is not associative. Compare
/// within one slot count.
pub fn open_model_runner(
    model_dir: &Path,
    slots: usize,
) -> Result<(RealForwardRunner, MfTokenizer), String> {
    open_model_runner_with_context(model_dir, slots, PROTOCOL_MAX_CONTEXT)
}

/// [`open_model_runner`] asking for a speculative drafter.
///
/// Every other opener in this crate reaches `open_with_options`, which pins
/// drafting OFF -- a head allocates its own KV and an M-row verify scratch,
/// and a frozen footprint row must not acquire either because the install it
/// happens to point at carries a head. The two MTP probes and the generation
/// gate are the callers that genuinely want one, so they say so here.
///
/// Takes the FULL policy pair so a DFlash2 probe asks through the same
/// door; `DraftPolicies::mtp` wraps the MTP-only callers.
pub fn open_model_runner_speculative(
    model_dir: &Path,
    slots: usize,
    speculation: runtime::DraftPolicies,
) -> Result<(RealForwardRunner, MfTokenizer), String> {
    let arch = repack::peek_manifest_arch(model_dir)?;
    let tokenizer = MfTokenizer::load_from_dir(model_dir).map_err(|e| {
        format!(
            "failed to load a tokenizer from {}: {e}",
            model_dir.display()
        )
    })?;
    let runner = RealForwardRunner::open_with_options_and_speculation(
        model_dir,
        arch,
        PROTOCOL_MAX_CONTEXT as usize,
        slots,
        speculation,
    )
    .map_err(|e| e.to_string())?;
    Ok((runner, tokenizer))
}

/// [`open_model_runner`] carrying a directional-steering policy
/// (`docs/OBLITERATION.md`).
///
/// A SEPARATE entry point for the reason [`open_model_runner_speculative`]
/// is one, and the reason is sharper here than there: steering CHANGES THE
/// TOKENS. A frozen digest or a frozen perplexity that silently acquired an
/// edit to every layer's residual stream would be a different measurement
/// wearing the old row's name, and unlike a slot count -- which is a
/// throughput axis only (crate Gotcha 5) -- nothing about the output would
/// be expected to survive it. Both memory oracles and all six quality gates
/// reach `open_model_runner*` above and therefore cannot get here by
/// defaulting into anything.
///
/// The set arrives already PARSED, from `repack::control_vector`, because
/// `crates/runtime` cannot reach the crate that owns the GGUF parser
/// (AGENTS.md Gotcha 8). `SteeringPolicy::off()` allocates and encodes
/// nothing, so passing it is identical to the plain opener -- which is what
/// the probe's null control exists to check rather than assume.
pub fn open_model_runner_steered(
    model_dir: &Path,
    slots: usize,
    steering: runtime::SteeringPolicy,
) -> Result<(RealForwardRunner, MfTokenizer), String> {
    let arch = repack::peek_manifest_arch(model_dir)?;
    let tokenizer = MfTokenizer::load_from_dir(model_dir).map_err(|e| {
        format!(
            "failed to load a tokenizer from {}: {e}",
            model_dir.display()
        )
    })?;
    let runner = RealForwardRunner::open_with_slot_policy_speculation_and_steering(
        model_dir,
        arch,
        PROTOCOL_MAX_CONTEXT as usize,
        runtime::ExpertCacheSlots::Fixed(slots),
        runtime::DraftPolicies::off(),
        steering,
    )
    .map_err(|e| e.to_string())?;
    Ok((runner, tokenizer))
}

/// [`open_model_runner`] with the KV window named explicitly.
///
/// THE PROTOCOL'S 4K IS A PROPERTY OF THE HARNESS, NOT OF THE PROMPTS, and
/// one family already needs a different one (ROADMAP M4). The protocol fixes
/// the PROSE, and how many tokens that prose becomes is the checkpoint's
/// tokenizer's answer: `long-synthesis` is 2,842 tokens under Qwen3-30B-A3B's
/// 152k vocab and 3,444 under Mistral 7B's 32k, and `3444 + PROTOCOL_MAX_NEW`
/// does not fit 4,096. The case then fails to run at all, and an oracle that
/// asserts `endOfTurn` on every case cannot be written for that family.
///
/// Raising `PROTOCOL_MAX_CONTEXT` itself is NOT the fix: KV is sized
/// `max_context * kv_stride` at open, so it would move every already-frozen
/// peak in every other family's rows. A per-family window in that family's
/// own oracle target moves only its own, which is why this is a parameter
/// rather than a constant.
///
/// The number is load-bearing for the row that uses it: KV is most of a
/// dense install's counted footprint (AGENTS.md Gotcha 40), so a row
/// measured at one window says nothing about another. Record it beside the
/// ceiling.
pub fn open_model_runner_with_context(
    model_dir: &Path,
    slots: usize,
    max_context: u32,
) -> Result<(RealForwardRunner, MfTokenizer), String> {
    let arch = repack::peek_manifest_arch(model_dir)?;
    open_with_arch(model_dir, arch, slots, max_context)
}

/// [`open_model_runner_with_context`] plus attaching a standalone vision
/// sidecar to the opened (text-only) trunk (vision memory sidecar, Part A4).
///
/// This is what Part A6's real-install gate uses to compare a
/// sidecar-attached text-only trunk against the combined
/// `qwen38-27b-vision.gturbo` install: the two must agree on vision output
/// byte for byte, which needs a trunk opened exactly the way every other
/// gate in this crate opens one (same `open_model_runner_with_context`
/// path) with the sidecar attached on top, rather than a bespoke open.
///
/// The attach happens AFTER open, matching
/// `RealForwardRunner::attach_vision_sidecar`'s own contract ("call this
/// once, before the first image -- there is no supported way to detach or
/// replace a sidecar once attached"). No image is processed here and no
/// tokenizer marker check runs: that is `MfTokenizer::verify_image_markers`'s
/// job, which a caller reaches through the returned tokenizer if it wants
/// that check ahead of an image, the same as any other opener in this crate
/// leaves shaping and sampling decisions to its caller.
pub fn open_model_runner_with_context_and_vision_sidecar(
    model_dir: &Path,
    slots: usize,
    max_context: u32,
    vision_sidecar: &Path,
) -> Result<(RealForwardRunner, MfTokenizer), String> {
    let arch = repack::peek_manifest_arch(model_dir)?;
    let (mut runner, tokenizer) = open_with_arch(model_dir, arch, slots, max_context)?;
    runner
        .attach_vision_sidecar(vision_sidecar)
        .map_err(|e| format!("{}: {e}", vision_sidecar.display()))?;
    Ok((runner, tokenizer))
}

/// The body both entry points share, taking an already-peeked `ArchConfig`
/// so [`open_model_runner_for_protocol`] reads `manifest.json` once.
pub(crate) fn open_with_arch(
    model_dir: &Path,
    arch: ArchConfig,
    slots: usize,
    max_context: u32,
) -> Result<(RealForwardRunner, MfTokenizer), String> {
    let tokenizer = MfTokenizer::load_from_dir(model_dir).map_err(|e| {
        format!(
            "failed to load a tokenizer from {}: {e}",
            model_dir.display()
        )
    })?;
    let runner = RealForwardRunner::open_with_options(model_dir, arch, max_context as usize, slots)
        .map_err(|e| e.to_string())?;
    Ok((runner, tokenizer))
}

/// [`open_model_runner`] with the protocol's parameters resolved from the
/// install's family, returned alongside the runner.
///
/// **Returning them together is the point.** The window is needed at open
/// (KV is sized there) and again at run (the generation loop enforces its own
/// limit), and the two diverging is silent: opening at 8,192 and running at
/// 4,096 refuses the long case exactly as the shared default did, with
/// nothing pointing at the mismatch. Handing back one value that both call
/// sites read makes that unrepresentable.
pub fn open_model_runner_for_protocol(
    model_dir: &Path,
    slots: usize,
) -> Result<(RealForwardRunner, MfTokenizer, ProtocolParameters), String> {
    open_model_runner_for_protocol_speculative(model_dir, slots, runtime::DraftPolicies::off())
}

/// [`open_model_runner_for_protocol`] with a drafter.
///
/// A SEPARATE entry point rather than a parameter on the one above, for the
/// reason `RealForwardRunner::open_with_options` is separate from
/// `open_with_slot_policy` (AGENTS.md Gotcha 35 and crate Gotcha 5): every
/// MEASURING caller -- both memory oracles, all four quality gates -- goes
/// through the plain one and therefore cannot acquire a drafter by
/// inheriting a default. A frozen peak or a frozen digest that silently
/// gained a speculative decode would be a different measurement wearing the
/// old row's name.
pub fn open_model_runner_for_protocol_speculative(
    model_dir: &Path,
    slots: usize,
    speculation: runtime::DraftPolicies,
) -> Result<(RealForwardRunner, MfTokenizer, ProtocolParameters), String> {
    let arch = repack::peek_manifest_arch(model_dir)?;
    let params = protocol_parameters(arch.family);
    let tokenizer = MfTokenizer::load_from_dir(model_dir).map_err(|e| {
        format!(
            "failed to load a tokenizer from {}: {e}",
            model_dir.display()
        )
    })?;
    let runner = RealForwardRunner::open_with_options_and_speculation(
        model_dir,
        arch,
        params.max_context as usize,
        slots,
        speculation,
    )
    .map_err(|e| e.to_string())?;
    Ok((runner, tokenizer, params))
}
