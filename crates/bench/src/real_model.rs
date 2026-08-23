//! Real-install benchmark flow (`turbospark-bench --model <dir>`): opens a
//! `.gturbo` install with `RealForwardRunner` and drives the frozen
//! community-protocol cases through the real prefill+decode loop,
//! chat-formatted exactly as the CLI formats them (the IT checkpoint needs
//! its turn markup; raw prompts babble). Reports the split
//! prefill/decode seconds from `RawDecodeResult` (not wall time) plus the
//! peak `phys_footprint` sampled at the Swift cadence: before prefill, on
//! every 8th decoded token, and once after the run.

use std::path::Path;

use model_io::ArchConfig;
use runtime::{
    run_raw_completion, run_raw_completion_speculative, GenerationConfig, RateControl,
    RawDecodeProgress, RealForwardRunner, StopReason,
};
use selection::ShapingConfig;
use tokenizer::{Message, MfTokenizer, Role};

use crate::memory::AppMemorySampler;
use crate::protocol::{
    ProtocolCase, PROTOCOL_MAX_CONTEXT, PROTOCOL_MAX_NEW, PROTOCOL_TEMPERATURE, PROTOCOL_TOP_K,
    PROTOCOL_TOP_P,
};

pub struct CaseResult {
    pub case_id: &'static str,
    pub prompt_tokens: usize,
    pub prefill_seconds: f64,
    pub new_tokens: usize,
    pub decode_seconds: f64,
    pub peak_footprint_bytes: Option<u64>,
    pub reason: StopReason,
}

impl CaseResult {
    pub fn tokens_per_second(&self) -> f64 {
        if self.decode_seconds > 0.0 {
            self.new_tokens as f64 / self.decode_seconds
        } else {
            0.0
        }
    }
}

/// Open a real `.gturbo` install for the protocol: arch reconstructed from
/// its own `manifest.json`, tokenizer loaded from the same directory (the
/// usual checkpoint bundling convention), KV sized to the protocol's 4K.
///
/// `slots` is the per-layer routed-expert cache size (allowed 8/16/24/32,
/// same set the CLI's `--expert-cache-slots` takes). Output is NOT
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

/// The body both entry points share, taking an already-peeked `ArchConfig`
/// so [`open_model_runner_for_protocol`] reads `manifest.json` once.
fn open_with_arch(
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

pub use crate::real_model_params::{protocol_parameters, ProtocolParameters};

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

/// One run of one protocol case (the caller decides whether it is a
/// discarded warmup or the measured run). The sampler is shared across
/// cases on purpose: the oracle asserts on the whole session's peak.
pub fn run_protocol_case(
    runner: &mut RealForwardRunner,
    tokenizer: &MfTokenizer,
    case: &ProtocolCase,
    sampler: &mut AppMemorySampler,
    rate: RateControl,
) -> Result<CaseResult, String> {
    run_protocol_case_with_context(runner, tokenizer, case, sampler, rate, PROTOCOL_MAX_CONTEXT)
}

/// [`run_protocol_case`] with the context window named explicitly.
///
/// It MUST be the same window the runner was opened with. The generation
/// loop enforces its own limit independently of the KV allocation, so
/// opening at 8,192 and running at the 4,096 default refuses the long case
/// exactly as before, with nothing pointing at the mismatch.
pub fn run_protocol_case_with_context(
    runner: &mut RealForwardRunner,
    tokenizer: &MfTokenizer,
    case: &ProtocolCase,
    sampler: &mut AppMemorySampler,
    rate: RateControl,
    max_context: u32,
) -> Result<CaseResult, String> {
    run_protocol_case_with_budget(
        runner,
        tokenizer,
        case,
        sampler,
        rate,
        max_context,
        PROTOCOL_MAX_NEW,
    )
}

/// [`run_protocol_case_with_context`] with the GENERATION BUDGET named too.
///
/// The second per-family parameter, and it exists for the same reason the
/// first does: the protocol freezes the PROSE, and how many tokens a model
/// spends answering it is the model's property, not the protocol's.
///
/// `gpt-oss` is what forced it (ROADMAP M5). Harmony puts the model's
/// reasoning in an `analysis` channel BEFORE its answer, so the three cases
/// need 818 / 1,780 / 1,211 tokens to reach `<|return|>` where every other
/// family here finishes inside `PROTOCOL_MAX_NEW`. At 1,024 two of the three
/// stop on `maxTokens`, which the oracle's validity gate refuses -- correctly,
/// since a truncated run is not comparable to a completed one, and wrongly
/// diagnosed, since nothing is broken. Raising the SHARED constant is not an
/// option: it would let every other family's frozen row generate further and
/// move peaks that are already published.
///
/// Read the budget with the number, as with the window: a peak measured at a
/// larger budget saw more KV rows and more expert-slot warming.
#[allow(clippy::too_many_arguments)]
pub fn run_protocol_case_with_budget(
    runner: &mut RealForwardRunner,
    tokenizer: &MfTokenizer,
    case: &ProtocolCase,
    sampler: &mut AppMemorySampler,
    rate: RateControl,
    max_context: u32,
    max_new: u32,
) -> Result<CaseResult, String> {
    run_protocol_case_speculating(
        runner,
        tokenizer,
        case,
        sampler,
        rate,
        max_context,
        max_new,
        ProtocolShaping::Sampled,
        None,
    )
}

/// How a protocol case shapes its sampling.
///
/// A SEPARATE axis from speculation, and separating them is the whole point:
/// `Greedy` alone is a valid arm, so a speculation A/B can hold shaping
/// FIXED and vary one thing. Folding greedy into `--speculative` (which the
/// first draft of this did) makes `spec` against `nospec` a two-variable
/// comparison whose delta nobody can attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtocolShaping {
    /// The frozen protocol's own: temperature 0.2, top-k 64, top-p 0.95, a
    /// seed per case. Every published row is this.
    Sampled,
    /// Temperature EXACTLY 0. The only shaping speculation can serve.
    Greedy,
}

/// [`run_protocol_case_with_budget`] with the shaping and an optional
/// speculative block spelled out.
///
/// **`Greedy` IS A DIFFERENT WORKLOAD from the frozen protocol**, not a
/// switch flipped on it: it changes the token stream, the stop points and
/// therefore the token count, so its joules-per-token is NOT comparable to
/// any row in `docs/POWER_BASELINE.md`. It is comparable to ANOTHER greedy
/// arm in the same capture, which is what a drafter A/B needs.
///
/// Speculation with `Sampled` is REFUSED rather than promoted to greedy.
/// Acceptance here is `argmax(target) == proposal`, exact only at
/// temperature 0; `run_raw_completion_speculative` refuses it too, and
/// silently changing the caller's shaping to make their flag work is how a
/// measurement harness reports one workload under another's name.
#[allow(clippy::too_many_arguments)]
pub fn run_protocol_case_speculating(
    runner: &mut RealForwardRunner,
    tokenizer: &MfTokenizer,
    case: &ProtocolCase,
    sampler: &mut AppMemorySampler,
    rate: RateControl,
    max_context: u32,
    max_new: u32,
    shaping_mode: ProtocolShaping,
    speculative_block: Option<usize>,
) -> Result<CaseResult, String> {
    if speculative_block.is_some() && shaping_mode != ProtocolShaping::Greedy {
        return Err(
            "speculative decoding needs greedy shaping (acceptance is exact only at \
             temperature 0); pass --shaping greedy alongside --speculative"
                .to_string(),
        );
    }
    // Chat-format exactly as the CLI does: the dialect template renders
    // the turn markup (and its own <bos>, hence add_bos false).
    let messages = [Message::new(Role::User, case.content)];
    let rendered = tokenizer
        .apply_chat_template(&messages)
        .map_err(|e| format!("chat template: {e}"))?;
    let prompt_ids = tokenizer.encode(&rendered, false);

    let shaping = match shaping_mode {
        ProtocolShaping::Sampled => ShapingConfig::new(
            PROTOCOL_TEMPERATURE,
            PROTOCOL_TOP_K,
            Some(PROTOCOL_TOP_P),
            1.0,
            Some(case.seed),
        ),
        // EXACTLY 0.0 and not the smoke's 0.0001: `is_deterministic`
        // compares against zero, so 0.0001 walks the full sampler and the
        // speculative loop refuses it.
        ProtocolShaping::Greedy => ShapingConfig::new(0.0, 1, None, 1.0, Some(case.seed)),
    };
    let config = GenerationConfig {
        shaping: shaping.map_err(|e| e.to_string())?,
        max_new_tokens: max_new,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate,
    };

    // Read before the mutable borrow: the logits width is the MODEL's, not
    // the tokenizer dialect's (`RealForwardRunner::vocab_size`).
    let vocab_size = runner.vocab_size();
    sampler.sample();
    // The Swift runtime samples every 8th decoded token. Shared by both
    // loops so the memory sampling cadence is not an axis of the A/B.
    let progress = |event: RawDecodeProgress| {
        if let RawDecodeProgress::Token { index, .. } = event {
            if index % 8 == 0 {
                sampler.sample();
            }
        }
    };
    let result = match speculative_block {
        None => run_raw_completion(
            runner,
            tokenizer,
            &prompt_ids,
            &config,
            max_context,
            vocab_size,
            progress,
        ),
        // REFUSES rather than falling back when the install cannot serve a
        // drafter, which is the same contract `--speculative` gives on the
        // CLI: a power capture that quietly measured the non-speculative
        // engine and reported it under a `spec` arm label is the exact
        // failure `scripts/power.sh`'s resolved-parameter header exists to
        // prevent.
        Some(block) => run_raw_completion_speculative(
            runner,
            tokenizer,
            &prompt_ids,
            &config,
            max_context,
            vocab_size,
            block,
            progress,
        ),
    }
    .map_err(|e| e.to_string())?;
    sampler.sample();

    Ok(CaseResult {
        case_id: case.id,
        prompt_tokens: result.prompt_tokens,
        prefill_seconds: result.prefill_seconds,
        new_tokens: result.new_tokens,
        decode_seconds: result.decode_seconds,
        peak_footprint_bytes: sampler.peak_bytes(),
        reason: result.reason,
    })
}
