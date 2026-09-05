//! Real-install benchmark flow (`turbospark-bench --model <dir>`): opens a
//! `.gturbo` install with `RealForwardRunner` and drives the frozen
//! community-protocol cases through the real prefill+decode loop,
//! chat-formatted exactly as the CLI formats them (the IT checkpoint needs
//! its turn markup; raw prompts babble). Reports the split
//! prefill/decode seconds from `RawDecodeResult` (not wall time) plus the
//! peak `phys_footprint` sampled at the Swift cadence: before prefill, on
//! every 8th decoded token, and once after the run.

use runtime::{
    run_raw_completion, run_raw_completion_chunked, run_raw_completion_speculative,
    GenerationConfig, RateControl, RawDecodeProgress, RealForwardRunner, StopReason,
};
use selection::ShapingConfig;
use tokenizer::{Message, MfTokenizer, Role};

use crate::memory::AppMemorySampler;
use crate::protocol::{
    ProtocolCase, PROTOCOL_MAX_CONTEXT, PROTOCOL_MAX_NEW, PROTOCOL_TEMPERATURE, PROTOCOL_TOP_K,
    PROTOCOL_TOP_P,
};

/// Benchmark measurements for a single protocol evaluation case.
pub struct CaseResult {
    pub case_id: &'static str,
    pub prompt_tokens: usize,
    pub prefill_seconds: f64,
    /// The chunk size this run's prefill ACTUALLY used, `None` for the
    /// sequential loop.
    ///
    /// REPORTED rather than inferred, and that is the point: byte-identity
    /// against the sequential path is the chunked driver's OWN contract, so
    /// a silent fallback to `run_raw_completion` passes an identity check
    /// trivially. Without this field the only thing distinguishing an
    /// engaged chunked arm from a silently sequential one is the seconds
    /// column, which is exactly the number under measurement.
    pub prefill_chunk_tokens: Option<usize>,
    pub new_tokens: usize,
    pub decode_seconds: f64,
    pub peak_footprint_bytes: Option<u64>,
    pub reason: StopReason,
}

impl CaseResult {
    /// Computed decode throughput in tokens per second.
    pub fn tokens_per_second(&self) -> f64 {
        if self.decode_seconds > 0.0 {
            self.new_tokens as f64 / self.decode_seconds
        } else {
            0.0
        }
    }
}

pub use crate::real_model_open::{
    open_model_runner, open_model_runner_for_protocol, open_model_runner_for_protocol_speculative,
    open_model_runner_speculative, open_model_runner_steered, open_model_runner_with_context,
};
pub use crate::real_model_params::{protocol_parameters, ProtocolParameters};

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
        // The sequential prefill loop, explicitly. This wrapper's signature
        // does not change, which is what keeps every `oracle_common` call
        // site an untouched diff line and every frozen memory-oracle row
        // measured on the path it was frozen on.
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
    prefill_chunk: Option<usize>,
) -> Result<CaseResult, String> {
    // NOT COMPOSABLE, and refused rather than silently ordered. The
    // speculative loop owns its own prefill and has no chunked variant, so
    // honouring one flag and dropping the other would report one workload
    // under the other's arm name -- the same failure the speculation
    // refusal below exists to prevent. `crates/cli` lets speculation win
    // here instead, correctly: it serves a user, this labels an artifact.
    if speculative_block.is_some() && prefill_chunk.is_some() {
        return Err(
            "speculative decoding and chunked prefill are not composable here: \
                    the speculative loop has its own prefill and no chunked variant, \
                    so a run with both would measure one of them under the other's \
                    label"
                .to_string(),
        );
    }
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
    let result = match (speculative_block, prefill_chunk) {
        // UNCHANGED, deliberately byte for byte: this is the arm every
        // frozen row in this crate was measured on, and the guarantee that
        // the new flag moves none of them is that `None` still reaches this
        // exact call with these exact arguments.
        (None, None) => run_raw_completion(
            runner,
            tokenizer,
            &prompt_ids,
            &config,
            max_context,
            vocab_size,
            progress,
        ),
        // The chunked prefill driver, which is what makes
        // `TURBOSPARK_ROUTED_BATCH` and `TURBOSPARK_BATCHED_GEMV` reachable
        // from this harness at all: both are read INSIDE the runtime's chunk
        // drivers, so neither needs wiring here, only engaging.
        (None, Some(chunk)) => run_raw_completion_chunked(
            runner,
            tokenizer,
            &prompt_ids,
            &config,
            max_context,
            vocab_size,
            chunk,
            progress,
        ),
        // REFUSES rather than falling back when the install cannot serve a
        // drafter, which is the same contract `--speculative` gives on the
        // CLI: a power capture that quietly measured the non-speculative
        // engine and reported it under a `spec` arm label is the exact
        // failure `scripts/power.sh`'s resolved-parameter header exists to
        // prevent.
        (Some(block), _) => run_raw_completion_speculative(
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
        prefill_chunk_tokens: prefill_chunk,
        new_tokens: result.new_tokens,
        decode_seconds: result.decode_seconds,
        peak_footprint_bytes: sampler.peak_bytes(),
        reason: result.reason,
    })
}
