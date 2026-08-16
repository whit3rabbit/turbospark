//! Real-install benchmark flow (`turbospark-bench --model <dir>`): opens a
//! `.gturbo` install with `RealForwardRunner` and drives the frozen
//! community-protocol cases through the real prefill+decode loop,
//! chat-formatted exactly as the CLI formats them (the IT checkpoint needs
//! its turn markup; raw prompts babble). Reports the split
//! prefill/decode seconds from `RawDecodeResult` (not wall time) plus the
//! peak `phys_footprint` sampled at the Swift cadence: before prefill, on
//! every 8th decoded token, and once after the run.

use std::path::Path;

use model_io::{ArchConfig, ModelFamily};
use runtime::{
    run_raw_completion, GenerationConfig, RateControl, RawDecodeProgress, RealForwardRunner,
    StopReason,
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

/// The KV window and the generation budget the frozen protocol runs a given
/// family at. Both are PER-FAMILY parameters and neither is a knob: see
/// [`open_model_runner_with_context`] and [`run_protocol_case_with_budget`]
/// for why each one had to stop being a shared constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProtocolParameters {
    /// The family these came from, so a caller can NAME it in a header
    /// without peeking `manifest.json` a second time. Both numbers below are
    /// meaningless without it.
    pub family: ModelFamily,
    /// KV window the runner is opened at, and the limit the generation loop
    /// enforces. MUST be the same number in both places.
    pub max_context: u32,
    /// New-token budget per case.
    pub max_new: u32,
}

/// The dense `llama` window (`mistral_memory_oracle.rs`'s). `long-synthesis`
/// is 3,444 tokens under Mistral's 32k vocab and `3444 + PROTOCOL_MAX_NEW`
/// does not fit 4,096, so the case does not run at all at the shared window.
const DENSE_LLAMA_MAX_CONTEXT: u32 = 8192;
/// The `gpt-oss` window (`gptoss_memory_oracle.rs`'s). Follows from the
/// budget below: `long-synthesis` is 2,839 tokens under o200k and
/// `2839 + 3072` does not fit 4,096.
const GPTOSS_MAX_CONTEXT: u32 = 8192;
/// The `gpt-oss` budget (`gptoss_memory_oracle.rs`'s). Harmony puts the
/// model's reasoning in an `analysis` channel BEFORE its answer, so the three
/// cases need 818 / 2,153 / 1,108 sampled tokens to reach `<|return|>`.
const GPTOSS_MAX_NEW: u32 = 3072;
/// The `muse_glimmer` window (`museglimmer_memory_oracle.rs`'s). Follows from
/// the budget below: `long-synthesis` is 2,820 tokens once the template's
/// system preamble is counted, and `2820 + 2048` does not fit 4,096.
const MUSE_GLIMMER_MAX_CONTEXT: u32 = 8192;
/// The `muse_glimmer` budget. The model reasons to a `to=self` message before
/// its `to=user` answer, so the three cases need 1,054 / 1,378 / 1,246
/// sampled tokens to reach `<|eot|>` -- the SHORT one already exceeds the
/// shared 1,024.
const MUSE_GLIMMER_MAX_NEW: u32 = 2048;

/// Resolve the protocol's two per-family parameters from the install's own
/// declared family.
///
/// **The match is exhaustive with NO wildcard arm, and that is the guard
/// rather than a style choice.** A `_ =>` here would let a seventh family
/// silently inherit Gemma's window and budget, which is the shape of bug
/// AGENTS.md Gotchas 24, 37 and 39 are all instances of: a default is a
/// claim about what silence means, and a table keyed by X holding a property
/// of Y stays invisible for exactly as long as the mapping is injective.
/// Adding a family should not compile until someone has answered this.
///
/// The two moved rows are the oracles' own numbers, asserted equal by
/// `mistral_memory_oracle.rs` and `gptoss_memory_oracle.rs` so the binary and
/// the oracles cannot drift apart. Read a peak or a tok/s row WITH these two
/// numbers; a row taken at one window says nothing about another (crate
/// Gotchas 11 and 12).
///
/// `const` so the two oracle targets can assert agreement in a `const`
/// block, which fails the BUILD rather than only firing on the rare
/// occasions those `#[ignore]`d targets run with an install present.
pub const fn protocol_parameters(family: ModelFamily) -> ProtocolParameters {
    match family {
        // The shared protocol, and what every frozen row in
        // `docs/BENCHMARKS.md` was measured at.
        // `qwen3_5` is here on the TOKENIZER's evidence rather than on a
        // measurement, and this comment is the answer the doc above asks
        // for. The window is decided by how many tokens the frozen prose
        // becomes, which belongs to the checkpoint's tokenizer -- and this
        // one declares `vocab_size: 248320`, Qwen 3.6's exactly, under the
        // same ChatML dialect. So the protocol's ~2.8k-token long case fits
        // 4,096 for the same reason it does there, and NOT for the reason
        // it fails on the dense `llama` half below (Mistral's 32k
        // sentencepiece vocab makes the same prose 3,444 tokens).
        // UNVERIFIED until an install exists: the first memory-oracle run
        // is what confirms it, and a `long-synthesis` that stops on
        // maxTokens is what would refute it.
        ModelFamily::Gemma4
        | ModelFamily::QwenGdnMoe
        | ModelFamily::QwenGdnDense
        | ModelFamily::Qwen3Moe
        | ModelFamily::DeepseekV4Flash => ProtocolParameters {
            family,
            max_context: PROTOCOL_MAX_CONTEXT,
            max_new: PROTOCOL_MAX_NEW,
        },
        // ONE FAMILY, BOTH HALVES OF THE ARCHITECTURE STRING. The window is
        // the dense half's requirement, measured on Mistral-7B-Instruct-v0.3
        // (ROADMAP M4). Mixtral shares that checkpoint's 32k sentencepiece
        // tokenizer, so the same window is right for it by the same
        // arithmetic -- not measured there, and it will not be: Mixtral
        // cannot stream on this engine at any useful slot count and runs the
        // protocol at 0.16 tok/s (AGENTS.md Gotcha 36).
        ModelFamily::Llama => ProtocolParameters {
            family,
            max_context: DENSE_LLAMA_MAX_CONTEXT,
            max_new: PROTOCOL_MAX_NEW,
        },
        // THE SECOND FAMILY THAT MOVES BOTH, and it moves them for gpt-oss's
        // reason: it REASONS BEFORE ANSWERING. `muse_glimmer`'s system
        // preamble carries `Reasoning strength: high.` and the model emits a
        // `to=self` message before its `to=user` one, so the shared 1,024
        // budget truncates the SHORT case, never mind the long one.
        //
        // **AN EARLIER DRAFT OF THIS FILE PUT THIS FAMILY IN THE SHARED
        // GROUP, and the mistake is worth keeping written down.** The prompt
        // side was measured properly -- the three frozen prompts encode to
        // 46 / 408 / 2,764 tokens under this checkpoint's own
        // `tokenizer.json`, so `2764 + 1024` fits 4,096 -- and the OUTPUT
        // side was assumed, in a comment that said "the model emits no
        // reasoning channel, so the shared 1,024 budget stands". Measuring
        // the prompt and assuming the completion is exactly the shape of
        // AGENTS.md Gotchas 37 to 39: half the question answered from the
        // file and half from expectation.
        //
        // Measured end to end at 8,192 / 3,072, greedy, on the real install
        // (2026-08-15), all three stopping endOfTurn:
        //   short-explanation   102 prompt + 1,054 new = 1,156
        //   medium-review       464 prompt + 1,378 new = 1,842
        //   long-synthesis    2,820 prompt + 1,246 new = 4,066
        // Note the prompts are LONGER than the raw prose tokenizes to (102
        // against 46) because the rendered template adds ~56 tokens of
        // system preamble, which is the other half of what a raw
        // tokenization misses.
        //
        // 2,048 is 1.49x the largest observed completion, the same margin
        // gpt-oss's 3,072 takes over its 2,153. The WINDOW has to be 8,192
        // because `2,820 + 2,048` does not fit 4,096 -- and note the largest
        // total observed (4,066) fits 4,096 with 30 tokens to spare, which
        // is exactly the kind of margin that turns a frozen row red on a
        // checkpoint revision.
        ModelFamily::MuseGlimmer => ProtocolParameters {
            family,
            max_context: MUSE_GLIMMER_MAX_CONTEXT,
            max_new: MUSE_GLIMMER_MAX_NEW,
        },
        // The only family that moves BOTH.
        ModelFamily::GptOss => ProtocolParameters {
            family,
            max_context: GPTOSS_MAX_CONTEXT,
            max_new: GPTOSS_MAX_NEW,
        },
    }
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
    let arch = repack::peek_manifest_arch(model_dir)?;
    let params = protocol_parameters(arch.family);
    let (runner, tokenizer) = open_with_arch(model_dir, arch, slots, params.max_context)?;
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
    // Chat-format exactly as the CLI does: the dialect template renders
    // the turn markup (and its own <bos>, hence add_bos false).
    let messages = [Message::new(Role::User, case.content)];
    let rendered = tokenizer
        .apply_chat_template(&messages)
        .map_err(|e| format!("chat template: {e}"))?;
    let prompt_ids = tokenizer.encode(&rendered, false);

    let config = GenerationConfig {
        shaping: ShapingConfig::new(
            PROTOCOL_TEMPERATURE,
            PROTOCOL_TOP_K,
            Some(PROTOCOL_TOP_P),
            1.0,
            Some(case.seed),
        )
        .map_err(|e| e.to_string())?,
        max_new_tokens: max_new,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate,
    };

    // Read before the mutable borrow: the logits width is the MODEL's, not
    // the tokenizer dialect's (`RealForwardRunner::vocab_size`).
    let vocab_size = runner.vocab_size();
    sampler.sample();
    let result = run_raw_completion(
        runner,
        tokenizer,
        &prompt_ids,
        &config,
        max_context,
        vocab_size,
        |event| {
            // The Swift runtime samples every 8th decoded token.
            if let RawDecodeProgress::Token { index, .. } = event {
                if index % 8 == 0 {
                    sampler.sample();
                }
            }
        },
    )
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The families whose published rows were measured at the shared
    /// protocol. If one of these ever moves, every number in
    /// `docs/BENCHMARKS.md` for that family is at a different workload.
    #[test]
    fn four_families_run_the_shared_protocol_parameters() {
        for family in [
            ModelFamily::Gemma4,
            ModelFamily::QwenGdnMoe,
            ModelFamily::Qwen3Moe,
            ModelFamily::DeepseekV4Flash,
        ] {
            let params = protocol_parameters(family);
            assert_eq!(
                params.max_context,
                PROTOCOL_MAX_CONTEXT,
                "{} is measured at the shared window; moving it invalidates its frozen rows",
                family.as_str()
            );
            assert_eq!(
                params.max_new,
                PROTOCOL_MAX_NEW,
                "{} is measured at the shared budget; moving it invalidates its frozen rows",
                family.as_str()
            );
        }
    }

    /// The dense `llama` window, which `mistral_memory_oracle.rs` freezes a
    /// 1,300 MiB ceiling against.
    #[test]
    fn the_llama_family_runs_at_the_dense_window() {
        let params = protocol_parameters(ModelFamily::Llama);
        assert_eq!(params.max_context, 8192, "mistral_memory_oracle.rs's row");
        assert_eq!(params.max_new, PROTOCOL_MAX_NEW);
        assert!(
            params.max_context > PROTOCOL_MAX_CONTEXT,
            "the point of the row is that `long-synthesis` does not fit 4,096 \
             under a 32k-vocab tokenizer"
        );
    }

    /// The one family that moves BOTH parameters, which
    /// `gptoss_memory_oracle.rs` freezes a 5,700 MiB ceiling against.
    #[test]
    fn the_gptoss_family_moves_both_parameters() {
        let params = protocol_parameters(ModelFamily::GptOss);
        assert_eq!(params.max_context, 8192, "gptoss_memory_oracle.rs's row");
        assert_eq!(params.max_new, 3072, "gptoss_memory_oracle.rs's row");
        assert!(
            params.max_new > PROTOCOL_MAX_NEW,
            "Harmony's reasoning channel needs 2,153 sampled tokens on \
             `medium-review`; at the shared budget two of three cases stop on \
             maxTokens"
        );
    }
}
