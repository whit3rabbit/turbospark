//! Real-install benchmark flow (`turbospark-bench --model <dir>`): opens a
//! `.gturbo` install with `RealForwardRunner` and drives the frozen
//! community-protocol cases through the real prefill+decode loop,
//! chat-formatted exactly as the CLI formats them (the IT checkpoint needs
//! its turn markup; raw prompts babble). Reports the split
//! prefill/decode seconds from `RawDecodeResult` (not wall time) plus the
//! peak `phys_footprint` sampled at the Swift cadence: before prefill, on
//! every 8th decoded token, and once after the run.

use std::path::Path;

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
    let arch = repack::peek_manifest_arch(model_dir)?;
    let tokenizer = MfTokenizer::load_from_dir(model_dir).map_err(|e| {
        format!(
            "failed to load a tokenizer from {}: {e}",
            model_dir.display()
        )
    })?;
    let runner =
        RealForwardRunner::open_with_options(model_dir, arch, PROTOCOL_MAX_CONTEXT as usize, slots)
            .map_err(|e| e.to_string())?;
    Ok((runner, tokenizer))
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
        max_new_tokens: PROTOCOL_MAX_NEW,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate,
    };

    sampler.sample();
    let result = run_raw_completion(
        runner,
        tokenizer,
        &prompt_ids,
        &config,
        PROTOCOL_MAX_CONTEXT,
        tokenizer.vocab_size,
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
