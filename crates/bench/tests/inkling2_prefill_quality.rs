#![cfg(target_os = "macos")]
//! Reference-answer perplexity through Inkling 2's DEFAULT prefill (chunked,
//! matrix units on a resident native-expert install), on a real install.
//!
//! The Phase 6 quality gates (`inkling2*_quality_gate.rs`) open streamed,
//! which prefills one token at a time, so they never exercise the chunked
//! prefill the CLI and app take on a resident install. This scores the same
//! reference answer in the same assistant slot three ways:
//! - sequential on a resident runner (`produce` per token, bit-exact with
//!   the exact prefill arm);
//! - the prompt through `prefill_chunk` (the default: matrix-unit GEMMs,
//!   experts and attention), then the answer teacher-forced;
//! - sequential on a streamed runner: the floor, a deviation the project
//!   already ships (host routing rounds `exp` differently).
//!
//! Bound, set before the first run: |chunked - sequential| at most
//! max(3 x |streamed - sequential|, 0.1% of the sequential perplexity). The
//! 0.1% term keeps the bound defined if the floor happens to land near zero;
//! it is one unit in the fourth decimal the frozen rows record.
//!
//!   TURBOSPARK_INKLING2_INSTALL_DIR=~/models/inkling2-mini-gturbo \
//!     cargo test -p turbospark-bench --test inkling2_prefill_quality --release -- --ignored --nocapture
//!
//! It opens a resident runner (about 17.7 GB) and then a streamed one: do not
//! run it beside another resident-install test on a 36 GB machine.

mod quality_common;

use foundation::LogitValue;
use runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};

const INKLING2_ASSISTANT_PREFIX: &str = "<|message_model|><|content_text|>";
const FLOOR_HEADROOM: f64 = 3.0;
const MIN_RELATIVE: f64 = 1e-3;

fn nll(logits: &[LogitValue], target: i32) -> f64 {
    let max = logits
        .iter()
        .fold(f32::NEG_INFINITY, |a, &v| a.max(v.to_f32())) as f64;
    assert!(max.is_finite(), "no finite logit");
    let sum: f64 = logits
        .iter()
        .map(|&v| (v.to_f32() as f64 - max).exp())
        .sum();
    (max + sum.ln()) - logits[target as usize].to_f32() as f64
}

/// Perplexity of `answer` after `prompt`: the prompt one token at a time
/// (`chunked` false) or through `prefill_chunk` (true), the answer
/// teacher-forced either way.
fn perplexity(
    runner: &mut RealForwardRunner,
    prompt: &[i32],
    answer: &[i32],
    chunked: bool,
) -> f64 {
    runner.reset();
    let mut logits = vec![LogitValue::from_f32(0.0); runner.vocab_size()];
    if chunked {
        runner
            .prefill_chunk(prompt, 0, &mut logits)
            .expect("prefill_chunk");
    } else {
        for (pos, &t) in prompt.iter().enumerate() {
            runner.produce(t, pos, &mut logits).expect("produce");
        }
    }
    // `logits` now predicts answer[0].
    let mut sum = nll(&logits, answer[0]);
    for i in 0..answer.len() - 1 {
        runner
            .produce(answer[i], prompt.len() + i, &mut logits)
            .expect("produce");
        sum += nll(&logits, answer[i + 1]);
    }
    (sum / answer.len() as f64).exp()
}

fn open(dir: &std::path::Path, residency: runtime::ExpertResidency) -> RealForwardRunner {
    RealForwardRunner::open_with_residency(
        dir,
        model_io::inkling2_mini(),
        8192,
        runtime::ExpertCacheSlots::Auto,
        runtime::DraftPolicies::off(),
        runtime::SteeringPolicy::off(),
        1,
        runtime::KvQuant::Off,
        residency,
    )
    .expect("open")
}

#[test]
#[ignore = "needs a real Inkling 2 .gturbo install (TURBOSPARK_INKLING2_INSTALL_DIR)"]
fn default_prefill_keeps_reference_perplexity_within_the_floor_bound() {
    let Some(dir) = std::env::var_os("TURBOSPARK_INKLING2_INSTALL_DIR") else {
        eprintln!("TURBOSPARK_INKLING2_INSTALL_DIR is not set; skipping.");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let tokenizer = tokenizer::MfTokenizer::load_from_dir(&dir).expect("tokenizer");
    let mut prompt = quality_common::user_turn_ids(&tokenizer);
    let prefix = tokenizer.encode(INKLING2_ASSISTANT_PREFIX, false);
    assert_eq!(
        prefix.len(),
        2,
        "the assistant prefix is two special tokens"
    );
    prompt.extend(&prefix);
    let answer = tokenizer.encode(quality_common::REFERENCE_ANSWER, false);
    assert!(answer.len() > 16);
    eprintln!(
        "prompt {} tokens, answer {} tokens",
        prompt.len(),
        answer.len()
    );

    let mut resident = open(&dir, runtime::ExpertResidency::Resident);
    assert!(
        resident.supports_chunked_prefill(),
        "this install has no chunked prefill to gate"
    );
    let seq = perplexity(&mut resident, &prompt, &answer, false);
    let chunk = perplexity(&mut resident, &prompt, &answer, true);
    drop(resident);
    let mut streamed = open(&dir, runtime::ExpertResidency::Streamed);
    let floor = perplexity(&mut streamed, &prompt, &answer, false);

    let (d, f) = ((chunk - seq).abs(), (floor - seq).abs());
    let bound = (FLOOR_HEADROOM * f).max(MIN_RELATIVE * seq);
    eprintln!(
        "reference perplexity: sequential {seq:.4}, chunked default {chunk:.4} (diff {d:.4}), \
         streamed floor {floor:.4} (diff {f:.4}); bound {bound:.4}"
    );
    assert!(
        d <= bound,
        "the default prefill moved the reference perplexity by {d:.4}, over {bound:.4}"
    );
}
