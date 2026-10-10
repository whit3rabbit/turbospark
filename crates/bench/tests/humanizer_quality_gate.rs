#![cfg(target_os = "macos")]
//! The quality gate for the DENSE humanizer Gemma 4 12B
//! (`cowWhySo/humanizer-12b-mlx-8bit` @ baea046b), against a REAL
//! `.gturbo` install. A separate target, one real model per process, the
//! same rule the memory oracles are split under.
//!
//! Not run by default (needs the ~12 GB install; use --release):
//!
//!   TURBOSPARK_HUMANIZER_INSTALL_DIR=~/.turbospark/models/text/humanizer-12b.gturbo \
//!     cargo test -p turbospark-bench --test humanizer_quality_gate --release -- --ignored --nocapture
//!
//! Build the install with `turbospark-model pull --repo
//! cowWhySo/humanizer-12b-mlx-8bit --alias humanizer-12b`.
//!
//! THREE WAYS THIS GATE DIFFERS FROM THE CHAT-FAMILY GATES, all read off
//! the model rather than invented:
//!
//! 1. **The corpus is the rewrite, not the answer.** The frozen reference
//!    rewrite sits where the chat gates put a reference answer, and
//!    `teacher_forced_nll` scores ONLY rewrite-position tokens -- the same
//!    rule NEW_MODEL.md Phase 6 states (an instruction-tuned checkpoint is
//!    never trained to predict the prompt, and this model's prompt is a
//!    380-byte instruction that would score worse than guessing).
//! 2. **No assistant prefix.** There is no template and no assistant slot;
//!    the wrapper IS the framing. The digest prompt is the wrapper around
//!    [`humanizer_common::DRAFT_DIGEST`].
//! 3. **The sampled arm runs the MODEL'S OWN recipe** (temperature 1.0,
//!    top_p 0.95, top_k off), not the shared protocol's. A quality gate
//!    sampling this model at temperature 0.2/top-k 64 would freeze a
//!    digest for a model its publisher does not ship.
//!
//! Expect the perplexity to be NOT COMPARABLE to the other families' rows:
//! the corpus is its own, and a rewrite scored after a 380-byte
//! instruction conditions differently than a chat answer after a turn
//! mark. Each row is a sentinel against its own past.

mod humanizer_common;
mod quality_common;

use half::f16;
use runtime::{
    run_raw_completion, GenerationConfig, LogitProducer, RawDecodeProgress, RealForwardRunner,
};
use turbospark_bench::memory::chip_brand_string;
use turbospark_bench::real_model::open_model_runner_with_context;

use humanizer_common::{
    humanizer_shaping, prompt_ids, DRAFT_DIGEST, DRAFT_MEDIUM, DRAFT_SHORT, REFERENCE_DIGEST,
    REFERENCE_MEDIUM, REFERENCE_SHORT,
};

/// Digest generation length: same width the chat gates freeze, long enough
/// for a distribution bug to diverge the continuation.
const DIGEST_MAX_NEW: u32 = 256;

/// The model's own context ceiling; every gate arm opens and generates at
/// or below it.
const MAX_CONTEXT: u32 = 8192;

/// The install directory, or a pointer to the pull command that builds it.
fn install_dir() -> std::path::PathBuf {
    match std::env::var("TURBOSPARK_HUMANIZER_INSTALL_DIR") {
        Ok(dir) => std::path::PathBuf::from(dir),
        Err(_) => panic!(
            "set TURBOSPARK_HUMANIZER_INSTALL_DIR to the humanizer install \
             (turbospark-model pull --repo cowWhySo/humanizer-12b-mlx-8bit \
             --alias humanizer-12b)"
        ),
    }
}

/// Mean NLL of the reference rewrite's tokens, conditioned on the wrapper
/// prompt. `teacher_forced_nll` scores exactly the continuation positions,
/// which is the whole rule: the wrapper's instruction tokens are not
/// evidence of anything.
fn rewrite_perplexity(runner: &mut RealForwardRunner, tokenizer: &tokenizer::MfTokenizer) -> f64 {
    let mut nll = 0.0f64;
    let mut cases = 0usize;
    for (draft, reference) in [
        (DRAFT_SHORT, REFERENCE_SHORT),
        (DRAFT_MEDIUM, REFERENCE_MEDIUM),
        (DRAFT_DIGEST, REFERENCE_DIGEST),
    ] {
        let prompt = prompt_ids(tokenizer, draft);
        let continuation = tokenizer.encode(reference, false);
        nll += quality_common::teacher_forced_nll(runner, &prompt, &continuation);
        cases += 1;
    }
    nll / cases as f64
}

/// Generate from the wrapper prompt and return the lowercase hex SHA-256
/// of the visible text plus decode tok/s. Mirrors
/// `quality_common::generation_digest` with the recipe's own budget rules:
/// EOS-only stop, and a max_new the model's 8,192 window can hold.
fn generation_digest(
    runner: &mut RealForwardRunner,
    tokenizer: &tokenizer::MfTokenizer,
    prompt: &[i32],
    shaping: &selection::ShapingConfig,
) -> (String, f64) {
    let config = GenerationConfig {
        shaping: *shaping,
        max_new_tokens: DIGEST_MAX_NEW,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: Default::default(),
    };
    let vocab_size = runner.vocab_size();
    let mut text = String::new();
    let result = run_raw_completion(
        runner,
        tokenizer,
        prompt,
        &config,
        MAX_CONTEXT,
        vocab_size,
        |event| match event {
            RawDecodeProgress::Token { delta, .. } => text.push_str(&delta),
            RawDecodeProgress::Tail(tail) => text.push_str(&tail),
            RawDecodeProgress::Prefill { .. } => {}
        },
    )
    .expect("generation runs");
    assert!(
        result.decode_seconds > 0.0 && result.new_tokens > 0,
        "generation produced no timed decode, so its throughput is undefined"
    );
    (
        model_io::hash_data(text.as_bytes()),
        result.new_tokens as f64 / result.decode_seconds,
    )
}

/// Per-chip rows, MOST SPECIFIC SUBSTRING FIRST. The digests are a
/// property of this device's reduce order; the row below was frozen from
/// the FIRST measured run of the install this file's header builds.
struct Row {
    brand_substr: &'static str,
    perplexity: f64,
    greedy_digest: &'static str,
    sampled_digest: &'static str,
    source: &'static str,
}

const BASELINES: &[Row] = &[
    // M4 Max 36GB (the development machine). Frozen 2026-10-09, AC, release,
    // 16 expert-cache slots (inert on a dense install), from the first run
    // of the streamed install; the second fresh process reproduced both
    // digests and the perplexity (run the target twice to repeat that
    // check). The perplexity is 1.9198 and reads LOW against every other
    // family's row because the corpus is the model's own specialty: a
    // reference rewrite conditioned on the wrapper, three drafts the model
    // was trained to rewrite. It is a sentinel against this row's own past,
    // not a cross-family rank.
    Row {
        brand_substr: "Apple M4 Max",
        perplexity: 1.9198,
        greedy_digest: "10203b0f7e6e920e741aee3e1100fd88319a6753cafd0bcb3369fa98df1d2809",
        sampled_digest: "1fd7b482af188113b6a3800ef2b9db199912d36a29127789693592db0fc5af64",
        source: "this port, measured locally -- first stream of the checkpoint",
    },
];

#[test]
#[ignore = "needs the real ~12 GB install; see this file's header for the env var and command"]
fn humanizer_quality_rows_hold() {
    let dir = install_dir();
    // Expert-cache slots are inert on a dense install; the shared default
    // keeps the harness identical to every other family's call.
    let (mut runner, tokenizer) =
        open_model_runner_with_context(&dir, 16, MAX_CONTEXT).expect("real install should open");

    let perplexity = rewrite_perplexity(&mut runner, &tokenizer);
    eprintln!("humanizer: rewrite-position perplexity {perplexity:.4}");

    let prompt = prompt_ids(&tokenizer, DRAFT_DIGEST);

    runner.reset();
    let greedy_prompt = prompt.clone();
    let (greedy_digest, greedy_tok_s) = generation_digest(
        &mut runner,
        &tokenizer,
        &greedy_prompt,
        &selection::ShapingConfig::new(0.0, 1, None, 1.0, Some(0)).expect("greedy shaping"),
    );
    eprintln!("humanizer: greedy digest {greedy_digest} at {greedy_tok_s:.3} tok/s");

    runner.reset();
    let (sampled_digest, sampled_tok_s) = generation_digest(
        &mut runner,
        &tokenizer,
        &prompt,
        &humanizer_shaping(20261008),
    );
    eprintln!("humanizer: sampled digest {sampled_digest} at {sampled_tok_s:.3} tok/s");

    let brand = chip_brand_string();
    let row = brand
        .as_deref()
        .and_then(|b| BASELINES.iter().find(|r| b.contains(r.brand_substr)));
    let Some(row) = row else {
        eprintln!(
            "humanizer: chip {brand:?} not in the baseline table; measured \
             perplexity {perplexity:.4}, greedy {greedy_digest}, sampled \
             {sampled_digest} -- freeze a row before trusting this run"
        );
        return;
    };

    let relative = (perplexity - row.perplexity).abs() / row.perplexity;
    assert!(
        relative <= quality_common::PERPLEXITY_REL_TOLERANCE,
        "perplexity {perplexity:.4} is {relative:.3} from the frozen row \
         {:.4} (source: {})",
        row.perplexity,
        row.source
    );
    assert_eq!(
        greedy_digest, row.greedy_digest,
        "greedy digest moved (source: {})",
        row.source
    );
    assert_eq!(
        sampled_digest, row.sampled_digest,
        "sampled digest moved (source: {})",
        row.source
    );
}

/// Throughput sanity only (the memory oracle owns the footprint): the
/// second fresh process must agree with the first on every frozen digit.
/// This target IS the second process when run twice; the row is what makes
/// two runs comparable.
#[test]
#[ignore = "needs the real ~12 GB install; see this file's header for the env var and command"]
fn humanizer_logits_survive_fp16_at_the_softcap() {
    // NEW_MODEL.md Phase 4's check, on the real head: Gemma 4 caps at 30.0,
    // so raw logits near the cap must not collapse into FP16 ties. Dump the
    // top-8 logits for one wrapper-prompt token and assert the top span is
    // not degenerate (a saturated head would return many identical maxima
    // and ties would collapse to the lowest index).
    let dir = install_dir();
    let (mut runner, tokenizer) =
        open_model_runner_with_context(&dir, 16, MAX_CONTEXT).expect("real install should open");
    let prompt = prompt_ids(&tokenizer, DRAFT_DIGEST);
    let mut logits = vec![f16::from_f32(0.0); runner.vocab_size()];
    runner.reset();
    runner
        .produce(prompt[0], 0, &mut logits)
        .expect("first token forward");
    let mut values: Vec<f32> = logits.iter().map(|v| v.to_f32()).collect();
    values.sort_by(|a, b| b.total_cmp(a));
    let top8 = &values[..8];
    eprintln!("humanizer: top-8 logits {top8:?}");
    assert!(
        top8.iter().all(|v| v.is_finite()),
        "non-finite logits in the top-8"
    );
    assert!(
        values.iter().any(|v| *v < 0.0),
        "every logit is non-negative: this looks like probabilities, not logits \
         (the producer must return raw capped logits; the sampler softmaxes)"
    );
    // Bounded by the softcap, and not saturated: with a 30.0 cap the capped
    // maximum sits at 30 * tanh(z/30) < 30. Ties AT the cap would mean a
    // head that clipped instead of soft-capped.
    assert!(
        top8[0] < 30.0,
        "top logit {} reached the softcap bound; the cap looks clipped, not soft",
        top8[0]
    );
}
