#![cfg(target_os = "macos")]
//! End-to-end proof that a real (not scripted) forward pass exists: builds
//! a tiny synthetic Gemma-4-shaped `.gturbo` install (real INT4-affine
//! quantized weights, deterministic but not trained), opens it with
//! `RealForwardRunner`, and drives it through the same `run_raw_completion`
//! loop every other `LogitProducer` in this crate uses. Runs the real GPU
//! kernel stack on real Metal hardware. Since the weights are synthetic,
//! the generated token ids are not semantically meaningful — only the
//! pipeline (embedding lookup, per-layer projections, RoPE, attention,
//! FFN, final softcapped-softmax, sampling, detokenization, stop handling)
//! is real, see `real_forward.rs`'s own module docs and `DEVIATIONS.md`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use mrefrust_repack::{build_synthetic_gemma4_install, build_synthetic_gemma4_moe_install};
use mrefrust_runtime::{
    run_raw_completion, GenerationConfig, RawDecodeProgress, RealForwardRunner,
};
use selection::ShapingConfig;
use tokenizer::MfTokenizer;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir =
        std::env::temp_dir().join(format!("mrefrust-real-forward-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn load_tokenizer() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

fn greedy_config(max_new_tokens: u32) -> GenerationConfig {
    GenerationConfig {
        shaping: ShapingConfig::new(0.0, 0, None, 1.0, None).unwrap(),
        max_new_tokens,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
    }
}

#[test]
fn real_forward_runner_generates_real_tokens_deterministically() {
    let tokenizer = load_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    let dir = temp_dir();
    let arch = build_synthetic_gemma4_install(&dir, vocab_size as i64, 2, "real-forward-test")
        .expect("synthetic install should write");

    let prompt_ids = tokenizer.encode("hi", false);
    assert!(!prompt_ids.is_empty());

    let mut runner = RealForwardRunner::open(&dir, arch).expect("install should open");
    let config = greedy_config(4);

    let mut first_tokens = Vec::new();
    let first = run_raw_completion(
        &mut runner,
        &tokenizer,
        &prompt_ids,
        &config,
        4096,
        vocab_size,
        |e| {
            if let RawDecodeProgress::Token { id, .. } = e {
                first_tokens.push(id);
            }
        },
    )
    .expect("real forward pass should run to a stop condition");

    assert!(first.new_tokens >= 1, "should generate at least one token");
    assert_eq!(first.prompt_tokens, prompt_ids.len());
    assert!(
        !first_tokens.is_empty(),
        "the loop should have emitted at least one real (non-scripted) token"
    );

    // Determinism: re-running the same runner (which `run_raw_completion`
    // resets internally) over the same prompt at temperature 0 reaches the
    // same generated tokens, since the weights and KV history are both
    // deterministic.
    let mut second_tokens = Vec::new();
    let second = run_raw_completion(
        &mut runner,
        &tokenizer,
        &prompt_ids,
        &config,
        4096,
        vocab_size,
        |e| {
            if let RawDecodeProgress::Token { id, .. } = e {
                second_tokens.push(id);
            }
        },
    )
    .expect("second real forward pass should also run to a stop condition");

    assert_eq!(first.new_tokens, second.new_tokens);
    assert_eq!(first_tokens, second_tokens);
    assert_eq!(first.reason, second.reason);
}

#[test]
fn real_forward_runner_generates_real_tokens_through_a_moe_layer() {
    let tokenizer = load_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    let dir = temp_dir();
    // 4 experts, top-2: exercises real router GEMV dispatch, host-side
    // top-k selection, and per-selected-expert FFN combine.
    let arch = build_synthetic_gemma4_moe_install(&dir, vocab_size as i64, 2, 4, 2, "moe-test")
        .expect("synthetic MoE install should write");

    let prompt_ids = tokenizer.encode("hi", false);
    assert!(!prompt_ids.is_empty());

    let mut runner = RealForwardRunner::open(&dir, arch).expect("MoE install should open");
    let config = greedy_config(4);

    let mut first_tokens = Vec::new();
    let first = run_raw_completion(
        &mut runner,
        &tokenizer,
        &prompt_ids,
        &config,
        4096,
        vocab_size,
        |e| {
            if let RawDecodeProgress::Token { id, .. } = e {
                first_tokens.push(id);
            }
        },
    )
    .expect("MoE forward pass should run to a stop condition");
    assert!(first.new_tokens >= 1);

    let mut second_tokens = Vec::new();
    let second = run_raw_completion(
        &mut runner,
        &tokenizer,
        &prompt_ids,
        &config,
        4096,
        vocab_size,
        |e| {
            if let RawDecodeProgress::Token { id, .. } = e {
                second_tokens.push(id);
            }
        },
    )
    .expect("second MoE forward pass should also run to a stop condition");

    assert_eq!(
        first_tokens, second_tokens,
        "MoE routing must be deterministic"
    );
    assert_eq!(first.reason, second.reason);
}
