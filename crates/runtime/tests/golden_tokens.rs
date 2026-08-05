#![cfg(target_os = "macos")]
//! Golden token-sequence guard for the memory-path rework.
//!
//! `real_forward.rs` asserts two runs of the SAME binary agree; this file
//! pins the exact greedy token sequences the pre-rework (copy-everything)
//! forward pass produced, so the zero-copy/persistent-KV rewrites can be
//! proven numerically identical, not just self-consistent. The synthetic
//! weights, the tokenizer fixture, and temperature-0 selection are all
//! deterministic, so these sequences are stable across machines; if a
//! change legitimately alters kernel math (it should not — the memory
//! rework must be bit-identical), this test is the tripwire.

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
    let dir = std::env::temp_dir().join(format!("mrefrust-golden-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn load_tokenizer() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

fn generate(runner: &mut RealForwardRunner, tokenizer: &MfTokenizer, max_new: u32) -> Vec<i32> {
    let config = GenerationConfig {
        shaping: ShapingConfig::new(0.0, 0, None, 1.0, None).unwrap(),
        max_new_tokens: max_new,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
    };
    let prompt_ids = tokenizer.encode("golden fixture prompt", false);
    assert!(!prompt_ids.is_empty());
    let mut tokens = Vec::new();
    run_raw_completion(
        runner,
        tokenizer,
        &prompt_ids,
        &config,
        4096,
        tokenizer.vocab_size,
        |e| {
            if let RawDecodeProgress::Token { id, .. } = e {
                tokens.push(id);
            }
        },
    )
    .expect("golden generation should run to a stop condition");
    tokens
}

#[test]
fn dense_golden_token_sequence_is_unchanged() {
    let tokenizer = load_tokenizer();
    let dir = temp_dir();
    let arch = build_synthetic_gemma4_install(&dir, tokenizer.vocab_size as i64, 2, "golden-dense")
        .expect("synthetic install should write");
    let mut runner = RealForwardRunner::open(&dir, arch).expect("install should open");
    let tokens = generate(&mut runner, &tokenizer, 8);
    eprintln!("dense golden tokens: {tokens:?}");
    assert_eq!(tokens, GOLDEN_DENSE, "dense forward pass output changed");
}

#[test]
fn moe_golden_token_sequence_is_unchanged() {
    let tokenizer = load_tokenizer();
    let dir = temp_dir();
    let arch = build_synthetic_gemma4_moe_install(
        &dir,
        tokenizer.vocab_size as i64,
        2,
        4,
        2,
        "golden-moe",
    )
    .expect("synthetic MoE install should write");
    let mut runner = RealForwardRunner::open(&dir, arch).expect("MoE install should open");
    let tokens = generate(&mut runner, &tokenizer, 8);
    eprintln!("moe golden tokens: {tokens:?}");
    assert_eq!(tokens, GOLDEN_MOE, "MoE forward pass output changed");
}

// Captured from the pre-rework copy-everything forward pass (see module
// docs). Regenerate ONLY if a deliberate numeric change is made, and note
// it in DEVIATIONS.md.
// Both shapes greedily settle on token 116 with these untrained synthetic
// weights; the guard is that the value and count stay exactly this, run
// after run, through every memory-path change.
const GOLDEN_DENSE: &[i32] = &[116, 116, 116, 116, 116, 116, 116, 116];
const GOLDEN_MOE: &[i32] = &[116, 116, 116, 116, 116, 116, 116, 116];
