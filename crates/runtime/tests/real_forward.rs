#![cfg(target_os = "macos")]
//! End-to-end proof that a real (not scripted) forward pass exists: builds
//! a tiny synthetic Gemma-4-shaped `.gturbo` install (real INT4-affine
//! quantized weights, deterministic but not trained), opens it with
//! `RealForwardRunner`, and drives it through the same `run_raw_completion`
//! loop every other `LogitProducer` in this crate uses. Runs the real GPU
//! kernel stack on real Metal hardware. Since the weights are synthetic,
//! the generated token ids are not semantically meaningful — only the
//! pipeline (embedding lookup, per-layer projections, RoPE, attention,
//! FFN, final logit softcap, host sampling, detokenization, stop handling)
//! is real, see `real_forward.rs`'s own module docs and `DEVIATIONS.md`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use mrefrust_repack::{
    build_synthetic_gemma4_install, build_synthetic_gemma4_moe_install,
    build_synthetic_gemma4_moe_streamed_install, build_synthetic_gemma4_swa_install,
};
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

/// A mixed sliding-window/full-attention install (mask alternating 0/1,
/// the shape real Gemma 4 has) runs end to end and deterministically:
/// mask-0 layers attend only the trailing `sliding_window` positions via
/// `kv_start` (parity-tested against the CPU window reference in
/// `crates/gpu/tests/attention_swa.rs`).
#[test]
fn mixed_swa_and_full_attention_layers_generate_deterministically() {
    let tokenizer = load_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    let dir = temp_dir();
    // Window of 4 with a longer prompt: late decode steps genuinely
    // exercise the windowed path (seq_len > window).
    let arch = build_synthetic_gemma4_swa_install(&dir, vocab_size as i64, 4, 4, "swa-test")
        .expect("SWA install should write");
    let mut runner = RealForwardRunner::open(&dir, arch).expect("SWA install should open");
    let prompt_ids = tokenizer.encode("a longer prompt to overflow the window", false);
    assert!(prompt_ids.len() > 4, "prompt must exceed the window");
    let config = greedy_config(6);

    let mut first_tokens = Vec::new();
    run_raw_completion(
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
    .expect("SWA generation should run to a stop condition");
    assert!(!first_tokens.is_empty());

    let mut second_tokens = Vec::new();
    run_raw_completion(
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
    .expect("second SWA generation should run");
    assert_eq!(first_tokens, second_tokens);
}

/// The fp16 SWA KV ring produces the same tokens as a linear layout: the
/// same SWA install opened with a small ring override (wraps during
/// prefill and every decode step) and with a max_context-sized override
/// (identity slot mapping, linear pipeline) must generate identical
/// streams, since the window (4) always fits inside the ring (16).
#[test]
fn swa_kv_ring_wrap_matches_linear_layout() {
    let tokenizer = load_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    let dir = temp_dir();
    let arch = build_synthetic_gemma4_swa_install(&dir, vocab_size as i64, 4, 4, "swa-ring-test")
        .expect("SWA install should write");

    let prompt_ids = tokenizer
        .encode("a much longer prompt that must overflow a sixteen slot kv ring buffer during prefill so the wrapped path really runs", false);
    assert!(
        prompt_ids.len() > 16,
        "prompt ({} tokens) must exceed the 16-slot ring so the wrap happens",
        prompt_ids.len()
    );
    let config = greedy_config(8);

    let run = |ring_override: Option<usize>| -> Vec<i32> {
        let mut runner = RealForwardRunner::open_with_kv_ring_override(
            &dir,
            arch.clone(),
            4096,
            16,
            ring_override,
        )
        .expect("SWA install should open");
        let mut tokens = Vec::new();
        run_raw_completion(
            &mut runner,
            &tokenizer,
            &prompt_ids,
            &config,
            4096,
            vocab_size,
            |e| {
                if let RawDecodeProgress::Token { id, .. } = e {
                    tokens.push(id);
                }
            },
        )
        .expect("SWA generation should run to a stop condition");
        tokens
    };

    // Some(4096) caps the ring at max_context: identity slot mapping and
    // the linear pipeline (seq_len never exceeds the ring), i.e. the old
    // linear layout. Some(16) wraps from the 17th position on.
    let linear_tokens = run(Some(4096));
    let ring_tokens = run(Some(16));
    assert!(!linear_tokens.is_empty());
    assert_eq!(
        linear_tokens, ring_tokens,
        "ring KV layout must not change generated tokens"
    );
}

/// The dense decode hot path must allocate ZERO Metal buffers: weights
/// are the one zero-copy resident buffer, KV and activation scratch are
/// preallocated at open. This is the steady-state memory guarantee the
/// Swift original's design rests on, asserted exactly rather than via a
/// noisy RSS threshold.
#[test]
fn dense_decode_allocates_no_gpu_buffers_per_token() {
    let tokenizer = load_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    let dir = temp_dir();
    let arch = build_synthetic_gemma4_install(&dir, vocab_size as i64, 2, "steady-state")
        .expect("synthetic install should write");
    let mut runner = RealForwardRunner::open(&dir, arch).expect("install should open");
    let prompt_ids = tokenizer.encode("hi", false);

    // Warmup: one full generation to fault in every code path.
    run_raw_completion(
        &mut runner,
        &tokenizer,
        &prompt_ids,
        &greedy_config(2),
        4096,
        vocab_size,
        |_| {},
    )
    .expect("warmup generation");

    let before = runner.gpu_buffer_allocations();
    run_raw_completion(
        &mut runner,
        &tokenizer,
        &prompt_ids,
        &greedy_config(16),
        4096,
        vocab_size,
        |_| {},
    )
    .expect("steady-state generation");
    let after = runner.gpu_buffer_allocations();

    assert_eq!(
        before, after,
        "dense decode must not allocate Metal buffers per token"
    );
}

/// A streamed-expert install (experts in packed_experts/layer files, read
/// through the PreadExpertStreamer's aligned slot cache with parallel
/// pread) carries the SAME deterministic weights as the resident-expert
/// install, so it must generate the exact same token sequence.
#[test]
fn streamed_expert_install_matches_resident_expert_install() {
    let tokenizer = load_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    let config = greedy_config(6);
    let prompt_ids = tokenizer.encode("hi", false);

    let resident_dir = temp_dir();
    let resident_arch =
        build_synthetic_gemma4_moe_install(&resident_dir, vocab_size as i64, 2, 4, 2, "moe-a")
            .expect("resident MoE install should write");
    let mut resident_runner =
        RealForwardRunner::open(&resident_dir, resident_arch).expect("resident install opens");
    let mut resident_tokens = Vec::new();
    run_raw_completion(
        &mut resident_runner,
        &tokenizer,
        &prompt_ids,
        &config,
        4096,
        vocab_size,
        |e| {
            if let RawDecodeProgress::Token { id, .. } = e {
                resident_tokens.push(id);
            }
        },
    )
    .expect("resident MoE generation should run");

    let streamed_dir = temp_dir();
    let streamed_arch = build_synthetic_gemma4_moe_streamed_install(
        &streamed_dir,
        vocab_size as i64,
        2,
        4,
        2,
        "moe-a",
    )
    .expect("streamed MoE install should write");
    let mut streamed_runner =
        RealForwardRunner::open(&streamed_dir, streamed_arch).expect("streamed install opens");
    let mut streamed_tokens = Vec::new();
    run_raw_completion(
        &mut streamed_runner,
        &tokenizer,
        &prompt_ids,
        &config,
        4096,
        vocab_size,
        |e| {
            if let RawDecodeProgress::Token { id, .. } = e {
                streamed_tokens.push(id);
            }
        },
    )
    .expect("streamed MoE generation should run");

    assert!(!resident_tokens.is_empty());
    assert_eq!(
        resident_tokens, streamed_tokens,
        "streamed experts must be numerically identical to resident experts"
    );
}
