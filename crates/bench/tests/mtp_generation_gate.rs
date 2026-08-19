#![cfg(target_os = "macos")]
//! Does the WIRED speculative generation loop produce what the sequential
//! one does, on the real drafter?
//!
//! `mtp_accept_length_probe.rs` measured that speculation pays (1.44x at
//! block 2, `docs/MTP.md`) by driving the primitives directly. This gate
//! covers the layer between that probe and a user:
//! [`runtime::run_raw_completion_speculative`], which is what the CLI and the
//! server reach. The probe's loop and this one are different code, so the
//! probe's losslessness result says nothing about this one.
//!
//! `crates/runtime/tests/speculative.rs` proves the same property against a
//! SCRIPTED drafter, at drafter qualities a real one cannot be steered to
//! (always right, always wrong) and at every corner of the stop ladder. What
//! it cannot prove is that the trait implementation over `RealForwardRunner`
//! forwards to the primitives correctly -- a scripted producer implements the
//! trait too, so a wrong forward there is invisible to it. That is this
//! file's whole job, which is why it asserts identity rather than measuring
//! anything.
//!
//! ```sh
//! TURBOSPARK_MTP_INSTALL_DIR=~/models/qwen38-27b-mtp.gturbo \
//!   cargo test -p turbospark-bench --test mtp_generation_gate --release -- --ignored --nocapture
//! ```

use runtime::{
    run_raw_completion, run_raw_completion_speculative, GenerationConfig, RateControl,
    RawDecodeProgress, RawDecodeResult,
};
use selection::ShapingConfig;
use tokenizer::{Message, Role};
use turbospark_bench::real_model::open_model_runner_speculative;

/// Draft depth this file asks for. Named here rather than set through
/// `MFERENCE_MTP_DRAFT` because the policy is now a PARAMETER: an unset
/// env var means `Auto`, which resolves to a depth too small for the
/// blocks below and would fail deep in the verify rather than at open.
const MTP_DEPTH: usize = 5;

const SLOTS: usize = 16;
const GENERATE: u32 = 96;
/// Block 2 is the measured optimum; 4 is included because it exercises a
/// deeper accepted prefix and a rollback replay of more than one row.
const BLOCKS: [usize; 2] = [2, 4];

fn greedy_config() -> GenerationConfig {
    GenerationConfig {
        // Temperature 0 EXACTLY, not the 0.0001 the smoke scripts use:
        // `is_deterministic` is `== 0.0`, and a speculative run is refused
        // at anything else (acceptance by argmax is exact only here).
        shaping: ShapingConfig::new(0.0, 0, None, 1.0, None).unwrap(),
        max_new_tokens: GENERATE,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        // Explicit, for `crates/bench` Gotcha 4's reason: an inherited rate
        // cap would pace both arms and mask nothing, but it would make the
        // wall clock printed below meaningless.
        rate: RateControl::default(),
    }
}

fn tokens_of(events: &[RawDecodeProgress]) -> Vec<i32> {
    events
        .iter()
        .filter_map(|e| match e {
            RawDecodeProgress::Token { id, .. } => Some(*id),
            _ => None,
        })
        .collect()
}

fn text_of(events: &[RawDecodeProgress]) -> String {
    events
        .iter()
        .map(|e| match e {
            RawDecodeProgress::Token { delta, .. } => delta.as_str(),
            RawDecodeProgress::Tail(t) => t.as_str(),
            _ => "",
        })
        .collect()
}

#[test]
#[ignore = "needs a real MTP install via TURBOSPARK_MTP_INSTALL_DIR"]
fn the_wired_speculative_loop_reproduces_the_sequential_stream() {
    // Read at `open`, so it has to be set before the runner is built.
    // `BLOCKS`'s largest plus the extra KV-row step.

    let dir = std::path::PathBuf::from(
        std::env::var_os("TURBOSPARK_MTP_INSTALL_DIR").expect("TURBOSPARK_MTP_INSTALL_DIR"),
    );
    let (mut runner, tokenizer) = open_model_runner_speculative(
        &dir,
        SLOTS,
        runtime::DraftPolicies::mtp(runtime::MtpDraftPolicy::Fixed(MTP_DEPTH)),
    )
    .expect("install opens");
    assert!(
        runner.mtp_draft_depth() > 0,
        "this install carries no MTP head; the gate would compare a run against itself"
    );
    // The MODEL's width, never the tokenizer's (AGENTS.md Gotcha 37).
    let vocab = runner.vocab_size();
    let config = greedy_config();

    let prompt = {
        let rendered = tokenizer
            .apply_chat_template(&[Message::new(
                Role::User,
                "Write a Python function that merges two sorted lists, then explain \
                 its time and space complexity in detail.",
            )])
            .expect("chat template renders");
        tokenizer.encode(&rendered, false)
    };

    // The reference arm, and it is the NON-SPECULATIVE loop rather than a
    // second speculative one: comparing speculative arms against each other
    // agrees whenever both are wrong the same way.
    //
    // Run twice; the second is the one whose clock is quoted. The first
    // decode of a process is on a cold GPU at low DVFS clocks, worth up to
    // 53% (AGENTS.md Gotcha 20), and it would land on the denominator.
    let mut warm = Vec::new();
    run_raw_completion(
        &mut runner,
        &tokenizer,
        &prompt,
        &config,
        4096,
        vocab,
        |e| warm.push(e),
    )
    .expect("warmup run");

    let mut seq_events = Vec::new();
    let seq_started = std::time::Instant::now();
    let seq: RawDecodeResult = run_raw_completion(
        &mut runner,
        &tokenizer,
        &prompt,
        &config,
        4096,
        vocab,
        |e| seq_events.push(e),
    )
    .expect("sequential run");
    let seq_secs = seq_started.elapsed().as_secs_f64();
    let seq_tokens = tokens_of(&seq_events);
    let seq_text = text_of(&seq_events);

    println!(
        "\nprompt {} tokens, generating {GENERATE}, greedy\n\
         sequential: {} tokens in {:.3}s ({:.2} tok/s), stop={:?}\n",
        prompt.len(),
        seq.new_tokens,
        seq_secs,
        seq.new_tokens as f64 / seq.decode_seconds,
        seq.reason
    );

    for block in BLOCKS {
        let mut events = Vec::new();
        let started = std::time::Instant::now();
        let spec = run_raw_completion_speculative(
            &mut runner,
            &tokenizer,
            &prompt,
            &config,
            4096,
            vocab,
            block,
            |e| events.push(e),
        )
        .expect("speculative run");
        let secs = started.elapsed().as_secs_f64();
        let spec_tokens = tokens_of(&events);

        println!(
            "block {block}: {} tokens in {secs:.3}s ({:.2} tok/s), stop={:?}, speedup {:.2}x",
            spec.new_tokens,
            spec.new_tokens as f64 / spec.decode_seconds,
            spec.reason,
            seq.decode_seconds / spec.decode_seconds
        );

        // THE GATE. Not a throughput claim -- the clock above is printed for
        // information and asserted on by nothing, because a gate that failed
        // on machine state would be worse than no gate (AGENTS.md Gotcha 43).
        assert_eq!(
            spec_tokens, seq_tokens,
            "block {block} produced a different token stream"
        );
        assert_eq!(
            text_of(&events),
            seq_text,
            "block {block} produced different text"
        );
        assert_eq!(spec.new_tokens, seq.new_tokens, "block {block}");
        assert_eq!(spec.reason, seq.reason, "block {block}");
        assert_eq!(
            spec.kv_backed_token_ids, seq.kv_backed_token_ids,
            "block {block} left a different cache behind"
        );
        assert_eq!(
            spec.kv_position,
            spec.kv_backed_token_ids.len(),
            "block {block} reported a cursor its own token list contradicts"
        );
    }

    println!("\nlossless at every block\n");
}

#[test]
#[ignore = "needs a real MTP install via TURBOSPARK_MTP_INSTALL_DIR"]
fn a_sampled_request_is_refused_on_the_real_install() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("TURBOSPARK_MTP_INSTALL_DIR").expect("TURBOSPARK_MTP_INSTALL_DIR"),
    );
    let (mut runner, tokenizer) = open_model_runner_speculative(
        &dir,
        SLOTS,
        runtime::DraftPolicies::mtp(runtime::MtpDraftPolicy::Fixed(MTP_DEPTH)),
    )
    .expect("install opens");
    let vocab = runner.vocab_size();

    let config = GenerationConfig {
        shaping: ShapingConfig::new(0.2, 64, Some(0.95), 1.0, Some(1)).unwrap(),
        max_new_tokens: 16,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: RateControl::default(),
    };
    let prompt = tokenizer.encode("hello", false);

    // The CLI's own default is sampled, so this is the path a user is most
    // likely to take. It must refuse rather than quietly decode greedily,
    // which would change what the model writes while reporting success.
    let err = run_raw_completion_speculative(
        &mut runner,
        &tokenizer,
        &prompt,
        &config,
        4096,
        vocab,
        2,
        |_| {},
    )
    .expect_err("a sampled speculative request must be refused");
    println!("refused as it must be: {err}");
}
