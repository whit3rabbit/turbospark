#![cfg(target_os = "macos")]
//! Does the WIRED speculative loop reproduce the sequential stream on the
//! real DFlash2 drafter, chain and tree? `mtp_generation_gate.rs`'s
//! question for the block drafter, owed since the flat verify made depth 7
//! the serving shape (`docs/KAPITSA_KERNELS.md`, levers 3 and 4).
//!
//! Run it once per tree setting; both switches are read once per process:
//!
//! ```sh
//! TURBOSPARK_DFLASH_INSTALL_DIR=~/models/qwen38-27b-dflash2.gturbo \
//!   TURBOSPARK_INT4_MMA8=1 TURBOSPARK_BATCHED_GEMV=0 \
//!   cargo test -p turbospark-bench --test dflash_generation_gate --release -- --ignored --nocapture
//! # and again with TURBOSPARK_DFLASH_TREE=0 (the chain)
//! ```
//!
//! `TURBOSPARK_BATCHED_GEMV=0` is required, not advice. The speculative
//! prefill takes the chunked driver (`Priming::Dflash`), whose default arm
//! runs Steel and is not bit-exact against the token walk the sequential
//! loop takes with a drafter open (`docs/DFLASH2.md`, section 6). Pinning
//! the per-token chunked arm makes the two prefills identical, so a
//! divergence below is the decode loop's. `TURBOSPARK_INT4_MMA8=1` runs the
//! tile in decode and verify alike (lossless by its exactness class); `=verify`
//! would part the streams by design.

use runtime::{
    run_raw_completion, run_raw_completion_speculative, DflashDraftPolicy, DraftPolicies,
    GenerationConfig, MtpDraftPolicy, RateControl, RawDecodeProgress,
};
use selection::ShapingConfig;
use tokenizer::{Message, Role};
use turbospark_bench::real_model::open_model_runner_speculative;

const SLOTS: usize = 16;
const GENERATE: u32 = 96;
/// The drafter's own depth; block 2 runs truncated rounds on it.
const DEPTH: usize = 7;
const BLOCKS: [usize; 2] = [2, 7];

fn tokens_of(events: &[RawDecodeProgress]) -> Vec<i32> {
    events
        .iter()
        .filter_map(|e| match e {
            RawDecodeProgress::Token { id, .. } => Some(*id),
            _ => None,
        })
        .collect()
}

#[test]
#[ignore = "needs a real DFlash2 install via TURBOSPARK_DFLASH_INSTALL_DIR"]
fn the_wired_dflash_loop_reproduces_the_sequential_stream() {
    gate(DflashDraftPolicy::Fixed(DEPTH), DEPTH, &BLOCKS);
}

/// What `auto` ships: the drafter opened at `dflash_serving_block()` (3 with
/// the 8-row tile off, 7 with it on), its rounds at that block, so the tree
/// is the one `auto` serves rather than a truncated round of a deeper one.
#[test]
#[ignore = "needs a real DFlash2 install via TURBOSPARK_DFLASH_INSTALL_DIR"]
fn the_auto_depth_reproduces_the_sequential_stream() {
    let depth = runtime::dflash_serving_block();
    gate(DflashDraftPolicy::Auto, depth, &[depth]);
}

fn gate(policy: DflashDraftPolicy, depth: usize, blocks: &[usize]) {
    assert_eq!(
        std::env::var("TURBOSPARK_BATCHED_GEMV").as_deref(),
        Ok("0"),
        "set TURBOSPARK_BATCHED_GEMV=0: the speculative prefill's default Steel arm is not \
         bit-exact against the sequential walk (this file's header)"
    );
    let tree = std::env::var("TURBOSPARK_DFLASH_TREE").as_deref() != Ok("0");
    let dir = std::path::PathBuf::from(
        std::env::var_os("TURBOSPARK_DFLASH_INSTALL_DIR").expect("TURBOSPARK_DFLASH_INSTALL_DIR"),
    );
    let (mut runner, tokenizer) = open_model_runner_speculative(
        &dir,
        SLOTS,
        DraftPolicies {
            mtp: MtpDraftPolicy::Off,
            dflash: policy,
        },
    )
    .expect("install opens");
    assert!(
        runner.dflash_speculation_blocker().is_none(),
        "this install cannot speculate with DFlash2"
    );
    if tree {
        assert_eq!(
            runner.dflash_tree_nodes(),
            Some(depth),
            "trees are on but this runner refuses them; the run would test chains"
        );
    }
    let vocab = runner.vocab_size();
    let config = GenerationConfig {
        shaping: ShapingConfig::new(0.0, 0, None, 1.0, None).unwrap(),
        max_new_tokens: GENERATE,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: RateControl::default(),
    };
    for text in [
        "Write a Python function that merges two sorted lists, then explain its time and \
         space complexity in detail.",
        "Explain why the sky is blue to a ten-year-old, in three short paragraphs.",
    ] {
        let rendered = tokenizer
            .apply_chat_template(&[Message::new(Role::User, text)])
            .expect("chat template renders");
        let prompt = tokenizer.encode(&rendered, false);

        let mut seq_events = Vec::new();
        let seq = run_raw_completion(
            &mut runner,
            &tokenizer,
            &prompt,
            &config,
            4096,
            vocab,
            |e| seq_events.push(e),
        )
        .expect("sequential run");
        let seq_tokens = tokens_of(&seq_events);
        println!(
            "\nprompt {} tokens, tree {tree}: sequential {} tokens, {:.2} tok/s",
            prompt.len(),
            seq.new_tokens,
            seq.new_tokens as f64 / seq.decode_seconds
        );
        for &block in blocks {
            let mut events = Vec::new();
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
            println!(
                "block {block}: {} tokens, {:.2} tok/s, {:.2}x",
                spec.new_tokens,
                spec.new_tokens as f64 / spec.decode_seconds,
                seq.decode_seconds / spec.decode_seconds
            );
            let spec_tokens = tokens_of(&events);
            let first = spec_tokens
                .iter()
                .zip(&seq_tokens)
                .position(|(a, b)| a != b);
            assert_eq!(
                spec_tokens, seq_tokens,
                "block {block}, tree {tree}: the streams part at token {first:?}"
            );
            assert_eq!(spec.reason, seq.reason, "block {block}");
            assert_eq!(
                spec.kv_backed_token_ids, seq.kv_backed_token_ids,
                "block {block} left a different cache behind"
            );
        }
    }
    println!("\nlossless at blocks {blocks:?}, tree {tree}\n");
}
