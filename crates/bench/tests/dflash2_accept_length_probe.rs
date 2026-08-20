//! DFlash2 acceptance and speedup against the non-speculative decode, on
//! the real install (`docs/DFLASH2.md`).
//!
//! The MTP probe's shape (`mtp_accept_length_probe.rs`), one drafter over:
//! per-position acceptance, rollbacks, and the wall-clock speedup each
//! block pays, with every arm asserted byte-identical to the plain greedy
//! stream. ONE structural difference: the DFlash2 drafter's block is fixed
//! at OPEN, so each block arm is a fresh runner (and the reference arm runs
//! on a drafter-off open of its own, dropped before the block arms so peak
//! memory is one runner at a time).
//!
//! ```sh
//! TURBOSPARK_DFLASH2_INSTALL_DIR=~/models/qwen38-27b-dflash2.gturbo \
//!   cargo test -p turbospark-bench --test dflash2_accept_length_probe --release -- --ignored --nocapture
//! ```

use foundation::LogitValue;
use runtime::{DflashDraftPolicy, DraftPolicies, LogitProducer, MtpDraftPolicy, RealForwardRunner};
use tokenizer::{Message, Role};
use turbospark_bench::real_model::open_model_runner_speculative;

/// The blocks worth a row, TRAINED WIDTH FIRST. The checkpoint's
/// `block_size: 8` bounds the drafter's ROW count, not its proposal count
/// (llama.cpp clamps at `min(tokens_per_block, dflash_block_size)` and emits
/// proposals for rows `1 ..= block_size - 1`; vLLM quotes its headline "at 7
/// draft tokens" and sizes its conv at `1 + num_speculative_tokens`), so 7
/// proposals plus the bonus row IS the trained shape and 8 is one row past
/// it. That extra row is off-distribution twice over: the drafter never saw
/// it, and the reference's conv masks tap 1 there (its tap mask is `position
/// % block_size >= tap`, which wraps at row 8) where this port applies it.
const BLOCKS: [usize; 4] = [7, 8, 4, 2];
const GENERATE: usize = 256;
const SLOTS: usize = 16;

fn argmax(logits: &[LogitValue]) -> i32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, v) in logits.iter().enumerate() {
        let f = v.to_f32();
        if f > best_v {
            best_v = f;
            best = i;
        }
    }
    best as i32
}

#[derive(Default)]
struct Stats {
    rounds: usize,
    accepted: usize,
    offered: Vec<usize>,
    matched: Vec<usize>,
    rollbacks: usize,
}

/// One block arm: prefill priming the drafter, then rounds of
/// draft/verify/accept/rollback/replay/rewind, exactly the loop's shape.
/// Returns the stats, the generated stream, and the decode wall clock.
fn run_dflash(
    runner: &mut RealForwardRunner,
    prompt: &[i32],
    vocab: usize,
    block: usize,
) -> (Stats, Vec<i32>, f64) {
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    let mut batch_logits = vec![LogitValue::from_f32(0.0); (block + 1) * vocab];
    runner.reset();
    for (i, &token) in prompt.iter().enumerate() {
        runner.produce(token, i, &mut logits).expect("produce");
        if i + 1 < prompt.len() {
            runner
                .dflash_prime_from_capture(i)
                .expect("prime the drafter over the prompt");
        }
    }
    let started = std::time::Instant::now();

    let mut history: Vec<i32> = prompt.to_vec();
    let mut next = argmax(&logits);
    let mut stats = Stats {
        offered: vec![0; block],
        matched: vec![0; block],
        ..Default::default()
    };
    let mut generated: Vec<i32> = Vec::new();

    while generated.len() < GENERATE {
        history.push(next);
        generated.push(next);
        stats.rounds += 1;
        let base = history.len() - 1;

        let mut proposals = Vec::with_capacity(block);
        runner
            .dflash_draft_block(next, base, &mut proposals)
            .expect("the draft block runs");
        if stats.rounds <= 3 {
            eprintln!(
                "[round {}] base {base} anchor {next} proposals {:?}",
                stats.rounds, proposals
            );
        }

        let feed: Vec<i32> = std::iter::once(next)
            .chain(proposals.iter().copied())
            .collect();
        let point = runner.checkpoint();
        runner
            .produce_batched(&feed, base, &mut batch_logits)
            .expect("batched verify");
        let mut accepted = 0usize;
        for (i, &proposal) in proposals.iter().enumerate() {
            stats.offered[i] += 1;
            let row = &batch_logits[i * vocab..(i + 1) * vocab];
            if stats.rounds <= 3 {
                eprintln!(
                    "[round {}] row {i} argmax {} vs proposal {proposal}",
                    stats.rounds,
                    argmax(row)
                );
            }
            if argmax(row) != proposal {
                break;
            }
            stats.matched[i] += 1;
            accepted += 1;
        }
        let bonus = argmax(&batch_logits[accepted * vocab..(accepted + 1) * vocab]);
        if accepted < proposals.len() {
            stats.rollbacks += 1;
            runner.rollback(&point);
            let keep = accepted + 1;
            runner
                .produce_batched(&feed[..keep], base, &mut batch_logits[..keep * vocab])
                .expect("replay the accepted prefix");
        }
        stats.accepted += accepted;
        runner
            .dflash_rewind_to(base + accepted)
            .expect("rewind the drafter to the accepted end");
        for &t in proposals.iter().take(accepted) {
            history.push(t);
            generated.push(t);
        }
        next = bonus;
    }
    (stats, generated, started.elapsed().as_secs_f64())
}

/// The non-speculative reference: plain greedy, no drafter state.
fn run_plain(runner: &mut RealForwardRunner, prompt: &[i32], vocab: usize) -> (Vec<i32>, f64) {
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    runner.reset();
    for (i, &token) in prompt.iter().enumerate() {
        runner.produce(token, i, &mut logits).expect("produce");
    }
    let started = std::time::Instant::now();
    let mut generated = Vec::new();
    let mut next = argmax(&logits);
    while generated.len() < GENERATE {
        generated.push(next);
        runner
            .produce(next, prompt.len() + generated.len() - 1, &mut logits)
            .expect("produce");
        next = argmax(&logits);
    }
    (generated, started.elapsed().as_secs_f64())
}

#[test]
#[ignore = "needs a real DFlash2 install via TURBOSPARK_DFLASH2_INSTALL_DIR"]
fn dflash2_accept_length_and_speedup() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("TURBOSPARK_DFLASH2_INSTALL_DIR").expect("TURBOSPARK_DFLASH2_INSTALL_DIR"),
    );

    // Reference first, on its own open, dropped before the block arms so
    // the peak is one runner at a time. Run TWICE and take the second
    // clock: the first decode of a process is a cold GPU (Gotcha 20).
    let (plain, plain_secs) = {
        let (mut runner, tokenizer) = open_model_runner_speculative(
            &dir,
            SLOTS,
            DraftPolicies {
                mtp: MtpDraftPolicy::Off,
                dflash: DflashDraftPolicy::Off,
            },
        )
        .expect("install opens with every drafter off");
        let vocab = runner.vocab_size();
        let prompt = protocol_prompt(&tokenizer);
        let _ = run_plain(&mut runner, &prompt, vocab);
        let out = run_plain(&mut runner, &prompt, vocab);
        println!(
            "\nprompt {} tokens, generating {GENERATE}, greedy",
            prompt.len()
        );
        println!(
            "reference (no drafter): {:.2} s, {:.2} tok/s\n",
            out.1,
            GENERATE as f64 / out.1
        );
        (out.0, out.1)
    };

    println!(" block  rounds  accepted/rd  committed/rd  rollbacks  seconds  MEASURED");
    for block in BLOCKS {
        let (mut runner, tokenizer) = open_model_runner_speculative(
            &dir,
            SLOTS,
            DraftPolicies {
                mtp: MtpDraftPolicy::Off,
                dflash: DflashDraftPolicy::Fixed(block),
            },
        )
        .expect("install opens with the drafter at the block");
        let vocab = runner.vocab_size();
        let prompt = protocol_prompt(&tokenizer);
        let (s, generated, secs) = run_dflash(&mut runner, &prompt, vocab, block);
        drop(runner);

        let per_round = s.accepted as f64 / s.rounds.max(1) as f64;
        let committed = per_round + 1.0;
        let measured = plain_secs / secs;
        println!(
            "{block:>6}  {:>6}  {:>11.2}  {:>12.2}  {:>9}  {:>7.2}  {:>7.2}x",
            s.rounds, per_round, committed, s.rollbacks, secs, measured,
        );
        let curve: Vec<f64> = s
            .offered
            .iter()
            .zip(&s.matched)
            .map(|(o, m)| if *o == 0 { 0.0 } else { *m as f64 / *o as f64 })
            .collect();
        println!(
            "        per-position acceptance: {}",
            curve
                .iter()
                .map(|p| format!("{p:.2}"))
                .collect::<Vec<_>>()
                .join(" ")
        );

        // Losslessness, on the common prefix (a round overshoots GENERATE
        // by construction; the reference lands on it exactly).
        let n = plain.len().min(generated.len());
        assert!(
            n >= GENERATE.min(plain.len()),
            "block {block} produced only {n} comparable tokens"
        );
        assert_eq!(
            plain[..n],
            generated[..n],
            "block {block} diverged from the non-speculative greedy stream: \
             speculation is not lossless"
        );

        // The functional-drafter guard, the MTP probe's own: near-zero
        // first-proposal acceptance is a broken drafter (a convention read
        // wrongly somewhere in the port), not a verdict about DFlash2.
        let first = s.matched[0] as f64 / s.offered[0].max(1) as f64;
        assert!(
            first > 0.02,
            "block {block}: the drafter's FIRST proposal was accepted {}/{} times \
             ({first:.4}). That is a broken port, not a low accept length -- the \
             published drafter's first-position acceptance is ~0.9 (docs/DFLASH2.md)",
            s.matched[0],
            s.offered[0]
        );
    }
}

fn protocol_prompt(tokenizer: &tokenizer::MfTokenizer) -> Vec<i32> {
    let rendered = tokenizer
        .apply_chat_template(&[Message::new(
            Role::User,
            "Write a Python function that merges two sorted lists, then explain \
             its time and space complexity in detail.",
        )])
        .expect("chat template renders");
    tokenizer.encode(&rendered, false)
}
