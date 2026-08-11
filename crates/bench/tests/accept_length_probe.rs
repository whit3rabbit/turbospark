#![cfg(target_os = "macos")]
//! How many proposed tokens does the target actually accept? (ROADMAP's
//! speculative-decoding item, the last unknown in its cost model.)
//!
//! Everything else in that model is now measured: `c(M)` from
//! `gemv_bandwidth_bench.rs`, the expert-union multiplier from
//! `router_window.py`, and the per-family compute split from
//! `MFERENCE_DISPATCH_PROFILE=1`. They say a batched verify of 8 tokens
//! costs roughly 4.4 decode steps, so speculation pays only if more than
//! ~4.4 of every 8 proposals survive. Accept length was the one term still
//! being guessed at.
//!
//! It needs NO batched kernels, which is why it comes first: only the
//! RATIO matters, so the verify pass here runs the proposals through
//! `produce` one at a time. Slow, and irrelevant to the answer.
//!
//! The drafter is n-gram / prompt-lookup: find the most recent earlier
//! occurrence of the last few tokens and propose whatever followed it.
//! Zero weights, zero download. Its accept length is a LOWER BOUND on what
//! a trained DFlash drafter would reach, so a number that already clears
//! break-even settles the question, while a number below it does not
//! condemn DFlash -- it just means the cheap drafter is not the one to use.
//!
//! This also exercises the accept walk and `RealForwardRunner::rollback`
//! end to end, which is the rest of the speculative loop minus the speed.
//!
//! ```sh
//! TURBOSPARK_PROBE_INSTALL_DIR=~/models/qwen36.gturbo \
//!   cargo test -p turbospark-bench --test accept_length_probe --release -- --ignored --nocapture
//! ```

use foundation::LogitValue;
use runtime::{LogitProducer, RealForwardRunner};
use tokenizer::{Message, Role};
use turbospark_bench::real_model::open_model_runner;

/// Proposals per speculative round. 8 is where the cost model is least
/// unfavourable; 4 and 16 are reported beside it because break-even moves
/// with M.
const BLOCKS: [usize; 3] = [4, 8, 16];
/// Tokens to generate per arm. Long enough for the n-gram table to have
/// something to match against.
const GENERATE: usize = 300;
/// Context length matched when looking for an earlier occurrence. Longer
/// is a more confident match and fires less often.
const NGRAM: usize = 3;

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

/// Most recent earlier occurrence of the last [`NGRAM`] tokens; returns
/// what followed it, up to `want` tokens.
fn propose(history: &[i32], want: usize) -> Vec<i32> {
    if history.len() < NGRAM + 1 {
        return Vec::new();
    }
    let tail = &history[history.len() - NGRAM..];
    // Skip the tail's own position, then walk backwards for the newest
    // match: recent context predicts better than ancient context.
    let last = history.len() - NGRAM;
    for start in (0..last).rev() {
        if &history[start..start + NGRAM] == tail {
            let from = start + NGRAM;
            let take = want.min(history.len().saturating_sub(from));
            if take > 0 {
                return history[from..from + take].to_vec();
            }
        }
    }
    Vec::new()
}

struct Stats {
    rounds: usize,
    proposed_rounds: usize,
    proposed: usize,
    accepted: usize,
}

fn run_block(
    runner: &mut RealForwardRunner,
    prompt: &[i32],
    vocab: usize,
    block: usize,
) -> (Stats, Vec<i32>) {
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    runner.reset();

    let mut history: Vec<i32> = prompt.to_vec();
    for (i, &token) in prompt.iter().enumerate() {
        runner.produce(token, i, &mut logits).expect("produce");
    }
    // The token the target would emit next with no speculation at all.
    let mut next = argmax(&logits);

    let mut stats = Stats {
        rounds: 0,
        proposed_rounds: 0,
        proposed: 0,
        accepted: 0,
    };
    let mut generated: Vec<i32> = Vec::new();

    while generated.len() < GENERATE {
        history.push(next);
        generated.push(next);
        stats.rounds += 1;

        // `next` is committed but not yet fed to the model; feeding it is
        // the first step of the verify block either way.
        let draft = if block == 0 {
            Vec::new()
        } else {
            propose(&history, block)
        };
        if draft.is_empty() {
            let pos = history.len() - 1;
            runner.produce(next, pos, &mut logits).expect("produce");
            next = argmax(&logits);
            continue;
        }
        stats.proposed_rounds += 1;
        stats.proposed += draft.len();

        let point = runner.checkpoint();
        let base = history.len() - 1;
        // Feed [next, draft...] and read each position's argmax. Position
        // p's logits predict p+1, so `next`'s logits are checked against
        // draft[0], and so on.
        let mut accepted = 0usize;
        let mut bonus = None;
        let mut token = next;
        for (i, &proposal) in draft.iter().enumerate() {
            runner
                .produce(token, base + i, &mut logits)
                .expect("produce");
            let target = argmax(&logits);
            if target != proposal {
                bonus = Some(target);
                break;
            }
            accepted += 1;
            token = proposal;
        }
        if bonus.is_none() {
            // Every proposal matched; the last position still gives a free
            // token, which is the block's bonus.
            runner
                .produce(token, base + draft.len(), &mut logits)
                .expect("produce");
            bonus = Some(argmax(&logits));
        }
        stats.accepted += accepted;

        // Commit the accepted prefix. Rolling back and replaying is what a
        // real loop does; here it also proves the rollback path is exercised
        // by something other than its own probe.
        runner.rollback(&point);
        let keep: Vec<i32> = std::iter::once(next)
            .chain(draft.iter().copied().take(accepted))
            .collect();
        for (i, &t) in keep.iter().enumerate() {
            runner.produce(t, base + i, &mut logits).expect("produce");
        }
        for &t in draft.iter().take(accepted) {
            history.push(t);
            generated.push(t);
        }
        next = bonus.expect("bonus token");
    }
    (stats, generated)
}

#[test]
#[ignore = "needs a real install via TURBOSPARK_PROBE_INSTALL_DIR"]
fn ngram_accept_length_against_the_break_even_it_has_to_clear() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("TURBOSPARK_PROBE_INSTALL_DIR").expect("TURBOSPARK_PROBE_INSTALL_DIR"),
    );
    let (mut runner, tokenizer) = open_model_runner(&dir, 16).expect("install opens");
    let vocab = tokenizer.vocab_size;
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

    // Break-even accepted-token counts, from the cost model in
    // docs/BENCHMARKS.md: batched verify cost in decode-steps, which the
    // accepted count has to exceed for speculation to pay.
    let break_even = |m: usize| match m {
        4 => 2.6,
        8 => 4.4,
        _ => 7.5,
    };

    // Ground truth: the same greedy generation with speculation switched
    // off entirely (block 1 never proposes). Comparing the speculative
    // arms against EACH OTHER would pass even if all of them were wrong
    // the same way.
    let (_, plain) = run_block(&mut runner, &prompt, vocab, 0);

    println!("\n block   rounds  fired   proposed/round  accepted/round   break-even  verdict");
    let mut reference: Option<Vec<i32>> = Some(plain);
    for block in BLOCKS {
        let (s, generated) = run_block(&mut runner, &prompt, vocab, block);
        let per_round = s.accepted as f64 / s.proposed_rounds.max(1) as f64;
        // Tokens gained per verify pass: the accepted prefix plus the
        // always-free bonus token.
        let gained = per_round + 1.0;
        let target = break_even(block);
        println!(
            "{block:>6}   {:>6}  {:>4.0}%   {:>13.2}   {:>13.2}   {:>10.1}  {}",
            s.rounds,
            100.0 * s.proposed_rounds as f64 / s.rounds as f64,
            s.proposed as f64 / s.proposed_rounds.max(1) as f64,
            gained,
            target,
            if gained > target { "PAYS" } else { "loses" }
        );

        // Speculation must not change the output. Same prompt, same greedy
        // settings, different block size: the token stream has to match.
        assert_eq!(
            reference.as_ref().expect("non-speculative reference"),
            &generated,
            "block {block} diverged from the non-speculative greedy stream: \
             speculation is not lossless"
        );
    }
    println!(
        "\n'accepted/round' counts the accepted prefix PLUS the bonus token every\n\
         verify yields for free. n-gram drafting is a LOWER bound on a trained\n\
         drafter; a block that already pays here settles the question, one that\n\
         does not says only that this drafter is too weak.\n"
    );
}
