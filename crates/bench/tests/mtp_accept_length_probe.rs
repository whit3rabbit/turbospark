#![cfg(target_os = "macos")]
//! How many of the MTP head's proposals does the trunk accept?
//! (`docs/MTP_SPECULATIVE.md`, step 3 -- the last unknown in its cost model.)
//!
//! Everything else in that model is measured: `c(M)` from
//! `gemv_bandwidth_bench.rs`, the dense compute split from
//! `TURBOSPARK_DISPATCH_PROFILE=1`, and the head's own cost from its tensor
//! total. They say a round of block M costs `(M+1) x c(M+1) + M x 0.015`
//! decode-steps, so speculation pays only if the round yields more committed
//! tokens than that. [`BREAK_EVEN`] is that column, copied from the page.
//!
//! It needs NO batched kernels, which is why it comes before them: only the
//! RATIO matters, so the verify pass here runs the proposals through `produce`
//! one at a time. Slow, and irrelevant to the answer.
//!
//! # What separates this from `accept_length_probe.rs`
//!
//! That one drafts with n-gram / prompt-lookup, whose accept length is a LOWER
//! bound on a trained drafter and which fires on only ~10% of rounds. This one
//! drafts with the checkpoint's OWN multi-token-prediction head, so it fires
//! on every round and its number is the real one for this target rather than a
//! bound. Both keep the same two disciplines, and the second is the load-
//! bearing one: the arms are gated against a NON-SPECULATIVE reference stream,
//! because comparing speculative arms against each other passes even when all
//! of them are wrong the same way.
//!
//! # The head's KV is the part that is easy to get silently wrong
//!
//! `encode_full_attention_block` takes its attention span from the `position`
//! ARGUMENT (`position + 1`), never from the head's cursor. So a draft taken
//! at decode position `P` off an unprimed head attends over `P` rows nobody
//! wrote -- no error, finite logits, plausible tokens, and a depressed accept
//! length that would read as a verdict about MTP. Two things prevent it here:
//! the head is PRIMED across the prompt, and `mtp_draft_step` refuses a step
//! off its own cursor. See `crates/runtime/src/families/qwen/mtp.rs`.
//!
//! ```sh
//! TURBOSPARK_MTP_INSTALL_DIR=~/models/qwen38-27b-mtp.gturbo \
//!   cargo test -p turbospark-bench --test mtp_accept_length_probe --release -- --ignored --nocapture
//! ```

use foundation::LogitValue;
use runtime::{LogitProducer, RealForwardRunner};
use tokenizer::{Message, Role};
use turbospark_bench::real_model::open_model_runner_speculative;

/// Draft depth this file asks for. Named here rather than set through
/// `TURBOSPARK_MTP_DRAFT` because the policy is now a PARAMETER: an unset
/// env var means `Auto`, which resolves to a depth too small for the
/// blocks below and would fail deep in the verify rather than at open.
const MTP_DEPTH: usize = 16;

/// Proposals per speculative round. These are the rows of the block table in
/// `docs/MTP_SPECULATIVE.md`; 15 is the largest legal one, because a batched
/// verify of `M + 1` positions has to fit `MAX_BATCH_ROWS = 16`.
const BLOCKS: [usize; 4] = [2, 4, 8, 15];
/// Tokens to generate per arm. Every arm generates the same stream, so this
/// trades runtime against the stability of the per-position curve.
const GENERATE: usize = 256;
/// Expert-cache slots. Pinned rather than defaulted into, for
/// `crates/bench/CLAUDE.md` Gotcha 5's reason -- though a dense install has no
/// routed experts, so this is inert here and passed for uniformity.
const SLOTS: usize = 16;

/// Verify cost in decode-steps, from `docs/MTP_SPECULATIVE.md`'s composed
/// table. A round has to COMMIT more tokens than this to pay.
fn break_even(block: usize) -> f64 {
    match block {
        2 => 1.71,
        4 => 2.86,
        8 => 4.53,
        _ => 7.96,
    }
}

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
    proposed: usize,
    accepted: usize,
    /// `offered[d]` counts rounds that reached proposal `d`; `matched[d]`
    /// counts those where it was accepted. The RATIO is the per-position
    /// acceptance curve, which is what decides the block size -- a mean alone
    /// cannot say whether a longer block would have paid.
    offered: Vec<usize>,
    matched: Vec<usize>,
    /// Rounds that had to rewind the trunk. Batched-verify only, and it is
    /// the cost the composed break-even column never modelled: a rejected
    /// batched pass has absorbed positions it must give back, and the gated-
    /// DeltaNet state can only be restored wholesale.
    rollbacks: usize,
}

/// Walks the prompt through the trunk, priming the head as it goes.
///
/// The head's input pair at `position` is `(h_position, token_at_position+1)`,
/// and during a prompt both are known: `produce` has just left `h` in
/// `scratch.x` and the next token is the next prompt token. So priming is
/// exactly the decode-time call with the guesswork removed.
fn prefill(runner: &mut RealForwardRunner, prompt: &[i32], logits: &mut [LogitValue], mtp: bool) {
    for (i, &token) in prompt.iter().enumerate() {
        runner.produce(token, i, logits).expect("produce");
        if mtp && i + 1 < prompt.len() {
            runner
                .mtp_prime_step(prompt[i + 1], i)
                .expect("prime the head over the prompt");
        }
    }
}

/// How a round's proposals are checked against the trunk.
///
/// The two arms must produce an IDENTICAL token stream -- this is an A/B
/// seam, not a feature flag -- so the losslessness gate covers both.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Verify {
    /// One `produce` per position, stopping at the first rejection. Needs no
    /// batched kernel, which is why step 3 could measure the accept length
    /// before step 4 existed.
    Sequential,
    /// One `produce_batched` over the confirmed token plus every proposal
    /// (`docs/MTP_SPECULATIVE.md` step 4). This is what the break-even
    /// column was always a projection OF.
    Batched,
}

/// `block == 0` is the non-speculative reference: plain greedy, no head.
///
/// Returns the stats, the generated stream, and the DECODE wall clock. The
/// clock excludes prefill and the reset, so the arms differ only in how a
/// round is verified.
fn run_block(
    runner: &mut RealForwardRunner,
    prompt: &[i32],
    vocab: usize,
    block: usize,
    verify: Verify,
) -> (Stats, Vec<i32>, f64) {
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    let mut draft_logits = vec![LogitValue::from_f32(0.0); vocab];
    // `block + 1` rows: the confirmed token plus every proposal.
    let mut batch_logits = vec![LogitValue::from_f32(0.0); (block + 1).max(1) * vocab];
    runner.reset();
    prefill(runner, prompt, &mut logits, block > 0);
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
        // `next` occupies `base`, and the head is at `base - 1`: its rows
        // cover every position the trunk has a hidden state for.
        let base = history.len() - 1;

        if block == 0 {
            runner.produce(next, base, &mut logits).expect("produce");
            next = argmax(&logits);
            continue;
        }

        // -- Draft. `block + 1` steps for `block` proposals: the last one is
        //    taken for its KV ROW alone, because a round where every proposal
        //    is accepted needs a head row at `base + block` that the
        //    proposal-producing steps do not write. Without it the fully-
        //    accepted case is the one that desyncs, i.e. the case a good
        //    drafter hits most often.
        let mut proposals: Vec<i32> = Vec::with_capacity(block);
        let mut chained = next;
        for d in 0..=block {
            runner
                .mtp_draft_step(chained, base - 1 + d, &mut draft_logits)
                .expect("draft step");
            chained = argmax(&draft_logits);
            if d < block {
                proposals.push(chained);
            }
        }
        stats.proposed += proposals.len();

        // -- Verify. Position p's logits predict p + 1, so `next`'s logits
        //    are checked against proposals[0] and the row after the last
        //    accepted proposal carries the free bonus token.
        //
        //    NEITHER ARM ROLLS BACK ON THE COMMON PATH, and working out why
        //    is what makes the wall-clock comparison mean anything. A round
        //    commits `next` plus its accepted proposals, and BOTH arms leave
        //    the engine having absorbed exactly those: the sequential loop
        //    stops at the first rejection, and a fully-accepted batched pass
        //    fed exactly the committed tokens. Only a batched pass with a
        //    REJECTION has absorbed positions that are about to be dropped,
        //    and only that case pays a rollback. The trunk's `rollback` is
        //    the expensive half -- `checkpoint` copies the whole gated-
        //    DeltaNet state, because a recurrent layer cannot be rewound
        //    incrementally the way a KV cursor can.
        let mut accepted = 0usize;
        let bonus;
        match verify {
            Verify::Sequential => {
                let mut stop = None;
                let mut token = next;
                for (i, &proposal) in proposals.iter().enumerate() {
                    runner
                        .produce(token, base + i, &mut logits)
                        .expect("produce");
                    let target = argmax(&logits);
                    stats.offered[i] += 1;
                    if target != proposal {
                        stop = Some(target);
                        break;
                    }
                    stats.matched[i] += 1;
                    accepted += 1;
                    token = proposal;
                }
                bonus = match stop {
                    Some(target) => target,
                    None => {
                        // Every proposal matched; the last position still
                        // yields a free token, which is the block's bonus.
                        runner
                            .produce(token, base + proposals.len(), &mut logits)
                            .expect("produce");
                        argmax(&logits)
                    }
                };
            }
            Verify::Batched => {
                let feed: Vec<i32> = std::iter::once(next)
                    .chain(proposals.iter().copied())
                    .collect();
                let point = runner.checkpoint();
                runner
                    .produce_batched(&feed, base, &mut batch_logits[..feed.len() * vocab])
                    .expect("batched verify");
                for (i, &proposal) in proposals.iter().enumerate() {
                    let row = &batch_logits[i * vocab..(i + 1) * vocab];
                    stats.offered[i] += 1;
                    if argmax(row) != proposal {
                        break;
                    }
                    stats.matched[i] += 1;
                    accepted += 1;
                }
                bonus = argmax(&batch_logits[accepted * vocab..(accepted + 1) * vocab]);
                if accepted < proposals.len() {
                    stats.rollbacks += 1;
                    runner.rollback(&point);
                    let keep = accepted + 1;
                    runner
                        .produce_batched(&feed[..keep], base, &mut batch_logits[..keep * vocab])
                        .expect("replay the accepted prefix");
                }
            }
        }
        stats.accepted += accepted;

        // -- The head rewinds to where the block ENDED and continues, where
        //    the trunk (when it rewinds at all) goes back to where the block
        //    STARTED. Two different targets, which is why one call cannot do
        //    both (`mtp_rewind_to`'s doc comment).
        runner
            .mtp_rewind_to(base + accepted)
            .expect("rewind the head to the accepted end");
        for &t in proposals.iter().take(accepted) {
            history.push(t);
            generated.push(t);
        }
        next = bonus;
    }
    (stats, generated, started.elapsed().as_secs_f64())
}

#[test]
#[ignore = "needs a real MTP install via TURBOSPARK_MTP_INSTALL_DIR"]
fn mtp_accept_length_against_the_break_even_it_has_to_clear() {
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
        "this install carries no MTP head; the probe would measure nothing"
    );
    // The MODEL's width, never the tokenizer's (AGENTS.md Gotcha 37).
    let vocab = runner.vocab_size();

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

    // Ground truth FIRST: the same greedy generation with the head switched
    // off entirely. Every arm below is asserted against this stream.
    //
    // Run TWICE, and the second reading is the reference clock. The first
    // decode of the process is on a cold GPU at low DVFS clocks, which is
    // worth up to 53% (AGENTS.md Gotcha 20) and would land entirely on the
    // denominator of every speedup below.
    let (_, _, _) = run_block(&mut runner, &prompt, vocab, 0, Verify::Sequential);
    let (_, plain, plain_secs) = run_block(&mut runner, &prompt, vocab, 0, Verify::Sequential);

    println!(
        "\nprompt {} tokens, generating {GENERATE}, greedy\n",
        prompt.len()
    );
    println!(
        "reference (no head): {plain_secs:.2} s for {} tokens, {:.2} tok/s\n",
        plain.len(),
        plain.len() as f64 / plain_secs
    );
    println!(
        " block  verify      rounds  accepted/rd  committed/rd  rollbacks  seconds  MEASURED  projected"
    );
    let mut curves: Vec<(usize, Vec<f64>)> = Vec::new();
    for block in BLOCKS {
        for verify in [Verify::Sequential, Verify::Batched] {
            let (s, generated, secs) = run_block(&mut runner, &prompt, vocab, block, verify);
            let per_round = s.accepted as f64 / s.rounds.max(1) as f64;
            // Committed per round: the accepted prefix plus the bonus token every
            // verify yields for free.
            let committed = per_round + 1.0;
            let target = break_even(block);
            // THE MEASURED SPEEDUP IS WALL CLOCK, and it is what step 4 exists to
            // produce. `projected` is the step-3 column: committed tokens against
            // a COMPOSED verify cost. The two answer different questions, and
            // where they disagree the measurement wins -- the projection has no
            // term for the rollback a rejected batched round pays, and none for
            // the per-round cost of snapshotting the recurrent state.
            let measured = plain_secs / secs;
            let projected = committed / target;
            println!(
                "{block:>6}  {:<10}  {:>6}  {:>11.2}  {:>12.2}  {:>9}  {:>7.2}  {:>7.2}x  {:>8.2}x",
                format!("{verify:?}"),
                s.rounds,
                per_round,
                committed,
                s.rollbacks,
                secs,
                measured,
                projected,
            );
            // Once per block, not once per arm: the two verify strategies are
            // lossless against each other, so their curves are identical by
            // construction and the assertion below is what proves it.
            if verify == Verify::Batched {
                curves.push((
                    block,
                    s.offered
                        .iter()
                        .zip(&s.matched)
                        .map(|(o, m)| if *o == 0 { 0.0 } else { *m as f64 / *o as f64 })
                        .collect(),
                ));
            }

            // Speculation must not change the output. Same prompt, same greedy
            // settings, different block size: the token stream has to match.
            //
            // On the COMMON PREFIX, because the arms stop at different lengths by
            // construction: a speculative round commits `accepted + 1` tokens, so
            // it overshoots `GENERATE`, while the reference commits exactly one
            // per round and lands on it exactly. Comparing full vectors fails on
            // a tail the reference never generated, which is a property of the
            // loop bound and not a divergence. (This only became visible once the
            // drafter started working: at zero acceptance both arms commit one
            // token per round and the lengths matched by accident.)
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

            // THE DRAFTER HAS TO BE FUNCTIONAL BEFORE ITS ACCEPT LENGTH MEANS
            // ANYTHING. A weak drafter still lands common tokens, so a rate at or
            // near zero is a broken head rather than a verdict about MTP -- and
            // reported as a verdict it would close the question this page exists
            // to keep open, which is precisely the mistake recorded at the top of
            // docs/MTP_SPECULATIVE.md. The bar is deliberately far below any
            // interesting threshold: it separates "drafting" from "not drafting",
            // not "pays" from "loses", and the table above is printed either way.
            let first = s.matched[0] as f64 / s.offered[0].max(1) as f64;
            assert!(
                first > 0.02,
                "block {block}: the head's FIRST proposal was accepted {}/{} times \
             ({first:.4}). That is a broken drafter, not a low accept length -- \
             run `mtp_head_probe` (it reports the rank of the true token in the \
             head's own distribution) before reading any row above as a result.",
                s.matched[0],
                s.offered[0]
            );
        }
    }

    println!("\nper-position acceptance (share of rounds reaching position d that accept it):");
    for (block, curve) in &curves {
        let shown: Vec<String> = curve.iter().take(8).map(|p| format!("{p:.2}")).collect();
        println!("  block {block:>2}: {}", shown.join(" "));
    }
    println!(
        "\n'committed/rd' is the accepted prefix PLUS the bonus token. 'break-even' is\n\
         the verify cost in decode-steps from docs/MTP_SPECULATIVE.md's composed table,\n\
         so speedup is the ratio of the two. Unlike accept_length_probe.rs's n-gram\n\
         drafter this is the checkpoint's own head, so these are the target's real\n\
         numbers rather than a lower bound.\n"
    );
}
