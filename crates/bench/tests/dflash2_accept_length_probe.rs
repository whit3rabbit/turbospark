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
use turbospark_bench::protocol::PROTOCOL_CASES;
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
/// Long enough to REACH the divergence rather than stop short of it.
///
/// This was 256, and 256 is why byte-identity to a sequential decode looked
/// like a property of this engine for a day: on the protocol's prose case
/// the two streams part at 154 tokens, so a 256-token assertion had ~100
/// tokens of margin over a divergence it was too short to reach on the
/// prompt that has one -- and the ad-hoc CLI check that corroborated it ran
/// 200 and stopped one token short of the same thing. A losslessness
/// assertion is only as long as its generation AND only as searching as its
/// prompt; this file now varies both.
const GENERATE: usize = 600;

/// How many leading tokens a speculative arm must match the sequential
/// stream for.
///
/// NOT a bit-identity claim, which is measured FALSE (`docs/DFLASH2.md`): a
/// batched verify runs `dequant_int4_gemm_simd` where a decode step runs
/// `dequant_int4_gemv_simd`, the two accumulate differently, and the streams
/// part at the first near-tie. What this floor catches is the failure that
/// matters -- a drafter or a rollback that corrupts state, which diverges in
/// the first handful of tokens rather than the second hundred.
///
/// **64 AND NOT THE OBSERVED 154, because the divergence point is DATA
/// DEPENDENT and has no principled lower bound.** It is wherever the first
/// near-tie happens to fall, so a different prompt, a re-quantized
/// checkpoint or a kernel change could legitimately move it earlier without
/// anything being wrong. A floor set just under the one measurement in hand
/// is a test that fails on data rather than on a defect -- the same reasoning
/// the memory oracle's tok/s floor uses (set from the SLOWEST reading with
/// margin, `crates/bench/CLAUDE.md` Gotcha 15), and the opposite of what a
/// tight bound would buy here.
const COMMON_PREFIX_FLOOR: usize = 64;
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
    let (plain, plain_secs, plain_prose) = {
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
        // The prose reference rides the SAME open, which is why the second
        // prompt costs a decode and not a model load.
        let prose = run_plain(&mut runner, &prose_prompt(&tokenizer), vocab).0;
        println!(
            "\nprompt {} tokens, generating {GENERATE}, greedy",
            prompt.len()
        );
        println!(
            "reference (no drafter): {:.2} s, {:.2} tok/s\n",
            out.1,
            GENERATE as f64 / out.1
        );
        (out.0, out.1, prose)
    };

    println!(" block  rounds  accepted/rd  committed/rd  rollbacks  seconds  MEASURED");
    let mut streams: Vec<(usize, Vec<i32>)> = Vec::new();
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
        // THE PROSE ARM, on the serving block and on the runner already
        // open. This is the losslessness check with teeth: the sweep's own
        // prompt tracks the sequential stream for all 600 tokens whatever
        // the engine does, so a floor asserted there is a decoration. This
        // prompt parts from it at ~200 (`docs/DFLASH2.md`), which means the
        // floor is measuring something.
        if block == runtime::DFLASH_SERVING_BLOCK {
            let prose = prose_prompt(&tokenizer);
            let (_, prose_gen, _) = run_dflash(&mut runner, &prose, vocab, block);
            let n = plain_prose.len().min(prose_gen.len());
            let prefix = (0..n)
                .take_while(|&i| plain_prose[i] == prose_gen[i])
                .count();
            println!(
                "\nprose case (protocol short-explanation), block {block}: matches the \
                 sequential stream for {prefix} of {n} tokens"
            );
            assert!(
                prefix >= COMMON_PREFIX_FLOOR.min(n),
                "the prose arm left the sequential greedy stream after only {prefix} \
                 tokens, under the {COMMON_PREFIX_FLOOR} floor. Divergence in the \
                 low hundreds is expected (the batched verify and the decode GEMV \
                 accumulate differently); divergence this early is corrupted state"
            );
        }
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

        // How far this arm tracks the sequential stream, REPORTED rather
        // than assumed. A round overshoots GENERATE by construction, so the
        // comparison is over the common length.
        let n = plain.len().min(generated.len());
        let prefix = (0..n).take_while(|&i| plain[i] == generated[i]).count();
        if prefix == n {
            println!("        matches the sequential stream for all {n} compared tokens");
        } else {
            println!(
                "        matches the sequential stream for {prefix} tokens, then                  diverges (expected past ~200; see docs/DFLASH2.md)"
            );
        }
        assert!(
            prefix >= COMMON_PREFIX_FLOOR.min(n),
            "block {block} left the sequential greedy stream after only {prefix} \
             tokens, under the {COMMON_PREFIX_FLOOR} floor. Late divergence is \
             expected (the batched verify and the decode GEMV accumulate \
             differently); divergence THIS early is a corrupted drafter or \
             rollback, not reassociation"
        );
        streams.push((block, generated.clone()));

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

    // THE BLOCK IS A THROUGHPUT KNOB AND NOT A MATH KNOB, and unlike
    // byte-identity against a SEQUENTIAL decode this one is measured TRUE:
    // blocks 2, 4, 7 and 8 produce the same tokens, because every one of
    // them commits rows from the same batched kernel and only the batch
    // SIZE differs. That is also what says the divergence above is the
    // kernel pair rather than the batch width -- if M were the variable,
    // these four would disagree with each other too.
    //
    // It is the strongest exact assertion this file can make, and it is
    // strictly stronger than what the old `GENERATE = 256` byte-identity
    // check was really testing: this one has no length below which it stops
    // looking.
    for pair in streams.windows(2) {
        let (a_block, a) = &pair[0];
        let (b_block, b) = &pair[1];
        let n = a.len().min(b.len());
        assert_eq!(
            a[..n],
            b[..n],
            "blocks {a_block} and {b_block} generated different text; the block \
             size must change throughput and nothing else"
        );
    }
    println!(
        "\nall {} block sizes generated identical text; the divergence from the \
         sequential stream is the batched kernel, not the block",
        streams.len()
    );
}

/// The PROTOCOL's own `short-explanation` case, rendered the same way.
///
/// A SECOND prompt, and it exists for one assertion the block sweep's prompt
/// cannot make. The sweep runs a code-shaped answer, whose greedy stream
/// stays byte-identical to a sequential decode for all 600 tokens -- so on
/// that prompt alone the losslessness check can never fail, whatever the
/// engine does. This one diverges at ~200 tokens on the same install
/// (`docs/DFLASH2.md`), which is what makes the floor below a live check
/// rather than a decoration. Taken from `PROTOCOL_CASES` rather than
/// retyped, so it cannot drift from the case the rest of the harness runs.
fn prose_prompt(tokenizer: &tokenizer::MfTokenizer) -> Vec<i32> {
    let case = PROTOCOL_CASES
        .iter()
        .find(|c| c.id == "short-explanation")
        .expect("the protocol carries short-explanation");
    let rendered = tokenizer
        .apply_chat_template(&[Message::new(Role::User, case.content)])
        .expect("chat template renders");
    tokenizer.encode(&rendered, false)
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
