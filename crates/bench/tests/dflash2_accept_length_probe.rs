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
//! THREE WORKLOADS, each swept across every block, because a serving default
//! read off one distribution is a default that has not been tested. Measured
//! per-position acceptance spans **prose 0.65-0.74, code 0.84-0.94, math
//! 0.88-0.98**, so the three bracket the range this checkpoint produces
//! rather than sampling one corner of it.
//!
//! `prose` is the one that DECIDES. It has the lowest acceptance, it is the
//! only workload on which the serving block does not pay, and it is where a
//! large block is catastrophic rather than merely worse -- so
//! `DFLASH_SERVING_BLOCK`'s justification rests on its row, and it is swept
//! here so that row is reproducible from this repo rather than from a CLI
//! measurement in a gitignored file. It is also the only prompt whose greedy
//! stream leaves the sequential one at all, which is what makes the
//! common-prefix floor a live check rather than a decoration.
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

/// The shaping the shipped loop runs under, character for character
/// `real_model.rs`'s `ProtocolShaping::Greedy`: temperature EXACTLY 0.0 (not
/// the smoke's 0.0001 -- `is_deterministic` compares against zero, and the
/// speculative loop REFUSES anything else).
fn greedy_shaping() -> selection::ShapingConfig {
    selection::ShapingConfig::new(0.0, 1, None, 1.0, Some(1)).expect("greedy shaping is valid")
}

/// Picks a token the way `run_raw_completion_speculative` picks one.
///
/// **THIS USED TO BE A LOCAL `argmax` AND THAT WAS A REAL GAP.** The shipped
/// loop calls `selection::select` for the draft chain, for the acceptance
/// comparison and for the bonus row; a probe that argmaxed directly agreed
/// with it at temperature 0 and could not see a regression anywhere in the
/// sampler -- repetition penalty, the top-k path, the deterministic fast
/// path's own guard. Routing through `select` costs nothing here (at
/// temperature 0 it takes that fast path) and makes the probe's acceptance
/// the acceptance a user gets rather than one that resembles it.
///
/// `history` and `generated` are passed rather than stubbed for the same
/// reason: they are what a step-dependent rule would read, so handing it
/// empties would restore exactly the blindness this replaces.
fn pick(
    logits: &[LogitValue],
    shaping: &selection::ShapingConfig,
    history: &[i32],
    generated: usize,
) -> i32 {
    selection::select(
        foundation::LogitsView::new(logits),
        shaping,
        history,
        generated as u64,
    )
    .expect("selection")
}

#[derive(Default)]
struct Stats {
    rounds: usize,
    accepted: usize,
    offered: Vec<usize>,
    matched: Vec<usize>,
    rollbacks: usize,
}

/// One workload's sequential reference: the stream every speculative arm on
/// that prompt is compared against, and the decode clock its speedup column
/// is read against.
///
/// Per workload rather than one shared clock, because the speedup a block
/// pays is a ratio against the SAME prompt decoded sequentially. Reading a
/// math arm against the code prompt's reference would compare two different
/// generations and call the difference speculation.
struct Reference {
    stream: Vec<i32>,
    secs: f64,
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

    let shaping = greedy_shaping();
    let mut history: Vec<i32> = prompt.to_vec();
    let mut next = pick(&logits, &shaping, &history, 0);
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
            // The history the SHIPPED loop would have here: it commits each
            // accepted proposal before sampling the next row, so row `i` is
            // picked against the prefix plus the `i` already accepted.
            let target = pick(row, &shaping, &history, generated.len() + i);
            if stats.rounds <= 3 {
                eprintln!(
                    "[round {}] row {i} target {target} vs proposal {proposal}",
                    stats.rounds
                );
            }
            if target != proposal {
                break;
            }
            stats.matched[i] += 1;
            accepted += 1;
        }
        let bonus = pick(
            &batch_logits[accepted * vocab..(accepted + 1) * vocab],
            &shaping,
            &history,
            generated.len() + accepted,
        );
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
    let shaping = greedy_shaping();
    let mut history: Vec<i32> = prompt.to_vec();
    let mut generated = Vec::new();
    let mut next = pick(&logits, &shaping, &history, 0);
    while generated.len() < GENERATE {
        generated.push(next);
        history.push(next);
        runner
            .produce(next, prompt.len() + generated.len() - 1, &mut logits)
            .expect("produce");
        // The REFERENCE arm goes through the same sampler as the speculative
        // ones. Leaving it on a bare argmax would make the byte-identity gate
        // compare two different sampling rules and call the difference a
        // speculation bug.
        next = pick(&logits, &shaping, &history, generated.len());
    }
    (generated, started.elapsed().as_secs_f64())
}

/// Prints one arm's row and its acceptance curve, reports how far it tracked
/// the sequential stream, and makes the two assertions every arm owes.
///
/// Factored out because there are three workloads now and the assertions are
/// the interesting part: a per-arm copy is a per-arm opportunity for one of
/// them to be dropped, and the one that would go unnoticed is the
/// functional-drafter guard, which is silent on a healthy run.
fn report_arm(
    label: &str,
    block: usize,
    s: &Stats,
    generated: &[i32],
    secs: f64,
    reference: &Reference,
) {
    let per_round = s.accepted as f64 / s.rounds.max(1) as f64;
    println!(
        "{label:>6}  {block:>5}  {:>6}  {:>11.2}  {:>12.2}  {:>9}  {:>7.2}  {:>7.2}x",
        s.rounds,
        per_round,
        per_round + 1.0,
        s.rollbacks,
        secs,
        reference.secs / secs,
    );
    let curve: Vec<String> = s
        .offered
        .iter()
        .zip(&s.matched)
        .map(|(o, m)| format!("{:.2}", if *o == 0 { 0.0 } else { *m as f64 / *o as f64 }))
        .collect();
    println!("        per-position acceptance: {}", curve.join(" "));

    // How far this arm tracks the sequential stream, REPORTED rather than
    // assumed. A round overshoots GENERATE by construction, so the comparison
    // is over the common length.
    let n = reference.stream.len().min(generated.len());
    let prefix = (0..n)
        .take_while(|&i| reference.stream[i] == generated[i])
        .count();
    if prefix == n {
        println!("        matches the sequential stream for all {n} compared tokens");
    } else {
        println!(
            "        matches the sequential stream for {prefix} of {n} tokens, then \
             diverges (expected; see docs/DFLASH2.md)"
        );
    }
    assert!(
        prefix >= COMMON_PREFIX_FLOOR.min(n),
        "{label} block {block} left the sequential greedy stream after only \
         {prefix} tokens, under the {COMMON_PREFIX_FLOOR} floor. Late divergence \
         is expected (the batched verify and the decode GEMV accumulate \
         differently); divergence THIS early is a corrupted drafter or rollback, \
         not reassociation"
    );

    // The functional-drafter guard, the MTP probe's own: near-zero
    // first-proposal acceptance is a broken drafter (a convention read wrongly
    // somewhere in the port), not a verdict about DFlash2.
    let first = s.matched[0] as f64 / s.offered[0].max(1) as f64;
    assert!(
        first > 0.02,
        "{label} block {block}: the drafter's FIRST proposal was accepted {}/{} \
         times ({first:.4}). That is a broken port, not a low accept length -- the \
         published drafter's first-position acceptance is ~0.9 (docs/DFLASH2.md)",
        s.matched[0],
        s.offered[0]
    );
}

/// Every block size must generate IDENTICAL text to every other, on one
/// workload. Strictly stronger than byte-identity against a sequential decode
/// (which is measured FALSE) and, unlike it, has no length below which it
/// stops looking.
fn assert_blocks_agree(label: &str, streams: &[(usize, Vec<i32>)]) {
    for pair in streams.windows(2) {
        let (a_block, a) = &pair[0];
        let (b_block, b) = &pair[1];
        let n = a.len().min(b.len());
        assert_eq!(
            a[..n],
            b[..n],
            "{label}: blocks {a_block} and {b_block} generated different text; the \
             block size must change throughput and nothing else"
        );
    }
    println!(
        "all {} block sizes generated identical {label} text; the divergence from \
         the sequential stream is the batched kernel, not the block",
        streams.len()
    );
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
    let (code_ref, prose_ref, math_ref) = {
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
        let (stream, secs) = run_plain(&mut runner, &prompt, vocab);
        // The other two references ride the SAME open, which is why each
        // costs a decode and not a model load. They need no warmup of their
        // own: the GPU is warm by the time they run.
        let (prose_stream, prose_secs) = run_plain(&mut runner, &prose_prompt(&tokenizer), vocab);
        let (math_stream, math_secs) = run_plain(&mut runner, &math_prompt(&tokenizer), vocab);
        println!(
            "\ncode prompt {} tokens, generating {GENERATE}, greedy",
            prompt.len()
        );
        println!(
            "reference (no drafter): code {:.2} s ({:.2} tok/s), prose {:.2} s, math {:.2} s\n",
            secs,
            GENERATE as f64 / secs,
            prose_secs,
            math_secs
        );
        (
            Reference { stream, secs },
            Reference {
                stream: prose_stream,
                secs: prose_secs,
            },
            Reference {
                stream: math_stream,
                secs: math_secs,
            },
        )
    };

    println!("prompt  block  rounds  accepted/rd  committed/rd  rollbacks  seconds  MEASURED");
    let mut streams: Vec<(usize, Vec<i32>)> = Vec::new();
    let mut math_streams: Vec<(usize, Vec<i32>)> = Vec::new();
    let mut prose_streams: Vec<(usize, Vec<i32>)> = Vec::new();
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
        report_arm("code", block, &s, &generated, secs, &code_ref);
        streams.push((block, generated));

        // THE MATH ARM, swept at every block, and that is what makes this
        // file able to re-open the block choice rather than only re-check it.
        // The two workloads the serving default was set from are a code-shaped
        // answer accepting 0.93-0.98 and prose accepting 0.66-0.83, both well
        // above the GSM8K figures vLLM and llama.cpp publish for this drafter
        // -- so the ordering could have been an artifact of two easy
        // distributions. Multi-step arithmetic is the cheap third opinion:
        // its numeric tokens are the least predictable this checkpoint emits.
        let math = math_prompt(&tokenizer);
        let (ms, math_gen, math_secs) = run_dflash(&mut runner, &math, vocab, block);
        report_arm("math", block, &ms, &math_gen, math_secs, &math_ref);
        math_streams.push((block, math_gen));

        // THE PROSE ARM, and it is SWEPT rather than run at the serving block
        // alone, because it is the workload that decides the default. Its
        // acceptance is the lowest of the three, it is the only one on which
        // the serving block does not pay, and it is where a large block is
        // catastrophic rather than merely worse -- so the const's
        // justification rests on THIS row, and until it was swept here that
        // row existed only in a CLI measurement nothing in the repo could
        // reproduce.
        //
        // It doubles as the losslessness check with teeth: the other two
        // prompts track the sequential stream for all 600 tokens whatever the
        // engine does, so a floor asserted on them alone is a decoration.
        // This one parts from it at ~150 (`docs/DFLASH2.md`).
        let prose = prose_prompt(&tokenizer);
        let (ps, prose_gen, prose_secs) = run_dflash(&mut runner, &prose, vocab, block);
        report_arm("prose", block, &ps, &prose_gen, prose_secs, &prose_ref);
        prose_streams.push((block, prose_gen));
        drop(runner);
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
    // looking. Asserted on BOTH swept workloads, because a batch-width
    // dependence that happened to be invisible on one prompt is exactly the
    // kind of thing one prompt cannot rule out.
    println!();
    assert_blocks_agree("code", &streams);
    assert_blocks_agree("math", &math_streams);
    assert_blocks_agree("prose", &prose_streams);

    // ASSERT THE THIRD WORKLOAD DISCRIMINATES, before any row above is
    // believed. `math_prompt` is one careless edit away from rendering what
    // `protocol_prompt` renders, and if it did, every assertion in this file
    // would still pass while the "third opinion" was the first one twice --
    // the same hazard Gotchas 48, 50 and 51 record on three other axes. Two
    // streams from one model at temperature 0 are equal only if their prompts
    // were.
    let code_first = &streams[0].1;
    let math_first = &math_streams[0].1;
    assert_ne!(
        code_first[..code_first.len().min(math_first.len())],
        math_first[..code_first.len().min(math_first.len())],
        "the math arm generated the code arm's text, so the two prompts are the \
         same and this file measures one workload three times"
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

/// A THIRD prompt: multi-step arithmetic.
///
/// **IT WAS CHOSEN AS THE HARD CASE AND MEASURED AS THE EASY ONE.** The
/// reasoning was that a drafter predicts a NUMBER far worse than it predicts
/// the next word of an explanation, so a chain of arithmetic should be where
/// a block ordering set on easy text comes apart. Measured per-position
/// acceptance says otherwise: **math 0.88-0.98, code 0.84-0.94, prose
/// 0.65-0.74**. Arithmetic working is the most TEMPLATED thing this
/// checkpoint writes -- `Monday's revenue:`, `120 x $3.25 = $390.00`,
/// restated totals -- and the few genuinely unpredictable digits sit in a
/// large majority of scaffolding the drafter gets right.
///
/// So this arm's value is not the one it was added for. It is a third
/// independent workload that AGREES, and it widened the measured acceptance
/// band rather than extending it downward: `prose` was and remains the hard
/// case, and the one the serving default rests on.
///
/// Kept, and kept honest, rather than swapped for something harder. A
/// workload that confirms is evidence; a doc comment that still predicted
/// what this one refuted would be the hedge that outlives its own
/// resolution, which `docs/DFLASH2.md` section 8 already complains about
/// once.
///
/// Written here rather than added to `PROTOCOL_CASES`: a case there is
/// frozen protocol, and adding one would move published rows in every family
/// (`crates/bench/CLAUDE.md` Gotcha 11). Note this checkpoint renders with
/// `enable_thinking: false` (AGENTS.md Gotcha 56), so the working is the
/// ANSWER rather than a `<think>` channel, and every token of it counts
/// toward the sweep.
fn math_prompt(tokenizer: &tokenizer::MfTokenizer) -> Vec<i32> {
    let rendered = tokenizer
        .apply_chat_template(&[Message::new(
            Role::User,
            "A bakery sells croissants at $3.25, muffins at $2.40, and loaves at \
             $5.75. On Monday it sold 120 croissants, 96 muffins, and 54 loaves. \
             On Tuesday croissant sales rose 15%, muffin sales fell by 12, and \
             loaf sales doubled. Ingredients cost 38% of revenue, and wages are \
             $420 per day. Showing every step: compute Monday's revenue, \
             Tuesday's revenue, the two-day total, the two-day ingredient cost, \
             and the two-day profit after wages. Then work out how many extra \
             loaves Tuesday would have needed for the two-day profit to reach \
             $1,500.",
        )])
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
