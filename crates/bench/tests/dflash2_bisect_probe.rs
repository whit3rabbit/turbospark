//! DFlash2 bisect probe: the rank of the TRUE next token in the drafter's
//! own distribution, teacher-forced (`docs/DFLASH2.md`).
//!
//! The acceptance probe's functional-drafter guard is the entry to this
//! file: 0-of-256 first-proposal acceptance is a broken port, and this is
//! the instrument that localizes it, in the MTP effort's own shape
//! (`mtp_bisect.py`'s cousin). A median rank near 0 says the backbone and
//! context write are right and the suspicion moves to the selector; a
//! median rank in the tens of thousands says the drafter is computing
//! garbage and the suspicion moves upstream (aux capture, fc, norms, rope,
//! conv). It prints the number rather than asserting, because EITHER
//! answer routes work rather than ends it.
//!
//! ```sh
//! TURBOSPARK_DFLASH2_INSTALL_DIR=~/models/qwen38-27b-dflash2.gturbo \
//!   cargo test -p turbospark-bench --test dflash2_bisect_probe --release -- --ignored --nocapture
//! ```

use foundation::LogitValue;
use runtime::{DflashDraftPolicy, DraftPolicies, LogitProducer, MtpDraftPolicy};
use tokenizer::{Message, Role};
use turbospark_bench::real_model::open_model_runner_speculative;

const BLOCK: usize = 8;
const SLOTS: usize = 16;
/// Steps of teacher-forced drafting. Enough for a stable median; the walk
/// is sequential so each step costs one produce plus one draft pass.
const STEPS: usize = 48;

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

fn rank_of(logits: &[LogitValue], token: i32) -> usize {
    let target = logits[token as usize].to_f32();
    logits.iter().filter(|&&v| v.to_f32() > target).count()
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

#[test]
#[ignore = "needs a real DFlash2 install via TURBOSPARK_DFLASH2_INSTALL_DIR"]
fn true_token_rank_in_the_drafters_own_distribution() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("TURBOSPARK_DFLASH2_INSTALL_DIR").expect("TURBOSPARK_DFLASH2_INSTALL_DIR"),
    );

    // The truth walk, on a drafter-off runner.
    let (mut truth, tokenizer) = open_model_runner_speculative(
        &dir,
        SLOTS,
        DraftPolicies {
            mtp: MtpDraftPolicy::Off,
            dflash: DflashDraftPolicy::Off,
        },
    )
    .expect("truth runner opens");
    let vocab = truth.vocab_size();
    let prompt = protocol_prompt(&tokenizer);
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    for (i, &t) in prompt.iter().enumerate() {
        truth.produce(t, i, &mut logits).expect("truth prefill");
    }
    // The true continuation, one token at a time.
    let mut true_stream: Vec<i32> = Vec::new();
    let mut next = argmax(&logits);
    for _ in 0..STEPS + 2 {
        true_stream.push(next);
        truth
            .produce(next, prompt.len() + true_stream.len() - 1, &mut logits)
            .expect("truth produce");
        next = argmax(&logits);
    }
    drop(truth);

    // The drafter runner: prefill primed, then teacher-forced rounds where
    // the ANCHOR is the true token and the question is where the true
    // NEXT-next token ranks in the drafter's row-1 logits.
    let (mut runner, _tokenizer) = open_model_runner_speculative(
        &dir,
        SLOTS,
        DraftPolicies {
            mtp: MtpDraftPolicy::Off,
            dflash: DflashDraftPolicy::Fixed(BLOCK),
        },
    )
    .expect("drafter runner opens");
    let mut dlogits = vec![LogitValue::from_f32(0.0); vocab];
    for (i, &t) in prompt.iter().enumerate() {
        runner.produce(t, i, &mut logits).expect("drafter prefill");
        if i + 1 < prompt.len() {
            runner
                .dflash_prime_from_capture(i)
                .expect("prime over the prompt");
        }
    }

    let mut ranks: Vec<usize> = Vec::new();
    // The LOOP's alignment, asked beside the bisect's: no produce of the
    // anchor first, draft at base = the anchor's own position, context
    // [0, pos). Row 1 should rank t_{pos+1} #1 if the loop's shape matches
    // the drafter's trained shape; row 2 ranking it instead means the rows
    // are one position off in the loop.
    let mut loop_row1: Vec<usize> = Vec::new();
    let mut loop_row2: Vec<usize> = Vec::new();
    let mut walk_vs_unary: Vec<usize> = Vec::new();
    let mut proposals = Vec::new();
    for step in 0..STEPS {
        // The trunk absorbs the true token at position `pos`, which also
        // refreshes the capture the context write reads.
        let pos = prompt.len() + step;
        let anchor = true_stream[step];
        runner.produce(anchor, pos, &mut logits).expect("produce");
        let true_next = true_stream[step + 1];

        // (a) The bisect alignment: context including the anchor, rows at
        //     [pos+1 ..]. Known-good by construction of this probe.
        runner
            .dflash_draft_block(anchor, pos + 1, &mut proposals)
            .expect("draft block");
        runner
            .dflash_probe_logits(1, &mut dlogits)
            .expect("read row 1");
        ranks.push(rank_of(&dlogits, true_next));

        // (b) The LOOP alignment: rewind the drafter one (the context
        //     write will re-cover [pos, pos+1)), draft at base = pos with
        //     the anchor's embedding row AT the anchor's own position.
        runner
            .dflash_rewind_to(pos)
            .expect("rewind to the anchor's position");
        runner
            .dflash_draft_block(anchor, pos, &mut proposals)
            .expect("draft block at the loop's base");
        runner
            .dflash_probe_logits(1, &mut dlogits)
            .expect("read row 1");
        loop_row1.push(rank_of(&dlogits, true_next));
        runner
            .dflash_probe_logits(2, &mut dlogits)
            .expect("read row 2");
        loop_row2.push(rank_of(&dlogits, true_next));
        // Selector-vs-unary: does the walk's first proposal agree with the
        // drafter's own top-1? If it never does, the edge score is
        // overriding a correct unary with garbage.
        runner
            .dflash_probe_logits(1, &mut dlogits)
            .expect("read row 1 again");
        walk_vs_unary.push((proposals[0] == argmax(&dlogits)) as usize);
    }

    let report = |name: &str, v: &[usize]| {
        let mut s = v.to_vec();
        s.sort_unstable();
        let top1 = s.iter().filter(|&&r| r == 0).count();
        println!(
            "  {name}: median {}, p90 {}, top-1 {}/{}",
            s[s.len() / 2],
            s[(s.len() as f64 * 0.9) as usize],
            top1,
            s.len()
        );
    };
    ranks.sort_unstable();
    println!("\nrank of the true token, {STEPS} teacher-forced steps, three alignments:");
    report("bisect (ctx incl anchor, base=pos+1), row 1", &ranks);
    report("loop   (ctx excl anchor, base=pos),     row 1", &loop_row1);
    report("loop   (ctx excl anchor, base=pos),     row 2", &loop_row2);
    println!(
        "  walk's first proposal equals the row-1 argmax: {}/{} steps",
        walk_vs_unary.iter().sum::<usize>(),
        walk_vs_unary.len()
    );
    println!(
        "\nif the loop's row 1 ranks the true token #1, the backbone is right in the\n\
         loop's shape too and the bug is the selector; if the loop's row 2 does,\n\
         the rows are one position off under the loop's base."
    );
}

/// Per-row diagnostics off one draft pass: enough to separate "the block
/// forward wrote nothing" (a CONSTANT row) from "it wrote NaN" (which
/// `top_k`'s `v <= val[k-1]` admits, since every NaN comparison is false,
/// and which then makes `score > best_score` false at every candidate so
/// the walk keeps `cand[0]`) from "it computed something and the selector
/// chose badly".
/// One arm of the single-variable experiment below.
struct Arm {
    name: &'static str,
    proposals: Vec<i32>,
    row1: Vec<LogitValue>,
    stats: Vec<RowStats>,
}

struct RowStats {
    min: f32,
    max: f32,
    nans: usize,
    argmax: i32,
}

fn row_stats(logits: &[LogitValue]) -> RowStats {
    let mut s = RowStats {
        min: f32::INFINITY,
        max: f32::NEG_INFINITY,
        nans: 0,
        argmax: 0,
    };
    let mut best = f32::NEG_INFINITY;
    for (i, v) in logits.iter().enumerate() {
        let f = v.to_f32();
        if f.is_nan() {
            s.nans += 1;
            continue;
        }
        s.min = s.min.min(f);
        s.max = s.max.max(f);
        if f > best {
            best = f;
            s.argmax = i as i32;
        }
    }
    s
}

/// THE SINGLE-VARIABLE EXPERIMENT the bisect above cannot run, because its
/// two alignments differ in the base AND in whether a context write happens
/// AND in whether the anchor was produced first.
///
/// Here one runner, one prefill, one anchor and one base are drafted TWICE:
///
/// - **C1** is the loop's own round 1 exactly -- no `produce` of the anchor
///   first, `base = prompt.len()`, and a context write of the one captured
///   row at `prompt.len() - 1` (the last prompt token, which the prefill's
///   priming deliberately leaves for the first round to cover).
/// - **C2** repeats the identical call. `dflash_context_write` zeroes
///   `capture_rows` when it consumes the capture, so the second call
///   computes `ctx_rows == 0` and skips the write. Same base, same anchor,
///   same cursor, same everything else.
///
/// So C1 minus C2 IS the context write, and nothing else. If C1 proposes
/// zeros and C2 proposes the true token, the fault is in the write executed
/// in the same round as the draft; if both propose zeros, it is not.
#[test]
#[ignore = "needs a real DFlash2 install via TURBOSPARK_DFLASH2_INSTALL_DIR"]
fn the_context_write_is_the_only_variable() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("TURBOSPARK_DFLASH2_INSTALL_DIR").expect("TURBOSPARK_DFLASH2_INSTALL_DIR"),
    );
    let (mut runner, tokenizer) = open_model_runner_speculative(
        &dir,
        SLOTS,
        DraftPolicies {
            mtp: MtpDraftPolicy::Off,
            dflash: DflashDraftPolicy::Fixed(BLOCK),
        },
    )
    .expect("drafter runner opens");
    let vocab = runner.vocab_size();
    let prompt = protocol_prompt(&tokenizer);
    let base = prompt.len();
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];

    // The loop's prefill, priming as it goes and leaving the LAST prompt
    // token's row for the first round's context write.
    for (i, &t) in prompt.iter().enumerate() {
        runner.produce(t, i, &mut logits).expect("prefill");
        if i + 1 < prompt.len() {
            runner.dflash_prime_from_capture(i).expect("prime");
        }
    }
    let anchor = argmax(&logits);

    let mut row = vec![LogitValue::from_f32(0.0); vocab];
    let mut proposals = Vec::new();
    let mut arms: Vec<Arm> = Vec::new();
    for name in ["C1 (with the context write)", "C2 (write already consumed)"] {
        runner
            .dflash_draft_block(anchor, base, &mut proposals)
            .expect("the draft block runs");
        let stats: Vec<RowStats> = (0..=BLOCK)
            .map(|r| {
                runner.dflash_probe_logits(r, &mut row).expect("probe");
                row_stats(&row)
            })
            .collect();
        runner
            .dflash_probe_logits(1, &mut row)
            .expect("probe row 1");
        println!("\n{name}: stage scan over the persistent draft buffers");
        for (buf, nans, min, max) in runner.dflash_probe_buffers() {
            println!("  {buf:>13}: nan {nans:>7}  min {min:>12.4}  max {max:>12.4}");
        }
        arms.push(Arm {
            name,
            proposals: proposals.clone(),
            row1: row.clone(),
            stats,
        });
    }

    // The truth, taken AFTER both arms so the produce cannot refresh the
    // capture either of them read.
    runner.produce(anchor, base, &mut logits).expect("produce");
    let true_next = argmax(&logits);

    println!("\nprompt {base} tokens, anchor {anchor}, true next {true_next}, block {BLOCK}");
    for arm in &arms {
        println!("\n{}", arm.name);
        println!("  proposals: {:?}", arm.proposals);
        println!(
            "  rank of the true next token in row 1: {}",
            rank_of(&arm.row1, true_next)
        );
        for (r, s) in arm.stats.iter().enumerate() {
            println!(
                "  row {r}: min {:>10.4}  max {:>10.4}  nan {:>6}  argmax {}",
                s.min, s.max, s.nans, s.argmax
            );
        }
    }
    println!(
        "\nC1 is the loop's round 1 and C2 differs from it ONLY by the context\n\
         write. A difference here localizes the zero-proposal bug to that write."
    );
}
