#![cfg(target_os = "macos")]
//! How far does the drift-bound arm of `TURBOSPARK_INT4_MMA8` move verify
//! away from decode? (`docs/INT4_SMALL_M_MMA.md`.)
//!
//! Under `TURBOSPARK_INT4_MMA8=verify` the batched dispatcher runs the
//! matrix-unit tile and one-row decode keeps the GEMV, so a verify row is no
//! longer bit-identical to the decode step it stands in for. Two readings:
//!
//! 1. TEACHER-FORCED DRIFT. A greedy reference stream decoded one token at a
//!    time, then the same tokens replayed through batched passes of `B` rows,
//!    each committed in full (the round-end commit of a fully accepted
//!    verify). State carries across passes, so drift compounds through the
//!    KV and GDN state the way it would in a real speculative run. Per row:
//!    KL(decode || verify) in nats, top-1 agreement, max |logit diff|.
//! 2. THE WIRED LOOP. `run_raw_completion_speculative` against the
//!    sequential `run_raw_completion`: tok/s, and how many tokens the
//!    speculative stream matches before it first departs.
//!
//! Run it with the flag OFF too. That arm is the baseline, not an identity:
//! on the real 27B the batched path already differs from decode in the last
//! bits without the tile (the synthetic suites prove bit-identity only at
//! their 128-wide shapes), so the drift the tile ADDS is the verify arm
//! minus the control.
//!
//! ```sh
//! TURBOSPARK_INT4_MMA8=verify TURBOSPARK_MTP_INSTALL_DIR=~/models/qwen38-27b-mtp.gturbo \
//!   cargo test -p turbospark-bench --test int4_mma8_drift_probe --release -- --ignored --nocapture
//! ```

use foundation::LogitValue;
use runtime::{
    run_raw_completion, run_raw_completion_speculative, GenerationConfig, LogitProducer,
    RateControl, RawDecodeProgress, RealForwardRunner,
};
use selection::ShapingConfig;
use tokenizer::{Message, Role};
use turbospark_bench::protocol::PROTOCOL_CASES;
use turbospark_bench::real_model::open_model_runner_speculative;

/// Head capacity at open: the verify scratch holds `MTP_DEPTH + 1` rows.
const MTP_DEPTH: usize = 8;
/// Teacher-forced tokens per prompt.
const FORCED: usize = 256;
/// Rows per batched pass: a block-2 and a block-4 verify.
const PASS_ROWS: [usize; 2] = [3, 5];
/// Tokens per wired-loop run.
const GENERATE: u32 = 256;
const BLOCKS: [usize; 2] = [2, 4];
const SLOTS: usize = 16;

fn argmax(logits: &[LogitValue]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, v) in logits.iter().enumerate() {
        let f = v.to_f32();
        if f > best_v {
            best_v = f;
            best = i;
        }
    }
    best
}

/// KL(p || q) in nats over full-vocabulary softmaxes, f64.
fn kl(p_logits: &[LogitValue], q_logits: &[LogitValue]) -> f64 {
    let lse = |l: &[LogitValue]| {
        let m = l
            .iter()
            .map(|v| v.to_f32() as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        m + l
            .iter()
            .map(|v| (v.to_f32() as f64 - m).exp())
            .sum::<f64>()
            .ln()
    };
    let (lp, lq) = (lse(p_logits), lse(q_logits));
    p_logits
        .iter()
        .zip(q_logits)
        .map(|(a, b)| {
            let la = a.to_f32() as f64 - lp;
            let lb = b.to_f32() as f64 - lq;
            la.exp() * (la - lb)
        })
        .sum::<f64>()
        .max(0.0)
}

fn prefill(runner: &mut RealForwardRunner, prompt: &[i32], logits: &mut [LogitValue]) {
    runner.reset();
    for (i, &t) in prompt.iter().enumerate() {
        runner.produce(t, i, logits).expect("produce");
    }
}

#[derive(Default)]
struct Drift {
    rows: usize,
    kl_sum: f64,
    kl_max: f64,
    top1: usize,
    max_diff: f32,
    bit_equal: usize,
}

/// Part 1 for one prompt and one pass width.
fn teacher_forced(
    runner: &mut RealForwardRunner,
    prompt: &[i32],
    vocab: usize,
    rows: usize,
) -> Drift {
    // Reference: one token at a time, GEMV arithmetic in every mode.
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    prefill(runner, prompt, &mut logits);
    let mut stream = Vec::with_capacity(FORCED + 1);
    stream.push(argmax(&logits) as i32);
    let mut reference: Vec<Vec<LogitValue>> = Vec::with_capacity(FORCED);
    for j in 0..FORCED {
        runner
            .produce(stream[j], prompt.len() + j, &mut logits)
            .expect("produce");
        reference.push(logits.clone());
        stream.push(argmax(&logits) as i32);
    }

    // Replay: the same tokens in batched passes, each fully committed.
    prefill(runner, prompt, &mut logits);
    let mut out = vec![LogitValue::from_f32(0.0); rows * vocab];
    let mut d = Drift::default();
    let mut j = 0;
    while j + rows <= FORCED {
        runner
            .produce_batched(&stream[j..j + rows], prompt.len() + j, &mut out)
            .expect("batched pass");
        runner.commit_pending(rows).expect("round-end commit");
        for r in 0..rows {
            let got = &out[r * vocab..(r + 1) * vocab];
            let want = &reference[j + r];
            let k = kl(want, got);
            d.rows += 1;
            d.kl_sum += k;
            d.kl_max = d.kl_max.max(k);
            d.top1 += usize::from(argmax(got) == argmax(want));
            let diff = got
                .iter()
                .zip(want.iter())
                .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
                .fold(0.0f32, f32::max);
            d.max_diff = d.max_diff.max(diff);
            d.bit_equal += usize::from(
                got.iter()
                    .zip(want.iter())
                    .all(|(a, b)| a.to_f32().to_bits() == b.to_f32().to_bits()),
            );
        }
        j += rows;
    }
    d
}

fn greedy_config() -> GenerationConfig {
    GenerationConfig {
        shaping: ShapingConfig::new(0.0, 0, None, 1.0, None).unwrap(),
        max_new_tokens: GENERATE,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
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

fn render(tokenizer: &tokenizer::MfTokenizer, content: &str) -> Vec<i32> {
    let rendered = tokenizer
        .apply_chat_template(&[Message::new(Role::User, content)])
        .expect("chat template renders");
    tokenizer.encode(&rendered, false)
}

#[test]
#[ignore = "needs a real MTP install via TURBOSPARK_MTP_INSTALL_DIR"]
fn verify_drift_against_decode() {
    let mode = std::env::var("TURBOSPARK_INT4_MMA8").unwrap_or_default();
    let control = mode.is_empty() || mode == "0";
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

    let mut prompts: Vec<(&str, Vec<i32>)> = PROTOCOL_CASES[..2]
        .iter()
        .map(|c| (c.id, render(&tokenizer, c.content)))
        .collect();
    prompts.push((
        "code",
        render(
            &tokenizer,
            "Write a Python function that merges two sorted lists, then explain \
             its time and space complexity in detail.",
        ),
    ));

    println!(
        "\nTURBOSPARK_INT4_MMA8={:?} ({}), {FORCED} teacher-forced tokens per prompt\n",
        mode,
        if control { "control" } else { "drift arm" }
    );
    println!(
        "prompt             rows  mean KL (nats)   max KL    top-1     max |dlogit|  bit-equal rows"
    );
    for (id, prompt) in &prompts {
        for rows in PASS_ROWS {
            let d = teacher_forced(&mut runner, prompt, vocab, rows);
            println!(
                "{id:<18} {rows:>4}  {:>14.3e}  {:>9.3e}  {:>6.2}%  {:>12.4}  {}/{}",
                d.kl_sum / d.rows as f64,
                d.kl_max,
                100.0 * d.top1 as f64 / d.rows as f64,
                d.max_diff,
                d.bit_equal,
                d.rows
            );
            // Not asserted: on the real 27B the batched path already differs
            // from decode in the last bits with the flag off (measured
            // 2026-10-22), so the control is a baseline to subtract.
        }
    }

    println!("\nwired loop, {GENERATE} greedy tokens");
    println!("prompt             arm         tok/s   matches sequential (prefix / total)");
    let config = greedy_config();
    for (id, prompt) in &prompts {
        // Warm the clocks once per prompt (AGENTS.md Gotcha 20).
        run_raw_completion(
            &mut runner,
            &tokenizer,
            prompt,
            &config,
            4096,
            vocab,
            |_| {},
        )
        .expect("warmup");
        let mut ev = Vec::new();
        let seq = run_raw_completion(&mut runner, &tokenizer, prompt, &config, 4096, vocab, |e| {
            ev.push(e)
        })
        .expect("sequential");
        let seq_tokens = tokens_of(&ev);
        println!(
            "{id:<18} sequential  {:>6.2}",
            seq.new_tokens as f64 / seq.decode_seconds
        );
        for block in BLOCKS {
            let mut ev = Vec::new();
            let spec = run_raw_completion_speculative(
                &mut runner,
                &tokenizer,
                prompt,
                &config,
                4096,
                vocab,
                block,
                |e| ev.push(e),
            )
            .expect("speculative");
            let t = tokens_of(&ev);
            let n = t.len().min(seq_tokens.len());
            let prefix = (0..n).take_while(|&i| t[i] == seq_tokens[i]).count();
            println!(
                "{id:<18} block {block:<5} {:>6.2}   {prefix} / {n}",
                spec.new_tokens as f64 / spec.decode_seconds
            );
            if control && prefix != n {
                println!("  ^ the CONTROL departed from sequential decode");
            }
        }
    }
}
