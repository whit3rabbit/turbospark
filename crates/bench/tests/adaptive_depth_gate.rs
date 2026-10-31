#![cfg(target_os = "macos")]
//! Does adaptive MTP draft depth beat the fixed block the engine ships, and
//! is it lossless? (`docs/MTP_SPECULATIVE.md`)
//!
//! One install, opened ONCE at the cap, so every arm runs against the same
//! head allocation and the same scratch width: opening per arm would put the
//! allocation difference into the very number being compared.
//!
//! Arms are INTERLEAVED and their order rotated per repetition, because a
//! sequential A-then-B run on this machine measures thermal drift as much as
//! the arms (`CLAUDE.local.md`). One discarded warmup first. Greedy, because
//! acceptance is exact only at temperature 0.
//!
//! The assertion is identity of the token stream across every arm. The tok/s
//! table is printed for the reader and asserted on by nothing: a gate that
//! failed on machine state would be worse than no gate (AGENTS.md Gotcha 43).
//! Run it on AC power with a quiet host, redirecting output to a file.
//!
//! ```sh
//! TURBOSPARK_MTP_INSTALL_DIR=~/models/qwen38-27b.gturbo \
//!   cargo test -p turbospark-bench --test adaptive_depth_gate --release \
//!   -- --ignored --nocapture > adaptive.out 2> adaptive.err
//! ```

use runtime::{
    run_raw_completion, run_raw_completion_speculative, run_raw_completion_speculative_adaptive,
    GenerationConfig, RateControl, RawDecodeProgress,
};
use selection::ShapingConfig;
use tokenizer::{Message, Role};
use turbospark_bench::real_model::open_model_runner_speculative;

const SLOTS: usize = 16;
/// The head is opened this deep for EVERY arm, so the allocation is not a
/// difference between them. The deepest adaptive cap must fit.
const OPEN_DEPTH: usize = 8;
const GENERATE: u32 = 256;
const REPS: usize = 3;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    Fixed(usize),
    /// Adaptive with this cap; the head is opened at `OPEN_DEPTH` regardless.
    Adaptive(usize),
}

impl Arm {
    fn label(self) -> String {
        match self {
            Arm::Fixed(n) => format!("fixed {n}"),
            Arm::Adaptive(cap) => format!("adaptive 2..{cap}"),
        }
    }
}

const ARMS: [Arm; 5] = [
    Arm::Fixed(2),
    Arm::Fixed(3),
    Arm::Fixed(4),
    Arm::Adaptive(4),
    Arm::Adaptive(8),
];

const RECALL_PASSAGE: &str = "The committee met on the third of March to review the harbor \
    expansion. Three options were tabled: a deep-water quay on the north shore, a floating \
    breakwater off the eastern spit, and a phased dredging programme along the existing channel. \
    The quay was costed at forty-one million, the breakwater at twenty-eight million, and the \
    dredging programme at nineteen million spread over six years. After discussion the committee \
    resolved to commission a full environmental survey of the eastern spit before any vote.";

fn prompts() -> Vec<(&'static str, String)> {
    vec![
        (
            "prose",
            "Write a reflective essay of about two hundred words on why people keep \
             returning to the town where they grew up."
                .to_string(),
        ),
        (
            "reasoning",
            "A train leaves at 9:15 travelling 80 km/h. A second leaves the same station \
             at 10:00 travelling 110 km/h on the same line. At what time does the second \
             catch the first? Work through it step by step."
                .to_string(),
        ),
        (
            "code",
            "Write a Python function that merges two sorted lists, then write unit tests \
             for it using pytest."
                .to_string(),
        ),
        (
            "recall",
            format!(
                "Repeat the following text back to me exactly, word for word:\n\n{RECALL_PASSAGE}"
            ),
        ),
    ]
}

fn greedy_config() -> GenerationConfig {
    GenerationConfig {
        // Temperature 0 EXACTLY: `is_deterministic` is `== 0.0`.
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

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore = "needs a real MTP install via TURBOSPARK_MTP_INSTALL_DIR and a quiet host on AC power"]
fn adaptive_depth_against_fixed_blocks() {
    // Read inside the loop per call; set once so adaptive's depth histogram
    // lands on stderr beside each run.
    std::env::set_var("TURBOSPARK_SPEC_STATS", "1");

    let dir = std::path::PathBuf::from(
        std::env::var_os("TURBOSPARK_MTP_INSTALL_DIR").expect("TURBOSPARK_MTP_INSTALL_DIR"),
    );
    let (mut runner, tokenizer) = open_model_runner_speculative(
        &dir,
        SLOTS,
        runtime::DraftPolicies::mtp(runtime::MtpDraftPolicy::Fixed(OPEN_DEPTH)),
    )
    .expect("install opens");
    assert!(
        runner.mtp_draft_depth() >= OPEN_DEPTH,
        "this install carries no MTP head (or opened shallower than the cap); nothing to compare"
    );
    let vocab = runner.vocab_size();
    let config = greedy_config();

    let cases: Vec<(&str, Vec<i32>)> = prompts()
        .into_iter()
        .map(|(name, text)| {
            let rendered = tokenizer
                .apply_chat_template(&[Message::new(Role::User, &text)])
                .expect("chat template renders");
            (name, tokenizer.encode(&rendered, false))
        })
        .collect();

    // Discarded: the first decode of a process is on a cold GPU at low clocks
    // (AGENTS.md Gotcha 20) and would land on whichever arm ran first.
    run_raw_completion(
        &mut runner,
        &tokenizer,
        &cases[0].1,
        &config,
        4096,
        vocab,
        |_| {},
    )
    .expect("warmup run");

    // The reference is the NON-SPECULATIVE loop, per prompt, so every arm is
    // compared against something that is not itself speculative.
    let mut reference: Vec<Vec<i32>> = Vec::new();
    let mut seq_tps: Vec<f64> = Vec::new();
    for (name, prompt) in &cases {
        let mut events = Vec::new();
        let r = run_raw_completion(&mut runner, &tokenizer, prompt, &config, 4096, vocab, |e| {
            events.push(e)
        })
        .expect("sequential run");
        seq_tps.push(r.new_tokens as f64 / r.decode_seconds);
        println!(
            "sequential {name}: {} tokens, {:.2} tok/s",
            r.new_tokens,
            seq_tps.last().unwrap()
        );
        reference.push(tokens_of(&events));
    }

    // tps[prompt][arm] -> samples
    let mut tps = vec![vec![Vec::<f64>::new(); ARMS.len()]; cases.len()];
    for rep in 0..REPS {
        for (pi, (name, prompt)) in cases.iter().enumerate() {
            for k in 0..ARMS.len() {
                // Rotated so no arm is always first or always after the same
                // neighbour.
                let ai = (k + rep + pi) % ARMS.len();
                let arm = ARMS[ai];
                eprintln!("[arm] rep {rep} prompt {name} {}", arm.label());
                let mut events = Vec::new();
                let result = match arm {
                    Arm::Fixed(n) => run_raw_completion_speculative(
                        &mut runner,
                        &tokenizer,
                        prompt,
                        &config,
                        4096,
                        vocab,
                        n,
                        |e| events.push(e),
                    ),
                    Arm::Adaptive(cap) => run_raw_completion_speculative_adaptive(
                        &mut runner,
                        &tokenizer,
                        prompt,
                        &config,
                        4096,
                        vocab,
                        cap,
                        runtime::AdaptiveDepth::DEFAULT_FLOOR,
                        |e| events.push(e),
                    ),
                }
                .expect("speculative run");
                // THE GATE: identity against the non-speculative stream.
                assert_eq!(
                    tokens_of(&events),
                    reference[pi],
                    "{} diverged from the sequential stream on {name}",
                    arm.label()
                );
                tps[pi][ai].push(result.new_tokens as f64 / result.decode_seconds);
            }
        }
    }

    println!(
        "\nmedian decode tok/s over {REPS} reps, greedy, {GENERATE} tokens, head opened at {OPEN_DEPTH}\n"
    );
    print!("{:<10} {:>10}", "prompt", "sequential");
    for arm in ARMS {
        print!(" {:>16}", arm.label());
    }
    for arm in ARMS {
        if let Arm::Adaptive(cap) = arm {
            print!(" {:>9}", format!("a{cap}/f2"));
        }
    }
    println!();
    for (pi, (name, _)) in cases.iter().enumerate() {
        print!("{name:<10} {:>10.2}", seq_tps[pi]);
        let mut med = Vec::new();
        for arm_tps in tps[pi].iter().take(ARMS.len()) {
            let m = median(&mut arm_tps.clone());
            med.push(m);
            print!(" {m:>16.2}");
        }
        let f2 = med[ARMS.iter().position(|a| *a == Arm::Fixed(2)).unwrap()];
        // Every adaptive arm against fixed 2, one ratio column each.
        for (ai, arm) in ARMS.iter().enumerate() {
            if matches!(arm, Arm::Adaptive(_)) {
                print!(" {:>8.3}x", med[ai] / f2);
            }
        }
        println!();
    }
    println!("\nlossless against the sequential stream on every arm and prompt\n");
}
