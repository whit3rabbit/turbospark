#![cfg(target_os = "macos")]
//! Would a prompt-lookup draft source pay on the dense Qwen3.8-27B, and on
//! which kind of output? (`docs/KAPITSA_KERNELS.md`, "Prompt lookup".)
//!
//! `accept_length_probe.rs` already refuted lookup as a free drafter on prose
//! with the 35B MoE: it fired on 10-11% of rounds. What it never covered is
//! copy-heavy output, the one place lookup is expected to pay, and the dense
//! 27B whose verify is much cheaper than the MoE's. This probe asks that.
//!
//! It generates ONE greedy stream per case with plain decode, then REPLAYS the
//! stream offline. Greedy acceptance is exact string equality against that
//! stream, so no verify pass runs: whether a proposal would have been accepted
//! is read straight off the reference. The replay therefore costs one plain
//! generation per case and nothing else, and it cannot disagree with the real
//! loop about acceptance. It CAN be optimistic about cost, see [`verify_cost`].
//!
//! Round semantics match `run_speculative`: a round starts with a committed
//! token `next`, proposes up to `k` tokens from the history, verifies
//! `[next, proposals..]` in one pass, commits the accepted prefix, and the
//! bonus token starts the next round. A round with no proposal runs one
//! sequential step and pays nothing for speculation, which is the shape the
//! planned `Lookup` drafter would have.
//!
//! ```sh
//! TURBOSPARK_PROBE_INSTALL_DIR=~/models/qwen38-27b.gturbo \
//!   cargo test -p turbospark-bench --test pld_replay_probe --release -- --ignored --nocapture
//! ```

use foundation::LogitValue;
use runtime::{LogitProducer, RealForwardRunner};
use tokenizer::{Message, Role};
use turbospark_bench::real_model::open_model_runner;

/// Tokens generated per case. Long enough for a copied span to repeat.
const GENERATE: usize = 512;
/// N-gram suffix lengths tried. Longer fires less often and predicts better.
const NGRAMS: [usize; 3] = [2, 3, 4];
/// Proposal counts tried per round.
const DRAFTS: [usize; 3] = [2, 4, 8];

/// Verify cost in sequential decode steps for a pass of `rows` rows, from the
/// dense 27B break-even column of `docs/MTP.md` (3, 5, 9 and 16 rows read
/// 1.71, 2.86, 4.53 and 7.96), linearly interpolated between them.
///
/// **OPTIMISTIC.** That column was measured with the MTP head's draft cost
/// folded in, which makes these rows slightly dear for a head-free source, and
/// it carries no separate rollback term, which makes them cheap for a partial
/// round. The two do not cancel in general. Treat the speedups below as a
/// ceiling that decides go or no-go, not as a projection to quote.
fn verify_cost(rows: usize) -> f64 {
    const POINTS: [(f64, f64); 5] = [
        (1.0, 1.0),
        (3.0, 1.71),
        (5.0, 2.86),
        (9.0, 4.53),
        (16.0, 7.96),
    ];
    let x = rows as f64;
    for w in POINTS.windows(2) {
        let (x0, y0) = w[0];
        let (x1, y1) = w[1];
        if x <= x1 {
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
        }
    }
    POINTS[POINTS.len() - 1].1
}

/// What followed the most recent earlier occurrence of the last `n` tokens,
/// up to `want` tokens. The tail's own position is excluded, so a run of one
/// repeated token cannot match itself.
fn propose(history: &[i32], n: usize, want: usize) -> Vec<i32> {
    if n == 0 || want == 0 || history.len() < n + 1 {
        return Vec::new();
    }
    let tail = &history[history.len() - n..];
    let last = history.len() - n;
    for start in (0..last).rev() {
        if &history[start..start + n] == tail {
            let from = start + n;
            let take = want.min(history.len() - from);
            if take > 0 {
                return history[from..from + take].to_vec();
            }
        }
    }
    Vec::new()
}

#[derive(Debug, Default, PartialEq)]
struct Replay {
    /// Rounds run, fired or not.
    rounds: usize,
    /// Rounds that proposed something.
    fired: usize,
    proposed: usize,
    accepted: usize,
    /// Tokens committed, which is the stream length minus the prompt walk.
    committed: usize,
    /// Cost in sequential decode steps.
    cost: f64,
}

impl Replay {
    fn speedup(&self) -> f64 {
        self.committed as f64 / self.cost
    }
}

/// Replays `stream` (the greedy continuation of `prompt`) under lookup.
fn replay(prompt: &[i32], stream: &[i32], n: usize, k: usize) -> Replay {
    let mut out = Replay::default();
    let mut history: Vec<i32> = prompt.to_vec();
    let mut i = 0usize;
    // `stream[i]` is the committed token that opens round `i`.
    while i < stream.len() {
        history.push(stream[i]);
        out.rounds += 1;
        // Never propose past the end of the reference.
        let room = stream.len() - 1 - i;
        let draft = propose(&history, n, k.min(room));
        if draft.is_empty() {
            out.cost += 1.0;
            out.committed += 1;
            i += 1;
            continue;
        }
        let accepted = draft
            .iter()
            .zip(&stream[i + 1..])
            .take_while(|(d, s)| d == s)
            .count();
        out.fired += 1;
        out.proposed += draft.len();
        out.accepted += accepted;
        out.cost += verify_cost(1 + draft.len());
        // Next round's `next` is the bonus, so the accepted proposals are
        // committed here and the bonus is committed at the top of the loop.
        for &t in &stream[i + 1..=i + accepted] {
            history.push(t);
        }
        out.committed += 1 + accepted;
        i += 1 + accepted;
    }
    out
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

fn generate(runner: &mut RealForwardRunner, prompt: &[i32], vocab: usize) -> Vec<i32> {
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    runner.reset();
    for (i, &token) in prompt.iter().enumerate() {
        runner.produce(token, i, &mut logits).expect("produce");
    }
    let mut stream = Vec::with_capacity(GENERATE);
    let mut next = argmax(&logits);
    while stream.len() < GENERATE {
        stream.push(next);
        let pos = prompt.len() + stream.len() - 1;
        runner.produce(next, pos, &mut logits).expect("produce");
        next = argmax(&logits);
    }
    stream
}

/// A quote-back task: the answer repeats most of the prompt verbatim, which
/// is the shape of a code edit or a RAG answer.
const COPY_HEAVY: &str = "Rename the variable `buf` to `scratch` in the function below and \
return the COMPLETE function with nothing else changed.\n\n\
```rust\n\
fn merge_runs(left: &[u32], right: &[u32]) -> Vec<u32> {\n\
    let mut buf = Vec::with_capacity(left.len() + right.len());\n\
    let (mut i, mut j) = (0usize, 0usize);\n\
    while i < left.len() && j < right.len() {\n\
        if left[i] <= right[j] {\n\
            buf.push(left[i]);\n\
            i += 1;\n\
        } else {\n\
            buf.push(right[j]);\n\
            j += 1;\n\
        }\n\
    }\n\
    while i < left.len() {\n\
        buf.push(left[i]);\n\
        i += 1;\n\
    }\n\
    while j < right.len() {\n\
        buf.push(right[j]);\n\
        j += 1;\n\
    }\n\
    buf\n\
}\n\
```";

fn read_prompt(name: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("prompts/real-generation-v1")
        .join(format!("{name}.txt"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

#[test]
#[ignore = "needs a real install via TURBOSPARK_PROBE_INSTALL_DIR"]
fn lookup_replay_on_protocol_cases_and_a_copy_heavy_case() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("TURBOSPARK_PROBE_INSTALL_DIR").expect("TURBOSPARK_PROBE_INSTALL_DIR"),
    );
    let (mut runner, tokenizer) = open_model_runner(&dir, 16).expect("install opens");
    let vocab = tokenizer.vocab_size;

    let cases: Vec<(&str, String)> = vec![
        ("short-explanation", read_prompt("short-explanation")),
        ("medium-review", read_prompt("medium-review")),
        ("long-synthesis", read_prompt("long-synthesis")),
        ("copy-heavy", COPY_HEAVY.to_string()),
    ];

    for (name, text) in cases {
        let rendered = tokenizer
            .apply_chat_template(&[Message::new(Role::User, &text)])
            .expect("chat template renders");
        let prompt = tokenizer.encode(&rendered, false);
        let stream = generate(&mut runner, &prompt, vocab);

        println!(
            "\n== {name}: prompt {} tokens, {} generated",
            prompt.len(),
            stream.len()
        );
        println!(" ngram  draft  fired  proposed/rd  accepted/rd  committed/rd  ceiling-speedup");
        for n in NGRAMS {
            for k in DRAFTS {
                let r = replay(&prompt, &stream, n, k);
                println!(
                    "{n:>6} {k:>6} {:>5.0}% {:>12.2} {:>12.2} {:>13.2} {:>15.3}x",
                    100.0 * r.fired as f64 / r.rounds as f64,
                    r.proposed as f64 / r.fired.max(1) as f64,
                    r.accepted as f64 / r.fired.max(1) as f64,
                    r.committed as f64 / r.rounds as f64,
                    r.speedup(),
                );
            }
        }
    }
    println!(
        "\n'ceiling-speedup' is committed tokens over verify cost in sequential steps, with\n\
         no rollback term and an interpolated c(M) (see verify_cost). Under 1.0x a lookup\n\
         drafter would lose; a few percent over 1.0x is inside the cost model's error.\n"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn propose_prefers_the_most_recent_match() {
        // `5 6` occurs twice before the tail; the newer one is followed by 9.
        let h = [5, 6, 7, 1, 5, 6, 9, 2, 5, 6];
        assert_eq!(propose(&h, 2, 3), vec![9, 2, 5]);
    }

    #[test]
    fn propose_never_matches_the_tail_against_itself() {
        assert!(propose(&[3, 3, 3], 3, 4).is_empty());
        assert!(propose(&[1, 2, 3], 2, 4).is_empty());
    }

    #[test]
    fn propose_clamps_to_what_the_history_holds() {
        let h = [1, 2, 3, 1, 2];
        // Only one token follows the earlier `1 2`... then the tail itself.
        assert_eq!(propose(&h, 2, 8), vec![3, 1, 2]);
        assert!(propose(&h, 2, 0).is_empty());
    }

    #[test]
    fn a_pure_repeat_is_fully_accepted() {
        // prompt ends mid-cycle; the stream continues the cycle exactly.
        let prompt: Vec<i32> = (0..40).map(|i| i % 5).collect();
        let stream: Vec<i32> = (40..80).map(|i| i % 5).collect();
        let r = replay(&prompt, &stream, 3, 4);
        assert_eq!(r.committed, stream.len());
        assert!(r.accepted > 0 && r.accepted <= r.proposed);
        assert!(r.speedup() > 1.0, "{r:?}");
    }

    #[test]
    fn unrelated_text_never_fires_and_costs_one_step_per_token() {
        let prompt: Vec<i32> = (0..30).collect();
        let stream: Vec<i32> = (1000..1030).collect();
        let r = replay(&prompt, &stream, 2, 4);
        assert_eq!(r.fired, 0);
        assert_eq!(r.committed, 30);
        assert!((r.speedup() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_wrong_match_is_charged_for_its_verify() {
        // The history repeats `1 2 3`, but the stream breaks the pattern, so
        // the proposal is rejected and the round still pays a 3-row verify.
        let prompt = vec![1, 2, 3, 1, 2];
        let stream = vec![3, 9, 9, 9];
        let r = replay(&prompt, &stream, 2, 2);
        assert!(r.fired >= 1);
        assert!(r.speedup() < 1.0, "{r:?}");
    }

    #[test]
    fn verify_cost_matches_the_published_points() {
        assert!((verify_cost(1) - 1.0).abs() < 1e-9);
        assert!((verify_cost(3) - 1.71).abs() < 1e-9);
        assert!((verify_cost(9) - 4.53).abs() < 1e-9);
        assert!(verify_cost(4) > 1.71 && verify_cost(4) < 2.86);
    }
}
