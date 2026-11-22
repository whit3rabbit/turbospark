#![cfg(target_os = "macos")]
//! Gate for the Inkling 2 batched verify and its rollback, on a real install.
//!
//! ```text
//! INKLING2_INSTALL=~/models/inkling2-mini-gturbo cargo test --release \
//!     -p turbospark-runtime --test real_forward_inkling2_verify -- --ignored --nocapture
//! ```
//!
//! Reference: one sequential greedy pass (`produce`, one token per call) over
//! a prose prompt and its continuation, every position's logits kept. Then a
//! fresh session interleaves sequential steps, 8-row verifies and retaining
//! rollbacks, and every logit row it produces must equal the reference's row
//! for that position BIT FOR BIT. Blocks sit early, across the sliding-window
//! ring's wrap (rows 1031 = window + slack), and right after a rollback, so
//! KV slack, conv-tail restores and the cursor are all on the path: a wrong
//! restore shows up as a mismatch in the NEXT pass, not in the verify itself.

use std::path::PathBuf;

use half::f16;
use turbospark_runtime::{LogitProducer, RealForwardRunner, SpeculativeProducer};

fn argmax(row: &[f16]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
        .unwrap()
        .0
}

struct Diff {
    rows: usize,
    exact: usize,
    top1: usize,
    max_abs: f32,
}

impl Diff {
    fn new() -> Self {
        Self {
            rows: 0,
            exact: 0,
            top1: 0,
            max_abs: 0.0,
        }
    }
    fn add(&mut self, got: &[f16], want: &[f16]) {
        self.rows += 1;
        if got
            .iter()
            .zip(want)
            .all(|(a, b)| a.to_bits() == b.to_bits())
        {
            self.exact += 1;
        }
        if argmax(got) == argmax(want) {
            self.top1 += 1;
        }
        for (a, b) in got.iter().zip(want) {
            let (a, b) = (a.to_f32(), b.to_f32());
            if a.is_finite() && b.is_finite() {
                self.max_abs = self.max_abs.max((a - b).abs());
            }
        }
    }
}

#[test]
#[ignore = "needs INKLING2_INSTALL (a real Inkling 2 install, resident experts)"]
fn verify_and_rollback_match_sequential_decode_bit_for_bit() {
    let install = PathBuf::from(std::env::var("INKLING2_INSTALL").expect("INKLING2_INSTALL"));
    let arch = model_io::inkling2_mini();
    let vocab = arch.vocab_size as usize;
    let tok = tokenizer::MfTokenizer::load_from_dir(&install).expect("tokenizer");
    // About 1,000 tokens of prose from this repository, so the blocks below
    // cross the window ring's wrap.
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/INKLING2_PHASE0.md"),
    )
    .expect("prose");
    let mut ids = tok.encode(&text, false);
    let prompt_len = 1000usize.min(ids.len());
    ids.truncate(prompt_len);
    let total = 1060usize;
    let open = || {
        RealForwardRunner::open_with_residency(
            &install,
            arch.clone(),
            2048,
            turbospark_runtime::ExpertCacheSlots::Auto,
            turbospark_runtime::DraftPolicies::off(),
            turbospark_runtime::SteeringPolicy::off(),
            1,
            turbospark_runtime::KvQuant::Off,
            turbospark_runtime::ExpertResidency::Resident,
        )
        .expect("open resident")
    };

    // Reference: sequential, greedy past the prompt.
    let mut runner = open();
    let mut want: Vec<Vec<f16>> = Vec::with_capacity(total);
    let mut row = vec![f16::ZERO; vocab];
    let t = std::time::Instant::now();
    for pos in 0..total {
        runner.produce(ids[pos], pos, &mut row).expect("produce");
        want.push(row.clone());
        if pos + 1 >= ids.len() {
            ids.push(argmax(&row) as i32);
        }
    }
    eprintln!(
        "reference: {total} positions in {:.1} s",
        t.elapsed().as_secs_f32()
    );

    // Candidate: the same session again, through verifies and rollbacks.
    runner.reset();
    let mut d_verify = Diff::new();
    let mut d_after = Diff::new();
    let mut pos = 0usize;
    // (block start, rows kept), in order. Each start equals where the
    // previous step left the cursor.
    let plan: &[(usize, usize)] = &[
        (16, 3),
        (19, 8),
        (27, 1),
        (1020, 8),
        (1028, 5),
        (1033, 2),
        (1035, 8),
    ];
    let mut block = vec![f16::ZERO; 8 * vocab];
    for &(start, keep) in plan {
        while pos < start {
            runner.produce(ids[pos], pos, &mut row).expect("produce");
            d_after.add(&row, &want[pos]);
            pos += 1;
        }
        let point = runner.checkpoint_round();
        SpeculativeProducer::verify(&mut runner, &ids[start..start + 8], start, &mut block)
            .expect("verify");
        for r in 0..8 {
            d_verify.add(&block[r * vocab..(r + 1) * vocab], &want[start + r]);
        }
        if keep < 8 {
            runner.rollback_retaining(&point, keep).expect("rollback");
        }
        pos = start + keep;
    }
    while pos < total {
        runner.produce(ids[pos], pos, &mut row).expect("produce");
        d_after.add(&row, &want[pos]);
        pos += 1;
    }
    eprintln!(
        "verify rows: {} of {} bit-identical, top-1 {} of {}, max |diff| {}",
        d_verify.exact, d_verify.rows, d_verify.top1, d_verify.rows, d_verify.max_abs
    );
    eprintln!(
        "sequential rows around them: {} of {} bit-identical, top-1 {} of {}, max |diff| {}",
        d_after.exact, d_after.rows, d_after.top1, d_after.rows, d_after.max_abs
    );
    assert_eq!(
        d_verify.exact, d_verify.rows,
        "verify rows differ from sequential decode"
    );
    assert_eq!(
        d_after.exact, d_after.rows,
        "a rollback left state that changes later decode"
    );
}
