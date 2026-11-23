//! KL gate for the opt-in matrix-unit prefill GEMM
//! (`TURBOSPARK_INKLING2_PREFILL_MMA=1`), on a real install.
//!
//! ```text
//! INKLING2_INSTALL=~/models/inkling2-mini-gturbo cargo test --release \
//!     -p turbospark-runtime --test real_forward_inkling2_prefill_mma -- --ignored --nocapture
//! ```
//!
//! Its own test binary: the switch is read once per process, so this file
//! turns it on before anything opens and holds exactly one test.
//!
//! Reference: the prompt through `produce` one token at a time (the
//! bit-exact path), then 64 greedy decode tokens. Candidate: the same prompt
//! through `prefill_chunk` with the matrix units, then the reference's 64
//! tokens teacher-forced. Per row: KL(reference || candidate) over the real
//! vocabulary and top-1 agreement, bounded relative to a shipped deviation
//! measured in the same run: the streamed host-routed path against the
//! resident one (`docs/INKLING2_PHASE0.md`). Never widened to make a run pass. Separately, the panel kernel's row independence makes the
//! whole prefill independent of chunking, which is asserted BIT FOR BIT: two
//! different chunkings must leave the same last-row logits.

use std::path::PathBuf;

use half::f16;
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};

/// The bounds are RELATIVE to a floor measured in the same run (see the
/// floor below), so prompt and host do not move them. Measured 2026-11-23 on
/// the BF16 install, 1,100-token prompt, 65 rows: matrix units mean KL
/// 2.49e-3, max 3.17e-2, top-1 63/65; floor mean 1.93e-3, max 3.12e-2, top-1
/// 62/65. Headroom 2x on KL and one top-1 row.
const KL_HEADROOM: f64 = 2.0;

fn argmax(row: &[f16]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
        .unwrap()
        .0
}

fn log_softmax(row: &[f16]) -> Vec<f64> {
    let m = row
        .iter()
        .map(|v| v.to_f64())
        .fold(f64::NEG_INFINITY, f64::max);
    let z: f64 = row.iter().map(|v| (v.to_f64() - m).exp()).sum();
    let lz = z.ln() + m;
    row.iter().map(|v| v.to_f64() - lz).collect()
}

fn kl(p_row: &[f16], q_row: &[f16]) -> f64 {
    let (lp, lq) = (log_softmax(p_row), log_softmax(q_row));
    lp.iter()
        .zip(&lq)
        .map(|(a, b)| {
            if a.is_finite() {
                a.exp() * (a - b)
            } else {
                0.0
            }
        })
        .sum::<f64>()
        .max(0.0)
}

/// Mean and max KL(want || got) per row, top-1 agreement, and the KLs.
fn compare(want: &[Vec<f16>], got: &[Vec<f16>], real: usize) -> (f64, f64, usize, Vec<f64>) {
    let kls: Vec<f64> = want
        .iter()
        .zip(got)
        .map(|(w, g)| kl(&w[..real], &g[..real]))
        .collect();
    let top1 = want
        .iter()
        .zip(got)
        .filter(|(w, g)| argmax(&w[..real]) == argmax(&g[..real]))
        .count();
    let mean = kls.iter().sum::<f64>() / kls.len() as f64;
    let max = kls.iter().cloned().fold(0.0, f64::max);
    (mean, max, top1, kls)
}

fn p95(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[(s.len() * 95 / 100).min(s.len() - 1)]
}

#[test]
#[ignore = "needs INKLING2_INSTALL (a real Inkling 2 install, resident experts)"]
fn mma_prefill_stays_within_the_kl_bound_and_is_chunking_independent() {
    std::env::set_var("TURBOSPARK_INKLING2_PREFILL_MMA", "1");
    let install = PathBuf::from(std::env::var("INKLING2_INSTALL").expect("INKLING2_INSTALL"));
    let arch = model_io::inkling2_mini();
    let vocab = arch.vocab_size as usize;
    let tok = tokenizer::MfTokenizer::load_from_dir(&install).expect("tokenizer");
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/INKLING2_PHASE0.md"),
    )
    .expect("prose");
    let mut ids = tok.encode(&text, false);
    let prompt_len = 1100usize;
    assert!(ids.len() >= prompt_len);
    ids.truncate(prompt_len);
    let real = 200_058usize.min(vocab);
    let decode = 64usize;
    let mut runner = RealForwardRunner::open_with_residency(
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
    .expect("open resident");
    assert!(runner.supports_chunked_prefill());

    // Reference: sequential prefill and greedy decode.
    let mut row = vec![f16::ZERO; vocab];
    for (pos, &id) in ids.iter().enumerate() {
        runner.produce(id, pos, &mut row).expect("produce");
    }
    let mut want = vec![row.clone()];
    let mut stream = Vec::with_capacity(decode);
    for i in 0..decode {
        let next = argmax(&row) as i32;
        stream.push(next);
        runner
            .produce(next, prompt_len + i, &mut row)
            .expect("decode");
        want.push(row.clone());
    }

    // Candidate: matrix-unit chunked prefill, teacher-forced decode.
    let chunked = |runner: &mut RealForwardRunner, chunks: &[usize], row: &mut Vec<f16>| {
        runner.reset();
        let mut at = 0usize;
        for &n in chunks {
            runner
                .prefill_chunk(&ids[at..at + n], at, row)
                .expect("prefill_chunk");
            at += n;
        }
        assert_eq!(at, prompt_len);
    };
    chunked(&mut runner, &[13, 512, 300, 275], &mut row);
    let mut got = vec![row.clone()];
    for (i, &t) in stream.iter().enumerate() {
        runner.produce(t, prompt_len + i, &mut row).expect("decode");
        got.push(row.clone());
    }
    let (mean, max, top1, kls) = compare(&want, &got, real);
    eprintln!(
        "mma prefill vs sequential over {} rows (last prompt row + {decode} decode): \
         mean KL {mean:.3e}, max KL {max:.3e}, p95 {:.3e} (last prompt row {:.3e}), \
         top-1 {top1}/{}",
        kls.len(),
        p95(&kls),
        kls[0],
        kls.len()
    );

    // Chunking independence, bit for bit.
    let first_last = got[0].clone();
    chunked(&mut runner, &[1100], &mut row);
    let one = row.clone();
    chunked(&mut runner, &[31, 33, 1, 1035], &mut row);
    assert!(
        one.iter()
            .zip(&first_last)
            .all(|(a, b)| a.to_bits() == b.to_bits())
            && row
                .iter()
                .zip(&first_last)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
        "matrix-unit prefill depends on how the prompt is chunked"
    );

    // The floor: a deviation this project already ships. Resident experts
    // route on the GPU, the streamed path on the host, and the two round
    // `exp` differently; resident is the default where it fits. Same prompt,
    // same teacher-forced tokens, sequential on both.
    drop(runner);
    let mut streamed = RealForwardRunner::open_with_residency(
        &install,
        arch.clone(),
        2048,
        turbospark_runtime::ExpertCacheSlots::Auto,
        turbospark_runtime::DraftPolicies::off(),
        turbospark_runtime::SteeringPolicy::off(),
        1,
        turbospark_runtime::KvQuant::Off,
        turbospark_runtime::ExpertResidency::Streamed,
    )
    .expect("open streamed");
    for (pos, &id) in ids.iter().enumerate() {
        streamed.produce(id, pos, &mut row).expect("produce");
    }
    let mut floor = vec![row.clone()];
    for (i, &t) in stream.iter().enumerate() {
        streamed
            .produce(t, prompt_len + i, &mut row)
            .expect("decode");
        floor.push(row.clone());
    }
    let (f_mean, f_max, f_top1, f_kls) = compare(&want, &floor, real);
    eprintln!(
        "floor (streamed host-routed vs resident, both sequential): mean KL {f_mean:.3e}, \
         max KL {f_max:.3e}, p95 {:.3e}, top-1 {f_top1}/{}",
        p95(&f_kls),
        f_kls.len()
    );

    assert!(
        mean <= f_mean * KL_HEADROOM,
        "mean KL {mean:.3e} over {KL_HEADROOM}x the floor's {f_mean:.3e}"
    );
    assert!(
        max <= f_max * KL_HEADROOM,
        "max KL {max:.3e} over {KL_HEADROOM}x the floor's {f_max:.3e}"
    );
    assert!(
        top1 + 1 >= f_top1,
        "top-1 {top1} more than one row below the floor's {f_top1}"
    );
}
