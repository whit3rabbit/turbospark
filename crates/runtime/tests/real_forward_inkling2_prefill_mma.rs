//! KL gate for the matrix-unit prefill (the default prefill path of a
//! resident native-expert install; `TURBOSPARK_INKLING2_PREFILL_MMA=0` turns
//! it off), on a real install.
//!
//! ```text
//! INKLING2_INSTALL=~/models/inkling2-mini-gturbo cargo test --release \
//!     -p turbospark-runtime --test real_forward_inkling2_prefill_mma -- --ignored --nocapture
//! ```
//!
//! Its own test binary: the switches are read once per process, so this file
//! sets them before anything opens and holds exactly one test.
//!
//! Design (2026-11-23, replacing a one-prompt mean/max bound that one spike
//! row could move 3x between equally accurate kernels):
//! - Four prompts frozen as fixtures (`tests/fixtures/inkling2_kl/`, two
//!   prose, two code), 1,100 tokens each, so the prompts never move.
//! - Reference: the prompt through `produce` one token at a time (the
//!   bit-exact path), then 64 greedy decode tokens. Candidate: the prompt
//!   through `prefill_chunk` with the matrix units, then the reference's
//!   tokens teacher-forced. Floor: the same tokens sequentially on the
//!   streamed host-routed path, a deviation this project already ships
//!   (resident routes on the GPU and rounds `exp` differently).
//! - Statistics POOLED over all prompts' rows (4 x 65) and compared with the
//!   floor's pooled rows at the median and the 90th percentile, robust to a
//!   single chaotic row; pooled top-1 agreement; and each prompt's last
//!   prompt row, the direct measure of the prefill itself, against the
//!   floor's pooled p90.
//! - Chunking independence, bit for bit, on the first prompt.
//!
//! The bounds are below; their measurements are in `docs/INKLING2_PHASE0.md`.
//! Never widened to make a run pass.

use std::path::PathBuf;

use half::f16;
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};

/// Pooled median and p90 of the candidate's per-row KL, each at most this
/// multiple of the floor's.
const QUANTILE_HEADROOM: f64 = 3.0;
// A prompt's last prompt row (the prefill's own output): its KL at most
// QUANTILE_HEADROOM x the floor's POOLED p90. Not an absolute value: the
// first design's 1e-3 nats failed on a prompt whose last position is high
// entropy, where the shipped floor itself read 1.48e-3 (2026-11-23).
/// Pooled top-1 agreement at most this many rows below the floor's.
const TOP1_SLACK: usize = 2;

const PROMPTS: &[&str] = &[
    "prose_inkling2_notes.txt",
    "prose_lamq.txt",
    "code_inkling2_reference.rs.txt",
    "code_moe_shader.metal.txt",
];
const PROMPT_LEN: usize = 1100;
const DECODE: usize = 64;

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

fn quantile(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[((s.len() - 1) as f64 * q).round() as usize]
}

fn open(
    install: &std::path::Path,
    residency: turbospark_runtime::ExpertResidency,
) -> RealForwardRunner {
    RealForwardRunner::open_with_residency(
        install,
        model_io::inkling2_mini(),
        2048,
        turbospark_runtime::ExpertCacheSlots::Auto,
        turbospark_runtime::DraftPolicies::off(),
        turbospark_runtime::SteeringPolicy::off(),
        1,
        turbospark_runtime::KvQuant::Off,
        residency,
    )
    .expect("open")
}

/// One prompt: its ids, the greedy tokens, the reference rows (last prompt
/// row then the decode rows) and the candidate's rows.
struct PromptRun {
    name: &'static str,
    ids: Vec<i32>,
    stream: Vec<i32>,
    want: Vec<Vec<f16>>,
    got: Vec<Vec<f16>>,
}

/// `ids` one token at a time, then `forced` teacher-forced: the last prompt
/// row and every decode row.
fn sequential(
    runner: &mut RealForwardRunner,
    ids: &[i32],
    forced: &[i32],
    vocab: usize,
) -> Vec<Vec<f16>> {
    let mut row = vec![f16::ZERO; vocab];
    for (pos, &id) in ids.iter().enumerate() {
        runner.produce(id, pos, &mut row).expect("produce");
    }
    let mut rows = vec![row.clone()];
    for (i, &t) in forced.iter().enumerate() {
        runner.produce(t, ids.len() + i, &mut row).expect("decode");
        rows.push(row.clone());
    }
    rows
}

fn chunked(runner: &mut RealForwardRunner, ids: &[i32], chunks: &[usize], row: &mut [f16]) {
    runner.reset();
    let mut at = 0usize;
    for &n in chunks {
        runner
            .prefill_chunk(&ids[at..at + n], at, row)
            .expect("prefill_chunk");
        at += n;
    }
    assert_eq!(at, ids.len());
}

#[test]
#[ignore = "needs INKLING2_INSTALL (a real Inkling 2 install, resident experts)"]
fn mma_prefill_stays_within_the_pooled_kl_bound_and_is_chunking_independent() {
    // The default; set explicitly so an inherited `=0` cannot turn this gate
    // into a comparison of the exact path with itself.
    std::env::set_var("TURBOSPARK_INKLING2_PREFILL_MMA", "1");
    let install = PathBuf::from(std::env::var("INKLING2_INSTALL").expect("INKLING2_INSTALL"));
    let vocab = model_io::inkling2_mini().vocab_size as usize;
    let real = 200_058usize.min(vocab);
    let tok = tokenizer::MfTokenizer::load_from_dir(&install).expect("tokenizer");
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inkling2_kl");

    // Reference and candidate on the resident runner.
    let mut runner = open(&install, turbospark_runtime::ExpertResidency::Resident);
    assert!(runner.supports_chunked_prefill());
    let mut runs = Vec::new();
    for &name in PROMPTS {
        let text = std::fs::read_to_string(dir.join(name)).expect("fixture");
        let mut ids = tok.encode(&text, false);
        assert!(ids.len() >= PROMPT_LEN, "{name}: {} tokens", ids.len());
        ids.truncate(PROMPT_LEN);
        runner.reset();
        // Greedy reference: the forced tokens are its own argmaxes.
        let mut row = vec![f16::ZERO; vocab];
        for (pos, &id) in ids.iter().enumerate() {
            runner.produce(id, pos, &mut row).expect("produce");
        }
        let mut want = vec![row.clone()];
        let mut stream = Vec::with_capacity(DECODE);
        for i in 0..DECODE {
            let next = argmax(&row) as i32;
            stream.push(next);
            runner
                .produce(next, PROMPT_LEN + i, &mut row)
                .expect("decode");
            want.push(row.clone());
        }
        chunked(&mut runner, &ids, &[13, 512, 300, 275], &mut row);
        let mut got = vec![row.clone()];
        for (i, &t) in stream.iter().enumerate() {
            runner.produce(t, PROMPT_LEN + i, &mut row).expect("decode");
            got.push(row.clone());
        }
        runs.push(PromptRun {
            name,
            ids,
            stream,
            want,
            got,
        });
    }
    // Chunking independence, bit for bit, on the first prompt.
    {
        let r = &runs[0];
        let mut row = vec![f16::ZERO; vocab];
        for chunks in [&[PROMPT_LEN][..], &[31, 33, 1, 1035][..]] {
            chunked(&mut runner, &r.ids, chunks, &mut row);
            assert!(
                row.iter()
                    .zip(&r.got[0])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "matrix-unit prefill depends on how the prompt is chunked ({chunks:?})"
            );
        }
    }
    drop(runner);

    // Floor on the streamed runner, the same teacher-forced tokens.
    let mut streamed = open(&install, turbospark_runtime::ExpertResidency::Streamed);
    let (mut cand, mut floor) = (Vec::new(), Vec::new());
    let (mut top1_c, mut top1_f, mut rows) = (0usize, 0usize, 0usize);
    let mut last_rows = Vec::new();
    let kls = |a: &[Vec<f16>], b: &[Vec<f16>]| -> Vec<f64> {
        a.iter()
            .zip(b)
            .map(|(w, g)| kl(&w[..real], &g[..real]))
            .collect()
    };
    let top1 = |a: &[Vec<f16>], b: &[Vec<f16>]| -> usize {
        a.iter()
            .zip(b)
            .filter(|(w, g)| argmax(&w[..real]) == argmax(&g[..real]))
            .count()
    };
    for r in &runs {
        streamed.reset();
        let fl = sequential(&mut streamed, &r.ids, &r.stream, vocab);
        let (kc, kf) = (kls(&r.want, &r.got), kls(&r.want, &fl));
        let (tc, tf) = (top1(&r.want, &r.got), top1(&r.want, &fl));
        eprintln!(
            "{}: candidate median {:.3e} p90 {:.3e} max {:.3e} last-row {:.3e} top-1 {tc}/{}; \
             floor median {:.3e} p90 {:.3e} max {:.3e} last-row {:.3e} top-1 {tf}/{}",
            r.name,
            quantile(&kc, 0.5),
            quantile(&kc, 0.9),
            quantile(&kc, 1.0),
            kc[0],
            kc.len(),
            quantile(&kf, 0.5),
            quantile(&kf, 0.9),
            quantile(&kf, 1.0),
            kf[0],
            kf.len()
        );
        last_rows.push((r.name, kc[0]));
        top1_c += tc;
        top1_f += tf;
        rows += kc.len();
        cand.extend(kc);
        floor.extend(kf);
    }
    let (cm, c9) = (quantile(&cand, 0.5), quantile(&cand, 0.9));
    let (fm, f9) = (quantile(&floor, 0.5), quantile(&floor, 0.9));
    eprintln!(
        "pooled over {rows} rows: candidate median {cm:.3e} p90 {c9:.3e} top-1 {top1_c}; floor \
         median {fm:.3e} p90 {f9:.3e} top-1 {top1_f}; ratios {:.2}x, {:.2}x",
        cm / fm,
        c9 / f9
    );
    assert!(
        cm <= fm * QUANTILE_HEADROOM,
        "pooled median KL {cm:.3e} over {QUANTILE_HEADROOM}x the floor's {fm:.3e}"
    );
    assert!(
        c9 <= f9 * QUANTILE_HEADROOM,
        "pooled p90 KL {c9:.3e} over {QUANTILE_HEADROOM}x the floor's {f9:.3e}"
    );
    assert!(
        top1_c + TOP1_SLACK >= top1_f,
        "pooled top-1 {top1_c} more than {TOP1_SLACK} rows below the floor's {top1_f}"
    );
    for (name, k) in last_rows {
        assert!(
            k <= f9 * QUANTILE_HEADROOM,
            "{name}: last prompt row KL {k:.3e} over {QUANTILE_HEADROOM}x the floor's pooled \
             p90 {f9:.3e}"
        );
    }
}
