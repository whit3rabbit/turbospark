#![cfg(target_os = "macos")]
//! How far the default chunked-prefill arm (batched, INT4 GEMMs on the
//! vendored Steel kernel at a 64-row micro-batch) moves the logits away from
//! the per-token reference arm on the REAL Qwen3.8-27B install.
//!
//! The per-token arm is the byte-identical reference against sequential
//! prefill (`docs/BATCHED_PREFILL.md`); Steel rounds each INT4 weight to an
//! FP16 matrix operand and reduces in the matrix units' order, so it is held
//! to a distribution bound instead. For prompts from 64 to 3000 tokens, each
//! prefilled in 128-token chunks the way the CLI and the server do, it
//! compares the final position's next-token distribution: KL(reference ||
//! steel) in nats over the full vocabulary, the worst logit difference, and
//! top-1 and top-5 agreement.
//!
//! The Steel arm runs twice, with the multi-row GQA attention partial (the
//! default) and with the one-row kernel per row, and each is held to the
//! bound.
//!
//! One runner, the reference arm first: the Steel scratch is allocated the
//! first time the batched arm runs, so nothing the reference arm does can
//! see it. Not run by default (needs the ~16 GB install):
//!
//!   TURBOSPARK_QWEN38_INSTALL_DIR=~/models/qwen38-27b.gturbo \
//!     cargo test -p turbospark-bench --test qwen38_prefill_arm_probe --release -- --ignored --nocapture

use foundation::LogitValue;
use runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};
use turbospark_bench::real_model::open_model_runner_with_context;

const CHUNK: usize = 128;
const LENGTHS: [usize; 6] = [64, 200, 500, 1000, 2000, 3000];
/// Mean KL bound in nats. The scalar batched arm's documented drift is
/// 6e-8 to 1.5e-5 nats per row (`families/qwen/prefill.rs`); this is a
/// distribution-level bound for the rounding Steel adds, far below what
/// changes a sampled text's character.
const MEAN_KL_BOUND: f64 = 1e-3;

fn prefill(runner: &mut RealForwardRunner, ids: &[i32], vocab: usize) -> Vec<f64> {
    runner.reset();
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    let mut offset = 0;
    while offset < ids.len() {
        let take = (ids.len() - offset).min(CHUNK);
        runner
            .prefill_chunk(&ids[offset..offset + take], offset, &mut logits)
            .expect("chunked prefill");
        offset += take;
    }
    logits.iter().map(|v| v.to_f32() as f64).collect()
}

fn log_softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = logits.iter().map(|v| (v - max).exp()).sum();
    let log_z = max + sum.ln();
    logits.iter().map(|v| v - log_z).collect()
}

fn top_k(logits: &[f64], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.truncate(k);
    idx
}

#[test]
#[ignore = "needs a real ~16 GB Qwen3.8-27B .gturbo install (TURBOSPARK_QWEN38_INSTALL_DIR)"]
fn steel_prefill_stays_within_the_distribution_bound() {
    let Some(dir) = std::env::var_os("TURBOSPARK_QWEN38_INSTALL_DIR") else {
        eprintln!("TURBOSPARK_QWEN38_INSTALL_DIR is not set; skipping");
        return;
    };
    for var in [
        "TURBOSPARK_BATCHED_GEMV",
        "TURBOSPARK_PREFILL_STEEL",
        "TURBOSPARK_PREFILL_ROWS_ATTN",
    ] {
        assert!(
            std::env::var_os(var).is_none(),
            "{var} is set; this probe chooses the arms itself"
        );
    }
    let (mut runner, tokenizer) =
        open_model_runner_with_context(std::path::Path::new(&dir), 16, 4096).expect("open");
    let vocab = runner.vocab_size();
    let docs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs");
    let mut text = String::new();
    let mut files: Vec<_> = std::fs::read_dir(&docs)
        .expect("docs dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .collect();
    files.sort();
    for f in files.iter().take(12) {
        text.push_str(&std::fs::read_to_string(f).unwrap_or_default());
        text.push_str("\n\n");
    }
    let ids = tokenizer.encode(&text, false);
    assert!(ids.len() >= *LENGTHS.last().unwrap(), "corpus too short");

    // Different slices of the corpus per length, so the cases are not
    // prefixes of one another.
    let prompts: Vec<Vec<i32>> = LENGTHS
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            let start = (i * 997) % (ids.len() - n);
            ids[start..start + n].to_vec()
        })
        .collect();

    runner.set_batched_gemv_prefill(false);
    let reference: Vec<Vec<f64>> = prompts
        .iter()
        .map(|p| prefill(&mut runner, p, vocab))
        .collect();

    // Back to the default: the automatic batched pick with Steel on, with
    // the multi-row attention partial (the default) and without it.
    let mut steel_runner = runner;
    steel_runner.set_batched_gemv_prefill(true);
    steel_runner.set_prefill_steel(true);
    steel_runner.set_prefill_rows_attention(true);
    let steel: Vec<Vec<f64>> = prompts
        .iter()
        .map(|p| prefill(&mut steel_runner, p, vocab))
        .collect();
    steel_runner.set_prefill_rows_attention(false);
    let steel_per_row: Vec<Vec<f64>> = prompts
        .iter()
        .map(|p| prefill(&mut steel_runner, p, vocab))
        .collect();

    for (label, arm) in [
        ("steel, multi-row attention", &steel),
        ("steel, per-row attention", &steel_per_row),
    ] {
        let mean_kl = compare(label, &reference, arm);
        assert!(
            mean_kl < MEAN_KL_BOUND,
            "{label}: mean KL {mean_kl} nats over the {MEAN_KL_BOUND} bound"
        );
    }
}

/// Prints the per-length table for one arm against the reference and
/// returns its mean KL.
fn compare(label: &str, reference: &[Vec<f64>], arm: &[Vec<f64>]) -> f64 {
    let mut kls = Vec::new();
    let mut top1_agree = 0;
    eprintln!("{label}");
    eprintln!("tokens   KL(ref||arm)     max|dlogit|   top1   top5 overlap");
    for ((n, r), s) in LENGTHS.iter().zip(reference).zip(arm) {
        let (lr, ls) = (log_softmax(r), log_softmax(s));
        let kl: f64 = lr.iter().zip(&ls).map(|(a, b)| a.exp() * (a - b)).sum();
        let max_d = r
            .iter()
            .zip(s)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let (t1r, t1s) = (top_k(r, 1)[0], top_k(s, 1)[0]);
        let (t5r, t5s) = (top_k(r, 5), top_k(s, 5));
        let overlap = t5r.iter().filter(|i| t5s.contains(i)).count();
        top1_agree += usize::from(t1r == t1s);
        kls.push(kl);
        eprintln!(
            "{n:>6}   {kl:>14.3e}   {max_d:>11.4}   {:>4}   {overlap}/5",
            t1r == t1s
        );
    }
    let mean_kl = kls.iter().sum::<f64>() / kls.len() as f64;
    eprintln!(
        "mean KL {mean_kl:.3e} nats, top-1 agreement {top1_agree}/{}",
        LENGTHS.len()
    );
    mean_kl
}
