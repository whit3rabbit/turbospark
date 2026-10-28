#![cfg(target_os = "macos")]
//! How far the default `lfm3` chunked-prefill arm (Steel INT4 GEMMs from 32
//! rows, multi-row GQA attention on the full layers) moves the logits away
//! from sequential prefill on the REAL LFM3-1.7B MLX 4-bit install.
//!
//! The reference is the driver's exact arm (`TURBOSPARK_PREFILL_STEEL=0`,
//! `TURBOSPARK_PREFILL_ROWS_ATTN=0`), which is bit-identical to one `produce`
//! per token; the probe re-checks that on its two shortest prompts before
//! using it. For prompts from 64 to 3000 tokens, each prefilled in 128-token
//! chunks the way the CLI and the server do, it compares the final position's
//! next-token distribution: KL(reference || arm) in nats over the vocabulary,
//! the worst logit difference, and top-1 and top-5 agreement. Steel runs with
//! the multi-row attention (the default) and with the one-row kernel per row.
//!
//!   TURBOSPARK_LFM3_INSTALL_DIR=~/models/lfm3-1.7b-mlx-4bit \
//!     cargo test -p turbospark-bench --test lfm3_prefill_arm_probe --release -- --ignored --nocapture
mod lfm3_common;

use foundation::LogitValue;
use runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};
use turbospark_bench::logit_compare;
use turbospark_bench::real_model::open_model_runner_with_context;

const CHUNK: usize = 128;
const LENGTHS: [usize; 6] = [64, 200, 500, 1000, 2000, 3000];
/// Mean KL bound in nats, the qwen38 probe's: a distribution-level bound
/// for the rounding Steel adds.
const MEAN_KL_BOUND: f64 = 1e-3;

fn bits(logits: &[LogitValue]) -> Vec<u16> {
    logits.iter().map(|v| v.to_bits()).collect()
}

fn prefill(runner: &mut RealForwardRunner, ids: &[i32], vocab: usize) -> Vec<LogitValue> {
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
    logits
}

fn sequential(runner: &mut RealForwardRunner, ids: &[i32], vocab: usize) -> Vec<LogitValue> {
    runner.reset();
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    for (i, &t) in ids.iter().enumerate() {
        runner.produce(t, i, &mut logits).expect("produce");
    }
    logits
}

fn as_f64(logits: &[LogitValue]) -> Vec<f64> {
    logits.iter().map(|v| v.to_f32() as f64).collect()
}

#[test]
#[ignore = "needs the pinned LFM3-1.7B MLX 4-bit install via TURBOSPARK_LFM3_INSTALL_DIR"]
fn lfm3_steel_prefill_stays_within_the_distribution_bound() {
    for var in [
        "TURBOSPARK_LFM3_CHUNKED_PREFILL",
        "TURBOSPARK_PREFILL_STEEL",
        "TURBOSPARK_PREFILL_ROWS_ATTN",
    ] {
        assert!(
            std::env::var_os(var).is_none(),
            "{var} is set; this probe chooses the arms itself"
        );
    }
    let dir = lfm3_common::install_dir();
    let (mut runner, tokenizer) = open_model_runner_with_context(&dir, 16, 4096).expect("open");
    assert!(runner.supports_chunked_prefill());
    let vocab = runner.vocab_size();
    let docs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs");
    let mut files: Vec<_> = std::fs::read_dir(&docs)
        .expect("docs dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .collect();
    files.sort();
    let mut text = String::new();
    for f in files.iter().take(12) {
        text.push_str(&std::fs::read_to_string(f).unwrap_or_default());
        text.push_str("\n\n");
    }
    let ids = tokenizer.encode(&text, false);
    assert!(ids.len() >= *LENGTHS.last().unwrap(), "corpus too short");
    let prompts: Vec<Vec<i32>> = LENGTHS
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            let start = (i * 997) % (ids.len() - n);
            ids[start..start + n].to_vec()
        })
        .collect();

    runner.set_prefill_steel(false);
    runner.set_prefill_rows_attention(false);
    let reference: Vec<Vec<LogitValue>> = prompts
        .iter()
        .map(|p| prefill(&mut runner, p, vocab))
        .collect();
    for (p, r) in prompts.iter().zip(&reference).take(2) {
        assert_eq!(
            bits(&sequential(&mut runner, p, vocab)),
            bits(r),
            "the exact arm drifted from sequential prefill at {} tokens",
            p.len()
        );
    }

    runner.set_prefill_steel(true);
    runner.set_prefill_rows_attention(true);
    let steel: Vec<Vec<LogitValue>> = prompts
        .iter()
        .map(|p| prefill(&mut runner, p, vocab))
        .collect();
    runner.set_prefill_rows_attention(false);
    let steel_per_row: Vec<Vec<LogitValue>> = prompts
        .iter()
        .map(|p| prefill(&mut runner, p, vocab))
        .collect();

    let reference: Vec<Vec<f64>> = reference.iter().map(|r| as_f64(r)).collect();
    for (label, arm) in [
        ("steel, multi-row attention", &steel),
        ("steel, per-row attention", &steel_per_row),
    ] {
        let arm: Vec<Vec<f64>> = arm.iter().map(|r| as_f64(r)).collect();
        let mean_kl = compare(label, &reference, &arm);
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
        let c = logit_compare::compare(r, s);
        top1_agree += usize::from(c.top1_agree);
        kls.push(c.kl);
        eprintln!(
            "{n:>6}   {:>14.3e}   {:>11.4}   {:>4}   {}/5",
            c.kl, c.max_abs_dlogit, c.top1_agree, c.top5_overlap
        );
    }
    let mean_kl = kls.iter().sum::<f64>() / kls.len() as f64;
    eprintln!(
        "mean KL {mean_kl:.3e} nats, top-1 agreement {top1_agree}/{}",
        LENGTHS.len()
    );
    mean_kl
}
