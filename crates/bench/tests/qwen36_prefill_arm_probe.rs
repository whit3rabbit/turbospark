#![cfg(target_os = "macos")]
//! How far the default chunked-prefill arm of the qwen MoE driver (batched,
//! INT4 GEMMs on the vendored Steel kernel at a 64-row micro-batch) moves the
//! logits away from its per-token reference arm on the REAL Qwen 3.6 35B-A3B
//! install. The MoE sibling of `qwen38_prefill_arm_probe.rs`, with its corpus,
//! lengths and chunking, and its OWN bound (below).
//!
//! The per-token arm (`TURBOSPARK_BATCHED_GEMV=0`) makes the sequential
//! flow's dispatches and is byte-identical to sequential prefill. The batched
//! arm is not: its M-row GDN and attention blocks reduce in a different order,
//! and Steel rounds each INT4 weight to an FP16 operand. So it is held to a
//! distribution bound. For prompts from 64 to 3000 tokens, each prefilled in
//! 128-token chunks the way the CLI and the server do, this compares the
//! final position's next-token distribution: KL(reference || batched) in nats
//! over the full vocabulary, the worst logit difference, and top-1 and top-5
//! agreement. The routed experts run the same per-token loop in both arms, so
//! everything measured here comes from the resident half.
//!
//! Two arms are measured: the batched arm with the scalar GEMMs (bit-exact
//! against the per-token GEMVs, so its distance is the M-row GDN and
//! attention blocks alone) and with Steel (the default, adding Steel's
//! rounding). The Steel choice is fixed when the batched scratch is first
//! allocated, so the scalar arm shares the reference's runner, which is then
//! dropped before a second runner opens for Steel: one model is open at a
//! time. Not run by default (needs the ~17 GiB install):
//!
//!   TURBOSPARK_QWEN36_INSTALL_DIR=~/.turbospark/models/qwen36.gturbo \
//!     cargo test -p turbospark-bench --test qwen36_prefill_arm_probe --release -- --ignored --nocapture

use foundation::LogitValue;
use runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};
use turbospark_bench::real_model::open_model_runner_with_context;

const CHUNK: usize = 128;
const LENGTHS: [usize; 6] = [64, 200, 500, 1000, 2000, 3000];
/// Mean KL bound in nats: a WIRING-REGRESSION sentinel for this family, not
/// the dense family's 1e-3, which this arm does not meet.
///
/// Measured 2026-10-15 on the real install: 2.55e-3 (scalar) and 3.94e-3
/// (Steel), with single lengths from 2.5e-5 to 1.4e-2 and no trend with length.
/// The mechanism is routing, not arithmetic: on a 300-token prompt the two
/// arms picked identical top-8 experts at every layer for tokens 0-14, and
/// the first difference (token 15, layer 6) swapped the 8th-ranked expert
/// across a 4.6e-4 router-logit margin, against an arm-to-arm router drift of
/// at most 2.3e-3 (`docs/BATCHED_PREFILL.md`, "The qwen MoE batched arm").
/// Discrete top-k selection turns the M-row blocks' reduction-order floor
/// (6e-8 to 1.5e-5 nats on the dense family) into expert swaps, so the dense
/// bound does not transfer. Making this the default anyway was a recorded
/// decision; this bound catches what a wrong row, offset or seed produces
/// (KL of order 1, top-1 lost), with headroom over the measured spread.
const MEAN_KL_BOUND: f64 = 1e-2;
/// Top-1 must agree on at least this many of the six lengths. Measured 6/6
/// (scalar) and 5/6 (Steel); a near-tie flip at the final position is the
/// same mechanism as above, a wiring bug loses most of them.
const MIN_TOP1_AGREEMENT: usize = LENGTHS.len() / 2;

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
#[ignore = "needs a real ~17 GiB Qwen 3.6 35B-A3B .gturbo install (TURBOSPARK_QWEN36_INSTALL_DIR)"]
fn batched_moe_prefill_stays_within_the_distribution_bound() {
    let Some(dir) = std::env::var_os("TURBOSPARK_QWEN36_INSTALL_DIR") else {
        eprintln!("TURBOSPARK_QWEN36_INSTALL_DIR is not set; skipping");
        return;
    };
    for var in [
        "TURBOSPARK_BATCHED_GEMV",
        "TURBOSPARK_PREFILL_STEEL",
        "TURBOSPARK_QWEN_MOE_CHUNKED",
    ] {
        assert!(
            std::env::var_os(var).is_none(),
            "{var} is set; this probe chooses the arms itself"
        );
    }
    let install = std::path::Path::new(&dir);
    let (mut runner, tokenizer) = open_model_runner_with_context(install, 16, 4096).expect("open");
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

    runner.set_batched_gemv_prefill(true);
    runner.set_prefill_steel(false);
    let scalar: Vec<Vec<f64>> = prompts
        .iter()
        .map(|p| prefill(&mut runner, p, vocab))
        .collect();
    drop(runner);

    // The default arm: batched, Steel on, on a fresh runner.
    let (mut runner, _) = open_model_runner_with_context(install, 16, 4096).expect("reopen");
    runner.set_batched_gemv_prefill(true);
    runner.set_prefill_steel(true);
    let steel: Vec<Vec<f64>> = prompts
        .iter()
        .map(|p| prefill(&mut runner, p, vocab))
        .collect();

    let results: Vec<(&str, (f64, usize))> =
        [("batched, scalar", &scalar), ("batched, steel", &steel)]
            .into_iter()
            .map(|(label, arm)| (label, compare(label, &reference, arm)))
            .collect();
    for (label, (mean_kl, top1)) in results {
        assert!(
            mean_kl < MEAN_KL_BOUND,
            "{label}: mean KL {mean_kl} nats over the {MEAN_KL_BOUND} bound"
        );
        assert!(
            top1 >= MIN_TOP1_AGREEMENT,
            "{label}: top-1 agreed on {top1} of {} lengths, under {MIN_TOP1_AGREEMENT}",
            LENGTHS.len()
        );
    }
}

/// Prints the per-length table for one arm against the reference and
/// returns its mean KL and how many lengths agree on top-1.
fn compare(label: &str, reference: &[Vec<f64>], arm: &[Vec<f64>]) -> (f64, usize) {
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
    (mean_kl, top1_agree)
}
