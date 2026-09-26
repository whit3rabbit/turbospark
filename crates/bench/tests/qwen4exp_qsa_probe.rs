//! `qwen4_exp` QSA above the indexer budget: the sparse arm against the
//! force-dense diagnostic arm, on the real install.
//!
//! This is an internal implementation comparison, not an upstream quality
//! score. It teacher-forces the protocol's `long-synthesis` prompt
//! (~2,940 tokens under this vocab, past the 2,051-token point where block
//! selection starts dropping blocks) twice through one open runner: once
//! with `set_qsa_force_dense(true)` (every block attended, the dense kernel
//! above budget) and once sparse. At a sample of positions it reports
//! `KL(sparse || dense)` in nats and whether the argmax agrees.
//!
//! How to read it (AGENTS.md Gotcha 38: report the observable, do not
//! invent a threshold): a BROKEN sparse kernel reads as garbage -- KL in the
//! several-nats range and argmax agreement near chance -- while a WORKING
//! one reads as a small, nonzero KL with high argmax agreement, because QSA
//! drops the lowest-scoring 4-token blocks and the checkpoint was trained
//! under exactly that selection. What IS asserted: every logit finite,
//! positions AT OR BELOW the budget bitwise identical between the arms (the
//! below-budget exactness claim, on the real model rather than a fixture),
//! and the arms NOT identical everywhere above it (selection did something).
//!
//! Two sequential passes over ~2,940 tokens: several minutes.
//!
//!   TURBOSPARK_QWEN4EXP_IQ2_XS_INSTALL_DIR=/tmp/qwen4exp-swift-iq2-xs.gturbo \
//!     cargo test -p turbospark-bench --test qwen4exp_qsa_probe --release -- --ignored --nocapture

#![cfg(target_os = "macos")]

use foundation::LogitValue;
use runtime::{LogitProducer, RealForwardRunner};
use tokenizer::{Message, Role};
use turbospark_bench::protocol::{PROTOCOL_CASES, PROTOCOL_EXPERT_CACHE_SLOTS};
use turbospark_bench::real_model::open_model_runner_with_context;

const MAX_CONTEXT: u32 = 4096;

fn install_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("TURBOSPARK_QWEN4EXP_IQ2_XS_INSTALL_DIR")
        .or_else(|| std::env::var_os("TURBOSPARK_QWEN4EXP_INSTALL_DIR"))
        .map(std::path::PathBuf::from)
}

/// `log softmax` in f64, max-subtracted, over the whole vocab.
fn log_softmax(logits: &[LogitValue]) -> Vec<f64> {
    let max = logits
        .iter()
        .fold(f32::NEG_INFINITY, |acc, &v| acc.max(v.to_f32())) as f64;
    assert!(max.is_finite(), "no finite logit");
    let shifted: Vec<f64> = logits.iter().map(|&v| v.to_f32() as f64 - max).collect();
    let lse = shifted.iter().map(|x| x.exp()).sum::<f64>().ln();
    shifted.iter().map(|x| x - lse).collect()
}

fn kl_nats(p_log: &[f64], q_log: &[f64]) -> f64 {
    p_log
        .iter()
        .zip(q_log)
        .map(|(&lp, &lq)| lp.exp() * (lp - lq))
        .sum()
}

fn argmax(logits: &[LogitValue]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
        .map(|(i, _)| i)
        .unwrap()
}

/// Feeds `ids[..len-1]` and keeps the raw logits at `sample` positions.
fn teacher_force(
    runner: &mut RealForwardRunner,
    ids: &[i32],
    sample: &[usize],
    force_dense: bool,
) -> Vec<Vec<LogitValue>> {
    runner.reset();
    runner.set_qsa_force_dense(force_dense);
    let vocab = runner.vocab_size();
    let mut kept = Vec::with_capacity(sample.len());
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    for (position, &token) in ids.iter().take(ids.len() - 1).enumerate() {
        runner
            .produce(token, position, &mut logits)
            .unwrap_or_else(|e| panic!("produce failed at position {position}: {e}"));
        if sample.contains(&position) {
            assert!(
                logits.iter().all(|v| v.to_f32().is_finite()),
                "non-finite logit at position {position} (force_dense {force_dense})"
            );
            kept.push(logits.clone());
        }
    }
    kept
}

#[test]
#[ignore = "needs a real Qwen4Exp install via TURBOSPARK_QWEN4EXP_IQ2_XS_INSTALL_DIR"]
fn sparse_qsa_against_forced_dense_past_the_budget() {
    let Some(dir) = install_dir() else {
        eprintln!("qwen4exp_qsa_probe: TURBOSPARK_QWEN4EXP_INSTALL_DIR is not set; skipping");
        return;
    };
    let arch = repack::peek_manifest_arch(&dir).expect("manifest");
    let ca = &arch.compressed_attention;
    let compress = ca.csa_compress_rate as usize;
    let block_topk = ca.index_top_k as usize;
    // Selection first drops a block when `(position + 1) / compress >
    // block_topk`, i.e. at `visible == (block_topk + 1) * compress`.
    let first_sparse = (block_topk + 1) * compress - 1;

    let (mut runner, tokenizer) =
        open_model_runner_with_context(&dir, PROTOCOL_EXPERT_CACHE_SLOTS, MAX_CONTEXT)
            .expect("real install should open above the indexer budget");
    let long = PROTOCOL_CASES
        .iter()
        .find(|c| c.id == "long-synthesis")
        .expect("protocol has the long case");
    let rendered = tokenizer
        .apply_chat_template(&[Message::new(Role::User, long.content)])
        .expect("chat template renders");
    let ids = tokenizer.encode(&rendered, false);
    eprintln!(
        "qwen4exp_qsa_probe: {} prompt tokens, selection starts dropping blocks at position \
         {first_sparse} (index_top_k {block_topk} x compress {compress})",
        ids.len()
    );
    assert!(
        ids.len() > first_sparse + 64,
        "the long prompt must reach well past the budget to probe anything"
    );

    // Below-budget positions (must be bitwise identical), the first sparse
    // positions, then every 64th position to the end of the prompt.
    let mut sample: Vec<usize> = vec![first_sparse - 12, first_sparse - 1];
    sample.extend(first_sparse..first_sparse + 8);
    sample.extend((first_sparse + 64..ids.len() - 1).step_by(64));

    let dense = teacher_force(&mut runner, &ids, &sample, true);
    let sparse = teacher_force(&mut runner, &ids, &sample, false);
    assert_eq!(dense.len(), sample.len());
    assert_eq!(sparse.len(), sample.len());

    let mut kls = Vec::new();
    let mut agree = 0usize;
    let mut above = 0usize;
    let mut any_differs_above = false;
    for (i, &position) in sample.iter().enumerate() {
        let identical = dense[i]
            .iter()
            .zip(&sparse[i])
            .all(|(a, b)| a.to_bits() == b.to_bits());
        if position < first_sparse {
            assert!(
                identical,
                "position {position} is below the budget, yet the sparse and forced-dense arms \
                 differ: the indexer leaked into the trunk on the real install"
            );
            eprintln!("  position {position:>5}: below budget, bitwise identical");
            continue;
        }
        above += 1;
        any_differs_above |= !identical;
        let kl = kl_nats(&log_softmax(&sparse[i]), &log_softmax(&dense[i]));
        assert!(kl.is_finite(), "KL at position {position} is not finite");
        let same_top = argmax(&sparse[i]) == argmax(&dense[i]);
        agree += same_top as usize;
        kls.push(kl);
        eprintln!(
            "  position {position:>5}: KL(sparse||dense) {kl:.5} nats, argmax {}",
            if same_top { "agrees" } else { "DIFFERS" }
        );
    }
    assert!(
        any_differs_above,
        "no sampled position above the budget differs between the arms: selection dropped \
         nothing, or the dense kernel ran regardless"
    );
    let mut sorted = kls.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mean = kls.iter().sum::<f64>() / kls.len() as f64;
    let median = sorted[sorted.len() / 2];
    let max = *sorted.last().unwrap();
    eprintln!(
        "qwen4exp_qsa_probe: {above} positions above budget: KL mean {mean:.5}, median \
         {median:.5}, max {max:.5} nats; argmax agreement {agree}/{above}"
    );
}
