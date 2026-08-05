#![cfg(target_os = "macos")]
//! End-to-end proof of the real-checkpoint Gemma 4 decode flow: builds a
//! tiny install through the REAL repack pipeline (verbatim mlx-community
//! tensor naming, INT8 router + shared MLP, BF16 learned norms, per-layer
//! packed experts, mixed SWA/full layers), opens it with
//! `RealForwardRunner` (which selects the learned-weight flow on the
//! verbatim naming), and drives real decode steps on real Metal hardware.
//! Weights are deterministic but not trained: only the pipeline is being
//! proven, not semantics.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use mrefrust_repack::build_synthetic_gemma4_real_install;
use mrefrust_runtime::{LogitProducer, RealForwardRunner};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "mrefrust-real-forward-gemma4-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

const VOCAB: i64 = 128;

fn open_runner(dir: &std::path::Path) -> RealForwardRunner {
    let arch = build_synthetic_gemma4_real_install(dir, VOCAB, 2, 2, 2, 8, "tiny-gemma4-real")
        .expect("real-naming install builds");
    RealForwardRunner::open(dir, arch).expect("real-naming install opens")
}

fn greedy_decode(runner: &mut RealForwardRunner, steps: usize) -> Vec<i32> {
    runner.reset();
    let mut token = 5i32;
    let mut out = Vec::new();
    for position in 0..steps {
        let mut probs = vec![f16::from_f32(0.0); VOCAB as usize];
        runner
            .produce(token, position, &mut probs)
            .expect("produce succeeds");
        let sum: f32 = probs.iter().map(|p| p.to_f32()).sum();
        let bad: Vec<(usize, f32)> = probs
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.to_f32().is_finite())
            .map(|(i, p)| (i, p.to_f32()))
            .collect();
        assert!(
            bad.is_empty(),
            "non-finite probability at position {position}: {} bad of {}, first {:?}",
            bad.len(),
            probs.len(),
            &bad[..bad.len().min(8)]
        );
        assert!(
            (sum - 1.0).abs() < 0.05,
            "probabilities sum to {sum} at position {position}"
        );
        let argmax = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
            .map(|(i, _)| i as i32)
            .unwrap();
        out.push(argmax);
        token = argmax;
    }
    out
}

/// An install with more experts per layer than the runner will be given
/// cache slots, so every token after the first mixes cache hits and
/// misses: what the hit-expert phase-1 command buffer splits.
fn build_contended_install(dir: &std::path::Path) -> model_io::ArchConfig {
    build_synthetic_gemma4_real_install(dir, VOCAB, 2, 8, 4, 8, "tiny-gemma4-real")
        .expect("real-naming install builds")
}

/// Decodes a fixed token sequence (not argmax-fed, so two runners cannot
/// diverge through their own inputs) and returns every step's
/// probabilities.
fn decode_probs(runner: &mut RealForwardRunner, tokens: &[i32]) -> Vec<Vec<f32>> {
    runner.reset();
    tokens
        .iter()
        .enumerate()
        .map(|(position, &token)| {
            let mut probs = vec![f16::from_f32(0.0); VOCAB as usize];
            runner
                .produce(token, position, &mut probs)
                .expect("produce succeeds");
            probs.iter().map(|p| p.to_f32()).collect()
        })
        .collect()
}

const PROBE_TOKENS: [i32; 8] = [5, 9, 2, 7, 1, 3, 8, 4];

#[test]
fn hit_expert_command_buffer_does_not_change_output() {
    // Two fresh runners over one install, so both walk the same expert
    // cache history and differ only in whether the resident share of each
    // layer's experts is dispatched ahead of the pread.
    let dir = temp_dir();
    let arch = build_contended_install(&dir);
    let mut overlapped = RealForwardRunner::open_with_options(&dir, arch.clone(), 4096, 4)
        .expect("real-naming install opens");
    let mut serial = RealForwardRunner::open_with_options(&dir, arch, 4096, 4)
        .expect("real-naming install opens");
    overlapped.set_hit_cb_overlap(true);
    serial.set_hit_cb_overlap(false);

    let with = decode_probs(&mut overlapped, &PROBE_TOKENS);
    let without = decode_probs(&mut serial, &PROBE_TOKENS);

    let p = overlapped.phase_counters();
    assert!(
        p.expert_hits > 0 && p.expert_hits < p.expert_requests,
        "fixture must mix cache hits and misses to exercise the split: \
         {} hits of {} requests",
        p.expert_hits,
        p.expert_requests
    );
    assert_eq!(
        with, without,
        "the hit-expert command buffer runs the same kernels over the same \
         slots, so its output must be bit-identical"
    );
}

#[test]
fn real_naming_install_decodes_deterministically() {
    let dir = temp_dir();
    let mut runner = open_runner(&dir);

    let first = greedy_decode(&mut runner, 6);
    let second = greedy_decode(&mut runner, 6);
    assert_eq!(first, second, "two greedy runs must be identical");
}

#[test]
fn decode_hot_path_allocates_no_gpu_buffers() {
    let dir = temp_dir();
    let mut runner = open_runner(&dir);

    // Warm up: first token compiles pipelines and binds blobs.
    let _ = greedy_decode(&mut runner, 2);
    let before = runner.gpu_buffer_allocations();
    let _ = greedy_decode(&mut runner, 4);
    let after = runner.gpu_buffer_allocations();
    assert_eq!(
        before, after,
        "real-Gemma decode hot path must not allocate Metal buffers"
    );
}
