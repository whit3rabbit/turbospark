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
use turbospark_repack::build_synthetic_gemma4_real_install;
use turbospark_runtime::{LogitProducer, RealForwardRunner};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-real-forward-gemma4-{}-{n}",
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
        let mut head = vec![f16::from_f32(0.0); VOCAB as usize];
        runner
            .produce(token, position, &mut head)
            .expect("produce succeeds");
        let bad: Vec<(usize, f32)> = head
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.to_f32().is_finite())
            .map(|(i, v)| (i, v.to_f32()))
            .collect();
        assert!(
            bad.is_empty(),
            "non-finite logit at position {position}: {} bad of {}, first {:?}",
            bad.len(),
            head.len(),
            &bad[..bad.len().min(8)]
        );
        // `produce` returns SOFTCAPPED LOGITS, not probabilities: the
        // softmax belongs to `selection::select`. Guard the contract by the
        // softcap bound -- probabilities would all sit inside [0, 1].
        let softcap = 30.0f32;
        let max = head.iter().map(|v| v.to_f32()).fold(f32::MIN, f32::max);
        let min = head.iter().map(|v| v.to_f32()).fold(f32::MAX, f32::min);
        assert!(
            min >= -softcap && max <= softcap,
            "logits [{min}, {max}] escape the softcap bound at position {position}"
        );
        let argmax = head
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
fn routed_pipeline_seam_states_are_identical() {
    // Every combination of the two overlap seams must be bit-identical
    // to a fully-serial baseline: same kernels, same commit-relative order
    // of every host buffer write, only the overlap differs. Output
    // equality alone is weak teeth on this fixture (near-uniform synthetic
    // routers, AGENTS.md Gotcha 12), so the pipeline is also asserted
    // structurally through its phase bucket: the post-loop drain always
    // accrues wait time when the routed commit is hoisted, and never when
    // it is not.
    let dir = temp_dir();
    let arch = build_contended_install(&dir);
    let mut baseline = RealForwardRunner::open_with_options(&dir, arch.clone(), 4096, 4)
        .expect("real-naming install opens");
    baseline.set_routed_pipeline(false);
    baseline.set_shared_cb_overlap(false);
    let expected = decode_probs(&mut baseline, &PROBE_TOKENS);
    let p = baseline.phase_counters();
    assert!(
        p.expert_hits > 0 && p.expert_hits < p.expert_requests,
        "fixture must mix cache hits and misses: {} hits of {} requests",
        p.expert_hits,
        p.expert_requests
    );
    assert_eq!(
        p.pipeline_wait_nanos, 0,
        "the non-pipelined arm must never retire a pending routed buffer"
    );

    for combo in 0u8..4 {
        let (pipeline, shared) = (combo & 1 != 0, combo & 2 != 0);
        let mut runner = RealForwardRunner::open_with_options(&dir, arch.clone(), 4096, 4)
            .expect("real-naming install opens");
        runner.set_routed_pipeline(pipeline);
        runner.set_shared_cb_overlap(shared);
        let probs = decode_probs(&mut runner, &PROBE_TOKENS);
        assert_eq!(
            probs, expected,
            "seam combo (pipeline={pipeline}, shared={shared}) \
             must match the serial baseline bit for bit"
        );
        let p = runner.phase_counters();
        if pipeline {
            assert!(
                p.pipeline_wait_nanos > 0,
                "pipelined runner must have drained a pending routed buffer"
            );
        } else {
            assert_eq!(
                p.pipeline_wait_nanos, 0,
                "non-pipelined runner must never commit the routed work early"
            );
        }
    }
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
