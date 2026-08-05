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
