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
