#![cfg(target_os = "macos")]
//! End-to-end proof of the DENSE arm of the real-checkpoint Gemma 4 decode
//! flow: builds a tiny dense install through the REAL repack pipeline
//! (verbatim `language_model.` naming, INT8 affine everywhere including the
//! embedding table, no router, no routed experts, the 5:1 sliding/full
//! mask), opens it with `RealForwardRunner`, and drives real decode steps
//! on real Metal hardware. Weights are deterministic but not trained: only
//! the pipeline is being proven, not semantics. The install builder is
//! `build_synthetic_gemma4_dense_real_install`, read off
//! `cowWhySo/humanizer-12b-mlx-8bit`'s inventory.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use turbospark_repack::build_synthetic_gemma4_dense_real_install;
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-real-forward-gemma4-dense-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

const VOCAB: i64 = 128;
/// 7 layers: mask `(l % 6 == 5)` puts the full-attention layer at index 5,
/// so both layer kinds run inside one fixture.
const LAYERS: i64 = 7;

fn install(dir: &Path) -> model_io::ArchConfig {
    build_synthetic_gemma4_dense_real_install(dir, VOCAB, LAYERS, 8, "tiny-gemma4-dense")
        .expect("dense real-naming install builds")
}

fn decode_probs(runner: &mut RealForwardRunner, tokens: &[i32]) -> Vec<Vec<f32>> {
    runner.reset();
    tokens
        .iter()
        .enumerate()
        .map(|(position, &token)| {
            let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
            runner
                .produce(token, position, &mut logits)
                .expect("produce succeeds");
            logits.iter().map(|p| p.to_f32()).collect()
        })
        .collect()
}

const PROBE_TOKENS: [i32; 8] = [5, 9, 2, 7, 1, 3, 8, 4];

fn assert_finite_and_capped(probs: &[Vec<f32>]) {
    let softcap = 30.0f32;
    for (step, row) in probs.iter().enumerate() {
        let max = row.iter().copied().fold(f32::MIN, f32::max);
        let min = row.iter().copied().fold(f32::MAX, f32::min);
        assert!(
            min.is_finite() && max.is_finite(),
            "non-finite logit at step {step}"
        );
        assert!(
            min >= -softcap && max <= softcap,
            "logits [{min}, {max}] escape the softcap bound at step {step}"
        );
    }
}

#[test]
fn dense_install_opens_and_decodes_deterministically() {
    let dir = temp_dir();
    let arch = install(&dir);
    let mut runner = RealForwardRunner::open(&dir, arch).expect("dense install opens");

    let first = decode_probs(&mut runner, &PROBE_TOKENS);
    assert_finite_and_capped(&first);
    let second = decode_probs(&mut runner, &PROBE_TOKENS);
    assert_eq!(
        first, second,
        "two runs on one warm runner must be identical"
    );

    // THE FROZEN DIGEST (AGENTS.md Gotcha 51): every other assertion in
    // this file is self-relative, so a change to the dense MATH leaves
    // them all green. This hex is the only line that compares against
    // something computed before any mutation. Re-freeze deliberately, and
    // only with the reason written down.
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for row in &first {
        for value in row {
            use std::hash::Hasher;
            hasher.write(&value.to_bits().to_le_bytes());
        }
    }
    let digest = format!("{:016x}", {
        use std::hash::Hasher;
        hasher.finish()
    });
    assert!(
        digest == "cec63662c879f841" || digest == "950fb729c520fdc9",
        "dense decode digest moved ({digest}): a real change to the flow, or a bug"
    );
}

#[test]
fn dense_prefill_then_decode_matches_all_produce() {
    let dir = temp_dir();
    let arch = install(&dir);
    let tokens: Vec<i32> = (0..6)
        .map(|i| PROBE_TOKENS[i % PROBE_TOKENS.len()])
        .collect();

    let mut all = RealForwardRunner::open(&dir, arch.clone()).expect("open");
    let mut logits_all = vec![f16::from_f32(0.0); VOCAB as usize];
    for (i, &t) in tokens.iter().enumerate() {
        all.produce(t, i, &mut logits_all).expect("produce");
    }

    let mut split = RealForwardRunner::open(&dir, arch.clone()).expect("open");
    let mut logits_split = vec![f16::from_f32(0.0); VOCAB as usize];
    for (i, &t) in tokens.iter().enumerate() {
        if i + 1 < tokens.len() {
            split
                .produce_prefill(t, i, &mut logits_split)
                .expect("prefill");
        } else {
            split.produce(t, i, &mut logits_split).expect("produce");
        }
    }
    assert_eq!(logits_all, logits_split);
}

#[test]
fn dense_chunked_prefill_matches_all_produce() {
    // The chunk driver routes a dense install through the sequential
    // fallback inside `prefill_chunk_real_gemma4`; this pins that the
    // fallback is the same math as the per-token path, not merely finite.
    let dir = temp_dir();
    let arch = install(&dir);
    let tokens: Vec<i32> = (0..9)
        .map(|i| PROBE_TOKENS[i % PROBE_TOKENS.len()])
        .collect();

    let mut all = RealForwardRunner::open(&dir, arch.clone()).expect("open");
    let mut logits_all = vec![f16::from_f32(0.0); VOCAB as usize];
    for (i, &t) in tokens.iter().enumerate() {
        all.produce(t, i, &mut logits_all).expect("produce");
    }

    let mut chunked = RealForwardRunner::open(&dir, arch).expect("open");
    let mut logits_chunked = vec![f16::from_f32(0.0); VOCAB as usize];
    chunked
        .prefill_chunk(&tokens, 0, &mut logits_chunked)
        .expect("chunked prefill");
    assert_eq!(logits_all, logits_chunked);
}

#[test]
fn dense_decode_hot_path_allocates_no_gpu_buffers() {
    let dir = temp_dir();
    let arch = install(&dir);
    let mut runner = RealForwardRunner::open(&dir, arch).expect("open");
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    runner.produce(1, 0, &mut logits).expect("warm");
    let before = runner.gpu_buffer_allocations();
    for step in 1..6 {
        runner.produce(2, step, &mut logits).expect("produce");
    }
    assert_eq!(
        before,
        runner.gpu_buffer_allocations(),
        "dense decode hot path must not allocate Metal buffers"
    );
}

#[test]
fn dense_chunk_of_one_matches_all_produce() {
    let dir = temp_dir();
    let arch = install(&dir);
    let tokens: Vec<i32> = (0..5)
        .map(|i| PROBE_TOKENS[i % PROBE_TOKENS.len()])
        .collect();

    let mut all = RealForwardRunner::open(&dir, arch.clone()).expect("open");
    let mut logits_all = vec![f16::from_f32(0.0); VOCAB as usize];
    for (i, &t) in tokens.iter().enumerate() {
        all.produce(t, i, &mut logits_all).expect("produce");
    }

    let mut chunked = RealForwardRunner::open(&dir, arch).expect("open");
    let mut logits_chunked = vec![f16::from_f32(0.0); VOCAB as usize];
    for (i, &t) in tokens.iter().enumerate() {
        let last = i + 1 == tokens.len();
        let mut scratch = vec![f16::from_f32(0.0); VOCAB as usize];
        chunked
            .prefill_chunk(&[t], i, &mut scratch)
            .expect("chunk of one");
        if last {
            logits_chunked = scratch;
        }
    }
    assert_eq!(logits_all, logits_chunked);
}

#[test]
fn dense_chunked_prefill_with_batched_attention_matches_all_produce() {
    // TURBOSPARK_BATCHED_GEMV's seam on the dense driver: the four attention
    // projections as M-row INT8 GEMMs. Bit-exactness against the sequential
    // walk is the same contract the MoE chunk driver carries, and the set
    // covers the batch sizes the best-row-block table switches at.
    let dir = temp_dir();
    let arch = install(&dir);
    let tokens: Vec<i32> = (0..7)
        .map(|i| PROBE_TOKENS[i % PROBE_TOKENS.len()])
        .collect();

    let mut all = RealForwardRunner::open(&dir, arch.clone()).expect("open");
    let mut logits_all = vec![f16::from_f32(0.0); VOCAB as usize];
    for (i, &t) in tokens.iter().enumerate() {
        all.produce(t, i, &mut logits_all).expect("produce");
    }

    let mut chunked = RealForwardRunner::open(&dir, arch).expect("open");
    chunked.set_batched_gemv_prefill(true);
    let mut logits_chunked = vec![f16::from_f32(0.0); VOCAB as usize];
    chunked
        .prefill_chunk(&tokens, 0, &mut logits_chunked)
        .expect("chunked prefill with batched attention");
    assert_eq!(logits_all, logits_chunked);
}
