#![cfg(target_os = "macos")]
//! Llama 3.x rope scaling through the llama flow: `arch.ropeFreqFactors`
//! (one divisor per rotated pair) bound as `rope_neox_freqs`'s angle table.
//!
//! The synthetic weights are untrained, so nothing here asserts on tokens
//! (AGENTS.md Gotcha 12). What can be checked without a reference engine:
//!
//! - all-ones divisors reproduce the scalar-theta path, so the table and
//!   the kernel agree with the RoPE every other llama install already runs;
//! - real llama3 divisors leave position 0 alone (a zero angle is the
//!   identity whatever the frequency) and move every later position, so the
//!   table actually reaches the attention scores;
//! - the combinations the flow has not been checked under are refused at
//!   open rather than half-applied.
//!
//! The installs are written by the ordinary synthetic builder and then given
//! the field the way the GGUF intake writes it: `arch.ropeFreqFactors` plus
//! manifest minor 2.

use half::f16;
use turbospark_repack::{build_synthetic_dense_llama_install, build_synthetic_qwen2_install};
use turbospark_runtime::{LogitProducer, RealForwardRunner};

const VOCAB: i64 = 128;
const LAYERS: i64 = 2;
const POSITIONS: usize = 6;

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-llama3-rope-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Writes `factors` into the install's manifest as the GGUF intake does.
fn add_rope_factors(dir: &std::path::Path, factors: &[f32]) {
    let path = dir.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    manifest["arch"]["ropeFreqFactors"] = serde_json::json!(factors);
    manifest["versionMinor"] = serde_json::json!(model_io::MANIFEST_VERSION_MINOR);
    std::fs::write(&path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
}

fn dense_llama(tag: &str) -> std::path::PathBuf {
    let dir = temp_dir(tag);
    build_synthetic_dense_llama_install(&dir, VOCAB, LAYERS, "tiny-llama3").expect("builds");
    dir
}

fn pairs(dir: &std::path::Path) -> usize {
    let arch = turbospark_repack::peek_manifest_arch(dir).expect("peek");
    (arch.full_head_dim / 2) as usize
}

/// Fixed token sequence, logits per position.
fn logits(dir: &std::path::Path) -> Vec<Vec<f32>> {
    let arch = turbospark_repack::peek_manifest_arch(dir).expect("peek");
    let mut runner = RealForwardRunner::open(dir, arch).expect("opens");
    runner.reset();
    let tokens = [5i32, 17, 3, 99, 42, 7];
    (0..POSITIONS)
        .map(|position| {
            let mut head = vec![f16::from_f32(0.0); VOCAB as usize];
            runner
                .produce(tokens[position], position, &mut head)
                .expect("produce");
            head.iter().map(|v| v.to_f32()).collect()
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn unit_divisors_reproduce_the_scalar_theta_path() {
    let plain = dense_llama("plain");
    let ones = dense_llama("ones");
    add_rope_factors(&ones, &vec![1.0; pairs(&ones)]);

    let a = logits(&plain);
    let b = logits(&ones);
    for position in 0..POSITIONS {
        let diff = max_abs_diff(&a[position], &b[position]);
        // The table is CPU `powf` and the scalar kernel computes the same
        // power on the GPU, so the angles agree to an ulp or two, which f16
        // logits cannot resolve at these magnitudes.
        assert!(diff < 1e-2, "position {position}: max |dlogit| {diff}");
    }
}

#[test]
fn llama3_divisors_move_every_position_but_the_first() {
    let plain = dense_llama("plain-b");
    let scaled = dense_llama("scaled");
    let arch = turbospark_repack::peek_manifest_arch(&scaled).expect("peek");
    // original context 8 so most pairs of this small head land in the
    // scaled or smooth bands; factor 8 as Llama 3.1 publishes.
    let factors = compute::llama3_rope_factors(
        arch.full_head_dim as usize,
        arch.full_rope_theta,
        8.0,
        1.0,
        4.0,
        8.0,
    )
    .expect("factors");
    assert!(factors.iter().any(|&f| f > 1.0), "{factors:?}");
    add_rope_factors(&scaled, &factors);

    let a = logits(&plain);
    let b = logits(&scaled);
    let first = max_abs_diff(&a[0], &b[0]);
    assert!(first < 1e-2, "position 0 has a zero angle: {first}");
    for position in 1..POSITIONS {
        let diff = max_abs_diff(&a[position], &b[position]);
        assert!(
            diff > 1e-3,
            "position {position}: the divisors did not reach the logits ({diff})"
        );
    }
}

#[test]
fn malformed_or_misplaced_divisors_are_refused_at_open() {
    // One short of the rotated pairs.
    let short = dense_llama("short");
    let n = pairs(&short);
    add_rope_factors(&short, &vec![1.0; n - 1]);
    let arch = turbospark_repack::peek_manifest_arch(&short).expect("peek");
    assert!(RealForwardRunner::open(&short, arch).is_err());

    // A non-positive divisor.
    let zero = dense_llama("zero");
    let mut bad = vec![1.0f32; n];
    bad[1] = 0.0;
    add_rope_factors(&zero, &bad);
    let arch = turbospark_repack::peek_manifest_arch(&zero).expect("peek");
    assert!(RealForwardRunner::open(&zero, arch).is_err());

    // A family that shares the flow but never carries the field.
    let qwen2 = temp_dir("qwen2");
    build_synthetic_qwen2_install(&qwen2, VOCAB, LAYERS, "tiny-qwen2").expect("builds");
    let n = pairs(&qwen2);
    add_rope_factors(&qwen2, &vec![1.0; n]);
    let arch = turbospark_repack::peek_manifest_arch(&qwen2).expect("peek");
    let err = RealForwardRunner::open(&qwen2, arch)
        .err()
        .expect("qwen2 must refuse rope divisors");
    assert!(err.to_string().contains("ropeFreqFactors"), "{err}");
}
