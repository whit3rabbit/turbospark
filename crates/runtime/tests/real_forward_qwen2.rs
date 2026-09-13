#![cfg(target_os = "macos")]
//! Synthetic Qwen2 coverage for the shared Llama attention flow.

use half::f16;
use turbospark_repack::{build_synthetic_qwen2_install, build_synthetic_qwen2_install_with_bias};
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};

const VOCAB: i64 = 128;
const LAYERS: i64 = 4;

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("turbospark-qwen2-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn open(dir: &std::path::Path) -> RealForwardRunner {
    let arch = turbospark_repack::peek_manifest_arch(dir).expect("manifest arch peeks");
    RealForwardRunner::open(dir, arch).expect("Qwen2 install opens")
}

fn logits_after_prompt(dir: &std::path::Path, tokens: &[i32]) -> Vec<f16> {
    let mut runner = open(dir);
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    for (position, &token) in tokens.iter().enumerate() {
        if position + 1 == tokens.len() {
            runner
                .produce(token, position, &mut logits)
                .expect("Qwen2 produces");
        } else {
            runner
                .produce_prefill(token, position, &mut logits)
                .expect("Qwen2 prefill produces");
        }
    }
    logits
}

#[test]
fn qwen2_dense_install_opens_and_produces_finite_logits() {
    let dir = temp_dir("decode");
    let arch = build_synthetic_qwen2_install(&dir, VOCAB, LAYERS, "tiny-qwen2")
        .expect("Qwen2 fixture builds");
    assert_eq!(arch.family, model_io::ModelFamily::Qwen2Dense);
    assert_eq!(arch.num_experts, 0);

    let logits = logits_after_prompt(&dir, &[5]);
    assert!(
        logits.iter().all(|v| v.to_f32().is_finite()),
        "Qwen2 logits must remain finite"
    );
}

#[test]
fn qwen2_qkv_biases_reach_the_logits() {
    let zero = temp_dir("zero-bias");
    let nonzero = temp_dir("nonzero-bias");
    build_synthetic_qwen2_install_with_bias(&zero, VOCAB, LAYERS, "tiny-qwen2", 0.0)
        .expect("zero-bias fixture builds");
    build_synthetic_qwen2_install_with_bias(&nonzero, VOCAB, LAYERS, "tiny-qwen2", 0.25)
        .expect("nonzero-bias fixture builds");

    assert_ne!(
        logits_after_prompt(&zero, &[5, 7, 11]),
        logits_after_prompt(&nonzero, &[5, 7, 11]),
        "Qwen2 Q/K/V biases must affect the attention output"
    );
}

#[test]
fn qwen2_chunked_prefill_matches_sequential_prefill() {
    let dir = temp_dir("prefill");
    build_synthetic_qwen2_install(&dir, VOCAB, LAYERS, "tiny-qwen2").expect("Qwen2 fixture builds");
    let tokens = [5, 7, 11, 13, 17];

    let mut sequential = open(&dir);
    sequential.reset();
    let mut expected = vec![f16::from_f32(0.0); VOCAB as usize];
    for (position, &token) in tokens.iter().enumerate() {
        if position + 1 == tokens.len() {
            sequential
                .produce(token, position, &mut expected)
                .expect("sequential final token produces");
        } else {
            sequential
                .produce_prefill(token, position, &mut expected)
                .expect("sequential prefill token produces");
        }
    }

    let mut chunked = open(&dir);
    chunked.reset();
    let mut actual = vec![f16::from_f32(0.0); VOCAB as usize];
    chunked
        .prefill_chunk(&tokens, 0, &mut actual)
        .expect("chunked prefill succeeds");

    assert_eq!(actual, expected, "Qwen2 prefill must share the bias path");
}
