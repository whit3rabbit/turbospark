#![cfg(target_os = "macos")]
//! Synthetic Qwen2 coverage for the shared Llama attention flow.

use half::f16;
use turbospark_repack::{
    build_synthetic_qwen2_install, build_synthetic_qwen2_install_with_bias,
    build_synthetic_qwen3_vl_install,
};
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

/// Logits after `prompt`, then after one decoded token on top of it. The
/// second is what reads the K/V rows the prompt wrote.
fn per_token_then_one(dir: &std::path::Path, prompt: &[i32]) -> (Vec<f16>, Vec<f16>) {
    let mut runner = open(dir);
    let mut first = vec![f16::from_f32(0.0); VOCAB as usize];
    for (position, &token) in prompt.iter().enumerate() {
        if position + 1 == prompt.len() {
            runner.produce(token, position, &mut first)
        } else {
            runner.produce_prefill(token, position, &mut first)
        }
        .expect("sequential prefill");
    }
    let mut second = vec![f16::from_f32(0.0); VOCAB as usize];
    runner
        .produce(3, prompt.len(), &mut second)
        .expect("sequential decode");
    (first, second)
}

fn chunked_then_one(
    runner: &mut RealForwardRunner,
    prompt: &[i32],
    chunk: usize,
) -> (Vec<f16>, Vec<f16>) {
    runner.reset();
    let mut first = vec![f16::from_f32(0.0); VOCAB as usize];
    for (i, span) in prompt.chunks(chunk).enumerate() {
        runner
            .prefill_chunk(span, i * chunk, &mut first)
            .expect("chunked prefill");
    }
    let mut second = vec![f16::from_f32(0.0); VOCAB as usize];
    runner
        .produce(3, prompt.len(), &mut second)
        .expect("decode after prefill");
    (first, second)
}

/// 41 tokens: two full 16-row micro-batches and a partial one, and chunk
/// spans that leave single-row batches.
fn long_prompt() -> Vec<i32> {
    (0..41).map(|i| (i * 7 + 5) % VOCAB as i32).collect()
}

/// The batched arm must reproduce the per-token loop bit for bit. `forced`
/// makes a refusal an error, so a quiet fallback cannot pass for it.
fn assert_batched_matches(dir: &std::path::Path, forced: bool) {
    let prompt = long_prompt();
    let expected = per_token_then_one(dir, &prompt);
    assert!(expected.0.iter().any(|&v| v != expected.0[0]));
    let mut runner = open(dir);
    // The exact arm: the matrix arm is held to a distribution bound instead.
    runner.set_prefill_steel(false);
    if forced {
        runner.set_batched_gemv_prefill(true);
    }
    for chunk in [41, 17, 7, 2] {
        let actual = chunked_then_one(&mut runner, &prompt, chunk);
        assert_eq!(actual.0, expected.0, "chunk {chunk}: prompt logits");
        assert_eq!(actual.1, expected.1, "chunk {chunk}: next-token logits");
    }
}

#[test]
fn qwen2_batched_prefill_is_bit_identical_to_per_token() {
    let dir = temp_dir("batched");
    build_synthetic_qwen2_install_with_bias(&dir, VOCAB, LAYERS, "tiny-qwen2", 0.25)
        .expect("Qwen2 fixture builds");
    // The default arm, then the forced one.
    assert_batched_matches(&dir, false);
    assert_batched_matches(&dir, true);
}

/// Per-head q/k norm, which Qwen2 does not have.
#[test]
fn per_head_qk_norm_batched_prefill_is_bit_identical_to_per_token() {
    let dir = temp_dir("batched-qknorm");
    build_synthetic_qwen3_vl_install(&dir, VOCAB, LAYERS, "tiny-qwen3-vl")
        .expect("qwen3_vl fixture builds");
    assert_batched_matches(&dir, true);
}

/// The same equality on a real llama-flow install (any family the batched arm
/// serves), plus the paired prefill time of the two arms.
///
/// ```sh
/// TURBOSPARK_LLAMA_FLOW_INSTALL_DIR=~/.turbospark/models/text/qwen25-7b-4bit.gturbo \
///   cargo test -p turbospark-runtime --release --test real_forward_qwen2 -- --ignored --nocapture
/// ```
#[test]
#[ignore = "needs a real install in TURBOSPARK_LLAMA_FLOW_INSTALL_DIR"]
fn real_install_batched_prefill_is_bit_identical_and_timed() {
    let dir = std::path::PathBuf::from(
        std::env::var("TURBOSPARK_LLAMA_FLOW_INSTALL_DIR").expect("install dir"),
    );
    let arch = turbospark_repack::peek_manifest_arch(&dir).expect("manifest arch peeks");
    let vocab = arch.vocab_size as usize;
    let prompt: Vec<i32> = (0..1117u32)
        .map(|i| (100 + (i * 7919) % 20_000) as i32)
        .collect();
    let mut per_token = open(&dir);
    per_token.set_batched_gemv_prefill(false);
    let mut batched = open(&dir);
    batched.set_batched_gemv_prefill(true);
    // The exact arm; the matrix arm's gate is `qwen2_prefill_arm_probe`.
    batched.set_prefill_steel(false);

    // Chunks of 128, as the CLI feeds them, then four greedy tokens.
    let run = |runner: &mut RealForwardRunner| {
        runner.reset();
        let mut logits = vec![f16::from_f32(0.0); vocab];
        let started = std::time::Instant::now();
        for (i, span) in prompt.chunks(128).enumerate() {
            runner
                .prefill_chunk(span, i * 128, &mut logits)
                .expect("chunked prefill");
        }
        let seconds = started.elapsed().as_secs_f64();
        let mut outputs = vec![logits.clone()];
        for step in 0..4 {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
                .expect("vocab")
                .0 as i32;
            runner
                .produce(next, prompt.len() + step, &mut logits)
                .expect("decode");
            outputs.push(logits.clone());
        }
        (outputs, seconds)
    };

    // Round 0 is warmup; the arms interleave so drift hits both.
    for round in 0..4 {
        let (expected, per_token_seconds) = run(&mut per_token);
        let (actual, batched_seconds) = run(&mut batched);
        assert!(actual == expected, "round {round}: logits differ");
        println!(
            "round {round}: per-token {per_token_seconds:.2} s ({:.1} tok/s), batched \
             {batched_seconds:.2} s ({:.1} tok/s), {:.2}x",
            prompt.len() as f64 / per_token_seconds,
            prompt.len() as f64 / batched_seconds,
            per_token_seconds / batched_seconds
        );
    }
}

/// The matrix arm (Steel GEMMs, multi-row attention, 64-row micro-batches) is
/// not bit-exact, so the synthetic check is that it stays close, across a
/// micro-batch boundary and for the token decoded on its cache. Distribution
/// quality is the real-install probe's job (`qwen2_prefill_arm_probe`).
#[test]
fn qwen2_matrix_prefill_stays_close_to_per_token() {
    let dir = temp_dir("matrix");
    build_synthetic_qwen2_install_with_bias(&dir, VOCAB, LAYERS, "tiny-qwen2", 0.25)
        .expect("Qwen2 fixture builds");
    let prompt: Vec<i32> = (0..150).map(|i| (i * 7 + 5) % VOCAB as i32).collect();
    let expected = per_token_then_one(&dir, &prompt);
    let mut runner = open(&dir);
    for rows_attention in [true, false] {
        runner.set_prefill_rows_attention(rows_attention);
        let actual = chunked_then_one(&mut runner, &prompt, 150);
        // Equality would mean the exact arm ran and this proves nothing.
        assert_ne!(actual.0, expected.0, "the matrix arm did not engage");
        for (name, a, b) in [
            ("prompt", &actual.0, &expected.0),
            ("next token", &actual.1, &expected.1),
        ] {
            let worst = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x.to_f32() - y.to_f32()).abs())
                .fold(0.0f32, f32::max);
            let scale = b.iter().map(|y| y.to_f32().abs()).fold(0.0f32, f32::max);
            println!("rows_attention {rows_attention} {name}: worst {worst} of {scale}");
            assert!(worst <= 0.02 * scale, "{name}: {worst} against {scale}");
        }
    }
}
