#![cfg(target_os = "macos")]
//! Speculation on a K-quant GGUF trunk, end to end on a synthetic install.
//!
//! The pi checkpoint (`Qwen3.8-27B` Q4_K_M plus its llama.cpp MTP block) is
//! the motivating file and is 17 GB, so this drives the same path in
//! milliseconds: the GGUF walk ingests the `blk.<n>.nextn.*` head under
//! `mtp.*`, the open-time gate admits Q4_K / Q6_K / Q8_0, and the batched
//! verify runs the K-quant GEMMs over GDN and full-attention layers.
//!
//! What it can say: the head is ingested and drafts, the gate opens, and the
//! batched pass is BIT-IDENTICAL to the same tokens run one at a time, which
//! is the property that makes a speculative verify lossless. What it cannot
//! say: anything about draft acceptance. The weights are untrained.

use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use turbospark_repack::{
    build_synthetic_qwen35_dense_gguf, parse_gguf_header, write_gguf_install_streamed,
    MemoryRangeSource, GGUF_DEFAULT_MAX_HEADER_BYTES, QWEN35_GGUF_VOCAB,
};
use turbospark_runtime::{LogitProducer, RealForwardRunner};

const VOCAB: usize = QWEN35_GGUF_VOCAB as usize;

fn temp_dir(tag: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-gguf-qwen35-mtp-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn install(tag: &str, with_head: bool) -> (std::path::PathBuf, model_io::ArchConfig) {
    let (bytes, _) = build_synthetic_qwen35_dense_gguf(with_head);
    let header = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).expect("parse");
    let dir = temp_dir(tag);
    let arch = write_gguf_install_streamed(
        &dir,
        &header,
        &MemoryRangeSource::new(&bytes),
        "qwen35-gguf-mtp",
        |_| {},
    )
    .expect("write install");
    (dir, arch)
}

fn open(dir: &std::path::Path, arch: model_io::ArchConfig) -> RealForwardRunner {
    RealForwardRunner::open_with_options_and_speculation(
        dir,
        arch,
        4096,
        16,
        turbospark_runtime::DraftPolicies::mtp(turbospark_runtime::MtpDraftPolicy::Fixed(2)),
    )
    .expect("install opens with a draft depth")
}

fn argmax(logits: &[f16]) -> i32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
        .map(|(i, _)| i as i32)
        .unwrap()
}

fn assert_finite_and_varied(logits: &[f16], what: &str) {
    assert!(
        logits.iter().all(|v| v.to_f32().is_finite()),
        "{what}: non-finite logit"
    );
    let first = logits[0].to_f32();
    assert!(
        logits.iter().any(|v| v.to_f32() != first),
        "{what}: every logit is {first}"
    );
}

/// The walk wrote the head under the runtime's names, and the open-time gate
/// that used to refuse dtype 7 now admits this Q4_K_M-shaped trunk.
#[test]
fn a_kquant_gguf_with_a_head_can_speculate() {
    let (dir, arch) = install("gate", true);
    let index = model_io::load_resident_index(&dir.join("model_weights.bin")).expect("index");
    let head: Vec<_> = index
        .entries
        .keys()
        .filter(|k| k.starts_with("mtp."))
        .collect();
    assert_eq!(head.len(), 15, "{head:?}");
    assert!(index.entries.contains_key("mtp.fc.weight"));
    assert!(
        !index.entries.keys().any(|k| k.contains("layers.4.")),
        "the head must not land as a phantom trunk layer"
    );
    // The trunk keeps the K-quant tags the batched kernels serve.
    let q = &index.entries["language_model.model.layers.3.self_attn.q_proj.weight"];
    assert_eq!(q.dtype, 7, "Q4_K");
    // The install-only check a host labels models with agrees, unopened.
    assert_eq!(
        turbospark_runtime::install_speculation_blocker(&dir, &arch),
        None
    );
    let runner = open(&dir, arch);
    assert_eq!(runner.speculation_blocker(), None);
    std::fs::remove_dir_all(&dir).ok();
}

/// The same trunk without its head is refused for the head's absence, not
/// for its dtype: the K-quant arm is no longer the blocker.
#[test]
fn a_headless_kquant_gguf_is_refused_for_the_head_not_the_dtype() {
    let (dir, arch) = install("headless", false);
    let unopened = turbospark_runtime::install_speculation_blocker(&dir, &arch)
        .expect("the install-only check refuses a headless install too");
    assert!(
        unopened.contains("multi-token-prediction head"),
        "{unopened}"
    );
    let result = RealForwardRunner::open_with_options_and_speculation(
        &dir,
        arch.clone(),
        4096,
        16,
        turbospark_runtime::DraftPolicies::mtp(turbospark_runtime::MtpDraftPolicy::Fixed(2)),
    );
    let reason = match result {
        Err(e) => e.to_string(),
        Ok(runner) => runner
            .speculation_blocker()
            .expect("a headless install cannot speculate"),
    };
    assert!(!reason.contains("is dtype"), "{reason}");
    assert!(reason.contains("multi-token-prediction head"), "{reason}");
    std::fs::remove_dir_all(&dir).ok();
}

/// THE LOSSLESSNESS PROPERTY on a K-quant trunk: a batched pass over a block
/// of tokens, then the round-end commit, leaves logits and state
/// bit-identical to the same tokens produced one at a time. Layers 0..2 are
/// GDN and layer 3 full attention, so both batched paths run, through Q4_K,
/// Q6_K (`ffn_down`, `output`) and Q8_0 (`ssm_out`).
#[test]
fn a_kquant_batched_pass_is_bit_identical_to_sequential_produce() {
    let (dir, arch) = install("parity", true);
    let mut runner = open(&dir, arch);
    let tokens = [5i32, 9, 3, 7, 2, 11];
    /// Draft depth 2 plus the verified token: the verify scratch's width.
    const BATCH: usize = 3;

    runner.reset();
    let mut sequential: Vec<f16> = Vec::with_capacity(tokens.len() * VOCAB);
    let mut row = vec![f16::from_f32(0.0); VOCAB];
    for (position, &token) in tokens.iter().enumerate() {
        runner
            .produce(token, position, &mut row)
            .expect("sequential produce");
        sequential.extend_from_slice(&row);
    }
    assert_finite_and_varied(&sequential[..VOCAB], "sequential row 0");
    let sequential_cursor = runner.checkpoint().position();

    runner.reset();
    let mut batched = vec![f16::from_f32(0.0); BATCH * VOCAB];
    runner
        .produce_batched(&tokens[..BATCH], 0, &mut batched)
        .expect("batched pass on a K-quant trunk");
    runner.commit_pending(BATCH).expect("round-end commit");
    for (offset, &token) in tokens[BATCH..].iter().enumerate() {
        runner
            .produce(token, BATCH + offset, &mut row)
            .expect("produce after the batch");
        batched.extend_from_slice(&row);
    }

    let differing = sequential
        .iter()
        .zip(&batched)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        differing,
        0,
        "{differing} of {} logits differ: a K-quant verify built on this would not be lossless",
        sequential.len()
    );
    assert_eq!(sequential_cursor, runner.checkpoint().position());
    std::fs::remove_dir_all(&dir).ok();
}

/// The ingested head drafts: prime it over a few trunk tokens, then take a
/// draft step. Finite, varied logits say the head's tensors bound and ran.
#[test]
fn the_ingested_head_drafts() {
    let (dir, arch) = install("draft", true);
    let mut runner = open(&dir, arch);
    runner.reset();
    let steps = 4usize;
    let mut token = 5i32;
    let mut logits = vec![f16::from_f32(0.0); VOCAB];
    for position in 0..steps {
        runner
            .produce(token, position, &mut logits)
            .expect("trunk produce");
        token = argmax(&logits);
        if position + 1 < steps {
            runner.mtp_prime_step(token, position).expect("prime");
        }
    }
    let mut draft = vec![f16::from_f32(0.0); VOCAB];
    runner
        .mtp_draft_step(token, steps - 1, &mut draft)
        .expect("draft step on the GGUF head");
    assert_finite_and_varied(&draft, "draft");
    std::fs::remove_dir_all(&dir).ok();
}
