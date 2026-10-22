//! The `flint` decode flow, on a synthetic install built in the checkpoint's
//! own source form (U8 packing, raw BF16 router force-quantized to INT8).
//!
//! `crates/runtime/CLAUDE.md` Gotcha 11 shapes this file: the fixture's
//! weights are UNTRAINED, so "it decoded" is nearly worthless as evidence. A
//! dropped sink, a skipped q/k norm or an ignored shared expert all produce
//! meaningless output too. So each test perturbs one input in the install and
//! requires the logits to MOVE: a flow that silently drops that input is
//! exactly invariant to it, which a fixture CAN see.
//!
//! **BLIND SPOTS, stated rather than hoped away.** Nothing here can show that
//! RoPE is correctly ABSENT on the full-attention layers (a perturbation can
//! prove an input is read, not that an operation is skipped), and nothing
//! compares against a reference implementation. Both are the real-model
//! gates' job (`docs/FLINT_PHASE0.md`).

#![cfg(target_os = "macos")]

use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use turbospark_repack::build_synthetic_flint_install;
use turbospark_runtime::{LogitProducer, RealForwardRunner};

fn tempdir() -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("turbospark-flint-{}-{unique}", std::process::id()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn flint_install() -> (std::path::PathBuf, model_io::ArchConfig) {
    let dir = tempdir();
    let arch = build_synthetic_flint_install(&dir, 64, "flint-toy").expect("write install");
    (dir, arch)
}

/// Decodes `steps` greedy tokens and returns the last step's logits. `steps`
/// exceeds the fixture's 8-token window, so the sliding layer wraps its ring.
fn decode(dir: &std::path::Path, arch: &model_io::ArchConfig, steps: usize) -> Vec<f32> {
    let vocab = arch.vocab_size as usize;
    let mut runner = RealForwardRunner::open(dir, arch.clone()).expect("a flint install opens");
    runner.reset();
    let mut token = 5i32;
    let mut last = Vec::new();
    for position in 0..steps {
        let mut logits = vec![f16::from_f32(0.0); vocab];
        runner
            .produce(token, position, &mut logits)
            .unwrap_or_else(|e| panic!("produce failed at position {position}: {e}"));
        let bad = logits.iter().filter(|v| !v.to_f32().is_finite()).count();
        assert_eq!(bad, 0, "non-finite logits at position {position}");
        token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
            .map(|(i, _)| i as i32)
            .unwrap();
        last = logits.iter().map(|v| v.to_f32()).collect();
    }
    last
}

/// Overwrites one resident tensor's bytes in place with a repeated u16.
fn patch(dir: &std::path::Path, name: &str, value: u16) {
    let path = dir.join("model_weights.bin");
    let index = model_io::load_resident_index(&path).expect("index");
    let entry = index
        .entries
        .get(name)
        .unwrap_or_else(|| panic!("{name} is not in the resident index"));
    let mut bytes = std::fs::read(&path).unwrap();
    let start = entry.file_offset as usize;
    let end = start + entry.size_bytes as usize;
    for chunk in bytes[start..end].chunks_exact_mut(2) {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
    std::fs::write(&path, bytes).unwrap();
}

fn patch_manifest(dir: &std::path::Path, field: &str, value: serde_json::Value) {
    let path = dir.join("manifest.json");
    let mut m: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert!(
        m["arch"].get(field).is_some(),
        "manifest has no arch field `{field}`"
    );
    m["arch"][field] = value;
    std::fs::write(&path, serde_json::to_vec_pretty(&m).unwrap()).unwrap();
}

fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

fn layer(l: usize, suffix: &str) -> String {
    format!("language_model.model.layers.{l}.{suffix}")
}

/// BF16 8.0.
const BF16_EIGHT: u16 = 0x4100;
/// An arbitrary non-zero packed pattern: any change to a packed weight moves
/// the product it feeds.
const PACKED_NOISE: u16 = 0x9E37;

#[test]
fn a_flint_install_opens_and_decodes_past_its_sliding_window() {
    let (dir, arch) = flint_install();
    // Layers 0 and 1 are dense AND full-attention; 2 slides; 3 is full.
    assert_eq!(arch.full_attention_layer_mask, vec![1u8, 1, 0, 1]);
    assert_eq!(arch.num_dense_leading_layers, 2);

    let logits = decode(&dir, &arch, 20);
    assert!(20 > arch.sliding_window as usize, "must cross the window");
    let first = logits[0];
    assert!(
        logits.iter().any(|v| *v != first),
        "every logit is {first}: the head produced nothing"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn decoding_the_same_install_twice_is_bit_identical() {
    let (dir, arch) = flint_install();
    let a = decode(&dir, &arch, 12);
    let b = decode(&dir, &arch, 12);
    assert_eq!(a, b, "two fresh runners disagree: generation is not deterministic");
    std::fs::remove_dir_all(&dir).ok();
}

/// Every one of these changes the logits, or the named input never reached
/// the math. Checked one tensor group at a time, because "something moved"
/// is satisfied by a flow that wires one input and drops the rest.
#[test]
fn every_flint_specific_input_reaches_the_math() {
    let (dir, arch) = flint_install();
    let baseline = decode(&dir, &arch, 6);
    std::fs::remove_dir_all(&dir).ok();

    // (what, tensors to patch, pattern)
    let cases: Vec<(&str, Vec<String>, u16)> = vec![
        (
            "the learned attention sinks (softmax_offset)",
            (0..4).map(|l| layer(l, "self_attn.softmax_offset")).collect(),
            BF16_EIGHT,
        ),
        (
            "the per-head q norm",
            (0..4).map(|l| layer(l, "self_attn.q_norm.weight")).collect(),
            BF16_EIGHT,
        ),
        (
            "the per-head k norm",
            (0..4).map(|l| layer(l, "self_attn.k_norm.weight")).collect(),
            BF16_EIGHT,
        ),
        (
            "dense lead layer 0's FFN",
            vec![layer(0, "mlp.down_proj.weight")],
            PACKED_NOISE,
        ),
        (
            "dense lead layer 1's FFN",
            vec![layer(1, "mlp.down_proj.weight")],
            PACKED_NOISE,
        ),
        (
            "the shared expert",
            vec![layer(2, "mlp.shared_expert.down_proj.weight")],
            PACKED_NOISE,
        ),
        (
            "the INT4 lm_head",
            vec!["language_model.lm_head.weight".to_string()],
            PACKED_NOISE,
        ),
    ];
    for (what, tensors, pattern) in cases {
        let (dir, arch) = flint_install();
        for name in &tensors {
            patch(&dir, name, pattern);
        }
        let after = decode(&dir, &arch, 6);
        assert!(
            distance(&baseline, &after) > 0.0,
            "perturbing {what} left the logits bit-identical: it is not reaching the math"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// THE ROUTED EXPERTS AND THE ROUTER'S top-k REACH THE MATH. Narrowing top-k
/// from 2 to 1 changes both which experts run and their renormalised weight
/// (1.0 for a single expert), so a flow ignoring `top_k_experts` is invariant.
#[test]
fn narrowing_top_k_changes_the_output() {
    let (dir, arch) = flint_install();
    let before = decode(&dir, &arch, 6);
    let mut one = arch.clone();
    one.top_k_experts = 1;
    patch_manifest(&dir, "topKExperts", serde_json::json!(1));
    let after = decode(&dir, &one, 6);
    assert!(distance(&before, &after) > 0.0, "top_k is not reaching the router");
    std::fs::remove_dir_all(&dir).ok();
}

/// RoPE reaches the SLIDING layer. (That it is absent on the full layers is
/// a blind spot of this file; see the header.)
#[test]
fn the_sliding_layers_rope_theta_changes_the_output() {
    let (dir, arch) = flint_install();
    let before = decode(&dir, &arch, 8);
    let mut other = arch.clone();
    other.rope_theta = 500.0;
    patch_manifest(&dir, "ropeTheta", serde_json::json!(500.0));
    let after = decode(&dir, &other, 8);
    assert!(
        distance(&before, &after) > 0.0,
        "rope_theta left the logits unchanged: the sliding layer is not rotating"
    );
    std::fs::remove_dir_all(&dir).ok();
}

fn refusal(dir: &std::path::Path, arch: model_io::ArchConfig, what: &str) -> String {
    match RealForwardRunner::open(dir, arch) {
        Ok(_) => panic!("{what} must be refused, but it opened"),
        Err(e) => format!("{e}"),
    }
}

/// The backstops `RealFlintState::build` owes. Each case makes the manifest
/// AGREE with the doctored config first, because `arch_validation` otherwise
/// refuses the mismatch before the flow runs; agreement is the state a
/// hand-edited install would be in, which is what a backstop exists for.
#[test]
fn an_install_the_flow_cannot_run_is_refused_by_name() {
    // A router that is not sigmoid would decode fluently and wrongly.
    let (dir, arch) = flint_install();
    let mut softmax = arch.clone();
    softmax.router_scoring_func = "softmax".to_string();
    patch_manifest(&dir, "routerScoringFunc", serde_json::json!("softmax"));
    let err = refusal(&dir, softmax, "a softmax-router flint");
    assert!(err.contains("sigmoid"), "got: {err}");
    std::fs::remove_dir_all(&dir).ok();

    // A rope theta on the full layers would rotate where Flint carries none.
    let (dir, arch) = flint_install();
    let mut roped = arch.clone();
    roped.full_rope_theta = 10_000.0;
    patch_manifest(&dir, "fullRopeTheta", serde_json::json!(10_000.0));
    let err = refusal(&dir, roped, "a flint with a full-layer rope theta");
    assert!(err.contains("positional encoding"), "got: {err}");
    std::fs::remove_dir_all(&dir).ok();

    // A tied head.
    let (dir, arch) = flint_install();
    let mut tied = arch.clone();
    tied.tie_word_embeddings = true;
    patch_manifest(&dir, "tieWordEmbeddings", serde_json::json!(true));
    let err = refusal(&dir, tied, "a tied flint");
    assert!(err.contains("untied"), "got: {err}");
    std::fs::remove_dir_all(&dir).ok();
}
