#![cfg(target_os = "macos")]
//! End-to-end proof of the `qwen3moe` decode flow: a tiny Qwen3-MoE-shaped
//! install through the REAL repack pipeline, opened with `RealForwardRunner`,
//! driven on real Metal.
//!
//! **The flow itself is `families/llama/`**, shared with Mixtral, so most of
//! what a per-family test would assert is already asserted by
//! `real_forward_llama.rs`. What is NOT covered there, and is the reason this
//! file exists, is the two places the families diverge: the per-head q/k norms
//! and the RMS epsilon. Both are keyed on `ArchConfig.family`, so the decisive
//! test is that the SAME weights decode DIFFERENTLY under the two families --
//! if the family tag were ignored, they would not.
//!
//! Weights are deterministic but NOT trained, so nothing here asserts on
//! generated TEXT (AGENTS.md Gotcha 12): only ids, counts and invariants.
//!
//! **WHAT THESE TESTS CANNOT SEE, stated rather than implied.** Mutation-
//! checked: disabling `qk_norm`, and holding the epsilon equal so only
//! `qk_norm` varies, both redden `the_per_head_norms_change_...`. Moving the
//! norm to AFTER RoPE does NOT, and cannot: RoPE is a rotation and preserves
//! each head's RMS, so both orders differ from "no norm at all" by about as
//! much, and no fixture comparison against the other family can separate
//! them. Ordering is settled by the reference implementations and is covered
//! by the cross-engine KL against llama.cpp on identical real bytes, which is
//! the gate that sees this class of error (`scripts/kld_llamacpp.py`,
//! AGENTS.md Gotcha 34). The same holds for the epsilon's own value: 1e-6 vs
//! 1e-5 is real and is not separable on untrained weights.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use model_io::ModelFamily;
use turbospark_repack::{build_synthetic_gqa_moe_install, tiny_gqa_moe_arch};
use turbospark_runtime::{LogitProducer, RealForwardRunner};

static COUNTER: AtomicU64 = AtomicU64::new(0);

const VOCAB: i64 = 128;
const LAYERS: i64 = 4;
const EXPERTS: i64 = 8;

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-real-forward-qwen3moe-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn open_family(dir: &std::path::Path, family: ModelFamily) -> RealForwardRunner {
    let arch =
        build_synthetic_gqa_moe_install(dir, VOCAB, LAYERS, EXPERTS, "tiny-qwen3moe", family)
            .expect("install builds");
    RealForwardRunner::open(dir, arch).expect("install opens")
}

fn open_runner(dir: &std::path::Path) -> RealForwardRunner {
    open_family(dir, ModelFamily::Qwen3Moe)
}

/// Logits after `steps` fixed-token steps, i.e. AT A CONTEXT OF `steps`.
///
/// **Not at position 0, and that is the whole point.** A decode step at
/// position 0 attends over exactly one key, and a softmax over one element is
/// 1.0 whatever the logit, so the attention output there is V alone and is
/// independent of q and k entirely. Any test of a q/k transform read at
/// position 0 measures nothing about that transform -- it passed here only
/// because the RMS epsilon also feeds the hidden-state norms. Feeding a fixed
/// token rather than the argmax keeps both arms on the same input even when
/// they disagree.
fn logits_after(runner: &mut RealForwardRunner, steps: usize) -> Vec<f32> {
    assert!(steps > 1, "position 0 cannot see a q/k transform");
    runner.reset();
    let mut head = vec![f16::from_f32(0.0); VOCAB as usize];
    for position in 0..steps {
        runner
            .produce(5, position, &mut head)
            .expect("produce succeeds");
    }
    head.iter().map(|v| v.to_f32()).collect()
}

/// Argmax-fed greedy decode, asserting the logits contract on every step:
/// finite, and NOT a probability distribution (runtime crate Gotcha 1).
fn greedy_decode(runner: &mut RealForwardRunner, steps: usize) -> Vec<i32> {
    runner.reset();
    let mut token = 5i32;
    let mut out = Vec::new();
    for position in 0..steps {
        let mut head = vec![f16::from_f32(0.0); VOCAB as usize];
        runner
            .produce(token, position, &mut head)
            .expect("produce succeeds");
        assert!(
            head.iter().all(|v| v.to_f32().is_finite()),
            "non-finite logit at position {position}"
        );
        let sum: f32 = head.iter().map(|v| v.to_f32()).sum();
        let any_negative = head.iter().any(|v| v.to_f32() < 0.0);
        assert!(
            any_negative || (sum - 1.0).abs() > 1e-2,
            "the head returned something that looks like a probability \
             distribution (all non-negative, sums to {sum}) at position \
             {position}; `produce` must write raw logits"
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
fn qwen3moe_install_decodes_deterministically() {
    let dir = temp_dir();
    let mut runner = open_runner(&dir);

    let first = greedy_decode(&mut runner, 6);
    let second = greedy_decode(&mut runner, 6);
    assert_eq!(first, second, "two greedy runs must be identical");
}

/// THE TEST THIS FILE EXISTS FOR. Two installs of the SAME shape and the same
/// deterministic weights, differing only in the family tag, must compute
/// different functions -- because `qwen3moe` norms q and k per head before
/// RoPE and `llama` does not.
///
/// Compares the full logit vector rather than the argmax: an argmax over 128
/// untrained rows could coincide by luck, and this has to fail if the family
/// tag is ever ignored, not merely usually fail. Read at a context of 4, for
/// the reason [`logits_after`] documents.
#[test]
fn the_per_head_norms_change_the_function_the_family_tag_selects() {
    let qwen_dir = temp_dir();
    let llama_dir = temp_dir();
    let mut qwen = open_family(&qwen_dir, ModelFamily::Qwen3Moe);
    let mut llama = open_family(&llama_dir, ModelFamily::Llama);

    let with_norms = logits_after(&mut qwen, 4);
    let without = logits_after(&mut llama, 4);
    assert_eq!(with_norms.len(), without.len());
    assert!(
        with_norms.iter().all(|v| v.is_finite()) && without.iter().all(|v| v.is_finite()),
        "both flows must produce finite logits"
    );
    assert_ne!(
        with_norms, without,
        "the two families decoded identically: the q/k norms were not applied, \
         so `ArchConfig.family` is being ignored by the shared flow"
    );
}

/// The norms are required at OPEN, not discovered at token 1. A `qwen3moe`
/// install missing them is a broken install, and the alternative to refusing
/// is a decode that silently skips a normalization.
#[test]
fn a_qwen3moe_install_without_the_per_head_norms_is_refused_at_open() {
    let dir = temp_dir();
    // Built as `llama`, so the two `[head_dim]` norm tensors are absent.
    let arch = build_synthetic_gqa_moe_install(
        &dir,
        VOCAB,
        LAYERS,
        EXPERTS,
        "tiny-qwen3moe",
        ModelFamily::Llama,
    )
    .expect("install builds");

    // The manifest has to claim the family too, or `validate_arch` refuses
    // first and this never reaches the FLOW's check, which is what is under
    // test. Every other field is identical between the two families, which is
    // exactly why the missing TENSORS are the only thing left to catch it.
    let manifest_path = dir.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();
    manifest["arch"]["family"] = serde_json::json!("qwen3moe");
    std::fs::write(&manifest_path, serde_json::to_string(&manifest).unwrap()).unwrap();

    let mut claimed = arch;
    claimed.family = ModelFamily::Qwen3Moe;
    let err = RealForwardRunner::open(&dir, claimed)
        .err()
        .expect("an install with no q/k norms must be refused");
    let message = err.to_string();
    assert!(
        message.contains("q_norm") || message.contains("k_norm"),
        "the refusal must name the missing tensor: {message}"
    );
}

/// A layer's routed slots are dispatched in the ROUTER'S RANKING, so the same
/// prompt must decode the same way however warm the expert cache is. Running a
/// longer generation in between changes the cache state and nothing else
/// (AGENTS.md Gotcha 27).
#[test]
fn output_does_not_depend_on_expert_cache_state() {
    let dir = temp_dir();
    let mut runner = open_runner(&dir);

    let baseline = greedy_decode(&mut runner, 3);
    let _ = greedy_decode(&mut runner, 9);
    let again = greedy_decode(&mut runner, 3);
    assert_eq!(
        baseline, again,
        "output moved with expert-cache state: the routed dispatch order is \
         a function of something other than the route"
    );
}

#[test]
fn decode_hot_path_allocates_no_gpu_buffers() {
    let dir = temp_dir();
    let mut runner = open_runner(&dir);

    // Warm up: the first token compiles pipelines and binds blobs.
    let _ = greedy_decode(&mut runner, 2);
    let before = runner.gpu_buffer_allocations();
    let _ = greedy_decode(&mut runner, 4);
    assert_eq!(
        before,
        runner.gpu_buffer_allocations(),
        "the decode path must allocate no Metal buffer per token"
    );
}

/// `produce_prefill` skips the output head and must advance everything else,
/// so a prefill-then-decode run has to agree with an all-`produce` run on the
/// step they share (runtime crate Gotcha 2). Worth repeating for this family
/// because the per-head norms write IN PLACE into the KV slot, which is
/// exactly the kind of side effect a skipped head could strand.
#[test]
fn prefill_and_produce_agree_on_the_scored_token() {
    let dir = temp_dir();
    let mut runner = open_runner(&dir);

    let all_produce = greedy_decode(&mut runner, 3);

    runner.reset();
    let mut scratch = vec![f16::from_f32(0.0); VOCAB as usize];
    let mut token = 5i32;
    for (position, &next) in all_produce.iter().enumerate().take(2) {
        runner
            .produce_prefill(token, position, &mut scratch)
            .expect("prefill succeeds");
        token = next;
    }
    let mut head = vec![f16::from_f32(0.0); VOCAB as usize];
    runner.produce(token, 2, &mut head).expect("produce");
    let argmax = head
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
        .map(|(i, _)| i as i32)
        .unwrap();
    assert_eq!(argmax, all_produce[2], "prefill skipped more than the head");
}

/// The fixture is the contract the builder and the flow agree on. Pinning the
/// fields the flow branches on stops a fixture edit from silently changing
/// what is under test -- and pinning that the two families' configs are
/// otherwise EQUAL is what makes the divergence test above attributable.
#[test]
fn the_two_families_differ_only_in_the_family_tag() {
    let qwen = tiny_gqa_moe_arch(VOCAB, LAYERS, EXPERTS, ModelFamily::Qwen3Moe);
    let llama = tiny_gqa_moe_arch(VOCAB, LAYERS, EXPERTS, ModelFamily::Llama);

    assert_eq!(qwen.family, ModelFamily::Qwen3Moe);
    assert_ne!(qwen.family, llama.family);
    let mut retagged = qwen.clone();
    retagged.family = ModelFamily::Llama;
    assert_eq!(
        retagged, llama,
        "the two fixtures must differ ONLY in the family tag, or the \
         divergence test is measuring something else"
    );

    assert!(qwen.full_attention_layer_mask.iter().all(|&m| m == 1));
    assert!(!qwen.shared_expert_gated);
    assert!(!qwen.attn_output_gate);
    assert!(!qwen.ffn_sandwich_norms);
    assert_eq!(qwen.final_logit_softcap, 0.0);
}
