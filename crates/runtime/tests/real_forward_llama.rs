#![cfg(target_os = "macos")]
//! End-to-end proof of the `llama`-architecture decode flow (ROADMAP Phase
//! M2): builds a tiny Mixtral-shaped install through the REAL repack pipeline
//! (plain GQA attention, INT8 router, INT4 routed experts, no shared expert),
//! opens it with `RealForwardRunner` -- which selects the flow from
//! `ArchConfig.family` and not from tensor naming -- and drives real decode
//! steps on real Metal.
//!
//! Weights are deterministic but NOT trained, so nothing here asserts on
//! generated TEXT (AGENTS.md Gotcha 12): only token ids, counts, and
//! structural invariants.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use turbospark_repack::{build_synthetic_llama_real_install, tiny_llama_arch};
use turbospark_runtime::{LogitProducer, RealForwardRunner};

static COUNTER: AtomicU64 = AtomicU64::new(0);

const VOCAB: i64 = 128;
const LAYERS: i64 = 4;
const EXPERTS: i64 = 8;

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-real-forward-llama-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn open_runner(dir: &std::path::Path) -> RealForwardRunner {
    let arch = build_synthetic_llama_real_install(dir, VOCAB, LAYERS, EXPERTS, "tiny-mixtral")
        .expect("llama install builds");
    RealForwardRunner::open(dir, arch).expect("llama install opens")
}

/// Argmax-fed greedy decode, asserting the logits contract on every step:
/// finite, and NOT a probability distribution (crate Gotcha 1). This
/// architecture has no softcap, so there is no bound to check instead.
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
fn llama_install_decodes_deterministically() {
    let dir = temp_dir();
    let mut runner = open_runner(&dir);

    let first = greedy_decode(&mut runner, 6);
    let second = greedy_decode(&mut runner, 6);
    assert_eq!(first, second, "two greedy runs must be identical");
}

/// The determinism this flow can actually get wrong: a layer's routed slots
/// are dispatched in the ROUTER'S RANKING, so the same prompt must decode the
/// same way however warm the expert cache is. Running a longer generation in
/// between changes the cache state and nothing else (AGENTS.md Gotcha 27).
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
        "the llama decode path must allocate no Metal buffer per token"
    );
}

/// `produce_prefill` skips the output head and must advance everything else,
/// so a prefill-then-decode run has to agree with an all-`produce` run on the
/// step they share (crate Gotcha 2).
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

/// THE DENSE HALF IS REFUSED, and by name. One `general.architecture` covers
/// dense Llama 2/3.x, Mistral and the Mixtral MoEs; only the MoE half has a
/// flow here, and a dense install must say so at open rather than run on a
/// path that was never written for it.
#[test]
fn a_dense_llama_install_is_refused_at_open() {
    let dir = temp_dir();
    let arch = build_synthetic_llama_real_install(&dir, VOCAB, LAYERS, EXPERTS, "tiny-mixtral")
        .expect("install builds");

    // The manifest has to agree, or `validate_arch` refuses first and this
    // never reaches the FLOW's guard, which is the thing under test. Editing
    // the two counts on disk is cheaper than a second fixture and keeps the
    // failure attributable: without it the message is
    // "manifest.arch.numExperts = 8; expected 0".
    let manifest_path = dir.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();
    manifest["arch"]["numExperts"] = serde_json::json!(0);
    manifest["arch"]["topKExperts"] = serde_json::json!(0);
    std::fs::write(&manifest_path, serde_json::to_string(&manifest).unwrap()).unwrap();

    let mut dense = arch.clone();
    dense.num_experts = 0;
    dense.top_k_experts = 0;
    let err = RealForwardRunner::open(&dir, dense)
        .err()
        .expect("a dense llama install must be refused");
    let message = err.to_string();
    assert!(
        message.contains("DENSE") || message.contains("dense"),
        "the refusal must name the dense half: {message}"
    );
}

/// The tiny arch is the contract the fixture and the flow agree on; pinning
/// the two fields the flow branches on stops a fixture edit from silently
/// changing what is under test.
#[test]
fn the_fixture_is_moe_and_all_full_attention() {
    let arch = tiny_llama_arch(VOCAB, LAYERS, EXPERTS);
    assert_eq!(arch.top_k_experts, 2);
    assert!(arch.full_attention_layer_mask.iter().all(|&m| m == 1));
    assert!(!arch.shared_expert_gated);
    assert!(!arch.attn_output_gate);
}

/// A slot count above the expert count is capped rather than allocated, which
/// matters far more on a COARSE MoE than on a fine-grained one: the slot cache
/// is `slots x layers x expert_stride`, and Mixtral 8x7B's stride is ~109 MiB
/// against Gemma 4's ~3.2 MiB. Asking for 16 slots over 8 experts there would
/// pin 54.5 GiB of host memory to hold a table that is 27.2 GiB whole
/// (ROADMAP Phase M2).
///
/// The fixture is tiny, so this asserts the CAP rather than a byte count: 16
/// slots over 8 experts must behave exactly as 8 do, including in the bytes
/// they decode to.
#[test]
fn a_slot_count_above_the_expert_count_is_capped() {
    let dir = temp_dir();
    let arch = build_synthetic_llama_real_install(&dir, VOCAB, LAYERS, EXPERTS, "tiny-mixtral")
        .expect("install builds");

    const CONTEXT: usize = 512;
    let mut at_capacity =
        RealForwardRunner::open_with_options(&dir, arch.clone(), CONTEXT, EXPERTS as usize)
            .expect("opens with one slot per expert");
    let mut over_capacity =
        RealForwardRunner::open_with_options(&dir, arch, CONTEXT, 4 * EXPERTS as usize)
            .expect("a slot count above the expert count must be capped, not allocated");

    assert_eq!(
        greedy_decode(&mut at_capacity, 4),
        greedy_decode(&mut over_capacity, 4),
        "capping the slot count changed the output"
    );
}
