#![cfg(target_os = "macos")]
//! End-to-end proof of the Qwen 3.6 decode flow: builds a tiny install
//! through the REAL repack pipeline (verbatim Qwen tensor naming,
//! `linear_attn.*` gated-DeltaNet tensors, `.mlp.switch_mlp.` packed
//! experts, INT8 router and sigmoid-gated shared expert, layers
//! alternating linear and gated full attention), opens it with
//! `RealForwardRunner` -- which selects the Qwen flow from
//! `ArchConfig.family` -- and drives real decode steps on real Metal.
//!
//! Weights are deterministic but NOT trained, so nothing here asserts on
//! generated TEXT (AGENTS.md Gotcha 12): only token ids, counts, stop
//! reasons, and structural invariants.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use selection::ShapingConfig;
use tokenizer::MfTokenizer;
use turbospark_repack::build_synthetic_qwen_gdn_moe_install;
use turbospark_runtime::{
    run_raw_completion, DflashDraftPolicy, DraftPolicies, GenerationConfig, LogitProducer,
    MtpDraftPolicy, RawDecodeProgress, RealForwardRunner, SteeringPolicy,
    MOE_SPECULATION_BLOCKER_MARKER,
};

static COUNTER: AtomicU64 = AtomicU64::new(0);

const VOCAB: i64 = 128;
const LAYERS: i64 = 4;
const EXPERTS: i64 = 8;

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-real-forward-qwen-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn chatml_tokenizer() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer loads")
}

fn open_runner(dir: &std::path::Path, vocab: i64) -> RealForwardRunner {
    let arch = build_synthetic_qwen_gdn_moe_install(dir, vocab, LAYERS, EXPERTS, "tiny-qwen36")
        .expect("qwen install builds");
    RealForwardRunner::open(dir, arch).expect("qwen install opens")
}

/// Argmax-fed greedy decode. Also asserts the logits contract on every
/// step: real, finite, and NOT probabilities (Qwen has no softcap, so the
/// bound check the Gemma test uses is replaced by a not-in-[0,1]-simplex
/// check -- see crate Gotcha 1).
fn greedy_decode(runner: &mut RealForwardRunner, steps: usize, vocab: usize) -> Vec<i32> {
    runner.reset();
    let mut token = 5i32;
    let mut out = Vec::new();
    for position in 0..steps {
        let mut head = vec![f16::from_f32(0.0); vocab];
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
fn qwen_install_decodes_deterministically() {
    let dir = temp_dir();
    let mut runner = open_runner(&dir, VOCAB);

    let first = greedy_decode(&mut runner, 6, VOCAB as usize);
    let second = greedy_decode(&mut runner, 6, VOCAB as usize);
    assert_eq!(first, second, "two greedy runs must be identical");
}

/// The teeth on `reset()` for a linear-attention model. Repeating the run
/// above with a DIFFERENT number of leading steps would still pass if
/// `reset` only rewound the KV cache; this one would not, because the GDN
/// state and conv tail carried into the second run would differ from the
/// first run's zeros.
#[test]
fn reset_rewinds_the_gdn_state_and_conv_tail() {
    let dir = temp_dir();
    let mut runner = open_runner(&dir, VOCAB);

    let baseline = greedy_decode(&mut runner, 3, VOCAB as usize);
    // Run a longer generation in between, so any state that survives
    // `reset` is state from a strictly different history.
    let _ = greedy_decode(&mut runner, 9, VOCAB as usize);
    let again = greedy_decode(&mut runner, 3, VOCAB as usize);
    assert_eq!(
        baseline, again,
        "the recurrent GDN state survived reset(): a linear layer's whole \
         history lives there, not in the KV cache"
    );
}

#[test]
fn decode_hot_path_allocates_no_gpu_buffers() {
    let dir = temp_dir();
    let mut runner = open_runner(&dir, VOCAB);

    // Warm up: the first token compiles pipelines and binds blobs.
    let _ = greedy_decode(&mut runner, 2, VOCAB as usize);
    let before = runner.gpu_buffer_allocations();
    let _ = greedy_decode(&mut runner, 4, VOCAB as usize);
    assert_eq!(
        before,
        runner.gpu_buffer_allocations(),
        "the Qwen decode path must allocate no Metal buffer per token"
    );
}

/// `produce_prefill` skips the output head but must advance everything
/// else -- including, on this flow, the GDN recurrence. If it did not, the
/// first real `produce` after a prefill would see a different state than
/// the all-`produce` path and the two would disagree.
#[test]
fn prefill_then_decode_matches_all_produce() {
    let dir = temp_dir();
    let mut runner = open_runner(&dir, VOCAB);
    let prompt = [5i32, 9, 2, 7];

    let mut scratch = vec![f16::from_f32(0.0); VOCAB as usize];
    runner.reset();
    for (position, &token) in prompt.iter().enumerate() {
        runner
            .produce(token, position, &mut scratch)
            .expect("produce");
    }
    let want = scratch.clone();

    runner.reset();
    for (position, &token) in prompt.iter().enumerate() {
        if position + 1 == prompt.len() {
            runner
                .produce(token, position, &mut scratch)
                .expect("produce");
        } else {
            runner
                .produce_prefill(token, position, &mut scratch)
                .expect("produce_prefill");
        }
    }
    assert_eq!(
        scratch, want,
        "prefill-then-decode must land on the same logits as all-produce"
    );
}

/// Greedy is `argmax`, which is invariant under every monotone transform
/// of the distribution, so a greedy-only check cannot see a distribution
/// bug (AGENTS.md Gotcha 16). Run the sampler too.
#[test]
fn runs_through_the_raw_completion_loop_sampled() {
    let (tokenizer, dir) = (chatml_tokenizer(), temp_dir());
    let vocab_size = tokenizer.vocab_size;
    let mut runner = open_runner(&dir, vocab_size as i64);
    let config = GenerationConfig {
        shaping: ShapingConfig::new(0.2, 64, Some(0.95), 1.0, Some(20260721)).unwrap(),
        max_new_tokens: 8,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: Default::default(),
    };
    let prompt_ids = tokenizer.encode("hi", false);
    let mut tokens = Vec::new();
    let result = run_raw_completion(
        &mut runner,
        &tokenizer,
        &prompt_ids,
        &config,
        4096,
        vocab_size,
        |e| {
            if let RawDecodeProgress::Token { id, .. } = e {
                tokens.push(id);
            }
        },
    )
    .expect("sampled qwen forward pass runs to a stop condition");
    assert_eq!(tokens.len(), result.new_tokens);
    assert!(result.new_tokens >= 1);
}

#[test]
fn runs_through_the_raw_completion_loop() {
    let tokenizer = chatml_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    let dir = temp_dir();
    let mut runner = open_runner(&dir, vocab_size as i64);

    let prompt_ids = tokenizer.encode("hi", false);
    assert!(!prompt_ids.is_empty());
    let config = GenerationConfig {
        shaping: ShapingConfig::new(0.0, 0, None, 1.0, None).unwrap(),
        max_new_tokens: 4,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: Default::default(),
    };

    let mut tokens = Vec::new();
    let result = run_raw_completion(
        &mut runner,
        &tokenizer,
        &prompt_ids,
        &config,
        4096,
        vocab_size,
        |e| {
            if let RawDecodeProgress::Token { id, .. } = e {
                tokens.push(id);
            }
        },
    )
    .expect("qwen forward pass runs to a stop condition");

    // Token ids and counts only: these weights are untrained, so the text
    // is meaningless and can even be empty.
    assert_eq!(result.prompt_tokens, prompt_ids.len());
    assert!(result.new_tokens >= 1);
    assert_eq!(tokens.len(), result.new_tokens);
}

/// The mask-2 gate: a linear-attention layer is only accepted under the
/// qwen36 family, and compressed (DeepSeek) layers are refused outright.
#[test]
fn open_rejects_linear_layers_outside_qwen_and_all_compressed_layers() {
    let dir = temp_dir();
    let arch = build_synthetic_qwen_gdn_moe_install(&dir, VOCAB, LAYERS, EXPERTS, "tiny-qwen36")
        .expect("qwen install builds");

    let mut mislabelled = arch.clone();
    mislabelled.family = model_io::ModelFamily::Gemma4;
    let err = match RealForwardRunner::open(&dir, mislabelled) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("mask 2 under gemma4 must be refused"),
    };
    assert!(err.contains("qwen36"), "unexpected error: {err}");

    let mut compressed = arch;
    compressed.full_attention_layer_mask = vec![1, 3, 1, 4];
    let err = match RealForwardRunner::open(&dir, compressed) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("CSA/HCA layers must be refused"),
    };
    assert!(err.contains("compressed"), "unexpected error: {err}");
}

/// Direct evidence that the linear layers ran, not just that decoding
/// produced numbers: after a decode step every mask-2 layer's recurrent
/// state must be non-zero, and every mask-1 layer must have none.
#[test]
fn linear_layers_advance_a_non_zero_recurrent_state() {
    let dir = temp_dir();
    let mut runner = open_runner(&dir, VOCAB);
    for layer in 0..LAYERS as usize {
        assert_eq!(
            runner.gdn_state_abs_max(layer),
            if layer % 2 == 0 { Some(0.0) } else { None },
            "layer {layer} starts at the empty-context state"
        );
    }

    let _ = greedy_decode(&mut runner, 3, VOCAB as usize);
    for layer in (0..LAYERS as usize).step_by(2) {
        let max = runner.gdn_state_abs_max(layer).expect("linear layer");
        assert!(max > 0.0, "layer {layer} recurrent state stayed zero");
        assert!(max.is_finite(), "layer {layer} recurrent state is {max}");
    }

    runner.reset();
    for layer in (0..LAYERS as usize).step_by(2) {
        assert_eq!(runner.gdn_state_abs_max(layer), Some(0.0));
    }
}

/// The MoE half of the speculation capability gate.
///
/// `speculation_blocker` refuses `num_experts != 0` by name, so a MoE install
/// never speculates however good a drafter it acquires. **The reason is a
/// POLICY and not a capability since ROADMAP Phase 3** (`5640c3f`): the
/// batched routed pair exists, is driven by `families/qwen/moe_batch.rs` and
/// is bit-identical to M sequential `produce` calls, so what is missing is a
/// DRAFTER -- no published MoE conversion of this architecture carries one
/// this port can ingest. The blocker has to say THAT rather than "no head",
/// because a reader told the head is missing goes looking for a checkpoint
/// that carries one, and on this family there is none.
///
/// Note this says nothing about decoding. The same install decodes throughout
/// this file; only speculation is refused.
#[test]
fn a_moe_install_reports_the_architectural_blocker_and_not_the_missing_head() {
    let dir = temp_dir();
    let runner = open_runner(&dir, VOCAB);

    let blocker = runner
        .speculation_blocker()
        .expect("a MoE install cannot speculate");
    // The marker rather than a literal, so this cannot go on asserting a
    // sentence the engine has stopped producing (which is what happened to
    // its predecessor, `"dense-only"`).
    assert!(
        blocker.contains(MOE_SPECULATION_BLOCKER_MARKER),
        "the reason must name the missing drafter rather than this install's \
         missing head, got: {blocker}"
    );
    assert_eq!(runner.mtp_draft_depth(), 0);
}

#[test]
fn a_dflash_block_on_an_moe_install_is_refused_for_the_experts_not_the_missing_tensor() {
    let dir = temp_dir();
    let arch = build_synthetic_qwen_gdn_moe_install(&dir, VOCAB, LAYERS, EXPERTS, "tiny-qwen36")
        .expect("qwen install builds");

    let err = RealForwardRunner::open_with_slot_policy_speculation_steering_and_sessions(
        &dir,
        arch,
        4096,
        model_io::ExpertCacheSlots::Fixed(16),
        DraftPolicies {
            mtp: MtpDraftPolicy::Off,
            dflash: DflashDraftPolicy::Fixed(3),
        },
        SteeringPolicy::off(),
        1,
    )
    .err()
    .expect("dflash on MoE install must be refused at open");

    let text = err.to_string();
    assert!(
        text.contains(MOE_SPECULATION_BLOCKER_MARKER),
        "expected error to contain MOE_SPECULATION_BLOCKER_MARKER, got: {text}"
    );
    assert!(
        !text.contains("carries none"),
        "error must name the expert blocker rather than missing tensor, got: {text}"
    );
}
