#![cfg(target_os = "macos")]
//! Does the Gemma 4 flow dispatch the steering edit, on every path that can
//! feed `scratch.x` at a token slot other than 0?
//!
//! `real_forward_llama_steered.rs` is the shape this file copies: an
//! alpha-0 null control beside an alpha-1 arm, and a per-layer isolation
//! case. What this family adds is the THIRD axis neither qwen nor llama
//! has -- a chunked-prefill driver whose per-token routed loop and
//! batched-routed tail write EVERY micro-batch token's row into the SAME
//! `scratch.x` buffer the sequential path uses at offset 0, each at its
//! own slot offset. `steering::encode_steering` and
//! `resid_capture::encode_resid_capture` hardcoded offset 0 until this
//! family needed otherwise (every earlier caller had exactly one row and
//! it always sat at 0), so the case that actually exercises a non-zero
//! offset is the one that would have caught a caller that steered every
//! token's row 0 instead of its own -- fluent, finite, and wrong only in
//! the tokens past the first of a micro-batch.
//!
//! # Why byte-identity against SEQUENTIAL decode, not coherence
//!
//! Same bar `real_forward_gemma4_chunked.rs` sets for the unsteered case:
//! grouping tokens into a command buffer must not change the logits. Here
//! that bar is asked of the STEERED engine -- if the chunk driver steered
//! every token at row 0 of `scratch.x` (the bug this file exists to catch),
//! the chunked and sequential arms would still both be internally
//! consistent and both finite, and would simply disagree with each other,
//! which two chunked runs compared only to EACH OTHER could not show.

use half::f16;
use model_io::{ArchConfig, LayerDirection, SteeringSet};
use turbospark_repack::build_synthetic_gemma4_real_install;
use turbospark_runtime::{
    ChunkedPrefillRunner, ExpertCacheSlots, LogitProducer, RealForwardRunner, SteeringPolicy,
};

const VOCAB: i64 = 128;
const LAYERS: i64 = 2;
const EXPERTS: i64 = 8;
const TOP_K: i64 = 4;
const SLIDING_WINDOW: i64 = 8;
/// Six prompt tokens split into micro-batches of 4 by the tests below, so at
/// least one micro-batch has tokens at slots 1, 2 and 3 -- offsets a
/// hardcoded 0 would silently corrupt or skip.
const PROMPT: [i32; 6] = [3, 7, 1, 5, 9, 2];

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-gemma4-steered-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn build_install(dir: &std::path::Path) -> ArchConfig {
    build_synthetic_gemma4_real_install(
        dir,
        VOCAB,
        LAYERS,
        EXPERTS,
        TOP_K,
        SLIDING_WINDOW,
        "tiny-gemma4-steered",
    )
    .expect("real-naming gemma4 install builds")
}

/// A direction on every layer, deterministic and varying with BOTH the
/// layer and the element -- copied in shape from
/// `real_forward_llama_steered.rs` so the two families' fixtures compare.
fn direction_set(hidden: usize, layers: usize) -> SteeringSet {
    let per_layer = (0..layers)
        .map(|l| {
            let values = (0..hidden)
                .map(|i| {
                    let a = (i as f32 * 0.37 + l as f32 * 1.13).sin();
                    let b = (i as f32 * 0.11).cos();
                    0.5 * a + 0.25 * b
                })
                .collect::<Vec<f32>>();
            Some(LayerDirection::new(values))
        })
        .collect();
    SteeringSet {
        layers: per_layer,
        hidden,
        declared_mode: None,
        declared_arch: None,
    }
}

/// The same set with every layer but `keep` removed.
fn only_layer(hidden: usize, layers: usize, keep: usize) -> SteeringSet {
    let mut set = direction_set(hidden, layers);
    for (l, slot) in set.layers.iter_mut().enumerate() {
        if l != keep {
            *slot = None;
        }
    }
    set
}

fn policy(set: SteeringSet, alpha: f32) -> SteeringPolicy {
    SteeringPolicy {
        set: Some(set),
        mode: foundation::SteeringMode::Ablate,
        alpha,
        target: 0.0,
        gate_threshold: 0.0,
    }
}

fn open_steered(
    dir: &std::path::Path,
    arch: ArchConfig,
    steering: SteeringPolicy,
) -> RealForwardRunner {
    RealForwardRunner::open_with_slot_policy_speculation_and_steering(
        dir,
        arch,
        4096,
        // >= 2 * TOP_K, so the driver's pipelining path (`banks == ROUTED_BANKS`)
        // is reachable and `ExpertCache::plan` never has to abort on a
        // protect-set it cannot honour -- steering changes which experts a
        // later layer routes to (it edits the residual the router reads),
        // so a cache sized to only just cover the UNSTEERED routing can
        // fail placement under the steered one even where `top_k` and
        // `num_experts` are unchanged.
        ExpertCacheSlots::Fixed(16),
        // No drafter: this family has none, and the verify pass is not
        // what is under test.
        turbospark_runtime::DraftPolicies::off(),
        steering,
    )
    .expect("a steered gemma4 install opens")
}

fn differing(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// Every prompt token through `produce`/`produce_prefill`, matching
/// `run_raw_completion`'s own split: every token but the last skips the
/// head, the last one does not.
fn sequential(runner: &mut RealForwardRunner, tokens: &[i32]) -> Vec<f32> {
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let last = tokens.len() - 1;
    for (position, &token) in tokens.iter().enumerate() {
        if position == last {
            runner.produce(token, position, &mut logits)
        } else {
            runner.produce_prefill(token, position, &mut logits)
        }
        .expect("sequential decode succeeds");
    }
    logits.iter().map(|v| v.to_f32()).collect()
}

/// The same prompt through `prefill_chunk`, split into spans of `chunk`.
fn chunked(runner: &mut RealForwardRunner, tokens: &[i32], chunk: usize) -> Vec<f32> {
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let mut offset = 0usize;
    while offset < tokens.len() {
        let take = (tokens.len() - offset).min(chunk);
        runner
            .prefill_chunk(&tokens[offset..offset + take], offset, &mut logits)
            .expect("chunked prefill succeeds");
        offset += take;
    }
    logits.iter().map(|v| v.to_f32()).collect()
}

/// The sequential decode path dispatches the edit at all, and alpha 0 is
/// the exact identity.
///
/// **THE ALPHA-0 CASE IS NOT DECORATION.** Without it, a hook that
/// dispatched garbage would pass "the edit moves the logits" perfectly
/// well (`real_forward_llama_steered.rs`'s reasoning, AGENTS.md Gotcha 48).
#[test]
fn the_gemma4_flow_dispatches_the_steering_edit_in_sequential_decode() {
    let dir = temp_dir("seq");
    let arch = build_install(&dir);
    let hidden = arch.hidden_size as usize;
    let layers = arch.num_layers as usize;

    let plain = sequential(
        &mut RealForwardRunner::open(&dir, arch.clone()).expect("opens"),
        &PROMPT,
    );
    let zero = sequential(
        &mut open_steered(
            &dir,
            arch.clone(),
            policy(direction_set(hidden, layers), 0.0),
        ),
        &PROMPT,
    );
    let steered = sequential(
        &mut open_steered(
            &dir,
            arch.clone(),
            policy(direction_set(hidden, layers), 1.0),
        ),
        &PROMPT,
    );

    assert_eq!(
        differing(&plain, &zero),
        0,
        "alpha 0 is not the identity: {} of {} logits moved with the edit switched off",
        differing(&plain, &zero),
        plain.len()
    );
    let moved = differing(&plain, &steered);
    assert!(
        moved > 0,
        "the gemma4 flow did not dispatch the edit: a full-strength ablation on every layer \
         left all {} logits untouched",
        plain.len()
    );
    println!(
        "sequential: alpha 1.0 moved {moved} of {} logits",
        plain.len()
    );
}

/// A set covering ONE layer must steer that layer and no other, matching
/// `real_forward_llama_steered.rs`'s `a_one_layer_set_steers_that_layer_...`
/// case: this is what says the per-layer offset table is honoured rather
/// than one direction being applied everywhere.
#[test]
fn a_one_layer_set_steers_that_layer_and_not_another() {
    let dir = temp_dir("perlayer");
    let arch = build_install(&dir);
    let hidden = arch.hidden_size as usize;
    let layers = arch.num_layers as usize;
    assert!(layers >= 2, "the case needs two distinct layers to compare");

    let plain = sequential(
        &mut RealForwardRunner::open(&dir, arch.clone()).expect("opens"),
        &PROMPT,
    );
    let first = sequential(
        &mut open_steered(
            &dir,
            arch.clone(),
            policy(only_layer(hidden, layers, 0), 1.0),
        ),
        &PROMPT,
    );
    let last = sequential(
        &mut open_steered(
            &dir,
            arch.clone(),
            policy(only_layer(hidden, layers, layers - 1), 1.0),
        ),
        &PROMPT,
    );

    assert!(
        differing(&plain, &first) > 0,
        "steering layer 0 alone changed nothing"
    );
    assert!(
        differing(&plain, &last) > 0,
        "steering the last layer alone changed nothing"
    );
    let between = differing(&first, &last);
    assert!(
        between > 0,
        "steering layer 0 and steering layer {} produced IDENTICAL logits, so the per-layer \
         direction table is not being read",
        layers - 1
    );
}

/// THE CASE THIS FILE EXISTS FOR: a steered chunked prefill agrees with a
/// steered sequential decode, byte for byte, on a prompt whose micro-batch
/// puts tokens at slots other than 0.
///
/// If `encode_steering` or `encode_resid_capture` steered/captured row 0 of
/// `scratch.x` regardless of which token was being processed -- what a
/// hook copied from the qwen/llama call sites verbatim would do, since
/// neither of those families ever needed anything else -- this would still
/// run, still be finite, and disagree with the sequential arm only in the
/// tokens past the first of each micro-batch. Comparing two chunked runs to
/// each other cannot see that; both would be wrong the same way.
#[test]
fn a_steered_chunked_prefill_agrees_with_steered_sequential_decode() {
    let dir = temp_dir("chunked");
    let arch = build_install(&dir);
    let hidden = arch.hidden_size as usize;
    let layers = arch.num_layers as usize;
    let set = direction_set(hidden, layers);

    let seq = sequential(
        &mut open_steered(&dir, arch.clone(), policy(set.clone(), 1.0)),
        &PROMPT,
    );
    // chunk=4 against a 6-token prompt: the first micro-batch is tokens
    // 0..4 (slots 0, 1, 2, 3), the second is tokens 4..6 (slots 0, 1) --
    // between the two micro-batches, every slot but the very first is
    // exercised at least once.
    let ch = chunked(
        &mut open_steered(&dir, arch.clone(), policy(set, 1.0)),
        &PROMPT,
        4,
    );

    assert!(
        seq.iter().all(|v| v.is_finite()) && ch.iter().all(|v| v.is_finite()),
        "a non-finite logit reads as a PERFECT score on every rank instrument downstream \
         (AGENTS.md Gotcha 59)"
    );
    assert_eq!(
        seq, ch,
        "a steered chunked prefill diverged from a steered sequential decode on the SAME \
         prompt and the SAME direction set: the chunk driver is steering (or capturing) the \
         wrong row of a micro-batch token"
    );
}

/// The unsteered control for the case above: chunked and sequential must
/// already agree with steering OFF, or a divergence in the steered case
/// proves nothing about the edit specifically.
#[test]
fn an_unsteered_chunked_prefill_still_agrees_with_sequential_decode() {
    let dir = temp_dir("chunked-off");
    let arch = build_install(&dir);

    let seq = sequential(
        &mut RealForwardRunner::open(&dir, arch.clone()).expect("opens"),
        &PROMPT,
    );
    let ch = chunked(
        &mut RealForwardRunner::open(&dir, arch.clone()).expect("opens"),
        &PROMPT,
        4,
    );

    assert_eq!(
        seq, ch,
        "chunked and sequential prefill disagree even with steering off; the fixture or the \
         chunk driver is broken independent of steering"
    );
}

/// The BATCHED-routed tail (`moe_batch.rs`, `MFERENCE_ROUTED_BATCH`'s
/// programmatic form) is a SEPARATE call site from the per-token one above
/// and neither can see the other's absence -- exactly the reason
/// `real_forward_llama_steered.rs` keeps the dense and MoE halves as
/// separate cases.
#[test]
fn a_steered_batched_routed_prefill_agrees_with_steered_sequential_decode() {
    let dir = temp_dir("batched-routed");
    let arch = build_install(&dir);
    let hidden = arch.hidden_size as usize;
    let layers = arch.num_layers as usize;
    let set = direction_set(hidden, layers);

    let seq = sequential(
        &mut open_steered(&dir, arch.clone(), policy(set.clone(), 1.0)),
        &PROMPT,
    );

    let mut runner = open_steered(&dir, arch.clone(), policy(set, 1.0));
    runner.set_routed_batch_prefill(true);
    let ch = chunked(&mut runner, &PROMPT, 4);

    assert_eq!(
        seq, ch,
        "a steered batched-routed prefill diverged from steered sequential decode: \
         moe_batch.rs's per-row tail is steering (or capturing) the wrong row"
    );
}
