#![cfg(target_os = "macos")]
//! Does the `muse_glimmer` flow dispatch the steering edit?
//!
//! `families/museglimmer/` has exactly ONE call site: dense, no router, so no
//! mid-layer commit -- the whole token stays on one command buffer, and the
//! hook sits right after the FFN-half sandwich tail's residual add, which is
//! the layer's true output (the same boundary `families/llama/`'s dense
//! branch uses: a raw add, nothing rescaling the stream after it).
//!
//! # What a synthetic fixture can and cannot say here
//!
//! It cannot say the edit is the RIGHT one -- untrained weights make a
//! steered token as meaningless as an unsteered one. What it can say is that
//! the edit is REACHED, that it is the identity at alpha 0, and that it is a
//! function of the per-layer direction rather than of one direction applied
//! everywhere. Whether the boundary is the one llama.cpp uses is a question
//! about the real model and is answered in `docs/OBLITERATION.md`.
//!
//! **THE ALPHA-0 CASE IS NOT DECORATION.** Without it, a hook that dispatched
//! garbage would pass "the edit moves the logits" perfectly well. The pair is
//! the test (AGENTS.md Gotcha 48).

use half::f16;
use model_io::{ArchConfig, LayerDirection, SteeringSet};
use turbospark_repack::build_synthetic_muse_glimmer_install;
use turbospark_runtime::{LogitProducer, RealForwardRunner, SteeringPolicy};

const VOCAB: i64 = 64;
/// A multiple of 4 so the `[0, 0, 0, 1]` sliding-window pattern is whole and
/// the family carries at least two layers to isolate.
const LAYERS: i64 = 8;

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-muse-steered-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn muse_install(dir: &std::path::Path) -> ArchConfig {
    build_synthetic_muse_glimmer_install(dir, VOCAB, LAYERS, "muse-steered")
        .expect("the install writes")
}

/// A direction on every layer, deterministic and varying with BOTH the layer
/// and the element -- copied in shape from `real_forward_llama_steered.rs` so
/// every family's fixture here is comparable.
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
        // This family is dense: no experts, no slot cache. Fixed(8) is inert
        // here and kept only so this call site matches every other family's.
        turbospark_runtime::ExpertCacheSlots::Fixed(8),
        turbospark_runtime::DraftPolicies::off(),
        steering,
    )
    .expect("a steered muse_glimmer install opens")
}

/// Four tokens through a fresh runner, every logit returned.
///
/// More than one position on purpose: at position 0 a softmax over one key is
/// exactly 1.0, so several transforms are unobservable there (AGENTS.md
/// Gotcha 51's second half).
fn run(runner: &mut RealForwardRunner, vocab: usize) -> Vec<f16> {
    let mut out: Vec<f16> = Vec::with_capacity(4 * vocab);
    let mut row = vec![f16::from_f32(0.0); vocab];
    for (position, token) in [1i32, 2, 3, 4].into_iter().enumerate() {
        runner.produce(token, position, &mut row).expect("produce");
        assert!(
            row.iter().all(|v| v.is_finite()),
            "position {position}: a non-finite logit reads as a PERFECT score on every rank \
             instrument downstream (AGENTS.md Gotcha 59)"
        );
        out.extend_from_slice(&row);
    }
    out
}

fn differing(a: &[f16], b: &[f16]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

#[test]
fn the_museglimmer_flow_dispatches_the_steering_edit() {
    let dir = temp_dir("main");
    let arch = muse_install(&dir);
    let hidden = arch.hidden_size as usize;
    let layers = arch.num_layers as usize;
    let vocab = VOCAB as usize;

    let plain = run(
        &mut RealForwardRunner::open(&dir, arch.clone()).expect("opens"),
        vocab,
    );
    let zero = run(
        &mut open_steered(
            &dir,
            arch.clone(),
            policy(direction_set(hidden, layers), 0.0),
        ),
        vocab,
    );
    let steered = run(
        &mut open_steered(
            &dir,
            arch.clone(),
            policy(direction_set(hidden, layers), 1.0),
        ),
        vocab,
    );

    // THE NULL CONTROL. `alpha = 0.0` is the exact identity in every mode, so
    // this must be BIT-identical to an engine opened with no set at all.
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
        "the muse_glimmer flow did not dispatch the edit: a full-strength ablation on every \
         layer left all {} logits untouched",
        plain.len()
    );
    println!(
        "muse_glimmer: alpha 1.0 moved {moved} of {} logits",
        plain.len()
    );
}

/// A set covering ONE layer must steer that layer and no other.
///
/// This is what says the per-layer offset table is honoured rather than one
/// direction being applied everywhere -- a bug that would leave the case
/// above perfectly green, since a wrong offset still reads a valid direction
/// out of a valid buffer and still moves the logits.
#[test]
fn a_one_layer_set_steers_that_layer_and_not_another() {
    let dir = temp_dir("perlayer");
    let arch = muse_install(&dir);
    let hidden = arch.hidden_size as usize;
    let layers = arch.num_layers as usize;
    let vocab = VOCAB as usize;
    assert!(layers >= 2, "the case needs two distinct layers to compare");

    let plain = run(
        &mut RealForwardRunner::open(&dir, arch.clone()).expect("opens"),
        vocab,
    );
    let first = run(
        &mut open_steered(
            &dir,
            arch.clone(),
            policy(only_layer(hidden, layers, 0), 1.0),
        ),
        vocab,
    );
    let last = run(
        &mut open_steered(
            &dir,
            arch.clone(),
            policy(only_layer(hidden, layers, layers - 1), 1.0),
        ),
        vocab,
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
         direction table is not being read -- one direction is reaching every layer",
        layers - 1
    );
    println!(
        "per-layer: first vs last differ in {between} of {} logits",
        plain.len()
    );
}
