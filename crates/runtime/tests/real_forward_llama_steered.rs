#![cfg(target_os = "macos")]
//! Does the `llama` flow dispatch the steering edit, on BOTH of its halves?
//!
//! Steering was the qwen flow's alone until 2026-08-24 and `real_forward_open`
//! refused a direction set anywhere else BY NAME, on the argument that a set
//! which loaded, reported itself on the startup line and changed nothing is
//! the silent no-op the whole surface exists to avoid. Lifting that refusal
//! for a family means proving the dispatch is really there, and this file is
//! that proof for `families/llama/`.
//!
//! # Why the dense and MoE halves are separate cases rather than one
//!
//! **They are two call sites, and neither can see the other's absence.** The
//! dense branch `continue`s out of the layer loop before the routed code is
//! reached, so its hook sits above that `continue` while the MoE hook sits at
//! the bottom of the loop -- and the MoE one encodes into the ROUTED command
//! buffer, because the router's top-k forced a commit in between. Deleting
//! either leaves a family half-steered: the model stays fluent, every logit
//! stays finite, and only the tokens change. `crates/repack`'s Gotcha on
//! shared architectures (AGENTS.md Gotcha 61) is the same shape one layer up
//! -- a condition written for one half of a shared flow is a latent bug for
//! exactly as long as nobody runs the other half.
//!
//! The MoE case additionally covers `Qwen3Moe`, which runs this same flow.
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
use turbospark_repack::{build_synthetic_dense_llama_install, build_synthetic_llama_real_install};
use turbospark_runtime::{LogitProducer, RealForwardRunner, SteeringPolicy};

const VOCAB: i64 = 128;
const LAYERS: i64 = 4;
const EXPERTS: i64 = 4;

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-llama-steered-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A direction on every layer, deterministic and varying with BOTH the layer
/// and the element.
///
/// Varying with the layer is what makes `only_this_layer` below discriminate:
/// a constant direction would make every layer's edit the same vector, so a
/// bug that read layer 0's offset for every layer would still agree with
/// itself. Copied in shape from `real_forward_qwen35_steered_batched.rs` so
/// the two families' fixtures are comparable.
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
///
/// A SPARSE set is the normal case for this surface, so this is not an exotic
/// input: it is what steering a narrow band looks like on the wire.
fn only_layer(hidden: usize, layers: usize, keep: usize) -> SteeringSet {
    let mut set = direction_set(hidden, layers);
    for (l, slot) in set.layers.iter_mut().enumerate() {
        if l != keep {
            *slot = None;
        }
    }
    set
}

/// The width comes from the INSTALL, never from a literal: `SteeringSet::
/// validate` refuses a width that disagrees with the model, and a hardcoded
/// one fails for a reason unrelated to what the case asks.
fn policy(set: SteeringSet, alpha: f32) -> SteeringPolicy {
    SteeringPolicy::single(set, foundation::SteeringMode::Ablate, alpha, 0.0, 0.0)
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
        turbospark_runtime::ExpertCacheSlots::Fixed(8),
        // No drafter: this family has none, and the verify pass is not what
        // is under test.
        turbospark_runtime::DraftPolicies::off(),
        steering,
    )
    .expect("a steered llama install opens")
}

/// Four tokens through a fresh runner, every logit returned.
///
/// More than one position on purpose: at position 0 a softmax over one key is
/// exactly 1.0, so several transforms are unobservable there (AGENTS.md
/// Gotcha 51's second half). The edit itself would show at position 0, but a
/// case that only ever reads position 0 is the shape that has silently
/// measured nothing twice in this repo.
fn run(runner: &mut RealForwardRunner) -> Vec<f16> {
    let vocab = VOCAB as usize;
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

/// The dense half: Mistral, Llama 2/3.x. Its hook sits above the layer
/// loop's `continue`.
#[test]
fn the_dense_llama_flow_dispatches_the_steering_edit() {
    let dir = temp_dir("dense");
    let arch = build_synthetic_dense_llama_install(&dir, VOCAB, LAYERS, "tiny-mistral")
        .expect("dense llama install builds");
    let hidden = arch.hidden_size as usize;
    let layers = arch.num_layers as usize;

    let plain = run(&mut RealForwardRunner::open(&dir, arch.clone()).expect("opens"));
    let zero = run(&mut open_steered(
        &dir,
        arch.clone(),
        policy(direction_set(hidden, layers), 0.0),
    ));
    let steered = run(&mut open_steered(
        &dir,
        arch.clone(),
        policy(direction_set(hidden, layers), 1.0),
    ));

    // THE NULL CONTROL. `alpha = 0.0` is the exact identity in every mode, so
    // this must be BIT-identical to an engine opened with no set at all --
    // which is also what says the steered arm below is measuring the edit
    // rather than the presence of the machinery.
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
        "the dense llama flow did not dispatch the edit: a full-strength ablation on every \
         layer left all {} logits untouched",
        plain.len()
    );
    println!("dense: alpha 1.0 moved {moved} of {} logits", plain.len());
}

/// The MoE half: Mixtral, and `Qwen3Moe` through the same flow. Its hook sits
/// at the bottom of the layer loop and encodes into the ROUTED command
/// buffer, not cb1.
#[test]
fn the_moe_llama_flow_dispatches_the_steering_edit() {
    let dir = temp_dir("moe");
    let arch = build_synthetic_llama_real_install(&dir, VOCAB, LAYERS, EXPERTS, "tiny-mixtral")
        .expect("moe llama install builds");
    let hidden = arch.hidden_size as usize;
    let layers = arch.num_layers as usize;

    let plain = run(&mut RealForwardRunner::open(&dir, arch.clone()).expect("opens"));
    let zero = run(&mut open_steered(
        &dir,
        arch.clone(),
        policy(direction_set(hidden, layers), 0.0),
    ));
    let steered = run(&mut open_steered(
        &dir,
        arch.clone(),
        policy(direction_set(hidden, layers), 1.0),
    ));

    assert_eq!(
        differing(&plain, &zero),
        0,
        "alpha 0 is not the identity on the MoE half"
    );
    let moved = differing(&plain, &steered);
    assert!(
        moved > 0,
        "the MoE llama flow did not dispatch the edit: a full-strength ablation on every \
         layer left all {} logits untouched. Note the dense case can pass while this fails \
         -- they are two call sites",
        plain.len()
    );
    println!("moe: alpha 1.0 moved {moved} of {} logits", plain.len());
}

/// A set covering ONE layer must steer that layer and no other.
///
/// This is what says the per-layer offset table is honoured rather than one
/// direction being applied everywhere -- a bug that would leave both cases
/// above perfectly green, since a wrong offset still reads a valid direction
/// out of a valid buffer and still moves the logits.
///
/// The two arms must DIFFER FROM EACH OTHER as well as from unsteered.
/// Asserting only "each moves the logits" would pass against a flow that
/// ignored the table entirely.
#[test]
fn a_one_layer_set_steers_that_layer_and_not_another() {
    let dir = temp_dir("perlayer");
    let arch = build_synthetic_dense_llama_install(&dir, VOCAB, LAYERS, "tiny-mistral")
        .expect("dense llama install builds");
    let hidden = arch.hidden_size as usize;
    let layers = arch.num_layers as usize;
    assert!(layers >= 2, "the case needs two distinct layers to compare");

    let plain = run(&mut RealForwardRunner::open(&dir, arch.clone()).expect("opens"));
    let first = run(&mut open_steered(
        &dir,
        arch.clone(),
        policy(only_layer(hidden, layers, 0), 1.0),
    ));
    let last = run(&mut open_steered(
        &dir,
        arch.clone(),
        policy(only_layer(hidden, layers, layers - 1), 1.0),
    ));

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
