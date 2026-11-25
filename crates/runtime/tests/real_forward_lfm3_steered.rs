#![cfg(target_os = "macos")]
//! Does the `lfm3` flow dispatch the steering edit on both sequential decode
//! and chunked prefill?

use half::f16;
use model_io::{ArchConfig, LayerDirection, SteeringSet};
use turbospark_repack::{build_synthetic_lfm3_install, build_synthetic_lfm3_mlx_install};
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner, SteeringPolicy};

const VOCAB: i64 = 64;
const LAYERS: i64 = 4;

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-lfm3-steered-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

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
        turbospark_runtime::DraftPolicies::off(),
        steering,
    )
    .expect("a steered lfm3 install opens")
}

fn run(runner: &mut RealForwardRunner, vocab: usize) -> Vec<f16> {
    let mut out: Vec<f16> = Vec::with_capacity(4 * vocab);
    let mut row = vec![f16::from_f32(0.0); vocab];
    for (position, token) in [1i32, 2, 3, 4].into_iter().enumerate() {
        runner.produce(token, position, &mut row).expect("produce");
        assert!(
            row.iter().all(|v| v.is_finite()),
            "pos {position}: non-finite"
        );
        out.extend_from_slice(&row);
    }
    out
}

fn differing(a: &[f16], b: &[f16]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

#[test]
fn the_lfm3_flow_dispatches_the_steering_edit() {
    let dir = temp_dir("main");
    let arch = build_synthetic_lfm3_install(&dir, VOCAB, LAYERS, "lfm3-steered").expect("install");
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

    assert_eq!(differing(&plain, &zero), 0, "alpha 0 must be identity");
    assert!(
        differing(&plain, &steered) > 0,
        "alpha 1.0 must move lfm3 logits"
    );
}

#[test]
fn lfm3_chunked_prefill_dispatches_the_steering_edit() {
    let dir = temp_dir("prefill");
    let arch =
        build_synthetic_lfm3_mlx_install(&dir, VOCAB, LAYERS, "lfm3-steered", 64).expect("install");
    let hidden = arch.hidden_size as usize;
    let layers = arch.num_layers as usize;
    let vocab = VOCAB as usize;

    let mut plain_runner = RealForwardRunner::open(&dir, arch.clone()).expect("opens");
    let mut zero_runner = open_steered(
        &dir,
        arch.clone(),
        policy(direction_set(hidden, layers), 0.0),
    );
    let mut steered_runner = open_steered(
        &dir,
        arch.clone(),
        policy(direction_set(hidden, layers), 1.0),
    );
    let mut plain_logits = vec![f16::ZERO; vocab];
    let mut zero_logits = vec![f16::ZERO; vocab];
    let mut steered_logits = vec![f16::ZERO; vocab];
    plain_runner
        .prefill_chunk(&[1, 2, 3, 4], 0, &mut plain_logits)
        .expect("plain prefill");
    zero_runner
        .prefill_chunk(&[1, 2, 3, 4], 0, &mut zero_logits)
        .expect("zero prefill");
    steered_runner
        .prefill_chunk(&[1, 2, 3, 4], 0, &mut steered_logits)
        .expect("steered prefill");

    assert_eq!(
        differing(&plain_logits, &zero_logits),
        0,
        "alpha 0 chunked prefill must be identity"
    );
    assert!(
        differing(&plain_logits, &steered_logits) > 0,
        "chunked prefill must steer lfm3 logits"
    );
}

#[test]
fn a_one_layer_lfm3_set_steers_that_layer_and_not_another() {
    let dir = temp_dir("perlayer");
    let arch = build_synthetic_lfm3_install(&dir, VOCAB, LAYERS, "lfm3-steered").expect("install");
    let hidden = arch.hidden_size as usize;
    let layers = arch.num_layers as usize;
    let vocab = VOCAB as usize;

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

    assert!(differing(&plain, &first) > 0, "layer 0 must steer");
    assert!(differing(&plain, &last) > 0, "last layer must steer");
    assert!(differing(&first, &last) > 0, "layers must steer distinctly");
}
