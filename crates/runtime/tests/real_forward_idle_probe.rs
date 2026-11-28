//! tslab's idle-gap probe actions (KERN-05) on a synthetic dense install:
//! `Prewarm` is refused by name where there is no prewarm, `TouchAll` reads
//! exactly the trunk where there are no resident experts, and no action
//! changes the next step's logits.

#![cfg(target_os = "macos")]

use half::f16;
use turbospark_runtime::{IdleProbeAction, LogitProducer, RealForwardError, RealForwardRunner};

fn open_dense(tag: &str) -> RealForwardRunner {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-idle-probe-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    turbospark_repack::build_synthetic_dense_llama_install(&dir, 128, 2, "tiny-mistral")
        .expect("synthetic dense Llama install builds");
    let arch = turbospark_repack::peek_manifest_arch(&dir).expect("arch peeks");
    RealForwardRunner::open(&dir, arch).expect("synthetic dense install opens")
}

fn step(runner: &mut RealForwardRunner) -> Vec<f16> {
    let mut logits = vec![f16::ZERO; runner.vocab_size()];
    runner.reset();
    runner.produce(5, 0, &mut logits).expect("produce");
    logits
}

#[test]
fn prewarm_is_refused_without_resident_experts() {
    let mut runner = open_dense("prewarm");
    let probe = runner.idle_probe().expect("probe pipelines");
    match runner.run_idle_probe(&probe, IdleProbeAction::Prewarm) {
        Err(RealForwardError::Unsupported(why)) => assert!(why.contains("prewarm"), "{why}"),
        other => panic!("expected a named refusal, got {other:?}"),
    }
}

#[test]
fn touch_all_reads_the_trunk_alone_on_a_dense_install_and_no_action_moves_logits() {
    let mut runner = open_dense("touch");
    let probe = runner.idle_probe().expect("probe pipelines");
    let before = step(&mut runner);
    assert_eq!(
        runner
            .run_idle_probe(&probe, IdleProbeAction::Wake)
            .unwrap(),
        0
    );
    let trunk = runner
        .run_idle_probe(&probe, IdleProbeAction::TouchWeights)
        .unwrap();
    assert!(trunk > 0, "the trunk has pages");
    assert_eq!(
        runner
            .run_idle_probe(&probe, IdleProbeAction::TouchAll)
            .unwrap(),
        trunk
    );
    assert_eq!(step(&mut runner), before);
}
