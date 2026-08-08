//! Runs `logit_softcap_softmax` on real Metal hardware and checks it
//! against the CPU reference in `turbospark_compute::logit_softcap_softmax`.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{logit_softcap_softmax, MetalContext};

#[test]
fn matches_cpu_reference_within_fp16_tolerance() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let logits_f32: Vec<f32> = (0..512).map(|i| ((i as f32) - 256.0) * 0.1).collect();
    let logits_f16: Vec<f16> = logits_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let softcap = 30.0f32;

    let cpu = turbospark_compute::logit_softcap_softmax(&logits_f32, softcap);
    let gpu =
        logit_softcap_softmax(&mut context, &logits_f16, softcap).expect("GPU dispatch succeeds");

    assert_eq!(gpu.len(), cpu.len());
    let err = turbospark_compute::max_abs_diff(
        &gpu.iter().map(|v| v.to_f32()).collect::<Vec<f32>>(),
        &cpu,
    );
    assert!(
        err < turbospark_compute::Tolerance::FP16_REDUCTION,
        "err = {err}"
    );

    let sum: f32 = gpu.iter().map(|v| v.to_f32()).sum();
    assert!((sum - 1.0).abs() < 0.05, "sum = {sum}");
}

#[test]
fn small_vocab_does_not_produce_nan() {
    // Smaller than a full threadgroup (256 threads), exercising the
    // empty-SIMD-lane guard the kernel's own comments call out.
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    let logits: Vec<f16> = vec![f16::from_f32(1.0), f16::from_f32(-1.0), f16::from_f32(0.5)];
    let out = logit_softcap_softmax(&mut context, &logits, 30.0).unwrap();
    assert!(out.iter().all(|v| v.to_f32().is_finite()));
}
