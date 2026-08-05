//! Runs `rmsnorm_no_scale` on real Metal hardware and checks it against the
//! CPU reference in `mrefrust_compute::rms_norm`, proving the shader
//! compile -> dispatch -> readback path end to end.
#![cfg(target_os = "macos")]

use half::f16;
use mrefrust_gpu::{rms_norm_no_scale, MetalContext};

#[test]
fn matches_cpu_reference_within_fp16_tolerance() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let x_f32: Vec<f32> = (0..256).map(|i| ((i as f32) - 128.0) * 0.03).collect();
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let weight = vec![1.0f32; x_f32.len()];
    let eps = 1e-6f32;

    let cpu = mrefrust_compute::rms_norm(&x_f32, &weight, eps);
    let gpu = rms_norm_no_scale(&mut context, &x_f16, eps).expect("GPU dispatch succeeds");

    assert_eq!(gpu.len(), cpu.len());
    let err =
        mrefrust_compute::max_abs_diff(&gpu.iter().map(|v| v.to_f32()).collect::<Vec<f32>>(), &cpu);
    assert!(
        err < mrefrust_compute::Tolerance::FP16_REDUCTION,
        "err = {err}"
    );
}

#[test]
fn zero_input_produces_zero_output() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    let x = vec![f16::from_f32(0.0); 64];
    let out = rms_norm_no_scale(&mut context, &x, 1e-6).unwrap();
    assert!(out.iter().all(|v| v.to_f32() == 0.0));
}

#[test]
fn repeated_dispatch_reuses_the_cached_pipeline() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    let x: Vec<f16> = (0..32).map(|i| f16::from_f32(i as f32 * 0.1)).collect();
    let first = rms_norm_no_scale(&mut context, &x, 1e-6).unwrap();
    let second = rms_norm_no_scale(&mut context, &x, 1e-6).unwrap();
    assert_eq!(first, second);
}
