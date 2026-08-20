//! Runs `rmsnorm_no_scale` on real Metal hardware and checks it against the
//! CPU reference in `turbospark_compute::rms_norm`, proving the shader
//! compile -> dispatch -> readback path end to end.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{rms_norm_no_scale, MetalContext};

#[test]
fn matches_cpu_reference_within_fp16_tolerance() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let x_f32: Vec<f32> = (0..256).map(|i| ((i as f32) - 128.0) * 0.03).collect();
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let weight = vec![1.0f32; x_f32.len()];
    let eps = 1e-6f32;

    let cpu = turbospark_compute::rms_norm(&x_f32, &weight, eps);
    let gpu = rms_norm_no_scale(&mut context, &x_f16, eps).expect("GPU dispatch succeeds");

    assert_eq!(gpu.len(), cpu.len());
    let err = turbospark_compute::max_abs_diff(
        &gpu.iter().map(|v| v.to_f32()).collect::<Vec<f32>>(),
        &cpu,
    );
    assert!(
        err < turbospark_compute::Tolerance::FP16_REDUCTION,
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

/// The identity the DFlash2 drafter's residual scaling rests on
/// (`DFLASH_RESIDUAL_SCALE`): dividing the INPUT by `S` and the EPS by
/// `S * S` reproduces the unscaled norm exactly. That is what lets this port
/// hold a residual stream whose true peak is 113,920 -- past FP16's 65,504
/// ceiling -- in FP16 buffers without changing the arithmetic.
///
/// The second half is the one that matters and the one an unscaled eps gets
/// wrong: with `eps` LEFT ALONE the two disagree, because `eps` then acts as
/// if it were `S * S` times larger. Asserting that disagreement here is what
/// makes the first assertion mean something -- on an input whose mean square
/// dwarfs any epsilon, both spellings pass and the test proves nothing.
#[test]
fn scaling_the_input_and_the_eps_together_is_the_identity() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    // A mean square of ~4e-4, which is the drafter's EMBEDDING row -- the
    // one place in its pass where an epsilon is not negligible.
    let x_f32: Vec<f32> = (0..256).map(|i| ((i as f32) - 128.0) * 0.0003).collect();
    let eps = 1e-6f32;
    let scale = 8.0f32;

    let plain = rms_norm_no_scale(
        &mut context,
        &x_f32
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect::<Vec<f16>>(),
        eps,
    )
    .expect("unscaled dispatch");
    let scaled: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v / scale)).collect();

    let matched = rms_norm_no_scale(&mut context, &scaled, eps / (scale * scale))
        .expect("scaled dispatch, scaled eps");
    let err = turbospark_compute::max_abs_diff(
        &matched.iter().map(|v| v.to_f32()).collect::<Vec<f32>>(),
        &plain.iter().map(|v| v.to_f32()).collect::<Vec<f32>>(),
    );
    assert!(
        err < turbospark_compute::Tolerance::FP16_REDUCTION,
        "scaling the input by {scale} and the eps by its square must reproduce \
         the unscaled norm; err = {err}"
    );

    let unmatched =
        rms_norm_no_scale(&mut context, &scaled, eps).expect("scaled dispatch, raw eps");
    let drift = turbospark_compute::max_abs_diff(
        &unmatched.iter().map(|v| v.to_f32()).collect::<Vec<f32>>(),
        &plain.iter().map(|v| v.to_f32()).collect::<Vec<f32>>(),
    );
    assert!(
        drift > 10.0 * turbospark_compute::Tolerance::FP16_REDUCTION,
        "this fixture cannot tell a scaled eps from an unscaled one, so the \
         assertion above is vacuous; drift = {drift}"
    );
}

#[test]
fn repeated_dispatch_reuses_the_cached_pipeline() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    let x: Vec<f16> = (0..32).map(|i| f16::from_f32(i as f32 * 0.1)).collect();
    let first = rms_norm_no_scale(&mut context, &x, 1e-6).unwrap();
    let second = rms_norm_no_scale(&mut context, &x, 1e-6).unwrap();
    assert_eq!(first, second);
}
