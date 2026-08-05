//! Runs `rope_proportional_neox` on real Metal hardware and checks it
//! against the CPU reference in `mrefrust_compute::rope_neox`.
#![cfg(target_os = "macos")]

use half::f16;
use mrefrust_gpu::{rope_proportional_neox, MetalContext};

#[test]
fn matches_cpu_reference_within_fp16_tolerance() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let num_tokens = 1u32;
    let num_heads = 2u32;
    let head_dim = 8u32;
    let rotated_pairs = 4u32; // full head: head_dim / 2
    let position = 5u32;
    let theta = 10000.0f32;

    let len = (num_tokens * num_heads * head_dim) as usize;
    let input_f32: Vec<f32> = (0..len)
        .map(|i| (i as f32 - len as f32 / 2.0) * 0.05)
        .collect();
    let input_f16: Vec<f16> = input_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = mrefrust_compute::rope_neox(
        &input_f32,
        num_tokens as usize,
        num_heads as usize,
        head_dim as usize,
        rotated_pairs as usize,
        position as usize,
        theta,
    );
    let gpu = rope_proportional_neox(
        &mut context,
        &input_f16,
        position,
        num_tokens,
        num_heads,
        head_dim,
        rotated_pairs,
        theta,
    )
    .expect("GPU dispatch succeeds");

    assert_eq!(gpu.len(), cpu.len());
    let err =
        mrefrust_compute::max_abs_diff(&gpu.iter().map(|v| v.to_f32()).collect::<Vec<f32>>(), &cpu);
    assert!(
        err < mrefrust_compute::Tolerance::FP16_REDUCTION,
        "err = {err}"
    );
}

#[test]
fn zero_position_is_identity() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    let data: Vec<f16> = (0..8).map(|i| f16::from_f32(i as f32)).collect();
    let out = rope_proportional_neox(&mut context, &data, 0, 1, 1, 8, 4, 10000.0).unwrap();
    for (a, b) in out.iter().zip(data.iter()) {
        assert!((a.to_f32() - b.to_f32()).abs() < 1e-2, "a={a:?} b={b:?}");
    }
}
