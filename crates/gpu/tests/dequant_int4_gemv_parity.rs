//! Runs `dequant_int4_gemv_simd` on real Metal hardware and checks it
//! against the CPU reference in `mrefrust_compute::dequant_int4_gemv`.
#![cfg(target_os = "macos")]

use half::f16;
use mrefrust_gpu::{dequant_int4_gemv, Int4AffineRowGpu, MetalContext};

#[test]
fn matches_cpu_reference_within_quant_tolerance() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let n = 128usize; // two groups of 64
    let m = 5usize; // not a multiple of 8, exercises the early-return guard
    let rows_f32: Vec<Vec<f32>> = (0..m)
        .map(|r| {
            (0..n)
                .map(|i| ((i as f32 + r as f32 * 3.0) - (n as f32 / 2.0)) * 0.02)
                .collect()
        })
        .collect();
    let quantized: Vec<_> = rows_f32
        .iter()
        .map(|row| mrefrust_compute::quantize_int4_affine(row))
        .collect();
    let gpu_rows: Vec<Int4AffineRowGpu> = quantized
        .iter()
        .map(|q| Int4AffineRowGpu {
            packed: &q.packed,
            scales: &q.scales,
            biases: &q.biases,
        })
        .collect();

    let x_f32: Vec<f32> = (0..n)
        .map(|i| ((i as f32) - (n as f32 / 2.0)) * 0.03)
        .collect();
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = mrefrust_compute::dequant_int4_gemv(&quantized, &x_f32, n);
    let gpu = dequant_int4_gemv(&mut context, &gpu_rows, &x_f16, n).expect("GPU dispatch succeeds");

    assert_eq!(gpu.len(), cpu.len());
    let err =
        mrefrust_compute::max_abs_diff(&gpu.iter().map(|v| v.to_f32()).collect::<Vec<f32>>(), &cpu);
    // FP16 accumulation plus int4-affine quantization noise, over 128
    // elements; both this test and the CPU reference use the same
    // quantized (already lossy) rows, so the remaining error is FP16
    // rounding, not quantization.
    assert!(err < 1.0, "err = {err}");
}
