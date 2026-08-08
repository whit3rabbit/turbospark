//! Runs `dequant_q8_0_gemv_simd` on real Metal hardware against the CPU
//! reference in `mrefrust_compute::dequant_q8_0_gemv` (ROADMAP Phase G
//! Stage 2). The kernel rule: no quant kernel is trusted before this file
//! exists and passes.
//!
//! Q8_0 needs one test the affine siblings do not. Their quants are
//! unsigned, so a sign error is not a reachable bug; here it is the most
//! likely one, it does not crash, and on symmetric-ish weights it does not
//! even move the magnitude of the result much. It gets a dedicated case
//! with deliberately asymmetric weights rather than being left to the
//! general tolerance.
#![cfg(target_os = "macos")]

use half::f16;
use mrefrust_gpu::{
    dequant_q8_0_gemv, dequant_q8_0_gemv_resident, q8_0_row_bytes, MetalContext, Q8_0ResidentMatrix,
};

fn weights(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 23) as f32) - 1.0
        })
        .collect()
}

/// The CPU reference and the GPU are handed the SAME already-quantized
/// bytes, so quantization error cancels and what is left is FP16 rounding
/// and reduction order. Anything above that is a kernel bug.
fn assert_matches(label: &str, gpu: &[f16], cpu: &[f32], n: usize) {
    assert_eq!(gpu.len(), cpu.len());
    let gpu_f32: Vec<f32> = gpu.iter().map(|v| v.to_f32()).collect();
    let err = mrefrust_compute::max_abs_diff(&gpu_f32, cpu);
    let scale = cpu.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1.0);
    // FP16 has 10 mantissa bits, and the kernel rounds once at the end while
    // accumulating in FP32, so the bound scales with the result magnitude
    // and only weakly with N.
    let bound = scale * 1e-2 + (n as f32) * 1e-4;
    assert!(err < bound, "{label}: err = {err}, bound = {bound}");
}

#[test]
fn matches_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let n = 128usize; // four blocks of 32
    let m = 5usize; // not a multiple of 8: exercises the early-return guard
    let rows: Vec<Vec<u8>> = (0..m)
        .map(|r| mrefrust_compute::quantize_q8_0(&weights(n, 7 + r as u32)))
        .collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();

    let x_f32 = weights(n, 99);
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = mrefrust_compute::dequant_q8_0_gemv(&refs, &x_f32, n);
    let gpu = dequant_q8_0_gemv(&mut context, &refs, &x_f16, n).expect("GPU dispatch succeeds");
    assert_matches("m=5,n=128", &gpu, &cpu, n);
}

/// A single row of all-negative weights against all-positive activations.
/// The exact result is strongly negative. A kernel reading the quants as
/// UNSIGNED returns a strongly POSITIVE number instead, which no tolerance
/// on magnitude would catch, so the sign is asserted outright.
#[test]
fn signed_quants_are_read_as_signed() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let n = 64usize;
    let row = mrefrust_compute::quantize_q8_0(&vec![-0.5f32; n]);
    let refs: Vec<&[u8]> = vec![&row];
    let x_f32 = vec![1.0f32; n];
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = mrefrust_compute::dequant_q8_0_gemv(&refs, &x_f32, n);
    let gpu = dequant_q8_0_gemv(&mut context, &refs, &x_f16, n).expect("GPU dispatch succeeds");

    assert!(
        cpu[0] < -25.0,
        "reference is not decisively negative: {}",
        cpu[0]
    );
    assert!(
        gpu[0].to_f32() < 0.0,
        "the kernel read the quants as unsigned: got {} against {}",
        gpu[0].to_f32(),
        cpu[0]
    );
    assert_matches("all-negative row", &gpu, &cpu, n);
}

/// Many blocks per row, each with a different scale. A kernel that hoisted
/// the scale out of the block loop, or strided it wrongly, passes the
/// four-block case above and fails here.
#[test]
fn per_block_scales_are_not_hoisted() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let n = 1024usize; // 32 blocks
    let m = 8usize;
    // Each block deliberately spans a different magnitude, so the 32 scales
    // within a row differ by orders of magnitude rather than by rounding.
    let rows_f32: Vec<Vec<f32>> = (0..m)
        .map(|r| {
            weights(n, 41 + r as u32)
                .iter()
                .enumerate()
                .map(|(i, &w)| w * 10f32.powi((i / 32) as i32 % 5 - 2))
                .collect()
        })
        .collect();
    let rows: Vec<Vec<u8>> = rows_f32
        .iter()
        .map(|r| mrefrust_compute::quantize_q8_0(r))
        .collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();

    let x_f32 = weights(n, 5);
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = mrefrust_compute::dequant_q8_0_gemv(&refs, &x_f32, n);
    let gpu = dequant_q8_0_gemv(&mut context, &refs, &x_f16, n).expect("GPU dispatch succeeds");
    assert_matches("32 blocks, varying scales", &gpu, &cpu, n);
}

/// The offset-bound form has to land on the same answer as the copying one.
/// Its whole purpose is reading weights in place out of the resident
/// mapping, and an offset bug there reads a neighbouring tensor: finite,
/// plausible, and wrong.
#[test]
fn the_resident_form_matches_the_copying_form() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let n = 256usize;
    let m = 4usize;
    let rows: Vec<Vec<u8>> = (0..m)
        .map(|r| mrefrust_compute::quantize_q8_0(&weights(n, 13 + r as u32)))
        .collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();

    // Put the matrix at a non-zero offset inside a bigger buffer, with
    // garbage in front of it, so an offset of zero cannot pass by accident.
    let row_bytes = q8_0_row_bytes(n);
    let pad = 4096usize;
    let mut blob = vec![0xABu8; pad];
    for r in &rows {
        blob.extend_from_slice(r);
    }
    let buffer = context.new_buffer_with_data(&blob);

    let x_f32 = weights(n, 77);
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = mrefrust_compute::dequant_q8_0_gemv(&refs, &x_f32, n);
    let gpu = dequant_q8_0_gemv_resident(
        &mut context,
        &Q8_0ResidentMatrix {
            buffer: &buffer,
            weights_offset: pad as u64,
            rows: m,
            cols: n,
        },
        &x_f16,
    )
    .expect("GPU dispatch succeeds");

    assert_eq!(row_bytes * m + pad, blob.len());
    assert_matches("resident at offset 4096", &gpu, &cpu, n);
}
