//! Runs `dequant_q4_k_gemv_simd` on real Metal hardware against the CPU
//! reference in `mrefrust_compute::dequant_q4_k_gemv` (ROADMAP Phase G
//! Stage 2). The kernel rule: no quant kernel is trusted before this file
//! exists and passes.
//!
//! Q4_K needs cases its Q8_0 sibling does not, because its superblock has
//! two levels of scale and three separate ways to be finite-but-wrong:
//! the 6-bit sub-scales split across bytes for sub-blocks 4..8, the two
//! nibbles of a byte living 32 elements apart in two different sub-blocks,
//! and a subtracted min that a symmetric-quant habit drops entirely. Each
//! gets a case that a general tolerance would not catch on its own.
#![cfg(target_os = "macos")]

use half::f16;
use mrefrust_gpu::{
    dequant_q4_k_gemv, dequant_q4_k_gemv_resident, q4_k_row_bytes, MetalContext, Q4KResidentMatrix,
};

const SUPERBLOCK: usize = 256;
const SUB: usize = 32;

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

    let n = 2 * SUPERBLOCK;
    let m = 5usize; // not a multiple of 8: exercises the early-return guard
    let rows: Vec<Vec<u8>> = (0..m)
        .map(|r| mrefrust_compute::quantize_q4_k(&weights(n, 7 + r as u32)))
        .collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();

    let x_f32 = weights(n, 99);
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = mrefrust_compute::dequant_q4_k_gemv(&refs, &x_f32, n);
    let gpu = dequant_q4_k_gemv(&mut context, &refs, &x_f16, n).expect("GPU dispatch succeeds");
    assert_matches("m=5,n=512", &gpu, &cpu, n);
}

/// Sub-blocks whose ranges differ by 16x inside one superblock, with the
/// LARGEST at index 5. That puts the maximum 6-bit scale in the second half,
/// where it is stored split across two bytes, and leaves indices 3 and 7 on
/// scales whose only set bit is one of the two high ones. A kernel that
/// reads just the low nibble of `packed[8..12]` still returns finite,
/// correctly signed numbers here; they are simply scaled wrongly per
/// sub-block, which is what the tolerance below rejects.
#[test]
fn six_bit_sub_scales_are_unpacked_across_both_bytes() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    const FACTORS: [f32; 8] = [1.0, 0.5, 4.0, 2.0, 0.5, 8.0, 1.0, 2.0];
    let n = SUPERBLOCK;
    let m = 8usize;
    let rows: Vec<Vec<u8>> = (0..m)
        .map(|r| {
            let scaled: Vec<f32> = weights(n, 41 + r as u32)
                .iter()
                .enumerate()
                .map(|(e, &v)| v * FACTORS[(e / SUB) % 8])
                .collect();
            mrefrust_compute::quantize_q4_k(&scaled)
        })
        .collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();

    let x_f32 = weights(n, 5);
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = mrefrust_compute::dequant_q4_k_gemv(&refs, &x_f32, n);
    let gpu = dequant_q4_k_gemv(&mut context, &refs, &x_f16, n).expect("GPU dispatch succeeds");
    assert_matches("8 sub-blocks, 16x spread", &gpu, &cpu, n);
}

/// A row of all-negative weights against all-positive activations. The exact
/// result is strongly negative, and it can only be negative through the
/// SUBTRACTED sub-block min: Q4_K quants are unsigned. A kernel that drops
/// the min term returns a non-negative number instead, which no tolerance on
/// magnitude would catch, so the sign is asserted outright.
#[test]
fn the_subtracted_min_is_what_carries_the_sign() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let n = SUPERBLOCK;
    // Not a constant row: a flat one has zero range, hence a zero sub-scale,
    // and would be carried entirely by the min term rather than testing both.
    let w: Vec<f32> = (0..n).map(|e| -1.0 + (e % SUB) as f32 / 128.0).collect();
    let row = mrefrust_compute::quantize_q4_k(&w);
    let refs: Vec<&[u8]> = vec![&row];
    let x_f32 = vec![1.0f32; n];
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = mrefrust_compute::dequant_q4_k_gemv(&refs, &x_f32, n);
    let gpu = dequant_q4_k_gemv(&mut context, &refs, &x_f16, n).expect("GPU dispatch succeeds");

    assert!(
        cpu[0] < -200.0,
        "reference is not decisively negative: {}",
        cpu[0]
    );
    assert!(
        gpu[0].to_f32() < 0.0,
        "the kernel dropped the sub-block min: got {} against {}",
        gpu[0].to_f32(),
        cpu[0]
    );
    assert_matches("all-negative row", &gpu, &cpu, n);
}

/// Many superblocks per row, each with a different pair of super-scales. A
/// kernel that hoisted `d`/`dmin` out of the superblock loop, or strided the
/// 144-byte block wrongly, passes the two-superblock case above and fails
/// here.
#[test]
fn per_superblock_scales_are_not_hoisted() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let n = 16 * SUPERBLOCK;
    let m = 8usize;
    let rows: Vec<Vec<u8>> = (0..m)
        .map(|r| {
            // Each superblock spans a different magnitude, so the 16 pairs of
            // super-scales within a row differ by orders of magnitude rather
            // than by rounding.
            let scaled: Vec<f32> = weights(n, 61 + r as u32)
                .iter()
                .enumerate()
                .map(|(e, &w)| w * 10f32.powi((e / SUPERBLOCK) as i32 % 5 - 2))
                .collect();
            mrefrust_compute::quantize_q4_k(&scaled)
        })
        .collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();

    let x_f32 = weights(n, 17);
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = mrefrust_compute::dequant_q4_k_gemv(&refs, &x_f32, n);
    let gpu = dequant_q4_k_gemv(&mut context, &refs, &x_f16, n).expect("GPU dispatch succeeds");
    assert_matches("16 superblocks, varying scales", &gpu, &cpu, n);
}

/// The offset-bound form has to land on the same answer as the copying one.
/// Its whole purpose is reading weights in place out of the resident
/// mapping, and an offset bug there reads a neighbouring tensor: finite,
/// plausible, and wrong.
#[test]
fn the_resident_form_matches_the_copying_form() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let n = SUPERBLOCK;
    let m = 4usize;
    let rows: Vec<Vec<u8>> = (0..m)
        .map(|r| mrefrust_compute::quantize_q4_k(&weights(n, 13 + r as u32)))
        .collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();

    // Put the matrix at a non-zero offset inside a bigger buffer, with
    // garbage in front of it, so an offset of zero cannot pass by accident.
    let row_bytes = q4_k_row_bytes(n);
    let pad = 4096usize;
    let mut blob = vec![0xABu8; pad];
    for r in &rows {
        blob.extend_from_slice(r);
    }
    let buffer = context.new_buffer_with_data(&blob);

    let x_f32 = weights(n, 77);
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = mrefrust_compute::dequant_q4_k_gemv(&refs, &x_f32, n);
    let gpu = dequant_q4_k_gemv_resident(
        &mut context,
        &Q4KResidentMatrix {
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
