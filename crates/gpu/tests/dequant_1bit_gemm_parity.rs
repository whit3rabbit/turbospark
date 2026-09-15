//! Holds `dequant_int1_gemm_simd` to its parity contract on real Metal
//! (ROADMAP P3.3): for EVERY batch width the verify pass can ask for, B rows
//! of the batched kernel must be BIT-IDENTICAL to B calls of the GEMV.
//!
//! Bit-exactness is the gate that admits the kernel, not a nice-to-have: a
//! sequential-GEMV fallback is numerically identical, so "close enough"
//! cannot tell a real batched kernel from the sequential engine quietly
//! wearing its name (and measuring like it) -- the reason
//! `encode_gemm_any` refuses rather than loops. The transitively pinned
//! chain is the point: the GEMV is held against the CPU reference, the CPU
//! reference is pinned to a PrismML oracle, and this file holds the batch
//! kernel to the GEMV byte for byte -- so a field-order or companion-dtype
//! bug cannot re-enter through the batch path without reddening here.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_compute::quant_1bit::{quantize_int1_affine_symmetric, BONSAI_GROUP_SIZE};
use turbospark_gpu::{
    dequant_int1_gemm_resident, dequant_int1_gemv_resident, Int1ResidentMatrix, MetalContext,
};

fn pseudo(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 23) as f32) - 1.0
        })
        .collect()
}

fn to_f16(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}

/// One resident matrix, and the (batch x GEMV) vs (one batch call)
/// comparison at every width the verify can ask for.
///
/// The fixture deliberately exercises the guards the kernel actually has:
/// `m` is not a multiple of the rows-per-threadgroup, so the early-return
/// runs, and `n` spans four groups so a group-stride bug is reachable.
fn assert_batch_matches_sequential_gemvs(batch: usize) {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let g = BONSAI_GROUP_SIZE;
    let n = 4 * g;
    let m = 5usize;
    let rows: Vec<_> = (0..m)
        .map(|r| quantize_int1_affine_symmetric(&pseudo(n, 11 + r as u32), g))
        .collect();

    // Three planar regions at three nonzero offsets in one buffer, the
    // resident form the runtime actually dispatches: an offset bug here
    // reads a neighbouring tensor's bytes and decodes plausibly wrong.
    let pad = 4096usize;
    let mut blob = vec![0xABu8; pad];
    let weights_offset = blob.len() as u64;
    for r in &rows {
        blob.extend_from_slice(&r.packed);
    }
    blob.extend_from_slice(&[0xCDu8; 256]);
    let scales_offset = blob.len() as u64;
    for r in &rows {
        for &s in &r.scales {
            blob.extend_from_slice(&s.to_le_bytes());
        }
    }
    let biases_offset = blob.len() as u64;
    for r in &rows {
        for &b in &r.biases {
            blob.extend_from_slice(&b.to_le_bytes());
        }
    }
    let buffer = context.new_buffer_with_data(&blob);
    let w = Int1ResidentMatrix {
        buffer: &buffer,
        weights_offset,
        scales_offset,
        biases_offset,
        rows: m,
        cols: n,
        group_size: g,
    };

    // B independent GEMV calls, one per right-hand side.
    let x: Vec<Vec<f16>> = (0..batch)
        .map(|b| to_f16(&pseudo(n, 500 + b as u32)))
        .collect();
    let mut expected = Vec::with_capacity(batch * m);
    for xb in &x {
        expected.extend(dequant_int1_gemv_resident(&mut context, &w, xb).expect("gemv dispatch"));
    }

    let flat_x: Vec<f16> = x.iter().flatten().copied().collect();
    let batched =
        dequant_int1_gemm_resident(&mut context, &w, &flat_x, batch).expect("batch dispatch");

    assert_eq!(batched.len(), expected.len());
    let differing = batched
        .iter()
        .zip(expected.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        differing,
        0,
        "batch {batch}: {differing} of {} outputs differ from B GEMV calls. The batched \
         kernel's accumulation order is the GEMV's by construction, so ANY difference is \
         a bug in the batching, not rounding.",
        batched.len()
    );
}

#[test]
fn every_batch_width_is_bit_identical_to_b_gemv_calls() {
    for batch in 1..=turbospark_gpu::MAX_BATCH_ROWS {
        assert_batch_matches_sequential_gemvs(batch);
    }
}

/// The batch must also land where the CPU reference says, within the same
/// rounding tolerance the GEMV parity test uses. This is belt-and-braces on
/// top of bit-exactness with the GEMV -- if the GEMV itself were wrong in
/// the same way on both sides of a bad refactor, the comparison above would
/// stay green and only a reference would notice.
#[test]
fn the_batched_output_matches_the_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let g = BONSAI_GROUP_SIZE;
    let n = 2 * g;
    let m = 3usize;
    let batch = 4usize;
    let rows: Vec<_> = (0..m)
        .map(|r| quantize_int1_affine_symmetric(&pseudo(n, 21 + r as u32), g))
        .collect();
    let mut blob: Vec<u8> = Vec::new();
    for r in &rows {
        blob.extend_from_slice(&r.packed);
    }
    for r in &rows {
        for &s in &r.scales {
            blob.extend_from_slice(&s.to_le_bytes());
        }
    }
    for r in &rows {
        for &b in &r.biases {
            blob.extend_from_slice(&b.to_le_bytes());
        }
    }
    let buffer = context.new_buffer_with_data(&blob);
    let w = Int1ResidentMatrix {
        buffer: &buffer,
        weights_offset: 0,
        scales_offset: (n / 8 * m) as u64,
        biases_offset: (n / 8 * m + n / g * m * 2) as u64,
        rows: m,
        cols: n,
        group_size: g,
    };

    let x: Vec<Vec<f16>> = (0..batch)
        .map(|b| to_f16(&pseudo(n, 90 + b as u32)))
        .collect();
    let flat_x: Vec<f16> = x.iter().flatten().copied().collect();
    let batched =
        dequant_int1_gemm_resident(&mut context, &w, &flat_x, batch).expect("batch dispatch");

    for (bi, xb) in x.iter().enumerate() {
        let x_f32: Vec<f32> = xb.iter().map(|v| v.to_f32()).collect();
        let cpu = turbospark_compute::dequant_int1_gemv(&rows, &x_f32, n);
        let gpu: Vec<f32> = batched[bi * m..(bi + 1) * m]
            .iter()
            .map(|v| v.to_f32())
            .collect();
        let err = turbospark_compute::max_abs_diff(&gpu, &cpu);
        let scale = cpu.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1.0);
        let bound = scale * 1e-2 + (n as f32) * 1e-4;
        assert!(err < bound, "batch row {bi}: err = {err}, bound = {bound}");
    }
}
