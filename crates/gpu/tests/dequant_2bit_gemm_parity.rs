//! Holds `dequant_int2_gemm_simd` to its parity contract on real Metal
//! (ROADMAP P3.3): for EVERY batch width the verify pass can ask for, B rows
//! of the batched kernel must be BIT-IDENTICAL to B calls of the GEMV.
//!
//! Same gate and same transitive pin as the 1-bit file beside this one --
//! with one trap that is WORTH MORE here, not less: at two bits a wrong
//! field order permutes elements within a byte and leaves every magnitude,
//! every group scale and the whole level histogram untouched (AGENTS.md
//! Gotcha 48). The batch kernel copies the GEMV's field extraction, so a
//! permutation would have to enter BOTH to survive; bit-exactness with the
//! GEMV is what makes that impossible, and the GEMV's own field order is
//! pinned to the MLX oracle one reference away.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_compute::quant_2bit::{quantize_int2_affine_ternary, TERNARY_GROUP_SIZE};
use turbospark_gpu::{
    dequant_int2_gemm_resident, dequant_int2_gemv_resident, Int2ResidentMatrix, MetalContext,
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

fn assert_batch_matches_sequential_gemvs(batch: usize) {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let g = TERNARY_GROUP_SIZE;
    let n = 4 * g;
    let m = 5usize;
    let rows: Vec<_> = (0..m)
        .map(|r| quantize_int2_affine_ternary(&pseudo(n, 11 + r as u32), g))
        .collect();

    // Three planar regions at three nonzero offsets in one buffer, the
    // resident form the runtime actually dispatches.
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
    let w = Int2ResidentMatrix {
        buffer: &buffer,
        weights_offset,
        scales_offset,
        biases_offset,
        rows: m,
        cols: n,
        group_size: g,
    };

    let x: Vec<Vec<f16>> = (0..batch)
        .map(|b| to_f16(&pseudo(n, 500 + b as u32)))
        .collect();
    let mut expected = Vec::with_capacity(batch * m);
    for xb in &x {
        expected.extend(dequant_int2_gemv_resident(&mut context, &w, xb).expect("gemv dispatch"));
    }

    let flat_x: Vec<f16> = x.iter().flatten().copied().collect();
    let batched =
        dequant_int2_gemm_resident(&mut context, &w, &flat_x, batch).expect("batch dispatch");

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

/// The batched output must land where the CPU reference says, within the
/// GEMV parity test's own tolerance -- the reference check that stays
/// meaningful even if both GPU kernels were ever wrong in the same way.
#[test]
fn the_batched_output_matches_the_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let g = TERNARY_GROUP_SIZE;
    let n = 2 * g;
    let m = 3usize;
    let batch = 4usize;
    let rows: Vec<_> = (0..m)
        .map(|r| quantize_int2_affine_ternary(&pseudo(n, 21 + r as u32), g))
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
    let w = Int2ResidentMatrix {
        buffer: &buffer,
        weights_offset: 0,
        scales_offset: (n / 4 * m) as u64,
        biases_offset: (n / 4 * m + n / g * m * 2) as u64,
        rows: m,
        cols: n,
        group_size: g,
    };

    let x: Vec<Vec<f16>> = (0..batch)
        .map(|b| to_f16(&pseudo(n, 90 + b as u32)))
        .collect();
    let flat_x: Vec<f16> = x.iter().flatten().copied().collect();
    let batched =
        dequant_int2_gemm_resident(&mut context, &w, &flat_x, batch).expect("batch dispatch");

    for (bi, xb) in x.iter().enumerate() {
        let x_f32: Vec<f32> = xb.iter().map(|v| v.to_f32()).collect();
        let cpu = turbospark_compute::dequant_int2_gemv(&rows, &x_f32, n);
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

/// TWO batch widths, ONE `MetalContext`. Every other test here builds a
/// fresh context per width, so a pipeline-cache key that dropped the baked
/// `B` could not fire in this file: nothing would ever reuse another
/// width's pipeline. Sharing the context makes the collision reachable --
/// batch 3 compiles first, and a key without `B` serves its pipeline to
/// batch 7, which then reads seven rows of activations through a
/// three-row-shaped kernel and differs from its GEMVs. This is the test
/// that reddens when the key and the baked constants drift apart
/// (crate Gotcha 1's own shape).
#[test]
fn two_batch_widths_in_one_context_cannot_share_a_pipeline() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let g = TERNARY_GROUP_SIZE;
    let n = 2 * g;
    let m = 6usize;
    let rows: Vec<_> = (0..m)
        .map(|r| quantize_int2_affine_ternary(&pseudo(n, 31 + r as u32), g))
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
    let w = Int2ResidentMatrix {
        buffer: &buffer,
        weights_offset: 0,
        scales_offset: (n / 4 * m) as u64,
        biases_offset: (n / 4 * m + n / g * m * 2) as u64,
        rows: m,
        cols: n,
        group_size: g,
    };

    for batch in [3usize, 7] {
        let x: Vec<Vec<f16>> = (0..batch)
            .map(|b| to_f16(&pseudo(n, 700 + b as u32)))
            .collect();
        let mut expected = Vec::with_capacity(batch * m);
        for xb in &x {
            expected
                .extend(dequant_int2_gemv_resident(&mut context, &w, xb).expect("gemv dispatch"));
        }
        let flat_x: Vec<f16> = x.iter().flatten().copied().collect();
        let batched =
            dequant_int2_gemm_resident(&mut context, &w, &flat_x, batch).expect("batch dispatch");
        let differing = batched
            .iter()
            .zip(expected.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            differing, 0,
            "batch {batch} on a context already used by another width: {differing} \
             outputs differ. If the pipeline key dropped the baked B, this width was \
             served a pipeline compiled for a different one."
        );
    }
}
