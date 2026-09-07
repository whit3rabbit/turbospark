//! Runs `dequant_int4_gemv_simd` on real Metal hardware and checks it
//! against the CPU reference in `turbospark_compute::dequant_int4_gemv`.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{dequant_int4_gemv, Int4AffineRowGpu, MetalContext};

/// Deterministic, INDEPENDENT pseudo-random values in `[-1, 1)` (splitmix64
/// on the flat index, matching `attention_tq_parity.rs`'s `unit()`).
///
/// Bounded magnitude is deliberate: the original fixture's `(i - n/2) *
/// scale` grows the row and query values with N itself, so the
/// accumulated FP16 error grows with N for a reason that has nothing to do
/// with the kernel (measured: a probe sweeping N from 128 to 512 under
/// that fixture tracked N smoothly with no discontinuity at the
/// vectorized-block boundary, which is what a magnitude artifact looks
/// like and a real bug in the block loop would not).
///
/// INDEPENDENCE is the second, sharper requirement, and the first attempt
/// here missed it: `((i % 64) - 32) * scale` repeats every 64 elements,
/// which is EXACTLY `dequant_int4.metal`'s `kGroupSize` -- every 64-wide
/// affine-quantization group then holds the identical ramp, so groups
/// quantize to the identical packed byte pattern and a mutation that
/// mis-orders elements WITHIN one group is invisible: whichever element it
/// reads instead, that element's value (and the weight nibble beside it)
/// is the same one it should have read. A splitmix64 hash has no period
/// this kernel's block/group/lane strides can alias against.
fn unit(i: usize, salt: u64) -> f32 {
    let mut z = (i as u64)
        .wrapping_add(salt)
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn run_case(n: usize) {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let m = 5usize; // not a multiple of 8, exercises the early-return guard
    let rows_f32: Vec<Vec<f32>> = (0..m)
        .map(|r| (0..n).map(|i| unit(i, r as u64 * 7 + 1) * 0.6).collect())
        .collect();
    let quantized: Vec<_> = rows_f32
        .iter()
        .map(|row| turbospark_compute::quantize_int4_affine(row))
        .collect();
    let gpu_rows: Vec<Int4AffineRowGpu> = quantized
        .iter()
        .map(|q| Int4AffineRowGpu {
            packed: &q.packed,
            scales: &q.scales,
            biases: &q.biases,
        })
        .collect();

    let x_f32: Vec<f32> = (0..n).map(|i| unit(i, 99991)).collect();
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = turbospark_compute::dequant_int4_gemv(&quantized, &x_f32, n);
    let gpu = dequant_int4_gemv(&mut context, &gpu_rows, &x_f16, n).expect("GPU dispatch succeeds");

    assert_eq!(gpu.len(), cpu.len());
    let err = turbospark_compute::max_abs_diff(
        &gpu.iter().map(|v| v.to_f32()).collect::<Vec<f32>>(),
        &cpu,
    );
    // FP16 accumulation plus int4-affine quantization noise, over N
    // elements of BOUNDED magnitude (see `bounded`'s doc); both this test
    // and the CPU reference use the same quantized (already lossy) rows,
    // so the remaining error is FP16 rounding, not quantization, and does
    // not grow with N the way it would if the fixture's own values did.
    assert!(err < 0.3, "n={n}: err = {err}");
}

/// N=128: two 64-wide groups, `n_groups=2`, `full_blocks = n_groups/4 = 0`
/// (`dequant_int4.metal`'s `kGroupSize` is 64). The vectorized 4-group
/// block loop never runs here -- only the scalar remainder loop does, so
/// every other assertion in this file used to be GPU-vs-GPU on that one
/// path (T4: the block loop had no case of its own at all).
#[test]
fn matches_cpu_reference_scalar_remainder_only_two_groups() {
    run_case(128);
}

/// N=512: eight 64-wide groups, `n_groups=8`, `full_blocks=2`, remainder
/// `8 - 4*2 = 0`. Every group goes through the vectorized block loop and
/// the scalar remainder loop dispatches zero iterations -- the opposite
/// edge from the N=128 case above, pinning the block path without the
/// scalar one riding along to hide a break in it.
#[test]
fn matches_cpu_reference_one_full_vectorized_block_no_remainder() {
    run_case(512);
}

/// N=320: five 64-wide groups, `n_groups=5`, `full_blocks=1`, remainder
/// `5 - 4*1 = 1`. One full 4-group vectorized block PLUS a one-group
/// scalar remainder in the same dispatch -- the seam between the two
/// loops, which neither N=128 (scalar only) nor N=512 (vectorized only)
/// can see.
#[test]
fn matches_cpu_reference_one_full_block_plus_one_group_remainder() {
    run_case(320);
}
