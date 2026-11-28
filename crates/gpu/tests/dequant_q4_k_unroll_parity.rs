//! The KERN-04 Stage B unroll arms (`dequant_q4_k_gemv_wide_unroll`) against
//! the CPU reference, in their own file so `dequant_q4_k_gemv_parity.rs` stays
//! as the evaluator's base had it. `weights` and `assert_matches` are copied
//! from there, tolerance unchanged.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{dequant_q4_k_gemv_resident_wide_unroll, MetalContext, Q4KResidentMatrix};

const SUPERBLOCK: usize = 256;

fn weights(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 23) as f32) - 1.0
        })
        .collect()
}

fn assert_matches(label: &str, gpu: &[f16], cpu: &[f32], n: usize) {
    assert_eq!(gpu.len(), cpu.len());
    let gpu_f32: Vec<f32> = gpu.iter().map(|v| v.to_f32()).collect();
    let err = turbospark_compute::max_abs_diff(&gpu_f32, cpu);
    let scale = cpu.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1.0);
    // FP16 has 10 mantissa bits, and the kernel rounds once at the end while
    // accumulating in FP32, so the bound scales with the result magnitude
    // and only weakly with N.
    let bound = scale * 1e-2 + (n as f32) * 1e-4;
    assert!(err < bound, "{label}: err = {err}, bound = {bound}");
}

/// The Stage B unroll arms against the CPU reference. Five superblocks per
/// row, so U = 2 and U = 4 both run a tail through accumulator 0, and each U
/// runs in the same context: a pipeline key missing U would serve the first
/// U's pipeline to the others and the tail cases would fail.
#[test]
fn the_unrolled_wide_arms_match_cpu_reference_with_a_tail() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    let (m, n) = (11usize, 5 * SUPERBLOCK);
    let rows: Vec<Vec<u8>> = (0..m)
        .map(|r| turbospark_compute::quantize_q4_k(&weights(n, 600 + r as u32)))
        .collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
    let blob: Vec<u8> = rows.concat();
    let buffer = context.new_buffer_with_data(&blob);
    let x_f32 = weights(n, 4242);
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let cpu = turbospark_compute::dequant_q4_k_gemv(&refs, &x_f32, n);
    for unroll in 1..=4 {
        let gpu = dequant_q4_k_gemv_resident_wide_unroll(
            &mut context,
            &Q4KResidentMatrix {
                buffer: &buffer,
                weights_offset: 0,
                rows: m,
                cols: n,
            },
            &x_f16,
            unroll,
        )
        .expect("GPU dispatch succeeds");
        assert_matches(&format!("wide unroll {unroll}"), &gpu, &cpu, n);
    }
}
