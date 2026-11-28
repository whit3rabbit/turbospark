//! The KERN-04 Q8_0 bench arms (`dequant_q8_0_gemv_wide` and `_r2`) against
//! the CPU reference `turbospark_compute::dequant_q8_0_gemv`. `weights` and
//! `assert_matches` are copied from `dequant_q8_0_gemv_parity.rs`, tolerance
//! unchanged; that file is left as it is.
//!
//! Cases a wide Q8_0 kernel can get finite and wrong: block counts that are not
//! a multiple of the 4 blocks a simdgroup takes per step (the stride loop), an
//! odd row count for the two-row arm (the last row has no partner), a matrix at
//! a 2-byte (not 4-byte) aligned offset, which is where 34-byte blocks put half
//! of all block starts, and the int8 sign.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{dequant_q8_0_gemv_resident_wide, MetalContext, Q8_0ResidentMatrix};

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
    let bound = scale * 1e-2 + (n as f32) * 1e-4;
    assert!(err < bound, "{label}: err = {err}, bound = {bound}");
}

#[test]
fn both_wide_arms_match_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    // Blocks per row: 1, 5 and 9 (none a multiple of 4), and 64.
    for blocks in [1usize, 5, 9, 64] {
        let n = blocks * 32;
        for m in [1usize, 17] {
            let rows: Vec<Vec<u8>> = (0..m)
                .map(|r| {
                    turbospark_compute::quantize_q8_0(&weights(n, 31 * blocks as u32 + r as u32))
                })
                .collect();
            let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
            // 2 bytes of garbage in front: a 2-aligned, not 4-aligned, start.
            let pad = 2usize;
            let mut blob = vec![0x7Fu8; pad];
            for r in &rows {
                blob.extend_from_slice(r);
            }
            let buffer = context.new_buffer_with_data(&blob);
            let x_f32 = weights(n, 5000 + blocks as u32);
            let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();
            let cpu = turbospark_compute::dequant_q8_0_gemv(&refs, &x_f32, n);
            for two_rows in [false, true] {
                let gpu = dequant_q8_0_gemv_resident_wide(
                    &mut context,
                    &Q8_0ResidentMatrix {
                        buffer: &buffer,
                        weights_offset: pad as u64,
                        rows: m,
                        cols: n,
                    },
                    &x_f16,
                    two_rows,
                )
                .expect("GPU dispatch succeeds");
                assert_matches(
                    &format!("blocks={blocks} m={m} two_rows={two_rows}"),
                    &gpu,
                    &cpu,
                    n,
                );
            }
        }
    }
}

#[test]
fn an_odd_weights_offset_is_refused() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    let n = 32;
    let mut blob = vec![0u8; 1];
    blob.extend_from_slice(&turbospark_compute::quantize_q8_0(&weights(n, 1)));
    let buffer = context.new_buffer_with_data(&blob);
    let x = vec![f16::from_f32(0.25); n];
    let err = dequant_q8_0_gemv_resident_wide(
        &mut context,
        &Q8_0ResidentMatrix {
            buffer: &buffer,
            weights_offset: 1,
            rows: 1,
            cols: n,
        },
        &x,
        false,
    )
    .expect_err("odd offset");
    assert!(err.to_string().contains("aligned"), "{err}");
}
