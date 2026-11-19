//! Holds `bf16_gemm_rows` to its parity contract on real Metal: B rows of
//! the batched kernel are BIT-IDENTICAL to B calls of `bf16_gemv_rows` at
//! every width the verify can ask for.
//!
//! A GGUF `qwen35` trunk decodes its BF16 `in_proj_a`/`in_proj_b` through
//! that GEMV, so this equality is what keeps a speculative verify over them
//! lossless. ONE context serves every width, so a pipeline key missing B
//! would serve the first width's pipeline to the rest and fail here.
#![cfg(target_os = "macos")]

use half::{bf16, f16};
use turbospark_gpu::{
    bf16_gemm_resident, encode_bf16_gemv_resident, Bf16ResidentMatrix, MetalContext, MAX_BATCH_ROWS,
};

fn lcg(seed: u32) -> impl FnMut() -> f32 {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    move || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((s >> 8) as f32 / (1u32 << 23) as f32) - 1.0
    }
}

fn gemv(context: &mut MetalContext, w: &Bf16ResidentMatrix<'_>, x: &[f16]) -> Vec<f16> {
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let x_buffer = context.new_buffer_with_data(&x_bytes);
    let y_buffer = context.new_output_buffer((w.rows * 2) as u64);
    let pass = context.begin_pass();
    encode_bf16_gemv_resident(context, &pass, w, (&x_buffer, 0), (&y_buffer, 0))
        .expect("gemv dispatch");
    pass.commit_and_wait();
    turbospark_gpu::read_buffer_f16(&y_buffer, 0, w.rows)
}

/// `n` is not a multiple of 256, so some threads stride one element fewer
/// than others, and `m` is odd. The matrix sits behind filler bytes at a
/// nonzero offset, the resident form the runtime dispatches. The second
/// shape is the real 2B checkpoint's `in_proj_a` (16 x 2048).
#[test]
fn every_bf16_batch_width_is_bit_identical_to_b_gemv_calls() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    for (m, n) in [(13usize, 1000usize), (16, 2048)] {
        let mut next = lcg(7 + m as u32);
        let pad = 4096usize;
        let mut blob = vec![0xABu8; pad];
        for _ in 0..m * n {
            blob.extend_from_slice(&bf16::from_f32(next() * 0.5).to_le_bytes());
        }
        blob.extend_from_slice(&[0xCDu8; 256]);
        let buffer = context.new_buffer_with_data(&blob);
        let w = Bf16ResidentMatrix {
            buffer: &buffer,
            weights_offset: pad as u64,
            rows: m,
            cols: n,
        };
        for batch in 1..=MAX_BATCH_ROWS {
            let x: Vec<Vec<f16>> = (0..batch)
                .map(|b| {
                    let mut g = lcg(500 + b as u32);
                    (0..n).map(|_| f16::from_f32(g())).collect()
                })
                .collect();
            let mut expected = Vec::with_capacity(batch * m);
            for xb in &x {
                expected.extend(gemv(&mut context, &w, xb));
            }
            let flat: Vec<f16> = x.iter().flatten().copied().collect();
            let batched =
                bf16_gemm_resident(&mut context, &w, &flat, batch).expect("batch dispatch");
            assert!(expected.iter().any(|v| v.to_f32() != 0.0));
            let differing = batched
                .iter()
                .zip(&expected)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                differing,
                0,
                "{m}x{n} batch {batch}: {differing} of {} outputs differ from B GEMV calls",
                batched.len()
            );
        }
    }
}
