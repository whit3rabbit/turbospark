#![cfg(target_os = "macos")]
//! `dequant_int4_gemm_simd` (B right-hand sides, one dispatch) must agree
//! with `dequant_int4_gemv_simd` run B times (ROADMAP Phase D2).
//!
//! The GEMV is the reference rather than the CPU kernel on purpose. It is
//! already parity-tested against `turbospark_compute::dequant_int4_gemv`
//! (`dequant_int4_gemv_parity.rs`), so going through it chains to the same
//! ground truth AND pins the property that actually matters: a batched
//! verify must produce the identical bytes a sequential decode would, or
//! speculative decoding stops being lossless. That is an EXACT comparison,
//! not a tolerance -- both paths accumulate the same terms in the same
//! order into the same FP32 accumulator, so any difference is a bug.

use half::f16;
use turbospark_gpu::{
    autorelease_pool, encode_dequant_int4_gemm_resident, encode_dequant_int4_gemv_resident,
    Int4ResidentMatrix, MetalContext, MAX_BATCH_ROWS,
};

/// Deterministic pseudo-random bytes; the values only have to be varied,
/// and a fixed generator keeps the test reproducible without a fixture.
fn fill(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u8
        })
        .collect()
}

/// BF16 bit patterns in a small positive range, so the dequantized values
/// stay well inside FP16 and the comparison tests the kernel rather than
/// an overflow (the fixture trap recorded in CLAUDE.local.md).
fn bf16_small(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(len * 2);
    for _ in 0..len {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // 0.0078125 .. 0.0625, exact binary fractions.
        let exponent = 0x3C00u16 + (((state >> 40) as u16 & 3) << 7);
        out.extend_from_slice(&exponent.to_le_bytes());
    }
    out
}

fn half_bytes(values: &[f16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect()
}

#[test]
fn batched_gemm_matches_the_gemv_run_once_per_row() {
    let rows = 128usize;
    let cols = 256usize;
    let mut context = MetalContext::new().expect("Metal device");

    let scales_offset = (rows * cols / 2) as u64;
    let biases_offset = scales_offset + (rows * cols / 64 * 2) as u64;
    let total = biases_offset + (rows * cols / 64 * 2) as u64;

    let mut blob = fill(0xD1A5, scales_offset as usize);
    blob.extend_from_slice(&bf16_small(0xBEEF, rows * cols / 64));
    blob.extend_from_slice(&bf16_small(0xF00D, rows * cols / 64));
    assert_eq!(blob.len(), total as usize);
    let weights = context.new_buffer_with_data(&blob);

    for batch in [1usize, 2, 5, MAX_BATCH_ROWS] {
        // Each row of `x` differs, so a kernel that broadcast one row into
        // every accumulator would fail rather than pass by symmetry.
        let x_values: Vec<f16> = (0..batch * cols)
            .map(|i| f16::from_f32(((i % 17) as f32 - 8.0) / 32.0))
            .collect();
        let x = context.new_buffer_with_data(&half_bytes(&x_values));
        let y_batched = context.new_output_buffer((batch * rows * 2) as u64);
        let y_single = context.new_output_buffer((rows * 2) as u64);

        let matrix = || Int4ResidentMatrix {
            buffer: &weights,
            weights_offset: 0,
            scales_offset,
            biases_offset,
            rows,
            cols,
        };

        autorelease_pool(|| {
            let pass = context.begin_pass();
            encode_dequant_int4_gemm_resident(
                &mut context,
                &pass,
                &matrix(),
                (&x, 0),
                (&y_batched, 0),
                batch,
            )
            .expect("batched dispatch");
            pass.commit_and_wait();
        });
        let got = turbospark_gpu::read_buffer_f16(&y_batched, 0, batch * rows);

        for b in 0..batch {
            autorelease_pool(|| {
                let pass = context.begin_pass();
                encode_dequant_int4_gemv_resident(
                    &mut context,
                    &pass,
                    &matrix(),
                    (&x, (b * cols * 2) as u64),
                    (&y_single, 0),
                )
                .expect("single dispatch");
                pass.commit_and_wait();
            });
            let expected = turbospark_gpu::read_buffer_f16(&y_single, 0, rows);
            let slice = &got[b * rows..(b + 1) * rows];
            assert_eq!(
                slice.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "batch {batch}, row {b} differs from the GEMV"
            );
            // The fixture has to be able to distinguish rows at all.
            assert!(
                expected.iter().any(|v| v.to_f32() != 0.0),
                "fixture produced an all-zero output, so this proves nothing"
            );
        }
    }
}
