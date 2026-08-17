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

/// Builds a weight blob and runs the batched kernel against the GEMV for
/// every batch in `batches`, asserting bit equality. Shared so the shape
/// axis below exercises exactly the same comparison.
fn assert_parity_at(context: &mut MetalContext, rows: usize, cols: usize, batches: &[usize]) {
    let scales_offset = (rows * cols / 2) as u64;
    let biases_offset = scales_offset + (rows * cols / 64 * 2) as u64;
    let total = biases_offset + (rows * cols / 64 * 2) as u64;

    let mut blob = fill(0xD1A5, scales_offset as usize);
    blob.extend_from_slice(&bf16_small(0xBEEF, rows * cols / 64));
    blob.extend_from_slice(&bf16_small(0xF00D, rows * cols / 64));
    assert_eq!(blob.len(), total as usize);
    let weights = context.new_buffer_with_data(&blob);

    for &batch in batches {
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
                context,
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
                    context,
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
                "{rows}x{cols} batch {batch}, row {b} differs from the GEMV"
            );
            assert!(
                expected.iter().any(|v| v.to_f32() != 0.0),
                "fixture produced an all-zero output, so this proves nothing"
            );
        }
    }
}

/// M, N and B are baked into the pipeline as function constants, and
/// `MetalContext::pipeline` caches on (source address, name, key). If the
/// key ever stops carrying all three, the SECOND shape dispatched in a
/// process silently reuses the FIRST one's pipeline -- and a wrong baked N
/// reads the wrong fraction of every row while producing finite, plausible
/// output. That is crate Gotcha 1, and it is exactly the trap
/// `gemv_bandwidth_bench.rs::baking_m_n_...` fell into on its own first run
/// (it read +322% from a pipeline baked for a third of the real N).
///
/// Four shapes in ONE process, deliberately differing in M alone, in N
/// alone, and in both, so a key that drops either field is caught. The
/// batch axis is covered by the case above, which walks four batches at one
/// shape in one process for the same reason.
///
/// Mutation-checked: dropping `n` from the key in `specialized_constants`
/// reddens the third shape here, and dropping `m` reddens the second.
#[test]
fn a_second_shape_in_one_process_does_not_reuse_the_first_shapes_pipeline() {
    let mut context = MetalContext::new().expect("Metal device");
    for (rows, cols) in [(128usize, 256usize), (256, 256), (128, 512), (64, 128)] {
        assert_parity_at(&mut context, rows, cols, &[1, 4]);
    }
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
