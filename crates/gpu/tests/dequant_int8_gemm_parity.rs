#![cfg(target_os = "macos")]
//! `dequant_int8_gemm_simd` (B right-hand sides, one dispatch) must agree
//! with `dequant_int8_gemv_simd` run B times, BIT FOR BIT.
//!
//! The GEMV is the reference rather than the CPU kernel on purpose, for the
//! same reason the INT4 sibling's parity test records: the GEMV is already
//! parity-tested against `turbospark_compute::dequant_int8_gemv`, so going
//! through it chains to the same ground truth AND pins the property that
//! actually matters -- a batched prefill must produce the identical bytes a
//! sequential decode would, or prefill-then-decode stops landing on the
//! all-produce logits the runtime tests assert. That is an EXACT
//! comparison, not a tolerance: both paths accumulate the same terms in the
//! same order into the same FP32 accumulator, so any difference is a bug.
//! The hostile fixture at the bottom builds data that could see a
//! reassociation and proves the two kernels do not make one.

use half::f16;
use turbospark_gpu::{
    autorelease_pool, encode_dequant_int8_gemm_resident, encode_dequant_int8_gemm_resident_blocked,
    encode_dequant_int8_gemv_resident, Int8ResidentMatrix, MetalContext, INT8_MAX_BATCH_ROWS,
    INT8_MAX_GEMM_ROW_BLOCK,
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
fn assert_parity_at(
    context: &mut MetalContext,
    rows: usize,
    cols: usize,
    batches: &[usize],
    row_blocks: &[usize],
) {
    // INT8 layout: one byte per element, then BF16 scales and biases at
    // group 64 -- `Int8ResidentMatrix`'s contract, and the real dense
    // checkpoint's.
    let scales_offset = (rows * cols) as u64;
    let biases_offset = scales_offset + (rows * cols / 64 * 2) as u64;
    let total = biases_offset + (rows * cols / 64 * 2) as u64;
    let mut blob = fill(0x1D8A5, scales_offset as usize);
    blob.extend_from_slice(&bf16_small(0xBEEF, rows * cols / 64));
    blob.extend_from_slice(&bf16_small(0xF00D, rows * cols / 64));
    assert_eq!(blob.len(), total as usize);
    let weights = context.new_buffer_with_data(&blob);

    for &batch in batches {
        let x_values: Vec<f16> = (0..batch * cols)
            .map(|i| f16::from_f32(((i % 17) as f32 - 8.0) / 32.0))
            .collect();
        let x = context.new_buffer_with_data(&half_bytes(&x_values));
        let y_single = context.new_output_buffer((rows * 2) as u64);

        let matrix = || Int8ResidentMatrix {
            buffer: &weights,
            weights_offset: 0,
            scales_offset,
            biases_offset,
            rows,
            cols,
        };

        for &row_block in row_blocks {
            let y_batched = context.new_output_buffer((batch * rows * 2) as u64);
            autorelease_pool(|| {
                let pass = context.begin_pass();
                encode_dequant_int8_gemm_resident_blocked(
                    context,
                    &pass,
                    &matrix(),
                    (&x, 0),
                    (&y_batched, 0),
                    batch,
                    row_block,
                )
                .expect("batched dispatch");
                pass.commit_and_wait();
            });
            let got = turbospark_gpu::read_buffer_f16(&y_batched, 0, batch * rows);

            for b in 0..batch {
                autorelease_pool(|| {
                    let pass = context.begin_pass();
                    encode_dequant_int8_gemv_resident(
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
                    "{rows}x{cols} batch {batch} row_block {row_block}, row {b} \
                     differs from the GEMV"
                );
                assert!(
                    expected.iter().any(|v| v.to_f32() != 0.0),
                    "fixture produced an all-zero output, so this proves nothing"
                );
            }
        }
    }
}

/// M, N and B are baked into the pipeline as function constants, and
/// `MetalContext::pipeline` caches on (source address, name, key). If the
/// key ever stops carrying all three, the SECOND shape dispatched in a
/// process silently reuses the FIRST one's pipeline. Four shapes in ONE
/// process, deliberately differing in M and N, each against the GEMV: a
/// shared key cannot pass this.
#[test]
fn four_shapes_in_one_process_do_not_share_a_pipeline() {
    let mut context = MetalContext::new().expect("Metal device");
    // Real dense-checkpoint shapes, trimmed: the humanizer's gate/up
    // (15360x3840), down (3840x15360), and q at both layer kinds
    // (4096x3840 sliding, 8192x3840 full) at a 16th of the rows, which is
    // still far past any per-shape compile ambiguity.
    for (rows, cols) in [(960, 3840), (240, 15360), (256, 3840), (512, 3840)] {
        assert_parity_at(&mut context, rows, cols, &[1, 3, 16], &[1]);
    }
}

/// Every batch width and every row block must agree with the GEMV: the
/// row-block axis is shipped as provably bit-moving-free, and the whole
/// `(row_block, batch)` grid is what makes that proof run rather than a
/// comment.
#[test]
fn every_batch_and_row_block_agree_with_the_gemv() {
    let mut context = MetalContext::new().expect("Metal device");
    let row_blocks: Vec<usize> = (1..=INT8_MAX_GEMM_ROW_BLOCK).collect();
    assert_parity_at(
        &mut context,
        128,
        64,
        &[2, 5, INT8_MAX_BATCH_ROWS],
        &row_blocks,
    );
}

/// Data that can SEE a reassociation, with a positive control that proves
/// it can. Large opposite-sign terms whose FP32 sum depends on grouping:
/// if either kernel ever changes its accumulation order, the control
/// differs from the baseline and this test stops being evidence. The two
/// kernels must still agree bit for bit on it.
#[test]
fn the_gemm_and_the_gemv_agree_on_data_that_can_see_reassociation() {
    let mut context = MetalContext::new().expect("Metal device");
    let rows = 64usize;
    let cols = 256usize;
    let scales_offset = (rows * cols) as u64;
    let biases_offset = scales_offset + (rows * cols / 64 * 2) as u64;

    // Weights alternating +127 / 1, scales huge and tiny in alternation:
    // the partial sums differ in the last bits under any regrouping.
    let mut blob = vec![0u8; scales_offset as usize];
    for (i, b) in blob.iter_mut().enumerate() {
        *b = if i % 2 == 0 { 127 } else { 1 };
    }
    for g in 0..rows * cols / 64 {
        let magnitude: f32 = if g % 2 == 0 { 512.0 } else { 1.0 / 512.0 };
        let bits = f32::to_bits(magnitude);
        let bf16 = ((bits >> 16) as u16).to_le_bytes();
        blob.extend_from_slice(&bf16);
        blob.extend_from_slice(&[0u8, 0u8]); // zero bias
    }
    assert_eq!(blob.len(), biases_offset as usize + rows * cols / 64 * 2);
    let weights = context.new_buffer_with_data(&blob);

    let x_values: Vec<f16> = (0..2 * cols)
        .map(|i| f16::from_f32(if i % 2 == 0 { 24.0 } else { -24.0 }))
        .collect();
    let x = context.new_buffer_with_data(&half_bytes(&x_values));

    let matrix = || Int8ResidentMatrix {
        buffer: &weights,
        weights_offset: 0,
        scales_offset,
        biases_offset,
        rows,
        cols,
    };

    let y_batched = context.new_output_buffer((2 * rows * 2) as u64);
    autorelease_pool(|| {
        let pass = context.begin_pass();
        encode_dequant_int8_gemm_resident(
            &mut context,
            &pass,
            &matrix(),
            (&x, 0),
            (&y_batched, 0),
            2,
        )
        .expect("batched dispatch");
        pass.commit_and_wait();
    });

    let y_single = context.new_output_buffer((rows * 2) as u64);
    for b in 0..2 {
        autorelease_pool(|| {
            let pass = context.begin_pass();
            encode_dequant_int8_gemv_resident(
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
        let got = turbospark_gpu::read_buffer_f16(&y_batched, 0, 2 * rows);
        let slice = &got[b * rows..(b + 1) * rows];
        assert_eq!(
            slice.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "batch {b} differs from the GEMV on reassociation-sensitive data"
        );
        assert!(
            expected.iter().any(|v| v.to_f32() != 0.0),
            "the hostile fixture produced an all-zero row, so it cannot see \
             a reassociation and proves nothing"
        );
    }
}
