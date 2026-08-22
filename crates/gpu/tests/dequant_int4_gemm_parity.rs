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
//!
//! **THAT SENTENCE WAS AN ASSERTION UNTIL 2026-08-21 AND IS A MEASUREMENT
//! NOW**, because three documents had come to blame these two kernels for a
//! real divergence -- a greedy speculative stream parting from a greedy
//! sequential one at ~154 tokens on the real `qwen3_5` install -- on the
//! ground that the two cases here run on a fixture that cannot see a
//! reassociation. That objection was correct about the fixture and wrong
//! about the conclusion: `the_gemm_and_the_gemv_agree_on_data_that_can_see_
//! reassociation` at the bottom of this file builds one that CAN see it,
//! proves it can with a positive control, and the two kernels still agree
//! bit-for-bit at every batch width. The divergence is real and is
//! somewhere else; `docs/DFLASH2.md` carries what is known.

use half::f16;
use turbospark_gpu::{
    autorelease_pool, encode_dequant_int4_gemm_mma_resident, encode_dequant_int4_gemm_resident,
    encode_dequant_int4_gemv_resident, Int4ResidentMatrix, MetalContext, MAX_BATCH_ROWS,
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

// ---------------------------------------------------------------- hostile

/// BF16 bit patterns that are NOT binary fractions, so `s * dot` and
/// `b * sum` actually ROUND.
///
/// The pair above uses exponents alone (0.0078125 .. 0.0625), which scale a
/// sum by shifting its exponent and leave every mantissa bit untouched.
/// These carry seven mantissa bits of their own.
fn bf16_ragged(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(len * 2);
    for _ in 0..len {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Exponent 0x3C..0x3E (0.5 .. 2-ish) with a full 7-bit mantissa.
        let exponent = 0x3Cu16 + ((state >> 40) as u16 & 2);
        let mantissa = (state >> 20) as u16 & 0x7F;
        out.extend_from_slice(&((exponent << 8) | mantissa).to_le_bytes());
    }
    out
}

/// FP16 activations spanning a wide dynamic range with full mantissas, so
/// `e0 + e1 + ... + e7` loses low bits and the loss depends on the order.
fn f16_ragged(seed: u64, len: usize) -> Vec<f16> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // ~1e-3 .. ~1e2, alternating sign: catastrophic cancellation is
            // what makes an order difference show up in the high bits.
            let mag = 0.001f32 * 1.7f32.powi((state >> 32) as i32 % 24);
            let frac = 1.0 + ((state >> 12) & 0x3FF) as f32 / 1024.0;
            let sign = if (state >> 5) & 1 == 0 { 1.0 } else { -1.0 };
            f16::from_f32(sign * mag * frac)
        })
        .collect()
}

/// **THE `dequant_int4_gemm_simd` / `dequant_int4_gemv_simd` PAIR IS
/// BIT-IDENTICAL ON DATA THAT CAN SEE REASSOCIATION, and that is a
/// measurement rather than a restatement of the case above.**
///
/// It exists because the two files' documentation disagreed. This test's own
/// header says the two kernels "accumulate the same terms in the same order",
/// which reading the MSL confirms -- the eight `fma`s into `dot`, the
/// `e0 + ... + e7` into `sum`, the two `fma`s into `acc` and the closing
/// `simd_sum` are the same sequence in both. But `crates/gpu/CLAUDE.md` and
/// `docs/DFLASH2.md` both attribute a real, measured divergence -- a greedy
/// speculative stream parting from a greedy sequential one at ~154 tokens on
/// the real `qwen3_5` install -- to these two kernels accumulating
/// DIFFERENTLY, and record the fixture above as unable to see it.
///
/// Both cannot be true, and the cases above cannot settle it: their scales
/// and biases are exact binary fractions, their activations are `k/32`, and
/// their quants are small integers, so every intermediate is exactly
/// representable in FP32 and NOTHING rounds. A comparison in which no
/// rounding occurs is blind to reassociation by construction, whatever the
/// kernels do.
///
/// So this fixture is built to round, and then ASSERTED to round before its
/// result is believed (AGENTS.md Gotchas 48, 50, 51): a fixture that cannot
/// see the property is worse than no fixture, because it reports green.
#[test]
fn the_gemm_and_the_gemv_agree_on_data_that_can_see_reassociation() {
    let mut context = MetalContext::new().expect("Metal device");
    hostile_parity_at(&mut context, 128, 512, &[1, 2, 8, MAX_BATCH_ROWS]);
}

/// **THE SAME COMPARISON AT THE REAL MODEL'S SHAPES, because the kernel BAKES
/// M, N and B as function constants and a different N is a different
/// pipeline.** An agreement at 128x512 does not transfer to 17408x5120 on its
/// own: the two compile separately, the unroll and register pressure differ,
/// and `n_groups / 4` walks 2 full blocks in one and 68 in the other.
///
/// The shapes are `Qwen/Qwen3.8-27B`'s, read off the install's manifest --
/// hidden 5120, 24 q heads at head_dim 256, FFN 17408 -- so this covers a
/// packed q projection, the KV projections and both FFN orientations. The
/// weights are synthetic; only the DIMENSIONS have to be real for the
/// specialization to be.
///
/// It exists because an earlier elimination of this kernel was run at fixture
/// shapes alone and that gap was worth closing rather than arguing about.
#[test]
fn the_gemm_and_the_gemv_agree_at_the_real_models_shapes() {
    let mut context = MetalContext::new().expect("Metal device");
    // (rows, cols): q_packed, k/v, FFN gate/up, FFN down.
    for (rows, cols) in [(12288, 5120), (1024, 5120), (17408, 5120), (5120, 17408)] {
        println!("shape {rows}x{cols}:");
        hostile_parity_at(&mut context, rows, cols, &[1, 2]);
    }
}

fn hostile_parity_at(context: &mut MetalContext, rows: usize, cols: usize, batches: &[usize]) {
    let scales_offset = (rows * cols / 2) as u64;
    let biases_offset = scales_offset + (rows * cols / 64 * 2) as u64;
    let mut blob = fill(0x5EED, scales_offset as usize);
    blob.extend_from_slice(&bf16_ragged(0xC0FFEE, rows * cols / 64));
    blob.extend_from_slice(&bf16_ragged(0xDECAF, rows * cols / 64));
    let weights = context.new_buffer_with_data(&blob);

    for &batch in batches {
        let x_values = f16_ragged(0xA5A5 + batch as u64, batch * cols);

        // THE DATA MUST BE ORDER-SENSITIVE, and this is the first of two
        // checks that say so. The kernel's per-block activation sum is
        // `e0 + .. + e7`; summing the same eight values the other way round
        // must give DIFFERENT BITS, or this data cannot distinguish any
        // accumulation order from any other.
        let mut order_sensitive = 0usize;
        for chunk in x_values.chunks_exact(8) {
            let fwd = chunk.iter().fold(0f32, |a, v| a + v.to_f32());
            let rev = chunk.iter().rev().fold(0f32, |a, v| a + v.to_f32());
            if fwd.to_bits() != rev.to_bits() {
                order_sensitive += 1;
            }
        }
        assert!(
            order_sensitive * 4 >= x_values.len() / 8,
            "batch {batch}: only {order_sensitive} of {} blocks are order-sensitive; \
             this fixture cannot see reassociation and proves nothing",
            x_values.len() / 8
        );

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
            // Finite BEFORE compared: a NaN row hashes and compares as
            // stably as any other bit pattern, so an all-NaN fixture would
            // report perfect agreement (AGENTS.md Gotcha 59).
            assert!(
                expected.iter().all(|v| v.is_finite()) && slice.iter().all(|v| v.is_finite()),
                "batch {batch} row {b}: non-finite output, so the comparison means nothing"
            );
            let differing = slice
                .iter()
                .zip(expected.iter())
                .filter(|(g, e)| g.to_bits() != e.to_bits())
                .count();
            assert_eq!(
                differing, 0,
                "batch {batch}, row {b}: {differing} of {rows} outputs differ from the GEMV \
                 on reassociation-sensitive data"
            );
        }

        // **THE POSITIVE CONTROL, and without it the green above is worth
        // nothing.** The check on the CPU proves the DATA is order-sensitive;
        // it does not prove this GPU comparison would SHOW a reordering if
        // one happened. A source-level mutation cannot prove that either --
        // Metal's fast-math reassociates freely, so rewriting the sum as
        // `e7 + ... + e0` compiles to the identical code and the mutation
        // survives (verified 2026-08-21, all three cases stayed green).
        //
        // `dequant_int4_gemm_mma` is a kernel KNOWN to reduce in a different
        // order -- it hands K to `simdgroup_multiply_accumulate`, which
        // reduces in hardware in an order Apple does not document, and its
        // own parity test is the only non-exact one in the repo for exactly
        // that reason. Running it on this fixture and requiring it to DIFFER
        // is what says the fixture and the comparison together can see a
        // reassociation. If this ever stops differing, the fixture has gone
        // blind and the assertion above has quietly stopped meaning anything.
        let y_mma = context.new_output_buffer((batch * rows * 2) as u64);
        autorelease_pool(|| {
            let pass = context.begin_pass();
            encode_dequant_int4_gemm_mma_resident(
                context,
                &pass,
                &matrix(),
                (&x, 0),
                (&y_mma, 0),
                batch,
            )
            .expect("mma dispatch");
            pass.commit_and_wait();
        });
        let mma = turbospark_gpu::read_buffer_f16(&y_mma, 0, batch * rows);
        let mma_differing = mma
            .iter()
            .zip(got.iter())
            .filter(|(m, g)| m.to_bits() != g.to_bits())
            .count();
        assert!(
            mma_differing > 0,
            "batch {batch}: the MMA kernel, which reduces in a different order, agrees with \
             the exact kernel to the BIT on this fixture -- so the fixture cannot see a \
             reassociation and the equality asserted above proves nothing"
        );
        println!(
            "batch {batch}: gemm==gemv exactly; mma differs on {mma_differing}/{} outputs \
             ({order_sensitive}/{} activation blocks order-sensitive)",
            batch * rows,
            x_values.len() / 8
        );
    }
}
