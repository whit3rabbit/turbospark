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
    autorelease_pool, dequant_int4_gemm_pipeline_limits, encode_dequant_int4_gemm_mma_resident,
    encode_dequant_int4_gemm_resident, encode_dequant_int4_gemm_resident_blocked,
    encode_dequant_int4_gemv_resident, Int4ResidentMatrix, MetalContext, GEMM_THREADS_PER_GROUP,
    MAX_BATCH_ROWS, MAX_GEMM_ROW_BLOCK,
};

/// The row block every WIRED call site dispatches, and the shape this
/// kernel had before `FC_GEMM_R` existed. Cases that are not about the row
/// axis pass this, so a regression in the shipped shape reddens them.
const SHIPPED_ROW_BLOCK: &[usize] = &[1];

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
        let y_single = context.new_output_buffer((rows * 2) as u64);

        let matrix = || Int4ResidentMatrix {
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
                encode_dequant_int4_gemm_resident_blocked(
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
        assert_parity_at(&mut context, rows, cols, &[1, 4], SHIPPED_ROW_BLOCK);
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
    hostile_parity_at(
        &mut context,
        128,
        512,
        &[1, 2, 8, MAX_BATCH_ROWS],
        SHIPPED_ROW_BLOCK,
    );
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
        hostile_parity_at(&mut context, rows, cols, &[1, 2], SHIPPED_ROW_BLOCK);
    }
}

fn hostile_parity_at(
    context: &mut MetalContext,
    rows: usize,
    cols: usize,
    batches: &[usize],
    row_blocks: &[usize],
) {
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
        let y_single = context.new_output_buffer((rows * 2) as u64);
        let matrix = || Int4ResidentMatrix {
            buffer: &weights,
            weights_offset: 0,
            scales_offset,
            biases_offset,
            rows,
            cols,
        };

        // Kept outside the row-block loop so the MMA positive control below
        // has a `got` to differ from whichever widths were swept.
        let mut got = Vec::new();
        for &row_block in row_blocks {
            let y_batched = context.new_output_buffer((batch * rows * 2) as u64);
            autorelease_pool(|| {
                let pass = context.begin_pass();
                encode_dequant_int4_gemm_resident_blocked(
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
            got = turbospark_gpu::read_buffer_f16(&y_batched, 0, batch * rows);

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
                    "batch {batch} row_block {row_block} row {b}: non-finite output, \
                     so the comparison means nothing"
                );
                let differing = slice
                    .iter()
                    .zip(expected.iter())
                    .filter(|(g, e)| g.to_bits() != e.to_bits())
                    .count();
                assert_eq!(
                    differing, 0,
                    "batch {batch}, row_block {row_block}, row {b}: {differing} of {rows} \
                     outputs differ from the GEMV on reassociation-sensitive data"
                );
            }
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

/// **`FC_GEMM_R` CHANGES WHICH SIMD GROUP OWNS A ROW AND NOTHING ELSE, so
/// every width must be bit-identical to the GEMV, not merely close.**
///
/// The argument is that each output `(row, bi)` still walks the same blocks
/// in the same order into one FP32 accumulator and `simd_sum` still reduces
/// the same 32-lane partition of K. That is exactly the kind of claim
/// `the_gemm_and_the_gemv_agree_on_data_that_can_see_reassociation` exists
/// to stop anyone believing from a reading, so this runs on the SAME hostile
/// fixture with the SAME positive control: ragged BF16 companions,
/// full-mantissa FP16 activations, an asserted order-sensitivity floor, and
/// `dequant_int4_gemm_mma` required to differ.
///
/// **IT IS ALSO THE KEY GUARD FOR THE FOURTH BAKED CONSTANT**, and that is
/// why the widths are swept in ONE process at one shape rather than in three
/// processes. `MetalContext::pipeline` caches on (source address, name,
/// key), so if `r` were dropped from `specialized_constants_row_blocked`'s
/// key the R=2 dispatch would be served the R=1 pipeline -- while the HOST
/// had already divided the threadgroup count by 2, so half the rows would
/// never be written and would read back as whatever the fresh output buffer
/// held. Mutation-checked: dropping `r` from the key reddens the R=2 arm
/// here and nothing else in this file.
#[test]
fn row_blocking_does_not_move_a_single_bit() {
    let mut context = MetalContext::new().expect("Metal device");
    let widths: Vec<usize> = (1..=MAX_GEMM_ROW_BLOCK).collect();
    hostile_parity_at(&mut context, 128, 512, &[1, 2, 8, MAX_BATCH_ROWS], &widths);
    // A row count that is NOT a multiple of `8 * row_block` at every width,
    // so the per-row `row0 + r >= m_dim` guard is exercised rather than
    // being a branch no fixture reaches. 8*4 = 32 does not divide 72.
    hostile_parity_at(&mut context, 72, 256, &[1, 3], &widths);
    // **THE REMAINDER LOOP NEEDS ITS OWN SHAPE, and no other case here
    // reaches it at R > 1.** The kernel walks `n_groups / 4` full 4-group
    // blocks and then a scalar tail, so the tail only runs when `cols` is
    // not a multiple of 256 -- which 512, 5120, 6144 and 17408 all are. At
    // `cols = 448` there are 7 groups: one full block and a 3-group tail,
    // so both loops contribute to every output and their accumulation order
    // relative to each other is pinned too. The tail nests its loops the
    // other way round (groups outermost) and guards rows with a separate
    // `continue`, so it is genuinely different code.
    //
    // 448 rather than a tidier 128, which would be ALL tail, because the
    // fixture's own order-sensitivity floor rejected 128 at batch 3 (10 of
    // 48 blocks). That guard firing is the fixture working: a shape too
    // small to round cannot see a reassociation, and a case that cannot see
    // one proves nothing about accumulation order.
    hostile_parity_at(&mut context, 72, 448, &[1, 3], &widths);
}

/// **PIPELINE REFLECTION CANNOT SEE THIS KERNEL'S REGISTER PRESSURE ON THIS
/// DEVICE. MEASURED NEGATIVE, 2026-08-29, AND THE POINT OF THE CASE IS TO
/// STOP IT BEING RE-DERIVED.**
///
/// The register file is this kernel's binding constraint --
/// `dequant_int4_batch.metal`'s header records two optimizations that lost
/// to it and an unroll table whose worst row is labelled "spilling" -- and
/// every one of those statements was read off a TIMING. Apple lowers a
/// pipeline's `maxTotalThreadsPerThreadgroup` when a kernel's register
/// demand will not fit the threadgroup, so the obvious hope was a STATIC
/// answer to "does this width spill": a Metal device, no model, no install
/// and no clean clock.
///
/// It does not work. `maxTotalThreadsPerThreadgroup` reads **1024 on every
/// `(R, B)` shape**, at the real 17408x5120, and the discrimination check
/// that says so is the reason to believe it rather than the table: raising
/// `kMaxRowBlock` to 64 and probing R=64 at B=16 declares
/// `acc[64][16]`, a thousand floats per lane that can fit no register file
/// on any GPU, and it STILL reads 1024. An instrument that reports its
/// ceiling on a configuration that cannot possibly hold is at its rail, not
/// measuring (AGENTS.md Gotcha 59's shape: the best possible score on
/// garbage). Metal exposes no register or occupancy count publicly, so
/// there is no second instrument to reach for either.
///
/// **WHAT SURVIVES IS NARROWER AND STILL WORTH ASSERTING.** Two of the three
/// values do carry information about this kernel, and one of them guards a
/// documented dead end:
///
/// - `static_threadgroup_memory_length` is 0, and must stay 0. Note 1 in the
///   shader header is threadgroup staging of `x`, a measured loss that a
///   future reader will reach for again; re-adding it makes this nonzero, so
///   this line is the one cheap guard against it landing unmeasured.
/// - `thread_execution_width` is 32, which the whole `simd_sum` reduction
///   assumes.
/// - `maxTotalThreadsPerThreadgroup >= GEMM_THREADS_PER_GROUP` is kept as a
///   HARD-FAILURE backstop rather than as a spill gate: it cannot see
///   pressure, but a shape that genuinely could not be encoded at the width
///   the dispatch asks for would still trip it.
///
/// So `c(R, B)` on AC remains the only instrument that can price a row
/// block, and the table below is printed for the record rather than read.
#[test]
fn pipeline_reflection_cannot_see_this_kernels_register_pressure() {
    let mut context = MetalContext::new().expect("Metal device");
    // The real `Qwen/Qwen3.8-27B` FFN gate/up shape, where the register
    // pressure this is about actually occurs.
    let (rows, cols) = (17408usize, 5120usize);
    println!("\n{rows}x{cols}   R   B   maxThreads  execWidth  tgMemBytes");
    let mut distinct_max_threads = std::collections::BTreeSet::new();
    for row_block in 1..=MAX_GEMM_ROW_BLOCK {
        for batch in [1usize, 2, 4, 8, MAX_BATCH_ROWS] {
            let limits =
                dequant_int4_gemm_pipeline_limits(&mut context, rows, cols, batch, row_block)
                    .expect("pipeline");
            println!(
                "{rows}x{cols}  {row_block:>2}  {batch:>2}  {:>10}  {:>9}  {:>10}",
                limits.max_total_threads_per_threadgroup,
                limits.thread_execution_width,
                limits.static_threadgroup_memory_length,
            );
            distinct_max_threads.insert(limits.max_total_threads_per_threadgroup);
            assert!(
                limits.max_total_threads_per_threadgroup >= GEMM_THREADS_PER_GROUP,
                "R={row_block} B={batch} reports {} threads against the \
                 {GEMM_THREADS_PER_GROUP} the dispatch asks for -- this shape cannot be \
                 encoded at all",
                limits.max_total_threads_per_threadgroup
            );
            assert_eq!(
                limits.thread_execution_width, 32,
                "the simd_sum reduction assumes a 32-lane SIMD group"
            );
            assert_eq!(
                limits.static_threadgroup_memory_length, 0,
                "R={row_block} B={batch} declares threadgroup memory. This kernel uses \
                 none, and note 1 in dequant_int4_batch.metal's header is a MEASURED LOSS \
                 for adding some (staging `x`); if it is being re-added, re-measure \
                 c_of_m first"
            );
        }
    }
    // The negative, stated as an assertion so it cannot quietly stop being
    // true without anyone noticing. If a future OS or device DOES vary this
    // with register demand, this reddens and the case above becomes the
    // spill gate it was originally written to be.
    //
    // **Scoped to REAL Apple Silicon.** A virtualized Metal device (CI's
    // "Apple Paravirtual device") is a different GPU implementation, not a
    // future OS or a different real chip, and it genuinely does vary this
    // reflection with (R, B) -- measured, not assumed: 6 distinct values on
    // the exact sweep this test runs. That is not the "instrument became
    // useful" case the comment above anticipates (there is no real register
    // spill to find on a paravirtualized device the way there would be on
    // silicon), it is a device this canary was never calibrated against.
    // The per-shape checks above -- dispatchability, SIMD width, threadgroup
    // memory -- still run unconditionally on every device, real or not.
    let device_name = context.device().name();
    if device_name.contains("Paravirtual") {
        println!(
            "device {device_name:?} is virtualized, not the real Apple Silicon this canary is \
             calibrated against ({} distinct maxTotalThreadsPerThreadgroup values seen); \
             skipping the constant-across-shapes assertion",
            distinct_max_threads.len()
        );
        return;
    }
    assert_eq!(
        distinct_max_threads.len(),
        1,
        "maxTotalThreadsPerThreadgroup now VARIES across (R, B): {distinct_max_threads:?}. \
         It was constant when this was measured, which is why the doc says reflection \
         cannot price a row block -- re-read that doc, the instrument may have become useful"
    );
}
