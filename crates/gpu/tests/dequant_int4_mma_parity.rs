#![cfg(target_os = "macos")]
//! `dequant_int4_gemm_mma` against `dequant_int4_gemv_simd` run B times.
//!
//! **THIS IS THE ONE BATCHED-KERNEL TEST THAT IS NOT EXACT, and the
//! difference from `dequant_int4_gemm_parity.rs` next door is the whole
//! point of the file existing separately.** That kernel accumulates K in a
//! fixed sequence into one FP32 register and must match the GEMV
//! bit-for-bit; this one hands K to `simdgroup_multiply_accumulate`, which
//! reduces in hardware in an order Apple does not document. So the contract
//! here is a tolerance, and a caller who needs speculative output to be
//! provably identical to non-speculative output (AGENTS.md Gotcha 27) must
//! use the exact kernel instead.
//!
//! What that costs is measured rather than asserted: the test REPORTS the
//! worst deviation it saw beside the bound it enforces, so a future change
//! that quietly doubles the error is visible in the output before it
//! reaches the threshold.

use half::f16;
use turbospark_gpu::{
    autorelease_pool, encode_dequant_int4_gemm_mma_resident, encode_dequant_int4_gemv_resident,
    Int4ResidentMatrix, MetalContext, MAX_BATCH_ROWS,
};

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

/// BF16 companions with a NON-ZERO MANTISSA, unlike the exact-power-of-two
/// values `dequant_int4_gemm_parity.rs` uses.
///
/// **That difference is the whole reason this fixture is not shared.** The
/// exact kernel's test wants values whose products sum exactly in FP32, so
/// that any bit difference is unambiguously a bug. This test is measuring
/// what REASSOCIATION costs, and on exactly-summable data reassociation
/// costs nothing -- the first run of this file read 4096/4096 outputs
/// bit-identical and a worst relative deviation of 0.000000, which looks
/// like a superb result and is a fixture that cannot see the property it
/// claims to bound. `the_fixture_can_see_reassociation_at_all` below asserts
/// the difference before any number here is believed.
fn bf16_small(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(len * 2);
    for _ in 0..len {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let exponent = 0x3C00u16 + (((state >> 40) as u16 & 3) << 7);
        // Seven mantissa bits, so the value is not a binary fraction and a
        // 256-term sum of such products does not close exactly in FP32.
        let mantissa = ((state >> 24) as u16) & 0x7F;
        out.extend_from_slice(&(exponent | mantissa).to_le_bytes());
    }
    out
}

/// Activation values with a full FP16 mantissa rather than the /32 grid the
/// exact test uses, for the same reason as above.
fn activation(i: usize) -> f16 {
    let v = ((i * 37 % 251) as f32 - 125.0) / 251.0;
    f16::from_f32(v)
}

fn half_bytes(values: &[f16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect()
}

/// The bound, as a fraction of the OUTPUT VECTOR'S OWN DYNAMIC RANGE rather
/// than of each element.
///
/// **Relative-to-the-element is the wrong metric here and measuring it that
/// way is a mistake this repo has now made three times** (the router
/// transcode read 166, `A_log` read 0.196, both meaningless). A dot product
/// of 256 terms of magnitude ~0.5 that lands on -0.02 has CANCELLED, and
/// floating-point error in such a sum is bounded by the size of the terms,
/// not by the size of what survived them -- so dividing by the result
/// manufactures a large number out of a well-behaved computation. The first
/// discriminating run of this file failed at `mma -0.019500 vs gemv
/// -0.020217`, an absolute difference of 0.0007 against terms summing to
/// hundreds of times that.
///
/// Scaling by `max|expected|` over the row is the standard fix and keeps
/// the check meaningful as the shape grows.
const RANGE_TOLERANCE: f32 = 0.005;

#[test]
fn the_matrix_kernel_agrees_with_the_gemv_to_a_tolerance() {
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

    let mut worst_rel = 0.0f32;
    let mut exact_matches = 0usize;
    let mut compared = 0usize;

    for batch in [1usize, 2, 5, 8, MAX_BATCH_ROWS] {
        // `x` is sized for a WHOLE number of 8-token column tiles, which is
        // this kernel's documented precondition: it loads its right-hand
        // side a tile at a time and would read past a tightly-sized buffer.
        let col_tiles = batch.div_ceil(8);
        let x_rows = col_tiles * 8;
        let x_values: Vec<f16> = (0..x_rows * cols).map(activation).collect();
        let x = context.new_buffer_with_data(&half_bytes(&x_values));
        let y_mma = context.new_output_buffer((batch * rows * 2) as u64);
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
            encode_dequant_int4_gemm_mma_resident(
                &mut context,
                &pass,
                &matrix(),
                (&x, 0),
                (&y_mma, 0),
                batch,
            )
            .expect("mma dispatch");
            pass.commit_and_wait();
        });
        let got = turbospark_gpu::read_buffer_f16(&y_mma, 0, batch * rows);

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
            assert!(
                expected.iter().any(|v| v.to_f32() != 0.0),
                "fixture produced an all-zero output, so this proves nothing"
            );
            let range = expected
                .iter()
                .fold(0.0f32, |acc, v| acc.max(v.to_f32().abs()));
            assert!(range > 0.0, "output range is zero");
            for (m, (g, e)) in got[b * rows..(b + 1) * rows]
                .iter()
                .zip(expected.iter())
                .enumerate()
            {
                let (gf, ef) = (g.to_f32(), e.to_f32());
                compared += 1;
                if g.to_bits() == e.to_bits() {
                    exact_matches += 1;
                }
                let rel = (gf - ef).abs() / range;
                worst_rel = worst_rel.max(rel);
                assert!(
                    rel <= RANGE_TOLERANCE,
                    "batch {batch}, row {b}, col {m}: mma {gf} vs gemv {ef} \
                     ({rel:.5} of the row's {range} range, over {RANGE_TOLERANCE})"
                );
            }
        }
    }

    println!(
        "\nworst deviation {worst_rel:.6} of the output range against a {RANGE_TOLERANCE} bound; \
         {exact_matches}/{compared} outputs happened to land bit-identical.\n\
         The second figure is REPORTED, never asserted: this kernel reassociates\n\
         the K reduction and any exact agreement is a coincidence of the fixture."
    );
}

/// The guard the first draft of this file lacked: on data whose products
/// sum exactly in FP32, reassociating the sum changes nothing, so a
/// tolerance measured there bounds nothing. This reproduces the fixture's
/// arithmetic on the host and checks that summing forward and backward
/// actually disagree.
///
/// It is pure CPU and needs no device, which is the point -- it says
/// whether the FIXTURE is capable of exhibiting the effect, independently
/// of whether the kernel exhibits it.
#[test]
fn the_fixture_can_see_reassociation_at_all() {
    let cols = 256usize;
    let scales = bf16_small(0xBEEF, cols / 64);
    let biases = bf16_small(0xF00D, cols / 64);
    let packed = fill(0xD1A5, cols / 2);

    let bf16 = |b: &[u8], i: usize| -> f32 {
        let bits = u16::from_le_bytes([b[i * 2], b[i * 2 + 1]]);
        f32::from_bits((bits as u32) << 16)
    };

    let terms: Vec<f32> = (0..cols)
        .map(|n| {
            let byte = packed[n / 2];
            let q = if n % 2 == 0 { byte & 0x0F } else { byte >> 4 } as f32;
            let g = n / 64;
            let w = q * bf16(&scales, g) + bf16(&biases, g);
            w * activation(n).to_f32()
        })
        .collect();

    let forward = terms.iter().fold(0.0f32, |a, t| a + t);
    let backward = terms.iter().rev().fold(0.0f32, |a, t| a + t);
    assert_ne!(
        forward.to_bits(),
        backward.to_bits(),
        "this fixture sums exactly in FP32, so it cannot distinguish any \
         reduction order and the tolerance measured against it is vacuous"
    );
    println!("forward {forward} vs backward {backward}: the fixture discriminates");
}

/// **STAGING `x` THROUGH THREADGROUP MEMORY MOVES NO BITS.**
///
/// `FC_MMA_STAGE_X` (110) changes only where the right-hand side is read
/// from: the same values reach the same `simdgroup_load` in the same order,
/// so the two arms must agree BIT FOR BIT with each other. That is a
/// stronger claim than this file's other case, which allows a tolerance
/// against the GEMV because `simdgroup_multiply_accumulate` reduces K in an
/// order Apple does not document. The tolerance is about hardware
/// reduction; this is about a memory path, and a memory path has no licence
/// to change a value.
///
/// It also pins the pipeline-cache key. Both arms run in ONE process at the
/// same `(M, N, B)`, so if `stage_x` were missing from the key the second
/// dispatch would silently reuse the first's pipeline and the comparison
/// would be a shape against itself -- green, and proving nothing (crate
/// Gotcha 1). The widths deliberately include one that is NOT a multiple of
/// the 8-row tile (5), because the staging loop bounds itself on
/// `col_tiles * kMmaTile` rather than on `B` and an off-by-one there would
/// only show at a ragged width.
#[test]
fn staging_x_through_threadgroup_memory_moves_no_bits() {
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

    let mut compared = 0usize;
    for batch in [1usize, 2, 5, 8, MAX_BATCH_ROWS, 32, 64] {
        let col_tiles = batch.div_ceil(8);
        let x_rows = col_tiles * 8;
        let x_values: Vec<f16> = (0..x_rows * cols).map(activation).collect();
        let x = context.new_buffer_with_data(&half_bytes(&x_values));

        let run = |context: &mut MetalContext, stage_x: bool| {
            let y = context.new_output_buffer((batch * rows * 2) as u64);
            autorelease_pool(|| {
                let pass = context.begin_pass();
                turbospark_gpu::encode_dequant_int4_gemm_mma_resident_staged(
                    context,
                    &pass,
                    &Int4ResidentMatrix {
                        buffer: &weights,
                        weights_offset: 0,
                        scales_offset,
                        biases_offset,
                        rows,
                        cols,
                    },
                    (&x, 0),
                    (&y, 0),
                    batch,
                    stage_x,
                )
                .expect("mma dispatch");
                pass.commit_and_wait();
            });
            turbospark_gpu::read_buffer_f16(&y, 0, batch * rows)
        };

        let plain = run(&mut context, false);
        let staged = run(&mut context, true);
        assert_eq!(
            plain.len(),
            staged.len(),
            "batch {batch}: arms disagree on length"
        );
        for (i, (a, b)) in plain.iter().zip(staged.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "batch {batch}, element {i}: staged {b} != unstaged {a}"
            );
            compared += 1;
        }
    }

    // The fixture has to actually produce varied, non-zero output, or
    // bit-equality is a comparison of two runs of zeros.
    assert!(compared >= 128 * 7, "compared only {compared} outputs");
}

/// **THE SKIP-DEQUANT DIAGNOSTIC IS REACHABLE, AND ITS OUTPUT IS WRONG.**
///
/// `FC_MMA_SKIP_DEQUANT` (111) fills the weight tile with a constant so
/// `gemv_bandwidth_bench.rs` can time the matrix path with the unpack
/// removed. The whole value of that measurement rests on the constant
/// actually reaching the kernel: if it did not, the bench would compile one
/// pipeline, time it twice, and report a ratio of 1.00 that reads as
/// "the dequant is free" -- a wrong answer that looks like a finding.
///
/// So this asserts the two DIFFER, which is the opposite of every other
/// case in this file and is deliberate. It also pins the pipeline-cache
/// key: both run in one process at the same `(M, N, B)`, so a key missing
/// the flag would hand back the first pipeline and the assertion would
/// fail loudly rather than silently pass.
#[test]
fn the_skip_dequant_diagnostic_is_reachable_and_its_output_is_wrong() {
    let rows = 128usize;
    let cols = 256usize;
    let batch = 8usize;
    let mut context = MetalContext::new().expect("Metal device");

    let scales_offset = (rows * cols / 2) as u64;
    let biases_offset = scales_offset + (rows * cols / 64 * 2) as u64;
    let total = biases_offset + (rows * cols / 64 * 2) as u64;
    let mut blob = fill(0xD1A5, scales_offset as usize);
    blob.extend_from_slice(&bf16_small(0xBEEF, rows * cols / 64));
    blob.extend_from_slice(&bf16_small(0xF00D, rows * cols / 64));
    assert_eq!(blob.len() as u64, total);
    let weights = context.new_buffer_with_data(&blob);

    let x_values: Vec<f16> = (0..batch * cols).map(activation).collect();
    let x = context.new_buffer_with_data(&half_bytes(&x_values));

    let matrix = || Int4ResidentMatrix {
        buffer: &weights,
        weights_offset: 0,
        scales_offset,
        biases_offset,
        rows,
        cols,
    };

    let y_real = context.new_output_buffer((batch * rows * 2) as u64);
    let y_diag = context.new_output_buffer((batch * rows * 2) as u64);
    autorelease_pool(|| {
        let pass = context.begin_pass();
        encode_dequant_int4_gemm_mma_resident(
            &mut context,
            &pass,
            &matrix(),
            (&x, 0),
            (&y_real, 0),
            batch,
        )
        .expect("real");
        pass.commit_and_wait();
    });
    autorelease_pool(|| {
        let pass = context.begin_pass();
        turbospark_gpu::encode_dequant_int4_gemm_mma_resident_skip_dequant(
            &mut context,
            &pass,
            &matrix(),
            (&x, 0),
            (&y_diag, 0),
            batch,
        )
        .expect("diagnostic");
        pass.commit_and_wait();
    });

    let real = turbospark_gpu::read_buffer_f16(&y_real, 0, batch * rows);
    let diag = turbospark_gpu::read_buffer_f16(&y_diag, 0, batch * rows);
    let differing = real
        .iter()
        .zip(diag.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert!(
        differing > real.len() / 2,
        "only {differing} of {} outputs differ: the skip-dequant constant is \
         not reaching the kernel, so any timing taken with it measures the \
         unmodified kernel twice",
        real.len()
    );
    // And the real arm must not be trivially zero, or "they differ" is
    // satisfied by a fixture that proves nothing.
    assert!(
        real.iter().any(|v| v.to_f32().abs() > 1e-3),
        "the real arm produced no signal"
    );
}

/// **WHAT THE COMPILER MADE OF EACH TILE, READ OFF THE PIPELINE.** No clock,
/// no model, no clean machine -- which is what makes these answerable in a
/// session that cannot benchmark.
///
/// Three separate claims, and only the first is a gate:
///
/// 1. **The wide arm can be dispatched at 128 threads at all.** This is a
///    HARD PRECONDITION rather than a spill signal: asking for more threads
///    than the pipeline permits is invalid, so it has to hold before any
///    timing is taken. The caveat recorded on the SIMD sibling -- that this
///    field reads 1024 even on an impossible `acc[64][16]`, so it cannot
///    PRICE register pressure -- does not apply to a floor.
///
/// 2. **The wide arm's threadgroup allocation is what the header claims**
///    (`w_tile` 2,048 + `x_tile` 8,192 + `y_tile` 1,024 = 11,264 B), so the
///    "bytes per thread 296 -> 88" mechanism the re-tile is betting on is a
///    reading rather than an assumption.
///
/// 3. **`x_tile` IS eliminated when `FC_MMA_STAGE_X` is false, MEASURED
///    2026-09-05 and asserted here.** It is declared unconditionally, so the
///    question was open and load-bearing: had the compiler kept it, every
///    number in the shader header's first table would have been taken while
///    paying 8 KiB of dead threadgroup memory, and both the plateau and the
///    staging penalty would need rereading. It does not. The narrow arm reads
///    1,280 B un-staged (`w_tile` 1,024 + `y_tile` 256) against 9,472 staged.
///
///    **THAT MAKES THE BYTES-PER-THREAD ARGUMENT NARROWER THAN IT LOOKS, and
///    the direction is worth stating because it is not the flattering one.**
///    The wide arm is meant to run staged, at 11,264 B over 128 threads = 88
///    B/thread. Against the narrow STAGED arm (9,472 over 32 = 296) that is a
///    3.4x cut, and the narrow staged arm is the one Do Not Revisit 13
///    measured losing 3.3x to 5.9x -- so the mechanism story holds exactly
///    where that entry left it. Against the narrow arm as it is actually
///    MEASURED in the gate, which is un-staged at 1,280 over 32 = 40
///    B/thread, the wide arm uses MORE, not less. Occupancy is therefore not
///    a reason to expect the wide arm to beat the un-staged narrow control,
///    and if it does, that is not the mechanism.
#[test]
fn the_wide_pipeline_dispatches_at_128_threads_and_fits_threadgroup_memory() {
    let mut context = MetalContext::new().expect("Metal device");
    let cols = 256usize;

    for batch in [1usize, 2, 8, MAX_BATCH_ROWS, 32, 64] {
        for stage_x in [false, true] {
            let narrow = turbospark_gpu::dequant_int4_gemm_mma_pipeline_limits(
                &mut context,
                128,
                cols,
                batch,
                stage_x,
                false,
            )
            .expect("narrow pipeline");
            let wide = turbospark_gpu::dequant_int4_gemm_mma_pipeline_limits(
                &mut context,
                128,
                cols,
                batch,
                stage_x,
                true,
            )
            .expect("wide pipeline");
            println!(
                "batch {batch:>2} stage_x {stage_x:<5}  narrow: {:>4} threads, \
                 {:>6} B tg mem   wide: {:>4} threads, {:>6} B tg mem",
                narrow.max_total_threads_per_threadgroup,
                narrow.static_threadgroup_memory_length,
                wide.max_total_threads_per_threadgroup,
                wide.static_threadgroup_memory_length,
            );
            assert!(
                wide.max_total_threads_per_threadgroup >= 128,
                "batch {batch} stage_x {stage_x}: the wide pipeline permits only {} \
                 threads, so its 128-thread dispatch is invalid",
                wide.max_total_threads_per_threadgroup
            );
            // w_tile + y_tile, plus x_tile only when it is actually read.
            let expect = |wide: bool| -> u64 {
                let w_tile = if wide { 16 } else { 8 } * 64 * 2;
                let y_tile = if wide { 4 } else { 1 } * 8 * 8 * 4;
                let x_tile = if stage_x { 8 * 8 * 64 * 2 } else { 0 };
                w_tile + y_tile + x_tile
            };
            assert_eq!(
                narrow.static_threadgroup_memory_length,
                expect(false),
                "batch {batch} stage_x {stage_x}: narrow threadgroup allocation"
            );
            assert_eq!(
                wide.static_threadgroup_memory_length,
                expect(true),
                "batch {batch} stage_x {stage_x}: wide threadgroup allocation"
            );
        }
    }
}

/// A weight blob plus its two BF16 companion planes, at the fixture's own
/// deliberately-not-exactly-summable values.
fn mma_blob(rows: usize, cols: usize) -> (Vec<u8>, u64, u64) {
    let scales_offset = (rows * cols / 2) as u64;
    let biases_offset = scales_offset + (rows * cols / 64 * 2) as u64;
    let mut blob = fill(0xD1A5, scales_offset as usize);
    blob.extend_from_slice(&bf16_small(0xBEEF, rows * cols / 64));
    blob.extend_from_slice(&bf16_small(0xF00D, rows * cols / 64));
    (blob, scales_offset, biases_offset)
}

/// **THE RE-TILE MOVES NO BITS, AND THAT IS A CLAIM ABOUT THE ALGORITHM
/// RATHER THAN A TOLERANCE.**
///
/// The four-SIMD-group arm does not change the accumulation grouping. For any
/// output tile the `simdgroup_multiply_accumulate` sequence is `n0` ascending
/// then `kt` 0..7 over identical fragments in both shapes; what changes is
/// which SIMD group owns the accumulator and where the operands were staged,
/// and neither of those is arithmetic. **K is never split across SIMD
/// groups**, which is the property the whole claim rests on.
///
/// So a failure here does not mean "widen the tolerance". It means the K walk
/// changed, which is a design error -- and a variant that deliberately split
/// K would need a cross-simdgroup reduction and would have to demote this
/// case to `RANGE_TOLERANCE` at the same commit.
///
/// Running both arms in ONE process at the same `(M, N, B)` also pins the
/// pipeline-cache split: the two kernels differ only by NAME, so a cache
/// keyed without it would hand back whichever compiled first and this would
/// pass vacuously against one kernel run twice. That is why the ragged case
/// below matters as well -- it is the one where the two tiles genuinely
/// diverge in dispatch shape.
#[test]
fn the_wide_tile_produces_the_same_bits_as_the_narrow_one() {
    let rows = 128usize;
    let cols = 256usize;
    let mut context = MetalContext::new().expect("Metal device");
    let (blob, scales_offset, biases_offset) = mma_blob(rows, cols);
    let weights = context.new_buffer_with_data(&blob);

    let mut compared = 0usize;
    let mut worst = 0.0f32;
    for batch in [1usize, 2, 5, 8, MAX_BATCH_ROWS, 32, 64] {
        let col_tiles = batch.div_ceil(8);
        let x_values: Vec<f16> = (0..col_tiles * 8 * cols).map(activation).collect();
        let x = context.new_buffer_with_data(&half_bytes(&x_values));

        for stage_x in [false, true] {
            let run = |context: &mut MetalContext, wide: bool| {
                let y = context.new_output_buffer((batch * rows * 2) as u64);
                autorelease_pool(|| {
                    let pass = context.begin_pass();
                    let m = Int4ResidentMatrix {
                        buffer: &weights,
                        weights_offset: 0,
                        scales_offset,
                        biases_offset,
                        rows,
                        cols,
                    };
                    if wide {
                        turbospark_gpu::encode_dequant_int4_gemm_mma_resident_wide(
                            context,
                            &pass,
                            &m,
                            (&x, 0),
                            (&y, 0),
                            batch,
                            stage_x,
                        )
                    } else {
                        turbospark_gpu::encode_dequant_int4_gemm_mma_resident_staged(
                            context,
                            &pass,
                            &m,
                            (&x, 0),
                            (&y, 0),
                            batch,
                            stage_x,
                        )
                    }
                    .expect("mma dispatch");
                    pass.commit_and_wait();
                });
                turbospark_gpu::read_buffer_f16(&y, 0, batch * rows)
            };

            let narrow = run(&mut context, false);
            let wide = run(&mut context, true);
            let mut differing = 0usize;
            for (a, b) in narrow.iter().zip(wide.iter()) {
                if a.to_bits() != b.to_bits() {
                    differing += 1;
                    worst = worst.max((a.to_f32() - b.to_f32()).abs());
                }
                compared += 1;
            }
            // Report before asserting, so a failure carries a magnitude
            // rather than only a bit mismatch.
            println!(
                "batch {batch:>2} stage_x {stage_x:<5}  {differing} of {} outputs differ",
                narrow.len()
            );
            assert_eq!(
                differing,
                0,
                "batch {batch} stage_x {stage_x}: the re-tile moved {differing} of {} \\
                 outputs (worst absolute {worst:e}). The tile is not supposed to \\
                 change the K walk, so this is a design error and not a tolerance \\
                 to widen",
                narrow.len()
            );
        }
    }
    assert!(compared >= 128 * 14, "compared only {compared} outputs");
    assert!(worst == 0.0, "worst deviation {worst:e}");
}

/// **A ROW COUNT THAT IS NOT A MULTIPLE OF THE WIDE TILE.**
///
/// Every pre-existing case in this file uses `rows = 128`, a multiple of both
/// 8 and 16, so the file is BLIND BY CONSTRUCTION to the partial-row-tile
/// case (AGENTS.md Gotcha 51's shape). 136 is chosen so the two tiles
/// genuinely disagree: it is 17 whole 8-row tiles for the narrow kernel and
/// 8.5 for the wide one, so exactly one wide threadgroup runs with its
/// `wm = 1` SIMD groups entirely past `M`.
///
/// Those groups must NOT return early -- they still have to reach both
/// `threadgroup_barrier`s on every `n0` iteration, since a barrier some
/// threads skip is divergent participation and Metal leaves that undefined.
/// They zero-fill their share of the weight tile and predicate on `m < M` at
/// copy-out instead. So this case covers the copy-out predicate AND the
/// no-per-simdgroup-early-return rule at once, and its failure mode for the
/// second of those is a HANG rather than an assertion.
#[test]
fn a_ragged_row_count_still_writes_every_row_of_the_wide_tile() {
    let rows = 136usize;
    let cols = 256usize;
    assert_eq!(rows % 8, 0, "the narrow tile must divide it exactly");
    assert_ne!(rows % 16, 0, "the wide tile must NOT, or the case is blind");
    let mut context = MetalContext::new().expect("Metal device");
    let (blob, scales_offset, biases_offset) = mma_blob(rows, cols);
    let weights = context.new_buffer_with_data(&blob);

    for batch in [1usize, 8, MAX_BATCH_ROWS] {
        let col_tiles = batch.div_ceil(8);
        let x_values: Vec<f16> = (0..col_tiles * 8 * cols).map(activation).collect();
        let x = context.new_buffer_with_data(&half_bytes(&x_values));

        let run = |context: &mut MetalContext, wide: bool| {
            // Pre-fill with a sentinel, so a row the kernel never writes is
            // distinguishable from a row it writes as zero.
            let y =
                context.new_buffer_with_data(&half_bytes(&vec![f16::from_f32(-7.5); batch * rows]));
            autorelease_pool(|| {
                let pass = context.begin_pass();
                let m = Int4ResidentMatrix {
                    buffer: &weights,
                    weights_offset: 0,
                    scales_offset,
                    biases_offset,
                    rows,
                    cols,
                };
                if wide {
                    turbospark_gpu::encode_dequant_int4_gemm_mma_resident_wide(
                        context,
                        &pass,
                        &m,
                        (&x, 0),
                        (&y, 0),
                        batch,
                        true,
                    )
                } else {
                    encode_dequant_int4_gemm_mma_resident(
                        context,
                        &pass,
                        &m,
                        (&x, 0),
                        (&y, 0),
                        batch,
                    )
                }
                .expect("mma dispatch");
                pass.commit_and_wait();
            });
            turbospark_gpu::read_buffer_f16(&y, 0, batch * rows)
        };

        let narrow = run(&mut context, false);
        let wide = run(&mut context, true);
        let untouched = wide
            .iter()
            .filter(|v| v.to_bits() == f16::from_f32(-7.5).to_bits())
            .count();
        assert_eq!(
            untouched,
            0,
            "batch {batch}: {untouched} of {} outputs still carry the sentinel, so \\
             the wide tile's last threadgroup never wrote its live rows",
            wide.len()
        );
        for (i, (a, b)) in narrow.iter().zip(wide.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "batch {batch}, element {i} (row {}): narrow {a} != wide {b}",
                i % rows
            );
        }
    }
}

/// **ONE TOKEN TILE, SO HALF THE THREADGROUP HAS NO MATRIX WORK AT ALL.**
///
/// At `B <= 8` there is a single column tile, so the `wn = 1` SIMD groups
/// take zero trips through the MMA loop and zero through the copy-out. They
/// must still reach both `threadgroup_barrier`s on every `n0` iteration.
///
/// **THE FAILURE MODE HERE IS A HANG, NOT AN ASSERTION.** Divergent barrier
/// participation is undefined in Metal and presents as a command buffer that
/// never completes, so a timeout on this case is the bug rather than flaky
/// infrastructure. The shape is deliberately small so it hangs immediately.
#[test]
fn a_single_column_tile_still_reaches_every_barrier() {
    let rows = 32usize;
    let cols = 128usize;
    let mut context = MetalContext::new().expect("Metal device");
    let (blob, scales_offset, biases_offset) = mma_blob(rows, cols);
    let weights = context.new_buffer_with_data(&blob);

    for batch in [1usize, 3, 8] {
        assert_eq!(batch.div_ceil(8), 1, "batch {batch} is not a single tile");
        let x_values: Vec<f16> = (0..8 * cols).map(activation).collect();
        let x = context.new_buffer_with_data(&half_bytes(&x_values));
        let y = context.new_output_buffer((batch * rows * 2) as u64);
        autorelease_pool(|| {
            let pass = context.begin_pass();
            turbospark_gpu::encode_dequant_int4_gemm_mma_resident_wide(
                &mut context,
                &pass,
                &Int4ResidentMatrix {
                    buffer: &weights,
                    weights_offset: 0,
                    scales_offset,
                    biases_offset,
                    rows,
                    cols,
                },
                (&x, 0),
                (&y, 0),
                batch,
                true,
            )
            .expect("wide dispatch");
            pass.commit_and_wait();
        });
        let out = turbospark_gpu::read_buffer_f16(&y, 0, batch * rows);
        assert!(
            out.iter().any(|v| v.to_f32().abs() > 1e-3),
            "batch {batch}: the single-tile case produced no signal"
        );
    }
}

/// The staging claim, on the re-tiled kernel.
///
/// Same content as `staging_x_through_threadgroup_memory_moves_no_bits` one
/// tile over, and it additionally pins that constant 110's byte still reaches
/// the pipeline-cache key for the wide pipeline: both arms run in one process
/// at the same `(M, N, B)` and the same function NAME, so the flag is the
/// only discriminator left.
#[test]
fn staging_x_moves_no_bits_in_the_wide_kernel() {
    let rows = 128usize;
    let cols = 256usize;
    let mut context = MetalContext::new().expect("Metal device");
    let (blob, scales_offset, biases_offset) = mma_blob(rows, cols);
    let weights = context.new_buffer_with_data(&blob);

    let mut compared = 0usize;
    for batch in [1usize, 2, 5, 8, MAX_BATCH_ROWS, 32, 64] {
        let col_tiles = batch.div_ceil(8);
        let x_values: Vec<f16> = (0..col_tiles * 8 * cols).map(activation).collect();
        let x = context.new_buffer_with_data(&half_bytes(&x_values));

        let run = |context: &mut MetalContext, stage_x: bool| {
            let y = context.new_output_buffer((batch * rows * 2) as u64);
            autorelease_pool(|| {
                let pass = context.begin_pass();
                turbospark_gpu::encode_dequant_int4_gemm_mma_resident_wide(
                    context,
                    &pass,
                    &Int4ResidentMatrix {
                        buffer: &weights,
                        weights_offset: 0,
                        scales_offset,
                        biases_offset,
                        rows,
                        cols,
                    },
                    (&x, 0),
                    (&y, 0),
                    batch,
                    stage_x,
                )
                .expect("wide dispatch");
                pass.commit_and_wait();
            });
            turbospark_gpu::read_buffer_f16(&y, 0, batch * rows)
        };

        let plain = run(&mut context, false);
        let staged = run(&mut context, true);
        for (i, (a, b)) in plain.iter().zip(staged.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "batch {batch}, element {i}: staged {b} != unstaged {a}"
            );
            compared += 1;
        }
    }
    assert!(compared >= 128 * 7, "compared only {compared} outputs");
}

/// The skip-dequant diagnostic on the re-tiled kernel, for the reason its
/// narrow sibling gives: an unreachable diagnostic constant would let
/// `gemv_bandwidth_bench.rs` compile one pipeline, time it twice, and report
/// a ratio of 1.00 that reads as "the dequant is free".
#[test]
fn the_skip_dequant_diagnostic_is_reachable_in_the_wide_kernel() {
    let rows = 128usize;
    let cols = 256usize;
    let batch = 8usize;
    let mut context = MetalContext::new().expect("Metal device");
    let (blob, scales_offset, biases_offset) = mma_blob(rows, cols);
    let weights = context.new_buffer_with_data(&blob);
    let x_values: Vec<f16> = (0..batch * cols).map(activation).collect();
    let x = context.new_buffer_with_data(&half_bytes(&x_values));

    let mut run = |diag: bool| {
        let y = context.new_output_buffer((batch * rows * 2) as u64);
        autorelease_pool(|| {
            let pass = context.begin_pass();
            let m = Int4ResidentMatrix {
                buffer: &weights,
                weights_offset: 0,
                scales_offset,
                biases_offset,
                rows,
                cols,
            };
            if diag {
                turbospark_gpu::encode_dequant_int4_gemm_mma_resident_wide_skip_dequant(
                    &mut context,
                    &pass,
                    &m,
                    (&x, 0),
                    (&y, 0),
                    batch,
                )
            } else {
                turbospark_gpu::encode_dequant_int4_gemm_mma_resident_wide(
                    &mut context,
                    &pass,
                    &m,
                    (&x, 0),
                    (&y, 0),
                    batch,
                    false,
                )
            }
            .expect("wide dispatch");
            pass.commit_and_wait();
        });
        turbospark_gpu::read_buffer_f16(&y, 0, batch * rows)
    };

    let real = run(false);
    let diag = run(true);
    let differing = real
        .iter()
        .zip(diag.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert!(
        differing > real.len() / 2,
        "only {differing} of {} outputs differ: the skip-dequant constant is not \\
         reaching the wide kernel, so any timing taken with it measures the \\
         unmodified kernel twice",
        real.len()
    );
    assert!(
        real.iter().any(|v| v.to_f32().abs() > 1e-3),
        "the real arm produced no signal"
    );
}

/// The re-tiled arm against the GEMV, at the same bound the narrow arm is
/// held to.
///
/// **REDUNDANT WHILE `the_wide_tile_produces_the_same_bits_as_the_narrow_one`
/// HOLDS, AND KEPT FOR WHEN IT DOES NOT.** That case is the stronger claim
/// (bit identity, which follows from K not being split across SIMD groups)
/// and it makes this one true by transitivity through
/// `the_matrix_kernel_agrees_with_the_gemv_to_a_tolerance`. But the
/// bit-identity claim is a property of THIS tiling rather than of the family
/// of tilings: a variant that split K would need a cross-simdgroup reduction
/// and would have to be demoted to a tolerance. This is the case that would
/// survive that demotion, so it is the one that keeps a wide arm honest
/// against the reference rather than only against its sibling.
#[test]
fn the_wide_matrix_kernel_agrees_with_the_gemv_to_a_tolerance() {
    let rows = 128usize;
    let cols = 256usize;
    let batch = 8usize;
    let mut context = MetalContext::new().expect("Metal device");
    let (blob, scales_offset, biases_offset) = mma_blob(rows, cols);
    let weights = context.new_buffer_with_data(&blob);
    let x_values: Vec<f16> = (0..batch * cols).map(activation).collect();
    let x = context.new_buffer_with_data(&half_bytes(&x_values));

    let matrix = || Int4ResidentMatrix {
        buffer: &weights,
        weights_offset: 0,
        scales_offset,
        biases_offset,
        rows,
        cols,
    };

    let y_wide = context.new_output_buffer((batch * rows * 2) as u64);
    autorelease_pool(|| {
        let pass = context.begin_pass();
        turbospark_gpu::encode_dequant_int4_gemm_mma_resident_wide(
            &mut context,
            &pass,
            &matrix(),
            (&x, 0),
            (&y_wide, 0),
            batch,
            true,
        )
        .expect("wide dispatch");
        pass.commit_and_wait();
    });
    let wide = turbospark_gpu::read_buffer_f16(&y_wide, 0, batch * rows);

    // The GEMV, one row at a time, is the reference the whole file is
    // written against.
    let mut worst = 0.0f32;
    for b in 0..batch {
        let x_row = context.new_buffer_with_data(&half_bytes(&x_values[b * cols..(b + 1) * cols]));
        let y_row = context.new_output_buffer((rows * 2) as u64);
        autorelease_pool(|| {
            let pass = context.begin_pass();
            encode_dequant_int4_gemv_resident(
                &mut context,
                &pass,
                &matrix(),
                (&x_row, 0),
                (&y_row, 0),
            )
            .expect("gemv dispatch");
            pass.commit_and_wait();
        });
        let expected = turbospark_gpu::read_buffer_f16(&y_row, 0, rows);
        let range = expected
            .iter()
            .fold(0.0f32, |m, v| m.max(v.to_f32().abs()))
            .max(1e-6);
        for (i, e) in expected.iter().enumerate() {
            let d = (wide[b * rows + i].to_f32() - e.to_f32()).abs() / range;
            worst = worst.max(d);
        }
    }
    println!("wide vs gemv: worst deviation {worst:.6} of the row's own range");
    assert!(
        worst <= RANGE_TOLERANCE,
        "worst deviation {worst:.6} exceeds {RANGE_TOLERANCE}"
    );
}
