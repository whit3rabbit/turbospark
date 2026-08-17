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
