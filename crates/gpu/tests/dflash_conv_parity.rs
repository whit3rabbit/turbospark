#![cfg(target_os = "macos")]
//! Parity for `dflash_grouped_conv_fp16` (`docs/DFLASH2.md`'s conv), against
//! a CPU reference that implements the reference implementation's formula
//! directly:
//!
//! `out[i, c] = sum_t (base[side, t, c] + delta[i, t, g(c)]) * x[i - t, c]`
//!
//! with the term zero when `i < t`, `g(c) = c / 16`, and `delta` sliced out
//! of a `[rows, 2 * 2 * groups]` projection output by the same stride/base
//! arithmetic the host passes the kernel. The comparison is TOLERANT, not
//! bit-exact: the conv feeds a drafter whose output is verified by the
//! target, so it is a throughput axis and never a correctness one, and no
//! bit-identity claim is made on either side of this test.
//!
//! Every case self-validates: alongside the parity check, each input is
//! mutated once and the output must MOVE, because an oracle that cannot
//! fail is the shape of bug this repo has shipped before (the MTP head's
//! first acceptance probe read 0 of 7,168 and printed a table anyway).

use half::f16;
use turbospark_gpu::{encode_dflash_grouped_conv, MetalContext, DFLASH_TAPS};

const GROUP_SIZE: usize = 16;

fn to_le(v: &[f16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    out
}

fn bf16_le(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&turbospark_compute::quant::f32_to_bf16(*x).to_le_bytes());
    }
    out
}

fn read_halfs(buffer: &metal::Buffer, len: usize) -> Vec<f16> {
    let ptr = buffer.contents() as *const u16;
    let bits = unsafe { std::slice::from_raw_parts(ptr, len) };
    bits.iter().map(|&b| f16::from_bits(b)).collect()
}

/// A small deterministic PRNG so every run of this test sees the same
/// "random" data (xorshift64, the same shape the other parity tests use).
struct XorShift(u64);
impl XorShift {
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 40) as f32 / (1u64 << 24) as f32) - 1.0
    }
    fn halves(&mut self, n: usize) -> Vec<f16> {
        (0..n).map(|_| f16::from_f32(self.next_f32())).collect()
    }
    fn floats(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

struct ConvCase {
    rows: usize,
    hidden: usize,
}

/// The CPU reference: the formula plus the host's own slicing of `delta`.
fn reference(
    x: &[f16],
    delta: &[f16],
    base: &[f32],
    rows: usize,
    hidden: usize,
    side: usize,
) -> Vec<f32> {
    let groups = hidden / GROUP_SIZE;
    let stride = 2 * DFLASH_TAPS as usize * groups;
    let col_base = side * DFLASH_TAPS as usize * groups;
    let mut out = vec![0.0f32; rows * hidden];
    for i in 0..rows {
        for c in 0..hidden {
            let g = c / GROUP_SIZE;
            let mut acc = 0.0f32;
            for t in 0..DFLASH_TAPS as usize {
                if i < t {
                    break;
                }
                let coeff = base[(side * DFLASH_TAPS as usize + t) * hidden + c]
                    + delta[i * stride + col_base + t * groups + g].to_f32();
                acc += coeff * x[(i - t) * hidden + c].to_f32();
            }
            out[i * hidden + c] = acc;
        }
    }
    out
}

fn run_case(case: ConvCase, seed: u64) {
    let ConvCase { rows, hidden } = case;
    let groups = hidden / GROUP_SIZE;
    let mut rng = XorShift(seed);
    let x = rng.halves(rows * hidden);
    // The projection output: BOTH sides filled, so the kernel slicing the
    // side-1 columns out from behind side 0 is part of what is tested.
    let delta = rng.halves(rows * 2 * DFLASH_TAPS as usize * groups);
    let base = rng.floats(2 * DFLASH_TAPS as usize * hidden);

    let want: Vec<Vec<f32>> = (0..2)
        .map(|side| reference(&x, &delta, &base, rows, hidden, side))
        .collect();

    let mut context = MetalContext::new().expect("Metal device");
    let x_buf = context.new_buffer_with_data(&to_le(&x));
    let delta_buf = context.new_buffer_with_data(&to_le(&delta));
    let base_buf = context.new_buffer_with_data(&bf16_le(&base));
    let out_buf = context.new_output_buffer((rows * hidden * 2) as u64);

    let mut got = Vec::new();
    for side in 0..2 {
        let pass = context.begin_pass();
        encode_dflash_grouped_conv(
            &mut context,
            &pass,
            (&x_buf, 0),
            (&delta_buf, 0),
            (&base_buf, 0),
            (&out_buf, 0),
            rows as u32,
            hidden as u32,
            side as u32,
            1.0,
        )
        .expect("encode");
        pass.commit_and_wait();
        got.push(read_halfs(&out_buf, rows * hidden));
    }

    for side in 0..2 {
        let mut max_diff = 0.0f32;
        for i in 0..rows * hidden {
            let d = (got[side][i].to_f32() - want[side][i]).abs();
            max_diff = max_diff.max(d);
        }
        // FP16 output rounding against an FP32 reference: the bound scales
        // with the value magnitude, and the inputs are O(1) coefficients
        // over O(1) activations with 2 taps.
        let bound = 2e-2;
        assert!(
            max_diff <= bound,
            "side {side}, rows {rows}, hidden {hidden}: max diff {max_diff} > {bound}"
        );
    }

    // Self-validating oracle (side 0): mutate row 1 of x and require that
    // row 0's output is BIT-IDENTICAL (a tap cannot read forward) while
    // rows 1 and 2 MOVE (row 1 through tap 0, row 2 through tap 1).
    if rows >= 3 {
        let mut x_mut = x.clone();
        for c in 0..hidden {
            x_mut[hidden + c] = f16::from_f32(x_mut[hidden + c].to_f32() + 0.5);
        }
        let probe = context.new_buffer_with_data(&to_le(&x_mut));
        let pass = context.begin_pass();
        encode_dflash_grouped_conv(
            &mut context,
            &pass,
            (&probe, 0),
            (&delta_buf, 0),
            (&base_buf, 0),
            (&out_buf, 0),
            rows as u32,
            hidden as u32,
            0,
            1.0,
        )
        .expect("encode probe");
        pass.commit_and_wait();
        let mutated = read_halfs(&out_buf, rows * hidden);
        assert!(
            mutated[..hidden] == got[0][..hidden],
            "row 0 moved when only row 1's input changed: a tap is reading forward"
        );
        let moved = (1..3)
            .filter(|&r| {
                mutated[r * hidden..(r + 1) * hidden] != got[0][r * hidden..(r + 1) * hidden]
            })
            .count();
        assert_eq!(
            moved, 2,
            "mutating row 1 moved {moved} of rows 1..3; tap 0 or tap 1 is dead"
        );
    }
}

#[test]
fn grouped_conv_matches_the_formula() {
    // The real drafter's shape: a block of 8 proposals plus the bonus row.
    run_case(
        ConvCase {
            rows: 9,
            hidden: 5120,
        },
        0xD157_F1A5,
    );
    // The smallest block, and a NON-power-of-two row count.
    run_case(
        ConvCase {
            rows: 3,
            hidden: 5120,
        },
        0xB10C_0C0C,
    );
    run_case(
        ConvCase {
            rows: 5,
            hidden: 128,
        },
        0x5EED_1234,
    );
    // One row: every tap past 0 is masked, which is the i < t boundary.
    run_case(
        ConvCase {
            rows: 1,
            hidden: 128,
        },
        0x000F_F1CE,
    );
}

/// The two sides must DISAGREE on the same inputs: `prepare` and `finish`
/// read different halves of the base kernel and of the projection output,
/// and a kernel that ignored `side` would pass the parity check above
/// twice with the same answer.
#[test]
fn the_two_sides_are_distinct() {
    let rows = 4usize;
    let hidden = 256usize;
    let groups = hidden / GROUP_SIZE;
    let mut rng = XorShift(0x0005_1DE5);
    let x = rng.halves(rows * hidden);
    let delta = rng.halves(rows * 2 * DFLASH_TAPS as usize * groups);
    let base = rng.floats(2 * DFLASH_TAPS as usize * hidden);

    let mut context = MetalContext::new().expect("Metal device");
    let x_buf = context.new_buffer_with_data(&to_le(&x));
    let delta_buf = context.new_buffer_with_data(&to_le(&delta));
    let base_buf = context.new_buffer_with_data(&bf16_le(&base));
    let out_buf = context.new_output_buffer((rows * hidden * 2) as u64);

    let mut outs = Vec::new();
    for side in 0..2 {
        let pass = context.begin_pass();
        encode_dflash_grouped_conv(
            &mut context,
            &pass,
            (&x_buf, 0),
            (&delta_buf, 0),
            (&base_buf, 0),
            (&out_buf, 0),
            rows as u32,
            hidden as u32,
            side as u32,
            1.0,
        )
        .expect("encode");
        pass.commit_and_wait();
        outs.push(read_halfs(&out_buf, rows * hidden));
    }
    let differing = outs[0]
        .iter()
        .zip(outs[1].iter())
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        differing > rows * hidden / 2,
        "side 1 output equals side 0 on {differing} of {} elements; the \
         kernel is ignoring `side`",
        rows * hidden
    );
}

/// `out_scale` multiplies the result and nothing else. It is how the draft
/// pass keeps a residual stream whose true peak is 113,920 inside FP16's
/// 65,504 ceiling (`DFLASH_RESIDUAL_SCALE`): the `finish` call sites divide
/// their output by a power of two, and the RMS norms that read the stream
/// take a correspondingly divided eps, so the scale cancels.
///
/// A POWER OF TWO is asserted EXACTLY rather than within a tolerance,
/// because that is the property the residual scaling claims: scaling by
/// 2^-3 shifts the exponent and leaves the mantissa alone, so the stored
/// halves must be bit-for-bit the unscaled ones at another exponent. A
/// tolerance here would pass for a scale that quietly rounded.
#[test]
fn the_output_scale_is_an_exact_power_of_two_shift() {
    let rows = 9usize;
    let hidden = 256usize;
    let groups = hidden / GROUP_SIZE;
    let mut rng = XorShift(0x5eed_2b17_u64);
    let x = rng.halves(rows * hidden);
    let delta = rng.halves(rows * 2 * DFLASH_TAPS as usize * groups);
    let base = rng.floats(2 * DFLASH_TAPS as usize * hidden);

    let mut context = MetalContext::new().expect("Metal device");
    let x_buf = context.new_buffer_with_data(&to_le(&x));
    let delta_buf = context.new_buffer_with_data(&to_le(&delta));
    let base_buf = context.new_buffer_with_data(&bf16_le(&base));
    let out_buf = context.new_output_buffer((rows * hidden * 2) as u64);

    let mut outs = Vec::new();
    for scale in [1.0f32, 0.125] {
        let pass = context.begin_pass();
        encode_dflash_grouped_conv(
            &mut context,
            &pass,
            (&x_buf, 0),
            (&delta_buf, 0),
            (&base_buf, 0),
            (&out_buf, 0),
            rows as u32,
            hidden as u32,
            1,
            scale,
        )
        .expect("encode");
        pass.commit_and_wait();
        outs.push(read_halfs(&out_buf, rows * hidden));
    }

    let mut moved = 0usize;
    for (i, (plain, scaled)) in outs[0].iter().zip(outs[1].iter()).enumerate() {
        let plain = plain.to_f32();
        // Subnormals lose the exponent room a power-of-two shift needs, so
        // they are excluded from the exactness claim rather than weakening
        // it for every element.
        if plain == 0.0 || plain.abs() < 1e-3 {
            continue;
        }
        assert_eq!(
            scaled.to_f32(),
            plain * 0.125,
            "element {i}: scaling by 2^-3 must be exact, got {} against {}",
            scaled.to_f32(),
            plain * 0.125
        );
        moved += 1;
    }
    assert!(
        moved > rows * hidden / 4,
        "only {moved} of {} elements were large enough to check; this fixture \
         cannot see the scale",
        rows * hidden
    );
}
