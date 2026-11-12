#![cfg(target_os = "macos")]
//! Parity for LFM3's gated short conv (`shaders/short_conv.metal`) against
//! `turbospark_compute::short_conv_reference`, its only contract (the
//! kernel is port-local).
//!
//! The reference rounds `u = B * xx` and `o = C * y` through FP16 where the
//! kernels store them and sums in the kernels' order, so CPU parity is held
//! to about one FP16 ulp at the output's magnitude. Prefill against N decode steps
//! is BIT-exact (output and tail): a verify or prefill row that differs from
//! sequential decode in the last bit is how greedy speculation diverged on
//! GDN (`gdn_conv_order_parity.rs`).
//!
//! Shapes: LFM3-1.7B's real width (H = 2048, K = 7), plus a non-multiple-of-
//! 256 channel count so the grid's tail threadgroup is exercised.

use half::f16;
use turbospark_compute::short_conv_reference;
use turbospark_gpu::{
    encode_short_conv_decode, encode_short_conv_prefill, MetalContext, ShortConvStateManager,
};

const LFM3_HIDDEN: usize = 2048;
const LFM3_TAPS: usize = 7;

struct XorShift(u64);
impl XorShift {
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    /// FP16-representable values in [-scale, scale).
    fn halves(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| f16::from_f32(self.next_f32() * scale).to_f32())
            .collect()
    }
}

fn f16_le(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| f16::from_f32(*x).to_bits().to_le_bytes())
        .collect()
}

/// BF16-representable weights, so the reference reads what the kernel reads.
fn bf16_round(v: &[f32]) -> Vec<f32> {
    v.iter()
        .map(|x| turbospark_compute::bf16_to_f32(turbospark_compute::f32_to_bf16(*x)))
        .collect()
}

fn bf16_le(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| turbospark_compute::f32_to_bf16(*x).to_le_bytes())
        .collect()
}

fn read_halfs(buffer: &metal::Buffer, len: usize) -> Vec<f32> {
    assert!(buffer.length() as usize >= len * 2);
    // SAFETY: shared-storage buffer of at least `len` halves, read after the
    // pass that wrote it has completed.
    let bits = unsafe { std::slice::from_raw_parts(buffer.contents() as *const u16, len) };
    bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
}

struct Fixture {
    hidden: usize,
    taps: usize,
    in_proj: Vec<f32>, // [rows, 3H]
    weight: Vec<f32>,  // [H, K], BF16-representable
    tail: Vec<f32>,    // [K-1, H] initial, FP16-representable
}

fn fixture(hidden: usize, taps: usize, rows: usize, zero_tail: bool, seed: u64) -> Fixture {
    let mut rng = XorShift(seed);
    // Scales chosen so u and y stay well inside FP16 range but are not tiny.
    let in_proj = rng.halves(rows * 3 * hidden, 2.0);
    let weight = bf16_round(&rng.halves(hidden * taps, 0.75));
    let tail = if zero_tail {
        vec![0.0; (taps - 1) * hidden]
    } else {
        rng.halves((taps - 1) * hidden, 3.0)
    };
    Fixture {
        hidden,
        taps,
        in_proj,
        weight,
        tail,
    }
}

/// Max |got - want| over FP16 ulp at |want|, with a floor at the smallest
/// normal so near-zero outputs are not judged in subnormal ulps.
fn assert_within_one_ulp(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let ulp = f16::from_f32(w.abs().max(6.1e-5)).to_f32() * f32::EPSILON * 8192.0;
        assert!(
            (g - w).abs() <= ulp,
            "{what}[{i}]: gpu {g} vs reference {w} (ulp {ulp})"
        );
    }
}

/// GPU decode over every row in order; returns (outputs, final tail).
fn gpu_decode_steps(ctx: &mut MetalContext, f: &Fixture, rows: usize) -> (Vec<f32>, Vec<f32>) {
    let (h, k) = (f.hidden, f.taps);
    let tail = ctx.new_buffer_with_data(&f16_le(&f.tail));
    let weight = ctx.new_buffer_with_data(&bf16_le(&f.weight));
    let in_proj = ctx.new_buffer_with_data(&f16_le(&f.in_proj));
    let out = ctx.new_output_buffer((rows * h * 2) as u64);
    for t in 0..rows {
        let pass = ctx.begin_pass();
        encode_short_conv_decode(
            ctx,
            &pass,
            (&tail, 0),
            (&in_proj, (t * 3 * h * 2) as u64),
            (&weight, 0),
            (&out, (t * h * 2) as u64),
            h as u32,
            k as u32,
        )
        .expect("encode decode");
        pass.commit_and_wait();
    }
    (read_halfs(&out, rows * h), read_halfs(&tail, (k - 1) * h))
}

/// GPU prefill over consecutive chunks of the given sizes.
fn gpu_prefill_chunks(
    ctx: &mut MetalContext,
    f: &Fixture,
    chunks: &[usize],
) -> (Vec<f32>, Vec<f32>) {
    let (h, k) = (f.hidden, f.taps);
    let rows: usize = chunks.iter().sum();
    let tail = ctx.new_buffer_with_data(&f16_le(&f.tail));
    let weight = ctx.new_buffer_with_data(&bf16_le(&f.weight));
    let in_proj = ctx.new_buffer_with_data(&f16_le(&f.in_proj));
    let out = ctx.new_output_buffer((rows * h * 2) as u64);
    let mut at = 0;
    for &n in chunks {
        let pass = ctx.begin_pass();
        encode_short_conv_prefill(
            ctx,
            &pass,
            (&tail, 0),
            (&in_proj, (at * 3 * h * 2) as u64),
            (&weight, 0),
            (&out, (at * h * 2) as u64),
            h as u32,
            k as u32,
            n as u32,
        )
        .expect("encode prefill");
        pass.commit_and_wait();
        at += n;
    }
    (read_halfs(&out, rows * h), read_halfs(&tail, (k - 1) * h))
}

fn reference(f: &Fixture, rows: usize) -> (Vec<f32>, Vec<f32>) {
    let mut tail = f.tail.clone();
    let out = short_conv_reference(&f.in_proj, &f.weight, &mut tail, rows, f.hidden, f.taps);
    (out, tail)
}

fn bits(v: &[f32]) -> Vec<u16> {
    v.iter().map(|x| f16::from_f32(*x).to_bits()).collect()
}

#[test]
fn decode_matches_reference_lfm3_width() {
    let mut ctx = MetalContext::new().expect("Metal device");
    for zero_tail in [true, false] {
        let rows = 10; // > K - 1, so later steps read only real rows
        let f = fixture(LFM3_HIDDEN, LFM3_TAPS, rows, zero_tail, 0x5107_c0de);
        let (got, got_tail) = gpu_decode_steps(&mut ctx, &f, rows);
        let (want, want_tail) = reference(&f, rows);
        assert_within_one_ulp(&got, &want, "decode out");
        // Tail values are u rounded once on both sides: exact.
        assert_eq!(bits(&got_tail), bits(&want_tail), "decode tail");
        assert!(got.iter().any(|x| *x != 0.0), "decode output is all zero");
    }
}

#[test]
fn prefill_matches_reference_lfm3_width() {
    let mut ctx = MetalContext::new().expect("Metal device");
    for zero_tail in [true, false] {
        let rows = 33;
        let f = fixture(LFM3_HIDDEN, LFM3_TAPS, rows, zero_tail, 0xfeed_0007);
        let (got, got_tail) = gpu_prefill_chunks(&mut ctx, &f, &[rows]);
        let (want, want_tail) = reference(&f, rows);
        assert_within_one_ulp(&got, &want, "prefill out");
        assert_eq!(bits(&got_tail), bits(&want_tail), "prefill tail");
    }
}

/// Chunk-equals-N-decode-steps, bit-exact, over chunk splits that cover
/// T < K-1 (1, 2, 5), T == K-1 (6), a chunk spanning the tail (9), and a
/// non-multiple-of-256 channel count, from a zero and a full tail.
#[test]
fn prefill_chunks_equal_decode_steps() {
    let mut ctx = MetalContext::new().expect("Metal device");
    let splits: &[&[usize]] = &[
        &[1, 1, 1, 1, 1, 1, 1, 1],
        &[2, 5, 1],
        &[6, 2],
        &[9],
        &[1, 9, 3],
    ];
    for &hidden in &[LFM3_HIDDEN, 300] {
        for zero_tail in [true, false] {
            for (s, split) in splits.iter().enumerate() {
                let rows: usize = split.iter().sum();
                let f = fixture(hidden, LFM3_TAPS, rows, zero_tail, 0xabc0 + s as u64);
                let (dec, dec_tail) = gpu_decode_steps(&mut ctx, &f, rows);
                let (pre, pre_tail) = gpu_prefill_chunks(&mut ctx, &f, split);
                let what = format!("hidden {hidden} zero_tail {zero_tail} split {split:?}");
                assert_eq!(bits(&pre), bits(&dec), "{what}: prefill out != decode out");
                assert_eq!(
                    bits(&pre_tail),
                    bits(&dec_tail),
                    "{what}: prefill tail != decode tail"
                );
                let (want, _) = reference(&f, rows);
                assert_within_one_ulp(&pre, &want, &what);
            }
        }
    }
}

/// The state holder: sized `[K-1, H]` fp16 per conv layer, absent elsewhere,
/// zeroed on creation and by `reset`.
#[test]
fn state_manager_sizes_and_resets() {
    let mut ctx = MetalContext::new().expect("Metal device");
    let mask = [true, false, true];
    let mut state = ShortConvStateManager::new(ctx.device(), &mask, LFM3_HIDDEN, LFM3_TAPS);
    let tail_len = (LFM3_TAPS - 1) * LFM3_HIDDEN;
    assert_eq!(state.tail_bytes_per_layer, tail_len * 2);
    assert!(state.is_conv(0) && !state.is_conv(1) && state.is_conv(2));
    assert_eq!(state.tail_buffer(2).length() as usize, tail_len * 2);
    assert!(read_halfs(state.tail_buffer(0), tail_len)
        .iter()
        .all(|x| *x == 0.0));

    // Advance layer 0's tail by one real decode step, then reset.
    let f = fixture(LFM3_HIDDEN, LFM3_TAPS, 1, true, 7);
    let weight = ctx.new_buffer_with_data(&bf16_le(&f.weight));
    let in_proj = ctx.new_buffer_with_data(&f16_le(&f.in_proj));
    let out = ctx.new_output_buffer((LFM3_HIDDEN * 2) as u64);
    let pass = ctx.begin_pass();
    encode_short_conv_decode(
        &mut ctx,
        &pass,
        (state.tail_buffer(0), 0),
        (&in_proj, 0),
        (&weight, 0),
        (&out, 0),
        LFM3_HIDDEN as u32,
        LFM3_TAPS as u32,
    )
    .expect("encode decode");
    pass.commit_and_wait();
    assert!(read_halfs(state.tail_buffer(0), tail_len)
        .iter()
        .any(|x| *x != 0.0));
    state.reset();
    assert!(read_halfs(state.tail_buffer(0), tail_len)
        .iter()
        .all(|x| *x == 0.0));
}

#[test]
fn rejects_short_views() {
    let mut ctx = MetalContext::new().expect("Metal device");
    let (h, k) = (64u32, 7u32);
    let tail = ctx.new_output_buffer(u64::from((k - 1) * h * 2));
    let weight = ctx.new_output_buffer(u64::from(h * k * 2));
    let in_proj = ctx.new_output_buffer(u64::from(3 * h * 2));
    let out = ctx.new_output_buffer(u64::from(h * 2));
    let pass = ctx.begin_pass();
    // Two rows of in_proj into a one-row buffer.
    let err = encode_short_conv_prefill(
        &mut ctx,
        &pass,
        (&tail, 0),
        (&in_proj, 0),
        (&weight, 0),
        (&out, 0),
        h,
        k,
        2,
    );
    assert!(
        err.is_err(),
        "prefill accepted an in_proj view past its buffer"
    );
    let err = encode_short_conv_decode(
        &mut ctx,
        &pass,
        (&tail, 2),
        (&in_proj, 0),
        (&weight, 0),
        (&out, 0),
        h,
        k,
    );
    assert!(err.is_err(), "decode accepted a tail view past its buffer");
    pass.commit_and_wait();
}
