#![cfg(target_os = "macos")]
//! Parity test for `moe_gguf_ext.metal`: the routed-expert decode pair for the
//! element-addressed block types (legacy Q4_0/Q4_1/Q5_0/Q5_1 and the ext
//! family Q1_0/TQ2_0/TQ1_0/NVFP4) against `turbospark_compute`'s decoders and
//! gated activation, on real Metal hardware, reading expert blobs through a
//! real argument buffer exactly as the runtime does. The harness is the Q8_0
//! one's in `moe_gguf_parity.rs`: a GGUF blob is three contiguous byte runs
//! with the six scale/bias offsets zero.
//!
//! Bytes are BUILT, not quantized: every byte pattern is a valid block, so the
//! CPU reference and the GPU get identical bytes and only FP16 rounding and
//! reduction order may differ. Phase 1 and phase 2 take INDEPENDENT types in
//! the mixed cases, because a layer can read one layout in each.

use half::f16;
use turbospark_gpu::{
    encode_inkling2_gguf_phase1, encode_inkling2_gguf_phase2, encode_moe_phase1_ext,
    encode_moe_phase2_ext, Inkling2GgufExpertType, MetalContext, MoeExpertOffsets, MoeExtType,
    RoutedBlobsBuffer, MAX_STREAMED_EXPERTS,
};

const FIXED_SLOTS: usize = 8;
const DIMS: (usize, usize) = (256, 256);

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

fn f16_bytes(v: f32) -> [u8; 2] {
    f16::from_f32(v).to_bits().to_le_bytes()
}

/// One row of valid blocks. Scales stay small so a whole expert stays in FP16.
fn rnd(s: &mut u32, n: usize, out: &mut Vec<u8>) {
    for _ in 0..n {
        out.push((lcg(s) & 0xFF) as u8);
    }
}

/// Smooth weights in [-1, 1] for the types this port has an encoder for.
fn smooth(cols: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(7);
    (0..cols)
        .map(|_| ((lcg(&mut s) >> 8) as f32 / (1u32 << 23) as f32) - 1.0)
        .collect()
}

/// Rows for the K and IQ types. K-quants with an encoder use it. Q5_K and the
/// IQ types have none (an IQ encoder is a codebook search nothing calls), so
/// their rows are random VALID code points under a per-type scale chosen to
/// put the largest representable weight just under 1.0 (the dynamic-range
/// trap: a shared scale makes the gate-times-up chain overflow FP16 on both
/// sides, which reads as a kernel bug and is a fixture bug).
fn kiq_row(kind: MoeExtType, cols: usize, seed: u32) -> Option<Vec<u8>> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let (d_bits, payload, per_block): (u16, usize, usize) = match kind {
        MoeExtType::Q2K => return Some(turbospark_compute::quantize_q2_k(&smooth(cols, seed))),
        MoeExtType::Q3K => return Some(turbospark_compute::quantize_q3_k(&smooth(cols, seed))),
        MoeExtType::Q6K => return Some(turbospark_compute::quantize_q6_k(&smooth(cols, seed))),
        // d, dmin, 12 scale bytes, 32 qh, 128 qs; the second f16 is written below.
        MoeExtType::Q5K => (0x1000, 172, 256),
        MoeExtType::Iq4Nl => (0x2000, 16, 32),
        MoeExtType::Iq4Xs => (0x0C00, 134, 256),
        MoeExtType::Iq3Xxs => (0x1800, 96, 256),
        MoeExtType::Iq2Xxs => (0x2800, 64, 256),
        MoeExtType::Iq2S => (0x2800, 80, 256),
        MoeExtType::Iq2Xs => (0x2400, 72, 256),
        MoeExtType::Iq1S => (0x2800, 48, 256),
        MoeExtType::Iq3S => (0x1400, 108, 256),
        MoeExtType::Iq1M => (0, 0, 256),
        MoeExtType::Iq1Xs => (0x2800, 44, 256),
        MoeExtType::Iq1Xxs => (0x2800, 40, 256),
        MoeExtType::Iq1Xxxs => (0x2800, 36, 256),
        _ => return None,
    };
    let mut out = Vec::new();
    for _ in 0..cols / per_block {
        if kind == MoeExtType::Iq1M {
            for _ in 0..48 {
                out.push((lcg(&mut s) & 0xFF) as u8);
            }
            // IQ1_M stores d in the top nibbles of its four u16 sub-scales;
            // these reconstruct f16 0x3000 (0.125).
            let d = 0x3000u16;
            for shift in [0, 4, 8, 12] {
                out.extend_from_slice(&(((d >> shift) & 0xF) << 12).to_le_bytes());
            }
            continue;
        }
        // Placeholder scale (1.0); the real one is fitted below from the
        // row's decoded peak, because a fixed per-type constant overflowed FP16
        // through the gate-times-up-then-down chain for several types.
        let _ = d_bits;
        out.extend_from_slice(&0x3C00u16.to_le_bytes());
        if kind == MoeExtType::Q5K {
            out.extend_from_slice(&0x3C00u16.to_le_bytes());
        }
        for _ in 0..payload {
            out.push((lcg(&mut s) & 0xFF) as u8);
        }
    }
    if kind != MoeExtType::Iq1M {
        // Fit the scale: decode at d = 1, then choose the power of two that
        // puts the row's largest weight in [0.25, 0.5]. Every block's scale
        // sits at offset 0 (Q5_K's second f16 follows it).
        let peak = decode(kind, &out, cols)
            .iter()
            .fold(0f32, |m, v| m.max(v.abs()));
        let scale = 2f32.powi((0.5 / peak.max(1e-6)).log2().floor() as i32);
        let bits = f16::from_f32(scale).to_bits().to_le_bytes();
        let block_bytes = kind.block_bytes();
        for block in out.chunks_exact_mut(block_bytes) {
            block[..2].copy_from_slice(&bits);
            if kind == MoeExtType::Q5K {
                block[2..4].copy_from_slice(&bits);
            }
        }
    }
    Some(out)
}

fn row(kind: MoeExtType, cols: usize, seed: u32) -> Vec<u8> {
    if let Some(bytes) = kiq_row(kind, cols, seed) {
        assert_eq!(bytes.len(), kind.row_bytes(cols), "{kind:?} row size");
        return bytes;
    }
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut out = Vec::new();
    for _ in 0..cols / kind.block_elems() {
        let sign = if lcg(&mut s) & 1 == 0 { 1.0 } else { -1.0 };
        let d = sign * (0.01 + (lcg(&mut s) % 8) as f32 * 0.004);
        let m = ((lcg(&mut s) % 129) as f32 - 64.0) * 0.004;
        match kind {
            MoeExtType::Q4_0 => {
                out.extend_from_slice(&f16_bytes(d));
                rnd(&mut s, 16, &mut out);
            }
            MoeExtType::Q4_1 => {
                out.extend_from_slice(&f16_bytes(d));
                out.extend_from_slice(&f16_bytes(m));
                rnd(&mut s, 16, &mut out);
            }
            MoeExtType::Q5_0 => {
                out.extend_from_slice(&f16_bytes(d));
                rnd(&mut s, 4 + 16, &mut out);
            }
            MoeExtType::Q5_1 => {
                out.extend_from_slice(&f16_bytes(d));
                out.extend_from_slice(&f16_bytes(m));
                rnd(&mut s, 4 + 16, &mut out);
            }
            MoeExtType::Q1_0 | MoeExtType::Q2_0 => {
                out.extend_from_slice(&f16_bytes(d));
                rnd(&mut s, 16, &mut out);
            }
            MoeExtType::Tq2_0 | MoeExtType::Tq1_0 => {
                rnd(&mut s, kind.block_bytes() - 2, &mut out);
                out.extend_from_slice(&f16_bytes(d));
            }
            MoeExtType::Nvfp4 => {
                for _ in 0..4 {
                    // Scale bytes 0x20..0x2F (0.125 .. 0.47), bit 7 sometimes set.
                    let b = 0x20 + (lcg(&mut s) % 0x10) as u8;
                    out.push(if lcg(&mut s) & 1 == 0 { b } else { b | 0x80 });
                }
                rnd(&mut s, 32, &mut out);
            }
            _ => unreachable!("{kind:?} is built by kiq_row"),
        }
    }
    assert_eq!(out.len(), kind.row_bytes(cols));
    out
}

fn decode(kind: MoeExtType, bytes: &[u8], n: usize) -> Vec<f32> {
    match kind {
        MoeExtType::Q4_0 => turbospark_compute::dequantize_q4_0(bytes, n),
        MoeExtType::Q4_1 => turbospark_compute::dequantize_q4_1(bytes, n),
        MoeExtType::Q5_0 => turbospark_compute::dequantize_q5_0(bytes, n),
        MoeExtType::Q5_1 => turbospark_compute::dequantize_q5_1(bytes, n),
        MoeExtType::Q1_0 => turbospark_compute::dequantize_q1_0(bytes, n),
        MoeExtType::Q2_0 => turbospark_compute::dequantize_q2_0(bytes, n),
        MoeExtType::Tq2_0 => turbospark_compute::dequantize_tq2_0(bytes, n),
        MoeExtType::Tq1_0 => turbospark_compute::dequantize_tq1_0(bytes, n),
        MoeExtType::Nvfp4 => turbospark_compute::dequantize_nvfp4(bytes, n),
        MoeExtType::Q5K => turbospark_compute::dequantize_q5_k(bytes, n),
        MoeExtType::Q3K => turbospark_compute::dequantize_q3_k(bytes, n),
        MoeExtType::Q2K => turbospark_compute::dequantize_q2_k(bytes, n),
        MoeExtType::Q6K => turbospark_compute::dequantize_q6_k(bytes, n),
        MoeExtType::Iq2Xs => turbospark_compute::dequantize_iq2_xs(bytes, n),
        MoeExtType::Iq1S => turbospark_compute::dequantize_iq1_s(bytes, n),
        MoeExtType::Iq3S => turbospark_compute::dequantize_iq3_s(bytes, n),
        MoeExtType::Iq3Xxs => turbospark_compute::dequantize_iq3_xxs(bytes, n),
        MoeExtType::Iq4Xs => turbospark_compute::dequantize_iq4_xs(bytes, n),
        MoeExtType::Iq2S => turbospark_compute::dequantize_iq2_s(bytes, n),
        MoeExtType::Iq2Xxs => turbospark_compute::dequantize_iq2_xxs(bytes, n),
        MoeExtType::Iq1M => turbospark_compute::dequantize_iq1_m(bytes, n),
        MoeExtType::Iq4Nl => turbospark_compute::dequantize_iq4_nl(bytes, n),
        MoeExtType::Iq1Xs => turbospark_compute::dequantize_iq1_xs(bytes, n),
        MoeExtType::Iq1Xxs => turbospark_compute::dequantize_iq1_xxs(bytes, n),
        MoeExtType::Iq1Xxxs => turbospark_compute::dequantize_iq1_xxxs(bytes, n),
    }
}

fn gemv(kind: MoeExtType, rows: &[Vec<u8>], x: &[f32], n: usize) -> Vec<f32> {
    rows.iter()
        .map(|r| decode(kind, r, n).iter().zip(x).map(|(w, x)| w * x).sum())
        .collect()
}

fn to_le(v: &[f16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

fn read_f16(buffer: &metal::Buffer, n: usize) -> Vec<f32> {
    let ptr = buffer.contents() as *const u16;
    let bits = unsafe { std::slice::from_raw_parts(ptr, n) };
    bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
}

type Expert = (Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<Vec<u8>>);

fn run_case(gate_up: MoeExtType, down: MoeExtType, use_silu: bool, top_k: usize) {
    let mut context = MetalContext::new().expect("Metal device");
    let (d_dim, f_dim) = DIMS;
    let rows = |kind: MoeExtType, n: usize, cols: usize, seed: u32| -> Vec<Vec<u8>> {
        (0..n)
            .map(|r| row(kind, cols, seed + r as u32 * 97))
            .collect()
    };
    let experts: Vec<Expert> = (0..top_k)
        .map(|e| {
            let seed = 5000 + 100 * e as u32;
            (
                rows(gate_up, f_dim, d_dim, seed + 1),
                rows(gate_up, f_dim, d_dim, seed + 2),
                rows(down, d_dim, f_dim, seed + 3),
            )
        })
        .collect();

    let x32: Vec<f32> = (0..d_dim).map(|i| ((i as f32) * 0.21).sin()).collect();
    let x16: Vec<f16> = x32.iter().map(|&v| f16::from_f32(v)).collect();
    let x_rounded: Vec<f32> = x16.iter().map(|v| v.to_f32()).collect();
    let residual16: Vec<f16> = (0..d_dim)
        .map(|i| f16::from_f32(((i as f32) * 0.11).cos() * 0.5))
        .collect();
    let weights: Vec<f32> = (0..top_k)
        .map(|e| 0.9 - 0.5 * e as f32 / top_k as f32)
        .collect();

    let mut expected: Vec<f32> = residual16.iter().map(|v| v.to_f32()).collect();
    for (e, (g, u, d)) in experts.iter().enumerate() {
        let (go, uo) = (
            gemv(gate_up, g, &x_rounded, d_dim),
            gemv(gate_up, u, &x_rounded, d_dim),
        );
        let act: Vec<f32> = if use_silu {
            go.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
        } else {
            turbospark_compute::gelu_tanh(&go)
        };
        let acts: Vec<f32> = act
            .iter()
            .zip(&uo)
            .map(|(&a, &u)| f16::from_f32(a * u).to_f32())
            .collect();
        let out = gemv(down, d, &acts, f_dim);
        for (dst, o) in expected.iter_mut().zip(&out) {
            *dst += weights[e] * o;
        }
    }

    let mut blobs = Vec::new();
    let mut offsets = None;
    for (g, u, d) in &experts {
        let mut blob = Vec::new();
        let mut at = [0u32; 3];
        for (i, set) in [g, u, d].into_iter().enumerate() {
            at[i] = blob.len() as u32;
            for r in set {
                blob.extend_from_slice(r);
            }
        }
        offsets.get_or_insert(MoeExpertOffsets {
            gate_w: at[0],
            gate_s: 0,
            gate_b: 0,
            up_w: at[1],
            up_s: 0,
            up_b: 0,
            down_w: at[2],
            down_s: 0,
            down_b: 0,
        });
        blobs.push(context.new_buffer_with_data(&blob));
    }
    let offsets = offsets.unwrap();

    let x_buf = context.new_buffer_with_data(&to_le(&x16));
    let residual_buf = context.new_buffer_with_data(&to_le(&residual16));
    // All eight slots sized and zeroed: phase 2 reduces every slot
    // unconditionally and a garbage row times a zero weight is still NaN.
    let acts_buf = context.new_buffer_with_data(&vec![0u8; MAX_STREAMED_EXPERTS * f_dim * 2]);
    let y_buf = context.new_output_buffer((d_dim * 2) as u64);
    let mut routing = vec![f16::from_f32(0.0); MAX_STREAMED_EXPERTS];
    for (slot, &w) in weights.iter().enumerate() {
        routing[slot] = f16::from_f32(w);
    }
    let routing_buf = context.new_buffer_with_data(&to_le(&routing));

    let routed = RoutedBlobsBuffer::new(&mut context, use_silu).expect("arg buffer");
    let refs: Vec<(&metal::Buffer, u64)> = blobs.iter().map(|b| (b, 0u64)).collect();
    routed.bind(&mut context, use_silu, &refs).expect("bind");

    let pass = context.begin_pass();
    for blob in &blobs {
        pass.use_read_buffer(blob);
    }
    encode_moe_phase1_ext(
        &mut context,
        &pass,
        gate_up,
        &routed,
        &offsets,
        (&x_buf, 0),
        (&acts_buf, 0),
        d_dim as u32,
        f_dim as u32,
        top_k as u32,
        use_silu,
    )
    .expect("phase1");
    encode_moe_phase2_ext(
        &mut context,
        &pass,
        down,
        &routed,
        &offsets,
        (&acts_buf, 0),
        (&routing_buf, 0),
        (&residual_buf, 0),
        (&y_buf, 0),
        d_dim as u32,
        f_dim as u32,
        top_k as u32,
        use_silu,
    )
    .expect("phase2");
    pass.commit_and_wait();

    let got = read_f16(&y_buf, d_dim);
    let mut peak = 0f32;
    for d in 0..d_dim {
        let want = expected[d];
        peak = peak.max(want.abs());
        // The eight-slot cases keep exactly the thresholds they were set with.
        // Past eight slots (the ten-slot kernels) the fixtures' partial sums
        // can be thousands (NVFP4's scales) while an output lands near zero by
        // cancellation, and FP16 rounding of those partials is proportional to
        // THEIR size, not the result's: so the floor also gets a term of 0.1%
        // of the largest expected output. A layout or order bug is O(1) of the
        // output, not 0.1% of its peak.
        let mut tol = 5e-2_f32.max(want.abs() * 3e-2);
        if top_k > FIXED_SLOTS {
            let peak_all = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
            tol = tol.max(peak_all * 1e-3);
        }
        assert!(
            (got[d] - want).abs() <= tol,
            "{gate_up:?}/{down:?} silu={use_silu} top_k={top_k} d={d}: got {} want {want}",
            got[d]
        );
    }
    // A fixture that decodes to ~0 everywhere would pass any kernel.
    assert!(
        peak > 0.1,
        "{gate_up:?}/{down:?}: expected output too small to prove anything: {peak}"
    );
}

#[test]
fn every_ext_type_matches_the_cpu_reference_in_both_phases() {
    for kind in MoeExtType::ALL {
        run_case(kind, kind, false, 2);
    }
}

#[test]
fn the_silu_constant_reaches_every_ext_kernel() {
    for kind in MoeExtType::ALL {
        run_case(kind, kind, true, 2);
    }
}

#[test]
fn all_eight_slots_participate_for_every_ext_type() {
    for kind in MoeExtType::ALL {
        run_case(kind, kind, false, FIXED_SLOTS);
    }
}

/// Phase 1 and phase 2 read INDEPENDENT layouts: a kernel that reused its
/// phase-1 geometry in phase 2 would stride the wrong block.
#[test]
fn mixed_layouts_decode_independently() {
    run_case(MoeExtType::Q4_0, MoeExtType::Tq2_0, false, 3);
    run_case(MoeExtType::Tq1_0, MoeExtType::Q5_1, false, 3);
    run_case(MoeExtType::Nvfp4, MoeExtType::Q1_0, true, 3);
}

/// Exactly ten routed experts (Qwen3-Next's top-10) take the ten-slot kernel
/// for every macro-generated type, against the same CPU reference as the
/// eight-slot cases. The hand-written phase 2 types (IQ4_NL, Q6_K) and Q2_0
/// (its own `encode_moe_phase2_q2_0_top10`) have no ext k10 and are refused.
#[test]
fn ten_slots_participate_for_every_ext_type_with_a_k10_kernel() {
    for kind in MoeExtType::ALL {
        if matches!(kind, MoeExtType::Iq4Nl | MoeExtType::Q6K | MoeExtType::Q2_0) {
            continue;
        }
        run_case(kind, kind, false, 10);
    }
}

#[test]
#[should_panic]
fn ten_slots_are_refused_for_a_hand_written_phase_two_type() {
    run_case(MoeExtType::Q4_0, MoeExtType::Iq4Nl, false, 10);
}

#[test]
#[should_panic]
fn nine_slots_panics_rather_than_silently_truncating() {
    run_case(MoeExtType::Q4_0, MoeExtType::Q4_0, false, FIXED_SLOTS + 1);
}

// ---------------------------------------------------------------------------
// Inkling 2's mode over the same kernels (function constants 112-114): outer
// scales from a record at the front of each blob, the softcapped SwiGLU, an
// outer-scaled down value, and NO residual. The residual slot is bound to the
// acts buffer by the API, so a kernel that still added it would be off by an
// acts row, and the cases below would fail.
// ---------------------------------------------------------------------------

type Ink = Inkling2GgufExpertType;

fn ink_row(kind: Ink, cols: usize, seed: u32) -> Vec<u8> {
    match kind {
        Ink::Q8_0 => turbospark_compute::quantize_q8_0(&smooth(cols, seed)),
        Ink::Q4K => turbospark_compute::quantize_q4_k(&smooth(cols, seed)),
        Ink::Ext(e) => row(e, cols, seed),
        Ink::Mxfp4 => mxfp4_row(cols, seed),
    }
}

/// Random MXFP4 blocks: an E8M0 scale near 2^-5 then 16 bytes of E2M1
/// nibble pairs, so the dequantized weights are a few tenths at most.
fn mxfp4_row(cols: usize, seed: u32) -> Vec<u8> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut next = || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (s >> 8) as u8
    };
    let mut v = Vec::with_capacity(cols / 32 * 17);
    for _ in 0..cols / 32 {
        v.push(120 + next() % 4);
        for _ in 0..16 {
            v.push(next());
        }
    }
    v
}

fn ink_decode(kind: Ink, bytes: &[u8], n: usize) -> Vec<f32> {
    match kind {
        Ink::Q8_0 => turbospark_compute::dequantize_q8_0(bytes, n),
        Ink::Q4K => turbospark_compute::dequantize_q4_k(bytes, n),
        Ink::Ext(e) => decode(e, bytes, n),
        Ink::Mxfp4 => turbospark_compute::dequantize_mxfp4(bytes, n),
    }
}

fn ink_gemv(kind: Ink, rows: &[Vec<u8>], x: &[f32], n: usize) -> Vec<f32> {
    rows.iter()
        .map(|r| {
            ink_decode(kind, r, n)
                .iter()
                .zip(x)
                .map(|(w, x)| w * x)
                .sum()
        })
        .collect()
}

fn run_inkling_case(gate_up: Ink, down: Ink, top_k: usize) {
    let mut context = MetalContext::new().expect("Metal device");
    let (d_dim, f_dim) = DIMS;
    let (gate_cap, up_cap) = (4.0f32, 25.0f32);
    let rows = |kind: Ink, n: usize, cols: usize, seed: u32| -> Vec<Vec<u8>> {
        (0..n)
            .map(|r| ink_row(kind, cols, seed + r as u32 * 97))
            .collect()
    };
    // Outer scales differ per expert and per projection, so a kernel that
    // read the wrong field, or none, is off.
    let outers: Vec<[f32; 3]> = (0..top_k)
        .map(|e| {
            [
                1.5 + 0.25 * e as f32,
                0.75 + 0.125 * e as f32,
                0.5 + 0.0625 * e as f32,
            ]
        })
        .collect();
    let experts: Vec<Expert> = (0..top_k)
        .map(|e| {
            let seed = 7000 + 100 * e as u32;
            (
                rows(gate_up, f_dim, d_dim, seed + 1),
                rows(gate_up, f_dim, d_dim, seed + 2),
                rows(down, d_dim, f_dim, seed + 3),
            )
        })
        .collect();
    let x32: Vec<f32> = (0..d_dim)
        .map(|i| ((i as f32) * 0.21).sin() * 2.0)
        .collect();
    let x16: Vec<f16> = x32.iter().map(|&v| f16::from_f32(v)).collect();
    let xr: Vec<f32> = x16.iter().map(|v| v.to_f32()).collect();
    let weights: Vec<f32> = (0..top_k)
        .map(|e| 0.9 - 0.5 * e as f32 / top_k as f32)
        .collect();

    let mut expected = vec![0.0f32; d_dim];
    for (e, (g, u, d)) in experts.iter().enumerate() {
        let o = outers[e];
        let go = ink_gemv(gate_up, g, &xr, d_dim);
        let uo = ink_gemv(gate_up, u, &xr, d_dim);
        let acts: Vec<f32> = go
            .iter()
            .zip(&uo)
            .map(|(&g, &u)| {
                f16::from_f32(turbospark_compute::inkling2::situ(
                    g * o[0],
                    u * o[1],
                    gate_cap,
                    up_cap,
                ))
                .to_f32()
            })
            .collect();
        let out = ink_gemv(down, d, &acts, f_dim);
        for (dst, v) in expected.iter_mut().zip(&out) {
            *dst += f16::from_f32(weights[e]).to_f32() * v * o[2];
        }
    }

    let mut blobs = Vec::new();
    let mut offsets = None;
    for ((g, u, d), o) in experts.iter().zip(&outers) {
        let mut blob: Vec<u8> = o
            .iter()
            .chain(&[0.0f32])
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut at = [0u32; 3];
        for (i, set) in [g, u, d].into_iter().enumerate() {
            at[i] = blob.len() as u32;
            for r in set {
                blob.extend_from_slice(r);
            }
        }
        offsets.get_or_insert(MoeExpertOffsets {
            gate_w: at[0],
            gate_s: 0,
            gate_b: 0,
            up_w: at[1],
            up_s: 4,
            up_b: 0,
            down_w: at[2],
            down_s: 8,
            down_b: 0,
        });
        blobs.push(context.new_buffer_with_data(&blob));
    }
    // Phase 2 reduces all eight slots: pad with the last blob, zero weight.
    while blobs.len() < FIXED_SLOTS {
        let last = blobs.last().unwrap().clone();
        blobs.push(last);
    }
    let offsets = offsets.unwrap();
    let x_buf = context.new_buffer_with_data(&to_le(&x16));
    // Nonzero acts past top_k: a reduce that read them unweighted would show.
    let mut acts_init = vec![0u8; MAX_STREAMED_EXPERTS * f_dim * 2];
    for b in acts_init.chunks_exact_mut(2).skip(top_k * f_dim) {
        b.copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
    }
    let acts_buf = context.new_buffer_with_data(&acts_init);
    let y_buf = context.new_output_buffer((d_dim * 2) as u64);
    let mut routing = vec![f16::from_f32(0.0); MAX_STREAMED_EXPERTS];
    for (slot, &w) in weights.iter().enumerate() {
        routing[slot] = f16::from_f32(w);
    }
    let routing_buf = context.new_buffer_with_data(&to_le(&routing));
    let routed = RoutedBlobsBuffer::new_for(
        &mut context,
        turbospark_gpu::moe_gguf_source(),
        "moe_phase1_gate_up_act_q8_0",
        false,
    )
    .expect("arg buffer");
    let refs: Vec<(&metal::Buffer, u64)> = blobs.iter().map(|b| (b, 0u64)).collect();
    routed
        .bind_for(
            &mut context,
            turbospark_gpu::moe_gguf_source(),
            "moe_phase1_gate_up_act_q8_0",
            false,
            &refs,
        )
        .expect("bind");

    let pass = context.begin_pass();
    for blob in &blobs {
        pass.use_read_buffer(blob);
    }
    encode_inkling2_gguf_phase1(
        &mut context,
        &pass,
        gate_up,
        &routed,
        &offsets,
        (&x_buf, 0),
        (&acts_buf, 0),
        d_dim as u32,
        f_dim as u32,
        top_k as u32,
        gate_cap,
        up_cap,
    )
    .expect("phase1");
    encode_inkling2_gguf_phase2(
        &mut context,
        &pass,
        down,
        &routed,
        &offsets,
        (&acts_buf, 0),
        (&routing_buf, 0),
        (&y_buf, 0),
        d_dim as u32,
        f_dim as u32,
        top_k as u32,
    )
    .expect("phase2");
    pass.commit_and_wait();

    let got = read_f16(&y_buf, d_dim);
    let mut peak = 0f32;
    for d in 0..d_dim {
        let want = expected[d];
        peak = peak.max(want.abs());
        let tol = 5e-2_f32.max(want.abs() * 3e-2);
        assert!(
            (got[d] - want).abs() <= tol,
            "inkling {gate_up:?}/{down:?} top_k={top_k} d={d}: got {} want {want}",
            got[d]
        );
    }
    assert!(
        peak > 0.1,
        "inkling {gate_up:?}/{down:?}: expected output too small to prove anything: {peak}"
    );
}

fn ink_all() -> Vec<Ink> {
    let mut v = vec![Ink::Q8_0, Ink::Q4K];
    v.extend(MoeExtType::ALL.iter().map(|&e| Ink::Ext(e)));
    v
}

#[test]
fn inkling_mode_matches_the_reference_for_every_type() {
    for kind in ink_all() {
        // Phase 2 needs the expert width tiled by the block: types with
        // 256-element blocks run phase 2 at 256 here (DIMS), as phase 1 does.
        run_inkling_case(kind, kind, 2);
    }
}

#[test]
fn inkling_mode_reduces_all_eight_slots() {
    for kind in [
        Ink::Q8_0,
        Ink::Ext(MoeExtType::Iq4Nl),
        Ink::Ext(MoeExtType::Nvfp4),
        Ink::Ext(MoeExtType::Q2_0),
    ] {
        run_inkling_case(kind, kind, FIXED_SLOTS);
    }
}

/// The combinations the published Inkling GGUFs ship (gate-up / down).
#[test]
fn inkling_mode_published_mixes() {
    use MoeExtType as E;
    for (g, d) in [
        (Ink::Ext(E::Iq2Xxs), Ink::Ext(E::Iq4Nl)),
        (Ink::Ext(E::Iq2Xxs), Ink::Ext(E::Q2_0)),
        (Ink::Ext(E::Iq1S), Ink::Ext(E::Nvfp4)),
        (Ink::Ext(E::Iq3Xxs), Ink::Ext(E::Nvfp4)),
        (Ink::Q4K, Ink::Ext(E::Q5_0)),
        (Ink::Ext(E::Nvfp4), Ink::Q8_0),
        // UD-Q2_K_XL layer 20.
        (Ink::Ext(E::Iq2S), Ink::Mxfp4),
    ] {
        run_inkling_case(g, d, 3);
    }
}

/// MXFP4 down reduces all eight slots in the mode, like the other types.
#[test]
fn inkling_mode_mxfp4_down_reduces_all_eight_slots() {
    run_inkling_case(Ink::Q8_0, Ink::Mxfp4, FIXED_SLOTS);
}

/// An MXFP4 gate-up has no Inkling pair: phase 1 refuses it by name.
#[test]
fn inkling_mode_refuses_an_mxfp4_gate_up() {
    let mut context = MetalContext::new().expect("Metal device");
    let buf = context.new_output_buffer(4096);
    let pass = context.begin_pass();
    let offsets = MoeExpertOffsets {
        gate_w: 16,
        gate_s: 0,
        gate_b: 0,
        up_w: 16,
        up_s: 4,
        up_b: 0,
        down_w: 16,
        down_s: 8,
        down_b: 0,
    };
    let err = turbospark_gpu::encode_inkling2_gguf_phase1_table(
        &mut context,
        &pass,
        Ink::Mxfp4,
        (&buf, 0),
        &offsets,
        (&buf, 0),
        (&buf, 0),
        64,
        64,
        2,
        4.0,
        25.0,
    )
    .expect_err("an MXFP4 gate-up must be refused");
    assert!(format!("{err}").contains("MXFP4"), "{err}");
}
