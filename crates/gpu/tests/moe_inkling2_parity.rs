#![cfg(target_os = "macos")]
//! Parity test for `moe_inkling2.metal`: Inkling 2's routed-expert decode pair
//! against `turbospark_compute::inkling2`, on real Metal hardware, through a
//! real `RoutedBlobs` argument buffer.
//!
//! Experts are BUILT in the checkpoint's own HF layout (packed E2M1 nibbles
//! with E4M3 sub-block scales and a per-expert outer scale, or E4M3 bytes with
//! E8M0 exponents). The expected output decodes those HF bytes with the
//! reference decoders; the GPU reads them after the repack transcoders. So a
//! wrong nibble order, a dropped outer scale or a wrong activation all fail
//! here, not just a wrong kernel.

use half::f16;
use turbospark_compute::inkling2::{
    dequant_mxfp8_row, dequant_nvfp4_row, mxfp8_row_to_blocks, nvfp4_row_to_blocks, situ,
};
use turbospark_gpu::{
    encode_inkling2_moe_phase1, encode_inkling2_moe_phase1_indexed, encode_inkling2_moe_phase2,
    encode_inkling2_moe_phase2_indexed, moe_inkling2_source, Inkling2ExpertType,
    Inkling2ResidentLayer, MetalContext, MoeExpertOffsets, RoutedBlobsBuffer, INKLING2_MAX_SLOTS,
    INKLING2_OUTER_SCALE_BYTES, MAX_STREAMED_EXPERTS, MOE_INKLING2_REFLECT_FUNCTION,
};

/// Inkling 2 Mini: hidden 2304, expert width 1152.
const MINI: (usize, usize) = (2304, 1152);
const SMALL: (usize, usize) = (256, 128);
/// `mlp_gate_softcap`, `mlp_up_softcap`.
const MINI_CAPS: (f32, f32) = (4.0, 25.0);

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

/// One projection row in HF layout, plus its f32 reference decode.
struct HfRow {
    transcoded: Vec<u8>,
    reference: Vec<f32>,
}

/// An E4M3 byte with exponent field in `e_lo..=e_hi`, random mantissa, and a
/// random sign when `signed`.
fn e4m3(s: &mut u32, e_lo: u32, e_hi: u32, signed: bool) -> u8 {
    let r = lcg(s) >> 8;
    let e = e_lo + r % (e_hi - e_lo + 1);
    let m = (r >> 4) & 7;
    let sign = if signed && (r >> 8) & 1 == 1 { 0x80 } else { 0 };
    (sign | (e << 3) | m) as u8
}

fn hf_row(kind: Inkling2ExpertType, cols: usize, outer: f32, seed: u32) -> HfRow {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(11);
    let mut reference = vec![0.0; cols];
    let transcoded = match kind {
        Inkling2ExpertType::Nvfp4 => {
            let packed: Vec<u8> = (0..cols / 2).map(|_| (lcg(&mut s) >> 8) as u8).collect();
            // Scales 2^-6 .. ~2^-3: weights up to ~0.7 so a whole expert stays
            // well inside FP16 through gate-times-up-then-down.
            let scales: Vec<u8> = (0..cols / 16).map(|_| e4m3(&mut s, 1, 3, false)).collect();
            dequant_nvfp4_row(&packed, &scales, outer, &mut reference);
            nvfp4_row_to_blocks(&packed, &scales).unwrap()
        }
        Inkling2ExpertType::Mxfp8 => {
            assert_eq!(outer, 1.0, "MXFP8 experts have no outer scale");
            let bytes: Vec<u8> = (0..cols).map(|_| e4m3(&mut s, 4, 8, true)).collect();
            // 2^-5 or 2^-6 per block.
            let exps: Vec<u8> = (0..cols / 32)
                .map(|_| 121 + (lcg(&mut s) >> 9) as u8 % 2)
                .collect();
            dequant_mxfp8_row(&bytes, &exps, &mut reference);
            mxfp8_row_to_blocks(&bytes, &exps).unwrap()
        }
    };
    assert_eq!(transcoded.len(), kind.row_bytes(cols));
    HfRow {
        transcoded,
        reference,
    }
}

struct Expert {
    gate: Vec<HfRow>,
    up: Vec<HfRow>,
    down: Vec<HfRow>,
    outers: [f32; 3],
}

fn dot(w: &[f32], x: &[f32]) -> f32 {
    w.iter().zip(x).map(|(a, b)| a * b).sum()
}

fn to_le(v: &[f16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

fn read_f16(buffer: &metal::Buffer, n: usize) -> Vec<f32> {
    let ptr = buffer.contents() as *const u16;
    let bits = unsafe { std::slice::from_raw_parts(ptr, n) };
    bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
}

struct Case {
    gate_up: Inkling2ExpertType,
    down: Inkling2ExpertType,
    dims: (usize, usize),
    top_k: usize,
    caps: (f32, f32),
}

fn run_case(c: &Case) {
    let mut context = MetalContext::new().expect("Metal device");
    let (d_dim, f_dim) = c.dims;
    // NVFP4 outer scales are deliberately not powers of two, and differ
    // between w13 and w2, so dropping or swapping one cannot pass.
    let outer = |kind: Inkling2ExpertType, v: f32| match kind {
        Inkling2ExpertType::Nvfp4 => v,
        Inkling2ExpertType::Mxfp8 => 1.0,
    };
    let experts: Vec<Expert> = (0..c.top_k)
        .map(|e| {
            let seed = 7000 + 100 * e as u32;
            let w13 = outer(c.gate_up, 0.75 + 0.05 * e as f32);
            let w2 = outer(c.down, 1.3 - 0.03 * e as f32);
            let rows = |kind, n: usize, cols, o, sd: u32| -> Vec<HfRow> {
                (0..n)
                    .map(|r| hf_row(kind, cols, o, sd + r as u32 * 97))
                    .collect()
            };
            Expert {
                gate: rows(c.gate_up, f_dim, d_dim, w13, seed + 1),
                up: rows(c.gate_up, f_dim, d_dim, w13, seed + 2),
                down: rows(c.down, d_dim, f_dim, w2, seed + 3),
                outers: [w13, w13, w2],
            }
        })
        .collect();

    let x16: Vec<f16> = (0..d_dim)
        .map(|i| f16::from_f32(((i as f32) * 0.21).sin()))
        .collect();
    let x: Vec<f32> = x16.iter().map(|v| v.to_f32()).collect();
    // route_scale 8: the Mini's weights sum to 8.
    let raw: Vec<f32> = (0..c.top_k).map(|e| (c.top_k - e) as f32).collect();
    let total: f32 = raw.iter().sum();
    let weights16: Vec<f16> = raw.iter().map(|w| f16::from_f32(8.0 * w / total)).collect();

    let mut expected = vec![0.0f32; d_dim];
    for (e, ex) in experts.iter().enumerate() {
        let acts: Vec<f32> = ex
            .gate
            .iter()
            .zip(&ex.up)
            .map(|(g, u)| {
                let a = situ(
                    dot(&g.reference, &x),
                    dot(&u.reference, &x),
                    c.caps.0,
                    c.caps.1,
                );
                f16::from_f32(a).to_f32()
            })
            .collect();
        let w = weights16[e].to_f32();
        for (dst, row) in expected.iter_mut().zip(&ex.down) {
            *dst += w * dot(&row.reference, &acts);
        }
    }

    let mut blobs = Vec::new();
    let mut offsets = None;
    for ex in &experts {
        let mut blob = Vec::with_capacity(INKLING2_OUTER_SCALE_BYTES);
        for o in ex.outers {
            blob.extend_from_slice(&o.to_le_bytes());
        }
        blob.resize(INKLING2_OUTER_SCALE_BYTES, 0);
        let mut at = [0u32; 3];
        for (i, set) in [&ex.gate, &ex.up, &ex.down].into_iter().enumerate() {
            at[i] = blob.len() as u32;
            for r in set {
                blob.extend_from_slice(&r.transcoded);
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
    let offsets = offsets.unwrap();

    let x_buf = context.new_buffer_with_data(&to_le(&x16));
    // Every slot past top_k holds NaN activations and a zero weight: if phase
    // 2 let an unused simdgroup contribute, NaN * 0 would poison the output.
    let nan = f16::from_bits(0x7E00);
    let acts_buf = context.new_buffer_with_data(&to_le(&vec![nan; MAX_STREAMED_EXPERTS * f_dim]));
    let mut routing = vec![f16::from_f32(0.0); MAX_STREAMED_EXPERTS];
    routing[..c.top_k].copy_from_slice(&weights16);
    let routing_buf = context.new_buffer_with_data(&to_le(&routing));
    let y_buf = context.new_output_buffer((d_dim * 2) as u64);

    let source = moe_inkling2_source();
    let routed =
        RoutedBlobsBuffer::new_for(&mut context, source, MOE_INKLING2_REFLECT_FUNCTION, false)
            .expect("arg buffer");
    let refs: Vec<(&metal::Buffer, u64)> = blobs.iter().map(|b| (b, 0u64)).collect();
    routed
        .bind_for(
            &mut context,
            source,
            MOE_INKLING2_REFLECT_FUNCTION,
            false,
            &refs,
        )
        .expect("bind");

    let pass = context.begin_pass();
    for blob in &blobs {
        pass.use_read_buffer(blob);
    }
    encode_inkling2_moe_phase1(
        &mut context,
        &pass,
        c.gate_up,
        &routed,
        &offsets,
        (&x_buf, 0),
        (&acts_buf, 0),
        d_dim as u32,
        f_dim as u32,
        c.top_k as u32,
        c.caps.0,
        c.caps.1,
    )
    .expect("phase1");
    encode_inkling2_moe_phase2(
        &mut context,
        &pass,
        c.down,
        &routed,
        &offsets,
        (&acts_buf, 0),
        (&routing_buf, 0),
        (&y_buf, 0),
        d_dim as u32,
        f_dim as u32,
        c.top_k as u32,
    )
    .expect("phase2");
    pass.commit_and_wait();

    let got = read_f16(&y_buf, d_dim);

    // The indexed pair over the same experts: one layer buffer holding a
    // decoy expert (all 0xFF, which no `sel` entry names) and then the case's
    // experts in REVERSE order, so slot `i` must read expert `len - i`. Same
    // experts, same slot order: the output must match the blob pair bit for
    // bit.
    let stride = blobs[0].length() as usize;
    let mut layer = vec![0xFFu8; stride];
    for blob in blobs.iter().rev() {
        let bytes = unsafe { std::slice::from_raw_parts(blob.contents() as *const u8, stride) };
        layer.extend_from_slice(bytes);
    }
    let n_experts = (blobs.len() + 1) as u32;
    let sel: Vec<u32> = (0..c.top_k).map(|i| (blobs.len() - i) as u32).collect();
    let sel_bytes: Vec<u8> = sel.iter().flat_map(|v| v.to_le_bytes()).collect();
    let layer_buf = context.new_buffer_with_data(&layer);
    let sel_buf = context.new_buffer_with_data(&sel_bytes);
    let acts2 = context.new_buffer_with_data(&to_le(&vec![nan; MAX_STREAMED_EXPERTS * f_dim]));
    let y2 = context.new_output_buffer((d_dim * 2) as u64);
    let resident = Inkling2ResidentLayer {
        buffer: &layer_buf,
        stride: stride as u64,
        n_experts,
        sel: (&sel_buf, 0),
    };
    let pass = context.begin_pass();
    encode_inkling2_moe_phase1_indexed(
        &mut context,
        &pass,
        c.gate_up,
        &resident,
        &offsets,
        (&x_buf, 0),
        (&acts2, 0),
        d_dim as u32,
        f_dim as u32,
        c.top_k as u32,
        c.caps.0,
        c.caps.1,
    )
    .expect("indexed phase1");
    encode_inkling2_moe_phase2_indexed(
        &mut context,
        &pass,
        c.down,
        &resident,
        &offsets,
        (&acts2, 0),
        (&routing_buf, 0),
        (&y2, 0),
        d_dim as u32,
        f_dim as u32,
        c.top_k as u32,
    )
    .expect("indexed phase2");
    pass.commit_and_wait();
    let got2 = read_f16(&y2, d_dim);
    assert!(
        got.iter()
            .zip(&got2)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "{:?}/{:?} dims={:?} top_k={}: the indexed pair differs from the blob pair",
        c.gate_up,
        c.down,
        c.dims,
        c.top_k
    );
    let mut peak = 0f32;
    for d in 0..d_dim {
        let want = expected[d];
        peak = peak.max(want.abs());
        let tol = 5e-2_f32.max(want.abs() * 3e-2);
        assert!(
            (got[d] - want).abs() <= tol,
            "{:?}/{:?} dims={:?} top_k={} d={d}: got {} want {want}",
            c.gate_up,
            c.down,
            c.dims,
            c.top_k,
            got[d]
        );
    }
    // A fixture that decodes to ~0 everywhere would pass any kernel.
    assert!(
        peak > 0.5,
        "expected output too small to prove anything: {peak}"
    );
}

use Inkling2ExpertType::{Mxfp8, Nvfp4};

#[test]
fn nvfp4_pair_matches_the_hf_reference() {
    run_case(&Case {
        gate_up: Nvfp4,
        down: Nvfp4,
        dims: SMALL,
        top_k: 2,
        caps: MINI_CAPS,
    });
}

#[test]
fn mxfp8_pair_matches_the_hf_reference() {
    run_case(&Case {
        gate_up: Mxfp8,
        down: Mxfp8,
        dims: SMALL,
        top_k: 2,
        caps: MINI_CAPS,
    });
}

/// The Mini's real shape and top_k, for both layer kinds.
#[test]
fn all_eight_slots_at_mini_dims() {
    for kind in [Nvfp4, Mxfp8] {
        run_case(&Case {
            gate_up: kind,
            down: kind,
            dims: MINI,
            top_k: INKLING2_MAX_SLOTS,
            caps: MINI_CAPS,
        });
    }
}

/// Fewer slots than simdgroups: the NaN-filled unused slots must not reach y.
#[test]
fn unused_slots_contribute_nothing() {
    run_case(&Case {
        gate_up: Nvfp4,
        down: Mxfp8,
        dims: SMALL,
        top_k: 3,
        caps: MINI_CAPS,
    });
}

/// Tight caps make the softcaps bite, so a kernel that ignored the cap
/// arguments (or swapped them) would fail where the Mini's loose caps pass.
#[test]
fn the_cap_arguments_reach_the_kernel() {
    run_case(&Case {
        gate_up: Mxfp8,
        down: Nvfp4,
        dims: SMALL,
        top_k: 4,
        caps: (0.5, 0.25),
    });
}

#[test]
fn nine_slots_is_refused() {
    let mut context = MetalContext::new().expect("Metal device");
    let source = moe_inkling2_source();
    let routed =
        RoutedBlobsBuffer::new_for(&mut context, source, MOE_INKLING2_REFLECT_FUNCTION, false)
            .expect("arg buffer");
    let buf = context.new_buffer_with_data(&[0u8; 64]);
    let offsets = MoeExpertOffsets {
        gate_w: 0,
        gate_s: 0,
        gate_b: 0,
        up_w: 0,
        up_s: 0,
        up_b: 0,
        down_w: 0,
        down_s: 0,
        down_b: 0,
    };
    let pass = context.begin_pass();
    let p1 = encode_inkling2_moe_phase1(
        &mut context,
        &pass,
        Nvfp4,
        &routed,
        &offsets,
        (&buf, 0),
        (&buf, 0),
        64,
        64,
        9,
        4.0,
        25.0,
    );
    let p2 = encode_inkling2_moe_phase2(
        &mut context,
        &pass,
        Nvfp4,
        &routed,
        &offsets,
        (&buf, 0),
        (&buf, 0),
        (&buf, 0),
        64,
        64,
        9,
    );
    pass.commit_and_wait();
    assert!(p1.is_err() && p2.is_err());
}
