#![cfg(target_os = "macos")]
//! The matrix-unit expert-grouped pair (`encode_inkling2_moe_grouped_rows_mma`)
//! for Inkling 2 chunked prefill, on real Metal hardware.
//!
//! Exactness class "row-independent, KL-bounded": not the scalar grouped
//! pair's bits, but a token row's activations and output must depend only on
//! that row's input and routing, never on which other rows (and so which
//! other pairs) share the pass. So, at the Mini's dims with 16 experts and
//! heavily shared routing, both formats:
//! - accuracy against the scalar grouped pair (itself bit-identical to the
//!   decode pair);
//! - row independence: a row's bits alone equal its bits inside 9 and 32
//!   rows;
//! - determinism.

use half::f16;
use turbospark_compute::inkling2::{mxfp8_row_to_blocks, nvfp4_row_to_blocks};
use turbospark_gpu::{
    encode_inkling2_moe_grouped_rows, encode_inkling2_moe_grouped_rows_mma, Inkling2ExpertType,
    Inkling2GroupScratch, Inkling2ResidentLayer, MetalContext, MoeExpertOffsets,
    INKLING2_OUTER_SCALE_BYTES,
};

const D: usize = 2304;
const F: usize = 1152;
const TOP_K: usize = 8;
const EXPERTS: usize = 16;
const ROWS: usize = 32;

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

fn e4m3(s: &mut u32, e_lo: u32, e_hi: u32, signed: bool) -> u8 {
    let r = lcg(s) >> 8;
    let e = e_lo + r % (e_hi - e_lo + 1);
    let m = (r >> 4) & 7;
    let sign = if signed && (r >> 8) & 1 == 1 { 0x80 } else { 0 };
    (sign | (e << 3) | m) as u8
}

fn wrow(kind: Inkling2ExpertType, cols: usize, s: &mut u32) -> Vec<u8> {
    match kind {
        Inkling2ExpertType::Nvfp4 => {
            let packed: Vec<u8> = (0..cols / 2).map(|_| (lcg(s) >> 8) as u8).collect();
            let scales: Vec<u8> = (0..cols / 16).map(|_| e4m3(s, 1, 3, false)).collect();
            nvfp4_row_to_blocks(&packed, &scales).unwrap()
        }
        Inkling2ExpertType::Mxfp8 => {
            let bytes: Vec<u8> = (0..cols).map(|_| e4m3(s, 4, 8, true)).collect();
            let exps: Vec<u8> = (0..cols / 32)
                .map(|_| 121 + (lcg(s) >> 9) as u8 % 2)
                .collect();
            mxfp8_row_to_blocks(&bytes, &exps).unwrap()
        }
    }
}

fn halves(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| f16::from_f32(x).to_bits().to_le_bytes())
        .collect()
}

fn bits(buffer: &metal::Buffer, n: usize) -> Vec<u16> {
    unsafe { std::slice::from_raw_parts(buffer.contents() as *const u16, n) }.to_vec()
}

struct Fixture {
    kind: Inkling2ExpertType,
    layer: Vec<u8>,
    stride: usize,
    offsets: MoeExpertOffsets,
    sel: Vec<u32>,
    rw: Vec<f32>,
    x: Vec<f32>,
}

fn fixture(kind: Inkling2ExpertType, seed: u32) -> Fixture {
    let mut s = seed;
    let mut layer = Vec::new();
    let mut offsets = None;
    let mut stride = 0;
    for _ in 0..EXPERTS {
        let mut blob = Vec::new();
        for _ in 0..3 {
            let o = match kind {
                Inkling2ExpertType::Nvfp4 => {
                    0.7 + 0.6 * ((lcg(&mut s) >> 8) as f32 / (1u32 << 24) as f32)
                }
                Inkling2ExpertType::Mxfp8 => 1.0,
            };
            blob.extend_from_slice(&o.to_le_bytes());
        }
        blob.resize(INKLING2_OUTER_SCALE_BYTES, 0);
        let mut at = [0u32; 3];
        for (i, (rows, cols)) in [(F, D), (F, D), (D, F)].into_iter().enumerate() {
            at[i] = blob.len() as u32;
            for _ in 0..rows {
                blob.extend_from_slice(&wrow(kind, cols, &mut s));
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
        stride = blob.len();
        layer.extend_from_slice(&blob);
    }
    // Rows draw 8 distinct experts from a slowly sliding 12-expert window,
    // so experts are shared by many rows (the case grouping exists for).
    let mut sel = vec![0u32; ROWS * TOP_K];
    for r in 0..ROWS {
        let mut w: Vec<u32> = (0..12).map(|i| ((r / 3 + i) % EXPERTS) as u32).collect();
        let mut t = seed ^ (r as u32).wrapping_mul(2_654_435_761);
        for k in 0..TOP_K {
            let i = (lcg(&mut t) >> 8) as usize % w.len();
            sel[r * TOP_K + k] = w.remove(i);
        }
    }
    let rw = (0..ROWS * TOP_K)
        .map(|_| 0.1 + (lcg(&mut s) >> 8) as f32 / (1u32 << 24) as f32)
        .collect();
    let x = (0..ROWS * D)
        .map(|_| ((lcg(&mut s) >> 8) as f32 / (1u32 << 23) as f32) - 1.0)
        .collect();
    Fixture {
        kind,
        layer,
        stride,
        offsets: offsets.unwrap(),
        sel,
        rw,
        x,
    }
}

/// Runs one implementation over the fixture's rows `rows` (gathered into a
/// fresh pass in that order); returns per row (acts bits, y bits).
fn apply(f: &Fixture, rows: &[usize], mma: bool) -> Vec<(Vec<u16>, Vec<u16>)> {
    let mut context = MetalContext::new().expect("Metal device");
    let n = rows.len();
    let sel: Vec<u8> = rows
        .iter()
        .flat_map(|&r| f.sel[r * TOP_K..(r + 1) * TOP_K].iter())
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let rw: Vec<f32> = rows
        .iter()
        .flat_map(|&r| f.rw[r * TOP_K..(r + 1) * TOP_K].iter().copied())
        .collect();
    let x: Vec<f32> = rows
        .iter()
        .flat_map(|&r| f.x[r * D..(r + 1) * D].iter().copied())
        .collect();
    let layer_buf = context.new_buffer_with_data(&f.layer);
    let sel_buf = context.new_buffer_with_data(&sel);
    let rw_buf = context.new_buffer_with_data(&halves(&rw));
    let x_buf = context.new_buffer_with_data(&halves(&x));
    let acts = context.new_output_buffer((n * TOP_K * F * 2) as u64);
    let y = context.new_output_buffer((n * D * 2) as u64);
    let offs = context.new_output_buffer(((EXPERTS + 1) * 4) as u64);
    let pairs = context.new_output_buffer((n * TOP_K * 4) as u64);
    let terms = context.new_output_buffer((n * TOP_K * D * 4) as u64);
    let layer = Inkling2ResidentLayer {
        buffer: &layer_buf,
        stride: f.stride as u64,
        n_experts: EXPERTS as u32,
        sel: (&sel_buf, 0),
    };
    let scratch = Inkling2GroupScratch {
        offs: (&offs, 0),
        pairs: (&pairs, 0),
        terms: (&terms, 0),
    };
    let pass = context.begin_pass();
    let enc = if mma {
        encode_inkling2_moe_grouped_rows_mma
    } else {
        encode_inkling2_moe_grouped_rows
    };
    enc(
        &mut context,
        &pass,
        f.kind,
        &layer,
        &f.offsets,
        (&x_buf, 0),
        D as u32,
        (&acts, 0),
        (&rw_buf, 0),
        (&y, 0),
        &scratch,
        (D as u32, F as u32),
        TOP_K as u32,
        (4.0, 25.0),
        n as u32,
    )
    .unwrap();
    pass.commit_and_wait();
    let (a, yb) = (bits(&acts, n * TOP_K * F), bits(&y, n * D));
    (0..n)
        .map(|i| {
            (
                a[i * TOP_K * F..(i + 1) * TOP_K * F].to_vec(),
                yb[i * D..(i + 1) * D].to_vec(),
            )
        })
        .collect()
}

fn to_f(v: &[u16]) -> Vec<f64> {
    v.iter().map(|&b| f16::from_bits(b).to_f64()).collect()
}

fn check(kind: Inkling2ExpertType, seed: u32) {
    let f = fixture(kind, seed);
    let all: Vec<usize> = (0..ROWS).collect();
    let mma = apply(&f, &all, true);
    let scalar = apply(&f, &all, false);
    // Accuracy: relative to each row's output RMS (FP16 rounding of a sum of
    // eight expert terms dominates either way).
    let mut worst = 0f64;
    for (m, s) in mma.iter().zip(&scalar) {
        let (ym, ys) = (to_f(&m.1), to_f(&s.1));
        let rms = (ys.iter().map(|v| v * v).sum::<f64>() / ys.len() as f64).sqrt();
        assert!(
            rms > 0.1,
            "fixture output too small to prove anything: {rms}"
        );
        for (a, b) in ym.iter().zip(&ys) {
            worst = worst.max((a - b).abs() / rms);
        }
    }
    eprintln!("{kind:?}: mma vs scalar grouped pair, worst |diff| / row rms {worst:.2e}");
    assert!(
        worst < 5e-3,
        "{kind:?}: mma grouped pair off the scalar pair: {worst}"
    );

    // Row independence, bit for bit.
    let probe = 11usize;
    let alone = apply(&f, &[probe], true);
    let nine = [0usize, 3, 7, probe, 12, 13, 20, 30, 31];
    let in9 = apply(&f, &nine, true);
    assert_eq!(
        alone[0], mma[probe],
        "{kind:?}: a row alone differs from it in 32"
    );
    assert_eq!(
        in9[3], mma[probe],
        "{kind:?}: a row in 9 differs from it in 32"
    );
    // Determinism.
    assert_eq!(apply(&f, &all, true), mma, "{kind:?}: not deterministic");
}

#[test]
fn grouped_mma_nvfp4_is_accurate_and_row_independent() {
    for seed in [3u32, 101] {
        check(Inkling2ExpertType::Nvfp4, seed);
    }
}

#[test]
fn grouped_mma_mxfp8_is_accurate_and_row_independent() {
    check(Inkling2ExpertType::Mxfp8, 7);
}
