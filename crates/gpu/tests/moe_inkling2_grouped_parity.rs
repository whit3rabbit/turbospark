#![cfg(target_os = "macos")]
//! Bit parity for the expert-grouped multi-row pair
//! (`encode_inkling2_moe_grouped_rows`) against the `_idx_rows` pair it
//! replaces in chunked prefill, on real Metal hardware.
//!
//! A resident layer of 16 experts at the Mini's dims, rows routed so experts
//! are shared heavily (the case grouping exists for) and some rows share all
//! eight. Both formats. The activations and the summed output must be the
//! same bits: the grouped pair decodes each expert once for all its pairs but
//! must not change any pair's arithmetic or the slot reduction order.

use half::f16;
use turbospark_compute::inkling2::{mxfp8_row_to_blocks, nvfp4_row_to_blocks};
use turbospark_gpu::{
    encode_inkling2_moe_grouped_rows, encode_inkling2_moe_phase1_indexed_rows,
    encode_inkling2_moe_phase2_indexed_rows, Inkling2ExpertType, Inkling2GroupScratch,
    Inkling2ResidentLayer, MetalContext, MoeExpertOffsets, INKLING2_OUTER_SCALE_BYTES,
};

const D: usize = 2304;
const F: usize = 1152;
const TOP_K: usize = 8;
const EXPERTS: usize = 16;

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

fn row(kind: Inkling2ExpertType, cols: usize, s: &mut u32) -> Vec<u8> {
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

fn halves(v: impl Iterator<Item = f32>) -> Vec<u8> {
    v.flat_map(|x| f16::from_f32(x).to_bits().to_le_bytes())
        .collect()
}

fn bits(buffer: &metal::Buffer, n: usize) -> Vec<u16> {
    let ptr = buffer.contents() as *const u16;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

fn run(kind: Inkling2ExpertType, n: usize, seed: u32) {
    let mut s = seed;
    let mut context = MetalContext::new().expect("Metal device");
    let outer = |s: &mut u32| match kind {
        Inkling2ExpertType::Nvfp4 => 0.7 + 0.6 * ((lcg(s) >> 8) as f32 / (1u32 << 24) as f32),
        Inkling2ExpertType::Mxfp8 => 1.0,
    };
    // One layer buffer, expert e at e * stride.
    let mut layer = Vec::new();
    let mut offsets = None;
    let mut stride = 0usize;
    for _ in 0..EXPERTS {
        let mut blob = Vec::new();
        for _ in 0..3 {
            blob.extend_from_slice(&outer(&mut s).to_le_bytes());
        }
        blob.resize(INKLING2_OUTER_SCALE_BYTES, 0);
        let mut at = [0u32; 3];
        for (i, (rows, cols)) in [(F, D), (F, D), (D, F)].into_iter().enumerate() {
            at[i] = blob.len() as u32;
            for _ in 0..rows {
                blob.extend_from_slice(&row(kind, cols, &mut s));
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
    let offsets = offsets.unwrap();
    // Selection: row r draws 8 distinct experts from a window that slides
    // slowly, so neighbouring rows share most experts; rows 0 and 1 share all.
    let mut sel = vec![0u32; n * TOP_K];
    for r in 0..n {
        let base = if r == 1 { 0 } else { r / 3 };
        let mut pick: Vec<u32> = (0..EXPERTS as u32).collect();
        // A seeded shuffle of the 12-expert window starting at `base`.
        let window: Vec<u32> = (0..12).map(|i| ((base + i) % EXPERTS) as u32).collect();
        pick.clear();
        let mut w = window.clone();
        let mut t = seed ^ (if r == 1 { 0 } else { r as u32 });
        while pick.len() < TOP_K {
            let i = (lcg(&mut t) >> 8) as usize % w.len();
            pick.push(w.remove(i));
        }
        sel[r * TOP_K..(r + 1) * TOP_K].copy_from_slice(&pick);
    }
    let rw: Vec<f32> = (0..n * TOP_K)
        .map(|_| 0.1 + (lcg(&mut s) >> 8) as f32 / (1u32 << 24) as f32)
        .collect();
    let x: Vec<f32> = (0..n * D)
        .map(|_| ((lcg(&mut s) >> 8) as f32 / (1u32 << 23) as f32) - 1.0)
        .collect();

    let layer_buf = context.new_buffer_with_data(&layer);
    let sel_buf =
        context.new_buffer_with_data(&sel.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
    let rw_buf = context.new_buffer_with_data(&halves(rw.iter().copied()));
    let x_buf = context.new_buffer_with_data(&halves(x.iter().copied()));
    let acts_a = context.new_output_buffer((n * TOP_K * F * 2) as u64);
    let acts_b = context.new_output_buffer((n * TOP_K * F * 2) as u64);
    let y_a = context.new_output_buffer((n * D * 2) as u64);
    let y_b = context.new_output_buffer((n * D * 2) as u64);
    let offs = context.new_output_buffer(((EXPERTS + 1) * 4) as u64);
    let pairs = context.new_output_buffer((n * TOP_K * 4) as u64);
    let terms = context.new_output_buffer((n * TOP_K * D * 4) as u64);
    let resident = Inkling2ResidentLayer {
        buffer: &layer_buf,
        stride: stride as u64,
        n_experts: EXPERTS as u32,
        sel: (&sel_buf, 0),
    };
    let caps = (4.0f32, 25.0f32);
    let pass = context.begin_pass();
    encode_inkling2_moe_phase1_indexed_rows(
        &mut context,
        &pass,
        kind,
        &resident,
        &offsets,
        (&x_buf, 0),
        D as u32,
        (&acts_a, 0),
        D as u32,
        F as u32,
        TOP_K as u32,
        caps.0,
        caps.1,
        n as u32,
    )
    .unwrap();
    encode_inkling2_moe_phase2_indexed_rows(
        &mut context,
        &pass,
        kind,
        &resident,
        &offsets,
        (&acts_a, 0),
        (&rw_buf, 0),
        (&y_a, 0),
        D as u32,
        F as u32,
        TOP_K as u32,
        n as u32,
    )
    .unwrap();
    encode_inkling2_moe_grouped_rows(
        &mut context,
        &pass,
        kind,
        &resident,
        &offsets,
        (&x_buf, 0),
        D as u32,
        (&acts_b, 0),
        (&rw_buf, 0),
        (&y_b, 0),
        &Inkling2GroupScratch {
            offs: (&offs, 0),
            pairs: (&pairs, 0),
            terms: (&terms, 0),
        },
        (D as u32, F as u32),
        TOP_K as u32,
        caps,
        n as u32,
    )
    .unwrap();
    pass.commit_and_wait();

    let (a, b) = (bits(&acts_a, n * TOP_K * F), bits(&acts_b, n * TOP_K * F));
    let diff = a.iter().zip(&b).filter(|(p, q)| p != q).count();
    assert_eq!(diff, 0, "{kind:?} n={n}: {diff} of {} acts differ", a.len());
    let (a, b) = (bits(&y_a, n * D), bits(&y_b, n * D));
    let diff = a.iter().zip(&b).filter(|(p, q)| p != q).count();
    assert_eq!(
        diff,
        0,
        "{kind:?} n={n}: {diff} of {} outputs differ",
        a.len()
    );
    // Not a vacuous match: the output is not all zero.
    assert!(a.iter().filter(|&&v| v & 0x7fff != 0).count() > n * D / 2);
}

#[test]
fn grouped_nvfp4_matches_the_indexed_rows_pair() {
    // Many seeds: the pair must hold over whatever values reach a rounding
    // boundary, not on one fixture.
    for n in [1usize, 2, 8, 23, 32] {
        for seed in 1..=6u32 {
            run(Inkling2ExpertType::Nvfp4, n, seed * 7919 + n as u32);
        }
    }
}

#[test]
fn grouped_mxfp8_matches_the_indexed_rows_pair() {
    for n in [1usize, 9, 32] {
        for seed in 1..=6u32 {
            run(Inkling2ExpertType::Mxfp8, n, seed * 104_729 + n as u32);
        }
    }
}
