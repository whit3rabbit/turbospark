#![cfg(target_os = "macos")]
//! Resident GGUF experts for Inkling 2's multi-row forward, on real Metal
//! hardware. `encode_inkling2_gguf_bind_rows` writes one blob table per row
//! from that row's expert ids, and the streamed decode pair runs on it. Two
//! claims, both bit for bit:
//! - each GPU-written table equals the bytes `RoutedBlobsBuffer::bind_for`
//!   encodes for the same blobs (the layout the pair reflects);
//! - phase 1 and phase 2 over the tables equal the pair over a host-bound
//!   `RoutedBlobsBuffer`, row by row (Q8_0 experts at the Mini's top 8).

use half::f16;
use turbospark_gpu::{
    encode_inkling2_gguf_bind_rows, encode_inkling2_gguf_phase1, encode_inkling2_gguf_phase1_table,
    encode_inkling2_gguf_phase2, encode_inkling2_gguf_phase2_table, moe_gguf_source,
    Inkling2GgufExpertType, MetalContext, MoeExpertOffsets, RoutedBlobsBuffer,
    INKLING2_GGUF_TABLE_BYTES, INKLING2_MAX_SLOTS, INKLING2_OUTER_SCALE_BYTES,
};

const D: usize = 512;
const F: usize = 256;
const TOP_K: usize = 8;
const EXPERTS: usize = 12;
const ROWS: usize = 5;
const PHASE1: &str = "moe_phase1_gate_up_act_q8_0";

fn lcg(s: &mut u32) -> u32 {
    *s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *s
}

/// One Q8_0 row of `cols`: per 32 elements an FP16 scale then 32 int8.
fn q8_row(cols: usize, s: &mut u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(cols / 32 * 34);
    for _ in 0..cols / 32 {
        let d = 0.002 + 0.008 * ((lcg(s) >> 8) as f32 / (1u32 << 24) as f32);
        v.extend_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        for _ in 0..32 {
            v.push((lcg(s) >> 13) as u8);
        }
    }
    v
}

struct Fixture {
    layer: Vec<u8>,
    stride: usize,
    offsets: MoeExpertOffsets,
    sel: Vec<u32>,
    rw: Vec<u8>,
    x: Vec<u8>,
}

fn fixture(seed: u32) -> Fixture {
    let mut s = seed;
    let mut layer = Vec::new();
    let mut stride = 0;
    let mut at = [0u32; 3];
    for _ in 0..EXPERTS {
        let mut blob = Vec::new();
        for _ in 0..3 {
            let o = 0.7 + 0.6 * ((lcg(&mut s) >> 8) as f32 / (1u32 << 24) as f32);
            blob.extend_from_slice(&o.to_le_bytes());
        }
        blob.resize(INKLING2_OUTER_SCALE_BYTES, 0);
        for (i, (rows, cols)) in [(F, D), (F, D), (D, F)].into_iter().enumerate() {
            at[i] = blob.len() as u32;
            for _ in 0..rows {
                blob.extend_from_slice(&q8_row(cols, &mut s));
            }
        }
        stride = blob.len();
        layer.extend_from_slice(&blob);
    }
    let offsets = MoeExpertOffsets {
        gate_w: at[0],
        gate_s: 0,
        gate_b: 0,
        up_w: at[1],
        up_s: 4,
        up_b: 0,
        down_w: at[2],
        down_s: 8,
        down_b: 0,
    };
    // Rows of INKLING2_MAX_SLOTS ids, each a distinct top 8 of 12.
    let mut sel = vec![0u32; ROWS * INKLING2_MAX_SLOTS];
    for r in 0..ROWS {
        let mut pool: Vec<u32> = (0..EXPERTS as u32).collect();
        for k in 0..TOP_K {
            let i = (lcg(&mut s) >> 8) as usize % pool.len();
            sel[r * INKLING2_MAX_SLOTS + k] = pool.remove(i);
        }
    }
    let half = |n: usize, s: &mut u32, lo: f32, span: f32| -> Vec<u8> {
        (0..n)
            .flat_map(|_| {
                let v = lo + span * ((lcg(s) >> 8) as f32 / (1u32 << 24) as f32);
                f16::from_f32(v).to_bits().to_le_bytes()
            })
            .collect()
    };
    let rw = half(ROWS * INKLING2_MAX_SLOTS, &mut s, 0.05, 0.2);
    let x = half(ROWS * D, &mut s, -1.0, 2.0);
    Fixture {
        layer,
        stride,
        offsets,
        sel,
        rw,
        x,
    }
}

fn read(b: &metal::Buffer, off: usize, n: usize) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts((b.contents() as *const u8).add(off), n) }.to_vec()
}

#[test]
fn gpu_tables_match_bind_for_and_run_the_pair_bit_for_bit() {
    let kind = Inkling2GgufExpertType::Q8_0;
    let f = fixture(17);
    let mut context = MetalContext::new().expect("Metal device");
    let layer = context.new_buffer_with_data(&f.layer);
    let sel_bytes: Vec<u8> = f.sel.iter().flat_map(|v| v.to_le_bytes()).collect();
    let sel = context.new_buffer_with_data(&sel_bytes);
    let rw = context.new_buffer_with_data(&f.rw);
    let x = context.new_buffer_with_data(&f.x);
    let tb = INKLING2_GGUF_TABLE_BYTES as usize;
    let tables = context.new_output_buffer((ROWS * tb) as u64);
    let acts = context.new_output_buffer((ROWS * TOP_K * F * 2) as u64);
    let y = context.new_output_buffer((ROWS * D * 2) as u64);

    // GPU tables, then the pair per row on them, in one pass.
    let pass = context.begin_pass();
    pass.use_read_buffer(&layer);
    encode_inkling2_gguf_bind_rows(
        &mut context,
        &pass,
        &layer,
        f.stride as u64,
        (&sel, 0),
        INKLING2_MAX_SLOTS as u32,
        (&tables, 0),
        TOP_K as u32,
        ROWS as u32,
    )
    .unwrap();
    for r in 0..ROWS {
        let t = (&tables, (r * tb) as u64);
        let a = (&acts, (r * TOP_K * F * 2) as u64);
        encode_inkling2_gguf_phase1_table(
            &mut context,
            &pass,
            kind,
            t,
            &f.offsets,
            (&x, (r * D * 2) as u64),
            a,
            D as u32,
            F as u32,
            TOP_K as u32,
            4.0,
            25.0,
        )
        .unwrap();
        encode_inkling2_gguf_phase2_table(
            &mut context,
            &pass,
            kind,
            t,
            &f.offsets,
            a,
            (&rw, (r * INKLING2_MAX_SLOTS * 2) as u64),
            (&y, (r * D * 2) as u64),
            D as u32,
            F as u32,
            TOP_K as u32,
        )
        .unwrap();
    }
    pass.commit_and_wait();

    // The streamed reference: a host-bound buffer per row, one pass each.
    let routed =
        RoutedBlobsBuffer::new_for(&mut context, moe_gguf_source(), PHASE1, false).unwrap();
    let acts_ref = context.new_output_buffer((TOP_K * F * 2) as u64);
    let y_ref = context.new_output_buffer((D * 2) as u64);
    for r in 0..ROWS {
        let blobs: Vec<(&metal::Buffer, u64)> = (0..TOP_K)
            .map(|k| {
                (
                    &layer,
                    u64::from(f.sel[r * INKLING2_MAX_SLOTS + k]) * f.stride as u64,
                )
            })
            .collect();
        routed
            .bind_for(&mut context, moe_gguf_source(), PHASE1, false, &blobs)
            .unwrap();
        let enc = routed.buffer().length() as usize;
        assert_eq!(enc, tb, "argument-encoded table size");
        assert_eq!(
            read(&tables, r * tb, tb),
            read(routed.buffer(), 0, tb),
            "row {r}: GPU table differs from bind_for's"
        );
        let pass = context.begin_pass();
        pass.use_read_buffer(&layer);
        encode_inkling2_gguf_phase1(
            &mut context,
            &pass,
            kind,
            &routed,
            &f.offsets,
            (&x, (r * D * 2) as u64),
            (&acts_ref, 0),
            D as u32,
            F as u32,
            TOP_K as u32,
            4.0,
            25.0,
        )
        .unwrap();
        encode_inkling2_gguf_phase2(
            &mut context,
            &pass,
            kind,
            &routed,
            &f.offsets,
            (&acts_ref, 0),
            (&rw, (r * INKLING2_MAX_SLOTS * 2) as u64),
            (&y_ref, 0),
            D as u32,
            F as u32,
            TOP_K as u32,
        )
        .unwrap();
        pass.commit_and_wait();
        let ya = read(&y, r * D * 2, D * 2);
        assert_eq!(
            read(&acts, r * TOP_K * F * 2, TOP_K * F * 2),
            read(&acts_ref, 0, TOP_K * F * 2),
            "row {r}: acts differ"
        );
        assert_eq!(ya, read(&y_ref, 0, D * 2), "row {r}: y differs");
        // Not a vacuous pass: the output carries signal.
        let rms = (ya
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f64().powi(2))
            .sum::<f64>()
            / D as f64)
            .sqrt();
        assert!(rms.is_finite() && rms > 1e-3, "row {r}: output rms {rms}");
    }
}
