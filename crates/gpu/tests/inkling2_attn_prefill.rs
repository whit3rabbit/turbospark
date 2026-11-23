#![cfg(target_os = "macos")]
//! The tiled matrix-unit attention for Inkling 2 prefill panels
//! (`encode_inkling2_attention_prefill`) against the rows kernel it replaces
//! there (`encode_inkling2_attention_rows`, itself bit-identical to decode),
//! on real Metal hardware.
//!
//! Exactness class "row-independent, KL-bounded": close to the rows kernel,
//! and a row's bits depend only on its own query, keys, bias row and sink,
//! never on which rows share its 32-row block or where the block starts. A
//! sliding-window slot past two ring wraps and a full-attention slot, 45 rows
//! (one full block and a partial one), bias extent shorter than the window.

use half::f16;
use turbospark_gpu::{
    encode_inkling2_attention_prefill, encode_inkling2_attention_rows, Inkling2AttnShape,
    MetalContext,
};

const HEADS: usize = 24;
const KVH: usize = 8;
const HD: usize = 128;
const EXTENT: usize = 64;
const N: usize = 45;

fn lcg(s: &mut u32) -> u32 {
    *s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *s
}

fn unit(s: &mut u32) -> f32 {
    ((lcg(s) >> 8) as f32 / (1u32 << 23) as f32) - 1.0
}

fn h16(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| f16::from_f32(x).to_bits().to_le_bytes())
        .collect()
}

fn f32le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bits(b: &metal::Buffer, n: usize) -> Vec<u16> {
    unsafe { std::slice::from_raw_parts(b.contents() as *const u16, n) }.to_vec()
}

struct Case {
    window: usize,
    ring: usize,
    pos0: usize,
    k: Vec<f32>,
    v: Vec<f32>,
    q: Vec<f32>,
    table: Vec<f32>,
    sinks: Vec<f32>,
}

fn case(window: usize, ring: usize, pos0: usize, seed: u32) -> Case {
    let mut s = seed;
    let kvd = KVH * HD;
    // Queries scaled so scores spread over several units (a peaked softmax).
    Case {
        window,
        ring,
        pos0,
        k: (0..ring * kvd).map(|_| unit(&mut s)).collect(),
        v: (0..ring * kvd).map(|_| unit(&mut s)).collect(),
        q: (0..N * HEADS * HD).map(|_| unit(&mut s) * 2.0).collect(),
        table: (0..N * HEADS * EXTENT)
            .map(|_| unit(&mut s) * 1.5)
            .collect(),
        sinks: (0..N * HEADS).map(|_| unit(&mut s) * 2.0).collect(),
    }
}

/// Runs a kernel over rows `rows` of the case (each row keeps its own
/// position `pos0 + row`; the rows must be consecutive); returns per row the
/// output bits.
fn run(c: &Case, rows: std::ops::Range<usize>, tiled: bool) -> Vec<Vec<u16>> {
    let mut ctx = MetalContext::new().expect("Metal device");
    let n = rows.len();
    let qs = HEADS * HD;
    let k = ctx.new_buffer_with_data(&h16(&c.k));
    let v = ctx.new_buffer_with_data(&h16(&c.v));
    let q = ctx.new_buffer_with_data(&h16(&c.q[rows.start * qs..rows.end * qs]));
    let table = ctx.new_buffer_with_data(&f32le(
        &c.table[rows.start * HEADS * EXTENT..rows.end * HEADS * EXTENT],
    ));
    let sinks = ctx.new_buffer_with_data(&h16(&c.sinks[rows.start * HEADS..rows.end * HEADS]));
    let out = ctx.new_output_buffer((n * qs * 2) as u64);
    let shape = Inkling2AttnShape {
        heads: HEADS as u32,
        kv_heads: KVH as u32,
        head_dim: HD as u32,
        extent: EXTENT as u32,
        ring: c.ring as u32,
    };
    let pass = ctx.begin_pass();
    let enc = if tiled {
        encode_inkling2_attention_prefill
    } else {
        encode_inkling2_attention_rows
    };
    enc(
        &mut ctx,
        &pass,
        (&q, 0),
        (&k, 0),
        (&v, 0),
        (&table, 0),
        (&sinks, 0),
        (&out, 0),
        c.window as u32,
        (c.pos0 + rows.start) as u32,
        shape,
        n as u32,
    )
    .unwrap();
    pass.commit_and_wait();
    let b = bits(&out, n * qs);
    (0..n).map(|i| b[i * qs..(i + 1) * qs].to_vec()).collect()
}

fn check(c: &Case, label: &str) {
    let tiled = run(c, 0..N, true);
    let rows = run(c, 0..N, false);
    let mut worst = 0f32;
    for (a, b) in tiled.iter().zip(&rows) {
        for (x, y) in a.iter().zip(b) {
            let d = (f16::from_bits(*x).to_f32() - f16::from_bits(*y).to_f32()).abs();
            worst = worst.max(d);
        }
    }
    eprintln!("{label}: tiled vs rows kernel, max |diff| {worst:.2e}");
    // Outputs are softmax averages of values in [-1, 1]; FP16 rounding of
    // those is ~5e-4, and the two kernels sum in different orders.
    assert!(
        worst < 4e-3,
        "{label}: tiled attention off the rows kernel: {worst}"
    );
    for i in [0usize, 13, 31, 32, 44] {
        let alone = run(c, i..i + 1, true);
        assert_eq!(
            alone[0], tiled[i],
            "{label}: row {i} alone differs from it in the panel"
        );
    }
    // A block that starts elsewhere: rows 5..45.
    let shifted = run(c, 5..N, true);
    for (j, i) in (5..N).enumerate() {
        assert_eq!(
            shifted[j], tiled[i],
            "{label}: row {i} differs in a block starting at 5"
        );
    }
    assert_eq!(run(c, 0..N, true), tiled, "{label}: not deterministic");
}

#[test]
fn tiled_attention_sliding_window_past_two_wraps() {
    // Window 100, ring window + 127, positions 400.. (several wraps).
    check(&case(100, 227, 400, 3), "swa");
}

#[test]
fn tiled_attention_full_layer() {
    check(&case(0, 600, 300, 7), "full");
}

#[test]
fn tiled_attention_first_rows_of_a_sequence() {
    // Positions 0..45: windows that start at 0, bias covering every key.
    check(&case(100, 227, 0, 11), "start");
}
