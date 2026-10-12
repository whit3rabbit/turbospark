#![cfg(target_os = "macos")]
//! Parity for the vendored Steel INT4 GEMM (`dequant_int4_steel_*`, through
//! `encode_dequant_int4_gemm_steel`) on real Metal hardware.
//!
//! It is not bit-exact against the scalar kernels (each weight is rounded to
//! an FP16 operand and the matrix units reduce in their own order), so the
//! bar is an FP64 CPU reference within a row-relative bound, the same
//! `0.005 x max |row|` the isolated probe used (`docs/MLX_KERNELS.md`). The
//! reference's own layout assumptions (low nibble first, BF16 scale and bias
//! per 64-column group) are pinned first by checking it against
//! `dequant_int4_gemv_simd`, the kernel every install already runs.
//!
//! The cases cover both instantiated tiles, token counts that do not divide
//! the tile (the masked `load_safe` path), an input as wide as Qwen3.8's
//! hidden size, and weights at a 2-byte-aligned offset, which is where real
//! resident tensors sit.

use half::{bf16, f16};
use turbospark_gpu::{
    autorelease_pool, encode_dequant_int4_gemm_steel, encode_dequant_int4_gemv_resident,
    read_half_buffer, steel_supports, Int4ResidentMatrix, MetalContext, SteelTile,
};

fn lcg(seed: u64) -> impl FnMut() -> u64 {
    let mut state = seed | 1;
    move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state >> 33
    }
}

struct Fixture {
    blob: Vec<u8>,
    weights_offset: usize,
    scales_offset: usize,
    biases_offset: usize,
    rows: usize,
    cols: usize,
}

/// `pad` leading bytes put the weights at that offset; 2 reproduces the
/// resident layout's 2-byte (not 4-byte) alignment.
fn fixture(rows: usize, cols: usize, pad: usize) -> Fixture {
    let mut next = lcg(0x5EE1 ^ (rows * 31 + cols) as u64);
    let groups = rows * cols / 64;
    let mut blob = vec![0u8; pad];
    let weights_offset = blob.len();
    blob.extend((0..rows * cols / 2).map(|_| next() as u8));
    let scales_offset = blob.len();
    for _ in 0..groups {
        // 0.004 .. 0.03: dequantized weights stay well inside FP16.
        let s = 0.004 + (next() % 1000) as f32 * 2.6e-5;
        blob.extend_from_slice(&bf16::from_f32(s).to_bits().to_le_bytes());
    }
    let biases_offset = blob.len();
    for _ in 0..groups {
        let b = -0.12 + (next() % 1000) as f32 * 2.4e-4;
        blob.extend_from_slice(&bf16::from_f32(b).to_bits().to_le_bytes());
    }
    Fixture {
        blob,
        weights_offset,
        scales_offset,
        biases_offset,
        rows,
        cols,
    }
}

fn activations(batch: usize, cols: usize) -> Vec<f16> {
    let mut next = lcg(0xAC7 ^ (batch * 7 + cols) as u64);
    (0..batch * cols)
        .map(|_| f16::from_f32(((next() % 2001) as f32 - 1000.0) / 2000.0))
        .collect()
}

fn reference(fx: &Fixture, x: &[f16], batch: usize) -> Vec<f64> {
    let bf = |offset: usize| {
        bf16::from_bits(u16::from_le_bytes([fx.blob[offset], fx.blob[offset + 1]])).to_f64()
    };
    let mut y = vec![0.0f64; batch * fx.rows];
    for r in 0..fx.rows {
        for b in 0..batch {
            let mut acc = 0.0f64;
            for c in 0..fx.cols {
                let byte = fx.blob[fx.weights_offset + (r * fx.cols + c) / 2];
                let q = if c % 2 == 0 { byte & 15 } else { byte >> 4 } as f64;
                let g = (r * fx.cols + c) / 64;
                let w = q * bf(fx.scales_offset + 2 * g) + bf(fx.biases_offset + 2 * g);
                acc += w * x[b * fx.cols + c].to_f64();
            }
            y[b * fx.rows + r] = acc;
        }
    }
    y
}

fn bytes(v: &[f16]) -> Vec<u8> {
    v.iter().flat_map(|h| h.to_bits().to_le_bytes()).collect()
}

/// Worst `|got - want| / max |want row|` over every output row.
fn worst_row_error(got: &[f16], want: &[f64], rows: usize) -> f64 {
    want.chunks(rows)
        .zip(got.chunks(rows))
        .map(|(w, g)| {
            let scale = w.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-6);
            w.iter()
                .zip(g)
                .map(|(a, b)| (b.to_f64() - a).abs() / scale)
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max)
}

const BOUND: f64 = 0.005;

fn check(rows: usize, cols: usize, pad: usize, batches: &[usize], tiles: &[SteelTile]) {
    let mut context = MetalContext::new().expect("Metal device");
    let fx = fixture(rows, cols, pad);
    let buffer = context.new_buffer_with_data(&fx.blob);
    let matrix = Int4ResidentMatrix {
        buffer: &buffer,
        weights_offset: fx.weights_offset as u64,
        scales_offset: fx.scales_offset as u64,
        biases_offset: fx.biases_offset as u64,
        rows,
        cols,
    };
    for &batch in batches {
        let x = activations(batch, cols);
        let want = reference(&fx, &x, batch);
        let x_buf = context.new_buffer_with_data(&bytes(&x));

        // Pin the reference's layout assumptions to the kernel installs run.
        let gemv_out = context.new_output_buffer((rows * 2) as u64);
        autorelease_pool(|| {
            let pass = context.begin_pass();
            encode_dequant_int4_gemv_resident(
                &mut context,
                &pass,
                &matrix,
                (&x_buf, 0),
                (&gemv_out, 0),
            )
            .expect("gemv encodes");
            pass.commit_and_wait();
        });
        let gemv = read_half_buffer(&gemv_out, rows);
        let gemv_err = worst_row_error(&gemv, &want[..rows], rows);
        assert!(
            gemv_err < BOUND,
            "{rows}x{cols}: the CPU reference disagrees with the GEMV ({gemv_err})"
        );

        for &tile in tiles {
            assert!(
                steel_supports(&matrix, tile),
                "{rows}x{cols} must fit {tile:?}"
            );
            let y = context.new_output_buffer((batch * rows * 2) as u64);
            autorelease_pool(|| {
                let pass = context.begin_pass();
                encode_dequant_int4_gemm_steel(
                    &mut context,
                    &pass,
                    &matrix,
                    (&x_buf, 0),
                    (&y, 0),
                    batch,
                    tile,
                )
                .expect("steel encodes");
                pass.commit_and_wait();
            });
            let got = read_half_buffer(&y, batch * rows);
            assert!(
                got.iter().all(|v| v.to_f32().is_finite()),
                "{tile:?} batch {batch}: non-finite output"
            );
            let err = worst_row_error(&got, &want, rows);
            assert!(
                err < BOUND,
                "{rows}x{cols} pad {pad} batch {batch} {tile:?}: worst row error {err}"
            );
        }
    }
}

const BOTH: &[SteelTile] = &[SteelTile::M32K32N32, SteelTile::M64K32N64];

#[test]
fn small_shapes_and_ragged_batches_match_the_reference() {
    check(64, 128, 0, &[1, 5, 32, 33], BOTH);
}

#[test]
fn wider_batches_match_the_reference() {
    check(256, 512, 0, &[16, 64, 70], BOTH);
}

#[test]
fn a_qwen38_width_input_matches_the_reference() {
    check(128, 5120, 0, &[3, 64], BOTH);
}

#[test]
fn a_two_byte_aligned_weight_offset_matches_the_reference() {
    check(128, 1024, 2, &[16, 40], BOTH);
}

#[test]
fn shapes_the_tile_cannot_serve_are_reported() {
    let context = MetalContext::new().expect("Metal device");
    let buffer = context.new_buffer_with_data(&[0u8; 64]);
    let matrix = |rows, cols| Int4ResidentMatrix {
        buffer: &buffer,
        weights_offset: 0,
        scales_offset: 0,
        biases_offset: 0,
        rows,
        cols,
    };
    // Qwen3.8's 48-row GDN a/b projections: not a whole tile of outputs.
    assert!(!steel_supports(&matrix(48, 5120), SteelTile::M32K32N32));
    assert!(!steel_supports(&matrix(96, 5120), SteelTile::M64K32N64));
    assert!(steel_supports(&matrix(96, 5120), SteelTile::M32K32N32));
    // Not a whole quantization group of inputs.
    assert!(!steel_supports(&matrix(64, 96), SteelTile::M32K32N32));
}
