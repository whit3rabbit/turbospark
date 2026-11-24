#![cfg(target_os = "macos")]
//! The matrix-unit BF16 GEMM for Inkling 2 prefill panels
//! (`encode_inkling2_bf16_gemm_rows_mma`), on real Metal hardware.
//!
//! Its exactness class is "row-independent, KL-bounded": not the GEMV's bits,
//! but a token row's output must depend only on that row's input. So:
//! - accuracy against an f64 reference, and the distance to the bit-exact
//!   GEMV in FP16 ulps;
//! - row independence: a row's bits are the same alone, inside a 9-row
//!   panel at another position, and inside a full 32-row panel;
//! - run-to-run determinism;
//! - several segments, partial 8-row output tiles and partial token tiles.

use half::{bf16, f16};
use turbospark_gpu::{
    encode_inkling2_bf16_gemm_rows_mma, encode_inkling2_bf16_gemv_rows, Inkling2GemvRowsSegment,
    MetalContext,
};

fn lcg(s: &mut u32) -> u32 {
    *s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *s
}

fn unit(s: &mut u32) -> f32 {
    ((lcg(s) >> 8) as f32 / (1u32 << 23) as f32) - 1.0
}

struct Fixture {
    w: Vec<bf16>,
    /// (w offset in elements, rows, cols) per segment.
    segs: Vec<(usize, usize, usize)>,
    /// Token rows, each `xs` halfs (`cols` of the widest segment used).
    x: Vec<f16>,
    xs: usize,
}

/// Inkling shapes: an attention-input-like segment (2304 cols), a
/// partial-tile segment (rows not a multiple of 8 or 64), and an o-proj-like
/// one (3072 cols, which reads its own input).
fn fixture(n_tokens: usize, seed: u32) -> Fixture {
    let mut s = seed;
    let shapes = [(203usize, 2304usize), (37, 2304), (96, 3072)];
    let mut w = Vec::new();
    let mut segs = Vec::new();
    for (rows, cols) in shapes {
        segs.push((w.len(), rows, cols));
        w.extend((0..rows * cols).map(|_| bf16::from_f32(unit(&mut s) * 0.08)));
    }
    let xs = 3072;
    let x = (0..n_tokens * xs)
        .map(|i| f16::from_f32(unit(&mut s) * (0.5 + (i / xs) as f32 * 0.1)))
        .collect();
    Fixture { w, segs, x, xs }
}

fn bits(b: &metal::Buffer, n: usize) -> Vec<u16> {
    unsafe { std::slice::from_raw_parts(b.contents() as *const u16, n) }.to_vec()
}

/// Runs the kernel over token rows `rows` of the fixture (copied into a fresh
/// panel in that order) and returns, per segment, `[n][seg rows]` output bits.
fn run(f: &Fixture, rows: &[usize], mma: bool) -> Vec<Vec<u16>> {
    let mut ctx = MetalContext::new().expect("Metal device");
    let wbytes: Vec<u8> = f.w.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
    let wbuf = ctx.new_buffer_with_data(&wbytes);
    let panel: Vec<u8> = rows
        .iter()
        .flat_map(|&r| f.x[r * f.xs..(r + 1) * f.xs].iter())
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect();
    let xbuf = ctx.new_buffer_with_data(&panel);
    let n = rows.len();
    let ybufs: Vec<metal::Buffer> = f
        .segs
        .iter()
        .map(|&(_, r, _)| ctx.new_output_buffer((n * r * 2) as u64))
        .collect();
    let segs: Vec<Inkling2GemvRowsSegment<'_>> = f
        .segs
        .iter()
        .zip(&ybufs)
        .map(|(&(off, r, c), y)| Inkling2GemvRowsSegment {
            w_offset: (off * 2) as u64,
            rows: r as u32,
            cols: c as u32,
            x: 0,
            y: (y, 0),
            y_stride: r as u32,
        })
        .collect();
    let pass = ctx.begin_pass();
    let xsrc = [((&xbuf, 0u64), f.xs as u32)];
    if mma {
        encode_inkling2_bf16_gemm_rows_mma(&mut ctx, &pass, &wbuf, &xsrc, &segs, n as u32).unwrap();
    } else {
        encode_inkling2_bf16_gemv_rows(&mut ctx, &pass, &wbuf, &xsrc, &segs, n as u32).unwrap();
    }
    pass.commit_and_wait();
    f.segs
        .iter()
        .zip(&ybufs)
        .map(|(&(_, r, _), y)| bits(y, n * r))
        .collect()
}

fn reference(f: &Fixture, seg: usize, token: usize, row: usize) -> f64 {
    let (off, _, cols) = f.segs[seg];
    (0..cols)
        .map(|k| f.w[off + row * cols + k].to_f64() * f.x[token * f.xs + k].to_f64())
        .sum()
}

/// Accuracy against an f64 reference, judged against the bit-exact GEMV's own
/// error on the same data rather than a fixed ulp count: a near-zero output
/// can sit several FP16 ulps from the GEMV while both are equally accurate.
#[test]
fn mma_gemm_is_as_accurate_as_the_gemv() {
    let f = fixture(32, 0xa11ce);
    let rows: Vec<usize> = (0..32).collect();
    let got = run(&f, &rows, true);
    let gemv = run(&f, &rows, false);
    let (mut err_mma, mut err_gemv, mut worst_mma, mut ulp_max) = (0f64, 0f64, 0f64, 0u32);
    let mut n = 0u64;
    for (si, &(_, r, _)) in f.segs.iter().enumerate() {
        for t in 0..32 {
            for o in 0..r {
                let want = reference(&f, si, t, o);
                let m = f16::from_bits(got[si][t * r + o]).to_f64();
                let g = f16::from_bits(gemv[si][t * r + o]).to_f64();
                err_mma += (m - want).abs();
                err_gemv += (g - want).abs();
                worst_mma = worst_mma.max((m - want).abs() / want.abs().max(0.05));
                let d = (got[si][t * r + o] as i32 - gemv[si][t * r + o] as i32).unsigned_abs();
                ulp_max = ulp_max.max(d);
                n += 1;
            }
        }
    }
    let (em, eg) = (err_mma / n as f64, err_gemv / n as f64);
    eprintln!(
        "mean |err| vs f64: mma {em:.3e}, gemv {eg:.3e}; mma worst rel {worst_mma:.2e}; \
         max {ulp_max} FP16 ulps from the GEMV"
    );
    assert!(worst_mma < 2e-3, "mma gemm off the reference: {worst_mma}");
    // FP32 matrix-unit accumulation is not expected to be worse than the
    // GEMV's FP32 lanes; 10% headroom for which output rounds which way.
    assert!(
        em <= eg * 1.1,
        "mma gemm less accurate than the GEMV: {em} vs {eg}"
    );
}

#[test]
fn a_rows_bits_do_not_depend_on_the_panel() {
    let f = fixture(32, 0xbee5);
    let full: Vec<usize> = (0..32).collect();
    let all = run(&f, &full, true);
    let probe = 13usize;
    // Alone; at offset 5 of a 9-row panel; in a 17-row panel reversed.
    let alone = run(&f, &[probe], true);
    let nine: Vec<usize> = [0, 1, 2, 3, 4, probe, 20, 21, 22].to_vec();
    let in9 = run(&f, &nine, true);
    let rev: Vec<usize> = (5..22).rev().collect();
    let pos_rev = rev.iter().position(|&r| r == probe).unwrap();
    let in17 = run(&f, &rev, true);
    for (si, &(_, r, _)) in f.segs.iter().enumerate() {
        let want = &all[si][probe * r..(probe + 1) * r];
        assert_eq!(&alone[si][..r], want, "segment {si}: alone differs");
        assert_eq!(
            &in9[si][5 * r..6 * r],
            want,
            "segment {si}: 9-row panel differs"
        );
        assert_eq!(
            &in17[si][pos_rev * r..(pos_rev + 1) * r],
            want,
            "segment {si}: 17-row panel differs"
        );
    }
}

#[test]
fn mma_gemm_is_deterministic() {
    let f = fixture(23, 0xd00d);
    let rows: Vec<usize> = (0..23).collect();
    assert_eq!(run(&f, &rows, true), run(&f, &rows, true));
}

#[test]
fn mma_gemm_refuses_cols_not_a_multiple_of_64() {
    let mut ctx = MetalContext::new().expect("Metal device");
    let b = ctx.new_buffer_with_data(&[0u8; 4096]);
    let pass = ctx.begin_pass();
    let seg = Inkling2GemvRowsSegment {
        w_offset: 0,
        rows: 4,
        cols: 72,
        x: 0,
        y: (&b, 0),
        y_stride: 4,
    };
    assert!(
        encode_inkling2_bf16_gemm_rows_mma(&mut ctx, &pass, &b, &[((&b, 0), 72)], &[seg], 2)
            .is_err()
    );
    pass.commit_and_wait();
}
