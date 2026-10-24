//! Gates for the production small-M INT4 MMA GEMM
//! (`encode_dequant_int4_mma8_resident`, `docs/INT4_SMALL_M_MMA.md`).
//!
//! The kernel is not bit-identical to the GEMV, so these stand in for it:
//! accuracy against an f64 reference no worse than the shipped batched GEMM,
//! and a row's bits independent of how many rows share the tile, which is
//! the property decode and verify rely on to stay bit-exact against each
//! other when both run this kernel.

#![cfg(target_os = "macos")]

use turbospark_gpu::{
    encode_dequant_int4_gemm_resident, encode_dequant_int4_mma8_resident, read_buffer_f16,
    Int4ResidentMatrix, MetalBuffer, MetalContext,
};

fn xorshift(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

struct Fixture {
    buffer: MetalBuffer,
    offsets: (u64, u64, u64),
    rows: usize,
    cols: usize,
    packed: Vec<u8>,
    scales: Vec<u16>,
    biases: Vec<u16>,
    x: MetalBuffer,
    host_x: Vec<half::f16>,
}

impl Fixture {
    /// Weights at a 2-aligned (not 4-aligned) offset: the resident shape.
    fn new(context: &MetalContext, rows: usize, cols: usize, seed: u64) -> Self {
        let mut next = xorshift(seed);
        let (mut packed, mut scales, mut biases) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..rows {
            let row: Vec<f32> = (0..cols)
                .map(|_| ((next() % 2000) as f32 / 1000.0 - 1.0) * 0.5)
                .collect();
            let q = turbospark_compute::quantize_int4_affine(&row);
            packed.extend_from_slice(&q.packed);
            scales.extend_from_slice(&q.scales);
            biases.extend_from_slice(&q.biases);
        }
        let w_off = 2usize;
        let s_off = w_off + packed.len();
        let b_off = s_off + scales.len() * 2;
        let mut bytes = vec![0u8; b_off + biases.len() * 2];
        bytes[w_off..s_off].copy_from_slice(&packed);
        for (i, v) in scales.iter().enumerate() {
            bytes[s_off + 2 * i..][..2].copy_from_slice(&v.to_le_bytes());
        }
        for (i, v) in biases.iter().enumerate() {
            bytes[b_off + 2 * i..][..2].copy_from_slice(&v.to_le_bytes());
        }
        let host_x: Vec<half::f16> = (0..8 * cols)
            .map(|_| half::f16::from_f32((next() % 2000) as f32 / 1000.0 - 1.0))
            .collect();
        let xb: Vec<u8> = host_x
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        Self {
            buffer: context.new_buffer_with_data(&bytes),
            offsets: (w_off as u64, s_off as u64, b_off as u64),
            rows,
            cols,
            packed,
            scales,
            biases,
            x: context.new_buffer_with_data(&xb),
            host_x,
        }
    }

    fn matrix(&self) -> Int4ResidentMatrix<'_> {
        Int4ResidentMatrix {
            buffer: &self.buffer,
            weights_offset: self.offsets.0,
            scales_offset: self.offsets.1,
            biases_offset: self.offsets.2,
            rows: self.rows,
            cols: self.cols,
        }
    }

    fn reference(&self, r: usize) -> Vec<f64> {
        let bf = |b: u16| f32::from_bits((b as u32) << 16) as f64;
        let (n, k, g) = (self.rows, self.cols, self.cols / 64);
        let mut y = vec![0.0; r * n];
        for t in 0..r {
            for m in 0..n {
                let mut acc = 0.0f64;
                for i in 0..k {
                    let byte = self.packed[m * k / 2 + i / 2];
                    let q = if i & 1 == 0 { byte & 0xF } else { byte >> 4 } as f64;
                    let w = q * bf(self.scales[m * g + i / 64]) + bf(self.biases[m * g + i / 64]);
                    acc += w * self.host_x[t * k + i].to_f64();
                }
                y[t * n + m] = acc;
            }
        }
        y
    }

    fn run_mma8(&self, context: &mut MetalContext, r: usize) -> Vec<half::f16> {
        let y = context.new_output_buffer((r * self.rows * 2) as u64);
        let pass = context.begin_pass();
        encode_dequant_int4_mma8_resident(context, &pass, &self.matrix(), (&self.x, 0), (&y, 0), r)
            .unwrap();
        pass.commit_and_wait();
        read_buffer_f16(&y, 0, r * self.rows)
    }

    fn run_gemm(&self, context: &mut MetalContext, r: usize) -> Vec<half::f16> {
        let y = context.new_output_buffer((r * self.rows * 2) as u64);
        let pass = context.begin_pass();
        encode_dequant_int4_gemm_resident(context, &pass, &self.matrix(), (&self.x, 0), (&y, 0), r)
            .unwrap();
        pass.commit_and_wait();
        read_buffer_f16(&y, 0, r * self.rows)
    }
}

fn max_err(got: &[half::f16], want: &[f64]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (g.to_f64() - w).abs())
        .fold(0.0, f64::max)
}

/// Shapes cover every K-split arm (16, 8 and 4 slices) and a partial
/// 8-row block.
const SHAPES: [(usize, usize); 4] = [(44, 1024), (256, 2048), (600, 1024), (8200, 512)];

#[test]
fn mma8_is_as_accurate_as_the_shipped_gemm() {
    let mut context = MetalContext::new().expect("Metal device");
    for (n, k) in SHAPES {
        let f = Fixture::new(&context, n, k, 0xD8_000 + n as u64);
        for r in [1usize, 2, 5, 8] {
            let want = f.reference(r);
            let scale = want.iter().map(|v| v.abs()).fold(0.0, f64::max);
            let gemm_err = max_err(&f.run_gemm(&mut context, r), &want);
            let err = max_err(&f.run_mma8(&mut context, r), &want);
            // Two half ulps at the output's scale over the GEMM's own error:
            // past that is a layout or epilogue bug, not summation order.
            let bound = 2.0 * gemm_err + scale * 2.0f64.powi(-10);
            assert!(
                err <= bound,
                "{n}x{k} R={r}: max|err| {err:.3e} exceeds {bound:.3e} (gemm {gemm_err:.3e})"
            );
        }
    }
}

#[test]
fn a_rows_bits_do_not_depend_on_the_rows_beside_it() {
    let mut context = MetalContext::new().expect("Metal device");
    for (n, k) in SHAPES {
        let f = Fixture::new(&context, n, k, 0xD9_000 + n as u64);
        let alone = f.run_mma8(&mut context, 1);
        for r in [2usize, 3, 8] {
            let together = f.run_mma8(&mut context, r);
            for (i, (a, b)) in alone.iter().zip(&together[..n]).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{n}x{k}: row 0 output {i} differs between R=1 and R={r}"
                );
            }
        }
        // And a later row: row 2 run at R=3 against row 2 inside R=8.
        let three = f.run_mma8(&mut context, 3);
        let eight = f.run_mma8(&mut context, 8);
        assert!(
            three[2 * n..3 * n]
                .iter()
                .zip(&eight[2 * n..3 * n])
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{n}x{k}: row 2 differs between R=3 and R=8"
        );
    }
}

#[test]
fn out_of_range_calls_are_refused() {
    let mut context = MetalContext::new().expect("Metal device");
    let f = Fixture::new(&context, 16, 128, 1);
    let y = context.new_output_buffer(16 * 9 * 2);
    let pass = context.begin_pass();
    for r in [0usize, 9] {
        assert!(encode_dequant_int4_mma8_resident(
            &mut context,
            &pass,
            &f.matrix(),
            (&f.x, 0),
            (&y, 0),
            r
        )
        .is_err());
    }
    pass.commit_and_wait();
}
