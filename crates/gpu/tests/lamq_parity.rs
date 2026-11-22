#![cfg(target_os = "macos")]
//! Parity test for `lamq.metal` (Kronecker Hadamard rotation, trellis and
//! level GEMVs, one-row decode, bucket scatter) against
//! `turbospark_compute::lamq`, on real Metal hardware.
//!
//! Fixtures are random but cover what each kernel branches on: every code
//! (hashed groups of 4 and 2, row trellis at 1 to 4 bits, both level
//! packings), row counts that are not a multiple of the 8 rows a threadgroup
//! takes, every release width of the rotation in both directions, and FP16 as
//! well as F32 outputs.

use half::f16;
use turbospark_compute::lamq::{LamqCode, LamqLinear, Rht};
use turbospark_gpu::{
    encode_lamq_decode_row, encode_lamq_gemv, encode_lamq_rht, encode_lamq_rht_in_batch,
    encode_lamq_scatter, LamqMatrixView, LamqRhtDir, LamqRhtFactor, MetalContext,
};

fn xs(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn unit(s: &mut u32) -> f32 {
    (xs(s) >> 8) as f32 / (1u32 << 23) as f32 - 1.0
}

fn signs(n: usize, seed: u32) -> Vec<i8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| if xs(&mut s) & 1 == 0 { 1 } else { -1 })
        .collect()
}

fn read_f32(b: &metal::Buffer, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(b.contents() as *const f32, n) }.to_vec()
}

fn read_f16(b: &metal::Buffer, n: usize) -> Vec<f32> {
    let bits = unsafe { std::slice::from_raw_parts(b.contents() as *const u16, n) };
    bits.iter().map(|&v| f16::from_bits(v).to_f32()).collect()
}

fn le32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn le16(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| f16::from_f32(x).to_bits().to_le_bytes())
        .collect()
}

fn assert_close(got: &[f32], want: &[f64], abs: f64, rel: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let tol = abs + rel * w.abs();
        assert!(
            (f64::from(g) - w).abs() <= tol,
            "{what}[{i}]: got {g}, want {w} (tol {tol})"
        );
    }
}

#[test]
fn rotation_matches_the_reference_at_every_release_width_both_ways() {
    let mut context = MetalContext::new().expect("Metal device");
    for n in [384usize, 1024, 1152, 2304, 3072] {
        let factor = LamqRhtFactor::new(&context, n).unwrap();
        let rht = Rht::new(n).unwrap();
        let s = signs(n, n as u32);
        let s_buf = context.new_buffer_with_data(&s.iter().map(|&v| v as u8).collect::<Vec<_>>());
        let mut seed = 7 + n as u32;
        // Two vectors, to cover the per-threadgroup offset.
        let x: Vec<f32> = (0..2 * n).map(|_| unit(&mut seed)).collect();
        // FP16 in, F32 out (the input rotation of an activation).
        let x16: Vec<f32> = x.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
        let x_buf = context.new_buffer_with_data(&le16(&x));
        let y_buf = context.new_buffer_with_data(&vec![0u8; 2 * n * 4]);
        // F32 in, FP16 out (the output rotation of a GEMV result).
        let xf_buf = context.new_buffer_with_data(&le32(&x));
        let z_buf = context.new_buffer_with_data(&vec![0u8; 2 * n * 2]);
        let pass = context.begin_pass();
        encode_lamq_rht(
            &mut context,
            &pass,
            &factor,
            LamqRhtDir::In,
            (&x_buf, 0),
            true,
            (&s_buf, 0),
            (&y_buf, 0),
            false,
            2,
        )
        .unwrap();
        encode_lamq_rht(
            &mut context,
            &pass,
            &factor,
            LamqRhtDir::Out,
            (&xf_buf, 0),
            false,
            (&s_buf, 0),
            (&z_buf, 0),
            true,
            2,
        )
        .unwrap();
        pass.commit_and_wait();
        let got_in = read_f32(&y_buf, 2 * n);
        let got_out = read_f16(&z_buf, 2 * n);
        for v in 0..2 {
            let want_in = rht.apply_in(&x16[v * n..(v + 1) * n], &s);
            assert_close(
                &got_in[v * n..(v + 1) * n],
                &want_in,
                2e-5,
                1e-5,
                &format!("in n={n} v={v}"),
            );
            let xv: Vec<f64> = x[v * n..(v + 1) * n]
                .iter()
                .map(|&a| f64::from(a))
                .collect();
            let want_out = rht.apply_out(&xv, &s);
            assert_close(
                &got_out[v * n..(v + 1) * n],
                &want_out,
                1e-3,
                1e-3,
                &format!("out n={n} v={v}"),
            );
        }
    }
}

const CODES: [LamqCode; 8] = [
    LamqCode::Ht2,
    LamqCode::Ht3,
    LamqCode::Rtr1,
    LamqCode::Rtr2,
    LamqCode::Rtr3,
    LamqCode::Rtr4,
    LamqCode::Lm6,
    LamqCode::Lm8,
];

/// Random codes, scales and table for one `rows x cols` matrix.
struct Fixture {
    code: LamqCode,
    rows: usize,
    cols: usize,
    codes: Vec<u8>,
    scales: Vec<u16>,
    table: Vec<f32>,
}

impl Fixture {
    fn new(code: LamqCode, rows: usize, cols: usize, seed: u32) -> Self {
        let mut s = seed | 1;
        let rb = code.row_bytes(cols).unwrap();
        let codes: Vec<u8> = (0..rows * rb).map(|_| xs(&mut s) as u8).collect();
        let scales: Vec<u16> = (0..rows)
            .map(|_| f16::from_f32(0.01 + 0.05 * (unit(&mut s) + 1.0)).to_bits())
            .collect();
        // Trellis LUTs are FP16 on device; round them here so the reference
        // sees the same values.
        let table: Vec<f32> = (0..code.table_rows() * code.g())
            .map(|_| {
                let v = unit(&mut s) * 2.0;
                if code.is_trellis() {
                    f16::from_f32(v).to_f32()
                } else {
                    v
                }
            })
            .collect();
        Self {
            code,
            rows,
            cols,
            codes,
            scales,
            table,
        }
    }

    fn reference(&self) -> LamqLinear<'_> {
        LamqLinear {
            code: self.code,
            m: self.rows,
            n: self.cols,
            codes: &self.codes,
            scales: &self.scales,
            signs_in: &[],
            signs_out: None,
            table: &self.table,
        }
    }

    fn table_bytes(&self) -> Vec<u8> {
        if self.code.is_trellis() {
            le16(&self.table)
        } else {
            le32(&self.table)
        }
    }
}

#[test]
fn gemv_matches_the_reference_for_every_code() {
    let mut context = MetalContext::new().expect("Metal device");
    for (ci, &code) in CODES.iter().enumerate() {
        for (rows, cols) in [(37usize, 2304usize), (9, 1152), (3, 384)] {
            let f = Fixture::new(code, rows, cols, 1000 + ci as u32 * 31 + rows as u32);
            let mut seed = 99 + rows as u32;
            let x: Vec<f32> = (0..cols).map(|_| unit(&mut seed)).collect();
            let codes = context.new_buffer_with_data(&f.codes);
            let scales = context.new_buffer_with_data(
                &f.scales
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            let table = context.new_buffer_with_data(&f.table_bytes());
            let x_buf = context.new_buffer_with_data(&le32(&x));
            let y32 = context.new_buffer_with_data(&vec![0u8; rows * 4]);
            let y16 = context.new_buffer_with_data(&vec![0u8; rows * 2]);
            let view = LamqMatrixView {
                code,
                rows: rows as u32,
                cols: cols as u32,
                codes: (&codes, 0),
                scales: (&scales, 0),
                table: (&table, 0),
            };
            let pass = context.begin_pass();
            encode_lamq_gemv(&mut context, &pass, &view, (&x_buf, 0), (&y32, 0), false).unwrap();
            encode_lamq_gemv(&mut context, &pass, &view, (&x_buf, 0), (&y16, 0), true).unwrap();
            pass.commit_and_wait();
            let lin = f.reference();
            let want: Vec<f64> = (0..rows)
                .map(|r| {
                    lin.row(r)
                        .iter()
                        .zip(&x)
                        .map(|(&w, &v)| f64::from(w) * f64::from(v))
                        .sum()
                })
                .collect();
            let what = format!("{} {rows}x{cols}", code.name());
            assert_close(
                &read_f32(&y32, rows),
                &want,
                1e-4,
                1e-4,
                &format!("{what} f32"),
            );
            assert_close(
                &read_f16(&y16, rows),
                &want,
                2e-3,
                2e-3,
                &format!("{what} f16"),
            );
        }
    }
}

#[test]
fn decode_row_matches_the_reference_row() {
    let mut context = MetalContext::new().expect("Metal device");
    for (ci, &code) in CODES.iter().enumerate() {
        let (rows, cols) = (5usize, 2304usize);
        let f = Fixture::new(code, rows, cols, 77 + ci as u32);
        let codes = context.new_buffer_with_data(&f.codes);
        let scales = context.new_buffer_with_data(
            &f.scales
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let table = context.new_buffer_with_data(&f.table_bytes());
        let out = context.new_buffer_with_data(&vec![0u8; cols * 4]);
        let view = LamqMatrixView {
            code,
            rows: rows as u32,
            cols: cols as u32,
            codes: (&codes, 0),
            scales: (&scales, 0),
            table: (&table, 0),
        };
        for r in [0u32, 4] {
            let pass = context.begin_pass();
            encode_lamq_decode_row(&mut context, &pass, &view, r, (&out, 0)).unwrap();
            pass.commit_and_wait();
            let want: Vec<f64> = f
                .reference()
                .row(r as usize)
                .into_iter()
                .map(f64::from)
                .collect();
            assert_close(
                &read_f32(&out, cols),
                &want,
                0.0,
                1e-6,
                &format!("{} row {r}", code.name()),
            );
        }
        let pass = context.begin_pass();
        assert!(
            encode_lamq_decode_row(&mut context, &pass, &view, 5, (&out, 0)).is_err(),
            "row past the end"
        );
        pass.commit_and_wait();
    }
}

#[test]
fn scatter_writes_bucket_values_to_their_token_ids() {
    let mut context = MetalContext::new().expect("Metal device");
    let n = 1000usize;
    // A permutation: position p holds token (p * 7 + 3) % n (7 is coprime with 1000).
    let order: Vec<i32> = (0..n).map(|p| ((p * 7 + 3) % n) as i32).collect();
    let src: Vec<f32> = (0..n).map(|p| p as f32).collect();
    let src_buf = context.new_buffer_with_data(&le32(&src));
    let ord_buf = context.new_buffer_with_data(
        &order
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let dst = context.new_buffer_with_data(&vec![0u8; n * 2]);
    let pass = context.begin_pass();
    encode_lamq_scatter(
        &mut context,
        &pass,
        (&src_buf, 0),
        (&ord_buf, 0),
        (&dst, 0),
        n as u32,
    )
    .unwrap();
    pass.commit_and_wait();
    // Positions below 2048 are exact in FP16.
    let got = read_f16(&dst, n);
    for (p, &t) in order.iter().enumerate() {
        assert_eq!(got[t as usize], p as f32, "token {t}");
    }
}

#[test]
fn shapes_out_of_contract_are_refused() {
    let context = MetalContext::new().expect("Metal device");
    assert!(
        LamqRhtFactor::new(&context, 4096).is_err(),
        "wider than the kernel holds"
    );
    assert!(LamqRhtFactor::new(&context, 10).is_err(), "no Hadamard");
}

/// The batched input rotation is the single-vector kernel run on several
/// vectors at once: every job's output must equal rotating that input with
/// those signs on its own, bit for bit. Five jobs over three inputs (one
/// input used twice, one signs vector at a nonzero offset) at the two
/// batched widths.
#[test]
fn batched_rotation_equals_one_at_a_time_bit_for_bit() {
    let mut context = MetalContext::new().expect("Metal device");
    for n in [1152usize, 2304] {
        let factor = LamqRhtFactor::new(&context, n).unwrap();
        // Five sign vectors back to back in one buffer, like the weights.
        let all_signs: Vec<u8> = (0..5)
            .flat_map(|k| signs(n, 31 * (k + 1) + n as u32))
            .map(|v| v as u8)
            .collect();
        let s_buf = context.new_buffer_with_data(&all_signs);
        let mut seed = 11 + n as u32;
        let inputs: Vec<metal::Buffer> = (0..3)
            .map(|_| {
                let x: Vec<f32> = (0..n).map(|_| unit(&mut seed)).collect();
                context.new_buffer_with_data(&le16(&x))
            })
            .collect();
        // (input, signs vector)
        let jobs = [(0u32, 0usize), (1, 1), (2, 2), (0, 3), (1, 4)];
        let batched = context.new_buffer_with_data(&vec![0u8; jobs.len() * n * 4]);
        let single: Vec<metal::Buffer> = jobs
            .iter()
            .map(|_| context.new_buffer_with_data(&vec![0u8; n * 4]))
            .collect();
        let pass = context.begin_pass();
        let xs: Vec<(&metal::Buffer, u64)> = inputs.iter().map(|b| (b, 0)).collect();
        let table: Vec<(u32, u64)> = jobs.iter().map(|&(x, k)| (x, (k * n) as u64)).collect();
        encode_lamq_rht_in_batch(
            &mut context,
            &pass,
            &factor,
            &xs,
            &s_buf,
            &table,
            (&batched, 0),
        )
        .unwrap();
        for (i, &(x, k)) in jobs.iter().enumerate() {
            encode_lamq_rht(
                &mut context,
                &pass,
                &factor,
                LamqRhtDir::In,
                (&inputs[x as usize], 0),
                true,
                (&s_buf, (k * n) as u64),
                (&single[i], 0),
                false,
                1,
            )
            .unwrap();
        }
        pass.commit_and_wait();
        let got = read_f32(&batched, jobs.len() * n);
        for (i, one) in single.iter().enumerate() {
            let want = read_f32(one, n);
            let row = &got[i * n..(i + 1) * n];
            assert!(
                row.iter()
                    .zip(&want)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "n {n} job {i}: batched rotation differs from the single one"
            );
        }
        // Out-of-contract batches are refused: no jobs, and an input index
        // past the inputs bound.
        let refused = context.begin_pass();
        assert!(encode_lamq_rht_in_batch(
            &mut context,
            &refused,
            &factor,
            &xs,
            &s_buf,
            &[],
            (&batched, 0)
        )
        .is_err());
        assert!(encode_lamq_rht_in_batch(
            &mut context,
            &refused,
            &factor,
            &xs,
            &s_buf,
            &[(3, 0)],
            (&batched, 0)
        )
        .is_err());
        refused.commit_and_wait();
    }
}
