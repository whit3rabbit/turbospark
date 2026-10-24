//! Decode-vs-verify kernel parity at `Qwen/Qwen3.8-27B`'s REAL shapes.
//!
//! On the real install a batched verify pass is not bit-identical to the
//! decode steps it stands in for (measured 2026-10-22,
//! `crates/bench/tests/int4_mma8_drift_probe.rs`: 0 of 255 rows equal with
//! every flag at its default, greedy speculation departing at token 166 on
//! a prose prompt). The synthetic suites prove bit-identity only at their
//! 128-wide shapes, and the kernels BAKE their shapes as function
//! constants, so a real-shape pipeline is a different compile. This file
//! bisects the INT4 projections kernel by kernel, each decode-side form
//! against the form the verify side runs:
//!
//! - A: the one-row GEMV against the batched GEMM, at 3 and 5 rows (a
//!   block-2 and a block-4 verify), every projection shape including the
//!   head.
//! - B: decode's fused FFN (`gate_up_act` + `down_epi` with the residual
//!   epilogue) against the unfused GEMV chain, which A ties to verify's
//!   GEMM chain.
//! - C: decode's fused GDN in-proj against four separate GEMVs.
//!
//! Weights and inputs are synthetic; only the DIMENSIONS are real.
//!
//! ```sh
//! cargo test -p turbospark-gpu --test qwen38_decode_verify_parity --release -- --nocapture
//! ```

#![cfg(target_os = "macos")]

use turbospark_gpu::{
    encode_dequant_int4_gemm_resident, encode_dequant_int4_gemv_resident, encode_down_epi_gemv,
    encode_gate_up_act_gemv, encode_gdn_in_proj, encode_residual_add, encode_silu_mul,
    read_buffer_f16, DownEpilogue, Int4ResidentMatrix, MetalBuffer, MetalContext,
};

const HIDDEN: usize = 5120;
const INTER: usize = 17408;

fn xorshift(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

/// A random INT4 matrix with small bf16 scales and biases, laid out
/// weights|scales|biases in one buffer.
struct Mat {
    buf: MetalBuffer,
    rows: usize,
    cols: usize,
}

impl Mat {
    fn new(context: &MetalContext, rows: usize, cols: usize, seed: u64) -> Self {
        let mut next = xorshift(seed);
        let groups = rows * cols / 64;
        let mut bytes: Vec<u8> = (0..rows * cols / 2).map(|_| next() as u8).collect();
        for _ in 0..2 {
            for _ in 0..groups {
                // bf16 in roughly [-0.03, 0.03], sign and magnitude varied.
                let v = ((next() % 2001) as f32 / 1000.0 - 1.0) * 0.03;
                let b = (v.to_bits() >> 16) as u16;
                bytes.extend_from_slice(&b.to_le_bytes());
            }
        }
        Self {
            buf: context.new_buffer_with_data(&bytes),
            rows,
            cols,
        }
    }

    fn m(&self) -> Int4ResidentMatrix<'_> {
        let s = (self.rows * self.cols / 2) as u64;
        Int4ResidentMatrix {
            buffer: &self.buf,
            weights_offset: 0,
            scales_offset: s,
            biases_offset: s + (self.rows * self.cols / 64 * 2) as u64,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

fn halves(context: &MetalContext, n: usize, seed: u64, scale: f32) -> MetalBuffer {
    let mut next = xorshift(seed);
    let bytes: Vec<u8> = (0..n)
        .flat_map(|_| {
            half::f16::from_f32(((next() % 2001) as f32 / 1000.0 - 1.0) * scale)
                .to_bits()
                .to_le_bytes()
        })
        .collect();
    context.new_buffer_with_data(&bytes)
}

fn mismatches(a: &[half::f16], b: &[half::f16]) -> (usize, f32) {
    let mut n = 0;
    let mut worst = 0f32;
    for (x, y) in a.iter().zip(b) {
        if x.to_bits() != y.to_bits() {
            n += 1;
            worst = worst.max((x.to_f32() - y.to_f32()).abs());
        }
    }
    (n, worst)
}

/// A: GEMV row by row against the batched GEMM at 3 and 5 rows.
#[test]
fn a_the_gemv_and_the_gemm_agree_at_every_projection_shape() {
    let mut context = MetalContext::new().expect("Metal device");
    let shapes: [(&str, usize, usize); 10] = [
        ("packed_q", 12288, HIDDEN),
        ("k/v", 1024, HIDDEN),
        ("o_proj / gdn out", HIDDEN, 6144),
        ("gate/up", INTER, HIDDEN),
        ("down", HIDDEN, INTER),
        ("gdn qkv", 10240, HIDDEN),
        ("gdn z", 6144, HIDDEN),
        ("gdn a/b", 48, HIDDEN),
        ("mtp fc", HIDDEN, 2 * HIDDEN),
        ("head", 248320, HIDDEN),
    ];
    let mut failures = Vec::new();
    for (label, rows, cols) in shapes {
        let w = Mat::new(&context, rows, cols, 0xA000 + rows as u64);
        for batch in [3usize, 5] {
            let x = halves(&context, batch * cols, 0xB000 + batch as u64, 1.0);
            let y_gemm = context.new_output_buffer((batch * rows * 2) as u64);
            let y_gemv = context.new_output_buffer((batch * rows * 2) as u64);
            let pass = context.begin_pass();
            encode_dequant_int4_gemm_resident(
                &mut context,
                &pass,
                &w.m(),
                (&x, 0),
                (&y_gemm, 0),
                batch,
            )
            .unwrap();
            for t in 0..batch {
                encode_dequant_int4_gemv_resident(
                    &mut context,
                    &pass,
                    &w.m(),
                    (&x, (t * cols * 2) as u64),
                    (&y_gemv, (t * rows * 2) as u64),
                )
                .unwrap();
            }
            pass.commit_and_wait();
            let (n, worst) = mismatches(
                &read_buffer_f16(&y_gemm, 0, batch * rows),
                &read_buffer_f16(&y_gemv, 0, batch * rows),
            );
            println!("A {label:<17} {rows:>6}x{cols:<5} batch {batch}: {n} of {} differ (max {worst:.3e})", batch * rows);
            if n > 0 {
                failures.push(format!("{label} batch {batch}: {n} differ"));
            }
        }
    }
    assert!(failures.is_empty(), "GEMV vs GEMM: {failures:?}");
}

/// B: the fused FFN tail against the unfused GEMV chain decode would run
/// with `TURBOSPARK_QWEN_FUSED_FFN=0` (and that verify matches via A).
#[test]
fn b_the_fused_ffn_matches_the_unfused_chain_at_the_real_widths() {
    let mut context = MetalContext::new().expect("Metal device");
    let gate = Mat::new(&context, INTER, HIDDEN, 0xC001);
    let up = Mat::new(&context, INTER, HIDDEN, 0xC002);
    let down = Mat::new(&context, HIDDEN, INTER, 0xC003);
    let x = halves(&context, HIDDEN, 0xC004, 1.0);
    let residual0 = halves(&context, HIDDEN, 0xC005, 4.0);
    let res_bytes: Vec<u8> = read_buffer_f16(&residual0, 0, HIDDEN)
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect();

    // Fused.
    let act_f = context.new_output_buffer((INTER * 2) as u64);
    let res_f = context.new_buffer_with_data(&res_bytes);
    let pass = context.begin_pass();
    encode_gate_up_act_gemv(
        &mut context,
        &pass,
        &gate.m(),
        &up.m(),
        (&x, 0),
        (&act_f, 0),
        false,
    )
    .unwrap();
    encode_down_epi_gemv(
        &mut context,
        &pass,
        &down.m(),
        (&act_f, 0),
        (&res_f, 0),
        (&res_f, 0),
        DownEpilogue::Residual,
    )
    .unwrap();
    pass.commit_and_wait();

    // Unfused chain.
    let g = context.new_output_buffer((INTER * 2) as u64);
    let u = context.new_output_buffer((INTER * 2) as u64);
    let act_u = context.new_output_buffer((INTER * 2) as u64);
    let h2 = context.new_output_buffer((HIDDEN * 2) as u64);
    let res_u = context.new_buffer_with_data(&res_bytes);
    let pass = context.begin_pass();
    encode_dequant_int4_gemv_resident(&mut context, &pass, &gate.m(), (&x, 0), (&g, 0)).unwrap();
    encode_dequant_int4_gemv_resident(&mut context, &pass, &up.m(), (&x, 0), (&u, 0)).unwrap();
    encode_silu_mul(
        &mut context,
        &pass,
        (&g, 0),
        (&u, 0),
        (&act_u, 0),
        INTER as u32,
    )
    .unwrap();
    encode_dequant_int4_gemv_resident(&mut context, &pass, &down.m(), (&act_u, 0), (&h2, 0))
        .unwrap();
    encode_residual_add(&mut context, &pass, (&res_u, 0), (&h2, 0), HIDDEN as u32).unwrap();
    pass.commit_and_wait();

    let (na, wa) = mismatches(
        &read_buffer_f16(&act_f, 0, INTER),
        &read_buffer_f16(&act_u, 0, INTER),
    );
    let (nr, wr) = mismatches(
        &read_buffer_f16(&res_f, 0, HIDDEN),
        &read_buffer_f16(&res_u, 0, HIDDEN),
    );
    println!("B gate_up_act: {na} of {INTER} differ (max {wa:.3e}); down+residual: {nr} of {HIDDEN} differ (max {wr:.3e})");
    assert_eq!(
        (na, nr),
        (0, 0),
        "fused FFN differs from the unfused chain at the real widths"
    );
}

/// C: the fused GDN in-proj against four separate GEMVs.
#[test]
fn c_the_fused_gdn_in_proj_matches_four_gemvs_at_the_real_widths() {
    let mut context = MetalContext::new().expect("Metal device");
    let shapes = [10240usize, 6144, 48, 48];
    let mats: Vec<Mat> = shapes
        .iter()
        .enumerate()
        .map(|(i, &r)| Mat::new(&context, r, HIDDEN, 0xD000 + i as u64))
        .collect();
    let x = halves(&context, HIDDEN, 0xD100, 1.0);
    let fused: Vec<MetalBuffer> = shapes
        .iter()
        .map(|&r| context.new_output_buffer((r * 2) as u64))
        .collect();
    let plain: Vec<MetalBuffer> = shapes
        .iter()
        .map(|&r| context.new_output_buffer((r * 2) as u64))
        .collect();
    let pass = context.begin_pass();
    encode_gdn_in_proj(
        &mut context,
        &pass,
        &mats[0].m(),
        &mats[1].m(),
        &mats[2].m(),
        &mats[3].m(),
        (&x, 0),
        (&fused[0], 0),
        (&fused[1], 0),
        (&fused[2], 0),
        (&fused[3], 0),
    )
    .unwrap();
    for i in 0..4 {
        encode_dequant_int4_gemv_resident(
            &mut context,
            &pass,
            &mats[i].m(),
            (&x, 0),
            (&plain[i], 0),
        )
        .unwrap();
    }
    pass.commit_and_wait();
    let mut bad = Vec::new();
    for (i, name) in ["qkv", "z", "a", "b"].iter().enumerate() {
        let (n, w) = mismatches(
            &read_buffer_f16(&fused[i], 0, shapes[i]),
            &read_buffer_f16(&plain[i], 0, shapes[i]),
        );
        println!(
            "C in_proj {name}: {n} of {} differ (max {w:.3e})",
            shapes[i]
        );
        if n > 0 {
            bad.push(*name);
        }
    }
    assert!(
        bad.is_empty(),
        "fused GDN in-proj differs from GEMVs: {bad:?}"
    );
}
