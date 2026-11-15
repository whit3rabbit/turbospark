//! Runs the kernels in `shaders/dequant_ext.metal` on real Metal hardware
//! against the CPU decoders in `turbospark_compute` (held to ggml's own
//! `to_float`). Bytes are BUILT here: every byte pattern is a valid block, so
//! the CPU reference and the GPU get identical bytes and only FP16 rounding
//! and reduction order may differ. Scales carry a random sign on purpose.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{
    dequant_ext_gemv_resident, encode_embed_lookup_ext, ExtBlockType, ExtResidentMatrix,
};

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

fn f16_bytes(v: f32) -> [u8; 2] {
    f16::from_f32(v).to_bits().to_le_bytes()
}

fn blocks(kind: ExtBlockType, n: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n % kind.block_elems(), 0);
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut out = Vec::new();
    for _ in 0..n / kind.block_elems() {
        match kind {
            ExtBlockType::Q1_0 => {
                let sign = if lcg(&mut s) & 1 == 0 { 1.0 } else { -1.0 };
                out.extend_from_slice(&f16_bytes(sign * (0.2 + (lcg(&mut s) % 8) as f32 * 0.05)));
                for _ in 0..16 {
                    out.push((lcg(&mut s) & 0xFF) as u8);
                }
            }
            // The two ternary types and NVFP4: every byte pattern is a valid
            // code, so random bytes cover the code space; only the scale field
            // is shaped (a small f16 for TQ, E4M3 bytes for NVFP4 that avoid
            // the 0x7F NaN code and stay small enough for f16).
            ExtBlockType::Tq2_0 | ExtBlockType::Tq1_0 => {
                let body = kind.block_bytes() - 2;
                for _ in 0..body {
                    out.push((lcg(&mut s) & 0xFF) as u8);
                }
                let sign = if lcg(&mut s) & 1 == 0 { 1.0 } else { -1.0 };
                out.extend_from_slice(&f16_bytes(sign * (0.2 + (lcg(&mut s) % 8) as f32 * 0.05)));
            }
            ExtBlockType::Q2_0 => {
                let sign = if lcg(&mut s) & 1 == 0 { 1.0 } else { -1.0 };
                out.extend_from_slice(&f16_bytes(sign * (0.2 + (lcg(&mut s) % 8) as f32 * 0.05)));
                for _ in 0..16 {
                    out.push((lcg(&mut s) & 0xFF) as u8);
                }
            }
            ExtBlockType::Mxfp4 => {
                // Exponent bytes around 124..131: scale 2^-4 .. 2^3. Includes the
                // 0 and 1 subnormal bytes on some blocks so they are exercised.
                let e = match lcg(&mut s) % 10 {
                    0 => 0u8,
                    1 => 1u8,
                    k => 120 + k as u8,
                };
                out.push(e);
                for _ in 0..16 {
                    out.push((lcg(&mut s) & 0xFF) as u8);
                }
            }
            ExtBlockType::Iq1Xs | ExtBlockType::Iq1Xxs | ExtBlockType::Iq1Xxxs => {
                // Every byte pattern is a legal index/scale code. d is small so
                // the largest weight (15 * (1 + 0.125) * d) stays in FP16.
                out.extend_from_slice(&f16_bytes(0.02 + (lcg(&mut s) % 8) as f32 * 0.004));
                for _ in 0..kind.block_bytes() - 2 {
                    out.push((lcg(&mut s) & 0xFF) as u8);
                }
            }
            ExtBlockType::Nvfp4 => {
                for _ in 0..4 {
                    // 0x30..0x48 is scale 0.5..2.5; bit 7 is set on some so the
                    // ignored-sign-bit rule is exercised.
                    let scale = 0x30 + (lcg(&mut s) % 0x18) as u8;
                    out.push(if lcg(&mut s) & 1 == 0 {
                        scale
                    } else {
                        scale | 0x80
                    });
                }
                for _ in 0..32 {
                    out.push((lcg(&mut s) & 0xFF) as u8);
                }
            }
        }
        assert_eq!(out.len() % kind.block_bytes(), 0);
    }
    out
}

fn decode(kind: ExtBlockType, bytes: &[u8], n: usize) -> Vec<f32> {
    match kind {
        ExtBlockType::Q1_0 => turbospark_compute::dequantize_q1_0(bytes, n),
        ExtBlockType::Tq2_0 => turbospark_compute::dequantize_tq2_0(bytes, n),
        ExtBlockType::Tq1_0 => turbospark_compute::dequantize_tq1_0(bytes, n),
        ExtBlockType::Nvfp4 => turbospark_compute::dequantize_nvfp4(bytes, n),
        ExtBlockType::Q2_0 => turbospark_compute::dequantize_q2_0(bytes, n),
        ExtBlockType::Mxfp4 => turbospark_compute::dequantize_mxfp4(bytes, n),
        ExtBlockType::Iq1Xs => turbospark_compute::dequantize_iq1_xs(bytes, n),
        ExtBlockType::Iq1Xxs => turbospark_compute::dequantize_iq1_xxs(bytes, n),
        ExtBlockType::Iq1Xxxs => turbospark_compute::dequantize_iq1_xxxs(bytes, n),
    }
}

fn activations(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| ((lcg(&mut s) >> 8) as f32 / (1u32 << 23) as f32) - 1.0)
        .collect()
}

fn assert_matches(label: &str, gpu: &[f16], cpu: &[f32], n: usize) {
    assert_eq!(gpu.len(), cpu.len());
    let gpu_f32: Vec<f32> = gpu.iter().map(|v| v.to_f32()).collect();
    let err = turbospark_compute::max_abs_diff(&gpu_f32, cpu);
    let scale = cpu.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1.0);
    let bound = scale * 1e-2 + (n as f32) * 1e-4;
    assert!(err < bound, "{label}: err = {err}, bound = {bound}");
}

fn gemv_case(kind: ExtBlockType, rows: usize, n: usize, seed: u32) {
    let mut context =
        turbospark_gpu::MetalContext::new().expect("Metal device available on this machine");
    let mut table = Vec::new();
    for r in 0..rows {
        table.extend_from_slice(&blocks(kind, n, seed + r as u32));
    }
    let row_bytes = kind.row_bytes(n);
    assert_eq!(table.len(), rows * row_bytes);
    // A nonzero offset proves the matrix is addressed in place, not from 0.
    let pad = 96usize;
    let mut shared = vec![0xA5u8; pad];
    shared.extend_from_slice(&table);
    let buffer = context.new_buffer_with_data(&shared);

    let x_f32 = activations(n, 99 + seed);
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let x_ref: Vec<f32> = x_f16.iter().map(|v| v.to_f32()).collect();
    let want: Vec<f32> = (0..rows)
        .map(|r| {
            let w = decode(kind, &table[r * row_bytes..(r + 1) * row_bytes], n);
            w.iter().zip(&x_ref).map(|(a, b)| a * b).sum()
        })
        .collect();
    let w = ExtResidentMatrix {
        buffer: &buffer,
        weights_offset: pad as u64,
        rows,
        cols: n,
        kind,
    };
    let got = dequant_ext_gemv_resident(&mut context, &w, &x_f16).expect("gpu gemv");
    assert_matches(&format!("{kind:?} gemv {rows}x{n}"), &got, &want, n);
}

#[test]
fn gemv_matches_cpu_reference_for_every_type() {
    for kind in ExtBlockType::ALL {
        let b = kind.block_elems();
        // 5 rows: not a multiple of 8, so the early-return guard is exercised.
        gemv_case(kind, 5, 3 * b, 7);
        gemv_case(kind, 3, b, 11);
        // More than one threadgroup of rows.
        gemv_case(kind, 19, 8 * b, 21);
    }
}

/// The row is not row 0 (a wrong stride returns a neighbour's row) and the
/// scale is not 1 (an ignored scale returns a plausible row).
#[test]
fn embed_lookup_reads_the_right_row_and_scales_it_for_every_type() {
    for kind in ExtBlockType::ALL {
        let mut context =
            turbospark_gpu::MetalContext::new().expect("Metal device available on this machine");
        let (vocab, d) = (7usize, 2 * kind.block_elems());
        let mut table = Vec::new();
        for t in 0..vocab {
            table.extend_from_slice(&blocks(kind, d, 500 + t as u32));
        }
        let table_buffer = context.new_buffer_with_data(&table);
        let out = context.new_output_buffer((d * std::mem::size_of::<u16>()) as u64);
        let (token, out_scale) = (5u32, 4.0f32);
        let pass = context.begin_pass();
        encode_embed_lookup_ext(
            &mut context,
            &pass,
            kind,
            (&table_buffer, 0),
            (&out, 0),
            token,
            d as u32,
            out_scale,
        )
        .expect("GPU dispatch succeeds");
        pass.commit_and_wait();

        let row_bytes = kind.row_bytes(d);
        let want: Vec<f32> = decode(
            kind,
            &table[token as usize * row_bytes..(token as usize + 1) * row_bytes],
            d,
        )
        .iter()
        .map(|v| v * out_scale)
        .collect();
        let got: Vec<f16> = {
            let ptr = out.contents() as *const u16;
            let bits = unsafe { std::slice::from_raw_parts(ptr, d) };
            bits.iter().map(|&b| f16::from_bits(b)).collect()
        };
        assert_matches(&format!("{kind:?} embed row 5, scale 4"), &got, &want, d);
        let peak = want.iter().fold(0f32, |m, &v| m.max(v.abs()));
        assert!(
            peak > 0.5,
            "{kind:?}: row too small to prove the scale: {peak}"
        );
    }
}

#[test]
fn row_bytes_counts_whole_blocks() {
    assert_eq!(ExtBlockType::Q1_0.row_bytes(256), 36);
    assert_eq!(ExtBlockType::Tq2_0.row_bytes(512), 132);
    assert_eq!(ExtBlockType::Tq1_0.row_bytes(512), 108);
    assert_eq!(ExtBlockType::Nvfp4.row_bytes(128), 72);
    assert_eq!(ExtBlockType::Q2_0.row_bytes(128), 36);
    assert_eq!(ExtBlockType::Mxfp4.row_bytes(64), 34);
    assert_eq!(ExtBlockType::Iq1Xs.row_bytes(512), 92);
    assert_eq!(ExtBlockType::Iq1Xxs.row_bytes(512), 84);
    assert_eq!(ExtBlockType::Iq1Xxxs.row_bytes(512), 76);
}

#[test]
#[should_panic(expected = "whole number")]
fn a_partial_block_is_refused() {
    let _ = ExtBlockType::Q1_0.row_bytes(64);
}
