//! Runs the Q4_0 / Q4_1 / Q5_0 / Q5_1 kernels in `shaders/dequant_legacy.metal`
//! on real Metal hardware against the CPU decoders in `turbospark_compute`
//! (which `crates/repack` holds to ggml's own `to_float`). The kernel rule: no
//! quant kernel is trusted before this file exists and passes.
//!
//! Like the Q5_K parity file, the bytes are BUILT here rather than quantized:
//! every byte pattern is a valid block, so a builder suffices, and the CPU
//! reference and the GPU are handed identical bytes. What is left to differ
//! is FP16 rounding and reduction order; anything above that is a kernel bug.
//!
//! The scales are signed on purpose. ggml's Q4_0 and Q5_0 scale is
//! `max / -8` (or `-16`), negative whenever the largest magnitude is positive,
//! so a kernel that only ever saw positive scales would pass here and fail on
//! half of every real file.
//!
//! The traps are the ones the shader header lists, and each leaves a finite,
//! plausible row: nibble halves read as consecutive elements, the minimum
//! subtracted instead of added, `qh` indexed by byte rather than by element,
//! and the fifth bit dropped.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{
    dequant_legacy_gemv_resident, encode_embed_lookup_legacy, LegacyBlockType,
    LegacyResidentMatrix, LEGACY_BLOCK_ELEMS,
};

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

fn f16_bytes(v: f32) -> [u8; 2] {
    f16::from_f32(v).to_bits().to_le_bytes()
}

/// A run of valid blocks of `kind` covering `n` elements. `d` and `m` stay
/// small so reconstruction stays inside FP16, with a random sign on both.
fn blocks(kind: LegacyBlockType, n: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n % LEGACY_BLOCK_ELEMS, 0);
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut out = Vec::new();
    for _ in 0..n / LEGACY_BLOCK_ELEMS {
        let sign = if lcg(&mut s) & 1 == 0 { 1.0 } else { -1.0 };
        let d = sign * (0.012_5 + (lcg(&mut s) % 8) as f32 * 0.003);
        let m = ((lcg(&mut s) % 129) as f32 - 64.0) * 0.01;
        out.extend_from_slice(&f16_bytes(d));
        let has_min = matches!(kind, LegacyBlockType::Q4_1 | LegacyBlockType::Q5_1);
        let has_qh = matches!(kind, LegacyBlockType::Q5_0 | LegacyBlockType::Q5_1);
        if has_min {
            out.extend_from_slice(&f16_bytes(m));
        }
        if has_qh {
            out.extend_from_slice(&lcg(&mut s).to_le_bytes());
        }
        for _ in 0..16 {
            out.push((lcg(&mut s) & 0xFF) as u8);
        }
        assert_eq!(out.len() % kind.block_bytes(), 0);
    }
    out
}

fn decode(kind: LegacyBlockType, bytes: &[u8], n: usize) -> Vec<f32> {
    match kind {
        LegacyBlockType::Q4_0 => turbospark_compute::dequantize_q4_0(bytes, n),
        LegacyBlockType::Q4_1 => turbospark_compute::dequantize_q4_1(bytes, n),
        LegacyBlockType::Q5_0 => turbospark_compute::dequantize_q5_0(bytes, n),
        LegacyBlockType::Q5_1 => turbospark_compute::dequantize_q5_1(bytes, n),
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

fn gemv_case(kind: LegacyBlockType, rows: usize, n: usize, seed: u32) {
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
    // The CPU side multiplies by the same f16-rounded activations.
    let x_ref: Vec<f32> = x_f16.iter().map(|v| v.to_f32()).collect();

    let want: Vec<f32> = (0..rows)
        .map(|r| {
            let w = decode(kind, &table[r * row_bytes..(r + 1) * row_bytes], n);
            w.iter().zip(&x_ref).map(|(a, b)| a * b).sum()
        })
        .collect();
    let w = LegacyResidentMatrix {
        buffer: &buffer,
        weights_offset: pad as u64,
        rows,
        cols: n,
        kind,
    };
    let got = dequant_legacy_gemv_resident(&mut context, &w, &x_f16).expect("gpu gemv");
    assert_matches(&format!("{kind:?} gemv {rows}x{n}"), &got, &want, n);
}

#[test]
fn gemv_matches_cpu_reference_for_every_type() {
    for kind in LegacyBlockType::ALL {
        // 5 rows: not a multiple of 8, so the early-return guard is exercised.
        gemv_case(kind, 5, 3 * LEGACY_BLOCK_ELEMS, 7);
        // One block, the smallest shape the kernel accepts.
        gemv_case(kind, 3, LEGACY_BLOCK_ELEMS, 11);
        // Wider than one block run and more than one threadgroup of rows.
        gemv_case(kind, 19, 8 * LEGACY_BLOCK_ELEMS, 21);
    }
}

/// The embedding lookup against the CPU decoder on identical bytes. The row is
/// not row 0 (a wrong stride returns a neighbour's row) and the scale is not 1
/// (an ignored scale returns a plausible row); every element is compared.
#[test]
fn embed_lookup_reads_the_right_row_and_scales_it_for_every_type() {
    for kind in LegacyBlockType::ALL {
        let mut context =
            turbospark_gpu::MetalContext::new().expect("Metal device available on this machine");
        let (vocab, d) = (7usize, 4 * LEGACY_BLOCK_ELEMS);
        let mut table = Vec::new();
        for t in 0..vocab {
            table.extend_from_slice(&blocks(kind, d, 500 + t as u32));
        }
        let table_buffer = context.new_buffer_with_data(&table);
        let out = context.new_output_buffer((d * std::mem::size_of::<u16>()) as u64);

        let (token, out_scale) = (5u32, 4.0f32);
        let pass = context.begin_pass();
        encode_embed_lookup_legacy(
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
    assert_eq!(LegacyBlockType::Q4_0.row_bytes(64), 36);
    assert_eq!(LegacyBlockType::Q4_1.row_bytes(64), 40);
    assert_eq!(LegacyBlockType::Q5_0.row_bytes(64), 44);
    assert_eq!(LegacyBlockType::Q5_1.row_bytes(64), 48);
}

#[test]
#[should_panic(expected = "whole number")]
fn a_partial_block_is_refused() {
    let _ = LegacyBlockType::Q4_0.row_bytes(40);
}
