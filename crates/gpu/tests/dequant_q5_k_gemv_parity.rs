//! Runs `dequant_q5_k_gemv_simd` on real Metal hardware against the CPU
//! reference in `turbospark_compute::dequant_q5_k_gemv` (ROADMAP Phase M2).
//! The kernel rule: no quant kernel is trusted before this file exists and
//! passes.
//!
//! **The bytes are BUILT here rather than quantized**, unlike the Q4_K and
//! Q6_K parity files, and that is deliberate rather than a shortcut. Q5_K has
//! no `quantize_q5_k` sibling in `crates/compute`, because its layout is
//! pinned against ggml directly (`scripts/ggml_q5_k_oracle.c` has ggml produce
//! both the bytes and the expected floats). Every 176-byte pattern is a valid
//! superblock, so a builder is enough here, and the division of labour is
//! clean: the compute test says the LAYOUT is right, this file says the KERNEL
//! agrees with the CPU reference on identical bytes.
//!
//! The cases are chosen against Q5_K's own failure modes. Three stay finite
//! and plausible when wrong: ignoring the `qh` run (every quant loses its
//! fifth bit, so the row is compressed rather than broken), indexing `qh` by
//! the wrong bit (values swap between elements 32 apart), and dropping the
//! per-sub-block min (every weight turns non-negative, the Q6_K habit).
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{
    dequant_q5_k_gemv, dequant_q5_k_gemv_resident, q5_k_row_bytes, MetalContext, Q5KResidentMatrix,
    Q5_K_BLOCK_BYTES, Q5_K_BLOCK_ELEMS,
};

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

/// A run of valid Q5_K superblocks. `d` and `dmin` are small positives so the
/// reconstruction stays inside FP16 (the dynamic-range trap AGENTS.md records
/// for the IQ fixtures applies here too), and the 6-bit scale and min fields
/// are written through their real packing rather than as raw bytes, so every
/// sub-block genuinely differs.
fn blocks(n: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n % Q5_K_BLOCK_ELEMS, 0);
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut out = Vec::with_capacity(n / Q5_K_BLOCK_ELEMS * Q5_K_BLOCK_BYTES);

    for _ in 0..n / Q5_K_BLOCK_ELEMS {
        out.extend_from_slice(&f16::from_f32(0.012_5).to_bits().to_le_bytes());
        out.extend_from_slice(&f16::from_f32(0.031_25).to_bits().to_le_bytes());

        // The 12 packed bytes, written the way `q4_k_scale_min` reads them:
        // low six bits for sub-blocks 0..4, then the split form for 4..8.
        let mut packed = [0u8; 12];
        for j in 0..8usize {
            let sc = (lcg(&mut s) % 64) as u8;
            let m = (lcg(&mut s) % 64) as u8;
            if j < 4 {
                packed[j] |= sc;
                packed[j + 4] |= m;
            } else {
                packed[j + 4] = (sc & 0xF) | ((m & 0xF) << 4);
                packed[j - 4] |= (sc >> 4) << 6;
                packed[j] |= (m >> 4) << 6;
            }
        }
        out.extend_from_slice(&packed);

        for _ in 0..Q5_K_BLOCK_ELEMS / 8 {
            out.push((lcg(&mut s) & 0xFF) as u8);
        }
        for _ in 0..Q5_K_BLOCK_ELEMS / 2 {
            out.push((lcg(&mut s) & 0xFF) as u8);
        }
    }
    out
}

fn activations(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| ((lcg(&mut s) >> 8) as f32 / (1u32 << 23) as f32) - 1.0)
        .collect()
}

/// The CPU reference and the GPU are handed the SAME bytes, so there is no
/// quantization error to cancel and what is left is FP16 rounding and
/// reduction order. Anything above that is a kernel bug.
fn assert_matches(label: &str, gpu: &[f16], cpu: &[f32], n: usize) {
    assert_eq!(gpu.len(), cpu.len());
    let gpu_f32: Vec<f32> = gpu.iter().map(|v| v.to_f32()).collect();
    let err = turbospark_compute::max_abs_diff(&gpu_f32, cpu);
    let scale = cpu.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1.0);
    let bound = scale * 1e-2 + (n as f32) * 1e-4;
    assert!(err < bound, "{label}: err = {err}, bound = {bound}");
}

#[test]
fn matches_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let n = 2 * Q5_K_BLOCK_ELEMS;
    let m = 5usize; // not a multiple of 8: exercises the early-return guard
    let rows: Vec<Vec<u8>> = (0..m).map(|r| blocks(n, 7 + r as u32)).collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();

    let x_f32 = activations(n, 99);
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = turbospark_compute::dequant_q5_k_gemv(&refs, &x_f32, n);
    let gpu = dequant_q5_k_gemv(&mut context, &refs, &x_f16, n).expect("gpu gemv");
    assert_matches("q5_k gemv", &gpu, &cpu, n);
}

/// A single superblock, the smallest shape the kernel accepts, so an
/// off-by-one in the block loop has nowhere to hide behind a second block.
#[test]
fn matches_cpu_reference_on_one_superblock() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let n = Q5_K_BLOCK_ELEMS;
    let rows = [blocks(n, 3)];
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
    let x_f32 = activations(n, 5);
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = turbospark_compute::dequant_q5_k_gemv(&refs, &x_f32, n);
    let gpu = dequant_q5_k_gemv(&mut context, &refs, &x_f16, n).expect("gpu gemv");
    assert_matches("q5_k gemv, one superblock", &gpu, &cpu, n);
}

/// The offset-bound path the decode flow actually uses: the same rows sitting
/// at a non-zero offset inside one shared buffer, which is how a resident
/// weight is addressed in place.
#[test]
fn resident_matrix_matches_the_copying_path() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let n = 2 * Q5_K_BLOCK_ELEMS;
    let m = 3usize;
    let rows: Vec<Vec<u8>> = (0..m).map(|r| blocks(n, 21 + r as u32)).collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();

    // A leading pad, so `weights_offset` is exercised rather than defaulted.
    // 256 keeps the offset a multiple of the alignment a real resident index
    // hands out.
    let pad = 256usize;
    let mut shared = vec![0u8; pad];
    for row in &rows {
        shared.extend_from_slice(row);
    }
    let buffer = context.new_buffer_with_data(&shared);

    let x_f32 = activations(n, 77);
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let cpu = turbospark_compute::dequant_q5_k_gemv(&refs, &x_f32, n);
    let gpu = dequant_q5_k_gemv_resident(
        &mut context,
        &Q5KResidentMatrix {
            buffer: &buffer,
            weights_offset: pad as u64,
            rows: m,
            cols: n,
        },
        &x_f16,
    )
    .expect("resident gemv");
    assert_matches("q5_k gemv, resident", &gpu, &cpu, n);
}

/// The row stride is the one number a caller has to get right to address a
/// matrix in place, and it is not derivable from the element count without
/// the block size.
#[test]
fn row_bytes_counts_whole_superblocks() {
    assert_eq!(q5_k_row_bytes(Q5_K_BLOCK_ELEMS), Q5_K_BLOCK_BYTES);
    assert_eq!(q5_k_row_bytes(4 * Q5_K_BLOCK_ELEMS), 4 * Q5_K_BLOCK_BYTES);
}

#[test]
#[should_panic(expected = "whole number")]
fn a_partial_superblock_is_refused() {
    let _ = q5_k_row_bytes(Q5_K_BLOCK_ELEMS + 1);
}
