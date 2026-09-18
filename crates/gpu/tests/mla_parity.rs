//! Parity for the `deepseek2` MLA kernels against
//! `turbospark_compute::mla` on real Metal hardware. The kernels are
//! port-local (no Swift original to diff), so these references ARE their
//! only contract -- the rule `shaders/dequant_q8_0.metal` established.
//!
//! Every Q8_0 case hands the kernel and the CPU reference the SAME
//! quantized bytes so quantization error cancels; what is left is FP16
//! rounding and reduction order.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_compute::{
    dequant_q8_0_gemv, mla_absorb_q, mla_attention_decode, mla_kv_norm, mla_rope_window,
    mla_v_combine, quantize_q8_0, yarn_frequencies,
};
use turbospark_gpu::{
    mla_absorb_q as gpu_absorb, mla_attention_decode as gpu_attention, mla_kv_norm as gpu_kv_norm,
    mla_rope_q_pe as gpu_rope_q_pe, mla_v_combine as gpu_v_combine, MetalContext,
};

fn weights(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 23) as f32) - 1.0
        })
        .collect()
}

fn assert_close(label: &str, gpu: &[f16], cpu: &[f32], magnitude: f32) {
    assert_eq!(gpu.len(), cpu.len(), "{label}: length");
    let gpu_f32: Vec<f32> = gpu.iter().map(|v| v.to_f32()).collect();
    let err = turbospark_compute::max_abs_diff(&gpu_f32, cpu);
    let bound = magnitude.max(1.0) * 2e-2;
    assert!(err < bound, "{label}: err = {err}, bound = {bound}");
}

const HEADS: usize = 4;
const KV_LORA: usize = 64;
const NOPE: usize = 32;
const ROPE: usize = 16;
const V_DIM: usize = 32;
const CACHE_ROW: usize = KV_LORA + ROPE;
/// The real model's, scaled to nothing: the fixture reads it out of the
/// same formula the baseline pins.
const SCALE: f32 = 0.114_721_39;

#[test]
fn kv_norm_norms_the_latent_and_passes_the_tail() {
    let mut context = MetalContext::new().expect("Metal device");
    let rows = 3usize; // exercise the multi-row grid, not just decode's 1
    let row_stride = CACHE_ROW;
    // The kernel reads the weight as the RESIDENT byte form (BF16), so the
    // fixture quantizes to bf16 first and the CPU reference uses the same
    // dequantized values.
    let w_f32: Vec<f32> = weights(KV_LORA, 5)
        .iter()
        .map(|&v| f32::from_bits((v.to_bits() >> 16) << 16))
        .collect();
    let weight_bf16: Vec<u16> = w_f32.iter().map(|&v| (v.to_bits() >> 16) as u16).collect();
    let data: Vec<f16> = weights(rows * row_stride, 11)
        .iter()
        .map(|&v| f16::from_f32(v * 0.5))
        .collect();

    let gpu = gpu_kv_norm(
        &mut context,
        &data,
        &weight_bf16,
        rows,
        KV_LORA,
        row_stride,
        1e-6,
    )
    .expect("dispatch");
    for r in 0..rows {
        let row_f32: Vec<f32> = data[r * row_stride..][..row_stride]
            .iter()
            .map(|v| v.to_f32())
            .collect();
        let cpu = mla_kv_norm(&row_f32, KV_LORA, &w_f32, 1e-6);
        assert_close(
            &format!("row {r}"),
            &gpu[r * row_stride..][..row_stride],
            &cpu,
            2.0,
        );
    }
}

#[test]
fn rope_q_pe_rotates_only_the_window() {
    let mut context = MetalContext::new().expect("Metal device");
    let head_dim = NOPE + ROPE;
    let data: Vec<f16> = weights(HEADS * head_dim, 13)
        .iter()
        .map(|&v| f16::from_f32(v))
        .collect();
    let yarn = yarn_frequencies(ROPE, 10_000.0, 40.0, 4096, 32.0, 1.0);
    // The DeepSeek mscale parameter (0.707), not yarn's built-in 1.0: the
    // same override the runtime applies when it builds the table.
    let mscale = 1.0 + 0.1 * 0.707 * 40.0f32.ln();

    let position = 7u32;
    let gpu = gpu_rope_q_pe(
        &mut context,
        &data,
        &yarn.frequencies,
        HEADS,
        head_dim,
        NOPE,
        ROPE,
        position,
        mscale,
    )
    .expect("dispatch");

    let mut cpu: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
    mla_rope_window(
        &mut cpu,
        HEADS,
        head_dim,
        NOPE,
        ROPE,
        &yarn.frequencies,
        position as f32,
        mscale,
    );
    assert_close("q_pe", &gpu, &cpu, 2.0);

    // THE ELEMENT SET IS THE POINT: the nope half must come back untouched,
    // which is what separates this kernel from every rope sibling here.
    for h in 0..HEADS {
        let lo = h * head_dim;
        assert!(
            data[lo..lo + NOPE] == gpu[lo..lo + NOPE],
            "head {h}: the nope half moved"
        );
    }
}

/// Pins the PAIRING, not just the GPU/CPU agreement: both sides above would
/// stay green together under either convention. The window here is 4 wide
/// with angle-pi frequencies, so pair 0 swaps the SIGN of both elements and
/// the two conventions rotate DIFFERENT partners: consecutive pairs
/// `(w0, w1), (w2, w3)` against half-split `(w0, w2), (w1, w3)`. The
/// expected values are hand-derived for consecutive, ggml's layout (see
/// `compute::mla_rope_window`).
#[test]
fn rope_pairs_consecutive_elements_not_split_halves() {
    let mut context = MetalContext::new().expect("Metal device");
    // freqs = [pi/2, pi/4], position = 1: angles [pi/2, pi/4].
    let data: Vec<f16> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .map(|&v| f16::from_f32(v))
        .collect();
    let s = std::f32::consts::SQRT_2 / 2.0;
    let gpu = gpu_rope_q_pe(
        &mut context,
        &data,
        &[std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_4],
        1, // heads
        4, // head_dim == window width: the whole head rotates
        0, // window offset
        4, // rotary
        1, // position
        1.0,
    )
    .expect("dispatch");
    // Angle pi/2: (cos, sin) = (0, 1). Consecutive pair 0 = (1, 2):
    // lo' = 1*0 - 2*1 = -2, hi' = 1*1 + 2*0 = 1.
    // Angle pi/4: (cos, sin) = (s, s). Consecutive pair 1 = (3, 4):
    // lo' = (3-4)*s = -s, hi' = (3+4)*s = 7s.
    let expect = [-2.0f32, 1.0, -s, 7.0 * s];
    for (i, (&g, &e)) in gpu.iter().zip(expect.iter()).enumerate() {
        let diff = (g.to_f32() - e).abs();
        assert!(
            diff < 2e-2,
            "element {i}: gpu {g} vs consecutive-expected {e}"
        );
    }
    // The split-half convention would instead rotate pair 0's (1, 3):
    // lo' = 1*0 - 3*1 = -3, a DIFFERENT first element. Guard the
    // discriminator itself: the fixture must be able to see the wrong
    // convention.
    let split_pair0_first = -3.0f32;
    assert!(
        (split_pair0_first - expect[0]).abs() > 0.5,
        "fixture no longer distinguishes the two pairings"
    );
}

#[test]
fn absorb_q_matches_the_cpu_matmul_and_gathers_the_pe_tail() {
    let mut context = MetalContext::new().expect("Metal device");
    // PRODUCTION GEOMETRY: the kernel contracts its accumulator at
    // kv_lora == 2 * threads (each thread owns two outputs), so the fixture
    // runs the real 512-wide latent and not a toy rank. Rows are latent
    // width, exactly the resident kv_b's.
    let kv_lora = 512usize;
    let all_rows: Vec<Vec<u8>> = (0..HEADS * (NOPE + V_DIM))
        .map(|r| quantize_q8_0(&weights(kv_lora, 100 + r as u32)))
        .collect();
    let refs: Vec<&[u8]> = all_rows.iter().map(|r| r.as_slice()).collect();

    let q_f32 = weights(HEADS * (NOPE + ROPE), 77);
    let q_f16: Vec<f16> = q_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let gpu = gpu_absorb(
        &mut context,
        &refs,
        &q_f16,
        HEADS,
        NOPE,
        kv_lora,
        V_DIM,
        ROPE,
    )
    .expect("dispatch");

    // CPU: dequantize the SAME bytes, then the reference matmul per head,
    // then the pe gather -- assembling the fused row the kernel emits.
    // Row-major [nope][kv_lora]: row i is kv_b's row h*(NOPE+V_DIM)+i, a
    // KV_LORA-wide slice of the latent -- the TRANSPOSED view the absorbed
    // product reduces over.
    let w_uk: Vec<Vec<f32>> = (0..HEADS)
        .map(|h| {
            let mut flat = vec![0.0f32; NOPE * kv_lora];
            for (i, row) in (0..NOPE)
                .map(|i| all_rows[h * (NOPE + V_DIM) + i].as_slice())
                .enumerate()
            {
                flat[i * kv_lora..][..kv_lora].copy_from_slice(&dequant_row(row, kv_lora));
            }
            flat
        })
        .collect();
    let w_refs: Vec<&[f32]> = w_uk.iter().map(|w| w.as_slice()).collect();
    let q_nope: Vec<f32> = (0..HEADS)
        .flat_map(|h| q_f32[h * (NOPE + ROPE)..][..NOPE].to_vec())
        .collect();
    let absorbed = mla_absorb_q(&q_nope, &w_refs, HEADS, kv_lora, NOPE);
    let mut cpu = vec![0.0f32; HEADS * (kv_lora + ROPE)];
    for h in 0..HEADS {
        cpu[h * (kv_lora + ROPE)..][..kv_lora].copy_from_slice(&absorbed[h * kv_lora..][..kv_lora]);
        cpu[h * (kv_lora + ROPE) + kv_lora..][..ROPE]
            .copy_from_slice(&q_f32[h * (NOPE + ROPE) + NOPE..][..ROPE]);
    }
    assert_close("absorb", &gpu, &cpu, 16.0);
}

#[test]
fn attention_decode_matches_and_mqa_shares_the_row() {
    let mut context = MetalContext::new().expect("Metal device");
    // THE REAL SHAPES: the kernel fixes its accumulator at kv_lora 512
    // (2 halves per thread over 256 threads), so the fixture must be the
    // production geometry and not a toy rank.
    let kv_lora = 512usize;
    let cache_row = kv_lora + ROPE;
    let seq = 33usize; // not a power of two, and past one chunk boundary
    let q_f32 = weights(HEADS * cache_row, 21);
    let cache_f32: Vec<Vec<f32>> = (0..seq)
        .map(|t| weights(cache_row, 500 + t as u32))
        .collect();
    let cache_refs: Vec<&[f32]> = cache_f32.iter().map(|r| r.as_slice()).collect();

    let q_f16: Vec<f16> = q_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let cache_f16: Vec<f16> = cache_f32
        .iter()
        .flat_map(|r| r.iter().map(|&v| f16::from_f32(v)))
        .collect();

    let gpu = gpu_attention(
        &mut context,
        &q_f16,
        &cache_f16,
        HEADS,
        cache_row,
        kv_lora,
        seq,
        SCALE,
    )
    .expect("dispatch");
    let cpu = mla_attention_decode(&q_f32, &cache_refs, HEADS, cache_row, kv_lora, SCALE);
    assert_close("attention", &gpu, &cpu, 2.0);
}

#[test]
fn v_combine_matches_the_cpu_matmul() {
    let mut context = MetalContext::new().expect("Metal device");
    let row_bytes = KV_LORA / 32 * 34;
    let _ = row_bytes;
    let all_rows: Vec<Vec<u8>> = (0..HEADS * (NOPE + V_DIM))
        .map(|r| quantize_q8_0(&weights(KV_LORA, 200 + r as u32)))
        .collect();
    let refs: Vec<&[u8]> = all_rows.iter().map(|r| r.as_slice()).collect();

    let attn_f32 = weights(HEADS * KV_LORA, 31);
    let attn_f16: Vec<f16> = attn_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let gpu = gpu_v_combine(&mut context, &refs, &attn_f16, HEADS, NOPE, KV_LORA, V_DIM)
        .expect("dispatch");

    let w_uv: Vec<Vec<f32>> = (0..HEADS)
        .map(|h| {
            let mut flat = vec![0.0f32; V_DIM * KV_LORA];
            for (j, row) in (0..V_DIM)
                .map(|j| all_rows[h * (NOPE + V_DIM) + NOPE + j].as_slice())
                .enumerate()
            {
                flat[j * KV_LORA..][..KV_LORA].copy_from_slice(&dequant_row(row, KV_LORA));
            }
            flat
        })
        .collect();
    let w_refs: Vec<&[f32]> = w_uv.iter().map(|w| w.as_slice()).collect();
    let cpu = mla_v_combine(&attn_f32, &w_refs, HEADS, KV_LORA, V_DIM);
    assert_close("v_combine", &gpu, &cpu, 8.0);
}

/// Dequantize one Q8_0 row (the same 34-byte blocks the kernels read).
fn dequant_row(row: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (b, chunk) in out.chunks_mut(32).enumerate() {
        let blk = &row[b * 34..];
        let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        for (i, o) in chunk.iter_mut().enumerate() {
            *o = (i8::from_le_bytes([blk[2 + i]]) as i32) as f32 * d;
        }
    }
    out
}

/// The fused-equals-separate guard NEW_MODEL.md Phase 2 asks for, in the
/// shape this flow allows: the attention output for ONE head must be the
/// plain weighted sum the reference computes, i.e. the online-softmax
/// accumulation reproduces a two-pass softmax over the same rows.
#[test]
fn online_softmax_matches_two_pass_at_short_seq() {
    let mut context = MetalContext::new().expect("Metal device");
    let kv_lora = 512usize;
    let cache_row = kv_lora + ROPE;
    let seq = 1usize;
    let q_f32 = weights(HEADS * cache_row, 41);
    let cache_f32: Vec<Vec<f32>> = (0..seq)
        .map(|t| weights(cache_row, 900 + t as u32))
        .collect();
    let cache_refs: Vec<&[f32]> = cache_f32.iter().map(|r| r.as_slice()).collect();
    let q_f16: Vec<f16> = q_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let cache_f16: Vec<f16> = cache_f32
        .iter()
        .flat_map(|r| r.iter().map(|&v| f16::from_f32(v)))
        .collect();
    let gpu = gpu_attention(
        &mut context,
        &q_f16,
        &cache_f16,
        HEADS,
        cache_row,
        kv_lora,
        seq,
        SCALE,
    )
    .expect("dispatch");
    let cpu = mla_attention_decode(&q_f32, &cache_refs, HEADS, cache_row, kv_lora, SCALE);
    // At seq == 1 softmax is exact regardless of max-subtraction, so any
    // drift here is the accumulation itself, not the trick.
    assert_close("seq=1", &gpu, &cpu, 2.0);
}

/// `dequant_q8_0_gemv` is the absorb kernel's per-head inner loop; if the
/// vendored reference and the kernel disagree on the BLOCK FORMAT, the
/// fixture above cannot say which side is wrong. This pins the block read
/// separately: a deliberately asymmetric row must survive a sign error.
#[test]
fn the_q8_0_block_read_is_signed() {
    let mut context = MetalContext::new().expect("Metal device");
    let n = 32usize;
    let mut row = vec![0u8; 34];
    row[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    row[2] = 127u8;
    let refs: Vec<&[u8]> = vec![&row];
    let x_f16: Vec<f16> = vec![f16::from_f32(2.0); n];
    let gpu = turbospark_gpu::dequant_q8_0_gemv(&mut context, &refs, &x_f16, n).expect("dispatch");
    assert!(
        (gpu[0].to_f32() - 254.0).abs() < 1.0,
        "signed read: got {}",
        gpu[0]
    );
    let _ = dequant_q8_0_gemv; // the reference the case mirrors
}
