//! Parity tests, on real Metal hardware, for `gdn.metal`'s eight kernels
//! against `mrefrust_compute::GdnReference`. Mirrors Swift's
//! `GDNKernelTests`: a six-step decode chain versus the CPU model, a
//! seven-row prefill chunk versus seven sequential decode steps (state and
//! conv-tail carry), the fused input projection versus the four separate
//! INT4 GEMVs it replaces (which must be BIT-identical, since greedy
//! decode output depends on it), and the short-chunk tail-carry path.
#![cfg(target_os = "macos")]

use half::f16;
use mrefrust_compute::{bf16_to_f32, f32_to_bf16, quantize_int4_affine, GdnDims, GdnReference};
use mrefrust_gpu::{
    encode_dequant_int4_gemv_resident, encode_gdn_conv_decode, encode_gdn_conv_prefill,
    encode_gdn_conv_tail_update, encode_gdn_delta_decode, encode_gdn_delta_prefill,
    encode_gdn_gated_norm, encode_gdn_in_proj, encode_gdn_qk_norm, read_buffer_f16,
    read_f32_buffer, write_buffer_bytes, GdnShape, Int4ResidentMatrix, MetalBuffer, MetalContext,
};

const HK: usize = 2;
const HV: usize = 4;
const DK: usize = 32;
const DV: usize = 32;
const K: usize = 4;

fn dims() -> GdnDims {
    GdnDims {
        num_k_heads: HK,
        num_v_heads: HV,
        key_head_dim: DK,
        value_head_dim: DV,
        conv_kernel_size: K,
    }
}

fn shape() -> GdnShape {
    GdnShape {
        num_k_heads: HK as u32,
        num_v_heads: HV as u32,
        key_head_dim: DK as u32,
        value_head_dim: DV as u32,
        conv_kernel_size: K as u32,
    }
}

fn deterministic(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (((state % 2000) as f32 / 1000.0) - 1.0) * scale
        })
        .collect()
}

/// Weights the GPU reads as BF16: the reference must see the ROUNDED
/// values, not the f32 originals, or the comparison measures rounding.
fn bf16_pair(seed: u64, n: usize, scale: f32, center: f32) -> (Vec<u16>, Vec<f32>) {
    let bits: Vec<u16> = deterministic(seed, n, scale)
        .iter()
        .map(|&v| f32_to_bf16(center + v))
        .collect();
    let values = bits.iter().map(|&b| bf16_to_f32(b)).collect();
    (bits, values)
}

/// Activations the GPU reads as FP16, same reasoning.
fn f16_pair(seed: u64, n: usize, scale: f32) -> (Vec<f16>, Vec<f32>) {
    let halves: Vec<f16> = deterministic(seed, n, scale)
        .iter()
        .map(|&v| f16::from_f32(v))
        .collect();
    let values = halves.iter().map(|h| h.to_f32()).collect();
    (halves, values)
}

fn half_bytes(v: &[f16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn zeroed(context: &MetalContext, bytes: usize) -> MetalBuffer {
    let buffer = context.new_output_buffer(bytes as u64);
    write_buffer_bytes(&buffer, 0, &vec![0u8; bytes]);
    buffer
}

/// One layer's fixed weights, on both sides of the comparison.
struct Weights {
    conv_bits: Vec<u16>,
    conv: Vec<f32>,
    a_log_bits: Vec<u16>,
    a_log: Vec<f32>,
    dt_bias_bits: Vec<u16>,
    dt_bias: Vec<f32>,
    norm_bits: Vec<u16>,
    norm: Vec<f32>,
}

impl Weights {
    fn new(seed: u64) -> Self {
        let c = dims().qkv_dim();
        let (conv_bits, conv) = bf16_pair(seed, c * K, 0.4, 0.0);
        let (a_log_bits, a_log) = bf16_pair(seed + 1, HV, 1.25, 0.25);
        let (dt_bias_bits, dt_bias) = bf16_pair(seed + 2, HV, 0.5, 0.0);
        let (norm_bits, norm) = bf16_pair(seed + 3, DV, 0.5, 1.0);
        Self {
            conv_bits,
            conv,
            a_log_bits,
            a_log,
            dt_bias_bits,
            dt_bias,
            norm_bits,
            norm,
        }
    }

    fn reference(&self) -> GdnReference {
        GdnReference::new(dims(), &self.conv, &self.a_log, &self.dt_bias, &self.norm)
    }
}

/// Per-row inputs: the raw fused projection output plus the a/b/z gates.
struct Rows {
    qkv: Vec<(Vec<f16>, Vec<f32>)>,
    a: Vec<(Vec<f16>, Vec<f32>)>,
    b: Vec<(Vec<f16>, Vec<f32>)>,
    z: Vec<(Vec<f16>, Vec<f32>)>,
}

impl Rows {
    fn new(rows: usize, seed: u64) -> Self {
        let c = dims().qkv_dim();
        let vd = dims().value_dim();
        let per = |base: u64, n: usize| {
            (0..rows)
                .map(|r| f16_pair(base + r as u64 * 17, n, 1.0))
                .collect::<Vec<_>>()
        };
        Self {
            qkv: per(seed, c),
            a: per(seed + 5_000, HV),
            b: per(seed + 6_000, HV),
            z: per(seed + 7_000, vd),
        }
    }

    fn flat(pairs: &[(Vec<f16>, Vec<f32>)]) -> Vec<f16> {
        pairs.iter().flat_map(|(h, _)| h.iter().copied()).collect()
    }
}

/// Buffers a decode chain reuses across steps: the persistent tail/state
/// plus the per-step scratch.
struct DecodeBuffers {
    tail: MetalBuffer,
    state: MetalBuffer,
    conv_w: MetalBuffer,
    a_log: MetalBuffer,
    dt_bias: MetalBuffer,
    norm_w: MetalBuffer,
    conv_out: MetalBuffer,
    y: MetalBuffer,
    out: MetalBuffer,
}

impl DecodeBuffers {
    fn new(context: &MetalContext, w: &Weights) -> Self {
        let c = dims().qkv_dim();
        let vd = dims().value_dim();
        Self {
            tail: zeroed(context, (K - 1) * c * 2),
            state: zeroed(context, HV * DV * DK * 4),
            conv_w: context.new_buffer_with_data(&u16_bytes(&w.conv_bits)),
            a_log: context.new_buffer_with_data(&u16_bytes(&w.a_log_bits)),
            dt_bias: context.new_buffer_with_data(&u16_bytes(&w.dt_bias_bits)),
            norm_w: context.new_buffer_with_data(&u16_bytes(&w.norm_bits)),
            conv_out: zeroed(context, c * 2),
            y: zeroed(context, vd * 2),
            out: zeroed(context, vd * 2),
        }
    }
}

/// One GPU decode step through the whole chain; returns the gated output.
fn gpu_decode_step(
    context: &mut MetalContext,
    bufs: &DecodeBuffers,
    rows: &Rows,
    row: usize,
) -> Vec<f32> {
    let vd = dims().value_dim();
    let qkv = context.new_buffer_with_data(&half_bytes(&rows.qkv[row].0));
    let a = context.new_buffer_with_data(&half_bytes(&rows.a[row].0));
    let b = context.new_buffer_with_data(&half_bytes(&rows.b[row].0));
    let z = context.new_buffer_with_data(&half_bytes(&rows.z[row].0));

    let pass = context.begin_pass();
    encode_gdn_conv_decode(
        context,
        &pass,
        shape(),
        (&bufs.tail, 0),
        (&qkv, 0),
        (&bufs.conv_w, 0),
        (&bufs.conv_out, 0),
    )
    .expect("conv");
    encode_gdn_qk_norm(context, &pass, shape(), (&bufs.conv_out, 0), 1).expect("qk norm");
    encode_gdn_delta_decode(
        context,
        &pass,
        shape(),
        (&bufs.conv_out, 0),
        (&a, 0),
        (&b, 0),
        (&bufs.a_log, 0),
        (&bufs.dt_bias, 0),
        &bufs.state,
        (&bufs.y, 0),
    )
    .expect("delta");
    encode_gdn_gated_norm(
        context,
        &pass,
        shape(),
        (&bufs.y, 0),
        (&z, 0),
        (&bufs.norm_w, 0),
        (&bufs.out, 0),
        1,
    )
    .expect("gated norm");
    pass.commit_and_wait();
    read_buffer_f16(&bufs.out, 0, vd)
        .iter()
        .map(|h| h.to_f32())
        .collect()
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let tolerance = (w.abs() * 4e-2).max(2e-2);
        assert!(
            (g - w).abs() <= tolerance,
            "{what} element {i}: got {g}, want {w}"
        );
    }
}

#[test]
fn decode_chain_matches_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let rows_count = 6;
    let w = Weights::new(0x51D);
    let rows = Rows::new(rows_count, 0x51D_0000);
    let bufs = DecodeBuffers::new(&context, &w);
    let mut reference = w.reference();

    for row in 0..rows_count {
        let got = gpu_decode_step(&mut context, &bufs, &rows, row);
        let want = reference.step(
            &rows.qkv[row].1,
            &rows.a[row].1,
            &rows.b[row].1,
            &rows.z[row].1,
        );
        assert_close(&got, &want, &format!("row {row}"));
    }

    // The FP32 recurrent state is the thing that carries between tokens;
    // an output-only check would miss a state that is drifting.
    let state = read_f32_buffer(&bufs.state, HV * DV * DK);
    let max_err = state
        .iter()
        .zip(&reference.state)
        .map(|(&g, &w)| (g - w).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err <= 5e-2, "state divergence {max_err}");
}

#[test]
fn prefill_chunk_matches_sequential_decode() {
    let mut context = MetalContext::new().expect("Metal device");
    let rows_count = 7;
    let c = dims().qkv_dim();
    let vd = dims().value_dim();
    let w = Weights::new(0xBEEF);
    let rows = Rows::new(rows_count, 0xBEEF_0000);

    let decode_bufs = DecodeBuffers::new(&context, &w);
    let decode: Vec<Vec<f32>> = (0..rows_count)
        .map(|row| gpu_decode_step(&mut context, &decode_bufs, &rows, row))
        .collect();

    let tail = zeroed(&context, (K - 1) * c * 2);
    let state = zeroed(&context, HV * DV * DK * 4);
    let qkv_rows = context.new_buffer_with_data(&half_bytes(&Rows::flat(&rows.qkv)));
    let a_rows = context.new_buffer_with_data(&half_bytes(&Rows::flat(&rows.a)));
    let b_rows = context.new_buffer_with_data(&half_bytes(&Rows::flat(&rows.b)));
    let z_rows = context.new_buffer_with_data(&half_bytes(&Rows::flat(&rows.z)));
    let conv_out = zeroed(&context, rows_count * c * 2);
    let y = zeroed(&context, rows_count * vd * 2);
    let out = zeroed(&context, rows_count * vd * 2);
    let n = rows_count as u32;

    let pass = context.begin_pass();
    encode_gdn_conv_prefill(
        &mut context,
        &pass,
        shape(),
        (&tail, 0),
        (&qkv_rows, 0),
        (&decode_bufs.conv_w, 0),
        (&conv_out, 0),
        n,
    )
    .expect("conv prefill");
    encode_gdn_conv_tail_update(&mut context, &pass, shape(), (&tail, 0), (&qkv_rows, 0), n)
        .expect("tail update");
    encode_gdn_qk_norm(&mut context, &pass, shape(), (&conv_out, 0), n).expect("qk norm");
    encode_gdn_delta_prefill(
        &mut context,
        &pass,
        shape(),
        (&conv_out, 0),
        (&a_rows, 0),
        (&b_rows, 0),
        (&decode_bufs.a_log, 0),
        (&decode_bufs.dt_bias, 0),
        &state,
        (&y, 0),
        n,
    )
    .expect("delta prefill");
    encode_gdn_gated_norm(
        &mut context,
        &pass,
        shape(),
        (&y, 0),
        (&z_rows, 0),
        (&decode_bufs.norm_w, 0),
        (&out, 0),
        n,
    )
    .expect("gated norm");
    pass.commit_and_wait();

    let prefill: Vec<f32> = read_buffer_f16(&out, 0, rows_count * vd)
        .iter()
        .map(|h| h.to_f32())
        .collect();
    for row in 0..rows_count {
        assert_close(
            &prefill[row * vd..(row + 1) * vd],
            &decode[row],
            &format!("prefill row {row}"),
        );
    }

    // Both paths must leave the SAME carry-forward state, or a chunk
    // boundary would silently change every token after it.
    let decode_tail = read_buffer_f16(&decode_bufs.tail, 0, (K - 1) * c);
    let prefill_tail = read_buffer_f16(&tail, 0, (K - 1) * c);
    assert_eq!(decode_tail, prefill_tail, "conv tail differs");
    let decode_state = read_f32_buffer(&decode_bufs.state, HV * DV * DK);
    let prefill_state = read_f32_buffer(&state, HV * DV * DK);
    let max_err = decode_state
        .iter()
        .zip(&prefill_state)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err <= 5e-2, "state divergence {max_err}");
}

/// `T < conv_kernel_size - 1` takes `gdn_conv_tail_update`'s ordered-shift
/// branch, where one thread per channel rewrites the whole column.
#[test]
fn short_chunk_tail_carry_matches_one_long_chunk() {
    let mut context = MetalContext::new().expect("Metal device");
    let c = dims().qkv_dim();
    let rows = Rows::new(2, 0x7A11);
    let both = Rows::flat(&rows.qkv);

    let run = |context: &mut MetalContext, chunks: &[&[f16]]| -> Vec<f16> {
        let tail = zeroed(context, (K - 1) * c * 2);
        for chunk in chunks {
            let buffer = context.new_buffer_with_data(&half_bytes(chunk));
            let pass = context.begin_pass();
            encode_gdn_conv_tail_update(
                context,
                &pass,
                shape(),
                (&tail, 0),
                (&buffer, 0),
                (chunk.len() / c) as u32,
            )
            .expect("tail update");
            pass.commit_and_wait();
        }
        read_buffer_f16(&tail, 0, (K - 1) * c)
    };

    let split = run(&mut context, &[&both[..c], &both[c..]]);
    let joined = run(&mut context, &[&both]);
    assert_eq!(split, joined, "tail differs between 1+1 and 2 rows");
}

/// A packed INT4 projection laid out the way the repacker does:
/// `[pad | weights | scales | biases]`, so the weights offset is 2-byte but
/// not necessarily 4-byte aligned.
fn packed_projection(
    context: &MetalContext,
    rows: usize,
    n: usize,
    pad: usize,
    seed: u64,
) -> (MetalBuffer, usize, usize, usize) {
    let mut weights = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    for r in 0..rows {
        let q = quantize_int4_affine(&deterministic(seed + r as u64 * 97 + 1, n, 0.5));
        weights.extend_from_slice(&q.packed);
        scales.extend_from_slice(&q.scales);
        biases.extend_from_slice(&q.biases);
    }
    let weights_offset = pad;
    let scale_offset = weights_offset + weights.len();
    let bias_offset = scale_offset + scales.len() * 2;
    let mut bytes = vec![0u8; bias_offset + biases.len() * 2];
    bytes[weights_offset..weights_offset + weights.len()].copy_from_slice(&weights);
    bytes[scale_offset..scale_offset + scales.len() * 2].copy_from_slice(&u16_bytes(&scales));
    bytes[bias_offset..bias_offset + biases.len() * 2].copy_from_slice(&u16_bytes(&biases));
    (
        context.new_buffer_with_data(&bytes),
        weights_offset,
        scale_offset,
        bias_offset,
    )
}

fn expect_fused_in_proj_matches_separate(hidden: usize, pad: usize, seed: u64) {
    let mut context = MetalContext::new().expect("Metal device");
    let qkv_rows = dims().qkv_dim();
    let z_rows = dims().value_dim();
    let ab_rows = HV;

    let projections: Vec<(MetalBuffer, usize, usize, usize, usize)> =
        [(qkv_rows, 0u64), (z_rows, 1), (ab_rows, 2), (ab_rows, 3)]
            .iter()
            .map(|&(rows, i)| {
                let (buffer, w, s, b) =
                    packed_projection(&context, rows, hidden, pad, seed + i * 1_000);
                (buffer, w, s, b, rows)
            })
            .collect();
    let views: Vec<Int4ResidentMatrix<'_>> = projections
        .iter()
        .map(|(buffer, w, s, b, rows)| Int4ResidentMatrix {
            buffer,
            weights_offset: *w as u64,
            scales_offset: *s as u64,
            biases_offset: *b as u64,
            rows: *rows,
            cols: hidden,
        })
        .collect();

    let (x_halfs, _) = f16_pair(seed + 9_999, hidden, 1.0);
    let x = context.new_buffer_with_data(&half_bytes(&x_halfs));
    let reference: Vec<MetalBuffer> = views.iter().map(|v| zeroed(&context, v.rows * 2)).collect();
    let fused: Vec<MetalBuffer> = views.iter().map(|v| zeroed(&context, v.rows * 2)).collect();

    let pass = context.begin_pass();
    for (view, out) in views.iter().zip(&reference) {
        encode_dequant_int4_gemv_resident(&mut context, &pass, view, (&x, 0), (out, 0))
            .expect("separate gemv");
    }
    encode_gdn_in_proj(
        &mut context,
        &pass,
        &views[0],
        &views[1],
        &views[2],
        &views[3],
        (&x, 0),
        (&fused[0], 0),
        (&fused[1], 0),
        (&fused[2], 0),
        (&fused[3], 0),
    )
    .expect("fused in_proj");
    pass.commit_and_wait();

    for (i, (view, (want, got))) in views.iter().zip(reference.iter().zip(&fused)).enumerate() {
        assert_eq!(
            read_buffer_f16(want, 0, view.rows),
            read_buffer_f16(got, 0, view.rows),
            "projection {i} is not bit-identical to its separate GEMV"
        );
    }
}

#[test]
fn fused_in_proj_is_bit_identical_to_four_separate_gemvs() {
    expect_fused_in_proj_matches_separate(128, 0, 0x1D_0001);
}

/// 2-byte-but-not-4-byte-aligned weight offsets (the repacker's only
/// guarantee) with a row total that is not a multiple of the 8 rows per
/// threadgroup, so the trailing threadgroup runs partly out of range.
#[test]
fn fused_in_proj_handles_odd_offsets_and_ragged_rows() {
    expect_fused_in_proj_matches_separate(192, 2, 0x1D_0003);
}
