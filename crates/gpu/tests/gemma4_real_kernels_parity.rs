//! Parity tests, on real Metal hardware, for the kernels the real-Gemma-4
//! learned-weight decode path adds: per-head scaled/no-scale RMSNorm, the
//! offset-bound INT8 GEMV, the INT8 router GEMV with its BF16 effective
//! scale, and the port-local `scalar_mul_fp16`. Each is checked against
//! the matching `mrefrust_compute` reference math.
#![cfg(target_os = "macos")]

use half::f16;
use mrefrust_compute::{bf16_to_f32, f32_to_bf16, quantize_int8_affine, Tolerance};
use mrefrust_gpu::{
    dequant_int8_gemv_resident, encode_scalar_mul, read_buffer_f16, rms_norm_bf16w_perhead,
    rms_norm_no_scale_perhead, router_gemv_gemma4, Int8ResidentMatrix, MetalContext,
};

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

fn to_f16(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}

fn to_f32(v: &[f16]) -> Vec<f32> {
    v.iter().map(|x| x.to_f32()).collect()
}

#[test]
fn perhead_bf16w_norm_matches_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let num_heads = 4u32;
    let head_dim = 64usize;
    let x = deterministic(11, num_heads as usize * head_dim, 2.0);
    let weight: Vec<f32> = deterministic(12, head_dim, 0.5)
        .iter()
        .map(|w| 1.0 + w)
        .collect();
    let weight_bits: Vec<u16> = weight.iter().map(|&w| f32_to_bf16(w)).collect();
    let weight_bf16: Vec<f32> = weight_bits.iter().map(|&b| bf16_to_f32(b)).collect();

    let gpu = rms_norm_bf16w_perhead(&mut context, &to_f16(&x), &weight_bits, num_heads, 1e-6)
        .expect("dispatch");

    let mut cpu = Vec::new();
    for h in 0..num_heads as usize {
        let head = &x[h * head_dim..(h + 1) * head_dim];
        cpu.extend(mrefrust_compute::rms_norm(head, &weight_bf16, 1e-6));
    }
    let err = mrefrust_compute::max_abs_diff(&to_f32(&gpu), &cpu);
    assert!(err < Tolerance::FP16_REDUCTION, "err = {err}");
}

#[test]
fn perhead_no_scale_norm_matches_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let num_heads = 3u32;
    let head_dim = 64usize;
    let x = deterministic(21, num_heads as usize * head_dim, 1.5);
    let ones = vec![1.0f32; head_dim];

    let gpu =
        rms_norm_no_scale_perhead(&mut context, &to_f16(&x), num_heads, head_dim as u32, 1e-6)
            .expect("dispatch");

    let mut cpu = Vec::new();
    for h in 0..num_heads as usize {
        let head = &x[h * head_dim..(h + 1) * head_dim];
        cpu.extend(mrefrust_compute::rms_norm(head, &ones, 1e-6));
    }
    let err = mrefrust_compute::max_abs_diff(&to_f32(&gpu), &cpu);
    assert!(err < Tolerance::FP16_REDUCTION, "err = {err}");
}

#[test]
fn int8_resident_gemv_matches_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let rows = 6usize;
    let cols = 128usize;

    let mut packed = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    let mut cpu_rows = Vec::new();
    for r in 0..rows {
        let row = deterministic(100 + r as u64, cols, 1.0);
        let q = quantize_int8_affine(&row);
        packed.extend_from_slice(&q.packed);
        scales.extend_from_slice(&q.scales);
        biases.extend_from_slice(&q.biases);
        cpu_rows.push(q);
    }
    let x = deterministic(200, cols, 1.0);
    let x16 = to_f16(&x);
    let x_for_cpu = to_f32(&x16);

    // One shared buffer holding weights, then scales, then biases — the
    // resident-buffer shape, with offsets in between.
    let mut blob = packed.clone();
    let scales_off = blob.len() as u64;
    for s in &scales {
        blob.extend_from_slice(&s.to_le_bytes());
    }
    let biases_off = blob.len() as u64;
    for b in &biases {
        blob.extend_from_slice(&b.to_le_bytes());
    }
    let buffer = context.new_buffer_with_data(&blob);
    let w = Int8ResidentMatrix {
        buffer: &buffer,
        weights_offset: 0,
        scales_offset: scales_off,
        biases_offset: biases_off,
        rows,
        cols,
    };

    let gpu = dequant_int8_gemv_resident(&mut context, &w, &x16).expect("dispatch");
    let cpu = mrefrust_compute::dequant_int8_gemv(&cpu_rows, &x_for_cpu, cols);
    let err = mrefrust_compute::bounded_rel_error(&to_f32(&gpu), &cpu, 0.5);
    assert!(err < Tolerance::FP16_REDUCTION, "err = {err}");
}

#[test]
fn router_gemv_gemma4_matches_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let num_experts = 6usize;
    let d = 128usize;

    let mut w_bytes = Vec::new();
    let mut scale_bits = Vec::new();
    let mut bias_bits = Vec::new();
    let mut cpu_rows = Vec::new();
    for e in 0..num_experts {
        let row = deterministic(300 + e as u64, d, 1.0);
        let q = quantize_int8_affine(&row);
        w_bytes.extend_from_slice(&q.packed);
        scale_bits.extend_from_slice(&q.scales);
        bias_bits.extend_from_slice(&q.biases);
        cpu_rows.push(q);
    }
    let x = deterministic(400, d, 1.0);
    let x16 = to_f16(&x);
    let eff: Vec<f32> = deterministic(500, d, 0.3).iter().map(|v| 1.0 + v).collect();
    let eff_bits: Vec<u16> = eff.iter().map(|&v| f32_to_bf16(v)).collect();

    let gpu = router_gemv_gemma4(
        &mut context,
        &w_bytes,
        &scale_bits,
        &bias_bits,
        &x16,
        &eff_bits,
        num_experts as u32,
    )
    .expect("dispatch");

    // CPU reference: dequantized INT8 rows dotted with x * effective_scale
    // (both rounded the way the kernel reads them: x through FP16, the
    // effective scale through BF16).
    let scaled_x: Vec<f32> = x16
        .iter()
        .zip(eff_bits.iter())
        .map(|(x, &e)| x.to_f32() * bf16_to_f32(e))
        .collect();
    let cpu = mrefrust_compute::dequant_int8_gemv(&cpu_rows, &scaled_x, d);
    let err = mrefrust_compute::bounded_rel_error(&gpu, &cpu, 0.5);
    assert!(err < Tolerance::FP16_REDUCTION, "err = {err}");
}

#[test]
fn scalar_mul_matches_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let n = 96usize;
    let x = deterministic(600, n, 2.0);
    let x16 = to_f16(&x);
    let scalar = 0.8125f32; // exactly representable in half

    let mut bytes = Vec::with_capacity(n * 2);
    for v in &x16 {
        bytes.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    let buffer = context.new_buffer_with_data(&bytes);
    let pass = context.begin_pass();
    encode_scalar_mul(&mut context, &pass, (&buffer, 0), scalar, n as u32).expect("dispatch");
    pass.commit_and_wait();

    let gpu = read_buffer_f16(&buffer, 0, n);
    let cpu: Vec<f32> = x16
        .iter()
        .map(|v| (*v * f16::from_f32(scalar)).to_f32())
        .collect();
    let err = mrefrust_compute::max_abs_diff(&to_f32(&gpu), &cpu);
    assert!(err == 0.0, "err = {err}");
}
