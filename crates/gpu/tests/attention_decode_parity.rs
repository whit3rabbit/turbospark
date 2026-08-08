//! Runs the two-pass split-KV decode attention kernels
//! (`attention_decode_partial` + `attention_decode_combine`, `num_chunks
//! == 1`) on real Metal hardware and checks them against the CPU
//! reference in `turbospark_compute::causal_attention`.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{attention_decode, MetalContext};

fn to_f16(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}

fn to_f32(v: &[f16]) -> Vec<f32> {
    v.iter().map(|x| x.to_f32()).collect()
}

#[test]
fn matches_cpu_reference_for_grouped_query_attention() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let head_dim = 8usize;
    let num_q_heads = 4usize;
    let num_kv_heads = 2usize; // GQA: 2 Q heads share each KV head.
    let seq_len = 6usize;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q_f32: Vec<f32> = (0..num_q_heads * head_dim)
        .map(|i| ((i as f32) - (num_q_heads * head_dim) as f32 / 2.0) * 0.05)
        .collect();
    let k_f32: Vec<f32> = (0..seq_len * num_kv_heads * head_dim)
        .map(|i| ((i as f32 * 1.3).sin()) * 0.3)
        .collect();
    let v_f32: Vec<f32> = (0..seq_len * num_kv_heads * head_dim)
        .map(|i| ((i as f32 * 0.7).cos()) * 0.3)
        .collect();

    let cpu = turbospark_compute::causal_attention(
        &q_f32,
        &k_f32,
        &v_f32,
        head_dim,
        num_q_heads,
        num_kv_heads,
        seq_len,
        None,
        Some(scale),
    );

    let gpu = attention_decode(
        &mut context,
        &to_f16(&q_f32),
        &to_f16(&k_f32),
        &to_f16(&v_f32),
        head_dim as u32,
        num_q_heads as u32,
        num_kv_heads as u32,
        seq_len as u32,
        scale,
    )
    .expect("GPU dispatch succeeds");

    assert_eq!(gpu.len(), cpu.len());
    let err = turbospark_compute::max_abs_diff(&to_f32(&gpu), &cpu);
    // FP16 accumulation noise only (both sides use the same FP32 online-
    // softmax algorithm); this is a small, non-quantized problem.
    assert!(err < 0.02, "err = {err}");
}

/// A wider GQA ratio (8 Q / 2 KV is 4 Q per KV, versus 2 in the case
/// above) over a much longer history, so the per-Q-head recomputation of
/// the shared KV head is exercised at a ratio the other cases never
/// reach.
#[test]
fn matches_cpu_reference_for_wide_gqa_ratio() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let head_dim = 64usize;
    let num_q_heads = 8usize;
    let num_kv_heads = 2usize;
    let seq_len = 40usize;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q_f32: Vec<f32> = (0..num_q_heads * head_dim)
        .map(|i| ((i as f32) * 0.031).sin() * 0.6)
        .collect();
    let k_f32: Vec<f32> = (0..seq_len * num_kv_heads * head_dim)
        .map(|i| ((i as f32 * 1.3).sin()) * 0.3)
        .collect();
    // Offset well away from zero, on purpose. V values that average to
    // ~0 make a test like this vacuous: attention is a weighted average
    // of V rows, so the expected output sits near zero too, and a kernel
    // that writes NOTHING (leaving fresh zeroed scratch for the combine
    // pass) still passes. Found by mutation testing, where a zero-mean V
    // hid a dispatch that returned without writing.
    let v_f32: Vec<f32> = (0..seq_len * num_kv_heads * head_dim)
        .map(|i| 0.8 + ((i as f32 * 0.7).cos()) * 0.3)
        .collect();

    let cpu = turbospark_compute::causal_attention(
        &q_f32,
        &k_f32,
        &v_f32,
        head_dim,
        num_q_heads,
        num_kv_heads,
        seq_len,
        None,
        Some(scale),
    );
    let gpu = attention_decode(
        &mut context,
        &to_f16(&q_f32),
        &to_f16(&k_f32),
        &to_f16(&v_f32),
        head_dim as u32,
        num_q_heads as u32,
        num_kv_heads as u32,
        seq_len as u32,
        scale,
    )
    .expect("GPU dispatch succeeds");

    assert_eq!(gpu.len(), cpu.len());
    let err = turbospark_compute::max_abs_diff(&to_f32(&gpu), &cpu);
    assert!(err < 0.02, "err = {err}");
}

#[test]
fn matches_cpu_reference_for_a_single_kv_head_at_position_zero() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let head_dim = 4usize;
    let num_heads = 1usize;
    let seq_len = 1usize;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q_f32 = vec![0.1f32, -0.2, 0.3, -0.4];
    let k_f32 = vec![0.05f32, 0.1, -0.1, 0.2];
    let v_f32 = vec![1.0f32, 2.0, 3.0, 4.0];

    let cpu = turbospark_compute::causal_attention(
        &q_f32,
        &k_f32,
        &v_f32,
        head_dim,
        num_heads,
        num_heads,
        seq_len,
        None,
        Some(scale),
    );
    let gpu = attention_decode(
        &mut context,
        &to_f16(&q_f32),
        &to_f16(&k_f32),
        &to_f16(&v_f32),
        head_dim as u32,
        num_heads as u32,
        num_heads as u32,
        seq_len as u32,
        scale,
    )
    .unwrap();

    // Single KV position: attention collapses to exactly V (weight 1.0).
    let err = turbospark_compute::max_abs_diff(&to_f32(&gpu), &cpu);
    assert!(err < 0.01, "err = {err}");
    assert_eq!(gpu.len(), 4);
}
