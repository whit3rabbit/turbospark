#![cfg(target_os = "macos")]
//! Sliding-window decode-attention parity: the vendored
//! `attention_decode_partial` dispatched with a nonzero `kv_start`
//! (attending `[seq_len - window, seq_len)`) must match
//! `mrefrust_compute::causal_attention`'s `window: Some(w)` reference on
//! real hardware.

use half::f16;
use mrefrust_gpu::{AttentionScratch, MetalContext};

fn to_le(v: &[f16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    out
}

#[test]
fn swa_kv_start_matches_cpu_window_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let head_dim = 32u32;
    let num_q_heads = 4u32;
    let num_kv_heads = 2u32;
    let seq_len = 24u32;
    let window = 7u32;
    let scale = 0.125f32;
    let kv_start = seq_len - window;

    let q16: Vec<f16> = (0..(num_q_heads * head_dim) as usize)
        .map(|i| f16::from_f32(((i as f32) * 0.19).sin()))
        .collect();
    let kv_len = (seq_len * num_kv_heads * head_dim) as usize;
    let k16: Vec<f16> = (0..kv_len)
        .map(|i| f16::from_f32(((i as f32) * 0.07).cos()))
        .collect();
    let v16: Vec<f16> = (0..kv_len)
        .map(|i| f16::from_f32(((i as f32) * 0.11).sin()))
        .collect();

    let expected = mrefrust_compute::causal_attention(
        &q16.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        &k16.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        &v16.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        head_dim as usize,
        num_q_heads as usize,
        num_kv_heads as usize,
        seq_len as usize,
        Some(window as usize),
        Some(scale),
    );

    let q_buf = context.new_buffer_with_data(&to_le(&q16));
    let k_buf = context.new_buffer_with_data(&to_le(&k16));
    let v_buf = context.new_buffer_with_data(&to_le(&v16));
    let out_buf = context.new_output_buffer((num_q_heads * head_dim) as u64 * 2);
    let scratch = AttentionScratch::new(&context, num_q_heads, head_dim);

    let pass = context.begin_pass();
    mrefrust_gpu::encode_attention_decode(
        &mut context,
        &pass,
        (&q_buf, 0),
        &k_buf,
        &v_buf,
        &scratch,
        (&out_buf, 0),
        head_dim,
        num_q_heads,
        num_kv_heads,
        seq_len,
        kv_start,
        0,
        scale,
    )
    .expect("encode");
    pass.commit_and_wait();

    let got: Vec<f32> = {
        let ptr = out_buf.contents() as *const u16;
        let bits = unsafe { std::slice::from_raw_parts(ptr, (num_q_heads * head_dim) as usize) };
        bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
    };

    for i in 0..got.len() {
        let diff = (got[i] - expected[i]).abs();
        assert!(
            diff <= 2e-3_f32.max(expected[i].abs() * 1e-2),
            "i={i}: got {} want {}",
            got[i],
            expected[i]
        );
    }
}

/// Ring-layout parity: the same window over a wrapped ring buffer
/// (`FC_ATTN_RING_CAP = 9`, so K/V row for logical position `p` lives at
/// `p % 9`) must match the CPU reference computed on the linear layout.
/// The dead ring rows are poisoned with huge values to prove the kernel
/// never reads outside the live window's slots.
#[test]
fn swa_ring_layout_matches_cpu_window_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let head_dim = 32u32;
    let num_q_heads = 4u32;
    let num_kv_heads = 2u32;
    let seq_len = 24u32;
    let window = 7u32;
    let ring_capacity = 9u32;
    let scale = 0.125f32;
    let kv_start = seq_len - window;

    let q16: Vec<f16> = (0..(num_q_heads * head_dim) as usize)
        .map(|i| f16::from_f32(((i as f32) * 0.19).sin()))
        .collect();
    let kv_len = (seq_len * num_kv_heads * head_dim) as usize;
    let k16: Vec<f16> = (0..kv_len)
        .map(|i| f16::from_f32(((i as f32) * 0.07).cos()))
        .collect();
    let v16: Vec<f16> = (0..kv_len)
        .map(|i| f16::from_f32(((i as f32) * 0.11).sin()))
        .collect();

    let expected = mrefrust_compute::causal_attention(
        &q16.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        &k16.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        &v16.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        head_dim as usize,
        num_q_heads as usize,
        num_kv_heads as usize,
        seq_len as usize,
        Some(window as usize),
        Some(scale),
    );

    // Scatter the live rows into ring layout; poison everything else.
    let row = (num_kv_heads * head_dim) as usize;
    let ring_len = ring_capacity as usize * row;
    let poison = f16::from_f32(1.0e4);
    let mut k_ring = vec![poison; ring_len];
    let mut v_ring = vec![poison; ring_len];
    for p in kv_start..seq_len {
        let src = p as usize * row;
        let dst = (p % ring_capacity) as usize * row;
        k_ring[dst..dst + row].copy_from_slice(&k16[src..src + row]);
        v_ring[dst..dst + row].copy_from_slice(&v16[src..src + row]);
    }

    let q_buf = context.new_buffer_with_data(&to_le(&q16));
    let k_buf = context.new_buffer_with_data(&to_le(&k_ring));
    let v_buf = context.new_buffer_with_data(&to_le(&v_ring));
    let out_buf = context.new_output_buffer((num_q_heads * head_dim) as u64 * 2);
    let scratch = AttentionScratch::new(&context, num_q_heads, head_dim);

    let pass = context.begin_pass();
    mrefrust_gpu::encode_attention_decode(
        &mut context,
        &pass,
        (&q_buf, 0),
        &k_buf,
        &v_buf,
        &scratch,
        (&out_buf, 0),
        head_dim,
        num_q_heads,
        num_kv_heads,
        seq_len,
        kv_start,
        ring_capacity,
        scale,
    )
    .expect("encode");
    pass.commit_and_wait();

    let got: Vec<f32> = {
        let ptr = out_buf.contents() as *const u16;
        let bits = unsafe { std::slice::from_raw_parts(ptr, (num_q_heads * head_dim) as usize) };
        bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
    };

    for i in 0..got.len() {
        let diff = (got[i] - expected[i]).abs();
        assert!(
            diff <= 2e-3_f32.max(expected[i].abs() * 1e-2),
            "i={i}: got {} want {}",
            got[i],
            expected[i]
        );
    }
}
