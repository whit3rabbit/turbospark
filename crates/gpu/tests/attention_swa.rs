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
        scale,
        0,
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

/// The same window, but K/V live in a ring of `window + 1` rows instead of
/// a full `seq_len` buffer: the kernel must address them modulo the ring
/// capacity and land on the identical result. This is what lets
/// sliding-window layers hold `window + 1` rows instead of `max_context`.
#[test]
fn swa_ring_addressing_matches_the_linear_layout() {
    let mut context = MetalContext::new().expect("Metal device");
    let head_dim = 32u32;
    let num_q_heads = 4u32;
    let num_kv_heads = 2u32;
    let seq_len = 24u32;
    let window = 7u32;
    let ring_capacity = window + 1;
    let scale = 0.125f32;
    let kv_start = seq_len - window;
    let row = (num_kv_heads * head_dim) as usize;

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

    // Fold the last `ring_capacity` logical rows into their ring slots,
    // exactly as KvCacheManager's `position % capacity` writer does.
    let mut k_ring = vec![f16::from_f32(0.0); ring_capacity as usize * row];
    let mut v_ring = k_ring.clone();
    for p in (seq_len - ring_capacity)..seq_len {
        let slot = (p % ring_capacity) as usize;
        let src = p as usize * row;
        k_ring[slot * row..(slot + 1) * row].copy_from_slice(&k16[src..src + row]);
        v_ring[slot * row..(slot + 1) * row].copy_from_slice(&v16[src..src + row]);
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
        scale,
        ring_capacity,
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
