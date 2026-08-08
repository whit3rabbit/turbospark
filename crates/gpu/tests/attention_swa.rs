#![cfg(target_os = "macos")]
//! Sliding-window decode-attention parity: the vendored
//! `attention_decode_partial` dispatched with a nonzero `kv_start`
//! (attending `[seq_len - window, seq_len)`) must match
//! `turbospark_compute::causal_attention`'s `window: Some(w)` reference on
//! real hardware, in both linear and ring KV layout.

use half::f16;
use turbospark_gpu::{AttentionScratch, MetalContext};

fn to_le(v: &[f16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    out
}

/// Runs one windowed decode-attention dispatch against the CPU reference.
///
/// `ring_capacity == 0` keeps K/V in linear layout; nonzero scatters the
/// live rows to `p % ring_capacity` and POISONS every other ring row with
/// a huge value, so a kernel that reads outside the live window's slots
/// blows the tolerance instead of quietly passing.
fn assert_windowed_attention_matches_cpu(
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    window: u32,
    ring_capacity: u32,
) {
    let mut context = MetalContext::new().expect("Metal device");
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

    let expected = turbospark_compute::causal_attention(
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

    let (k_src, v_src) = if ring_capacity == 0 {
        (k16.clone(), v16.clone())
    } else {
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
        (k_ring, v_ring)
    };

    let q_buf = context.new_buffer_with_data(&to_le(&q16));
    let k_buf = context.new_buffer_with_data(&to_le(&k_src));
    let v_buf = context.new_buffer_with_data(&to_le(&v_src));
    let out_buf = context.new_output_buffer((num_q_heads * head_dim) as u64 * 2);
    let scratch = AttentionScratch::new(&context, num_q_heads, head_dim);

    let pass = context.begin_pass();
    turbospark_gpu::encode_attention_decode(
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

/// Linear KV layout, 2 Q heads per KV head.
#[test]
fn swa_kv_start_matches_cpu_window_reference() {
    assert_windowed_attention_matches_cpu(4, 2, 32, 24, 7, 0);
}

/// Wrapped ring buffer: the K/V row for logical position `p` lives at
/// `p % 9`.
#[test]
fn swa_ring_layout_matches_cpu_window_reference() {
    assert_windowed_attention_matches_cpu(4, 2, 32, 24, 7, 9);
}

/// A wider GQA ratio (8 Q / 2 KV is 4 Q per KV) over a wrapped ring and a
/// longer window than the cases above, so the kernel's per-Q-head
/// recomputation of the shared KV head is exercised together with ring
/// addressing rather than only on its own.
#[test]
fn wide_gqa_ratio_ring_layout_matches_cpu_window_reference() {
    assert_windowed_attention_matches_cpu(8, 2, 32, 40, 20, 24);
}

/// The cases above all have windows short enough that `chunks_for` keeps
/// the dispatch at one chunk. These two clear `MAX_CHUNKS *
/// MIN_POSITIONS_PER_CHUNK` positions, so they run the split-KV path:
/// 16 chunks across the window, the partials merged by the combine pass.
/// Without them nothing in the suite would dispatch more than one chunk.
#[test]
fn split_kv_linear_layout_matches_cpu_window_reference() {
    assert_windowed_attention_matches_cpu(4, 2, 32, 400, 400, 0);
}

#[test]
fn split_kv_ring_layout_matches_cpu_window_reference() {
    assert_windowed_attention_matches_cpu(8, 2, 32, 400, 300, 512);
}
