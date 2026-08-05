//! FP32 causal attention reference. Ported from
//! `Support/Reference/Attention/Attention.swift`.
//!
//! Materializes the full attention row per Q head rather than the tiled
//! online-softmax the GPU kernel uses; only the observable output is a
//! contract, not the accumulation order.

/// Q layout: `[num_q_heads, head_dim]`.
/// K, V layout: `[seq_len, num_kv_heads, head_dim]` (V may alias K).
/// Output: `[num_q_heads, head_dim]`.
///
/// `window`, when `Some(w)` and `seq_len > w`, restricts attention to the
/// last `w` key/value positions (sliding-window attention).
#[allow(clippy::too_many_arguments)]
pub fn causal_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    seq_len: usize,
    window: Option<usize>,
    scale: Option<f32>,
) -> Vec<f32> {
    assert!(
        num_q_heads % num_kv_heads == 0,
        "num_q_heads must be a multiple of num_kv_heads"
    );
    assert_eq!(q.len(), num_q_heads * head_dim);
    assert_eq!(k.len(), seq_len * num_kv_heads * head_dim);
    assert_eq!(v.len(), seq_len * num_kv_heads * head_dim);

    let group_size = num_q_heads / num_kv_heads;
    let scale = scale.unwrap_or(1.0 / (head_dim as f32).sqrt());
    let kv_start = match window {
        Some(w) if seq_len > w => seq_len - w,
        _ => 0,
    };

    let mut out = vec![0f32; num_q_heads * head_dim];

    for qh in 0..num_q_heads {
        let kv_head = qh / group_size;
        let q_base = qh * head_dim;
        let q_vec = &q[q_base..q_base + head_dim];

        let mut scores: Vec<f32> = (kv_start..seq_len)
            .map(|p| {
                let k_base = (p * num_kv_heads + kv_head) * head_dim;
                let dot: f32 = q_vec
                    .iter()
                    .zip(&k[k_base..k_base + head_dim])
                    .map(|(a, b)| a * b)
                    .sum();
                dot * scale
            })
            .collect();

        let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for s in scores.iter_mut() {
            *s = (*s - mx).exp();
        }
        let sum: f32 = scores.iter().sum();
        let inv_sum = 1.0 / sum;
        for s in scores.iter_mut() {
            *s *= inv_sum;
        }

        for d in 0..head_dim {
            let stride = num_kv_heads * head_dim;
            let v_column_start = (kv_start * num_kv_heads + kv_head) * head_dim + d;
            let acc: f32 = scores
                .iter()
                .enumerate()
                .map(|(i, p)| p * v[v_column_start + i * stride])
                .sum();
            out[qh * head_dim + d] = acc;
        }
    }
    out
}
