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
    causal_attention_with_sinks(
        q,
        k,
        v,
        head_dim,
        num_q_heads,
        num_kv_heads,
        seq_len,
        window,
        scale,
        None,
    )
}

/// The same, with ROADMAP M5's ATTENTION SINKS: one learned logit per query
/// head that joins the softmax denominator and nothing else.
///
/// A sink has no VALUE row, so it takes probability mass away from the real
/// keys and contributes nothing to the output -- the whole of its effect is
/// that the attention weights no longer sum to one. Adding it to the
/// numerator is the plausible wrong version and would make it an ordinary
/// extra key with a zero value, which is a different function.
///
/// Reference: ggml's `ggml_compute_forward_soft_max_f32` under
/// `ggml_soft_max_add_sinks`, which does `max = MAX(max, sk[head])` and then
/// `sum += expf(sk[head] - max)`.
#[allow(clippy::too_many_arguments)]
pub fn causal_attention_with_sinks(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    seq_len: usize,
    window: Option<usize>,
    scale: Option<f32>,
    sinks: Option<&[f32]>,
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

        let sink = sinks.map(|s| s[qh]);
        let mut mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if let Some(sink) = sink {
            mx = mx.max(sink);
        }
        for s in scores.iter_mut() {
            *s = (*s - mx).exp();
        }
        let mut sum: f32 = scores.iter().sum();
        if let Some(sink) = sink {
            sum += (sink - mx).exp();
        }
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
