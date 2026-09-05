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

/// Attention over an explicit SUBSET of key/value positions: the CPU
/// reference for `qwen4_exp`'s QSA attention application
/// (`docs/QWEN4_PHASE0.md` section 5, where the indexer's output "is a
/// boolean mask ANDed onto the causal mask"). PORT-LOCAL, no Swift
/// counterpart.
///
/// `positions` lists the key/value rows the query may attend to, oldest
/// first; every entry must be `< seq_len` where `seq_len` is
/// `k.len() / (num_kv_heads * head_dim)`. Rows NOT listed contribute
/// nothing: not to the softmax denominator, not to the output. The
/// reference is deliberately a GATHER followed by [`causal_attention`] over
/// the compact sequence, so the two functions cannot disagree on the
/// attention arithmetic itself, only on which rows enter it -- and with the
/// identity list (`0..seq_len`) the two are the same function, which
/// [`tests::identity_positions_reproduce_causal_attention_exactly`] pins.
///
/// `positions` is not required to be sorted or distinct here; the GPU
/// kernel that matches this reference takes the sorted, distinct list the
/// host builds from the indexer's mask.
#[allow(clippy::too_many_arguments)]
pub fn indexed_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    positions: &[usize],
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    scale: Option<f32>,
) -> Vec<f32> {
    let row = num_kv_heads * head_dim;
    assert!(row > 0, "num_kv_heads and head_dim must be positive");
    assert_eq!(k.len() % row, 0, "K must be a whole number of rows");
    assert_eq!(k.len(), v.len(), "K and V must have the same shape");
    assert!(
        !positions.is_empty(),
        "at least one position must be selected"
    );
    let seq_len = k.len() / row;

    let mut k_sel = Vec::with_capacity(positions.len() * row);
    let mut v_sel = Vec::with_capacity(positions.len() * row);
    for &p in positions {
        assert!(p < seq_len, "selected position {p} is outside 0..{seq_len}");
        k_sel.extend_from_slice(&k[p * row..(p + 1) * row]);
        v_sel.extend_from_slice(&v[p * row..(p + 1) * row]);
    }
    causal_attention(
        q,
        &k_sel,
        &v_sel,
        head_dim,
        num_q_heads,
        num_kv_heads,
        positions.len(),
        None,
        scale,
    )
}

#[cfg(test)]
mod tests {
    use super::{causal_attention, indexed_attention};

    fn fixture(seq_len: usize, num_kv_heads: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
        let n = seq_len * num_kv_heads * head_dim;
        let k: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.37).sin() * 0.5).collect();
        let v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.11).cos() * 0.5).collect();
        (k, v)
    }

    #[test]
    fn identity_positions_reproduce_causal_attention_exactly() {
        let (head_dim, nq, nkv, seq_len) = (8usize, 4usize, 2usize, 11usize);
        let q: Vec<f32> = (0..nq * head_dim)
            .map(|i| (i as f32 - 16.0) * 0.05)
            .collect();
        let (k, v) = fixture(seq_len, nkv, head_dim);
        let dense = causal_attention(&q, &k, &v, head_dim, nq, nkv, seq_len, None, None);
        let all: Vec<usize> = (0..seq_len).collect();
        let indexed = indexed_attention(&q, &k, &v, &all, head_dim, nq, nkv, None);
        assert_eq!(dense, indexed, "identity list must be the same function");
    }

    #[test]
    fn unselected_rows_do_not_enter_the_softmax_or_the_output() {
        // Selecting rows {1, 4} of a 6-row sequence must equal dense
        // attention over a 2-row sequence made of exactly those rows -- and
        // must DIFFER from dense attention over all 6, or the test could
        // pass on a reference that ignores `positions`.
        let (head_dim, nq, nkv, seq_len) = (4usize, 2usize, 1usize, 6usize);
        let q: Vec<f32> = vec![0.3, -0.2, 0.5, 0.1, -0.4, 0.2, 0.0, 0.6];
        let (k, v) = fixture(seq_len, nkv, head_dim);
        let picked = [1usize, 4usize];
        let indexed = indexed_attention(&q, &k, &v, &picked, head_dim, nq, nkv, None);

        let row = nkv * head_dim;
        let mut k2 = Vec::new();
        let mut v2 = Vec::new();
        for &p in &picked {
            k2.extend_from_slice(&k[p * row..(p + 1) * row]);
            v2.extend_from_slice(&v[p * row..(p + 1) * row]);
        }
        let compact = causal_attention(&q, &k2, &v2, head_dim, nq, nkv, 2, None, None);
        assert_eq!(indexed, compact);

        let dense = causal_attention(&q, &k, &v, head_dim, nq, nkv, seq_len, None, None);
        assert_ne!(indexed, dense, "fixture must make the subset observable");
    }

    #[test]
    #[should_panic(expected = "outside")]
    fn a_position_past_the_sequence_is_refused() {
        let (k, v) = fixture(3, 1, 2);
        let q = vec![1.0, 0.0];
        indexed_attention(&q, &k, &v, &[3], 2, 1, 1, None);
    }
}
