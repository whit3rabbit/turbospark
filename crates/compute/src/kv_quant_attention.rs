//! CPU reference for causal attention over TurboQuant-quantized K/V rows:
//! the contract [`crate::kv_quant`]'s packed rows are checked against, and
//! the contract `crates/gpu`'s `attention_tq.metal` pair
//! (`attention_decode_partial_tq` / `attention_decode_combine_tq`) is
//! checked against in turn.
//!
//! Mirrors mlx-vlm's `_TurboQuantMSECodec.score` / `weighted_sum`: the
//! query is rotated with the KEY rotation ONCE per query head, and scores
//! are computed in ROTATED space (`norm_k * sum_i q_rot[i] * cb_k[idx_i]`)
//! rather than by dequantizing K first. Values accumulate in rotated space
//! (`weight * norm_v * cb_v[idx]`) and the INVERSE value rotation is
//! applied ONCE, after the weighted sum over all positions -- not per row.
//! Sinks join the softmax denominator only, exactly as in
//! [`crate::attention::causal_attention_with_sinks`].

use crate::kv_quant::{rht_forward, rht_inverse, unpack_lsb_first, QuantizedRow};

/// Codebook, sign vector and bit width for one side (K or V) of a
/// TurboQuant-quantized layer. Built once per (head_dim, bits, seed) and
/// reused across every row and every decode step.
#[derive(Debug, Clone)]
pub struct TqTables {
    pub signs: Vec<f32>,
    pub codebook: Vec<f32>,
    pub midpoints: Vec<f32>,
    pub bits: u8,
}

impl TqTables {
    /// Builds the codebook, its midpoints, and the sign vector for one
    /// side of the codec at `dim` head_dim and `bits` width, seeded with
    /// [`crate::kv_quant::KEY_SEED`] or [`crate::kv_quant::VALUE_SEED`].
    pub fn new(dim: usize, bits: u8, seed: u64) -> Self {
        let codebook = crate::kv_quant::codebook(dim, bits);
        let midpoints = crate::kv_quant::midpoints(&codebook);
        let signs = crate::kv_quant::sign_vector(dim, seed);
        Self {
            signs,
            codebook,
            midpoints,
            bits,
        }
    }
}

/// Q layout: `[num_q_heads, head_dim]`. K/V layout: one [`QuantizedRow`]
/// per `(position, kv_head)`, row-major `[seq_len, num_kv_heads]` (`k[p *
/// num_kv_heads + kv_head]`). Output: `[num_q_heads, head_dim]`.
///
/// TurboQuant only ever quantizes FULL-ATTENTION layers (mlx-vlm's
/// `should_quantize_kv_layer`, ported at
/// `model_io::kv_quant::layer_is_quantized`), so this reference takes no
/// sliding-window argument -- unlike [`crate::attention::causal_attention`],
/// which does.
#[allow(clippy::too_many_arguments)]
pub fn causal_attention_tq(
    q: &[f32],
    k: &[QuantizedRow],
    v: &[QuantizedRow],
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    seq_len: usize,
    scale: Option<f32>,
    sinks: Option<&[f32]>,
    k_tables: &TqTables,
    v_tables: &TqTables,
) -> Vec<f32> {
    assert!(
        num_q_heads % num_kv_heads == 0,
        "num_q_heads must be a multiple of num_kv_heads"
    );
    assert_eq!(q.len(), num_q_heads * head_dim);
    assert_eq!(k.len(), seq_len * num_kv_heads);
    assert_eq!(v.len(), seq_len * num_kv_heads);

    let group_size = num_q_heads / num_kv_heads;
    let scale = scale.unwrap_or(1.0 / (head_dim as f32).sqrt());

    let mut out = vec![0f32; num_q_heads * head_dim];

    for qh in 0..num_q_heads {
        let kv_head = qh / group_size;
        let q_base = qh * head_dim;
        let q_vec = &q[q_base..q_base + head_dim];
        let q_rot = rht_forward(q_vec, &k_tables.signs);

        let mut scores: Vec<f32> = (0..seq_len)
            .map(|p| {
                let row = &k[p * num_kv_heads + kv_head];
                let indices = unpack_lsb_first(&row.words, k_tables.bits, head_dim);
                let dot: f32 = q_rot
                    .iter()
                    .zip(&indices)
                    .map(|(&qv, &idx)| qv * k_tables.codebook[idx as usize])
                    .sum();
                dot * row.norm * scale
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

        // Values accumulate in ROTATED space; the inverse rotation runs
        // ONCE, after the sum over every position, never per row.
        let mut acc_rot = vec![0f32; head_dim];
        for (p, &weight) in scores.iter().enumerate() {
            let row = &v[p * num_kv_heads + kv_head];
            let indices = unpack_lsb_first(&row.words, v_tables.bits, head_dim);
            let coeff = weight * row.norm;
            for (d, &idx) in indices.iter().enumerate() {
                acc_rot[d] += coeff * v_tables.codebook[idx as usize];
            }
        }
        let acc = rht_inverse(&acc_rot, &v_tables.signs);
        out[q_base..q_base + head_dim].copy_from_slice(&acc);
    }
    out
}
