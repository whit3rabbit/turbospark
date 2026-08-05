//! FP32 RoPE reference. Ported from
//! `Support/Reference/Attention/RoPE.swift`.

/// Paired-convention RoPE. Pairs in `[0, rotary_dim/2)` rotate; pairs from
/// `rotary_dim/2` to `head_dim/2` pass through unchanged. Setting
/// `rotary_dim == head_dim` recovers full rotation.
pub fn rope_paired(
    input: &[f32],
    num_tokens: usize,
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    position: usize,
    theta: f32,
) -> Vec<f32> {
    assert_eq!(
        input.len(),
        num_tokens * num_heads * head_dim,
        "input must be num_tokens * num_heads * head_dim"
    );
    assert!(rotary_dim % 2 == 0, "rotary_dim must be even");
    assert!(rotary_dim <= head_dim, "rotary_dim cannot exceed head_dim");

    let pairs = rotary_dim / 2;
    let log_theta = theta.ln();
    let position_f = position as f32;
    let angles: Vec<f32> = (0..pairs)
        .map(|k| {
            let exponent = -((2 * k) as f32) / rotary_dim as f32;
            position_f * (exponent * log_theta).exp()
        })
        .collect();
    let cos_table: Vec<f32> = angles.iter().map(|a| a.cos()).collect();
    let sin_table: Vec<f32> = angles.iter().map(|a| a.sin()).collect();

    let mut out = input.to_vec();
    for t in 0..num_tokens {
        for h in 0..num_heads {
            let base = (t * num_heads + h) * head_dim;
            for k in 0..pairs {
                let i0 = base + 2 * k;
                let i1 = i0 + 1;
                let x0 = input[i0];
                let x1 = input[i1];
                let c = cos_table[k];
                let s = sin_table[k];
                out[i0] = x0 * c - x1 * s;
                out[i1] = x0 * s + x1 * c;
            }
        }
    }
    out
}

/// NeoX-convention RoPE. Pairs `(x[i], x[i + head_dim/2])` for
/// `i in [0, rotated_pairs)`. Frequencies divide by `head_dim` (not
/// `2 * rotated_pairs`), matching HF Gemma 4's proportional-RoPE init.
/// Setting `rotated_pairs == head_dim / 2` recovers full NeoX rotation.
pub fn rope_neox(
    input: &[f32],
    num_tokens: usize,
    num_heads: usize,
    head_dim: usize,
    rotated_pairs: usize,
    position: usize,
    theta: f32,
) -> Vec<f32> {
    assert_eq!(
        input.len(),
        num_tokens * num_heads * head_dim,
        "input size mismatch"
    );
    assert!(
        rotated_pairs * 2 <= head_dim,
        "rotated_pairs * 2 must not exceed head_dim"
    );
    let half_dim = head_dim / 2;
    let log_theta = theta.ln();
    let position_f = position as f32;
    let angles: Vec<f32> = (0..rotated_pairs)
        .map(|i| {
            let exponent = -((2 * i) as f32) / head_dim as f32;
            position_f * (exponent * log_theta).exp()
        })
        .collect();
    let cos_table: Vec<f32> = angles.iter().map(|a| a.cos()).collect();
    let sin_table: Vec<f32> = angles.iter().map(|a| a.sin()).collect();

    let mut out = input.to_vec();
    for t in 0..num_tokens {
        for h in 0..num_heads {
            let base = (t * num_heads + h) * head_dim;
            for i in 0..rotated_pairs {
                let i0 = base + i;
                let i1 = base + half_dim + i;
                let x0 = input[i0];
                let x1 = input[i1];
                let c = cos_table[i];
                let s = sin_table[i];
                out[i0] = x0 * c - x1 * s;
                out[i1] = x0 * s + x1 * c;
            }
        }
    }
    out
}
