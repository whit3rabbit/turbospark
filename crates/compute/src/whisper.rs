//! FP32 reference kernels for whisper models.
//!
//! Ground truth for the whisper speech family: the conv front end over the
//! log-mel spectrogram, the sinusoidal positional table, Pre-LN encoder
//! blocks, and an incremental decoder step with self-attention KV cache and
//! cross-attention over the encoded window. These run the CPU fallback path
//! when no Metal device is present and are the parity partner for any
//! future whisper Metal kernels.
//!
//! Layout conventions match the Hugging Face whisper checkpoint: linear
//! weights are `[out, in]` row-major, conv weights `[out_ch, in_ch, k]`,
//! attention projections carry NO bias, and activations are exact-erf GELU
//! (`activation_function: "gelu"`). The encoder is PRE-LN -- `x + attn(LN(x))`
//! -- which is the structural difference from the BERT Post-LN reference in
//! `compute::encoder`.

use crate::vision::{bidirectional_attention, gelu_erf, layer_norm, matmul_bias};

/// Weights for the two-layer conv front end.
pub struct WhisperConvWeights<'a> {
    /// `[d_model, n_mels, 3]` flat.
    pub conv1_weight: &'a [f32],
    pub conv1_bias: &'a [f32],
    /// `[d_model, d_model, 3]` flat.
    pub conv2_weight: &'a [f32],
    pub conv2_bias: &'a [f32],
}

/// One dimensional convolution, kernel 3, symmetric zero padding `pad`,
/// stride `stride`, followed by exact-erf GELU: `[in_ch, t] -> [out_ch, t']`
/// with `t' = (t + 2*pad - 3)/stride + 1`.
///
/// Whisper's front end needs both shapes: conv1 runs stride 1, pad 1
/// (length preserving) and conv2 runs stride 2, pad 1 (downsampling by 2),
/// which is how 3000 mel frames become the 1500 positions
/// `max_source_positions` counts.
#[allow(clippy::too_many_arguments)]
pub fn conv1d3_gelu(
    input: &[f32],
    in_ch: usize,
    t_len: usize,
    weight: &[f32],
    bias: &[f32],
    out_ch: usize,
    stride: usize,
    pad: usize,
) -> Vec<f32> {
    assert_eq!(
        weight.len(),
        out_ch * in_ch * 3,
        "conv weight must be out*in*3"
    );
    assert_eq!(bias.len(), out_ch, "conv bias must be out_ch");
    assert!(stride >= 1, "conv stride must be positive");
    let padded = t_len + 2 * pad;
    assert!(padded >= 3, "conv1d kernel 3 needs at least 3 frames");
    let out_t = (padded - 3) / stride + 1;
    let at = |ch: usize, t: isize| -> f32 {
        if t < 0 || t >= t_len as isize {
            0.0
        } else {
            input[ch * t_len + t as usize]
        }
    };
    let mut out = vec![0.0f32; out_ch * out_t];
    for c in 0..out_ch {
        for t in 0..out_t {
            let mut acc = 0.0f64;
            for ic in 0..in_ch {
                for k in 0..3isize {
                    acc += f64::from(at(ic, t as isize * stride as isize + k - pad as isize))
                        * f64::from(weight[(c * in_ch + ic) * 3 + k as usize]);
                }
            }
            out[c * out_t + t] = gelu_erf(&[(acc as f32) + bias[c]])[0];
        }
    }
    out
}

/// The full conv front end: mel `[n_mels, frames]` through conv1 (stride 1,
/// pad 1) and conv2 (stride 2, pad 1), each with GELU, producing
/// `[d_model, frames/2]` -- 3000 mel frames become 1500 positions.
pub fn whisper_conv_frontend(
    mel: &[f32],
    n_mels: usize,
    frames: usize,
    weights: &WhisperConvWeights,
    d_model: usize,
) -> Vec<f32> {
    let hidden = conv1d3_gelu(
        mel,
        n_mels,
        frames,
        weights.conv1_weight,
        weights.conv1_bias,
        d_model,
        1,
        1,
    );
    conv1d3_gelu(
        &hidden,
        d_model,
        frames,
        weights.conv2_weight,
        weights.conv2_bias,
        d_model,
        2,
        1,
    )
}

/// The openai whisper sinusoidal positional table, `[length, channels]`.
///
/// The checkpoint stores this table (`embed_positions.weight`), so the
/// loader reads it back; this generator exists for synthetic fixtures and
/// matches the reference formula exactly.
pub fn sinusoids(length: usize, channels: usize) -> Vec<f32> {
    assert!(channels % 2 == 0, "sinusoids needs an even channel count");
    let half = channels / 2;
    // The reference folds max_timescale (10000) into the log increment
    // alone: inv_timescale[0] is 1.0, NOT 10000. Multiplying here too
    // produced a 10000x-wrong table that the stored-table path never
    // noticed and the MLX-generated path decoded as fluent garbage.
    let log_inc = 10_000f64.ln() / (half - 1) as f64;
    let inv_timescale: Vec<f64> = (0..half).map(|i| (-(i as f64) * log_inc).exp()).collect();
    let mut out = vec![0.0f32; length * channels];
    for pos in 0..length {
        for i in 0..half {
            let scaled = pos as f64 * inv_timescale[i];
            out[pos * channels + i] = scaled.sin() as f32;
            out[pos * channels + half + i] = scaled.cos() as f32;
        }
    }
    out
}

/// Weights for one whisper encoder layer. Attention projections carry no
/// bias in the whisper checkpoint.
pub struct WhisperEncoderLayerWeights<'a> {
    /// `[d_model, d_model]` flat. The checkpoint carries NO k projection
    /// bias (openai's `MultiHeadAttention` builds `key` with
    /// `bias=False`); q, v, and out carry real biases.
    pub q_weight: &'a [f32],
    pub q_bias: &'a [f32],
    pub k_weight: &'a [f32],
    pub v_weight: &'a [f32],
    pub v_bias: &'a [f32],
    pub out_weight: &'a [f32],
    pub out_bias: &'a [f32],
    pub ln1_weight: &'a [f32],
    pub ln1_bias: &'a [f32],
    /// `[ffn, d_model]` flat.
    pub fc1_weight: &'a [f32],
    pub fc1_bias: &'a [f32],
    /// `[d_model, ffn]` flat.
    pub fc2_weight: &'a [f32],
    pub fc2_bias: &'a [f32],
    pub ln2_weight: &'a [f32],
    pub ln2_bias: &'a [f32],
}

/// One Pre-LN encoder block: `h = x + attn(LN1(x))`, `y = h + mlp(LN2(h))`.
#[allow(clippy::too_many_arguments)]
pub fn whisper_encoder_layer(
    x: &[f32],
    seq: usize,
    weights: &WhisperEncoderLayerWeights,
    d_model: usize,
    heads: usize,
    ffn: usize,
    eps: f32,
) -> Vec<f32> {
    let head_dim = d_model / heads;
    let scale = (head_dim as f32).powf(-0.5);
    let normed = layer_norm_rows(x, seq, d_model, weights.ln1_weight, weights.ln1_bias, eps);
    let q = matmul_bias(
        &normed,
        weights.q_weight,
        Some(weights.q_bias),
        seq,
        d_model,
        d_model,
    );
    let k = matmul_bias(&normed, weights.k_weight, None, seq, d_model, d_model);
    let v = matmul_bias(
        &normed,
        weights.v_weight,
        Some(weights.v_bias),
        seq,
        d_model,
        d_model,
    );
    let attn = bidirectional_attention(&q, &k, &v, seq, heads, head_dim, scale);
    let attn_out = matmul_bias(
        &attn,
        weights.out_weight,
        Some(weights.out_bias),
        seq,
        d_model,
        d_model,
    );

    let mut h = vec![0.0f32; seq * d_model];
    for i in 0..seq * d_model {
        h[i] = x[i] + attn_out[i];
    }

    let normed2 = layer_norm_rows(&h, seq, d_model, weights.ln2_weight, weights.ln2_bias, eps);
    let fc1 = matmul_bias(
        &normed2,
        weights.fc1_weight,
        Some(weights.fc1_bias),
        seq,
        d_model,
        ffn,
    );
    let activated = gelu_erf(&fc1);
    let fc2 = matmul_bias(
        &activated,
        weights.fc2_weight,
        Some(weights.fc2_bias),
        seq,
        ffn,
        d_model,
    );

    let mut out = vec![0.0f32; seq * d_model];
    for i in 0..seq * d_model {
        out[i] = h[i] + fc2[i];
    }
    out
}

/// Adds one positional row to every frame of a `[seq, d_model]` stream.
pub fn add_position_rows(
    hidden: &mut [f32],
    positions: &[f32],
    seq: usize,
    d_model: usize,
    start_row: usize,
) {
    assert!(
        (start_row + seq) * d_model <= positions.len(),
        "positional table too short for rows {start_row}..{start_row}+{seq}"
    );
    for t in 0..seq {
        let p = (start_row + t) * d_model;
        for i in 0..d_model {
            hidden[t * d_model + i] += positions[p + i];
        }
    }
}

/// Weights for one whisper decoder layer.
pub struct WhisperDecoderLayerWeights<'a> {
    pub self_q_weight: &'a [f32],
    pub self_q_bias: &'a [f32],
    pub self_k_weight: &'a [f32],
    pub self_v_weight: &'a [f32],
    pub self_v_bias: &'a [f32],
    pub self_out_weight: &'a [f32],
    pub self_out_bias: &'a [f32],
    pub ln_self_weight: &'a [f32],
    pub ln_self_bias: &'a [f32],
    pub cross_q_weight: &'a [f32],
    pub cross_q_bias: &'a [f32],
    pub cross_k_weight: &'a [f32],
    pub cross_v_weight: &'a [f32],
    pub cross_v_bias: &'a [f32],
    pub cross_out_weight: &'a [f32],
    pub cross_out_bias: &'a [f32],
    pub ln_cross_weight: &'a [f32],
    pub ln_cross_bias: &'a [f32],
    pub fc1_weight: &'a [f32],
    pub fc1_bias: &'a [f32],
    pub fc2_weight: &'a [f32],
    pub fc2_bias: &'a [f32],
    pub ln_fc_weight: &'a [f32],
    pub ln_fc_bias: &'a [f32],
}

/// Cached self-attention keys and values for one decoder layer.
#[derive(Debug, Clone, Default)]
pub struct WhisperSelfKv {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    /// Cached keys/values so far, in rows of d_model.
    pub len: usize,
}

/// Cached cross-attention keys and values for one decoder layer,
/// computed once per encoded window.
#[derive(Debug, Clone, Default)]
pub struct WhisperCrossKv {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

/// Projects the encoded window through each layer's cross K/V weights,
/// once per window: the cross-attention key/value cache.
pub fn cross_kv(
    encoder_out: &[f32],
    seq: usize,
    weights: &WhisperDecoderLayerWeights,
    d_model: usize,
) -> WhisperCrossKv {
    WhisperCrossKv {
        k: matmul_bias(
            encoder_out,
            weights.cross_k_weight,
            None,
            seq,
            d_model,
            d_model,
        ),
        v: matmul_bias(
            encoder_out,
            weights.cross_v_weight,
            Some(weights.cross_v_bias),
            seq,
            d_model,
            d_model,
        ),
    }
}

/// Single-query attention over a cached key/value history: `q [d]` against
/// `keys/vals [len, d]`, scaled softmax in f64, returns `d` values.
fn query_attention(
    q: &[f32],
    keys: &[f32],
    vals: &[f32],
    len: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    let mut scores = vec![0.0f64; len];
    let mut max_score = f64::NEG_INFINITY;
    for t in 0..len {
        let mut dot = 0.0f64;
        for i in 0..d {
            dot += f64::from(q[i]) * f64::from(keys[t * d + i]);
        }
        scores[t] = dot * f64::from(scale);
        max_score = max_score.max(scores[t]);
    }
    let mut total = 0.0f64;
    for s in &mut scores {
        *s = (*s - max_score).exp();
        total += *s;
    }
    let mut out = vec![0.0f32; d];
    for t in 0..len {
        let w = scores[t] / total;
        for i in 0..d {
            out[i] += (w * f64::from(vals[t * d + i])) as f32;
        }
    }
    out
}

/// One incremental decoder step for a single token position.
///
/// `x` is the embedding of the current token `[d_model]` at decoder
/// position `pos`; the layer appends its projected K/V to `state`, attends
/// causally over it, cross-attends over the cached encoder window, and
/// returns the updated stream `[d_model]` for the next layer.
#[allow(clippy::too_many_arguments)]
pub fn whisper_decoder_layer_step(
    x: &[f32],
    pos: usize,
    weights: &WhisperDecoderLayerWeights,
    state: &mut WhisperSelfKv,
    cross: &WhisperCrossKv,
    d_model: usize,
    heads: usize,
    eps: f32,
) -> Vec<f32> {
    let head_dim = d_model / heads;
    let scale = (head_dim as f32).powf(-0.5);

    // Pre-LN self attention with KV cache append.
    let normed = layer_norm(x, weights.ln_self_weight, weights.ln_self_bias, eps);
    let q = matmul_bias(
        &normed,
        weights.self_q_weight,
        Some(weights.self_q_bias),
        1,
        d_model,
        d_model,
    );
    let k = matmul_bias(&normed, weights.self_k_weight, None, 1, d_model, d_model);
    let v = matmul_bias(
        &normed,
        weights.self_v_weight,
        Some(weights.self_v_bias),
        1,
        d_model,
        d_model,
    );
    state.k.extend_from_slice(&k);
    state.v.extend_from_slice(&v);
    state.len += 1;
    debug_assert_eq!(state.len, pos + 1);

    let mut attn_mix = vec![0.0f32; d_model];
    for h in 0..heads {
        let q_head = &q[h * head_dim..(h + 1) * head_dim];
        // Per-head slices over the flat [len, d_model] caches.
        let mut k_head = Vec::with_capacity(state.len * head_dim);
        let mut v_head = Vec::with_capacity(state.len * head_dim);
        for t in 0..state.len {
            k_head.extend_from_slice(
                &state.k[t * d_model + h * head_dim..t * d_model + (h + 1) * head_dim],
            );
            v_head.extend_from_slice(
                &state.v[t * d_model + h * head_dim..t * d_model + (h + 1) * head_dim],
            );
        }
        let mixed = query_attention(q_head, &k_head, &v_head, state.len, head_dim, scale);
        attn_mix[h * head_dim..(h + 1) * head_dim].copy_from_slice(&mixed);
    }
    let attn_out = matmul_bias(
        &attn_mix,
        weights.self_out_weight,
        Some(weights.self_out_bias),
        1,
        d_model,
        d_model,
    );
    let mut h1 = vec![0.0f32; d_model];
    for i in 0..d_model {
        h1[i] = x[i] + attn_out[i];
    }

    // Cross attention over the encoded window.
    let normed2 = layer_norm(&h1, weights.ln_cross_weight, weights.ln_cross_bias, eps);
    let q = matmul_bias(
        &normed2,
        weights.cross_q_weight,
        Some(weights.cross_q_bias),
        1,
        d_model,
        d_model,
    );
    let mut attn_mix = vec![0.0f32; d_model];
    let seq = cross.k.len() / d_model;
    for h in 0..heads {
        let q_head = &q[h * head_dim..(h + 1) * head_dim];
        let mut k_head = Vec::with_capacity(seq * head_dim);
        let mut v_head = Vec::with_capacity(seq * head_dim);
        for t in 0..seq {
            k_head.extend_from_slice(
                &cross.k[t * d_model + h * head_dim..t * d_model + (h + 1) * head_dim],
            );
            v_head.extend_from_slice(
                &cross.v[t * d_model + h * head_dim..t * d_model + (h + 1) * head_dim],
            );
        }
        let mixed = query_attention(q_head, &k_head, &v_head, seq, head_dim, scale);
        attn_mix[h * head_dim..(h + 1) * head_dim].copy_from_slice(&mixed);
    }
    let cross_out = matmul_bias(
        &attn_mix,
        weights.cross_out_weight,
        Some(weights.cross_out_bias),
        1,
        d_model,
        d_model,
    );
    let mut h2 = vec![0.0f32; d_model];
    for i in 0..d_model {
        h2[i] = h1[i] + cross_out[i];
    }

    // FFN: fc1 is [ffn, d_model], fc2 is [d_model, ffn].
    let normed3 = layer_norm(&h2, weights.ln_fc_weight, weights.ln_fc_bias, eps);
    let fc1 = matmul_bias(
        &normed3,
        weights.fc1_weight,
        Some(weights.fc1_bias),
        1,
        d_model,
        weights.fc1_bias.len(),
    );
    let activated = gelu_erf(&fc1);
    let ffn_in = weights.fc2_weight.len() / d_model;
    let fc2 = matmul_bias(
        &activated,
        weights.fc2_weight,
        Some(weights.fc2_bias),
        1,
        ffn_in,
        d_model,
    );
    let mut out = vec![0.0f32; d_model];
    for i in 0..d_model {
        out[i] = h2[i] + fc2[i];
    }
    out
}

/// LayerNorm over every row of a `[seq, d_model]` stream.
fn layer_norm_rows(
    x: &[f32],
    seq: usize,
    d_model: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; seq * d_model];
    for t in 0..seq {
        let row = &x[t * d_model..(t + 1) * d_model];
        out[t * d_model..(t + 1) * d_model].copy_from_slice(&layer_norm(row, weight, bias, eps));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG scalar and matrix helpers for fixtures.
    pub(crate) fn lcg(rng: &mut u64) -> f32 {
        *rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*rng >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    fn rng_vec(rng: &mut u64, n: usize) -> Vec<f32> {
        (0..n).map(|_| lcg(rng)).collect()
    }

    /// A tiny deterministic conv fixture, verified against a hand-run of
    /// the same arithmetic with different loop order.
    #[test]
    fn conv1d3_gelu_matches_hand_computation() {
        // in_ch 2, out_ch 1, t 4; weights all small integers.
        let input = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0];
        let weight = vec![1.0, 0.5, 0.25, 1.0, 1.0, 1.0];
        let bias = vec![0.1];
        let got = conv1d3_gelu(&input, 2, 4, &weight, &bias, 1, 1, 0);
        assert_eq!(got.len(), 2);
        for t in 0..2 {
            let mut acc = 0.0f64;
            for k in 0..3 {
                acc += f64::from(input[k + t]) * f64::from(weight[k]);
                acc += f64::from(input[4 + k + t]) * f64::from(weight[3 + k]);
            }
            let want = gelu_erf(&[(acc as f32) + 0.1])[0];
            assert!((got[t] - want).abs() < 1e-6, "t={t}: {} vs {want}", got[t]);
        }
    }

    #[test]
    fn conv_shapes_match_whisper_frontend() {
        // conv1: stride 1 pad 1 preserves length; conv2: stride 2 pad 1
        // halves it -- 3000 mel frames must become 1500 positions.
        let frames = 3000usize;
        let mel = vec![0.25f32; 80 * frames];
        let d = 384usize;
        let mut rng = 9u64;
        let w1: Vec<f32> = (0..d * 80 * 3).map(|_| lcg(&mut rng) * 0.01).collect();
        let w2: Vec<f32> = (0..d * d * 3).map(|_| lcg(&mut rng) * 0.01).collect();
        let b1 = vec![0.0f32; d];
        let b2 = vec![0.0f32; d];
        let weights = WhisperConvWeights {
            conv1_weight: &w1,
            conv1_bias: &b1,
            conv2_weight: &w2,
            conv2_bias: &b2,
        };
        let out = whisper_conv_frontend(&mel, 80, frames, &weights, d);
        assert_eq!(out.len(), d * 1500);
    }

    #[test]
    fn sinusoids_matches_the_reference_formula() {
        let table = sinusoids(8, 4);
        assert_eq!(table.len(), 32);
        // Row 0: sin(0), sin(0), cos(0), cos(0).
        assert_eq!(&table[0..4], &[0.0, 0.0, 1.0, 1.0]);
        // inv_timescale[0] is 1.0 (max_timescale folds into the log
        // increment), so row 1 starts sin(1), cos(1).
        let sin1 = 1.0f64.sin() as f32;
        let cos1 = 1.0f64.cos() as f32;
        assert!((table[4] - sin1).abs() < 1e-6, "sin {}", table[4]);
        assert!((table[4 + 2] - cos1).abs() < 1e-6, "cos {}", table[4 + 2]);
    }

    #[test]
    fn encoder_layer_matches_a_naive_independent_reference() {
        // Deterministic pseudo-random fixture, d_model 4, heads 2, ffn 6.
        let d = 4usize;
        let seq = 3usize;
        let ffn = 6usize;
        let mut rng = 12345u64;
        let x = rng_vec(&mut rng, seq * d);
        let (qw, kw, vw, ow) = (
            rng_vec(&mut rng, d * d),
            rng_vec(&mut rng, d * d),
            rng_vec(&mut rng, d * d),
            rng_vec(&mut rng, d * d),
        );
        let (qb, vb, ob) = (
            rng_vec(&mut rng, d),
            rng_vec(&mut rng, d),
            rng_vec(&mut rng, d),
        );
        let fc1 = rng_vec(&mut rng, ffn * d);
        let fc2 = rng_vec(&mut rng, d * ffn);
        let b1 = rng_vec(&mut rng, ffn);
        let b2 = rng_vec(&mut rng, d);
        let ln1w: Vec<f32> = (0..d).map(|_| 0.5 + lcg(&mut rng).abs() * 0.5).collect();
        let ln1b = rng_vec(&mut rng, d);
        let ln2w: Vec<f32> = (0..d).map(|_| 0.5 + lcg(&mut rng).abs() * 0.5).collect();
        let ln2b = rng_vec(&mut rng, d);
        let eps = 1e-5;

        let got = whisper_encoder_layer(
            &x,
            seq,
            &WhisperEncoderLayerWeights {
                q_weight: &qw,
                q_bias: &qb,
                k_weight: &kw,
                v_weight: &vw,
                v_bias: &vb,
                out_weight: &ow,
                out_bias: &ob,
                ln1_weight: &ln1w,
                ln1_bias: &ln1b,
                fc1_weight: &fc1,
                fc1_bias: &b1,
                fc2_weight: &fc2,
                fc2_bias: &b2,
                ln2_weight: &ln2w,
                ln2_bias: &ln2b,
            },
            d,
            2,
            ffn,
            eps,
        );

        // Naive reference in f64 with explicit loops, written against the
        // formulas rather than the kernels.
        let ln = |row: &[f64], w: &[f32], b: &[f32]| -> Vec<f64> {
            let mean = row.iter().sum::<f64>() / row.len() as f64;
            let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / row.len() as f64;
            row.iter()
                .enumerate()
                .map(|(i, v)| {
                    ((v - mean) / (var + f64::from(eps)).sqrt()) * f64::from(w[i]) + f64::from(b[i])
                })
                .collect()
        };
        let linear =
            |row: &[f64], w: &[f32], rows: usize, cols: usize, bias: Option<&[f32]>| -> Vec<f64> {
                (0..rows)
                    .map(|r| {
                        let mut acc = bias.map(|b| f64::from(b[r])).unwrap_or(0.0);
                        for c in 0..cols {
                            acc += row[c] * f64::from(w[r * cols + c]);
                        }
                        acc
                    })
                    .collect()
            };
        let gelu = |v: f64| 0.5 * v * (1.0 + erf(v / 2.0f64.sqrt()));
        for t in 0..seq {
            let row: Vec<f64> = x[t * d..(t + 1) * d]
                .iter()
                .map(|&v| f64::from(v))
                .collect();
            // Attention needs the whole sequence, so build normed rows for
            // every position first.
            let normed: Vec<Vec<f64>> = (0..seq)
                .map(|s| {
                    ln(
                        &x[s * d..(s + 1) * d]
                            .iter()
                            .map(|&v| f64::from(v))
                            .collect::<Vec<_>>(),
                        &ln1w,
                        &ln1b,
                    )
                })
                .collect();
            let scale = (d as f64 / 2.0).powf(-0.5);
            let mut attn_mix = vec![0.0f64; d];
            for h in 0..2 {
                // head h covers dims [h*2, h*2+2)
                let qh = &linear(&normed[t], &qw, d, d, Some(&qb))[h * 2..h * 2 + 2];
                let mut scores = Vec::new();
                for normed_row in &normed {
                    let kh = &linear(normed_row, &kw, d, d, None)[h * 2..h * 2 + 2];
                    let dot: f64 = qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f64>() * scale;
                    scores.push(dot);
                }
                let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f64 = exps.iter().sum();
                for (s, &w) in exps.iter().enumerate() {
                    let vh = &linear(&normed[s], &vw, d, d, Some(&vb))[h * 2..h * 2 + 2];
                    for i in 0..2 {
                        attn_mix[h * 2 + i] += w / total * vh[i];
                    }
                }
            }
            let attn_out = linear(&attn_mix, &ow, d, d, Some(&ob));
            let mut h1 = vec![0.0f64; d];
            for i in 0..d {
                h1[i] = row[i] + attn_out[i];
            }
            let ln_h1 = ln(&h1, &ln2w, &ln2b);
            let f1 = linear(&ln_h1, &fc1, ffn, d, Some(&b1));
            let act: Vec<f64> = f1.iter().map(|&v| gelu(v)).collect();
            let f2 = linear(&act, &fc2, d, ffn, Some(&b2));
            for i in 0..d {
                let want = h1[i] + f2[i];
                assert!(
                    (f64::from(got[t * d + i]) - want).abs() < 2e-4,
                    "pos {t} dim {i}: {} vs {want}",
                    got[t * d + i]
                );
            }
        }
    }

    /// Abramowitz-Stegun erf approximation, the test's own independent erf.
    fn erf(x: f64) -> f64 {
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + 0.327_591_1 * x);
        let y = 1.0
            - (-x * x).exp()
                * (t * (0.254_829_592
                    + t * (-0.284_496_736
                        + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429)))));
        sign * y
    }

    #[test]
    fn decoder_step_incremental_matches_full_recompute() {
        // The classic KV-cache oracle: stepping one position at a time
        // must produce the same stream as recomputing the full causal
        // sequence each time, for every layer weight fixture.
        let d = 4usize;
        let heads = 2usize;
        let seq = 4usize;
        let ffn = 6usize;
        let mut rng = 777u64;
        let (sq, sk, sv, so) = (
            rng_vec(&mut rng, d * d),
            rng_vec(&mut rng, d * d),
            rng_vec(&mut rng, d * d),
            rng_vec(&mut rng, d * d),
        );
        let (cq, ck, cv, co) = (
            rng_vec(&mut rng, d * d),
            rng_vec(&mut rng, d * d),
            rng_vec(&mut rng, d * d),
            rng_vec(&mut rng, d * d),
        );
        let (sqb, svb, sob) = (
            rng_vec(&mut rng, d),
            rng_vec(&mut rng, d),
            rng_vec(&mut rng, d),
        );
        let (cqb, cvb, cob) = (
            rng_vec(&mut rng, d),
            rng_vec(&mut rng, d),
            rng_vec(&mut rng, d),
        );
        let fc1 = rng_vec(&mut rng, ffn * d);
        let fc2 = rng_vec(&mut rng, d * ffn);
        let b1 = rng_vec(&mut rng, ffn);
        let b2 = rng_vec(&mut rng, d);
        let lnw = |scale: f32| -> Vec<f32> { (0..d).map(|_| scale).collect() };
        let lnb = |v: f32| -> Vec<f32> { (0..d).map(|_| v).collect() };
        let weights = WhisperDecoderLayerWeights {
            self_q_weight: &sq,
            self_q_bias: &sqb,
            self_k_weight: &sk,
            self_v_weight: &sv,
            self_v_bias: &svb,
            self_out_weight: &so,
            self_out_bias: &sob,
            ln_self_weight: &lnw(1.0),
            ln_self_bias: &lnb(0.0),
            cross_q_weight: &cq,
            cross_q_bias: &cqb,
            cross_k_weight: &ck,
            cross_v_weight: &cv,
            cross_v_bias: &cvb,
            cross_out_weight: &co,
            cross_out_bias: &cob,
            ln_cross_weight: &lnw(1.0),
            ln_cross_bias: &lnb(0.0),
            fc1_weight: &fc1,
            fc1_bias: &b1,
            fc2_weight: &fc2,
            fc2_bias: &b2,
            ln_fc_weight: &lnw(1.0),
            ln_fc_bias: &lnb(0.0),
        };
        let tokens = rng_vec(&mut rng, seq * d);
        let enc = rng_vec(&mut rng, seq * d);
        let cross = cross_kv(&enc, seq, &weights, d);

        let mut state = WhisperSelfKv::default();
        for pos in 0..seq {
            let got = whisper_decoder_layer_step(
                &tokens[pos * d..(pos + 1) * d],
                pos,
                &weights,
                &mut state,
                &cross,
                d,
                heads,
                1e-5,
            );
            // Full recompute of the same position: identical weights, the
            // self K/V history equals what the cache accumulated.
            let full = full_causal_step(
                &tokens[..(pos + 1) * d],
                pos,
                &weights,
                &cross,
                d,
                heads,
                ffn,
            );
            for i in 0..d {
                assert!(
                    (got[i] - full[i]).abs() < 1e-5,
                    "pos {pos} dim {i}: {} vs {}",
                    got[i],
                    full[i]
                );
            }
        }
    }

    /// A recompute-style decoder step (rebuilds K/V from the whole prefix)
    /// used as the incremental path's oracle.
    fn full_causal_step(
        prefix: &[f32],
        pos: usize,
        w: &WhisperDecoderLayerWeights,
        cross: &WhisperCrossKv,
        d: usize,
        heads: usize,
        ffn: usize,
    ) -> Vec<f32> {
        let len = pos + 1;
        let head_dim = d / heads;
        let scale = (head_dim as f32).powf(-0.5);
        let normed = layer_norm_rows(prefix, len, d, w.ln_self_weight, w.ln_self_bias, 1e-5);
        let q = matmul_bias(
            &normed[(len - 1) * d..len * d],
            w.self_q_weight,
            Some(w.self_q_bias),
            1,
            d,
            d,
        );
        let k = matmul_bias(&normed, w.self_k_weight, None, len, d, d);
        let v = matmul_bias(&normed, w.self_v_weight, Some(w.self_v_bias), len, d, d);
        let mut attn_mix = vec![0.0f32; d];
        for h in 0..heads {
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            let mut kh = Vec::with_capacity(len * head_dim);
            let mut vh = Vec::with_capacity(len * head_dim);
            for t in 0..len {
                kh.extend_from_slice(&k[t * d + h * head_dim..t * d + (h + 1) * head_dim]);
                vh.extend_from_slice(&v[t * d + h * head_dim..t * d + (h + 1) * head_dim]);
            }
            let mixed = query_attention(qh, &kh, &vh, len, head_dim, scale);
            attn_mix[h * head_dim..(h + 1) * head_dim].copy_from_slice(&mixed);
        }
        let attn_out = matmul_bias(&attn_mix, w.self_out_weight, Some(w.self_out_bias), 1, d, d);
        let mut h1 = vec![0.0f32; d];
        for i in 0..d {
            h1[i] = prefix[pos * d + i] + attn_out[i];
        }
        let normed2 = layer_norm(&h1, w.ln_cross_weight, w.ln_cross_bias, 1e-5);
        let q = matmul_bias(&normed2, w.cross_q_weight, Some(w.cross_q_bias), 1, d, d);
        let enc_seq = cross.k.len() / d;
        let mut attn_mix = vec![0.0f32; d];
        for h in 0..heads {
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            let mut kh = Vec::with_capacity(enc_seq * head_dim);
            let mut vh = Vec::with_capacity(enc_seq * head_dim);
            for t in 0..enc_seq {
                kh.extend_from_slice(&cross.k[t * d + h * head_dim..t * d + (h + 1) * head_dim]);
                vh.extend_from_slice(&cross.v[t * d + h * head_dim..t * d + (h + 1) * head_dim]);
            }
            let mixed = query_attention(qh, &kh, &vh, enc_seq, head_dim, scale);
            attn_mix[h * head_dim..(h + 1) * head_dim].copy_from_slice(&mixed);
        }
        let cross_out = matmul_bias(
            &attn_mix,
            w.cross_out_weight,
            Some(w.cross_out_bias),
            1,
            d,
            d,
        );
        let mut h2 = vec![0.0f32; d];
        for i in 0..d {
            h2[i] = h1[i] + cross_out[i];
        }
        let normed3 = layer_norm(&h2, w.ln_fc_weight, w.ln_fc_bias, 1e-5);
        let fc1 = matmul_bias(&normed3, w.fc1_weight, Some(w.fc1_bias), 1, d, ffn);
        let act = gelu_erf(&fc1);
        let fc2 = matmul_bias(&act, w.fc2_weight, Some(w.fc2_bias), 1, ffn, d);
        let mut out = vec![0.0f32; d];
        for i in 0..d {
            out[i] = h2[i] + fc2[i];
        }
        out
    }

    #[test]
    fn add_position_rows_broadcasts() {
        let mut hidden = vec![1.0f32; 4];
        let positions = vec![0.0f32, 0.5, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        add_position_rows(&mut hidden, &positions, 2, 2, 0);
        assert_eq!(hidden, vec![1.0, 1.5, 2.0, 1.0]);
    }
}
