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

/// Qwen's partial NeoX RoPE: rotation confined to the FIRST `rotary_dim`
/// elements of each head, pairing `(x[i], x[rotary_dim/2 + i])` inside that
/// window, with the frequency divisor `rotary_dim`. Elements at or past
/// `rotary_dim` pass through untouched.
///
/// Distinct from [`rope_neox`] on two counts, both of which change the
/// numbers: the pair partner is `rotary_dim/2` away, not `head_dim/2`, and
/// frequencies divide by `rotary_dim`, not `head_dim`. Gemma's proportional
/// variant rotates a prefix of the pairs across the FULL head; this one
/// rotates all the pairs of a prefix of the head.
pub fn rope_neox_subdim(
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
        "input size mismatch"
    );
    assert!(rotary_dim % 2 == 0, "rotary_dim must be even");
    assert!(rotary_dim <= head_dim, "rotary_dim cannot exceed head_dim");

    let pairs = rotary_dim / 2;
    let log_theta = theta.ln();
    let position_f = position as f32;
    let angles: Vec<f32> = (0..pairs)
        .map(|i| {
            let exponent = -((2 * i) as f32) / rotary_dim as f32;
            position_f * (exponent * log_theta).exp()
        })
        .collect();

    let mut out = input.to_vec();
    for t in 0..num_tokens {
        for head in 0..num_heads {
            let base = (t * num_heads + head) * head_dim;
            for (i, angle) in angles.iter().enumerate() {
                let (s, c) = angle.sin_cos();
                let i0 = base + i;
                let i1 = base + pairs + i;
                let x0 = input[i0];
                let x1 = input[i1];
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

/// YaRN rope scaling (ROADMAP M5, the `gpt-oss` family), reduced to the two
/// things a kernel needs: a per-dimension frequency table and one magnitude
/// scale.
///
/// **This exists because the three rope kernels here take a SCALAR theta and
/// YaRN is not expressible as one.** It interpolates, per dimension pair,
/// between extrapolating the trained frequency and interpolating it by
/// `1/factor`, along a ramp between two correction dimensions. The result is
/// still position-INDEPENDENT, though, which is the whole reason this can be
/// a table computed once at open rather than a kernel that reruns the ramp
/// every token: ggml's `rope_yarn` is linear in `theta_extrap`, and
/// `theta_extrap = position * base^(-2i/n_dims)`, so
///
/// ```text
/// theta_i = position * base^(-2i/n_dims) * (freq_scale*(1 - mix_i) + mix_i)
/// ```
///
/// and everything after `position *` is what [`yarn_frequencies`] returns.
///
/// THE MAGNITUDE SCALE IS THE PART THAT LOOKS LIKE IT CANCELS AND DOES NOT.
/// llama.cpp computes `attn_factor = 1 + 0.1*ln(factor)` and then divides by
/// the same quantity, which reads as a cancellation to 1.0; it is there to
/// stop `rope_yarn` applying the factor TWICE, because `rope_yarn` multiplies
/// cos and sin by `1 + 0.1*ln(1/freq_scale)` itself. Net, q and k are each
/// scaled ONCE by [`YarnSpec::mscale`], and `q.k` therefore by its square.
/// Dropping it leaves a model that is finite, fluent and wrong.
///
/// References, read rather than recalled: `rope_yarn`,
/// `rope_yarn_ramp` and `ggml_rope_yarn_corr_dims` in ggml, and the
/// `yarn_attn_factor` block in llama.cpp's `llama-context.cpp`.
#[derive(Debug, Clone, PartialEq)]
pub struct YarnSpec {
    /// Per dimension PAIR, so `rotary_dim / 2` entries. Multiply by the
    /// position to get that pair's rotation angle.
    pub frequencies: Vec<f32>,
    /// Multiplies both cos and sin, i.e. scales q and k.
    pub mscale: f32,
}

/// The two correction dimensions the ramp runs between, in PAIR units
/// scaled by two (ggml keeps them in element units and compares against
/// `i0/2`, which this reproduces exactly rather than simplifying).
fn yarn_corr_dim(n_dims: usize, n_ctx_orig: i64, n_rot: f32, base: f32) -> f32 {
    n_dims as f32 * (n_ctx_orig as f32 / (n_rot * 2.0 * std::f32::consts::PI)).ln()
        / (2.0 * base.ln())
}

/// `1` at the fast end of the ramp, `0` at the slow end, linear between.
fn yarn_ramp(low: f32, high: f32, i0: usize) -> f32 {
    let y = (i0 as f32 / 2.0 - low) / (high - low).max(0.001);
    1.0 - y.clamp(0.0, 1.0)
}

/// Build the frequency table and magnitude scale for one rope base.
///
/// `factor <= 0` means no scaling, and the result is then the ordinary
/// unscaled frequency table with `mscale = 1.0` -- so a caller can use this
/// path unconditionally and a family without YaRN gets exactly what
/// `rope_proportional_neox` computes inline.
#[must_use]
pub fn yarn_frequencies(
    rotary_dim: usize,
    base: f32,
    factor: f32,
    original_context: i64,
    beta_fast: f32,
    beta_slow: f32,
) -> YarnSpec {
    assert!(rotary_dim % 2 == 0, "rotary_dim must be even");
    let pairs = rotary_dim / 2;
    let active = factor > 0.0;
    let freq_scale = if active { 1.0 / factor } else { 1.0 };

    // ggml clamps to `[0, n_dims - 1]` in ELEMENT units.
    let (low, high) = if active {
        let start = yarn_corr_dim(rotary_dim, original_context, beta_fast, base).floor();
        let end = yarn_corr_dim(rotary_dim, original_context, beta_slow, base).ceil();
        (start.max(0.0), end.min(rotary_dim as f32 - 1.0))
    } else {
        (0.0, 0.0)
    };

    let frequencies = (0..pairs)
        .map(|p| {
            let i0 = 2 * p;
            let extrap = base.powf(-(i0 as f32) / rotary_dim as f32);
            if !active {
                return extrap;
            }
            // `ext_factor` is 1.0 whenever the file declares yarn, which is
            // llama.cpp's own default for that scaling type; this port has no
            // user knob to set it otherwise, so it is folded in rather than
            // carried.
            let mix = yarn_ramp(low, high, i0);
            extrap * (freq_scale * (1.0 - mix) + mix)
        })
        .collect();

    let mscale = if active { 1.0 + 0.1 * factor.ln() } else { 1.0 };
    YarnSpec {
        frequencies,
        mscale,
    }
}
