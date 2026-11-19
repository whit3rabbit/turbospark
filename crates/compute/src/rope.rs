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

/// Which of `(t, h, w)` drives each frequency pair under Qwen 3.8's
/// INTERLEAVED mRoPE (`docs/VISION_PHASE0.md` item 2). Returns one component
/// index in `0..3` per pair, `freq_dim == rotary_dim / 2` entries.
///
/// A transcription of `_interleaved_position_selector`
/// (`mlx_vlm/models/rope_utils.py:350-355`): every pair defaults to `t`, then
/// `h` claims residue 1 and `w` residue 2, each stopping at
/// `min(section[dim] * 3, freq_dim)`.
///
/// **BOTH CLAMPS ARE REAL AND THE `i % 3` COLLAPSE IS NOT THE RULE.** On this
/// family they never bind -- `head_dim` 256 at `partial_rotary_factor` 0.25
/// gives `freq_dim` 32, and the declared `[11, 11, 10]` tiles it exactly, so
/// the selector reduces to `i % 3`. That is a property of one config rather
/// than of the format: sections summing to less than `freq_dim` leave a tail
/// of pairs on `t`, and the collapse would silently hand them to `h` and `w`.
///
/// `section[0]` is read by nothing, here or in the reference. `t` is what a
/// pair gets when neither of the other two claims it, so stating its width
/// would be a second, drifting copy of `freq_dim - (claimed by h) - (claimed
/// by w)`.
pub fn mrope_component_selector(section: [usize; 3], freq_dim: usize) -> Vec<u8> {
    let mut selector = vec![0u8; freq_dim];
    // `h` takes residue 1, `w` residue 2. The two index sets are disjoint, so
    // the reference's loop order is not load-bearing -- worth knowing before
    // anyone reorders them.
    for (dim, offset) in [(1usize, 1usize), (2usize, 2usize)] {
        let limit = (section[dim] * 3).min(freq_dim);
        let mut idx = offset;
        while idx < limit {
            selector[idx] = dim as u8;
            idx += 3;
        }
    }
    selector
}

/// Interleaved mRoPE: [`rope_neox_subdim`] with the scalar position replaced
/// by a per-pair choice among `(t, h, w)` through
/// [`mrope_component_selector`].
///
/// Same pairing, same frequency divisor, same angle arithmetic. **At
/// `t == h == w` this is [`rope_neox_subdim`] evaluated at that position**,
/// which is not a coincidence to be measured but the whole reason the trunk's
/// existing kernel stays exact for every TEXT token of a mixed prompt
/// (`docs/VISION_PHASE0.md` item 2). Only an image's own tokens diverge.
#[allow(clippy::too_many_arguments)]
pub fn rope_mrope_interleaved(
    input: &[f32],
    num_tokens: usize,
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    positions: [usize; 3],
    section: [usize; 3],
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
    let selector = mrope_component_selector(section, pairs);
    let log_theta = theta.ln();
    // Spelled exactly as `rope_neox_subdim` spells it, so the degenerate case
    // agrees to the bit rather than to a tolerance.
    let angles: Vec<f32> = (0..pairs)
        .map(|i| {
            let exponent = -((2 * i) as f32) / rotary_dim as f32;
            positions[selector[i] as usize] as f32 * (exponent * log_theta).exp()
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

/// ggml's YaRN ramp mix: `1` at the fast end (the shortest-wavelength pairs
/// keep the EXTRAPOLATED angle), `0` at the slow end (the longest-wavelength
/// pairs take the full `1/factor` interpolation), linear between. That
/// direction reads backwards off the YaRN paper's own prose, and a draft
/// that "fixed" it interpolated the fast end instead; it passed every
/// self-referential fixture and degraded long-position attention against
/// llama.cpp. The mixing is `extrap * (freq_scale * (1 - mix) + mix)`, so
/// mix here must be 1 exactly where ggml's `rope_yarn_ramp` is 1.
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

/// Llama 3.x rope scaling (`rope_scaling.rope_type == "llama3"`) as one
/// DIVISOR per rotated pair, the shape GGUF ships as `rope_freqs.weight`.
///
/// The angle for pair `p` at position `t` is then
/// `t * base^(-2p/rotary_dim) / factors[p]`, which is llama.cpp's reading of
/// `rope_freqs.weight` (`theta / freq_factors[i]` in `ggml_rope_ext`) and
/// the vector `convert_hf_to_gguf.py` writes from these same five numbers.
/// One representation for both intakes is the point: a GGUF Llama 3.1 file
/// carries ONLY the vector (no `rope.scaling.*` keys), so the HF path builds
/// the vector too rather than carrying the formula.
///
/// Bands, by wavelength `2*pi / freq`:
/// - shorter than `original_context / high_freq_factor`: divisor 1 (kept);
/// - longer than `original_context / low_freq_factor`: divisor `factor`;
/// - between: the smooth blend `1 / ((1 - s) / factor + s)`.
///
/// Computed in f64 and narrowed once, so the f32 result does not depend on
/// summation order. Returns `None` for parameters HF itself would reject
/// (non-positive factor or context, `high_freq_factor <= low_freq_factor`),
/// so a malformed config is refused rather than turned into a table.
#[must_use]
pub fn llama3_rope_factors(
    rotary_dim: usize,
    base: f64,
    factor: f64,
    low_freq_factor: f64,
    high_freq_factor: f64,
    original_context: f64,
) -> Option<Vec<f32>> {
    let valid = rotary_dim > 0
        && rotary_dim % 2 == 0
        && base > 1.0
        && factor > 0.0
        && low_freq_factor > 0.0
        && high_freq_factor > low_freq_factor
        && original_context > 0.0
        && [
            base,
            factor,
            low_freq_factor,
            high_freq_factor,
            original_context,
        ]
        .iter()
        .all(|v| v.is_finite());
    if !valid {
        return None;
    }
    let low_freq_wavelen = original_context / low_freq_factor;
    let high_freq_wavelen = original_context / high_freq_factor;
    let factors = (0..rotary_dim / 2)
        .map(|p| {
            let freq = base.powf(-((2 * p) as f64) / rotary_dim as f64);
            let wavelen = 2.0 * std::f64::consts::PI / freq;
            let divisor = if wavelen < high_freq_wavelen {
                1.0
            } else if wavelen > low_freq_wavelen {
                factor
            } else {
                let smooth = (original_context / wavelen - low_freq_factor)
                    / (high_freq_factor - low_freq_factor);
                1.0 / ((1.0 - smooth) / factor + smooth)
            };
            divisor as f32
        })
        .collect();
    Some(factors)
}

/// The per-pair angle table `rope_neox_freqs` binds, from a base and one
/// divisor per pair (see [`llama3_rope_factors`]). With every divisor 1 this
/// is exactly [`yarn_frequencies`]' unscaled table, which is what a test
/// pins so the two cannot drift apart.
///
/// # Panics
/// If `factors.len() != rotary_dim / 2`: a table one pair short rotates the
/// last pair with whatever the buffer holds past the end.
#[must_use]
pub fn rope_frequencies_with_factors(rotary_dim: usize, base: f32, factors: &[f32]) -> Vec<f32> {
    assert!(rotary_dim % 2 == 0, "rotary_dim must be even");
    assert_eq!(
        factors.len(),
        rotary_dim / 2,
        "one rope divisor per rotated pair"
    );
    factors
        .iter()
        .enumerate()
        .map(|(p, &divisor)| base.powf(-((2 * p) as f32) / rotary_dim as f32) / divisor)
        .collect()
}

#[cfg(test)]
mod llama3_tests {
    use super::*;

    /// Llama 3.1/3.2's published scaling: head_dim 128, theta 500000,
    /// factor 8, low 1, high 4, original 8192. Expected values were computed
    /// from HF's `_compute_llama3_parameters` formula in f64 and agree with
    /// the `rope_freqs.weight` llama.cpp's converter writes for Llama 3.1
    /// (29 ones, a six-step ramp, then 8s).
    #[test]
    fn llama31_factors_match_the_published_profile() {
        let f = llama3_rope_factors(128, 500000.0, 8.0, 1.0, 4.0, 8192.0).unwrap();
        assert_eq!(f.len(), 64);
        assert!(f[..29].iter().all(|&x| x == 1.0), "{:?}", &f[..29]);
        let ramp = [1.207484, 1.553415, 2.026313, 2.69453, 3.684253, 5.257327];
        for (got, want) in f[29..35].iter().zip(ramp) {
            assert!((got - want).abs() < 1e-5, "{got} vs {want}");
        }
        assert!(f[35..].iter().all(|&x| x == 8.0), "{:?}", &f[35..]);
    }

    /// The audio crate's tiny llama3 fixture
    /// (`crates/audio/testdata/lm/llama_llama3_rope`): head_dim 8, theta
    /// 10000, factor 4, original context 8. Pair 0 sits in the smooth band.
    #[test]
    fn tiny_fixture_factors() {
        let f = llama3_rope_factors(8, 10000.0, 4.0, 1.0, 4.0, 8.0).unwrap();
        let want = [std::f32::consts::PI, 4.0, 4.0, 4.0];
        for (got, want) in f.iter().zip(want) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
    }

    #[test]
    fn unit_divisors_reproduce_the_unscaled_table() {
        let ones = vec![1.0f32; 64];
        let got = rope_frequencies_with_factors(128, 500000.0, &ones);
        let want = yarn_frequencies(128, 500000.0, 0.0, 0, 0.0, 0.0).frequencies;
        assert_eq!(got, want);
    }

    #[test]
    fn divisors_divide_the_frequency() {
        let f = llama3_rope_factors(128, 500000.0, 8.0, 1.0, 4.0, 8192.0).unwrap();
        let scaled = rope_frequencies_with_factors(128, 500000.0, &f);
        let plain = rope_frequencies_with_factors(128, 500000.0, &[1.0; 64]);
        assert_eq!(scaled[0], plain[0]);
        assert!((scaled[63] * 8.0 - plain[63]).abs() <= plain[63] * 1e-6);
    }

    #[test]
    fn malformed_parameters_are_refused() {
        assert!(llama3_rope_factors(128, 500000.0, 0.0, 1.0, 4.0, 8192.0).is_none());
        assert!(llama3_rope_factors(128, 500000.0, 8.0, 4.0, 4.0, 8192.0).is_none());
        assert!(llama3_rope_factors(128, 500000.0, 8.0, 1.0, 4.0, 0.0).is_none());
        assert!(llama3_rope_factors(127, 500000.0, 8.0, 1.0, 4.0, 8192.0).is_none());
        assert!(llama3_rope_factors(128, f64::NAN, 8.0, 1.0, 4.0, 8192.0).is_none());
    }
}
