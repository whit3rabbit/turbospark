//! The BS.1770 loudness stage of the outetts `DacInterface`.
//!
//! Port of `process_audio_array` and the `mlx_audio/dsp.py` helpers it
//! calls (`integrated_loudness`, `normalize_loudness`, `normalize_peak`,
//! `_k_weight_audio`, `_biquad_coefficients`, `lfilter`) at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/dsp.py).
//! Everything runs in f64 like the numpy reference and is portable (no
//! Metal or macOS API).
//!
//! Shape contract: the reference hands `process_audio_array` a
//! `[1, 1, samples]` array whose channel branch takes the squeeze path
//! down to the mono samples (MLX convs are channels-last, so the
//! reconstructed chunk arrives `[1, samples, 1]` and the decode path
//! swapaxes it into the squeeze branch first). This port takes the mono
//! samples directly and runs the identical pad, measure, normalize,
//! peak-limit, and trim steps. One documented deviation in
//! [`super::decode_codes_to_audio`]: the reference concatenates its
//! 4096-code decode chunks along the size-1 channel axis, which only
//! works while the decode is a single chunk.

/// K-weighting high-shelf frequency (Hz).
const SHELF_FREQ: f64 = 1681.974450955533;
/// K-weighting high-shelf Q.
const SHELF_Q: f64 = 0.7071752369554196;
/// K-weighting high-shelf gain (dB).
const SHELF_GAIN_DB: f64 = 3.999843853973347;
/// K-weighting high-pass frequency (Hz).
const HIGHPASS_FREQ: f64 = 38.13547087602444;
/// K-weighting high-pass Q.
const HIGHPASS_Q: f64 = 0.5003270373238773;

/// `process_audio_array` for mono samples: pad to one loudness block,
/// measure integrated loudness, normalize to the target LUFS, apply the
/// peak limit, and trim the pad. The reference defaults pass through the
/// outetts call site (`target_loudness` -18, `peak_limit` -1 dB,
/// `block_size` 0.4 s).
pub fn process_audio_array(
    samples: &[f32],
    sample_rate: u32,
    target_loudness: f32,
    peak_limit_db: f32,
    block_size: f32,
) -> Vec<f32> {
    let original_length = samples.len();
    if original_length == 0 {
        return Vec::new();
    }
    let min_samples = (block_size * sample_rate as f32) as usize;
    let mut padded: Vec<f64> = samples.iter().map(|&x| f64::from(x)).collect();
    padded.resize(min_samples.max(original_length), 0.0);

    let measured = integrated_loudness(&padded, sample_rate, block_size);
    let gain = 10.0f64.powf((f64::from(target_loudness) - measured) / 20.0);
    let mut normalized: Vec<f64> = padded.iter().map(|&x| gain * x).collect();

    // Peak limiting: `normalize_peak` scales to the target dBFS peak.
    let peak_value = normalized.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
    let threshold = 10.0f64.powf(f64::from(peak_limit_db) / 20.0);
    if peak_value > threshold {
        let peak_gain = threshold / peak_value;
        for value in &mut normalized {
            *value *= peak_gain;
        }
    }

    normalized[..original_length]
        .iter()
        .map(|&x| x as f32)
        .collect()
}

/// The gated integrated loudness (LUFS) of mono samples, the BS.1770
/// K-weighted block-gating measurement. Returns negative infinity for
/// digital silence (the reference `log10(0)` path), which
/// [`process_audio_array`] folds into the peak limiter.
pub fn integrated_loudness(samples: &[f64], sample_rate: u32, block_size: f32) -> f64 {
    let rate = f64::from(sample_rate);
    let weighted = k_weight_audio(samples, sample_rate);
    let block_samples = (f64::from(block_size) * rate).round() as usize;
    let hop_samples = (f64::from(block_size) * 0.25 * rate).round() as usize;
    if weighted.len() < block_samples || block_samples == 0 || hop_samples == 0 {
        return f64::NEG_INFINITY;
    }
    let num_blocks = 1 + (weighted.len() - block_samples) / hop_samples;
    let mut mean_square = vec![0.0f64; num_blocks];
    for (block_index, slot) in mean_square.iter_mut().enumerate() {
        let lower = block_index * hop_samples;
        let upper = lower + block_samples;
        let sum: f64 = weighted[lower..upper].iter().map(|&x| x * x).sum();
        *slot = sum / block_samples as f64;
    }
    // Mono channel gain 1.0 (the reference table's first entry).
    let block_loudness: Vec<f64> = mean_square
        .iter()
        .map(|&ms| -0.691 + 10.0 * ms.max(0.0).log10())
        .collect();
    let absolute_threshold = -70.0;
    let gated: Vec<usize> = block_loudness
        .iter()
        .enumerate()
        .filter(|(_, &loudness)| loudness > absolute_threshold)
        .map(|(index, _)| index)
        .collect();
    let gated_mean_square = if gated.is_empty() {
        0.0
    } else {
        gated.iter().map(|&i| mean_square[i]).sum::<f64>() / gated.len() as f64
    };
    let relative_threshold = -0.691 + 10.0 * gated_mean_square.max(0.0).log10() - 10.0;
    let gated: Vec<usize> = block_loudness
        .iter()
        .enumerate()
        .filter(|(_, &loudness)| loudness > relative_threshold && loudness > absolute_threshold)
        .map(|(index, _)| index)
        .collect();
    let gated_mean_square = if gated.is_empty() {
        0.0
    } else {
        gated.iter().map(|&i| mean_square[i]).sum::<f64>() / gated.len() as f64
    };
    -0.691 + 10.0 * gated_mean_square.max(0.0).log10()
}

/// The two K-weighting biquad stages in series, computed in f64 like the
/// reference (`weighted = np.array(data, dtype=np.float64)`).
fn k_weight_audio(samples: &[f64], sample_rate: u32) -> Vec<f64> {
    let (shelf_b, shelf_a) =
        biquad_coefficients(SHELF_GAIN_DB, SHELF_Q, SHELF_FREQ, sample_rate, true);
    let (highpass_b, highpass_a) =
        biquad_coefficients(0.0, HIGHPASS_Q, HIGHPASS_FREQ, sample_rate, false);
    let after_shelf = lfilter(&shelf_b, &shelf_a, samples);
    lfilter(&highpass_b, &highpass_a, &after_shelf)
}

/// `_biquad_coefficients`: the BS.1770 K-weighting stages. The shelf
/// numerator follows the libebur128-style design (with the
/// `shelf_gain ** 0.4996667741545416` midpoint exponent); the high-pass
/// numerator is exactly `[1, -2, 1]`.
fn biquad_coefficients(
    gain_db: f64,
    q_factor: f64,
    center_freq: f64,
    sample_rate: u32,
    high_shelf: bool,
) -> ([f64; 3], [f64; 3]) {
    let k = (std::f64::consts::PI * center_freq / f64::from(sample_rate)).tan();
    let denominator = 1.0 + k / q_factor + k * k;
    let a = [
        1.0,
        2.0 * (k * k - 1.0) / denominator,
        (1.0 - k / q_factor + k * k) / denominator,
    ];
    let b = if high_shelf {
        let shelf_gain = 10.0f64.powf(gain_db / 20.0);
        let mid_gain = shelf_gain.powf(0.4996667741545416);
        [
            (shelf_gain + mid_gain * k / q_factor + k * k) / denominator,
            2.0 * (k * k - shelf_gain) / denominator,
            (shelf_gain - mid_gain * k / q_factor + k * k) / denominator,
        ]
    } else {
        [1.0, -2.0, 1.0]
    };
    (b, a)
}

/// `lfilter`: the direct-form II transposed recurrence, normalized by
/// `a[0]`. Ported verbatim from the reference helper (state length 2 for
/// the biquads, with the tail assignment it performs after the loop).
fn lfilter(b: &[f64], a: &[f64], data: &[f64]) -> Vec<f64> {
    let b0 = b[0] / a[0];
    let b: Vec<f64> = b.iter().map(|&x| x / a[0]).collect();
    let a: Vec<f64> = a.iter().map(|&x| x / a[0]).collect();
    let state_len = a.len().max(b.len()) - 1;
    if state_len == 0 {
        return data.iter().map(|&x| b0 * x).collect();
    }
    let mut state = vec![0.0f64; state_len];
    let mut out = Vec::with_capacity(data.len());
    for &sample in data {
        let output = b[0] * sample + state[0];
        for i in 1..state_len {
            let feedforward = if i < b.len() { b[i] * sample } else { 0.0 };
            let feedback = if i < a.len() { a[i] * output } else { 0.0 };
            state[i - 1] = state[i] + feedforward - feedback;
        }
        let feedforward = if state_len < b.len() {
            b[state_len] * sample
        } else {
            0.0
        };
        let feedback = if state_len < a.len() {
            a[state_len] * output
        } else {
            0.0
        };
        state[state_len - 1] = feedforward - feedback;
        out.push(output);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 440 Hz tone at 0.9 amplitude measures about -4.57 LUFS with
    /// the BS.1770 K-weighting; the value is the reference
    /// `integrated_loudness` on the same f64 signal (measured parity
    /// 1.6e-14), so a biquad or gating regression moves the number.
    #[test]
    fn full_scale_tone_measures_near_zero_lufs() {
        let rate = 24000;
        let n = rate; // one second
        let samples: Vec<f64> = (0..n)
            .map(|i| 0.9 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / rate as f64).sin())
            .collect();
        let loudness = integrated_loudness(&samples, rate, 0.4);
        assert!(
            (loudness - -4.566574915509744).abs() < 1e-6,
            "tone loudness {loudness}"
        );
    }

    /// Silence measures negative infinity (the reference `log10(0)` path).
    #[test]
    fn silence_measures_negative_infinity() {
        let samples = vec![0.0f64; 24000];
        let loudness = integrated_loudness(&samples, 24000, 0.4);
        assert!(loudness.is_infinite() && loudness < 0.0);
    }

    /// A signal with a high crest factor (two impulses in silence) is
    /// loudness-normalized past full scale and the peak limiter trims it
    /// to exactly the -1 dBFS target (0.8912509), the reference
    /// `process_audio_array` output for the same signal.
    #[test]
    fn high_crest_signal_is_peak_limited() {
        let rate = 24000u32;
        let mut samples = vec![0.0f32; rate as usize];
        samples[500] = 0.99;
        samples[501] = -0.99;
        let out = process_audio_array(&samples, rate, -18.0, -1.0, 0.4);
        assert_eq!(out.len(), samples.len());
        let peak = out.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        assert!((peak - 0.8912509).abs() < 1e-4, "peak {peak}");
    }

    /// Zero-length input stays zero-length (no loudness call).
    #[test]
    fn empty_input_stays_empty() {
        assert!(process_audio_array(&[], 24000, -18.0, -1.0, 0.4).is_empty());
    }
}
