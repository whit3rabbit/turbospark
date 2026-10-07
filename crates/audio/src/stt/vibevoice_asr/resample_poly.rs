//! The scipy `resample_poly` frontend the pinned VibeVoice-ASR checkpoint
//! requires.
//!
//! mlx-audio resamples non-native input with `scipy.signal.resample_poly`
//! and a "kaiser_best" FIR (`mlx_audio/resample.py`): `firwin` with a
//! Kaiser window (64 zeros, rolloff 0.9475937167399596, beta
//! 14.769656459379492) applied at the upsampled rate, `padtype="edge"`.
//! The crate-level [`crate::resample`] kernels are torchaudio-compatible
//! sinc-Hann and linear designs, which are different filters, so this
//! family carries the exact scipy design.
//!
//! The port reproduces the scipy 1.17 algorithm literally: left-pad the
//! freshly designed filter by `down - half_len % down` zeros, edge-extend
//! the input by `(len(h_full) - 1) / 2` samples on each side, zero-stuff by
//! `up`, correlate with the filter, subsample by `down`, and drop the first
//! `n_pre_remove` outputs. Filter design and accumulation run in f64; the
//! output is rounded through f32 exactly like the reference
//! `.astype(np.float32)`.

use crate::error::AudioError;

/// Design parameters of `mlx_audio.resample._polyphase_filter`.
const NUM_ZEROS: usize = 64;
const ROLLOFF: f64 = 0.947593716_7399596;
const BETA: f64 = 14.769656459_379492;

/// Modified Bessel function of the first kind, order zero, as the power
/// series `sum (x/2)^(2k) / (k!)^2` in ascending powers, stopping once the
/// term is negligible against the sum.
fn bessel_i0(x: f64) -> f64 {
    let half = x / 2.0;
    let mut term = 1.0f64;
    let mut sum = 1.0f64;
    let mut k = 1.0f64;
    while term > sum * 1.0e-17 {
        term *= (half * half) / (k * k);
        sum += term;
        k += 1.0;
    }
    sum
}

/// `np.kaiser(num_taps, beta)`.
fn kaiser_window(num_taps: usize, beta: f64) -> Vec<f64> {
    let alpha = (num_taps - 1) as f64 / 2.0;
    let denom = bessel_i0(beta);
    (0..num_taps)
        .map(|n| {
            let ratio = (n as f64 - alpha) / alpha;
            let scaled = beta * (1.0 - ratio * ratio).max(0.0).sqrt();
            bessel_i0(scaled) / denom
        })
        .collect()
}

/// `scipy.signal.firwin(numtaps, cutoff, window=("kaiser", beta))` for the
/// type I (odd, symmetric) low-pass case with scale=True: a sinc at the
/// cutoff times the Kaiser window, divided by the DC sum.
pub fn firwin_kaiser_lowpass(num_taps: usize, cutoff: f64, beta: f64) -> Vec<f64> {
    let window = kaiser_window(num_taps, beta);
    let center = (num_taps - 1) as f64 / 2.0;
    let mut h: Vec<f64> = (0..num_taps)
        .map(|n| {
            let x = (n as f64 - center) * cutoff;
            // np.sinc: sin(pi x) / (pi x), one at zero.
            if x == 0.0 {
                1.0
            } else {
                let pi_x = std::f64::consts::PI * x;
                pi_x.sin() / pi_x
            }
        })
        .zip(window)
        .map(|(sinc, window)| sinc * window)
        .collect();
    let sum: f64 = h.iter().sum();
    for value in &mut h {
        *value /= sum;
    }
    h
}

/// The shared `mlx_audio.resample` filter for one rate pair: `firwin` at
/// the upsampled Nyquist with cutoff `rolloff / max(up, down)`.
pub fn mlx_audio_polyphase_fir(up: usize, down: usize) -> Vec<f64> {
    let max_rate = up.max(down);
    firwin_kaiser_lowpass(
        2 * NUM_ZEROS * max_rate + 1,
        ROLLOFF / max_rate as f64,
        BETA,
    )
}

/// `scipy.signal.resample_poly(input, up, down, window=fir, padtype="edge")`
/// rounded through f32, with `up`/`down` already coprime as the callers
/// pass reduced pairs.
pub fn resample_poly_edge(input: &[f32], up: usize, down: usize, fir: &[f64]) -> Vec<f32> {
    assert!(up >= 1 && down >= 1, "resample_poly rates must be positive");
    if up == down {
        return input.to_vec();
    }
    let n_in = input.len();
    // The filter is applied after zero-stuffing, so it gains a factor of up.
    let h: Vec<f64> = fir.iter().map(|value| value * up as f64).collect();
    let half_len = (h.len() - 1) / 2;
    let n_pre_pad = down - half_len % down;
    let mut n_post_pad = 0;
    let n_pre_remove = (half_len + n_pre_pad) / down;
    let n_out = (n_in * up).div_ceil(down);
    // Grow the filter tail until the padded correlation covers the whole
    // requested output (scipy's `while _output_len(...) < n_out + ...`).
    let output_len = |filter_len: usize, input_len: usize| {
        ((input_len - 1).saturating_mul(up) + filter_len - 1) / down + 1
    };
    while output_len(h.len() + n_pre_pad + n_post_pad, n_in) < n_out + n_pre_remove {
        n_post_pad += 1;
    }
    let mut h_full = vec![0.0f64; n_pre_pad];
    h_full.extend_from_slice(&h);
    h_full.extend(std::iter::repeat_n(0.0, n_post_pad));
    let full_len = h_full.len();

    let mut pad = (full_len - 1).div_ceil(up);
    while (pad * up) % down != 0 {
        pad += 1;
    }
    let xpad_len = n_in + 2 * pad;
    let stuffed_len = xpad_len * up;
    let mut stuffed = vec![0.0f64; stuffed_len];
    let first = f64::from(input[0]);
    for index in 0..pad {
        stuffed[index * up] = first;
    }
    for (index, value) in input.iter().enumerate() {
        stuffed[(index + pad) * up] = f64::from(*value);
    }
    let last = f64::from(*input.last().unwrap());
    for index in 0..pad {
        stuffed[(n_in + pad + index) * up] = last;
    }

    let offset = (pad * up) / down;
    let total = ((xpad_len - 1) * up + full_len - 1) / down + 1;
    let mut out = Vec::with_capacity(n_out);
    for m in 0..total {
        let base = m * down;
        let mut acc = 0.0f64;
        for (k, &tap) in h_full.iter().enumerate() {
            let Some(q) = base.checked_sub(k) else { break };
            if q % up == 0 && q < stuffed_len {
                acc += tap * stuffed[q];
            }
        }
        if m >= offset + n_pre_remove {
            out.push(acc as f32);
            if out.len() == n_out {
                break;
            }
        }
    }
    out
}

/// Resamples mono audio from `source_rate` to `target_rate` with the
/// mlx-audio polyphase design (reduced by their GCD first, like scipy).
pub fn resample_mono_polyphase(
    input: &[f32],
    source_rate: u32,
    target_rate: u32,
) -> Result<Vec<f32>, AudioError> {
    if source_rate == 0 || target_rate == 0 {
        return Err(AudioError::InvalidParameter {
            name: "sample rate".into(),
            value: format!("{source_rate} -> {target_rate}"),
            why: "resample rates must be positive".into(),
        });
    }
    if input.is_empty() || source_rate == target_rate {
        return Ok(input.to_vec());
    }
    let gcd = gcd(source_rate, target_rate);
    let up = target_rate as usize / gcd as usize;
    let down = source_rate as usize / gcd as usize;
    let fir = mlx_audio_polyphase_fir(up, down);
    Ok(resample_poly_edge(input, up, down, &fir))
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

#[cfg(test)]
mod tests {
    use super::{firwin_kaiser_lowpass, mlx_audio_polyphase_fir, resample_poly_edge};

    /// scipy 1.17 reference values for the shared design, generated with
    /// `scipy.signal.firwin(385, rolloff/3, window=("kaiser", beta))`:
    /// center tap 0.31586457129951045, fir[0] 5.467061006722385e-09,
    /// fir[1] 8.71801131881767e-09, fir[100] -0.000112990040196451, and an
    /// exactly unit DC sum.
    #[test]
    fn fir_head_and_dc_normalization_match_scipy() {
        let fir = mlx_audio_polyphase_fir(3, 2);
        assert_eq!(fir.len(), 385);
        assert!(
            (fir[192] - 0.315864571_29951045).abs() < 5.0e-16,
            "center {}",
            fir[192]
        );
        assert!(
            (fir[0] - 5.467_061_006_722_385e-9).abs() < 1.0e-20,
            "fir[0] {}",
            fir[0]
        );
        assert!(
            (fir[1] - 8.718_011_318_817_67e-9).abs() < 1.0e-20,
            "fir[1] {}",
            fir[1]
        );
        assert!(
            (fir[100] - (-1.129_900_401_964_51e-4)).abs() < 1.0e-14,
            "fir[100] {}",
            fir[100]
        );
        let sum: f64 = fir.iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-12);
        // The standalone design call reproduces the shared filter bitwise.
        let again = firwin_kaiser_lowpass(385, 0.947593716_7399596 / 3.0, 14.769656459_379492);
        for (got, want) in fir.iter().zip(&again) {
            assert_eq!(got.to_bits(), want.to_bits());
        }
    }

    #[test]
    fn identity_rates_copy_the_input() {
        let input = vec![0.25f32, -0.5, 0.75];
        assert_eq!(
            resample_poly_edge(&input, 1, 1, &mlx_audio_polyphase_fir(1, 1)),
            input
        );
    }

    /// 3/2 upsampling of a constant signal stays constant (edge extension
    /// and DC-pass filter), the cheapest end-to-end invariant of the
    /// scipy-aligned path.
    #[test]
    fn constant_signal_stays_constant_under_3_2() {
        let input = vec![0.5f32; 97];
        let out = resample_poly_edge(&input, 3, 2, &mlx_audio_polyphase_fir(3, 2));
        assert_eq!(out.len(), (97usize * 3).div_ceil(2));
        for value in &out {
            assert!((value - 0.5).abs() < 1.0e-5, "got {value}");
        }
    }

    /// Interleaved test tones (250 Hz and 3 kHz at 16 kHz) resampled to
    /// 24 kHz keep their frequency and amplitude within the passband
    /// ripple; the dominant tone's period is the structural check.
    #[test]
    fn tone_frequency_survives_16k_to_24k() {
        let n: usize = 1600;
        let input: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                0.4 * (2.0 * std::f32::consts::PI * 250.0 * t).sin()
                    + 0.1 * (2.0 * std::f32::consts::PI * 3_000.0 * t).sin()
            })
            .collect();
        let out = super::resample_mono_polyphase(&input, 16_000, 24_000).unwrap();
        assert_eq!(out.len(), (n * 3).div_ceil(2));
        // Peaks of the dominant 250 Hz tone (amplitude 0.4, above the 0.1
        // secondary tone) repeat every 4 ms, now at 24 kHz: 48 samples.
        let mut peaks = Vec::new();
        for i in 1..out.len() - 1 {
            if out[i] > 0.3
                && out[i - 1] <= out[i]
                && out[i] >= out[i + 1]
                && peaks.last().is_none_or(|&last| i - last > 48)
            {
                peaks.push(i);
            }
        }
        assert!(peaks.len() >= 4, "too few dominant peaks: {peaks:?}");
        for window in peaks.windows(2) {
            let spacing = (window[1] - window[0]) as f64 / 24_000.0;
            assert!((spacing - 0.004).abs() < 2.0e-4, "spacing {spacing}");
        }
    }

    #[test]
    fn zero_input_refuses_and_empty_input_copies() {
        assert!(super::resample_mono_polyphase(&[0.0], 0, 24_000).is_err());
        assert!(super::resample_mono_polyphase(&[], 16_000, 24_000)
            .unwrap()
            .is_empty());
    }
}
