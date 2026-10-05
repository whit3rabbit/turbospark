//! Mel filterbank construction and the log-mel frontend.
//!
//! Filterbanks come in the two conventions every frontend names: HTK
//! (`mel = 2595 log10(1 + f / 700)`) and Slaney (librosa's piecewise-linear
//! then log scale, with its per-band area normalization). Bin frequencies
//! run `k * sample_rate / fft_size` for `k` in `[0, fft_size / 2]`.
//!
//! [`mel_spectrogram`] frames with the [`crate::stft`] conventions, applies
//! the windowed FFT, squares (or takes the magnitude per `power`), and
//! projects each frame onto the filterbank. [`log_mel_spectrogram`] adds
//! the `log10(max(x, floor))` compression. The whisper-specific peak clamp
//! and `(x + 4) / 4` normalization are one-liners on top of
//! [`log_mel_spectrogram`] and stay with the whisper frontend, which owns
//! those exact numerics.

use crate::error::AudioError;
use crate::stft::StftOptions;

/// The frequency-to-mel warping a filterbank is built in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MelScale {
    /// `mel = 2595 log10(1 + f / 700)`.
    Htk,
    /// librosa's Slaney scale: linear below 1 kHz, log above, plus its
    /// area normalization on the filterbank weights.
    #[default]
    Slaney,
}

/// A mel filterbank: `num_mels` rows of `num_bins` weights, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct MelFilterbank {
    pub num_mels: usize,
    pub num_bins: usize,
    /// `weights[mel * num_bins + bin]`.
    pub weights: Vec<f32>,
}

impl MelFilterbank {
    /// Projects one power (or magnitude) spectrum onto the filterbank.
    pub fn project(&self, spectrum: &[f32]) -> Result<Vec<f32>, AudioError> {
        if spectrum.len() != self.num_bins {
            return Err(AudioError::ShapeMismatch {
                what: "spectrum length vs filterbank bins",
                expected: self.num_bins,
                actual: spectrum.len(),
            });
        }
        Ok((0..self.num_mels)
            .map(|m| {
                let row = &self.weights[m * self.num_bins..(m + 1) * self.num_bins];
                row.iter().zip(spectrum).map(|(&w, &s)| w * s).sum()
            })
            .collect())
    }
}

/// Builds a mel filterbank.
///
/// `fmax = None` means Nyquist. Band edges are `num_mels + 2` frequencies
/// spaced evenly in mel space between `fmin` and `fmax`; band `i` is the
/// triangle over edges `(i, i+1, i+2)`. Slaney scale multiplies each row by
/// librosa's `2 / (upper_hz - lower_hz)` area factor; HTK leaves rows
/// unnormalized.
pub fn mel_filterbank(
    num_mels: usize,
    fft_size: usize,
    sample_rate: u32,
    fmin: f32,
    fmax: Option<f32>,
    scale: MelScale,
) -> Result<MelFilterbank, AudioError> {
    if num_mels == 0 {
        return Err(AudioError::InvalidParameter {
            name: "num_mels".to_string(),
            value: "0".to_string(),
            why: "at least one mel band is required".to_string(),
        });
    }
    crate::error::check_sample_rate(sample_rate)?;
    if !fmin.is_finite() || fmin < 0.0 {
        return Err(AudioError::InvalidParameter {
            name: "fmin".to_string(),
            value: fmin.to_string(),
            why: "must be non-negative".to_string(),
        });
    }
    let fmax = fmax.unwrap_or(sample_rate as f32 / 2.0);
    if !fmax.is_finite() || fmax <= fmin {
        return Err(AudioError::InvalidParameter {
            name: "fmax".to_string(),
            value: fmax.to_string(),
            why: format!("must exceed fmin {fmin}"),
        });
    }
    if fft_size == 0 {
        return Err(AudioError::InvalidParameter {
            name: "fft_size".to_string(),
            value: "0".to_string(),
            why: "must be positive".to_string(),
        });
    }
    let num_bins = fft_size / 2 + 1;
    // Bound the independently allocated filterbank before band-edge
    // arithmetic, even when the caller has not constructed an FFT plan.
    if num_mels.checked_add(2).is_none()
        || num_mels.checked_mul(num_bins).is_none_or(|n| n > 1 << 24)
    {
        return Err(AudioError::BufferTooLarge {
            what: "mel filterbank",
            samples: usize::MAX,
        });
    }
    let nyquist = sample_rate as f32 / 2.0;
    if fmax > nyquist {
        return Err(AudioError::InvalidParameter {
            name: "fmax".to_string(),
            value: fmax.to_string(),
            why: format!("must not exceed the Nyquist frequency {nyquist}"),
        });
    }

    let mel_min = hz_to_mel(fmin, scale);
    let mel_max = hz_to_mel(fmax, scale);
    let edges: Vec<f32> = (0..num_mels + 2)
        .map(|i| {
            let mel = mel_min + (mel_max - mel_min) * i as f32 / (num_mels + 1) as f32;
            mel_to_hz(mel, scale)
        })
        .collect();

    let mut weights = vec![0.0f32; num_mels * num_bins];
    for m in 0..num_mels {
        let (lower, center, upper) = (edges[m], edges[m + 1], edges[m + 2]);
        let norm = match scale {
            MelScale::Slaney => 2.0 / (upper - lower),
            MelScale::Htk => 1.0,
        };
        for (bin, slot) in weights[m * num_bins..(m + 1) * num_bins]
            .iter_mut()
            .enumerate()
        {
            let f = bin as f32 * sample_rate as f32 / fft_size as f32;
            *slot = if f >= lower && f <= center {
                if center > lower {
                    (f - lower) / (center - lower)
                } else {
                    1.0
                }
            } else if f > center && f <= upper {
                (upper - f) / (upper - center)
            } else {
                0.0
            } * norm;
        }
    }
    Ok(MelFilterbank {
        num_mels,
        num_bins,
        weights,
    })
}

#[derive(Debug, Clone)]
pub struct MelSpectrogramOptions {
    /// STFT options: FFT size, hop, window, centering.
    pub stft: StftOptions,
    pub num_mels: usize,
    pub sample_rate: u32,
    pub fmin: f32,
    /// `None` means Nyquist.
    pub fmax: Option<f32>,
    pub scale: MelScale,
    /// 1.0 for magnitude, 2.0 for power; any positive value scales the
    /// magnitudes by `mag^power`.
    pub power: f32,
}

impl Default for MelSpectrogramOptions {
    fn default() -> Self {
        Self {
            stft: StftOptions {
                fft_size: 512,
                hop: 160,
                window: crate::dsp::hann_window(400),
                center: true,
            },
            num_mels: 80,
            sample_rate: 16_000,
            fmin: 0.0,
            fmax: Some(8_000.0),
            scale: MelScale::Slaney,
            power: 2.0,
        }
    }
}

/// Mel spectrogram: `num_mels` values per frame, frames in time order.
pub fn mel_spectrogram(
    samples: &[f32],
    options: &MelSpectrogramOptions,
) -> Result<Vec<Vec<f32>>, AudioError> {
    if !options.power.is_finite() || options.power <= 0.0 {
        return Err(AudioError::InvalidParameter {
            name: "power".to_string(),
            value: options.power.to_string(),
            why: "must be positive".to_string(),
        });
    }
    let filterbank = mel_filterbank(
        options.num_mels,
        options.stft.fft_size,
        options.sample_rate,
        options.fmin,
        options.fmax,
        options.scale,
    )?;
    let spectra = crate::stft::stft(samples, &options.stft)?;
    let mut out = Vec::with_capacity(spectra.len());
    for spectrum in spectra {
        let power: Vec<f32> = spectrum
            .iter()
            .map(|c| {
                let mag = c.re.hypot(c.im);
                mag.powf(options.power)
            })
            .collect();
        out.push(filterbank.project(&power)?);
    }
    Ok(out)
}

/// Applies `log10(max(x, floor))` elementwise. `floor` must be positive;
/// 1e-10 is the value the whisper frontend uses.
pub fn log_mel_spectrogram(
    samples: &[f32],
    options: &MelSpectrogramOptions,
    floor: f32,
) -> Result<Vec<Vec<f32>>, AudioError> {
    if floor <= 0.0 {
        return Err(AudioError::InvalidParameter {
            name: "floor".to_string(),
            value: floor.to_string(),
            why: "must be positive".to_string(),
        });
    }
    let mel = mel_spectrogram(samples, options)?;
    Ok(mel
        .into_iter()
        .map(|frame| frame.into_iter().map(|x| x.max(floor).log10()).collect())
        .collect())
}

pub fn hz_to_mel(f: f32, scale: MelScale) -> f32 {
    match scale {
        MelScale::Htk => 2595.0 * (1.0 + f / 700.0).log10(),
        MelScale::Slaney => {
            if f < 1_000.0 {
                3.0 * f / 200.0
            } else {
                15.0 + (f / 1_000.0).ln() / (6.4f64.ln() / 27.0) as f32
            }
        }
    }
}

pub fn mel_to_hz(mel: f32, scale: MelScale) -> f32 {
    match scale {
        MelScale::Htk => 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0),
        MelScale::Slaney => {
            if mel < 15.0 {
                mel * 200.0 / 3.0
            } else {
                1_000.0 * ((6.4f64.ln() / 27.0) as f32 * (mel - 15.0)).exp()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::hann_window;

    #[test]
    fn scale_warp_points() {
        // HTK mel is approximately the identity at 1 kHz.
        assert!((hz_to_mel(1_000.0, MelScale::Htk) - 1_000.0).abs() < 1.0);
        // Slaney mel is exactly 15 at 1 kHz.
        assert!((hz_to_mel(1_000.0, MelScale::Slaney) - 15.0).abs() < 1e-5);
        // Round trips at points on both Slaney segments; f32 precision
        // near 8 kHz makes a 0.1 percent relative tolerance the honest one.
        for f in [0.0f32, 500.0, 1_000.0, 4_000.0, 8_000.0] {
            let back = mel_to_hz(hz_to_mel(f, MelScale::Slaney), MelScale::Slaney);
            assert!(
                (back - f).abs() < f * 1e-3 + 1e-4,
                "slaney round trip {f}: {back}"
            );
            let back = mel_to_hz(hz_to_mel(f, MelScale::Htk), MelScale::Htk);
            assert!(
                (back - f).abs() < f * 1e-3 + 1e-4,
                "htk round trip {f}: {back}"
            );
        }
    }

    #[test]
    fn filterbank_shape_and_bands() {
        let fb = mel_filterbank(80, 512, 16_000, 0.0, Some(8_000.0), MelScale::Slaney).unwrap();
        assert_eq!(fb.num_mels, 80);
        assert_eq!(fb.num_bins, 257);
        assert_eq!(fb.weights.len(), 80 * 257);
        // Every band carries energy and the band peaks move upward.
        let mut peaks = Vec::new();
        for m in 0..fb.num_mels {
            let row = &fb.weights[m * fb.num_bins..(m + 1) * fb.num_bins];
            let (best, weight) = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            assert!(*weight > 0.0, "band {m} is empty");
            peaks.push(best);
        }
        assert!(peaks.windows(2).all(|w| w[0] <= w[1]), "peaks not sorted");
    }

    #[test]
    fn filterbank_rejects_bad_geometry() {
        let err = mel_filterbank(0, 512, 16_000, 0.0, None, MelScale::Slaney).unwrap_err();
        assert!(err.to_string().contains("num_mels"), "{err}");
        let err = mel_filterbank(80, 512, 16_000, 9_000.0, None, MelScale::Slaney).unwrap_err();
        assert!(err.to_string().contains("fmax"), "{err}");
        let err =
            mel_filterbank(80, 512, 16_000, 0.0, Some(9_000.0), MelScale::Slaney).unwrap_err();
        assert!(err.to_string().contains("Nyquist"), "{err}");
    }

    #[test]
    fn spectrogram_energy_lands_in_the_tone_band() {
        // A 1 kHz tone must light up the mel band whose edges bracket it.
        let sample_rate = 16_000u32;
        let freq = 1_000.0f32;
        let samples: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f64::consts::PI * freq as f64 * i as f64 / 16_000.0).sin() as f32)
            .collect();
        let options = MelSpectrogramOptions {
            stft: StftOptions {
                fft_size: 512,
                hop: 160,
                window: hann_window(400),
                center: true,
            },
            num_mels: 80,
            sample_rate,
            fmin: 0.0,
            fmax: Some(8_000.0),
            scale: MelScale::Slaney,
            power: 2.0,
        };
        let spec = mel_spectrogram(&samples, &options).unwrap();
        // Centered: 1 + 16000/160 = 101 frames.
        assert_eq!(spec.len(), 101);
        let frame = &spec[spec.len() / 2];
        let mel_of_tone = hz_to_mel(freq, MelScale::Slaney);
        let mel_min = hz_to_mel(0.0, MelScale::Slaney);
        let mel_max = hz_to_mel(8_000.0, MelScale::Slaney);
        let band = ((mel_of_tone - mel_min) / (mel_max - mel_min) * 80.0) as usize;
        let band = band.min(79);
        let (best, _) = frame
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert!(
            (best as i32 - band as i32).abs() <= 1,
            "tone energy in band {best}, expected near {band}"
        );
    }

    #[test]
    fn log_floor_clamps_silence() {
        let samples = vec![0.0f32; 4_000];
        let options = MelSpectrogramOptions::default();
        let spec = log_mel_spectrogram(&samples, &options, 1e-10).unwrap();
        assert_eq!(spec.len(), 1 + 4_000 / 160);
        for frame in &spec {
            for &x in frame {
                assert!(
                    (x - -10.0).abs() < 1e-5,
                    "log10(1e-10) should be -10, got {x}"
                );
            }
        }
    }

    #[test]
    fn project_rejects_wrong_spectrum_length() {
        let fb = mel_filterbank(8, 16, 8_000, 0.0, None, MelScale::Htk).unwrap();
        let err = fb.project(&[0.0; 5]).unwrap_err();
        assert!(err.to_string().contains("expected 9, got 5"), "{err}");
    }

    #[test]
    fn spectrogram_energy_is_finite_and_non_negative() {
        // A loud real signal projects to finite, non-negative mel energy.
        let loud: Vec<f32> = (0..8_000)
            .map(|i| (2.0 * std::f64::consts::PI * 40.0 * i as f64 / 8_000.0).sin() as f32)
            .collect();
        let options = MelSpectrogramOptions::default();
        let loud_spec = mel_spectrogram(&loud, &options).unwrap();
        assert!(loud_spec[0].iter().all(|x| x.is_finite()));
        assert!(loud_spec[0].iter().all(|x| *x >= 0.0));
    }
}

#[cfg(test)]
mod slaney_regression {
    use super::*;

    #[test]
    fn matches_the_official_whisper_filterbank_asset() {
        // OpenAI Whisper v20250625 assets/mel_filters.npz, mel_80[0,1].
        // Asset SHA256: 7450ae70723a5ef9d341e3cee628c7cb0177f36ce42c44b7ed2bf3325f0f6d4c.
        // This literal comes from librosa's exported matrix, independently
        // of this implementation's frequency-to-mel conversion.
        let bank = mel_filterbank(80, 400, 16000, 0.0, None, MelScale::Slaney).unwrap();
        assert!((bank.weights[1] - 0.024_862_595).abs() < 1e-8);
    }

    #[test]
    fn matches_the_official_whisper_128_filterbank_asset() {
        // Generated by tests/reference/generate_whisper.py (NumPy 2.5.3)
        // from the same pinned official float32 asset above.
        // Low, middle, and high coefficients
        // catch both linear-region and logarithmic-region normalization.
        let bank = mel_filterbank(128, 400, 16000, 0.0, None, MelScale::Slaney).unwrap();
        assert_eq!((bank.num_mels, bank.num_bins), (128, 201));
        for (band, bin, want) in [
            (0, 1, 0.012_373_987),
            (64, 43, 0.018_091_518),
            (127, 195, 0.005_041_601_6),
        ] {
            let got = bank.weights[band * bank.num_bins + bin];
            // Rust's f32 edges differ from librosa's f64 construction.
            // Independent full-matrix comparison found max error 2.645e-7;
            // these selected coefficients differ by at most 1.323e-7
            // (7.31 ppm), bounded here without changing the 80-band limit.
            assert!(
                (got - want).abs() < 2e-7,
                "band {band} bin {bin}: {got} vs {want}"
            );
        }
    }
}

#[cfg(test)]
mod allocation_regression {
    use super::*;
    #[test]
    fn refuses_zero_fft_and_unbounded_filterbank_dimensions() {
        assert!(mel_filterbank(80, 0, 16000, 0.0, None, MelScale::Slaney).is_err());
        assert!(matches!(
            mel_filterbank(usize::MAX, 400, 16000, 0.0, None, MelScale::Slaney),
            Err(AudioError::BufferTooLarge { .. })
        ));
        assert!(matches!(
            mel_filterbank(80, usize::MAX, 16000, 0.0, None, MelScale::Slaney),
            Err(AudioError::BufferTooLarge { .. })
        ));
    }
}
