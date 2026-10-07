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

use crate::dsp::TableCache;
use crate::error::AudioError;
use crate::stft::StftOptions;
use std::sync::Arc;

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

/// A filterbank plus the nonzero span of each triangular row, so a frame
/// projection touches a few dozen bins per band instead of all of them.
///
/// [`MelFilterbank::project`] is the dense reference. [`Self::project_into`]
/// returns the same bits for every input, including the sign of a zero
/// result and non-finite spectra (see its comments for why skipping the
/// exactly-zero weights is only safe under those two conditions).
#[derive(Debug, Clone)]
pub struct MelProjector {
    bank: Arc<MelFilterbank>,
    /// Per band: `[first, end)` covering every weight whose bits are not
    /// +0.0. `(0, 0)` for an all-zero row.
    spans: Vec<(usize, usize)>,
}

impl MelProjector {
    pub fn new(bank: Arc<MelFilterbank>) -> Self {
        let spans = (0..bank.num_mels)
            .map(|m| {
                let row = &bank.weights[m * bank.num_bins..(m + 1) * bank.num_bins];
                // Bits, not `!= 0.0`: -0.0 and NaN weights stay in the span.
                let first = row.iter().position(|w| w.to_bits() != 0);
                match first {
                    None => (0, 0),
                    Some(first) => {
                        let last = row.iter().rposition(|w| w.to_bits() != 0).unwrap_or(first);
                        (first, last + 1)
                    }
                }
            })
            .collect();
        Self { bank, spans }
    }

    pub fn filterbank(&self) -> &MelFilterbank {
        &self.bank
    }

    /// Projects one spectrum into `out` (cleared, then `num_mels` values),
    /// bit-identical to [`MelFilterbank::project`].
    ///
    /// The dense projection folds `w * s` over every bin from the neutral
    /// element of `Iterator::sum`. A bin outside a row's span has weight
    /// +0.0, so for finite `s` its term is exactly +-0.0 and can change the
    /// accumulator only through the sign of a zero. That sign is rebuilt from
    /// two facts about the spectrum, found once per frame: the first and
    /// last bins whose sign bit is clear (a +0.0 term turns a -0.0
    /// accumulator into +0.0; a -0.0 term never changes anything but a
    /// -0.0 start). A non-finite bin makes `0.0 * s` NaN in the dense sum,
    /// so such a frame takes the dense path.
    pub fn project_into(&self, spectrum: &[f32], out: &mut Vec<f32>) -> Result<(), AudioError> {
        let bank = &*self.bank;
        if spectrum.len() != bank.num_bins {
            return Err(AudioError::ShapeMismatch {
                what: "spectrum length vs filterbank bins",
                expected: bank.num_bins,
                actual: spectrum.len(),
            });
        }
        let n = bank.num_bins;
        out.clear();
        let mut first_nonneg = n;
        let mut last_nonneg: Option<usize> = None;
        let mut finite = true;
        for (i, &s) in spectrum.iter().enumerate() {
            if !s.is_finite() {
                finite = false;
                break;
            }
            if s.is_sign_positive() {
                if last_nonneg.is_none() {
                    first_nonneg = i;
                }
                last_nonneg = Some(i);
            }
        }
        if !finite {
            out.extend((0..bank.num_mels).map(|m| {
                let row = &bank.weights[m * n..(m + 1) * n];
                row.iter().zip(spectrum).map(|(&w, &s)| w * s).sum::<f32>()
            }));
            return Ok(());
        }
        // The neutral element `Iterator::sum` starts from (-0.0 on current
        // toolchains, +0.0 on older ones): read it from the same impl rather
        // than assuming it.
        let init: f32 = std::iter::empty::<f32>().sum();
        let init_is_neg_zero = init.to_bits() == (-0.0f32).to_bits();
        for (m, &(first, end)) in self.spans.iter().enumerate() {
            let row = &bank.weights[m * n..(m + 1) * n];
            let mut acc = init;
            // Leading zero terms: a -0.0 start survives only if every one
            // of them is -0.0, i.e. no non-negative bin before the span.
            if init_is_neg_zero && first > first_nonneg {
                acc = 0.0;
            }
            for i in first..end {
                acc += row[i] * spectrum[i];
            }
            // Trailing zero terms: same rule for a -0.0 accumulator.
            if acc.to_bits() == (-0.0f32).to_bits() && last_nonneg.is_some_and(|l| l >= end) {
                acc = 0.0;
            }
            out.push(acc);
        }
        Ok(())
    }
}

/// Filterbanks above this many weights are built per call, not cached.
const MAX_CACHED_FILTERBANK_WEIGHTS: usize = 1 << 20;

/// Returns a shared [`MelProjector`] for the exact filterbank parameters,
/// building it on first use.
///
/// Building the bank costs a per-cell f32 division across `num_mels x
/// num_bins` cells, and frontends used to redo it on every utterance. The
/// bank is a pure function of its parameters, so a cached one is the same
/// bits as a fresh [`mel_filterbank`]. Invalid parameters return the same
/// error as `mel_filterbank` and are never cached.
pub fn mel_projector_cached(
    num_mels: usize,
    fft_size: usize,
    sample_rate: u32,
    fmin: f32,
    fmax: Option<f32>,
    scale: MelScale,
) -> Result<Arc<MelProjector>, AudioError> {
    type Key = (usize, usize, u32, u32, Option<u32>, bool);
    static BANKS: TableCache<Key, MelProjector> = TableCache::new(16);
    let key: Key = (
        num_mels,
        fft_size,
        sample_rate,
        fmin.to_bits(),
        fmax.map(f32::to_bits),
        scale == MelScale::Htk,
    );
    // The size bound is checked from the parameters, not the built bank, so
    // an oversized request never enters the map.
    let cacheable = num_mels
        .checked_mul(fft_size / 2 + 1)
        .is_some_and(|n| n <= MAX_CACHED_FILTERBANK_WEIGHTS);
    BANKS.get_or_build(key, cacheable, || {
        let bank = mel_filterbank(num_mels, fft_size, sample_rate, fmin, fmax, scale)?;
        Ok(MelProjector::new(Arc::new(bank)))
    })
}

/// [`mel_filterbank`] through the shared cache; see [`mel_projector_cached`].
pub fn mel_filterbank_cached(
    num_mels: usize,
    fft_size: usize,
    sample_rate: u32,
    fmin: f32,
    fmax: Option<f32>,
    scale: MelScale,
) -> Result<Arc<MelFilterbank>, AudioError> {
    let projector = mel_projector_cached(num_mels, fft_size, sample_rate, fmin, fmax, scale)?;
    Ok(Arc::clone(&projector.bank))
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
    let projector = mel_projector_cached(
        options.num_mels,
        options.stft.fft_size,
        options.sample_rate,
        options.fmin,
        options.fmax,
        options.scale,
    )?;
    // Stream frames through reusable power/mel buffers rather than holding
    // the whole complex spectrogram.
    let mut power: Vec<f32> = Vec::new();
    let mut out = Vec::new();
    crate::stft::stft_each(
        samples,
        &options.stft,
        crate::stft::StftPaddingMode::Reflect,
        crate::stft::StftWindowPlacement::Left,
        |spectrum| {
            power.clear();
            power.extend(spectrum.iter().map(|c| {
                let mag = c.re.hypot(c.im);
                mag.powf(options.power)
            }));
            let mut row = Vec::with_capacity(options.num_mels);
            projector.project_into(&power, &mut row)?;
            out.push(row);
            Ok(())
        },
    )?;
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

/// Bitwise parity of the cached/streaming mel paths and the span-limited
/// projection against the previous implementations, retained verbatim in
/// `old`. The projection cases target the sign of zero and non-finite bins,
/// the only places skipping +0.0-weight terms could differ.
#[cfg(test)]
mod parity_tests {
    use super::*;
    use crate::dsp::TestRng;

    mod old {
        use super::super::*;

        /// The dense `MelFilterbank::project` body.
        pub(super) fn project(bank: &MelFilterbank, spectrum: &[f32]) -> Vec<f32> {
            (0..bank.num_mels)
                .map(|m| {
                    let row = &bank.weights[m * bank.num_bins..(m + 1) * bank.num_bins];
                    row.iter().zip(spectrum).map(|(&w, &s)| w * s).sum()
                })
                .collect()
        }

        /// `mel_spectrogram` before streaming and caching.
        pub(super) fn mel_spectrogram(
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
    }

    fn assert_bits(what: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "{what}[{i}]: {g:?} vs {w:?}");
        }
    }

    fn bank(
        num_mels: usize,
        fft: usize,
        rate: u32,
        fmax: Option<f32>,
        scale: MelScale,
    ) -> Arc<MelFilterbank> {
        Arc::new(mel_filterbank(num_mels, fft, rate, 0.0, fmax, scale).unwrap())
    }

    /// Spectrum values that expose zero-sign handling: both zeros, values
    /// whose product with a small weight underflows to -0.0, and ordinary
    /// magnitudes of both signs.
    fn tricky_spectrum(rng: &mut TestRng, n: usize, mode: u64) -> Vec<f32> {
        const POOL: [f32; 9] = [0.0, -0.0, 1.0, -1.0, 2.5, -3.5, 1e-30, -1e-30, 1e30];
        (0..n)
            .map(|_| match mode {
                0 => POOL[(rng.next() % 9) as usize],
                // All negative or negative zero: every zero term is -0.0.
                1 => -(POOL[(rng.next() % 9) as usize].abs()),
                // Nonnegative power-like.
                2 => rng.value().abs(),
                // Mostly -0.0 with a rare +0.0 or value.
                3 => {
                    if rng.next() % 40 == 0 {
                        POOL[(rng.next() % 9) as usize]
                    } else {
                        -0.0
                    }
                }
                _ => 0.0,
            })
            .collect()
    }

    fn check_banks() -> Vec<Arc<MelFilterbank>> {
        let mut banks = vec![
            bank(80, 400, 16_000, Some(8_000.0), MelScale::Slaney),
            bank(128, 400, 16_000, Some(8_000.0), MelScale::Slaney),
            bank(8, 16, 8_000, None, MelScale::Htk),
            bank(40, 512, 22_050, None, MelScale::Slaney),
            // More bands than bins: duplicated edges, empty rows.
            bank(30, 16, 16_000, None, MelScale::Htk),
            bank(1, 2, 16_000, None, MelScale::Slaney),
        ];
        // Hand-built rows: all-zero, single weight at each end, -0.0 and NaN
        // weights, a row spanning everything, a zero gap inside a row.
        let n = 12;
        let mut weights = vec![0.0f32; 8 * n];
        weights[n] = 0.5;
        weights[2 * n + n - 1] = 0.25;
        weights[3 * n + 4] = -0.0;
        weights[4 * n + 5] = f32::NAN;
        for w in &mut weights[5 * n..6 * n] {
            *w = 0.125;
        }
        weights[6 * n + 2] = 1.0;
        weights[6 * n + 9] = 2.0;
        weights[7 * n + 6] = 1e-30;
        banks.push(Arc::new(MelFilterbank {
            num_mels: 8,
            num_bins: n,
            weights,
        }));
        banks
    }

    #[test]
    fn span_projection_matches_dense_including_zero_sign() {
        let mut rng = TestRng(0xABCD_0123_4567_89EF);
        for (b, bank) in check_banks().into_iter().enumerate() {
            let projector = MelProjector::new(Arc::clone(&bank));
            let mut out = vec![7.0f32; 3];
            for trial in 0..400u64 {
                let spectrum = tricky_spectrum(&mut rng, bank.num_bins, trial % 5);
                projector.project_into(&spectrum, &mut out).unwrap();
                let want = old::project(&bank, &spectrum);
                assert_bits(&format!("bank {b} trial {trial}"), &out, &want);
                assert_bits(
                    &format!("bank {b} trial {trial} (project)"),
                    &bank.project(&spectrum).unwrap(),
                    &want,
                );
            }
        }
    }

    #[test]
    fn span_projection_matches_dense_on_non_finite_spectra() {
        let mut rng = TestRng(0x5555_AAAA_1357_9BDF);
        for (b, bank) in check_banks().into_iter().enumerate() {
            let projector = MelProjector::new(Arc::clone(&bank));
            let mut out = Vec::new();
            for trial in 0..120usize {
                let mut spectrum = tricky_spectrum(&mut rng, bank.num_bins, 2);
                let at = trial % bank.num_bins;
                spectrum[at] = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN][trial % 3];
                projector.project_into(&spectrum, &mut out).unwrap();
                let want = old::project(&bank, &spectrum);
                // NaN payloads and signs compared as bits as well.
                assert_bits(&format!("bank {b} nonfinite {trial}"), &out, &want);
            }
        }
    }

    #[test]
    fn span_projection_matches_dense_on_realistic_power_spectra() {
        let mut rng = TestRng(0x7777_1111_3333_5555);
        let bank = bank(80, 400, 16_000, Some(8_000.0), MelScale::Slaney);
        let projector = MelProjector::new(Arc::clone(&bank));
        let mut out = Vec::new();
        for trial in 0..200 {
            let spectrum: Vec<f32> = (0..bank.num_bins)
                .map(|_| {
                    let v = rng.value();
                    v * v
                })
                .collect();
            projector.project_into(&spectrum, &mut out).unwrap();
            assert_bits(
                &format!("power {trial}"),
                &out,
                &old::project(&bank, &spectrum),
            );
        }
        // Wrong length is refused with the dense path's error.
        let short = vec![0.0f32; bank.num_bins - 1];
        let want = bank.project(&short).unwrap_err().to_string();
        assert_eq!(
            projector
                .project_into(&short, &mut out)
                .unwrap_err()
                .to_string(),
            want
        );
    }

    #[test]
    fn cached_filterbank_is_bit_identical_and_errors_pass_through() {
        let cases: [(usize, usize, u32, f32, Option<f32>, MelScale); 5] = [
            (80, 400, 16_000, 0.0, Some(8_000.0), MelScale::Slaney),
            (128, 400, 16_000, 0.0, None, MelScale::Slaney),
            (8, 16, 8_000, 0.0, None, MelScale::Htk),
            (40, 512, 22_050, 100.0, Some(7_000.0), MelScale::Slaney),
            (40, 512, 22_050, 100.0, Some(7_000.0), MelScale::Htk),
        ];
        for (num_mels, fft, rate, fmin, fmax, scale) in cases {
            let fresh = mel_filterbank(num_mels, fft, rate, fmin, fmax, scale).unwrap();
            // Twice: the second call is a cache hit.
            for _ in 0..2 {
                let cached = mel_filterbank_cached(num_mels, fft, rate, fmin, fmax, scale).unwrap();
                assert_eq!(
                    (cached.num_mels, cached.num_bins),
                    (fresh.num_mels, fresh.num_bins)
                );
                assert_bits("cached weights", &cached.weights, &fresh.weights);
            }
        }
        // Parameters that differ only in scale or fmax must not collide.
        let slaney = mel_filterbank_cached(8, 16, 8_000, 0.0, None, MelScale::Slaney).unwrap();
        let htk = mel_filterbank_cached(8, 16, 8_000, 0.0, None, MelScale::Htk).unwrap();
        assert_ne!(slaney.weights, htk.weights);
        let explicit =
            mel_filterbank_cached(8, 16, 8_000, 0.0, Some(3_000.0), MelScale::Htk).unwrap();
        assert_ne!(explicit.weights, htk.weights);
        // Invalid parameters return the uncached error every time.
        for _ in 0..2 {
            let want = mel_filterbank(0, 512, 16_000, 0.0, None, MelScale::Slaney).unwrap_err();
            let got =
                mel_filterbank_cached(0, 512, 16_000, 0.0, None, MelScale::Slaney).unwrap_err();
            assert_eq!(got.to_string(), want.to_string());
            let want =
                mel_filterbank(80, 512, 16_000, 9_000.0, None, MelScale::Slaney).unwrap_err();
            let got = mel_filterbank_cached(80, 512, 16_000, 9_000.0, None, MelScale::Slaney)
                .unwrap_err();
            assert_eq!(got.to_string(), want.to_string());
        }
        assert!(
            mel_filterbank_cached(usize::MAX, 400, 16_000, 0.0, None, MelScale::Slaney).is_err()
        );
    }

    #[test]
    fn mel_spectrogram_matches_old_bitwise() {
        let mut rng = TestRng(0x3141_5926_5358_9793);
        let default = MelSpectrogramOptions::default();
        let variants = [
            default.clone(),
            MelSpectrogramOptions {
                stft: StftOptions {
                    fft_size: 400,
                    hop: 160,
                    window: crate::dsp::hann_window(400),
                    center: true,
                },
                num_mels: 80,
                fmax: Some(8_000.0),
                ..default.clone()
            },
            MelSpectrogramOptions {
                stft: StftOptions {
                    fft_size: 64,
                    hop: 16,
                    window: crate::dsp::hann_window(48),
                    center: false,
                },
                num_mels: 12,
                sample_rate: 8_000,
                fmax: None,
                scale: MelScale::Htk,
                power: 1.0,
                ..default.clone()
            },
            MelSpectrogramOptions {
                stft: StftOptions {
                    fft_size: 128,
                    hop: 32,
                    window: crate::dsp::hann_window(128),
                    center: true,
                },
                num_mels: 20,
                sample_rate: 16_000,
                fmax: Some(6_000.0),
                power: 1.7,
                ..default.clone()
            },
        ];
        for (v, options) in variants.iter().enumerate() {
            for len in [2_000usize, 5_000] {
                let samples = rng.vec(len);
                let got = mel_spectrogram(&samples, options).unwrap();
                let want = old::mel_spectrogram(&samples, options).unwrap();
                assert_eq!(got.len(), want.len());
                for (f, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_bits(&format!("variant {v} len {len} frame {f}"), g, w);
                }
            }
            // Silence exercises the all-+0.0 power path.
            let silent = vec![0.0f32; 1_000];
            let got = mel_spectrogram(&silent, options).unwrap();
            let want = old::mel_spectrogram(&silent, options).unwrap();
            for (g, w) in got.iter().zip(&want) {
                assert_bits(&format!("variant {v} silence"), g, w);
            }
        }
        // Error precedence is unchanged: power, then filterbank, then STFT.
        let bad_power = MelSpectrogramOptions {
            power: 0.0,
            ..default.clone()
        };
        assert_eq!(
            mel_spectrogram(&[0.0; 10], &bad_power)
                .unwrap_err()
                .to_string(),
            old::mel_spectrogram(&[0.0; 10], &bad_power)
                .unwrap_err()
                .to_string()
        );
        let bad_bank = MelSpectrogramOptions {
            num_mels: 0,
            ..default.clone()
        };
        assert_eq!(
            mel_spectrogram(&[0.0; 10], &bad_bank)
                .unwrap_err()
                .to_string(),
            old::mel_spectrogram(&[0.0; 10], &bad_bank)
                .unwrap_err()
                .to_string()
        );
        assert_eq!(
            mel_spectrogram(&[0.0; 10], &default)
                .unwrap_err()
                .to_string(),
            old::mel_spectrogram(&[0.0; 10], &default)
                .unwrap_err()
                .to_string()
        );
    }
}
