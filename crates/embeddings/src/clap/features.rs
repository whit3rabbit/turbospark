//! `ClapFeatureExtractor` for the unfused checkpoint: 48 kHz mono PCM to a
//! `[frames, mels]` log-mel (dB) matrix.
//!
//! Reference: transformers 4.57.1 `feature_extraction_clap.py` with
//! `truncation="rand_trunc"`, `padding="repeatpad"`, and
//! `audio_utils.spectrogram` / `mel_filter_bank` / `power_to_db`:
//!
//! 1. Longer than `max_length_s`: crop a `max_samples` window (the
//!    reference draws the offset with `np.random.randint`; here the caller
//!    picks it through [`LongAudioCrop`], default [`LongAudioCrop::Head`]).
//!    Shorter: tile `floor(max / len)` times, then zero-pad to `max`.
//! 2. Reflect-pad `n_fft / 2` both sides (numpy `reflect`, no edge repeat),
//!    frame at `hop`, multiply by a periodic Hann window
//!    (`np.hanning(n + 1)[:-1]`).
//! 3. FFT in f64, rounded through complex64 (the reference stores the
//!    spectrum as `np.complex64`), then `|X|^2` in f64.
//! 4. Slaney-scale, Slaney-normalized filterbank (`mel_filters_slaney`;
//!    rand_trunc uses it, fusion uses the HTK one), floor at 1e-10.
//! 5. `10 * log10(max(x, 1e-10))` (reference 1.0, no `top_db`), cast to
//!    f32.
//!
//! The whole chain runs in f64 like the numpy reference. The shared
//! `turbospark_audio` STFT and mel filterbank are f32; the f32 chain is
//! measured against the same goldens in the tests (see the family README
//! for the numbers) and is kept out of the production path because its
//! error lands in the low-energy bins the dB scale magnifies.

use std::f64::consts::PI;
use std::path::Path;

use turbospark_audio::conversion::{to_mono_resampled, MonoResampleStrategy};
use turbospark_audio::resample::SincHannOptions;

use super::config::FeatureConfig;
use crate::error::{bad_config, input_error, EmbeddingError};
use crate::Result;

/// `mel_floor` and `power_to_db` `min_value`.
const MEL_FLOOR: f64 = 1e-10;

/// Where to crop a clip longer than `max_length_s`.
///
/// The reference (`rand_trunc`) draws `offset ~ U[0, len - max]` with
/// numpy's global RNG, so its features are not reproducible run to run.
/// This port makes the choice explicit and deterministic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LongAudioCrop {
    /// Keep the first `max_samples` (offset 0).
    #[default]
    Head,
    /// Keep the middle window (offset `(len - max) / 2`, floored).
    Center,
    /// Explicit sample offset; must be at most `len - max`.
    Offset(usize),
}

/// Extracted features for one clip.
#[derive(Debug, Clone, PartialEq)]
pub struct MelFeatures {
    /// Row-major `[frames, mels]` log-mel in dB.
    pub data: Vec<f32>,
    pub frames: usize,
    pub mels: usize,
    /// The reference's `is_longer` flag (input exceeded `max_length_s`).
    /// The unfused model ignores it; it only gates the fusion branch.
    pub is_longer: bool,
    /// Sample offset of the crop (0 when not cropped).
    pub crop_offset: usize,
}

/// The CLAP log-mel frontend.
#[derive(Debug, Clone)]
pub struct ClapFeatureExtractor {
    config: FeatureConfig,
    /// `[mels, bins]` Slaney filterbank, f64 like the reference.
    filters: Vec<f64>,
    window: Vec<f64>,
    fft: Fft64,
}

impl ClapFeatureExtractor {
    pub fn new(config: FeatureConfig) -> Result<Self> {
        config.validate()?;
        let n = config.fft_window_size;
        if config.max_samples() <= n / 2 {
            return Err(bad_config(
                "max_length_s",
                "window shorter than half an FFT frame cannot be reflect-padded",
            ));
        }
        let filters = slaney_filterbank(
            n / 2 + 1,
            config.feature_size,
            config.frequency_min,
            config.frequency_max,
            config.sampling_rate,
        );
        // np.hanning(n + 1)[:-1]: 0.5 - 0.5 cos(2 pi k / n).
        let window = (0..n)
            .map(|k| 0.5 - 0.5 * (2.0 * PI * k as f64 / n as f64).cos())
            .collect();
        Ok(Self {
            fft: Fft64::new(n),
            config,
            filters,
            window,
        })
    }

    /// Loads `preprocessor_config.json`.
    pub fn from_file(path: &Path) -> Result<Self> {
        Self::new(FeatureConfig::from_file(path)?)
    }

    pub fn config(&self) -> &FeatureConfig {
        &self.config
    }

    /// `[mels, bins]` filterbank, for inspection.
    pub fn filterbank(&self) -> &[f64] {
        &self.filters
    }

    /// Log-mel features of mono PCM at `config().sampling_rate`.
    pub fn extract(&self, pcm: &[f32], crop: LongAudioCrop) -> Result<MelFeatures> {
        let max = self.config.max_samples();
        if pcm.is_empty() {
            // The reference divides by the length (repeatpad) and fails.
            return Err(input_error("empty audio"));
        }
        if let Some(i) = pcm.iter().position(|v| !v.is_finite()) {
            return Err(input_error(format!("non-finite sample at index {i}")));
        }
        let (clip, is_longer, crop_offset): (Vec<f64>, bool, usize) = if pcm.len() > max {
            let overflow = pcm.len() - max;
            let offset = match crop {
                LongAudioCrop::Head => 0,
                LongAudioCrop::Center => overflow / 2,
                LongAudioCrop::Offset(o) if o <= overflow => o,
                LongAudioCrop::Offset(o) => {
                    return Err(input_error(format!(
                        "crop offset {o} exceeds the {overflow} samples past max_length"
                    )))
                }
            };
            let clip = pcm[offset..offset + max]
                .iter()
                .map(|&v| f64::from(v))
                .collect();
            (clip, true, offset)
        } else {
            // repeatpad: tile floor(max / len) times, then zero-pad.
            let repeats = max / pcm.len();
            let mut clip = Vec::with_capacity(max);
            for _ in 0..repeats {
                clip.extend(pcm.iter().map(|&v| f64::from(v)));
            }
            clip.resize(max, 0.0);
            (clip, false, 0)
        };
        let data = self.log_mel(&clip);
        Ok(MelFeatures {
            frames: data.len() / self.config.feature_size,
            mels: self.config.feature_size,
            data,
            is_longer,
            crop_offset,
        })
    }

    /// Steps 2-5 over an already cropped or padded clip.
    fn log_mel(&self, clip: &[f64]) -> Vec<f32> {
        let n = self.config.fft_window_size;
        let half = n / 2;
        let hop = self.config.hop_length;
        let len = clip.len();
        // numpy "reflect": mirror about the edge sample without repeating it.
        let padded: Vec<f64> = (0..len + 2 * half)
            .map(|i| {
                if i < half {
                    clip[half - i]
                } else if i < half + len {
                    clip[i - half]
                } else {
                    clip[2 * len - 2 - (i - half)]
                }
            })
            .collect();
        let frames = 1 + (padded.len() - n) / hop;
        let bins = half + 1;
        let mels = self.config.feature_size;
        let mut out = vec![0.0f32; frames * mels];
        let mut buf = vec![(0.0f64, 0.0f64); n];
        let mut power = vec![0.0f64; bins];
        for (t, row) in out.chunks_exact_mut(mels).enumerate() {
            let start = t * hop;
            for (k, slot) in buf.iter_mut().enumerate() {
                *slot = (padded[start + k] * self.window[k], 0.0);
            }
            self.fft.forward(&mut buf);
            for (p, &(re, im)) in power.iter_mut().zip(&buf[..bins]) {
                // The reference stores the spectrum as complex64, then
                // takes np.abs in f64 and squares.
                let mag = f64::from(re as f32).hypot(f64::from(im as f32));
                *p = mag * mag;
            }
            for (m, slot) in row.iter_mut().enumerate() {
                let w = &self.filters[m * bins..(m + 1) * bins];
                let e: f64 = w.iter().zip(&power).map(|(a, b)| a * b).sum();
                *slot = (10.0 * e.max(MEL_FLOOR).log10()) as f32;
            }
        }
        out
    }
}

/// `audio_utils.hertz_to_mel(.., "slaney")`.
fn hz_to_mel_slaney(f: f64) -> f64 {
    let logstep = 27.0 / 6.4f64.ln();
    if f >= 1000.0 {
        15.0 + (f / 1000.0).ln() * logstep
    } else {
        3.0 * f / 200.0
    }
}

/// `audio_utils.mel_to_hertz(.., "slaney")`.
fn mel_to_hz_slaney(m: f64) -> f64 {
    let logstep = 6.4f64.ln() / 27.0;
    if m >= 15.0 {
        1000.0 * (logstep * (m - 15.0)).exp()
    } else {
        200.0 * m / 3.0
    }
}

/// `np.linspace(start, stop, num)`: `i * step + start`, last pinned to
/// `stop`.
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let step = (stop - start) / (num - 1) as f64;
    let mut v: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
    v[num - 1] = stop;
    v
}

/// `audio_utils.mel_filter_bank(norm="slaney", mel_scale="slaney")`,
/// transposed to `[mels, bins]`.
fn slaney_filterbank(bins: usize, mels: usize, fmin: f64, fmax: f64, sr: u32) -> Vec<f64> {
    let mel_pts = linspace(hz_to_mel_slaney(fmin), hz_to_mel_slaney(fmax), mels + 2);
    let f: Vec<f64> = mel_pts.into_iter().map(mel_to_hz_slaney).collect();
    let fft_freqs = linspace(0.0, f64::from(sr / 2), bins);
    let mut out = vec![0.0f64; mels * bins];
    for m in 0..mels {
        let enorm = 2.0 / (f[m + 2] - f[m]);
        for (k, &x) in fft_freqs.iter().enumerate() {
            let down = -(f[m] - x) / (f[m + 1] - f[m]);
            let up = (f[m + 2] - x) / (f[m + 2] - f[m + 1]);
            out[m * bins + k] = 0.0f64.max(down.min(up)) * enorm;
        }
    }
    out
}

/// Iterative radix-2 complex FFT in f64 (forward, unscaled).
#[derive(Debug, Clone)]
struct Fft64 {
    n: usize,
    twiddles: Vec<(f64, f64)>,
    rev: Vec<usize>,
}

impl Fft64 {
    fn new(n: usize) -> Self {
        debug_assert!(n.is_power_of_two());
        let bits = n.trailing_zeros();
        let rev = (0..n)
            .map(|i| {
                if bits == 0 {
                    0
                } else {
                    i.reverse_bits() >> (usize::BITS - bits)
                }
            })
            .collect();
        let twiddles = (0..n / 2)
            .map(|k| {
                let a = -2.0 * PI * k as f64 / n as f64;
                (a.cos(), a.sin())
            })
            .collect();
        Self { n, twiddles, rev }
    }

    fn forward(&self, buf: &mut [(f64, f64)]) {
        let n = self.n;
        for i in 0..n {
            let j = self.rev[i];
            if i < j {
                buf.swap(i, j);
            }
        }
        let mut size = 2;
        while size <= n {
            let half = size / 2;
            let stride = n / size;
            for start in (0..n).step_by(size) {
                for k in 0..half {
                    let (wr, wi) = self.twiddles[k * stride];
                    let (ar, ai) = buf[start + k];
                    let (br, bi) = buf[start + k + half];
                    let (tr, ti) = (br * wr - bi * wi, br * wi + bi * wr);
                    buf[start + k] = (ar + tr, ai + ti);
                    buf[start + k + half] = (ar - tr, ai - ti);
                }
            }
            size *= 2;
        }
    }
}

/// Reads a WAV file and returns mono PCM at `target_rate` (48 kHz for
/// CLAP): channels are averaged (f64 accumulation), then resampled with
/// the torchaudio-compatible sinc-Hann polyphase filter at its defaults.
///
/// The reference has no single resampler: the HF zero-shot pipeline
/// decodes files through ffmpeg, and most callers use librosa or
/// torchaudio. Features of resampled audio therefore match the reference
/// only as closely as the caller's resampler matches this one; 48 kHz
/// input skips resampling entirely.
pub fn load_wav_mono(path: &Path, target_rate: u32) -> Result<Vec<f32>> {
    let wave = turbospark_audio::wav::read_wav_f32(path).map_err(|e| EmbeddingError::Io {
        path: path.display().to_string(),
        why: e.to_string(),
    })?;
    let strategy = MonoResampleStrategy::SincHann(SincHannOptions::default());
    to_mono_resampled(&wave, target_rate, &strategy).map_err(|e| EmbeddingError::Kernel(e.into()))
}

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn fft_matches_naive_dft() {
        let n = 16;
        let fft = Fft64::new(n);
        let x: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 11) as f64 - 5.0).collect();
        let mut buf: Vec<(f64, f64)> = x.iter().map(|&v| (v, 0.0)).collect();
        fft.forward(&mut buf);
        for (k, &(re, im)) in buf.iter().enumerate() {
            let (mut er, mut ei) = (0.0, 0.0);
            for (t, &v) in x.iter().enumerate() {
                let a = -2.0 * PI * (k * t) as f64 / n as f64;
                er += v * a.cos();
                ei += v * a.sin();
            }
            assert!(
                (re - er).abs() < 1e-12 && (im - ei).abs() < 1e-12,
                "bin {k}"
            );
        }
    }

    #[test]
    fn repeatpad_and_crop_policies() {
        let mut c = FeatureConfig {
            truncation: "rand_trunc".into(),
            ..FeatureConfig::default()
        };
        c.sampling_rate = 4000;
        c.max_length_s = 1;
        c.fft_window_size = 64;
        c.hop_length = 40;
        c.feature_size = 8;
        c.frequency_min = 50.0;
        c.frequency_max = 2000.0;
        let fe = ClapFeatureExtractor::new(c).unwrap();
        // 1500 samples: tiled 2x into 3000, then 1000 zeros.
        let pcm: Vec<f32> = (0..1500).map(|i| ((i % 37) as f32 - 18.0) / 40.0).collect();
        let short = fe.extract(&pcm, LongAudioCrop::Head).unwrap();
        assert!(!short.is_longer);
        assert_eq!(short.frames, 101);
        let mut manual: Vec<f64> = pcm.iter().chain(&pcm).map(|&v| f64::from(v)).collect();
        manual.resize(4000, 0.0);
        assert_eq!(short.data, fe.log_mel(&manual));

        let long: Vec<f32> = (0..4100).map(|i| ((i % 53) as f32 - 26.0) / 60.0).collect();
        let head = fe.extract(&long, LongAudioCrop::Head).unwrap();
        let tail = fe.extract(&long, LongAudioCrop::Offset(100)).unwrap();
        let mid = fe.extract(&long, LongAudioCrop::Center).unwrap();
        assert!(head.is_longer && head.crop_offset == 0);
        assert_eq!(mid.crop_offset, 50);
        assert_eq!(
            tail.data,
            fe.extract(&long[100..], LongAudioCrop::Head).unwrap().data
        );
        assert!(fe.extract(&long, LongAudioCrop::Offset(101)).is_err());
        assert!(fe.extract(&[], LongAudioCrop::Head).is_err());
        assert!(fe.extract(&[f32::NAN], LongAudioCrop::Head).is_err());
    }
}
