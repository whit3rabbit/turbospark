//! Gemma4 audio log-mel frontend used by the EmbeddingGemma 2 audio tower.
//!
//! Reference: transformers `Gemma4AudioFeatureExtractor`
//! (`src/transformers/models/gemma4/feature_extraction_gemma4.py`) at commit
//! `28034545c35acac05793695bd5c878f66e6b544e`, configured by the
//! `google/embeddinggemma-2` `preprocessor_config.json`: 16 kHz, 320-sample
//! periodic Hann frames, 160-sample hop, 512-point real FFT, 128 HTK mel
//! bands over 0..8000 Hz with no area normalization, magnitude (not power)
//! spectrum, `log(mel + 1e-3)`, no preemphasis, dither, or per-bin
//! normalization.
//!
//! The contract details that differ from the other frontends in this crate:
//!
//! - Semicausal padding: `frame_length / 2` zeros are prepended, nothing is
//!   appended, and frames come from a `frame_length + 1` unfold whose last
//!   sample is dropped (it only feeds preemphasis, which is off).
//! - The floor is additive (`log(x + floor)`), not a clamp.
//! - A frame is valid only when the last sample of its unfold window is real
//!   audio; invalid frames are zeroed after the log.
//!
//! The shipped `hann_320.f32` and `mel_filters_257x128.f32` tables were
//! checked against this contract (periodic Hann, `[bin][mel]` HTK
//! filterbank); see the family README for the measured differences.

use std::path::Path;
use std::sync::Arc;

use crate::dsp::hann_window;
use crate::fft::{RealFftPlan, RealFftScratch};
use crate::mel::MelFilterbank;
use crate::nn::bad_config;
use crate::{Result, SpeechError};

pub const SAMPLE_RATE: u32 = 16_000;
pub const FRAME_LENGTH: usize = 320;
pub const HOP_LENGTH: usize = 160;
pub const FFT_LENGTH: usize = 512;
pub const NUM_BINS: usize = FFT_LENGTH / 2 + 1;
pub const NUM_MELS: usize = 128;
pub const MEL_FLOOR: f32 = 1e-3;
pub const MIN_FREQUENCY_HZ: f32 = 0.0;
pub const MAX_FREQUENCY_HZ: f32 = 8_000.0;

/// File name of the shipped periodic Hann window (320 little-endian f32).
pub const HANN_FILE: &str = "hann_320.f32";
/// File name of the shipped filterbank (`[257][128]` little-endian f32).
pub const MEL_FILTERS_FILE: &str = "mel_filters_257x128.f32";

/// Log-mel features, row-major `[frames][NUM_MELS]`, with the per-frame
/// validity mask. Masked frames are zero.
#[derive(Debug, Clone, PartialEq)]
pub struct LogMel {
    pub frames: usize,
    pub values: Vec<f32>,
    pub mask: Vec<bool>,
}

impl LogMel {
    /// Number of frames whose mask is set.
    pub fn valid_frames(&self) -> usize {
        self.mask.iter().filter(|&&valid| valid).count()
    }

    /// Pads with masked zero frames, or truncates, to exactly `frames`.
    pub fn fit_to(mut self, frames: usize) -> Self {
        self.values.resize(frames * NUM_MELS, 0.0);
        self.mask.resize(frames, false);
        self.frames = frames;
        self
    }

    /// The mask as the `0.0` / `1.0` floats the compiled encoder consumes.
    pub fn mask_f32(&self) -> Vec<f32> {
        self.mask
            .iter()
            .map(|&valid| if valid { 1.0 } else { 0.0 })
            .collect()
    }
}

/// The Gemma4 log-mel frontend with its window and filterbank.
#[derive(Debug, Clone)]
pub struct LogMelFrontend {
    window: Vec<f32>,
    filters: MelFilterbank,
    plan: Arc<RealFftPlan>,
}

impl LogMelFrontend {
    /// Builds the frontend from the reference contract: a periodic Hann
    /// window and an unnormalized HTK filterbank.
    pub fn new() -> Result<Self> {
        Self::with_parts(hann_window(FRAME_LENGTH), htk_filterbank_f64())
    }

    /// Builds the frontend from explicit tables: a `FRAME_LENGTH` window and
    /// a filterbank in the shipped `[NUM_BINS][NUM_MELS]` (bin-major) layout.
    pub fn from_tables(window: &[f32], bin_major_filters: &[f32]) -> Result<Self> {
        if window.len() != FRAME_LENGTH {
            return Err(bad_config(
                HANN_FILE,
                format!("holds {} taps, expected {FRAME_LENGTH}", window.len()),
            ));
        }
        if bin_major_filters.len() != NUM_BINS * NUM_MELS {
            return Err(bad_config(
                MEL_FILTERS_FILE,
                format!(
                    "holds {} weights, expected {NUM_BINS} x {NUM_MELS}",
                    bin_major_filters.len()
                ),
            ));
        }
        if window
            .iter()
            .chain(bin_major_filters)
            .any(|value| !value.is_finite())
        {
            return Err(bad_config(
                MEL_FILTERS_FILE,
                "frontend tables must be finite",
            ));
        }
        let mut weights = vec![0.0f32; NUM_MELS * NUM_BINS];
        for bin in 0..NUM_BINS {
            for mel in 0..NUM_MELS {
                weights[mel * NUM_BINS + bin] = bin_major_filters[bin * NUM_MELS + mel];
            }
        }
        Self::with_parts(
            window.to_vec(),
            MelFilterbank {
                num_mels: NUM_MELS,
                num_bins: NUM_BINS,
                weights,
            },
        )
    }

    /// Loads the shipped `hann_320.f32` and `mel_filters_257x128.f32`.
    pub fn load(asset_dir: &Path) -> Result<Self> {
        let window = read_f32_table(&asset_dir.join(HANN_FILE), HANN_FILE)?;
        let filters = read_f32_table(&asset_dir.join(MEL_FILTERS_FILE), MEL_FILTERS_FILE)?;
        Self::from_tables(&window, &filters)
    }

    fn with_parts(window: Vec<f32>, filters: MelFilterbank) -> Result<Self> {
        Ok(Self {
            window,
            filters,
            plan: RealFftPlan::cached(FFT_LENGTH)?,
        })
    }

    pub fn window(&self) -> &[f32] {
        &self.window
    }

    /// Mel-major `[NUM_MELS][NUM_BINS]` filterbank weights.
    pub fn filters(&self) -> &MelFilterbank {
        &self.filters
    }

    /// Log-mel features of `samples` right-padded with zeros to
    /// `padded_len` samples (`padded_len < samples.len()` means no padding).
    ///
    /// Frame count and mask follow `_extract_spectrogram`: after the
    /// semicausal left pad of `FRAME_LENGTH / 2`, frames are taken every
    /// `HOP_LENGTH` samples while a `FRAME_LENGTH + 1` window fits, and a
    /// frame is valid when the last sample of that window is real audio.
    /// The extractor's `__call__` pads to a multiple of 128 samples before
    /// this step; callers reproduce it by passing that length.
    pub fn features(&self, samples: &[f32], padded_len: usize) -> Result<LogMel> {
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(SpeechError::Input {
                why: "audio samples must be finite".into(),
            });
        }
        let pad_left = FRAME_LENGTH / 2;
        let unfold = FRAME_LENGTH + 1;
        let total = pad_left + samples.len().max(padded_len);
        let frames = if total < unfold {
            0
        } else {
            (total - unfold) / HOP_LENGTH + 1
        };
        let real_end = pad_left + samples.len();
        let sample_at = |index: usize| -> f32 {
            if index < pad_left || index >= real_end {
                0.0
            } else {
                samples[index - pad_left]
            }
        };

        let mut values = vec![0.0f32; frames * NUM_MELS];
        let mut mask = vec![false; frames];
        let mut frame = vec![0.0f32; FFT_LENGTH];
        let mut spectrum = Vec::new();
        let mut magnitude = vec![0.0f32; NUM_BINS];
        let mut scratch = RealFftScratch::default();
        for index in 0..frames {
            let start = index * HOP_LENGTH;
            let last = start + unfold - 1;
            mask[index] = last >= pad_left && last < real_end;
            if !mask[index] {
                continue;
            }
            for (tap, slot) in frame[..FRAME_LENGTH].iter_mut().enumerate() {
                *slot = sample_at(start + tap) * self.window[tap];
            }
            frame[FRAME_LENGTH..].fill(0.0);
            self.plan
                .forward_into(&frame, &mut spectrum, &mut scratch)?;
            for (slot, bin) in magnitude.iter_mut().zip(&spectrum) {
                *slot = (f64::from(bin.re).hypot(f64::from(bin.im))) as f32;
            }
            let row = &mut values[index * NUM_MELS..(index + 1) * NUM_MELS];
            for (mel, slot) in row.iter_mut().enumerate() {
                let weights = &self.filters.weights[mel * NUM_BINS..(mel + 1) * NUM_BINS];
                let energy: f64 = weights
                    .iter()
                    .zip(&magnitude)
                    .map(|(&weight, &value)| f64::from(weight) * f64::from(value))
                    .sum();
                *slot = (energy + f64::from(MEL_FLOOR)).ln() as f32;
            }
        }
        Ok(LogMel {
            frames,
            values,
            mask,
        })
    }
}

/// The transformers `mel_filter_bank(norm=None, mel_scale="htk")` for this
/// geometry, computed in f64 like the reference and rounded once.
///
/// [`crate::mel::mel_filterbank`] builds the same triangles in f32 and lands
/// up to 2.2e-5 from the reference weights, which moves low-energy log-mel
/// values by more than the fixture tolerance; this keeps the f64 band edges.
fn htk_filterbank_f64() -> MelFilterbank {
    let hz_to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f64| 700.0 * (10f64.powf(mel / 2595.0) - 1.0);
    let mel_min = hz_to_mel(f64::from(MIN_FREQUENCY_HZ));
    let mel_max = hz_to_mel(f64::from(MAX_FREQUENCY_HZ));
    let edges: Vec<f64> = (0..NUM_MELS + 2)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (NUM_MELS + 1) as f64))
        .collect();
    let nyquist = f64::from(SAMPLE_RATE) / 2.0;
    let mut weights = vec![0.0f32; NUM_MELS * NUM_BINS];
    for mel in 0..NUM_MELS {
        let (lower, center, upper) = (edges[mel], edges[mel + 1], edges[mel + 2]);
        for bin in 0..NUM_BINS {
            let hz = nyquist * bin as f64 / (NUM_BINS - 1) as f64;
            let down = (hz - lower) / (center - lower);
            let up = (upper - hz) / (upper - center);
            weights[mel * NUM_BINS + bin] = down.min(up).max(0.0) as f32;
        }
    }
    MelFilterbank {
        num_mels: NUM_MELS,
        num_bins: NUM_BINS,
        weights,
    }
}

/// Reads a headerless little-endian f32 table.
pub(crate) fn read_f32_table(path: &Path, name: &str) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).map_err(|error| bad_config(name, error.to_string()))?;
    if bytes.len() % 4 != 0 {
        return Err(bad_config(
            name,
            format!("{} bytes is not a whole number of f32 values", bytes.len()),
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_signal() -> Vec<f32> {
        (0..4000)
            .map(|i| {
                let t = i as f64 / 16_000.0;
                let tone = 0.4 * (2.0 * std::f64::consts::PI * 440.0 * t).sin();
                let chirp = 0.2 * (2.0 * std::f64::consts::PI * (200.0 + 0.5 * i as f64) * t).sin();
                let ramp = ((i * 7 % 31) as f64 - 15.0) / 160.0;
                (tone + chirp + ramp) as f32
            })
            .collect()
    }

    #[test]
    fn features_match_the_pinned_transformers_fixture() {
        let golden: serde_json::Value = serde_json::from_str(include_str!(
            "../../../testdata/embeddinggemma2_features.json"
        ))
        .expect("fixture is valid JSON");
        let samples = fixture_signal();
        assert_eq!(golden["samples"].as_u64(), Some(samples.len() as u64));
        // `__call__` pads to a multiple of 128 samples before extraction.
        let padded = samples.len().div_ceil(128) * 128;
        let actual = LogMelFrontend::new()
            .unwrap()
            .features(&samples, padded)
            .unwrap();
        let frames = golden["frames"].as_u64().unwrap() as usize;
        assert_eq!(actual.frames, frames);
        let mask: Vec<bool> = golden["mask"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_bool().unwrap())
            .collect();
        assert_eq!(actual.mask, mask);
        assert!(
            mask.iter().any(|valid| !valid),
            "fixture must pin a masked frame"
        );
        // The reference transforms in f64; this frontend runs the crate's
        // f32 FFT, whose round-off (up to 2e-6 absolute in mel energy on
        // this fixture) is magnified by the log in bands that sit near the
        // 1e-3 floor. So the energy (`exp`) domain carries the tight bound,
        // 3e-6 absolute plus f32-relative 1e-6, and the log domain a looser
        // one.
        let mut log_error = 0.0f32;
        let mut energy_error = 0.0f64;
        for (row, golden_row) in golden["features"].as_array().unwrap().iter().enumerate() {
            let golden_row = golden_row.as_array().unwrap();
            assert_eq!(golden_row.len(), NUM_MELS);
            for (mel, expected) in golden_row.iter().enumerate() {
                let expected = expected.as_f64().unwrap() as f32;
                let got = actual.values[row * NUM_MELS + mel];
                log_error = log_error.max((got - expected).abs());
                if mask[row] {
                    let reference = f64::from(expected).exp();
                    let excess = (f64::from(got).exp() - reference).abs() - 1e-6 * reference;
                    energy_error = energy_error.max(excess);
                } else {
                    assert_eq!(got, 0.0, "masked frame {row} must be zero");
                }
            }
        }
        assert!(
            energy_error <= 3e-6,
            "mel energy error beyond f32 relative {energy_error}"
        );
        assert!(log_error <= 5e-4, "log-mel max abs error {log_error}");
    }

    #[test]
    fn shipped_table_layout_round_trips_into_the_same_frontend() {
        let reference = LogMelFrontend::new().unwrap();
        let mut bin_major = vec![0.0f32; NUM_BINS * NUM_MELS];
        for mel in 0..NUM_MELS {
            for bin in 0..NUM_BINS {
                bin_major[bin * NUM_MELS + mel] = reference.filters.weights[mel * NUM_BINS + bin];
            }
        }
        let loaded = LogMelFrontend::from_tables(reference.window(), &bin_major).unwrap();
        assert_eq!(loaded.filters, reference.filters);
        let samples = fixture_signal();
        assert_eq!(
            loaded.features(&samples, 0).unwrap(),
            reference.features(&samples, 0).unwrap()
        );
    }

    #[test]
    fn f64_filterbank_stays_close_to_the_shared_f32_builder() {
        let shared = crate::mel::mel_filterbank(
            NUM_MELS,
            FFT_LENGTH,
            SAMPLE_RATE,
            MIN_FREQUENCY_HZ,
            Some(MAX_FREQUENCY_HZ),
            crate::mel::MelScale::Htk,
        )
        .unwrap();
        let ours = htk_filterbank_f64();
        let worst = shared
            .weights
            .iter()
            .zip(&ours.weights)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-4,
            "f32 and f64 HTK filterbanks differ by {worst}"
        );
    }

    #[test]
    fn refuses_misshaped_tables_and_non_finite_audio() {
        let frontend = LogMelFrontend::new().unwrap();
        assert!(LogMelFrontend::from_tables(&[0.5; 319], &[0.0; NUM_BINS * NUM_MELS]).is_err());
        assert!(LogMelFrontend::from_tables(frontend.window(), &[0.0; 10]).is_err());
        assert!(frontend.features(&[0.0, f32::NAN], 0).is_err());
    }

    #[test]
    fn twenty_second_window_yields_1999_valid_frames() {
        // 320000 samples: 1999 frames, all valid. The compiled encoder takes
        // 2000, so the window path appends one masked frame.
        let frontend = LogMelFrontend::new().unwrap();
        let features = frontend.features(&vec![0.0; 320_000], 0).unwrap();
        assert_eq!(features.frames, 1999);
        assert_eq!(features.valid_frames(), 1999);
        let fitted = features.fit_to(2000);
        assert_eq!(fitted.values.len(), 2000 * NUM_MELS);
        assert_eq!(fitted.valid_frames(), 1999);
    }

    #[test]
    fn short_partial_window_masks_the_padding() {
        let frontend = LogMelFrontend::new().unwrap();
        let features = frontend.features(&vec![0.1; 16_000], 320_000).unwrap();
        assert_eq!(features.frames, 1999);
        // Frame i is valid while i * 160 + 320 < 16000 + 160.
        assert_eq!(features.valid_frames(), 99);
        assert!(features.values[99 * NUM_MELS..].iter().all(|&v| v == 0.0));
    }
}
