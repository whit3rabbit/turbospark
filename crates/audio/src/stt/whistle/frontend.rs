//! Whistle's centered, zero-padded log-mel frontend.

use crate::fft::{RealFftPlan, RealFftScratch};
use crate::{Result, SpeechError};

pub(super) const RATE: usize = 16_000;
pub(super) const BINS: usize = 80;
const FFT: usize = 512;
const WINDOW: usize = 400;
const HOP: usize = 160;

pub(super) fn validate_pcm(pcm: &[f32]) -> Result<()> {
    if pcm.len() < HOP
        || pcm.len() > RATE * 30
        || pcm.iter().any(|v| !v.is_finite() || v.abs() > 1.0)
    {
        return Err(SpeechError::Input {
            why: "Whistle expects 10 ms to 30 s of finite normalized 16 kHz mono audio in [-1,1]"
                .into(),
        });
    }
    Ok(())
}

pub(super) struct Frontend {
    bank: Vec<f32>,
    window: Vec<f32>,
    fft: RealFftPlan,
}

impl Frontend {
    pub fn new(bank: Vec<f32>) -> Result<Self> {
        if bank.len() != (FFT / 2 + 1) * BINS || bank.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(SpeechError::BadConfig {
                field: "mel_bank".into(),
                why: "Whistle mel bank must be [257,80]".into(),
            });
        }
        // The released checkpoint uses a symmetric Hann window, not Whisper's
        // periodic window. Keeping this separate prevents accidental reuse.
        let window = (0..WINDOW)
            .map(|i| {
                (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / (WINDOW - 1) as f64).cos()) as f32
            })
            .collect();
        Ok(Self {
            bank,
            window,
            fft: RealFftPlan::new(FFT).map_err(|e| SpeechError::Audio(e.to_string()))?,
        })
    }

    pub fn features(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        validate_pcm(pcm)?;
        let frames = pcm.len() / HOP;
        let gain = level_gain(pcm);
        let mut features = vec![0.0; frames * BINS];
        let mut input = vec![0.0; FFT];
        let mut spectrum = Vec::new();
        let mut scratch = RealFftScratch::default();
        for t in 0..frames {
            input.fill(0.0);
            for (i, &window) in self.window.iter().enumerate() {
                let pos = (t * HOP + i) as isize - (WINDOW / 2) as isize;
                if pos >= 0 && (pos as usize) < pcm.len() {
                    input[i] = pcm[pos as usize] * gain * window;
                }
            }
            self.fft
                .forward_into(&input, &mut spectrum, &mut scratch)
                .map_err(|e| SpeechError::Audio(e.to_string()))?;
            for (k, value) in spectrum.iter().enumerate() {
                let power = value.re * value.re + value.im * value.im;
                for c in 0..BINS {
                    features[t * BINS + c] += power * self.bank[k * BINS + c];
                }
            }
            for v in &mut features[t * BINS..(t + 1) * BINS] {
                *v = (*v + f32::from_bits(0x3380_0000)).ln();
            }
        }
        // The reference accumulates mean and sample variance in f64, then
        // rounds the mean and deviation to f32 before normalization.
        for c in 0..BINS {
            let mean = (0..frames)
                .map(|t| features[t * BINS + c] as f64)
                .sum::<f64>()
                / frames as f64;
            let variance = (0..frames)
                .map(|t| {
                    let d = features[t * BINS + c] as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / frames.saturating_sub(1).max(1) as f64;
            let scale = (variance.sqrt() as f32 + 1e-5).recip();
            for t in 0..frames {
                features[t * BINS + c] = (features[t * BINS + c] - mean as f32) * scale;
            }
        }
        Ok(features)
    }
}

fn level_gain(pcm: &[f32]) -> f32 {
    // Use the 99th percentile of 20 ms RMS windows. This levels speech without
    // allowing long quiet regions to drive the gain.
    let mut rms: Vec<f32> = pcm
        .chunks_exact(2 * HOP)
        .map(|chunk| {
            (chunk.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / (2 * HOP) as f64).sqrt()
                as f32
        })
        .collect();
    if rms.is_empty() {
        return 1.0;
    }
    let index = ((rms.len() - 1) as f32 * 0.99) as usize;
    let (_, peak, _) = rms.select_nth_unstable_by(index, f32::total_cmp);
    if *peak > 1e-6 {
        0.1 / *peak
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_empty_overlong_nonfinite_and_unnormalized_pcm() {
        for pcm in [
            vec![],
            vec![0.0; 159],
            vec![0.0; RATE * 30 + 1],
            vec![f32::INFINITY; 160],
            vec![1.1; 160],
        ] {
            assert!(validate_pcm(&pcm).is_err());
        }
        assert!(validate_pcm(&[1.0; 160]).is_ok());
        assert!(validate_pcm(&vec![0.0; RATE * 30]).is_ok());
        assert!(Frontend::new(vec![-1.0; 257 * 80]).is_err());
    }

    #[test]
    fn level_gain_uses_high_percentile_and_ignores_trailing_partial_window() {
        let mut pcm = vec![0.0; 320 * 101];
        pcm[320 * 98..320 * 99].fill(0.5);
        pcm[320 * 99..320 * 100].fill(0.25);
        assert_eq!(level_gain(&pcm), 0.4);
        pcm.extend([1.0; 100]);
        assert_eq!(level_gain(&pcm), 0.4);
    }

    #[test]
    fn silence_stays_finite_and_frame_count_drops_partial_hop() {
        let frontend = Frontend::new(vec![0.0; 257 * BINS]).unwrap();
        let features = frontend.features(&vec![0.0; HOP * 5 + 3]).unwrap();
        assert_eq!(features, vec![0.0; 5 * BINS]);
        assert!(frontend.features(&[f32::NAN; HOP]).is_err());
    }
}
