//! Kaldi FBANK and low frame rate stacking used by SenseVoice.

use crate::fft::RealFftPlan;
use crate::{Result, SpeechError};

#[derive(Debug, Clone, Copy)]
pub(in crate::stt) struct FrontendConfig {
    pub sample_rate: usize,
    pub num_mels: usize,
    pub frame_length_ms: usize,
    pub frame_shift_ms: usize,
    pub lfr_m: usize,
    pub lfr_n: usize,
}

pub(super) fn extract_features(
    samples: &[f32],
    config: FrontendConfig,
    cmvn_means: &[f32],
    cmvn_istd: &[f32],
) -> Result<(Vec<f32>, usize)> {
    if samples.is_empty() {
        return Err(SpeechError::Input {
            why: "audio must contain at least one sample".into(),
        });
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(SpeechError::Input {
            why: "audio samples must be finite".into(),
        });
    }
    if config.num_mels == 0 || config.lfr_m == 0 || config.lfr_n == 0 {
        return Err(bad_frontend("mel and LFR dimensions must be positive"));
    }
    if cmvn_means.len() != config.num_mels * config.lfr_m
        || cmvn_istd.len() != config.num_mels * config.lfr_m
    {
        return Err(bad_frontend(
            "CMVN vectors do not match the stacked feature width",
        ));
    }

    let fbank = compute_fbank(samples, config)?;
    if fbank.is_empty() {
        return Err(SpeechError::Input {
            why: "audio is shorter than one complete SenseVoice analysis frame".into(),
        });
    }
    let fbank_frames = fbank.len() / config.num_mels;
    let output_frames = fbank_frames.div_ceil(config.lfr_n);
    let left_pad = (config.lfr_m - 1) / 2;
    let feature_width = config.num_mels * config.lfr_m;
    let mut features = vec![0.0; output_frames * feature_width];

    for out_frame in 0..output_frames {
        for stack_index in 0..config.lfr_m {
            let source = (out_frame * config.lfr_n + stack_index)
                .saturating_sub(left_pad)
                .min(fbank_frames - 1);
            let source_row = &fbank[source * config.num_mels..(source + 1) * config.num_mels];
            let target_start = out_frame * feature_width + stack_index * config.num_mels;
            for mel in 0..config.num_mels {
                let col = stack_index * config.num_mels + mel;
                features[target_start + mel] = (source_row[mel] + cmvn_means[col]) * cmvn_istd[col];
            }
        }
    }
    Ok((features, output_frames))
}

pub(in crate::stt) fn compute_fbank(samples: &[f32], config: FrontendConfig) -> Result<Vec<f32>> {
    let window_size = config.sample_rate * config.frame_length_ms / 1000;
    let window_shift = config.sample_rate * config.frame_shift_ms / 1000;
    if window_size < 2 || window_shift == 0 {
        return Err(bad_frontend(
            "analysis frame and shift must be at least two and one samples",
        ));
    }
    if samples.len() < window_size {
        return Ok(Vec::new());
    }
    let frame_count = 1 + (samples.len() - window_size) / window_shift;
    let fft_size = window_size.next_power_of_two();
    let fft = RealFftPlan::new(fft_size).map_err(SpeechError::from)?;
    let mel_weights = kaldi_mel_weights(config.num_mels, fft_size, config.sample_rate);
    let mut frame = vec![0.0f32; fft_size];
    let mut output = vec![0.0f32; frame_count * config.num_mels];

    for frame_index in 0..frame_count {
        let start = frame_index * window_shift;
        let input = &samples[start..start + window_size];
        let mean = input.iter().map(|&sample| sample * 32768.0).sum::<f32>() / window_size as f32;
        for index in 0..window_size {
            let current = input[index] * 32768.0 - mean;
            let emphasized = if index == 0 {
                current
            } else {
                current - 0.97 * (input[index - 1] * 32768.0 - mean)
            };
            let phase = 2.0 * std::f32::consts::PI * index as f32 / (window_size - 1) as f32;
            let hamming = 0.54 - 0.46 * phase.cos();
            frame[index] = emphasized * hamming;
        }
        frame[window_size..].fill(0.0);

        let spectrum = fft.forward(&frame).map_err(SpeechError::from)?;
        for mel in 0..config.num_mels {
            let weights = &mel_weights[mel * (fft_size / 2)..(mel + 1) * (fft_size / 2)];
            let energy = spectrum[..fft_size / 2]
                .iter()
                .zip(weights)
                .map(|(bin, weight)| (bin.re * bin.re + bin.im * bin.im) * weight)
                .sum::<f32>();
            output[frame_index * config.num_mels + mel] = energy.max(1e-8).ln();
        }
    }
    Ok(output)
}

fn kaldi_mel_weights(num_mels: usize, fft_size: usize, sample_rate: usize) -> Vec<f32> {
    let nyquist = sample_rate as f32 * 0.5;
    let low_freq = 20.0f32;
    let low_mel = kaldi_mel_scale(low_freq);
    let high_mel = kaldi_mel_scale(nyquist);
    let mel_delta = (high_mel - low_mel) / (num_mels + 1) as f32;
    let bins = fft_size / 2;
    let mut weights = vec![0.0f32; num_mels * bins];
    for mel in 0..num_mels {
        let left = low_mel + mel as f32 * mel_delta;
        let center = low_mel + (mel + 1) as f32 * mel_delta;
        let right = low_mel + (mel + 2) as f32 * mel_delta;
        for bin in 0..bins {
            let freq = bin as f32 * sample_rate as f32 / fft_size as f32;
            let mel_freq = kaldi_mel_scale(freq);
            let up = (mel_freq - left) / (center - left);
            let down = (right - mel_freq) / (right - center);
            weights[mel * bins + bin] = up.min(down).max(0.0);
        }
    }
    weights
}

fn kaldi_mel_scale(freq: f32) -> f32 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

fn bad_frontend(why: &str) -> SpeechError {
    SpeechError::BadConfig {
        field: "frontend_conf".into(),
        why: why.into(),
    }
}
