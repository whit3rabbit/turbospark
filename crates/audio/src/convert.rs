//! Any supported file to WAV, with optional trim, downmix and resample.
//!
//! This is the one conversion path: speech-model normalization
//! ([`ConvertOptions::speech`]), the preview pane's "Use selection" trim,
//! and WAV export all call [`convert`]. AAC output is the app's job (Apple's
//! encoder through `AVAudioFile`); it reads the WAV this writes.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::decode::DecodedStream;
use crate::error::AudioError;
use crate::resample::{downmix, Resampler};
use crate::wav::{WavSampleFormat, WavWriter};
use crate::SPEECH_SAMPLE_RATE;

/// What to write. Every field is optional on the wire; absent keeps the
/// source's value.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ConvertOptions {
    /// Output rate in Hz.
    pub sample_rate: Option<u32>,
    /// 1 downmixes to mono; the source count is the only other allowed
    /// value (no upmixing: inventing channels is not conversion).
    pub channels: Option<u16>,
    pub sample_format: WavSampleFormat,
    /// Seconds into the source to start at.
    pub start_seconds: Option<f64>,
    /// Seconds into the source to stop at (exclusive).
    pub end_seconds: Option<f64>,
}

impl ConvertOptions {
    /// 16 kHz mono 16-bit: the input shape speech models read.
    pub fn speech() -> Self {
        Self {
            sample_rate: Some(SPEECH_SAMPLE_RATE),
            channels: Some(1),
            sample_format: WavSampleFormat::Int16,
            ..Self::default()
        }
    }

    fn validate(&self, source_channels: usize) -> Result<(), AudioError> {
        if let Some(rate) = self.sample_rate {
            if !(8_000..=192_000).contains(&rate) {
                return Err(AudioError::InvalidOption(format!(
                    "sampleRate must be 8000..=192000, got {rate}"
                )));
            }
        }
        if let Some(channels) = self.channels {
            if channels != 1 && usize::from(channels) != source_channels {
                return Err(AudioError::InvalidOption(format!(
                    "channels must be 1 or the source's {source_channels}, got {channels}"
                )));
            }
        }
        for (name, value) in [
            ("startSeconds", self.start_seconds),
            ("endSeconds", self.end_seconds),
        ] {
            if let Some(v) = value {
                if !v.is_finite() || v < 0.0 {
                    return Err(AudioError::InvalidOption(format!(
                        "{name} must be a non-negative number"
                    )));
                }
            }
        }
        if let (Some(start), Some(end)) = (self.start_seconds, self.end_seconds) {
            if end <= start {
                return Err(AudioError::EmptyRange);
            }
        }
        Ok(())
    }
}

/// What [`convert`] wrote.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConvertReport {
    pub sample_rate: u32,
    pub channels: u32,
    pub frames: u64,
    pub duration_seconds: f64,
}

/// Decodes `source` and writes a WAV to `destination`, overwriting it.
///
/// Trim is applied in SOURCE frames before resampling, so the cut lands on
/// the sample the user selected rather than on a resampled neighbour.
pub fn convert(
    source: &Path,
    destination: &Path,
    options: &ConvertOptions,
) -> Result<ConvertReport, AudioError> {
    let mut stream = DecodedStream::open(source)?;
    let source_rate = stream.sample_rate();
    let source_channels = stream.channels();
    options.validate(source_channels)?;

    let output_rate = options.sample_rate.unwrap_or(source_rate);
    let mono = options.channels == Some(1) && source_channels > 1;
    let output_channels = if mono { 1 } else { source_channels };
    let start_frame = options
        .start_seconds
        .map_or(0, |s| (s * f64::from(source_rate)) as u64);
    let end_frame = options
        .end_seconds
        .map(|s| (s * f64::from(source_rate)) as u64);

    let mut resamplers: Vec<Resampler> = (0..output_channels)
        .map(|_| Resampler::new(source_rate, output_rate))
        .collect();
    let channel_count = u16::try_from(output_channels)
        .map_err(|_| AudioError::InvalidOption("too many channels".to_string()))?;
    let mut writer = WavWriter::create(
        destination,
        output_rate,
        channel_count,
        options.sample_format,
    )?;

    let mut position = 0u64;
    let mut written = 0u64;
    let mut outputs: Vec<Vec<f32>> = vec![Vec::new(); output_channels];
    while let Some(chunk) = stream.next_chunk()? {
        let length = chunk.first().map_or(0, Vec::len) as u64;
        let chunk_start = position;
        position += length;
        if position <= start_frame {
            continue;
        }
        if end_frame.is_some_and(|end| chunk_start >= end) {
            break;
        }
        let from = start_frame.saturating_sub(chunk_start) as usize;
        let to = end_frame.map_or(length, |end| end.min(position) - chunk_start) as usize;
        let slice: Vec<Vec<f32>> = chunk.iter().map(|c| c[from..to].to_vec()).collect();
        let planes = if mono { vec![downmix(&slice)] } else { slice };
        for (index, plane) in planes.iter().enumerate() {
            resamplers[index].push(plane, &mut outputs[index]);
        }
        written += flush(&mut writer, &mut outputs)?;
    }
    for (index, resampler) in resamplers.iter_mut().enumerate() {
        resampler.finish(&mut outputs[index]);
    }
    written += flush(&mut writer, &mut outputs)?;
    writer.finish()?;

    if written == 0 {
        let _ = std::fs::remove_file(destination);
        return Err(AudioError::EmptyRange);
    }
    Ok(ConvertReport {
        sample_rate: output_rate,
        channels: output_channels as u32,
        frames: written,
        duration_seconds: written as f64 / f64::from(output_rate),
    })
}

/// Writes the frames every channel has produced so far and keeps any
/// excess. Resamplers advance in lockstep, so in practice all channels hold
/// the same count; taking the minimum makes that an invariant rather than an
/// assumption.
fn flush(writer: &mut WavWriter, outputs: &mut [Vec<f32>]) -> Result<u64, AudioError> {
    let ready = outputs.iter().map(Vec::len).min().unwrap_or(0);
    if ready == 0 {
        return Ok(0);
    }
    let block: Vec<Vec<f32>> = outputs
        .iter_mut()
        .map(|o| o.drain(..ready).collect())
        .collect();
    writer.write_planar(&block)?;
    Ok(ready as u64)
}

/// Decodes `source` straight to 16 kHz mono samples in memory: the input a
/// [`crate::speech::SpeechToText`] model takes.
///
/// In memory on purpose, unlike [`convert`]: a model consumes the whole
/// utterance at once, and 16 kHz mono is 64 KB per second, so even the
/// app's 30-minute clip cap is about 115 MB.
pub fn load_speech_samples(source: &Path) -> Result<Vec<f32>, AudioError> {
    let mut stream = DecodedStream::open(source)?;
    let mut resampler = Resampler::new(stream.sample_rate(), SPEECH_SAMPLE_RATE);
    let mut samples = Vec::new();
    while let Some(chunk) = stream.next_chunk()? {
        resampler.push(&downmix(&chunk), &mut samples);
    }
    resampler.finish(&mut samples);
    Ok(samples)
}
