//! Minimal streaming RIFF/WAVE writer: interleaved 16-bit PCM or 32-bit
//! float, sizes patched on `finish`.
//!
//! Hand-written because the format is two fixed headers and a data run, and
//! because the reader side is symphonia's: the round-trip tests decode what
//! this writes, which checks both halves against each other.

use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use serde::Deserialize;

use crate::error::AudioError;

/// Sample encoding in the written file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum WavSampleFormat {
    /// 16-bit signed integer. What speech pipelines and most tools expect.
    #[default]
    Int16,
    /// 32-bit IEEE float. No clipping and no quantization, for exports.
    Float32,
}

impl WavSampleFormat {
    fn bytes_per_sample(self) -> u16 {
        match self {
            WavSampleFormat::Int16 => 2,
            WavSampleFormat::Float32 => 4,
        }
    }
}

/// Writes one WAV file. Drop without `finish` leaves a file whose header
/// still claims zero data, which every reader treats as empty rather than
/// as a size it would read past.
pub struct WavWriter {
    out: BufWriter<File>,
    channels: u16,
    format: WavSampleFormat,
    data_bytes: u64,
}

/// RIFF sizes are 32-bit; past this the file cannot describe itself.
const MAX_DATA_BYTES: u64 = u32::MAX as u64 - 64;

impl WavWriter {
    pub fn create(
        path: &Path,
        sample_rate: u32,
        channels: u16,
        format: WavSampleFormat,
    ) -> Result<Self, AudioError> {
        if sample_rate == 0 || channels == 0 {
            return Err(AudioError::InvalidOption(
                "WAV needs a non-zero sample rate and channel count".to_string(),
            ));
        }
        let mut out = BufWriter::new(File::create(path)?);
        let bytes_per_sample = format.bytes_per_sample();
        let block_align = channels * bytes_per_sample;
        let format_tag: u16 = match format {
            WavSampleFormat::Int16 => 1,
            WavSampleFormat::Float32 => 3,
        };
        out.write_all(b"RIFF")?;
        out.write_all(&0u32.to_le_bytes())?; // patched in finish
        out.write_all(b"WAVE")?;
        out.write_all(b"fmt ")?;
        out.write_all(&16u32.to_le_bytes())?;
        out.write_all(&format_tag.to_le_bytes())?;
        out.write_all(&channels.to_le_bytes())?;
        out.write_all(&sample_rate.to_le_bytes())?;
        out.write_all(&(sample_rate * u32::from(block_align)).to_le_bytes())?;
        out.write_all(&block_align.to_le_bytes())?;
        out.write_all(&(bytes_per_sample * 8).to_le_bytes())?;
        out.write_all(b"data")?;
        out.write_all(&0u32.to_le_bytes())?; // patched in finish
        Ok(Self {
            out,
            channels,
            format,
            data_bytes: 0,
        })
    }

    /// Appends planar frames; every channel slice must be the same length
    /// and there must be exactly `channels` of them.
    pub fn write_planar(&mut self, channels: &[Vec<f32>]) -> Result<(), AudioError> {
        if channels.len() != usize::from(self.channels) {
            return Err(AudioError::InvalidOption(format!(
                "expected {} channels, got {}",
                self.channels,
                channels.len()
            )));
        }
        let frames = channels.first().map_or(0, Vec::len);
        if channels.iter().any(|c| c.len() != frames) {
            return Err(AudioError::InvalidOption(
                "channel buffers differ in length".to_string(),
            ));
        }
        let added =
            frames as u64 * u64::from(self.channels) * u64::from(self.format.bytes_per_sample());
        if self.data_bytes + added > MAX_DATA_BYTES {
            return Err(AudioError::InvalidOption(
                "output exceeds the 4 GB WAV limit".to_string(),
            ));
        }
        for i in 0..frames {
            for channel in channels {
                let sample = channel[i];
                match self.format {
                    WavSampleFormat::Int16 => {
                        let clamped = sample.clamp(-1.0, 1.0);
                        let value = (clamped * 32_767.0).round() as i16;
                        self.out.write_all(&value.to_le_bytes())?;
                    }
                    WavSampleFormat::Float32 => self.out.write_all(&sample.to_le_bytes())?,
                }
            }
        }
        self.data_bytes += added;
        Ok(())
    }

    /// Patches the RIFF and data sizes and flushes. Returns data bytes.
    pub fn finish(mut self) -> Result<u64, AudioError> {
        self.out.flush()?;
        let mut file = self
            .out
            .into_inner()
            .map_err(|e| AudioError::Io(e.into_error()))?;
        let data = self.data_bytes as u32;
        file.seek(SeekFrom::Start(4))?;
        file.write_all(&(36 + data).to_le_bytes())?;
        file.seek(SeekFrom::Start(40))?;
        file.write_all(&data.to_le_bytes())?;
        file.flush()?;
        Ok(self.data_bytes)
    }
}

/// Writes a whole mono buffer. Convenience for TTS output and tests.
pub fn write_mono(
    path: &Path,
    samples: &[f32],
    sample_rate: u32,
    format: WavSampleFormat,
) -> Result<(), AudioError> {
    let mut writer = WavWriter::create(path, sample_rate, 1, format)?;
    writer.write_planar(&[samples.to_vec()])?;
    writer.finish()?;
    Ok(())
}
