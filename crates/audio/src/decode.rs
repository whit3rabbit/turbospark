//! Streaming decode of any supported file into planar `f32` chunks.
//!
//! Streaming, never whole-file: a 30-minute 48 kHz stereo capture is about
//! 690 MB as `f32`, and every consumer here (peaks, resample, WAV write)
//! needs only a running window. A caller that wants the whole clip collects
//! chunks itself and owns that memory decision.

use std::fs::File;
use std::path::Path;

use serde::Serialize;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{Decoder, DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::error::AudioError;
use crate::formats::{classify_extension, FileClass};

/// One decoded run of audio: `channels[c][i]` is frame `i` of channel `c`.
/// Every channel has the same length.
pub type PlanarChunk = Vec<Vec<f32>>;

/// What a file is, without decoding more than needed.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProbeReport {
    pub sample_rate: u32,
    pub channels: u32,
    /// Total frames per channel.
    pub frames: u64,
    pub duration_seconds: f64,
    /// Codec short name from the decoder registry, for example `aac`.
    pub codec: String,
}

/// An open file yielding planar chunks until exhausted.
pub struct DecodedStream {
    format: Box<dyn FormatReader>,
    decoder: Box<dyn Decoder>,
    track_id: u32,
    sample_rate: u32,
    channels: usize,
    codec: String,
    declared_frames: Option<u64>,
    /// The first chunk, decoded eagerly in `open` so the true rate and
    /// channel count are known before the caller builds anything.
    pending: Option<PlanarChunk>,
    finished: bool,
}

impl DecodedStream {
    /// Opens `path`, picks its first audio track, and decodes the first
    /// chunk. Refuses a known-undecodable extension up front with
    /// [`AudioError::Unsupported`] rather than a probe error.
    pub fn open(path: &Path) -> Result<Self, AudioError> {
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if classify_extension(extension) == FileClass::Refused {
            return Err(AudioError::Unsupported(format!(".{extension}")));
        }
        let file = File::open(path)?;
        let stream = MediaSourceStream::new(Box::new(file), Default::default());
        let mut hint = Hint::new();
        if !extension.is_empty() {
            hint.with_extension(extension);
        }
        let probed = symphonia::default::get_probe()
            .format(
                &hint,
                stream,
                &FormatOptions::default(),
                &MetadataOptions::default(),
            )
            .map_err(|e| map_open_error(e, extension))?;
        let format = probed.format;
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or_else(|| AudioError::Decode("the file has no audio track".to_string()))?;
        let track_id = track.id;
        let params = track.codec_params.clone();
        let codec = symphonia::default::get_codecs()
            .get_codec(params.codec)
            .map(|d| d.short_name.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let decoder = symphonia::default::get_codecs()
            .make(&params, &DecoderOptions::default())
            .map_err(|_| AudioError::Unsupported(format!("{codec} audio")))?;

        let mut stream = Self {
            format,
            decoder,
            track_id,
            sample_rate: params.sample_rate.unwrap_or(0),
            channels: params.channels.map(|c| c.count()).unwrap_or(0),
            codec,
            declared_frames: params.n_frames,
            pending: None,
            finished: false,
        };
        // Decode ahead so `sample_rate`/`channels` reflect what the codec
        // actually produces; container headers are allowed to be absent or
        // wrong (raw AAC, some MP3s).
        stream.pending = stream.decode_next()?;
        if stream.sample_rate == 0 || stream.channels == 0 {
            return Err(AudioError::Decode(
                "the file declares no sample rate or channel layout".to_string(),
            ));
        }
        Ok(stream)
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn codec(&self) -> &str {
        &self.codec
    }

    /// Frames per channel the container declares, when it declares any.
    pub fn declared_frames(&self) -> Option<u64> {
        self.declared_frames
    }

    /// The next chunk, or `None` at end of stream.
    pub fn next_chunk(&mut self) -> Result<Option<PlanarChunk>, AudioError> {
        if let Some(chunk) = self.pending.take() {
            return Ok(Some(chunk));
        }
        self.decode_next()
    }

    fn decode_next(&mut self) -> Result<Option<PlanarChunk>, AudioError> {
        while !self.finished {
            let packet = match self.format.next_packet() {
                Ok(packet) => packet,
                Err(SymphoniaError::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    self.finished = true;
                    return Ok(None);
                }
                Err(SymphoniaError::ResetRequired) => {
                    // A chained stream changing parameters mid-file. Rare in
                    // the formats here; stopping is safer than mixing two
                    // layouts into one buffer.
                    self.finished = true;
                    return Ok(None);
                }
                Err(e) => return Err(AudioError::Decode(e.to_string())),
            };
            if packet.track_id() != self.track_id {
                continue;
            }
            let decoded = match self.decoder.decode(&packet) {
                Ok(decoded) => decoded,
                // One corrupt packet should cost one packet, not the file.
                Err(SymphoniaError::DecodeError(_)) => continue,
                Err(SymphoniaError::Unsupported(what)) => {
                    return Err(AudioError::Unsupported(what.to_string()))
                }
                Err(e) => return Err(AudioError::Decode(e.to_string())),
            };
            let spec = *decoded.spec();
            let frames = decoded.frames();
            if frames == 0 {
                continue;
            }
            let channels = spec.channels.count();
            if self.sample_rate == 0 {
                self.sample_rate = spec.rate;
            }
            if self.channels == 0 {
                self.channels = channels;
            }
            let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
            buffer.copy_planar_ref(decoded);
            let samples = buffer.samples();
            let chunk = (0..channels)
                .map(|c| samples[c * frames..(c + 1) * frames].to_vec())
                .collect();
            return Ok(Some(chunk));
        }
        Ok(None)
    }
}

fn map_open_error(error: SymphoniaError, extension: &str) -> AudioError {
    match error {
        SymphoniaError::IoError(e) => AudioError::Io(e),
        SymphoniaError::Unsupported(_) => AudioError::Unsupported(if extension.is_empty() {
            "unknown container".to_string()
        } else {
            format!(".{extension}")
        }),
        other => AudioError::Decode(other.to_string()),
    }
}

/// Reads the format, rate, channels and length of `path`.
///
/// Uses the container's declared frame count when it has one, and decodes
/// to count otherwise (an MP3 with no Xing header declares nothing).
pub fn probe(path: &Path) -> Result<ProbeReport, AudioError> {
    let mut stream = DecodedStream::open(path)?;
    let frames = match stream.declared_frames() {
        Some(frames) => frames,
        None => {
            let mut counted = 0u64;
            while let Some(chunk) = stream.next_chunk()? {
                counted += chunk.first().map_or(0, |c| c.len()) as u64;
            }
            counted
        }
    };
    let sample_rate = stream.sample_rate();
    Ok(ProbeReport {
        sample_rate,
        channels: stream.channels() as u32,
        frames,
        duration_seconds: frames as f64 / f64::from(sample_rate.max(1)),
        codec: stream.codec().to_string(),
    })
}
