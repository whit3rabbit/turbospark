//! Strict RIFF/WAVE reader and writer.
//!
//! Port of upstream audio.cpp's `wav_reader.cpp` and `wav_writer.cpp`
//! behavior: a container parser that validates every field it acts on and
//! reports the failure with the offending value. Decoded widths are the ones
//! the reader implements: PCM 8 (unsigned), 16, 24, and 32, plus IEEE float
//! 32. `WAVE_FORMAT_EXTENSIBLE` (0xFFFE) is accepted when its sub-format GUID
//! names PCM or IEEE float.
//!
//! Reading takes bytes, not a path, so the crate performs no I/O; the path
//! wrapper is a thin `std::fs` convenience. A hostile or corrupt stream fails
//! with a typed [`AudioError`] before any large allocation: the data chunk
//! length is checked against a cap before the sample buffer is reserved.

use std::path::Path;

use crate::error::AudioError;
use crate::waveform::Waveform;

/// Upper bound on the data chunk this reader will decode into one buffer.
/// One GiB of samples is far past any frontend need (30 stereo minutes at
/// 48 kHz f32 is about 69 MiB) while still refusing multi-GiB hostile
/// lengths before they allocate.
pub const MAX_WAV_DATA_BYTES: usize = 1 << 30;

const FORMAT_PCM: u16 = 0x0001;
const FORMAT_IEEE_FLOAT: u16 = 0x0003;
const FORMAT_EXTENSIBLE: u16 = 0xFFFE;

const ID_RIFF: u32 = u32::from_be_bytes(*b"RIFF");
const ID_WAVE: u32 = u32::from_be_bytes(*b"WAVE");
const ID_FMT: u32 = u32::from_be_bytes(*b"fmt ");
const ID_DATA: u32 = u32::from_be_bytes(*b"data");

/// Decodes a RIFF/WAVE byte stream into an interleaved f32 [`Waveform`].
///
/// Integer widths map to f32 by dividing by the full-scale magnitude
/// (i16 by 32768, i24 by 8388608, i32 by 2147483648), so the negative
/// full-scale sample reaches exactly -1.0. PCM 8-bit is unsigned with a
/// 128 midpoint. Float 32 is passed through unchanged.
pub fn read_wav_f32_bytes(bytes: &[u8]) -> Result<Waveform, AudioError> {
    let mut cursor = Bytes::new(bytes);

    let riff = cursor
        .take_u32be()
        .ok_or_else(|| not_wav("the stream is shorter than a RIFF header"))?;
    if riff != ID_RIFF {
        return Err(not_wav(&format!(
            "stream starts with {:?}",
            String::from_utf8_lossy(&riff.to_be_bytes())
        )));
    }
    // The RIFF size is informational for streaming files; the chunk walk
    // below stops at the end of the actual bytes, so it is not validated.
    let _riff_size = cursor
        .take_u32le()
        .ok_or_else(|| not_wav("the stream ends inside the RIFF header"))?;
    let wave = cursor
        .take_u32be()
        .ok_or_else(|| not_wav("the stream ends inside the RIFF header"))?;
    if wave != ID_WAVE {
        return Err(not_wav(&format!(
            "RIFF form id is {:?}, expected WAVE",
            String::from_utf8_lossy(&wave.to_be_bytes())
        )));
    }

    let mut format_tag = None;
    let mut channels = None;
    let mut sample_rate = None;
    let mut bits = None;

    loop {
        let Some(id) = cursor.take_u32be() else {
            return Err(AudioError::MalformedWav {
                detail: "file ended before a data chunk".to_string(),
            });
        };
        let Some(len) = cursor.take_u32le() else {
            return Err(AudioError::MalformedWav {
                detail: "chunk header truncated before the length field".to_string(),
            });
        };
        let len = len as usize;
        match id {
            ID_FMT => {
                let body = cursor.take(len).ok_or_else(|| AudioError::MalformedWav {
                    detail: format!("fmt chunk length {len} runs past the end of the file"),
                })?;
                parse_fmt(
                    body,
                    &mut format_tag,
                    &mut channels,
                    &mut sample_rate,
                    &mut bits,
                )?;
            }
            ID_DATA => {
                let (tag, ch, rate, bits) = match (format_tag, channels, sample_rate, bits) {
                    (Some(t), Some(c), Some(r), Some(b)) => (t, c, r, b),
                    _ => {
                        return Err(AudioError::MalformedWav {
                            detail: "data chunk appears before the fmt chunk".to_string(),
                        })
                    }
                };
                if len > MAX_WAV_DATA_BYTES {
                    return Err(AudioError::BufferTooLarge {
                        what: "WAV data chunk",
                        samples: len,
                    });
                }
                let body = cursor.take(len).ok_or_else(|| AudioError::MalformedWav {
                    detail: format!("data chunk length {len} runs past the end of the file"),
                })?;
                let width = bits as usize / 8;
                let block = width * ch;
                if len % block != 0 {
                    return Err(AudioError::MalformedWav {
                        detail: format!(
                            "data chunk is {len} bytes, not a whole number of {block}-byte frames"
                        ),
                    });
                }
                let samples = decode_body(body, tag, width)?;
                // Trailing chunks after data are legal and ignored, so the
                // walk ends here; the pad byte after an odd data chunk is
                // trailing padding this reader never needs to consume.
                return Waveform::new(rate, ch as u16, samples);
            }
            _ => {
                // Unknown chunk: skip its body; the word-alignment pad byte
                // that follows an odd-length body is handled after the match.
                cursor.take(len).ok_or_else(|| AudioError::MalformedWav {
                    detail: format!("chunk length {len} runs past the end of the file"),
                })?;
            }
        }
        if len % 2 == 1 {
            // Chunk bodies are word-aligned; consume the pad byte if it is
            // present. A missing pad byte at end-of-file is tolerable here
            // because the data chunk is the only one that matters.
            let _ = cursor.take(1);
        }
    }
}

/// Decodes a RIFF/WAVE file from disk.
pub fn read_wav_f32(path: &Path) -> Result<Waveform, AudioError> {
    let bytes = std::fs::read(path).map_err(|e| AudioError::NotWav {
        detail: format!("could not read {}: {e}", path.display()),
    })?;
    read_wav_f32_bytes(&bytes)
}

/// Encodes a [`Waveform`] as a 32-bit IEEE float WAVE file (format tag 3).
pub fn write_wav_f32(wave: &Waveform) -> Vec<u8> {
    encode_wav(wave, FORMAT_IEEE_FLOAT, 32, |s, data| {
        data.extend_from_slice(&s.to_le_bytes());
    })
}

/// Encodes a [`Waveform`] as a 16-bit PCM WAVE file (format tag 1).
///
/// Samples scale by 32768 with round-to-nearest and a clamp at 32767, the
/// mirror of the reader's `i16 / 32768` mapping: -1.0 encodes to -32768 and
/// decodes back to exactly -1.0, while +1.0 clips to the representable top
/// endpoint 32767.
pub fn write_wav_i16(wave: &Waveform) -> Vec<u8> {
    encode_wav(wave, FORMAT_PCM, 16, |s, data| {
        let scaled = (s * 32768.0).round().clamp(-32768.0, 32767.0) as i16;
        data.extend_from_slice(&scaled.to_le_bytes());
    })
}

fn encode_wav(
    wave: &Waveform,
    format_tag: u16,
    bits: u16,
    // Appends one sample's little-endian bytes (exactly `bits / 8` of them)
    // straight into the data chunk; a per-sample `Vec` costs an allocation
    // for every sample of every encode.
    encode_sample: impl Fn(f32, &mut Vec<u8>),
) -> Vec<u8> {
    let width = bits as usize / 8;
    let mut data = Vec::with_capacity(wave.samples.len() * width);
    for &s in &wave.samples {
        encode_sample(s, &mut data);
    }
    debug_assert_eq!(data.len(), wave.samples.len() * width);
    let pad = data.len() % 2;
    // 4 "WAVE" + 8 fmt header + 16 fmt body + 8 data header + data + pad.
    let riff_size = 4 + 8 + 16 + 8 + data.len() + pad;
    let mut out = Vec::with_capacity(8 + riff_size);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(riff_size as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&format_tag.to_le_bytes());
    out.extend_from_slice(&wave.channels.to_le_bytes());
    out.extend_from_slice(&wave.sample_rate.to_le_bytes());
    let block_align = wave.channels as usize * width;
    let byte_rate = wave.sample_rate as usize * block_align;
    out.extend_from_slice(&(byte_rate as u32).to_le_bytes());
    out.extend_from_slice(&(block_align as u16).to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&data);
    if pad == 1 {
        out.push(0);
    }
    out
}

fn parse_fmt(
    body: &[u8],
    format_tag: &mut Option<u16>,
    channels: &mut Option<usize>,
    sample_rate: &mut Option<u32>,
    bits: &mut Option<u16>,
) -> Result<(), AudioError> {
    if body.len() < 16 {
        return Err(AudioError::MalformedWav {
            detail: format!("fmt chunk is {} bytes, at least 16 required", body.len()),
        });
    }
    let tag = u16::from_le_bytes([body[0], body[1]]);
    let mut effective_tag = tag;
    let effective_bits = u16::from_le_bytes([body[14], body[15]]);
    if tag == FORMAT_EXTENSIBLE {
        if body.len() < 40 {
            return Err(AudioError::MalformedWav {
                detail: format!(
                    "extensible fmt chunk is {} bytes, at least 40 required",
                    body.len()
                ),
            });
        }
        // Match the complete PCM/IEEE-float GUID. Looking at only its
        // low tag bytes can misinterpret an unrelated extensible codec.
        const GUID_TAIL: [u8; 14] = [0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71];
        let extension_size = u16::from_le_bytes([body[16], body[17]]) as usize;
        if extension_size < 22 || extension_size > body.len() - 18 || body[26..40] != GUID_TAIL {
            return Err(AudioError::MalformedWav {
                detail: "invalid extensible format size or sub-format GUID".to_string(),
            });
        }
        effective_tag = u16::from_le_bytes([body[24], body[25]]);
        let valid_bits = u16::from_le_bytes([body[18], body[19]]);
        if valid_bits != 0 && valid_bits != effective_bits {
            return Err(AudioError::UnsupportedBitsPerSample {
                format_tag: effective_tag,
                bits: effective_bits,
            });
        }
    }
    if effective_tag != FORMAT_PCM && effective_tag != FORMAT_IEEE_FLOAT {
        return Err(AudioError::UnsupportedWavFormat {
            format_tag: effective_tag,
        });
    }
    let ch = u16::from_le_bytes([body[2], body[3]]) as usize;
    let rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
    crate::error::check_channel_count(ch)?;
    crate::error::check_sample_rate(rate)?;
    match (effective_tag, effective_bits) {
        (FORMAT_PCM, 8 | 16 | 24 | 32) | (FORMAT_IEEE_FLOAT, 32) => {}
        _ => {
            return Err(AudioError::UnsupportedBitsPerSample {
                format_tag: effective_tag,
                bits: effective_bits,
            })
        }
    }
    *format_tag = Some(effective_tag);
    *channels = Some(ch);
    *sample_rate = Some(rate);
    *bits = Some(effective_bits);
    Ok(())
}

fn decode_body(body: &[u8], format_tag: u16, width: usize) -> Result<Vec<f32>, AudioError> {
    let mut samples = Vec::with_capacity(body.len() / width);
    match (format_tag, width) {
        (FORMAT_PCM, 1) => {
            for &b in body {
                samples.push((f32::from(b) - 128.0) / 128.0);
            }
        }
        (FORMAT_PCM, 2) => {
            for pair in body.chunks_exact(2) {
                samples.push(f32::from(i16::from_le_bytes([pair[0], pair[1]])) / 32768.0);
            }
        }
        (FORMAT_PCM, 3) => {
            for triple in body.chunks_exact(3) {
                // Little-endian 24-bit value, sign-extended from bit 23.
                let raw = u32::from(triple[0])
                    | (u32::from(triple[1]) << 8)
                    | (u32::from(triple[2]) << 16);
                let signed = if raw & 0x0080_0000 != 0 {
                    raw | 0xFF00_0000
                } else {
                    raw
                };
                samples.push(signed as i32 as f32 / 8_388_608.0);
            }
        }
        (FORMAT_PCM, 4) => {
            for quad in body.chunks_exact(4) {
                samples.push(
                    i32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]) as f32
                        / 2_147_483_648.0,
                );
            }
        }
        (FORMAT_IEEE_FLOAT, 4) => {
            for quad in body.chunks_exact(4) {
                samples.push(f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]));
            }
        }
        _ => {
            return Err(AudioError::UnsupportedBitsPerSample {
                format_tag,
                bits: (width * 8) as u16,
            })
        }
    }
    Ok(samples)
}

/// Cursor over the whole byte stream. A `take` returns the requested slice
/// or `None` when the stream cannot satisfy it in full, so callers never see
/// a short body.
struct Bytes<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Bytes<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        if end > self.bytes.len() {
            return None;
        }
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Some(out)
    }

    fn take_u32le(&mut self) -> Option<u32> {
        self.take(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn take_u32be(&mut self) -> Option<u32> {
        self.take(4)
            .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }
}

fn not_wav(detail: &str) -> AudioError {
    AudioError::NotWav {
        detail: detail.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(wave: &Waveform) -> Waveform {
        let bytes = write_wav_f32(wave);
        read_wav_f32_bytes(&bytes).unwrap()
    }

    #[test]
    fn f32_roundtrip_preserves_samples() {
        let wave = Waveform::new(44_100, 2, vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.25]).unwrap();
        let back = roundtrip(&wave);
        assert_eq!(back, wave);
    }

    #[test]
    fn i16_writer_clamps_at_full_scale() {
        let wave = Waveform::mono(16_000, vec![1.0, -1.0, 0.5]).unwrap();
        let bytes = write_wav_i16(&wave);
        let back = read_wav_f32_bytes(&bytes).unwrap();
        // -1.0 is exact through the 32768 path; +1.0 clips to 32767/32768.
        assert!((back.samples[0] - 32767.0 / 32768.0).abs() < 1e-6);
        assert!((back.samples[1] + 1.0).abs() < 1e-7);
        assert!((back.samples[2] - 0.5).abs() < 1e-7);
    }

    #[test]
    fn reader_rejects_non_riff() {
        let err = read_wav_f32_bytes(b"NOTA WAVE junk").unwrap_err();
        assert!(
            err.to_string().starts_with("not a RIFF/WAVE stream"),
            "{err}"
        );
    }

    #[test]
    fn reader_rejects_unsupported_format_tag() {
        // fmt: tag 0x0011 (IMA ADPCM), mono, 8 kHz, 16 bits.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&28u32.to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&0x0011u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&8_000u32.to_le_bytes());
        bytes.extend_from_slice(&16_000u32.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&0u32.to_le_bytes());
        let err = read_wav_f32_bytes(&bytes).unwrap_err();
        assert!(
            err.to_string().contains("format tag 0x0011"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn reader_rejects_truncated_data() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&64u32.to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&8_000u32.to_le_bytes());
        bytes.extend_from_slice(&16_000u32.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 10]);
        let err = read_wav_f32_bytes(&bytes).unwrap_err();
        assert!(
            err.to_string().contains("runs past the end of the file"),
            "unexpected: {err}"
        );
    }
}
