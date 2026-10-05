//! Byte-level WAV coverage: containers built by hand, one field at a time,
//! so a decode failure points at the exact header field this crate
//! misparses. Complements the unit tests, which mostly round-trip through
//! this crate's own writer.

use turbospark_audio::read_wav_f32_bytes;
use turbospark_audio::wav::{write_wav_f32, write_wav_i16};
use turbospark_audio::waveform::Waveform;

/// Assembles a minimal WAVE container from parts.
fn container(fmt_body: &[u8], data_body: &[u8], extra_chunk: Option<(&[u8; 4], &[u8])>) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    let riff_size = 4
        + 8
        + fmt_body.len()
        + 8
        + data_body.len()
        + (data_body.len() % 2)
        + extra_chunk.map_or(0, |(_id, body)| 8 + body.len() + (body.len() % 2));
    out.extend_from_slice(&(riff_size as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&(fmt_body.len() as u32).to_le_bytes());
    out.extend_from_slice(fmt_body);
    if let Some((id, body)) = extra_chunk {
        out.extend_from_slice(id);
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(body);
        if body.len() % 2 == 1 {
            out.push(0);
        }
    }
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_body.len() as u32).to_le_bytes());
    out.extend_from_slice(data_body);
    if data_body.len() % 2 == 1 {
        out.push(0);
    }
    out
}

fn fmt_body_pcm(channels: u16, rate: u32, bits: u16) -> Vec<u8> {
    let width = bits / 8;
    let mut body = Vec::new();
    body.extend_from_slice(&1u16.to_le_bytes());
    body.extend_from_slice(&channels.to_le_bytes());
    body.extend_from_slice(&rate.to_le_bytes());
    body.extend_from_slice(&(rate * u32::from(channels) * u32::from(width)).to_le_bytes());
    body.extend_from_slice(&(channels * width).to_le_bytes());
    body.extend_from_slice(&bits.to_le_bytes());
    body
}

#[test]
fn decodes_24_bit_pcm() {
    // Two stereo frames; full-scale negative, then a quarter positive on
    // both channels. 8388608 = 2^23, 2097152 = 2^21.
    let mut data = Vec::new();
    for raw in [0x800000u32, 0x200000, 0x800000, 0x200000] {
        let v = raw.to_le_bytes();
        data.extend_from_slice(&v[..3]);
    }
    let bytes = container(&fmt_body_pcm(2, 48_000, 24), &data, None);
    let wave = read_wav_f32_bytes(&bytes).unwrap();
    assert_eq!(wave.sample_rate, 48_000);
    assert_eq!(wave.channels, 2);
    assert_eq!(wave.frame_count(), 2);
    assert!((wave.samples[0] + 1.0).abs() < 1e-7, "{}", wave.samples[0]);
    assert!((wave.samples[1] - 0.25).abs() < 1e-7, "{}", wave.samples[1]);
}

#[test]
fn decodes_8_bit_unsigned() {
    // Midpoint 128 decodes to silence; 255 is the top endpoint.
    let data = [128u8, 255];
    let bytes = container(&fmt_body_pcm(1, 8_000, 8), &data, None);
    let wave = read_wav_f32_bytes(&bytes).unwrap();
    assert!((wave.samples[0]).abs() < 1e-7);
    assert!((wave.samples[1] - 127.0 / 128.0).abs() < 1e-7);
}

#[test]
fn decodes_extensible_pcm() {
    // WAVE_FORMAT_EXTENSIBLE wrapper whose sub-format GUID names PCM 16.
    let mut fmt = Vec::new();
    fmt.extend_from_slice(&0xFFFEu16.to_le_bytes());
    fmt.extend_from_slice(&1u16.to_le_bytes()); // channels
    fmt.extend_from_slice(&16_000u32.to_le_bytes());
    fmt.extend_from_slice(&32_000u32.to_le_bytes()); // byte rate
    fmt.extend_from_slice(&2u16.to_le_bytes()); // block align
    fmt.extend_from_slice(&16u16.to_le_bytes()); // bits
    fmt.extend_from_slice(&22u16.to_le_bytes()); // cbSize
    fmt.extend_from_slice(&16u16.to_le_bytes()); // valid bits
    fmt.extend_from_slice(&0u32.to_le_bytes()); // channel mask
                                                // KSDATAFORMAT_SUBTYPE_PCM: 00000001-0000-0010-8000-00aa00389b71
    fmt.extend_from_slice(&[
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b,
        0x71,
    ]);
    let data = [0u8; 4];
    let bytes = container(&fmt, &data, None);
    let wave = read_wav_f32_bytes(&bytes).unwrap();
    assert_eq!(wave.frame_count(), 2);
    assert_eq!(wave.samples, vec![0.0, 0.0]);
}

#[test]
fn skips_unknown_chunks_before_data() {
    // A LIST chunk with an odd body length forces the pad-byte path.
    let bytes = container(
        &fmt_body_pcm(1, 8_000, 16),
        &0i16.to_le_bytes(),
        Some((b"LIST", b"INFOabc")),
    );
    let wave = read_wav_f32_bytes(&bytes).unwrap();
    assert_eq!(wave.frame_count(), 1);
}

#[test]
fn writer_output_has_riff_sizes_that_match_the_bytes() {
    // Mono i16 data is even; a single odd sample forces the data pad byte,
    // which must be counted in the RIFF size.
    let wave = Waveform::mono(8_000, vec![0.5]).unwrap();
    let bytes = write_wav_i16(&wave);
    let riff_size = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    assert_eq!(bytes.len(), 8 + riff_size);
    let back = read_wav_f32_bytes(&bytes).unwrap();
    assert_eq!(back.frame_count(), 1);

    // Three mono f32 samples are 12 bytes (even); the writer's own sizes
    // must agree with the byte count too.
    let wave = Waveform::mono(8_000, vec![0.1, 0.2, 0.3]).unwrap();
    let bytes = write_wav_f32(&wave);
    let riff_size = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    assert_eq!(bytes.len(), 8 + riff_size);
    let back = read_wav_f32_bytes(&bytes).unwrap();
    assert_eq!(back.samples.len(), 3);
}

#[test]
#[ignore = "exercises std::fs; run explicitly where filesystem writes are allowed"]
fn reads_from_disk() {
    let wave = Waveform::mono(16_000, vec![0.0, 0.5, -0.5]).unwrap();
    let path = std::env::temp_dir().join("turbospark-audio-roundtrip.wav");
    std::fs::write(&path, write_wav_f32(&wave)).unwrap();
    let back = turbospark_audio::read_wav_f32(&path).unwrap();
    assert_eq!(back, wave);
    std::fs::remove_file(&path).ok();
}

#[test]
fn decodes_signed_32_bit_pcm_at_the_documented_scale() {
    let data: Vec<u8> = [i32::MIN, 536870912, 0]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect();
    let wave = read_wav_f32_bytes(&container(&fmt_body_pcm(1, 16000, 32), &data, None)).unwrap();
    assert_eq!(wave.samples, vec![-1.0, 0.25, 0.0]);
}

#[test]
fn refuses_extensible_guid_suffix_and_extension_size_corruption() {
    let mut fmt = fmt_body_pcm(1, 16000, 16);
    fmt[..2].copy_from_slice(&0xFFFEu16.to_le_bytes());
    fmt.extend_from_slice(&22u16.to_le_bytes());
    fmt.extend_from_slice(&16u16.to_le_bytes());
    fmt.extend_from_slice(&0u32.to_le_bytes());
    fmt.extend_from_slice(&[
        1, 0, 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71,
    ]);
    read_wav_f32_bytes(&container(&fmt, &[0, 0], None)).unwrap();
    let mut corrupt = fmt.clone();
    corrupt[39] ^= 1;
    assert!(read_wav_f32_bytes(&container(&corrupt, &[0, 0], None)).is_err());
    corrupt = fmt;
    corrupt[16..18].copy_from_slice(&0u16.to_le_bytes());
    assert!(read_wav_f32_bytes(&container(&corrupt, &[0, 0], None)).is_err());
}
