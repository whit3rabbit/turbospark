//! Sortformer diarization demo: `cargo run -p turbospark-audio --example
//! sortformer_diarize -- <model_dir> <wav-or-npy> [preds_out.npy]`.
//!
//! Prints the speaker segments (RTTM-like) so the output can be diffed
//! against the Python reference (`/tmp/sortformer_ref.py`). Input must be
//! mono 16 kHz WAV, or a raw f32 `.npy` of samples for bit-identical
//! comparison with the reference. Set `TURBOSPEECH_SORTFORMER_DUMP=<dir>`
//! to dump per-stage intermediates as raw little-endian f32 files.

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(args.next().expect("model dir"));
    let wav = args.next().expect("wav path");
    let preds_out = args.next();

    let model = turbospark_audio::models::sortformer::Sortformer::load(&model_dir)?;
    let proc = model.config().processor.clone();

    let samples: Vec<f32> = if wav.ends_with(".npy") {
        read_npy_f32(&wav)?
    } else {
        let w = turbospark_audio::read_wav_f32(std::path::Path::new(&wav))?;
        turbospark_audio::to_mono_resampled(
            &w,
            proc.sampling_rate,
            &turbospark_audio::MonoResampleStrategy::SincHann(Default::default()),
        )?
    };

    let out = model.generate(&samples, proc.sampling_rate, &Default::default())?;
    println!(
        "frames {}",
        out.speaker_probs.len() / model.config().modules.num_speakers
    );
    for seg in &out.segments {
        println!(
            "SPEAKER audio 1 {:.3} {:.3} <NA> <NA> speaker_{} <NA> <NA>",
            seg.start,
            seg.end - seg.start,
            seg.speaker
        );
    }
    println!("num_speakers {}", out.num_speakers);

    if let Some(out_path) = preds_out {
        write_npy_f32(out_path.as_str(), &out.speaker_probs)?;
    }
    Ok(())
}

/// Minimal .npy v1.0 writer: f32 little-endian, 1-D.
pub fn write_npy_f32(path: &str, data: &[f32]) -> std::io::Result<()> {
    let json = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}",
        data.len()
    );
    let header_len = {
        let mut len = json.len() + 1;
        while (10 + len) % 64 != 0 {
            len += 1;
        }
        len
    };
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"\x93NUMPY");
    bytes.extend_from_slice(&[1, 0]);
    bytes.extend_from_slice(&(header_len as u16).to_le_bytes());
    bytes.extend_from_slice(json.as_bytes());
    bytes.extend(std::iter::repeat_n(b' ', header_len - json.len() - 1));
    bytes.push(b'\n');
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes)
}

/// Minimal .npy reader for f32 arrays of any rank saved by numpy
/// (little-endian, not fortran order).
pub fn read_npy_f32(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if &bytes[..6] != b"\x93NUMPY" {
        return Err("not an npy file".into());
    }
    let major = bytes[6];
    let header_len = if major == 1 {
        u16::from_le_bytes([bytes[8], bytes[9]]) as usize
    } else {
        u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize
    };
    let offset = if major == 1 { 10 } else { 12 };
    let header = std::str::from_utf8(&bytes[offset..offset + header_len])?;
    if !header.contains("<f4") {
        return Err(format!("only little-endian f4 supported, header: {header}").into());
    }
    if header.contains("fortran_order': True") {
        return Err("fortran order not supported".into());
    }
    let data_start = offset + header_len;
    let count = (bytes.len() - data_start) / 4;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let b = &bytes[data_start + i * 4..data_start + i * 4 + 4];
        out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    }
    Ok(out)
}
