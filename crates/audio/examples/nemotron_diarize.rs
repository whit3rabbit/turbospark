//! Nemotron 3 Diarization demo: `cargo run -p turbospark-audio --example
//! nemotron_diarize -- <model_dir> <mode> <wav-or-npy> [out_prefix]`.
//!
//! Modes: `offline` (generate), `stream` (generate_stream), `window`
//! (single-window forward on extracted mel features). Prints speaker
//! segments (RTTM-like) so the output can be diffed against the Python
//! reference (`/tmp/nemotron_ref.py`). Input must be mono 16 kHz WAV, or
//! a raw f32 `.npy` of samples for bit-identical comparison with the
//! reference. Set `TURBOSPEECH_NEMOTRON_DUMP=<dir>` to dump per-stage
//! intermediates as raw little-endian f32 files for stage bisection.

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(args.next().expect("model dir"));
    let mode = args.next().expect("mode: offline|stream|window");
    let wav = args.next().expect("wav path");
    let dump_dir = std::env::var("TURBOSPEECH_NEMOTRON_DUMP").ok();

    let model =
        turbospark_audio::models::vad::nemotron_diarization::NemotronDiarization::load(&model_dir)?;
    let samples: Vec<f32> = if wav.ends_with(".npy") {
        read_npy_f32(&wav)?
    } else {
        let w = turbospark_audio::read_wav_f32(std::path::Path::new(&wav))?;
        turbospark_audio::to_mono_resampled(
            &w,
            model.config().processor.sampling_rate,
            &turbospark_audio::MonoResampleStrategy::SincHann(Default::default()),
        )?
    };
    let n_spk = model.config().num_speakers;
    let opts = turbospark_audio::models::vad::sortformer::GenerateOptions::default();

    match mode.as_str() {
        "window" => {
            let total = samples.len();
            let count = total / model.config().processor.hop_length;
            let mel = model.mel_features(&samples, 0, count, 0, total)?;
            if let Some(dir) = &dump_dir {
                dump(&format!("{dir}/rust_mel.bin"), &mel);
            }
            let preds = model.forward_window(&mel, count, count);
            if let Some(dir) = &dump_dir {
                dump(&format!("{dir}/rust_window_preds.bin"), &preds);
            }
            print_segments(&preds, n_spk, 0.01);
        }
        "stream" => {
            let (results, state) = model.generate_stream(&samples, &opts)?;
            let mut all = Vec::new();
            for (i, r) in results.iter().enumerate() {
                println!(
                    "chunk {i}: frames {} segments {}",
                    r.speaker_probs.len() / n_spk,
                    r.segments.len()
                );
                if let Some(dir) = &dump_dir {
                    dump(&format!("{dir}/rust_feed_{i}.bin"), &r.speaker_probs);
                }
                all.extend_from_slice(&r.speaker_probs);
            }
            if let Some(dir) = &dump_dir {
                dump(&format!("{dir}/rust_spkcache.bin"), &state.spkcache);
                dump(
                    &format!("{dir}/rust_spkcache_preds.bin"),
                    &state.spkcache_preds,
                );
                dump(&format!("{dir}/rust_fifo.bin"), &state.fifo);
                dump(&format!("{dir}/rust_fifo_preds.bin"), &state.fifo_preds);
            }
            println!(
                "state: frames_processed {} spkcache {} fifo {} compressed {}",
                state.frames_processed,
                state.spkcache.len() / model.config().encoder.d_model,
                state.fifo.len() / model.config().encoder.d_model,
                state.spkcache_compressed
            );
            print_segments(&all, n_spk, 0.01);
        }
        "offline" => {
            let out = model.generate(&samples, &opts)?;
            if let Some(dir) = &dump_dir {
                dump(&format!("{dir}/rust_offline_preds.bin"), &out.speaker_probs);
            }
            println!("num_speakers {}", out.num_speakers);
            print_segments(&out.speaker_probs, n_spk, 0.01);
        }
        other => return Err(format!("unknown mode {other}").into()),
    }
    Ok(())
}

fn print_segments(preds: &[f32], n_spk: usize, stride: f64) {
    let segs = turbospark_audio::models::vad::sortformer::preds_to_segments(
        preds, n_spk, stride, 0.5, 0.0, 0.0,
    );
    for seg in &segs {
        println!(
            "SPEAKER audio 1 {:.3} {:.3} <NA> <NA> speaker_{} <NA> <NA>",
            seg.start,
            seg.end - seg.start,
            seg.speaker
        );
    }
    println!("num_segments {}", segs.len());
}

fn dump(path: &str, data: &[f32]) {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    if let Err(e) = std::fs::write(path, bytes) {
        eprintln!("dump {path} failed: {e}");
    }
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
