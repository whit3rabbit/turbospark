//! Silero VAD end-to-end demo: `cargo run -p turbospark-audio --example
//! silero_vad_demo -- <model_dir> <wav> [probs_out.npy]`. Prints per-chunk
//! probabilities and speech timestamps so the output can be diffed against
//! the Python reference (`/tmp/silero_ref.py`).

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(args.next().expect("model dir"));
    let wav = args.next().expect("wav path");
    let probs_out = args.next();

    let samples = turbospark_audio::read_wav_f32(std::path::Path::new(&wav))?;
    let mono = turbospark_audio::to_mono_resampled(
        &samples,
        16_000,
        &turbospark_audio::MonoResampleStrategy::SincHann(Default::default()),
    )?;
    let model = turbospark_audio::models::silero_vad::SileroVad::load(&model_dir)?;
    let probs = model.predict_proba(&mono, 16_000)?;
    let ts = model.timestamps(&probs, mono.len(), 16_000);
    println!("probs {}", probs.len());
    for (i, p) in probs.iter().enumerate().take(8) {
        println!("p[{i}] {:.4}", p);
    }
    println!("timestamps {:?}", ts);
    if let Some(out) = probs_out {
        // Minimal .npy v1.0 writer: f32 little-endian, 1-D. The header
        // dict ends with a newline and the 10-byte prefix plus header
        // pads to a 64-byte boundary.
        let json = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}",
            probs.len()
        );
        let header_len = {
            let mut len = json.len() + 1; // + newline
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
        for p in &probs {
            bytes.extend_from_slice(&p.to_le_bytes());
        }
        std::fs::write(out, bytes)?;
    }
    Ok(())
}
