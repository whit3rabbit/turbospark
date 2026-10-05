//! Kokoro end-to-end gate: `cargo run -p turbospark-audio --example
//! kokoro_demo -- <model_dir> <voice.safetensors> <phonemes> [out.f32]`.
//! Prints durations and writes the raw f32 waveform so the output can be
//! compared against the Python reference.

use std::path::{Path, PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(args.next().expect("model dir"));
    let voice_path = args.next().expect("voice safetensors");
    let phonemes = args.next().expect("phoneme string");
    let out = args.next();

    let model = turbospark_audio::models::kokoro::Kokoro::open(&model_dir)?;
    eprintln!("phoneme chars: {}", phonemes.chars().count());
    let ids = model.phoneme_ids(&phonemes);
    eprintln!("ids: {} {:?}", ids.len(), ids);
    let voice_file =
        turbospark_model_io::safetensors::SafetensorsFile::open(Path::new(&voice_path))?;
    let voice = voice_file.load_as_f32("voice")?;
    let style_dim = 256;
    let idx = phonemes.chars().count() - 1;
    let ref_s = voice[idx * style_dim..(idx + 1) * style_dim].to_vec();
    let audio = model.generate(&ids, &ref_s, 1.0)?;
    println!("samples {}", audio.len());
    println!(
        "audio std {:.6} peak {:.6}",
        {
            let m = audio.iter().sum::<f32>() / audio.len() as f32;
            (audio.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / audio.len() as f32).sqrt()
        },
        audio.iter().fold(0.0f32, |a, v| a.max(v.abs()))
    );
    if let Some(out) = out {
        let bytes: Vec<u8> = audio.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(out, bytes)?;
    }
    Ok(())
}
