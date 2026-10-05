//! DeepFilterNet enhancement gate: `cargo run -p turbospark-audio
//! --example dfn_enhance -- <model_dir> <48k_wav> <out.f32>`. Reads a
//! 48 kHz mono wav, runs the enhancement, writes raw f32 samples.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = std::path::PathBuf::from(args.next().expect("model dir"));
    let wav = args.next().expect("48k wav");
    let out = args.next().expect("out path");

    let samples = turbospark_audio::read_wav_f32(std::path::Path::new(&wav))?;
    if samples.sample_rate != 48_000 {
        return Err("expected 48 kHz input".into());
    }
    let mono = samples.samples;
    let model = turbospark_audio::models::deepfilternet::DeepFilterNet::open(&model_dir)?;
    let enhanced = model.enhance(&mono)?;
    println!("in {} out {} std {:.6}", mono.len(), enhanced.len(), {
        let m = enhanced.iter().sum::<f32>() / enhanced.len() as f32;
        (enhanced.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / enhanced.len() as f32).sqrt()
    });
    let bytes: Vec<u8> = enhanced.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(out, bytes)?;
    Ok(())
}
