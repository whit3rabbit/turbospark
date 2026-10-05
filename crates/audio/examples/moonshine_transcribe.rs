//! Moonshine CPU reference: `cargo run --release -p turbospark-audio
//! --example moonshine_transcribe -- <model_dir> <wav>`.

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = std::path::PathBuf::from(args.next().expect("model dir"));
    let wav = args.next().expect("wav path");
    let input = turbospark_audio::read_wav_f32(std::path::Path::new(&wav))?;
    let samples = turbospark_audio::to_mono_resampled(
        &input,
        16_000,
        &turbospark_audio::MonoResampleStrategy::SincHann(Default::default()),
    )?;
    let opened = Instant::now();
    let model = turbospark_audio::models::stt::moonshine::Moonshine::open(&model_dir)?;
    let open_ms = opened.elapsed().as_secs_f64() * 1_000.0;
    let started = Instant::now();
    let text = model.transcribe(&samples)?;
    println!("TEXT: {text}");
    eprintln!(
        "moonshine_cpu open_ms={open_ms:.2} transcribe_ms={:.2} samples={}",
        started.elapsed().as_secs_f64() * 1_000.0,
        samples.len()
    );
    Ok(())
}
