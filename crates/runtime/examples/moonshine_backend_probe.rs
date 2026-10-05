//! Compare the portable and opt-in Metal Moonshine runners on a WAV clip.
//! Set TURBOSPARK_MOONSHINE_DEVICE=metal to select the experimental backend.

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = std::path::PathBuf::from(args.next().expect("model dir"));
    let wav = args.next().expect("wav path");
    let open_only = args.next().as_deref() == Some("--open-only");
    let opened = Instant::now();
    let model = turbospark_runtime::moonshine::MoonshineRunner::open(&model_dir)?;
    let open_ms = opened.elapsed().as_secs_f64() * 1_000.0;
    if open_only {
        eprintln!(
            "moonshine_{} open_ms={open_ms:.2}",
            if model.using_metal() { "metal" } else { "cpu" }
        );
        return Ok(());
    }
    let input = audio::read_wav_f32(std::path::Path::new(&wav))?;
    let samples = audio::to_mono_resampled(
        &input,
        16_000,
        &audio::MonoResampleStrategy::SincHann(Default::default()),
    )?;
    let started = Instant::now();
    let text = model.transcribe(&samples)?;
    println!("TEXT: {text}");
    eprintln!(
        "moonshine_{} open_ms={open_ms:.2} transcribe_ms={:.2} samples={}",
        if model.using_metal() { "metal" } else { "cpu" },
        started.elapsed().as_secs_f64() * 1_000.0,
        samples.len()
    );
    Ok(())
}
