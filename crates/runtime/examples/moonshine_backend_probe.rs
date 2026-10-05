//! Compare the portable and opt-in Metal Moonshine runners on a WAV clip.
//! Set TURBOSPARK_MOONSHINE_DEVICE=metal to select the experimental backend.

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = std::path::PathBuf::from(args.next().expect("model dir"));
    let wav = args.next().expect("wav path");
    let mode = args.next();
    if !matches!(
        mode.as_deref(),
        None | Some("--open-only") | Some("--warmup")
    ) || args.next().is_some()
    {
        return Err("expected <model-dir> <wav> [--open-only|--warmup]".into());
    }
    let opened = Instant::now();
    let model = turbospark_runtime::moonshine::MoonshineRunner::open(&model_dir)?;
    let open_ms = opened.elapsed().as_secs_f64() * 1_000.0;
    if mode.as_deref() == Some("--open-only") {
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
    let warmup = if mode.as_deref() == Some("--warmup") {
        let started = Instant::now();
        let text = model.transcribe(&samples)?;
        Some((text, started.elapsed().as_secs_f64() * 1_000.0))
    } else {
        None
    };
    let started = Instant::now();
    let text = model.transcribe(&samples)?;
    let transcribe_ms = started.elapsed().as_secs_f64() * 1_000.0;
    if let Some((warmup_text, _)) = &warmup {
        if warmup_text != &text {
            return Err("Moonshine warmup and measured transcripts differ".into());
        }
    }
    println!("TEXT: {text}");
    eprintln!(
        "moonshine_{} open_ms={open_ms:.2} warmup_ms={:.2} transcribe_ms={:.2} samples={}",
        if model.using_metal() { "metal" } else { "cpu" },
        warmup.as_ref().map_or(0.0, |(_, elapsed)| *elapsed),
        transcribe_ms,
        samples.len()
    );
    Ok(())
}
