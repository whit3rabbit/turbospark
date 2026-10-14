//! Transcribes a WAV with the Core ML Parakeet export.
//!
//! Usage: `cargo run --release -p turbospark-runtime --example
//! parakeet_coreml_transcribe -- <model_dir> <wav>`
//!
//! The model directory is a FluidInference `parakeet-redux-coreml`
//! checkout. Any WAV the audio crate reads is resampled to 16 kHz mono.
//! Encoder placement follows `TURBOSPARK_PARAKEET_COREML_ENCODER_UNITS`
//! (default GPU). macOS-only.

#![cfg(target_os = "macos")]

use std::path::Path;

use audio::conversion::{to_mono_resampled, MonoResampleStrategy};
use audio::wav::read_wav_f32_bytes;
use turbospark_runtime::coreml::{ParakeetCoreMl, ParakeetCoreMlOptions};

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let model_dir = args
        .next()
        .ok_or("usage: parakeet_coreml_transcribe <model_dir> <wav>")?;
    let wav_path = args.next().ok_or("missing <wav>")?;
    let opts = ParakeetCoreMlOptions::from_env()?;

    let started = std::time::Instant::now();
    let model = ParakeetCoreMl::open(Path::new(&model_dir), &opts)?;
    eprintln!("opened {model:?} with {opts:?} in {:?}", started.elapsed());

    let bytes = std::fs::read(&wav_path).map_err(|e| format!("read {wav_path}: {e}"))?;
    let wave = read_wav_f32_bytes(&bytes).map_err(|e| format!("decode wav: {e}"))?;
    let mono = to_mono_resampled(
        &wave,
        model.sample_rate(),
        &MonoResampleStrategy::SincHann(Default::default()),
    )
    .map_err(|e| format!("resample: {e}"))?;
    let seconds = mono.len() as f64 / f64::from(model.sample_rate());

    // The first prediction per program includes Core ML's one-time
    // specialization, so report it separately from a warm run.
    let mut result = None;
    for label in ["cold", "warm"] {
        let started = std::time::Instant::now();
        result = Some(model.decode(&mono, model.sample_rate(), &|| false, |_, _| true)?);
        let elapsed = started.elapsed();
        eprintln!(
            "{label}: {seconds:.2}s of audio in {elapsed:?} (RTFx {:.1})",
            seconds / elapsed.as_secs_f64()
        );
    }
    let result = result.expect("ran at least once");
    for s in &result.sentences {
        println!("[{:8.2} - {:8.2}] {}", s.start, s.end, s.text.trim());
    }
    println!("text: {}", result.text);
    Ok(())
}
