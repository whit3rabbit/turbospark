//! Real-model transcription gate for the whisper CPU reference path.
//!
//! Usage: `cargo run -p turbospark-runtime --example whisper_transcribe --
//! <model_dir> <wav> [language]`
//!
//! The model directory holds config.json, model.safetensors, and
//! tokenizer.json (openai/whisper-tiny.en or compatible); the wav is any
//! sample-rate PCM file the audio crate can read, resampled to 16 kHz mono
//! on the way in. macOS-only, like the whisper module it exercises; on
//! other targets this file compiles empty.

#![cfg(target_os = "macos")]

use std::path::Path;

use audio::conversion::{to_mono_resampled, MonoResampleStrategy};
use audio::wav::read_wav_f32_bytes;
use turbospark_runtime::WhisperRunner;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let model_dir = args
        .next()
        .ok_or("usage: whisper_transcribe <model_dir> <wav> [language]")?;
    let wav_path = args.next().ok_or("missing <wav>")?;
    let language = args.next();

    let started = std::time::Instant::now();
    let runner = WhisperRunner::open(Path::new(&model_dir))?;
    eprintln!(
        "opened {} (d_model {}, {} enc layers, {} mel bands) in {:?}; device: {}",
        model_dir,
        runner.config.d_model,
        runner.config.encoder_layers(),
        runner.config.n_mels,
        started.elapsed(),
        if runner.using_metal() {
            "metal"
        } else {
            "cpu-reference"
        }
    );

    let bytes = std::fs::read(&wav_path).map_err(|e| format!("read {wav_path}: {e}"))?;
    let wave = read_wav_f32_bytes(&bytes).map_err(|e| format!("decode wav: {e}"))?;
    let mono = to_mono_resampled(
        &wave,
        audio::whisper::WHISPER_SAMPLE_RATE,
        &MonoResampleStrategy::SincHann(Default::default()),
    )
    .map_err(|e| format!("resample: {e}"))?;
    eprintln!(
        "audio: {} ch @ {} Hz, {:.2}s -> 16 kHz mono",
        wave.channels,
        wave.sample_rate,
        wave.duration_seconds()
    );

    let started = std::time::Instant::now();
    let transcription = runner.transcribe(&mono, language.as_deref())?;
    eprintln!("transcribed in {:?}", started.elapsed());
    for segment in &transcription.segments {
        println!(
            "[{:8.2} - {:8.2}] {}",
            segment.start_seconds, segment.end_seconds, segment.text
        );
    }
    println!("language: {}", transcription.language);
    Ok(())
}
