//! Real-model gate for Fluid Speech Pro on Core ML. Ignored by default.
//!
//! Model directories:
//! - `TURBOSPARK_FLUID_SPEECH_COREML_MODEL` (default
//!   `~/models/fluid-speech-pro-coreml`): `FluidVoice-Labs/fluid-speech-pro-coreml`
//!   at `3b9d38a93d606f84011e5a0a669b7ea146375a22`.
//! - `TURBOSPEECH_FLUID_SPEECH_MODEL` (default `~/models/fluid-speech-pro-mlx`):
//!   the CPU oracle, `audio::stt::parakeet::PARAKEET_FLUID_SPEECH_PRO`, whose
//!   stage outputs match the publisher's MLX loader (`fluid_parity` tests).
//!
//! The two exports quantize differently (Core ML: int8 per output channel
//! with fp16 activations; MLX: 8-bit groups of 64 dequantized to f32 here),
//! so the gate compares transcripts, not logits. Clips are synthesized with
//! macOS `say` at 16 kHz. A test skips (passes with a note) when a model or
//! `say` is missing.
//!
//!     cargo test --release -p turbospark-runtime --test coreml_fluid_speech -- --ignored --nocapture

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::process::Command;

use audio::stt::parakeet::ParakeetTdt;
use audio::wav::read_wav_f32;
use turbospark_runtime::coreml::{ParakeetCoreMl, ParakeetCoreMlOptions};

fn model_dir(env: &str, default: &str) -> Option<PathBuf> {
    let p = std::env::var_os(env)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(default));
    if p.is_dir() {
        Some(p)
    } else {
        eprintln!("skipping: {} not found (set {env})", p.display());
        None
    }
}

fn say(dir: &Path, name: &str, text: &str) -> Option<Vec<f32>> {
    let path = dir.join(format!("{name}.wav"));
    let ok = Command::new("say")
        .args(["-o"])
        .arg(&path)
        .args(["--data-format=LEI16@16000", text])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("skipping: `say` failed");
        return None;
    }
    let w = read_wav_f32(&path).unwrap();
    assert_eq!((w.sample_rate, w.channels), (16_000, 1));
    Some(w.samples)
}

const CLIPS: [&str; 4] = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "On Tuesday, the committee approved a budget of four million dollars for the new library.",
    "Um, so, uh, I I think we we should, you know, go to the the store tomorrow.",
];

#[test]
#[ignore = "needs the Fluid Speech Pro Core ML and MLX checkpoints"]
fn coreml_transcripts_match_cpu_fluid() {
    let Some(coreml_dir) = model_dir(
        "TURBOSPARK_FLUID_SPEECH_COREML_MODEL",
        "models/fluid-speech-pro-coreml",
    ) else {
        return;
    };
    let Some(cpu_dir) = model_dir(
        "TURBOSPEECH_FLUID_SPEECH_MODEL",
        "models/fluid-speech-pro-mlx",
    ) else {
        return;
    };
    assert_eq!(
        turbospark_runtime::coreml::parakeet::ParakeetExport::detect(&coreml_dir),
        Some(turbospark_runtime::coreml::parakeet::ParakeetExport::FluidSpeechPro)
    );
    let opts = ParakeetCoreMlOptions::for_model(&coreml_dir).unwrap();
    // An undeclared-on-GPU int8 encoder must default off the GPU.
    assert_eq!(
        opts.encoder_units,
        turbospark_coreml::ComputeUnits::CpuAndNeuralEngine
    );
    let coreml = ParakeetCoreMl::open(&coreml_dir, &opts).unwrap();
    let cpu = ParakeetTdt::open(&cpu_dir).unwrap();
    let dir = std::env::temp_dir().join(format!("ts-fluid-coreml-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut mismatches = 0;
    for (i, text) in CLIPS.iter().enumerate() {
        let Some(pcm) = say(&dir, &format!("clip{i}"), text) else {
            return;
        };
        let started = std::time::Instant::now();
        let got = coreml.decode(&pcm, 16_000, &|| false, |_, _| true).unwrap();
        let elapsed = started.elapsed();
        let want = cpu.decode(&pcm, 16_000).unwrap();
        println!(
            "clip {i}: coreml {:?} ({:.0} ms) | cpu {:?}",
            got.text,
            elapsed.as_secs_f64() * 1e3,
            want.text
        );
        if got.text != want.text {
            mismatches += 1;
        }
        // Timestamps stay ordered on the 80 ms grid.
        let starts: Vec<f32> = got
            .sentences
            .iter()
            .flat_map(|s| s.tokens.iter().map(|t| t.start))
            .collect();
        assert!(starts.windows(2).all(|w| w[0] <= w[1]), "clip {i}");
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(mismatches, 0, "Core ML and CPU Fluid transcripts differ");
}
