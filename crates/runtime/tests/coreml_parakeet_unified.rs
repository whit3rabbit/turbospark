//! Real-model gates for live transcription with Parakeet Unified EN on Core
//! ML. Ignored by default.
//!
//! Model directories:
//! - `TURBOSPARK_PARAKEET_UNIFIED_MODEL` (default
//!   `~/models/coreml/parakeet-unified-en-coreml`): FluidInference
//!   `parakeet-unified-en-0.6b-coreml` at
//!   `d32e972dd4315f1dc3f6be28fb2aab0ab3e80358`, the pinned subset (config,
//!   vocab, decoder, joint decision, the 70_7_7 int8 streaming encoder).
//! - `TURBOSPARK_PARAKEET_COREML_MODEL` (default
//!   `~/models/coreml/parakeet-ultra-coreml`): the batch model the live text
//!   is compared with, as the app's final pass would run it.
//!
//! Clips are synthesized with macOS `say` at 16 kHz. A test skips (passes
//! with a note) when a model or `say` is missing.
//!
//!     cargo test --release -p turbospark-runtime --test coreml_parakeet_unified -- --ignored --nocapture

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use audio::wav::read_wav_f32;
use turbospark_runtime::coreml::parakeet_unified_stream::StreamingModel;
use turbospark_runtime::coreml::{
    ParakeetCoreMl, ParakeetCoreMlOptions, ParakeetUnifiedCoreMl, UnifiedOptions, UnifiedStream,
};

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

fn scratch() -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "ts-parakeet-unified-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// 16 kHz mono PCM of `text` spoken by the default system voice.
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

#[path = "common/wer.rs"]
mod wer;
use wer::wer;

const TEXT: &str = "The history of the lighthouse begins with fires lit on hilltops to guide \
    sailors home. Later, towers were built along dangerous coasts, and keepers tended oil \
    lamps through the night. Today most lighthouses are automated, but many are preserved as \
    museums, and visitors can climb the spiral stairs to look out over the water.";

/// Streams `pcm` in `step`-sample appends, returning the transcript, the
/// per-append compute times, and how far behind the audio each closed word
/// was committed (received audio minus the word's end).
fn stream(model: &ParakeetUnifiedCoreMl, pcm: &[f32], step: usize) -> (String, Vec<f64>, Vec<f64>) {
    let mut s = UnifiedStream::new(model.geometry());
    let (mut text, mut step_ms, mut lag) = (String::new(), Vec::new(), Vec::new());
    for part in pcm.chunks(step) {
        let t = Instant::now();
        let u = s.append(model, part, &|| false).unwrap();
        step_ms.push(t.elapsed().as_secs_f64() * 1e3);
        text.push_str(&u.appended_text);
        lag.extend(
            u.closed_words
                .iter()
                .map(|w| u.received_seconds - w.end_seconds),
        );
    }
    let u = s.finish(model, &|| false).unwrap();
    text.push_str(&u.appended_text);
    (text, step_ms, lag)
}

fn percentile(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

#[test]
#[ignore = "needs the Core ML Parakeet Unified checkpoint"]
fn live_text_is_accurate_and_independent_of_append_size() {
    let Some(dir) = model_dir(
        "TURBOSPARK_PARAKEET_UNIFIED_MODEL",
        "models/coreml/parakeet-unified-en-coreml",
    ) else {
        return;
    };
    let opened = Instant::now();
    let model = ParakeetUnifiedCoreMl::open(&dir, &UnifiedOptions::from_env().unwrap()).unwrap();
    eprintln!(
        "opened {model:?} in {:.1}s (first Neural Engine load compiles; later ones are cached)",
        opened.elapsed().as_secs_f64()
    );
    let scratch = scratch();
    let Some(pcm) = say(&scratch, "lighthouse", TEXT) else {
        return;
    };
    let seconds = pcm.len() as f64 / 16_000.0;

    // 100 ms appends, as the app's live feed sends them.
    let started = Instant::now();
    let (live, mut step_ms, mut lag) = stream(&model, &pcm, 1_600);
    let wall = started.elapsed().as_secs_f64();
    let w = wer(TEXT, &live);
    eprintln!(
        "{seconds:.1}s streamed in {wall:.2}s ({:.0}x), wer {w:.3}\n{live}",
        seconds / wall
    );
    eprintln!(
        "append compute ms: median {:.1} p95 {:.1} max {:.1}",
        percentile(&mut step_ms, 0.5),
        percentile(&mut step_ms, 0.95),
        percentile(&mut step_ms, 1.0)
    );
    eprintln!(
        "word commit lag s: median {:.2} p95 {:.2} (geometry latency {:.2})",
        percentile(&mut lag, 0.5),
        percentile(&mut lag, 0.95),
        model.geometry().latency_seconds()
    );
    assert!(w <= 0.05, "live transcript wer {w:.3}");
    assert!(
        live.chars().any(|c| c.is_ascii_punctuation()),
        "Unified adds punctuation: {live}"
    );

    // The windows depend only on the audio, so the transcript must not
    // change with how the audio arrives.
    for step in [16_000, 32_000] {
        let (other, _, _) = stream(&model, &pcm, step);
        assert_eq!(other, live, "append size {step} changed the transcript");
    }

    // The app's final pass runs the batch model on the same audio; live text
    // should already be close to it.
    if let Some(batch_dir) = model_dir(
        "TURBOSPARK_PARAKEET_COREML_MODEL",
        "models/coreml/parakeet-ultra-coreml",
    ) {
        let opts = ParakeetCoreMlOptions::for_model(&batch_dir).unwrap();
        let batch = ParakeetCoreMl::open(&batch_dir, &opts).unwrap();
        let final_text = batch.transcribe(&pcm, 16_000).unwrap();
        let between = wer(&final_text, &live);
        eprintln!(
            "batch ({opts:?}) wer {:.3}; live vs batch {between:.3}\n{final_text}",
            wer(TEXT, &final_text)
        );
        assert!(between <= 0.05, "live and batch disagree by {between:.3}");
    }
    std::fs::remove_dir_all(&scratch).unwrap();
}
