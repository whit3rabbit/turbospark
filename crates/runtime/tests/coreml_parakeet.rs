//! Real-model gates for the Core ML Parakeet pipeline. Ignored by default.
//!
//! Model directories:
//! - `TURBOSPARK_PARAKEET_COREML_MODEL` (default
//!   `~/models/coreml/parakeet-redux-coreml`): FluidInference
//!   `parakeet-redux-coreml` at `8c5ef97a29cd120dc76b354b3f22b7fec3b486f9`.
//! - `TURBOSPARK_PARAKEET_REDUX_MODEL` (default `~/models/parakeet-redux`):
//!   the CPU oracle, `moondream/parakeet-redux` at the revision pinned in
//!   `audio::stt::parakeet::PARAKEET_REDUX`.
//!
//! The CPU parity checks run only on a Redux export (`base_model` in its
//! `config.json`). Another export, such as `parakeet-ultra-coreml` at
//! `95eaa59a39d4394f047a4dc5cce480388a60d1b6`, is held to the long-clip WER
//! bound and timestamp order only.
//!
//! Clips are synthesized with macOS `say` at 16 kHz, so no audio fixture is
//! committed. A test skips (passes with a note) when a model or `say` is
//! missing.
//!
//!     cargo test --release -p turbospark-runtime --test coreml_parakeet -- --ignored --nocapture

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::process::Command;

use audio::stt::nemo::AlignedResult;
use audio::stt::parakeet::ParakeetTdt;
use audio::wav::read_wav_f32;
use turbospark_runtime::coreml::{ParakeetCoreMl, ParakeetCoreMlOptions};

fn base_model(dir: &Path) -> Option<String> {
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).ok()?).ok()?;
    config.get("base_model")?.as_str().map(str::to_owned)
}

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
        "ts-parakeet-coreml-{}-{}",
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

fn token_table(r: &AlignedResult) -> Vec<(i32, f32)> {
    r.sentences
        .iter()
        .flat_map(|s| s.tokens.iter().map(|t| (t.id, t.start)))
        .collect()
}

const SHORT: [&str; 3] = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "On Tuesday, the committee approved a budget of four million dollars for the new library.",
];

#[test]
#[ignore = "needs the Core ML and CPU Parakeet Redux checkpoints"]
fn short_clips_match_cpu_redux() {
    let Some(coreml_dir) = model_dir(
        "TURBOSPARK_PARAKEET_COREML_MODEL",
        "models/coreml/parakeet-redux-coreml",
    ) else {
        return;
    };
    // The CPU oracle is Redux; comparing another export against it would
    // fail on correct output.
    let cpu_dir = (base_model(&coreml_dir).as_deref() == Some("moondream/parakeet-redux"))
        .then(|| model_dir("TURBOSPARK_PARAKEET_REDUX_MODEL", "models/parakeet-redux"))
        .flatten();
    let opts = ParakeetCoreMlOptions::for_model(&coreml_dir).unwrap();
    let coreml = ParakeetCoreMl::open(&coreml_dir, &opts).unwrap();
    let cpu = cpu_dir.map(|d| ParakeetTdt::open(&d).unwrap());
    let dir = scratch();
    for (i, text) in SHORT.iter().enumerate() {
        let Some(pcm) = say(&dir, &format!("short{i}"), text) else {
            return;
        };
        assert!(pcm.len() <= 240_000, "clip {i} exceeds one window");
        let got = coreml.decode(&pcm, 16_000, &|| false, |_, _| true).unwrap();
        let w = wer(text, &got.text);
        eprintln!("coreml [{i}] wer {w:.3}: {}", got.text);
        if let Some(cpu) = &cpu {
            let want = cpu.decode(&pcm, 16_000).unwrap();
            let (a, b) = (token_table(&got), token_table(&want));
            let same_ids = a.iter().map(|t| t.0).eq(b.iter().map(|t| t.0));
            let max_dt = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x.1 - y.1).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "cpu    [{i}] {}\n       tokens coreml {} cpu {} identical ids {same_ids} max start delta {max_dt:.3}s",
                want.text,
                a.len(),
                b.len()
            );
            assert_eq!(
                got.text, want.text,
                "clip {i}: Core ML and CPU Redux disagree"
            );
            assert!(same_ids, "clip {i}: token ids differ");
            assert_eq!(max_dt, 0.0, "clip {i}: token frames differ");
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[ignore = "needs the Core ML Parakeet Redux checkpoint"]
fn long_clip_merges_windows_without_duplicates() {
    let Some(coreml_dir) = model_dir(
        "TURBOSPARK_PARAKEET_COREML_MODEL",
        "models/coreml/parakeet-redux-coreml",
    ) else {
        return;
    };
    let opts = ParakeetCoreMlOptions::for_model(&coreml_dir).unwrap();
    let coreml = ParakeetCoreMl::open(&coreml_dir, &opts).unwrap();
    let text = "The history of the lighthouse begins with fires lit on hilltops to guide \
        sailors home. Later, towers were built along dangerous coasts, and keepers \
        tended oil lamps through the night. In the nineteenth century, a French \
        engineer designed a lens that could focus the light into a narrow beam, \
        visible for many miles. Today most lighthouses are automated, but many are \
        preserved as museums, and visitors can climb the spiral stairs to look out \
        over the water. Some keepers kept detailed logs of storms, shipwrecks, and \
        rescues, and those records now help historians understand the weather of \
        past centuries.";
    let dir = scratch();
    let Some(pcm) = say(&dir, "long", text) else {
        return;
    };
    let seconds = pcm.len() as f64 / 16_000.0;
    assert!(seconds > 30.0, "clip is only {seconds:.1}s");
    let mut windows = 0;
    let got = coreml
        .decode(&pcm, 16_000, &|| false, |_, total| {
            windows = total;
            true
        })
        .unwrap();
    let w = wer(text, &got.text);
    eprintln!(
        "long clip {seconds:.1}s in {windows} windows, wer {w:.3}\n{}",
        got.text
    );
    assert!(windows >= 3);
    // Measured 0.010 here (only "19th" vs "nineteenth"). The seam bug this
    // guards against (tokens past the overlap dropped) measured 0.232.
    assert!(w < 0.05, "windowed transcript lost words at a seam");
    // Timestamps must stay monotonic across window seams.
    let starts: Vec<f32> = got
        .sentences
        .iter()
        .flat_map(|s| s.tokens.iter().map(|t| t.start))
        .collect();
    assert!(
        starts.windows(2).all(|p| p[0] <= p[1]),
        "timestamps go backwards"
    );
    let cpu_dir = (base_model(&coreml_dir).as_deref() == Some("moondream/parakeet-redux"))
        .then(|| model_dir("TURBOSPARK_PARAKEET_REDUX_MODEL", "models/parakeet-redux"))
        .flatten();
    if let Some(cpu_dir) = cpu_dir {
        let cpu = ParakeetTdt::open(&cpu_dir).unwrap();
        let want = cpu.decode(&pcm, 16_000).unwrap();
        let vs_cpu = wer(&want.text, &got.text);
        eprintln!(
            "cpu (single pass) wer {:.3}; coreml vs cpu wer {vs_cpu:.3}",
            wer(text, &want.text)
        );
        // Measured 0.000: windowing plus merging reproduces the CPU
        // decoder's single pass over the whole clip.
        assert_eq!(
            vs_cpu, 0.0,
            "windowed Core ML differs from the CPU single pass"
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
