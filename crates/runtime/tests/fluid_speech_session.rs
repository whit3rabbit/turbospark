//! Real-model gate for Fluid Speech Pro on the native audio session (CPU).
//! Ignored by default.
//!
//! - `TURBOSPEECH_FLUID_SPEECH_MODEL` (default `~/models/fluid-speech-pro-mlx`):
//!   `FluidVoice-Labs/fluid-speech-pro-mlx` at
//!   `aeb1d4e43487586b15d86c0a1dd4511e1cf6b7a0` (8-bit file is enough).
//!
//! Checks the session surface the app and server use: the portable opt-in,
//! the family string, both transcript styles on one session (the second
//! reopens the model), windowed decoding of a clip longer than one 15 s
//! window, and the refusal of an unknown style. Clips come from macOS `say`.
//!
//!     cargo test --release -p turbospark-runtime --test fluid_speech_session -- --ignored --nocapture

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::process::Command;

use audio::wav::read_wav_f32;
use turbospark_runtime::native_audio::{AudioCancel, AudioRequest, AudioSession, AudioTask};

fn model_dir() -> Option<PathBuf> {
    let p = std::env::var_os("TURBOSPEECH_FLUID_SPEECH_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default())
                .join("models/fluid-speech-pro-mlx")
        });
    if p.join("config.json").is_file() {
        Some(p)
    } else {
        eprintln!("skipping: {} not found", p.display());
        None
    }
}

fn say(dir: &Path, text: &str) -> Option<Vec<f32>> {
    let path = dir.join("clip.wav");
    let ok = Command::new("say")
        .arg("-o")
        .arg(&path)
        .args(["--data-format=LEI16@16000", text])
        .status()
        .is_ok_and(|s| s.success());
    if !ok {
        eprintln!("skipping: `say` failed");
        return None;
    }
    Some(read_wav_f32(&path).unwrap().samples)
}

fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

const SCRIPT: &str = "Um, so, uh, I I think we we should, you know, go to the the store \
    tomorrow. The committee met on Tuesday and approved the budget for the new library. \
    After that, uh, we we talked about the schedule for the spring, and, um, everyone \
    agreed that the the opening should happen in May. Finally, the chair thanked the \
    volunteers who had organized the book sale.";

#[test]
#[ignore = "needs the Fluid Speech Pro MLX checkpoint"]
fn session_runs_both_styles_over_several_windows() {
    let Some(dir) = model_dir() else { return };
    assert!(
        AudioSession::open(dir.clone(), AudioTask::SpeechToText, false).is_err(),
        "the CPU path needs the portable opt-in"
    );
    let session = AudioSession::open(dir, AudioTask::SpeechToText, true).unwrap();
    assert_eq!(session.family(), "fluid_speech_pro");
    let scratch = std::env::temp_dir().join(format!("ts-fluid-session-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let Some(pcm) = say(&scratch, SCRIPT) else {
        return;
    };
    let _ = std::fs::remove_dir_all(&scratch);
    assert!(
        pcm.len() > 16_000 * 20,
        "clip must span more than one window"
    );

    let run = |style: Option<&str>| {
        let request = AudioRequest {
            transcript_style: style.map(str::to_owned),
            ..AudioRequest::default()
        };
        let mut windows = 0;
        let started = std::time::Instant::now();
        let result = session
            .execute(request, pcm.clone(), AudioCancel::default(), |e| {
                if matches!(e, turbospark_runtime::native_audio::AudioEvent::Progress(_)) {
                    windows += 1;
                }
            })
            .unwrap();
        let t = result.transcript.unwrap();
        println!(
            "{style:?}: {} windows, {:.1} s: {:?}",
            windows,
            started.elapsed().as_secs_f64(),
            t.text
        );
        assert!(windows >= 2, "expected several windows, saw {windows}");
        let starts: Vec<f64> = t.segments.iter().filter_map(|s| s.start_seconds).collect();
        assert!(starts.windows(2).all(|w| w[0] <= w[1]));
        t.text
    };
    let verbatim = run(None);
    let clean = run(Some("clean"));
    let again = run(Some("verbatim"));
    assert_eq!(verbatim, again, "switching back reopens the same model");
    // The overlap merge keeps every sentence: the closing words survive.
    for text in [&verbatim, &clean] {
        let w = words(text);
        assert!(w.ends_with(&["book".into(), "sale".into()]), "{text}");
        assert!(w.contains(&"library".to_string()), "{text}");
    }
    let fillers = |t: &str| {
        words(t)
            .iter()
            .filter(|w| ["um", "uh"].contains(&w.as_str()))
            .count()
    };
    assert!(fillers(&verbatim) > 0, "verbatim keeps fillers: {verbatim}");
    assert_eq!(fillers(&clean), 0, "clean drops fillers: {clean}");

    let bad = AudioRequest {
        transcript_style: Some("tidy".into()),
        ..AudioRequest::default()
    };
    let err = session
        .execute(bad, pcm, AudioCancel::default(), |_| {})
        .unwrap_err();
    assert!(err.contains("transcript_style"), "{err}");
}
