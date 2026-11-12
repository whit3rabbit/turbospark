//! Real-model gate for offline speaker diarization on Core ML. Ignored by
//! default.
//!
//! Model directory: `TURBOSPARK_DIARIZATION_COREML_MODEL` (default
//! `~/models/coreml/speaker-diarization-coreml`), FluidInference
//! `speaker-diarization-coreml` at `7eab74f3463c736418d0eb10296261ae581146c9`,
//! the pinned community-1 subset (Segmentation, FBank, Embedding, PldaRho,
//! `plda-parameters.json`).
//!
//! Clips are synthesized with macOS `say` at 16 kHz in two voices, trimmed
//! of their own silence and joined with known gaps, so the reference turns
//! are exact sample ranges. A test skips (passes with a note) when the model
//! or a voice is missing.
//!
//!     cargo test --release -p turbospark-runtime --test coreml_diarization -- --ignored --nocapture

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use audio::wav::read_wav_f32;
use turbospark_runtime::coreml::{DiarizationCoreMl, DiarizationOptions, SpeakerTurn};

const RATE: usize = 16_000;

fn model_dir() -> Option<PathBuf> {
    let env = "TURBOSPARK_DIARIZATION_COREML_MODEL";
    let p = std::env::var_os(env).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("models/coreml/speaker-diarization-coreml")
    });
    if p.is_dir() {
        Some(p)
    } else {
        eprintln!("skipping: {} not found (set {env})", p.display());
        None
    }
}

fn scratch() -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "ts-diarization-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// 16 kHz mono PCM of `text` in `voice`, without its leading and trailing
/// silence.
fn say(dir: &Path, name: &str, voice: &str, text: &str) -> Option<Vec<f32>> {
    // Concurrent `say -o` runs fail with error -54, which would skip a test.
    static SAY: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _one = SAY.lock().unwrap_or_else(|p| p.into_inner());
    let path = dir.join(format!("{name}.wav"));
    let ok = Command::new("say")
        .args(["-v", voice, "-o"])
        .arg(&path)
        .args(["--data-format=LEI16@16000", text])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("skipping: `say -v {voice}` failed");
        return None;
    }
    let w = read_wav_f32(&path).unwrap();
    assert_eq!((w.sample_rate, w.channels), (16_000, 1));
    let loud = |s: &f32| s.abs() > 1e-3;
    let start = w.samples.iter().position(loud)?;
    let end = w.samples.iter().rposition(loud)? + 1;
    Some(w.samples[start..end].to_vec())
}

/// Reference turn: speaker index and sample range.
type RefTurn = (usize, usize, usize);

/// Joins `(voice index, text)` turns with `gap` seconds of silence.
fn conversation(
    dir: &Path,
    voices: &[&str],
    script: &[(usize, &str)],
    gap: f64,
) -> Option<(Vec<f32>, Vec<RefTurn>)> {
    let silence = vec![0.0f32; (gap * RATE as f64) as usize];
    let mut pcm = silence.clone();
    let mut turns = Vec::new();
    for (i, (speaker, text)) in script.iter().enumerate() {
        let clip = say(dir, &format!("turn{i}"), voices[*speaker], text)?;
        turns.push((*speaker, pcm.len(), pcm.len() + clip.len()));
        pcm.extend_from_slice(&clip);
        pcm.extend_from_slice(&silence);
    }
    Some((pcm, turns))
}

/// Frame-level diarization error rate at 10 ms, under the best one-to-one
/// mapping of hypothesis to reference speakers. Frames within `collar`
/// seconds of a reference boundary are not scored (NIST md-eval style).
/// Returns (DER, missed, false alarm, confusion) as fractions of scored
/// reference speech.
fn der(
    reference: &[RefTurn],
    hyp: &[SpeakerTurn],
    len: usize,
    collar: f64,
) -> (f64, f64, f64, f64) {
    let step = RATE / 100;
    let frames = len / step;
    let mut r = vec![None; frames];
    let mut scored = vec![true; frames];
    let c = (collar * 100.0).round() as isize;
    for &(s, a, b) in reference {
        for slot in r.iter_mut().take(b / step).skip(a / step) {
            *slot = Some(s);
        }
        for edge in [a / step, b / step] {
            for f in (edge as isize - c).max(0)..(edge as isize + c).min(frames as isize) {
                scored[f as usize] = false;
            }
        }
    }
    let mut h = vec![None; frames];
    for t in hyp {
        let a = (t.start * 100.0).round() as usize;
        let b = ((t.end * 100.0).round() as usize).min(frames);
        for slot in h.iter_mut().take(b).skip(a) {
            *slot = Some(t.speaker);
        }
    }
    let ref_speakers = reference.iter().map(|t| t.0).max().map_or(0, |m| m + 1);
    let hyp_speakers = hyp.iter().map(|t| t.speaker).max().map_or(0, |m| m + 1);
    let mut best = (f64::INFINITY, 0.0, 0.0, 0.0);
    for mapping in permutations(hyp_speakers, ref_speakers.max(hyp_speakers)) {
        let (mut speech, mut miss, mut fa, mut conf) = (0.0, 0.0, 0.0, 0.0);
        for f in 0..frames {
            if !scored[f] {
                continue;
            }
            match (r[f], h[f].map(|s| mapping[s])) {
                (Some(_), None) => {
                    speech += 1.0;
                    miss += 1.0;
                }
                (None, Some(_)) => fa += 1.0,
                (Some(a), Some(b)) => {
                    speech += 1.0;
                    if a != b {
                        conf += 1.0;
                    }
                }
                (None, None) => {}
            }
        }
        let e = (miss + fa + conf) / speech;
        if e < best.0 {
            best = (e, miss / speech, fa / speech, conf / speech);
        }
    }
    best
}

/// Every injective map of `n` hypothesis speakers onto `k >= n` labels.
fn permutations(n: usize, k: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    for rest in permutations(n - 1, k) {
        for v in 0..k {
            if !rest.contains(&v) {
                let mut m = rest.clone();
                m.push(v);
                out.push(m);
            }
        }
    }
    out
}

/// The hypothesis speaker covering most of each reference turn.
fn turn_speakers(reference: &[RefTurn], hyp: &[SpeakerTurn]) -> Vec<Option<usize>> {
    reference
        .iter()
        .map(|&(_, a, b)| {
            let (a, b) = (a as f64 / RATE as f64, b as f64 / RATE as f64);
            let mut cover = std::collections::HashMap::new();
            for t in hyp {
                let o = t.end.min(b) - t.start.max(a);
                if o > 0.0 {
                    *cover.entry(t.speaker).or_insert(0.0) += o;
                }
            }
            cover
                .into_iter()
                .max_by(|x, y| x.1.total_cmp(&y.1))
                .map(|(s, _)| s)
        })
        .collect()
}

const SCRIPT: [(usize, &str); 10] = [
    (0, "Good morning everyone. Thanks for joining the planning call. Today we need to settle the release date and the testing schedule."),
    (1, "Morning. Before we pick a date, I want to flag that the database migration is still running slower than we expected on the staging cluster."),
    (0, "How much slower are we talking about? Last week you said it would finish in under an hour."),
    (1, "Closer to three hours right now. We found an index that gets rebuilt twice, and fixing that should cut the time roughly in half."),
    (0, "Okay. Can you have the fix ready by Thursday so we can run it again before the weekend?"),
    (1, "Thursday works if nothing else breaks. I will send the numbers to the channel as soon as the second run is done."),
    (0, "Great. Then let us tentatively plan the release for the following Tuesday, and we will confirm on Friday."),
    (1, "That sounds reasonable. One more thing, the support team asked for release notes at least two days ahead of time."),
    (0, "Good point. I will draft the notes on Monday and share them with support and with the documentation folks."),
    (1, "Perfect. That covers everything on my side. Talk to you on Friday."),
];

#[test]
#[ignore = "needs the Core ML diarization model and macOS say voices"]
fn two_voice_meeting_is_split_into_the_right_turns() {
    let Some(dir) = model_dir() else { return };
    let tmp = scratch();
    let Some((pcm, reference)) = conversation(&tmp, &["Samantha", "Daniel"], &SCRIPT, 0.5) else {
        return;
    };
    let seconds = pcm.len() as f64 / RATE as f64;
    let t = Instant::now();
    let opts = DiarizationOptions::from_env().unwrap();
    let model = DiarizationCoreMl::open(&dir, &opts).unwrap();
    let open = t.elapsed();
    let t = Instant::now();
    let d = model.diarize(&pcm, 16_000, &|| false, |_, _| true).unwrap();
    let run = t.elapsed();
    for turn in &d.turns {
        eprintln!(
            "  S{} {:7.2} {:7.2}",
            turn.speaker + 1,
            turn.start,
            turn.end
        );
    }
    let (e0, miss0, fa0, conf0) = der(&reference, &d.turns, pcm.len(), 0.0);
    let (e25, miss25, fa25, conf25) = der(&reference, &d.turns, pcm.len(), 0.25);
    let by_turn = turn_speakers(&reference, &d.turns);
    eprintln!(
        "clip {seconds:.1} s, {} reference turns; open {open:.2?}, diarize {run:.2?} ({:.0}x real time)",
        reference.len(),
        seconds / run.as_secs_f64()
    );
    eprintln!(
        "speakers {}; turn majority speakers {by_turn:?}",
        d.speakers
    );
    eprintln!(
        "DER collar 0: {e0:.4} (miss {miss0:.4}, false alarm {fa0:.4}, confusion {conf0:.4})"
    );
    eprintln!("DER collar 0.25 s: {e25:.4} (miss {miss25:.4}, false alarm {fa25:.4}, confusion {conf25:.4})");
    assert_eq!(d.speakers, 2, "two voices");
    // Every turn goes to its own voice: the first speaker heard is S1.
    let expected: Vec<Option<usize>> = reference.iter().map(|t| Some(t.0)).collect();
    assert_eq!(by_turn, expected, "turn speakers");
    assert!(e25 <= 0.05, "DER {e25:.4} with a 0.25 s collar");
    assert!(
        d.turns.windows(2).all(|w| w[0].end <= w[1].start + 1e-9),
        "turns overlap"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
#[ignore = "needs the Core ML diarization model and macOS say voices"]
fn one_voice_monologue_stays_one_speaker() {
    let Some(dir) = model_dir() else { return };
    let tmp = scratch();
    let script: Vec<(usize, &str)> = SCRIPT.iter().map(|&(_, text)| (0, text)).take(6).collect();
    let Some((pcm, reference)) = conversation(&tmp, &["Samantha"], &script, 0.5) else {
        return;
    };
    let model = DiarizationCoreMl::open(&dir, &DiarizationOptions::from_env().unwrap()).unwrap();
    let d = model.diarize(&pcm, 16_000, &|| false, |_, _| true).unwrap();
    let (e25, ..) = der(&reference, &d.turns, pcm.len(), 0.25);
    eprintln!(
        "monologue {:.1} s: speakers {}, DER collar 0.25 s {e25:.4}",
        pcm.len() as f64 / RATE as f64,
        d.speakers
    );
    assert_eq!(d.speakers, 1, "{:?}", d.turns);
    assert!(e25 <= 0.05, "DER {e25:.4}");
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
#[ignore = "needs the Core ML diarization model"]
fn cancellation_stops_before_the_first_window() {
    let Some(dir) = model_dir() else { return };
    let model = DiarizationCoreMl::open(&dir, &DiarizationOptions::default()).unwrap();
    let pcm = vec![0.01f32; 16_000 * 40];
    let e = model
        .diarize(&pcm, 16_000, &|| true, |_, _| true)
        .unwrap_err();
    assert!(e.contains("cancelled"), "{e}");
    let e = model
        .diarize(&pcm, 8_000, &|| false, |_, _| true)
        .unwrap_err();
    assert!(e.contains("16000 Hz"), "{e}");
}
