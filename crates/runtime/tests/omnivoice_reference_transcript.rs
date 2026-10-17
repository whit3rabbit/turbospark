//! OmniVoice reference-transcript A/B on real clones. Ignored by default.
//!
//! k2-fsa/OmniVoice conditions every clone on a reference transcript: when
//! the caller gives none it runs Whisper on the reference to get one. The
//! pinned mlx-audio port and this crate have no ASR fallback and pass None,
//! so the text block then covers only the target while the audio block
//! still holds the reference speech. This gate clones the same targets at
//! the same seeds four ways per reference: no transcript, the exact one,
//! the oracle's transcript of the reference (the app sends a take's ASR
//! transcript, and upstream its own Whisper pass), and a stale one (the
//! next reference's words: a transcript that no longer matches the audio).
//! The target length comes from the target text alone, so every variant
//! generates the same frame count.
//!
//! Each clip is transcribed with the CPU Parakeet Redux oracle and scored
//! against the target (`common/wer.rs` normalization). "Leading lost" is the
//! number of target words before the first word the transcript gets right,
//! the expected failure when the reference speech has no words in the text
//! block. "Leaked" counts heard words from the transcript sent (or the
//! reference's own when none was) that the target does not contain. The
//! gate asserts finite output only; the comparison is the measurement,
//! recorded in `crates/audio/src/tts/omnivoice/README.md`.
//!
//! - `TURBOSPARK_TEST_OMNIVOICE`: an OmniVoice install
//!   (`mlx-community/OmniVoice-8bit`).
//! - `TURBOSPARK_PARAKEET_REDUX_MODEL` (default `~/models/parakeet-redux`).
//! - `TURBOSPARK_TEST_LIBRISPEECH_WAV` (default
//!   `~/models/librispeech/LibriSpeech/wav16k`): 16 kHz mono WAVs of
//!   LibriSpeech test-clean, named `<utterance id>.wav`.
//! - `TURBOSPARK_TEST_OMNIVOICE_AB_OUT`, optional: a directory for the
//!   24 kHz clips.
//!
//!     cargo test --release -p turbospark-runtime --test omnivoice_reference_transcript -- --ignored --nocapture

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use audio::stt::parakeet::ParakeetTdt;
use audio::wav::{read_wav_f32, write_wav_f32};
use turbospark_runtime::omnivoice::{Device, OmniVoiceRequest, OmniVoiceRunner};

#[path = "common/wer.rs"]
mod wer;

/// LibriSpeech test-clean references (two female, two male, 5.7 to 8.1 s)
/// with their transcripts in sentence case, as a user or an ASR pass would
/// write them.
const REFERENCES: [(&str, &str); 4] = [
    (
        "1284-1180-0000",
        "He wore blue silk stockings, blue knee pants with gold buckles, a blue ruffled waist and a jacket of bright blue braided with gold.",
    ),
    (
        "2830-3979-0000",
        "We want you to help us publish some leading work of Luther's for the general American market. Will you do it?",
    ),
    (
        "4992-23283-0000",
        "But the more forgetfulness had then prevailed, the more powerful was the force of remembrance when she awoke.",
    ),
    (
        "7176-88083-0000",
        "All about him was a tumult of bright and broken color, scattered in broad splashes.",
    ),
];

/// Targets that share no content words with any reference, so a reference
/// word in a transcript is a leak and not a coincidence.
const TARGETS: [&str; 2] = [
    "Please remember to close the kitchen window before the evening rain begins.",
    "Our neighbors planted a small garden behind the old stone wall last spring.",
];

const SEEDS: [u64; 3] = [7, 11, 23];
const STEPS: usize = 32;
const VARIANTS: [&str; 4] = ["none", "exact", "asr", "stale"];

fn dir_from(env: &str, default: &str) -> Option<PathBuf> {
    let path = std::env::var_os(env)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(default));
    if path.is_dir() {
        Some(path)
    } else {
        eprintln!("skipping: {} not found (set {env})", path.display());
        None
    }
}

/// The oracle's words with its British spellings folded to the targets'
/// (Parakeet writes "neighbours" for a clean "neighbors").
fn folded(hyp: &str) -> String {
    wer::normalize(hyp)
        .into_iter()
        .map(|w| {
            if w == "neighbours" {
                "neighbors".into()
            } else {
                w
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Heard words that belong to `source` (the transcript sent, or the
/// reference's own when none was) and not to `target`: the model speaking
/// the reference text instead of the script.
fn leaked(source: &str, target: &str, hyp: &str) -> usize {
    let source = wer::normalize(source);
    let target = wer::normalize(target);
    wer::normalize(hyp)
        .iter()
        .filter(|w| source.contains(w) && !target.contains(w))
        .count()
}

/// Target words before the first exact match in a minimum-edit alignment
/// of `hyp` against `target` (all of them when nothing matches).
fn leading_lost(target: &str, hyp: &str) -> usize {
    let (r, h) = (wer::normalize(target), wer::normalize(hyp));
    let mut cost = vec![vec![0usize; h.len() + 1]; r.len() + 1];
    for (i, row) in cost.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in cost[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=r.len() {
        for j in 1..=h.len() {
            cost[i][j] = (cost[i - 1][j - 1] + usize::from(r[i - 1] != h[j - 1]))
                .min(cost[i - 1][j] + 1)
                .min(cost[i][j - 1] + 1);
        }
    }
    // Walk back from the end; the last exact match seen is the first one
    // in reading order.
    let (mut i, mut j, mut first) = (r.len(), h.len(), r.len());
    while i > 0 && j > 0 {
        let same = r[i - 1] == h[j - 1];
        if cost[i][j] == cost[i - 1][j - 1] + usize::from(!same) {
            if same {
                first = i - 1;
            }
            i -= 1;
            j -= 1;
        } else if cost[i][j] == cost[i - 1][j] + 1 {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    first
}

#[test]
fn leading_lost_counts_target_words_before_the_first_match() {
    let target = "please remember to close the window";
    assert_eq!(
        leading_lost(target, "Please remember to close the window."),
        0
    );
    assert_eq!(leading_lost(target, "to close the window"), 2);
    // A misheard first word counts as lost, the same as a dropped one.
    assert_eq!(leading_lost(target, "member to close the window"), 2);
    // Leaked words before the target do not hide a clean start.
    assert_eq!(
        leading_lost(target, "gold please remember to close the window"),
        0
    );
    assert_eq!(leading_lost(target, "nothing alike"), 6);
}

#[test]
fn folding_and_leaks_score_words_not_spelling() {
    assert_eq!(folded("Our neighbours planted,"), "our neighbors planted");
    let source = "We want you to help us publish.";
    let target = "Please remember to close the window.";
    // "to" is in both texts, so it is not a leak.
    assert_eq!(leaked(source, target, "We want you to close the window"), 3);
    assert_eq!(leaked(source, target, target), 0);
}

#[derive(Default)]
struct Tally {
    errors: f64,
    words: usize,
    clean: usize,
    lost_clips: usize,
    lost_words: usize,
    leaked_clips: usize,
    leaked_words: usize,
    clips: usize,
}

#[test]
#[ignore = "requires TURBOSPARK_TEST_OMNIVOICE, Parakeet Redux and LibriSpeech WAVs"]
fn reference_transcript_ab() {
    let Some(install) = std::env::var_os("TURBOSPARK_TEST_OMNIVOICE").map(PathBuf::from) else {
        eprintln!("skipping: TURBOSPARK_TEST_OMNIVOICE is unset");
        return;
    };
    let Some(parakeet) = dir_from("TURBOSPARK_PARAKEET_REDUX_MODEL", "models/parakeet-redux")
    else {
        return;
    };
    let Some(wavs) = dir_from(
        "TURBOSPARK_TEST_LIBRISPEECH_WAV",
        "models/librispeech/LibriSpeech/wav16k",
    ) else {
        return;
    };
    let out = std::env::var_os("TURBOSPARK_TEST_OMNIVOICE_AB_OUT").map(PathBuf::from);
    if let Some(out) = &out {
        std::fs::create_dir_all(out).expect("out dir");
    }

    let mut runner = OmniVoiceRunner::open(&install, Device::Metal).expect("OmniVoice opens");
    let oracle = ParakeetTdt::open(&parakeet).expect("Parakeet opens");
    let rate = runner.sample_rate();
    eprintln!(
        "device {:?}, {STEPS} steps, seeds {SEEDS:?}",
        runner.device()
    );

    let mut tallies: [Tally; 4] = Default::default();
    let started = std::time::Instant::now();
    for (index, (id, transcript)) in REFERENCES.iter().enumerate() {
        let stale = REFERENCES[(index + 1) % REFERENCES.len()].1;
        let reference = read_wav_f32(&wavs.join(format!("{id}.wav"))).expect("reference wav");
        assert_eq!(reference.channels, 1, "{id} must be mono");
        assert_eq!(reference.sample_rate, 16_000, "{id} must be 16 kHz");
        let asr = oracle
            .transcribe(&reference.samples, 16_000)
            .expect("transcribe reference");
        eprintln!(
            "ASR\t{id}\twer {:.3}\t{asr}",
            wer::wer(transcript, &folded(&asr))
        );
        for (t, target) in TARGETS.iter().enumerate() {
            for seed in SEEDS {
                for (v, reference_text) in
                    [None, Some(*transcript), Some(asr.as_str()), Some(stale)]
                        .into_iter()
                        .enumerate()
                {
                    let request = OmniVoiceRequest {
                        text: target,
                        language: Some("en"),
                        instruction: None,
                        reference: Some(&reference.samples),
                        reference_rate: reference.sample_rate,
                        reference_text,
                        steps: STEPS,
                        speed: 1.0,
                        seed,
                    };
                    let (_, clip) = runner
                        .synthesize(&request, &mut |_, _| true)
                        .expect("real clone");
                    assert!(!clip.is_empty() && clip.iter().all(|s| s.is_finite()));
                    let wave = audio::Waveform {
                        sample_rate: rate,
                        channels: 1,
                        samples: clip,
                    };
                    let pcm16 = audio::to_mono_resampled(
                        &wave,
                        16_000,
                        &audio::MonoResampleStrategy::SincHann(Default::default()),
                    )
                    .expect("resample");
                    let heard = oracle.transcribe(&pcm16, 16_000).expect("transcribe");
                    let words = wer::normalize(target).len();
                    let scored = folded(&heard);
                    let rate_err = wer::wer(target, &scored);
                    let lost = leading_lost(target, &scored);
                    let leak = leaked(reference_text.unwrap_or(transcript), target, &scored);
                    let tally = &mut tallies[v];
                    tally.errors += rate_err * words as f64;
                    tally.words += words;
                    tally.clean += usize::from(rate_err == 0.0);
                    tally.lost_clips += usize::from(lost > 0);
                    tally.lost_words += lost;
                    tally.leaked_clips += usize::from(leak > 0);
                    tally.leaked_words += leak;
                    tally.clips += 1;
                    eprintln!(
                        "CLIP\t{id}\tt{t}\tseed {seed}\t{}\twer {rate_err:.3}\tlost {lost}\tleaked {leak}\t{heard}",
                        VARIANTS[v]
                    );
                    if let Some(out) = &out {
                        let name = format!("{id}-t{t}-s{seed}-{}.wav", VARIANTS[v]);
                        std::fs::write(out.join(name), write_wav_f32(&wave)).expect("clip writes");
                    }
                }
            }
        }
    }
    eprintln!("total {:.0} s", started.elapsed().as_secs_f64());
    for (name, tally) in VARIANTS.iter().zip(&tallies) {
        eprintln!(
            "SUMMARY\t{name}\twer {:.2}% ({:.0}/{})\tclean {}/{}\tleading loss {} clips, {} words\tleaks {} clips, {} words",
            100.0 * tally.errors / tally.words.max(1) as f64,
            tally.errors,
            tally.words,
            tally.clean,
            tally.clips,
            tally.lost_clips,
            tally.lost_words,
            tally.leaked_clips,
            tally.leaked_words,
        );
    }
}
