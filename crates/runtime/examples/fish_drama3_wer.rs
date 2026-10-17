//! Fish Audio Drama 3 word error rate: synthesize a corpus, transcribe each
//! clip with the in-repo Parakeet TDT (CPU), and score it.
//!
//! ```sh
//! cargo run --release -p turbospark-runtime --example fish_drama3_wer -- \
//!     <install-dir> <corpus.txt | --smoke> <parakeet-dir> [seed=0] [expert-slots=16] \
//!     [out-dir | -] [recipe=fish|reference]
//! ```
//!
//! `recipe` picks the main sampler: `fish` (the default, Fish Audio's serving
//! recipe, `SamplingParams::FISH_SERVING`) or `reference` (mlx-audio's,
//! `SamplingParams::REFERENCE`), so the two can be compared on one corpus.
//! Each clip also prints every chunk's frames against its limit and how it
//! stopped, and whether the runner reports the take as stopped early.
//!
//! `<corpus.txt>` holds one sentence per line (blank lines and lines starting
//! with `#` are skipped). A line may carry `[cues]` and speaker tags: the
//! score compares the transcript with the words a chunk voices
//! (`chunking::spoken_text`), not with the markup. `<parakeet-dir>` is a
//! `moondream/parakeet-redux` checkout (`~/models/parakeet-redux`).
//! `out-dir` keeps each clip as a 44.1 kHz WAV.
//!
//! `--smoke` runs a few sentences to show the harness works. It is NOT the
//! evaluation set (the 28-sentence corpus), and a WER from it is not a
//! quality result. The run is labelled so in its output.
//!
//! A clip fails by the rule in `docs/FISH_DRAMA3_PHASE0.md` (Warm runs): its
//! transcript misses most of the text AND it ends in at least
//! [`FAIL_SILENT_FRAMES`] silent frames. "Most" is [`FAIL_WER`], a threshold
//! the doc does not state, so each clip's WER and tail length are printed too.
//! A clip that meets one half of the rule only is flagged, because the doc
//! records that the two went together.
//!
//! Timing is not measured here. Run it on a host that is not loading the
//! GPU if the frame counts' wall time matters; the frame count is fixed by
//! the seed and the words.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};

use audio::stt::parakeet::ParakeetTdt;
use audio::tts::fish_drama3::chunking::{split_text, spoken_text};
use audio::tts::fish_drama3::sampling::SamplingParams;
use turbospark_runtime::fish_drama3::{
    Drama3Request, Drama3Sampling, FishDrama3Runner, MAX_CONTEXT,
};
use turbospark_runtime::{ExpertCacheSlots, ExpertResidency};

#[path = "../tests/common/wer.rs"]
mod wer;

/// Samples per codec frame (46.4 ms at 44.1 kHz).
const FRAME_SAMPLES: usize = 2048;
/// A frame is silent under this RMS (dBFS, full scale 1.0).
const SILENT_DBFS: f32 = -45.0;
/// Trailing silent frames at or above which the tail counts as a failure
/// (`docs/FISH_DRAMA3_PHASE0.md`: every failed clip ended in 458 or more,
/// every other clip in at most 8).
const FAIL_SILENT_FRAMES: usize = 458;
/// "Misses most of the text": a WER of at least this. The doc states no
/// number, so this is the plain reading of "most".
const FAIL_WER: f64 = 0.5;

/// Sentences to show the harness works. Not the evaluation corpus.
const SMOKE: [&str; 3] = [
    "The quick brown fox jumps over the lazy dog.",
    "Please call Stella and ask her to bring these things with her from the store.",
    "[sighs] Fine. If we miss it, it's on you.",
];

/// Trailing frames of `samples` whose RMS is under [`SILENT_DBFS`]. A final
/// partial frame counts as a frame.
fn trailing_silent_frames(samples: &[f32]) -> usize {
    let floor = 10f32.powf(SILENT_DBFS / 20.0);
    samples
        .chunks(FRAME_SAMPLES)
        .rev()
        .take_while(|frame| {
            let power = frame.iter().map(|v| v * v).sum::<f32>() / frame.len() as f32;
            power.sqrt() < floor
        })
        .count()
}

#[derive(Debug, PartialEq)]
enum Verdict {
    Pass,
    /// Both halves of the rule.
    Fail,
    /// The transcript missed most of the text and the tail is not silent.
    MissWithoutSilence,
    /// A long silent tail and a transcript that kept most of the text.
    SilenceWithoutMiss,
}

fn verdict(wer: f64, tail_frames: usize) -> Verdict {
    match (wer >= FAIL_WER, tail_frames >= FAIL_SILENT_FRAMES) {
        (true, true) => Verdict::Fail,
        (true, false) => Verdict::MissWithoutSilence,
        (false, true) => Verdict::SilenceWithoutMiss,
        (false, false) => Verdict::Pass,
    }
}

/// The words a request voices: every chunk's text without tags and cues.
fn spoken_words(text: &str) -> String {
    split_text(text)
        .iter()
        .map(|chunk| spoken_text(chunk).words)
        .collect::<Vec<_>>()
        .join(" ")
}

fn read_corpus(path: &str) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    Ok(parse_corpus(&text))
}

fn parse_corpus(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let install = PathBuf::from(args.next().ok_or("install dir")?);
    let corpus_arg = args.next().ok_or("corpus.txt or --smoke")?;
    let stt_dir = PathBuf::from(args.next().ok_or("parakeet dir")?);
    let seed: u64 = args.next().map_or(Ok(0), |s| s.parse())?;
    let slots: usize = args.next().map_or(Ok(16), |s| s.parse())?;
    let out_dir = args.next().filter(|d| d != "-").map(PathBuf::from);
    let (recipe_name, recipe) = match args.next().as_deref() {
        None | Some("fish") => ("fish", SamplingParams::FISH_SERVING),
        Some("reference") => ("reference", SamplingParams::REFERENCE),
        Some(other) => return Err(format!("recipe {other:?}: fish or reference").into()),
    };
    let sampling = Drama3Sampling {
        temperature: Some(recipe.temperature),
        top_p: Some(recipe.top_p),
        top_k: Some(recipe.top_k),
        max_frames: None,
    };
    eprintln!("recipe {recipe_name}: {recipe:?}");

    let smoke = corpus_arg == "--smoke";
    let corpus: Vec<String> = if smoke {
        SMOKE.iter().map(|s| s.to_string()).collect()
    } else {
        read_corpus(&corpus_arg)?
    };
    if corpus.is_empty() {
        return Err("the corpus has no sentences".into());
    }
    if smoke {
        eprintln!(
            "SMOKE CORPUS ({} sentences): this shows the harness runs. It is not the \
             evaluation set and its WER is not a quality result.",
            corpus.len()
        );
    }
    if let Some(dir) = &out_dir {
        std::fs::create_dir_all(dir)?;
    }

    let stt = ParakeetTdt::open(&stt_dir)?;
    let mut runner = FishDrama3Runner::open(
        &install,
        ExpertCacheSlots::Fixed(slots),
        ExpertResidency::Streamed,
        MAX_CONTEXT,
    )?;
    let rate = runner.sample_rate();

    let mut failed = 0usize;
    let mut flagged = 0usize;
    let mut early = 0usize;
    let mut wers = Vec::new();
    let mut passing_wers = Vec::new();
    // The largest share of its limit an EOS chunk of a passing clip used: the
    // text cap keeps every normal clip under half of it.
    let mut widest_share = 0f64;
    for (i, text) in corpus.iter().enumerate() {
        let mut samples = Vec::new();
        let request = Drama3Request {
            text,
            reference: None,
            seed,
            sampling,
        };
        let take = runner.synthesize(&request, &|| false, |pcm| {
            samples.extend(pcm);
            true
        })?;
        let chunk_list = take
            .chunks
            .iter()
            .map(|c| format!("{}/{} {:?}", c.frames, c.limit, c.stop))
            .collect::<Vec<_>>()
            .join(", ");
        let frames = samples.len().div_ceil(FRAME_SAMPLES);
        let tail = trailing_silent_frames(&samples);
        let wave = audio::Waveform {
            sample_rate: rate,
            channels: 1,
            samples,
        };
        if let Some(dir) = &out_dir {
            write_clip(dir, i, &wave)?;
        }
        let mono16 = audio::to_mono_resampled(
            &wave,
            16_000,
            &audio::MonoResampleStrategy::SincHann(Default::default()),
        )?;
        let transcript = if mono16.is_empty() {
            String::new()
        } else {
            stt.transcribe(&mono16, 16_000)?
        };
        let score = wer::wer(&spoken_words(text), &transcript);
        let verdict = verdict(score, tail);
        match verdict {
            Verdict::Pass => {
                passing_wers.push(score);
                for c in &take.chunks {
                    if c.stop == audio::tts::fish_drama3::generate::ChunkStop::Eos {
                        widest_share = widest_share.max(c.frames as f64 / c.limit as f64);
                    }
                }
            }
            Verdict::Fail => failed += 1,
            _ => flagged += 1,
        }
        if take.stopped_early() {
            early += 1;
        }
        wers.push(score);
        println!(
            "clip {i:>3} seed {seed}: {frames:>4} frames, tail {tail:>4} silent, WER {score:.3}, {verdict:?}\n  chunks: {chunk_list}; longest silence {} frames; stopped early {}\n  text: {text}\n  heard: {transcript}",
            take.longest_silence_frames,
            take.stopped_early()
        );
    }
    let mean = wers.iter().sum::<f64>() / wers.len() as f64;
    let passing_mean = passing_wers.iter().sum::<f64>() / passing_wers.len().max(1) as f64;
    println!(
        "{} clips at seed {seed}, recipe {recipe_name}: {failed} failed, {flagged} flagged (one \
         half of the rule), {early} reported stopped early, mean WER {mean:.3}, mean WER of \
         passing clips {passing_mean:.3}, widest EOS chunk {:.0}% of its limit{}",
        corpus.len(),
        widest_share * 100.0,
        if smoke {
            " (SMOKE CORPUS, not a quality result)"
        } else {
            ""
        }
    );
    Ok(())
}

fn write_clip(dir: &Path, index: usize, wave: &audio::Waveform) -> std::io::Result<()> {
    std::fs::write(
        dir.join(format!("clip{index:03}.wav")),
        audio::write_wav_f32(wave),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(frames: usize, amplitude: f32) -> Vec<f32> {
        vec![amplitude; frames * FRAME_SAMPLES]
    }

    #[test]
    fn the_tail_counts_only_trailing_silent_frames() {
        // -45 dBFS is an RMS of about 0.0056.
        let mut samples = tone(10, 0.1);
        samples.extend(tone(5, 0.001));
        assert_eq!(trailing_silent_frames(&samples), 5);
        // Silence inside the clip is not the tail.
        let mut inner = tone(3, 0.001);
        inner.extend(tone(4, 0.1));
        assert_eq!(trailing_silent_frames(&inner), 0);
        assert_eq!(trailing_silent_frames(&[]), 0);
    }

    #[test]
    fn the_silence_floor_is_minus_45_dbfs() {
        assert_eq!(trailing_silent_frames(&tone(2, 0.0050)), 2);
        assert_eq!(trailing_silent_frames(&tone(2, 0.0063)), 0);
    }

    #[test]
    fn a_partial_final_frame_counts_as_a_frame() {
        let mut samples = tone(2, 0.1);
        samples.extend(vec![0.0; 100]);
        assert_eq!(trailing_silent_frames(&samples), 1);
    }

    #[test]
    fn a_clip_fails_only_when_both_halves_of_the_rule_hold() {
        assert_eq!(verdict(0.9, 458), Verdict::Fail);
        assert_eq!(verdict(0.5, 1024), Verdict::Fail);
        assert_eq!(verdict(0.9, 457), Verdict::MissWithoutSilence);
        assert_eq!(verdict(0.49, 458), Verdict::SilenceWithoutMiss);
        assert_eq!(verdict(0.0, 8), Verdict::Pass);
    }

    #[test]
    fn the_score_reads_the_voiced_words_not_the_markup() {
        assert_eq!(
            wer::normalize(&spoken_words("[sighs] Fine. If we miss it, it's on you.")),
            wer::normalize("Fine. If we miss it, it's on you.")
        );
        assert_eq!(
            wer::wer(
                &spoken_words("[sighs] Fine. If we miss it, it's on you."),
                "fine if we miss it it's on you"
            ),
            0.0
        );
    }

    #[test]
    fn a_corpus_skips_blank_and_comment_lines() {
        assert_eq!(
            parse_corpus("# eval\n\none\n  two  \n#three\n"),
            ["one", "two"]
        );
    }
}
