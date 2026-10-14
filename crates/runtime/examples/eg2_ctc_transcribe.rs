//! Transcribes a WAV with the EmbeddingGemma 2 word-CTC programs (eg2demo
//! v5) on Core ML. Experiment.
//!
//! Usage: `cargo run --release -p turbospark-runtime --example
//! eg2_ctc_transcribe -- <asset_dir> <wav> [--cpu-scorer] [--recall]`
//!
//! - `--cpu-scorer` replaces the compiled scorer with the portable f32 CPU
//!   reference of the same program.
//! - `--recall` also runs both scorers on every window and reports, over the
//!   frames with real audio, how often the 64 shortlisted candidates hold
//!   the exhaustive best word, and how often the two scorers' reranked
//!   words agree. The exhaustive search is slow (229457 rows per frame).
//!
//! Placement follows `TURBOSPARK_EG2_CTC_{ENCODER,READER,SCORER}_UNITS`
//! (default `all`). macOS 15 or newer.

#![cfg(target_os = "macos")]

use std::path::Path;
use std::time::Instant;

use audio::conversion::{to_mono_resampled, MonoResampleStrategy};
use audio::stt::embeddinggemma2::frontend::{HOP_LENGTH, SAMPLE_RATE};
use audio::stt::embeddinggemma2::scorer::{exhaustive_best, rerank, CpuBlockScorer, BLOCK_SIZE};
use audio::stt::embeddinggemma2::windows::{
    plan_windows, valid_encoder_frames, ENCODER_FRAMES, MEL_FRAMES, WINDOW_SAMPLES,
};
use audio::stt::embeddinggemma2::{
    AcousticBackend, BlockShortlister, Calibration, LogMelFrontend, Transcript, WordTable,
    EMBEDDING_DIM,
};
use audio::wav::read_wav_f32_bytes;
use turbospark_runtime::coreml::embeddinggemma2_ctc::{
    open_with, CoreMlAcoustic, CoreMlScorer, Eg2CtcOptions,
};

fn main() -> Result<(), String> {
    let mut positional = Vec::new();
    let (mut cpu_scorer, mut recall) = (false, false);
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--cpu-scorer" => cpu_scorer = true,
            "--recall" => recall = true,
            _ => positional.push(arg),
        }
    }
    let [asset_dir, wav_path] = positional.as_slice() else {
        return Err("usage: eg2_ctc_transcribe <asset_dir> <wav> [--cpu-scorer] [--recall]".into());
    };
    let dir = Path::new(asset_dir);
    let opts = Eg2CtcOptions::from_env()?;

    let bytes = std::fs::read(wav_path).map_err(|e| format!("read {wav_path}: {e}"))?;
    let wave = read_wav_f32_bytes(&bytes).map_err(|e| format!("decode wav: {e}"))?;
    let mono = to_mono_resampled(
        &wave,
        SAMPLE_RATE,
        &MonoResampleStrategy::SincHann(Default::default()),
    )
    .map_err(|e| format!("resample: {e}"))?;
    let seconds = mono.len() as f64 / f64::from(SAMPLE_RATE);

    let started = Instant::now();
    let backend = CoreMlAcoustic::open(dir, &opts)?;
    eprintln!(
        "encoder + reader loaded with {opts:?} in {:?}",
        started.elapsed()
    );

    if cpu_scorer {
        let started = Instant::now();
        let scorer = CpuBlockScorer::load(dir).map_err(|e| e.to_string())?;
        let mut model = open_with(dir, backend, scorer)?;
        eprintln!("CPU scorer + vocabulary loaded in {:?}", started.elapsed());
        let transcript = timed_runs(seconds, || {
            model.transcribe(&mono).map_err(|e| e.to_string())
        })?;
        print_transcript(&transcript);
        report_acoustic(&model.backend);
        if recall {
            let mut coreml = CoreMlScorer::open(dir, opts.scorer_units)?;
            measure_recall(
                &mono,
                &mut model.backend,
                &mut coreml,
                &mut model.shortlister,
                (
                    &model.frontend,
                    &model.table,
                    model.calibration,
                    model.tail_padding,
                ),
            )?;
        }
    } else {
        let started = Instant::now();
        let scorer = CoreMlScorer::open(dir, opts.scorer_units)?;
        let mut model = open_with(dir, backend, scorer)?;
        eprintln!(
            "Core ML scorer + vocabulary loaded in {:?}",
            started.elapsed()
        );
        let transcript = timed_runs(seconds, || {
            model.transcribe(&mono).map_err(|e| e.to_string())
        })?;
        print_transcript(&transcript);
        report_acoustic(&model.backend);
        let scorer = model.shortlister.time;
        eprintln!("scorer: {} calls, {:?} total", scorer.calls, scorer.total);
        if recall {
            let mut cpu = CpuBlockScorer::load(dir).map_err(|e| e.to_string())?;
            measure_recall(
                &mono,
                &mut model.backend,
                &mut model.shortlister,
                &mut cpu,
                (
                    &model.frontend,
                    &model.table,
                    model.calibration,
                    model.tail_padding,
                ),
            )?;
        }
    }
    Ok(())
}

/// Runs twice: the first prediction per program includes Core ML's one-time
/// specialization, so a warm run is reported separately.
fn timed_runs(
    seconds: f64,
    mut run: impl FnMut() -> Result<Transcript, String>,
) -> Result<Transcript, String> {
    let mut last = None;
    for label in ["cold", "warm"] {
        let started = Instant::now();
        last = Some(run()?);
        let elapsed = started.elapsed();
        eprintln!(
            "{label}: {seconds:.2}s of audio in {elapsed:?} (RTFx {:.1})",
            seconds / elapsed.as_secs_f64()
        );
    }
    Ok(last.expect("ran at least once"))
}

fn print_transcript(transcript: &Transcript) {
    for word in &transcript.words {
        println!(
            "[{:7.2} - {:7.2}] {:>6} {}",
            word.start_seconds, word.end_seconds, word.word_id, word.text
        );
    }
    let emitted = transcript.frame_path.iter().filter(|l| l.is_some()).count();
    println!(
        "windows: {}, frames: {}, word frames: {emitted}",
        transcript.windows,
        transcript.frame_path.len()
    );
    println!("text: {}", transcript.text);
}

fn report_acoustic(backend: &CoreMlAcoustic) {
    eprintln!(
        "encoder: {} calls, {:?} total; reader: {} calls, {:?} total",
        backend.encode_time.calls,
        backend.encode_time.total,
        backend.read_time.calls,
        backend.read_time.total
    );
}

/// Shortlist recall against the exhaustive search, for both scorers, over
/// every kept frame (all of them carry real audio), and over the frames
/// whose exhaustive winner beats blank (the frames that emit a word). A
/// "changed" frame is one whose decision differs from the exhaustive one.
fn measure_recall(
    mono: &[f32],
    backend: &mut CoreMlAcoustic,
    coreml: &mut dyn BlockShortlister,
    cpu: &mut dyn BlockShortlister,
    model: (&LogMelFrontend, &WordTable, Calibration, usize),
) -> Result<(), String> {
    #[derive(Default)]
    struct Counts {
        hits: usize,
        word_hits: usize,
        changed: usize,
    }
    let (frontend, table, calibration, tail_padding) = model;
    // The same input the transcriber windows: audio plus its tail padding.
    let mono = &[mono, &vec![0.0; tail_padding]].concat()[..];
    let plans = plan_windows(mono.len(), |len| {
        valid_encoder_frames(len.saturating_sub(HOP_LENGTH).div_ceil(HOP_LENGTH))
    });
    let (mut frames, mut word_frames, mut agree) = (0usize, 0usize, 0usize);
    let (mut a_counts, mut b_counts) = (Counts::default(), Counts::default());
    let started = Instant::now();
    for plan in &plans {
        let window = &mono[plan.sample_start..plan.sample_start + plan.sample_len];
        let features = frontend
            .features(window, WINDOW_SAMPLES)
            .map_err(|e| e.to_string())?
            .fit_to(MEL_FRAMES);
        let valid = valid_encoder_frames(features.valid_frames());
        let hidden = backend
            .encode(&features.values, &features.mask_f32())
            .map_err(|e| e.to_string())?;
        let mask: Vec<f32> = (0..ENCODER_FRAMES)
            .map(|f| if f < valid { 1.0 } else { 0.0 })
            .collect();
        let reader = backend.read(&hidden, &mask).map_err(|e| e.to_string())?;
        let a = coreml
            .shortlist(&reader.vectors, ENCODER_FRAMES)
            .map_err(|e| e.to_string())?;
        let b = cpu
            .shortlist(&reader.vectors, ENCODER_FRAMES)
            .map_err(|e| e.to_string())?;
        for frame in plan.keep.clone() {
            let vector = &reader.vectors[frame * EMBEDDING_DIM..(frame + 1) * EMBEDDING_DIM];
            let blank = reader.blank[frame];
            let best = exhaustive_best(vector, table).ok_or("empty table")?;
            let truth = calibration.decide(Some(best), blank);
            let block = best.word_id / BLOCK_SIZE as u32;
            frames += 1;
            word_frames += usize::from(truth.is_some());
            let tally = |counts: &mut Counts, row: &[u32]| {
                let hit = row.contains(&block);
                counts.hits += usize::from(hit);
                counts.word_hits += usize::from(hit && truth.is_some());
                let decision = calibration.decide(rerank(vector, row, table), blank);
                counts.changed += usize::from(decision != truth);
                decision
            };
            let da = tally(&mut a_counts, a.row(frame));
            let db = tally(&mut b_counts, b.row(frame));
            agree += usize::from(da == db);
        }
    }
    let pct = |n: usize, of: usize| 100.0 * n as f64 / of.max(1) as f64;
    for (name, c) in [("Core ML", &a_counts), ("CPU", &b_counts)] {
        println!(
            "{name} scorer: recall {}/{frames} ({:.2}%) all frames, {}/{word_frames} ({:.2}%) word frames; decisions changed vs exhaustive {}",
            c.hits,
            pct(c.hits, frames),
            c.word_hits,
            pct(c.word_hits, word_frames),
            c.changed
        );
    }
    println!(
        "Core ML and CPU decisions agree on {agree}/{frames} frames [{:?}]",
        started.elapsed()
    );
    Ok(())
}
