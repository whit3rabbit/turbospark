//! The worker's audio arm: open an audio install, run its job once warm,
//! then once measured, and record wall time, real-time factor, per-stage
//! times and a digest of the output.
//!
//! Wired pipelines (`audio_kind::WORKER_KINDS`): MiniMax Music 3, whose
//! runner returns per-stage times, and Whisper. Others are refused by name
//! rather than run through some other path under their label.
//!
//! The digest is what the audio gates compare: the SHA-256 of the
//! waveform's f32 bytes (bit-exact output) or of the transcript text
//! (text-exact output, where a numerics change may move logits but not the
//! words). Same seed and same job on both arms, so the bytes are a function
//! of the arithmetic alone.

use std::path::Path;
use std::time::Instant;

use serde_json::{json, Value};
use turbospark_bench::memory::{task_fault_counters, AppMemorySampler};

use crate::audio_kind::{self, AudioKind};
use crate::provenance;
use crate::worker::{RunRecord, WorkerRequest, MEASURING_MARKER};
use crate::workload::AudioJob;

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

pub fn run(req: &WorkerRequest, job: &AudioJob) -> Result<RunRecord, String> {
    let dir = catalog::resolve_model_arg(&req.model);
    let config_path = dir.join("config.json");
    let cfg: Value = serde_json::from_str(
        &std::fs::read_to_string(&config_path)
            .map_err(|e| format!("{}: {e}", config_path.display()))?,
    )
    .map_err(|e| format!("{}: {e}", config_path.display()))?;
    let kind = audio_kind::kind(&cfg).ok_or_else(|| {
        format!(
            "{}: not an audio pipeline tslab knows ({:?})",
            dir.display(),
            audio_kind::model_type(&cfg)
        )
    })?;
    if !audio_kind::runs_in_worker(kind) {
        return Err(format!(
            "{} has no worker path yet; its kernels are benched by tslab kernel",
            kind.as_str()
        ));
    }
    let mut sampler = AppMemorySampler::new();
    sampler.sample();
    let (open_s, warm_s, audio) = match (kind, job) {
        (AudioKind::Music3, AudioJob::Music { .. }) => music3(&dir, job, &mut sampler)?,
        (AudioKind::Whisper, AudioJob::Transcribe { .. }) => whisper(&dir, job, &mut sampler)?,
        (k, j) => {
            return Err(format!(
                "a {} job does not fit a {} install",
                match j {
                    AudioJob::Music { .. } => "music",
                    AudioJob::Transcribe { .. } => "transcribe",
                },
                k.as_str()
            ))
        }
    };
    sampler.sample();
    Ok(RunRecord {
        run_id: req.run_id.clone(),
        arm: req.arm.name.clone(),
        mode: req.mode,
        status: "ok".into(),
        refusal: None,
        family: Some(kind.as_str().to_string()),
        model_dir: Some(dir.display().to_string()),
        manifest_sha256: provenance::sha256_file(&config_path),
        open_s: Some(open_s),
        warm_s: Some(warm_s),
        resolved_slots: None,
        resolved_residency: None,
        chunked: None,
        prompts: Vec::new(),
        totals: None,
        peak_footprint_bytes: sampler.peak_bytes(),
        faults: task_fault_counters(),
        dispatch_profile: None,
        provenance: provenance::collect(),
        audio: Some(audio),
    })
}

fn music3(
    dir: &Path,
    job: &AudioJob,
    sampler: &mut AppMemorySampler,
) -> Result<(f64, f64, Value), String> {
    let AudioJob::Music {
        caption,
        lyrics,
        duration_s,
        steps,
        seed,
    } = job
    else {
        unreachable!("matched by the caller")
    };
    let request = |duration: f64, steps: Option<usize>| {
        let mut r =
            audio::music::minimax_music3::TextGenerateRequest::new(caption.clone(), lyrics.clone());
        r.duration_seconds = Some(duration);
        r.steps = steps;
        r.seed = Some(*seed);
        r
    };
    let t = Instant::now();
    let runner = runtime::Music3Runner::open(dir).map_err(|e| e.to_string())?;
    let open_s = t.elapsed().as_secs_f64();
    sampler.sample();
    // A short, one-step warm pass compiles every pipeline the job uses.
    let t = Instant::now();
    runner
        .generate_text(&request(duration_s.min(2.0), Some(1)))
        .map_err(|e| e.to_string())?;
    let warm_s = t.elapsed().as_secs_f64();
    sampler.sample();
    eprintln!("{MEASURING_MARKER}");
    let t = Instant::now();
    let (generation, st) = runner
        .generate_text_timed(&request(*duration_s, *steps))
        .map_err(|e| e.to_string())?;
    let run_s = t.elapsed().as_secs_f64();
    sampler.sample();
    // Interleaved stereo: two values per frame.
    let frames = generation.waveform.len() / 2;
    let audio_s = frames as f64 / generation.sample_rate.max(1) as f64;
    let bytes: Vec<u8> = generation
        .waveform
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    Ok((
        open_s,
        warm_s,
        json!({
            "task": "music", "run_s": run_s, "audio_s": audio_s,
            "rtf": run_s / audio_s.max(f64::MIN_POSITIVE),
            "stages": {
                "tokenize": ms(st.tokenize), "prefill": ms(st.prefill), "lm_head": ms(st.lm_head),
                "sampling": ms(st.sampling), "depth": ms(st.depth), "lm_decode": ms(st.lm_decode),
                "condition": ms(st.condition), "dit": ms(st.dit), "vocoder": ms(st.vocoder),
                "stitch": ms(st.stitch),
            },
            "frames": generation.frames, "sample_rate": generation.sample_rate,
            "output_digest": model_io::hash_data(&bytes),
        }),
    ))
}

fn whisper(
    dir: &Path,
    job: &AudioJob,
    sampler: &mut AppMemorySampler,
) -> Result<(f64, f64, Value), String> {
    let AudioJob::Transcribe { wav, language } = job else {
        unreachable!("matched by the caller")
    };
    let wave = audio::read_wav_f32(wav).map_err(|e| format!("{}: {e}", wav.display()))?;
    // Resampling is part of a frontend's arithmetic; the lab does not pick a
    // resampler on the runtime's behalf.
    if wave.sample_rate != 16_000 || wave.channels != 1 {
        return Err(format!(
            "{}: {} Hz, {} channel(s); give a 16 kHz mono WAV",
            wav.display(),
            wave.sample_rate,
            wave.channels
        ));
    }
    let pcm = &wave.samples;
    let t = Instant::now();
    let runner = runtime::WhisperRunner::open(dir)?;
    let open_s = t.elapsed().as_secs_f64();
    if !runner.using_metal() {
        return Err("Whisper opened on the CPU path; this arm would not measure Metal".into());
    }
    sampler.sample();
    let t = Instant::now();
    runner.transcribe(&pcm[..pcm.len().min(32_000)], language.as_deref())?;
    let warm_s = t.elapsed().as_secs_f64();
    sampler.sample();
    eprintln!("{MEASURING_MARKER}");
    let t = Instant::now();
    let out = runner.transcribe(pcm, language.as_deref())?;
    let run_s = t.elapsed().as_secs_f64();
    sampler.sample();
    let text: String = out
        .segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let audio_s = pcm.len() as f64 / 16_000.0;
    Ok((
        open_s,
        warm_s,
        json!({
            "task": "transcribe", "run_s": run_s, "audio_s": audio_s,
            "rtf": run_s / audio_s.max(f64::MIN_POSITIVE),
            "stages": {"transcribe": run_s * 1e3},
            "segments": out.segments.len(), "language": out.language,
            "text_digest": model_io::hash_data(text.as_bytes()),
            "output_digest": model_io::hash_data(text.as_bytes()),
        }),
    ))
}
