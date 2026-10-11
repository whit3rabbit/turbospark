//! Run MegaTTS3 clone requests on one opened runner and print a JSON line per
//! request.
//!
//! Usage: megatts3_bench <model-dir> --jobs JOBS.json [--device metal|cpu] [--warmup]
//!
//! JOBS.json is an array of `{"reference": WAV, "text": TEXT, "output": WAV,
//! "steps": N, "seed": N, "p_w": X, "t_w": X}`. Only the first three are
//! required; the rest default to `SynthParams::default()` and seed 0. The model
//! opens once, so one process covers many requests. A prepared voice is cached
//! per (reference, seed) as in the app, and each line says whether its request
//! prepared the voice or reused it.
//!
//! `front_ms` is the time until the first diffusion progress event (reference
//! preparation when not cached, G2P and the duration model). `total_ms` covers
//! the whole request, including the WavVAE decode.
//!
//! `admission_bytes` is what runtime admission checks before the request: the
//! process footprint at that moment plus the request's activation reserve. A
//! request whose measured peak footprint exceeds it would be admitted on a
//! machine it does not fit.
//!
//! `--warmup` runs one discarded 1-step request first so Metal pipeline
//! compilation stays out of the timings. Peak process memory is not measured
//! here: run under `/usr/bin/time -l` and read "peak memory footprint".

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use audio::tts::megatts3::{
        activation_reserve_bytes, plan_chunks, resample_reference, SynthParams, SAMPLE_RATE,
    };
    use std::path::{Path, PathBuf};
    use std::time::Instant;
    use turbospark_runtime::megatts3::{CloneProgress, CloneRequest, Device, MegaTts3Runner};

    fn ms(d: std::time::Duration) -> f64 {
        d.as_secs_f64() * 1e3
    }
    /// Mono 24 kHz, exactly as `turbospark-clone` loads a reference.
    fn load(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let wave = audio::read_wav_f32(path)?;
        let channels = usize::from(wave.channels).max(1);
        let mono: Vec<f32> = wave
            .samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect();
        Ok(resample_reference(&mono, wave.sample_rate)?)
    }

    let mut args = std::env::args().skip(1);
    let model = PathBuf::from(args.next().ok_or("model directory required")?);
    let (mut jobs_path, mut device, mut warmup) = (None, Device::Metal, false);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--warmup" => warmup = true,
            "--jobs" => jobs_path = Some(PathBuf::from(args.next().ok_or("--jobs needs a path")?)),
            "--device" => {
                device = match args.next().as_deref() {
                    Some("metal") => Device::Metal,
                    Some("cpu") => Device::Cpu,
                    other => return Err(format!("unknown device {other:?}").into()),
                }
            }
            other => return Err(format!("unknown option {other}").into()),
        }
    }
    let jobs: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(jobs_path.ok_or("--jobs is required")?)?)?;
    if jobs.is_empty() {
        return Err("no jobs".into());
    }
    let text = |job: &serde_json::Value, key: &str| -> Result<String, String> {
        job[key]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| format!("job needs a string {key:?}"))
    };
    let device_name = match device {
        Device::Metal => "metal",
        Device::Cpu => "cpu",
    };

    let started = Instant::now();
    let mut runner = MegaTts3Runner::open(&model, device)?;
    eprintln!(
        "{}",
        serde_json::json!({
            "open_ms": ms(started.elapsed()),
            "device": device_name,
            "footprint_after_open_bytes": gpu::process_footprint(),
        })
    );
    if warmup {
        let reference = load(Path::new(&text(&jobs[0], "reference")?))?;
        let request = CloneRequest {
            text: "Warm up.",
            reference: &reference,
            reference_rate: SAMPLE_RATE,
            seed: u64::MAX,
            params: SynthParams {
                steps: 1,
                ..SynthParams::default()
            },
        };
        runner.synthesize_controlled(&request, &mut |_| Ok(()), &mut |_| true)?;
    }

    for (index, job) in jobs.iter().enumerate() {
        let reference = load(Path::new(&text(job, "reference")?))?;
        let defaults = SynthParams::default();
        let params = SynthParams {
            steps: job["steps"].as_u64().map_or(defaults.steps, |s| s as usize),
            p_w: job["p_w"].as_f64().map_or(defaults.p_w, |v| v as f32),
            t_w: job["t_w"].as_f64().map_or(defaults.t_w, |v| v as f32),
            ..defaults
        };
        let words = text(job, "text")?;
        let request = CloneRequest {
            text: &words,
            reference: &reference,
            reference_rate: SAMPLE_RATE,
            seed: job["seed"].as_u64().unwrap_or(0),
            params: params.clone(),
        };
        // The runtime's own run estimate (`native_audio`), for the reference
        // already at 24 kHz.
        let longest = plan_chunks(&words, params.max_chunk_bytes)?
            .iter()
            .map(String::len)
            .max()
            .unwrap_or(0);
        let reserve = activation_reserve_bytes(reference.len(), longest, params.dur_alpha);
        let footprint_before = gpu::process_footprint();
        let (mut prepared, mut front) = (false, None);
        let mut samples = Vec::new();
        let started = Instant::now();
        runner.synthesize_controlled(
            &request,
            &mut |progress| {
                match progress {
                    CloneProgress::Reference => prepared = true,
                    CloneProgress::Diffusion { .. } if front.is_none() => {
                        front = Some(started.elapsed())
                    }
                    CloneProgress::Diffusion { .. } => {}
                }
                Ok(())
            },
            &mut |pcm| {
                samples.extend(pcm);
                true
            },
        )?;
        let total = started.elapsed();
        let output = text(job, "output")?;
        let wave = audio::Waveform::mono(SAMPLE_RATE, samples)?;
        std::fs::write(&output, audio::write_wav_f32(&wave))?;
        let audio_seconds = wave.duration_seconds();
        let report = serde_json::json!({
            "job": index,
            "device": device_name,
            "steps": params.steps,
            "p_w": params.p_w,
            "t_w": params.t_w,
            "seed": request.seed,
            "prepared_voice": prepared,
            "reference_seconds": reference.len() as f64 / f64::from(SAMPLE_RATE),
            "audio_seconds": audio_seconds,
            "front_ms": front.map(ms),
            "total_ms": ms(total),
            "rtf": total.as_secs_f64() / audio_seconds,
            "reserve_bytes": reserve,
            "footprint_before_bytes": footprint_before,
            "admission_bytes": footprint_before.map(|f| f + reserve),
            "output": output,
        });
        println!("{report}");
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("megatts3_bench requires macOS and a Metal device");
}
