//! Time one MiniMax Music 3 Metal request and print a JSON line.
//!
//! Usage: music3_bench <model-dir> --caption TEXT --lyrics TEXT --output WAV
//!        [--duration S] [--steps N] [--seed N] [--precision checkpoint|float32] [--warmup]
//!
//! `--flow-only` skips the AR stage and feeds the flow stage deterministic
//! synthetic frame hiddens, for iterating on DiT and vocoder kernels. It is a
//! timing aid: the audio is not music and no quality statistic applies.
//!
//! `--warmup` runs one discarded 1 s, 1 step request first so Metal pipeline
//! compilation and first-touch costs stay out of the measured request. Peak
//! process memory is not measured here: run under `/usr/bin/time -l` (the
//! `crates/audio/scripts/benchmark_music3.py` driver does).

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use audio::music::minimax_music3::{output_stats, Music3Precision, TextGenerateRequest};
    use std::time::{Duration, Instant};

    fn ms(d: Duration) -> f64 {
        d.as_secs_f64() * 1e3
    }

    let mut args = std::env::args().skip(1);
    let model = std::path::PathBuf::from(args.next().ok_or("model directory required")?);
    let (mut caption, mut lyrics, mut output) = (None, None, None);
    let (mut duration, mut steps, mut seed) = (None, None, None);
    let mut precision = Music3Precision::Checkpoint;
    let mut warmup = false;
    let mut flow_only = false;
    while let Some(flag) = args.next() {
        if flag == "--warmup" {
            warmup = true;
            continue;
        }
        if flag == "--flow-only" {
            flow_only = true;
            continue;
        }
        let value = args
            .next()
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--caption" => caption = Some(value),
            "--lyrics" => lyrics = Some(value),
            "--output" => output = Some(std::path::PathBuf::from(value)),
            "--duration" => duration = Some(value.parse::<f64>()?),
            "--steps" => steps = Some(value.parse::<usize>()?),
            "--seed" => seed = Some(value.parse::<u64>()?),
            "--precision" => {
                precision = match value.as_str() {
                    "checkpoint" => Music3Precision::Checkpoint,
                    "float32" => Music3Precision::Float32,
                    other => return Err(format!("unknown precision {other}").into()),
                }
            }
            other => return Err(format!("unknown option {other}").into()),
        }
    }
    let caption = caption.ok_or("--caption required")?;
    let lyrics = lyrics.ok_or("--lyrics required")?;
    let output = output.ok_or("--output required")?;
    let mut request = TextGenerateRequest::new(caption, lyrics);
    request.duration_seconds = duration;
    request.steps = steps;
    request.seed = seed;

    let opened = Instant::now();
    let runner = turbospark_runtime::Music3Runner::open_with_precision(&model, precision)?;
    let open = opened.elapsed();

    let mut warmup_ms = None;
    if warmup {
        let mut small = request.clone();
        small.duration_seconds = Some(1.0);
        small.steps = Some(1);
        let started = Instant::now();
        runner.generate_text(&small)?;
        warmup_ms = Some(ms(started.elapsed()));
    }

    runner.reset_dispatch_profile();
    if flow_only {
        let config = runner.config();
        let frames =
            (request.duration_seconds.unwrap_or(60.0) * config.frame_rate).trunc() as usize;
        let fused = config.num_codebooks * config.hidden_size;
        let mut state = 0x9e3779b97f4a7c15u64;
        let hiddens: Vec<f32> = (0..frames * fused)
            .map(|_| {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                let unit =
                    (state.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32 / (1u64 << 24) as f32;
                0.5 * (2.0 * unit - 1.0)
            })
            .collect();
        let started = Instant::now();
        let planar = runner.run_flow(
            &hiddens,
            frames,
            request.steps.unwrap_or(30),
            request.seed.unwrap_or(0),
        )?;
        let wall = started.elapsed();
        let dispatch: Vec<_> = runner
            .dispatch_profile()
            .into_iter()
            .map(|d| {
                serde_json::json!({
                    "op": d.op, "shape": d.shape, "calls": d.calls,
                    "total_ms": ms(d.total), "mean_ms": ms(d.total) / d.calls as f64,
                })
            })
            .collect();
        let hash = planar.iter().fold(0xcbf29ce484222325u64, |h, v| {
            (h ^ u64::from(v.to_bits())).wrapping_mul(0x100000001b3)
        });
        println!(
            "{}",
            serde_json::json!({
                "flow_only": true, "frames": frames, "steps": request.steps.unwrap_or(30),
                "flow_ms": ms(wall), "samples": planar.len() / 2,
                "waveform_fnv": format!("{hash:016x}"), "dispatch": dispatch,
            })
        );
        return Ok(());
    }
    let started = Instant::now();
    let (generation, timings) = runner.generate_text_timed(&request)?;
    let wall = started.elapsed();

    let wave = audio::Waveform::new(generation.sample_rate, 2, generation.waveform.clone())?;
    std::fs::write(&output, audio::write_wav_i16(&wave))?;

    let dispatch: Vec<_> = runner
        .dispatch_profile()
        .into_iter()
        .map(|d| {
            serde_json::json!({
                "op": d.op,
                "shape": d.shape,
                "calls": d.calls,
                "total_ms": ms(d.total),
                "mean_ms": ms(d.total) / d.calls as f64,
            })
        })
        .collect();
    let audio_seconds = generation.samples as f64 / f64::from(generation.sample_rate);
    let stats = output_stats(&generation.waveform);
    let report = serde_json::json!({
        "open_ms": ms(open),
        "warmup_ms": warmup_ms,
        "generate_ms": ms(wall),
        "audio_seconds": audio_seconds,
        "rtf": wall.as_secs_f64() / audio_seconds,
        "frames": generation.frames,
        "samples": generation.samples,
        "sample_rate": generation.sample_rate,
        "resident_weight_bytes": runner.resident_weight_bytes(),
        "stage_ms": {
            "tokenize": ms(timings.tokenize),
            "prefill": ms(timings.prefill),
            "lm_head": ms(timings.lm_head),
            "sampling": ms(timings.sampling),
            "depth": ms(timings.depth),
            "lm_decode": ms(timings.lm_decode),
            "condition": ms(timings.condition),
            "dit": ms(timings.dit),
            "vocoder": ms(timings.vocoder),
            "stitch": ms(timings.stitch),
            "total": ms(timings.total),
            "chunks": timings.chunks,
        },
        "dispatch": dispatch,
        "output": {
            "all_finite": stats.all_finite,
            "peak": stats.peak,
            "rms": stats.rms,
            "clip_ratio": stats.clip_ratio,
            "silence_ratio": stats.silence_ratio,
            "dc_left": stats.dc_left,
            "dc_right": stats.dc_right,
            "channel_correlation": stats.channel_correlation,
        },
    });
    println!("{report}");
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("music3_bench requires macOS and a Metal device");
}
