//! Compare the portable CPU and opt-in Metal Qwen3-ASR runners on a WAV clip.
//! The Metal decoder never dispatches by default, so the backend is an
//! explicit argument rather than an environment switch.

use std::time::Instant;

enum Runner {
    Cpu(audio::stt::qwen3_asr::Qwen3Asr),
    Metal(turbospark_runtime::qwen3_asr_metal::Qwen3AsrMetalEngine),
}

impl Runner {
    fn open(
        backend: &str,
        model_dir: &std::path::Path,
    ) -> Result<(Self, f64), Box<dyn std::error::Error>> {
        let opened = Instant::now();
        let runner = match backend {
            "cpu" => Self::Cpu(audio::stt::qwen3_asr::Qwen3Asr::load(model_dir)?),
            "metal" => Self::Metal(
                turbospark_runtime::qwen3_asr_metal::Qwen3AsrMetalEngine::open(model_dir)?,
            ),
            other => return Err(format!("unknown backend {other}: expected cpu or metal").into()),
        };
        Ok((runner, opened.elapsed().as_secs_f64() * 1_000.0))
    }

    fn transcribe(&mut self, samples: &[f32]) -> Result<String, Box<dyn std::error::Error>> {
        match self {
            Self::Cpu(model) => Ok(model.transcribe(samples)?),
            Self::Metal(engine) => Ok(engine.transcribe(samples)?),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = std::path::PathBuf::from(args.next().expect("model dir"));
    let wav = args.next().expect("wav path");
    let backend = args.next().expect("backend: cpu or metal");
    let mode = args.next();
    if !matches!(backend.as_str(), "cpu" | "metal")
        || !matches!(mode.as_deref(), None | Some("--warmup"))
        || args.next().is_some()
    {
        return Err("expected <model-dir> <wav> <cpu|metal> [--warmup]".into());
    }
    let (mut runner, open_ms) = Runner::open(&backend, &model_dir)?;
    let input = audio::read_wav_f32(std::path::Path::new(&wav))?;
    let samples = audio::to_mono_resampled(
        &input,
        16_000,
        &audio::MonoResampleStrategy::SincHann(Default::default()),
    )?;
    let warmup = if mode.as_deref() == Some("--warmup") {
        let started = Instant::now();
        let text = runner.transcribe(&samples)?;
        Some((text, started.elapsed().as_secs_f64() * 1_000.0))
    } else {
        None
    };
    let started = Instant::now();
    let text = runner.transcribe(&samples)?;
    let transcribe_ms = started.elapsed().as_secs_f64() * 1_000.0;
    if let Some((warmup_text, _)) = &warmup {
        if warmup_text != &text {
            return Err("Qwen3-ASR warmup and measured transcripts differ".into());
        }
    }
    println!("TEXT: {text}");
    eprintln!(
        "qwen3_asr_{backend} open_ms={open_ms:.2} warmup_ms={:.2} \
         transcribe_ms={transcribe_ms:.2} samples={}",
        warmup.as_ref().map_or(0.0, |(_, elapsed)| *elapsed),
        samples.len()
    );
    Ok(())
}
