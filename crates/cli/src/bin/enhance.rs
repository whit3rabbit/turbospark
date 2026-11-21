//! `turbospark-enhance`: speech enhancement (noise suppression) of a WAV file.
//!
//! The runtime owns family detection, resampling and bounds
//! (`runtime::native_audio`, task `enhancement`); this binary reads the WAV,
//! mixes it to mono, runs one job and writes the 48 kHz voice, and optionally
//! the removed background (input minus voice at 48 kHz).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use audio::Waveform;
use catalog::Store;
use runtime::native_audio::{
    enhancement_pcm, validate_request, AudioCancel, AudioEvent, AudioRequest, AudioSession,
    AudioTask,
};

const USAGE: &str = "\
turbospark-enhance: clean speech in a recording (noise suppression)

USAGE:
    turbospark-enhance --model MODEL --input WAV --output WAV \\
        [--background WAV] [--wav-format pcm16|float32]

--model is an installed audio alias (mossformer2-se-48k-fp16) or a model
directory. --input is any WAV, mono or multichannel at any rate from 8 kHz to
192 kHz; it is mixed to mono. --output gets the cleaned voice as mono 48 kHz.
--background also writes what was removed (input resampled to 48 kHz, minus
the voice). At most 661 seconds of input. CPU only for now.
";

#[derive(Debug, Clone, PartialEq)]
struct EnhanceOptions {
    model: PathBuf,
    input: PathBuf,
    output: PathBuf,
    background: Option<PathBuf>,
    wav_format: WavFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WavFormat {
    Pcm16,
    Float32,
}

#[derive(Debug)]
enum Error {
    Usage(String),
    Failed(String),
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match parse_args(&args).and_then(|options| enhance(&options)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(Error::Usage(message)) => {
            eprintln!("{message}\n\n{USAGE}");
            ExitCode::from(2)
        }
        Err(Error::Failed(message)) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn set_once<T>(slot: &mut Option<T>, value: T, flag: &str) -> Result<(), Error> {
    if slot.is_some() {
        return Err(Error::Usage(format!("{flag} may be specified only once")));
    }
    *slot = Some(value);
    Ok(())
}

fn required<T>(slot: Option<T>, flag: &str) -> Result<T, Error> {
    slot.ok_or_else(|| Error::Usage(format!("{flag} is required")))
}

fn parse_args(args: &[String]) -> Result<EnhanceOptions, Error> {
    if args.is_empty() || (args.len() == 1 && (args[0] == "--help" || args[0] == "-h")) {
        return Err(Error::Usage(USAGE.to_string()));
    }
    let (mut model, mut input, mut output, mut background, mut wav_format) =
        (None, None, None, None, None);
    let mut index = 0usize;
    while index < args.len() {
        let flag = args[index].as_str();
        let value = || -> Result<&str, Error> {
            args.get(index + 1)
                .map(String::as_str)
                .ok_or_else(|| Error::Usage(format!("{flag} requires a value")))
        };
        match flag {
            "--model" => set_once(&mut model, PathBuf::from(value()?), flag)?,
            "--input" => set_once(&mut input, PathBuf::from(value()?), flag)?,
            "--output" => set_once(&mut output, PathBuf::from(value()?), flag)?,
            "--background" => set_once(&mut background, PathBuf::from(value()?), flag)?,
            "--wav-format" => {
                let selected = match value()? {
                    "pcm16" => WavFormat::Pcm16,
                    "float32" => WavFormat::Float32,
                    other => {
                        return Err(Error::Usage(format!(
                            "unknown WAV format {other:?}; expected pcm16 or float32"
                        )))
                    }
                };
                set_once(&mut wav_format, selected, flag)?;
            }
            "--help" | "-h" => return Err(Error::Usage(USAGE.to_string())),
            other if other.starts_with('-') => {
                return Err(Error::Usage(format!("unknown option {other:?}")))
            }
            other => return Err(Error::Usage(format!("unexpected argument {other:?}"))),
        }
        index += 2;
    }
    let options = EnhanceOptions {
        model: required(model, "--model")?,
        input: required(input, "--input")?,
        output: required(output, "--output")?,
        background,
        wav_format: wav_format.unwrap_or(WavFormat::Pcm16),
    };
    // Writing over the input would lose the only copy of the recording.
    if options.output == options.input || options.background.as_ref() == Some(&options.input) {
        return Err(Error::Usage("output paths must differ from --input".into()));
    }
    if options.background.as_ref() == Some(&options.output) {
        return Err(Error::Usage(
            "--background must differ from --output".into(),
        ));
    }
    Ok(options)
}

/// Mono samples at the file's own rate; the runtime resamples.
fn load_input(path: &Path) -> Result<(Vec<f32>, u32), Error> {
    let wave = audio::read_wav_f32(path)
        .map_err(|e| Error::Failed(format!("reading {}: {e}", path.display())))?;
    let channels = usize::from(wave.channels).max(1);
    let mono = wave
        .samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    Ok((mono, wave.sample_rate))
}

/// A directory as given, else an installed audio alias.
fn resolve_model(model: &Path) -> Result<PathBuf, Error> {
    if model.is_dir() {
        return Ok(model.to_path_buf());
    }
    let name = model.to_string_lossy();
    let store = Store::default_store().map_err(Error::Failed)?;
    store.resolve_audio(&name).ok_or_else(|| {
        Error::Failed(format!(
            "{name:?} is neither a directory nor an installed audio alias"
        ))
    })
}

fn request(rate: u32) -> AudioRequest {
    AudioRequest {
        task: AudioTask::Enhancement,
        input_sample_rate: Some(rate),
        ..AudioRequest::default()
    }
}

/// The removed part at the voice's rate: the input exactly as the model saw
/// it, minus the voice.
fn background(request: &AudioRequest, input: &[f32], voice: &[f32]) -> Result<Vec<f32>, Error> {
    let at_rate = enhancement_pcm(request, input).map_err(Error::Failed)?;
    if at_rate.len() != voice.len() {
        return Err(Error::Failed(format!(
            "voice has {} samples but the resampled input {}",
            voice.len(),
            at_rate.len()
        )));
    }
    Ok(at_rate.iter().zip(voice).map(|(x, v)| x - v).collect())
}

fn write(path: &Path, rate: u32, samples: Vec<f32>, format: WavFormat) -> Result<(), Error> {
    let wave = Waveform::new(rate, 1, samples)
        .map_err(|e| Error::Failed(format!("assembling the output: {e}")))?;
    let bytes = match format {
        WavFormat::Pcm16 => audio::write_wav_i16(&wave),
        WavFormat::Float32 => audio::write_wav_f32(&wave),
    };
    std::fs::write(path, bytes)
        .map_err(|e| Error::Failed(format!("writing {}: {e}", path.display())))
}

fn enhance(options: &EnhanceOptions) -> Result<(), Error> {
    let model = resolve_model(&options.model)?;
    let (input, rate) = load_input(&options.input)?;
    let request = request(rate);

    let started = Instant::now();
    let session = AudioSession::open(model.clone(), AudioTask::Enhancement, false)
        .map_err(|e| Error::Failed(format!("opening {}: {e}", model.display())))?;
    // Bounds are the runtime's, checked before the job queues.
    validate_request(session.family(), &request, &input).map_err(Error::Usage)?;
    let opened = started.elapsed();
    let mut voice = Vec::new();
    let result = session
        .execute(
            request.clone(),
            input.clone(),
            AudioCancel::default(),
            |event| match event {
                AudioEvent::Pcm(chunk) => voice.extend_from_slice(&chunk),
                AudioEvent::Progress(p) if p.stage == "enhance" => {
                    if let Some(total) = p.total {
                        eprintln!("enhance {}/{total}", p.completed);
                    }
                }
                AudioEvent::Progress(_) => {}
            },
        )
        .map_err(|e| Error::Failed(format!("enhancement failed: {e}")))?;
    let format = result
        .pcm_format
        .ok_or_else(|| Error::Failed("the engine reported no PCM format".to_string()))?;
    if format.channels != 1 {
        return Err(Error::Failed(format!(
            "{} output channels",
            format.channels
        )));
    }
    if let Some(path) = &options.background {
        let removed = background(&request, &input, &voice)?;
        write(path, format.sample_rate, removed, options.wav_format)?;
    }
    let seconds = voice.len() as f64 / f64::from(format.sample_rate);
    write(
        &options.output,
        format.sample_rate,
        voice,
        options.wav_format,
    )?;
    eprintln!(
        "open {:.1}s, enhance {:.1}s",
        opened.as_secs_f64(),
        (started.elapsed() - opened).as_secs_f64()
    );
    println!(
        "wrote {seconds:.2} s of mono {} Hz voice ({}) to {}",
        format.sample_rate,
        session.family(),
        options.output.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    fn usage(result: Result<EnhanceOptions, Error>) -> String {
        match result {
            Err(Error::Usage(m)) => m,
            other => panic!("expected a usage error, got {other:?}"),
        }
    }

    #[test]
    fn parses_required_and_optional_flags() {
        let parsed = parse_args(&args(&[
            "--model",
            "m",
            "--input",
            "in.wav",
            "--output",
            "out.wav",
            "--background",
            "bg.wav",
            "--wav-format",
            "float32",
        ]))
        .unwrap();
        assert_eq!(
            parsed,
            EnhanceOptions {
                model: "m".into(),
                input: "in.wav".into(),
                output: "out.wav".into(),
                background: Some("bg.wav".into()),
                wav_format: WavFormat::Float32,
            }
        );
        let plain = parse_args(&args(&[
            "--model", "m", "--input", "a.wav", "--output", "b.wav",
        ]))
        .unwrap();
        assert_eq!(
            (plain.background, plain.wav_format),
            (None, WavFormat::Pcm16)
        );
    }

    #[test]
    fn refuses_missing_repeated_unknown_and_overwriting_flags() {
        assert!(
            usage(parse_args(&args(&["--model", "m", "--input", "a.wav"])))
                .contains("--output is required")
        );
        assert!(usage(parse_args(&args(&[
            "--model", "m", "--model", "n", "--input", "a.wav", "--output", "b.wav"
        ])))
        .contains("only once"));
        assert!(usage(parse_args(&args(&[
            "--model", "m", "--input", "a.wav", "--output", "b.wav", "--speed", "2"
        ])))
        .contains("unknown option"));
        assert!(usage(parse_args(&args(&[
            "--model", "m", "--input", "a.wav", "--output", "a.wav"
        ])))
        .contains("differ from --input"));
        assert!(usage(parse_args(&args(&[
            "--model",
            "m",
            "--input",
            "a.wav",
            "--output",
            "b.wav",
            "--background",
            "b.wav"
        ])))
        .contains("differ from --output"));
        assert!(usage(parse_args(&args(&["--help"]))).contains("USAGE"));
    }

    #[test]
    fn background_is_resampled_input_minus_voice() {
        let input: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.01).sin()).collect();
        let req = request(16_000);
        let at_rate = enhancement_pcm(&req, &input).unwrap();
        assert_eq!(at_rate.len(), 4800);
        let voice: Vec<f32> = at_rate.iter().map(|v| v * 0.25).collect();
        let removed = background(&req, &input, &voice).unwrap();
        for ((r, x), v) in removed.iter().zip(&at_rate).zip(&voice) {
            assert_eq!(*r, x - v);
        }
        assert!(background(&req, &input, &voice[1..]).is_err());
    }
}
