//! `turbospark-clone`: zero-shot voice cloning with MegaTTS3.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use audio::tts::megatts3::{resample_reference, MegaTts3, SeededRng, SynthParams, SAMPLE_RATE};
use audio::Waveform;
use catalog::Store;

const USAGE: &str = "\
turbospark-clone: clone a voice from a short recording and speak new English text

USAGE:
    turbospark-clone speak --model MODEL --reference WAV (--text TEXT | --text-file PATH) \\
        --output WAV --consent [--steps N] [--p-w X] [--t-w X] [--seed N] \\
        [--wav-format pcm16|float32] [--metal]

--model is an installed audio alias or a converted MegaTTS3 directory (aligner_lm/,
diffusion_transformer/, duration_lm/, wavvae/, g2p/) produced by
crates/audio/tools/convert_megatts3.py.
--reference is a clean recording of the target voice, mono or multichannel at any
sample rate; at most the first 28 seconds are used. Text is English only.
--steps (default 32, 1 through 100) trades speed for quality. --p-w (default 1.4,
0.1 through 5) raises intelligibility and --t-w (default 3.0, 0.1 through 10)
raises similarity to the reference. --seed (default 0) makes a run repeatable.
--metal (macOS, experimental) runs the diffusion transformer and the WavVAE
decoder on the GPU; the default is the CPU reference path.

Voice cloning needs the community WavVAE encoder, which ByteDance chose not to
release; its origin and license are unverified. Only clone a voice you have the
right to use. --consent states that you do.
";

#[derive(Debug, Clone, PartialEq)]
struct SpeakOptions {
    model: PathBuf,
    reference: PathBuf,
    text: String,
    output: PathBuf,
    steps: usize,
    p_w: f32,
    t_w: f32,
    seed: u64,
    wav_format: WavFormat,
    metal: bool,
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

/// Upper bound on `--text-file`, in bytes.
const MAX_TEXT_BYTES: usize = 32_768;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match parse_args(&args).and_then(|options| speak(&options)) {
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

fn parse_number<T: std::str::FromStr>(flag: &str, text: &str, what: &str) -> Result<T, Error> {
    text.parse::<T>()
        .map_err(|_| Error::Usage(format!("{flag} {text:?} is not {what}")))
}

fn in_range(flag: &str, value: f32, low: f32, high: f32) -> Result<f32, Error> {
    if value.is_finite() && (low..=high).contains(&value) {
        Ok(value)
    } else {
        Err(Error::Usage(format!(
            "{flag} {value} is outside {low} through {high}"
        )))
    }
}

fn parse_args(args: &[String]) -> Result<SpeakOptions, Error> {
    if args.is_empty() || (args.len() == 1 && (args[0] == "--help" || args[0] == "-h")) {
        return Err(Error::Usage(USAGE.to_string()));
    }
    if args[0] != "speak" {
        return Err(Error::Usage(format!("unknown command {:?}", args[0])));
    }
    let (mut model, mut reference, mut text, mut text_file, mut output) =
        (None, None, None, None, None);
    let (mut steps, mut p_w, mut t_w, mut seed, mut wav_format) = (None, None, None, None, None);
    let mut consent = false;
    let mut metal = false;
    let mut index = 1usize;
    while index < args.len() {
        let flag = args[index].as_str();
        if flag == "--consent" {
            if consent {
                return Err(Error::Usage("--consent may be specified only once".into()));
            }
            consent = true;
            index += 1;
            continue;
        }
        if flag == "--metal" {
            if metal {
                return Err(Error::Usage("--metal may be specified only once".into()));
            }
            metal = true;
            index += 1;
            continue;
        }
        let value = || -> Result<&str, Error> {
            args.get(index + 1)
                .map(String::as_str)
                .ok_or_else(|| Error::Usage(format!("{flag} requires a value")))
        };
        match flag {
            "--model" => set_once(&mut model, PathBuf::from(value()?), flag)?,
            "--reference" => set_once(&mut reference, PathBuf::from(value()?), flag)?,
            "--text" => set_once(&mut text, value()?.to_string(), flag)?,
            "--text-file" => set_once(&mut text_file, PathBuf::from(value()?), flag)?,
            "--output" => set_once(&mut output, PathBuf::from(value()?), flag)?,
            "--steps" => {
                let n: usize = parse_number(flag, value()?, "a positive integer")?;
                if !(1..=100).contains(&n) {
                    return Err(Error::Usage(format!(
                        "--steps {n} is outside 1 through 100"
                    )));
                }
                set_once(&mut steps, n, flag)?;
            }
            "--p-w" => {
                let x: f32 = parse_number(flag, value()?, "a number")?;
                set_once(&mut p_w, in_range(flag, x, 0.1, 5.0)?, flag)?;
            }
            "--t-w" => {
                let x: f32 = parse_number(flag, value()?, "a number")?;
                set_once(&mut t_w, in_range(flag, x, 0.1, 10.0)?, flag)?;
            }
            "--seed" => set_once(
                &mut seed,
                parse_number::<u64>(flag, value()?, "an unsigned integer")?,
                flag,
            )?,
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
    if !consent {
        return Err(Error::Usage(
            "--consent is required: pass it to confirm you have the right to clone this voice"
                .to_string(),
        ));
    }
    if text.is_some() == text_file.is_some() {
        return Err(Error::Usage(
            "provide exactly one of --text or --text-file".to_string(),
        ));
    }
    let text = match (text, text_file) {
        (Some(t), _) => t,
        (None, Some(path)) => read_text(&path)?,
        (None, None) => unreachable!("checked above"),
    };
    if text.trim().is_empty() || text.len() > MAX_TEXT_BYTES {
        return Err(Error::Usage(format!(
            "text must be 1 through {MAX_TEXT_BYTES} bytes"
        )));
    }
    let defaults = SynthParams::default();
    Ok(SpeakOptions {
        model: required(model, "--model")?,
        reference: required(reference, "--reference")?,
        text,
        output: required(output, "--output")?,
        steps: steps.unwrap_or(defaults.steps),
        p_w: p_w.unwrap_or(defaults.p_w),
        t_w: t_w.unwrap_or(defaults.t_w),
        seed: seed.unwrap_or(0),
        wav_format: wav_format.unwrap_or(WavFormat::Pcm16),
        metal,
    })
}

fn read_text(path: &Path) -> Result<String, Error> {
    let meta = std::fs::metadata(path)
        .map_err(|e| Error::Failed(format!("reading {}: {e}", path.display())))?;
    if meta.len() as usize > MAX_TEXT_BYTES {
        return Err(Error::Usage(format!(
            "{} is larger than {MAX_TEXT_BYTES} bytes",
            path.display()
        )));
    }
    std::fs::read_to_string(path)
        .map_err(|e| Error::Failed(format!("reading {}: {e}", path.display())))
}

/// Mono 24 kHz samples from any WAV the audio crate can read.
fn load_reference(path: &Path) -> Result<Vec<f32>, Error> {
    let wave = audio::read_wav_f32(path)
        .map_err(|e| Error::Failed(format!("reading {}: {e}", path.display())))?;
    let channels = usize::from(wave.channels).max(1);
    let mono: Vec<f32> = wave
        .samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    resample_reference(&mono, wave.sample_rate)
        .map_err(|e| Error::Failed(format!("resampling the reference: {e}")))
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

/// The CPU reference path, straight through the audio crate.
fn synthesize_cpu(
    model: &Path,
    reference: &[f32],
    options: &SpeakOptions,
    params: &SynthParams,
) -> Result<Vec<f32>, Error> {
    let started = Instant::now();
    let mut tts = MegaTts3::open(model)
        .map_err(|e| Error::Failed(format!("opening {}: {e}", model.display())))?;
    let opened = started.elapsed();
    let mut rng = SeededRng::new(options.seed);
    let prompt = tts
        .prepare_prompt(reference, &mut rng)
        .map_err(|e| Error::Failed(format!("preparing the reference voice: {e}")))?;
    let prepared = started.elapsed();
    let samples = tts
        .synthesize(&prompt, &options.text, params, &mut rng)
        .map_err(|e| Error::Failed(format!("synthesis failed: {e}")))?;
    let total = started.elapsed();
    eprintln!(
        "open {:.1}s, prepare reference {:.1}s, synthesize {:.1}s",
        opened.as_secs_f64(),
        (prepared - opened).as_secs_f64(),
        (total - prepared).as_secs_f64()
    );
    Ok(samples)
}

/// The runtime's Metal path: the DiT and the WavVAE decoder on the GPU. The
/// same seed draws the same random values as the CPU path; the audio differs
/// only by the device's rounding.
#[cfg(target_os = "macos")]
fn synthesize_metal(
    model: &Path,
    reference: &[f32],
    options: &SpeakOptions,
    params: &SynthParams,
) -> Result<Vec<f32>, Error> {
    use runtime::megatts3::{CloneRequest, Device, MegaTts3Runner};
    let started = Instant::now();
    let mut runner = MegaTts3Runner::open(model, Device::Metal)
        .map_err(|e| Error::Failed(format!("opening {}: {e}", model.display())))?;
    let opened = started.elapsed();
    let request = CloneRequest {
        text: &options.text,
        reference,
        reference_rate: SAMPLE_RATE,
        seed: options.seed,
        params: params.clone(),
    };
    let mut samples = Vec::new();
    runner
        .synthesize_controlled(&request, &mut |_| Ok(()), &mut |pcm| {
            samples.extend(pcm);
            true
        })
        .map_err(|e| Error::Failed(format!("synthesis failed: {e}")))?;
    eprintln!(
        "open {:.1}s, prepare reference and synthesize {:.1}s (Metal)",
        opened.as_secs_f64(),
        (started.elapsed() - opened).as_secs_f64()
    );
    Ok(samples)
}

#[cfg(not(target_os = "macos"))]
fn synthesize_metal(
    _model: &Path,
    _reference: &[f32],
    _options: &SpeakOptions,
    _params: &SynthParams,
) -> Result<Vec<f32>, Error> {
    Err(Error::Usage("--metal needs macOS".to_string()))
}

fn speak(options: &SpeakOptions) -> Result<(), Error> {
    let reference = load_reference(&options.reference)?;
    let model = resolve_model(&options.model)?;
    let params = SynthParams {
        steps: options.steps,
        p_w: options.p_w,
        t_w: options.t_w,
        ..SynthParams::default()
    };
    let samples = if options.metal {
        synthesize_metal(&model, &reference, options, &params)?
    } else {
        synthesize_cpu(&model, &reference, options, &params)?
    };
    let wave = Waveform::mono(SAMPLE_RATE, samples)
        .map_err(|e| Error::Failed(format!("assembling the output: {e}")))?;
    let encoded = match options.wav_format {
        WavFormat::Pcm16 => audio::write_wav_i16(&wave),
        WavFormat::Float32 => audio::write_wav_f32(&wave),
    };
    std::fs::write(&options.output, encoded)
        .map_err(|e| Error::Failed(format!("writing {}: {e}", options.output.display())))?;
    println!(
        "wrote {:.2} s of audio ({} samples at {} Hz, mono) to {}",
        wave.duration_seconds(),
        wave.frame_count(),
        SAMPLE_RATE,
        options.output.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(extra: &[&str]) -> Vec<String> {
        [
            "speak",
            "--model",
            "m",
            "--reference",
            "r.wav",
            "--output",
            "o.wav",
            "--consent",
        ]
        .iter()
        .chain(extra)
        .map(|s| s.to_string())
        .collect()
    }

    fn with_text(extra: &[&str]) -> Vec<String> {
        let mut a = args(extra);
        a.extend(["--text".to_string(), "hello there".to_string()]);
        a
    }

    #[test]
    fn parses_defaults_and_overrides() {
        let o = parse_args(&with_text(&[])).unwrap();
        assert_eq!((o.steps, o.p_w, o.t_w, o.seed), (32, 1.4, 3.0, 0));
        assert_eq!(o.wav_format, WavFormat::Pcm16);
        assert_eq!(o.text, "hello there");
        let o = parse_args(&with_text(&[
            "--steps",
            "10",
            "--p-w",
            "2.5",
            "--t-w",
            "4",
            "--seed",
            "9",
            "--wav-format",
            "float32",
        ]))
        .unwrap();
        assert_eq!((o.steps, o.p_w, o.t_w, o.seed), (10, 2.5, 4.0, 9));
        assert_eq!(o.wav_format, WavFormat::Float32);
    }

    #[test]
    fn consent_is_mandatory() {
        let mut a = with_text(&[]);
        a.retain(|s| s != "--consent");
        let err = parse_args(&a).unwrap_err();
        assert!(
            matches!(&err, Error::Usage(m) if m.contains("--consent")),
            "{err:?}"
        );
    }

    #[test]
    fn metal_is_an_opt_in_switch() {
        assert!(
            !parse_args(&with_text(&[])).unwrap().metal,
            "CPU is the default"
        );
        assert!(parse_args(&with_text(&["--metal"])).unwrap().metal);
    }

    #[test]
    fn rejects_bad_arguments() {
        for bad in [
            vec!["--steps", "0"],
            vec!["--steps", "101"],
            vec!["--steps", "x"],
            vec!["--p-w", "0"],
            vec!["--p-w", "nan"],
            vec!["--t-w", "11"],
            vec!["--seed", "-1"],
            vec!["--wav-format", "mp3"],
            vec!["--steps", "5", "--steps", "6"],
            vec!["--metal", "--metal"],
            vec!["--bogus", "1"],
            vec!["stray"],
        ] {
            assert!(parse_args(&with_text(&bad)).is_err(), "{bad:?} accepted");
        }
        // Missing required flags and the text/text-file exclusivity.
        assert!(parse_args(&args(&[])).is_err(), "no text");
        let mut both = with_text(&[]);
        both.extend(["--text-file".to_string(), "t.txt".to_string()]);
        assert!(parse_args(&both).is_err(), "text and text-file");
        let mut blank = args(&[]);
        blank.extend(["--text".to_string(), "   ".to_string()]);
        assert!(parse_args(&blank).is_err(), "blank text");
        assert!(parse_args(&[]).is_err());
        assert!(parse_args(&["listen".to_string()]).is_err());
    }

    #[test]
    fn reads_text_files_within_the_limit() {
        let dir = std::env::temp_dir().join(format!("clone-text-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ok = dir.join("ok.txt");
        std::fs::write(&ok, "from a file").unwrap();
        let mut a = args(&[]);
        a.extend(["--text-file".to_string(), ok.to_string_lossy().into_owned()]);
        assert_eq!(parse_args(&a).unwrap().text, "from a file");
        let big = dir.join("big.txt");
        std::fs::write(&big, vec![b'a'; MAX_TEXT_BYTES + 1]).unwrap();
        let mut b = args(&[]);
        b.extend([
            "--text-file".to_string(),
            big.to_string_lossy().into_owned(),
        ]);
        assert!(parse_args(&b).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn missing_model_directory_fails_cleanly() {
        let reference = std::env::temp_dir().join(format!("clone-ref-{}.wav", std::process::id()));
        let wave = Waveform::mono(24_000, vec![0.1; 4_800]).unwrap();
        std::fs::write(&reference, audio::write_wav_f32(&wave)).unwrap();
        let options = SpeakOptions {
            model: PathBuf::from("/nonexistent/megatts3"),
            reference: reference.clone(),
            text: "hello".to_string(),
            output: std::env::temp_dir().join("clone-never-written.wav"),
            steps: 1,
            p_w: 1.4,
            t_w: 3.0,
            seed: 0,
            wav_format: WavFormat::Pcm16,
            metal: false,
        };
        // Neither a directory nor an installed alias: refused before loading.
        let err = speak(&options).unwrap_err();
        assert!(
            matches!(&err, Error::Failed(m) if m.contains("/nonexistent/megatts3")),
            "{err:?}"
        );
        assert!(!options.output.exists());
        let _ = std::fs::remove_file(reference);
    }

    #[test]
    fn reference_loading_downmixes_and_resamples() {
        let dir = std::env::temp_dir().join(format!("clone-load-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Stereo 48 kHz: left 0.5, right -0.1 -> mono 0.2 at 24 kHz, half as many frames.
        let frames = 4_800usize;
        let samples: Vec<f32> = (0..frames).flat_map(|_| [0.5f32, -0.1]).collect();
        let path = dir.join("stereo48k.wav");
        std::fs::write(
            &path,
            audio::write_wav_f32(&Waveform::new(48_000, 2, samples).unwrap()),
        )
        .unwrap();
        let mono = load_reference(&path).unwrap();
        assert!(
            (mono.len() as i64 - 2_400).abs() <= 1,
            "{} samples",
            mono.len()
        );
        let mid = &mono[mono.len() / 4..3 * mono.len() / 4];
        assert!(
            mid.iter().all(|v| (v - 0.2).abs() < 1e-3),
            "mid-clip not 0.2"
        );
        let same = dir.join("mono24k.wav");
        std::fs::write(
            &same,
            audio::write_wav_f32(&Waveform::mono(24_000, vec![0.25; 100]).unwrap()),
        )
        .unwrap();
        assert_eq!(load_reference(&same).unwrap(), vec![0.25; 100]);
        assert!(load_reference(&dir.join("absent.wav")).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }
}
