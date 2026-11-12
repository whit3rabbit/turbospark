//! `turbospark-speak`: speech with Fish Audio Drama 3 (44.1 kHz), optionally in
//! a cloned voice.
//!
//! The model family comes from the directory's `config.json`; anything but
//! `fish_drama3` is refused for now. `turbospark-clone` stays MegaTTS3-only.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use audio::Waveform;
use catalog::Store;
use runtime::native_audio::{
    tts_model_type, validate_request, AudioCancel, AudioEvent, AudioRequest, AudioSession,
    AudioTask, FISH_DRAMA3_FAMILY,
};

const USAGE: &str = "\
turbospark-speak: speak text with Fish Audio Drama 3

USAGE:
    turbospark-speak --model MODEL (--text TEXT | --text-file PATH) --output WAV \\
        [--reference WAV --reference-text TEXT --consent] [--seed N] \\
        [--temperature X] [--top-p X] [--top-k N] [--max-frames N] \\
        [--wav-format pcm16|float32]

--model is an installed audio alias or a Drama 3 install directory (config.json,
tokenizer.json, codec.safetensors, tts_side.safetensors, trunk/), made by
`turbospark-model pull-drama3`. Output is mono 44.1 kHz. Metal only.
Drama 3 takes delivery cues inline in the text, such as [whispers]. There is no
speed control. Without --reference the model picks the voice.
--reference is a recording of the voice to clone, 1 through 30 seconds, mono or
multichannel at any sample rate. It needs --reference-text, the exact words
spoken in it, and --consent, which states you have the right to use that voice.
--seed N makes a run repeatable. Without it each run draws its own seed and
prints it, so running again gives a new take. --temperature (default 1.0, finite
and not negative), --top-p (default 0.9, above 0 through 1), --top-k (default 30,
at least 1) and --max-frames (default and most 1024; a frame is 46.4 ms) override
Fish Audio's sampler recipe.
";

/// Upper bound on `--text-file`, in bytes.
const MAX_TEXT_BYTES: usize = 32_768;

#[derive(Debug, Clone, PartialEq)]
struct SpeakOptions {
    model: PathBuf,
    text: String,
    output: PathBuf,
    reference: Option<PathBuf>,
    reference_text: Option<String>,
    consent: bool,
    /// None draws a fresh seed per run (`fresh_seed`).
    seed: Option<u64>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    max_frames: Option<usize>,
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

fn parse_args(args: &[String]) -> Result<SpeakOptions, Error> {
    if args.is_empty() || (args.len() == 1 && (args[0] == "--help" || args[0] == "-h")) {
        return Err(Error::Usage(USAGE.to_string()));
    }
    let (mut model, mut text, mut text_file, mut output) = (None, None, None, None);
    let (mut reference, mut reference_text) = (None, None);
    let (mut seed, mut temperature, mut top_p, mut top_k) = (None, None, None, None);
    let (mut max_frames, mut wav_format) = (None, None);
    let mut consent = false;
    let mut index = 0usize;
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
        let value = || -> Result<&str, Error> {
            args.get(index + 1)
                .map(String::as_str)
                .ok_or_else(|| Error::Usage(format!("{flag} requires a value")))
        };
        match flag {
            "--model" => set_once(&mut model, PathBuf::from(value()?), flag)?,
            "--text" => set_once(&mut text, value()?.to_string(), flag)?,
            "--text-file" => set_once(&mut text_file, PathBuf::from(value()?), flag)?,
            "--output" => set_once(&mut output, PathBuf::from(value()?), flag)?,
            "--reference" => set_once(&mut reference, PathBuf::from(value()?), flag)?,
            "--reference-text" => set_once(&mut reference_text, value()?.to_string(), flag)?,
            "--seed" => set_once(
                &mut seed,
                parse_number::<u64>(flag, value()?, "an unsigned integer")?,
                flag,
            )?,
            // Range checks are the runtime's (`validate_request`), so the CLI
            // cannot drift from what the engine accepts.
            "--temperature" => set_once(
                &mut temperature,
                parse_number::<f32>(flag, value()?, "a number")?,
                flag,
            )?,
            "--top-p" => set_once(
                &mut top_p,
                parse_number::<f32>(flag, value()?, "a number")?,
                flag,
            )?,
            "--top-k" => set_once(
                &mut top_k,
                parse_number::<usize>(flag, value()?, "a positive integer")?,
                flag,
            )?,
            "--max-frames" => set_once(
                &mut max_frames,
                parse_number::<usize>(flag, value()?, "a positive integer")?,
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
    // A reference is a clone: it needs the transcript and the consent, and
    // neither means anything without it.
    match (&reference, &reference_text) {
        (Some(_), None) => {
            return Err(Error::Usage(
                "--reference needs --reference-text, the exact words spoken in the recording"
                    .to_string(),
            ))
        }
        (None, Some(_)) => {
            return Err(Error::Usage(
                "--reference-text needs --reference".to_string(),
            ))
        }
        _ => {}
    }
    if reference.is_some() && !consent {
        return Err(Error::Usage(
            "--consent is required with --reference: pass it to confirm you have the right to clone this voice"
                .to_string(),
        ));
    }
    if reference.is_none() && consent {
        return Err(Error::Usage("--consent needs --reference".to_string()));
    }
    Ok(SpeakOptions {
        model: required(model, "--model")?,
        text,
        output: required(output, "--output")?,
        reference,
        reference_text,
        consent,
        seed,
        temperature,
        top_p,
        top_k,
        max_frames,
        wav_format: wav_format.unwrap_or(WavFormat::Pcm16),
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

/// Mono samples at the file's own rate. The runner resamples a reference to
/// 44.1 kHz itself, so this does not.
fn load_reference(path: &Path) -> Result<(Vec<f32>, u32), Error> {
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

/// Only Drama 3 for now. Read from `config.json`, never from tensor names.
fn require_drama3(model: &Path) -> Result<(), Error> {
    match tts_model_type(model).as_deref() {
        Some(FISH_DRAMA3_FAMILY) => Ok(()),
        Some(other) => Err(Error::Failed(format!(
            "{} is a {other:?} model; turbospark-speak speaks only {FISH_DRAMA3_FAMILY:?} for now \
             (turbospark-clone covers MegaTTS3)",
            model.display()
        ))),
        None => Err(Error::Failed(format!(
            "{} has no config.json model_type; expected a {FISH_DRAMA3_FAMILY:?} install",
            model.display()
        ))),
    }
}

/// A seed for a run that names none. The runtime takes every seed
/// literally, so a fixed default repeated one take on every run, and some
/// Drama 3 seeds go silent after the first sentence.
fn fresh_seed() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    // Two seeds drawn in one process always differ, whatever the clock.
    static DRAWN: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    nanos ^ (u64::from(std::process::id()) << 32) ^ DRAWN.fetch_add(1, Ordering::Relaxed)
}

/// The request and the reference PCM it travels with, at `seed`.
fn build_request(
    options: &SpeakOptions,
    seed: u64,
    reference: Option<(Vec<f32>, u32)>,
) -> (AudioRequest, Vec<f32>) {
    let (samples, rate) = match reference {
        Some((samples, rate)) => (samples, Some(rate)),
        None => (Vec::new(), None),
    };
    let request = AudioRequest {
        task: AudioTask::TextToSpeech,
        text: options.text.clone(),
        seed: Some(seed),
        reference_text: options.reference_text.clone(),
        reference_sample_rate: rate,
        consent: options.consent,
        temperature: options.temperature,
        top_p: options.top_p,
        top_k: options.top_k,
        max_frames: options.max_frames,
        ..AudioRequest::default()
    };
    (request, samples)
}

/// Loads the reference and builds the request, refusing the caller's mistakes
/// (a range, a missing transcript, a clip outside 1 through 30 s) before the
/// model loads.
fn prepare_request(options: &SpeakOptions) -> Result<(AudioRequest, Vec<f32>), Error> {
    let reference = options
        .reference
        .as_deref()
        .map(load_reference)
        .transpose()?;
    let seed = options.seed.unwrap_or_else(fresh_seed);
    let (request, samples) = build_request(options, seed, reference);
    validate_request(FISH_DRAMA3_FAMILY, &request, &samples).map_err(Error::Usage)?;
    Ok((request, samples))
}

fn encode(wave: &Waveform, format: WavFormat) -> Vec<u8> {
    match format {
        WavFormat::Pcm16 => audio::write_wav_i16(wave),
        WavFormat::Float32 => audio::write_wav_f32(wave),
    }
}

fn speak(options: &SpeakOptions) -> Result<(), Error> {
    let model = resolve_model(&options.model)?;
    require_drama3(&model)?;
    let (request, samples) = prepare_request(options)?;

    let started = Instant::now();
    let session = AudioSession::open(model.clone(), AudioTask::TextToSpeech, false)
        .map_err(|e| Error::Failed(format!("opening {}: {e}", model.display())))?;
    let opened = started.elapsed();
    let mut pcm = Vec::new();
    let mut reported = 0usize;
    let result = session
        .execute(
            request,
            samples,
            AudioCancel::default(),
            |event| match event {
                AudioEvent::Pcm(chunk) => pcm.extend_from_slice(&chunk),
                AudioEvent::Progress(progress) => {
                    if progress.completed >= reported + 25 {
                        reported = progress.completed;
                        eprintln!("{} {}", progress.stage, progress.completed);
                    }
                }
            },
        )
        .map_err(|e| Error::Failed(format!("synthesis failed: {e}")))?;
    let format = result
        .pcm_format
        .ok_or_else(|| Error::Failed("the engine reported no PCM format".to_string()))?;
    let channels = u16::try_from(format.channels)
        .map_err(|_| Error::Failed(format!("{} output channels", format.channels)))?;
    let wave = Waveform::new(format.sample_rate, channels, pcm)
        .map_err(|e| Error::Failed(format!("assembling the output: {e}")))?;
    std::fs::write(&options.output, encode(&wave, options.wav_format))
        .map_err(|e| Error::Failed(format!("writing {}: {e}", options.output.display())))?;
    eprintln!(
        "open {:.1}s, synthesize {:.1}s",
        opened.as_secs_f64(),
        (started.elapsed() - opened).as_secs_f64()
    );
    println!(
        "wrote {:.2} s of audio ({} samples at {} Hz, {} channel{}) to {}, seed {}",
        wave.duration_seconds(),
        wave.frame_count(),
        format.sample_rate,
        channels,
        if channels == 1 { "" } else { "s" },
        options.output.display(),
        result
            .seed
            .map_or_else(|| "unknown".to_string(), |s| s.to_string())
    );
    if result.stopped_early {
        eprintln!(
            "warning: the take likely stopped speaking before the end of the text; run again \
             without --seed for a new take"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Vec<String> {
        ["--model", "m", "--output", "o.wav", "--text", "hello there"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn with(extra: &[&str]) -> Vec<String> {
        let mut a = base();
        a.extend(extra.iter().map(|s| s.to_string()));
        a
    }

    fn clone_args(extra: &[&str]) -> Vec<String> {
        with(
            &[
                &["--reference", "r.wav", "--reference-text", "the words"][..],
                extra,
            ]
            .concat(),
        )
    }

    fn usage_message(result: Result<SpeakOptions, Error>) -> String {
        match result {
            Err(Error::Usage(m)) => m,
            other => panic!("expected a usage error, got {other:?}"),
        }
    }

    fn tempdir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("speak-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn parses_defaults_and_every_override() {
        let o = parse_args(&base()).unwrap();
        assert_eq!(o.text, "hello there");
        assert_eq!(o.seed, None, "no --seed draws a fresh one per run");
        assert_eq!(o.wav_format, WavFormat::Pcm16);
        assert!(o.reference.is_none() && !o.consent);
        assert_eq!(
            (o.temperature, o.top_p, o.top_k, o.max_frames),
            (None, None, None, None)
        );
        let o = parse_args(&clone_args(&[
            "--consent",
            "--seed",
            "9",
            "--temperature",
            "0.5",
            "--top-p",
            "0.9",
            "--top-k",
            "40",
            "--max-frames",
            "200",
            "--wav-format",
            "float32",
        ]))
        .unwrap();
        assert_eq!(o.seed, Some(9));
        assert_eq!(o.temperature, Some(0.5));
        assert_eq!(o.top_p, Some(0.9));
        assert_eq!(o.top_k, Some(40));
        assert_eq!(o.max_frames, Some(200));
        assert_eq!(o.wav_format, WavFormat::Float32);
        assert_eq!(o.reference_text.as_deref(), Some("the words"));
        assert!(o.consent);
    }

    #[test]
    fn a_reference_needs_consent() {
        let m = usage_message(parse_args(&clone_args(&[])));
        assert!(m.contains("--consent"), "{m}");
        assert!(parse_args(&clone_args(&["--consent"])).is_ok());
    }

    #[test]
    fn a_reference_needs_its_transcript() {
        let a = with(&["--reference", "r.wav", "--consent"]);
        let m = usage_message(parse_args(&a));
        assert!(m.contains("--reference-text"), "{m}");
    }

    #[test]
    fn a_transcript_or_consent_without_a_reference_is_refused() {
        let m = usage_message(parse_args(&with(&["--reference-text", "words"])));
        assert!(m.contains("needs --reference"), "{m}");
        let m = usage_message(parse_args(&with(&["--consent"])));
        assert!(m.contains("needs --reference"), "{m}");
    }

    #[test]
    fn rejects_bad_arguments() {
        for bad in [
            vec!["--seed", "-1"],
            vec!["--temperature", "hot"],
            vec!["--top-p", "x"],
            vec!["--top-k", "-3"],
            vec!["--max-frames", "1.5"],
            vec!["--wav-format", "mp3"],
            vec!["--seed", "1", "--seed", "2"],
            vec!["--temperature", "0.5", "--temperature", "0.6"],
            vec!["--bogus", "1"],
            vec!["stray"],
            vec!["--top-k"],
        ] {
            assert!(parse_args(&with(&bad)).is_err(), "{bad:?} accepted");
        }
        // Text rules: exactly one source, not blank.
        let none: Vec<String> = ["--model", "m", "--output", "o"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(parse_args(&none).is_err(), "no text");
        let mut both = base();
        both.extend(["--text-file".to_string(), "t.txt".to_string()]);
        assert!(parse_args(&both).is_err(), "text and text-file");
        let mut blank = none.clone();
        blank.extend(["--text".to_string(), "   ".to_string()]);
        assert!(parse_args(&blank).is_err(), "blank text");
        let mut no_model = base();
        no_model.drain(0..2);
        assert!(parse_args(&no_model).is_err(), "no model");
        assert!(parse_args(&[]).is_err());
    }

    #[test]
    fn reads_text_files_within_the_limit() {
        let dir = tempdir("text");
        let ok = dir.join("ok.txt");
        std::fs::write(&ok, "from a file").unwrap();
        let a: Vec<String> = [
            "--model",
            "m",
            "--output",
            "o",
            "--text-file",
            ok.to_str().unwrap(),
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(parse_args(&a).unwrap().text, "from a file");
        let big = dir.join("big.txt");
        std::fs::write(&big, vec![b'a'; MAX_TEXT_BYTES + 1]).unwrap();
        let mut b = a.clone();
        *b.last_mut().unwrap() = big.to_string_lossy().into_owned();
        assert!(parse_args(&b).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn requests_carry_exactly_what_was_asked() {
        let o = parse_args(&clone_args(&["--consent", "--seed", "3", "--top-k", "12"])).unwrap();
        let (request, samples) = build_request(&o, 3, Some((vec![0.1; 44_100 * 2], 44_100)));
        assert_eq!(samples.len(), 88_200);
        assert_eq!(request.task, AudioTask::TextToSpeech);
        assert_eq!(request.seed, Some(3));
        assert_eq!(request.top_k, Some(12));
        assert_eq!(request.temperature, None);
        assert_eq!(request.reference_sample_rate, Some(44_100));
        assert_eq!(request.reference_text.as_deref(), Some("the words"));
        assert!(request.consent);
        assert_eq!(request.speed, 1.0, "Drama 3 refuses any other speed");
        assert!(validate_request(FISH_DRAMA3_FAMILY, &request, &samples).is_ok());

        // Plain speech: no PCM, no rate, and the runtime accepts it.
        let plain = parse_args(&base()).unwrap();
        let (request, samples) = build_request(&plain, 5, None);
        assert!(samples.is_empty() && request.reference_sample_rate.is_none());
        assert!(validate_request(FISH_DRAMA3_FAMILY, &request, &samples).is_ok());
    }

    #[test]
    fn the_runtime_refuses_what_the_cli_passes_through() {
        // Ranges are the runtime's to judge; the CLI must forward them.
        for extra in [
            vec!["--temperature", "-1"],
            vec!["--top-p", "0"],
            vec!["--top-p", "1.5"],
            vec!["--top-k", "0"],
            vec!["--max-frames", "0"],
            vec!["--max-frames", "1025"],
        ] {
            let o = parse_args(&with(&extra)).unwrap();
            let (request, samples) = build_request(&o, 0, None);
            assert!(
                validate_request(FISH_DRAMA3_FAMILY, &request, &samples).is_err(),
                "{extra:?} accepted by the runtime"
            );
        }
        // A clip under one second is refused before any model loads.
        let o = parse_args(&clone_args(&["--consent"])).unwrap();
        let (request, samples) = build_request(&o, 0, Some((vec![0.1; 1_000], 44_100)));
        assert!(validate_request(FISH_DRAMA3_FAMILY, &request, &samples).is_err());
    }

    /// No --seed is a new take each run. The runtime takes a seed
    /// literally, so the old default of 0 repeated one take, and some Drama 3
    /// seeds go silent after the first sentence. A seed the caller names,
    /// 0 included, is used as given.
    #[test]
    fn no_seed_draws_a_fresh_one_and_a_named_seed_is_kept() {
        let mut options = parse_args(&base()).unwrap();
        let first = prepare_request(&options).unwrap().0.seed;
        let second = prepare_request(&options).unwrap().0.seed;
        assert!(first.is_some() && second.is_some());
        assert_ne!(first, second, "two runs without --seed are two takes");
        for named in [0u64, 9] {
            options.seed = Some(named);
            assert_eq!(prepare_request(&options).unwrap().0.seed, Some(named));
        }
    }

    #[test]
    fn prepare_refuses_before_any_model_loads() {
        let dir = tempdir("prepare");
        let mut options = parse_args(&with(&["--top-p", "1.5"])).unwrap();
        assert!(matches!(prepare_request(&options), Err(Error::Usage(_))));
        options.top_p = Some(0.9);
        let (request, samples) = prepare_request(&options).unwrap();
        assert_eq!((request.top_p, samples.len()), (Some(0.9), 0));
        // A half-second reference is under the 1 s floor.
        let short = dir.join("short.wav");
        std::fs::write(
            &short,
            audio::write_wav_f32(&Waveform::mono(16_000, vec![0.1; 8_000]).unwrap()),
        )
        .unwrap();
        options.reference = Some(short.clone());
        options.reference_text = Some("words".to_string());
        options.consent = true;
        assert!(matches!(prepare_request(&options), Err(Error::Usage(m)) if m.contains("seconds")));
        // Two seconds at 16 kHz goes through at its own rate.
        std::fs::write(
            &short,
            audio::write_wav_f32(&Waveform::mono(16_000, vec![0.1; 32_000]).unwrap()),
        )
        .unwrap();
        let (request, samples) = prepare_request(&options).unwrap();
        assert_eq!(
            (samples.len(), request.reference_sample_rate),
            (32_000, Some(16_000))
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn only_drama3_installs_are_spoken() {
        let dir = tempdir("family");
        std::fs::write(dir.join("config.json"), r#"{"model_type":"fish_drama3"}"#).unwrap();
        assert!(require_drama3(&dir).is_ok());
        std::fs::write(dir.join("config.json"), r#"{"model_type":"megatts3"}"#).unwrap();
        let Err(Error::Failed(m)) = require_drama3(&dir) else {
            panic!("a megatts3 directory was accepted");
        };
        assert!(
            m.contains("megatts3") && m.contains("turbospark-clone"),
            "{m}"
        );
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        assert!(require_drama3(&dir).is_err());
        std::fs::remove_file(dir.join("config.json")).unwrap();
        assert!(require_drama3(&dir).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_missing_model_fails_before_any_output_is_written() {
        let output = std::env::temp_dir().join(format!("speak-never-{}.wav", std::process::id()));
        let options = SpeakOptions {
            model: PathBuf::from("/nonexistent/drama3"),
            text: "hello".to_string(),
            output: output.clone(),
            reference: None,
            reference_text: None,
            consent: false,
            seed: Some(0),
            temperature: None,
            top_p: None,
            top_k: None,
            max_frames: None,
            wav_format: WavFormat::Pcm16,
        };
        let err = speak(&options).unwrap_err();
        assert!(
            matches!(&err, Error::Failed(m) if m.contains("/nonexistent/drama3")),
            "{err:?}"
        );
        assert!(!output.exists());
    }

    #[test]
    fn reference_loading_downmixes_and_keeps_the_native_rate() {
        let dir = tempdir("load");
        // Stereo 48 kHz: left 0.5, right -0.1 -> mono 0.2, same frame count and rate.
        let frames = 4_800usize;
        let samples: Vec<f32> = (0..frames).flat_map(|_| [0.5f32, -0.1]).collect();
        let path = dir.join("stereo48k.wav");
        std::fs::write(
            &path,
            audio::write_wav_f32(&Waveform::new(48_000, 2, samples).unwrap()),
        )
        .unwrap();
        let (mono, rate) = load_reference(&path).unwrap();
        assert_eq!((mono.len(), rate), (frames, 48_000));
        assert!(mono.iter().all(|v| (v - 0.2).abs() < 1e-3));
        assert!(load_reference(&dir.join("absent.wav")).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn output_wavs_keep_the_engines_rate_in_both_formats() {
        let wave = Waveform::mono(44_100, vec![0.25; 441]).unwrap();
        for format in [WavFormat::Pcm16, WavFormat::Float32] {
            let back = audio::read_wav_f32_bytes(&encode(&wave, format)).unwrap();
            assert_eq!((back.sample_rate, back.channels), (44_100, 1), "{format:?}");
            assert_eq!(back.samples.len(), 441);
            assert!((back.samples[0] - 0.25).abs() < 1e-3);
        }
    }
}
