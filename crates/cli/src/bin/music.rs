//! `turbospark-music`: the MiniMax Music 3 generation command.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use audio::music::minimax_music3::{Model, Music3Precision, TextGenerateRequest};
use audio::Waveform;
use catalog::Store;

const USAGE: &str = "\
turbospark-music: generate MiniMax Music 3 audio

USAGE:
    turbospark-music generate --model ALIAS_OR_DIR --caption TEXT \\
        (--lyrics TEXT | --lyrics-file PATH) --output WAV \\
        [--duration SECONDS] [--steps N] [--seed N] [--precision checkpoint|float32]

The model must be a pinned MiniMax Music 3 profile installed by
`turbospark-model pull-audio`, or a local converted model directory.
Lyrics are required. Use `[instrumental]` explicitly for instrumental output.
";

#[derive(Debug, Clone, PartialEq)]
struct GenerateOptions {
    model: String,
    caption: String,
    lyrics: String,
    duration: Option<f64>,
    steps: Option<usize>,
    seed: Option<u64>,
    output: PathBuf,
    precision: Music3Precision,
}

#[derive(Debug)]
enum Error {
    Usage(String),
    Failed(String),
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
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

fn run(args: &[String]) -> Result<(), Error> {
    let options = parse_args(args)?;
    let store = Store::default_store().map_err(Error::Failed)?;
    generate_to_file(&store, &options)
}

fn parse_args(args: &[String]) -> Result<GenerateOptions, Error> {
    if args.is_empty() || (args.len() == 1 && (args[0] == "--help" || args[0] == "-h")) {
        return Err(Error::Usage(USAGE.to_string()));
    }
    if args[0] != "generate" {
        return Err(Error::Usage(format!("unknown command {:?}", args[0])));
    }

    let mut model = None;
    let mut caption = None;
    let mut lyrics = None;
    let mut lyrics_file = None;
    let mut duration = None;
    let mut steps = None;
    let mut seed = None;
    let mut output = None;
    let mut precision = None;
    let mut index = 1usize;
    while index < args.len() {
        let flag = args[index].as_str();
        let value = || -> Result<&str, Error> {
            args.get(index + 1)
                .map(String::as_str)
                .ok_or_else(|| Error::Usage(format!("{flag} requires a value")))
        };
        match flag {
            "--model" => set_once(&mut model, value()?.to_string(), flag)?,
            "--caption" => set_once(&mut caption, value()?.to_string(), flag)?,
            "--lyrics" => set_once(&mut lyrics, value()?.to_string(), flag)?,
            "--lyrics-file" => set_once(&mut lyrics_file, value()?.to_string(), flag)?,
            "--duration" => {
                let text = value()?;
                let parsed = text
                    .parse::<f64>()
                    .map_err(|_| Error::Usage(format!("--duration {text:?} is not a number")))?;
                set_once(&mut duration, parsed, flag)?;
            }
            "--steps" => {
                let text = value()?;
                let parsed = text.parse::<usize>().map_err(|_| {
                    Error::Usage(format!("--steps {text:?} is not a positive integer"))
                })?;
                set_once(&mut steps, parsed, flag)?;
            }
            "--seed" => {
                let text = value()?;
                let parsed = text.parse::<u64>().map_err(|_| {
                    Error::Usage(format!("--seed {text:?} is not an unsigned integer"))
                })?;
                set_once(&mut seed, parsed, flag)?;
            }
            "--precision" => {
                let selected = match value()? {
                    "checkpoint" => Music3Precision::Checkpoint,
                    "float32" => Music3Precision::Float32,
                    other => {
                        return Err(Error::Usage(format!(
                            "unknown precision {other:?}; expected checkpoint or float32"
                        )))
                    }
                };
                set_once(&mut precision, selected, flag)?;
            }
            "--output" => set_once(&mut output, PathBuf::from(value()?), flag)?,
            "--help" | "-h" => return Err(Error::Usage(USAGE.to_string())),
            other if other.starts_with('-') => {
                return Err(Error::Usage(format!("unknown option {other:?}")))
            }
            other => return Err(Error::Usage(format!("unexpected argument {other:?}"))),
        }
        index += 2;
    }

    if lyrics.is_some() == lyrics_file.is_some() {
        return Err(Error::Usage(
            "provide exactly one of --lyrics or --lyrics-file".to_string(),
        ));
    }
    let lyrics = if let Some(lyrics) = lyrics {
        lyrics
    } else {
        let path = lyrics_file.expect("exactly one lyrics source was checked");
        std::fs::read_to_string(&path)
            .map_err(|error| Error::Usage(format!("reading lyrics file {path}: {error}")))?
    };
    let options = GenerateOptions {
        model: required(model, "--model")?,
        caption: required(caption, "--caption")?,
        lyrics,
        duration,
        steps,
        seed,
        output: required(output, "--output")?,
        precision: precision.unwrap_or(Music3Precision::Checkpoint),
    };
    text_request(&options)
        .validate()
        .map_err(|error| Error::Usage(error.to_string()))?;
    if options.output.as_os_str().is_empty() {
        return Err(Error::Usage("--output cannot be empty".to_string()));
    }
    Ok(options)
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

fn text_request(options: &GenerateOptions) -> TextGenerateRequest {
    TextGenerateRequest {
        caption: options.caption.clone(),
        lyrics: options.lyrics.clone(),
        duration_seconds: options.duration,
        steps: options.steps,
        seed: options.seed,
    }
}

fn resolve_audio_model(store: &Store, model: &str) -> Result<PathBuf, Error> {
    store.resolve_audio(model).ok_or_else(|| {
        Error::Failed(format!(
            "{model:?} is neither a directory nor an installed audio alias; run `turbospark-model list-audio`"
        ))
    })
}

fn is_tiny_fixture(path: &Path) -> Result<bool, Error> {
    let config_path = path.join("config.json");
    let bytes = std::fs::read(&config_path)
        .map_err(|error| Error::Failed(format!("reading {}: {error}", config_path.display())))?;
    let config: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| Error::Failed(format!("parsing {}: {error}", config_path.display())))?;
    Ok(config
        .get("hidden_size")
        .and_then(serde_json::Value::as_u64)
        .is_some_and(|size| size <= 64)
        && config
            .get("vocab_size")
            .and_then(serde_json::Value::as_u64)
            .is_some_and(|size| size <= 512))
}

fn generate_to_file(store: &Store, options: &GenerateOptions) -> Result<(), Error> {
    let model_path = resolve_audio_model(store, &options.model)?;
    let request = text_request(options);
    let generated = if is_tiny_fixture(&model_path)? {
        Model::load_converted_with_precision(&model_path, options.precision)
            .and_then(|model| model.generate_text(&request))
            .map_err(|error| Error::Failed(error.to_string()))?
    } else {
        generate_metal(&model_path, &request, options.precision)?
    };
    let waveform = Waveform::new(generated.sample_rate, 2, generated.waveform)
        .map_err(|error| Error::Failed(error.to_string()))?;
    std::fs::write(&options.output, audio::write_wav_i16(&waveform))
        .map_err(|error| Error::Failed(format!("writing {}: {error}", options.output.display())))?;
    println!(
        "wrote {} samples at {} Hz, stereo, to {}",
        generated.samples,
        generated.sample_rate,
        options.output.display()
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn generate_metal(
    path: &Path,
    request: &TextGenerateRequest,
    precision: Music3Precision,
) -> Result<audio::music::minimax_music3::Generation, Error> {
    let runner = runtime::Music3Runner::open_with_precision(path, precision)
        .map_err(|error| Error::Failed(error.to_string()))?;
    eprintln!(
        "Music 3 Metal: {:.2} GiB resident weights",
        runner.resident_weight_bytes() as f64 / 1073741824.0
    );
    runner
        .generate_text(request)
        .map_err(|error| Error::Failed(error.to_string()))
}

#[cfg(not(target_os = "macos"))]
fn generate_metal(
    _path: &Path,
    _request: &TextGenerateRequest,
    _precision: Music3Precision,
) -> Result<audio::music::minimax_music3::Generation, Error> {
    Err(Error::Failed(
        "full-size Music 3 generation requires macOS and a Metal device".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_args(model: String) -> Vec<String> {
        [
            "generate".to_string(),
            "--model".to_string(),
            model,
            "--caption".to_string(),
            "soft piano".to_string(),
            "--lyrics".to_string(),
            "[instrumental]".to_string(),
            "--duration".to_string(),
            "0.04".to_string(),
            "--steps".to_string(),
            "1".to_string(),
            "--seed".to_string(),
            "7".to_string(),
            "--output".to_string(),
            "song.wav".to_string(),
        ]
        .to_vec()
    }

    #[test]
    fn parser_requires_exactly_one_lyrics_source_and_valid_options() {
        let args = base_args("tiny-model".to_string());
        assert_eq!(parse_args(&args).unwrap().seed, Some(7));
        for lyrics_args in [
            Vec::<String>::new(),
            vec!["--lyrics-file".to_string(), "lyrics.txt".to_string()],
            vec![
                "--lyrics-file".to_string(),
                "lyrics.txt".to_string(),
                "--lyrics".to_string(),
                "[instrumental]".to_string(),
            ],
        ] {
            let mut invalid = args.clone();
            invalid.retain(|arg| arg != "--lyrics" && arg != "[instrumental]");
            invalid.extend(lyrics_args);
            assert!(parse_args(&invalid).is_err());
        }
        for (flag, value) in [("--steps", "0"), ("--duration", "NaN"), ("--seed", "-1")] {
            let mut invalid = args.clone();
            let index = invalid.iter().position(|arg| arg == flag).unwrap();
            invalid[index + 1] = value.to_string();
            assert!(parse_args(&invalid).is_err(), "{flag} {value}");
        }
    }

    #[test]
    fn precision_defaults_and_rejects_ambiguous_selection() {
        let args = base_args("tiny-model".to_string());
        assert_eq!(
            parse_args(&args).unwrap().precision,
            Music3Precision::Checkpoint
        );
        for (value, expected) in [
            ("checkpoint", Music3Precision::Checkpoint),
            ("float32", Music3Precision::Float32),
        ] {
            let mut selected = args.clone();
            selected.extend(["--precision".into(), value.into()]);
            assert_eq!(parse_args(&selected).unwrap().precision, expected);
            selected.extend(["--precision".into(), value.into()]);
            assert!(parse_args(&selected).is_err());
        }
        let mut invalid = args;
        invalid.extend(["--precision".into(), "bf16".into()]);
        assert!(parse_args(&invalid).is_err());
    }

    #[test]
    fn model_resolution_accepts_audio_aliases_and_explicit_directories() {
        let root =
            std::env::temp_dir().join(format!("turbospark-music-path-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let store = Store::new(&root);
        let alias_path = store.audio_install_path("minimax-music3-test");
        std::fs::create_dir_all(&alias_path).unwrap();
        let explicit = root.join("explicit-model");
        std::fs::create_dir_all(&explicit).unwrap();
        assert_eq!(
            resolve_audio_model(&store, "minimax-music3-test").unwrap(),
            alias_path
        );
        assert_eq!(
            resolve_audio_model(&store, explicit.to_str().unwrap()).unwrap(),
            explicit
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn tiny_generation_routes_selected_precision() {
        let model = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../audio/testdata/minimax_music3/precision/bf16");
        let mut outputs = vec![];
        for precision in ["checkpoint", "float32"] {
            let output = std::env::temp_dir().join(format!(
                "music3-precision-{}-{precision}.wav",
                std::process::id()
            ));
            let mut args = base_args(model.to_string_lossy().into_owned());
            let index = args.iter().position(|s| s == "song.wav").unwrap();
            args[index] = output.to_string_lossy().into_owned();
            args.extend(["--precision".into(), precision.into()]);
            generate_to_file(
                &Store::new(std::env::temp_dir()),
                &parse_args(&args).unwrap(),
            )
            .unwrap();
            outputs.push(std::fs::read(&output).unwrap());
            std::fs::remove_file(output).unwrap();
        }
        assert_ne!(
            outputs[0], outputs[1],
            "precision option must change the execution route"
        );
    }

    #[test]
    fn tiny_generation_writes_a_stereo_44100_wav() {
        let model = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../audio/testdata/minimax_music3/converted_plain");
        let output = std::env::temp_dir().join(format!(
            "turbospark-music-output-{}.wav",
            std::process::id()
        ));
        let mut args = base_args(model.to_string_lossy().into_owned());
        let output_index = args.iter().position(|arg| arg == "song.wav").unwrap();
        args[output_index] = output.to_string_lossy().into_owned();
        let options = parse_args(&args).unwrap();
        generate_to_file(&Store::new(std::env::temp_dir()), &options).unwrap();
        let bytes = std::fs::read(&output).unwrap();
        let decoded = audio::read_wav_f32_bytes(&bytes).unwrap();
        assert_eq!(decoded.sample_rate, 44_100);
        assert_eq!(decoded.channels, 2);
        assert!(!decoded.samples.is_empty());
        std::fs::remove_file(output).unwrap();
    }
}
