//! `turbospark-music`: the music generation command (MiniMax Music 3 and
//! ACE-Step 1.5 Turbo, chosen by what the model directory holds).

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use audio::music::minimax_music3::{
    is_tiny_fixture_dims, FlowGuidance, FlowProfile, FlowSolver, Generation, Model,
    Music3Precision, StageTimings, TextGenerateRequest,
};
use audio::Waveform;
use catalog::Store;

const USAGE: &str = "\
turbospark-music: generate music (MiniMax Music 3 or ACE-Step 1.5 Turbo)

USAGE:
    turbospark-music generate --model ALIAS_OR_DIR --caption TEXT \\
        (--lyrics TEXT | --lyrics-file PATH) --output WAV \\
        [--duration SECONDS] [--steps N] [--seed N] [--precision checkpoint|float32] \\
        [--wav-format pcm16|float32] [--timings] [--language CODE] \\
        [--flow-profile fast|exact|turbo] [--flow-uncond-interval N] \\
        [--flow-uncond-warmup N] [--flow-solver euler|ab2]

The model must be a pinned music profile installed by
`turbospark-model pull-audio` (minimax-music3-*, ace-step15-turbo), or a
local model directory.

ACE-Step 1.5 Turbo (48 kHz stereo, Metal): --duration is required (up to
600 s), --language sets the vocal language code (default en), and the
schedule is fixed at 8 steps with upstream's DCW correction. The Music 3
options (--precision, --flow-*, --steps other than 8) are refused for it.
The rest of this text describes MiniMax Music 3 (44.1 kHz stereo).
Lyrics are required. Use `[instrumental]` explicitly for instrumental output.
--wav-format pcm16 (default) rounds to 16-bit integers; float32 writes the
generated samples unrounded. --timings prints per-stage wall time and the
real-time factor to stderr.
With no --flow-profile, the checkpoint's declared profile is used. Ordinary
checkpoints default to fast. The marked Turbo candidate defaults to 8-step
Euler with both guidance branches on every step. --flow-profile fast integrates
the flow with second-order Adams-Bashforth, reuses the unconditional guidance
branch (interval 3, warmup 2), and runs 21 steps when --steps is not given: 30
DiT forwards instead of 60, about half the DiT time. It changes the audio and
does not follow MLX's trajectory. --flow-profile exact restores the reference:
Euler, both guidance branches every step, 30 steps by default.
--flow-uncond-interval N, --flow-uncond-warmup N and --flow-solver euler|ab2
override that part of the chosen profile. An explicit --steps is always used.
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
    wav_format: WavFormat,
    timings: bool,
    flow_profile: Option<FlowProfile>,
    flow_uncond_interval: Option<usize>,
    flow_uncond_warmup: Option<usize>,
    flow_solver: Option<FlowSolver>,
    /// Whether --precision was given (ACE-Step refuses it).
    precision_explicit: bool,
    /// ACE-Step vocal language code; Music 3 refuses it.
    language: Option<String>,
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
    let mut wav_format = None;
    let mut timings = false;
    let mut flow_profile = None;
    let mut uncond_interval = None;
    let mut uncond_warmup = None;
    let mut flow_solver = None;
    let mut language = None;
    let mut index = 1usize;
    while index < args.len() {
        let flag = args[index].as_str();
        if flag == "--timings" {
            if timings {
                return Err(Error::Usage(
                    "--timings may be specified only once".to_string(),
                ));
            }
            timings = true;
            index += 1;
            continue;
        }
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
            "--flow-uncond-interval" | "--flow-uncond-warmup" => {
                let text = value()?;
                let parsed = text.parse::<usize>().map_err(|_| {
                    Error::Usage(format!("{flag} {text:?} is not a non-negative integer"))
                })?;
                if flag == "--flow-uncond-interval" {
                    set_once(&mut uncond_interval, parsed, flag)?;
                } else {
                    set_once(&mut uncond_warmup, parsed, flag)?;
                }
            }
            "--flow-profile" => {
                let selected = match value()? {
                    "fast" => FlowProfile::Fast,
                    "exact" => FlowProfile::Exact,
                    "turbo" => FlowProfile::Turbo,
                    other => {
                        return Err(Error::Usage(format!(
                            "unknown flow profile {other:?}; expected fast, exact or turbo"
                        )))
                    }
                };
                set_once(&mut flow_profile, selected, flag)?;
            }
            "--flow-solver" => {
                let selected = match value()? {
                    "euler" => FlowSolver::Euler,
                    "ab2" => FlowSolver::AdamsBashforth2,
                    other => {
                        return Err(Error::Usage(format!(
                            "unknown flow solver {other:?}; expected euler or ab2"
                        )))
                    }
                };
                set_once(&mut flow_solver, selected, flag)?;
            }
            "--output" => set_once(&mut output, PathBuf::from(value()?), flag)?,
            "--language" => set_once(&mut language, value()?.to_string(), flag)?,
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
    // Explicit flags override the profile's solver and guidance schedule.
    let options = GenerateOptions {
        model: required(model, "--model")?,
        caption: required(caption, "--caption")?,
        lyrics,
        duration,
        steps,
        seed,
        output: required(output, "--output")?,
        precision: precision.unwrap_or(Music3Precision::Checkpoint),
        wav_format: wav_format.unwrap_or(WavFormat::Pcm16),
        timings,
        flow_profile,
        flow_uncond_interval: uncond_interval,
        flow_uncond_warmup: uncond_warmup,
        flow_solver,
        precision_explicit: precision.is_some(),
        language,
    };
    if options.flow_uncond_interval == Some(0) {
        return Err(Error::Usage(
            "--flow-uncond-interval must be at least 1".to_string(),
        ));
    }
    // Family-specific bounds are checked once the model is known; these
    // hold for every music model.
    if options
        .duration
        .is_some_and(|d| !(d.is_finite() && d > 0.0))
    {
        return Err(Error::Usage(
            "--duration must be a positive number".to_string(),
        ));
    }
    if options.steps == Some(0) {
        return Err(Error::Usage("--steps must be at least 1".to_string()));
    }
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
        max_frames: None,
        context_size: None,
        steps: options.steps,
        seed: options.seed,
    }
}

fn resolve_audio_model(store: &Store, model: &str) -> Result<PathBuf, Error> {
    // An alias on a drive that is not plugged in says which drive (10.7)
    // rather than that no such model is installed.
    if let Some(message) = catalog::not_connected_message(model, catalog::ModelModality::Audio) {
        return Err(Error::Failed(message));
    }
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
    let dimension = |key: &str| {
        config
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
    };
    Ok(match (dimension("hidden_size"), dimension("vocab_size")) {
        (Some(hidden), Some(vocab)) => is_tiny_fixture_dims(hidden, vocab),
        _ => false,
    })
}

fn generate_to_file(store: &Store, options: &GenerateOptions) -> Result<(), Error> {
    let model_path = resolve_audio_model(store, &options.model)?;
    #[cfg(target_os = "macos")]
    if runtime::ace_step15::detect(&model_path) {
        return generate_ace(&model_path, options);
    }
    if options.language.is_some() {
        return Err(Error::Usage(
            "--language is an ACE-Step setting; MiniMax Music 3 does not take it".to_string(),
        ));
    }
    let request = text_request(options);
    request
        .validate()
        .map_err(|error| Error::Usage(error.to_string()))?;
    let (generated, timings) = if is_tiny_fixture(&model_path)? {
        Model::load_converted_with_precision(&model_path, options.precision)
            .and_then(|model| {
                let profile = options
                    .flow_profile
                    .unwrap_or_else(|| model.preferred_runner_profile());
                model.set_flow_profile(profile);
                if let Some(guidance) = guidance_override(options, profile) {
                    model.set_flow_guidance(guidance)?;
                }
                if let Some(solver) = options.flow_solver {
                    model.set_flow_solver(solver);
                }
                model.generate_text_timed(&request)
            })
            .map_err(|error| Error::Failed(error.to_string()))?
    } else {
        generate_metal(&model_path, &request, options.precision, options)?
    };
    let waveform = Waveform::new(generated.sample_rate, 2, generated.waveform)
        .map_err(|error| Error::Failed(error.to_string()))?;
    let encoded = match options.wav_format {
        WavFormat::Pcm16 => audio::write_wav_i16(&waveform),
        WavFormat::Float32 => audio::write_wav_f32(&waveform),
    };
    std::fs::write(&options.output, encoded)
        .map_err(|error| Error::Failed(format!("writing {}: {error}", options.output.display())))?;
    if options.timings {
        print_timings(&timings, generated.samples, generated.sample_rate);
    }
    println!(
        "wrote {} samples at {} Hz, stereo, to {}",
        generated.samples,
        generated.sample_rate,
        options.output.display()
    );
    Ok(())
}

/// The Music 3 flags ACE-Step would otherwise ignore, which would look
/// like they worked.
fn ace_refusals(options: &GenerateOptions) -> Result<(), Error> {
    let music3_only = [
        (options.precision_explicit, "--precision"),
        (options.flow_profile.is_some(), "--flow-profile"),
        (
            options.flow_uncond_interval.is_some(),
            "--flow-uncond-interval",
        ),
        (options.flow_uncond_warmup.is_some(), "--flow-uncond-warmup"),
        (options.flow_solver.is_some(), "--flow-solver"),
    ];
    if let Some((_, flag)) = music3_only.iter().find(|(set, _)| *set) {
        return Err(Error::Usage(format!(
            "{flag} is a MiniMax Music 3 option; ACE-Step does not take it"
        )));
    }
    if options.steps.is_some_and(|s| s != 8) {
        return Err(Error::Usage(
            "ACE-Step Turbo runs a fixed 8-step schedule; leave --steps unset or 8".to_string(),
        ));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn generate_ace(path: &Path, options: &GenerateOptions) -> Result<(), Error> {
    use audio::music::ace_step15::Stage;
    ace_refusals(options)?;
    let duration = options
        .duration
        .ok_or_else(|| Error::Usage("ACE-Step needs --duration SECONDS (up to 600)".to_string()))?;
    if duration > audio::music::ace_step15::pipeline::MAX_DURATION_SECONDS {
        return Err(Error::Usage(format!(
            "--duration {duration} exceeds ACE-Step's {} s",
            audio::music::ace_step15::pipeline::MAX_DURATION_SECONDS
        )));
    }
    let language = options.language.as_deref().unwrap_or("en");
    if language.is_empty()
        || language.len() > 16
        || !language
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err(Error::Usage(
            "--language must be a short code such as en, zh or ja".to_string(),
        ));
    }
    let started = std::time::Instant::now();
    let mut runner =
        runtime::ace_step15::AceStep15Runner::open(path, runtime::ace_step15::Device::Metal)
            .map_err(|error| Error::Failed(error.to_string()))?;
    let loaded = started.elapsed();
    let request = runtime::ace_step15::Request {
        caption: &options.caption,
        lyrics: &options.lyrics,
        language,
        duration_seconds: duration,
        seed: options.seed.unwrap_or(0),
    };
    let started = std::time::Instant::now();
    let rendered = runner
        .generate(&request, |stage| {
            if let Stage::Step { index, total } = stage {
                eprintln!("[ace-step] step {}/{total}", index + 1);
            }
            true
        })
        .map_err(|error| Error::Failed(error.to_string()))?
        .ok_or_else(|| Error::Failed("ACE-Step generation stopped".to_string()))?;
    let elapsed = started.elapsed();
    if rendered.caption_truncated || rendered.lyrics_truncated {
        eprintln!(
            "note: the {} was cut to ACE-Step's token limit (256 caption, 2048 lyric tokens)",
            if rendered.caption_truncated {
                "caption"
            } else {
                "lyrics"
            }
        );
    }
    let samples = rendered.interleaved.len() / 2;
    let waveform = Waveform::new(rendered.sample_rate, 2, rendered.interleaved)
        .map_err(|error| Error::Failed(error.to_string()))?;
    let encoded = match options.wav_format {
        WavFormat::Pcm16 => audio::write_wav_i16(&waveform),
        WavFormat::Float32 => audio::write_wav_f32(&waveform),
    };
    std::fs::write(&options.output, encoded)
        .map_err(|error| Error::Failed(format!("writing {}: {error}", options.output.display())))?;
    if options.timings {
        let audio_seconds = samples as f64 / f64::from(rendered.sample_rate);
        eprintln!(
            "timings: load {:.1} s, generate {:.1} s for {audio_seconds:.2} s of audio (RTF {:.2})",
            loaded.as_secs_f64(),
            elapsed.as_secs_f64(),
            elapsed.as_secs_f64() / audio_seconds
        );
    }
    println!(
        "wrote {samples} samples at {} Hz, stereo, to {}",
        rendered.sample_rate,
        options.output.display()
    );
    Ok(())
}

fn print_timings(timings: &StageTimings, samples: usize, sample_rate: u32) {
    let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
    let audio_seconds = samples as f64 / f64::from(sample_rate);
    eprintln!(
        "timings: total {:.0} ms for {audio_seconds:.2} s of audio (RTF {:.2}), {} AR frames, {} flow chunks",
        ms(timings.total),
        timings.total.as_secs_f64() / audio_seconds,
        timings.frames,
        timings.chunks
    );
    eprintln!(
        "  ar {:.0} ms: prefill {:.0}, lm_head {:.0}, sampling {:.0}, depth {:.0}, lm_decode {:.0}",
        ms(timings.autoregressive()),
        ms(timings.prefill),
        ms(timings.lm_head),
        ms(timings.sampling),
        ms(timings.depth),
        ms(timings.lm_decode)
    );
    eprintln!(
        "  flow {:.0} ms: condition {:.0}, dit {:.0}, vocoder {:.0}; stitch {:.0}; tokenize {:.0}",
        ms(timings.flow()),
        ms(timings.condition),
        ms(timings.dit),
        ms(timings.vocoder),
        ms(timings.stitch),
        ms(timings.tokenize)
    );
}

#[cfg(target_os = "macos")]
fn generate_metal(
    path: &Path,
    request: &TextGenerateRequest,
    precision: Music3Precision,
    options: &GenerateOptions,
) -> Result<(Generation, StageTimings), Error> {
    let runner = runtime::Music3Runner::open_with_precision(path, precision)
        .map_err(|error| Error::Failed(error.to_string()))?;
    let profile = options
        .flow_profile
        .unwrap_or_else(|| runner.preferred_runner_profile());
    if options.flow_profile.is_some() {
        runner.set_flow_profile(profile);
    }
    if let Some(guidance) = guidance_override(options, profile) {
        runner
            .set_flow_guidance(guidance)
            .map_err(|error| Error::Failed(error.to_string()))?;
    }
    if let Some(solver) = options.flow_solver {
        runner.set_flow_solver(solver);
    }
    eprintln!(
        "Music 3 Metal: {:.2} GiB resident weights",
        runner.resident_weight_bytes() as f64 / 1073741824.0
    );
    runner
        .generate_text_timed(request)
        .map_err(|error| Error::Failed(error.to_string()))
}

#[cfg(not(target_os = "macos"))]
fn generate_metal(
    _path: &Path,
    _request: &TextGenerateRequest,
    _precision: Music3Precision,
    _options: &GenerateOptions,
) -> Result<(Generation, StageTimings), Error> {
    Err(Error::Failed(
        "full-size Music 3 generation requires macOS and a Metal device".into(),
    ))
}

fn guidance_override(options: &GenerateOptions, profile: FlowProfile) -> Option<FlowGuidance> {
    if options.flow_uncond_interval.is_none() && options.flow_uncond_warmup.is_none() {
        return None;
    }
    let mut guidance = profile.guidance();
    guidance.uncond_interval = options
        .flow_uncond_interval
        .unwrap_or(guidance.uncond_interval);
    guidance.uncond_warmup = options.flow_uncond_warmup.unwrap_or(guidance.uncond_warmup);
    Some(guidance)
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
    fn parser_leaves_checkpoint_profile_intact_and_accepts_named_overrides() {
        let args = base_args("tiny-model".to_string());
        let default = parse_args(&args).unwrap();
        assert_eq!(default.flow_profile, None);
        assert_eq!(default.flow_solver, None);
        assert_eq!(guidance_override(&default, FlowProfile::Fast), None);
        let mut exact = args.clone();
        exact.extend(["--flow-profile", "exact"].map(String::from));
        let exact = parse_args(&exact).unwrap();
        assert_eq!(exact.flow_profile, Some(FlowProfile::Exact));
        assert_eq!(exact.flow_solver, None);
        let mut turbo_args = args.clone();
        turbo_args.extend(["--flow-profile", "turbo"].map(String::from));
        let turbo = parse_args(&turbo_args).unwrap();
        assert_eq!(turbo.flow_profile, Some(FlowProfile::Turbo));
        assert_eq!(turbo.flow_profile.unwrap().solver(), FlowSolver::Euler);
        assert_eq!(guidance_override(&turbo, FlowProfile::Turbo), None);
        for bad in [["--flow-profile", "wrong"], ["--flow-profile", ""]] {
            let mut invalid = args.clone();
            invalid.extend(bad.map(String::from));
            assert!(parse_args(&invalid).is_err(), "{bad:?}");
        }
        let mut twice = args.clone();
        twice.extend(["--flow-profile", "fast", "--flow-profile", "exact"].map(String::from));
        assert!(parse_args(&twice).is_err());
    }

    #[test]
    fn explicit_flow_flags_override_the_profile_in_either_order() {
        let args = base_args("tiny-model".to_string());
        let mut reuse = args.clone();
        reuse.extend(["--flow-uncond-interval", "1", "--flow-solver", "euler"].map(String::from));
        let parsed = parse_args(&reuse).unwrap();
        assert_eq!(parsed.flow_profile, None);
        assert_eq!(parsed.flow_solver, Some(FlowSolver::Euler));
        assert_eq!(
            (
                guidance_override(&parsed, FlowProfile::Fast)
                    .unwrap()
                    .uncond_interval,
                guidance_override(&parsed, FlowProfile::Fast)
                    .unwrap()
                    .uncond_warmup
            ),
            (1, 2)
        );
        // An override given before --flow-profile exact still applies on top of it.
        let mut before = args.clone();
        before.extend(
            [
                "--flow-solver",
                "ab2",
                "--flow-uncond-warmup",
                "4",
                "--flow-profile",
                "exact",
            ]
            .map(String::from),
        );
        let parsed = parse_args(&before).unwrap();
        assert_eq!(parsed.flow_solver, Some(FlowSolver::AdamsBashforth2));
        assert_eq!(
            (
                guidance_override(&parsed, FlowProfile::Exact)
                    .unwrap()
                    .uncond_interval,
                guidance_override(&parsed, FlowProfile::Exact)
                    .unwrap()
                    .uncond_warmup
            ),
            (1, 4)
        );
        for bad in [
            ["--flow-uncond-interval", "0"],
            ["--flow-uncond-warmup", "x"],
            ["--flow-solver", "rk4"],
        ] {
            let mut invalid = args.clone();
            invalid.extend(bad.map(String::from));
            assert!(parse_args(&invalid).is_err(), "{bad:?}");
        }
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
    fn wav_format_and_timings_flags_parse_once() {
        let args = base_args("tiny-model".to_string());
        let defaults = parse_args(&args).unwrap();
        assert_eq!(defaults.wav_format, WavFormat::Pcm16);
        assert!(!defaults.timings);
        let mut selected = args.clone();
        selected.extend(["--wav-format".into(), "float32".into(), "--timings".into()]);
        let options = parse_args(&selected).unwrap();
        assert_eq!(options.wav_format, WavFormat::Float32);
        assert!(options.timings);
        let mut duplicate = selected.clone();
        duplicate.push("--timings".into());
        assert!(parse_args(&duplicate).is_err());
        let mut invalid = args;
        invalid.extend(["--wav-format".into(), "mp3".into()]);
        assert!(parse_args(&invalid).is_err());
    }

    #[test]
    fn float32_wav_keeps_samples_that_pcm16_rounds() {
        let model = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../audio/testdata/minimax_music3/converted_plain");
        let mut decoded = vec![];
        for format in ["pcm16", "float32"] {
            let output = std::env::temp_dir()
                .join(format!("music3-format-{}-{format}.wav", std::process::id()));
            let mut args = base_args(model.to_string_lossy().into_owned());
            let index = args.iter().position(|s| s == "song.wav").unwrap();
            args[index] = output.to_string_lossy().into_owned();
            args.extend(["--wav-format".into(), format.into()]);
            generate_to_file(
                &Store::new(std::env::temp_dir()),
                &parse_args(&args).unwrap(),
            )
            .unwrap();
            decoded.push(audio::read_wav_f32_bytes(&std::fs::read(&output).unwrap()).unwrap());
            std::fs::remove_file(output).unwrap();
        }
        assert_eq!(decoded[0].samples.len(), decoded[1].samples.len());
        // PCM16 lands on multiples of 1/32768; the float file does not.
        let on_grid = |v: f32| (v * 32768.0 - (v * 32768.0).round()).abs() < 1e-3;
        assert!(decoded[0].samples.iter().copied().all(on_grid));
        assert!(!decoded[1].samples.iter().copied().all(on_grid));
        let worst = decoded[0]
            .samples
            .iter()
            .zip(&decoded[1].samples)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst <= 0.5 / 32768.0 + 1e-7, "pcm16 error {worst}");
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

    #[test]
    fn language_parses_and_ace_refuses_music3_options() {
        let mut args = base_args("ace".to_string());
        // base_args carries --steps 1, a Music 3 value; ACE needs 8 or none.
        let at = args.iter().position(|a| a == "--steps").unwrap();
        args.drain(at..at + 2);
        args.extend(["--language".to_string(), "zh".to_string()]);
        let options = parse_args(&args).unwrap();
        assert_eq!(options.language.as_deref(), Some("zh"));
        assert!(!options.precision_explicit);
        assert!(ace_refusals(&options).is_ok());
        for extra in [
            vec!["--precision", "float32"],
            vec!["--flow-profile", "exact"],
            vec!["--flow-uncond-interval", "2"],
            vec!["--flow-uncond-warmup", "1"],
            vec!["--flow-solver", "ab2"],
            vec!["--steps", "30"],
        ] {
            let mut with = args.clone();
            with.extend(extra.iter().map(|v| v.to_string()));
            let parsed = parse_args(&with).unwrap();
            assert!(
                ace_refusals(&parsed).is_err(),
                "{extra:?} should be refused for ACE"
            );
        }
        let mut eight = args.clone();
        eight.extend(["--steps".to_string(), "8".to_string()]);
        assert!(ace_refusals(&parse_args(&eight).unwrap()).is_ok());
        let mut twice = args.clone();
        twice.extend(["--language".to_string(), "en".to_string()]);
        assert!(parse_args(&twice).is_err());
    }
}
