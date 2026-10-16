//! Parakeet TDT on Core ML (FluidInference `parakeet-redux-coreml` and
//! `parakeet-ultra-coreml`, which share the v3 component contract).
//!
//! The export splits the model into four compiled programs:
//!
//! - `Preprocessor`: 15 s of 16 kHz PCM to a NeMo log-mel `[1,128,1501]`.
//! - `Encoder`: the FastConformer, mel to `[1,1024,188]` (ternary weights,
//!   MIL ios18, so macOS 15+).
//! - `Decoder`: one LSTM prediction step with explicit `h`/`c` state.
//! - `JointDecisionv3`: joint network plus argmax over tokens and durations.
//!
//! This module supplies what the export leaves to the host: 15 s windows
//! with overlap, the TDT greedy loop, and merging windows on their shared
//! tokens. The loop follows the Redux branch of the CPU decoder
//! (`audio::stt::parakeet`, `redux_mode`): a blank advances at least one
//! frame, and the step count is capped at `max_symbols * frames`. That CPU
//! decoder on `moondream/parakeet-redux` is the parity oracle.

use std::path::Path;

use audio::stt::nemo::{
    merge_longest_common_subsequence, merge_longest_contiguous, sentences_to_result,
    tokens_to_sentences, AlignedResult, AlignedToken,
};
use audio::stt::parakeet::{adjust_sentencepiece_digit_spacing, is_special_piece};
use turbospark_coreml::{ComputeUnits, CoreMlModel, FeatureKind, LoadOptions, Tensor};

pub const SAMPLE_RATE: u32 = 16_000;
pub const VOCAB_FILE: &str = "parakeet_vocab.json";
pub const REQUIRED_DIRS: [&str; 4] = [
    "Preprocessor.mlmodelc",
    "Encoder.mlmodelc",
    "Decoder.mlmodelc",
    "JointDecisionv3.mlmodelc",
];
/// The `contract` value FluidInference writes into `config.json` for this
/// export. A different contract may use a different vocabulary or graph,
/// so it is refused rather than guessed at.
pub const KNOWN_CONTRACT: &str = "parakeet-tdt-0.6b-v3-coreml";

/// Hop 160 times subsampling 8: one encoder frame per 80 ms. Same grid as
/// the CPU decoder (`subsampling_factor * hop / sample_rate`).
const FRAME_SAMPLES: usize = 1280;
/// From `moondream/parakeet-redux` `config.json` at the pinned revision
/// (`durations`, `max_symbols_per_step`). FluidAudio uses the same values.
const DURATIONS: [usize; 5] = [0, 1, 2, 3, 4];
const MAX_SYMBOLS: usize = 10;
/// Target overlap between windows. Rounded down to whole encoder frames.
const OVERLAP_SAMPLES: usize = 2 * SAMPLE_RATE as usize;

const ENCODER_UNITS_ENV: &str = "TURBOSPARK_PARAKEET_COREML_ENCODER_UNITS";
/// Lets a subprocess probe place an unverified encoder on the GPU. Nothing
/// else should set it: a bad placement aborts the process, it does not fail.
const ALLOW_UNVERIFIED_GPU_ENV: &str = "TURBOSPARK_COREML_ALLOW_QUANTIZED_GPU";

/// Encoder weight formats measured on the GPU here. FluidAudio reports that
/// Core ML hands int8 encoders to MPSGraph under GPU placement, whose MLIR
/// pass fails and aborts the process (`StreamingUnifiedAsrManager` coerces
/// int8 to CPU and Neural Engine for that reason). On an M4 Max with macOS
/// 26.6 (2026-10-16) Ultra's int8 encoder did not abort on the GPU, but ran
/// at 18x real time against 122x on the Neural Engine, so it is no GPU
/// candidate either way. Anything not listed runs on the Neural Engine.
const GPU_VERIFIED_WEIGHT_FORMATS: [&str; 1] = [
    // parakeet-redux-coreml: docs/COREML.md bench, GPU encoder 150x.
    "ternary_lut2_int8_blockwise_fp16_scale_g128",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParakeetCoreMlOptions {
    /// The encoder is the only large program. [`Self::for_model`] picks it
    /// from the declared weight format: the GPU where measured (Redux's
    /// Neural Engine first load compiles for several minutes and cannot be
    /// cancelled), the Neural Engine otherwise, because an int8 encoder on
    /// the GPU can abort the process.
    pub encoder_units: ComputeUnits,
    /// Decoder and joint run once per decode step on tiny inputs, where the
    /// CPU avoids per-call accelerator dispatch.
    pub step_units: ComputeUnits,
}

impl Default for ParakeetCoreMlOptions {
    fn default() -> Self {
        Self {
            encoder_units: ComputeUnits::CpuAndGpu,
            step_units: ComputeUnits::CpuOnly,
        }
    }
}

impl ParakeetCoreMlOptions {
    /// Options for the export in `dir`: the encoder placement follows its
    /// declared weight format ([`default_encoder_units`]), then
    /// `TURBOSPARK_PARAKEET_COREML_ENCODER_UNITS` (`cpu`, `cpu_gpu`,
    /// `cpu_ane`, `all`) overrides it. An unknown value is an error, not a
    /// silent default; an unsafe GPU choice is refused at open.
    pub fn for_model(dir: &Path) -> Result<Self, String> {
        let format = encoder_weight_format(dir)?;
        let mut o = Self {
            encoder_units: default_encoder_units(format.as_deref()),
            ..Self::default()
        };
        if let Ok(v) = std::env::var(ENCODER_UNITS_ENV) {
            o.encoder_units = ComputeUnits::parse(v.trim()).ok_or_else(|| {
                format!("{ENCODER_UNITS_ENV}={v} is not one of cpu, cpu_gpu, cpu_ane, all")
            })?;
        }
        Ok(o)
    }
}

/// `encoders["Encoder.mlmodelc"].weight_format` from the export's
/// `config.json`, or `None` when the export does not declare one.
pub fn encoder_weight_format(dir: &Path) -> Result<Option<String>, String> {
    let Some(config) = read_config(dir)? else {
        return Ok(None);
    };
    Ok(config
        .get("encoders")
        .and_then(|e| e.get(REQUIRED_DIRS[1]))
        .and_then(|e| e.get("weight_format"))
        .and_then(|f| f.as_str())
        .map(str::to_owned))
}

/// Encoder placement for a declared weight format. The GPU is the default
/// only where it was measured: it loads in about a second, while the Neural
/// Engine's first compile takes minutes. An export with no declared format
/// keeps the GPU default it had before formats were read.
pub fn default_encoder_units(format: Option<&str>) -> ComputeUnits {
    match format {
        None => ComputeUnits::CpuAndGpu,
        Some(f) if GPU_VERIFIED_WEIGHT_FORMATS.contains(&f) => ComputeUnits::CpuAndGpu,
        Some(_) => ComputeUnits::CpuAndNeuralEngine,
    }
}

/// Refuses a GPU-capable placement for a declared format that was never
/// measured there, unless `allow_unverified` (the subprocess probe flag).
fn check_encoder_units(
    format: Option<&str>,
    units: ComputeUnits,
    allow_unverified: bool,
) -> Result<(), String> {
    let gpu = matches!(units, ComputeUnits::CpuAndGpu | ComputeUnits::All);
    let verified = format.is_none_or(|f| GPU_VERIFIED_WEIGHT_FORMATS.contains(&f));
    if !gpu || verified || allow_unverified {
        return Ok(());
    }
    Err(format!(
        "the Core ML Parakeet encoder ({}) is not verified on the GPU, where Core ML can abort the \
         process; use cpu_ane or cpu",
        format.unwrap_or_default()
    ))
}

/// Shapes read from the loaded programs and checked against each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Dims {
    window_samples: usize,
    mel_bins: usize,
    mel_frames: usize,
    encoder_hidden: usize,
    encoder_frames: usize,
    pred_layers: usize,
    pred_hidden: usize,
}

pub struct ParakeetCoreMl {
    preprocessor: CoreMlModel,
    encoder: CoreMlModel,
    decoder: CoreMlModel,
    joint: CoreMlModel,
    vocabulary: Vec<String>,
    dims: Dims,
}

impl std::fmt::Debug for ParakeetCoreMl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParakeetCoreMl")
            .field("dims", &self.dims)
            .field("vocabulary", &self.vocabulary.len())
            .finish()
    }
}

fn fixed_shape(m: &CoreMlModel, name: &str, inputs: bool) -> Result<Vec<usize>, String> {
    let specs = if inputs { m.inputs() } else { m.outputs() };
    let spec = specs
        .iter()
        .find(|s| s.name == name)
        .ok_or_else(|| format!("{}: missing feature {name}", m.path().display()))?;
    match &spec.kind {
        FeatureKind::MultiArray {
            shape: Some(s),
            dtype: Some(_),
        } => Ok(s.clone()),
        other => Err(format!(
            "{}: feature {name} is {other:?}, expected a fixed-shape multi-array",
            m.path().display()
        )),
    }
}

/// `shape` must have `rank` axes and size 1 on axis `batch_axis`.
fn expect_rank(name: &str, shape: &[usize], rank: usize, batch_axis: usize) -> Result<(), String> {
    if shape.len() != rank || shape[batch_axis] != 1 {
        return Err(format!(
            "{name} has shape {shape:?}, expected rank {rank} with size 1 on axis {batch_axis}"
        ));
    }
    Ok(())
}

fn read_dims(
    pre: &CoreMlModel,
    enc: &CoreMlModel,
    dec: &CoreMlModel,
    joint: &CoreMlModel,
) -> Result<Dims, String> {
    let audio = fixed_shape(pre, "audio_signal", true)?;
    let mel = fixed_shape(pre, "mel", false)?;
    let enc_in = fixed_shape(enc, "mel", true)?;
    let enc_out = fixed_shape(enc, "encoder", false)?;
    let h = fixed_shape(dec, "h_in", true)?;
    let dec_out = fixed_shape(dec, "decoder", false)?;
    let j_enc = fixed_shape(joint, "encoder_step", true)?;
    let j_dec = fixed_shape(joint, "decoder_step", true)?;
    expect_rank("audio_signal", &audio, 2, 0)?;
    expect_rank("mel", &mel, 3, 0)?;
    expect_rank("encoder", &enc_out, 3, 0)?;
    expect_rank("h_in", &h, 3, 1)?;
    expect_rank("decoder", &dec_out, 3, 0)?;
    let dims = Dims {
        window_samples: audio[1],
        mel_bins: mel[1],
        mel_frames: mel[2],
        encoder_hidden: enc_out[1],
        encoder_frames: enc_out[2],
        pred_layers: h[0],
        pred_hidden: h[2],
    };
    if enc_in != mel {
        return Err(format!(
            "encoder input {enc_in:?} does not match preprocessor output {mel:?}"
        ));
    }
    if dec_out != [1, dims.pred_hidden, 1] || j_dec != dec_out {
        return Err(format!(
            "decoder output {dec_out:?} and joint decoder_step {j_dec:?} disagree with hidden {}",
            dims.pred_hidden
        ));
    }
    if j_enc != [1, dims.encoder_hidden, 1] {
        return Err(format!(
            "joint encoder_step {j_enc:?} does not match encoder hidden {}",
            dims.encoder_hidden
        ));
    }
    if dims.window_samples.div_ceil(FRAME_SAMPLES) != dims.encoder_frames {
        return Err(format!(
            "{} samples per window do not give {} encoder frames at {FRAME_SAMPLES} samples per frame",
            dims.window_samples, dims.encoder_frames
        ));
    }
    if dims.window_samples <= OVERLAP_SAMPLES + FRAME_SAMPLES {
        return Err(format!(
            "window of {} samples is too short for the overlap",
            dims.window_samples
        ));
    }
    Ok(dims)
}

/// Parses `parakeet_vocab.json`: an object of `"id": "piece"` whose ids
/// must be exactly `0..n`. Blank is `n`.
fn read_vocabulary(path: &Path) -> Result<Vec<String>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut vocab = vec![None; map.len()];
    for (k, v) in map {
        let id: usize = k
            .parse()
            .map_err(|_| format!("{}: key {k} is not an id", path.display()))?;
        let piece = v
            .as_str()
            .ok_or_else(|| format!("{}: id {id} is not a string", path.display()))?;
        let slot = vocab
            .get_mut(id)
            .ok_or_else(|| format!("{}: ids are not contiguous", path.display()))?;
        if slot.replace(piece.to_string()).is_some() {
            return Err(format!("{}: duplicate id {id}", path.display()));
        }
    }
    vocab
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| format!("{}: ids are not contiguous", path.display()))
}

/// The export's `config.json`, or `None` when it has none. FluidInference's
/// is a few hundred bytes; the cap keeps a foreign file from being read
/// whole.
fn read_config(dir: &Path) -> Result<Option<serde_json::Value>, String> {
    use std::io::Read;
    let path = dir.join("config.json");
    let Ok(file) = std::fs::File::open(&path) else {
        return Ok(None);
    };
    let mut bytes = Vec::new();
    file.take((1 << 20) + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() > 1 << 20 {
        return Err(format!("{}: exceeds the 1 MiB limit", path.display()));
    }
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn check_contract(dir: &Path) -> Result<(), String> {
    let Some(v) = read_config(dir)? else {
        return Ok(());
    };
    match v.get("contract").and_then(|c| c.as_str()) {
        None => Ok(()),
        Some(KNOWN_CONTRACT) => Ok(()),
        Some(other) => Err(format!(
            "Core ML Parakeet contract {other:?} is not supported (expected {KNOWN_CONTRACT:?})"
        )),
    }
}

impl ParakeetCoreMl {
    pub fn open(dir: &Path, opts: &ParakeetCoreMlOptions) -> Result<Self, String> {
        let (major, _, _) = turbospark_coreml::os_version();
        // Ultra declares macOS 14, but only macOS 15 has run any export here.
        if major < 15 {
            return Err("Parakeet on Core ML requires macOS 15 or newer".into());
        }
        check_contract(dir)?;
        check_encoder_units(
            encoder_weight_format(dir)?.as_deref(),
            opts.encoder_units,
            std::env::var(ALLOW_UNVERIFIED_GPU_ENV).as_deref() == Ok("1"),
        )?;
        let vocabulary = read_vocabulary(&dir.join(VOCAB_FILE))?;
        let cache = super::cache_root(dir);
        let load = |name: &str, units: ComputeUnits| {
            CoreMlModel::open(&dir.join(name), &cache, &LoadOptions::new(units))
                .map_err(|e| e.to_string())
        };
        // The preprocessor runs on the CPU in FluidAudio too: it is a small
        // STFT whose output feeds every later stage, so it stays exact.
        let preprocessor = load(REQUIRED_DIRS[0], ComputeUnits::CpuOnly)?;
        let encoder = load(REQUIRED_DIRS[1], opts.encoder_units)?;
        let decoder = load(REQUIRED_DIRS[2], opts.step_units)?;
        let joint = load(REQUIRED_DIRS[3], opts.step_units)?;
        let dims = read_dims(&preprocessor, &encoder, &decoder, &joint)?;
        Ok(Self {
            preprocessor,
            encoder,
            decoder,
            joint,
            vocabulary,
            dims,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    pub fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<String, String> {
        Ok(self
            .decode(samples, sample_rate, &|| false, |_, _| true)?
            .text)
    }

    /// Transcribes mono PCM. `progress(window, total)` runs before each
    /// window and stops the run by returning `false`, as `cancelled` does.
    pub fn decode(
        &self,
        samples: &[f32],
        sample_rate: u32,
        cancelled: &dyn Fn() -> bool,
        mut progress: impl FnMut(usize, usize) -> bool,
    ) -> Result<AlignedResult, String> {
        if sample_rate != SAMPLE_RATE {
            return Err(format!(
                "Parakeet expects {SAMPLE_RATE} Hz PCM, received {sample_rate} Hz"
            ));
        }
        if samples.is_empty() || samples.iter().any(|s| !s.is_finite()) {
            return Err("Parakeet input must be nonempty finite mono PCM".into());
        }
        let plan = plan_windows(samples.len(), self.dims.window_samples);
        let overlap_seconds = plan.overlap as f32 / SAMPLE_RATE as f32;
        let mut merged: Vec<AlignedToken> = Vec::new();
        for (i, &start) in plan.starts.iter().enumerate() {
            if cancelled() || !progress(i, plan.starts.len()) {
                return Err("transcription cancelled".into());
            }
            let end = (start + self.dims.window_samples).min(samples.len());
            let tokens = self.window_tokens(&samples[start..end], start)?;
            merged = if i == 0 {
                tokens
            } else {
                merge_windows(merged, tokens, overlap_seconds)
            };
        }
        Ok(sentences_to_result(tokens_to_sentences(merged)))
    }

    /// Runs one window (at most `window_samples` long, zero-padded) and
    /// returns its tokens on the absolute timeline.
    fn window_tokens(
        &self,
        chunk: &[f32],
        start_sample: usize,
    ) -> Result<Vec<AlignedToken>, String> {
        let d = self.dims;
        let mut padded = vec![0.0f32; d.window_samples];
        padded[..chunk.len()].copy_from_slice(chunk);
        let audio = Tensor::f32(vec![1, d.window_samples], padded).map_err(err)?;
        let audio_len = Tensor::i32(vec![1], vec![chunk.len() as i32]).map_err(err)?;
        let pre = self
            .preprocessor
            .predict(&[("audio_signal", &audio), ("audio_length", &audio_len)])
            .map_err(err)?;
        let mel = take(&pre, "mel")?;
        let mel_len = take(&pre, "mel_length")?;
        let enc = self
            .encoder
            .predict(&[("mel", mel), ("mel_length", mel_len)])
            .map_err(err)?;
        let features = take(&enc, "encoder")?
            .as_f32()
            .ok_or("encoder output is not float32")?;
        let enc_len = take(&enc, "encoder_length")?
            .as_i32()
            .and_then(|v| v.first().copied())
            .ok_or("encoder_length is not int32")?;
        let frames = usize::try_from(enc_len).unwrap_or(0).min(d.encoder_frames);

        let stepper = CoreMlStepper {
            decoder: &self.decoder,
            joint: &self.joint,
            features,
            dims: d,
        };
        let blank = self.vocabulary.len();
        let emitted = tdt_greedy_redux(&stepper, frames, blank, &DURATIONS, MAX_SYMBOLS)?;
        let frame_seconds = FRAME_SAMPLES as f32 / SAMPLE_RATE as f32;
        let offset = start_sample as f32 / SAMPLE_RATE as f32;
        let mut accumulated_text = String::new();
        Ok(emitted
            .into_iter()
            .filter_map(|e| {
                let piece = &self.vocabulary[e.token];
                (!is_special_piece(piece)).then(|| {
                    let mut text = piece.replace('\u{2581}', " ");
                    adjust_sentencepiece_digit_spacing(&mut text, &accumulated_text);
                    accumulated_text.push_str(&text);
                    AlignedToken::new(
                        e.token as i32,
                        text,
                        offset + e.frame as f32 * frame_seconds,
                        e.duration as f32 * frame_seconds,
                    )
                })
            })
            .collect())
    }
}

/// Groups sentencepiece tokens into words: a piece that starts with a
/// space opens a new word. Returns `(text, start, end)` in seconds.
pub fn words(tokens: &[AlignedToken]) -> Vec<(String, f32, f32)> {
    let mut out: Vec<(String, f32, f32)> = Vec::new();
    for t in tokens {
        match out.last_mut() {
            Some(w) if !t.text.starts_with(' ') => {
                w.0.push_str(&t.text);
                w.2 = t.end;
            }
            _ => out.push((t.text.trim_start().to_string(), t.start, t.end)),
        }
    }
    out.retain(|w| !w.0.is_empty());
    out
}

fn err(e: turbospark_coreml::CoreMlError) -> String {
    e.to_string()
}

fn take<'a>(
    m: &'a std::collections::HashMap<String, Tensor>,
    name: &str,
) -> Result<&'a Tensor, String> {
    m.get(name)
        .ok_or_else(|| format!("Core ML output {name} is missing"))
}

/// Joins a window onto the transcript so far. The contiguous-run merge is
/// tried first and the subsequence merge is its fallback, the same order
/// mlx-audio's chunked Parakeet decode uses.
fn merge_windows(a: Vec<AlignedToken>, b: Vec<AlignedToken>, overlap: f32) -> Vec<AlignedToken> {
    match merge_longest_contiguous(a.clone(), b.clone(), overlap) {
        Ok(m) => m,
        Err(_) => merge_longest_common_subsequence(a, b, overlap)
            .expect("the subsequence merge falls back to a cutoff instead of failing"),
    }
}

#[derive(Debug, PartialEq, Eq)]
struct WindowPlan {
    starts: Vec<usize>,
    overlap: usize,
}

/// Window start samples. Starts are whole encoder frames apart, so every
/// window's frames land on the same 80 ms grid as the first.
fn plan_windows(len: usize, window: usize) -> WindowPlan {
    let hop = (window - OVERLAP_SAMPLES) / FRAME_SAMPLES * FRAME_SAMPLES;
    let mut starts = vec![0];
    while starts[starts.len() - 1] + window < len {
        starts.push(starts[starts.len() - 1] + hop);
    }
    WindowPlan {
        starts,
        overlap: window - hop,
    }
}

/// One step of the TDT decoder, abstracted so the loop is testable without
/// a model.
pub(crate) trait TdtStepper {
    type State;
    fn initial(&self) -> Self::State;
    /// Prediction-network output for `token` from `state`, and the state it
    /// advances to if `token` is accepted.
    fn predict(&self, token: usize, state: &Self::State)
        -> Result<(Vec<f32>, Self::State), String>;
    /// `(token, duration_index)` for encoder frame `frame`.
    fn joint(&self, frame: usize, prediction: &[f32]) -> Result<(usize, usize), String>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Emitted {
    pub token: usize,
    pub frame: usize,
    pub duration: usize,
}

/// Greedy TDT decode with Redux semantics. The prediction for the current
/// `(last token, state)` is computed once and reused until a token is
/// accepted, as the CPU decoder caches its proposal.
pub(crate) fn tdt_greedy_redux<S: TdtStepper>(
    s: &S,
    frames: usize,
    blank: usize,
    durations: &[usize],
    max_symbols: usize,
) -> Result<Vec<Emitted>, String> {
    let mut last = blank;
    let mut state = s.initial();
    let mut proposal: Option<(Vec<f32>, S::State)> = None;
    let mut time = 0usize;
    let mut out = Vec::new();
    for _ in 0..max_symbols.saturating_mul(frames) {
        if time >= frames {
            break;
        }
        if proposal.is_none() {
            proposal = Some(s.predict(last, &state)?);
        }
        let prediction = &proposal.as_ref().expect("set above").0;
        let (token, duration_index) = s.joint(time, prediction)?;
        let mut duration = *durations
            .get(duration_index)
            .ok_or_else(|| format!("duration index {duration_index} is out of range"))?;
        if token > blank {
            return Err(format!("token {token} is beyond blank {blank}"));
        }
        if token == blank {
            duration = duration.max(1);
        } else {
            last = token;
            state = proposal.take().expect("set above").1;
            out.push(Emitted {
                token,
                frame: time,
                duration,
            });
        }
        time = time.saturating_add(duration);
    }
    Ok(out)
}

struct CoreMlStepper<'a> {
    decoder: &'a CoreMlModel,
    joint: &'a CoreMlModel,
    /// Encoder output `[1, hidden, frames]`, channel-major.
    features: &'a [f32],
    dims: Dims,
}

impl TdtStepper for CoreMlStepper<'_> {
    type State = (Tensor, Tensor);

    fn initial(&self) -> Self::State {
        let shape = vec![self.dims.pred_layers, 1, self.dims.pred_hidden];
        (
            Tensor::zeros(turbospark_coreml::DType::F32, shape.clone()),
            Tensor::zeros(turbospark_coreml::DType::F32, shape),
        )
    }

    fn predict(
        &self,
        token: usize,
        state: &Self::State,
    ) -> Result<(Vec<f32>, Self::State), String> {
        let targets = Tensor::i32(vec![1, 1], vec![token as i32]).map_err(err)?;
        let len = Tensor::i32(vec![1], vec![1]).map_err(err)?;
        let mut out = self
            .decoder
            .predict(&[
                ("targets", &targets),
                ("target_length", &len),
                ("h_in", &state.0),
                ("c_in", &state.1),
            ])
            .map_err(err)?;
        let prediction = out
            .remove("decoder")
            .and_then(|t| t.to_f32_vec())
            .ok_or("decoder output is missing or not float")?;
        let h = out.remove("h_out").ok_or("decoder h_out is missing")?;
        let c = out.remove("c_out").ok_or("decoder c_out is missing")?;
        Ok((prediction, (h, c)))
    }

    fn joint(&self, frame: usize, prediction: &[f32]) -> Result<(usize, usize), String> {
        let (hidden, frames) = (self.dims.encoder_hidden, self.dims.encoder_frames);
        let column: Vec<f32> = (0..hidden)
            .map(|c| self.features[c * frames + frame])
            .collect();
        let enc = Tensor::f32(vec![1, hidden, 1], column).map_err(err)?;
        let dec =
            Tensor::f32(vec![1, self.dims.pred_hidden, 1], prediction.to_vec()).map_err(err)?;
        let out = self
            .joint
            .predict(&[("encoder_step", &enc), ("decoder_step", &dec)])
            .map_err(err)?;
        let scalar = |name: &str| -> Result<usize, String> {
            take(&out, name)?
                .as_i32()
                .and_then(|v| v.first().copied())
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| format!("joint output {name} is not a non-negative int32"))
        };
        Ok((scalar("token_id")?, scalar("duration")?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    /// Joint answers come from a script, keyed by call order; `predict`
    /// counts calls and threads the accepted token through the state.
    struct Scripted {
        joint: RefCell<std::vec::IntoIter<(usize, usize)>>,
        predicts: Cell<usize>,
        frames_seen: RefCell<Vec<usize>>,
    }

    impl Scripted {
        fn new(script: Vec<(usize, usize)>) -> Self {
            Self {
                joint: RefCell::new(script.into_iter()),
                predicts: Cell::new(0),
                frames_seen: RefCell::new(Vec::new()),
            }
        }
    }

    impl TdtStepper for Scripted {
        type State = Vec<usize>;
        fn initial(&self) -> Vec<usize> {
            Vec::new()
        }
        fn predict(
            &self,
            token: usize,
            state: &Vec<usize>,
        ) -> Result<(Vec<f32>, Vec<usize>), String> {
            self.predicts.set(self.predicts.get() + 1);
            let mut next = state.clone();
            next.push(token);
            Ok((vec![token as f32], next))
        }
        fn joint(&self, frame: usize, _p: &[f32]) -> Result<(usize, usize), String> {
            self.frames_seen.borrow_mut().push(frame);
            Ok(self.joint.borrow_mut().next().unwrap_or((9, 1)))
        }
    }

    const BLANK: usize = 9;

    #[test]
    fn blank_with_zero_duration_still_advances() {
        // Blank at duration index 0 (zero frames) must move one frame.
        let s = Scripted::new(vec![(BLANK, 0), (BLANK, 0), (BLANK, 0)]);
        let out = tdt_greedy_redux(&s, 3, BLANK, &DURATIONS, 10).unwrap();
        assert!(out.is_empty());
        assert_eq!(*s.frames_seen.borrow(), vec![0, 1, 2]);
        // Only the initial prediction: no token was accepted.
        assert_eq!(s.predicts.get(), 1);
    }

    #[test]
    fn tokens_can_stay_on_a_frame_and_re_predict_only_on_emission() {
        // Two tokens on frame 0 (duration 0), then one that jumps 2 frames.
        let s = Scripted::new(vec![(3, 0), (4, 0), (5, 2), (BLANK, 1)]);
        let out = tdt_greedy_redux(&s, 3, BLANK, &DURATIONS, 10).unwrap();
        assert_eq!(
            out,
            vec![
                Emitted {
                    token: 3,
                    frame: 0,
                    duration: 0
                },
                Emitted {
                    token: 4,
                    frame: 0,
                    duration: 0
                },
                Emitted {
                    token: 5,
                    frame: 0,
                    duration: 2
                },
            ]
        );
        assert_eq!(*s.frames_seen.borrow(), vec![0, 0, 0, 2]);
        // Initial plus one per accepted token.
        assert_eq!(s.predicts.get(), 4);
    }

    #[test]
    fn step_cap_stops_a_stuck_decoder() {
        // A decoder that emits forever on frame 0 stops at max_symbols * frames.
        let s = Scripted::new(vec![(1, 0); 100]);
        let out = tdt_greedy_redux(&s, 2, BLANK, &DURATIONS, 3).unwrap();
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn bad_joint_outputs_are_errors() {
        let s = Scripted::new(vec![(BLANK + 1, 1)]);
        assert!(tdt_greedy_redux(&s, 2, BLANK, &DURATIONS, 3).is_err());
        let s = Scripted::new(vec![(1, 7)]);
        assert!(tdt_greedy_redux(&s, 2, BLANK, &DURATIONS, 3).is_err());
    }

    #[test]
    fn windows_cover_input_on_the_frame_grid() {
        let w = 240_000;
        assert_eq!(plan_windows(1, w).starts, vec![0]);
        assert_eq!(plan_windows(w, w).starts, vec![0]);
        let p = plan_windows(w + 1, w);
        assert_eq!(p.starts.len(), 2);
        let hop = p.starts[1];
        assert_eq!(hop % FRAME_SAMPLES, 0);
        assert!(p.overlap >= OVERLAP_SAMPLES);
        assert!(p.overlap < OVERLAP_SAMPLES + FRAME_SAMPLES);
        // The last window always reaches the end of the input.
        for len in [w + 1, 3 * w, 10 * w + 12_345] {
            let p = plan_windows(len, w);
            assert!(p.starts.last().unwrap() + w >= len);
            assert!(p.starts.windows(2).all(|s| s[1] - s[0] == hop));
        }
    }

    #[test]
    fn merge_drops_duplicated_overlap_tokens() {
        let t = |id: i32, start: f32| AlignedToken::new(id, format!(" w{id}"), start, 0.08);
        let a = vec![t(1, 10.0), t(2, 12.5), t(3, 13.0), t(4, 13.5)];
        let b = vec![t(2, 12.5), t(3, 13.0), t(4, 13.5), t(5, 14.0)];
        let ids: Vec<i32> = merge_windows(a, b, 2.04).iter().map(|t| t.id).collect();
        assert_eq!(ids, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn words_join_continuation_pieces() {
        let t = |s: &str, start: f32| AlignedToken::new(0, s, start, 0.08);
        let w = words(&[
            t(" Hel", 0.0),
            t("lo", 0.08),
            t(" world", 0.16),
            t(".", 0.24),
        ]);
        assert_eq!(
            w,
            vec![
                ("Hello".to_string(), 0.0, 0.16),
                ("world.".to_string(), 0.16, 0.32)
            ]
        );
    }

    #[test]
    fn encoder_units_follow_weight_format() {
        use ComputeUnits::{All, CpuAndGpu, CpuAndNeuralEngine, CpuOnly};
        let redux = GPU_VERIFIED_WEIGHT_FORMATS[0];
        let ultra = "int8_linear_per_channel";
        assert_eq!(default_encoder_units(None), CpuAndGpu);
        assert_eq!(default_encoder_units(Some(redux)), CpuAndGpu);
        assert_eq!(default_encoder_units(Some(ultra)), CpuAndNeuralEngine);
        assert_eq!(
            default_encoder_units(Some("palettized_lut6")),
            CpuAndNeuralEngine
        );
        // GPU placement of an unmeasured format is refused, not attempted:
        // the failure mode is a process abort.
        for units in [CpuAndGpu, All] {
            assert!(check_encoder_units(Some(ultra), units, false).is_err());
            assert!(check_encoder_units(Some(ultra), units, true).is_ok());
            assert!(check_encoder_units(Some(redux), units, false).is_ok());
            assert!(check_encoder_units(None, units, false).is_ok());
        }
        for units in [CpuAndNeuralEngine, CpuOnly] {
            assert!(check_encoder_units(Some(ultra), units, false).is_ok());
        }
    }

    #[test]
    fn weight_format_is_read_from_config() {
        let d = std::env::temp_dir().join(format!(
            "ts-parakeet-format-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        assert_eq!(encoder_weight_format(&d).unwrap(), None);
        std::fs::write(
            d.join("config.json"),
            r#"{"contract":"parakeet-tdt-0.6b-v3-coreml","encoders":{"Encoder.mlmodelc":{"weight_format":"int8_linear_per_channel"}}}"#,
        )
        .unwrap();
        assert_eq!(
            encoder_weight_format(&d).unwrap().as_deref(),
            Some("int8_linear_per_channel")
        );
        let opts = ParakeetCoreMlOptions::for_model(&d).unwrap();
        if std::env::var_os(ENCODER_UNITS_ENV).is_none() {
            assert_eq!(opts.encoder_units, ComputeUnits::CpuAndNeuralEngine);
        }
        std::fs::write(
            d.join("config.json"),
            r#"{"contract":"parakeet-tdt-0.6b-v3-coreml"}"#,
        )
        .unwrap();
        assert_eq!(encoder_weight_format(&d).unwrap(), None);
        std::fs::write(d.join("config.json"), "{").unwrap();
        assert!(encoder_weight_format(&d).is_err());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn vocabulary_must_be_contiguous() {
        let d = std::env::temp_dir().join(format!(
            "ts-parakeet-vocab-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join(VOCAB_FILE);
        std::fs::write(&p, r#"{"0":"<unk>","1":"▁a"}"#).unwrap();
        assert_eq!(read_vocabulary(&p).unwrap(), vec!["<unk>", "\u{2581}a"]);
        std::fs::write(&p, r#"{"0":"<unk>","2":"b"}"#).unwrap();
        assert!(read_vocabulary(&p).is_err());
        std::fs::write(&p, r#"{"0":1}"#).unwrap();
        assert!(read_vocabulary(&p).is_err());
        std::fs::write(d.join("config.json"), r#"{"contract":"other"}"#).unwrap();
        assert!(check_contract(&d).is_err());
        std::fs::write(
            d.join("config.json"),
            format!(r#"{{"contract":"{KNOWN_CONTRACT}"}}"#),
        )
        .unwrap();
        assert!(check_contract(&d).is_ok());
        std::fs::remove_dir_all(&d).unwrap();
    }
}
