//! Parakeet Unified EN 0.6B streaming on Core ML (FluidInference
//! `parakeet-unified-en-0.6b-coreml`): English with punctuation and case.
//!
//! The install holds one streaming encoder bundle,
//! `parakeet_unified_encoder_streaming_<left>_<chunk>_<right>[_int8]`, whose
//! chunked-attention mask fixes the latency tier, plus the RNNT prediction
//! network (`parakeet_unified_decoder`) and the single-step joint with argmax
//! (`parakeet_unified_joint_decision_single_step`). The log-mel front end is
//! computed here with `audio::nemo_mel` (NeMo `per_feature` normalization
//! over the window's valid frames), as FluidAudio does instead of shipping
//! the flexible-shape Core ML preprocessor that would pin the CPU. Windowing
//! and the RNNT loop are in [`super::parakeet_unified_stream`].
//!
//! Placement: an int8 encoder runs on CPU and Neural Engine only. FluidAudio
//! (`StreamingUnifiedAsrManager`) reports that Core ML hands int8 ops to
//! MPSGraph under GPU placement, which fails its MLIR pass and aborts the
//! process; the same-format Parakeet Ultra encoder did not abort on an M4
//! Max but ran about 7x slower on the GPU than on the Neural Engine. The
//! decoder and joint run once per token on tiny inputs, on the CPU.

use std::path::Path;

use audio::nemo_mel::{nemo_log_mel_spectrogram, NemoMelOptions};
use turbospark_coreml::{ComputeUnits, CoreMlModel, DType, FeatureKind, LoadOptions, Tensor};

use super::parakeet_unified_stream::{
    EncodedWindow, Geometry, RnntStepper, StreamingModel, FRAME_SAMPLES,
};

pub const CONFIG: &str = "config.json";
pub const VOCAB: &str = "vocab.json";
pub const DECODER: &str = "parakeet_unified_decoder.mlmodelc";
pub const JOINT: &str = "parakeet_unified_joint_decision_single_step.mlmodelc";
pub const ENCODER_PREFIX: &str = "parakeet_unified_encoder_streaming_";
/// `model_id` in the export's `config.json`; anything else is refused.
pub const MODEL_ID: &str = "nvidia/parakeet-unified-en-0.6b";
const HOP: usize = 160;
const MEL_BINS: usize = 128;

const ENCODER_UNITS_ENV: &str = "TURBOSPARK_PARAKEET_UNIFIED_ENCODER_UNITS";
const ALLOW_UNVERIFIED_GPU_ENV: &str = "TURBOSPARK_COREML_ALLOW_QUANTIZED_GPU";

/// A streaming encoder bundle found in an install.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderBundle {
    pub dir_name: String,
    pub geometry: Geometry,
    pub int8: bool,
}

/// Parses `parakeet_unified_encoder_streaming_70_7_7_int8.mlmodelc`.
pub fn parse_encoder_bundle(name: &str) -> Option<EncoderBundle> {
    let stem = name
        .strip_prefix(ENCODER_PREFIX)?
        .strip_suffix(".mlmodelc")?;
    let (sizes, int8) = match stem.strip_suffix("_int8") {
        Some(s) => (s, true),
        None => (stem, false),
    };
    let mut parts = sizes.split('_').map(|p| p.parse::<usize>().ok());
    let (left, chunk, right) = (parts.next()??, parts.next()??, parts.next()??);
    if parts.next().is_some() || chunk == 0 {
        return None;
    }
    Some(EncoderBundle {
        dir_name: name.to_owned(),
        geometry: Geometry { left, chunk, right },
        int8,
    })
}

/// Every streaming encoder bundle in `dir`, sorted by name.
pub fn encoder_bundles(dir: &Path) -> Vec<EncoderBundle> {
    let mut out: Vec<EncoderBundle> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| parse_encoder_bundle(&e.file_name().to_string_lossy()))
        .collect();
    out.sort_by(|a, b| a.dir_name.cmp(&b.dir_name));
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnifiedOptions {
    pub encoder_units: ComputeUnits,
}

impl Default for UnifiedOptions {
    fn default() -> Self {
        Self {
            encoder_units: ComputeUnits::CpuAndNeuralEngine,
        }
    }
}

impl UnifiedOptions {
    /// Defaults with `TURBOSPARK_PARAKEET_UNIFIED_ENCODER_UNITS` (`cpu`,
    /// `cpu_gpu`, `cpu_ane`, `all`) applied; a GPU choice is checked at open.
    pub fn from_env() -> Result<Self, String> {
        let mut o = Self::default();
        if let Ok(v) = std::env::var(ENCODER_UNITS_ENV) {
            o.encoder_units = ComputeUnits::parse(v.trim()).ok_or_else(|| {
                format!("{ENCODER_UNITS_ENV}={v} is not one of cpu, cpu_gpu, cpu_ane, all")
            })?;
        }
        Ok(o)
    }
}

/// Refuses GPU-capable placement for an int8 encoder unless the subprocess
/// probe flag allows it: the failure mode there is a process abort.
fn check_units(int8: bool, units: ComputeUnits, allow: bool) -> Result<(), String> {
    if int8 && matches!(units, ComputeUnits::CpuAndGpu | ComputeUnits::All) && !allow {
        return Err(
            "the int8 Parakeet Unified encoder must not run on the GPU, where Core ML can abort \
             the process; use cpu_ane or cpu"
                .into(),
        );
    }
    Ok(())
}

pub struct ParakeetUnifiedCoreMl {
    encoder: CoreMlModel,
    decoder: CoreMlModel,
    joint: CoreMlModel,
    vocabulary: Vec<String>,
    bundle: EncoderBundle,
    blank: usize,
    mel_frames: usize,
    hidden: usize,
    pred_layers: usize,
    pred_hidden: usize,
}

impl std::fmt::Debug for ParakeetUnifiedCoreMl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParakeetUnifiedCoreMl")
            .field("bundle", &self.bundle)
            .field("vocabulary", &self.vocabulary.len())
            .finish()
    }
}

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn fixed_shape(m: &CoreMlModel, name: &str, inputs: bool) -> Result<Vec<usize>, String> {
    let specs = if inputs { m.inputs() } else { m.outputs() };
    let spec = specs
        .iter()
        .find(|s| s.name == name)
        .ok_or_else(|| format!("{}: missing feature {name}", m.path().display()))?;
    match &spec.kind {
        FeatureKind::MultiArray { shape: Some(s), .. } => Ok(s.clone()),
        other => Err(format!(
            "{}: feature {name} is {other:?}, expected a fixed-shape multi-array",
            m.path().display()
        )),
    }
}

fn read_json(path: &Path) -> Result<serde_json::Value, String> {
    use std::io::Read;
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut bytes = Vec::new();
    file.take((1 << 20) + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() > 1 << 20 {
        return Err(format!("{}: exceeds the 1 MiB limit", path.display()));
    }
    serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// `vocab.json` as an id-to-piece array or an `{"id": "piece"}` object;
/// ids must be contiguous from zero.
fn read_vocabulary(path: &Path) -> Result<Vec<String>, String> {
    let value = read_json(path)?;
    let pieces: Vec<Option<String>> = match value {
        serde_json::Value::Array(items) => items
            .into_iter()
            .map(|v| v.as_str().map(str::to_owned))
            .collect(),
        serde_json::Value::Object(map) => {
            let mut v = vec![None; map.len()];
            for (k, piece) in map {
                let id: usize = k
                    .parse()
                    .map_err(|_| format!("{}: key {k:?} is not an id", path.display()))?;
                *v.get_mut(id)
                    .ok_or_else(|| format!("{}: ids are not contiguous", path.display()))? =
                    piece.as_str().map(str::to_owned);
            }
            v
        }
        _ => return Err(format!("{}: not an array or object", path.display())),
    };
    pieces
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| format!("{}: a piece is missing or not a string", path.display()))
}

fn config_usize(config: &serde_json::Value, key: &str) -> Result<usize, String> {
    config
        .get(key)
        .and_then(|v| v.as_u64())
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| format!("config.json: {key} is missing or not a count"))
}

impl ParakeetUnifiedCoreMl {
    pub fn open(dir: &Path, opts: &UnifiedOptions) -> Result<Self, String> {
        let (major, _, _) = turbospark_coreml::os_version();
        if major < 15 {
            return Err("Parakeet Unified on Core ML requires macOS 15 or newer".into());
        }
        let config = read_json(&dir.join(CONFIG))?;
        let model_id = config.get("model_id").and_then(|v| v.as_str());
        if model_id != Some(MODEL_ID) {
            return Err(format!(
                "config.json model_id {model_id:?} is not {MODEL_ID:?}; another export may use a \
                 different vocabulary or graph"
            ));
        }
        if config_usize(&config, "sample_rate")? != 16_000
            || config_usize(&config, "subsampling_factor")? * HOP != FRAME_SAMPLES
            || config_usize(&config, "feature_dim")? != MEL_BINS
        {
            return Err("config.json sample rate, subsampling, or features differ from the 16 kHz, x8, 128-mel export this runner supports".into());
        }
        let blank = config_usize(&config, "blank_idx")?;
        let vocabulary = read_vocabulary(&dir.join(VOCAB))?;
        if vocabulary.len() != config_usize(&config, "vocab_size")? || blank != vocabulary.len() {
            return Err(format!(
                "vocab.json has {} pieces; config.json expects vocab_size and blank_idx to match",
                vocabulary.len()
            ));
        }
        let bundles = encoder_bundles(dir);
        let bundle = match bundles.as_slice() {
            [one] => one.clone(),
            [] => return Err(format!("{}: no {ENCODER_PREFIX}* bundle", dir.display())),
            many => {
                return Err(format!(
                    "{}: {} streaming encoders; install exactly one",
                    dir.display(),
                    many.len()
                ))
            }
        };
        check_units(
            bundle.int8,
            opts.encoder_units,
            std::env::var(ALLOW_UNVERIFIED_GPU_ENV).as_deref() == Ok("1"),
        )?;

        let cache = super::cache_root(dir);
        let load = |name: &str, units: ComputeUnits| {
            CoreMlModel::open(&dir.join(name), &cache, &LoadOptions::new(units)).map_err(err)
        };
        let encoder = load(&bundle.dir_name, opts.encoder_units)?;
        let decoder = load(DECODER, ComputeUnits::CpuOnly)?;
        let joint = load(JOINT, ComputeUnits::CpuOnly)?;

        // The bundle name fixes the window; its mel input must agree.
        let window = bundle.geometry.window_samples();
        let mel = fixed_shape(&encoder, "mel", true)?;
        if mel.len() != 3 || mel[0] != 1 || mel[1] != MEL_BINS || mel[2] != window / HOP + 1 {
            return Err(format!(
                "{}: mel input {mel:?} does not match its name (expected [1, {MEL_BINS}, {}])",
                bundle.dir_name,
                window / HOP + 1
            ));
        }
        let enc = fixed_shape(&encoder, "encoder", false)?;
        if enc.len() != 3 || enc[0] != 1 {
            return Err(format!("encoder output has shape {enc:?}"));
        }
        let h = fixed_shape(&decoder, "h_in", true)?;
        if h.len() != 3 || h[1] != 1 {
            return Err(format!("decoder h_in has shape {h:?}"));
        }
        let step = fixed_shape(&joint, "encoder_step", true)?;
        let pred = fixed_shape(&joint, "decoder_step", true)?;
        if step != [1, enc[1], 1] || pred != [1, h[2], 1] {
            return Err(format!(
                "joint inputs {step:?} and {pred:?} do not match encoder {enc:?} and decoder {h:?}"
            ));
        }
        Ok(Self {
            encoder,
            decoder,
            joint,
            vocabulary,
            bundle,
            blank,
            mel_frames: mel[2],
            hidden: enc[1],
            pred_layers: h[0],
            pred_hidden: h[2],
        })
    }

    pub fn bundle(&self) -> &EncoderBundle {
        &self.bundle
    }
}

/// Channel-major `[128 * frames]` log-mel for one zero-padded window and
/// the count of valid frames, NeMo `per_feature` over the valid frames.
pub fn unified_mel(window: &[f32], valid: usize) -> Result<(Vec<f32>, usize), String> {
    let options = NemoMelOptions {
        pad_to: window.len(),
        normalize_valid_frames: true,
        ..NemoMelOptions::default()
    };
    // Only the valid samples go in, so `pad_to` zero-fills the rest and the
    // statistics come from `floor(valid / hop)` frames, as FluidAudio's
    // `UnifiedMelExtractor` computes them.
    let rows = nemo_log_mel_spectrogram(&window[..valid], &options).map_err(err)?;
    let frames = rows.len();
    let mut out = vec![0.0f32; MEL_BINS * frames];
    for (t, row) in rows.iter().enumerate() {
        for (m, value) in row.iter().enumerate() {
            out[m * frames + t] = *value;
        }
    }
    Ok((out, (valid / HOP).min(frames)))
}

impl RnntStepper for ParakeetUnifiedCoreMl {
    type State = (Tensor, Tensor);

    fn initial(&self) -> Result<Self::State, String> {
        let shape = vec![self.pred_layers, 1, self.pred_hidden];
        Ok((
            Tensor::zeros(DType::F32, shape.clone()),
            Tensor::zeros(DType::F32, shape),
        ))
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

    fn joint(&self, encoder_step: &[f32], prediction: &[f32]) -> Result<(usize, f32), String> {
        let enc = Tensor::f32(vec![1, self.hidden, 1], encoder_step.to_vec()).map_err(err)?;
        let dec = Tensor::f32(vec![1, self.pred_hidden, 1], prediction.to_vec()).map_err(err)?;
        let out = self
            .joint
            .predict(&[("encoder_step", &enc), ("decoder_step", &dec)])
            .map_err(err)?;
        let token = out
            .get("token_id")
            .and_then(|t| t.as_i32())
            .and_then(|v| v.first().copied())
            .and_then(|v| usize::try_from(v).ok())
            .ok_or("joint token_id is missing or not a non-negative int32")?;
        if token > self.blank {
            return Err(format!(
                "joint token_id {token} is past the blank id {}",
                self.blank
            ));
        }
        let probability = out
            .get("token_prob")
            .and_then(|t| t.to_f32_vec())
            .and_then(|v| v.first().copied())
            .unwrap_or(0.0);
        Ok((token, probability))
    }
}

impl StreamingModel for ParakeetUnifiedCoreMl {
    fn geometry(&self) -> Geometry {
        self.bundle.geometry
    }

    fn blank(&self) -> usize {
        self.blank
    }

    fn encode(&self, window: &[f32], valid: usize) -> Result<EncodedWindow, String> {
        let (mel, mel_valid) = unified_mel(window, valid)?;
        if mel.len() != MEL_BINS * self.mel_frames {
            return Err(format!(
                "log-mel has {} frames; the encoder takes {}",
                mel.len() / MEL_BINS,
                self.mel_frames
            ));
        }
        let mel = Tensor::f32(vec![1, MEL_BINS, self.mel_frames], mel).map_err(err)?;
        let mel_len = Tensor::i32(vec![1], vec![mel_valid as i32]).map_err(err)?;
        let out = self
            .encoder
            .predict(&[("mel", &mel), ("mel_length", &mel_len)])
            .map_err(err)?;
        let data = out
            .get("encoder")
            .and_then(|t| t.to_f32_vec())
            .ok_or("encoder output is missing or not float")?;
        let frames = data.len() / self.hidden;
        let valid = out
            .get("encoder_length")
            .and_then(|t| t.as_i32())
            .and_then(|v| v.first().copied())
            .and_then(|v| usize::try_from(v).ok())
            .ok_or("encoder_length is missing or not int32")?;
        Ok(EncodedWindow {
            hidden: self.hidden,
            frames,
            valid: valid.min(frames),
            data,
        })
    }

    fn piece(&self, token: usize) -> Option<&str> {
        let piece = self.vocabulary.get(token)?;
        (!audio::stt::parakeet::is_special_piece(piece)).then_some(piece.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_bundle_names_parse() {
        let b = parse_encoder_bundle("parakeet_unified_encoder_streaming_70_7_7_int8.mlmodelc")
            .unwrap();
        assert_eq!(
            b.geometry,
            Geometry {
                left: 70,
                chunk: 7,
                right: 7
            }
        );
        assert!(b.int8);
        let b =
            parse_encoder_bundle("parakeet_unified_encoder_streaming_70_13_13.mlmodelc").unwrap();
        assert!(!b.int8);
        assert!((b.geometry.latency_seconds() - 2.08).abs() < 1e-9);
        for bad in [
            "parakeet_unified_encoder.mlmodelc",
            "parakeet_unified_encoder_streaming_70_7.mlmodelc",
            "parakeet_unified_encoder_streaming_70_0_7.mlmodelc",
            "parakeet_unified_encoder_streaming_70_7_7_8.mlmodelc",
            "parakeet_unified_encoder_streaming_70_7_7_int8",
        ] {
            assert_eq!(parse_encoder_bundle(bad), None, "{bad}");
        }
    }

    #[test]
    fn int8_encoder_is_kept_off_the_gpu() {
        use ComputeUnits::{All, CpuAndGpu, CpuAndNeuralEngine, CpuOnly};
        for units in [CpuAndGpu, All] {
            assert!(check_units(true, units, false).is_err());
            assert!(check_units(true, units, true).is_ok());
            assert!(check_units(false, units, false).is_ok());
        }
        for units in [CpuAndNeuralEngine, CpuOnly] {
            assert!(check_units(true, units, false).is_ok());
        }
        assert_eq!(UnifiedOptions::default().encoder_units, CpuAndNeuralEngine);
    }

    #[test]
    fn mel_matches_the_encoder_input_and_zeroes_padding() {
        let g = Geometry {
            left: 70,
            chunk: 7,
            right: 7,
        };
        let window_len = g.window_samples();
        let mut window = vec![0.0f32; window_len];
        let valid = 20_000;
        for (i, s) in window[..valid].iter_mut().enumerate() {
            *s = (i as f32 * 0.05).sin() * 0.3;
        }
        let (mel, mel_valid) = unified_mel(&window, valid).unwrap();
        let frames = window_len / HOP + 1;
        assert_eq!(frames, 673, "70_7_7 encoder input");
        assert_eq!(mel.len(), MEL_BINS * frames);
        assert_eq!(mel_valid, valid / HOP);
        for m in 0..MEL_BINS {
            for t in mel_valid..frames {
                assert_eq!(mel[m * frames + t], 0.0, "padded frame {t} bin {m}");
            }
            let mean: f32 =
                (0..mel_valid).map(|t| mel[m * frames + t]).sum::<f32>() / mel_valid as f32;
            assert!(mean.abs() < 1e-3, "bin {m} is not centered: {mean}");
        }
    }

    #[test]
    fn vocabulary_accepts_array_or_contiguous_object() {
        let d = std::env::temp_dir().join(format!(
            "ts-unified-vocab-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join(VOCAB);
        std::fs::write(&p, r#"["<unk>","▁a","b"]"#).unwrap();
        assert_eq!(read_vocabulary(&p).unwrap(), ["<unk>", "\u{2581}a", "b"]);
        std::fs::write(&p, r#"{"1":"▁a","0":"<unk>"}"#).unwrap();
        assert_eq!(read_vocabulary(&p).unwrap(), ["<unk>", "\u{2581}a"]);
        std::fs::write(&p, r#"{"0":"<unk>","2":"b"}"#).unwrap();
        assert!(read_vocabulary(&p).is_err());
        std::fs::write(&p, r#"[1]"#).unwrap();
        assert!(read_vocabulary(&p).is_err());
        std::fs::remove_dir_all(&d).unwrap();
    }
}
