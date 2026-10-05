//! Parakeet TDT speech recognition.
//!
//! Reference: `mlx_audio/stt/models/parakeet/` at mlx-audio 0.5.7,
//! commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`. The port shares
//! the NeMo FastConformer encoder with Sortformer and implements Parakeet's
//! recurrent transducer predictor, joint token/duration head, and greedy
//! TDT decode loop. The MLX-community v2 and v3 checkpoints use the same
//! encoder and checkpoint tensor layout.

use std::path::Path;

use serde_json::Value;
use turbospark_audio::nemo_mel::{nemo_log_mel_spectrogram, NemoMelNormalization, NemoMelOptions};
use turbospark_audio::stft::StftOptions;
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::models::vad::sortformer::{FastConformer, FcEncoderConfig};
use crate::ops;
use crate::{Result, SpeechError};

mod redux;

/// Immutable upstream MLX reference profiles. Redux uses a distinct ternary
/// weight format that the Rust loader expands to temporary F32 tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParakeetProfile {
    pub name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
}

pub const PARAKEET_TDT_V2: ParakeetProfile = ParakeetProfile {
    name: "Parakeet TDT v2",
    repository: "mlx-community/parakeet-tdt-0.6b-v2",
    revision: "8ae155301e23d820d82aa60d24817c900e69e487",
};

pub const PARAKEET_TDT_V3: ParakeetProfile = ParakeetProfile {
    name: "Parakeet TDT v3",
    repository: "mlx-community/parakeet-tdt-0.6b-v3",
    revision: "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15",
};

pub const PARAKEET_REDUX: ParakeetProfile = ParakeetProfile {
    name: "Parakeet Redux",
    repository: "moondream/parakeet-redux",
    revision: "2bf128600aac4b16946f7ed8372e56117fe5e23b",
};

/// Parsed config for the supported NeMo FastConformer-RNNT-TDT layout.
#[derive(Debug, Clone)]
pub struct ParakeetConfig {
    pub sample_rate: u32,
    pub mel: NemoMelOptions,
    pub encoder: FcEncoderConfig,
    pub subsampling_factor: usize,
    pub vocab_size: usize,
    pub decoder_hidden: usize,
    pub decoder_layers: usize,
    pub joint_hidden: usize,
    pub num_extra_outputs: usize,
    pub durations: Vec<usize>,
    pub max_symbols: usize,
    pub vocabulary: Vec<String>,
    redux_mode: bool,
}

fn bad(field: impl Into<String>, why: impl Into<String>) -> SpeechError {
    SpeechError::BadConfig {
        field: field.into(),
        why: why.into(),
    }
}

fn required<'a>(v: &'a Value, key: &str) -> Result<&'a Value> {
    v.get(key)
        .ok_or_else(|| bad(key, "missing from config.json"))
}

fn usize_field(v: &Value, key: &str) -> Result<usize> {
    required(v, key)?
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| bad(key, "must be a positive integer fitting usize"))
}

fn bool_field(v: &Value, key: &str) -> Result<bool> {
    required(v, key)?
        .as_bool()
        .ok_or_else(|| bad(key, "must be a boolean"))
}

fn text_field(v: &Value, key: &str) -> Result<String> {
    required(v, key)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| bad(key, "must be a string"))
}

fn symmetric_hann(size: usize) -> Vec<f32> {
    if size <= 1 {
        return vec![1.0; size];
    }
    (0..size)
        .map(|i| {
            (0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (size - 1) as f64).cos())) as f32
        })
        .collect()
}

impl ParakeetConfig {
    /// Parses the pinned MLX checkpoint's NeMo-derived `config.json`.
    /// Unsupported graph switches fail during open rather than silently
    /// selecting a similar but incorrect encoder.
    pub fn from_json(v: &Value) -> Result<Self> {
        Self::parse(v, false)
    }

    fn parse(v: &Value, redux_mode: bool) -> Result<Self> {
        let sample_rate = u32::try_from(usize_field(required(v, "preprocessor")?, "sample_rate")?)
            .map_err(|_| bad("preprocessor.sample_rate", "does not fit u32"))?;
        let pre = required(v, "preprocessor")?;
        let enc = required(v, "encoder")?;
        let dec = required(v, "decoder")?;
        let pred = required(dec, "prednet")?;
        let joint = required(v, "joint")?;
        let jointnet = required(joint, "jointnet")?;
        let decoding = required(v, "decoding")?;

        if sample_rate != 16_000 {
            return Err(bad("preprocessor.sample_rate", "only 16 kHz is supported"));
        }
        for (key, expected) in [("window", "hann"), ("normalize", "per_feature")] {
            if text_field(pre, key)? != expected {
                return Err(bad(
                    format!("preprocessor.{key}"),
                    format!("only {expected} is supported"),
                ));
            }
        }
        if !bool_field(pre, "log")? || usize_field(pre, "frame_splicing")? != 1 {
            return Err(bad(
                "preprocessor",
                "log-mel features and frame_splicing=1 are required",
            ));
        }
        if text_field(enc, "subsampling")? != "dw_striding"
            || text_field(enc, "self_attention_model")? != "rel_pos"
            || text_field(enc, "conv_norm_type")? != "batch_norm"
            || enc.get("causal_downsampling").and_then(Value::as_bool) == Some(true)
        {
            return Err(SpeechError::Unsupported {
                why: "Parakeet requires non-causal depthwise-striding relative-position FastConformer with batch normalization".into(),
            });
        }
        if enc.get("reduction").is_some_and(|x| !x.is_null())
            || enc.get("reduction_position").is_some_and(|x| !x.is_null())
            || enc.get("conv_context_size").is_some_and(|x| !x.is_null())
        {
            return Err(SpeechError::Unsupported {
                why:
                    "Parakeet encoder reduction and convolution context variants are not supported"
                        .into(),
            });
        }
        let subsampling_factor = usize_field(enc, "subsampling_factor")?;
        let feature_size = usize_field(pre, "features")?;
        let n_fft = usize_field(pre, "n_fft")?;
        let window_samples = ((required(pre, "window_size")?
            .as_f64()
            .ok_or_else(|| bad("preprocessor.window_size", "must be numeric"))?
            * f64::from(sample_rate)) as usize)
            .max(1);
        let hop = ((required(pre, "window_stride")?
            .as_f64()
            .ok_or_else(|| bad("preprocessor.window_stride", "must be numeric"))?
            * f64::from(sample_rate)) as usize)
            .max(1);
        if n_fft < window_samples || n_fft % 2 != 0 {
            return Err(bad(
                "preprocessor.n_fft",
                "must be even and at least the analysis-window length",
            ));
        }
        let mel = NemoMelOptions {
            stft: StftOptions {
                fft_size: n_fft,
                hop,
                window: symmetric_hann(window_samples),
                center: true,
            },
            sample_rate,
            num_mels: feature_size,
            normalize: NemoMelNormalization::PerFeature,
            preemphasis: pre.get("preemph").and_then(Value::as_f64).unwrap_or(0.97) as f32,
            log_zero_guard_value: pre
                .get("log_zero_guard_value")
                .and_then(Value::as_f64)
                .unwrap_or(2.0f64.powi(-24)) as f32,
            pad_to: pre.get("pad_to").and_then(Value::as_u64).unwrap_or(0) as usize,
            pad_value: pre.get("pad_value").and_then(Value::as_f64).unwrap_or(0.0) as f32,
            normalize_valid_frames: pre
                .get("normalize_valid_frames")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        };

        let hidden_size = usize_field(enc, "d_model")?;
        let heads = usize_field(enc, "n_heads")?;
        let ff_expansion = usize_field(enc, "ff_expansion_factor")?;
        let conv_kernel_size = usize_field(enc, "conv_kernel_size")?;
        let encoder = FcEncoderConfig {
            hidden_size,
            num_hidden_layers: usize_field(enc, "n_layers")?,
            num_attention_heads: heads,
            intermediate_size: hidden_size
                .checked_mul(ff_expansion)
                .ok_or_else(|| bad("encoder.ff_expansion_factor", "dimension overflow"))?,
            num_mel_bins: feature_size,
            conv_kernel_size,
            subsampling_conv_channels: usize_field(enc, "subsampling_conv_channels")?,
            subsampling_conv_kernel_size: 3,
            subsampling_conv_stride: 2,
            attention_bias: bool_field(enc, "use_bias")?,
            scale_input: bool_field(enc, "xscaling")?,
        };
        if subsampling_factor != 8
            || heads == 0
            || hidden_size % heads != 0
            || conv_kernel_size % 2 == 0
            || !matches!(feature_size, 80 | 128)
        {
            return Err(SpeechError::Unsupported {
                why: "only an 8x, odd-kernel FastConformer with 80 or 128 mel bins is supported"
                    .into(),
            });
        }
        if encoder.attention_bias {
            return Err(SpeechError::Unsupported {
                why: "Parakeet's MLX checkpoints use bias-free Conformer linear layers".into(),
            });
        }
        if !bool_field(dec, "blank_as_pad")? || text_field(decoding, "model_type")? != "tdt" {
            return Err(SpeechError::Unsupported {
                why: "only blank-as-pad TDT decoding is supported".into(),
            });
        }

        let vocab_size = usize_field(dec, "vocab_size")?;
        let decoder_hidden = usize_field(pred, "pred_hidden")?;
        let decoder_layers = usize_field(pred, "pred_rnn_layers")?;
        let joint_hidden = usize_field(jointnet, "joint_hidden")?;
        let num_extra_outputs = joint
            .get("num_extra_outputs")
            .and_then(Value::as_u64)
            .and_then(|x| usize::try_from(x).ok())
            .ok_or_else(|| bad("joint.num_extra_outputs", "must be an integer"))?;
        let vocabulary: Vec<String> = required(joint, "vocabulary")?
            .as_array()
            .ok_or_else(|| bad("joint.vocabulary", "must be an array"))?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| bad("joint.vocabulary", "entries must be strings"))
            })
            .collect::<Result<_>>()?;
        let durations: Vec<usize> = required(decoding, "durations")?
            .as_array()
            .ok_or_else(|| bad("decoding.durations", "must be an array"))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|x| usize::try_from(x).ok())
                    .ok_or_else(|| bad("decoding.durations", "entries must be unsigned integers"))
            })
            .collect::<Result<_>>()?;
        let max_symbols = decoding
            .get("greedy")
            .and_then(|x| x.get("max_symbols"))
            .and_then(Value::as_u64)
            .and_then(|x| usize::try_from(x).ok())
            .filter(|&x| x > 0)
            .ok_or_else(|| bad("decoding.greedy.max_symbols", "must be positive"))?;
        if vocabulary.len() != vocab_size
            || usize_field(joint, "num_classes")? != vocab_size
            || usize_field(jointnet, "encoder_hidden")? != hidden_size
            || usize_field(jointnet, "pred_hidden")? != decoder_hidden
            || joint_hidden != decoder_hidden
            || num_extra_outputs != durations.len()
            || durations.is_empty()
            || durations[0] != 0
            || durations.windows(2).any(|pair| pair[0] >= pair[1])
            || text_field(jointnet, "activation")? != "relu"
        {
            return Err(bad(
                "joint",
                "decoder, vocabulary, duration head, and encoder dimensions disagree",
            ));
        }
        Ok(Self {
            sample_rate,
            mel,
            encoder,
            subsampling_factor,
            vocab_size,
            decoder_hidden,
            decoder_layers,
            joint_hidden,
            num_extra_outputs,
            durations,
            max_symbols,
            vocabulary,
            redux_mode,
        })
    }
}

struct LstmLayer {
    input_weight: Vec<f32>,
    recurrent_weight: Vec<f32>,
    bias: Vec<f32>,
}

struct TdtDecoder {
    embedding: Vec<f32>,
    layers: Vec<LstmLayer>,
    joint_encoder: Vec<f32>,
    joint_encoder_bias: Vec<f32>,
    joint_predictor: Vec<f32>,
    joint_predictor_bias: Vec<f32>,
    joint_output: Vec<f32>,
    joint_output_bias: Vec<f32>,
}

fn load_checked(file: &SafetensorsFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let desc = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
        name: name.to_string(),
        why: "missing from safetensors".into(),
    })?;
    if desc.shape != shape {
        return Err(SpeechError::Tensor {
            name: name.to_string(),
            why: format!("expected shape {shape:?}, found {:?}", desc.shape),
        });
    }
    file.load_as_f32(name).map_err(|e| SpeechError::Tensor {
        name: name.to_string(),
        why: format!("load failed: {e}"),
    })
}

impl TdtDecoder {
    fn load(file: &SafetensorsFile, cfg: &ParakeetConfig) -> Result<Self> {
        let h = cfg.decoder_hidden;
        let vocab_rows = cfg.vocab_size + 1;
        let embedding = load_checked(file, "decoder.prediction.embed.weight", &[vocab_rows, h])?;
        let layers = (0..cfg.decoder_layers)
            .map(|i| {
                let prefix = format!("decoder.prediction.dec_rnn.lstm.{i}");
                Ok(LstmLayer {
                    input_weight: load_checked(file, &format!("{prefix}.Wx"), &[4 * h, h])?,
                    recurrent_weight: load_checked(file, &format!("{prefix}.Wh"), &[4 * h, h])?,
                    bias: load_checked(file, &format!("{prefix}.bias"), &[4 * h])?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let joint_h = cfg.joint_hidden;
        let output_rows = cfg.vocab_size + 1 + cfg.num_extra_outputs;
        Ok(Self {
            embedding,
            layers,
            joint_encoder: load_checked(
                file,
                "joint.enc.weight",
                &[joint_h, cfg.encoder.hidden_size],
            )?,
            joint_encoder_bias: load_checked(file, "joint.enc.bias", &[joint_h])?,
            joint_predictor: load_checked(file, "joint.pred.weight", &[joint_h, h])?,
            joint_predictor_bias: load_checked(file, "joint.pred.bias", &[joint_h])?,
            joint_output: load_checked(file, "joint.joint_net.2.weight", &[output_rows, joint_h])?,
            joint_output_bias: load_checked(file, "joint.joint_net.2.bias", &[output_rows])?,
        })
    }

    fn step(
        &self,
        token: usize,
        hidden: &[Vec<f32>],
        cell: &[Vec<f32>],
        h: usize,
    ) -> (Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let blank = self.embedding.len() / h - 1;
        let input = if token == blank {
            vec![0.0; h]
        } else {
            self.embedding[token * h..(token + 1) * h].to_vec()
        };
        let mut next_hidden = Vec::with_capacity(self.layers.len());
        let mut next_cell = Vec::with_capacity(self.layers.len());
        let mut layer_input = input;
        for (i, layer) in self.layers.iter().enumerate() {
            let mut gates = ops::linear(
                &layer_input,
                &layer.input_weight,
                Some(&layer.bias),
                1,
                h,
                4 * h,
            );
            let recurrent = ops::linear(&hidden[i], &layer.recurrent_weight, None, 1, h, 4 * h);
            for (gate, r) in gates.iter_mut().zip(recurrent) {
                *gate += r;
            }
            let mut h_out = vec![0.0; h];
            let mut c_out = vec![0.0; h];
            for j in 0..h {
                let input_gate = sigmoid(gates[j]);
                let forget_gate = sigmoid(gates[h + j]);
                let candidate = gates[2 * h + j].tanh();
                let output_gate = sigmoid(gates[3 * h + j]);
                c_out[j] = forget_gate * cell[i][j] + input_gate * candidate;
                h_out[j] = output_gate * c_out[j].tanh();
            }
            layer_input = h_out.clone();
            next_hidden.push(h_out);
            next_cell.push(c_out);
        }
        (layer_input, next_hidden, next_cell)
    }

    fn joint(&self, encoder: &[f32], predictor: &[f32], cfg: &ParakeetConfig) -> Vec<f32> {
        let mut enc = ops::linear(
            encoder,
            &self.joint_encoder,
            Some(&self.joint_encoder_bias),
            1,
            cfg.encoder.hidden_size,
            cfg.joint_hidden,
        );
        let pred = ops::linear(
            predictor,
            &self.joint_predictor,
            Some(&self.joint_predictor_bias),
            1,
            cfg.decoder_hidden,
            cfg.joint_hidden,
        );
        for (x, p) in enc.iter_mut().zip(pred) {
            *x = (*x + p).max(0.0);
        }
        ops::linear(
            &enc,
            &self.joint_output,
            Some(&self.joint_output_bias),
            1,
            cfg.joint_hidden,
            cfg.vocab_size + 1 + cfg.num_extra_outputs,
        )
    }
}

/// Loaded Parakeet TDT model with the v2/v3 or ternary Redux checkpoint.
pub struct ParakeetTdt {
    config: ParakeetConfig,
    encoder: FastConformer,
    decoder: TdtDecoder,
}

impl ParakeetTdt {
    /// Opens a local v2/v3 MLX install or the pinned raw Redux export.
    /// Redux weights are converted to a temporary F32 checkpoint at load and
    /// that temporary file is deleted before this method returns.
    pub fn open(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_value: Value =
            serde_json::from_slice(&std::fs::read(&config_path).map_err(|e| {
                bad(
                    "config.json",
                    format!("failed to read {}: {e}", config_path.display()),
                )
            })?)
            .map_err(|e| bad("config.json", format!("invalid JSON: {e}")))?;
        if config_value.get("ternary_modules").is_some() {
            let normalized = redux::normalized_config(model_dir, &config_value)?;
            let config = ParakeetConfig::parse(&normalized, true)?;
            let converted_path = redux::convert_checkpoint(model_dir)?;
            let loaded = Self::open_with_config(&converted_path, config);
            match loaded {
                Ok(model) => {
                    std::fs::remove_file(&converted_path).map_err(|e| {
                        bad(
                            "temporary Redux checkpoint",
                            format!("failed to remove {}: {e}", converted_path.display()),
                        )
                    })?;
                    Ok(model)
                }
                Err(error) => {
                    let _ = std::fs::remove_file(converted_path);
                    Err(error)
                }
            }
        } else {
            let config = ParakeetConfig::from_json(&config_value)?;
            Self::open_with_config(&model_dir.join("model.safetensors"), config)
        }
    }

    fn open_with_config(model_path: &Path, config: ParakeetConfig) -> Result<Self> {
        let file = SafetensorsFile::open(model_path)?;
        let encoder = FastConformer::load_parakeet(&file, &config.encoder)?;
        let decoder = TdtDecoder::load(&file, &config)?;
        Ok(Self {
            config,
            encoder,
            decoder,
        })
    }

    pub fn config(&self) -> &ParakeetConfig {
        &self.config
    }

    /// Mel frontend plus FastConformer encoder as a t-major feature matrix.
    fn encoder_features(&self, samples: &[f32]) -> Result<(Vec<f32>, usize)> {
        let mel = nemo_log_mel_spectrogram(samples, &self.config.mel)?;
        let mel_frames = mel.len();
        let mut channel_major = vec![0.0; self.config.encoder.num_mel_bins * mel_frames];
        for (t, row) in mel.iter().enumerate() {
            if row.len() != self.config.encoder.num_mel_bins {
                return Err(SpeechError::Audio(
                    "unexpected NeMo mel feature width".into(),
                ));
            }
            for (m, &value) in row.iter().enumerate() {
                channel_major[m * mel_frames + t] = value;
            }
        }
        self.encoder
            .encode_mel(&channel_major, mel_frames, mel_frames, &self.config.encoder)
    }

    /// Transcribes finite mono PCM at the configured sample rate.
    pub fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<String> {
        if sample_rate != self.config.sample_rate {
            return Err(SpeechError::Input {
                why: format!(
                    "Parakeet expects {} Hz PCM, received {sample_rate} Hz",
                    self.config.sample_rate
                ),
            });
        }
        if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
            return Err(SpeechError::Input {
                why: "Parakeet input must be nonempty finite mono PCM".into(),
            });
        }
        let (features, frame_count) = self.encoder_features(samples)?;
        self.decode(&features, frame_count)
    }

    fn decode(&self, features: &[f32], frame_count: usize) -> Result<String> {
        let cfg = &self.config;
        let blank_id = cfg.vocab_size;
        let mut last_token = blank_id;
        let mut hidden = vec![vec![0.0f32; cfg.decoder_hidden]; cfg.decoder_layers];
        let mut cell = vec![vec![0.0f32; cfg.decoder_hidden]; cfg.decoder_layers];
        let mut time = 0usize;
        let mut zero_duration_symbols = 0usize;
        let mut pieces = Vec::new();
        if cfg.redux_mode {
            for _ in 0..cfg.max_symbols.saturating_mul(frame_count) {
                if time >= frame_count {
                    break;
                }
                let encoder_start = time * cfg.encoder.hidden_size;
                let encoder_frame =
                    &features[encoder_start..encoder_start + cfg.encoder.hidden_size];
                let (predictor, proposed_hidden, proposed_cell) =
                    self.decoder
                        .step(last_token, &hidden, &cell, cfg.decoder_hidden);
                let logits = self.decoder.joint(encoder_frame, &predictor, cfg);
                let token = argmax(&logits[..=blank_id]);
                let mut duration = cfg.durations[argmax(&logits[blank_id + 1..])];
                if token == blank_id {
                    duration = duration.max(1);
                } else {
                    last_token = token;
                    hidden = proposed_hidden;
                    cell = proposed_cell;
                    let piece = &cfg.vocabulary[token];
                    if !is_special_piece(piece) {
                        pieces.push(piece.replace('▁', " "));
                    }
                }
                time = time.saturating_add(duration);
            }
            return Ok(pieces.concat().trim().to_string());
        }
        while time < frame_count {
            let encoder_start = time * cfg.encoder.hidden_size;
            let encoder_frame = &features[encoder_start..encoder_start + cfg.encoder.hidden_size];
            let (predictor, proposed_hidden, proposed_cell) =
                self.decoder
                    .step(last_token, &hidden, &cell, cfg.decoder_hidden);
            let logits = self.decoder.joint(encoder_frame, &predictor, cfg);
            let token = argmax(&logits[..=blank_id]);
            let duration = cfg.durations[argmax(&logits[blank_id + 1..])];
            if token != blank_id {
                last_token = token;
                hidden = proposed_hidden;
                cell = proposed_cell;
                let piece = &cfg.vocabulary[token];
                if !is_special_piece(piece) {
                    pieces.push(piece.replace('▁', " "));
                }
            }
            time = time.saturating_add(duration);
            zero_duration_symbols += 1;
            if duration != 0 {
                zero_duration_symbols = 0;
            } else if zero_duration_symbols >= cfg.max_symbols {
                time += 1;
                zero_duration_symbols = 0;
            }
        }
        Ok(pieces.concat().trim().to_string())
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    best
}

fn is_special_piece(piece: &str) -> bool {
    (piece.starts_with("<|") && piece.ends_with("|>")) || piece == "<unk>" || piece == "<pad>"
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn testdata(name: &str) -> String {
        format!("{}/testdata/parakeet/{name}", env!("CARGO_MANIFEST_DIR"))
    }

    fn read_npy(path: &str) -> (Vec<usize>, Vec<f32>) {
        let bytes = std::fs::read(path).expect(path);
        assert_eq!(&bytes[..6], b"\x93NUMPY");
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
        let shape: Vec<usize> = {
            let start = header.find('(').unwrap() + 1;
            let end = header[start..].find(')').unwrap() + start;
            header[start..end]
                .split(',')
                .filter_map(|p| p.trim().parse::<usize>().ok())
                .collect()
        };
        let data_start = 10 + header_len;
        let data = bytes[data_start..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        (shape, data)
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn pinned_config() -> ParakeetConfig {
        let value: Value =
            serde_json::from_str(&std::fs::read_to_string(testdata("config.json")).unwrap())
                .unwrap();
        ParakeetConfig::from_json(&value).unwrap()
    }

    fn model_dir() -> Option<String> {
        let dir = std::env::var("TURBOSPEECH_PARAKEET_MODEL").unwrap_or_else(|_| {
            format!(
                "{}/models/parakeet-tdt-0.6b-v2",
                std::env::var("HOME").unwrap()
            )
        });
        if std::path::Path::new(&dir)
            .join("model.safetensors")
            .exists()
        {
            Some(dir)
        } else {
            eprintln!("skipping: checkpoint not present at {dir}");
            None
        }
    }

    #[test]
    fn profiles_are_immutable_and_distinct() {
        assert_eq!(PARAKEET_TDT_V2.revision.len(), 40);
        assert_eq!(PARAKEET_TDT_V3.revision.len(), 40);
        assert_eq!(PARAKEET_REDUX.revision.len(), 40);
        assert_ne!(PARAKEET_TDT_V2.revision, PARAKEET_TDT_V3.revision);
    }

    #[test]
    fn argmax_keeps_the_first_tie() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
    }

    #[test]
    fn special_piece_filter_matches_reference() {
        assert!(is_special_piece("<|lang|>"));
        assert!(is_special_piece("<unk>"));
        assert!(is_special_piece("<pad>"));
        assert!(!is_special_piece("▁word"));
    }

    #[test]
    fn config_rejects_a_different_encoder_graph() {
        let mut cfg = json!({
            "preprocessor": {
                "sample_rate": 16000,
                "normalize": "per_feature",
                "window_size": 0.025,
                "window_stride": 0.01,
                "window": "hann",
                "features": 80,
                "n_fft": 512,
                "log": true,
                "frame_splicing": 1
            },
            "encoder": {
                "subsampling": "dw_striding",
                "self_attention_model": "rel_pos",
                "conv_norm_type": "batch_norm",
                "causal_downsampling": false,
                "feat_in": 80,
                "d_model": 8,
                "n_layers": 1,
                "n_heads": 2,
                "ff_expansion_factor": 2,
                "conv_kernel_size": 3,
                "subsampling_conv_channels": 4,
                "subsampling_factor": 8,
                "use_bias": false,
                "xscaling": false,
                "reduction": null,
                "reduction_position": null,
                "conv_context_size": null
            },
            "decoder": {
                "blank_as_pad": true,
                "vocab_size": 2,
                "prednet": {"pred_hidden": 4, "pred_rnn_layers": 1}
            },
            "joint": {
                "num_classes": 2,
                "num_extra_outputs": 2,
                "vocabulary": ["a", "b"],
                "jointnet": {
                    "joint_hidden": 4,
                    "encoder_hidden": 8,
                    "pred_hidden": 4,
                    "activation": "relu"
                }
            },
            "decoding": {"model_type": "tdt", "durations": [0, 1], "greedy": {"max_symbols": 3}}
        });
        cfg["encoder"]["subsampling"] = json!("striding");
        assert!(ParakeetConfig::from_json(&cfg).is_err());
    }

    /// The pinned v2 profile parses from the committed config fixture with
    /// the dimensions the fixtures were generated against.
    #[test]
    fn pinned_config_fixture_parses_with_pinned_dimensions() {
        let cfg = pinned_config();
        assert_eq!(cfg.sample_rate, 16_000);
        assert_eq!(cfg.encoder.num_mel_bins, 128);
        assert_eq!(cfg.encoder.hidden_size, 1024);
        assert_eq!(cfg.encoder.num_hidden_layers, 24);
        assert_eq!(cfg.vocab_size, 1024);
        assert_eq!(cfg.decoder_hidden, 640);
        assert_eq!(cfg.decoder_layers, 2);
        assert_eq!(cfg.durations, vec![0, 1, 2, 3, 4]);
        assert_eq!(cfg.max_symbols, 10);
        assert_eq!(cfg.num_extra_outputs, 5);
        assert!(!cfg.redux_mode);
    }

    /// Mel frontend vs the pinned reference golden on the committed speech
    /// excerpt; needs no checkpoint weights.
    #[test]
    fn mel_frontend_matches_reference() {
        let cfg = pinned_config();
        let (_, samples) = read_npy(&testdata("speech_samples.npy"));
        let (mel_shape, golden) = read_npy(&testdata("speech_mel.npy"));
        assert_eq!(mel_shape, vec![326, 128]);

        let mel = nemo_log_mel_spectrogram(&samples, &cfg.mel).unwrap();
        assert_eq!(mel.len(), 326);
        let flat: Vec<f32> = mel.concat();
        let diff = max_abs_diff(&flat, &golden);
        assert!(diff <= 2e-4, "mel mismatch, max abs diff {diff}");
    }

    /// FastConformer encoder output vs the pinned reference golden;
    /// skipped when the checkpoint is not installed (set
    /// TURBOSPEECH_PARAKEET_MODEL to override).
    #[test]
    fn encoder_forward_matches_reference() {
        let Some(dir) = model_dir() else { return };
        let model = ParakeetTdt::open(std::path::Path::new(&dir)).unwrap();
        let (_, samples) = read_npy(&testdata("speech_samples.npy"));
        let (enc_shape, golden) = read_npy(&testdata("speech_encoder.npy"));
        assert_eq!(enc_shape, vec![41, 1024]);

        let (features, frame_count) = model.encoder_features(&samples).unwrap();
        assert_eq!(frame_count, 41);
        let diff = max_abs_diff(&features, &golden);
        assert!(diff <= 1e-4, "encoder mismatch, max abs diff {diff}");
    }

    /// The greedy TDT decode against the pinned reference: per-step
    /// predictor/joint tensors, the full (time, token, decision) step
    /// table, emitted token ids, and the final transcript; skipped when
    /// the checkpoint is not installed.
    #[test]
    fn tdt_decode_matches_reference_steps() {
        let Some(dir) = model_dir() else { return };
        let model = ParakeetTdt::open(std::path::Path::new(&dir)).unwrap();
        let (_, samples) = read_npy(&testdata("speech_samples.npy"));
        let manifest: Value =
            serde_json::from_str(&std::fs::read_to_string(testdata("manifest.json")).unwrap())
                .unwrap();
        let (pred_shape, pred_golden) = read_npy(&testdata("speech_steps_pred.npy"));
        let (joint_shape, joint_golden) = read_npy(&testdata("speech_steps_joint.npy"));
        let captured = manifest["captured_steps"].as_u64().unwrap() as usize;
        assert_eq!(pred_shape, vec![captured as usize, 640]);
        assert_eq!(joint_shape, vec![captured as usize, 1, 1030]);

        let (features, frame_count) = model.encoder_features(&samples).unwrap();
        let cfg = &model.config;
        let blank_id = cfg.vocab_size;
        let h = cfg.decoder_hidden;
        let mut last_token = blank_id;
        let mut hidden = vec![vec![0.0f32; h]; cfg.decoder_layers];
        let mut cell = vec![vec![0.0f32; h]; cfg.decoder_layers];
        let mut time = 0usize;
        let mut zero_duration_symbols = 0usize;
        let mut step_index = 0usize;
        let mut emitted = Vec::new();
        while time < frame_count {
            let encoder_start = time * cfg.encoder.hidden_size;
            let encoder_frame = &features[encoder_start..encoder_start + cfg.encoder.hidden_size];
            let (predictor, proposed_hidden, proposed_cell) =
                model.decoder.step(last_token, &hidden, &cell, h);
            let logits = model.decoder.joint(encoder_frame, &predictor, cfg);
            if step_index < captured {
                let pred_diff = max_abs_diff(
                    &predictor,
                    &pred_golden[step_index * h..(step_index + 1) * h],
                );
                let joint_row = &joint_golden[step_index * 1030..(step_index + 1) * 1030];
                let joint_diff = max_abs_diff(&logits, joint_row);
                assert!(
                    pred_diff <= 1e-5,
                    "step {step_index} predictor diff {pred_diff}"
                );
                assert!(
                    joint_diff <= 2e-3,
                    "step {step_index} joint diff {joint_diff}"
                );
            }
            let token = argmax(&logits[..=blank_id]);
            let decision = argmax(&logits[blank_id + 1..]);
            let duration = cfg.durations[decision];
            let expected = &manifest["steps"][step_index];
            assert_eq!(
                time as u64,
                expected["time"].as_u64().unwrap(),
                "step {step_index} time"
            );
            assert_eq!(
                last_token as u64,
                expected["last_token"].as_u64().unwrap(),
                "step {step_index} last_token"
            );
            assert_eq!(
                token as u64,
                expected["pred_token"].as_u64().unwrap(),
                "step {step_index} pred_token"
            );
            assert_eq!(
                decision as u64,
                expected["decision"].as_u64().unwrap(),
                "step {step_index} decision"
            );
            if token != blank_id {
                last_token = token;
                hidden = proposed_hidden;
                cell = proposed_cell;
                emitted.push(token);
            }
            time = time.saturating_add(duration);
            zero_duration_symbols += 1;
            if duration != 0 {
                zero_duration_symbols = 0;
            } else if zero_duration_symbols >= cfg.max_symbols {
                time += 1;
                zero_duration_symbols = 0;
            }
            step_index += 1;
        }
        assert_eq!(step_index as u64, manifest["n_steps"].as_u64().unwrap());

        let expected_tokens: Vec<usize> = manifest["emitted_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(emitted, expected_tokens);
        assert_eq!(
            model.transcribe(&samples, 16_000).unwrap(),
            manifest["transcript"].as_str().unwrap()
        );
    }
}
