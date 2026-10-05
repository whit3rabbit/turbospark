//! GLM-ASR-Nano audio encoder and Llama decoder.
//!
//! Reference: `mlx_audio/stt/models/glmasr/` at mlx-audio 0.5.7,
//! commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`.

use std::fs;
use std::path::Path;

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;
use turbospark_tokenizer::Tokenizer as AudioTokenizer;

use crate::models::stt::qwen3_asr::{
    config::TextConfig,
    decoder::{Decoder, Linear},
};
use crate::quant::QuantScheme;
use crate::{ops, Result, SpeechError};

const SAMPLE_RATE: usize = 16_000;
const FFT_SIZE: usize = 400;
const HOP_LENGTH: usize = 160;
const MEL_BINS: usize = 128;
const AUDIO_HIDDEN: usize = 1280;
const AUDIO_HEADS: usize = 20;
const AUDIO_ROTARY_DIM: usize = 32;
const AUDIO_FFN: usize = 5120;
const AUDIO_LAYERS: usize = 32;
const MERGE_FACTOR: usize = 4;
const MAX_AUDIO_FRAMES: usize = 1500;
const DEFAULT_MAX_TOKENS: usize = 128;
const QUANT_BITS: u32 = 4;
const QUANT_GROUP: usize = 64;
const MAX_AUDIO_SECONDS: usize = 30;

/// Immutable Hugging Face checkpoint profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlmAsrProfile {
    pub name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
}

pub const GLM_ASR_NANO_2512: GlmAsrProfile = GlmAsrProfile {
    name: "GLM-ASR-Nano-2512 4-bit",
    repository: "mlx-community/GLM-ASR-Nano-2512-4bit",
    revision: "35553fa5bebfcc3ece3ce7d47b98827cb0ac9eef",
};

#[derive(Debug, Clone, PartialEq)]
struct Config {
    merge_factor: usize,
    max_whisper_length: usize,
    quant_bits: u32,
    quant_group: usize,
    text: TextConfig,
}

fn bad_config(field: &str, why: impl Into<String>) -> SpeechError {
    SpeechError::BadConfig {
        field: field.to_owned(),
        why: why.into(),
    }
}

fn positive(value: &Value, field: &str) -> Result<usize> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|number| usize::try_from(number).ok())
        .filter(|&number| number > 0)
        .ok_or_else(|| bad_config(field, "must be a positive integer"))
}

fn integer_or(value: &Value, field: &str, default: usize) -> Result<usize> {
    value
        .get(field)
        .map(|_| positive(value, field))
        .unwrap_or(Ok(default))
}

impl Config {
    fn from_json(root: &Value) -> Result<Self> {
        if root.get("model_type").and_then(Value::as_str) != Some("glmasr") {
            return Err(bad_config("model_type", "expected glmasr"));
        }
        if root.get("adapter_type").and_then(Value::as_str) != Some("mlp")
            || root.get("mlp_adapter_act").and_then(Value::as_str) != Some("gelu")
            || root.get("use_rope").and_then(Value::as_bool) != Some(true)
        {
            return Err(SpeechError::Unsupported {
                why: "GLM-ASR supports the pinned RoPE MLP-adapter profile".into(),
            });
        }

        let quant = root
            .get("quantization_config")
            .or_else(|| root.get("quantization"))
            .ok_or_else(|| bad_config("quantization_config", "missing from config.json"))?;
        let bits = positive(quant, "bits")? as u32;
        let quant_group = positive(quant, "group_size")?;
        if bits != QUANT_BITS
            || quant_group != QUANT_GROUP
            || quant.get("mode").and_then(Value::as_str) != Some("affine")
        {
            return Err(SpeechError::Unsupported {
                why: "GLM-ASR supports the pinned affine 4-bit, group-64 checkpoint".into(),
            });
        }

        let merge_factor = integer_or(root, "merge_factor", MERGE_FACTOR)?;
        let max_whisper_length = integer_or(root, "max_whisper_length", MAX_AUDIO_FRAMES)?;
        if merge_factor != MERGE_FACTOR || max_whisper_length != MAX_AUDIO_FRAMES {
            return Err(SpeechError::Unsupported {
                why: "GLM-ASR supports merge_factor=4 and a 1500-frame encoder limit".into(),
            });
        }

        // mlx-audio 0.5.7 supplies these defaults for the pinned repository,
        // whose config.json intentionally stores only the top-level adapter.
        let text = TextConfig {
            vocab_size: 59_264,
            hidden_size: 2048,
            intermediate_size: 6144,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 128,
            rotary_dim: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            qk_norm: false,
            tie_word_embeddings: false,
        };
        if text.num_attention_heads % text.num_key_value_heads != 0
            || text.hidden_size != text.num_attention_heads * text.head_dim
            || text.rotary_dim > text.head_dim
            || text.rotary_dim % 2 != 0
        {
            return Err(bad_config("lm_config", "invalid pinned Llama geometry"));
        }

        Ok(Self {
            merge_factor,
            max_whisper_length,
            quant_bits: bits,
            quant_group,
            text,
        })
    }

    fn quant_scheme(&self) -> QuantScheme {
        QuantScheme {
            bits: self.quant_bits,
            group_size: self.quant_group,
        }
    }
}

/// Loaded GLM-ASR-Nano profile. Loading is local-only; callers download the
/// pinned snapshot separately and pass its directory to [`GlmAsr::load`].
pub struct GlmAsr {
    config: Config,
    audio_encoder: AudioEncoder,
    decoder: Decoder,
    tokenizer: AudioTokenizer,
    stop_ids: Vec<u32>,
}

impl GlmAsr {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let root: Value = serde_json::from_slice(&fs::read(&config_path).map_err(|error| {
            SpeechError::Input {
                why: format!("cannot read {}: {error}", config_path.display()),
            }
        })?)
        .map_err(|error| bad_config("config.json", error.to_string()))?;
        let config = Config::from_json(&root)?;
        let weights = SafetensorsFile::open(&model_dir.join("model.safetensors"))?;
        let audio_encoder = AudioEncoder::load(&weights, &config)?;
        let decoder = Decoder::load(&weights, &config.text, config.quant_scheme())?;
        if decoder.hidden_size != config.text.hidden_size
            || decoder.vocab_size != config.text.vocab_size
        {
            return Err(bad_config(
                "lm_config",
                "Llama checkpoint geometry mismatch",
            ));
        }

        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = AudioTokenizer::from_file(&tokenizer_path)
            .map_err(|error| bad_config("tokenizer.json", error.to_string()))?;
        let stop_ids = ["<|endoftext|>", "<|user|>", "<|observation|>"]
            .into_iter()
            .map(|token| {
                tokenizer
                    .token_to_id(token)
                    .ok_or_else(|| bad_config("tokenizer.json", format!("missing token {token}")))
            })
            .collect::<Result<Vec<_>>>()?;
        for token in [
            "<|user|>",
            "<|begin_of_audio|>",
            "<|end_of_audio|>",
            "<|assistant|>",
        ] {
            if tokenizer.token_to_id(token).is_none() {
                return Err(bad_config(
                    "tokenizer.json",
                    format!("missing token {token}"),
                ));
            }
        }

        Ok(Self {
            audio_encoder,
            decoder,
            tokenizer,
            stop_ids,
            config,
        })
    }

    pub fn profile(&self) -> GlmAsrProfile {
        GLM_ASR_NANO_2512
    }

    /// Transcribe a mono waveform sampled at 16 kHz using greedy decoding.
    /// Inputs longer than 30 seconds are refused; upstream's silence-aware
    /// multi-chunk policy is not part of this first offline profile.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        self.transcribe_with_max_tokens(samples, DEFAULT_MAX_TOKENS)
    }

    pub fn transcribe_with_max_tokens(&self, samples: &[f32], max_tokens: usize) -> Result<String> {
        if max_tokens == 0 {
            return Err(SpeechError::Input {
                why: "GLM-ASR max_tokens must be positive".into(),
            });
        }
        let audio = self.encode_audio(samples, false)?;
        self.decode_audio(&audio.features, audio.rows, max_tokens)
    }

    fn encode_audio(&self, samples: &[f32], capture_trace: bool) -> Result<EncodedAudio> {
        if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
            return Err(SpeechError::Input {
                why: "audio must be non-empty and contain only finite samples".into(),
            });
        }
        let max_samples = MAX_AUDIO_SECONDS * SAMPLE_RATE;
        if samples.len() > max_samples {
            return Err(SpeechError::Input {
                why: "GLM-ASR currently accepts one audio chunk of at most 30 seconds".into(),
            });
        }

        let mut padded = samples.to_vec();
        if padded.len() < SAMPLE_RATE {
            padded.resize(SAMPLE_RATE, 0.0);
        }
        let (mel, mel_rows) = log_mel(&padded)?;
        let (encoded, encoder_trace) =
            self.audio_encoder
                .whisper
                .forward(&mel, mel_rows, capture_trace)?;
        let encoder_rows = encoded.len() / AUDIO_HIDDEN;
        if encoder_rows > self.config.max_whisper_length || encoder_rows < self.config.merge_factor
        {
            return Err(SpeechError::Input {
                why: format!("GLM-ASR encoder produced unsupported sequence length {encoder_rows}"),
            });
        }

        let mut normalized = encoded;
        self.audio_encoder
            .layer_norm
            .apply(&mut normalized, encoder_rows);
        let rows = ((encoder_rows - self.config.merge_factor) / self.config.merge_factor + 1)
            .min(self.config.max_whisper_length / self.config.merge_factor);
        let merged_width = AUDIO_HIDDEN * self.config.merge_factor;
        let mut merged = vec![0.0; rows * merged_width];
        for row in 0..rows {
            let source = row * self.config.merge_factor * AUDIO_HIDDEN;
            let count = self.config.merge_factor * AUDIO_HIDDEN;
            merged[row * merged_width..(row + 1) * merged_width]
                .copy_from_slice(&normalized[source..source + count]);
        }
        let mut hidden = self.audio_encoder.adapting_in.forward(&merged, rows);
        ops::gelu_erf(&mut hidden);
        let features = self.audio_encoder.adapting_out.forward(&hidden, rows);
        let trace = capture_trace.then(|| AudioTrace {
            mel: snapshot(&mel, &[mel_rows, MEL_BINS]),
            conv1: encoder_trace
                .as_ref()
                .expect("trace requested")
                .conv1
                .clone(),
            conv2: encoder_trace
                .as_ref()
                .expect("trace requested")
                .conv2
                .clone(),
            encoder_first: encoder_trace
                .as_ref()
                .expect("trace requested")
                .encoder_first
                .clone(),
            encoder_last: encoder_trace
                .as_ref()
                .expect("trace requested")
                .encoder_last
                .clone(),
            normalized: snapshot(&normalized, &[encoder_rows, AUDIO_HIDDEN]),
            merged: snapshot(&merged, &[rows, merged_width]),
            adapted: snapshot(&features, &[rows, self.config.text.hidden_size]),
        });
        Ok(EncodedAudio {
            features,
            rows,
            trace,
        })
    }

    fn decode_audio(
        &self,
        features: &[f32],
        audio_rows: usize,
        max_tokens: usize,
    ) -> Result<String> {
        let hidden_size = self.decoder.hidden_size;
        if audio_rows == 0 || features.len() != audio_rows * hidden_size {
            return Err(SpeechError::Tensor {
                name: "GLM-ASR audio adapter output".into(),
                why: "audio row count does not match the Llama hidden width".into(),
            });
        }

        let prefix = self
            .tokenizer
            .encode("<|user|>\n<|begin_of_audio|>", false)
            .map_err(|error| SpeechError::Input {
                why: format!("GLM-ASR prompt tokenization failed: {error}"),
            })?;
        let suffix = self
            .tokenizer
            .encode(
                "<|end_of_audio|>\nPlease transcribe this audio into text<|assistant|>\n",
                false,
            )
            .map_err(|error| SpeechError::Input {
                why: format!("GLM-ASR prompt tokenization failed: {error}"),
            })?;
        let mut token_ids = prefix
            .get_ids()
            .iter()
            .map(|&id| {
                i32::try_from(id).map_err(|_| SpeechError::Input {
                    why: "GLM-ASR prompt token id exceeds signed 32-bit range".into(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let audio_start = token_ids.len();
        token_ids.resize(audio_start + audio_rows, 0);
        token_ids.extend(
            suffix
                .get_ids()
                .iter()
                .map(|&id| {
                    i32::try_from(id).map_err(|_| SpeechError::Input {
                        why: "GLM-ASR prompt token id exceeds signed 32-bit range".into(),
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        );

        let mut embeddings = self.decoder.embed(&token_ids)?;
        for row in 0..audio_rows {
            let position = audio_start + row;
            let dst = position * hidden_size;
            let src = row * hidden_size;
            embeddings[dst..dst + hidden_size].copy_from_slice(&features[src..src + hidden_size]);
        }
        let prompt_rows = embeddings.len() / hidden_size;
        let (mut last_hidden, mut cache) = self.decoder.prefill(&embeddings, prompt_rows);
        let mut generated = Vec::new();
        for _ in 0..max_tokens {
            let next = argmax(&self.decoder.logits(&last_hidden));
            if self.stop_ids.contains(&(next as u32)) {
                break;
            }
            let token = i32::try_from(next).map_err(|_| SpeechError::Input {
                why: "GLM-ASR generated token id exceeds signed 32-bit range".into(),
            })?;
            generated.push(next as u32);
            let embedding = self.decoder.embed(&[token])?;
            last_hidden = self.decoder.step(&embedding, &mut cache);
        }
        self.tokenizer
            .decode(&generated, true)
            .map(|text| text.trim().to_owned())
            .map_err(|error| SpeechError::Input {
                why: format!("GLM-ASR output detokenization failed: {error}"),
            })
    }
}

struct EncodedAudio {
    features: Vec<f32>,
    rows: usize,
    trace: Option<AudioTrace>,
}

#[derive(Debug, Clone)]
struct Snapshot {
    shape: Vec<usize>,
    indices: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Debug, Clone)]
struct AudioTrace {
    mel: Snapshot,
    conv1: Snapshot,
    conv2: Snapshot,
    encoder_first: Snapshot,
    encoder_last: Snapshot,
    normalized: Snapshot,
    merged: Snapshot,
    adapted: Snapshot,
}

fn snapshot(values: &[f32], shape: &[usize]) -> Snapshot {
    let count = values.len().min(16);
    let indices = if count <= 1 {
        vec![0]
    } else {
        (0..count)
            .map(|slot| slot * (values.len() - 1) / (count - 1))
            .collect()
    };
    Snapshot {
        shape: shape.to_vec(),
        values: indices.iter().map(|&index| values[index]).collect(),
        indices,
    }
}

fn log_mel(samples: &[f32]) -> Result<(Vec<f32>, usize)> {
    let window: Vec<f32> = (0..FFT_SIZE)
        .map(|index| {
            0.5 - 0.5 * (2.0 * std::f32::consts::PI * index as f32 / (FFT_SIZE - 1) as f32).cos()
        })
        .collect();
    let spectra = turbospark_audio::stft::stft(
        samples,
        &turbospark_audio::stft::StftOptions {
            fft_size: FFT_SIZE,
            hop: HOP_LENGTH,
            window,
            center: true,
        },
    )
    .map_err(|error| SpeechError::Input {
        why: format!("GLM-ASR STFT failed: {error}"),
    })?;
    if spectra.len() < 2 {
        return Err(SpeechError::Input {
            why: "audio is too short for the GLM-ASR frontend".into(),
        });
    }
    let filterbank = turbospark_audio::mel::mel_filterbank(
        MEL_BINS,
        FFT_SIZE,
        SAMPLE_RATE as u32,
        0.0,
        Some(SAMPLE_RATE as f32 / 2.0),
        turbospark_audio::mel::MelScale::Slaney,
    )
    .map_err(|error| SpeechError::Input {
        why: format!("GLM-ASR mel filterbank failed: {error}"),
    })?;

    // mlx-audio drops its final centered STFT frame before global normalization.
    let retained = spectra.len() - 1;
    let mut mel = Vec::with_capacity(retained * MEL_BINS);
    let mut peak = f32::NEG_INFINITY;
    for spectrum in spectra.iter().take(retained) {
        let power: Vec<f32> = spectrum
            .iter()
            .map(|bin| bin.re * bin.re + bin.im * bin.im)
            .collect();
        let projected = filterbank
            .project(&power)
            .map_err(|error| SpeechError::Input {
                why: format!("GLM-ASR mel projection failed: {error}"),
            })?;
        for value in projected {
            peak = peak.max(value.max(1e-10).log10());
            mel.push(value);
        }
    }
    if !peak.is_finite() {
        return Err(SpeechError::Input {
            why: "GLM-ASR frontend produced no finite mel energy".into(),
        });
    }
    let clamp_at = peak - 8.0;
    for value in &mut mel {
        *value = (value.max(1e-10).log10().max(clamp_at) + 4.0) / 4.0;
    }
    Ok((mel, retained))
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| {
            if value > best.1 {
                (index, value)
            } else {
                best
            }
        })
        .0
}

fn load_tensor(file: &SafetensorsFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let descriptor = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
        name: name.to_owned(),
        why: "tensor is missing".into(),
    })?;
    if descriptor.shape != shape {
        return Err(SpeechError::Tensor {
            name: name.to_owned(),
            why: format!("expected shape {shape:?}, got {:?}", descriptor.shape),
        });
    }
    Ok(file.load_as_f32(name)?)
}

struct Conv1d {
    weight: Vec<f32>,
    bias: Vec<f32>,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
}

impl Conv1d {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        // MLX Conv1d stores [out, kernel, in], while the portable kernel
        // consumes PyTorch [out, in, kernel].
        let mlx = load_tensor(file, &format!("{prefix}.weight"), &[output, kernel, input])?;
        let mut weight = vec![0.0; mlx.len()];
        for out in 0..output {
            for channel in 0..input {
                for tap in 0..kernel {
                    weight[(out * input + channel) * kernel + tap] =
                        mlx[(out * kernel + tap) * input + channel];
                }
            }
        }
        let bias = load_tensor(file, &format!("{prefix}.bias"), &[output])?;
        Ok(Self {
            weight,
            bias,
            input,
            output,
            kernel,
            stride,
            padding,
        })
    }

    fn forward(&self, input: &[f32], rows: usize) -> (Vec<f32>, usize) {
        debug_assert_eq!(input.len(), rows * self.input);
        let mut channels_first = vec![0.0; input.len()];
        for row in 0..rows {
            for channel in 0..self.input {
                channels_first[channel * rows + row] = input[row * self.input + channel];
            }
        }
        let output_rows = (rows + 2 * self.padding - self.kernel) / self.stride + 1;
        let output = ops::conv1d(
            &channels_first,
            &self.weight,
            Some(&self.bias),
            self.input,
            self.output,
            self.kernel,
            self.stride,
            self.padding,
            1,
            1,
        );
        let mut time_major = vec![0.0; output.len()];
        for row in 0..output_rows {
            for channel in 0..self.output {
                time_major[row * self.output + channel] = output[channel * output_rows + row];
            }
        }
        (time_major, output_rows)
    }
}

struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    width: usize,
}

impl LayerNorm {
    fn load(file: &SafetensorsFile, prefix: &str, width: usize) -> Result<Self> {
        Ok(Self {
            weight: load_tensor(file, &format!("{prefix}.weight"), &[width])?,
            bias: load_tensor(file, &format!("{prefix}.bias"), &[width])?,
            width,
        })
    }

    fn apply(&self, values: &mut [f32], rows: usize) {
        ops::layernorm(
            values,
            rows,
            self.width,
            &self.weight,
            Some(&self.bias),
            1e-5,
        );
    }
}

struct WhisperAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
}

impl WhisperAttention {
    fn load(file: &SafetensorsFile, prefix: &str) -> Result<Self> {
        Ok(Self {
            q_proj: Linear::load(
                file,
                &format!("{prefix}.q_proj"),
                AUDIO_HIDDEN,
                AUDIO_HIDDEN,
                QuantScheme {
                    bits: QUANT_BITS,
                    group_size: QUANT_GROUP,
                },
            )?,
            k_proj: Linear::load(
                file,
                &format!("{prefix}.k_proj"),
                AUDIO_HIDDEN,
                AUDIO_HIDDEN,
                QuantScheme {
                    bits: QUANT_BITS,
                    group_size: QUANT_GROUP,
                },
            )?,
            v_proj: Linear::load(
                file,
                &format!("{prefix}.v_proj"),
                AUDIO_HIDDEN,
                AUDIO_HIDDEN,
                QuantScheme {
                    bits: QUANT_BITS,
                    group_size: QUANT_GROUP,
                },
            )?,
            out_proj: Linear::load(
                file,
                &format!("{prefix}.out_proj"),
                AUDIO_HIDDEN,
                AUDIO_HIDDEN,
                QuantScheme {
                    bits: QUANT_BITS,
                    group_size: QUANT_GROUP,
                },
            )?,
            heads: AUDIO_HEADS,
            head_dim: AUDIO_HIDDEN / AUDIO_HEADS,
            rotary_dim: AUDIO_ROTARY_DIM,
        })
    }

    fn forward(&self, input: &[f32], rows: usize) -> Vec<f32> {
        let q = self.q_proj.forward(input, rows);
        let k = self.k_proj.forward(input, rows);
        let v = self.v_proj.forward(input, rows);
        let mut q = transpose_heads(&q, rows, self.heads, self.head_dim);
        let mut k = transpose_heads(&k, rows, self.heads, self.head_dim);
        let v = transpose_heads(&v, rows, self.heads, self.head_dim);
        apply_rope_traditional(&mut q, self.heads, rows, self.head_dim, self.rotary_dim);
        apply_rope_traditional(&mut k, self.heads, rows, self.head_dim, self.rotary_dim);

        let mut attended = vec![0.0; rows * self.heads * self.head_dim];
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        for head in 0..self.heads {
            let head_start = head * rows * self.head_dim;
            let query_head = &q[head_start..head_start + rows * self.head_dim];
            let key_head = &k[head_start..head_start + rows * self.head_dim];
            let value_head = &v[head_start..head_start + rows * self.head_dim];
            for position in 0..rows {
                let query = &query_head[position * self.head_dim..(position + 1) * self.head_dim];
                let result = ops::sdpa(
                    query,
                    key_head,
                    value_head,
                    None,
                    1,
                    rows,
                    self.head_dim,
                    self.head_dim,
                    scale,
                );
                let target = position * self.heads * self.head_dim + head * self.head_dim;
                attended[target..target + self.head_dim].copy_from_slice(&result);
            }
        }
        self.out_proj.forward(&attended, rows)
    }
}

struct WhisperLayer {
    attention: WhisperAttention,
    attention_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_norm: LayerNorm,
}

impl WhisperLayer {
    fn load(file: &SafetensorsFile, index: usize) -> Result<Self> {
        let prefix = format!("audio_encoder.whisper.layers.{index}");
        let scheme = QuantScheme {
            bits: QUANT_BITS,
            group_size: QUANT_GROUP,
        };
        Ok(Self {
            attention: WhisperAttention::load(file, &format!("{prefix}.self_attn"))?,
            attention_norm: LayerNorm::load(
                file,
                &format!("{prefix}.self_attn_layer_norm"),
                AUDIO_HIDDEN,
            )?,
            fc1: Linear::load(
                file,
                &format!("{prefix}.fc1"),
                AUDIO_HIDDEN,
                AUDIO_FFN,
                scheme,
            )?,
            fc2: Linear::load(
                file,
                &format!("{prefix}.fc2"),
                AUDIO_FFN,
                AUDIO_HIDDEN,
                scheme,
            )?,
            final_norm: LayerNorm::load(file, &format!("{prefix}.final_layer_norm"), AUDIO_HIDDEN)?,
        })
    }

    fn forward(&self, input: &[f32], rows: usize) -> Vec<f32> {
        let mut normalized = input.to_vec();
        self.attention_norm.apply(&mut normalized, rows);
        let attention = self.attention.forward(&normalized, rows);
        let mut hidden: Vec<f32> = input.iter().zip(attention).map(|(&x, a)| x + a).collect();
        let mut normalized = hidden.clone();
        self.final_norm.apply(&mut normalized, rows);
        let mut feed_forward = self.fc1.forward(&normalized, rows);
        ops::gelu_erf(&mut feed_forward);
        let feed_forward = self.fc2.forward(&feed_forward, rows);
        for (value, add) in hidden.iter_mut().zip(feed_forward) {
            *value += add;
        }
        hidden
    }
}

struct WhisperEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    layers: Vec<WhisperLayer>,
}

impl WhisperEncoder {
    fn load(file: &SafetensorsFile) -> Result<Self> {
        let prefix = "audio_encoder.whisper";
        let conv1 = Conv1d::load(
            file,
            &format!("{prefix}.conv1"),
            MEL_BINS,
            AUDIO_HIDDEN,
            3,
            1,
            1,
        )?;
        let conv2 = Conv1d::load(
            file,
            &format!("{prefix}.conv2"),
            AUDIO_HIDDEN,
            AUDIO_HIDDEN,
            3,
            2,
            1,
        )?;
        let layers = (0..AUDIO_LAYERS)
            .map(|index| WhisperLayer::load(file, index))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv1,
            conv2,
            layers,
        })
    }

    fn forward(
        &self,
        mel: &[f32],
        mel_rows: usize,
        capture_trace: bool,
    ) -> Result<(Vec<f32>, Option<EncoderTrace>)> {
        if mel.len() != mel_rows * MEL_BINS {
            return Err(SpeechError::Tensor {
                name: "GLM-ASR mel features".into(),
                why: "mel frame count does not match the pinned 128-band frontend".into(),
            });
        }
        let (mut hidden, conv1_rows) = self.conv1.forward(mel, mel_rows);
        ops::gelu_erf(&mut hidden);
        let conv1 = capture_trace.then(|| snapshot(&hidden, &[conv1_rows, AUDIO_HIDDEN]));
        let (mut hidden, rows) = self.conv2.forward(&hidden, conv1_rows);
        ops::gelu_erf(&mut hidden);
        let conv2 = capture_trace.then(|| snapshot(&hidden, &[rows, AUDIO_HIDDEN]));
        let mut first = None;
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, rows);
            if index == 0 && capture_trace {
                first = Some(snapshot(&hidden, &[rows, AUDIO_HIDDEN]));
            }
        }
        let trace = capture_trace.then(|| EncoderTrace {
            conv1: conv1.expect("trace requested"),
            conv2: conv2.expect("trace requested"),
            encoder_first: first.expect("at least one Whisper encoder layer"),
            encoder_last: snapshot(&hidden, &[rows, AUDIO_HIDDEN]),
        });
        Ok((hidden, trace))
    }
}

struct EncoderTrace {
    conv1: Snapshot,
    conv2: Snapshot,
    encoder_first: Snapshot,
    encoder_last: Snapshot,
}

struct AudioEncoder {
    whisper: WhisperEncoder,
    layer_norm: LayerNorm,
    adapting_in: Linear,
    adapting_out: Linear,
}

impl AudioEncoder {
    fn load(file: &SafetensorsFile, config: &Config) -> Result<Self> {
        let scheme = config.quant_scheme();
        Ok(Self {
            whisper: WhisperEncoder::load(file)?,
            layer_norm: LayerNorm::load(file, "audio_encoder.layer_norm", AUDIO_HIDDEN)?,
            adapting_in: Linear::load(
                file,
                "audio_encoder.adapting.fc1",
                AUDIO_HIDDEN * config.merge_factor,
                config.text.hidden_size * 2,
                scheme,
            )?,
            adapting_out: Linear::load(
                file,
                "audio_encoder.adapting.fc2",
                config.text.hidden_size * 2,
                config.text.hidden_size,
                scheme,
            )?,
        })
    }
}

fn transpose_heads(input: &[f32], rows: usize, heads: usize, dim: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        for head in 0..heads {
            let src = (row * heads + head) * dim;
            let dst = (head * rows + row) * dim;
            output[dst..dst + dim].copy_from_slice(&input[src..src + dim]);
        }
    }
    output
}

fn apply_rope_traditional(
    values: &mut [f32],
    heads: usize,
    rows: usize,
    dim: usize,
    rotary: usize,
) {
    let pairs = rotary / 2;
    let (cos, sin) = ops::rope_tables(rows, rotary, 10_000.0);
    for head in 0..heads {
        for row in 0..rows {
            let base = (head * rows + row) * dim;
            for pair in 0..pairs {
                let left = pair * 2;
                let right = left + 1;
                let a = values[base + left];
                let b = values[base + right];
                let table = row * pairs + pair;
                values[base + left] = a * cos[table] - b * sin[table];
                values[base + right] = a * sin[table] + b * cos[table];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_pinned_profile_and_rejects_wrong_quantization() {
        let root = serde_json::json!({
            "model_type":"glmasr",
            "adapter_type":"mlp",
            "mlp_adapter_act":"gelu",
            "use_rope":true,
            "merge_factor":4,
            "max_whisper_length":1500,
            "quantization_config":{"bits":4,"group_size":64,"mode":"affine"}
        });
        let config = Config::from_json(&root).unwrap();
        assert_eq!(config.text.hidden_size, 2048);
        assert!(!config.text.qk_norm);
        assert!(!config.text.tie_word_embeddings);
        assert!(Config::from_json(&serde_json::json!({
            "model_type":"glmasr","adapter_type":"mlp","mlp_adapter_act":"gelu",
            "use_rope":true,"quantization_config":{"bits":8,"group_size":64,"mode":"affine"}
        }))
        .is_err());
    }

    #[test]
    fn traditional_rope_leaves_the_unrotated_head_suffix_untouched() {
        let mut values = vec![1.0; 64];
        let suffix = values[32..].to_vec();
        apply_rope_traditional(&mut values, 1, 1, 64, AUDIO_ROTARY_DIM);
        assert_eq!(&values[32..], suffix);
    }

    #[test]
    #[ignore = "requires the pinned GLM-ASR snapshot in TURBOSPARK_GLMASR_DIR"]
    fn pinned_checkpoint_matches_mlx_stages_and_transcript() {
        let model_dir = std::env::var_os("TURBOSPARK_GLMASR_DIR")
            .map(std::path::PathBuf::from)
            .expect("set TURBOSPARK_GLMASR_DIR to the pinned model snapshot");
        let model = GlmAsr::load(&model_dir).expect("pinned GLM-ASR checkpoint loads");
        let wav_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_forced_aligner_reference.wav");
        let audio = turbospark_audio::wav::read_wav_f32(&wav_path).expect("reference WAV loads");
        let encoded = model
            .encode_audio(&audio.samples, true)
            .expect("audio encoder runs");
        let trace = encoded.trace.as_ref().expect("trace requested");
        let reference: Value =
            serde_json::from_str(include_str!("../../../testdata/glmasr_reference.json"))
                .expect("GLM-ASR MLX fixture is valid JSON");
        let stages = reference.get("stages").expect("fixture has stage samples");
        assert_snapshot("mel", &trace.mel, &stages["mel"]);
        assert_snapshot("conv1", &trace.conv1, &stages["conv1"]);
        assert_snapshot("conv2", &trace.conv2, &stages["conv2"]);
        assert_snapshot(
            "encoder_first",
            &trace.encoder_first,
            &stages["encoder_first"],
        );
        assert_snapshot("encoder_last", &trace.encoder_last, &stages["encoder_last"]);
        assert_snapshot("normalized", &trace.normalized, &stages["normalized"]);
        assert_snapshot("merged", &trace.merged, &stages["merged"]);
        assert_snapshot("adapted", &trace.adapted, &stages["adapted"]);
        let transcript = model
            .decode_audio(&encoded.features, encoded.rows, DEFAULT_MAX_TOKENS)
            .expect("GLM-ASR greedy decoding succeeds");
        assert_eq!(transcript, reference["transcript"].as_str().unwrap());
    }

    fn assert_snapshot(name: &str, actual: &Snapshot, expected: &Value) {
        let shape: Vec<usize> = expected["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as usize)
            .collect();
        let shape_matches =
            actual.shape == shape || (shape.first() == Some(&1) && actual.shape == shape[1..]);
        assert!(
            shape_matches,
            "{name} shape: Rust {:?}, MLX {shape:?}",
            actual.shape
        );
        let indices: Vec<usize> = expected["indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(actual.indices, indices, "{name} sampled indices");
        let values = expected["values"].as_array().unwrap();
        assert_eq!(actual.values.len(), values.len(), "{name} sample count");
        // Small backend reduction and activation differences accumulate over
        // 32 f32 encoder layers; keep the frontend and first-layer checks tight.
        let tolerance = if matches!(name, "encoder_last" | "normalized" | "merged" | "adapted") {
            3e-2
        } else {
            1e-2
        };
        for (index, (&got, expected)) in actual.values.iter().zip(values).enumerate() {
            let expected = expected.as_f64().unwrap() as f32;
            assert!(
                (got - expected).abs() <= tolerance,
                "{name}[{}]: Rust {got}, MLX {expected}",
                actual.indices[index]
            );
        }
    }
}
