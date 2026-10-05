//! Fun-ASR-Nano speech recognition using its SANM encoder and Qwen3 decoder.
//!
//! Reference: `mlx_audio/stt/models/fun_asr_nano/` at mlx-audio 0.5.7,
//! commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`.

use std::fs;
use std::path::Path;

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;
use turbospark_tokenizer::Tokenizer as AudioTokenizer;

use crate::models::stt::qwen3_asr::{config::TextConfig, decoder::Decoder};
use crate::quant::QuantScheme;
use crate::stt::sensevoice::frontend::{compute_fbank, FrontendConfig};
use crate::{ops, Result, SpeechError};

/// Immutable Hugging Face checkpoint profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FunAsrNanoProfile {
    pub name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
}

pub const FUN_ASR_NANO_2512: FunAsrNanoProfile = FunAsrNanoProfile {
    name: "Fun-ASR-Nano-2512",
    repository: "mlx-community/Fun-ASR-Nano-2512",
    revision: "a7bc96fceaafce39ed6748e0c0fa9a9508b67f86",
};

#[derive(Debug, Clone, PartialEq, Eq)]
struct Config {
    input_size: usize,
    sample_rate: usize,
    num_mels: usize,
    frame_length_ms: usize,
    frame_shift_ms: usize,
    lfr_m: usize,
    lfr_n: usize,
    output_size: usize,
    attention_heads: usize,
    linear_units: usize,
    num_blocks: usize,
    tp_blocks: usize,
    fsmn_kernel: usize,
    sanm_shift: usize,
    adaptor_downsample_rate: usize,
    adaptor_ffn_dim: usize,
    adaptor_num_layers: usize,
    adaptor_attention_heads: usize,
    tokenizer_path: String,
    default_max_tokens: usize,
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

impl Config {
    fn from_json(root: &Value) -> Result<Self> {
        if root.get("model_type").and_then(Value::as_str) != Some("fun_asr_nano") {
            return Err(bad_config("model_type", "expected fun_asr_nano"));
        }
        let frontend = root
            .get("frontend_conf")
            .ok_or_else(|| bad_config("frontend_conf", "missing from config.json"))?;
        if frontend.get("window").and_then(Value::as_str) != Some("hamming") {
            return Err(SpeechError::Unsupported {
                why: "Fun-ASR-Nano port supports the pinned Hamming-window frontend".into(),
            });
        }
        let encoder = root
            .get("audio_encoder_conf")
            .ok_or_else(|| bad_config("audio_encoder_conf", "missing from config.json"))?;
        let adaptor = root
            .get("audio_adaptor_conf")
            .ok_or_else(|| bad_config("audio_adaptor_conf", "missing from config.json"))?;
        let config = Self {
            input_size: positive(root, "input_size")?,
            sample_rate: positive(frontend, "fs")?,
            num_mels: positive(frontend, "n_mels")?,
            frame_length_ms: positive(frontend, "frame_length")?,
            frame_shift_ms: positive(frontend, "frame_shift")?,
            lfr_m: positive(frontend, "lfr_m")?,
            lfr_n: positive(frontend, "lfr_n")?,
            output_size: positive(encoder, "output_size")?,
            attention_heads: positive(encoder, "attention_heads")?,
            linear_units: positive(encoder, "linear_units")?,
            num_blocks: positive(encoder, "num_blocks")?,
            tp_blocks: encoder
                .get("tp_blocks")
                .and_then(Value::as_u64)
                .and_then(|number| usize::try_from(number).ok())
                .ok_or_else(|| {
                    bad_config("audio_encoder_conf.tp_blocks", "must be non-negative")
                })?,
            fsmn_kernel: positive(encoder, "kernel_size")?,
            sanm_shift: encoder
                .get("sanm_shift")
                .and_then(Value::as_u64)
                .and_then(|number| usize::try_from(number).ok())
                .ok_or_else(|| {
                    bad_config("audio_encoder_conf.sanm_shift", "must be non-negative")
                })?,
            adaptor_downsample_rate: positive(adaptor, "downsample_rate")?,
            adaptor_ffn_dim: positive(adaptor, "ffn_dim")?,
            adaptor_num_layers: positive(adaptor, "n_layer")?,
            adaptor_attention_heads: positive(adaptor, "attention_heads")?,
            tokenizer_path: root
                .get("qwen_tokenizer_path")
                .and_then(Value::as_str)
                .ok_or_else(|| bad_config("qwen_tokenizer_path", "must be a string"))?
                .to_owned(),
            default_max_tokens: positive(root, "default_max_tokens")?,
        };
        let normalize_before = encoder
            .get("normalize_before")
            .and_then(Value::as_bool)
            .ok_or_else(|| {
                bad_config("audio_encoder_conf.normalize_before", "must be a boolean")
            })?;
        let use_low_frame_rate = adaptor
            .get("use_low_frame_rate")
            .and_then(Value::as_bool)
            .ok_or_else(|| {
                bad_config("audio_adaptor_conf.use_low_frame_rate", "must be a boolean")
            })?;
        let adaptor_encoder_dim = positive(adaptor, "encoder_dim")?;
        let adaptor_llm_dim = positive(adaptor, "llm_dim")?;
        if config.sample_rate != 16_000
            || config.num_mels != 80
            || config.frame_length_ms != 25
            || config.frame_shift_ms != 10
            || config.lfr_m != 7
            || config.lfr_n != 6
            || config.input_size != config.num_mels * config.lfr_m
            || config.output_size != 512
            || config.attention_heads != 4
            || config.linear_units != 2048
            || config.num_blocks != 50
            || config.tp_blocks != 20
            || config.fsmn_kernel != 11
            || config.sanm_shift != 0
            || !normalize_before
            || config.adaptor_downsample_rate != 1
            || config.adaptor_ffn_dim != 2048
            || config.adaptor_num_layers != 2
            || config.adaptor_attention_heads != 8
            || adaptor_encoder_dim != config.output_size
            || adaptor_llm_dim != 1024
            || !use_low_frame_rate
            || config.tokenizer_path != "Qwen3-0.6B"
        {
            return Err(SpeechError::Unsupported {
                why: "only the pinned Fun-ASR-Nano-2512 audio and adaptor geometry is supported"
                    .into(),
            });
        }
        Ok(config)
    }

    fn frontend(&self) -> FrontendConfig {
        FrontendConfig {
            sample_rate: self.sample_rate,
            num_mels: self.num_mels,
            frame_length_ms: self.frame_length_ms,
            frame_shift_ms: self.frame_shift_ms,
            lfr_m: self.lfr_m,
            lfr_n: self.lfr_n,
        }
    }
}

/// Loaded Fun-ASR-Nano profile. It reads only an already-downloaded local
/// snapshot and does not fetch model assets implicitly.
pub struct FunAsrNano {
    config: Config,
    audio_encoder: AudioEncoder,
    audio_adaptor: AudioAdaptor,
    decoder: Decoder,
    tokenizer: AudioTokenizer,
}

struct AudioStages {
    features: Vec<f32>,
    encoder: Vec<f32>,
    adaptor: Vec<f32>,
    feature_rows: usize,
    audio_token_count: usize,
}

impl FunAsrNano {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let root: Value = serde_json::from_slice(&fs::read(&config_path).map_err(|error| {
            SpeechError::Input {
                why: format!("cannot read {}: {error}", config_path.display()),
            }
        })?)
        .map_err(|error| bad_config("config.json", error.to_string()))?;
        let config = Config::from_json(&root)?;
        let text_config = TextConfig::from_root(&root)?;
        if !text_config.tie_word_embeddings || text_config.hidden_size != 1024 {
            return Err(SpeechError::Unsupported {
                why: "Fun-ASR-Nano requires the tied 1024-wide Qwen3 decoder profile".into(),
            });
        }
        let weights = SafetensorsFile::open(&model_dir.join("model.safetensors"))?;
        let audio_encoder = AudioEncoder::load(&weights, &config)?;
        let audio_adaptor = AudioAdaptor::load(&weights, &config)?;
        let decoder = Decoder::load(
            &weights,
            &text_config,
            QuantScheme {
                bits: 4,
                group_size: 128,
            },
        )?;
        if decoder.hidden_size != text_config.hidden_size
            || decoder.vocab_size != text_config.vocab_size
        {
            return Err(bad_config("text_config", "Qwen3 decoder geometry mismatch"));
        }
        let tokenizer_dir = model_dir.join(&config.tokenizer_path);
        let tokenizer_path = tokenizer_dir.join("tokenizer.json");
        let tokenizer = AudioTokenizer::from_file(&tokenizer_path)
            .map_err(|error| bad_config("Qwen3 tokenizer.json", error.to_string()))?;
        for token in ["<|im_end|>", "<|endoftext|>"] {
            if tokenizer.token_to_id(token).is_none() {
                return Err(bad_config(
                    "Qwen3 tokenizer.json",
                    format!("required token {token} is missing"),
                ));
            }
        }
        Ok(Self {
            config,
            audio_encoder,
            audio_adaptor,
            decoder,
            tokenizer,
        })
    }

    pub fn profile(&self) -> FunAsrNanoProfile {
        FUN_ASR_NANO_2512
    }

    /// Transcribe one mono waveform sampled at 16 kHz using greedy decoding.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        self.transcribe_with_max_tokens(samples, self.config.default_max_tokens)
    }

    pub fn transcribe_with_max_tokens(&self, samples: &[f32], max_tokens: usize) -> Result<String> {
        if max_tokens == 0 {
            return Err(SpeechError::Input {
                why: "Fun-ASR-Nano max_tokens must be positive".into(),
            });
        }
        let stages = self.encode_audio(samples)?;
        self.decode_audio(
            &stages.adaptor,
            stages.feature_rows,
            stages.audio_token_count,
            max_tokens,
        )
    }

    fn encode_audio(&self, samples: &[f32]) -> Result<AudioStages> {
        if samples.is_empty() {
            return Err(SpeechError::Input {
                why: "audio must contain at least one sample".into(),
            });
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(SpeechError::Input {
                why: "audio samples must be finite".into(),
            });
        }

        let fbank = compute_fbank(samples, self.config.frontend())?;
        let (features, feature_rows) = stack_lfr(
            &fbank,
            self.config.num_mels,
            self.config.lfr_m,
            self.config.lfr_n,
        )?;
        let encoded = self.audio_encoder.forward(&features, feature_rows)?;
        let adapted = self.audio_adaptor.forward(&encoded, feature_rows)?;
        Ok(AudioStages {
            features,
            encoder: encoded,
            adaptor: adapted,
            feature_rows,
            audio_token_count: fake_token_length(feature_rows),
        })
    }

    fn decode_audio(
        &self,
        adapted: &[f32],
        feature_rows: usize,
        audio_token_count: usize,
        max_tokens: usize,
    ) -> Result<String> {
        if audio_token_count == 0
            || audio_token_count > feature_rows
            || adapted.len() != feature_rows * self.decoder.hidden_size
        {
            return Err(SpeechError::Tensor {
                name: "Fun-ASR-Nano audio adaptor output".into(),
                why: "audio token lengths do not match encoded feature rows".into(),
            });
        }
        let prefix = self
            .tokenizer
            .encode(prompt_prefix().as_str(), false)
            .map_err(|error| SpeechError::Input {
                why: format!("Fun-ASR-Nano prompt tokenization failed: {error}"),
            })?;
        let suffix = self
            .tokenizer
            .encode(prompt_suffix(), false)
            .map_err(|error| SpeechError::Input {
                why: format!("Fun-ASR-Nano prompt tokenization failed: {error}"),
            })?;
        let mut token_ids = prefix
            .get_ids()
            .iter()
            .map(|&id| {
                i32::try_from(id).map_err(|_| SpeechError::Input {
                    why: "Qwen3 prompt token id exceeds signed 32-bit range".into(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let audio_start = token_ids.len();
        // The reference inserts token ID 0 placeholders, then replaces their
        // embeddings with projected audio features before the Qwen prefill.
        token_ids.resize(audio_start + audio_token_count, 0);
        let suffix_ids = suffix
            .get_ids()
            .iter()
            .map(|&id| {
                i32::try_from(id).map_err(|_| SpeechError::Input {
                    why: "Qwen3 suffix token id exceeds signed 32-bit range".into(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        token_ids.extend(suffix_ids);

        let mut embeddings = self.decoder.embed(&token_ids)?;
        for audio_row in 0..audio_token_count {
            let position = audio_start + audio_row;
            let target = position * self.decoder.hidden_size;
            let source = audio_row * self.decoder.hidden_size;
            embeddings[target..target + self.decoder.hidden_size]
                .copy_from_slice(&adapted[source..source + self.decoder.hidden_size]);
        }
        let rows = embeddings.len() / self.decoder.hidden_size;
        let (mut last_hidden, mut cache) = self.decoder.prefill(&embeddings, rows);
        let stop_ids = [
            self.tokenizer.token_to_id("<|im_end|>").unwrap(),
            self.tokenizer.token_to_id("<|endoftext|>").unwrap(),
        ];
        let mut generated = Vec::new();
        for _ in 0..max_tokens {
            let logits = self.decoder.logits(&last_hidden);
            let next = argmax(&logits);
            if stop_ids.contains(&(next as u32)) {
                break;
            }
            let token_id = i32::try_from(next).map_err(|_| SpeechError::Input {
                why: "Qwen3 generated token id exceeds signed 32-bit range".into(),
            })?;
            generated.push(next as u32);
            let next_embedding = self.decoder.embed(&[token_id])?;
            last_hidden = self.decoder.step(&next_embedding, &mut cache);
        }
        self.tokenizer
            .decode(&generated, false)
            .map(|text| text.trim().to_owned())
            .map_err(|error| SpeechError::Input {
                why: format!("Fun-ASR-Nano output detokenization failed: {error}"),
            })
    }
}

fn prompt_prefix() -> String {
    format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}\u{ff1a}",
        "\u{8bed}\u{97f3}\u{8f6c}\u{5199}"
    )
}

fn prompt_suffix() -> &'static str {
    "<|im_end|>\n<|im_start|>assistant\n"
}

fn fake_token_length(speech_length: usize) -> usize {
    speech_length.div_ceil(2).div_ceil(2).div_ceil(2).max(1)
}

fn stack_lfr(
    fbank: &[f32],
    num_mels: usize,
    lfr_m: usize,
    lfr_n: usize,
) -> Result<(Vec<f32>, usize)> {
    if num_mels == 0 || lfr_m == 0 || lfr_n == 0 || fbank.len() % num_mels != 0 {
        return Err(bad_config("frontend_conf", "invalid FBANK or LFR geometry"));
    }
    let fbank_rows = fbank.len() / num_mels;
    if fbank_rows == 0 {
        return Err(SpeechError::Input {
            why: "audio is shorter than one complete Fun-ASR-Nano analysis frame".into(),
        });
    }
    let output_rows = fbank_rows.div_ceil(lfr_n);
    let width = num_mels * lfr_m;
    let left_pad = (lfr_m - 1) / 2;
    let mut output = vec![0.0; output_rows * width];
    for row in 0..output_rows {
        for stack in 0..lfr_m {
            let source = (row * lfr_n + stack)
                .saturating_sub(left_pad)
                .min(fbank_rows - 1);
            let src = &fbank[source * num_mels..(source + 1) * num_mels];
            let dst = row * width + stack * num_mels;
            output[dst..dst + num_mels].copy_from_slice(src);
        }
    }
    Ok((output, output_rows))
}

fn add(left: &[f32], right: &[f32]) -> Vec<f32> {
    debug_assert_eq!(left.len(), right.len());
    left.iter().zip(right).map(|(&a, &b)| a + b).collect()
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

struct Linear {
    weight: Vec<f32>,
    bias: Vec<f32>,
    input: usize,
    output: usize,
}

impl Linear {
    fn load(file: &SafetensorsFile, prefix: &str, input: usize, output: usize) -> Result<Self> {
        Ok(Self {
            weight: load_tensor(file, &format!("{prefix}.weight"), &[output, input])?,
            bias: load_tensor(file, &format!("{prefix}.bias"), &[output])?,
            input,
            output,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        ops::linear(
            x,
            &self.weight,
            Some(&self.bias),
            rows,
            self.input,
            self.output,
        )
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

struct SanmAttention {
    qkv: Linear,
    output: Linear,
    fsmn_weight: Vec<f32>,
    width: usize,
    heads: usize,
    kernel: usize,
    left_padding: usize,
}

impl SanmAttention {
    fn load(file: &SafetensorsFile, prefix: &str, input: usize, config: &Config) -> Result<Self> {
        let width = config.output_size;
        let kernel = config.fsmn_kernel;
        let left_padding = (kernel - 1) / 2 + config.sanm_shift;
        if width % config.attention_heads != 0 || left_padding > kernel - 1 {
            return Err(bad_config(
                "audio_encoder_conf",
                "invalid SANM attention geometry",
            ));
        }
        Ok(Self {
            qkv: Linear::load(file, &format!("{prefix}.linear_q_k_v"), input, 3 * width)?,
            output: Linear::load(file, &format!("{prefix}.linear_out"), width, width)?,
            fsmn_weight: load_tensor(
                file,
                &format!("{prefix}.fsmn_block.weight"),
                &[width, kernel, 1],
            )?,
            width,
            heads: config.attention_heads,
            kernel,
            left_padding,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let width = self.width;
        let head_width = width / self.heads;
        let qkv = self.qkv.forward(x, rows);
        let mut attended = vec![0.0f32; rows * width];
        let mut scores = vec![0.0f32; rows];
        let scale = (head_width as f32).sqrt().recip();
        for time in 0..rows {
            for head in 0..self.heads {
                for source in 0..rows {
                    let mut dot = 0.0f32;
                    for dim in 0..head_width {
                        let col = head * head_width + dim;
                        dot += qkv[time * 3 * width + col] * qkv[source * 3 * width + width + col];
                    }
                    scores[source] = dot * scale;
                }
                ops::softmax_row(&mut scores);
                for dim in 0..head_width {
                    let col = head * head_width + dim;
                    let mut sum = 0.0f32;
                    for source in 0..rows {
                        sum += scores[source] * qkv[source * 3 * width + 2 * width + col];
                    }
                    attended[time * width + col] = sum;
                }
            }
        }
        let mut output = self.output.forward(&attended, rows);
        for time in 0..rows {
            for channel in 0..width {
                let mut memory = 0.0f32;
                for kernel_index in 0..self.kernel {
                    let source = time as isize + kernel_index as isize - self.left_padding as isize;
                    if source >= 0 && source < rows as isize {
                        let value = qkv[source as usize * 3 * width + 2 * width + channel];
                        memory += value * self.fsmn_weight[channel * self.kernel + kernel_index];
                    }
                }
                let value = qkv[time * 3 * width + 2 * width + channel];
                output[time * width + channel] += memory + value;
            }
        }
        output
    }
}

struct EncoderLayer {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attention: SanmAttention,
    ff1: Linear,
    ff2: Linear,
    input: usize,
    output: usize,
}

impl EncoderLayer {
    fn load(file: &SafetensorsFile, prefix: &str, input: usize, config: &Config) -> Result<Self> {
        let output = config.output_size;
        let attention_prefix = format!("{prefix}.self_attn");
        Ok(Self {
            norm1: LayerNorm::load(file, &format!("{prefix}.norm1"), input)?,
            norm2: LayerNorm::load(file, &format!("{prefix}.norm2"), output)?,
            attention: SanmAttention::load(file, &attention_prefix, input, config)?,
            ff1: Linear::load(
                file,
                &format!("{prefix}.feed_forward.w_1"),
                output,
                config.linear_units,
            )?,
            ff2: Linear::load(
                file,
                &format!("{prefix}.feed_forward.w_2"),
                config.linear_units,
                output,
            )?,
            input,
            output,
        })
    }

    fn forward(&self, input: &[f32], rows: usize) -> Vec<f32> {
        let mut normalized = input.to_vec();
        self.norm1.apply(&mut normalized, rows);
        let attended = self.attention.forward(&normalized, rows);
        let mut hidden = if self.input == self.output {
            add(input, &attended)
        } else {
            attended
        };
        let residual = hidden.clone();
        self.norm2.apply(&mut hidden, rows);
        let mut feedforward = self.ff1.forward(&hidden, rows);
        for value in &mut feedforward {
            *value = value.max(0.0);
        }
        add(&residual, &self.ff2.forward(&feedforward, rows))
    }
}

struct AudioEncoder {
    first: EncoderLayer,
    layers: Vec<EncoderLayer>,
    after_norm: LayerNorm,
    tp_layers: Vec<EncoderLayer>,
    tp_norm: LayerNorm,
    config: Config,
}

impl AudioEncoder {
    fn load(file: &SafetensorsFile, config: &Config) -> Result<Self> {
        let prefix = "audio_encoder";
        let first = EncoderLayer::load(
            file,
            &format!("{prefix}.encoders0.0"),
            config.input_size,
            config,
        )?;
        let mut layers = Vec::with_capacity(config.num_blocks - 1);
        for index in 0..config.num_blocks - 1 {
            layers.push(EncoderLayer::load(
                file,
                &format!("{prefix}.encoders.{index}"),
                config.output_size,
                config,
            )?);
        }
        let after_norm =
            LayerNorm::load(file, &format!("{prefix}.after_norm"), config.output_size)?;
        let mut tp_layers = Vec::with_capacity(config.tp_blocks);
        for index in 0..config.tp_blocks {
            tp_layers.push(EncoderLayer::load(
                file,
                &format!("{prefix}.tp_encoders.{index}"),
                config.output_size,
                config,
            )?);
        }
        let tp_norm = LayerNorm::load(file, &format!("{prefix}.tp_norm"), config.output_size)?;
        Ok(Self {
            first,
            layers,
            after_norm,
            tp_layers,
            tp_norm,
            config: config.clone(),
        })
    }

    fn forward(&self, features: &[f32], rows: usize) -> Result<Vec<f32>> {
        if features.len() != rows * self.config.input_size {
            return Err(SpeechError::Tensor {
                name: "Fun-ASR-Nano FBANK/LFR features".into(),
                why: "feature size does not match configured input width".into(),
            });
        }
        let mut hidden = features.to_vec();
        add_sinusoidal_position(
            &mut hidden,
            rows,
            self.config.input_size,
            self.config.output_size,
        );
        hidden = self.first.forward(&hidden, rows);
        for layer in &self.layers {
            hidden = layer.forward(&hidden, rows);
        }
        self.after_norm.apply(&mut hidden, rows);
        for layer in &self.tp_layers {
            hidden = layer.forward(&hidden, rows);
        }
        self.tp_norm.apply(&mut hidden, rows);
        Ok(hidden)
    }
}

fn add_sinusoidal_position(values: &mut [f32], rows: usize, width: usize, scale_width: usize) {
    let half = width / 2;
    let log_timescale_increment = 10_000.0f32.ln() / (half - 1) as f32;
    let scale = (scale_width as f32).sqrt();
    for row in 0..rows {
        for dim in 0..half {
            let inverse_timescale = (-(dim as f32) * log_timescale_increment).exp();
            let scaled_time = (row + 1) as f32 * inverse_timescale;
            let sin_index = row * width + dim;
            let cos_index = row * width + half + dim;
            values[sin_index] = values[sin_index] * scale + scaled_time.sin();
            values[cos_index] = values[cos_index] * scale + scaled_time.cos();
        }
    }
}

struct SelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    width: usize,
    heads: usize,
}

impl SelfAttention {
    fn load(file: &SafetensorsFile, prefix: &str, width: usize, heads: usize) -> Result<Self> {
        if width % heads != 0 {
            return Err(bad_config(
                "audio_adaptor_conf",
                "attention width must divide by heads",
            ));
        }
        Ok(Self {
            query: Linear::load(file, &format!("{prefix}.linear_q"), width, width)?,
            key: Linear::load(file, &format!("{prefix}.linear_k"), width, width)?,
            value: Linear::load(file, &format!("{prefix}.linear_v"), width, width)?,
            output: Linear::load(file, &format!("{prefix}.linear_out"), width, width)?,
            width,
            heads,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let width = self.width;
        let head_width = width / self.heads;
        let query = self.query.forward(x, rows);
        let key = self.key.forward(x, rows);
        let value = self.value.forward(x, rows);
        let mut attended = vec![0.0f32; rows * width];
        let mut scores = vec![0.0f32; rows];
        let scale = (head_width as f32).sqrt().recip();
        for time in 0..rows {
            for head in 0..self.heads {
                let offset = head * head_width;
                for source in 0..rows {
                    let mut dot = 0.0;
                    for dim in 0..head_width {
                        dot +=
                            query[time * width + offset + dim] * key[source * width + offset + dim];
                    }
                    scores[source] = dot * scale;
                }
                ops::softmax_row(&mut scores);
                for dim in 0..head_width {
                    let mut sum = 0.0;
                    for source in 0..rows {
                        sum += scores[source] * value[source * width + offset + dim];
                    }
                    attended[time * width + offset + dim] = sum;
                }
            }
        }
        self.output.forward(&attended, rows)
    }
}

struct AdaptorLayer {
    norm1: LayerNorm,
    attention: SelfAttention,
    norm2: LayerNorm,
    ff1: Linear,
    ff2: Linear,
    width: usize,
}

impl AdaptorLayer {
    fn load(file: &SafetensorsFile, prefix: &str, width: usize, heads: usize) -> Result<Self> {
        let attention_prefix = format!("{prefix}.self_attn");
        let hidden_units = width / 4;
        Ok(Self {
            norm1: LayerNorm::load(file, &format!("{prefix}.norm1"), width)?,
            attention: SelfAttention::load(file, &attention_prefix, width, heads)?,
            norm2: LayerNorm::load(file, &format!("{prefix}.norm2"), width)?,
            ff1: Linear::load(
                file,
                &format!("{prefix}.feed_forward.w_1"),
                width,
                hidden_units,
            )?,
            ff2: Linear::load(
                file,
                &format!("{prefix}.feed_forward.w_2"),
                hidden_units,
                width,
            )?,
            width,
        })
    }

    fn forward(&self, values: &[f32], rows: usize) -> Vec<f32> {
        let mut normalized = values.to_vec();
        self.norm1.apply(&mut normalized, rows);
        let hidden = add(values, &self.attention.forward(&normalized, rows));
        let mut normalized = hidden.clone();
        self.norm2.apply(&mut normalized, rows);
        let mut feedforward = self.ff1.forward(&normalized, rows);
        for value in &mut feedforward {
            *value = value.max(0.0);
        }
        debug_assert_eq!(self.width, self.ff2.output);
        add(&hidden, &self.ff2.forward(&feedforward, rows))
    }
}

struct AudioAdaptor {
    linear1: Linear,
    linear2: Linear,
    blocks: Vec<AdaptorLayer>,
    downsample_rate: usize,
    encoder_dim: usize,
    llm_dim: usize,
}

impl AudioAdaptor {
    fn load(file: &SafetensorsFile, config: &Config) -> Result<Self> {
        let root = "audio_adaptor";
        let k = config.adaptor_downsample_rate;
        let encoder_dim = config.output_size;
        let llm_dim = 1024;
        let mut blocks = Vec::with_capacity(config.adaptor_num_layers);
        for index in 0..config.adaptor_num_layers {
            blocks.push(AdaptorLayer::load(
                file,
                &format!("{root}.blocks.{index}"),
                llm_dim,
                config.adaptor_attention_heads,
            )?);
        }
        Ok(Self {
            linear1: Linear::load(
                file,
                &format!("{root}.linear1"),
                encoder_dim * k,
                config.adaptor_ffn_dim,
            )?,
            linear2: Linear::load(
                file,
                &format!("{root}.linear2"),
                config.adaptor_ffn_dim,
                llm_dim,
            )?,
            blocks,
            downsample_rate: k,
            encoder_dim,
            llm_dim,
        })
    }

    fn forward(&self, features: &[f32], rows: usize) -> Result<Vec<f32>> {
        if features.len() != rows * self.encoder_dim {
            return Err(SpeechError::Tensor {
                name: "Fun-ASR-Nano audio encoder output".into(),
                why: "feature size does not match adaptor input width".into(),
            });
        }
        let groups = rows.div_ceil(self.downsample_rate);
        let mut grouped = vec![0.0; groups * self.encoder_dim * self.downsample_rate];
        for group in 0..groups {
            for offset in 0..self.downsample_rate {
                let row = group * self.downsample_rate + offset;
                if row < rows {
                    let source = row * self.encoder_dim;
                    let target = (group * self.downsample_rate + offset) * self.encoder_dim;
                    grouped[target..target + self.encoder_dim]
                        .copy_from_slice(&features[source..source + self.encoder_dim]);
                }
            }
        }
        let mut adapted = self.linear1.forward(&grouped, groups);
        for value in &mut adapted {
            *value = value.max(0.0);
        }
        adapted = self.linear2.forward(&adapted, groups);
        for block in &self.blocks {
            adapted = block.forward(&adapted, groups);
        }
        debug_assert_eq!(adapted.len(), groups * self.llm_dim);
        Ok(adapted)
    }
}

#[cfg(test)]
mod tests {
    use super::{fake_token_length, stack_lfr};
    use serde_json::Value;
    use std::path::Path;

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../testdata/fun_asr_nano_reference.json"
        ))
        .expect("valid Fun-ASR-Nano fixture")
    }

    #[test]
    fn fake_audio_token_length_matches_two_stride_two_convolutions_and_pooling() {
        assert_eq!(fake_token_length(1), 1);
        assert_eq!(fake_token_length(8), 1);
        assert_eq!(fake_token_length(9), 2);
        assert_eq!(fake_token_length(47), 6);
    }

    #[test]
    fn lfr_stack_repeats_the_first_and_last_frame_at_boundaries() {
        let fbank = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
        let (stacked, rows) = stack_lfr(&fbank, 2, 3, 2).unwrap();
        assert_eq!(rows, 2);
        assert_eq!(
            stacked,
            vec![1.0, 10.0, 1.0, 10.0, 2.0, 20.0, 2.0, 20.0, 3.0, 30.0, 3.0, 30.0,]
        );
    }

    #[test]
    fn config_rejects_other_model_families() {
        let error =
            super::Config::from_json(&serde_json::json!({"model_type":"qwen3_asr"})).unwrap_err();
        assert!(error.to_string().contains("fun_asr_nano"));
    }

    #[test]
    #[ignore = "requires the pinned Fun-ASR-Nano snapshot in TURBOSPARK_FUN_ASR_NANO_DIR"]
    fn pinned_checkpoint_matches_mlx_transcript() {
        let model_dir = std::env::var_os("TURBOSPARK_FUN_ASR_NANO_DIR")
            .expect("set TURBOSPARK_FUN_ASR_NANO_DIR to the pinned local snapshot");
        let model = super::FunAsrNano::load(Path::new(&model_dir)).expect("checkpoint loads");
        let audio_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_forced_aligner_reference.wav");
        let audio = crate::wav::read_wav_f32(&audio_path).expect("reference WAV loads");
        assert_eq!(audio.sample_rate, 16_000);
        assert_eq!(audio.channels, 1);
        let expected = fixture();
        let stages = model.encode_audio(&audio.samples).unwrap();
        assert_eq!(
            stages.feature_rows,
            expected["feature_frames"].as_u64().unwrap() as usize
        );
        assert_eq!(
            stages.audio_token_count,
            expected["fake_audio_tokens"].as_u64().unwrap() as usize
        );
        assert_sampled_stage(
            &stages.features,
            stages.feature_rows,
            model.config.input_size,
            &expected["lfr_features"],
        );
        assert_sampled_stage(
            &stages.encoder,
            stages.feature_rows,
            model.config.output_size,
            &expected["audio_encoder"],
        );
        assert_sampled_stage(
            &stages.adaptor,
            stages.feature_rows,
            model.audio_adaptor.llm_dim,
            &expected["audio_adaptor"],
        );
        let text = model
            .decode_audio(
                &stages.adaptor,
                stages.feature_rows,
                stages.audio_token_count,
                32,
            )
            .unwrap();
        assert_eq!(text, expected["transcript"].as_str().unwrap());
    }

    fn assert_sampled_stage(values: &[f32], rows: usize, columns: usize, expected: &Value) {
        assert_eq!(expected["shape"][0].as_u64().unwrap() as usize, rows);
        assert_eq!(expected["shape"][1].as_u64().unwrap() as usize, columns);
        let row_ids: Vec<usize> = serde_json::from_value(expected["rows"].clone()).unwrap();
        let column_ids: Vec<usize> = serde_json::from_value(expected["columns"].clone()).unwrap();
        let expected_values: Vec<Vec<f32>> =
            serde_json::from_value(expected["values"].clone()).unwrap();
        let mut max_abs = 0.0f32;
        for (row_index, &row) in row_ids.iter().enumerate() {
            for (column_index, &column) in column_ids.iter().enumerate() {
                max_abs = max_abs.max(
                    (values[row * columns + column] - expected_values[row_index][column_index])
                        .abs(),
                );
            }
        }
        assert!(max_abs < 0.01, "stage max abs difference was {max_abs}");
    }
}
