//! IndexTTS (GPT-style mel-code TTS with a BigVGAN vocoder).
//!
//! Port of `mlx_audio/tts/models/indextts/` (indextts.py, gpt2.py,
//! conformer.py, perceiver.py, attention.py, mel.py, bigvgan.py, and
//! the local `ecapa_tdnn` speaker encoder) at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/indextts):
//! a gpt2 backbone over the shared [`crate::lm`] port with BOTH
//! embedding tables patched to identity (embeddings arrive
//! precomputed), a Conformer + PerceiverResampler conditioning stack
//! over the 100-bin log-mel reference, per-position learned position
//! embeddings for the text and mel streams, and a
//! speaker-conditioned BigVGAN vocoder that turns every step's
//! post-norm hidden rows into the waveform. See [README.md](README.md)
//! for the verification evidence and open items.
//!
//! Scope of this port, matching the reference single-prompt `generate`:
//! batch size 1, non-streaming decode, and prompt construction from
//! pre-tokenized ids. Text tokenization (sentencepiece over the
//! reference's CJK-char text normalization) is a caller concern; the
//! reference audio path takes the reference log-mel directly.

use serde_json::Value;

use crate::lm;
use crate::lm::engine::Input;
use crate::music::minimax_music3::rng;
use crate::nn::LayerNorm;
use crate::tts::sampling;
use crate::{quant, Result, SpeechError};

mod conformer;
mod perceiver;
mod speaker;
mod vocoder;

pub use conformer::ConformerArgs;
pub use speaker::EcapaTdnnArgs;
pub use vocoder::BigvganConditioningConfig;

use conformer::Conformer;
use perceiver::PerceiverResampler;
use speaker::SpeakerEncoder;
use vocoder::BigvganConditioning;

/// Reference sample rate (the deployed vocoder emits 24 kHz mono).
pub const DEFAULT_SAMPLE_RATE: u32 = 24000;
/// Log-mel bins per reference frame (the conformer input and the
/// `num_mels` of the reference extractor).
pub const REF_MEL_BINS: usize = 100;

/// The GPT-side settings (`GPTConfig`).
#[derive(Debug, Clone)]
pub struct GptConfig {
    pub model_dim: usize,
    pub heads: usize,
    pub layers: usize,
    pub max_mel_tokens: usize,
    pub max_text_tokens: usize,
    pub number_text_tokens: usize,
    pub number_mel_codes: usize,
    pub start_mel_token: i32,
    pub stop_mel_token: i32,
    pub start_text_token: i32,
    pub stop_text_token: i32,
    /// Parsed and required true: the reference raises
    /// NotImplementedError otherwise.
    pub use_mel_codes_as_input: bool,
    pub mel_length_compression: usize,
    /// Parsed and required `"conformer_perceiver"`: the reference raises
    /// NotImplementedError otherwise.
    pub condition_type: String,
    pub condition_module: ConformerArgs,
    pub max_conditioning_inputs: usize,
    pub condition_num_latent: usize,
}

impl GptConfig {
    /// The `gpt` config dict; every field except the two conditioner
    /// sizing defaults is required, matching the reference dataclass.
    pub fn from_json(value: &Value) -> Result<Self> {
        let u = |key: &str| -> Result<usize> {
            value
                .get(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .ok_or_else(|| SpeechError::BadConfig {
                    field: key.to_string(),
                    why: "required".into(),
                })
        };
        let i = |key: &str| -> Result<i32> {
            value
                .get(key)
                .and_then(Value::as_i64)
                .map(|n| n as i32)
                .ok_or_else(|| SpeechError::BadConfig {
                    field: key.to_string(),
                    why: "required".into(),
                })
        };
        // Required: the reference dataclass default (`input_layer: "conv2d"`)
        // fails its own CONV_LAYERS lookup, so every loadable checkpoint
        // carries the block.
        let condition_module = match value.get("condition_module") {
            Some(module) => ConformerArgs::from_json(module)?,
            None => {
                return Err(SpeechError::BadConfig {
                    field: "gpt.condition_module".into(),
                    why: "required (the reference dataclass default input_layer                           \"conv2d\" fails its own CONV_LAYERS lookup)"
                        .into(),
                })
            }
        };
        Ok(GptConfig {
            model_dim: u("model_dim")?,
            heads: u("heads")?,
            layers: u("layers")?,
            max_mel_tokens: u("max_mel_tokens")?,
            max_text_tokens: u("max_text_tokens")?,
            number_text_tokens: u("number_text_tokens")?,
            number_mel_codes: u("number_mel_codes")?,
            start_mel_token: i("start_mel_token")?,
            stop_mel_token: i("stop_mel_token")?,
            start_text_token: i("start_text_token")?,
            stop_text_token: i("stop_text_token")?,
            use_mel_codes_as_input: value
                .get("use_mel_codes_as_input")
                .and_then(Value::as_bool)
                .ok_or_else(|| SpeechError::BadConfig {
                    field: "use_mel_codes_as_input".into(),
                    why: "required".into(),
                })?,
            mel_length_compression: u("mel_length_compression")?,
            condition_type: value
                .get("condition_type")
                .and_then(Value::as_str)
                .ok_or_else(|| SpeechError::BadConfig {
                    field: "condition_type".into(),
                    why: "required".into(),
                })?
                .to_owned(),
            condition_module,
            max_conditioning_inputs: value
                .get("max_conditioning_inputs")
                .and_then(Value::as_u64)
                .unwrap_or(1) as usize,
            condition_num_latent: value
                .get("condition_num_latent")
                .and_then(Value::as_u64)
                .unwrap_or(32) as usize,
        })
    }
}

/// The top-level `ModelArgs`.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub bigvgan: BigvganConditioningConfig,
    pub gpt: GptConfig,
    /// The sentencepiece tokenizer the reference resolves from the Hub
    /// or a local directory; caller-owned here.
    pub tokenizer_name: Option<String>,
    pub sample_rate: u32,
}

impl ModelConfig {
    /// `ModelArgs` from a `config.json` dict (missing optional fields
    /// take the dataclass defaults).
    pub fn from_json(value: &Value) -> Result<Self> {
        let bigvgan = value
            .get("bigvgan")
            .map(BigvganConditioningConfig::from_json)
            .transpose()?
            .ok_or_else(|| SpeechError::BadConfig {
                field: "bigvgan".into(),
                why: "required".into(),
            })?;
        let gpt = value
            .get("gpt")
            .map(GptConfig::from_json)
            .transpose()?
            .ok_or_else(|| SpeechError::BadConfig {
                field: "gpt".into(),
                why: "required".into(),
            })?;
        let config = ModelConfig {
            bigvgan,
            gpt,
            tokenizer_name: value
                .get("tokenizer_name")
                .and_then(Value::as_str)
                .map(str::to_owned),
            sample_rate: value
                .get("sample_rate")
                .and_then(Value::as_u64)
                .unwrap_or(DEFAULT_SAMPLE_RATE as u64) as u32,
        };
        if !config.gpt.use_mel_codes_as_input {
            return Err(SpeechError::Unsupported {
                why: "use_mel_codes_as_input=false is not supported by the reference \
                      (NotImplementedError); the deployed checkpoints run with true"
                    .into(),
            });
        }
        if config.gpt.condition_type != "conformer_perceiver" {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "condition_type={:?} is not supported by the reference \
                     (NotImplementedError); the deployed checkpoints run with \
                     conformer_perceiver",
                    config.gpt.condition_type
                ),
            });
        }
        Ok(config)
    }
}

/// Sampling controls for one generation (`make_sampler(temp=0.8,
/// top_k=30)`, the reference default, plus the loop cap).
#[derive(Debug, Clone)]
pub struct GenerateParams {
    /// 0.0 selects the exact greedy path (`make_sampler(0, ...)`).
    pub temperature: f32,
    /// The reference passes raw logits through the sampler chain; a
    /// top-p filter there would `exp` unnormalized logits (a reference
    /// quirk this port reproduces when enabled). The default sampler
    /// leaves it off.
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub max_tokens: usize,
}

impl Default for GenerateParams {
    fn default() -> Self {
        GenerateParams {
            temperature: 0.8,
            top_p: None,
            top_k: Some(30),
            max_tokens: 5000,
        }
    }
}

/// Why the generate loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The sampler drew `stop_mel_token`.
    StopMelToken,
    /// The token budget ran out.
    MaxTokens,
}

/// One generated utterance: the sampled mel codes (without the stop
/// token), the per-step post-norm hidden rows the vocoder consumed, and
/// the decoded mono waveform.
#[derive(Debug, Clone)]
pub struct Generated {
    pub tokens: Vec<i32>,
    /// `[steps, model_dim]` row-major latent states (one row per loop
    /// iteration, including the stop-token step, like the reference
    /// `latent_states`).
    pub latent_states: Vec<f32>,
    pub samples: Vec<f32>,
    pub stopped_by: StopReason,
}

/// The loaded family model (`Model`).
pub struct Model {
    pub config: ModelConfig,
    text_embedding: Vec<f32>,
    mel_embedding: Vec<f32>,
    /// Learned position tables (`LearnedPositionEncoding.emb`).
    mel_pos_embedding: Vec<f32>,
    text_pos_embedding: Vec<f32>,
    /// Parsed and unused by the generate loop (the reference builds it
    /// for text decoding it never runs).
    #[allow(dead_code)]
    text_head: crate::nn::Linear,
    mel_head: crate::nn::Linear,
    conditioning_encoder: Conformer,
    perceiver_encoder: PerceiverResampler,
    gpt: lm::Gpt2Model,
    final_norm: LayerNorm,
    bigvgan: BigvganConditioning,
    speaker: SpeakerEncoder,
}

impl Model {
    /// Opens a checkpoint directory (`config.json` + safetensors).
    pub fn open(dir: &std::path::Path) -> Result<Self> {
        let config_value = quant::read_json(&dir.join("config.json"))?;
        let config = ModelConfig::from_json(&config_value)?;
        let weights_path = dir.join("model.safetensors");
        let gpt_weights = lm::Weights::open_single(&weights_path)?;
        let file = turbospark_model_io::safetensors::SafetensorsFile::open(&weights_path)?;
        Self::load(&file, &gpt_weights, &config)
    }

    /// Loads from one checkpoint shard (plus its [`lm::Weights`] view
    /// for the gpt2 block) and the parsed config.
    pub fn load(
        file: &turbospark_model_io::safetensors::SafetensorsFile,
        gpt_weights: &lm::Weights,
        config: &ModelConfig,
    ) -> Result<Self> {
        use crate::codec::conv::{load_bias, BiasMode};
        use crate::codec::wnconv::load_f32_shaped;
        let gpt = &config.gpt;
        let dim = gpt.model_dim;
        let linear = |name: &str, input: usize, output: usize| -> Result<crate::nn::Linear> {
            let weight = load_f32_shaped(file, &format!("{name}.weight"), &[output, input])?;
            let bias = load_bias(file, &format!("{name}.bias"), output, BiasMode::Required)?;
            Ok(crate::nn::Linear::new(weight, bias, input, output))
        };
        let embedding = |name: &str, rows: usize| -> Result<Vec<f32>> {
            load_f32_shaped(file, &format!("{name}.weight"), &[rows, dim])
        };
        // The gpt2 backbone over the family's `gpt.` names, position and
        // token tables patched away (Identity in the reference).
        let gpt_args = lm::gpt2::Gpt2Config {
            model_type: "gpt2".into(),
            n_ctx: 1,
            n_embd: dim,
            n_head: gpt.heads,
            n_layer: gpt.layers,
            n_positions: 1,
            layer_norm_epsilon: 1e-5,
            vocab_size: 1,
            num_key_value_heads: gpt.heads,
        };
        let gpt_model =
            lm::Gpt2Model::from_config_without_tables_at_root(gpt_weights, gpt_args, "gpt.")?
                .without_position_embeddings();
        let conditioning = Conformer::load(file, &gpt.condition_module, "conditioning_encoder.")?;
        let perceiver = PerceiverResampler::load(
            file,
            "perceiver_encoder.",
            dim,
            gpt.condition_module.output_size,
            gpt.condition_num_latent,
            gpt.condition_module.attention_heads,
            64,
            gpt.condition_module.perceiver_mult,
        )?;
        let bigvgan = BigvganConditioning::load(file, config.bigvgan.clone(), "bigvgan.")?;
        let speaker = SpeakerEncoder::load(
            file,
            "bigvgan.speaker_encoder.",
            &EcapaTdnnArgs::new(
                config.bigvgan.num_mels,
                config.bigvgan.speaker_embedding_dim,
            ),
        )?;
        Ok(Model {
            config: config.clone(),
            text_embedding: embedding("text_embedding", gpt.number_text_tokens + 1)?,
            mel_embedding: embedding("mel_embedding", gpt.number_mel_codes)?,
            mel_pos_embedding: embedding(
                "mel_pos_embedding.emb",
                gpt.max_mel_tokens + 2 + gpt.max_conditioning_inputs,
            )?,
            text_pos_embedding: embedding("text_pos_embedding.emb", gpt.max_text_tokens + 2)?,
            text_head: linear("text_head", dim, gpt.number_text_tokens + 1)?,
            mel_head: linear("mel_head", dim, gpt.number_mel_codes)?,
            conditioning_encoder: conditioning,
            perceiver_encoder: perceiver,
            gpt: gpt_model,
            final_norm: LayerNorm::new(
                load_f32_shaped(file, "final_norm.weight", &[dim])?,
                Some(load_f32_shaped(file, "final_norm.bias", &[dim])?),
                1e-5,
            ),
            bigvgan,
            speaker,
        })
    }

    /// `get_conditioning`: the reference log-mel rows `[T, 100]`
    /// through the Conformer and the perceiver resampler, yielding the
    /// `[condition_num_latent, model_dim]` context rows.
    pub fn get_conditioning(&self, mel: &[f32], frames: usize) -> Result<Vec<f32>> {
        // The encoder trusts `frames`; a flat buffer of any other length
        // would be read with the wrong row stride.
        if frames == 0 || mel.len() != frames * REF_MEL_BINS {
            return Err(SpeechError::Input {
                why: format!(
                    "reference mel holds {} values, expected {frames} rows of {REF_MEL_BINS} bins",
                    mel.len()
                ),
            });
        }
        let latent = self.conditioning_encoder.forward(mel, frames)?;
        let context_frames = latent.len() / self.config.gpt.condition_module.output_size;
        self.perceiver_encoder.forward(&latent, context_frames)
    }

    /// `log_mel_spectrogram` (mel.py): Hann STFT (n_fft 1024, hop 256,
    /// centered, reflect-padded), magnitudes through the unnormalized
    /// HTK 100-band filterbank, `log(max(mel, 1e-5))`. Returns
    /// `[T, 100]` rows. The reference `stft(window="hann")` resolves to
    /// the SYMMETRIC hanning window (denominator `size - 1`), not the
    /// periodic [`crate::dsp::hann_window`].
    pub fn log_mel_spectrogram(&self, samples: &[f32]) -> Result<Vec<Vec<f32>>> {
        let fft_size = 1024usize;
        let window: Vec<f32> = (0..fft_size)
            .map(|i| {
                (0.5 * (1.0
                    - (2.0 * std::f64::consts::PI * i as f64 / (fft_size - 1) as f64).cos()))
                    as f32
            })
            .collect();
        let options = crate::mel::MelSpectrogramOptions {
            stft: crate::stft::StftOptions {
                fft_size,
                hop: 256,
                window,
                center: true,
            },
            num_mels: 100,
            sample_rate: self.config.sample_rate,
            fmin: 0.0,
            fmax: None,
            scale: crate::mel::MelScale::Htk,
            power: 1.0,
        };
        let mel =
            crate::mel::mel_spectrogram(samples, &options).map_err(|e| SpeechError::Input {
                why: format!("log mel spectrogram failed: {e}"),
            })?;
        Ok(mel
            .into_iter()
            .map(|frame| frame.into_iter().map(|x| x.max(1e-5).ln()).collect())
            .collect())
    }

    /// `prepare_input_embedding` for one prompt given the already
    /// computed conditioning rows: the conditioning rows, then the text
    /// rows `[start_text] tokens [stop_text, start_mel]` with their
    /// learned positions. Returns the embedding block and its length.
    pub fn prepare_input_embedding(
        &self,
        conditioning: &[f32],
        text_ids: &[i32],
    ) -> Result<(Vec<f32>, usize)> {
        let gpt = &self.config.gpt;
        let conditioning_rows = conditioning.len() / gpt.model_dim;
        let mut tokens = Vec::with_capacity(text_ids.len() + 3);
        tokens.push(gpt.start_text_token);
        tokens.extend_from_slice(text_ids);
        tokens.push(gpt.stop_text_token);
        tokens.push(gpt.start_mel_token);
        let length = tokens.len();
        if length > gpt.max_text_tokens + 2 {
            return Err(SpeechError::Input {
                why: format!(
                    "prompt of {length} text positions exceeds max_text_tokens + 2 = {}",
                    gpt.max_text_tokens + 2
                ),
            });
        }
        let dim = gpt.model_dim;
        let mut embedding = Vec::with_capacity((conditioning_rows + length) * dim);
        embedding.extend_from_slice(conditioning);
        embedding.extend(crate::lm::engine::lookup_embeddings(
            &self.text_embedding,
            gpt.number_text_tokens + 1,
            dim,
            &tokens,
        )?);
        // + text_pos_embedding(tokens): arange(0, len) rows.
        for (index, _token) in tokens.iter().enumerate() {
            let row = position_row(&self.text_pos_embedding, index, dim)?;
            let base = (conditioning_rows + index) * dim;
            for (c, value) in row.iter().enumerate() {
                embedding[base + c] += *value;
            }
        }
        Ok((embedding, conditioning_rows + length))
    }

    /// The full reference `generate` (single prompt): conditioning from
    /// the reference mel rows, the prompt embedding, the mel-token
    /// loop, then the conditioned BigVGAN decode over every step's
    /// post-norm last row. `seed` only feeds the sampled draws;
    /// temperature 0 never draws and is fully deterministic.
    pub fn generate(
        &mut self,
        ref_mel: &[Vec<f32>],
        text_ids: &[i32],
        params: &GenerateParams,
        seed: u64,
    ) -> Result<Generated> {
        // Negative temperatures invert the distribution and NaN passes
        // every comparison. Zero is the greedy path, handled below.
        if !params.temperature.is_finite() || params.temperature < 0.0 {
            return Err(SpeechError::Input {
                why: format!(
                    "temperature must be finite and >= 0, got {}",
                    params.temperature
                ),
            });
        }
        let gpt = &self.config.gpt;
        let dim = gpt.model_dim;
        let frames = ref_mel.len();
        if frames == 0 {
            return Err(SpeechError::Input {
                why: "empty reference mel".into(),
            });
        }
        // Every row must be a full mel frame. The flat concat below assumes
        // a fixed stride, so a ragged row would shift all later frames.
        if let Some(row) = ref_mel.iter().find(|row| row.len() != REF_MEL_BINS) {
            return Err(SpeechError::Input {
                why: format!(
                    "reference mel row has {} bins, expected {REF_MEL_BINS}",
                    row.len()
                ),
            });
        }
        let mel_rows: Vec<f32> = ref_mel.concat();
        let conditioning = self.get_conditioning(&mel_rows, frames)?;
        let (embedding, embedding_len) = self.prepare_input_embedding(&conditioning, text_ids)?;

        let mut caches = lm::make_prompt_cache(gpt.layers);
        let mut hidden = self
            .gpt
            .forward_hidden(Input::Embeddings(&embedding), Some(&mut caches))?;
        self.final_norm.apply(&mut hidden, embedding_len);

        let mut latent_states: Vec<f32> = Vec::new();
        let mut tokens: Vec<i32> = Vec::new();
        let mut stopped_by = StopReason::MaxTokens;
        let mut keys = rng::KeySequence::new(seed);
        let vocab = gpt.number_mel_codes;
        for step in 0..params.max_tokens {
            // This step's latent: the post-final-norm last row.
            let last = hidden[hidden.len() - dim..].to_vec();
            latent_states.extend_from_slice(&last);
            let mut logits = self.mel_head.forward(&last, 1);
            // make_sampler: top-p filter first (when enabled), then
            // top-k, then the temperature-scaled categorical draw.
            if let Some(top_p) = params.top_p {
                if top_p > 0.0 && top_p < 1.0 {
                    sampling::apply_top_p(&mut logits, top_p);
                }
            }
            if let Some(top_k) = params.top_k {
                if top_k > 0 {
                    if top_k as usize >= vocab {
                        return Err(SpeechError::Input {
                            why: format!("top_k {top_k} outside the (0, {vocab}) interval"),
                        });
                    }
                    sampling::apply_top_k(&mut logits, top_k as usize)?;
                }
            }
            let next = if params.temperature == 0.0 {
                sampling::argmax(&logits)
            } else {
                sampling::sample_categorical(&logits, params.temperature, keys.next())
            };
            let next = next as i32;
            if next == gpt.stop_mel_token {
                stopped_by = StopReason::StopMelToken;
                break;
            }
            tokens.push(next);
            if step + 1 == params.max_tokens {
                stopped_by = StopReason::MaxTokens;
                break;
            }
            // mel_emb = mel_embedding(next) + mel_pos(offset); the next
            // inputs are that single row.
            let mut input =
                crate::lm::engine::lookup_embeddings(&self.mel_embedding, vocab, dim, &[next])?;
            let position = embedding_len + step;
            if position >= gpt.max_mel_tokens + 2 + gpt.max_conditioning_inputs {
                return Err(SpeechError::Input {
                    why: format!("mel position {position} outside the position table"),
                });
            }
            let pos_row = position_row(&self.mel_pos_embedding, position, dim)?;
            for (c, value) in pos_row.iter().enumerate() {
                input[c] += *value;
            }
            hidden = self
                .gpt
                .forward_hidden(Input::Embeddings(&input), Some(&mut caches))?;
            self.final_norm.apply(&mut hidden, 1);
        }

        let steps = latent_states.len() / dim;
        let samples = if steps == 0 {
            Vec::new()
        } else {
            let speaker = self.speaker.forward(&mel_rows, frames)?;
            self.bigvgan.forward(&latent_states, steps, &speaker)?
        };
        Ok(Generated {
            tokens,
            latent_states,
            samples,
            stopped_by,
        })
    }
}

fn position_row(table: &[f32], position: usize, dim: usize) -> Result<Vec<f32>> {
    let start = position
        .checked_mul(dim)
        .filter(|&s| s + dim <= table.len())
        .ok_or_else(|| SpeechError::Input {
            why: format!(
                "position {position} outside a learned position table of {} rows",
                table.len() / dim
            ),
        })?;
    Ok(table[start..start + dim].to_vec())
}

#[cfg(test)]
mod tests;
