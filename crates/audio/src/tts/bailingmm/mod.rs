//! Ming Omni TTS (BailingMM) family: MoE and dense variants.
//!
//! Port of `mlx_audio/tts/models/bailingmm/bailingmm.py` at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/bailingmm),
//! covering the sibling `dense/` directory's `Model` too (the dense
//! variant is the same implementation with
//! `_is_moe_llm_config` forced to false; the `dense` directory rides
//! this module per the MODELS.md alias policy). The reference
//! `ming_omni_tts` architecture: an LLM backbone (bailing MoE over the
//! shared [`crate::lm`] port, or the Ming Qwen2 dense variant) read for
//! hidden states only, an AudioVAE tokenizer (Ming Qwen2 encoder and
//! decoder with a streaming ISTFT head), a DiT flow-matching head
//! (`FlowLoss` + CFM solver) that emits one latent patch per step, a
//! DiT aggregator projecting patches back into the LLM stream, and a
//! two-class stop head. There are NO generated vocabulary tokens: the
//! loop is hidden states -> flow sample -> decode, stopping when the
//! stop head's softmax probability clears 0.5 after step 3.
//!
//! Scope of this port, matching the reference generate path: batch 1,
//! non-streaming audio assembly (the reference decodes per step with
//! its streaming state and concatenates the chunks; the chunk state is
//! implemented exactly, the chunks are joined once at the end), and
//! prompt construction from pre-tokenized segments. Text tokenization
//! is a caller concern (the real checkpoints ship a Hugging Face
//! tokenizer with the `<audio>` / `<audioPatch>` specials the portable
//! crate does not embed); [`MarkerIds`] carries the tokenized marker
//! strings.
//!
//! Checkpoint layout: the LLM tensors sit under `model.model.` (the
//! reference sanitize strips one `model.` off the raw keys and re-adds
//! it), the audio/flow/aggregator/head tensors at the top level. Open
//! the weights with the `model.model.` prefix ([`Model::open`] does) so
//! the shared bailing MoE loader sees its names; every other load here
//! is prefix-free by absolute name.

use serde_json::Value;

use crate::lm::{self, cache::LayerCaches, Bias, Weights};
use crate::nn::Linear;
use crate::tts::bailingmm::dit::{Aggregator, FlowLoss, FlowNoise, FlowParams, SeededNoise};
use crate::{ops, quant, Result, SpeechError};

use crate::tts::bailingmm::audio_vae::{Decoder, Encoder, IstftState, UpsampleState};
use crate::tts::bailingmm::ming_qwen2::{MingQwen2, SlidingConfig};

pub mod audio_vae;
pub mod dit;
pub mod ming_qwen2;

/// The reference default prompt when the caller does not override it.
pub const DEFAULT_PROMPT: &str = "Please generate speech based on the following description.\n";

/// Why the generate loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    StopHead,
    MaxDecodeSteps,
}

/// One generated utterance: the per-step latent patches, the joined
/// waveform, and why the loop stopped.
#[derive(Debug, Clone)]
pub struct Generated {
    pub latents: Vec<f32>,
    pub latent_steps: usize,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub stopped_by: StopReason,
}

/// Generation controls with the reference `generate` defaults
/// (`cfg_scale 2.0`, `sigma 0.25`, `temperature 0.0`, `ddpm_steps 10`,
/// `max_tokens` -> `max_decode_steps 200`).
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub cfg_scale: f32,
    pub sigma: f32,
    pub temperature: f32,
    pub flow_steps: usize,
    pub max_decode_steps: usize,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            cfg_scale: 2.0,
            sigma: 0.25,
            temperature: 0.0,
            flow_steps: 10,
            max_decode_steps: 200,
        }
    }
}

/// The raw checkpoint sub-configurations, kept as parsed JSON and
/// resolved into the concrete stacks at load time.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_type: String,
    /// True when the checkpoint is the `dense` directory variant (its
    /// config forces the Qwen2 backbone path regardless of the llm
    /// config's MoE keys).
    pub dense_variant: bool,
    pub llm_config: Value,
    pub audio_tokenizer_config: Value,
    pub ditar_config: Value,
    pub aggregator_config: Value,
}

impl ModelConfig {
    /// The reference `from_dict`: `llm_config` (or `text_config`, which
    /// the dense variant reads) plus the three audio sub-configs.
    pub fn from_json(value: &Value) -> Result<Self> {
        let model_type = value
            .get("model_type")
            .and_then(Value::as_str)
            .unwrap_or("ming_omni_tts")
            .to_owned();
        let llm_config = value
            .get("llm_config")
            .or_else(|| value.get("text_config"))
            .cloned()
            .ok_or_else(|| SpeechError::BadConfig {
                field: "llm_config".into(),
                why: "the LLM sub-config is required".into(),
            })?;
        let required = |key: &str| -> Result<Value> {
            value
                .get(key)
                .cloned()
                .ok_or_else(|| SpeechError::BadConfig {
                    field: key.to_owned(),
                    why: "this sub-config is required".into(),
                })
        };
        Ok(Self {
            dense_variant: model_type == "dense",
            model_type,
            llm_config,
            audio_tokenizer_config: required("audio_tokenizer_config")?,
            ditar_config: required("ditar_config")?,
            aggregator_config: required("aggregator_config")?,
        })
    }

    /// The reference `_is_moe_llm_config`: every MoE key present. The
    /// dense variant overrides this to false unconditionally.
    fn is_moe(&self) -> bool {
        if self.dense_variant {
            return false;
        }
        [
            "moe_intermediate_size",
            "num_experts",
            "num_shared_experts",
            "norm_topk_prob",
            "num_experts_per_tok",
            "first_k_dense_replace",
        ]
        .iter()
        .all(|key| self.llm_config.get(key).is_some_and(|v| !v.is_null()))
    }
}

/// The tokenized prompt markers the reference encodes from literal
/// strings (`_prepare_input_embed`). The caller tokenizes once with the
/// checkpoint tokenizer and passes the ids; `audio_token` and
/// `patch_token` come from `convert_tokens_to_ids`.
#[derive(Debug, Clone)]
pub struct MarkerIds {
    pub role_human: Vec<i32>,
    /// The `" Text input:\n"` header (skipped for the music-style
    /// prompts; see [`music_style_text`]).
    pub text_input_header: Vec<i32>,
    pub role_assistant: Vec<i32>,
    /// The `<|endoftext|>` id appended after an instruction.
    pub end_of_text: i32,
    /// `convert_tokens_to_ids("<audio>")`.
    pub audio_token: i32,
    /// `convert_tokens_to_ids("<audioPatch>")`.
    pub patch_token: i32,
}

/// The reference music-style detection on the raw text: when all five
/// fields are present the `" Text input:\n"` header is dropped.
pub fn music_style_text(text: &str) -> bool {
    ["Genre: ", "Mood: ", "Instrument: ", "Theme: ", "Duration: "]
        .iter()
        .all(|marker| text.contains(marker))
}

/// The pre-tokenized prompt segments ([`Model::prepare_input_embeds`]).
#[derive(Debug, Clone)]
pub struct PromptSegments<'a> {
    /// The tokenized prompt string (reference default
    /// [`DEFAULT_PROMPT`]).
    pub prompt: &'a [i32],
    /// The tokenized text to speak.
    pub text: &'a [i32],
    /// The tokenized instruction plus `<|endoftext|>` is appended by
    /// the model; pass the instruction ids only.
    pub instruction: Option<&'a [i32]>,
    /// The reference transcript ids for voice cloning.
    pub prompt_text: Option<&'a [i32]>,
    /// The voice-clone latent `[frames, latent_dim]` from
    /// [`Encoder::encode_latent`] on the padded reference audio.
    pub prompt_latent: Option<&'a [f32]>,
    /// Set from [`music_style_text`] on the raw text.
    pub skip_text_input_header: bool,
}

/// The loaded family model.
pub struct Model {
    pub config: ModelConfig,
    llm: Llm,
    encoder: Encoder,
    decoder: Decoder,
    linear_proj_audio: Aggregator,
    flowloss: FlowLoss,
    stop_head: Linear,
    /// Loaded for checkpoint parity; the reference defines it but never
    /// calls it on the TTS generate path.
    #[allow(dead_code)]
    spk_head: Option<Linear>,
    latent_dim: usize,
    patch_size: usize,
    history_patch_size: usize,
    sample_rate: u32,
}

enum Llm {
    Moe(Box<lm::BailingMoeModel>),
    Dense(Box<MingQwen2>),
}

impl Model {
    /// Opens a checkpoint directory. The LLM tensors sit one `model.`
    /// above the names the shared bailing MoE loader builds
    /// (`model.` + `model.layers.N...` = `model.model.layers.N...`),
    /// so the LLM weights carry that prefix; every other load is
    /// prefix-free by absolute name.
    pub fn open(dir: &std::path::Path) -> Result<Self> {
        let config_value = quant::read_json(&dir.join("config.json"))?;
        let llm_weights = Weights::open(dir)?.with_prefix("model.");
        let flat_weights = Weights::open(dir)?;
        Self::load(&llm_weights, &flat_weights, &config_value)
    }

    /// Loads from opened weights: `llm_weights` must carry the `model.`
    /// prefix (the shared bailing MoE loader builds `model.`-prefixed
    /// names itself, so `model.` + `model.layers.N...` reaches
    /// `model.model.layers.N...`); every other tensor loads by absolute
    /// name from `flat_weights`.
    pub fn load(llm_weights: &Weights, flat_weights: &Weights, config: &Value) -> Result<Self> {
        let config = ModelConfig::from_json(config)?;
        // LLM backbone.
        let llm = if config.is_moe() {
            let moe_config = lm::bailing_moe::from_json(&config.llm_config)?;
            Llm::Moe(Box::new(lm::BailingMoeModel::from_config(
                llm_weights,
                moe_config,
            )?))
        } else {
            let dense_config = lm::qwen2::from_json(&config.llm_config)?;
            let sliding = SlidingConfig::from_dict(&config.llm_config);
            Llm::Dense(Box::new(MingQwen2::load_absolute(
                flat_weights,
                "model.model.",
                dense_config,
                &sliding,
            )?))
        };
        // Audio tokenizer config.
        let audio = &config.audio_tokenizer_config;
        let sample_rate = audio
            .get("sample_rate")
            .and_then(Value::as_u64)
            .ok_or_else(|| SpeechError::BadConfig {
                field: "audio_tokenizer_config.sample_rate".into(),
                why: "must be a number".into(),
            })? as u32;
        let audio_patch_size = audio
            .get("patch_size")
            .and_then(Value::as_i64)
            .unwrap_or(-1);
        if audio_patch_size != -1 && audio_patch_size < 0 {
            return Err(SpeechError::BadConfig {
                field: "audio_tokenizer_config.patch_size".into(),
                why: format!("unsupported patch size {audio_patch_size}"),
            });
        }
        let audio_patch_size = audio_patch_size.max(0) as usize;
        let enc = audio
            .get("enc_kwargs")
            .ok_or_else(|| SpeechError::BadConfig {
                field: "enc_kwargs".into(),
                why: "the encoder kwargs are required".into(),
            })?;
        let dec = audio
            .get("dec_kwargs")
            .ok_or_else(|| SpeechError::BadConfig {
                field: "dec_kwargs".into(),
                why: "the decoder kwargs are required".into(),
            })?;
        let input_dim = usize_field(enc, "input_dim")?;
        let hop_size = enc
            .get("hop_size")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(320);
        let latent_dim = usize_field(enc, "latent_dim")?;
        let output_dim = usize_field(dec, "output_dim")?;
        let dec_latent_dim = usize_field(dec, "latent_dim")?;
        if dec_latent_dim != latent_dim {
            return Err(SpeechError::BadConfig {
                field: "latent_dim".into(),
                why: format!(
                    "encoder latent {latent_dim} does not match decoder latent {dec_latent_dim}"
                ),
            });
        }
        let enc_backbone = backbone_config(enc.get("backbone"))?;
        let dec_backbone = backbone_config(dec.get("backbone"))?;
        let encoder = Encoder::load_absolute(
            flat_weights,
            "audio.encoder.",
            "audio.encoder.encoder.",
            "audio.encoder.aggregator.",
            "audio.encoder.cls_embed",
            enc_backbone.0,
            &enc_backbone.1,
            input_dim,
            hop_size,
            latent_dim,
            audio_patch_size,
        )?;
        let decoder = Decoder::load_absolute(
            flat_weights,
            "audio.decoder.",
            "audio.decoder.decoder.",
            dec_backbone.0,
            &dec_backbone.1,
            latent_dim,
            output_dim,
            audio_patch_size,
        )?;
        // DiTar / aggregator configs.
        let ditar = &config.ditar_config;
        let patch_size = usize_field(ditar, "patch_size")?;
        let history_patch_size = ditar
            .get("history_patch_size")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(patch_size);
        check_patch_geometry(latent_dim, patch_size, history_patch_size)?;
        let ditar_hidden = usize_field(ditar, "hidden_size")?;
        let ditar_depth = usize_field(ditar, "depth")?;
        let ditar_heads = usize_field(ditar, "num_heads")?;
        let mlp_ratio = ditar
            .get("mlp_ratio")
            .and_then(Value::as_f64)
            .unwrap_or(4.0) as f32;
        let flowloss = dit::load_flow_loss(
            flat_weights,
            "flowloss.",
            ditar_depth,
            ditar_heads,
            mlp_ratio,
            latent_dim,
            ditar_hidden,
            ditar_hidden,
        )?;
        let agg = &config.aggregator_config;
        let agg_hidden = usize_field(agg, "hidden_size")?;
        let agg_depth = usize_field(agg, "depth")?;
        let agg_heads = usize_field(agg, "num_heads")?;
        let agg_ratio = agg.get("mlp_ratio").and_then(Value::as_f64).unwrap_or(4.0) as f32;
        let llm_hidden = match &llm {
            Llm::Moe(model) => model.config.hidden_size,
            Llm::Dense(model) => model.config.hidden_size,
        };
        let linear_proj_audio = Aggregator::load_absolute(
            flat_weights,
            "linear_proj_audio.",
            agg_depth,
            agg_heads,
            agg_ratio,
            latent_dim,
            llm_hidden,
            agg_hidden,
        )?;
        let stop_head = flat_weights.linear_absolute("stop_head", llm_hidden, 2, Bias::Required)?;
        let spk_head = if flat_weights.contains_absolute("spk_head") {
            Some(flat_weights.linear_absolute("spk_head", 192, llm_hidden, Bias::Required)?)
        } else {
            None
        };
        Ok(Self {
            config,
            llm,
            encoder,
            decoder,
            linear_proj_audio,
            flowloss,
            stop_head,
            spk_head,
            latent_dim,
            patch_size,
            history_patch_size,
            sample_rate,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    /// Reference `_pad_prompt_waveform`: pad the reference audio to a
    /// multiple of `(1 / 12.5) * patch_size * sample_rate` samples (the
    /// 12.5 Hz tokenizer frame alignment).
    pub fn pad_prompt_waveform(&self, waveform: &[f32]) -> Vec<f32> {
        let pad_align = ((1.0 / 12.5) * self.patch_size as f32 * self.sample_rate as f32) as usize;
        if pad_align == 0 || waveform.len() % pad_align == 0 {
            return waveform.to_vec();
        }
        let new_len = waveform.len().div_ceil(pad_align) * pad_align;
        let mut padded = waveform.to_vec();
        padded.resize(new_len, 0.0);
        padded
    }

    /// Encodes the reference audio into the voice-clone latent. The
    /// noise draws (`mu + std * normal`) come from the closure in
    /// row-major consumption order, so a recorded replay reproduces the
    /// reference draw.
    pub fn encode_prompt_latent(
        &mut self,
        ref_audio: &[f32],
        noise: &mut dyn FnMut() -> f32,
    ) -> Result<(Vec<f32>, usize)> {
        let padded = self.pad_prompt_waveform(ref_audio);
        let (latent, rows) = self.encoder.encode_latent(&padded, noise)?;
        if rows % self.patch_size != 0 {
            return Err(SpeechError::Input {
                why: format!(
                    "voice-clone latent rows {rows} are not a multiple of patch size {}",
                    self.patch_size
                ),
            });
        }
        Ok((latent, rows))
    }

    /// Reference `_prepare_input_embed`: assemble the id sequence
    ///
    /// `[HUMAN] [prompt] [header] [prompt text] [text] [ASSISTANT]
    /// [instruction EOT] [audio] [patch ids]`
    ///
    /// embed it through the LLM's word embeddings, and splice the
    /// projected voice-clone latent after the first `<audio>` position.
    /// Returns the input embeddings `[sequence, hidden]`.
    pub fn prepare_input_embeds(
        &self,
        markers: &MarkerIds,
        segments: &PromptSegments<'_>,
    ) -> Result<(Vec<f32>, usize)> {
        let hidden = match &self.llm {
            Llm::Moe(model) => model.config.hidden_size,
            Llm::Dense(model) => model.config.hidden_size,
        };
        let word_embeddings: &[f32] = match &self.llm {
            Llm::Moe(model) => model.word_embeddings(),
            Llm::Dense(model) => &model.embed_tokens,
        };
        let vocab_width = match &self.llm {
            Llm::Moe(model) => model.config.vocab_size,
            Llm::Dense(model) => model.config.vocab_size,
        };
        let mut ids: Vec<i32> = Vec::new();
        ids.extend_from_slice(&markers.role_human);
        ids.extend_from_slice(segments.prompt);
        if !segments.skip_text_input_header {
            ids.extend_from_slice(&markers.text_input_header);
        }
        if let Some(prompt_text) = segments.prompt_text {
            ids.extend_from_slice(prompt_text);
        }
        ids.extend_from_slice(segments.text);
        ids.extend_from_slice(&markers.role_assistant);
        if let Some(instruction) = segments.instruction {
            ids.extend_from_slice(instruction);
            ids.push(markers.end_of_text);
        }
        ids.push(markers.audio_token);
        // Project the voice-clone latent through the aggregator patch
        // ([reference] reshape(-1, patch, latent) -> aggregator).
        let prompt_latent_embeds;
        let num_patches;
        if let Some(latent) = segments.prompt_latent {
            if latent.len() % self.latent_dim != 0 {
                return Err(SpeechError::Input {
                    why: "prompt latent is not a multiple of the latent dim".into(),
                });
            }
            let rows = latent.len() / self.latent_dim;
            if rows % self.patch_size != 0 {
                return Err(SpeechError::Input {
                    why: format!(
                        "prompt latent rows {rows} are not a multiple of patch size {}",
                        self.patch_size
                    ),
                });
            }
            num_patches = rows / self.patch_size;
            prompt_latent_embeds =
                self.linear_proj_audio
                    .forward(latent, rows / self.patch_size, self.patch_size);
            for _ in 0..num_patches {
                ids.push(markers.patch_token);
            }
        } else {
            num_patches = 0;
            prompt_latent_embeds = Vec::new();
        }
        // Embed the ids (rows x hidden).
        let mut embeds = Vec::with_capacity(ids.len() * hidden);
        for &id in &ids {
            let row = usize::try_from(id).map_err(|_| SpeechError::Input {
                why: format!("negative token id {id}"),
            })?;
            if row * hidden >= word_embeddings.len() || row >= vocab_width {
                return Err(SpeechError::Input {
                    why: format!("token id {id} outside the word embedding table"),
                });
            }
            embeds.extend_from_slice(&word_embeddings[row * hidden..(row + 1) * hidden]);
        }
        // Splice the projected latent after the first <audio> position.
        if num_patches > 0 {
            let audio_position = ids
                .iter()
                .position(|&id| id == markers.audio_token)
                .ok_or_else(|| SpeechError::Input {
                    why: "audio token missing from the assembled prompt".into(),
                })?;
            let start = audio_position + 1;
            if start + num_patches > ids.len() {
                return Err(SpeechError::Input {
                    why: "not enough patch positions after the audio token".into(),
                });
            }
            let splice_at = start * hidden;
            let splice_len = num_patches * hidden;
            embeds.splice(
                splice_at..splice_at + splice_len,
                prompt_latent_embeds.iter().copied(),
            );
        }
        Ok((embeds, num_patches))
    }

    /// The reference `sample` / `generate` loop over the prepared input
    /// embeddings: flow-sample one latent patch per LLM pass, decode it
    /// through the cached audio decoder, and stop on the stop head
    /// (probability above 0.5 after step 3) or `max_decode_steps`.
    /// The initial solver noise comes from `noise` in reference draw
    /// order; pass [`SeededNoise`] for standalone seeded runs.
    pub fn generate(
        &mut self,
        input_embeds: &[f32],
        prompt_latent: Option<&[f32]>,
        params: &GenerateParams,
        noise: &mut dyn FlowNoise,
    ) -> Result<Generated> {
        if params.max_decode_steps == 0 {
            return Err(SpeechError::Input {
                why: "No audio chunks were generated".into(),
            });
        }
        // The loop reads the last hidden row with `len - hidden`, which
        // underflows when there are no input rows.
        if input_embeds.is_empty() {
            return Err(SpeechError::Input {
                why: "input embeddings are empty".into(),
            });
        }
        let hidden = match &self.llm {
            Llm::Moe(model) => model.config.hidden_size,
            Llm::Dense(model) => model.config.hidden_size,
        };
        let num_layers = match &self.llm {
            Llm::Moe(model) => model.config.num_hidden_layers,
            Llm::Dense(model) => model.config.num_hidden_layers,
        };
        let mut caches: LayerCaches = lm::make_prompt_cache(num_layers);
        let mut decoder_caches: LayerCaches = lm::make_prompt_cache(
            match &self.config.audio_tokenizer_config.get("dec_kwargs") {
                Some(dec) => dec
                    .get("backbone")
                    .and_then(|b| b.get("num_hidden_layers"))
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize,
                None => 0,
            },
        );
        let mut upsample_state = UpsampleState::default();
        let mut istft_state = IstftState::default();

        // latent_history = zeros(history_patch_size, latent_dim) with the
        // voice-clone latent flush right.
        let mut latent_history = vec![0.0f32; self.history_patch_size * self.latent_dim];
        if let Some(latent) = prompt_latent {
            // A misaligned latent would make the copies below length-mismatch.
            if latent.len() % self.latent_dim != 0 {
                return Err(SpeechError::Input {
                    why: "prompt latent is not a multiple of the latent dim".into(),
                });
            }
            let rows = latent.len() / self.latent_dim;
            let start = self.history_patch_size as i64 - rows as i64;
            if start < 0 {
                let keep = (-start) as usize * self.latent_dim;
                latent_history.copy_from_slice(&latent[keep..]);
            } else {
                let offset = start as usize * self.latent_dim;
                latent_history[offset..].copy_from_slice(latent);
            }
        }

        let mut embeds = input_embeds.to_vec();
        let mut chunks: Vec<Vec<f32>> = Vec::new();
        let mut latents: Vec<f32> = Vec::new();
        let mut stopped_by = StopReason::MaxDecodeSteps;
        for step in 0..params.max_decode_steps {
            let hidden_states = match &mut self.llm {
                Llm::Moe(model) => model.forward_hidden(
                    crate::lm::engine::Input::Embeddings(&embeds),
                    Some(&mut caches),
                )?,
                Llm::Dense(model) => model.forward_hidden(&embeds, Some(&mut caches))?,
            };
            // z_diff: the last position's post-norm hidden row.
            let z_diff = &hidden_states[hidden_states.len() - hidden..];
            let flow = FlowParams {
                cfg: params.cfg_scale,
                patch: self.patch_size,
                sigma: params.sigma,
                temperature: params.temperature,
                steps: params.flow_steps,
                use_epss: true,
                sway_sampling_coef: -1.0,
            };
            let sampled = self.flowloss.sample(z_diff, &latent_history, &flow, noise);
            latents.extend_from_slice(&sampled);

            // stop_prob = softmax(stop_head(z_diff))[1]; is_last needs
            // step > 3 as well.
            let stop_logits = self.stop_head.forward(z_diff, 1);
            let mut scores = stop_logits;
            ops::softmax_row(&mut scores);
            let is_last = scores[1] > 0.5 && step > 3;

            let patch_rows = self.patch_size;
            let chunk = self.decoder.low_level_reconstruct(
                &sampled,
                patch_rows,
                &mut decoder_caches,
                &mut upsample_state,
                &mut istft_state,
                is_last,
            )?;
            chunks.push(chunk);
            if is_last {
                stopped_by = StopReason::StopHead;
                break;
            }
            // Project the patch into the LLM stream and slide the
            // latent history window.
            embeds = self.linear_proj_audio.forward(&sampled, 1, self.patch_size);
            let keep_from = self.patch_size * self.latent_dim;
            latent_history.drain(..keep_from);
            latent_history.extend_from_slice(&sampled);
        }
        let samples = chunks.concat();
        let latent_steps = latents.len() / (self.patch_size * self.latent_dim);
        Ok(Generated {
            latents,
            latent_steps,
            samples,
            sample_rate: self.sample_rate,
            stopped_by,
        })
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn flowloss_z_probe(
        &self,
        x: &[f32],
        t: f32,
        c: &[f32],
        cfg: f32,
        history: &[f32],
        batch: usize,
        history_len: usize,
        patch: usize,
    ) -> Vec<f32> {
        self.flowloss
            .model_forward_with_cfg(x, t, c, cfg, history, batch, history_len, patch)
    }

    /// [`Self::generate`] with a seeded crate-local noise source.
    pub fn generate_seeded(
        &mut self,
        input_embeds: &[f32],
        prompt_latent: Option<&[f32]>,
        params: &GenerateParams,
        seed: u64,
    ) -> Result<Generated> {
        let mut noise = SeededNoise::new(seed);
        self.generate(input_embeds, prompt_latent, params, &mut noise)
    }
}

/// The patch geometry the generate loop relies on. Zero sizes divide by
/// zero in the latent-step count and in the patch row math, and a history
/// shorter than one patch makes `latent_history.drain(..patch * latent_dim)`
/// run past its end.
fn check_patch_geometry(
    latent_dim: usize,
    patch_size: usize,
    history_patch_size: usize,
) -> Result<()> {
    if latent_dim == 0 {
        return Err(SpeechError::BadConfig {
            field: "latent_dim".into(),
            why: "latent_dim must be > 0".into(),
        });
    }
    if patch_size == 0 {
        return Err(SpeechError::BadConfig {
            field: "ditar_config.patch_size".into(),
            why: "patch_size must be > 0".into(),
        });
    }
    if history_patch_size < patch_size {
        return Err(SpeechError::BadConfig {
            field: "ditar_config.history_patch_size".into(),
            why: format!(
                "history_patch_size {history_patch_size} must be >= patch_size {patch_size}"
            ),
        });
    }
    Ok(())
}

fn usize_field(value: &Value, key: &str) -> Result<usize> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .ok_or_else(|| SpeechError::BadConfig {
            field: key.to_owned(),
            why: "must be a non-negative number".into(),
        })
}

/// A resolved audio backbone: the shared dense config plus its Ming
/// sliding-window extras.
fn backbone_config(value: Option<&Value>) -> Result<(lm::DenseConfig, SlidingConfig)> {
    let value = value.ok_or_else(|| SpeechError::BadConfig {
        field: "backbone".into(),
        why: "the audio backbone config is required".into(),
    })?;
    // The reference builds these through Qwen2ModelArgs.from_dict; the
    // model_type check it performs is carried by the required field.
    if value.get("model_type").and_then(Value::as_str).is_none() {
        return Err(SpeechError::BadConfig {
            field: "backbone.model_type".into(),
            why: "the audio backbone must name its model_type".into(),
        });
    }
    Ok((
        lm::qwen2::from_json(value)?,
        SlidingConfig::from_dict(value),
    ))
}

#[cfg(test)]
mod tests;
