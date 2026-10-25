//! Higgs Audio v2 TTS family (Boson AI).
//!
//! Port of `mlx_audio/tts/models/higgs_audio/` (higgs_audio.py,
//! generation.py, model.py, serve.py) at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/higgs_audio):
//! a llama-3.2-style backbone whose every layer is a dual-FFN block (text
//! and audio paths share attention; the input/post norms and MLPs are
//! per path, routed by the `audio_out_mask`), a separate audio codebook
//! embedding table, and a two-head projector (text logits plus
//! `[K, codebook_size + 2]` audio logits). Audio frames are emitted in a
//! delay pattern (codebook i lags by i frames), the first frame is the
//! synthetic all-BOS AUDIO_INIT, any codebook EOS starts a K-frame
//! ramp-out, and the aligned codes decode through the Higgs audio codec
//! ([`crate::codec::higgs_audio`], ported separately). See
//! [README.md](README.md) for the verification evidence and open items.
//!
//! Scope of this port, matching the reference single-prompt path: batch
//! size 1, non-streaming decode (streaming overlap-add is out of scope),
//! and prompt construction from pre-tokenized ids. Text tokenization is a
//! caller concern (the reference loads the checkpoint's Hugging Face
//! tokenizer); the exact prompt strings are exposed as functions so the
//! caller tokenizes them verbatim.

use serde_json::Value;

use crate::codec::higgs_audio::HiggsAudioTokenizer;
use crate::lm;
use crate::lm::cache::KvCache;
use crate::lm::engine::{Attention, Input, Mlp};
use crate::music::minimax_music3::rng;
use crate::nn::{Linear, RmsNorm};
use crate::{quant, Result, SpeechError};

/// Reference codec repository the framework `post_load_hook` downloads.
pub const DEFAULT_CODEC_REPO: &str = "mlx-community/higgs-audio-v2-tokenizer";
/// Reference sample rate (the deployed codec emits 24 kHz mono).
pub const DEFAULT_SAMPLE_RATE: u32 = 24000;

/// The llama-3.2 backbone settings (`HiggsTextConfig`).
#[derive(Debug, Clone)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    /// Parsed for config parity; the reference model never builds a tied
    /// head (the text logits always come from
    /// `audio_decoder_proj.text_lm_head`), so this field is unused.
    pub tie_word_embeddings: bool,
    pub rope_scaling: Option<Value>,
}

impl Default for TextConfig {
    fn default() -> Self {
        TextConfig {
            hidden_size: 3072,
            num_hidden_layers: 28,
            num_attention_heads: 24,
            num_key_value_heads: 8,
            intermediate_size: 8192,
            vocab_size: 128256,
            rope_theta: 500000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: true,
            // The dataclass default llama3 dict; `from_dict` instead passes
            // the raw `rope_scaling` key through, so a config.json without
            // the key runs the unscaled default rope.
            rope_scaling: Some(serde_json::json!({
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            })),
        }
    }
}

impl TextConfig {
    /// `HiggsTextConfig` fields from the `text_config` dict, keyed exactly
    /// like the reference `from_dict` (missing keys take the dataclass
    /// defaults, except `rope_scaling` which passes `None` through).
    pub fn from_json(value: &Value) -> Self {
        let defaults = TextConfig::default();
        let u = |key: &str, fallback: usize| -> usize {
            value
                .get(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(fallback)
        };
        TextConfig {
            hidden_size: u("hidden_size", defaults.hidden_size),
            num_hidden_layers: u("num_hidden_layers", defaults.num_hidden_layers),
            num_attention_heads: u("num_attention_heads", defaults.num_attention_heads),
            num_key_value_heads: u("num_key_value_heads", defaults.num_key_value_heads),
            intermediate_size: u("intermediate_size", defaults.intermediate_size),
            vocab_size: u("vocab_size", defaults.vocab_size),
            rope_theta: value
                .get("rope_theta")
                .and_then(Value::as_f64)
                .unwrap_or(defaults.rope_theta as f64) as f32,
            rms_norm_eps: value
                .get("rms_norm_eps")
                .and_then(Value::as_f64)
                .unwrap_or(defaults.rms_norm_eps as f64) as f32,
            tie_word_embeddings: value
                .get("tie_word_embeddings")
                .and_then(Value::as_bool)
                .unwrap_or(defaults.tie_word_embeddings),
            rope_scaling: value.get("rope_scaling").cloned().filter(|v| !v.is_null()),
        }
    }
}

/// The top-level `HiggsAudioConfig`.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub text_config: TextConfig,
    pub audio_num_codebooks: usize,
    pub audio_codebook_size: usize,
    /// Per-codebook stream BOS (`audio_codebook_size` on the defaults).
    pub audio_stream_bos_id: i32,
    /// Per-codebook stream EOS (`audio_codebook_size + 1` on the defaults).
    pub audio_stream_eos_id: i32,
    /// Parsed for config parity; the reference builds one dual-FFN block
    /// per backbone layer regardless of this list.
    pub audio_dual_ffn_layers: Vec<usize>,
    /// Parsed and refused when true: the reference ignores the flag and
    /// builds no audio attention layers, so a config enabling it describes
    /// an architecture the reference itself cannot load.
    pub use_audio_out_self_attention: bool,
    /// Parsed and refused when nonzero, for the same reason as
    /// [`ModelConfig::use_audio_out_self_attention`].
    pub audio_decoder_proj_num_layers: usize,
    pub use_delay_pattern: bool,
    pub audio_in_token_idx: Option<i32>,
    pub audio_out_token_idx: Option<i32>,
    pub audio_out_bos_token_id: Option<i32>,
    pub audio_eos_token_id: Option<i32>,
    pub pad_token_id: Option<i32>,
    /// The framework `Model` pins this to 24000; parsed as a family field
    /// so configs can carry it, with the reference default as fallback.
    pub sample_rate: u32,
}

impl ModelConfig {
    /// `HiggsAudioConfig.from_dict` over a `config.json` dict.
    pub fn from_json(value: &Value) -> Result<Self> {
        let defaults_text = TextConfig::default();
        let text_config = match value.get("text_config") {
            Some(text) => TextConfig::from_json(text),
            None => defaults_text,
        };
        // head_dim is `hidden_size / num_attention_heads` (llama default), so
        // zero heads divides by zero and a non-divisor silently truncates.
        if text_config.num_attention_heads == 0
            || text_config.hidden_size % text_config.num_attention_heads != 0
        {
            return Err(SpeechError::BadConfig {
                field: "text_config.num_attention_heads".into(),
                why: format!(
                    "must be non-zero and divide hidden_size {} (got {})",
                    text_config.hidden_size, text_config.num_attention_heads
                ),
            });
        }
        let u = |key: &str, fallback: u64| -> usize {
            value.get(key).and_then(Value::as_u64).unwrap_or(fallback) as usize
        };
        let config = ModelConfig {
            text_config,
            audio_num_codebooks: u("audio_num_codebooks", 8),
            audio_codebook_size: u("audio_codebook_size", 1024),
            audio_stream_bos_id: value
                .get("audio_stream_bos_id")
                .and_then(Value::as_i64)
                .unwrap_or(1024) as i32,
            audio_stream_eos_id: value
                .get("audio_stream_eos_id")
                .and_then(Value::as_i64)
                .unwrap_or(1025) as i32,
            audio_dual_ffn_layers: match value.get("audio_dual_ffn_layers") {
                Some(Value::Array(entries)) => entries
                    .iter()
                    .filter_map(|entry| entry.as_u64().map(|n| n as usize))
                    .collect(),
                _ => (0..value
                    .get("text_config")
                    .and_then(|t| t.get("num_hidden_layers"))
                    .and_then(Value::as_u64)
                    .unwrap_or(28) as usize)
                    .collect(),
            },
            use_audio_out_self_attention: value
                .get("use_audio_out_self_attention")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            audio_decoder_proj_num_layers: u("audio_decoder_proj_num_layers", 0),
            use_delay_pattern: value
                .get("use_delay_pattern")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            audio_in_token_idx: value
                .get("audio_in_token_idx")
                .and_then(Value::as_i64)
                .map(|n| n as i32),
            audio_out_token_idx: value
                .get("audio_out_token_idx")
                .and_then(Value::as_i64)
                .map(|n| n as i32),
            audio_out_bos_token_id: value
                .get("audio_out_bos_token_id")
                .and_then(Value::as_i64)
                .map(|n| n as i32),
            audio_eos_token_id: value
                .get("audio_eos_token_id")
                .and_then(Value::as_i64)
                .map(|n| n as i32),
            pad_token_id: value
                .get("pad_token_id")
                .and_then(Value::as_i64)
                .map(|n| n as i32),
            sample_rate: value
                .get("sample_rate")
                .and_then(Value::as_u64)
                .unwrap_or(DEFAULT_SAMPLE_RATE as u64) as u32,
        };
        if config.use_audio_out_self_attention {
            return Err(SpeechError::Unsupported {
                why: "use_audio_out_self_attention=true needs the separate audio attention \
                      layers the reference never builds; the pinned v2 3B checkpoint runs \
                      with the flag false"
                    .into(),
            });
        }
        if config.audio_decoder_proj_num_layers != 0 {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "audio_decoder_proj_num_layers={} needs the extra projector transformer \
                     the reference never builds; the pinned v2 3B checkpoint runs with 0",
                    config.audio_decoder_proj_num_layers
                ),
            });
        }
        if config.audio_num_codebooks == 0 || config.audio_codebook_size == 0 {
            return Err(SpeechError::BadConfig {
                field: "audio_num_codebooks".into(),
                why: "audio_num_codebooks and audio_codebook_size must be positive".into(),
            });
        }
        Ok(config)
    }

    /// Per-codebook stride of the shared audio embedding table and audio
    /// head (`codebook_size + 2`: values 0..C-1 plus stream BOS and EOS).
    pub fn codebook_stride(&self) -> usize {
        self.audio_codebook_size + 2
    }
}

/// Sampling controls for one generation (`HiggsAudioModel.generate`
/// defaults; the framework `Model.generate` default frame cap of 1200).
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub max_new_frames: usize,
    /// 0.0 (or a `sample_audio` temperature of 0) selects the greedy path.
    pub temperature: f32,
    /// `None` or values outside (0, 1) disable the nucleus filter.
    pub top_p: Option<f32>,
    /// `None` or `<= 0` disables the top-k filter.
    pub top_k: Option<i32>,
    /// `None` or 0 disables repetition-avoidance sampling.
    pub ras_win_len: Option<usize>,
    pub ras_max_repeat: usize,
    /// Frames sampled greedily before the temperature path takes over.
    pub sampling_warmup_frames: usize,
    /// Linear fade over the decoded waveform edges, the framework
    /// `Model.generate` defaults.
    pub fade_in_ms: f32,
    pub fade_out_ms: f32,
}

impl Default for GenerateParams {
    fn default() -> Self {
        GenerateParams {
            max_new_frames: 1200,
            temperature: 0.7,
            top_p: Some(0.95),
            top_k: None,
            ras_win_len: Some(7),
            ras_max_repeat: 2,
            sampling_warmup_frames: 0,
            fade_in_ms: 30.0,
            fade_out_ms: 15.0,
        }
    }
}

/// Why the generate loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The EOS ramp-out finished (`eos-ramp-complete-at-frame-N`).
    EosRamp,
    /// The frame budget ran out (`max-frames`).
    MaxFrames,
}

/// One generated utterance: the raw delayed frames, the aligned codec
/// codes, and the decoded mono waveform (faded).
#[derive(Debug, Clone)]
pub struct Generated {
    /// One entry per generated frame in delay-pattern form, each
    /// `audio_num_codebooks` codes.
    pub frames: Vec<Vec<i32>>,
    /// Codebook-major aligned view `[K][T]` after revert, boundary trim,
    /// and the codec-range clip; the codec decode input.
    pub codes: Vec<Vec<i32>>,
    pub samples: Vec<f32>,
    pub num_frames_raw: usize,
    pub num_frames_aligned: usize,
    pub stopped_by: StopReason,
}

/// Prompt embeddings plus the per-position audio routing mask
/// (`build_prompt` output, batch of one).
#[derive(Debug, Clone)]
pub struct PromptEmbeds {
    /// Row-major `[T, hidden_size]`.
    pub embeds: Vec<f32>,
    /// Length T; true at positions routed through the audio dual-FFN path.
    pub audio_mask: Vec<bool>,
}

/// Precomputed reference-audio prompt pieces (`encode_reference` output).
#[derive(Debug, Clone)]
pub struct ReferenceContext {
    /// `[prefix_len, hidden]`: the text prefix through `<|audio_out_bos|>`.
    pub prefix_emb: Vec<f32>,
    /// `[T_ref_delayed, hidden]`: the ref codes as summed codebook embeds.
    pub audio_emb: Vec<f32>,
    pub prefix_len: usize,
    pub t_ref_delayed: usize,
    /// Raw ref-code length, pre-delay-pattern.
    pub t_ref: usize,
    pub ref_text: String,
}

/// One dual-FFN decoder layer: shared attention, per-path input/post
/// norms and MLPs (`HiggsDualFFNDecoderLayer`).
struct DualFfnLayer {
    input_layernorm: RmsNorm,
    audio_input_layernorm: RmsNorm,
    attention: Attention,
    post_attention_layernorm: RmsNorm,
    audio_post_attention_layernorm: RmsNorm,
    mlp: Mlp,
    audio_mlp: Mlp,
}

/// The loaded family model (`HiggsAudioModel`).
pub struct Model {
    pub config: ModelConfig,
    embed_tokens: Vec<f32>,
    audio_codebook_embeddings: Vec<f32>,
    layers: Vec<DualFfnLayer>,
    norm: RmsNorm,
    text_lm_head: Linear,
    audio_lm_head: Linear,
}

impl Model {
    /// Opens a checkpoint directory (`config.json` + safetensors).
    pub fn open(dir: &std::path::Path) -> Result<Self> {
        let config_value = quant::read_json(&dir.join("config.json"))?;
        let weights = lm::Weights::open(dir)?;
        Self::load(&weights, &config_value)
    }

    /// Loads from opened weights and the raw config dict. The backbone
    /// tensors carry the reference module-tree names (`embed_tokens`,
    /// `layers.*`, `norm`); a bosonai-style save that nests them under
    /// `model.` is detected once and followed for the backbone tensors,
    /// and the family tensors (`audio_codebook_embeddings`,
    /// `audio_decoder_proj.*`) are probed for their own prefix.
    pub fn load(weights: &lm::Weights, config: &Value) -> Result<Self> {
        let config = ModelConfig::from_json(config)?;
        let backbone_prefix = if weights.contains("layers.0.self_attn.q_proj") {
            ""
        } else if weights.contains("model.layers.0.self_attn.q_proj") {
            "model."
        } else {
            return Err(SpeechError::Tensor {
                name: "layers.0.self_attn.q_proj.weight".into(),
                why: "tensor is missing (neither the module-tree nor the model. \
                      layout is present)"
                    .into(),
            });
        };
        let family_prefix = if weights.contains_absolute("audio_codebook_embeddings") {
            ""
        } else if weights.contains_absolute("model.audio_codebook_embeddings") {
            "model."
        } else {
            return Err(SpeechError::Tensor {
                name: "audio_codebook_embeddings.weight".into(),
                why: "tensor is missing".into(),
            });
        };
        let bp = |name: &str| format!("{backbone_prefix}{name}");
        let fp = |name: &str| format!("{family_prefix}{name}");
        let embed = |name: &str, rows: usize, cols: usize| -> Result<Vec<f32>> {
            weights.tensor_absolute(&format!("{name}.weight"), &[rows, cols])
        };
        let text = &config.text_config;
        let hidden = text.hidden_size;
        let kv_width = text.num_key_value_heads * head_dim(text);
        let q_width = text.num_attention_heads * head_dim(text);
        let rope = lm::rope::Rope::initialize(
            head_dim(text),
            text.rope_theta,
            false,
            text.rope_scaling.as_ref(),
            None,
        )?;
        // No pre-sizing from `num_hidden_layers`: it is an untrusted config
        // value, and the loop fails on the first missing tensor before any
        // large allocation matters.
        let mut layers = Vec::new();
        for index in 0..text.num_hidden_layers {
            let layer = bp(&format!("layers.{index}."));
            layers.push(DualFfnLayer {
                input_layernorm: RmsNorm::new(
                    weights
                        .tensor_absolute(&format!("{layer}input_layernorm.weight"), &[hidden])?,
                    text.rms_norm_eps,
                ),
                audio_input_layernorm: RmsNorm::new(
                    weights.tensor_absolute(
                        &format!("{layer}audio_input_layernorm.weight"),
                        &[hidden],
                    )?,
                    text.rms_norm_eps,
                ),
                attention: Attention {
                    q_proj: weights.linear_absolute(
                        &format!("{layer}self_attn.q_proj"),
                        hidden,
                        q_width,
                        lm::Bias::None,
                    )?,
                    k_proj: weights.linear_absolute(
                        &format!("{layer}self_attn.k_proj"),
                        hidden,
                        kv_width,
                        lm::Bias::None,
                    )?,
                    v_proj: weights.linear_absolute(
                        &format!("{layer}self_attn.v_proj"),
                        hidden,
                        kv_width,
                        lm::Bias::None,
                    )?,
                    o_proj: weights.linear_absolute(
                        &format!("{layer}self_attn.o_proj"),
                        q_width,
                        hidden,
                        lm::Bias::None,
                    )?,
                    q_norm: None,
                    k_norm: None,
                    n_heads: text.num_attention_heads,
                    n_kv_heads: text.num_key_value_heads,
                    head_dim: head_dim(text),
                    scale: (head_dim(text) as f64).powf(-0.5) as f32,
                    rope: Some(rope.clone()),
                    window: None,
                    bidirectional: false,
                    exec: None,
                },
                post_attention_layernorm: RmsNorm::new(
                    weights.tensor_absolute(
                        &format!("{layer}post_attention_layernorm.weight"),
                        &[hidden],
                    )?,
                    text.rms_norm_eps,
                ),
                audio_post_attention_layernorm: RmsNorm::new(
                    weights.tensor_absolute(
                        &format!("{layer}audio_post_attention_layernorm.weight"),
                        &[hidden],
                    )?,
                    text.rms_norm_eps,
                ),
                mlp: load_mlp(
                    weights,
                    &bp(&format!("layers.{index}.mlp")),
                    hidden,
                    text.intermediate_size,
                )?,
                audio_mlp: load_mlp(
                    weights,
                    &bp(&format!("layers.{index}.audio_mlp")),
                    hidden,
                    text.intermediate_size,
                )?,
            });
        }
        let codebooks = config.audio_num_codebooks;
        let stride = config.codebook_stride();
        Ok(Model {
            embed_tokens: embed(&bp("embed_tokens"), text.vocab_size, hidden)?,
            audio_codebook_embeddings: embed(
                &fp("audio_codebook_embeddings"),
                codebooks * stride,
                hidden,
            )?,
            layers,
            norm: RmsNorm::new(
                weights.tensor_absolute(&format!("{}norm.weight", bp("")), &[hidden])?,
                text.rms_norm_eps,
            ),
            text_lm_head: weights.linear_absolute(
                &fp("audio_decoder_proj.text_lm_head"),
                hidden,
                text.vocab_size,
                lm::Bias::None,
            )?,
            audio_lm_head: weights.linear_absolute(
                &fp("audio_decoder_proj.audio_lm_head"),
                hidden,
                codebooks * stride,
                lm::Bias::None,
            )?,
            config,
        })
    }

    pub fn audio_num_codebooks(&self) -> usize {
        self.config.audio_num_codebooks
    }

    /// The shared audio codebook embedding table
    /// `[num_codebooks * (codebook_size + 2), hidden_size]`.
    pub fn audio_codebook_embeddings(&self) -> &[f32] {
        &self.audio_codebook_embeddings
    }

    /// The attention forward over row-major embeddings with per-position
    /// dual-FFN routing, then the final norm. `audio_mask` selects the
    /// audio norms and MLPs; `None` synthesizes the reference all-false
    /// mask. Returns the post-norm hidden states `[rows, hidden]`.
    pub fn forward_hidden(
        &mut self,
        input: Input<'_>,
        audio_mask: Option<&[bool]>,
        caches: &mut [KvCache],
    ) -> Result<Vec<f32>> {
        let hidden = self.config.text_config.hidden_size;
        let rows = input.len(hidden)?;
        if rows == 0 {
            return Err(SpeechError::Input {
                why: "empty forward input".into(),
            });
        }
        if caches.len() != self.layers.len() {
            return Err(SpeechError::Input {
                why: format!(
                    "{} layer caches for {} layers",
                    caches.len(),
                    self.layers.len()
                ),
            });
        }
        let routing: Vec<bool> = match audio_mask {
            Some(mask) => {
                if mask.len() != rows {
                    return Err(SpeechError::Input {
                        why: format!("audio mask of {} for {rows} positions", mask.len()),
                    });
                }
                mask.to_vec()
            }
            None => vec![false; rows],
        };
        let mut h = match input {
            Input::Tokens(tokens) => crate::lm::engine::lookup_embeddings(
                &self.embed_tokens,
                self.config.text_config.vocab_size,
                hidden,
                tokens,
            )?,
            Input::Embeddings(values) => values.to_vec(),
        };
        let offset = caches.first().map(|c| c.offset).unwrap_or(0);
        for (index, layer) in self.layers.iter_mut().enumerate() {
            // Pre-attention norm: both paths, selected by the mask into
            // the attention input; the residual keeps the unnormalized h.
            let mut normed = h.clone();
            layer.input_layernorm.apply(&mut normed, rows);
            let mut audio_normed = h.clone();
            layer.audio_input_layernorm.apply(&mut audio_normed, rows);
            for (row, route) in routing.iter().enumerate() {
                if *route {
                    let base = row * hidden;
                    normed[base..base + hidden].copy_from_slice(&audio_normed[base..base + hidden]);
                }
            }
            let attn = layer.attention.forward(
                &normed,
                rows,
                offset,
                Some(&mut caches[index] as &mut dyn lm::cache::LayerKv),
            )?;
            for (row, out) in attn.chunks_exact(hidden).enumerate() {
                let base = row * hidden;
                for (col, value) in out.iter().enumerate() {
                    h[base + col] += value;
                }
            }
            // Post-attention: split norm + split MLP by mask.
            let mut post_text = h.clone();
            layer.post_attention_layernorm.apply(&mut post_text, rows);
            let mut post_audio = h.clone();
            layer
                .audio_post_attention_layernorm
                .apply(&mut post_audio, rows);
            let mlp_text = layer.mlp.forward(&post_text, rows);
            let mlp_audio = layer.audio_mlp.forward(&post_audio, rows);
            for (row, route) in routing.iter().enumerate() {
                let base = row * hidden;
                let chosen = if *route { &mlp_audio } else { &mlp_text };
                for (col, value) in chosen[base..base + hidden].iter().enumerate() {
                    h[base + col] += value;
                }
            }
        }
        self.norm.apply(&mut h, rows);
        Ok(h)
    }

    /// The full forward: hidden states through both heads. `audio_mask`
    /// present also computes the audio logits `[rows, K, C+2]` (the
    /// reference computes them for every position and lets the caller
    /// filter by mask); `None` skips the audio head.
    pub fn forward(
        &mut self,
        input: Input<'_>,
        audio_mask: Option<&[bool]>,
        caches: &mut [KvCache],
    ) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        let hidden = self.config.text_config.hidden_size;
        let projected = self.forward_hidden(input, audio_mask, caches)?;
        let rows = projected.len() / hidden;
        let text_logits = self.text_lm_head.forward(&projected, rows);
        let audio_logits = audio_mask.map(|_| self.audio_lm_head.forward(&projected, rows));
        Ok((text_logits, audio_logits))
    }

    /// `lookup_audio_embedding` over this model's table: `[K][T]` codebook
    /// ids to `[T, hidden]` summed rows (codebook k rides the table rows
    /// `k * stride + token`; per-codebook embeddings sum).
    pub fn lookup_audio_embedding(&self, codebook_ids: &[Vec<i32>]) -> Result<Vec<f32>> {
        lookup_audio_embedding(
            &self.audio_codebook_embeddings,
            codebook_ids,
            self.config.codebook_stride(),
            self.config.text_config.hidden_size,
        )
    }

    /// The single-time-step variant the generate loop uses: one `[K]`
    /// frame to one hidden row.
    fn lookup_frame_embedding(&self, frame: &[i32]) -> Result<Vec<f32>> {
        let hidden = self.config.text_config.hidden_size;
        let stride = self.config.codebook_stride();
        if frame.len() != self.config.audio_num_codebooks {
            return Err(SpeechError::Input {
                why: format!(
                    "frame of {} codes for {} codebooks",
                    frame.len(),
                    self.config.audio_num_codebooks
                ),
            });
        }
        let mut out = vec![0.0f32; hidden];
        for (codebook, &token) in frame.iter().enumerate() {
            let row = embed_row(
                &self.audio_codebook_embeddings,
                codebook * stride + index_of(token)?,
                hidden,
            )?;
            for (col, value) in out.iter_mut().enumerate() {
                *value += row[col];
            }
        }
        Ok(out)
    }

    /// Runs the raw frame loop (`_generate_raw_frames`) from a prepared
    /// prompt: prefill, the synthetic all-BOS first frame, the ramp-in,
    /// greedy-warmup then sampled steps, RAS, and the EOS ramp-out. The
    /// caches are fresh for every call, matching the reference.
    pub fn generate_frames(
        &mut self,
        prompt: &PromptEmbeds,
        params: &GenerateParams,
        seed: u64,
    ) -> Result<(Vec<Vec<i32>>, StopReason)> {
        let k = self.config.audio_num_codebooks;
        let bos = self.config.audio_stream_bos_id;
        let eos = self.config.audio_stream_eos_id;
        let width = self.config.codebook_stride();

        let mut caches = lm::make_prompt_cache(self.config.text_config.num_hidden_layers);
        self.forward_hidden(
            Input::Embeddings(&prompt.embeds),
            Some(&prompt.audio_mask),
            &mut caches,
        )?;

        // Frame 0 is the synthetic all-BOS AUDIO_INIT; the prefill logits
        // are discarded (the reference never samples from them).
        let mut frames: Vec<Vec<i32>> = Vec::new();
        let mut prev_frame = vec![bos; k];
        frames.push(prev_frame.clone());
        let mut num_delay = 0usize;
        let mut num_remaining_delays: Option<usize> = None;
        let mut stopped_by = StopReason::MaxFrames;

        let ras_enabled = params.ras_win_len.is_some_and(|len| len > 0);
        let ras_win = params.ras_win_len.unwrap_or(0);
        let mut ras_window: Vec<Vec<i32>> = if ras_enabled {
            vec![vec![bos]; k]
        } else {
            Vec::new()
        };

        let mut keys = rng::KeySequence::new(seed);
        for step in 0..params.max_new_frames {
            let embed = self.lookup_frame_embedding(&prev_frame)?;
            let (_, audio_logits) =
                self.forward(Input::Embeddings(&embed), Some(&[true]), &mut caches)?;
            let audio_logits = audio_logits.ok_or_else(|| SpeechError::Tensor {
                name: "audio_logits".into(),
                why: "the audio head did not run".into(),
            })?;
            // The step input is one position: `[K, C+2]` rows.
            let mut tokens: Vec<i32> = if step < params.sampling_warmup_frames {
                greedy_sample_audio(&audio_logits, k, width)
            } else {
                let key = keys.next();
                sample_audio(&audio_logits, k, width, params, key)?
            };

            // RAS: per codebook, resample greedily when the sampled token
            // already fills the recent window.
            if ras_enabled {
                let mut needs: Vec<bool> = Vec::with_capacity(k);
                let mut any = false;
                for (codebook, &token) in tokens.iter().enumerate() {
                    let window = &ras_window[codebook];
                    let start = window.len().saturating_sub(ras_win);
                    let count = window[start..].iter().filter(|&&v| v == token).count();
                    let need = count >= params.ras_max_repeat;
                    needs.push(need);
                    any |= need;
                }
                if any {
                    let greedy = greedy_sample_audio(&audio_logits, k, width);
                    for codebook in 0..k {
                        if needs[codebook] {
                            tokens[codebook] = greedy[codebook];
                        }
                    }
                }
            }

            if self.config.use_delay_pattern {
                // Ramp-in: tail codebooks forced to BOS until the delay
                // budget fills.
                if num_delay + 1 < k {
                    for token in tokens.iter_mut().skip(num_delay + 1) {
                        *token = bos;
                    }
                    num_delay += 1;
                }
                // Ramp-out in progress: the EOS prefix grows.
                if let Some(remaining) = num_remaining_delays {
                    let force_until = k - remaining;
                    for token in tokens.iter_mut().take(force_until) {
                        *token = eos;
                    }
                    num_remaining_delays = Some(remaining - 1);
                } else {
                    let eos_positions: Vec<usize> = tokens
                        .iter()
                        .enumerate()
                        .filter(|(_, &token)| token == eos)
                        .map(|(index, _)| index)
                        .collect();
                    if let Some(&last_eos) = eos_positions.last() {
                        for token in tokens.iter_mut().take(last_eos) {
                            *token = eos;
                        }
                        num_remaining_delays = Some(k - last_eos - 1);
                    }
                }
            }

            if ras_enabled {
                for (codebook, &token) in tokens.iter().enumerate() {
                    ras_window[codebook].push(token);
                    if ras_window[codebook].len() > ras_win + 4 {
                        let overflow = ras_window[codebook].len() - ras_win;
                        ras_window[codebook].drain(..overflow);
                    }
                }
            }

            frames.push(tokens.clone());
            prev_frame = tokens;

            if let Some(remaining) = num_remaining_delays {
                if self.config.use_delay_pattern && remaining == 0 {
                    stopped_by = StopReason::EosRamp;
                    break;
                }
            }
        }
        Ok((frames, stopped_by))
    }

    /// Full generation from a prepared prompt: the frame loop, the delay
    /// revert with boundary trim and clip, and the codec decode plus the
    /// edge fades. `seed` only feeds the sampled (temperature > 0) draws;
    /// greedy runs never draw and are fully deterministic.
    pub fn generate(
        &mut self,
        codec: &HiggsAudioTokenizer,
        prompt: &PromptEmbeds,
        params: &GenerateParams,
        seed: u64,
    ) -> Result<Generated> {
        let (frames, stopped_by) = self.generate_frames(prompt, params, seed)?;
        let raw = frames.len();
        let mut codes = revert_delay_pattern(&frames_to_sequence(
            &frames,
            self.config.audio_num_codebooks,
        ));
        if codes.first().is_some_and(|book| book.len() >= 2) {
            for book in codes.iter_mut() {
                book.remove(0);
                book.pop();
            }
        }
        let aligned = codes.first().map_or(0, |book: &Vec<i32>| book.len());
        let clipped = clip_codes(&codes, self.config.audio_codebook_size);
        let mut samples = if clipped.first().is_none_or(|book| book.is_empty()) {
            Vec::new()
        } else {
            codec.decode_tokens(&clipped)?
        };
        apply_fades(
            &mut samples,
            params.fade_in_ms,
            params.fade_out_ms,
            self.config.sample_rate,
        );
        Ok(Generated {
            frames,
            codes: clipped,
            samples,
            num_frames_raw: raw,
            num_frames_aligned: aligned,
            stopped_by,
        })
    }
}

fn head_dim(text: &TextConfig) -> usize {
    // The reference `_llama_args_from_text_config` passes no head_dim, so
    // the llama default `hidden_size // num_attention_heads` applies.
    text.hidden_size / text.num_attention_heads
}

fn load_mlp(weights: &lm::Weights, name: &str, hidden: usize, intermediate: usize) -> Result<Mlp> {
    Ok(Mlp {
        gate_proj: weights.linear_absolute(
            &format!("{name}.gate_proj"),
            hidden,
            intermediate,
            lm::Bias::None,
        )?,
        up_proj: weights.linear_absolute(
            &format!("{name}.up_proj"),
            hidden,
            intermediate,
            lm::Bias::None,
        )?,
        down_proj: weights.linear_absolute(
            &format!("{name}.down_proj"),
            intermediate,
            hidden,
            lm::Bias::None,
        )?,
        exec: None,
    })
}

fn codebooks_stride(codebook: usize, stride: usize) -> Result<usize> {
    codebook
        .checked_mul(stride)
        .ok_or_else(|| SpeechError::Input {
            why: "codebook index overflows the table stride".into(),
        })
}

fn index_of(token: i32) -> Result<usize> {
    usize::try_from(token).map_err(|_| SpeechError::Input {
        why: format!("negative codebook token {token}"),
    })
}

fn embed_row(table: &[f32], row: usize, width: usize) -> Result<&[f32]> {
    let start = row
        .checked_mul(width)
        .filter(|&s| s + width <= table.len())
        .ok_or_else(|| SpeechError::Input {
            why: format!(
                "embedding row {row} outside a table of {} rows",
                table.len() / width
            ),
        })?;
    Ok(&table[start..start + width])
}

/// `lookup_audio_embedding`: `[K][T]` codebook ids to `[T, hidden]` summed
/// rows over the shared table.
pub fn lookup_audio_embedding(
    table: &[f32],
    codebook_ids: &[Vec<i32>],
    stride: usize,
    hidden: usize,
) -> Result<Vec<f32>> {
    if codebook_ids.is_empty() {
        return Err(SpeechError::Input {
            why: "no codebooks".into(),
        });
    }
    let steps = codebook_ids[0].len();
    if steps == 0 {
        return Err(SpeechError::Input {
            why: "empty codebook timeline".into(),
        });
    }
    for book in codebook_ids {
        if book.len() != steps {
            return Err(SpeechError::Input {
                why: "ragged codebook timeline".into(),
            });
        }
    }
    let mut out = vec![0.0f32; steps * hidden];
    for (codebook, book) in codebook_ids.iter().enumerate() {
        let shift = codebooks_stride(codebook, stride)?;
        for (step, &token) in book.iter().enumerate() {
            // Each codebook owns `stride` rows. A larger code on a non-last
            // book would otherwise read the next codebook's row silently; the
            // table bound only catches the last book.
            let index = index_of(token)?;
            if index >= stride {
                return Err(SpeechError::Input {
                    why: format!("codebook {codebook} code {token} is outside its {stride} rows"),
                });
            }
            let row = embed_row(table, shift + index, hidden)?;
            for (col, value) in out[step * hidden..(step + 1) * hidden]
                .iter_mut()
                .enumerate()
            {
                *value += row[col];
            }
        }
    }
    Ok(out)
}

/// `revert_delay_pattern`: the delayed codebook-major `[K][L]` sequence
/// (each codebook offset by its index) into the aligned `[K][L - K + 1]`
/// view. Fewer raw frames than codebooks minus one produce an empty
/// timeline.
pub fn revert_delay_pattern(data: &[Vec<i32>]) -> Vec<Vec<i32>> {
    let codebooks = data.len();
    let mut out = vec![Vec::new(); codebooks];
    if codebooks == 0 {
        return out;
    }
    let total = data[0].len();
    for book in data {
        if book.len() != total {
            return vec![Vec::new(); codebooks];
        }
    }
    if total + 1 < codebooks {
        return out;
    }
    for (codebook, book) in data.iter().enumerate() {
        let end = total - codebooks + 1;
        out[codebook] = book[codebook..codebook + end].to_vec();
    }
    out
}

/// Transposes time-major frames `[N][K]` into the codebook-major
/// `[K][N]` layout the delay revert consumes.
fn frames_to_sequence(frames: &[Vec<i32>], codebooks: usize) -> Vec<Vec<i32>> {
    let mut out = vec![Vec::with_capacity(frames.len()); codebooks];
    for frame in frames {
        for (codebook, token) in frame.iter().take(codebooks).enumerate() {
            out[codebook].push(*token);
        }
    }
    out
}

/// `apply_delay_pattern`: aligned `[K][L]` codes into the delayed
/// `[K][L + K - 1]` view, BOS-filled ahead of each codebook's start.
pub fn apply_delay_pattern(codebook_ids: &[Vec<i32>], bos_id: i32) -> Result<Vec<Vec<i32>>> {
    if codebook_ids.is_empty() {
        return Err(SpeechError::Input {
            why: "no codebooks".into(),
        });
    }
    let steps = codebook_ids[0].len();
    for book in codebook_ids {
        if book.len() != steps {
            return Err(SpeechError::Input {
                why: "ragged codebook timeline".into(),
            });
        }
    }
    let codebooks = codebook_ids.len();
    let mut out = vec![vec![bos_id; steps + codebooks - 1]; codebooks];
    for (codebook, book) in codebook_ids.iter().enumerate() {
        for (step, &token) in book.iter().enumerate() {
            out[codebook][codebook + step] = token;
        }
    }
    Ok(out)
}

/// `build_delay_pattern_mask`: aligned `[K][L]` prompt codes into the
/// delayed `[K][L + K - 1]` mask with BOS below the diagonal and
/// `pad_token_id` past each codebook's content end.
pub fn build_delay_pattern_mask(
    input_ids: &[Vec<i32>],
    bos_token_id: i32,
    pad_token_id: i32,
) -> Result<Vec<Vec<i32>>> {
    apply_delay_pattern(input_ids, bos_token_id).map(|mut delayed| {
        let steps = input_ids[0].len();
        for (codebook, row) in delayed.iter_mut().enumerate() {
            for value in row.iter_mut().skip(steps + codebook) {
                *value = pad_token_id;
            }
        }
        delayed
    })
}

/// Clips aligned codes into the codec's valid range `[0, codebook_size)`,
/// the reference `mx.clip(aligned, 0, codebook_size - 1)` that folds the
/// boundary-trimmed stream BOS/EOS sentinels.
pub fn clip_codes(codes: &[Vec<i32>], codebook_size: usize) -> Vec<Vec<i32>> {
    let max = (codebook_size as i32) - 1;
    codes
        .iter()
        .map(|book| book.iter().map(|&token| token.clamp(0, max)).collect())
        .collect()
}

/// The per-codebook greedy draw (`greedy_sample_audio`): argmax over each
/// `[C+2]` logit row of the `[K, C+2]` step block.
pub fn greedy_sample_audio(audio_logits: &[f32], codebooks: usize, width: usize) -> Vec<i32> {
    (0..codebooks)
        .map(|codebook| {
            let row = &audio_logits[codebook * width..(codebook + 1) * width];
            crate::tts::sampling::argmax(row) as i32
        })
        .collect()
}

/// The per-codebook sampled draw (`sample_audio`): temperature scaling,
/// the kth-value top-k mask (ties kept), the descending-cumsum top-p mask,
/// and the Gumbel-max argmax. `key` feeds the crate-local threefry
/// uniforms the reference replaces with `mx.random.uniform`; a sampled
/// Rust run therefore diverges token-by-token from mlx at the same seed
/// while the filter math is pinned by fixtures. A temperature of 0 or
/// below returns the greedy draw without consuming the key.
pub fn sample_audio(
    audio_logits: &[f32],
    codebooks: usize,
    width: usize,
    params: &GenerateParams,
    key: rng::Key,
) -> Result<Vec<i32>> {
    if params.temperature <= 0.0 {
        return Ok(greedy_sample_audio(audio_logits, codebooks, width));
    }
    let scale = 1.0 / params.temperature;
    let mut rows: Vec<Vec<f32>> = (0..codebooks)
        .map(|codebook| {
            audio_logits[codebook * width..(codebook + 1) * width]
                .iter()
                .map(|&logit| logit * scale)
                .collect()
        })
        .collect();
    let top_k = params.top_k.filter(|&k| k > 0).map(|k| k as usize);
    if let Some(top_k) = top_k {
        for row in rows.iter_mut() {
            if top_k < row.len() {
                let kth = rng::kth_largest(row, top_k);
                for logit in row.iter_mut() {
                    if *logit < kth {
                        *logit = f32::NEG_INFINITY;
                    }
                }
            }
        }
    }
    let top_p = params.top_p.filter(|&p| p > 0.0 && p < 1.0);
    if let Some(top_p) = top_p {
        for row in rows.iter_mut() {
            apply_top_p_row(row, top_p);
        }
    }
    // Gumbel-max categorical draw, one uniform per logit in row-major
    // order (the reference draws the whole `[1, 1, K, C+2]` at once).
    let uniforms = rng::uniform01(key, codebooks * width);
    let mut out = Vec::with_capacity(codebooks);
    for (codebook, row) in rows.iter().enumerate() {
        let mut best = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for (index, &logit) in row.iter().enumerate() {
            let u = uniforms[codebook * width + index];
            let g = -((-(u + 1e-20).ln() + 1e-20).ln());
            let score = logit + g;
            if score > best_value {
                best = index;
                best_value = score;
            }
        }
        out.push(best as i32);
    }
    Ok(out)
}

/// The top-p half of `sample_audio`: descending softmax probabilities,
/// cumulative sum, keep where the PREVIOUS cumulative mass is below
/// `top_p` (index 0 always survives), mask the rest to negative infinity.
/// Public for the fixture tests that pin the filter on fixed logits.
pub fn apply_top_p_row(logits: &mut [f32], top_p: f32) {
    let len = logits.len();
    let mut order: Vec<usize> = (0..len).collect();
    order.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let total: f32 = probs.iter().sum();
    let mut cumulative = 0.0f32;
    let mut shifted = vec![0.0f32; len];
    for (position, &index) in order.iter().enumerate() {
        // `shifted[index]` is the cumulative mass BEFORE this entry.
        shifted[index] = cumulative;
        if position + 1 < len {
            cumulative += probs[index] / total;
        }
    }
    for (logit, &previous) in logits.iter_mut().zip(&shifted) {
        if previous < top_p {
            continue;
        }
        *logit = f32::NEG_INFINITY;
    }
}

/// Linear edge fades over the decoded waveform: the reference
/// `pcm[:n_in] *= np.linspace(0, 1, n_in)` and the mirrored descending
/// `pcm[-n_out:] *= np.linspace(1, 0, n_out)`. A fade longer than the
/// waveform does not touch it.
pub fn apply_fades(samples: &mut [f32], fade_in_ms: f32, fade_out_ms: f32, sample_rate: u32) {
    let n_in = (fade_in_ms * sample_rate as f32 / 1000.0) as usize;
    let n_out = (fade_out_ms * sample_rate as f32 / 1000.0) as usize;
    if n_in > 0 && samples.len() > n_in {
        for (index, sample) in samples[..n_in].iter_mut().enumerate() {
            *sample *= linspace_ramp(index, n_in);
        }
    }
    if n_out > 0 && samples.len() > n_out {
        let start = samples.len() - n_out;
        for (index, sample) in samples[start..].iter_mut().enumerate() {
            *sample *= linspace_ramp(n_out - 1 - index, n_out);
        }
    }
}

/// `np.linspace` value at `index` of an `n`-point 0-to-1 ramp, computed
/// in float64 then cast, matching the reference fade ramps sample for
/// sample.
fn linspace_ramp(index: usize, n: usize) -> f32 {
    if n <= 1 {
        return 0.0;
    }
    let step = 1.0f64 / (n - 1) as f64;
    (index as f64 * step) as f32
}

/// `smart_voice` prompt string (`_build_prompt_smart_voice`): the target
/// text alone; the assistant turn starts at `<|audio_out_bos|>`. The
/// caller tokenizes with `add_special_tokens=False`.
pub fn smart_voice_prompt_string(target_text: &str) -> String {
    format!(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n\
         {target_text}<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n\n\
         <|audio_out_bos|>"
    )
}

/// Voice-clone reference prefix string (`encode_reference`): the
/// reference transcript through `<|audio_out_bos|>`.
pub fn voice_clone_prefix_string(ref_text: &str) -> String {
    format!(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n\
         {ref_text}<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n\n\
         <|audio_out_bos|>"
    )
}

/// Voice-clone middle string (`_build_prompt_voice_clone`): the audio
/// turn closes, then the target turn opens.
pub fn voice_clone_middle_string(target_text: &str) -> String {
    format!(
        "<|audio_eos|><|eot_id|>\
         <|start_header_id|>user<|end_header_id|>\n\n\
         {target_text}<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n\n\
         <|audio_out_bos|>"
    )
}

impl Model {
    /// `build_prompt` for the smart-voice path: text ids only, every
    /// position on the text dual-FFN path.
    pub fn build_prompt_smart_voice(&self, prompt_ids: &[i32]) -> Result<PromptEmbeds> {
        let hidden = self.config.text_config.hidden_size;
        let embeds = crate::lm::engine::lookup_embeddings(
            &self.embed_tokens,
            self.config.text_config.vocab_size,
            hidden,
            prompt_ids,
        )?;
        Ok(PromptEmbeds {
            embeds,
            audio_mask: vec![false; prompt_ids.len()],
        })
    }

    /// `encode_reference` for a pre-encoded reference: the BOS/EOS-wrapped
    /// codes go through the delay mask and the summed codebook embedding,
    /// and the transcript prefix through the text embedding table. The
    /// codec `encode` step itself needs the wav2vec2 semantic tower (out
    /// of the portable codec contract), so codes arrive precomputed.
    pub fn encode_reference(
        &self,
        ref_text_ids: &[i32],
        ref_codes: &[Vec<i32>],
        ref_text: &str,
    ) -> Result<ReferenceContext> {
        let k = self.config.audio_num_codebooks;
        if ref_codes.len() != k {
            return Err(SpeechError::Input {
                why: format!("{} reference codebooks for {k} codebooks", ref_codes.len()),
            });
        }
        let hidden = self.config.text_config.hidden_size;
        let prefix_emb = crate::lm::engine::lookup_embeddings(
            &self.embed_tokens,
            self.config.text_config.vocab_size,
            hidden,
            ref_text_ids,
        )?;
        let mut wrapped: Vec<Vec<i32>> = Vec::with_capacity(k);
        let t_ref = ref_codes.first().map_or(0, |book| book.len());
        for book in ref_codes {
            let mut column = vec![self.config.audio_stream_bos_id];
            column.extend_from_slice(book);
            column.push(self.config.audio_stream_eos_id);
            wrapped.push(column);
        }
        let delayed = build_delay_pattern_mask(
            &wrapped,
            self.config.audio_stream_bos_id,
            self.config.audio_stream_eos_id,
        )?;
        let t_ref_delayed = delayed.first().map_or(0, |book| book.len());
        let audio_emb = self.lookup_audio_embedding(&delayed)?;
        Ok(ReferenceContext {
            prefix_len: ref_text_ids.len(),
            prefix_emb,
            audio_emb,
            t_ref_delayed,
            t_ref,
            ref_text: ref_text.to_owned(),
        })
    }

    /// `build_prompt` for the voice-clone path: the cached reference
    /// pieces, then the middle ids. The audio positions of the delayed
    /// reference ride the audio dual-FFN path.
    pub fn build_prompt_voice_clone(
        &self,
        target_ids: &[i32],
        reference: &ReferenceContext,
    ) -> Result<PromptEmbeds> {
        let hidden = self.config.text_config.hidden_size;
        let middle_emb = crate::lm::engine::lookup_embeddings(
            &self.embed_tokens,
            self.config.text_config.vocab_size,
            hidden,
            target_ids,
        )?;
        let mut embeds = Vec::with_capacity(
            reference.prefix_emb.len() + reference.audio_emb.len() + middle_emb.len(),
        );
        embeds.extend_from_slice(&reference.prefix_emb);
        embeds.extend_from_slice(&reference.audio_emb);
        embeds.extend_from_slice(&middle_emb);
        let mut audio_mask = vec![false; reference.prefix_len];
        audio_mask.resize(reference.prefix_len + reference.t_ref_delayed, true);
        audio_mask.resize(
            reference.prefix_len + reference.t_ref_delayed + target_ids.len(),
            false,
        );
        Ok(PromptEmbeds { embeds, audio_mask })
    }
}

#[cfg(test)]
mod tests;
