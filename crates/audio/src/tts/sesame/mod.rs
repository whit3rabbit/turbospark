//! Sesame CSM TTS family (CSM-1B / MisoTTS-style wrapper).
//!
//! Port of `mlx_audio/tts/models/sesame/` at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/sesame):
//! the CSM two-transformer architecture. A llama backbone over summed
//! per-codebook audio embeddings plus a text embedding predicts codebook 0;
//! a smaller depth decoder conditioned on the backbone hidden state (through
//! a projection) predicts codebooks 1..K-1 through the `audio_head` tensor.
//! Each frame of `audio_num_codebooks` codes decodes through the 24 kHz Mimi
//! codec, and the loop stops when a frame is all zeros. The reference
//! attention variant (`attention.py`) rotates queries and keys with an
//! interleaved (GPT-J style) Llama3-scaled RoPE; the backbone and decoder
//! transformers therefore live in this module, built from the shared
//! [`crate::lm`] attention/MLP math and rope constructions with the
//! interleaved pair convention. See [README.md](README.md) for the
//! verification evidence and open items.
//!
//! Scope of this port, matching the reference single-prompt path: batch
//! size 1, non-streaming decode (the frames decode to one waveform after
//! generation), and prompt construction from pre-tokenized ids. Text
//! tokenization is a caller concern (the reference loads the Llama-3.2
//! tokenizer from Hugging Face); [`text_prompt_string`] builds the exact
//! `"[{speaker}]{text}"` string the caller tokenizes.

use serde_json::Value;

use crate::codec::mimi::Mimi;
use crate::lm;
use crate::lm::cache::KvCache;
use crate::lm::engine::{Attention, Mlp};
use crate::music::minimax_music3::rng;
use crate::nn::{Linear, RmsNorm};
use crate::tts::sampling;
use crate::{quant, Result, SpeechError};

/// Reference Mimi checkpoint the audio tokenizer loads from.
pub const MIMI_REPO: &str = "kyutai/moshiko-pytorch-bf16";
/// Reference text tokenizer repo (Llama-3.2 1B).
pub const TOKENIZER_REPO: &str = "unsloth/Llama-3.2-1B";
/// Reference sample rate (the deployed Mimi codec).
pub const DEFAULT_SAMPLE_RATE: u32 = 24000;
/// Reference generation caps: 80 ms per frame (12.5 Hz) and a 2048-token
/// context the prompt must stay under once the frame budget is reserved.
pub const MS_PER_FRAME: f32 = 80.0;
pub const MAX_SEQ_LEN: usize = 2048;

/// The public CSM-1B watermark key constant
/// (`watermarking.CSM_1B_GH_WATERMARK`).
pub const CSM_1B_GH_WATERMARK: [u32; 5] = [212, 211, 146, 56, 201];

/// One llama flavor preset (`create_llama_model_args`), used when a
/// checkpoint config carries only `backbone_flavor`/`decoder_flavor`.
#[derive(Debug, Clone)]
pub struct LlamaFlavor {
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

/// The four reference flavor presets.
pub fn llama_flavor(flavor: &str) -> Result<LlamaFlavor> {
    let preset = match flavor {
        // hidden 2048, 16 layers, 32 heads, 8 kv heads, head_dim 64.
        "llama-1B" => LlamaFlavor {
            num_hidden_layers: 16,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 64,
            hidden_size: 2048,
            intermediate_size: 8192,
        },
        "llama-8B" => LlamaFlavor {
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            hidden_size: 4096,
            intermediate_size: 14336,
        },
        "llama-100M" => LlamaFlavor {
            num_hidden_layers: 4,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim: 128,
            hidden_size: 1024,
            intermediate_size: 8192,
        },
        "llama-300M" => LlamaFlavor {
            num_hidden_layers: 8,
            num_attention_heads: 24,
            num_key_value_heads: 6,
            head_dim: 64,
            hidden_size: 1536,
            intermediate_size: 6912,
        },
        other => {
            return Err(SpeechError::BadConfig {
                field: "flavor".into(),
                why: format!("unknown llama flavor {other}"),
            })
        }
    };
    Ok(preset)
}

fn flavor_dense(
    preset: &LlamaFlavor,
    rope_theta: f64,
    rope_scaling: Option<Value>,
) -> lm::DenseConfig {
    llama_dense(
        preset.num_hidden_layers,
        preset.num_attention_heads,
        preset.num_key_value_heads,
        preset.head_dim,
        preset.hidden_size,
        preset.intermediate_size,
        1e-5,
        128_256,
        2048,
        false,
        false,
        rope_theta,
        rope_scaling,
    )
}

#[allow(clippy::too_many_arguments)]
fn llama_dense(
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    intermediate_size: usize,
    rms_norm_eps: f64,
    vocab_size: usize,
    max_position_embeddings: usize,
    attention_bias: bool,
    mlp_bias: bool,
    rope_theta: f64,
    rope_scaling: Option<Value>,
) -> lm::DenseConfig {
    lm::DenseConfig {
        model_type: "llama".into(),
        hidden_size,
        num_hidden_layers,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        rms_norm_eps: rms_norm_eps as f32,
        vocab_size,
        qk_norms: false,
        attention_bias,
        mlp_bias,
        rope_theta: rope_theta as f32,
        rope_traditional: false,
        rope_scaling,
        max_position_embeddings: Some(max_position_embeddings),
        tie_word_embeddings: true,
        sliding_window: None,
    }
}

/// The fields `create_llama_model_args_for_backbone` reads off the config;
/// a missing field falls back to the flavor presets, exactly like the
/// reference try/except.
struct BackboneFields {
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    intermediate_size: usize,
    rms_norm_eps: f64,
    max_position_embeddings: usize,
    attention_bias: bool,
    mlp_bias: bool,
    rope_theta: f64,
}

fn backbone_fields(value: &Value) -> Option<BackboneFields> {
    let u = |key: &str| value.get(key).and_then(Value::as_u64);
    Some(BackboneFields {
        num_hidden_layers: u("num_hidden_layers")? as usize,
        num_attention_heads: u("num_attention_heads")? as usize,
        num_key_value_heads: u("num_key_value_heads")? as usize,
        head_dim: u("head_dim")? as usize,
        hidden_size: u("hidden_size")? as usize,
        intermediate_size: u("intermediate_size")? as usize,
        rms_norm_eps: value.get("rms_norm_eps")?.as_f64()?,
        max_position_embeddings: u("max_position_embeddings")? as usize,
        attention_bias: value.get("attention_bias")?.as_bool()?,
        mlp_bias: value.get("mlp_bias")?.as_bool()?,
        rope_theta: value.get("rope_theta")?.as_f64()?,
    })
}

/// The TTS `ModelConfig`: the two llama backbone configurations plus the
/// audio vocabulary geometry and the generate-behavior flags the reference
/// `Model.__init__` reads off the config dict.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub backbone: lm::DenseConfig,
    pub decoder: lm::DenseConfig,
    pub text_vocab_size: usize,
    pub audio_vocab_size: usize,
    pub audio_num_codebooks: usize,
    /// `"[0] text"` (with the space) instead of `"[0]text"` when set.
    pub speaker_prefix_space: bool,
    pub use_default_voice_prompt: bool,
    pub voice_match: bool,
}

impl ModelConfig {
    pub fn from_json(value: &Value) -> Result<Self> {
        let text_vocab_size = value
            .get("text_vocab_size")
            .and_then(Value::as_u64)
            .ok_or_else(|| SpeechError::BadConfig {
                field: "text_vocab_size".into(),
                why: "must be a positive integer".into(),
            })? as usize;
        let audio_vocab_size = value
            .get("audio_vocab_size")
            .and_then(Value::as_u64)
            .ok_or_else(|| SpeechError::BadConfig {
                field: "audio_vocab_size".into(),
                why: "must be a positive integer".into(),
            })? as usize;
        let audio_num_codebooks = value
            .get("audio_num_codebooks")
            .and_then(Value::as_u64)
            .ok_or_else(|| SpeechError::BadConfig {
                field: "audio_num_codebooks".into(),
                why: "must be a positive integer".into(),
            })? as usize;
        if audio_num_codebooks < 1 {
            return Err(SpeechError::BadConfig {
                field: "audio_num_codebooks".into(),
                why: "must be at least 1".into(),
            });
        }

        // Reference flow: the explicit backbone/depth-decoder fields are
        // tried first; ANY missing field (including rope_scaling inside the
        // depth dict) falls back to the flavor presets.
        let explicit = (|| {
            let backbone = backbone_fields(value)?;
            let depth = value.get("depth_decoder_config")?;
            let depth_u = |key: &str| depth.get(key).and_then(Value::as_u64);
            let rope_scaling = value.get("rope_scaling").cloned().filter(|v| !v.is_null());
            let decoder_scaling = depth.get("rope_scaling")?.clone();
            let decoder = llama_dense(
                depth_u("num_hidden_layers")? as usize,
                depth_u("num_attention_heads")? as usize,
                depth_u("num_key_value_heads")? as usize,
                depth_u("head_dim")? as usize,
                depth_u("hidden_size")? as usize,
                depth_u("intermediate_size")? as usize,
                depth.get("rms_norm_eps")?.as_f64()?,
                depth_u("vocab_size")? as usize,
                depth_u("max_position_embeddings")? as usize,
                depth.get("attention_bias")?.as_bool()?,
                depth.get("mlp_bias")?.as_bool()?,
                depth.get("rope_theta")?.as_f64()?,
                Some(decoder_scaling),
            );
            let backbone = llama_dense(
                backbone.num_hidden_layers,
                backbone.num_attention_heads,
                backbone.num_key_value_heads,
                backbone.head_dim,
                backbone.hidden_size,
                backbone.intermediate_size,
                backbone.rms_norm_eps,
                text_vocab_size,
                backbone.max_position_embeddings,
                backbone.attention_bias,
                backbone.mlp_bias,
                backbone.rope_theta,
                rope_scaling,
            );
            Some((backbone, decoder))
        })();

        let (backbone, decoder) = match explicit {
            Some(pair) => pair,
            None => {
                let backbone_flavor = value
                    .get("backbone_flavor")
                    .and_then(Value::as_str)
                    .ok_or_else(|| SpeechError::BadConfig {
                        field: "backbone".into(),
                        why: "config has neither the backbone fields nor backbone_flavor".into(),
                    })?;
                let decoder_flavor = value
                    .get("decoder_flavor")
                    .and_then(Value::as_str)
                    .ok_or_else(|| SpeechError::BadConfig {
                        field: "decoder".into(),
                        why: "config has neither the depth decoder fields nor decoder_flavor"
                            .into(),
                    })?;
                // The reference presets share one rope_scaling dict read
                // from whatever config key exists; the flavor presets carry
                // their own llama3 scaling, so the config dict (if any)
                // only supplies theta-consistent factor metadata.
                let scaling = || {
                    Some(serde_json::json!({
                        "factor": 32.0,
                        "low_freq_factor": 1.0,
                        "high_freq_factor": 4.0,
                        "original_max_position_embeddings": 8192,
                        "rope_type": "llama3",
                    }))
                };
                (
                    flavor_dense(&llama_flavor(backbone_flavor)?, 500_000.0, scaling()),
                    flavor_dense(&llama_flavor(decoder_flavor)?, 500_000.0, scaling()),
                )
            }
        };

        Ok(Self {
            backbone,
            decoder,
            text_vocab_size,
            audio_vocab_size,
            audio_num_codebooks,
            speaker_prefix_space: value
                .get("speaker_prefix_space")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            use_default_voice_prompt: value
                .get("use_default_voice_prompt")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            voice_match: value
                .get("voice_match")
                .and_then(Value::as_bool)
                .unwrap_or(true),
        })
    }
}

/// Sampling and generation options for one prompt, with the reference
/// `generate` defaults (`make_sampler(temp=0.9, top_k=50)`, 90 s cap).
#[derive(Debug, Clone)]
pub struct GenerateParams {
    /// 0.0 selects the exact greedy path (`make_sampler(0, ...)`).
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    /// `<= 0` disables, matching the reference `make_sampler(top_k=...)`.
    pub top_k: i32,
    /// `max_audio_length_ms`: the frame budget is `ms / 80`.
    pub max_audio_length_ms: f32,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            temperature: 0.9,
            top_p: 0.0,
            min_p: 0.0,
            top_k: 50,
            max_audio_length_ms: 90_000.0,
        }
    }
}

/// Why the generate loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The sampled frame was all zeros (the reference eos).
    AllZeroFrame,
    /// The frame budget ran out.
    MaxFrames,
}

/// One generated prompt: the sampled frames, the codebook-major codes,
/// and the decoded mono waveform.
#[derive(Debug, Clone)]
pub struct Generated {
    /// One entry per generated frame, each `audio_num_codebooks` codes.
    pub frames: Vec<Vec<i32>>,
    /// Codebook-major view of the frames (`[K][T]`), the Mimi decode input.
    pub codes: Vec<Vec<i32>>,
    pub samples: Vec<f32>,
    pub stopped_by: StopReason,
}

/// The prompt frame tensors (`_tokenize_segment` output): ids and mask,
/// each `[rows, audio_num_codebooks + 1]` row-major.
#[derive(Debug, Clone)]
pub struct Prompt {
    pub tokens: Vec<i32>,
    pub mask: Vec<bool>,
    pub rows: usize,
    /// `audio_num_codebooks + 1`.
    pub width: usize,
}

impl Prompt {
    /// Concatenates section frames (`mx.concat([text, audio], axis=0)`).
    pub fn concat(sections: &[Prompt]) -> Result<Prompt> {
        let first = sections.first().ok_or_else(|| SpeechError::Input {
            why: "empty prompt sections".into(),
        })?;
        let width = first.width;
        let mut tokens = Vec::new();
        let mut mask = Vec::new();
        let mut rows = 0;
        for section in sections {
            if section.width != width {
                return Err(SpeechError::Input {
                    why: format!(
                        "prompt section width {} does not match {width}",
                        section.width
                    ),
                });
            }
            tokens.extend_from_slice(&section.tokens);
            mask.extend_from_slice(&section.mask);
            rows += section.rows;
        }
        Ok(Prompt {
            tokens,
            mask,
            rows,
            width,
        })
    }
}

/// The exact string the reference tokenizes for a text segment
/// (`_tokenize_text_segment`): `"[{speaker}]{text}"`, with a space after
/// the bracket when `speaker_prefix_space` is set. The caller tokenizes
/// this with the Llama-3 tokenizer (BOS/EOS are added by the tokenizer
/// template, so those ids belong in `text_ids` too).
pub fn text_prompt_string(speaker: i32, text: &str, speaker_prefix_space: bool) -> String {
    let prompt_text = if speaker_prefix_space {
        text.trim_start()
    } else {
        text
    };
    if speaker_prefix_space {
        format!("[{speaker}] {prompt_text}")
    } else {
        format!("[{speaker}]{prompt_text}")
    }
}

/// One transformer stack (backbone or depth decoder): llama blocks over
/// the family attention (interleaved Llama3-scaled rope), no embeddings
/// and no LM head (the reference replaces `embed_tokens` with `Identity`
/// and never applies a head).
struct Transformer {
    config: lm::DenseConfig,
    layers: Vec<Block>,
    norm: RmsNorm,
}

struct Block {
    attention: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

/// Builds the interleaved Llama3-scaled rope of `attention.py`
/// (`Llama3ScaledRoPE`): the llama3 frequency scaling with the attention
/// module's own defaults (low 1, high 4, old context 8192, factor from
/// the config `rope_scaling`), applied with the adjacent-pair convention.
fn family_rope(config: &lm::DenseConfig) -> Result<lm::rope::Rope> {
    // Absent rope_scaling means unscaled RoPE. A present block without a
    // numeric factor is an error: defaulting to 1.0 would silently drop
    // the scaling the checkpoint asked for.
    let factor = match config.rope_scaling.as_ref() {
        None => 1.0,
        Some(scaling) => scaling
            .get("factor")
            .and_then(Value::as_f64)
            .ok_or_else(|| SpeechError::BadConfig {
                field: "rope_scaling.factor".into(),
                why: "rope_scaling is present but factor is missing or not a number".into(),
            })?,
    };
    let scaling = serde_json::json!({
        "rope_type": "llama3",
        "factor": factor,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192.0,
    });
    lm::rope::Rope::initialize(
        config.head_dim,
        config.rope_theta,
        true,
        Some(&scaling),
        None,
    )
}

impl Transformer {
    /// Loads one stack (`backbone` or `decoder`) under the `model.`
    /// checkpoint prefix. Every attention projection carries the config
    /// bias, including `o_proj` (the reference `Attention.__init__`).
    fn load(weights: &lm::Weights, name: &str, config: &lm::DenseConfig) -> Result<Self> {
        let rope = family_rope(config)?;
        let hidden = config.hidden_size;
        let kv_width = config.num_key_value_heads * config.head_dim;
        let q_width = config.num_attention_heads * config.head_dim;
        let attention_bias = if config.attention_bias {
            lm::Bias::Required
        } else {
            lm::Bias::None
        };
        let mlp_bias = if config.mlp_bias {
            lm::Bias::Required
        } else {
            lm::Bias::None
        };
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let layer = format!("{name}.layers.{index}.");
            let attention = Attention {
                q_proj: weights.linear(
                    &format!("{layer}self_attn.q_proj"),
                    hidden,
                    q_width,
                    attention_bias,
                )?,
                k_proj: weights.linear(
                    &format!("{layer}self_attn.k_proj"),
                    hidden,
                    kv_width,
                    attention_bias,
                )?,
                v_proj: weights.linear(
                    &format!("{layer}self_attn.v_proj"),
                    hidden,
                    kv_width,
                    attention_bias,
                )?,
                o_proj: weights.linear(
                    &format!("{layer}self_attn.o_proj"),
                    q_width,
                    hidden,
                    attention_bias,
                )?,
                q_norm: None,
                k_norm: None,
                n_heads: config.num_attention_heads,
                n_kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                scale: (config.head_dim as f64).powf(-0.5) as f32,
                rope: Some(rope.clone()),
                window: None,
                bidirectional: false,
                exec: None,
            };
            let mlp = Mlp {
                gate_proj: weights.linear(
                    &format!("{layer}mlp.gate_proj"),
                    hidden,
                    config.intermediate_size,
                    mlp_bias,
                )?,
                up_proj: weights.linear(
                    &format!("{layer}mlp.up_proj"),
                    hidden,
                    config.intermediate_size,
                    mlp_bias,
                )?,
                down_proj: weights.linear(
                    &format!("{layer}mlp.down_proj"),
                    config.intermediate_size,
                    hidden,
                    mlp_bias,
                )?,
                exec: None,
            };
            layers.push(Block {
                attention,
                mlp,
                input_layernorm: RmsNorm::new(
                    weights.tensor(&format!("{layer}input_layernorm"), &[hidden])?,
                    config.rms_norm_eps,
                ),
                post_attention_layernorm: RmsNorm::new(
                    weights.tensor(&format!("{layer}post_attention_layernorm"), &[hidden])?,
                    config.rms_norm_eps,
                ),
            });
        }
        Ok(Self {
            config: config.clone(),
            layers,
            norm: RmsNorm::new(
                weights.tensor(&format!("{name}.norm"), &[hidden])?,
                config.rms_norm_eps,
            ),
        })
    }

    /// Runs the stack over row-major embeddings `[rows, hidden]` with the
    /// layer caches advancing, returning the post-norm hidden states.
    fn forward(&mut self, h: &[f32], caches: &mut [KvCache]) -> Result<Vec<f32>> {
        let rows = h.len() / self.config.hidden_size;
        if rows == 0 {
            return Err(SpeechError::Input {
                why: "empty transformer input".into(),
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
        let offset = caches.first().map(|c| c.offset).unwrap_or(0);
        let mut h = h.to_vec();
        for (index, block) in self.layers.iter_mut().enumerate() {
            let mut attended = h.clone();
            block.input_layernorm.apply(&mut attended, rows);
            let attn = block.attention.forward(
                &attended,
                rows,
                offset,
                Some(&mut caches[index] as &mut dyn lm::cache::LayerKv),
            )?;
            for (row, out) in attn.chunks_exact(self.config.hidden_size).enumerate() {
                let base = row * self.config.hidden_size;
                for (col, value) in out.iter().enumerate() {
                    h[base + col] += value;
                }
            }
            let mut fed = h.clone();
            block.post_attention_layernorm.apply(&mut fed, rows);
            let mlp = block.mlp.forward(&fed, rows);
            for (row, out) in mlp.chunks_exact(self.config.hidden_size).enumerate() {
                let base = row * self.config.hidden_size;
                for (col, value) in out.iter().enumerate() {
                    h[base + col] += value;
                }
            }
        }
        self.norm.apply(&mut h, rows);
        Ok(h)
    }
}

/// The loaded family model: the two transformers, the embedding tables,
/// the codebook heads, and the rolling caches the reference keeps on the
/// model (`setup_caches` at construction, `reset_caches` per generate).
pub struct Model {
    pub config: ModelConfig,
    backbone: Transformer,
    decoder: Transformer,
    text_embeddings: Vec<f32>,
    audio_embeddings: Vec<f32>,
    projection: Linear,
    codebook0_head: Linear,
    /// `[(K - 1), decoder_dim, audio_vocab_size]`, applied by matmul.
    audio_head: Vec<f32>,
    backbone_caches: Vec<KvCache>,
    decoder_caches: Vec<KvCache>,
}

impl Model {
    /// Opens a checkpoint directory (`config.json` + safetensors).
    pub fn open(dir: &std::path::Path) -> Result<Self> {
        let config_value = quant::read_json(&dir.join("config.json"))?;
        let weights = lm::Weights::open(dir)?.with_prefix("model.");
        Self::load(&weights, &config_value)
    }

    /// Loads from opened weights and the raw config dict.
    pub fn load(weights: &lm::Weights, config: &Value) -> Result<Self> {
        let config = ModelConfig::from_json(config)?;
        let backbone_dim = config.backbone.hidden_size;
        let decoder_dim = config.decoder.hidden_size;
        let codebooks = config.audio_num_codebooks;
        let audio_vocab = config.audio_vocab_size;
        let backbone = Transformer::load(weights, "backbone", &config.backbone)?;
        let decoder = Transformer::load(weights, "decoder", &config.decoder)?;
        let text_embeddings =
            weights.embedding("text_embeddings", config.text_vocab_size, backbone_dim)?;
        let audio_embeddings =
            weights.embedding("audio_embeddings", audio_vocab * codebooks, backbone_dim)?;
        let projection = weights.linear("projection", backbone_dim, decoder_dim, lm::Bias::None)?;
        let codebook0_head =
            weights.linear("codebook0_head", backbone_dim, audio_vocab, lm::Bias::None)?;
        let audio_head = weights.raw_tensor(
            "audio_head",
            &[codebooks.saturating_sub(1), decoder_dim, audio_vocab],
        )?;
        let mut model = Self {
            backbone_caches: lm::make_prompt_cache(config.backbone.num_hidden_layers),
            decoder_caches: lm::make_prompt_cache(config.decoder.num_hidden_layers),
            backbone,
            decoder,
            text_embeddings,
            audio_embeddings,
            projection,
            codebook0_head,
            audio_head,
            config,
        };
        model.reset_caches();
        Ok(model)
    }

    pub fn audio_num_codebooks(&self) -> usize {
        self.config.audio_num_codebooks
    }

    /// The reference frame width: one row per time step holds the K audio
    /// codes plus one text column.
    pub fn frame_size(&self) -> usize {
        self.config.audio_num_codebooks + 1
    }

    /// Reference `reset_caches`: fresh caches for both stacks. The
    /// generate entry points call this, matching `generate`.
    pub fn reset_caches(&mut self) {
        self.backbone_caches = lm::make_prompt_cache(self.config.backbone.num_hidden_layers);
        self.decoder_caches = lm::make_prompt_cache(self.config.decoder.num_hidden_layers);
    }

    fn embedding_row<'a>(&self, table: &'a [f32], width: usize, row: usize) -> Result<&'a [f32]> {
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

    /// Reference `_embed_tokens` plus the mask-sum: every row of the
    /// `[rows, K + 1]` id block sums its K masked audio-codebook
    /// embeddings (codebook c rides row `code * audio_vocab + c * audio_vocab`)
    /// with the masked text embedding in the last column.
    fn embed_rows(&self, tokens: &[i32], mask: &[bool], rows: usize) -> Result<Vec<f32>> {
        let k = self.config.audio_num_codebooks;
        let width = k + 1;
        let dim = self.config.backbone.hidden_size;
        let audio_vocab = self.config.audio_vocab_size;
        let mut out = vec![0.0f32; rows * dim];
        for row in 0..rows {
            let base = row * width;
            for codebook in 0..k {
                if !mask[base + codebook] {
                    continue;
                }
                let token = tokens[base + codebook];
                // Range-check before indexing: a code at or past
                // audio_vocab would read the next codebook's rows.
                let code = usize::try_from(token)
                    .ok()
                    .filter(|&c| c < audio_vocab)
                    .ok_or_else(|| SpeechError::Input {
                        why: format!("audio token {token} outside codebook of {audio_vocab}"),
                    })?;
                let index = codebook * audio_vocab + code;
                let embed = self.embedding_row(&self.audio_embeddings, dim, index)?;
                for (col, value) in out[row * dim..(row + 1) * dim].iter_mut().enumerate() {
                    *value += embed[col];
                }
            }
            if mask[base + k] {
                let token = tokens[base + k];
                let index = usize::try_from(token).map_err(|_| SpeechError::Input {
                    why: format!("negative text token {token}"),
                })?;
                let embed = self.embedding_row(&self.text_embeddings, dim, index)?;
                for (col, value) in out[row * dim..(row + 1) * dim].iter_mut().enumerate() {
                    *value += embed[col];
                }
            }
        }
        Ok(out)
    }

    /// The per-codebook sampler step (`sampler(logits)` on the raw
    /// logits): greedy argmax at temperature 0, otherwise the
    /// `make_sampler` filter order (top-p, min-p, then top-k) and the
    /// categorical draw over the raw (softmax-normalized) values.
    fn sample(&self, logits: &[f32], params: &GenerateParams, key: rng::Key) -> Result<usize> {
        if params.temperature == 0.0 {
            return Ok(sampling::argmax(logits));
        }
        let mut filtered = logits.to_vec();
        if params.top_p > 0.0 && params.top_p < 1.0 {
            sampling::apply_top_p(&mut filtered, params.top_p);
        }
        if params.min_p != 0.0 {
            sampling::apply_min_p(&mut filtered, params.min_p, 1)?;
        }
        if params.top_k > 0 {
            sampling::apply_top_k(&mut filtered, params.top_k as usize)?;
        }
        Ok(sampling::sample_categorical(
            &filtered,
            params.temperature,
            key,
        ))
    }

    /// Reference `generate_frame` for the single row this port runs:
    /// embed and sum the frame block, run the backbone, sample codebook 0
    /// from the backbone hidden state, then grow the (fresh) depth decoder
    /// over `[hidden, codebook 0 embed, codebook 1 embed, ...]` sampling
    /// codebooks 1..K-1 through `audio_head`. Returns the K sampled codes.
    pub fn generate_frame(
        &mut self,
        tokens: &[i32],
        mask: &[bool],
        params: &GenerateParams,
        keys: &mut rng::KeySequence,
    ) -> Result<Vec<i32>> {
        let k = self.config.audio_num_codebooks;
        let rows = tokens.len() / (k + 1);
        if tokens.len() != rows * (k + 1) || mask.len() != tokens.len() {
            return Err(SpeechError::Input {
                why: format!(
                    "frame block must be [{rows}, {}] ids and masks, got {} ids / {} masks",
                    k + 1,
                    tokens.len(),
                    mask.len()
                ),
            });
        }

        // Backbone pass over the summed embeddings; the last position
        // predicts codebook 0.
        let embeds = self.embed_rows(tokens, mask, rows)?;
        let hidden = self.backbone.forward(&embeds, &mut self.backbone_caches)?;
        let dim = self.config.backbone.hidden_size;
        let last = &hidden[(rows - 1) * dim..rows * dim];
        let c0_logits = self.codebook0_head.forward(last, 1);
        let c0 = self.sample(&c0_logits, params, keys.next())?;

        // Depth decoder: the first call sees [projected hidden, projected
        // codebook 0 embed]; every later call sees one position, the
        // projected embed of the previous codebook's sample. The decoder
        // cache resets for every new frame (the reference rewrites
        // `decoder_cache` inside `generate_frame`).
        self.decoder_caches = lm::make_prompt_cache(self.config.decoder.num_hidden_layers);
        let decoder_dim = self.config.decoder.hidden_size;
        let audio_vocab = self.config.audio_vocab_size;
        let c0_embed = self.embedding_row(&self.audio_embeddings, dim, c0)?;
        let mut curr_rows = self.projection.forward(last, 1);
        curr_rows.extend_from_slice(&self.projection.forward(c0_embed, 1));
        let mut frame = Vec::with_capacity(k);
        frame.push(c0 as i32);
        for codebook in 1..k {
            let decoder_hidden = self.decoder.forward(&curr_rows, &mut self.decoder_caches)?;
            let decoder_rows = decoder_hidden.len() / decoder_dim;
            let last_row =
                &decoder_hidden[(decoder_rows - 1) * decoder_dim..decoder_rows * decoder_dim];
            let head = &self.audio_head
                [(codebook - 1) * decoder_dim * audio_vocab..codebook * decoder_dim * audio_vocab];
            let mut logits = vec![0.0f32; audio_vocab];
            for (d, &value) in last_row.iter().enumerate() {
                let base = d * audio_vocab;
                for (v, logit) in logits.iter_mut().enumerate() {
                    *logit += value * head[base + v];
                }
            }
            let next = self.sample(&logits, params, keys.next())?;
            frame.push(next as i32);
            let index = codebook * audio_vocab + next;
            let embed = self.embedding_row(&self.audio_embeddings, dim, index)?;
            curr_rows = self.projection.forward(embed, 1);
        }
        Ok(frame)
    }

    /// Reference `_tokenize_text_segment` for pre-tokenized ids: one frame
    /// row per text token, all audio columns zero and masked off, the text
    /// id in the last column and masked on.
    pub fn tokenize_text_segment(&self, text_ids: &[i32]) -> Prompt {
        let width = self.frame_size();
        let mut tokens = vec![0i32; text_ids.len() * width];
        let mut mask = vec![false; text_ids.len() * width];
        for (row, &id) in text_ids.iter().enumerate() {
            tokens[row * width + width - 1] = id;
            mask[row * width + width - 1] = true;
        }
        Prompt {
            tokens,
            mask,
            rows: text_ids.len(),
            width,
        }
    }

    /// Reference `_tokenize_audio` on codec codes `[K][T]`: one frame row
    /// per time step with the K codes in the audio columns (masked on) and
    /// a zero text column; `add_eos` appends one all-zero frame. The
    /// codebook count must match `audio_num_codebooks`.
    pub fn tokenize_audio(&self, codes: &[Vec<i32>], add_eos: bool) -> Result<Prompt> {
        let k = self.config.audio_num_codebooks;
        if codes.len() != k {
            return Err(SpeechError::Input {
                why: format!(
                    "audio tokenizer returned {} codebooks, expected {k}",
                    codes.len()
                ),
            });
        }
        let frames = codes.first().map_or(0, |book| book.len());
        for book in codes {
            if book.len() != frames {
                return Err(SpeechError::Input {
                    why: "ragged codebooks".into(),
                });
            }
        }
        let audio_vocab = self.config.audio_vocab_size;
        for book in codes {
            if let Some(&bad) = book.iter().find(|&&c| c < 0 || c as usize >= audio_vocab) {
                return Err(SpeechError::Input {
                    why: format!("audio code {bad} outside codebook of {audio_vocab}"),
                });
            }
        }
        let rows = frames + usize::from(add_eos);
        let width = self.frame_size();
        let mut tokens = vec![0i32; rows * width];
        let mut mask = vec![false; rows * width];
        for frame in 0..frames {
            for (codebook, book) in codes.iter().enumerate() {
                tokens[frame * width + codebook] = book[frame];
            }
            mask[frame * width..frame * width + k].fill(true);
        }
        if add_eos {
            mask[(rows - 1) * width..(rows - 1) * width + k].fill(true);
        }
        Ok(Prompt {
            tokens,
            mask,
            rows,
            width,
        })
    }

    /// Reference `_tokenize_segment`: the text rows followed by the audio
    /// rows.
    pub fn tokenize_segment(
        &self,
        text_ids: &[i32],
        codes: &[Vec<i32>],
        add_eos: bool,
    ) -> Result<Prompt> {
        let text = self.tokenize_text_segment(text_ids);
        let audio = self.tokenize_audio(codes, add_eos)?;
        Prompt::concat(&[text, audio])
    }

    /// Runs one prompt end to end (`generate` for a single non-streaming
    /// prompt): fresh caches, up to `max_audio_length_ms / 80` frames,
    /// stop on an all-zero frame, then decode every generated frame
    /// through Mimi. `seed` only feeds the categorical draws; temperature
    /// 0 never draws and is fully deterministic. An immediate all-zero
    /// frame produces empty frames and an empty waveform (the reference
    /// yields no result in that case).
    ///
    /// The decode is one full Mimi pass over the codebook-major codes; the
    /// reference decodes through the stateful `MimiStreamingDecoder` in
    /// 62-frame chunks, which produces the same samples (the streaming
    /// decoder carries the same state a full pass consumes).
    pub fn generate(
        &mut self,
        mimi: &Mimi,
        prompt: &Prompt,
        params: &GenerateParams,
        seed: u64,
    ) -> Result<Generated> {
        let (frames, stopped_by) = self.generate_frames(prompt, params, seed)?;
        let codes = frames_to_codes(&frames, self.config.audio_num_codebooks);
        let samples = if frames.is_empty() {
            Vec::new()
        } else {
            mimi.decode(&codes)?
        };
        Ok(Generated {
            frames,
            codes,
            samples,
            stopped_by,
        })
    }

    /// The frame loop alone: fresh caches, the reference eos and frame
    /// budget checks, and the sampled frames. No decode. Callers that
    /// stream intervals or that feed deliberately non-audio prompts drive
    /// this directly.
    pub fn generate_frames(
        &mut self,
        prompt: &Prompt,
        params: &GenerateParams,
        seed: u64,
    ) -> Result<(Vec<Vec<i32>>, StopReason)> {
        if prompt.rows == 0 {
            return Err(SpeechError::Input {
                why: "empty prompt".into(),
            });
        }
        let max_audio_frames = (params.max_audio_length_ms / MS_PER_FRAME) as usize;
        let max_seq_len = MAX_SEQ_LEN.saturating_sub(max_audio_frames);
        if prompt.rows >= max_seq_len {
            return Err(SpeechError::Input {
                why: format!(
                    "inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}"
                ),
            });
        }
        self.reset_caches();
        let width = self.frame_size();
        if prompt.width != width {
            return Err(SpeechError::Input {
                why: format!(
                    "prompt width {} does not match frame size {width}",
                    prompt.width
                ),
            });
        }
        let mut keys = rng::KeySequence::new(seed);
        let mut frames: Vec<Vec<i32>> = Vec::new();
        let mut stopped_by = StopReason::MaxFrames;
        let mut curr_tokens = prompt.tokens.clone();
        let mut curr_mask = prompt.mask.clone();
        for _ in 0..max_audio_frames {
            let frame = self.generate_frame(&curr_tokens, &curr_mask, params, &mut keys)?;
            if frame.iter().all(|&token| token == 0) {
                stopped_by = StopReason::AllZeroFrame;
                break;
            }
            // The next input row is the sampled frame plus a zero (masked)
            // text column.
            let mut row = frame.clone();
            row.push(0);
            curr_tokens = row;
            let mut row_mask = vec![true; width];
            row_mask[width - 1] = false;
            curr_mask = row_mask;
            frames.push(frame);
        }
        Ok((frames, stopped_by))
    }
}

/// Transposes the generated frames `[T][K]` into the codebook-major
/// `[K][T]` layout the Mimi decode consumes. An empty frame list stays
/// empty (the reference decode never runs without frames).
pub fn frames_to_codes(frames: &[Vec<i32>], codebooks: usize) -> Vec<Vec<i32>> {
    if frames.is_empty() {
        return Vec::new();
    }
    let mut codes = vec![Vec::with_capacity(frames.len()); codebooks];
    for frame in frames {
        for (book, code) in frame.iter().take(codebooks).enumerate() {
            codes[book].push(*code);
        }
    }
    codes
}

/// Reference `encode_audio`: codec-encode mono samples into the
/// `[K][T]` codebook layout the prompt builders consume.
pub fn encode_audio(mimi: &Mimi, samples: &[f32]) -> Result<Vec<Vec<i32>>> {
    if samples.is_empty() {
        return Err(SpeechError::Input {
            why: "empty waveform".into(),
        });
    }
    mimi.encode(samples)
}

/// Reference `watermark` (`sesame/watermarking.py`): REFUSED. The stage
/// needs the silentcipher 44.1 kHz watermark model, a separate
/// Hugging Face download outside the pinned mlx-audio tree; this port
/// returns generated audio unwatermarked and this entry point fails with
/// a descriptive error instead of porting blind.
pub fn watermark(audio: &[f32], sample_rate: u32, key: &[u32]) -> Result<Vec<f32>> {
    let _ = (audio.len(), sample_rate, key.len());
    Err(SpeechError::Unsupported {
        why: "the CSM audio watermark requires the silentcipher 44.1k watermark \
              model, a separately downloaded checkpoint; the watermark stage is \
              not ported and generated audio is returned unwatermarked"
            .into(),
    })
}

#[cfg(test)]
mod tests;
