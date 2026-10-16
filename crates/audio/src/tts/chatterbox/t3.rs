//! The chatterbox T3 token generator: a llama backbone fed composite
//! `[conditioning | text | BOS | BOS]` embeddings, with text and speech
//! embedding tables, learned position embeddings, dedicated text and
//! speech output heads, a conditioning encoder (speaker projection,
//! perceiver resampler, emotion scalar), and the classifier-free
//! guidance sampling loop.
//!
//! Reference: `chatterbox/t3/` (`t3.py`, `cond_enc.py`,
//! `perceiver.py`, `learned_pos_emb.py`) plus the `chatterbox/config.py`
//! T3/llama configuration, all at the pinned commit recorded in the
//! parent module.
//!
//! Batch handling: the reference runs CFG as a batch of two through the
//! transformer. This workspace's LM engine is single-batch, so the two
//! rows run as two forward passes over independent prompt caches - the
//! same math, because attention never crosses the batch axis (the
//! recorded fixture decision margins keep the f32 reordering honest).
//! The llama head is never applied: T3 carries its own text/speech
//! heads, so the backbone loads without its embedding table and with a
//! tied-head config regardless of the checkpoint's tie flag.
//!
//! Sampling: the reference filters and draws on the RAW post-CFG,
//! post-penalty logits (the shared sampler helpers are input-agnostic
//! and match that contract exactly: top-p compares unnormalized
//! exponential sums, min-p is shift invariant, and the categorical draw
//! renormalizes internally). Temperature 0 takes the plain argmax after
//! the penalty, exactly like `make_sampler(0, ...)`.

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::lm::{self, engine::Input};
use crate::music::minimax_music3::rng;
use crate::nn::load_tensor;
use crate::tts::sampling;
use crate::{Result, SpeechError};

use super::s3gen::{SaLayerNorm, SaLinear};

/// Pinned chatterbox T3 defaults (`config.py::T3Config`).
#[derive(Debug, Clone)]
pub struct T3Config {
    pub text_tokens_dict_size: usize,
    pub start_text_token: i32,
    pub stop_text_token: i32,
    pub max_text_tokens: usize,
    pub speech_tokens_dict_size: usize,
    pub start_speech_token: i32,
    pub stop_speech_token: i32,
    pub max_speech_tokens: usize,
    pub input_pos_emb: String,
    pub speech_cond_prompt_len: usize,
    pub encoder_type: String,
    pub speaker_embed_size: usize,
    pub use_perceiver_resampler: bool,
    pub emotion_adv: bool,
    /// The llama backbone configuration (`LLAMA_CONFIGS` entry merged
    /// into the T3 dict); parsed through the shared llama config.
    pub llama: Value,
}

impl Default for T3Config {
    fn default() -> Self {
        T3Config {
            text_tokens_dict_size: 704,
            start_text_token: 255,
            stop_text_token: 0,
            max_text_tokens: 2048,
            speech_tokens_dict_size: 8194,
            start_speech_token: 6561,
            stop_speech_token: 6562,
            max_speech_tokens: 4096,
            input_pos_emb: "learned".to_string(),
            speech_cond_prompt_len: 150,
            encoder_type: "voice_encoder".to_string(),
            speaker_embed_size: 256,
            use_perceiver_resampler: true,
            emotion_adv: true,
            llama: llama_520m_config(),
        }
    }
}

/// The pinned `LLAMA_520M_CONFIG` llama configuration dict.
pub fn llama_520m_config() -> Value {
    serde_json::json!({
        "model_type": "llama",
        "vocab_size": 4000,
        "hidden_size": 1024,
        "num_hidden_layers": 30,
        "intermediate_size": 4096,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 64,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-05,
        "rope_theta": 500000.0,
        "rope_scaling": {
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        "attention_bias": false,
        "mlp_bias": false,
        "tie_word_embeddings": false,
    })
}

impl T3Config {
    /// Parses the reference `T3Config` fields; missing or null fields
    /// keep the defaults. The llama configuration comes from the
    /// `llama_config` key when present (the fixture injects a tiny
    /// backbone there); otherwise the pinned `LLAMA_520M_CONFIG`.
    pub fn from_json_with_defaults(value: &Value, defaults: T3Config) -> Result<Self> {
        let int = |name: &str, fallback: i64| -> Result<i64> {
            match value.get(name) {
                None | Some(Value::Null) => Ok(fallback),
                Some(v) => v.as_i64().ok_or_else(|| SpeechError::BadConfig {
                    field: name.to_string(),
                    why: "expected an integer".to_string(),
                }),
            }
        };
        let usize_field = |name: &str, fallback: usize| -> Result<usize> {
            Ok(int(name, fallback as i64)? as usize)
        };
        let text = |name: &str, fallback: &str| -> Result<String> {
            match value.get(name) {
                None | Some(Value::Null) => Ok(fallback.to_string()),
                Some(v) => v
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| SpeechError::BadConfig {
                        field: name.to_string(),
                        why: "expected a string".to_string(),
                    }),
            }
        };
        let boolean = |name: &str, fallback: bool| -> Result<bool> {
            match value.get(name) {
                None | Some(Value::Null) => Ok(fallback),
                Some(v) => v.as_bool().ok_or_else(|| SpeechError::BadConfig {
                    field: name.to_string(),
                    why: "expected a bool".to_string(),
                }),
            }
        };
        let llama = match value.get("llama_config") {
            None | Some(Value::Null) => defaults.llama.clone(),
            Some(v) => v.clone(),
        };
        Ok(T3Config {
            text_tokens_dict_size: usize_field(
                "text_tokens_dict_size",
                defaults.text_tokens_dict_size,
            )?,
            start_text_token: int("start_text_token", defaults.start_text_token as i64)? as i32,
            stop_text_token: int("stop_text_token", defaults.stop_text_token as i64)? as i32,
            max_text_tokens: usize_field("max_text_tokens", defaults.max_text_tokens)?,
            speech_tokens_dict_size: usize_field(
                "speech_tokens_dict_size",
                defaults.speech_tokens_dict_size,
            )?,
            start_speech_token: int("start_speech_token", defaults.start_speech_token as i64)?
                as i32,
            stop_speech_token: int("stop_speech_token", defaults.stop_speech_token as i64)? as i32,
            max_speech_tokens: usize_field("max_speech_tokens", defaults.max_speech_tokens)?,
            input_pos_emb: text("input_pos_emb", &defaults.input_pos_emb)?,
            speech_cond_prompt_len: usize_field(
                "speech_cond_prompt_len",
                defaults.speech_cond_prompt_len,
            )?,
            encoder_type: text("encoder_type", &defaults.encoder_type)?,
            speaker_embed_size: usize_field("speaker_embed_size", defaults.speaker_embed_size)?,
            use_perceiver_resampler: boolean(
                "use_perceiver_resampler",
                defaults.use_perceiver_resampler,
            )?,
            emotion_adv: boolean("emotion_adv", defaults.emotion_adv)?,
            llama,
        })
    }
}

/// T3 conditioning (`t3/cond_enc.py::T3Cond`).
#[derive(Debug, Clone)]
pub struct T3Cond {
    /// Speaker embedding, `[1, speaker_embed_size]`.
    pub speaker_emb: Vec<f32>,
    /// Speech prompt token ids, `[1, T]` flattened; `None` disables the
    /// perceiver branch.
    pub cond_prompt_speech_tokens: Option<Vec<i32>>,
    /// Emotion exaggeration scalar (0-1; the reference default 0.5).
    pub emotion_adv: f32,
}

/// Learned position embeddings
/// (`t3/learned_pos_emb.py::LearnedPositionEmbeddings`).
#[derive(Debug, Clone)]
struct LearnedPosEmb {
    emb: Vec<f32>,
    seq_len: usize,
    dim: usize,
}

impl LearnedPosEmb {
    fn load(file: &SafetensorsFile, prefix: &str, seq_len: usize, dim: usize) -> Result<Self> {
        Ok(LearnedPosEmb {
            emb: load_tensor(file, &format!("{prefix}.emb.weight"), &[seq_len, dim])?,
            seq_len,
            dim,
        })
    }

    /// `get_fixed_embedding(idx)`: row `idx` of the table.
    fn at(&self, index: usize) -> Result<Vec<f32>> {
        if index >= self.seq_len {
            return Err(SpeechError::Input {
                why: format!(
                    "position {index} exceeds the {}-row position table",
                    self.seq_len
                ),
            });
        }
        Ok(self.emb[index * self.dim..(index + 1) * self.dim].to_vec())
    }
}

/// The perceiver resampler (`t3/perceiver.py::Perceiver`): 32 learned
/// query tokens cross-attend the variable-length speech prompt, then
/// self-attend, through one shared attention block.
#[derive(Debug, Clone)]
struct Perceiver {
    query: Vec<f32>,
    query_tokens: usize,
    attn: AttentionBlock,
}

impl Perceiver {
    fn load(file: &SafetensorsFile, prefix: &str, dim: usize) -> Result<Self> {
        const QUERY_TOKENS: usize = 32;
        Ok(Perceiver {
            query: load_tensor(
                file,
                &format!("{prefix}.pre_attention_query"),
                &[1, QUERY_TOKENS, dim],
            )?,
            query_tokens: QUERY_TOKENS,
            attn: AttentionBlock::load(file, &format!("{prefix}.attn"), dim)?,
        })
    }

    /// Rows `[T, dim]` -> rows `[32, dim]`.
    fn forward(&self, h: &[f32], t: usize) -> Vec<f32> {
        let query = self.query.clone();
        let pre_att = self.attn.forward(&query, self.query_tokens, h, t);
        self.attn
            .forward(&pre_att, self.query_tokens, &pre_att, self.query_tokens)
    }
}

/// `AttentionBlock` + `AttentionQKV` (`t3/perceiver.py`): LayerNorm on
/// both inputs, biased q/k/v/out projections, 4-head softmax attention.
#[derive(Debug, Clone)]
struct AttentionBlock {
    norm: SaLayerNorm,
    to_q: SaLinear,
    to_k: SaLinear,
    to_v: SaLinear,
    proj_out: SaLinear,
    heads: usize,
    head_dim: usize,
    dim: usize,
}

const PERCEIVER_HEADS: usize = 4;

impl AttentionBlock {
    fn load(file: &SafetensorsFile, prefix: &str, dim: usize) -> Result<Self> {
        Ok(AttentionBlock {
            norm: SaLayerNorm::load(file, &format!("{prefix}.norm"), dim)?,
            to_q: SaLinear::load(file, &format!("{prefix}.to_q"), dim, dim)?,
            to_k: SaLinear::load(file, &format!("{prefix}.to_k"), dim, dim)?,
            to_v: SaLinear::load(file, &format!("{prefix}.to_v"), dim, dim)?,
            proj_out: SaLinear::load(file, &format!("{prefix}.proj_out"), dim, dim)?,
            heads: PERCEIVER_HEADS,
            head_dim: dim / PERCEIVER_HEADS,
            dim,
        })
    }

    /// Cross-attention of `x1` `[T1, dim]` onto `x2` `[T2, dim]`;
    /// returns `x1 + proj(attention)` rows `[T1, dim]`.
    fn forward(&self, x1: &[f32], t1: usize, x2: &[f32], t2: usize) -> Vec<f32> {
        let mut q_norm = x1.to_vec();
        self.norm.forward(&mut q_norm, t1, 1e-5);
        let mut k_norm = x2.to_vec();
        self.norm.forward(&mut k_norm, t2, 1e-5);
        let q = self.to_q.forward(&q_norm, t1);
        let k = self.to_k.forward(&k_norm, t2);
        let v = self.to_v.forward(&k_norm, t2);
        let scale = (self.head_dim as f32).powf(-0.5);
        let mut out = vec![0.0f32; t1 * self.dim];
        let mut scores = vec![0.0f32; t2];
        for h in 0..self.heads {
            let off = h * self.head_dim;
            for i in 0..t1 {
                for (j, s) in scores.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for d in 0..self.head_dim {
                        acc += q[i * self.dim + off + d] * k[j * self.dim + off + d];
                    }
                    *s = acc * scale;
                }
                crate::ops::softmax_row(&mut scores);
                for (j, &s) in scores.iter().enumerate() {
                    for d in 0..self.head_dim {
                        out[i * self.dim + off + d] += s * v[j * self.dim + off + d];
                    }
                }
            }
        }
        let projected = self.proj_out.forward(&out, t1);
        let mut result = x1.to_vec();
        for (r, p) in result.iter_mut().zip(&projected) {
            *r += p;
        }
        result
    }
}

/// The conditioning encoder (`t3/cond_enc.py::T3CondEnc`): speaker
/// projection, perceiver resampler, emotion scalar.
#[derive(Debug, Clone)]
struct T3CondEnc {
    spkr_enc: SaLinear,
    emotion_adv_fc: Option<SaLinear>,
    perceiver: Option<Perceiver>,
}

impl T3CondEnc {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        config: &T3Config,
        n_channels: usize,
    ) -> Result<Self> {
        if config.encoder_type != "voice_encoder" {
            return Err(SpeechError::Unsupported {
                why: format!("encoder_type '{}' is not ported", config.encoder_type),
            });
        }
        Ok(T3CondEnc {
            spkr_enc: SaLinear::load(
                file,
                &format!("{prefix}.spkr_enc"),
                config.speaker_embed_size,
                n_channels,
            )?,
            emotion_adv_fc: if config.emotion_adv {
                Some(SaLinear::load_no_bias(
                    file,
                    &format!("{prefix}.emotion_adv_fc"),
                    1,
                    n_channels,
                )?)
            } else {
                None
            },
            perceiver: if config.use_perceiver_resampler {
                Some(Perceiver::load(
                    file,
                    &format!("{prefix}.perceiver"),
                    n_channels,
                )?)
            } else {
                None
            },
        })
    }
}

/// Generation options with the reference `generate` defaults.
#[derive(Debug, Clone)]
pub struct T3GenParams {
    pub max_new_tokens: usize,
    /// 0.0 selects the exact greedy path (`make_sampler(0, ...)`).
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    /// Classifier-free guidance weight; 0 disables the second row.
    pub cfg_weight: f32,
}

impl Default for T3GenParams {
    fn default() -> Self {
        T3GenParams {
            max_new_tokens: 1000,
            temperature: 0.8,
            top_p: 1.0,
            min_p: 0.05,
            repetition_penalty: 1.2,
            cfg_weight: 0.5,
        }
    }
}

/// Why the generate loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    StopSpeechToken,
    MaxTokens,
}

/// The loaded T3 generator.
pub struct T3 {
    pub config: T3Config,
    /// Backbone hidden width (`n_channels`).
    dim: usize,
    backbone: lm::Backbone,
    text_emb: Vec<f32>,
    speech_emb: Vec<f32>,
    text_pos_emb: Option<LearnedPosEmb>,
    speech_pos_emb: Option<LearnedPosEmb>,
    speech_head: Vec<f32>,
    cond_enc: T3CondEnc,
}

impl T3 {
    /// Loads the `t3.` subtree from one checkpoint file. `llama` must be
    /// the resolved dense config for the backbone (the T3 dict's llama
    /// configuration parsed through the shared llama module).
    pub fn load(
        weights_path: &std::path::Path,
        file: &SafetensorsFile,
        config: &T3Config,
    ) -> Result<Self> {
        let mut llama_config = lm::llama::from_json(&config.llama)?;
        let dim = llama_config.hidden_size;
        // The checkpoint stores the backbone under `t3.tfmr.model.`; the
        // T3 never reads the llama token embedding table (inputs are
        // precomputed embeddings) and never applies the llama head, so
        // the backbone loads without both (the tied-head flag keeps the
        // loader from demanding an `lm_head` tensor).
        llama_config.tie_word_embeddings = true;
        let weights = lm::Weights::open_single(weights_path)?.with_prefix("t3.tfmr.model.");
        let backbone = lm::Backbone::load_without_embed_table(&weights, llama_config)?;
        let text_emb = load_tensor(
            file,
            "t3.text_emb.weight",
            &[config.text_tokens_dict_size, dim],
        )?;
        let speech_emb = load_tensor(
            file,
            "t3.speech_emb.weight",
            &[config.speech_tokens_dict_size, dim],
        )?;
        let text_pos_emb = if config.input_pos_emb == "learned" {
            Some(LearnedPosEmb::load(
                file,
                "t3.text_pos_emb",
                config.max_text_tokens + 2,
                dim,
            )?)
        } else {
            None
        };
        let speech_pos_emb = if config.input_pos_emb == "learned" {
            Some(LearnedPosEmb::load(
                file,
                "t3.speech_pos_emb",
                config.max_speech_tokens + 2 + 2,
                dim,
            )?)
        } else {
            None
        };
        let speech_head = load_tensor(
            file,
            "t3.speech_head.weight",
            &[config.speech_tokens_dict_size, dim],
        )?;
        let cond_enc = T3CondEnc::load(file, "t3.cond_enc", config, dim)?;
        Ok(T3 {
            config: config.clone(),
            dim,
            backbone,
            text_emb,
            speech_emb,
            text_pos_emb,
            speech_pos_emb,
            speech_head,
            cond_enc,
        })
    }

    /// `prepare_conditioning`: rows `[len_cond, dim]` (speaker 1,
    /// perceiver 32, emotion 1).
    pub fn prepare_conditioning(&mut self, cond: &T3Cond) -> Result<Vec<f32>> {
        if cond.speaker_emb.len() != self.config.speaker_embed_size {
            return Err(SpeechError::Input {
                why: format!(
                    "speaker embedding has {} dims, expected {}",
                    cond.speaker_emb.len(),
                    self.config.speaker_embed_size
                ),
            });
        }
        let n_channels = self.dim;
        let mut rows = vec![0.0f32; n_channels * (1 + 32 + 1)];
        // Speaker projection (1 row).
        let cond_spkr = self.cond_enc.spkr_enc.forward(&cond.speaker_emb, 1);
        rows[..n_channels].copy_from_slice(&cond_spkr);
        // Perceiver resampled speech prompt (32 rows).
        if let Some(tokens) = &cond.cond_prompt_speech_tokens {
            let perceiver = self
                .cond_enc
                .perceiver
                .as_ref()
                .ok_or_else(|| SpeechError::Input {
                    why: "speech prompt tokens arrived but the perceiver is disabled".to_string(),
                })?;
            if tokens.is_empty() {
                return Err(SpeechError::Input {
                    why: "empty speech prompt tokens".to_string(),
                });
            }
            let speech_pos = self
                .speech_pos_emb
                .as_ref()
                .ok_or_else(|| SpeechError::Input {
                    why: "learned speech positions are required".to_string(),
                })?;
            let mut emb = vec![0.0f32; tokens.len() * n_channels];
            for (i, &token) in tokens.iter().enumerate() {
                let idx = usize::try_from(token).map_err(|_| SpeechError::Input {
                    why: format!("negative speech token {token}"),
                })?;
                if idx >= self.config.speech_tokens_dict_size {
                    return Err(SpeechError::Input {
                        why: format!("speech token {token} outside the embedding table"),
                    });
                }
                let base = i * n_channels;
                emb[base..base + n_channels]
                    .copy_from_slice(&self.speech_emb[idx * n_channels..(idx + 1) * n_channels]);
                let pos = speech_pos.at(i)?;
                for (c, p) in emb[base..base + n_channels].iter_mut().zip(&pos) {
                    *c += p;
                }
            }
            let resampled = perceiver.forward(&emb, tokens.len());
            rows[n_channels..33 * n_channels].copy_from_slice(&resampled);
        }
        // Emotion scalar (1 row).
        if let Some(fc) = &self.cond_enc.emotion_adv_fc {
            let projected = fc.forward(&[cond.emotion_adv], 1);
            rows[33 * n_channels..34 * n_channels].copy_from_slice(&projected);
        }
        Ok(rows)
    }

    /// Assembles the composite prompt embeddings
    /// (`prepare_input_embeds` + `_prepare_inference_context`): rows
    /// `[conditioning | text(+pos) | BOS(+pos 0) | BOS(+pos 0)]`. When
    /// `cfg_weight > 0` a second row is returned whose text half is the
    /// position embeddings only - the reference zeroes the text
    /// embedding BEFORE adding positions, so the unconditional half
    /// keeps the learned positions.
    fn prepare_inference_context(
        &self,
        cond_rows: &[f32],
        text_tokens: &[i32],
        len_cond: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let dim = self.dim;
        let text_len = text_tokens.len();
        let prompt_len = len_cond + text_len + 2;
        let mut row = vec![0.0f32; prompt_len * dim];
        let mut uncond = vec![0.0f32; prompt_len * dim];
        row[..len_cond * dim].copy_from_slice(cond_rows);
        uncond[..len_cond * dim].copy_from_slice(cond_rows);
        let text_pos = self
            .text_pos_emb
            .as_ref()
            .ok_or_else(|| SpeechError::Input {
                why: format!(
                    "input_pos_emb '{}' requires the learned text positions",
                    self.config.input_pos_emb
                ),
            })?;
        for (i, &token) in text_tokens.iter().enumerate() {
            let idx = usize::try_from(token).map_err(|_| SpeechError::Input {
                why: format!("negative text token {token}"),
            })?;
            if idx >= self.config.text_tokens_dict_size {
                return Err(SpeechError::Input {
                    why: format!("text token {token} outside the embedding table"),
                });
            }
            let pos = text_pos.at(i)?;
            // Conditional half: embedding + position.
            let base = (len_cond + i) * dim;
            row[base..base + dim].copy_from_slice(&self.text_emb[idx * dim..(idx + 1) * dim]);
            for (c, p) in row[base..base + dim].iter_mut().zip(&pos) {
                *c += p;
            }
            // Unconditional half: position only (zeroed embedding).
            uncond[base..base + dim].copy_from_slice(&pos);
        }
        // Two BOS rows, each the start speech embedding plus position 0.
        let bos = self.speech_token_embed(self.config.start_speech_token, 0)?;
        for b in 0..2 {
            let base = (len_cond + text_len + b) * dim;
            row[base..base + dim].copy_from_slice(&bos);
            uncond[base..base + dim].copy_from_slice(&bos);
        }
        Ok((row, uncond))
    }

    /// Speech embedding row `token` plus speech position `pos`.
    fn speech_token_embed(&self, token: i32, pos: usize) -> Result<Vec<f32>> {
        let dim = self.dim;
        let idx = usize::try_from(token).map_err(|_| SpeechError::Input {
            why: format!("negative speech token {token}"),
        })?;
        if idx >= self.config.speech_tokens_dict_size {
            return Err(SpeechError::Input {
                why: format!("speech token {token} outside the embedding table"),
            });
        }
        let mut out = self.speech_emb[idx * dim..(idx + 1) * dim].to_vec();
        let speech_pos = self
            .speech_pos_emb
            .as_ref()
            .ok_or_else(|| SpeechError::Input {
                why: "learned speech positions are required".to_string(),
            })?;
        let pos_row = speech_pos.at(pos)?;
        for (c, p) in out.iter_mut().zip(&pos_row) {
            *c += p;
        }
        Ok(out)
    }

    /// The speech head over one hidden row.
    fn speech_logits(&self, hidden_row: &[f32]) -> Vec<f32> {
        crate::ops::linear(
            hidden_row,
            &self.speech_head,
            None,
            1,
            self.dim,
            self.config.speech_tokens_dict_size,
        )
    }

    /// The generate loop (`t3.py::inference`). Returns the generated id
    /// stream including the leading start token and, when hit, the
    /// trailing stop token, plus why the loop stopped. `keys` feeds the
    /// categorical draws; temperature 0 never draws.
    pub fn inference(
        &mut self,
        cond: &T3Cond,
        text_tokens: &[i32],
        params: &T3GenParams,
        keys: &mut rng::KeySequence,
    ) -> Result<(Vec<i32>, StopReason)> {
        if text_tokens.is_empty() {
            return Err(SpeechError::Input {
                why: "empty text token prompt".to_string(),
            });
        }
        let cfg = params.cfg_weight;
        let cond_rows = self.prepare_conditioning(cond)?;
        let len_cond = cond_rows.len() / self.dim;
        let (row, uncond_row) =
            self.prepare_inference_context(&cond_rows, text_tokens, len_cond)?;
        let use_cfg = cfg > 0.0;

        let mut caches = lm::make_prompt_cache(self.backbone.config.num_hidden_layers);
        let mut uncond_caches = lm::make_prompt_cache(self.backbone.config.num_hidden_layers);
        let hidden = self
            .backbone
            .forward_hidden(Input::Embeddings(&row), Some(&mut caches))?;
        let uncond_hidden = if use_cfg {
            self.backbone
                .forward_hidden(Input::Embeddings(&uncond_row), Some(&mut uncond_caches))?
        } else {
            Vec::new()
        };
        let dim = self.dim;
        let mut cond_logits = self.speech_logits(&hidden[hidden.len() - dim..]);
        let mut uncond_logits = if use_cfg {
            self.speech_logits(&uncond_hidden[uncond_hidden.len() - dim..])
        } else {
            Vec::new()
        };

        let mut generated: Vec<i32> = vec![self.config.start_speech_token];
        let mut stopped_by = StopReason::MaxTokens;
        for _step in 0..params.max_new_tokens {
            let mut logits = if use_cfg {
                cond_logits
                    .iter()
                    .zip(&uncond_logits)
                    .map(|(&c, &u)| c + cfg * (c - u))
                    .collect::<Vec<f32>>()
            } else {
                std::mem::take(&mut cond_logits)
            };
            // The reference repetition penalty sees every generated id
            // (the context size is the full generation cap).
            if params.repetition_penalty != 0.0 {
                sampling::apply_repetition_penalty(
                    &mut logits,
                    &generated,
                    params.repetition_penalty,
                    params.max_new_tokens,
                )?;
            }
            let next: usize = if params.temperature == 0.0 {
                sampling::argmax(&logits)
            } else {
                let mut filtered = logits;
                if params.top_p > 0.0 && params.top_p < 1.0 {
                    sampling::apply_top_p(&mut filtered, params.top_p);
                }
                if params.min_p != 0.0 {
                    sampling::apply_min_p(&mut filtered, params.min_p, 1)?;
                }
                sampling::sample_categorical(&filtered, params.temperature, keys.next())
            };
            let next = i32::try_from(next).map_err(|_| SpeechError::Input {
                why: format!("sampled token {next} does not fit i32"),
            })?;
            generated.push(next);
            if next == self.config.stop_speech_token {
                stopped_by = StopReason::StopSpeechToken;
                break;
            }
            // Embed the sampled token at decode position step + 1 and
            // step both cache chains.
            let pos_index = generated.len() - 1;
            let embed = self.speech_token_embed(next, pos_index)?;
            let hidden = self
                .backbone
                .forward_hidden(Input::Embeddings(&embed), Some(&mut caches))?;
            cond_logits = self.speech_logits(&hidden[hidden.len() - dim..]);
            if use_cfg {
                let uncond_hidden = self
                    .backbone
                    .forward_hidden(Input::Embeddings(&embed), Some(&mut uncond_caches))?;
                uncond_logits = self.speech_logits(&uncond_hidden[uncond_hidden.len() - dim..]);
            }
        }
        Ok((generated, stopped_by))
    }
}
