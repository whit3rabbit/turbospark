//! The chatterbox_turbo T3 token generator.
//!
//! Reference: `chatterbox_turbo/models/t3/` (`t3.py`, `cond_enc.py`,
//! `t3_config.py`) and `t3/gpt2.py` (the GPT-2 medium backbone the
//! turbo generator rides), all at the pinned commit recorded in the
//! parent module.
//!
//! Structure: composite embeddings `[conditioning | text |
//! start_speech_token]` feed a GPT-2 backbone whose learned position
//! table is added inside the forward from the cache offset (the T3 has
//! no per-token position embeddings of its own: `input_pos_emb` is
//! `None`). The conditioning encoder is a single speaker projection:
//! turbo drops the sibling family's perceiver resampler and emotion
//! scalar (`use_perceiver_resampler = False`, `emotion_adv = False`).
//! Text and speech embedding tables and a dedicated speech head (with
//! bias) carry the token spaces; the backbone's own token table
//! (`tfmr.wte`) and text head are never read by the forward, so the
//! loader skips both (real installs save ~200 MB of dead weight).
//!
//! Generation loop (`inference_turbo`): no classifier-free guidance
//! (unlike the sibling family). Per step: speech-head logits on the
//! last position, the sign-aware repetition penalty over the UNIQUE
//! generated ids (`np.unique` before the gather), the temperature
//! divide, the top-k filter that keeps every logit at or above the
//! k-th largest value (ties survive), and the descending top-p filter
//! that masks where the sorted inclusive cumulative probability
//! exceeds `top_p` shifted one right (the head of the nucleus always
//! survives). The reference then draws through `mx.random.categorical`
//! on the still-unnormalized filtered logits; the Rust draw uses the
//! crate-local MLX-compatible threefry RNG, so a seeded Python stream
//! replays exactly. `temperature == 0` is the explicit greedy mode:
//! the plain post-penalty argmax (top-k and top-p never mask the
//! maximum, so the filters cannot move it); the reference itself has
//! no greedy path and always samples.

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::wnconv::load_f32_shaped;
use crate::lm;
use crate::lm::engine::Input;
use crate::music::minimax_music3::rng;
use crate::tts::chatterbox::s3gen::SaLinear;
use crate::tts::sampling;
use crate::{Result, SpeechError};

/// Pinned turbo T3 defaults (`t3/t3_config.py::T3Config.turbo`).
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
    pub speech_cond_prompt_len: usize,
    pub encoder_type: String,
    pub speaker_embed_size: usize,
    pub use_perceiver_resampler: bool,
    pub emotion_adv: bool,
    /// The GPT-2 backbone configuration (`create_gpt2_config()` at the
    /// reference geometry; the fixture injects a tiny backbone here).
    pub gpt2: Value,
}

impl Default for T3Config {
    fn default() -> Self {
        T3Config {
            text_tokens_dict_size: 50_276,
            start_text_token: 255,
            stop_text_token: 0,
            max_text_tokens: 2048,
            speech_tokens_dict_size: 6563,
            start_speech_token: 6561,
            stop_speech_token: 6562,
            max_speech_tokens: 4096,
            speech_cond_prompt_len: 375,
            encoder_type: "voice_encoder".to_string(),
            speaker_embed_size: 256,
            use_perceiver_resampler: false,
            emotion_adv: false,
            gpt2: gpt2_medium_config(),
        }
    }
}

/// The pinned GPT-2 medium configuration (`t3/gpt2.py::
/// create_gpt2_config`); `n_inner` defaults to `4 * n_embd`.
pub fn gpt2_medium_config() -> Value {
    serde_json::json!({
        "model_type": "gpt2",
        "vocab_size": 50276,
        "n_ctx": 8196,
        "n_positions": 8196,
        "n_embd": 1024,
        "n_layer": 24,
        "n_head": 16,
        "activation_function": "gelu_new",
        "layer_norm_epsilon": 1e-5,
    })
}

impl T3Config {
    /// Parses the reference `T3Config` fields; missing or null fields
    /// keep the defaults. The GPT-2 configuration comes from the
    /// `gpt2_config` key when present.
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
        let gpt2 = match value.get("gpt2_config") {
            None | Some(Value::Null) => defaults.gpt2.clone(),
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
            gpt2,
        })
    }
}

/// T3 conditioning (`t3/cond_enc.py::T3Cond`); the CLAP, perceiver, and
/// emotion fields the dataclass carries are all unused by turbo.
#[derive(Debug, Clone)]
pub struct T3Cond {
    /// Speaker embedding, `[1, speaker_embed_size]`.
    pub speaker_emb: Vec<f32>,
    /// Speech prompt token ids, `[1, T]` flattened.
    pub cond_prompt_speech_tokens: Option<Vec<i32>>,
}

/// Generation options with the reference `generate` defaults
/// (`chatterbox_turbo.generate` passes these into `inference_turbo`).
#[derive(Debug, Clone)]
pub struct T3GenParams {
    pub max_gen_len: usize,
    /// 0.0 selects the explicit greedy path (post-penalty argmax).
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
}

impl Default for T3GenParams {
    fn default() -> Self {
        T3GenParams {
            max_gen_len: 800,
            temperature: 0.8,
            top_k: 1000,
            top_p: 0.95,
            repetition_penalty: 1.2,
        }
    }
}

/// The sign-aware repetition penalty over the unique generated ids
/// (`t3.py::_apply_repetition_penalty`): `np.unique` first, so every
/// token is penalized once regardless of how often it repeats, and
/// tokens outside the vocabulary are skipped.
pub fn apply_repetition_penalty(logits: &mut [f32], generated: &[i32], penalty: f32) {
    if penalty == 1.0 || generated.is_empty() {
        return;
    }
    let mut seen: Vec<usize> = Vec::new();
    for &token in generated {
        if let Ok(index) = usize::try_from(token) {
            if index < logits.len() && !seen.contains(&index) {
                seen.push(index);
            }
        }
    }
    for index in seen {
        logits[index] = if logits[index] < 0.0 {
            logits[index] * penalty
        } else {
            logits[index] / penalty
        };
    }
}

/// Top-k filter (`t3.py::_top_k_filtering`): keep every logit at or
/// above the k-th largest value; ties at the threshold survive, so the
/// kept set can exceed `k`. The negated comparisons mirror the
/// reference `mx.where` NaN handling exactly (NaN compares false and is
/// masked), which clippy would otherwise flag.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn apply_top_k_threshold(logits: &mut [f32], top_k: usize) {
    if top_k == 0 {
        return;
    }
    let k = top_k.min(logits.len());
    let mut sorted = logits.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let kth = sorted[k - 1];
    for v in logits.iter_mut() {
        // `mx.where(logits >= kth, logits, -inf)` negated: NaN compares
        // false, so this masks it like the reference does.
        if !(*v >= kth) {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Top-p filter (`t3.py::_top_p_filtering`): sort descending, softmax
/// over the sorted logits, mask where the inclusive cumulative
/// probability exceeds `top_p`, shift the mask one right (the nucleus
/// head always survives), and scatter back. The negated comparison
/// mirrors the reference `mx.where` NaN handling exactly.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn apply_top_p_shift(logits: &mut [f32], top_p: f32) {
    if top_p >= 1.0 {
        return;
    }
    let n = logits.len();
    let mut order: Vec<usize> = (0..n).collect();
    // Descending value order; the fixture vectors have no ties, which
    // is also where the reference argsort tie order is unspecified.
    order.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // Softmax over the sorted (post-top-k) logits; masked entries
    // contribute zero probability.
    let mut total = 0.0f32;
    let mut exps = vec![0.0f32; n];
    for &index in &order {
        let e = (logits[index] - max).exp();
        exps[index] = e;
        total += e;
    }
    let mut cumulative = 0.0f32;
    let mut keep = vec![true; n];
    for (k, &index) in order.iter().enumerate() {
        if k > 0 {
            // `mx.where(cumulative > top_p, -inf, sorted)` negated: a
            // NaN cumulative compares false and keeps the entry, as in
            // the reference.
            keep[index] = !(cumulative > top_p);
        }
        cumulative += exps[index] / total;
    }
    for (v, &k) in logits.iter_mut().zip(&keep) {
        if !k {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// The loaded T3 generator.
pub struct T3 {
    pub config: T3Config,
    /// Backbone hidden width (`n_embd`).
    dim: usize,
    backbone: lm::Gpt2Model,
    text_emb: Vec<f32>,
    speech_emb: Vec<f32>,
    speech_head_weight: Vec<f32>,
    speech_head_bias: Vec<f32>,
    /// The conditioning encoder (`cond_enc.spkr_enc`); turbo builds no
    /// perceiver and no emotion head.
    spkr_enc: SaLinear,
}

impl T3 {
    /// Loads the `t3.` subtree from one checkpoint file. `gpt2` must be
    /// the resolved backbone configuration (the T3 dict's gpt2
    /// configuration parsed through the shared gpt2 module).
    pub fn load(
        weights_path: &std::path::Path,
        file: &SafetensorsFile,
        config: &T3Config,
    ) -> Result<Self> {
        if config.encoder_type != "voice_encoder" {
            return Err(SpeechError::Unsupported {
                why: format!("encoder_type '{}' is not ported", config.encoder_type),
            });
        }
        let gpt2_config = lm::gpt2::from_json(&config.gpt2)?;
        let dim = gpt2_config.n_embd;
        // The checkpoint stores the backbone under `t3.tfmr.`. The T3
        // never reads the gpt2 token table (inputs are precomputed
        // embeddings) but does consume the learned positions.
        let weights = lm::Weights::open_single(weights_path)?.with_prefix("t3.tfmr.");
        let backbone =
            lm::Gpt2Model::from_config_without_tables_at_root(&weights, gpt2_config, "")?
                .with_position_table_at_root(&weights, "")?;
        let text_emb = load_f32_shaped(
            file,
            "t3.text_emb.weight",
            &[config.text_tokens_dict_size, dim],
        )?;
        let speech_emb = load_f32_shaped(
            file,
            "t3.speech_emb.weight",
            &[config.speech_tokens_dict_size, dim],
        )?;
        let speech_head_weight = load_f32_shaped(
            file,
            "t3.speech_head.weight",
            &[config.speech_tokens_dict_size, dim],
        )?;
        let speech_head_bias = load_f32_shaped(
            file,
            "t3.speech_head.bias",
            &[config.speech_tokens_dict_size],
        )?;
        let spkr_enc =
            SaLinear::load(file, "t3.cond_enc.spkr_enc", config.speaker_embed_size, dim)?;
        Ok(T3 {
            config: config.clone(),
            dim,
            backbone,
            text_emb,
            speech_emb,
            speech_head_weight,
            speech_head_bias,
            spkr_enc,
        })
    }

    /// `prepare_conditioning`: rows `[1 + T_prompt, dim]` (speaker
    /// projection, then the embedded speech prompt; no perceiver, no
    /// emotion row).
    pub fn prepare_conditioning(&self, cond: &T3Cond) -> Result<Vec<f32>> {
        if cond.speaker_emb.len() != self.config.speaker_embed_size {
            return Err(SpeechError::Input {
                why: format!(
                    "speaker embedding has {} dims, expected {}",
                    cond.speaker_emb.len(),
                    self.config.speaker_embed_size
                ),
            });
        }
        let dim = self.dim;
        let prompt = cond.cond_prompt_speech_tokens.as_deref().unwrap_or(&[]);
        let mut rows = vec![0.0f32; (1 + prompt.len()) * dim];
        let spkr = self.spkr_enc.forward(&cond.speaker_emb, 1);
        rows[..dim].copy_from_slice(&spkr);
        for (i, &token) in prompt.iter().enumerate() {
            let index = self.speech_token_index(token)?;
            let base = (1 + i) * dim;
            rows[base..base + dim]
                .copy_from_slice(&self.speech_emb[index * dim..(index + 1) * dim]);
        }
        Ok(rows)
    }

    fn speech_token_index(&self, token: i32) -> Result<usize> {
        let index = usize::try_from(token).map_err(|_| SpeechError::Input {
            why: format!("negative speech token {token}"),
        })?;
        if index >= self.config.speech_tokens_dict_size {
            return Err(SpeechError::Input {
                why: format!("speech token {token} outside the embedding table"),
            });
        }
        Ok(index)
    }

    fn text_token_index(&self, token: i32) -> Result<usize> {
        let index = usize::try_from(token).map_err(|_| SpeechError::Input {
            why: format!("negative text token {token}"),
        })?;
        if index >= self.config.text_tokens_dict_size {
            return Err(SpeechError::Input {
                why: format!("text token {token} outside the embedding table"),
            });
        }
        Ok(index)
    }

    /// `prepare_input_embeds`: rows `[conditioning | text |
    /// start_speech_token]`. No learned positions are added here; the
    /// backbone adds its own from the cache offset.
    pub(crate) fn prepare_input_embeds(
        &self,
        cond_rows: &[f32],
        text_tokens: &[i32],
    ) -> Result<Vec<f32>> {
        let dim = self.dim;
        let len_cond = cond_rows.len() / dim;
        let mut embeds = vec![0.0f32; (len_cond + text_tokens.len() + 1) * dim];
        embeds[..cond_rows.len()].copy_from_slice(cond_rows);
        for (i, &token) in text_tokens.iter().enumerate() {
            let index = self.text_token_index(token)?;
            let base = (len_cond + i) * dim;
            embeds[base..base + dim]
                .copy_from_slice(&self.text_emb[index * dim..(index + 1) * dim]);
        }
        let start = self.speech_token_index(self.config.start_speech_token)?;
        let base = (len_cond + text_tokens.len()) * dim;
        embeds[base..base + dim].copy_from_slice(&self.speech_emb[start * dim..(start + 1) * dim]);
        Ok(embeds)
    }

    /// The speech head over one hidden row (the head carries a bias,
    /// unlike the sibling family's).
    fn speech_logits(&self, hidden_row: &[f32]) -> Vec<f32> {
        crate::ops::linear(
            hidden_row,
            &self.speech_head_weight,
            Some(&self.speech_head_bias),
            1,
            self.dim,
            self.config.speech_tokens_dict_size,
        )
    }

    /// `inference_turbo`: generate speech tokens from text tokens.
    /// Returns the sampled id stream with the trailing stop token
    /// removed when hit (the leading start token is an input, never a
    /// generated id). `keys` feeds the categorical draws; greedy mode
    /// (`temperature == 0`) never draws.
    pub fn inference_turbo(
        &mut self,
        cond: &T3Cond,
        text_tokens: &[i32],
        params: &T3GenParams,
        keys: &mut rng::KeySequence,
    ) -> Result<Vec<i32>> {
        if text_tokens.is_empty() {
            return Err(SpeechError::Input {
                why: "empty text token prompt".to_string(),
            });
        }
        let dim = self.dim;
        let cond_rows = self.prepare_conditioning(cond)?;
        let embeds = self.prepare_input_embeds(&cond_rows, text_tokens)?;

        let mut caches = lm::make_prompt_cache(self.backbone.config.n_layer);
        let hidden = self
            .backbone
            .forward_hidden(Input::Embeddings(&embeds), Some(&mut caches))?;
        let mut logits = self.speech_logits(&hidden[hidden.len() - dim..]);

        let mut generated: Vec<i32> = Vec::new();
        let mut next = self.sample(&logits, &generated, params, keys)?;
        generated.push(next);
        for _step in 0..params.max_gen_len {
            if next == self.config.stop_speech_token {
                break;
            }
            let index = self.speech_token_index(next)?;
            let embed = self.speech_emb[index * dim..(index + 1) * dim].to_vec();
            let hidden = self
                .backbone
                .forward_hidden(Input::Embeddings(&embed), Some(&mut caches))?;
            logits = self.speech_logits(&hidden[hidden.len() - dim..]);
            next = self.sample(&logits, &generated, params, keys)?;
            generated.push(next);
        }
        // The reference drops the trailing stop token after the loop.
        if generated.last() == Some(&self.config.stop_speech_token) {
            generated.pop();
        }
        Ok(generated)
    }

    /// `_sample_token`: repetition penalty, temperature divide, top-k,
    /// top-p, then the categorical draw (greedy mode takes the argmax).
    fn sample(
        &self,
        logits: &[f32],
        generated: &[i32],
        params: &T3GenParams,
        keys: &mut rng::KeySequence,
    ) -> Result<i32> {
        let mut filtered = logits.to_vec();
        if params.repetition_penalty != 1.0 && !generated.is_empty() {
            apply_repetition_penalty(&mut filtered, generated, params.repetition_penalty);
        }
        if params.temperature > 0.0 && params.temperature != 1.0 {
            for v in &mut filtered {
                *v /= params.temperature;
            }
        }
        if params.top_k > 0 {
            apply_top_k_threshold(&mut filtered, params.top_k);
        }
        if params.top_p < 1.0 {
            apply_top_p_shift(&mut filtered, params.top_p);
        }
        let next = if params.temperature == 0.0 {
            sampling::argmax(&filtered)
        } else {
            rng::categorical(&filtered, keys.next())
        };
        i32::try_from(next).map_err(|_| SpeechError::Input {
            why: format!("sampled token {next} does not fit i32"),
        })
    }
}
