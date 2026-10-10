//! Higgs Audio v3 TTS family (Boson AI).
//!
//! Port of `mlx_audio/tts/models/higgs_audio_v3/` (model.py, config.py,
//! generation.py, prompt.py) at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/higgs_audio_v3):
//! a qwen3 backbone (shared [`crate::lm`] port) whose every frame is one
//! codebook vector, a single fused `multimodal_embedding` table that is
//! BOTH the audio input embedding (codebook k rides rows
//! `k * codebook_size + code`) and the audio logits head (the tied
//! `as_linear` view reshaped to `[K, V]`), and an SGLang-style delayed
//! multi-codebook sampler: codebook i starts i frames late (ramp-in tail
//! forced to BOC), the first codebook's EOC starts a fixed `K - 2` frame
//! ramp-out countdown, and the aligned diagonal decodes through the
//! Higgs audio codec ([`crate::codec::higgs_audio`]). See
//! [README.md](README.md) for the verification evidence and open items.
//!
//! Scope of this port, matching the reference single-prompt `generate`:
//! batch size 1, non-streaming decode (the whole waveform decodes after
//! generation; the `continuous_batching` batch engine is out of scope),
//! and prompt construction from pre-tokenized ids. Text tokenization is
//! a caller concern (the reference loads the checkpoint tokenizer.json
//! in `post_load_hook`); the prompt layout is exposed through
//! [`PromptBuilder`] so the caller tokenizes the pieces verbatim.

use serde_json::Value;

use crate::codec::higgs_audio::HiggsAudioTokenizer;
use crate::lm;
use crate::lm::engine::Input;
use crate::music::minimax_music3::rng;
use crate::tts::sampling;
use crate::tts::sampling::argmax;
use crate::{quant, Result, SpeechError};

/// The prompt placeholder id the audio segments leave in the token ids
/// (prompt.py `AUDIO_PLACEHOLDER_ID`).
pub const AUDIO_PLACEHOLDER_ID: i32 = -100;
/// Reference sample rate (the deployed codec emits 24 kHz mono).
pub const DEFAULT_SAMPLE_RATE: u32 = 24000;
/// The reference `_resolve_generation_limit` fallback frame cap.
pub const DEFAULT_MAX_NEW_FRAMES: usize = 2048;

/// The qwen3 backbone settings (`HiggsAudioV3TextConfig`), with the
/// reference dataclass defaults.
#[derive(Debug, Clone)]
pub struct TextConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    /// Parsed for config parity; the reference builds the inner
    /// `Qwen3Model` (whose forward ends at the final norm) and reads the
    /// audio logits from the fused embedding table, so no text head ever
    /// runs regardless of this flag.
    pub tie_word_embeddings: bool,
    pub rope_scaling: Option<Value>,
}

impl Default for TextConfig {
    fn default() -> Self {
        TextConfig {
            model_type: "qwen3".into(),
            hidden_size: 2560,
            num_hidden_layers: 36,
            intermediate_size: 9728,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            max_position_embeddings: 32768,
            rope_theta: 1_000_000.0,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            vocab_size: 151936,
            tie_word_embeddings: true,
            rope_scaling: None,
        }
    }
}

impl TextConfig {
    /// `HiggsAudioV3TextConfig.from_dict`: missing keys take the
    /// dataclass defaults; `rope_theta` falls back to the
    /// `rope_parameters.rope_theta` alias before the default.
    pub fn from_json(value: &Value) -> Self {
        let defaults = TextConfig::default();
        let data = match value {
            Value::Null => &Value::Null,
            other => other,
        };
        let u = |key: &str, fallback: usize| -> usize {
            data.get(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(fallback)
        };
        let rope_parameters = data.get("rope_parameters");
        TextConfig {
            model_type: data
                .get("model_type")
                .and_then(Value::as_str)
                .unwrap_or(&defaults.model_type)
                .to_owned(),
            hidden_size: u("hidden_size", defaults.hidden_size),
            num_hidden_layers: u("num_hidden_layers", defaults.num_hidden_layers),
            intermediate_size: u("intermediate_size", defaults.intermediate_size),
            num_attention_heads: u("num_attention_heads", defaults.num_attention_heads),
            num_key_value_heads: u("num_key_value_heads", defaults.num_key_value_heads),
            max_position_embeddings: u("max_position_embeddings", defaults.max_position_embeddings),
            rope_theta: data
                .get("rope_theta")
                .and_then(Value::as_f64)
                .or_else(|| {
                    rope_parameters
                        .and_then(|p| p.get("rope_theta"))
                        .and_then(Value::as_f64)
                })
                .unwrap_or(defaults.rope_theta as f64) as f32,
            head_dim: u("head_dim", defaults.head_dim),
            rms_norm_eps: data
                .get("rms_norm_eps")
                .and_then(Value::as_f64)
                .unwrap_or(defaults.rms_norm_eps as f64) as f32,
            vocab_size: u("vocab_size", defaults.vocab_size),
            tie_word_embeddings: data
                .get("tie_word_embeddings")
                .and_then(Value::as_bool)
                .unwrap_or(defaults.tie_word_embeddings),
            rope_scaling: data.get("rope_scaling").cloned().filter(|v| !v.is_null()),
        }
    }

    /// `to_qwen3_args` mapped onto the shared engine configuration. The
    /// reference feeds the inner `Qwen3Model`, whose only model-level
    /// knobs are these.
    pub fn to_dense_config(&self) -> lm::DenseConfig {
        lm::DenseConfig {
            model_type: self.model_type.clone(),
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            intermediate_size: self.intermediate_size,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            rms_norm_eps: self.rms_norm_eps,
            vocab_size: self.vocab_size,
            qk_norms: true,
            attention_bias: false,
            mlp_bias: false,
            rope_theta: self.rope_theta,
            rope_traditional: false,
            rope_scaling: self.rope_scaling.clone(),
            max_position_embeddings: Some(self.max_position_embeddings),
            tie_word_embeddings: self.tie_word_embeddings,
            sliding_window: None,
        }
    }
}

/// The top-level `HiggsAudioV3Config`.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub text_config: TextConfig,
    /// Parsed for config parity (default -100); the reference reads the
    /// placeholder from the prompt module constant, not this field.
    pub audio_token_id: i32,
    pub audio_num_codebooks: usize,
    pub audio_codebook_size: usize,
    /// Per-codebook begin-of-chunk sentinel (codebook_size - 2 on the
    /// defaults).
    pub audio_boc_token_id: i32,
    /// Per-codebook end-of-chunk sentinel (codebook_size - 1 on the
    /// defaults).
    pub audio_eoc_token_id: i32,
    /// Parsed for config parity; the reference prompt and decode paths
    /// always apply the delay pattern, so the flag is never read.
    pub use_delay_pattern: bool,
    pub sample_rate: u32,
}

impl ModelConfig {
    /// `HiggsAudioV3Config.from_dict` over a `config.json` dict. The
    /// `audio_encoder_config` block backs the codebook geometry, BOC/EOC
    /// ids, and delay flag when the top-level keys are absent.
    pub fn from_json(value: &Value) -> Result<Self> {
        let text_config = match value.get("text_config") {
            Some(text) => TextConfig::from_json(text),
            None => TextConfig::default(),
        };
        let enc = value
            .get("audio_encoder_config")
            .cloned()
            .unwrap_or(Value::Null);
        let codebook_size = value
            .get("audio_codebook_size")
            .and_then(Value::as_u64)
            .or_else(|| enc.get("vocab_size").and_then(Value::as_u64))
            .unwrap_or(1026) as usize;
        let config = ModelConfig {
            text_config,
            audio_token_id: value
                .get("audio_token_id")
                .and_then(Value::as_i64)
                .unwrap_or(-100) as i32,
            audio_num_codebooks: value
                .get("audio_num_codebooks")
                .and_then(Value::as_u64)
                .or_else(|| enc.get("num_codebooks").and_then(Value::as_u64))
                .unwrap_or(8) as usize,
            audio_codebook_size: codebook_size,
            audio_boc_token_id: value
                .get("audio_boc_token_id")
                .and_then(Value::as_i64)
                .unwrap_or(codebook_size as i64 - 2) as i32,
            audio_eoc_token_id: value
                .get("audio_eoc_token_id")
                .and_then(Value::as_i64)
                .unwrap_or(codebook_size as i64 - 1) as i32,
            use_delay_pattern: value
                .get("use_delay_pattern")
                .and_then(Value::as_bool)
                .or_else(|| enc.get("use_delay_pattern").and_then(Value::as_bool))
                .unwrap_or(true),
            sample_rate: value
                .get("sample_rate")
                .and_then(Value::as_u64)
                .unwrap_or(DEFAULT_SAMPLE_RATE as u64) as u32,
        };
        if config.audio_num_codebooks == 0 || config.audio_codebook_size == 0 {
            return Err(SpeechError::BadConfig {
                field: "audio_num_codebooks".into(),
                why: "audio_num_codebooks and audio_codebook_size must be positive".into(),
            });
        }
        Ok(config)
    }

    /// Rows of the fused embedding table per codebook (`codebook_size`):
    /// codebook k rides rows `k * codebook_size + code`, unlike the v2
    /// family whose stride carries the two stream sentinels. Here BOC and
    /// EOC are ordinary classes inside every codebook's vocabulary.
    pub fn codebook_stride(&self) -> usize {
        self.audio_codebook_size
    }
}

/// Sampling controls for one generation (the reference `generate`
/// defaults). The reference resolves the frame cap through
/// `max_new_tokens` -> `max_new_frames` -> `max_tokens` -> 2048; this
/// port takes the resolved value directly.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub max_new_frames: usize,
    /// 0.0 (or any value <= 1e-5, or `top_k == 1`) selects the exact
    /// greedy path (`sample_independent` argmax).
    pub temperature: f32,
    /// `None` or values outside (0, 1) disable the nucleus filter.
    pub top_p: Option<f32>,
    /// `None` or values <= 0 or >= the codebook width disable the top-k
    /// filter.
    pub top_k: Option<i32>,
    /// Linear fade over the decoded waveform edges (the reference
    /// `generate` defaults).
    pub fade_in_ms: f32,
    pub fade_out_ms: f32,
}

impl Default for GenerateParams {
    fn default() -> Self {
        GenerateParams {
            max_new_frames: DEFAULT_MAX_NEW_FRAMES,
            temperature: 1.0,
            top_p: None,
            top_k: None,
            fade_in_ms: 30.0,
            fade_out_ms: 15.0,
        }
    }
}

/// Why the generate loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The first codebook emitted EOC and the `K - 2` frame countdown
    /// ran out.
    EocRamp,
    /// The frame budget ran out.
    MaxSteps,
}

/// One generated utterance: the raw delayed frames, the aligned codec
/// codes, and the decoded mono waveform (faded).
#[derive(Debug, Clone)]
pub struct Generated {
    /// One entry per generated frame in delay-pattern form, each
    /// `audio_num_codebooks` codes (time-major, the reference
    /// `delayed_rows`).
    pub frames: Vec<Vec<i32>>,
    /// The delay-reversed diagonal `[T][K]`, time-major like the
    /// reference `reverse_delay_pattern` output; the codec decode input
    /// (the Rust codec consumes the transpose internally).
    pub codes: Vec<Vec<i32>>,
    pub samples: Vec<f32>,
    pub num_frames_raw: usize,
    pub num_frames_aligned: usize,
    pub stopped_by: StopReason,
}

/// Special-token ids the caller resolves from the checkpoint tokenizer
/// (prompt.py `HiggsAudioV3PromptBuilder`).
#[derive(Debug, Clone)]
pub struct PromptBuilder {
    pub tts_id: i32,
    pub ref_audio_id: i32,
    pub text_id: i32,
    pub audio_id: i32,
    /// Present only when the tokenizer defines `<|ref_text|>`.
    pub ref_text_id: Option<i32>,
}

/// One voice-clone reference: its optional transcript ids and the
/// ALREADY delayed code rows `[T + K - 1][K]` (the reference stores
/// `encode_reference_audio` output, which delays before building the
/// prompt).
#[derive(Debug, Clone)]
pub struct ReferenceCodes {
    pub text_ids: Option<Vec<i32>>,
    pub delayed_codes: Vec<Vec<i32>>,
}

/// The prompt layout (`PromptParts`): the token ids (audio positions
/// carry [`AUDIO_PLACEHOLDER_ID`]) plus each audio segment's token
/// offset and delayed code rows.
#[derive(Debug, Clone)]
pub struct PromptParts {
    pub token_ids: Vec<i32>,
    pub audio_segments: Vec<(usize, Vec<Vec<i32>>)>,
}

impl PromptBuilder {
    /// `build_prompt`: `<|tts|>`, then per reference the optional
    /// `<|ref_text|>` transcript, `<|ref_audio|>` and its placeholder
    /// rows, then `<|text|>` target text `<|audio|>`.
    pub fn build_prompt(
        &self,
        text_ids: &[i32],
        references: &[ReferenceCodes],
    ) -> Result<PromptParts> {
        let mut ids: Vec<i32> = Vec::with_capacity(text_ids.len() + 8);
        let mut segments = Vec::with_capacity(references.len());
        ids.push(self.tts_id);
        for reference in references {
            if let (Some(text_ids), Some(ref_text_id)) =
                (reference.text_ids.as_ref(), self.ref_text_id)
            {
                ids.push(ref_text_id);
                ids.extend_from_slice(text_ids);
            }
            ids.push(self.ref_audio_id);
            let start = ids.len();
            let num_rows = reference.delayed_codes.len();
            ids.resize(ids.len() + num_rows, AUDIO_PLACEHOLDER_ID);
            segments.push((start, reference.delayed_codes.clone()));
        }
        ids.push(self.text_id);
        ids.extend_from_slice(text_ids);
        ids.push(self.audio_id);
        Ok(PromptParts {
            token_ids: ids,
            audio_segments: segments,
        })
    }
}

/// `apply_delay_pattern`: raw codec codes `[T][K]` into the delayed rows
/// `[T + K - 1][K]`, BOC-filled ahead of each codebook's start and
/// EOC-filled behind its end.
pub fn apply_delay_pattern(codes: &[Vec<i32>], boc_id: i32, eoc_id: i32) -> Result<Vec<Vec<i32>>> {
    let num_codebooks = codes.first().map_or(0, |row| row.len());
    if num_codebooks == 0 {
        return Err(SpeechError::Input {
            why: "no codebooks".into(),
        });
    }
    if codes.iter().any(|row| row.len() != num_codebooks) {
        return Err(SpeechError::Input {
            why: "ragged codebook timeline".into(),
        });
    }
    let t = codes.len();
    let mut out = vec![vec![eoc_id; num_codebooks]; t + num_codebooks - 1];
    for codebook in 0..num_codebooks {
        if codebook > 0 {
            for row in out.iter_mut().take(codebook) {
                row[codebook] = boc_id;
            }
        }
        for (step, row) in codes.iter().enumerate() {
            out[codebook + step][codebook] = row[codebook];
        }
    }
    Ok(out)
}

/// `reverse_delay_pattern`: the delayed rows `[L][K]` back into the raw
/// codec codes `[L - K + 1][K]` taken along the diagonal.
pub fn reverse_delay_pattern(delayed: &[Vec<i32>]) -> Result<Vec<Vec<i32>>> {
    let num_codebooks = delayed.first().map_or(0, |row| row.len());
    if num_codebooks == 0 {
        return Err(SpeechError::Input {
            why: "no codebooks".into(),
        });
    }
    if delayed.iter().any(|row| row.len() != num_codebooks) {
        return Err(SpeechError::Input {
            why: "ragged delayed timeline".into(),
        });
    }
    let length = delayed.len();
    let t = length
        .checked_sub(num_codebooks)
        .ok_or_else(|| SpeechError::Input {
            why: format!("delayed rows have L={length}, K={num_codebooks}; need L >= K"),
        })?
        + 1;
    Ok((0..t)
        .map(|step| {
            (0..num_codebooks)
                .map(|codebook| delayed[step + codebook][codebook])
                .collect()
        })
        .collect())
}

/// One delayed multi-codebook sampler step's state
/// (`HiggsSamplerState`).
#[derive(Debug, Clone)]
pub struct SamplerState {
    pub num_codebooks: usize,
    pub delay_count: usize,
    pub eoc_countdown: Option<i32>,
    pub generation_done: bool,
    pub last_codes: Option<Vec<i32>>,
}

impl SamplerState {
    pub fn new(num_codebooks: usize) -> Self {
        SamplerState {
            num_codebooks,
            delay_count: 0,
            eoc_countdown: None,
            generation_done: false,
            last_codes: None,
        }
    }
}

/// The per-row greedy draw (`sample_independent` argmax): argmax over
/// each `[V]` logit row of the `[K, V]` step block.
pub fn greedy_sample_audio(audio_logits: &[f32], codebooks: usize, width: usize) -> Vec<i32> {
    (0..codebooks)
        .map(|codebook| argmax(&audio_logits[codebook * width..(codebook + 1) * width]) as i32)
        .collect()
}

/// `sample_independent`'s sampled path: temperature scaling, then the
/// reference `_apply_top_k` / `_apply_top_p` masks (each recomputes the
/// log-softmax, filters the log-probs with the shared
/// [`sampling`] port, and maps the masked entries back to negative
/// infinity on the scaled logits), then the categorical draw. The draw
/// uses the crate-local threefry uniforms in row-major order, so a
/// sampled Rust run diverges token-by-token from mlx at the same seed
/// while the filter masks are pinned by fixtures.
pub fn sample_independent(
    audio_logits: &[f32],
    codebooks: usize,
    width: usize,
    params: &GenerateParams,
    key: rng::Key,
) -> Result<Vec<i32>> {
    let greedy = params.temperature <= 1e-5 || params.top_k == Some(1);
    if greedy {
        return Ok(greedy_sample_audio(audio_logits, codebooks, width));
    }
    let scale = 1.0 / params.temperature;
    let mut scaled: Vec<f32> = audio_logits.iter().map(|&l| l * scale).collect();
    let top_k = params
        .top_k
        .filter(|&k| k > 0 && (k as usize) < width)
        .map(|k| k as usize);
    if let Some(top_k) = top_k {
        for codebook in 0..codebooks {
            let row = &mut scaled[codebook * width..(codebook + 1) * width];
            apply_masked_filter(row, &|logprobs| sampling::apply_top_k(logprobs, top_k))?;
        }
    }
    let top_p = params.top_p.filter(|&p| p > 0.0 && p < 1.0);
    if let Some(top_p) = top_p {
        for codebook in 0..codebooks {
            let row = &mut scaled[codebook * width..(codebook + 1) * width];
            apply_masked_filter(row, &|logprobs| {
                sampling::apply_top_p(logprobs, top_p);
                Ok(())
            })?;
        }
    }
    // mx.random.categorical over the [K, V] block draws K * V uniforms;
    // each row's inverse-CDF draw consumes its row's first uniform.
    let uniforms = rng::uniform01(key, codebooks * width);
    let mut out = Vec::with_capacity(codebooks);
    for codebook in 0..codebooks {
        let row = &scaled[codebook * width..(codebook + 1) * width];
        out.push(categorical_with_uniform(row, uniforms[codebook * width]) as i32);
    }
    Ok(out)
}

/// One `_apply_top_k` / `_apply_top_p` link: log-softmax, run the filter
/// on the log-probs, and mask the logits where the filtered log-probs
/// went to negative infinity (`_mask_logits_from_logprobs`).
fn apply_masked_filter(
    logits: &mut [f32],
    filter: &dyn Fn(&mut [f32]) -> Result<()>,
) -> Result<()> {
    let mut logprobs = sampling::log_softmax(logits);
    filter(&mut logprobs)?;
    for (logit, &logprob) in logits.iter_mut().zip(&logprobs) {
        if logprob.is_infinite() && logprob < 0.0 {
            *logit = f32::NEG_INFINITY;
        }
    }
    Ok(())
}

/// `mx.random.categorical` for one row with an external uniform:
/// exclusive cumsum of `exp(x - max)`, `u = uniform * total`, then the
/// count of CDF entries `<= u` minus one (f64 comparisons, matching
/// [`rng::categorical`]).
fn categorical_with_uniform(logits: &[f32], uniform: f32) -> usize {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f64> = logits
        .iter()
        .map(|&x| (x - max) as f64)
        .map(|v| v.exp())
        .collect();
    let total: f64 = weights.iter().sum();
    let u = uniform as f64 * total;
    let mut cumulative = 0.0f64;
    let mut count = 0usize;
    for &w in &weights {
        if cumulative > u {
            break;
        }
        count += 1;
        cumulative += w;
    }
    count.saturating_sub(1).min(weights.len() - 1)
}

/// The reference `step`: one sampler step over the `[K, V]` logits with
/// the ramp-in BOC forcing, the EOC countdown, and the done flag. When
/// `generation_done` was already set the row is all `STOP_CODE` (-1);
/// the generate loop always breaks before drawing again, so those rows
/// never reach the codec.
pub fn step(
    audio_logits: &[f32],
    state: &mut SamplerState,
    params: &GenerateParams,
    boc_id: i32,
    eoc_id: i32,
    key: rng::Key,
) -> Result<Vec<i32>> {
    let n = state.num_codebooks;
    if state.generation_done {
        return Ok(vec![-1; n]);
    }
    if n == 0 || audio_logits.len() % n != 0 {
        return Err(SpeechError::Input {
            why: format!(
                "logits of {} values do not split into {n} codebook rows",
                audio_logits.len()
            ),
        });
    }
    let width = audio_logits.len() / n;
    let mut codes = sample_independent(audio_logits, n, width, params, key)?;

    if state.delay_count < n {
        let next_codebook = state.delay_count + 1;
        if next_codebook < n {
            for (codebook, token) in codes.iter_mut().enumerate() {
                if codebook >= next_codebook {
                    *token = boc_id;
                }
            }
        }
        state.delay_count += 1;
    } else if let Some(countdown) = state.eoc_countdown.as_mut() {
        *countdown -= 1;
        if *countdown <= 0 {
            state.generation_done = true;
        }
    } else if codes[0] == eoc_id {
        if n <= 2 {
            state.generation_done = true;
        } else {
            state.eoc_countdown = Some(n as i32 - 2);
        }
    }

    if !state.generation_done {
        state.last_codes = Some(codes.clone());
    }
    Ok(codes)
}

/// The loaded family model (`Model`): the qwen3 backbone plus the fused
/// multimodal embedding table.
pub struct Model {
    pub config: ModelConfig,
    backbone: lm::Backbone,
    /// `[K * codebook_size, hidden_size]`; the audio input embedding and
    /// (tied) the audio logits head.
    multimodal_embedding: Vec<f32>,
}

impl Model {
    /// Opens a checkpoint directory (`config.json` + safetensors).
    pub fn open(dir: &std::path::Path) -> Result<Self> {
        let config_value = quant::read_json(&dir.join("config.json"))?;
        let weights = lm::Weights::open(dir)?;
        Self::load(weights, &config_value)
    }

    /// Loads from opened weights and the raw config dict. Four tensor
    /// layouts are recognized: the bare module tree (`layers.*`,
    /// `embed_tokens`, `multimodal_embedding`), the full-model save
    /// (`model.` prefix), the MLX-sanitized save (`backbone.` prefix),
    /// and the raw bosonai checkpoint (`body.layers.*` with the text
    /// embedding at `tied.embedding.text_embedding` and the fused table
    /// at `tied.embedding.modality_embeddings.0.embedding`, mirroring
    /// the reference `sanitize` renames; the skipped `tied.head.*` and
    /// `modality_embeddings.0.model.*` tensors are simply never read).
    pub fn load(weights: lm::Weights, config: &Value) -> Result<Self> {
        let config = ModelConfig::from_json(config)?;
        // Absolute probes: `Weights::open` may carry its auto-detected
        // `model.` prefix, which would make the prefixed `contains` hit
        // on a full-model save and misdetect it as the bare layout.
        let backbone_prefix = if weights.contains_absolute("layers.0.self_attn.q_proj") {
            ""
        } else if weights.contains_absolute("backbone.layers.0.self_attn.q_proj") {
            "backbone."
        } else if weights.contains_absolute("body.layers.0.self_attn.q_proj") {
            // Raw bosonai layout: `Weights::open` never sees its synthetic
            // `model.` prefix here, so the prefix is the raw `body.` and
            // only the text embedding keeps its tied name.
            "body."
        } else {
            // `Weights::open` auto-detected the full-model `model.` prefix.
            "model."
        };
        let (embed_table, had_bare_or_prefixed_embed) =
            if weights.contains_absolute(&format!("{backbone_prefix}embed_tokens")) {
                ("embed_tokens", true)
            } else {
                ("tied.embedding.text_embedding", false)
            };
        if !had_bare_or_prefixed_embed
            && !weights.contains_absolute("tied.embedding.text_embedding")
        {
            return Err(SpeechError::Tensor {
                name: "embed_tokens.weight".into(),
                why: "text embedding tensor is missing (neither the module-tree \
                      nor the raw tied.embedding layout is present)"
                    .into(),
            });
        }
        let multimodal_candidates = [
            "multimodal_embedding",
            "model.multimodal_embedding",
            "backbone.multimodal_embedding",
            "tied.embedding.modality_embeddings.0.embedding",
        ];
        let multimodal_name = multimodal_candidates
            .iter()
            .find(|name| weights.contains_absolute(name))
            .ok_or_else(|| SpeechError::Tensor {
                name: "multimodal_embedding.weight".into(),
                why: "tensor is missing".into(),
            })?;
        let hidden = config.text_config.hidden_size;
        let rows = config.audio_num_codebooks * config.codebook_stride();
        let multimodal_embedding =
            weights.tensor_absolute(&format!("{multimodal_name}.weight"), &[rows, hidden])?;
        let backbone = lm::Backbone::load_with_embed_table(
            &weights.with_prefix(backbone_prefix),
            config.text_config.to_dense_config(),
            embed_table,
        )?;
        Ok(Model {
            config,
            backbone,
            multimodal_embedding,
        })
    }

    /// `_embed_audio_codes` for one frame: the `[K]` codes shift by the
    /// per-codebook offsets and sum their embedding rows into one hidden
    /// vector.
    pub fn embed_audio_codes(&self, codes: &[i32]) -> Result<Vec<f32>> {
        let k = self.config.audio_num_codebooks;
        if codes.len() != k {
            return Err(SpeechError::Input {
                why: format!("frame of {} codes for {k} codebooks", codes.len()),
            });
        }
        let hidden = self.config.text_config.hidden_size;
        let stride = self.config.codebook_stride();
        let mut out = vec![0.0f32; hidden];
        for (codebook, &code) in codes.iter().enumerate() {
            let row = embed_row(
                &self.multimodal_embedding,
                codebook * stride + index_of(code)?,
                hidden,
            )?;
            for (col, value) in out.iter_mut().enumerate() {
                *value += row[col];
            }
        }
        Ok(out)
    }

    /// `_embed_audio_codes` for a `[T][K]` block: one summed row per
    /// frame.
    pub fn embed_audio_codes_rows(&self, codes: &[Vec<i32>]) -> Result<Vec<f32>> {
        let hidden = self.config.text_config.hidden_size;
        let mut out = Vec::with_capacity(codes.len() * hidden);
        for frame in codes {
            out.extend_from_slice(&self.embed_audio_codes(frame)?);
        }
        Ok(out)
    }

    /// `_audio_logits`: the tied embedding table as a linear over
    /// post-norm hidden states, `[rows, K * codebook_size]` in row-major
    /// `[row, codebook, class]` order.
    pub fn audio_logits(&self, hidden: &[f32]) -> Vec<f32> {
        let hidden_size = self.config.text_config.hidden_size;
        let rows = hidden.len() / hidden_size;
        let outputs = self.multimodal_embedding.len() / hidden_size;
        crate::ops::linear(
            hidden,
            &self.multimodal_embedding,
            None,
            rows,
            hidden_size,
            outputs,
        )
    }

    /// `_build_prompt_embeddings`: text rows from the backbone embedding
    /// table, audio segment rows from the fused table, concatenated.
    /// Returns the `[T, hidden]` row-major block. An unresolved
    /// placeholder in the tail is refused, like the reference.
    pub fn build_prompt_embeddings(&self, parts: &PromptParts) -> Result<Vec<f32>> {
        let hidden = self.config.text_config.hidden_size;
        let vocab = self.config.text_config.vocab_size;
        let mut pieces: Vec<Vec<f32>> = Vec::new();
        let mut cursor = 0usize;
        for (start, delayed_codes) in &parts.audio_segments {
            pieces.push(crate::lm::engine::lookup_embeddings(
                self.backbone.embed_table(),
                vocab,
                hidden,
                &parts.token_ids[cursor..*start],
            )?);
            pieces.push(self.embed_audio_codes_rows(delayed_codes)?);
            cursor = start + delayed_codes.len();
        }
        let tail = &parts.token_ids[cursor..];
        if tail.contains(&AUDIO_PLACEHOLDER_ID) {
            return Err(SpeechError::Input {
                why: "internal prompt error: unresolved audio placeholder".into(),
            });
        }
        pieces.push(crate::lm::engine::lookup_embeddings(
            self.backbone.embed_table(),
            vocab,
            hidden,
            tail,
        )?);
        Ok(pieces.concat())
    }

    /// The full reference `generate` (single prompt): prefill the prompt
    /// embeddings, then the delay-pattern sampler loop, then the delay
    /// reversal, codec decode, and edge fades. `seed` only feeds the
    /// sampled (temperature > 1e-5, top_k != 1) draws; greedy runs never
    /// draw and are fully deterministic.
    pub fn generate(
        &mut self,
        codec: &HiggsAudioTokenizer,
        prompt_embeds: &[f32],
        params: &GenerateParams,
        seed: u64,
    ) -> Result<Generated> {
        let (frames, stopped_by) = self.generate_frames(prompt_embeds, params, seed)?;
        let raw = frames.len();
        let aligned = if raw < self.config.audio_num_codebooks {
            // The reference `_decode_audio` returns an empty waveform
            // before the reversal when fewer rows than codebooks exist.
            Vec::new()
        } else {
            reverse_delay_pattern(&frames)?
        };
        let samples = if aligned.is_empty() {
            Vec::new()
        } else {
            // The Rust codec consumes codebook-major `[K][T]`.
            let num_codebooks = self.config.audio_num_codebooks;
            let mut codec_codes: Vec<Vec<i32>> =
                vec![Vec::with_capacity(aligned.len()); num_codebooks];
            for frame in &aligned {
                for (codebook, &code) in frame.iter().take(num_codebooks).enumerate() {
                    codec_codes[codebook].push(code);
                }
            }
            codec.decode_tokens(&codec_codes)?
        };
        let mut samples = samples;
        crate::tts::higgs_audio::apply_fades(
            &mut samples,
            params.fade_in_ms,
            params.fade_out_ms,
            self.config.sample_rate,
        );
        let num_frames_aligned = aligned.len();
        Ok(Generated {
            frames,
            codes: aligned,
            num_frames_raw: raw,
            num_frames_aligned,
            samples,
            stopped_by,
        })
    }

    /// The raw frame loop alone: prefill, the sampled/greedy steps with
    /// ramp-in and EOC countdown, and the stop conditions. Returns the
    /// delayed rows and why the loop stopped, without decoding.
    pub fn generate_frames(
        &mut self,
        prompt_embeds: &[f32],
        params: &GenerateParams,
        seed: u64,
    ) -> Result<(Vec<Vec<i32>>, StopReason)> {
        if prompt_embeds.is_empty() {
            return Err(SpeechError::Input {
                why: "empty forward input".into(),
            });
        }
        let hidden_size = self.config.text_config.hidden_size;
        if prompt_embeds.len() % hidden_size != 0 {
            return Err(SpeechError::Input {
                why: format!(
                    "prompt embedding length {} is not a multiple of hidden size {hidden_size}",
                    prompt_embeds.len()
                ),
            });
        }
        let mut caches = lm::make_prompt_cache(self.config.text_config.num_hidden_layers);
        let mut hidden = self
            .backbone
            .forward_hidden(Input::Embeddings(prompt_embeds), Some(&mut caches))?;
        let mut state = SamplerState::new(self.config.audio_num_codebooks);
        let mut frames: Vec<Vec<i32>> = Vec::new();
        let mut stopped_by = StopReason::MaxSteps;
        let mut keys = rng::KeySequence::new(seed);
        let boc = self.config.audio_boc_token_id;
        let eoc = self.config.audio_eoc_token_id;
        for _ in 0..params.max_new_frames {
            let rows = hidden.len() / hidden_size;
            let last = &hidden[(rows - 1) * hidden_size..];
            let logits = self.audio_logits(last);
            let codes = step(&logits, &mut state, params, boc, eoc, keys.next())?;
            if state.generation_done {
                frames.push(codes);
                stopped_by = StopReason::EocRamp;
                break;
            }
            let embed = self.embed_audio_codes(&codes)?;
            frames.push(codes);
            hidden = self
                .backbone
                .forward_hidden(Input::Embeddings(&embed), Some(&mut caches))?;
        }
        Ok((frames, stopped_by))
    }
}

fn index_of(code: i32) -> Result<usize> {
    usize::try_from(code).map_err(|_| SpeechError::Input {
        why: format!("negative audio code {code}"),
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

#[cfg(test)]
mod tests;
