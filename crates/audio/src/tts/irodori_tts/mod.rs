//! Irodori TTS (Japanese flow-matching TTS, repo family
//! `mlx-community/Irodori-TTS-*`).
//!
//! Port of `mlx_audio/tts/models/irodori_tts/` at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/irodori_tts)
//! (MLX 0.32.3). A Rectified Flow DiT over continuous DACVAE latents at
//! 48 kHz: gated joint attention over latent self-tokens plus text and
//! speaker/caption context (per-branch key/value projections, half
//! interleaved RoPE on the self keys/queries, low-rank AdaLN with tanh
//! gates), a scratch or ModernBERT text encoder (the v4 checkpoints
//! bundle the ModernBERT backbone and feed separate text/caption
//! projectors), a reference-latent speaker encoder, an optional
//! token-sum duration predictor with AdaRN-Zero conditioning, and the
//! Euler RF sampler (per-condition CFG in independent or joint mode,
//! sway schedule, optional temporal score rescaling and speaker-KV
//! scaling) plus the MeanFlow distilled sampler. The waveform decodes
//! through the shared [DACVAE codec port](../../codec/dacvae/README.md).
//!
//! Scope, matching the reference single-prompt path: single batch,
//! non-streaming, caller-owned Japanese text normalization (including
//! the NFKC pass) and tokenization (hand over the padded token ids and
//! masks; the bundled llm-jp tokenizer is not ported). The reference's
//! batched CFG bundles run as sequential single-row forwards (rows are
//! independent); the `alternating` CFG mode is refused. See
//! [README.md](README.md) for the verification evidence and open items.

use std::path::Path;

use serde_json::Value;

use crate::codec::dacvae::Dacvae;
use crate::lm::{Bias, Weights};
use crate::music::minimax_music3::rng;
use crate::nn::Linear;
use crate::ops;
use crate::{quant, Result, SpeechError};

#[cfg(test)]
mod tests;

/// The flow noise source: one standard-normal draw of
/// `sequence_length * patched_latent_dim` values per generate call (the
/// initial Rectified Flow state, after the optional truncation factor).
pub trait FlowNoise {
    fn normal(&mut self, n: usize) -> Vec<f32>;
}

/// Seeded crate-local threefry noise for standalone runs.
pub struct SeededFlowNoise {
    keys: rng::KeySequence,
}

impl SeededFlowNoise {
    pub fn new(seed: u64) -> Self {
        Self {
            keys: rng::KeySequence::new(seed),
        }
    }
}

impl FlowNoise for SeededFlowNoise {
    fn normal(&mut self, n: usize) -> Vec<f32> {
        rng::normal(self.keys.next(), n)
    }
}

fn bad_config(field: &str, why: impl Into<String>) -> SpeechError {
    SpeechError::BadConfig {
        field: field.to_owned(),
        why: why.into(),
    }
}

fn usize_field(value: &Value, key: &str, default: usize) -> Result<usize> {
    match value.get(key).and_then(Value::as_u64) {
        Some(number) => usize::try_from(number).map_err(|_| bad_config(key, "does not fit usize")),
        None => Ok(default),
    }
}

fn f32_field(value: &Value, key: &str, default: f32) -> f32 {
    value
        .get(key)
        .and_then(Value::as_f64)
        .map(|v| v as f32)
        .unwrap_or(default)
}

fn bool_field(value: &Value, key: &str, default: bool) -> bool {
    value.get(key).and_then(Value::as_bool).unwrap_or(default)
}

fn optional_f32(value: &Value, key: &str) -> Option<f32> {
    value.get(key).and_then(Value::as_f64).map(|v| v as f32)
}

fn text_field(value: &Value, key: &str, default: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_owned()
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// The DiT configuration (`IrodoriDiTConfig`); the resolved `caption_*`
/// and ratio properties read through at use sites.
#[derive(Debug, Clone)]
pub struct DitConfig {
    pub flow_parameterization: String,
    pub latent_dim: usize,
    pub latent_patch_size: usize,
    pub model_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub mlp_ratio: f32,
    pub text_mlp_ratio: Option<f32>,
    pub speaker_mlp_ratio: Option<f32>,
    pub text_vocab_size: usize,
    pub text_add_bos: bool,
    pub text_dim: usize,
    pub text_layers: usize,
    pub text_heads: usize,
    /// `scratch` or `pretrained`.
    pub text_encoder_type: String,
    pub text_encoder_config: Option<Value>,
    pub pretrained_projector_type: String,
    pub pretrained_projector_hidden_ratio: f32,
    pub speaker_dim: usize,
    pub speaker_layers: usize,
    pub speaker_heads: usize,
    pub speaker_patch_size: usize,
    pub timestep_embed_dim: usize,
    pub adaln_rank: usize,
    pub norm_eps: f32,
    pub use_caption_condition: bool,
    pub use_speaker_condition: Option<bool>,
    pub caption_vocab_size: Option<usize>,
    pub caption_dim: Option<usize>,
    pub caption_layers: Option<usize>,
    pub caption_heads: Option<usize>,
    pub caption_mlp_ratio: Option<f32>,
    pub use_duration_predictor: bool,
    pub duration_aux_dim: usize,
    pub duration_hidden_dim: usize,
    pub duration_layers: usize,
    pub duration_attention_heads: usize,
    pub duration_architecture: String,
    pub duration_token_init_frames: f32,
}

impl DitConfig {
    pub fn from_json(value: &Value) -> Result<Self> {
        Ok(Self {
            flow_parameterization: text_field(value, "flow_parameterization", "rf_velocity"),
            latent_dim: usize_field(value, "latent_dim", 32)?,
            latent_patch_size: usize_field(value, "latent_patch_size", 1)?,
            model_dim: usize_field(value, "model_dim", 1280)?,
            num_layers: usize_field(value, "num_layers", 12)?,
            num_heads: usize_field(value, "num_heads", 20)?,
            mlp_ratio: f32_field(value, "mlp_ratio", 2.875),
            // The dataclass defaults these ratios to 2.6; an explicit
            // null falls back to `mlp_ratio` through the resolver.
            text_mlp_ratio: match value.get("text_mlp_ratio") {
                Some(Value::Null) | None => Some(2.6),
                Some(v) => v.as_f64().map(|n| n as f32),
            },
            speaker_mlp_ratio: match value.get("speaker_mlp_ratio") {
                Some(Value::Null) | None => Some(2.6),
                Some(v) => v.as_f64().map(|n| n as f32),
            },
            text_vocab_size: usize_field(value, "text_vocab_size", 99574)?,
            text_add_bos: bool_field(value, "text_add_bos", true),
            text_dim: usize_field(value, "text_dim", 512)?,
            text_layers: usize_field(value, "text_layers", 10)?,
            text_heads: usize_field(value, "text_heads", 8)?,
            text_encoder_type: text_field(value, "text_encoder_type", "scratch"),
            text_encoder_config: value
                .get("text_encoder_config")
                .cloned()
                .filter(|v| !v.is_null()),
            pretrained_projector_type: text_field(value, "pretrained_projector_type", "linear"),
            pretrained_projector_hidden_ratio: f32_field(
                value,
                "pretrained_projector_hidden_ratio",
                2.0,
            ),
            speaker_dim: usize_field(value, "speaker_dim", 768)?,
            speaker_layers: usize_field(value, "speaker_layers", 8)?,
            speaker_heads: usize_field(value, "speaker_heads", 12)?,
            speaker_patch_size: usize_field(value, "speaker_patch_size", 1)?,
            timestep_embed_dim: usize_field(value, "timestep_embed_dim", 512)?,
            adaln_rank: usize_field(value, "adaln_rank", 192)?,
            norm_eps: f32_field(value, "norm_eps", 1e-5),
            use_caption_condition: bool_field(value, "use_caption_condition", false),
            use_speaker_condition: value.get("use_speaker_condition").and_then(Value::as_bool),
            caption_vocab_size: value
                .get("caption_vocab_size")
                .and_then(Value::as_u64)
                .map(|v| v as usize),
            caption_dim: value
                .get("caption_dim")
                .and_then(Value::as_u64)
                .map(|v| v as usize),
            caption_layers: value
                .get("caption_layers")
                .and_then(Value::as_u64)
                .map(|v| v as usize),
            caption_heads: value
                .get("caption_heads")
                .and_then(Value::as_u64)
                .map(|v| v as usize),
            caption_mlp_ratio: optional_f32(value, "caption_mlp_ratio"),
            use_duration_predictor: bool_field(value, "use_duration_predictor", false),
            duration_aux_dim: usize_field(value, "duration_aux_dim", 14)?,
            duration_hidden_dim: usize_field(value, "duration_hidden_dim", 1024)?,
            duration_layers: usize_field(value, "duration_layers", 3)?,
            duration_attention_heads: usize_field(value, "duration_attention_heads", 8)?,
            duration_architecture: text_field(
                value,
                "duration_architecture",
                "token_sum_adarn_zero_no_aux",
            ),
            duration_token_init_frames: f32_field(value, "duration_token_init_frames", 9.0),
        })
    }

    pub fn use_pretrained_text_encoder(&self) -> bool {
        self.text_encoder_type
            .trim()
            .eq_ignore_ascii_case("pretrained")
    }

    pub fn use_meanflow(&self) -> bool {
        self.flow_parameterization
            .trim()
            .eq_ignore_ascii_case("meanflow")
    }

    pub fn use_speaker_condition_resolved(&self) -> bool {
        match self.use_speaker_condition {
            Some(value) => value,
            None => !self.use_caption_condition,
        }
    }

    pub fn caption_vocab_size_resolved(&self) -> usize {
        self.caption_vocab_size.unwrap_or(self.text_vocab_size)
    }

    pub fn caption_dim_resolved(&self) -> usize {
        self.caption_dim.unwrap_or(self.text_dim)
    }

    pub fn caption_layers_resolved(&self) -> usize {
        self.caption_layers.unwrap_or(self.text_layers)
    }

    pub fn caption_heads_resolved(&self) -> usize {
        self.caption_heads.unwrap_or(self.text_heads)
    }

    pub fn caption_mlp_ratio_resolved(&self) -> f32 {
        self.caption_mlp_ratio
            .unwrap_or(self.text_mlp_ratio_resolved())
    }

    pub fn patched_latent_dim(&self) -> usize {
        self.latent_dim * self.latent_patch_size
    }

    pub fn speaker_patched_latent_dim(&self) -> usize {
        self.patched_latent_dim() * self.speaker_patch_size
    }

    pub fn text_mlp_ratio_resolved(&self) -> f32 {
        self.text_mlp_ratio.unwrap_or(self.mlp_ratio)
    }

    pub fn speaker_mlp_ratio_resolved(&self) -> f32 {
        self.speaker_mlp_ratio.unwrap_or(self.mlp_ratio)
    }
}

/// The sampler configuration (`SamplerConfig`).
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub num_steps: usize,
    pub cfg_scale_text: f32,
    pub cfg_scale_speaker: f32,
    pub cfg_scale_caption: f32,
    pub cfg_guidance_mode: String,
    pub cfg_min_t: f32,
    pub cfg_max_t: f32,
    pub truncation_factor: Option<f32>,
    pub rescale_k: Option<f32>,
    pub rescale_sigma: Option<f32>,
    pub context_kv_cache: bool,
    pub speaker_kv_scale: Option<f32>,
    pub speaker_kv_min_t: Option<f32>,
    pub speaker_kv_max_layers: Option<usize>,
    pub sequence_length: usize,
    pub t_schedule_mode: String,
    pub sway_coeff: f32,
    pub min_seconds: f32,
    pub max_seconds: f32,
}

impl SamplerConfig {
    pub fn from_json(value: &Value) -> Result<Self> {
        Ok(Self {
            num_steps: usize_field(value, "num_steps", 40)?,
            cfg_scale_text: f32_field(value, "cfg_scale_text", 3.0),
            cfg_scale_speaker: f32_field(value, "cfg_scale_speaker", 5.0),
            cfg_scale_caption: f32_field(value, "cfg_scale_caption", 3.0),
            cfg_guidance_mode: text_field(value, "cfg_guidance_mode", "independent"),
            cfg_min_t: f32_field(value, "cfg_min_t", 0.5),
            cfg_max_t: f32_field(value, "cfg_max_t", 1.0),
            truncation_factor: optional_f32(value, "truncation_factor"),
            rescale_k: optional_f32(value, "rescale_k"),
            rescale_sigma: optional_f32(value, "rescale_sigma"),
            context_kv_cache: bool_field(value, "context_kv_cache", true),
            speaker_kv_scale: optional_f32(value, "speaker_kv_scale"),
            speaker_kv_min_t: optional_f32(value, "speaker_kv_min_t"),
            speaker_kv_max_layers: value
                .get("speaker_kv_max_layers")
                .and_then(Value::as_u64)
                .map(|v| v as usize),
            sequence_length: usize_field(value, "sequence_length", 750)?,
            t_schedule_mode: text_field(value, "t_schedule_mode", "linear"),
            sway_coeff: f32_field(value, "sway_coeff", -1.0),
            min_seconds: f32_field(value, "min_seconds", 0.5),
            max_seconds: f32_field(value, "max_seconds", 30.0),
        })
    }
}

/// The top-level model configuration (`ModelConfig`).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_type: String,
    pub sample_rate: u32,
    pub max_text_length: usize,
    pub max_caption_length: usize,
    pub max_speaker_latent_length: usize,
    pub ref_max_seconds: f32,
    pub audio_downsample_factor: usize,
    pub dit: DitConfig,
    pub sampler: SamplerConfig,
}

impl ModelConfig {
    pub fn from_json(value: &Value) -> Result<Self> {
        Ok(Self {
            model_type: text_field(value, "model_type", "irodori_tts"),
            sample_rate: usize_field(value, "sample_rate", 48000)? as u32,
            max_text_length: usize_field(value, "max_text_length", 256)?,
            max_caption_length: usize_field(value, "max_caption_length", 512)?,
            max_speaker_latent_length: usize_field(value, "max_speaker_latent_length", 6400)?,
            ref_max_seconds: f32_field(value, "ref_max_seconds", 30.0),
            audio_downsample_factor: usize_field(value, "audio_downsample_factor", 1920)?,
            dit: DitConfig::from_json(value.get("dit").unwrap_or(&Value::Null))?,
            sampler: SamplerConfig::from_json(value.get("sampler").unwrap_or(&Value::Null))?,
        })
    }
}

// ---------------------------------------------------------------------------
// Shared math
// ---------------------------------------------------------------------------

/// Per-head RMSNorm over `[rows, heads * head_dim]` activations with the
/// checkpoint's `(heads, head_dim)` weight: the mean runs over the head
/// dim only, not the full row.
fn rms_norm_per_head(
    x: &mut [f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    weight: &[f32],
    eps: f32,
) {
    for row in 0..rows {
        for h in 0..heads {
            let base = row * heads * head_dim + h * head_dim;
            let mut sum = 0.0f32;
            for value in &x[base..base + head_dim] {
                sum += value * value;
            }
            let scale = 1.0 / (sum / head_dim as f32 + eps).sqrt();
            for (offset, value) in x[base..base + head_dim].iter_mut().enumerate() {
                *value = *value * scale * weight[h * head_dim + offset];
            }
        }
    }
}

/// Plain RMSNorm over rows with a flat weight (the crate norm, kept here
/// under the reference name's semantics).
fn rms_norm(x: &mut [f32], rows: usize, weight: &[f32], eps: f32) {
    crate::nn::RmsNorm::new(weight.to_vec(), eps).apply(x, rows);
}

/// `precompute_freqs_cis`: interleaved RoPE tables `(cos, sin)` of
/// `[end, dim / 2]` at `theta ** (-2i / dim)`.
fn rope_tables(dim: usize, end: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = dim / 2;
    let mut cos = vec![0.0f32; end * half];
    let mut sin = vec![0.0f32; end * half];
    for position in 0..end {
        for (i, cos_value) in cos[position * half..(position + 1) * half]
            .iter_mut()
            .enumerate()
        {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / dim as f32);
            let angle = position as f32 * freq;
            *cos_value = angle.cos();
            sin[position * half + i] = angle.sin();
        }
    }
    (cos, sin)
}

/// `apply_rotary_emb` over `[rows, heads * head_dim]` activations: the
/// interleaved (even, odd) rotation runs per head over head_dim with
/// the `[rows, head_dim / 2]` tables.
fn apply_rope_interleaved_per_head(
    x: &mut [f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    let half = head_dim / 2;
    for row in 0..rows {
        for h in 0..heads {
            let base = (row * heads + h) * head_dim;
            for i in 0..half {
                let even = x[base + 2 * i];
                let odd = x[base + 2 * i + 1];
                x[base + 2 * i] = even * cos[row * half + i] - odd * sin[row * half + i];
                x[base + 2 * i + 1] = odd * cos[row * half + i] + even * sin[row * half + i];
            }
        }
    }
}

/// `get_timestep_embedding`: `(cos, sin)` halves at
/// `1000 * exp(-ln(10000) * i / half)`.
fn timestep_embedding(t: f32, embed_size: usize) -> Vec<f32> {
    let half = embed_size / 2;
    let mut out = vec![0.0f32; embed_size];
    for i in 0..half {
        let freq = 1000.0 * (-10000.0f32.ln() * i as f32 / half as f32).exp();
        let angle = t * freq;
        out[i] = angle.cos();
        out[half + i] = angle.sin();
    }
    out
}

#[allow(
    clippy::too_many_arguments,
    reason = "attention kernels carry the tensor shape explicitly"
)]
/// Non-causal single-batch attention over `[sq, heads * head_dim]`
/// queries and `[sk, heads * head_dim]` keys/values with an additive
/// `[sq, sk]` mask broadcast over heads (the reference's
/// `_bool_to_additive_mask` layout, -1e9 floor included in the mask).
fn masked_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    sq: usize,
    sk: usize,
    heads: usize,
    head_dim: usize,
    mask: Option<&[f32]>,
) -> Vec<f32> {
    let scale = (head_dim as f64).powf(-0.5) as f32;
    let mut out = vec![0.0f32; sq * heads * head_dim];
    let mut scores = vec![0.0f32; sk];
    for h in 0..heads {
        for i in 0..sq {
            let q_row = &q[(i * heads + h) * head_dim..(i * heads + h + 1) * head_dim];
            let mut max = f32::NEG_INFINITY;
            for (j, score) in scores.iter_mut().enumerate() {
                let k_row = &k[(j * heads + h) * head_dim..(j * heads + h + 1) * head_dim];
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_row[d] * k_row[d];
                }
                let mut value = dot * scale;
                if let Some(mask) = mask {
                    value += mask[i * sk + j];
                }
                *score = value;
                max = max.max(value);
            }
            let mut sum = 0.0f32;
            for score in &mut scores {
                *score = (*score - max).exp();
                sum += *score;
            }
            let inv = 1.0 / sum;
            for (j, &weight) in scores.iter().enumerate() {
                let v_row = &v[(j * heads + h) * head_dim..(j * heads + h + 1) * head_dim];
                let out_row = &mut out[(i * heads + h) * head_dim..(i * heads + h + 1) * head_dim];
                for (o, &vv) in out_row.iter_mut().zip(v_row) {
                    *o += weight * inv * vv;
                }
            }
        }
    }
    out
}

/// Boolean key mask `[sk]` to the additive float mask the attention
/// applies (reference `_bool_to_additive_mask`, `-1e9` floor).
fn additive_mask(mask: &[bool]) -> Vec<f32> {
    mask.iter()
        .map(|&valid| if valid { 0.0 } else { -1e9 })
        .collect()
}

/// SwiGLU feed-forward: `w2(silu(w1(x)) * w3(x))`.
struct SwiGlu {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl SwiGlu {
    fn load(weights: &Weights, prefix: &str, dim: usize, hidden: usize) -> Result<Self> {
        Ok(Self {
            w1: weights.linear(&format!("{prefix}w1"), dim, hidden, Bias::None)?,
            w2: weights.linear(&format!("{prefix}w2"), hidden, dim, Bias::None)?,
            w3: weights.linear(&format!("{prefix}w3"), dim, hidden, Bias::None)?,
        })
    }

    fn forward(&self, x: &mut [f32], rows: usize) {
        let gate = self.w1.forward(x, rows);
        let up = self.w3.forward(x, rows);
        let fused: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| g * (1.0 / (1.0 + (-g).exp())) * u)
            .collect();
        let down = self.w2.forward(&fused, rows);
        x.copy_from_slice(&down);
    }
}

// ---------------------------------------------------------------------------
// ModernBERT text backbone (v4 pretrained encoder)
// ---------------------------------------------------------------------------

/// `ModernBertConfig`.
#[derive(Debug, Clone)]
pub struct ModernBertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// `gelu` (exact) or the tanh approximation.
    pub hidden_activation: String,
    pub norm_eps: f32,
    pub norm_bias: bool,
    pub attention_bias: bool,
    pub mlp_bias: bool,
    pub global_attn_every_n_layers: usize,
    pub local_attention: usize,
    pub global_rope_theta: f32,
    pub local_rope_theta: f32,
}

impl ModernBertConfig {
    pub fn from_json(value: &Value) -> Result<Self> {
        let mut resolved = value.clone();
        // transformers >= 5 nests the per-layer-type RoPE settings.
        if let Some(rope) = value.get("rope_parameters").and_then(Value::as_object) {
            if let Some(object) = resolved.as_object_mut() {
                if let Some(full) = rope.get("full_attention").and_then(|f| f.get("rope_theta")) {
                    object
                        .entry("global_rope_theta".to_owned())
                        .or_insert_with(|| full.clone());
                }
                if let Some(sliding) = rope
                    .get("sliding_attention")
                    .and_then(|f| f.get("rope_theta"))
                {
                    object
                        .entry("local_rope_theta".to_owned())
                        .or_insert_with(|| sliding.clone());
                }
            }
        }
        Ok(Self {
            vocab_size: usize_field(&resolved, "vocab_size", 102400)?,
            hidden_size: usize_field(&resolved, "hidden_size", 768)?,
            intermediate_size: usize_field(&resolved, "intermediate_size", 3072)?,
            num_hidden_layers: usize_field(&resolved, "num_hidden_layers", 25)?,
            num_attention_heads: usize_field(&resolved, "num_attention_heads", 12)?,
            hidden_activation: text_field(&resolved, "hidden_activation", "gelu"),
            norm_eps: f32_field(&resolved, "norm_eps", 1e-5),
            norm_bias: bool_field(&resolved, "norm_bias", false),
            attention_bias: bool_field(&resolved, "attention_bias", false),
            mlp_bias: bool_field(&resolved, "mlp_bias", false),
            global_attn_every_n_layers: usize_field(&resolved, "global_attn_every_n_layers", 3)?,
            local_attention: usize_field(&resolved, "local_attention", 128)?,
            global_rope_theta: f32_field(&resolved, "global_rope_theta", 160000.0),
            local_rope_theta: f32_field(&resolved, "local_rope_theta", 10000.0),
        })
    }

    pub fn is_global_layer(&self, layer: usize) -> bool {
        if self.global_attn_every_n_layers == 0 {
            return true;
        }
        layer % self.global_attn_every_n_layers == 0
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// LayerNorm with float32 accumulation and optional bias.
struct ModernLayerNorm {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    width: usize,
    eps: f32,
}

impl ModernLayerNorm {
    fn load(weights: &Weights, base: &str, width: usize, eps: f32, bias: bool) -> Result<Self> {
        Ok(Self {
            weight: weights.tensor(base, &[width])?,
            bias: if bias {
                Some(weights.raw_tensor(&format!("{base}.bias"), &[width])?)
            } else {
                None
            },
            width,
            eps,
        })
    }

    fn identity(width: usize) -> Self {
        Self {
            weight: Vec::new(),
            bias: None,
            width,
            eps: 0.0,
        }
    }

    fn is_identity(&self) -> bool {
        self.weight.is_empty()
    }

    fn apply(&self, x: &mut [f32], rows: usize) {
        if self.is_identity() {
            return;
        }
        for row in 0..rows {
            let base = row * self.width;
            let mut mean = 0.0f32;
            for value in &x[base..base + self.width] {
                mean += value;
            }
            mean /= self.width as f32;
            let mut var = 0.0f32;
            for value in &x[base..base + self.width] {
                var += (value - mean) * (value - mean);
            }
            var /= self.width as f32;
            let inv = 1.0 / (var + self.eps).sqrt();
            for (offset, value) in x[base..base + self.width].iter_mut().enumerate() {
                *value = (*value - mean) * inv * self.weight[offset]
                    + self.bias.as_ref().map_or(0.0, |b| b[offset]);
            }
        }
    }
}

struct ModernBertAttention {
    wqkv: Linear,
    wo: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl ModernBertAttention {
    fn load(weights: &Weights, prefix: &str, config: &ModernBertConfig) -> Result<Self> {
        Ok(Self {
            wqkv: weights.linear(
                &format!("{prefix}Wqkv"),
                config.hidden_size,
                3 * config.hidden_size,
                if config.attention_bias {
                    Bias::Required
                } else {
                    Bias::None
                },
            )?,
            wo: weights.linear(
                &format!("{prefix}Wo"),
                config.hidden_size,
                config.hidden_size,
                if config.attention_bias {
                    Bias::Required
                } else {
                    Bias::None
                },
            )?,
            num_heads: config.num_attention_heads,
            head_dim: config.head_dim(),
        })
    }

    /// HuggingFace-style rotate-half RoPE over `[rows, head_dim]` per
    /// head; the tables are `[rows, head_dim]` with the duplicated
    /// halves.
    fn apply_rotate_half(&self, x: &mut [f32], rows: usize, cos: &[f32], sin: &[f32]) {
        let half = self.head_dim / 2;
        for row in 0..rows {
            for h in 0..self.num_heads {
                let base = (row * self.num_heads + h) * self.head_dim;
                for i in 0..half {
                    let first = x[base + i];
                    let second = x[base + half + i];
                    x[base + i] = first * cos[row * self.head_dim + i]
                        - second * sin[row * self.head_dim + i];
                    x[base + half + i] = second * cos[row * self.head_dim + half + i]
                        + first * sin[row * self.head_dim + half + i];
                }
            }
        }
    }

    fn forward(
        &self,
        x: &[f32],
        rows: usize,
        cos: &[f32],
        sin: &[f32],
        mask: Option<&[f32]>,
    ) -> Vec<f32> {
        let width = self.num_heads * self.head_dim;
        let qkv = self.wqkv.forward(x, rows);
        let mut q = vec![0.0f32; rows * width];
        let mut k = vec![0.0f32; rows * width];
        let mut v = vec![0.0f32; rows * width];
        for row in 0..rows {
            q[row * width..(row + 1) * width]
                .copy_from_slice(&qkv[row * 3 * width..row * 3 * width + width]);
            k[row * width..(row + 1) * width]
                .copy_from_slice(&qkv[row * 3 * width + width..row * 3 * width + 2 * width]);
            v[row * width..(row + 1) * width]
                .copy_from_slice(&qkv[row * 3 * width + 2 * width..row * 3 * width + 3 * width]);
        }
        // The rope tables are per position over head_dim; q/k rows are
        // token-major so the table row matches the token index.
        self.apply_rotate_half(&mut q, rows, cos, sin);
        self.apply_rotate_half(&mut k, rows, cos, sin);

        // Rearrange to head-major per token for masked_attention: it
        // indexes (token, head) pairs.
        let out = masked_attention(&q, &k, &v, rows, rows, self.num_heads, self.head_dim, mask);
        self.wo.forward(&out, rows)
    }
}

struct ModernBertMlp {
    wi: Linear,
    wo: Linear,
    exact_gelu: bool,
}

impl ModernBertMlp {
    fn load(weights: &Weights, prefix: &str, config: &ModernBertConfig) -> Result<Self> {
        let activation = config.hidden_activation.to_lowercase();
        if !matches!(
            activation.as_str(),
            "gelu" | "gelu_new" | "gelu_pytorch_tanh"
        ) {
            return Err(bad_config(
                "hidden_activation",
                format!("unsupported ModernBERT activation {activation:?}"),
            ));
        }
        let bias = if config.mlp_bias {
            Bias::Required
        } else {
            Bias::None
        };
        Ok(Self {
            wi: weights.linear(
                &format!("{prefix}Wi"),
                config.hidden_size,
                2 * config.intermediate_size,
                bias,
            )?,
            wo: weights.linear(
                &format!("{prefix}Wo"),
                config.intermediate_size,
                config.hidden_size,
                bias,
            )?,
            exact_gelu: activation == "gelu",
        })
    }

    fn forward(&self, x: &mut [f32], rows: usize) {
        let gated = self.wi.forward(x, rows);
        let intermediate = gated.len() / (2 * rows);
        let mut act = vec![0.0f32; rows * intermediate];
        for row in 0..rows {
            for i in 0..intermediate {
                act[row * intermediate + i] = gated[row * 2 * intermediate + i];
            }
        }
        if self.exact_gelu {
            ops::gelu_erf(&mut act);
        } else {
            ops::gelu_tanh(&mut act);
        }
        let mut fused = vec![0.0f32; rows * intermediate];
        for row in 0..rows {
            for i in 0..intermediate {
                let gate = gated[row * 2 * intermediate + intermediate + i];
                fused[row * intermediate + i] = act[row * intermediate + i] * gate;
            }
        }
        let out = self.wo.forward(&fused, rows);
        x.copy_from_slice(&out);
    }
}

struct ModernBertLayer {
    attn_norm: ModernLayerNorm,
    attn: ModernBertAttention,
    mlp_norm: ModernLayerNorm,
    mlp: ModernBertMlp,
}

impl ModernBertLayer {
    fn load(
        weights: &Weights,
        prefix: &str,
        config: &ModernBertConfig,
        layer: usize,
    ) -> Result<Self> {
        let bias = config.norm_bias;
        let attn_norm = if layer == 0 {
            ModernLayerNorm::identity(config.hidden_size)
        } else {
            ModernLayerNorm::load(
                weights,
                &format!("{prefix}attn_norm"),
                config.hidden_size,
                config.norm_eps,
                bias,
            )?
        };
        Ok(Self {
            attn_norm,
            attn: ModernBertAttention::load(weights, &format!("{prefix}attn."), config)?,
            mlp_norm: ModernLayerNorm::load(
                weights,
                &format!("{prefix}mlp_norm"),
                config.hidden_size,
                config.norm_eps,
                bias,
            )?,
            mlp: ModernBertMlp::load(weights, &format!("{prefix}mlp."), config)?,
        })
    }

    fn forward(&self, x: &mut [f32], rows: usize, cos: &[f32], sin: &[f32], mask: Option<&[f32]>) {
        let mut attended = x.to_vec();
        self.attn_norm.apply(&mut attended, rows);
        let attention = self.attn.forward(&attended, rows, cos, sin, mask);
        for (value, add) in x.iter_mut().zip(attention) {
            *value += add;
        }
        let mut fed = x.to_vec();
        self.mlp_norm.apply(&mut fed, rows);
        self.mlp.forward(&mut fed, rows);
        for (value, add) in x.iter_mut().zip(fed) {
            *value += add;
        }
    }
}

/// `ModernBertEncoder`: bidirectional encoder with global/local
/// alternating attention windows.
pub struct ModernBertEncoder {
    tok_embeddings: Vec<f32>,
    embeddings_norm: ModernLayerNorm,
    layers: Vec<ModernBertLayer>,
    final_norm: ModernLayerNorm,
    config: ModernBertConfig,
}

impl ModernBertEncoder {
    fn load(weights: &Weights, prefix: &str, config: &ModernBertConfig) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|i| ModernBertLayer::load(weights, &format!("{prefix}layers.{i}."), config, i))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            tok_embeddings: weights.embedding(
                &format!("{prefix}embeddings.tok_embeddings"),
                config.vocab_size,
                config.hidden_size,
            )?,
            embeddings_norm: ModernLayerNorm::load(
                weights,
                &format!("{prefix}embeddings.norm"),
                config.hidden_size,
                config.norm_eps,
                config.norm_bias,
            )?,
            layers,
            final_norm: ModernLayerNorm::load(
                weights,
                &format!("{prefix}final_norm"),
                config.hidden_size,
                config.norm_eps,
                config.norm_bias,
            )?,
            config: config.clone(),
        })
    }

    fn forward(&self, input_ids: &[i32], mask: Option<&[bool]>) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let seq_len = input_ids.len();
        let mut x = crate::lm::engine::lookup_embeddings(
            &self.tok_embeddings,
            self.config.vocab_size,
            hidden,
            input_ids,
        )?;
        self.embeddings_norm.apply(&mut x, seq_len);

        let head_dim = self.config.head_dim();
        let (global_cos, global_sin) =
            rope_rotate_half_tables(head_dim, seq_len, self.config.global_rope_theta);
        let (local_cos, local_sin) =
            rope_rotate_half_tables(head_dim, seq_len, self.config.local_rope_theta);

        let global_mask = mask.map(|mask| {
            let row = additive_mask(mask);
            let mut full = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                full[i * seq_len..(i + 1) * seq_len].copy_from_slice(&row);
            }
            full
        });
        let local_mask = mask.map(|mask| self.local_additive_mask(mask, seq_len));

        for (i, layer) in self.layers.iter().enumerate() {
            let global = self.config.is_global_layer(i);
            let before = x.clone();
            if global {
                layer.forward(
                    &mut x,
                    seq_len,
                    &global_cos,
                    &global_sin,
                    global_mask.as_deref(),
                );
            } else {
                layer.forward(
                    &mut x,
                    seq_len,
                    &local_cos,
                    &local_sin,
                    local_mask.as_deref(),
                );
            }
            let _ = before;
        }
        self.final_norm.apply(&mut x, seq_len);
        Ok(x)
    }

    /// The local window additive mask: key mask AND window, with the
    /// diagonal always left open.
    fn local_additive_mask(&self, mask: &[bool], seq_len: usize) -> Vec<f32> {
        let window = self.config.local_attention / 2;
        let mut out = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let distance = (i as i64 - j as i64).unsigned_abs() as usize;
                let keep = (mask[j] && distance <= window) || distance == 0;
                out[i * seq_len + j] = if keep { 0.0 } else { f32::MIN };
            }
        }
        out
    }
}

/// HuggingFace-style rotate-half rope tables `[seq, head_dim]`.
fn rope_rotate_half_tables(head_dim: usize, seq_len: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0.0f32; seq_len * head_dim];
    let mut sin = vec![0.0f32; seq_len * head_dim];
    for position in 0..seq_len {
        for i in 0..half {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
            let angle = position as f32 * freq;
            for half_side in 0..2 {
                cos[position * head_dim + half_side * half + i] = angle.cos();
                sin[position * head_dim + half_side * half + i] = angle.sin();
            }
        }
    }
    (cos, sin)
}

// ---------------------------------------------------------------------------
// Condition encoders
// ---------------------------------------------------------------------------

struct TextBlock {
    attention_norm: Vec<f32>,
    attention: TextSelfAttention,
    mlp_norm: Vec<f32>,
    mlp: SwiGlu,
    norm_eps: f32,
}

impl TextBlock {
    fn load(
        weights: &Weights,
        prefix: &str,
        dim: usize,
        heads: usize,
        mlp_hidden: usize,
        norm_eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            attention_norm: weights.tensor(&format!("{prefix}attention_norm"), &[dim])?,
            attention: TextSelfAttention::load(
                weights,
                &format!("{prefix}attention."),
                dim,
                heads,
                norm_eps,
            )?,
            mlp_norm: weights.tensor(&format!("{prefix}mlp_norm"), &[dim])?,
            mlp: SwiGlu::load(weights, &format!("{prefix}mlp."), dim, mlp_hidden)?,
            norm_eps,
        })
    }

    fn forward(&self, x: &mut [f32], rows: usize, mask: Option<&[bool]>) -> Result<()> {
        let mut attended = x.to_vec();
        rms_norm(&mut attended, rows, &self.attention_norm, self.norm_eps);
        let attention = self.attention.forward(&attended, rows, mask)?;
        for (value, add) in x.iter_mut().zip(attention) {
            *value += add;
        }
        let mut fed = x.to_vec();
        rms_norm(&mut fed, rows, &self.mlp_norm, self.norm_eps);
        self.mlp.forward(&mut fed, rows);
        for (value, add) in x.iter_mut().zip(fed) {
            *value += add;
        }
        Ok(())
    }
}

/// Non-causal gated self-attention with per-head q/k norms and full
/// interleaved RoPE (encoder blocks).
struct TextSelfAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    gate: Linear,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    heads: usize,
    head_dim: usize,
    norm_eps: f32,
}

impl TextSelfAttention {
    fn load(
        weights: &Weights,
        prefix: &str,
        dim: usize,
        heads: usize,
        norm_eps: f32,
    ) -> Result<Self> {
        let head_dim = dim / heads;
        Ok(Self {
            wq: weights.linear(&format!("{prefix}wq"), dim, dim, Bias::None)?,
            wk: weights.linear(&format!("{prefix}wk"), dim, dim, Bias::None)?,
            wv: weights.linear(&format!("{prefix}wv"), dim, dim, Bias::None)?,
            wo: weights.linear(&format!("{prefix}wo"), dim, dim, Bias::None)?,
            gate: weights.linear(&format!("{prefix}gate"), dim, dim, Bias::None)?,
            q_norm: weights.raw_tensor(&format!("{prefix}q_norm.weight"), &[heads, head_dim])?,
            k_norm: weights.raw_tensor(&format!("{prefix}k_norm.weight"), &[heads, head_dim])?,
            heads,
            head_dim,
            norm_eps,
        })
    }

    fn forward(&self, x: &[f32], rows: usize, mask: Option<&[bool]>) -> Result<Vec<f32>> {
        let mut q = self.wq.forward(x, rows);
        let mut k = self.wk.forward(x, rows);
        let v = self.wv.forward(x, rows);
        let gate = self.gate.forward(x, rows);
        rms_norm_per_head(
            &mut q,
            rows,
            self.heads,
            self.head_dim,
            &self.q_norm,
            self.norm_eps,
        );
        rms_norm_per_head(
            &mut k,
            rows,
            self.heads,
            self.head_dim,
            &self.k_norm,
            self.norm_eps,
        );
        // The reference reshapes the projection to (heads, head_dim) and
        // rotates every head over head_dim with the (rows, head_dim/2)
        // tables.
        let (cos, sin) = rope_tables(self.head_dim, rows, 10000.0);
        apply_rope_interleaved_per_head(&mut q, rows, self.heads, self.head_dim, &cos, &sin);
        apply_rope_interleaved_per_head(&mut k, rows, self.heads, self.head_dim, &cos, &sin);
        let additive = mask.map(|mask| {
            // (Sq, Sk) broadcast of the same key mask.
            let mut full = vec![0.0f32; rows * rows];
            let row_mask = additive_mask(mask);
            for i in 0..rows {
                full[i * rows..(i + 1) * rows].copy_from_slice(&row_mask);
            }
            full
        });
        let out = masked_attention(
            &q,
            &k,
            &v,
            rows,
            rows,
            self.heads,
            self.head_dim,
            additive.as_deref(),
        );
        // Output gate: wo(out * sigmoid(gate)).
        let mut gated = out;
        for (value, gate) in gated.iter_mut().zip(&gate) {
            *value *= 1.0 / (1.0 + (-gate).exp());
        }
        Ok(self.wo.forward(&gated, rows))
    }
}

/// `TextEncoder`: embedding + non-causal blocks with mask zeroing after
/// every block. Used for text and caption in the scratch path.
struct TextEncoder {
    embedding: Vec<f32>,
    blocks: Vec<TextBlock>,
    vocab_size: usize,
    dim: usize,
}

impl TextEncoder {
    #[allow(
        clippy::too_many_arguments,
        reason = "the loader mirrors the reference constructor"
    )]
    fn load(
        weights: &Weights,
        prefix: &str,
        vocab_size: usize,
        dim: usize,
        heads: usize,
        layers: usize,
        mlp_ratio: f32,
        norm_eps: f32,
    ) -> Result<Self> {
        let mlp_hidden = (dim as f32 * mlp_ratio) as usize;
        Ok(Self {
            embedding: weights.embedding(&format!("{prefix}text_embedding"), vocab_size, dim)?,
            blocks: (0..layers)
                .map(|i| {
                    TextBlock::load(
                        weights,
                        &format!("{prefix}blocks.{i}."),
                        dim,
                        heads,
                        mlp_hidden,
                        norm_eps,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            vocab_size,
            dim,
        })
    }

    fn forward(&self, input_ids: &[i32], mask: Option<&[bool]>) -> Result<Vec<f32>> {
        let rows = input_ids.len();
        let mut x = crate::lm::engine::lookup_embeddings(
            &self.embedding,
            self.vocab_size,
            self.dim,
            input_ids,
        )?;
        if let Some(mask) = mask {
            for (row, &valid) in mask.iter().enumerate() {
                if valid {
                    continue;
                }
                for value in &mut x[row * self.dim..(row + 1) * self.dim] {
                    *value = 0.0;
                }
            }
            for block in &self.blocks {
                block.forward(&mut x, rows, Some(mask))?;
                for (row, &valid) in mask.iter().enumerate() {
                    if valid {
                        continue;
                    }
                    for value in &mut x[row * self.dim..(row + 1) * self.dim] {
                        *value = 0.0;
                    }
                }
            }
        } else {
            for block in &self.blocks {
                block.forward(&mut x, rows, None)?;
            }
        }
        Ok(x)
    }
}

/// `PretrainedConditionProjector` over the shared ModernBERT backbone
/// (`linear` or `residual_mlp`).
struct PretrainedProjector {
    projector: Linear,
    residual_norm: Option<Vec<f32>>,
    residual_up: Option<Linear>,
    residual_down: Option<Linear>,
    backbone_dim: usize,
    output_dim: usize,
    norm_eps: f32,
}

impl PretrainedProjector {
    fn load(
        weights: &Weights,
        prefix: &str,
        backbone_dim: usize,
        output_dim: usize,
        projector_type: &str,
        hidden_ratio: f32,
        norm_eps: f32,
    ) -> Result<Self> {
        let projector_type = projector_type.trim().to_lowercase();
        if !matches!(projector_type.as_str(), "linear" | "residual_mlp") {
            return Err(bad_config(
                "pretrained_projector_type",
                "must be 'linear' or 'residual_mlp'",
            ));
        }
        let (residual_norm, residual_up, residual_down) = if projector_type == "residual_mlp" {
            let hidden = ((output_dim as f32 * hidden_ratio).round() as usize).max(1);
            (
                Some(weights.tensor(&format!("{prefix}residual_norm"), &[backbone_dim])?),
                Some(weights.linear(
                    &format!("{prefix}residual_up"),
                    backbone_dim,
                    hidden,
                    Bias::Required,
                )?),
                Some(weights.linear(
                    &format!("{prefix}residual_down"),
                    hidden,
                    output_dim,
                    Bias::Required,
                )?),
            )
        } else {
            (None, None, None)
        };
        Ok(Self {
            projector: weights.linear(
                &format!("{prefix}projector"),
                backbone_dim,
                output_dim,
                Bias::Required,
            )?,
            residual_norm,
            residual_up,
            residual_down,
            backbone_dim,
            output_dim,
            norm_eps,
        })
    }

    fn forward(
        &self,
        backbone: &ModernBertEncoder,
        input_ids: &[i32],
        mask: Option<&[bool]>,
    ) -> Result<Vec<f32>> {
        let state = backbone.forward(input_ids, mask)?;
        let rows = state.len() / self.backbone_dim;
        let mut projected = self.projector.forward(&state, rows);
        if let (Some(norm), Some(up), Some(down)) =
            (&self.residual_norm, &self.residual_up, &self.residual_down)
        {
            let mut residual = state;
            rms_norm(&mut residual, rows, norm, self.norm_eps);
            let mut hidden = up.forward(&residual, rows);
            for value in &mut hidden {
                *value *= 1.0 / (1.0 + (-*value).exp());
            }
            let applied = down.forward(&hidden, rows);
            for (value, add) in projected.iter_mut().zip(applied) {
                *value += add;
            }
        }
        if let Some(mask) = mask {
            for (row, &valid) in mask.iter().enumerate() {
                if valid {
                    continue;
                }
                for value in &mut projected[row * self.output_dim..(row + 1) * self.output_dim] {
                    *value = 0.0;
                }
            }
        }
        Ok(projected)
    }
}

/// `ReferenceLatentEncoder` over patched speaker latents.
struct ReferenceLatentEncoder {
    in_proj: Linear,
    blocks: Vec<TextBlock>,
    dim: usize,
}

impl ReferenceLatentEncoder {
    #[allow(
        clippy::too_many_arguments,
        reason = "the loader mirrors the reference constructor"
    )]
    fn load(
        weights: &Weights,
        prefix: &str,
        in_dim: usize,
        dim: usize,
        heads: usize,
        layers: usize,
        mlp_ratio: f32,
        norm_eps: f32,
    ) -> Result<Self> {
        let mlp_hidden = (dim as f32 * mlp_ratio) as usize;
        Ok(Self {
            in_proj: weights.linear(&format!("{prefix}in_proj"), in_dim, dim, Bias::Required)?,
            blocks: (0..layers)
                .map(|i| {
                    TextBlock::load(
                        weights,
                        &format!("{prefix}blocks.{i}."),
                        dim,
                        heads,
                        mlp_hidden,
                        norm_eps,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            dim,
        })
    }

    fn forward(&self, latent: &[f32], rows: usize, mask: Option<&[bool]>) -> Result<Vec<f32>> {
        let mut x = self.in_proj.forward(latent, rows);
        for value in &mut x {
            *value /= 6.0;
        }
        if let Some(mask) = mask {
            for (row, &valid) in mask.iter().enumerate() {
                if valid {
                    continue;
                }
                for value in &mut x[row * self.dim..(row + 1) * self.dim] {
                    *value = 0.0;
                }
            }
            for block in &self.blocks {
                block.forward(&mut x, rows, Some(mask))?;
                for (row, &valid) in mask.iter().enumerate() {
                    if valid {
                        continue;
                    }
                    for value in &mut x[row * self.dim..(row + 1) * self.dim] {
                        *value = 0.0;
                    }
                }
            }
        } else {
            for block in &self.blocks {
                block.forward(&mut x, rows, None)?;
            }
        }
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// DiT components
// ---------------------------------------------------------------------------

/// `LowRankAdaLN`: shift/scale/gate from the timestep embedding with the
/// residual branch on each projection, tanh gate.
struct LowRankAdaLn {
    shift_down: Linear,
    scale_down: Linear,
    gate_down: Linear,
    shift_up: Linear,
    scale_up: Linear,
    gate_up: Linear,
    dim: usize,
    eps: f32,
}

impl LowRankAdaLn {
    fn load(weights: &Weights, prefix: &str, dim: usize, rank: usize, eps: f32) -> Result<Self> {
        let rank = rank.clamp(1, dim);
        Ok(Self {
            shift_down: weights.linear(&format!("{prefix}shift_down"), dim, rank, Bias::None)?,
            scale_down: weights.linear(&format!("{prefix}scale_down"), dim, rank, Bias::None)?,
            gate_down: weights.linear(&format!("{prefix}gate_down"), dim, rank, Bias::None)?,
            shift_up: weights.linear(&format!("{prefix}shift_up"), rank, dim, Bias::Required)?,
            scale_up: weights.linear(&format!("{prefix}scale_up"), rank, dim, Bias::Required)?,
            gate_up: weights.linear(&format!("{prefix}gate_up"), rank, dim, Bias::Required)?,
            dim,
            eps,
        })
    }

    /// Returns the conditioned `(normalized x, gate)` pair over the
    /// broadcast branch arrays.
    fn forward(&self, x: &[f32], rows: usize, cond: &LowRankBranch) -> (Vec<f32>, Vec<f32>) {
        let mut out = x.to_vec();
        for row in 0..rows {
            // RMSNorm over the row, then the scale/shift broadcast.
            let base = row * self.dim;
            let mut sum = 0.0f32;
            for value in &out[base..base + self.dim] {
                sum += value * value;
            }
            let norm = 1.0 / (sum / self.dim as f32 + self.eps).sqrt();
            for (d, value) in out[base..base + self.dim].iter_mut().enumerate() {
                *value = *value * norm * (1.0 + cond.scale[base + d]) + cond.shift[base + d];
            }
        }
        (out, cond.gate.clone())
    }
}

/// One precomputed AdaLN branch (shift, scale, gate) over `rows` rows.
struct LowRankBranch {
    shift: Vec<f32>,
    scale: Vec<f32>,
    gate: Vec<f32>,
}

impl LowRankAdaLn {
    fn branch(&self, cond_row: &[f32], rows: usize) -> LowRankBranch {
        let dim = self.dim;
        let third = cond_row.len() / 3;
        let branch = |down: &Linear, up: &Linear, slice: &[f32]| -> Vec<f32> {
            // up(down(silu(slice))) + slice.
            let mut activated = slice.to_vec();
            for value in &mut activated {
                *value *= 1.0 / (1.0 + (-*value).exp());
            }
            let downed = down.forward(&activated, 1);
            let mut projected = up.forward(&downed, 1);
            for (value, residual) in projected.iter_mut().zip(slice) {
                *value += residual;
            }
            // Broadcast the single conditioning row to every sequence row.
            let mut full = Vec::with_capacity(rows * dim);
            for _ in 0..rows {
                full.extend_from_slice(&projected);
            }
            full
        };
        let shift = branch(&self.shift_down, &self.shift_up, &cond_row[..third]);
        let scale = branch(
            &self.scale_down,
            &self.scale_up,
            &cond_row[third..2 * third],
        );
        let gate = branch(&self.gate_down, &self.gate_up, &cond_row[2 * third..]);
        LowRankBranch {
            shift,
            scale,
            gate: gate.iter().map(|g| g.tanh()).collect(),
        }
    }
}

/// One precomputed KV pair for a context branch.
#[derive(Clone)]
struct KvPair {
    k: Vec<f32>,
    v: Vec<f32>,
}

/// `JointAttention`: latent self-tokens attend themselves plus text and
/// speaker/caption context. Half interleaved RoPE on the self q/k.
struct JointAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wk_text: Linear,
    wv_text: Linear,
    wk_speaker: Option<Linear>,
    wv_speaker: Option<Linear>,
    wk_caption: Option<Linear>,
    wv_caption: Option<Linear>,
    gate: Linear,
    wo: Linear,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    heads: usize,
    head_dim: usize,
    dim: usize,
    norm_eps: f32,
}

impl JointAttention {
    #[allow(
        clippy::too_many_arguments,
        reason = "the loader mirrors the reference constructor"
    )]
    fn load(
        weights: &Weights,
        prefix: &str,
        dim: usize,
        heads: usize,
        text_ctx_dim: usize,
        speaker_ctx_dim: Option<usize>,
        caption_ctx_dim: Option<usize>,
        norm_eps: f32,
    ) -> Result<Self> {
        let head_dim = dim / heads;
        let (wk_speaker, wv_speaker) = match speaker_ctx_dim {
            Some(ctx) => (
                Some(weights.linear(&format!("{prefix}wk_speaker"), ctx, dim, Bias::None)?),
                Some(weights.linear(&format!("{prefix}wv_speaker"), ctx, dim, Bias::None)?),
            ),
            None => (None, None),
        };
        let (wk_caption, wv_caption) = match caption_ctx_dim {
            Some(ctx) => (
                Some(weights.linear(&format!("{prefix}wk_caption"), ctx, dim, Bias::None)?),
                Some(weights.linear(&format!("{prefix}wv_caption"), ctx, dim, Bias::None)?),
            ),
            None => (None, None),
        };
        Ok(Self {
            wq: weights.linear(&format!("{prefix}wq"), dim, dim, Bias::None)?,
            wk: weights.linear(&format!("{prefix}wk"), dim, dim, Bias::None)?,
            wv: weights.linear(&format!("{prefix}wv"), dim, dim, Bias::None)?,
            wk_text: weights.linear(&format!("{prefix}wk_text"), text_ctx_dim, dim, Bias::None)?,
            wv_text: weights.linear(&format!("{prefix}wv_text"), text_ctx_dim, dim, Bias::None)?,
            wk_speaker,
            wv_speaker,
            wk_caption,
            wv_caption,
            gate: weights.linear(&format!("{prefix}gate"), dim, dim, Bias::None)?,
            wo: weights.linear(&format!("{prefix}wo"), dim, dim, Bias::None)?,
            q_norm: weights.raw_tensor(&format!("{prefix}q_norm.weight"), &[heads, head_dim])?,
            k_norm: weights.raw_tensor(&format!("{prefix}k_norm.weight"), &[heads, head_dim])?,
            heads,
            head_dim,
            dim,
            norm_eps,
        })
    }

    /// Applies the reference `_apply_rotary_half`: RoPE over the FIRST
    /// HALF OF THE HEADS only (`y[..., :y.shape[-2] // 2, :]` slices the
    /// heads axis), with a full head-dim interleaved rotation.
    fn apply_rope_half(&self, x: &mut [f32], rows: usize, cos: &[f32], sin: &[f32]) {
        let rot_heads = self.heads / 2;
        let half = self.head_dim / 2;
        for row in 0..rows {
            for h in 0..rot_heads {
                let base = (row * self.heads + h) * self.head_dim;
                for i in 0..half {
                    let even = x[base + 2 * i];
                    let odd = x[base + 2 * i + 1];
                    x[base + 2 * i] = even * cos[row * half + i] - odd * sin[row * half + i];
                    x[base + 2 * i + 1] = odd * cos[row * half + i] + even * sin[row * half + i];
                }
            }
        }
    }

    fn kv_text(&self, state: &[f32], rows: usize) -> KvPair {
        let mut k = self.wk_text.forward(state, rows);
        let v = self.wv_text.forward(state, rows);
        rms_norm_per_head(
            &mut k,
            rows,
            self.heads,
            self.head_dim,
            &self.k_norm,
            self.norm_eps,
        );
        KvPair { k, v }
    }

    fn kv_speaker(&self, state: &[f32], rows: usize) -> Result<KvPair> {
        let wk = self
            .wk_speaker
            .as_ref()
            .ok_or_else(|| SpeechError::Unsupported {
                why: "speaker KV requested from a caption-only attention".into(),
            })?;
        let wv = self.wv_speaker.as_ref().unwrap();
        let mut k = wk.forward(state, rows);
        let v = wv.forward(state, rows);
        rms_norm_per_head(
            &mut k,
            rows,
            self.heads,
            self.head_dim,
            &self.k_norm,
            self.norm_eps,
        );
        Ok(KvPair { k, v })
    }

    fn kv_caption(&self, state: &[f32], rows: usize) -> Result<KvPair> {
        let wk = self
            .wk_caption
            .as_ref()
            .ok_or_else(|| SpeechError::Unsupported {
                why: "caption KV requested from a speaker-only attention".into(),
            })?;
        let wv = self.wv_caption.as_ref().unwrap();
        let mut k = wk.forward(state, rows);
        let v = wv.forward(state, rows);
        rms_norm_per_head(
            &mut k,
            rows,
            self.heads,
            self.head_dim,
            &self.k_norm,
            self.norm_eps,
        );
        Ok(KvPair { k, v })
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "joint attention carries its four branches explicitly"
    )]
    fn forward(
        &self,
        x: &[f32],
        rows: usize,
        text_mask: &[bool],
        freqs: &RopeTables,
        kv_text: &KvPair,
        kv_speaker: Option<&KvPair>,
        speaker_mask: Option<&[bool]>,
        kv_caption: Option<&KvPair>,
        caption_mask: Option<&[bool]>,
        start_pos: usize,
    ) -> Vec<f32> {
        let mut q = self.wq.forward(x, rows);
        let mut k_self = self.wk.forward(x, rows);
        let v_self = self.wv.forward(x, rows);
        let gate = self.gate.forward(x, rows);
        rms_norm_per_head(
            &mut q,
            rows,
            self.heads,
            self.head_dim,
            &self.q_norm,
            self.norm_eps,
        );
        rms_norm_per_head(
            &mut k_self,
            rows,
            self.heads,
            self.head_dim,
            &self.k_norm,
            self.norm_eps,
        );
        let (q_cos, q_sin) = freqs.slice(start_pos, rows);
        self.apply_rope_half(&mut q, rows, &q_cos, &q_sin);
        self.apply_rope_half(&mut k_self, rows, &q_cos, &q_sin);

        // Concatenate the self, text, and context branches along keys.
        let text_rows = kv_text.k.len() / self.dim;
        let speaker_rows = kv_speaker.map_or(0, |kv| kv.k.len() / self.dim);
        let caption_rows = kv_caption.map_or(0, |kv| kv.k.len() / self.dim);
        let total = rows + text_rows + speaker_rows + caption_rows;
        let mut k = vec![0.0f32; total * self.dim];
        let mut v = vec![0.0f32; total * self.dim];
        let mut valid = vec![true; total];
        let mut offset = 0usize;
        k[..rows * self.dim].copy_from_slice(&k_self);
        v[..rows * self.dim].copy_from_slice(&v_self);
        valid[..rows].fill(true);
        offset += rows;
        k[offset * self.dim..(offset + text_rows) * self.dim].copy_from_slice(&kv_text.k);
        v[offset * self.dim..(offset + text_rows) * self.dim].copy_from_slice(&kv_text.v);
        valid[offset..offset + text_rows].copy_from_slice(text_mask);
        offset += text_rows;
        if let (Some(kv), Some(mask)) = (kv_speaker, speaker_mask) {
            k[offset * self.dim..(offset + speaker_rows) * self.dim].copy_from_slice(&kv.k);
            v[offset * self.dim..(offset + speaker_rows) * self.dim].copy_from_slice(&kv.v);
            valid[offset..offset + speaker_rows].copy_from_slice(mask);
            offset += speaker_rows;
        }
        if let (Some(kv), Some(mask)) = (kv_caption, caption_mask) {
            k[offset * self.dim..(offset + caption_rows) * self.dim].copy_from_slice(&kv.k);
            v[offset * self.dim..(offset + caption_rows) * self.dim].copy_from_slice(&kv.v);
            valid[offset..offset + caption_rows].copy_from_slice(mask);
        }

        // Additive (Sq, Sk) mask: self rows always visible, context rows
        // by their masks.
        let mut mask = vec![0.0f32; rows * total];
        for i in 0..rows {
            for (j, &visible) in valid.iter().enumerate() {
                mask[i * total + j] = if visible { 0.0 } else { -1e9 };
            }
        }
        let out = masked_attention(
            &q,
            &k,
            &v,
            rows,
            total,
            self.heads,
            self.head_dim,
            Some(&mask),
        );
        let mut gated = out;
        for (value, g) in gated.iter_mut().zip(&gate) {
            *value *= 1.0 / (1.0 + (-g).exp());
        }
        self.wo.forward(&gated, rows)
    }
}

/// The precomputed interleaved RoPE tables for one forward pass.
struct RopeTables {
    cos: Vec<f32>,
    sin: Vec<f32>,
    half: usize,
}

impl RopeTables {
    fn new(head_dim: usize, end: usize) -> Self {
        let (cos, sin) = rope_tables(head_dim, end, 10000.0);
        Self {
            cos,
            sin,
            half: head_dim / 2,
        }
    }

    fn slice(&self, start: usize, len: usize) -> (Vec<f32>, Vec<f32>) {
        (
            self.cos[start * self.half..(start + len) * self.half].to_vec(),
            self.sin[start * self.half..(start + len) * self.half].to_vec(),
        )
    }
}

struct DiffusionBlock {
    attention: JointAttention,
    mlp: SwiGlu,
    attention_adaln: LowRankAdaLn,
    mlp_adaln: LowRankAdaLn,
    dim: usize,
}

impl DiffusionBlock {
    #[allow(
        clippy::too_many_arguments,
        reason = "the block forwards its four conditioning branches"
    )]
    fn forward(
        &self,
        x: &mut [f32],
        cond_row: &[f32],
        rows: usize,
        text_mask: &[bool],
        freqs: &RopeTables,
        kv_text: &KvPair,
        kv_speaker: Option<&KvPair>,
        speaker_mask: Option<&[bool]>,
        kv_caption: Option<&KvPair>,
        caption_mask: Option<&[bool]>,
        start_pos: usize,
    ) {
        let attn_cond = self.attention_adaln.branch(cond_row, rows);
        let (x_norm, gate) = self.attention_adaln.forward(x, rows, &attn_cond);
        let attended = self.attention.forward(
            &x_norm,
            rows,
            text_mask,
            freqs,
            kv_text,
            kv_speaker,
            speaker_mask,
            kv_caption,
            caption_mask,
            start_pos,
        );
        for ((value, add), g) in x.iter_mut().zip(&attended).zip(&gate) {
            *value += g * add;
        }
        let mlp_cond = self.mlp_adaln.branch(cond_row, rows);
        let (x_norm, gate) = self.mlp_adaln.forward(x, rows, &mlp_cond);
        let mut fed = x_norm;
        self.mlp.forward(&mut fed, rows);
        for ((value, add), g) in x.iter_mut().zip(&fed).zip(&gate) {
            *value += g * add;
        }
        let _ = self.dim;
    }
}

// ---------------------------------------------------------------------------
// Duration predictor
// ---------------------------------------------------------------------------

/// `build_duration_features` over the CALLER-normalized text (the
/// reference runs `normalize_text`, whose NFKC pass is caller-owned
/// here). Returns the 14-dim feature row.
pub fn build_duration_features(
    normalized_text: &str,
    token_count: usize,
    max_text_len: usize,
    has_speaker: bool,
) -> Vec<f32> {
    let chars: Vec<char> = normalized_text.chars().collect();
    let char_count = chars.len().max(1) as f32;
    let kana_count = chars.iter().filter(|c| is_kana(**c)).count() as f32;
    let kanji_count = chars.iter().filter(|c| is_kanji(**c)).count() as f32;
    let alnum_count = chars.iter().filter(|c| is_ascii_alnum(**c)).count() as f32;
    let emoji_count = count_annotation_emojis(normalized_text) as f32;
    let count = |needle: &str| normalized_text.matches(needle).count() as f32;
    let period_count = count("\u{3002}") + count(".");
    let comma_count = count("\u{3001}") + count(",");
    let long_vowel_count = count("\u{30FC}");
    let ellipsis_count = count("\u{2026}");
    let exclamation_count = count("\u{FF01}") + count("!");
    let question_count = count("\u{FF1F}") + count("?");
    vec![
        (token_count as f32).clamp(0.0, max_text_len as f32) / max_text_len as f32,
        log1p_cap(char_count, 512.0),
        token_count as f32 / char_count,
        log1p_cap(period_count, 8.0),
        log1p_cap(comma_count, 16.0),
        log1p_cap(long_vowel_count, 8.0),
        log1p_cap(ellipsis_count, 8.0),
        log1p_cap(exclamation_count, 8.0),
        log1p_cap(question_count, 8.0),
        log1p_cap(emoji_count, 8.0),
        kana_count / char_count,
        kanji_count / char_count,
        alnum_count / char_count,
        if has_speaker { 1.0 } else { 0.0 },
    ]
}

fn log1p_cap(value: f32, cap: f32) -> f32 {
    value.clamp(0.0, cap).ln_1p() / cap.ln_1p()
}

fn is_kana(ch: char) -> bool {
    let code = ch as u32;
    (0x3040..=0x309F).contains(&code) || (0x30A0..=0x30FF).contains(&code)
}

fn is_kanji(ch: char) -> bool {
    let code = ch as u32;
    (0x3400..=0x4DBF).contains(&code)
        || (0x4E00..=0x9FFF).contains(&code)
        || (0xF900..=0xFAFF).contains(&code)
        || (0x20000..=0x2FA1F).contains(&code)
}

fn is_ascii_alnum(ch: char) -> bool {
    ch.is_ascii_alphanumeric()
}

/// The annotation-emoji allowlist (`ALLOWED_ANNOTATION_EMOJIS`).
const ANNOTATION_EMOJIS: &[&str] = &[
    "\u{23E9}",                   // fast-forwards
    "\u{23F1}\u{FE0F}",           // stopwatch
    "\u{23F8}\u{FE0F}",           // pause
    "\u{1F32C}\u{FE0F}",          // wind face
    "\u{1F36D}",                  // lollipop
    "\u{1F39B}\u{FE0F}",          // control knobs
    "\u{1F3AD}",                  // performing arts
    "\u{1F3B5}",                  // musical note
    "\u{1F422}",                  // turtle
    "\u{1F431}",                  // cat face
    "\u{1F442}",                  // ear
    "\u{1F443}",                  // nose
    "\u{1F445}",                  // tongue
    "\u{1F44C}",                  // ok hand
    "\u{1F44F}",                  // clap
    "\u{1F48B}",                  // kiss mark
    "\u{1F4A5}",                  // collision
    "\u{1F4A6}",                  // sweat droplets
    "\u{1F4AA}",                  // flexed biceps
    "\u{1F4C4}",                  // page facing up
    "\u{1F4DE}",                  // telephone receiver
    "\u{1F4E2}",                  // loudspeaker
    "\u{1F4E3}",                  // megaphone
    "\u{1F606}",                  // grinning squinting face
    "\u{1F60A}",                  // smiling face
    "\u{1F60C}",                  // relieved face
    "\u{1F60E}",                  // smiling face with sunglasses
    "\u{1F60F}",                  // smirking face
    "\u{1F612}",                  // unamused face
    "\u{1F616}",                  // confounded face
    "\u{1F620}",                  // angry face
    "\u{1F62A}",                  // sleepy face
    "\u{1F62D}",                  // loudly crying face
    "\u{1F62E}",                  // face with open mouth
    "\u{1F62E}\u{200D}\u{1F4A8}", // face exhaling
    "\u{1F630}",                  // anxious face with sweat
    "\u{1F631}",                  // face screaming in fear
    "\u{1F632}",                  // astonished face
    "\u{1F634}",                  // sleeping face
    "\u{1F644}",                  // face with rolling eyes
    "\u{1F64F}",                  // folded hands
    "\u{1F910}",                  // zipper-mouth face
    "\u{1F914}",                  // thinking face
    "\u{1F922}",                  // nauseated face
    "\u{1F927}",                  // sneezing face
    "\u{1F92D}",                  // face with hand over mouth
    "\u{1F964}",                  // cup with straw
    "\u{1F971}",                  // yawning face
    "\u{1F974}",                  // woozy face
    "\u{1F975}",                  // hot face
    "\u{1F979}",                  // face holding back tears
    "\u{1F97A}",                  // pleading face
    "\u{1FAE3}",                  // face holding back tears (newer)
    "\u{1FAF6}",                  // heart hands
    "\u{1F4D6}",                  // open book
];

/// Longest-first non-overlapping emoji count (the reference regex
/// alternation semantics).
fn count_annotation_emojis(text: &str) -> usize {
    let chars: Vec<char> = text.chars().collect();
    let mut count = 0usize;
    let mut position = 0usize;
    'scan: while position < chars.len() {
        // Longest candidates first: the reference sorts by escaped
        // length, longest match wins at each position.
        let mut best: Option<usize> = None;
        let mut best_len = 0usize;
        for emoji in ANNOTATION_EMOJIS {
            let candidate: Vec<char> = emoji.chars().collect();
            if candidate.len() > best_len
                && position + candidate.len() <= chars.len()
                && chars[position..position + candidate.len()] == candidate[..]
            {
                best = Some(candidate.len());
                best_len = candidate.len();
            }
        }
        if let Some(len) = best {
            count += 1;
            position += len;
            continue 'scan;
        }
        position += 1;
    }
    count
}

/// AdaRN-Zero SwiGLU block for the token-sum duration predictor
/// (speaker and optional caption modulation summed).
struct DurationSwiGluBlock {
    norm_weight: Vec<f32>,
    mlp: SwiGlu,
    modulation: Option<Linear>,
    caption_modulation: Option<Linear>,
    dim: usize,
    norm_eps: f32,
}

impl DurationSwiGluBlock {
    fn load(
        weights: &Weights,
        prefix: &str,
        dim: usize,
        hidden: usize,
        norm_eps: f32,
        cond_dim: Option<usize>,
        caption_cond_dim: Option<usize>,
    ) -> Result<Self> {
        Ok(Self {
            norm_weight: weights.tensor(&format!("{prefix}norm"), &[dim])?,
            mlp: SwiGlu::load(weights, &format!("{prefix}mlp."), dim, hidden)?,
            modulation: cond_dim
                .map(|cond| {
                    weights.linear(
                        &format!("{prefix}modulation"),
                        cond,
                        dim * 3,
                        Bias::Required,
                    )
                })
                .transpose()?,
            caption_modulation: caption_cond_dim
                .map(|cond| {
                    weights.linear(
                        &format!("{prefix}caption_modulation"),
                        cond,
                        dim * 3,
                        Bias::Required,
                    )
                })
                .transpose()?,
            dim,
            norm_eps,
        })
    }

    fn split3(values: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let third = values.len() / 3;
        (
            values[..third].to_vec(),
            values[third..2 * third].to_vec(),
            values[2 * third..].to_vec(),
        )
    }

    fn forward(&self, x: &[f32], cond: Option<&[f32]>, caption_cond: Option<&[f32]>) -> Vec<f32> {
        let rows = x.len() / self.dim;
        let mut h = x.to_vec();
        rms_norm(&mut h, rows, &self.norm_weight, self.norm_eps);
        if self.modulation.is_some() || self.caption_modulation.is_some() {
            let mut shift = vec![0.0f32; h.len()];
            let mut scale = vec![0.0f32; h.len()];
            let mut gate = vec![0.0f32; h.len()];
            let apply_mod = |linear: &Linear,
                             values: &[f32],
                             shift: &mut [f32],
                             scale: &mut [f32],
                             gate: &mut [f32]| {
                let mut activated = linear.forward(values, 1);
                for value in &mut activated {
                    *value *= 1.0 / (1.0 + (-*value).exp());
                }
                let (ds, dsc, dg) = Self::split3(&activated);
                for row in 0..rows {
                    for d in 0..self.dim {
                        shift[row * self.dim + d] += ds[d];
                        scale[row * self.dim + d] += dsc[d];
                        gate[row * self.dim + d] += dg[d];
                    }
                }
            };
            if let (Some(linear), Some(values)) = (&self.modulation, cond) {
                apply_mod(linear, values, &mut shift, &mut scale, &mut gate);
            }
            if let (Some(linear), Some(values)) = (&self.caption_modulation, caption_cond) {
                apply_mod(linear, values, &mut shift, &mut scale, &mut gate);
            }
            for row in 0..rows {
                for d in 0..self.dim {
                    let index = row * self.dim + d;
                    h[index] = h[index] * (1.0 + scale[index]) + shift[index];
                    gate[index] = gate[index].tanh();
                }
            }
            let mut out = x.to_vec();
            self.mlp.forward(&mut h, rows);
            for (value, (add, g)) in out.iter_mut().zip(h.iter().zip(&gate)) {
                *value += g * add;
            }
            out
        } else {
            self.mlp.forward(&mut h, rows);
            let mut out = x.to_vec();
            for (value, add) in out.iter_mut().zip(&h) {
                *value += add;
            }
            out
        }
    }
}

/// The token-sum duration predictor paths (the v3/v4 default
/// architectures; pooled variants are refused at load).
struct DurationPredictor {
    token_input_proj: Linear,
    token_blocks: Vec<DurationSwiGluBlock>,
    token_out_norm: Vec<f32>,
    token_out_proj: Linear,
    speaker_dim: Option<usize>,
    caption_dim: Option<usize>,
    text_dim: usize,
    dual: bool,
    norm_eps: f32,
}

impl DurationPredictor {
    #[allow(
        clippy::too_many_arguments,
        reason = "the constructor mirrors the reference signature"
    )]
    fn load(
        weights: &Weights,
        prefix: &str,
        text_dim: usize,
        hidden_dim: usize,
        layers: usize,
        norm_eps: f32,
        speaker_dim: Option<usize>,
        caption_dim: Option<usize>,
        architecture: &str,
    ) -> Result<Self> {
        let dual = match architecture {
            "token_sum_adarn_zero_no_aux" => false,
            "token_sum_dual_adarn_zero_no_aux" => true,
            other => {
                return Err(SpeechError::Unsupported {
                    why: format!(
                        "duration_architecture {other:?}: the port implements the token-sum paths"
                    ),
                })
            }
        };
        Ok(Self {
            token_input_proj: weights.linear(
                &format!("{prefix}token_input_proj"),
                text_dim,
                hidden_dim,
                Bias::Required,
            )?,
            token_blocks: (0..layers)
                .map(|i| {
                    DurationSwiGluBlock::load(
                        weights,
                        &format!("{prefix}token_blocks.{i}."),
                        hidden_dim,
                        hidden_dim,
                        norm_eps,
                        speaker_dim,
                        if dual { caption_dim } else { None },
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            token_out_norm: weights.tensor(&format!("{prefix}token_out_norm"), &[hidden_dim])?,
            token_out_proj: weights.linear(
                &format!("{prefix}token_out_proj"),
                hidden_dim,
                1,
                Bias::Required,
            )?,
            speaker_dim,
            caption_dim: if dual { caption_dim } else { None },
            text_dim,
            dual,
            norm_eps,
        })
    }

    /// Returns `log1p(total_frames)` for the batch of one.
    #[allow(
        clippy::too_many_arguments,
        reason = "the forward mirrors the reference signature"
    )]
    fn forward(
        &self,
        text_state: &[f32],
        text_mask: &[bool],
        speaker_state: Option<&[f32]>,
        has_speaker: bool,
        caption_state: Option<&[f32]>,
        caption_mask: Option<&[bool]>,
        has_caption: bool,
    ) -> Result<f32> {
        let rows = text_state.len() / self.text_dim;
        let speaker_vec = match (self.speaker_dim, speaker_state) {
            (Some(dim), Some(state)) => {
                if has_speaker && !state.is_empty() {
                    state[..dim].to_vec()
                } else {
                    vec![0.0f32; dim]
                }
            }
            (Some(dim), None) => vec![0.0f32; dim],
            _ => {
                return Err(SpeechError::Unsupported {
                    why: "speaker modules missing".into(),
                })
            }
        };
        let caption_vec = if self.dual {
            match (self.caption_dim, caption_state) {
                (Some(dim), Some(state)) => {
                    if has_caption {
                        // Masked mean over the caption sequence.
                        let rows_caption = state.len() / dim;
                        let mask_f: Vec<f32> = caption_mask
                            .unwrap_or(&vec![true; rows_caption])
                            .iter()
                            .map(|&valid| if valid { 1.0 } else { 0.0 })
                            .collect();
                        let denom = mask_f.iter().sum::<f32>().max(1.0);
                        let mut mean = vec![0.0f32; dim];
                        for (row, &weight) in mask_f.iter().enumerate() {
                            for d in 0..dim {
                                mean[d] += weight * state[row * dim + d];
                            }
                        }
                        for value in &mut mean {
                            *value /= denom;
                        }
                        mean
                    } else {
                        vec![0.0f32; dim]
                    }
                }
                (Some(dim), None) => vec![0.0f32; dim],
                _ => {
                    return Err(SpeechError::Unsupported {
                        why: "caption modules missing".into(),
                    })
                }
            }
        } else {
            Vec::new()
        };

        let mut h = self.token_input_proj.forward(text_state, rows);
        for block in &self.token_blocks {
            h = block.forward(
                &h,
                Some(&speaker_vec),
                if self.dual { Some(&caption_vec) } else { None },
            );
        }
        rms_norm(&mut h, rows, &self.token_out_norm, self.norm_eps);
        let logits = self.token_out_proj.forward(&h, rows);
        let mut total = 0.0f32;
        for (row, &valid) in text_mask.iter().enumerate() {
            if !valid {
                continue;
            }
            let value = logits[row];
            // softplus == log(1 + exp(x)), matching the reference f32 math.
            total += (1.0 + value.exp()).ln();
        }
        Ok(total.max(0.0).ln_1p())
    }
}

// ---------------------------------------------------------------------------
// The DiT
// ---------------------------------------------------------------------------

/// `IrodoriDiT`: the loaded diffusion transformer plus its condition
/// encoders.
pub struct IrodoriDiT {
    config: DitConfig,
    text_backbone: Option<ModernBertEncoder>,
    text_encoder: Option<TextEncoder>,
    text_projector: Option<PretrainedProjector>,
    caption_encoder: Option<TextEncoder>,
    caption_projector: Option<PretrainedProjector>,
    speaker_encoder: Option<ReferenceLatentEncoder>,
    text_norm: Vec<f32>,
    speaker_norm: Option<Vec<f32>>,
    caption_norm: Option<Vec<f32>>,
    duration_predictor: Option<DurationPredictor>,
    cond_module: Vec<Linear>,
    delta_cond_module: Option<Vec<Linear>>,
    in_proj: Linear,
    blocks: Vec<DiffusionBlock>,
    out_norm: Vec<f32>,
    out_proj: Linear,
    head_dim: usize,
}

/// The encoded conditioning bundle (`encode_conditions_full`).
pub struct Conditions {
    pub text_state: Vec<f32>,
    pub text_mask: Vec<bool>,
    pub speaker_state: Option<Vec<f32>>,
    pub speaker_mask: Option<Vec<bool>>,
    pub caption_state: Option<Vec<f32>>,
    pub caption_mask: Option<Vec<bool>>,
}

/// Per-layer context KV caches (`build_kv_cache`).
pub struct KvCaches {
    text: Vec<KvPair>,
    speaker: Option<Vec<KvPair>>,
    caption: Option<Vec<KvPair>>,
}

impl IrodoriDiT {
    fn load(weights: &Weights, config: &DitConfig) -> Result<Self> {
        if !matches!(
            config.flow_parameterization.as_str(),
            "rf_velocity" | "meanflow"
        ) {
            return Err(bad_config(
                "flow_parameterization",
                "must be 'rf_velocity' or 'meanflow'",
            ));
        }
        let prefix = "model.";
        let head_dim = config.model_dim / config.num_heads;

        let (text_backbone, text_encoder, text_projector) = if config.use_pretrained_text_encoder()
        {
            let encoder_config_value = config.text_encoder_config.clone().ok_or_else(|| {
                bad_config(
                    "text_encoder_config",
                    "text_encoder_type='pretrained' requires text_encoder_config",
                )
            })?;
            let backbone_config = ModernBertConfig::from_json(&encoder_config_value)?;
            let backbone = ModernBertEncoder::load(
                weights,
                &format!("{prefix}pretrained_text_backbone.backbone."),
                &backbone_config,
            )?;
            let projector = PretrainedProjector::load(
                weights,
                &format!("{prefix}text_encoder."),
                backbone.config.hidden_size,
                config.text_dim,
                &config.pretrained_projector_type,
                config.pretrained_projector_hidden_ratio,
                config.norm_eps,
            )?;
            (Some(backbone), None, Some(projector))
        } else {
            (
                None,
                Some(TextEncoder::load(
                    weights,
                    &format!("{prefix}text_encoder."),
                    config.text_vocab_size,
                    config.text_dim,
                    config.text_heads,
                    config.text_layers,
                    config.text_mlp_ratio_resolved(),
                    config.norm_eps,
                )?),
                None,
            )
        };

        let (caption_encoder, caption_projector) = if config.use_caption_condition {
            if config.use_pretrained_text_encoder() {
                let backbone = text_backbone
                    .as_ref()
                    .ok_or_else(|| SpeechError::Unsupported {
                        why: "caption projector requires the shared backbone".into(),
                    })?;
                (
                    None,
                    Some(PretrainedProjector::load(
                        weights,
                        &format!("{prefix}caption_encoder."),
                        backbone.config.hidden_size,
                        config.caption_dim_resolved(),
                        &config.pretrained_projector_type,
                        config.pretrained_projector_hidden_ratio,
                        config.norm_eps,
                    )?),
                )
            } else {
                (
                    Some(TextEncoder::load(
                        weights,
                        &format!("{prefix}caption_encoder."),
                        config.caption_vocab_size_resolved(),
                        config.caption_dim_resolved(),
                        config.caption_heads_resolved(),
                        config.caption_layers_resolved(),
                        config.caption_mlp_ratio_resolved(),
                        config.norm_eps,
                    )?),
                    None,
                )
            }
        } else {
            (None, None)
        };

        let (speaker_encoder, speaker_norm) = if config.use_speaker_condition_resolved() {
            (
                Some(ReferenceLatentEncoder::load(
                    weights,
                    &format!("{prefix}speaker_encoder."),
                    config.speaker_patched_latent_dim(),
                    config.speaker_dim,
                    config.speaker_heads,
                    config.speaker_layers,
                    config.speaker_mlp_ratio_resolved(),
                    config.norm_eps,
                )?),
                Some(weights.tensor(&format!("{prefix}speaker_norm"), &[config.speaker_dim])?),
            )
        } else {
            (None, None)
        };

        let caption_norm = if config.use_caption_condition {
            Some(weights.tensor(
                &format!("{prefix}caption_norm"),
                &[config.caption_dim_resolved()],
            )?)
        } else {
            None
        };

        let duration_predictor = if config.use_duration_predictor {
            Some(DurationPredictor::load(
                weights,
                &format!("{prefix}duration_predictor."),
                config.text_dim,
                config.duration_hidden_dim,
                config.duration_layers,
                config.norm_eps,
                if config.use_speaker_condition_resolved() {
                    Some(config.speaker_dim)
                } else {
                    None
                },
                if config.use_caption_condition {
                    Some(config.caption_dim_resolved())
                } else {
                    None
                },
                &config.duration_architecture,
            )?)
        } else {
            None
        };

        let seq = |p: &str| -> Result<Vec<Linear>> {
            Ok(vec![
                weights.linear(
                    &format!("{p}layers.0"),
                    config.timestep_embed_dim,
                    config.model_dim,
                    Bias::None,
                )?,
                weights.linear(
                    &format!("{p}layers.2"),
                    config.model_dim,
                    config.model_dim,
                    Bias::None,
                )?,
                weights.linear(
                    &format!("{p}layers.4"),
                    config.model_dim,
                    config.model_dim * 3,
                    Bias::None,
                )?,
            ])
        };
        let cond_module = seq(&format!("{prefix}cond_module."))?;
        let delta_cond_module = if config.use_meanflow() {
            Some(seq(&format!("{prefix}delta_cond_module."))?)
        } else {
            None
        };

        let mlp_hidden = (config.model_dim as f32 * config.mlp_ratio) as usize;
        let speaker_ctx = if config.use_speaker_condition_resolved() {
            Some(config.speaker_dim)
        } else {
            None
        };
        let caption_ctx = if config.use_caption_condition {
            Some(config.caption_dim_resolved())
        } else {
            None
        };
        let blocks = (0..config.num_layers)
            .map(|i| {
                let block_prefix = format!("{prefix}blocks.{i}.");
                Ok(DiffusionBlock {
                    attention: JointAttention::load(
                        weights,
                        &format!("{block_prefix}attention."),
                        config.model_dim,
                        config.num_heads,
                        config.text_dim,
                        speaker_ctx,
                        caption_ctx,
                        config.norm_eps,
                    )?,
                    mlp: SwiGlu::load(
                        weights,
                        &format!("{block_prefix}mlp."),
                        config.model_dim,
                        mlp_hidden,
                    )?,
                    attention_adaln: LowRankAdaLn::load(
                        weights,
                        &format!("{block_prefix}attention_adaln."),
                        config.model_dim,
                        config.adaln_rank,
                        config.norm_eps,
                    )?,
                    mlp_adaln: LowRankAdaLn::load(
                        weights,
                        &format!("{block_prefix}mlp_adaln."),
                        config.model_dim,
                        config.adaln_rank,
                        config.norm_eps,
                    )?,
                    dim: config.model_dim,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            config: config.clone(),
            text_backbone,
            text_encoder,
            text_projector,
            caption_encoder,
            caption_projector,
            speaker_encoder,
            text_norm: weights.tensor(&format!("{prefix}text_norm"), &[config.text_dim])?,
            speaker_norm,
            caption_norm,
            duration_predictor,
            cond_module,
            delta_cond_module,
            in_proj: weights.linear(
                &format!("{prefix}in_proj"),
                config.patched_latent_dim(),
                config.model_dim,
                Bias::Required,
            )?,
            blocks,
            out_norm: weights.tensor(&format!("{prefix}out_norm"), &[config.model_dim])?,
            out_proj: weights.linear(
                &format!("{prefix}out_proj"),
                config.model_dim,
                config.patched_latent_dim(),
                Bias::Required,
            )?,
            head_dim,
        })
    }

    fn run_cond_module(&self, modules: &[Linear], t: f32) -> Vec<f32> {
        let mut h = timestep_embedding(t, self.config.timestep_embed_dim);
        for (index, linear) in modules.iter().enumerate() {
            let rows = h.len() / linear.input;
            let projected = linear.forward(&h, rows);
            h = projected;
            if index + 1 < modules.len() {
                for value in &mut h {
                    *value *= 1.0 / (1.0 + (-*value).exp());
                }
            }
        }
        h
    }

    /// `encode_conditions_full` over the caller-owned token ids (the
    /// speaker latents arrive already sliced; patching happens here).
    #[allow(
        clippy::too_many_arguments,
        reason = "the encode mirrors the reference signature"
    )]
    pub fn encode_conditions_full(
        &self,
        text_input_ids: &[i32],
        text_mask: &[bool],
        ref_latent: Option<(&[f32], usize)>,
        ref_mask: Option<&[bool]>,
        caption_input_ids: Option<&[i32]>,
        caption_mask: Option<&[bool]>,
    ) -> Result<Conditions> {
        let config = &self.config;
        let text_state_raw = match (
            &self.text_backbone,
            &self.text_projector,
            &self.text_encoder,
        ) {
            (Some(backbone), Some(projector), _) => {
                projector.forward(backbone, text_input_ids, Some(text_mask))?
            }
            (None, None, Some(encoder)) => encoder.forward(text_input_ids, Some(text_mask))?,
            _ => {
                return Err(SpeechError::Unsupported {
                    why: "no text encoder configured".into(),
                })
            }
        };
        let text_rows = text_state_raw.len() / config.text_dim;
        let mut text_state = text_state_raw;
        rms_norm(&mut text_state, text_rows, &self.text_norm, config.norm_eps);

        let (speaker_state, speaker_mask) = if config.use_speaker_condition_resolved() {
            match (ref_latent, ref_mask) {
                (Some((latent, latent_rows)), Some(mask)) => {
                    let (patched, patched_mask) = patch_sequence(
                        latent,
                        latent_rows,
                        mask,
                        config.speaker_patch_size,
                        config.speaker_patched_latent_dim(),
                    );
                    let rows = patched.len() / config.speaker_patched_latent_dim();
                    let encoder =
                        self.speaker_encoder
                            .as_ref()
                            .ok_or_else(|| SpeechError::Unsupported {
                                why: "speaker condition without a speaker encoder".into(),
                            })?;
                    let mut state = encoder.forward(&patched, rows, Some(&patched_mask))?;
                    rms_norm(
                        &mut state,
                        rows,
                        self.speaker_norm.as_ref().unwrap(),
                        config.norm_eps,
                    );
                    (Some(state), Some(patched_mask))
                }
                _ => (Some(vec![0.0f32; config.speaker_dim]), Some(vec![false])),
            }
        } else {
            (None, None)
        };

        let (caption_state, caption_mask) = if config.use_caption_condition {
            match (caption_input_ids, caption_mask) {
                (Some(ids), Some(mask)) => {
                    let state = match (
                        &self.text_backbone,
                        &self.caption_projector,
                        &self.caption_encoder,
                    ) {
                        (Some(backbone), Some(projector), _) => {
                            projector.forward(backbone, ids, Some(mask))?
                        }
                        (None, None, Some(encoder)) => encoder.forward(ids, Some(mask))?,
                        _ => {
                            return Err(SpeechError::Unsupported {
                                why: "no caption encoder configured".into(),
                            })
                        }
                    };
                    let rows = state.len() / config.caption_dim_resolved();
                    let mut normalized = state;
                    rms_norm(
                        &mut normalized,
                        rows,
                        self.caption_norm.as_ref().unwrap(),
                        config.norm_eps,
                    );
                    (Some(normalized), Some(mask.to_vec()))
                }
                _ => (None, None),
            }
        } else {
            (None, None)
        };

        Ok(Conditions {
            text_state,
            text_mask: text_mask.to_vec(),
            speaker_state,
            speaker_mask,
            caption_state,
            caption_mask,
        })
    }

    /// `build_kv_cache`: per-layer text/speaker/caption KV projections.
    pub fn build_kv_cache(&self, conditions: &Conditions) -> Result<KvCaches> {
        let text_rows = conditions.text_state.len() / self.config.text_dim;
        let text: Vec<KvPair> = self
            .blocks
            .iter()
            .map(|block| block.attention.kv_text(&conditions.text_state, text_rows))
            .collect();
        let speaker = match (&conditions.speaker_state, &self.speaker_encoder) {
            (Some(state), Some(_)) if self.config.use_speaker_condition_resolved() => {
                let rows = state.len() / self.config.speaker_dim;
                Some(
                    self.blocks
                        .iter()
                        .map(|block| block.attention.kv_speaker(state, rows))
                        .collect::<Result<Vec<_>>>()?,
                )
            }
            _ => None,
        };
        let caption = match (&conditions.caption_state, &self.caption_projector) {
            (Some(state), Some(_)) if self.config.use_caption_condition => {
                let rows = state.len() / self.config.caption_dim_resolved();
                Some(
                    self.blocks
                        .iter()
                        .map(|block| block.attention.kv_caption(state, rows))
                        .collect::<Result<Vec<_>>>()?,
                )
            }
            _ => None,
        };
        Ok(KvCaches {
            text,
            speaker,
            caption,
        })
    }

    /// `forward_with_conditions` for one bundle row.
    #[allow(
        clippy::too_many_arguments,
        reason = "the forward mirrors the reference signature"
    )]
    pub fn forward_with_conditions(
        &self,
        x_t: &[f32],
        t: f32,
        conditions: &Conditions,
        kv: Option<&KvCaches>,
        delta_t: Option<f32>,
    ) -> Result<Vec<f32>> {
        let config = &self.config;
        let rows = x_t.len() / config.patched_latent_dim();
        let mut cond = self.run_cond_module(&self.cond_module, t);
        if let (Some(modules), Some(delta)) = (&self.delta_cond_module, delta_t) {
            let embed = self.run_cond_module(modules, delta);
            for (value, add) in cond.iter_mut().zip(&embed) {
                *value += add;
            }
        } else if delta_t.is_some() {
            return Err(SpeechError::Input {
                why: "delta_t was provided to an RF velocity model".into(),
            });
        }

        let mut x = self.in_proj.forward(x_t, rows);
        let freqs = RopeTables::new(self.head_dim, rows);

        // The context KV caches: precomputed when available, otherwise
        // recomputed per block from the states.
        let text_rows = conditions.text_state.len() / config.text_dim;
        let speaker_state = conditions.speaker_state.as_ref();
        let speaker_rows = speaker_state.map_or(0, |s| s.len() / config.speaker_dim);
        let caption_state = conditions.caption_state.as_ref();
        let caption_rows = caption_state.map_or(0, |s| s.len() / config.caption_dim_resolved());

        let mut speaker_kv: Vec<Option<KvPair>> = Vec::with_capacity(self.blocks.len());
        let mut caption_kv: Vec<Option<KvPair>> = Vec::with_capacity(self.blocks.len());
        for (index, block) in self.blocks.iter().enumerate() {
            let cached_speaker = kv.and_then(|kv| kv.speaker.as_ref()).map(|c| &c[index]);
            speaker_kv.push(match cached_speaker {
                Some(pair) => Some(KvPair {
                    k: pair.k.clone(),
                    v: pair.v.clone(),
                }),
                None => match (speaker_state, block.attention.wk_speaker.as_ref()) {
                    (Some(state), Some(_)) => {
                        Some(block.attention.kv_speaker(state, speaker_rows)?)
                    }
                    _ => None,
                },
            });
            let cached_caption = kv.and_then(|kv| kv.caption.as_ref()).map(|c| &c[index]);
            caption_kv.push(match cached_caption {
                Some(pair) => Some(KvPair {
                    k: pair.k.clone(),
                    v: pair.v.clone(),
                }),
                None => match (caption_state, block.attention.wk_caption.as_ref()) {
                    (Some(state), Some(_)) => {
                        Some(block.attention.kv_caption(state, caption_rows)?)
                    }
                    _ => None,
                },
            });
        }

        // The text KV: cached or recomputed once.
        let kv_text_owned;
        let kv_text: &Vec<KvPair> = match kv {
            Some(kv) => &kv.text,
            None => {
                kv_text_owned = self
                    .blocks
                    .iter()
                    .map(|block| block.attention.kv_text(&conditions.text_state, text_rows))
                    .collect::<Vec<_>>();
                &kv_text_owned
            }
        };

        for (index, block) in self.blocks.iter().enumerate() {
            block.forward(
                &mut x,
                &cond,
                rows,
                &conditions.text_mask,
                &freqs,
                &kv_text[index],
                speaker_kv[index].as_ref(),
                conditions.speaker_mask.as_deref(),
                caption_kv[index].as_ref(),
                conditions.caption_mask.as_deref(),
                0,
            );
        }

        rms_norm(&mut x, rows, &self.out_norm, config.norm_eps);
        Ok(self.out_proj.forward(&x, rows))
    }
}

/// `patch_sequence_with_mask` over channel-last rows.
#[allow(clippy::too_many_arguments)]
fn patch_sequence(
    latent: &[f32],
    rows: usize,
    mask: &[bool],
    patch_size: usize,
    patched_dim: usize,
) -> (Vec<f32>, Vec<bool>) {
    if patch_size <= 1 {
        let mut flat = latent.to_vec();
        flat.truncate(rows * patched_dim);
        return (flat, mask.to_vec());
    }
    let dim = patched_dim / patch_size;
    let usable = (rows / patch_size) * patch_size;
    let groups = usable / patch_size;
    let mut out = vec![0.0f32; groups * patched_dim];
    for g in 0..groups {
        for p in 0..patch_size {
            out[g * patched_dim + p * dim..g * patched_dim + (p + 1) * dim].copy_from_slice(
                &latent[(g * patch_size + p) * dim..(g * patch_size + p + 1) * dim],
            );
        }
    }
    let mut patched_mask = Vec::with_capacity(groups);
    for g in 0..groups {
        patched_mask.push((g * patch_size..(g + 1) * patch_size).all(|p| mask[p]));
    }
    (out, patched_mask)
}

// ---------------------------------------------------------------------------
// Samplers
// ---------------------------------------------------------------------------

/// Temporal score rescaling (arXiv 2510.01184).
fn temporal_score_rescale(v_pred: &[f32], x_t: &[f32], t: f32, k: f32, sigma: f32) -> Vec<f32> {
    if t >= 1.0 {
        return v_pred.to_vec();
    }
    let one_minus_t = 1.0 - t;
    let snr = one_minus_t * one_minus_t / (t * t);
    let sigma_sq = sigma * sigma;
    let ratio = (snr * sigma_sq + 1.0) / (snr * sigma_sq / k + 1.0);
    v_pred
        .iter()
        .zip(x_t)
        .map(|(&v, &x)| (ratio * (one_minus_t * v + x) - x) / one_minus_t)
        .collect()
}

/// Sampler parameters the caller may override per generate call.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub rng_seed: u64,
    /// Manual duration in seconds (skips the duration predictor).
    pub seconds: Option<f32>,
    pub duration_scale: f32,
    pub min_seconds: f32,
    pub max_seconds: f32,
    /// Sampler overrides.
    pub num_steps: Option<usize>,
    pub cfg_scale_text: Option<f32>,
    pub cfg_scale_speaker: Option<f32>,
    pub cfg_scale_caption: Option<f32>,
    pub cfg_scale: Option<f32>,
    pub t_schedule_mode: Option<String>,
    pub sway_coeff: Option<f32>,
    pub cfg_min_t: Option<f32>,
    pub cfg_max_t: Option<f32>,
    pub truncation_factor: Option<f32>,
    pub rescale_k: Option<f32>,
    pub rescale_sigma: Option<f32>,
    pub context_kv_cache: bool,
    pub speaker_kv_scale: Option<f32>,
    pub speaker_kv_min_t: Option<f32>,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            rng_seed: 0,
            seconds: None,
            duration_scale: 1.0,
            min_seconds: 0.5,
            max_seconds: 30.0,
            num_steps: None,
            cfg_scale_text: None,
            cfg_scale_speaker: None,
            cfg_scale_caption: None,
            cfg_scale: None,
            t_schedule_mode: None,
            sway_coeff: None,
            cfg_min_t: None,
            cfg_max_t: None,
            truncation_factor: None,
            rescale_k: None,
            rescale_sigma: None,
            context_kv_cache: true,
            speaker_kv_scale: None,
            speaker_kv_min_t: None,
        }
    }
}

/// The caller reference-audio handle: channel-last latent rows, their
/// count, and the validity mask.
pub type RefLatent<'a> = (&'a [f32], usize, Option<&'a [bool]>);

/// One generated result: the latents `[steps, latent_dim]`, the frame
/// count before patching, and the decoded waveform.
#[derive(Debug, Clone)]
pub struct Generated {
    pub latent: Vec<f32>,
    pub latent_steps: usize,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// The sampler resolver consumed by [`Model::generate_latents`].
struct ResolvedSampler {
    cfg_guidance_mode: String,
    num_steps: usize,
    cfg_scale_text: f32,
    cfg_scale_speaker: f32,
    cfg_scale_caption: f32,
    cfg_min_t: f32,
    cfg_max_t: f32,
    truncation_factor: Option<f32>,
    rescale_k: Option<f32>,
    rescale_sigma: Option<f32>,
    context_kv_cache: bool,
    speaker_kv_scale: Option<f32>,
    speaker_kv_min_t: Option<f32>,
    t_schedule_mode: String,
    sway_coeff: f32,
}

fn resolve_sampler(config: &SamplerConfig, params: &GenerateParams) -> ResolvedSampler {
    let (text, speaker, caption) = match params.cfg_scale {
        Some(scale) => (scale, scale, scale),
        None => (
            params.cfg_scale_text.unwrap_or(config.cfg_scale_text),
            params.cfg_scale_speaker.unwrap_or(config.cfg_scale_speaker),
            params.cfg_scale_caption.unwrap_or(config.cfg_scale_caption),
        ),
    };
    ResolvedSampler {
        cfg_guidance_mode: config.cfg_guidance_mode.clone(),
        num_steps: params.num_steps.unwrap_or(config.num_steps),
        cfg_scale_text: text,
        cfg_scale_speaker: speaker,
        cfg_scale_caption: caption,
        cfg_min_t: params.cfg_min_t.unwrap_or(config.cfg_min_t),
        cfg_max_t: params.cfg_max_t.unwrap_or(config.cfg_max_t),
        truncation_factor: params.truncation_factor.or(config.truncation_factor),
        rescale_k: params.rescale_k.or(config.rescale_k),
        rescale_sigma: params.rescale_sigma.or(config.rescale_sigma),
        context_kv_cache: params.context_kv_cache && config.context_kv_cache,
        speaker_kv_scale: params.speaker_kv_scale.or(config.speaker_kv_scale),
        speaker_kv_min_t: params.speaker_kv_min_t.or(config.speaker_kv_min_t),
        t_schedule_mode: params
            .t_schedule_mode
            .clone()
            .unwrap_or_else(|| config.t_schedule_mode.clone()),
        sway_coeff: params.sway_coeff.unwrap_or(config.sway_coeff),
    }
}

/// Scales every layer of a speaker KV cache (reference
/// `_scale_kv_cache`).
fn scale_kv_cache(cache: &[KvPair], scale: f32, max_layers: Option<usize>) -> Vec<KvPair> {
    let n = max_layers.unwrap_or(cache.len()).min(cache.len());
    cache
        .iter()
        .enumerate()
        .map(|(i, pair)| {
            if i < n {
                KvPair {
                    k: pair.k.iter().map(|v| v * scale).collect(),
                    v: pair.v.iter().map(|v| v * scale).collect(),
                }
            } else {
                KvPair {
                    k: pair.k.clone(),
                    v: pair.v.clone(),
                }
            }
        })
        .collect()
}

/// The t schedule: linear from `0.999` to 0, or the sway warping.
fn t_schedule_with(sampler: &ResolvedSampler, num_steps: usize) -> Vec<f32> {
    let init_scale = 0.999f32;
    if sampler.t_schedule_mode.trim().eq_ignore_ascii_case("sway") {
        let mut u: Vec<f32> = (0..=num_steps)
            .map(|i| i as f32 / num_steps as f32)
            .collect();
        for value in &mut u {
            *value +=
                sampler.sway_coeff * ((0.5 * std::f32::consts::PI * *value).cos() + *value - 1.0);
            *value = value.clamp(0.0, 1.0);
        }
        return u.into_iter().map(|u| (1.0 - u) * init_scale).collect();
    }
    (0..=num_steps)
        .map(|i| init_scale * (1.0 - i as f32 / num_steps as f32))
        .collect()
}

/// The Euler RF sampler with classifier-free guidance (the reference
/// `sample_euler_cfg`), independent and joint guidance modes. The
/// reference batches the CFG bundles into one forward pass; the port
/// runs the bundles sequentially over the same weights (rows are
/// independent). The independent mode keeps the CONDITIONAL context KV
/// caches for every bundle, exactly like the reference
/// `context_kv_cache` path: the uncond bundles hide the context keys
/// with their zeroed masks.
#[allow(
    clippy::too_many_arguments,
    reason = "the sampler carries its conditions explicitly"
)]
fn sample_euler_cfg(
    dit: &IrodoriDiT,
    conditions: &Conditions,
    kv: Option<&KvCaches>,
    noise: &mut dyn FlowNoise,
    sequence_length: usize,
    latent_dim: usize,
    sampler: &ResolvedSampler,
) -> Result<Vec<f32>> {
    let mode = sampler.cfg_guidance_mode.trim().to_lowercase();
    if !matches!(mode.as_str(), "independent" | "joint") {
        if mode == "alternating" {
            return Err(SpeechError::Unsupported {
                why: "cfg_guidance_mode='alternating' is not ported; use independent or joint"
                    .into(),
            });
        }
        return Err(SpeechError::Input {
            why: format!("unknown cfg_guidance_mode={mode:?}"),
        });
    }

    let use_spk = dit.config.use_speaker_condition_resolved();
    let use_cap = dit.config.use_caption_condition;
    let is_dual = use_spk && use_cap;
    let cfg_scale_context = if !is_dual {
        if use_cap {
            sampler.cfg_scale_caption
        } else {
            sampler.cfg_scale_speaker
        }
    } else {
        sampler.cfg_scale_speaker
    };
    let has_text_cfg = sampler.cfg_scale_text > 0.0;
    let has_speaker_cfg = sampler.cfg_scale_speaker > 0.0 && use_spk;
    let has_caption_cfg = sampler.cfg_scale_caption > 0.0
        && use_cap
        && conditions
            .caption_mask
            .as_ref()
            .is_some_and(|mask| mask.iter().any(|&valid| valid));
    let has_context_cfg = if !is_dual {
        cfg_scale_context > 0.0
    } else {
        false
    };

    let mut x_t: Vec<f32> = noise.normal(sequence_length * latent_dim);
    if let Some(truncation) = sampler.truncation_factor {
        for value in &mut x_t {
            *value *= truncation;
        }
    }
    let schedule = t_schedule_with(sampler, sampler.num_steps);

    // The effective speaker cache: scaled from the start, rolled back
    // once across the cfg_min_t threshold (reference speaker_kv_scale).
    let mut speaker_kv_active = sampler.speaker_kv_scale.is_some();
    let mut speaker_kv_scaled: Option<Vec<KvPair>> = match (kv, sampler.speaker_kv_scale) {
        (Some(kv), Some(scale)) => kv
            .speaker
            .as_ref()
            .map(|cache| scale_kv_cache(cache, scale, None)),
        (Some(kv), None) => kv.speaker.as_ref().map(|cache| cache.to_vec()),
        _ => None,
    };

    for step in 0..sampler.num_steps {
        let t = schedule[step];
        let t_next = schedule[step + 1];
        let use_cfg = (has_text_cfg || has_speaker_cfg || has_caption_cfg)
            && (sampler.cfg_min_t <= t && t <= sampler.cfg_max_t);

        // The effective caches for this step: the conditional ones with
        // the current speaker scaling.
        let step_kv_owned;
        let step_kv: Option<&KvCaches> = match kv {
            Some(kv) => {
                step_kv_owned = KvCaches {
                    text: kv.text.clone(),
                    speaker: speaker_kv_scaled.clone(),
                    caption: kv.caption.clone(),
                };
                Some(&step_kv_owned)
            }
            None => None,
        };

        let v_pred = if use_cfg {
            let cond_velocity = dit.forward_with_conditions(&x_t, t, conditions, step_kv, None)?;
            let mut v_pred = cond_velocity.clone();
            let mut guided = |bundle: Conditions, scale: f32| -> Result<()> {
                let v_uncond = dit.forward_with_conditions(&x_t, t, &bundle, step_kv, None)?;
                for (value, (cond, uncond)) in
                    v_pred.iter_mut().zip(cond_velocity.iter().zip(&v_uncond))
                {
                    *value += scale * (cond - uncond);
                }
                Ok(())
            };
            if mode == "independent" {
                if !is_dual {
                    if has_text_cfg {
                        guided(
                            bundle_with(conditions, BundleKind::Text),
                            sampler.cfg_scale_text,
                        )?;
                    }
                    if has_context_cfg {
                        guided(
                            bundle_with(conditions, BundleKind::Context),
                            cfg_scale_context,
                        )?;
                    }
                } else {
                    if has_text_cfg {
                        guided(
                            bundle_with(conditions, BundleKind::Text),
                            sampler.cfg_scale_text,
                        )?;
                    }
                    if has_speaker_cfg {
                        guided(
                            bundle_with(conditions, BundleKind::Speaker),
                            sampler.cfg_scale_speaker,
                        )?;
                    }
                    if has_caption_cfg {
                        guided(
                            bundle_with(conditions, BundleKind::Caption),
                            sampler.cfg_scale_caption,
                        )?;
                    }
                }
            } else {
                // joint: one combined unconditional branch with its own
                // caches built from the zeroed states.
                let joint_scale = if is_dual {
                    [has_text_cfg, has_speaker_cfg, has_caption_cfg]
                        .iter()
                        .zip([
                            sampler.cfg_scale_text,
                            sampler.cfg_scale_speaker,
                            sampler.cfg_scale_caption,
                        ])
                        .filter(|(active, _)| **active)
                        .map(|(_, scale)| scale)
                        .next()
                        .unwrap_or(sampler.cfg_scale_text)
                } else if has_text_cfg && has_context_cfg {
                    if (sampler.cfg_scale_text - cfg_scale_context).abs() > 1e-6 {
                        return Err(SpeechError::Input {
                            why: "cfg_guidance_mode='joint' requires equal text/speaker scales"
                                .into(),
                        });
                    }
                    sampler.cfg_scale_text
                } else if has_text_cfg {
                    sampler.cfg_scale_text
                } else {
                    cfg_scale_context
                };
                let uncond_conditions = Conditions {
                    text_state: vec![0.0f32; conditions.text_state.len()],
                    text_mask: vec![false; conditions.text_mask.len()],
                    speaker_state: conditions
                        .speaker_state
                        .as_ref()
                        .map(|s| vec![0.0f32; s.len()]),
                    speaker_mask: conditions
                        .speaker_mask
                        .as_ref()
                        .map(|m| vec![false; m.len()]),
                    caption_state: conditions
                        .caption_state
                        .as_ref()
                        .map(|s| vec![0.0f32; s.len()]),
                    caption_mask: conditions
                        .caption_mask
                        .as_ref()
                        .map(|m| vec![false; m.len()]),
                };
                let uncond_kv = dit.build_kv_cache(&uncond_conditions)?;
                let v_uncond = dit.forward_with_conditions(
                    &x_t,
                    t,
                    &uncond_conditions,
                    Some(&uncond_kv),
                    None,
                )?;
                for (value, (cond, uncond)) in
                    v_pred.iter_mut().zip(cond_velocity.iter().zip(&v_uncond))
                {
                    *value += joint_scale * (cond - uncond);
                }
            }
            v_pred
        } else {
            dit.forward_with_conditions(&x_t, t, conditions, step_kv, None)?
        };

        let mut v_pred = v_pred;
        if let (Some(k), Some(sigma)) = (sampler.rescale_k, sampler.rescale_sigma) {
            v_pred = temporal_score_rescale(&v_pred, &x_t, t, k, sigma);
        }

        if speaker_kv_active
            && sampler.speaker_kv_scale.is_some()
            && sampler
                .speaker_kv_min_t
                .is_some_and(|min_t| t_next < min_t && min_t <= t)
        {
            if let Some(scale) = sampler.speaker_kv_scale {
                let inv = 1.0 / scale;
                if let Some(cache) = &mut speaker_kv_scaled {
                    *cache = scale_kv_cache(cache, inv, None);
                }
                speaker_kv_active = false;
            }
        }

        // Euler update: x_{t-dt} = x_t + v * (t_next - t).
        let dt = t_next - t;
        for (value, velocity) in x_t.iter_mut().zip(&v_pred) {
            *value += velocity * dt;
        }
    }
    Ok(x_t)
}

/// Bundle kinds for the per-condition unconditional branches.
enum BundleKind {
    Text,
    Context,
    Speaker,
    Caption,
}

/// Builds the unconditional bundle: the reference concatenates state
/// variants per branch; the single-row port picks the right mix.
fn bundle_with(conditions: &Conditions, kind: BundleKind) -> Conditions {
    match kind {
        BundleKind::Text => Conditions {
            text_state: vec![0.0f32; conditions.text_state.len()],
            text_mask: vec![false; conditions.text_mask.len()],
            speaker_state: conditions.speaker_state.clone(),
            speaker_mask: conditions.speaker_mask.clone(),
            caption_state: conditions.caption_state.clone(),
            caption_mask: conditions.caption_mask.clone(),
        },
        BundleKind::Context => Conditions {
            text_state: conditions.text_state.clone(),
            text_mask: conditions.text_mask.clone(),
            speaker_state: conditions
                .speaker_state
                .as_ref()
                .map(|s| vec![0.0f32; s.len()]),
            speaker_mask: conditions
                .speaker_mask
                .as_ref()
                .map(|m| vec![false; m.len()]),
            caption_state: conditions
                .caption_state
                .as_ref()
                .map(|s| vec![0.0f32; s.len()]),
            caption_mask: conditions
                .caption_mask
                .as_ref()
                .map(|m| vec![false; m.len()]),
        },
        BundleKind::Speaker => Conditions {
            text_state: conditions.text_state.clone(),
            text_mask: conditions.text_mask.clone(),
            speaker_state: conditions
                .speaker_state
                .as_ref()
                .map(|s| vec![0.0f32; s.len()]),
            speaker_mask: conditions
                .speaker_mask
                .as_ref()
                .map(|m| vec![false; m.len()]),
            caption_state: conditions.caption_state.clone(),
            caption_mask: conditions.caption_mask.clone(),
        },
        BundleKind::Caption => Conditions {
            text_state: conditions.text_state.clone(),
            text_mask: conditions.text_mask.clone(),
            speaker_state: conditions.speaker_state.clone(),
            speaker_mask: conditions.speaker_mask.clone(),
            caption_state: conditions
                .caption_state
                .as_ref()
                .map(|s| vec![0.0f32; s.len()]),
            caption_mask: conditions
                .caption_mask
                .as_ref()
                .map(|m| vec![false; m.len()]),
        },
    }
}

/// The MeanFlow Euler sampler (reference `sample_euler_meanflow`): one
/// conditional branch per step, `delta_t = t - t_next` conditioning.
fn sample_euler_meanflow(
    dit: &IrodoriDiT,
    conditions: &Conditions,
    kv: &KvCaches,
    noise: &mut dyn FlowNoise,
    sequence_length: usize,
    latent_dim: usize,
    num_steps: usize,
) -> Result<Vec<f32>> {
    if num_steps == 0 {
        return Err(SpeechError::Input {
            why: "MeanFlow num_steps must be positive".into(),
        });
    }
    let mut x_t: Vec<f32> = noise.normal(sequence_length * latent_dim);
    let mut t = 1.0f32;
    for step in 1..=num_steps {
        let t_next = 1.0f32 - step as f32 / num_steps as f32;
        let velocity =
            dit.forward_with_conditions(&x_t, t, conditions, Some(kv), Some(t - t_next))?;
        for (value, v) in x_t.iter_mut().zip(&velocity) {
            *value += v * (t_next - t);
        }
        t = t_next;
    }
    Ok(x_t)
}

// ---------------------------------------------------------------------------
// Model facade
// ---------------------------------------------------------------------------

/// `_find_silence_point`: the first window whose population std drops
/// under 0.05 with a |mean| under 0.1, over the channel-last latents.
fn find_silence_point(latent: &[f32], rows: usize, dim: usize, window_size: usize) -> usize {
    let mut padded = latent.to_vec();
    padded.resize((rows + window_size) * dim, 0.0);
    for i in 0..rows {
        let mut sum = 0.0f32;
        let mut sum_sq = 0.0f32;
        for step in i..i + window_size {
            for d in 0..dim {
                let value = padded[step * dim + d];
                sum += value;
                sum_sq += value * value;
            }
        }
        let count = (window_size * dim) as f32;
        let mean = sum / count;
        let std = (sum_sq / count - mean * mean).sqrt();
        if std < 0.05 && mean.abs() < 0.1 {
            return i;
        }
    }
    rows
}

/// The loaded Irodori TTS model.
pub struct Model {
    pub config: ModelConfig,
    dit: IrodoriDiT,
    dacvae: Option<Dacvae>,
}

impl Model {
    /// Opens a checkpoint directory: `config.json`, the DiT safetensors,
    /// and the bundled `dacvae/` codec folder
    /// (`post_load_hook`'s local path; a hub fallback is caller-owned).
    pub fn open(dir: &Path) -> Result<Self> {
        let config_value = quant::read_json(&dir.join("config.json"))?;
        let config = ModelConfig::from_json(&config_value)?;
        let weights = Weights::open(dir)?;
        let dit = IrodoriDiT::load(&weights, &config.dit)?;
        // The codec is optional at load (the reference `post_load_hook`
        // warns and leaves it None); the decode path refuses without it.
        let dacvae = if dir.join("dacvae").is_dir() {
            Some(Dacvae::open(&dir.join("dacvae"))?)
        } else {
            None
        };
        Ok(Self {
            config,
            dit,
            dacvae,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Decodes channel-first `[latent_dim, frames]` DACVAE latents to
    /// mono samples (the family tail of the reference generate; the
    /// chunked reference decode is a memory optimization this full pass
    /// replaces).
    pub fn decode_latents(&self, latents: &[f32]) -> Result<Vec<f32>> {
        let dacvae = self.dacvae.as_ref().ok_or_else(|| SpeechError::Input {
            why: "Irodori-TTS requires DACVAE to be loaded".into(),
        })?;
        dacvae.decode(latents)
    }

    /// `_encode_ref_audio`: DACVAE-encodes one reference waveform and
    /// trims to `max_ref_seconds` / the hard latent cap. Returns the
    /// channel-last latent rows `[frames, latent_dim]`.
    pub fn encode_ref_audio(
        &self,
        samples: &[f32],
        max_ref_seconds: Option<f32>,
    ) -> Result<(Vec<f32>, usize)> {
        let config = &self.config;
        let dacvae = self.dacvae.as_ref().ok_or_else(|| SpeechError::Input {
            why: "Irodori-TTS requires DACVAE to be loaded".into(),
        })?;
        let max_steps = self.max_ref_latent_steps(max_ref_seconds);
        let sliced = &samples[..samples
            .len()
            .min(max_steps * config.audio_downsample_factor)];
        let latent = dacvae.encode(sliced)?;
        let dim = dacvae.config.codebook_dim;
        let frames = latent.len() / dim;
        let actual_t = (sliced.len() / config.audio_downsample_factor).min(frames);
        let mut channel_last = vec![0.0f32; actual_t * dim];
        for t in 0..actual_t {
            for d in 0..dim {
                channel_last[t * dim + d] = latent[d * frames + t];
            }
        }
        Ok((channel_last, actual_t))
    }

    fn max_ref_latent_steps(&self, max_ref_seconds: Option<f32>) -> usize {
        let seconds = max_ref_seconds.unwrap_or(self.config.ref_max_seconds);
        let mut limit = self.config.max_speaker_latent_length;
        if seconds > 0.0 {
            let steps = (seconds * self.config.sample_rate as f32
                / self.config.audio_downsample_factor as f32) as usize;
            limit = limit.min(steps.max(1));
        }
        limit
    }

    /// `generate_latents`: text tokens (caller-owned), optional caption
    /// tokens, optional reference latent (channel-last rows), and the
    /// sampler parameters. Returns the latents `[frames, latent_dim]`
    /// plus the frame count before patching.
    #[allow(
        clippy::too_many_arguments,
        reason = "the entry point mirrors the reference signature"
    )]
    pub fn generate_latents(
        &self,
        text_input_ids: &[i32],
        text_mask: &[bool],
        caption_input_ids: Option<&[i32]>,
        caption_mask: Option<&[bool]>,
        ref_latent: Option<RefLatent<'_>>,
        params: &GenerateParams,
        duration_features: Option<&[f32]>,
        noise: &mut dyn FlowNoise,
    ) -> Result<Generated> {
        let config = &self.config;
        let dit_config = &config.dit;
        // An empty caption must arrive fully masked (the reference zeroes
        // the mask when the caption string is empty).
        let (caption_ids, caption_mask) = match (caption_input_ids, caption_mask) {
            (Some(ids), Some(mask)) if dit_config.use_caption_condition => (Some(ids), Some(mask)),
            _ => (None, None),
        };
        // Speaker defaults (the reference `generate_latents`): a zero
        // latent of one full speaker patch with an all-false mask when
        // the mode needs speaker input but none came; an all-true mask
        // when latents arrived without one.
        let zero_speaker;
        let true_mask;
        let false_mask;
        let (ref_latent, ref_mask): (&[f32], &[bool]) =
            if dit_config.use_speaker_condition_resolved() {
                match ref_latent {
                    Some((latent, _rows, Some(mask))) => (latent, mask),
                    Some((latent, rows, None)) => {
                        true_mask = vec![true; rows];
                        (latent, &true_mask)
                    }
                    None => {
                        zero_speaker =
                            vec![0.0f32; dit_config.speaker_patch_size * dit_config.latent_dim];
                        false_mask = vec![false; dit_config.speaker_patch_size];
                        (&zero_speaker, &false_mask)
                    }
                }
            } else {
                (&[], &[])
            };
        let speaker_rows = if dit_config.use_speaker_condition_resolved() {
            ref_latent.len() / dit_config.latent_dim.max(1)
        } else {
            0
        };

        // Sequence length.
        let latent_steps = if let Some(seconds) = params.seconds {
            let clamped = seconds.clamp(params.min_seconds, params.max_seconds);
            let target_samples = (clamped * config.sample_rate as f32) as usize;
            target_samples.div_ceil(config.audio_downsample_factor)
        } else if dit_config.use_duration_predictor {
            let features = duration_features.ok_or_else(|| SpeechError::Input {
                why: "duration_features required when the duration predictor is enabled".into(),
            })?;
            if features.len() != dit_config.duration_aux_dim {
                return Err(SpeechError::Input {
                    why: format!(
                        "duration_features must hold {} values, got {}",
                        dit_config.duration_aux_dim,
                        features.len()
                    ),
                });
            }
            let has_speaker = ref_mask.iter().any(|&valid| valid);
            let conditions = self.dit.encode_conditions_full(
                text_input_ids,
                text_mask,
                Some((ref_latent, speaker_rows)),
                Some(ref_mask),
                caption_ids,
                caption_mask,
            )?;
            let pred_log = self
                .dit
                .duration_predictor
                .as_ref()
                .ok_or_else(|| SpeechError::Unsupported {
                    why: "duration predictor disabled".into(),
                })?
                .forward(
                    &conditions.text_state,
                    &conditions.text_mask,
                    conditions.speaker_state.as_deref(),
                    has_speaker,
                    conditions.caption_state.as_deref(),
                    conditions.caption_mask.as_deref(),
                    caption_mask.is_some_and(|mask| mask.iter().any(|&valid| valid)),
                )?;
            let pred_frames = (pred_log.exp() - 1.0).max(0.0);
            let scaled = pred_frames * params.duration_scale;
            let min_frames = (params.min_seconds * config.sample_rate as f32
                / config.audio_downsample_factor as f32)
                .ceil()
                .max(1.0) as usize;
            let max_frames = (params.max_seconds * config.sample_rate as f32
                / config.audio_downsample_factor as f32)
                .floor()
                .max(1.0) as usize;
            (scaled.round() as usize).clamp(min_frames, max_frames)
        } else {
            config.sampler.sequence_length
        };

        let patched_steps = latent_steps.div_ceil(dit_config.latent_patch_size);
        let sampler = resolve_sampler(&config.sampler, params);
        let conditions = self.dit.encode_conditions_full(
            text_input_ids,
            text_mask,
            Some((ref_latent, speaker_rows)),
            Some(ref_mask),
            caption_ids,
            caption_mask,
        )?;
        let kv = if dit_config.use_meanflow()
            || sampler.context_kv_cache
            || sampler.speaker_kv_scale.is_some()
        {
            Some(self.dit.build_kv_cache(&conditions)?)
        } else {
            None
        };

        let latent_flat = if dit_config.use_meanflow() {
            let kv_ref = kv.as_ref().ok_or_else(|| SpeechError::Input {
                why: "meanflow path requires the context KV cache".into(),
            })?;
            sample_euler_meanflow(
                &self.dit,
                &conditions,
                kv_ref,
                noise,
                patched_steps,
                dit_config.patched_latent_dim(),
                sampler.num_steps,
            )?
        } else {
            if matches!(
                sampler.cfg_guidance_mode.trim().to_lowercase().as_str(),
                "alternating"
            ) {
                // Refused inside the sampler; keep the early error here so
                // the KV cache work is not wasted.
                return Err(SpeechError::Unsupported {
                    why: "cfg_guidance_mode='alternating' is not ported; use independent or joint"
                        .into(),
                });
            }
            sample_euler_cfg(
                &self.dit,
                &conditions,
                kv.as_ref(),
                noise,
                patched_steps,
                dit_config.patched_latent_dim(),
                &sampler,
            )?
        };

        Ok(Generated {
            latent: latent_flat,
            latent_steps,
            samples: Vec::new(),
            sample_rate: config.sample_rate,
        })
    }

    /// The full reference `generate`: latents, DACVAE decode (full pass;
    /// the reference's chunked decode is a memory optimization), the
    /// duration clamp, and the trailing-silence trim.
    #[allow(
        clippy::too_many_arguments,
        reason = "the entry point mirrors the reference signature"
    )]
    pub fn generate(
        &self,
        text_input_ids: &[i32],
        text_mask: &[bool],
        caption_input_ids: Option<&[i32]>,
        caption_mask: Option<&[bool]>,
        ref_latent: Option<RefLatent<'_>>,
        params: &GenerateParams,
        duration_features: Option<&[f32]>,
        noise: &mut dyn FlowNoise,
    ) -> Result<Generated> {
        let config = &self.config;
        let mut generated = self.generate_latents(
            text_input_ids,
            text_mask,
            caption_input_ids,
            caption_mask,
            ref_latent,
            params,
            duration_features,
            noise,
        )?;
        let dacvae = self.dacvae.as_ref().ok_or_else(|| SpeechError::Input {
            why: "Irodori-TTS requires DACVAE to be loaded".into(),
        })?;
        let dim = dacvae.config.codebook_dim;
        let frames = generated.latent.len() / dim;
        // The latents come back channel-last; decode wants channel-first.
        let mut channel_first = vec![0.0f32; generated.latent.len()];
        for t in 0..frames {
            for d in 0..dim {
                channel_first[d * frames + t] = generated.latent[t * dim + d];
            }
        }
        let samples = dacvae.decode(&channel_first)?;
        // Clamp to the predicted duration, then trim trailing silence.
        let target_samples = generated.latent_steps * config.audio_downsample_factor;
        let silence_rows = find_silence_point(&generated.latent, frames, dim, 20);
        let silence_samples = silence_rows * config.audio_downsample_factor;
        let trim = if silence_samples > 0 {
            target_samples.min(silence_samples)
        } else {
            target_samples
        };
        generated.samples = samples[..samples.len().min(trim)].to_vec();
        Ok(generated)
    }
}
