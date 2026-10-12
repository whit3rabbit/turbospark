//! Pocket TTS family (Kyutai flow-LM over continuous Mimi latents).
//!
//! Port of `mlx_audio/tts/models/pocket_tts/` (pocket_tts.py, flow_lm.py,
//! transformer.py, mlp.py, rope.py, conditioners.py, config.py,
//! utils.py) at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/pocket_tts):
//! a lookup-table text conditioner feeds a streaming transformer whose
//! last hidden state conditions an AdaLN MLP flow net; each step
//! integrates the flow from temperature-scaled Gaussian noise (Euler,
//! `lsd_decode_steps`) into the next Mimi latent, and an `out_eos` head
//! ends the run `frames_after_eos` frames after the threshold crosses.
//! See [README.md](README.md) for the verification evidence and open
//! items.
//!
//! Scope of this port, matching the reference single-chunk
//! `_generate_audio_stream_short_text`: batch size 1, non-streaming.
//! The run ends at the scaled latents (`next_latent * emb_std +
//! emb_mean`); the bundled pocket Mimi variant (a continuous-latent
//! streaming autoencoder with no Rust port; the shared
//! `src/codec/mimi` module is the RVQ codec) owns the waveform decode.
//! Text tokenization is a caller concern: the caller passes the token
//! ids the sentencepiece tokenizer would produce and the audio
//! conditioning embeddings a loaded voice (Mimi encode through the
//! `speaker_proj_weight`) would produce.

use serde_json::Value;

use crate::lm;
use crate::lm::cache::KvCache;
use crate::music::minimax_music3::rng;
use crate::nn::{LayerNorm, Linear};
use crate::ops;
use crate::{quant, Result, SpeechError};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// Reference defaults (`DEFAULT_*` in pocket_tts.py).
pub const DEFAULT_TEMPERATURE: f32 = 0.7;
pub const DEFAULT_LSD_DECODE_STEPS: usize = 1;
pub const DEFAULT_EOS_THRESHOLD: f32 = -4.0;
/// `DEFAULT_AUDIO_PROMPT`: the fallback voice is a separate download, so
/// the Rust caller always supplies conditioning embeddings.
pub const DEFAULT_AUDIO_PROMPT: &str = "alba";

/// `FlowConfig`.
#[derive(Debug, Clone)]
pub struct FlowConfig {
    pub dim: usize,
    pub depth: usize,
}

/// `FlowLMTransformerConfig`.
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub hidden_scale: usize,
    pub max_period: f32,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
}

/// `LookupTable` (the tokenizer fields are caller concerns; the table
/// geometry drives the embedding shape check).
#[derive(Debug, Clone)]
pub struct LookupTableConfig {
    pub dim: usize,
    pub n_bins: usize,
}

/// `FlowLMConfig`.
#[derive(Debug, Clone)]
pub struct FlowLMConfig {
    pub flow: FlowConfig,
    pub transformer: TransformerConfig,
    pub lookup_table: LookupTableConfig,
}

/// `QuantizerConfig`.
#[derive(Debug, Clone)]
pub struct QuantizerConfig {
    pub dimension: usize,
    pub output_dimension: usize,
}

/// `MimiConfig` reduced to the fields the latent path needs plus the
/// identity fields the family config records.
#[derive(Debug, Clone)]
pub struct MimiConfig {
    pub sample_rate: u32,
    pub frame_rate: f32,
    pub quantizer: QuantizerConfig,
}

/// The family `ModelConfig` (`ModelConfig.from_dict`).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_type: String,
    pub flow_lm: FlowLMConfig,
    pub mimi: MimiConfig,
}

impl ModelConfig {
    pub fn from_json(value: &Value) -> Result<Self> {
        let model_type = value
            .get("model_type")
            .and_then(Value::as_str)
            .unwrap_or("pocket_tts")
            .to_owned();
        let flow_lm_value = value.get("flow_lm").ok_or_else(|| SpeechError::BadConfig {
            field: "flow_lm".to_string(),
            why: "PocketTTS requires flow_lm config section".to_string(),
        })?;
        let mimi_value = value.get("mimi").ok_or_else(|| SpeechError::BadConfig {
            field: "mimi".to_string(),
            why: "PocketTTS requires mimi config section".to_string(),
        })?;
        let flow = |v: &Value| -> Result<FlowConfig> {
            let u = |k: &str| {
                v.get(k)
                    .and_then(Value::as_u64)
                    .map(|n| n as usize)
                    .ok_or_else(|| SpeechError::BadConfig {
                        field: k.to_string(),
                        why: "flow config field is missing".to_string(),
                    })
            };
            Ok(FlowConfig {
                dim: u("dim")?,
                depth: u("depth")?,
            })
        };
        let transformer = |v: &Value| -> Result<TransformerConfig> {
            let u = |k: &str| {
                v.get(k)
                    .and_then(Value::as_u64)
                    .map(|n| n as usize)
                    .ok_or_else(|| SpeechError::BadConfig {
                        field: k.to_string(),
                        why: "transformer config field is missing".to_string(),
                    })
            };
            let cfg = TransformerConfig {
                hidden_scale: u("hidden_scale")?,
                max_period: v
                    .get("max_period")
                    .and_then(Value::as_f64)
                    .unwrap_or(10000.0) as f32,
                d_model: u("d_model")?,
                num_heads: u("num_heads")?,
                num_layers: u("num_layers")?,
            };
            // head_dim is d_model / num_heads. Zero heads panics on the
            // division, and a remainder silently drops trailing channels
            // from every head, so both are rejected at load time.
            if cfg.num_heads == 0 {
                return Err(SpeechError::BadConfig {
                    field: "num_heads".to_string(),
                    why: "num_heads must be > 0".to_string(),
                });
            }
            if cfg.d_model % cfg.num_heads != 0 {
                return Err(SpeechError::BadConfig {
                    field: "d_model".to_string(),
                    why: format!(
                        "d_model {} must be divisible by num_heads {}",
                        cfg.d_model, cfg.num_heads
                    ),
                });
            }
            Ok(cfg)
        };
        let lookup_table = |v: &Value| -> Result<LookupTableConfig> {
            Ok(LookupTableConfig {
                dim: v.get("dim").and_then(Value::as_u64).unwrap_or(0) as usize,
                n_bins: v.get("n_bins").and_then(Value::as_u64).ok_or_else(|| {
                    SpeechError::BadConfig {
                        field: "n_bins".to_string(),
                        why: "lookup_table config field is missing".to_string(),
                    }
                })? as usize,
            })
        };
        let quantizer = |v: &Value| -> Result<QuantizerConfig> {
            let u = |k: &str| {
                v.get(k)
                    .and_then(Value::as_u64)
                    .map(|n| n as usize)
                    .ok_or_else(|| SpeechError::BadConfig {
                        field: k.to_string(),
                        why: "quantizer config field is missing".to_string(),
                    })
            };
            Ok(QuantizerConfig {
                dimension: u("dimension")?,
                output_dimension: u("output_dimension")?,
            })
        };
        Ok(Self {
            model_type,
            flow_lm: FlowLMConfig {
                flow: flow(flow_lm_value.get("flow").unwrap_or(&Value::Null))?,
                transformer: transformer(flow_lm_value.get("transformer").unwrap_or(&Value::Null))?,
                lookup_table: lookup_table(
                    flow_lm_value.get("lookup_table").unwrap_or(&Value::Null),
                )?,
            },
            mimi: MimiConfig {
                sample_rate: mimi_value
                    .get("sample_rate")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| SpeechError::BadConfig {
                        field: "sample_rate".to_string(),
                        why: "mimi config field is missing".to_string(),
                    })? as u32,
                frame_rate: mimi_value
                    .get("frame_rate")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| SpeechError::BadConfig {
                        field: "frame_rate".to_string(),
                        why: "mimi config field is missing".to_string(),
                    })? as f32,
                quantizer: quantizer(mimi_value.get("quantizer").unwrap_or(&Value::Null))?,
            },
        })
    }
}

/// Sampling and generation options with the reference defaults.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    /// Noise standard deviation is `sqrt(temperature)`.
    pub temperature: f32,
    pub lsd_decode_steps: usize,
    /// `None` clamps nothing (the reference default).
    pub noise_clamp: Option<f32>,
    pub eos_threshold: f32,
    /// `None` uses the short-text branch of `prepare_text_prompt` (3)
    /// plus 2, because the token-only generate path cannot recover the
    /// word count. Pass an explicit value for other texts.
    pub frames_after_eos: Option<usize>,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            temperature: DEFAULT_TEMPERATURE,
            lsd_decode_steps: DEFAULT_LSD_DECODE_STEPS,
            noise_clamp: None,
            eos_threshold: DEFAULT_EOS_THRESHOLD,
            frames_after_eos: None,
        }
    }
}

/// One generated run: the raw flow latents per step, the scaled
/// `decoding_input` latents the pocket Mimi decoder would consume, the
/// per-step EOS flags, and why the loop stopped.
#[derive(Debug, Clone)]
pub struct Generated {
    pub latents: Vec<Vec<f32>>,
    pub decoding_input: Vec<Vec<f32>>,
    pub eos_flags: Vec<bool>,
    pub eos_step: Option<usize>,
    pub stopped_by: StopReason,
    /// The transformer output rows (pre-norm) per call when captured:
    /// conditioning prefill rows, text prefill rows, then one row per
    /// step. Fixture-only.
    pub transformer_outputs: Option<Vec<Vec<f32>>>,
}

/// Why the generate loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    EndOfSpeech,
    MaxSteps,
}

/// The lookup-table text conditioner: `Embedding(n_bins + 1, dim)` plus
/// the optional output projection when `dim != d_model`.
pub struct LutConditioner {
    embed: Vec<f32>,
    embed_cols: usize,
    output_proj: Option<Linear>,
}

impl LutConditioner {
    fn load(weights: &lm::Weights, config: &FlowLMConfig, d_model: usize) -> Result<Self> {
        let table = &config.lookup_table;
        let embed = weights.tensor_absolute(
            "flow_lm.conditioner.embed.weight",
            &[table.n_bins + 1, table.dim],
        )?;
        let output_proj = if table.dim != d_model {
            Some(weights.linear_absolute(
                "flow_lm.conditioner.output_proj",
                table.dim,
                d_model,
                lm::Bias::None,
            )?)
        } else {
            None
        };
        Ok(Self {
            embed,
            embed_cols: table.dim,
            output_proj,
        })
    }

    /// Token ids (caller-tokenized text) to embeddings `[len, d_model]`.
    pub fn forward(&self, tokens: &[i32]) -> Result<Vec<f32>> {
        let dim = self.embed_cols;
        let mut out = Vec::with_capacity(tokens.len() * dim);
        for &token in tokens {
            let index = usize::try_from(token).map_err(|_| SpeechError::Input {
                why: format!("negative text token {token}"),
            })?;
            if (index + 1) * dim > self.embed.len() {
                return Err(SpeechError::Input {
                    why: format!(
                        "text token {token} outside the {}-row lookup table",
                        self.embed.len() / dim
                    ),
                });
            }
            out.extend_from_slice(&self.embed[index * dim..(index + 1) * dim]);
        }
        if let Some(proj) = &self.output_proj {
            let len = tokens.len();
            out = proj.forward(&out, len);
        }
        Ok(out)
    }
}

/// The reference `apply_rope` over interleaved pairs in head-major
/// `[heads, time, head_dim]` layout with per-position
/// `freq = max_period^(-2 d / head_dim)`.
fn apply_rope(
    q: &mut [f32],
    k: &mut [f32],
    time: usize,
    heads: usize,
    head_dim: usize,
    offset: usize,
    max_period: f32,
) {
    let half = head_dim / 2;
    let rotate = |x: &mut [f32]| {
        for head in 0..heads {
            for t in 0..time {
                let base = (head * time + t) * head_dim;
                for pair in 0..half {
                    let freq = (max_period).powf(-2.0 * pair as f32 / head_dim as f32);
                    let angle = (offset + t) as f32 * freq;
                    let rotr = angle.cos();
                    let roti = angle.sin();
                    let real = x[base + 2 * pair];
                    let imag = x[base + 2 * pair + 1];
                    x[base + 2 * pair] = real * rotr - imag * roti;
                    x[base + 2 * pair + 1] = real * roti + imag * rotr;
                }
            }
        }
    };
    rotate(q);
    rotate(k);
}

struct StreamingLayer {
    norm1: LayerNorm,
    in_proj: Linear,
    out_proj: Linear,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
    heads: usize,
    head_dim: usize,
    max_period: f32,
}

impl StreamingLayer {
    /// The reference `from_config` never passes a layer scale
    /// (`StreamingTransformer`'s `layer_scale` default), so the port
    /// always builds unscaled residuals.
    #[allow(clippy::too_many_arguments)]
    fn load(
        weights: &lm::Weights,
        prefix: &str,
        d_model: usize,
        dim_feedforward: usize,
        heads: usize,
        max_period: f32,
    ) -> Result<Self> {
        Ok(Self {
            norm1: LayerNorm::load(
                &weights.shards()[0],
                &format!("{prefix}norm1"),
                d_model,
                1e-5,
            )?,
            in_proj: weights.linear_absolute(
                &format!("{prefix}self_attn.in_proj"),
                d_model,
                3 * d_model,
                lm::Bias::None,
            )?,
            out_proj: weights.linear_absolute(
                &format!("{prefix}self_attn.out_proj"),
                d_model,
                d_model,
                lm::Bias::None,
            )?,
            norm2: LayerNorm::load(
                &weights.shards()[0],
                &format!("{prefix}norm2"),
                d_model,
                1e-5,
            )?,
            linear1: weights.linear_absolute(
                &format!("{prefix}linear1"),
                d_model,
                dim_feedforward,
                lm::Bias::None,
            )?,
            linear2: weights.linear_absolute(
                &format!("{prefix}linear2"),
                dim_feedforward,
                d_model,
                lm::Bias::None,
            )?,
            heads,
            head_dim: d_model / heads,
            max_period,
        })
    }

    /// `[time, d_model]` through the pre-norm block with a causal
    /// additive mask over the cache.
    fn forward(&self, x: &mut [f32], time: usize, cache: Option<&mut KvCache>) {
        let d_model = self.norm1.width();
        let mut attended = x.to_vec();
        self.norm1.apply(&mut attended, time);
        let projected = self.in_proj.forward(&attended, time);
        // The fused projection reshapes to [b, t, 3, heads, head_dim]:
        // q, k, v are contiguous thirds of the output width.
        let third = d_model;
        let mut qh = vec![0.0f32; time * d_model];
        let mut kh = vec![0.0f32; time * d_model];
        let mut vh = vec![0.0f32; time * d_model];
        for row in 0..time {
            qh[row * third..(row + 1) * third]
                .copy_from_slice(&projected[row * 3 * third..row * 3 * third + third]);
            kh[row * third..(row + 1) * third]
                .copy_from_slice(&projected[row * 3 * third + third..row * 3 * third + 2 * third]);
            vh[row * third..(row + 1) * third]
                .copy_from_slice(&projected[row * 3 * third + 2 * third..(row + 1) * 3 * third]);
        }
        // Head-major [heads, time, head_dim] for the rope, cache, and
        // attention below.
        let mut qh = lm::engine::head_major(&qh, time, self.heads, self.head_dim);
        let mut kh = lm::engine::head_major(&kh, time, self.heads, self.head_dim);
        let vh = lm::engine::head_major(&vh, time, self.heads, self.head_dim);
        let offset = cache.as_ref().map_or(0, |c| c.offset);
        apply_rope(
            &mut qh,
            &mut kh,
            time,
            self.heads,
            self.head_dim,
            offset,
            self.max_period,
        );
        let (keys, values) = match cache {
            Some(cache) => {
                cache.update_and_fetch(&kh, &vh, self.heads, time, self.head_dim);
                (cache.keys.clone(), cache.values.clone())
            }
            None => (kh.clone(), vh.clone()),
        };
        // Causal attention over [heads, total, head_dim].
        let total = keys.len() / (self.heads * self.head_dim);
        let mut scores = vec![0.0f32; total];
        let mut out = vec![0.0f32; time * d_model];
        for head in 0..self.heads {
            let q_head = &qh[head * time * self.head_dim..(head + 1) * time * self.head_dim];
            let k_head = &keys[head * total * self.head_dim..(head + 1) * total * self.head_dim];
            let v_head = &values[head * total * self.head_dim..(head + 1) * total * self.head_dim];
            for row in 0..time {
                let high = offset + row;
                let q_row = &q_head[row * self.head_dim..(row + 1) * self.head_dim];
                let mut max = f32::NEG_INFINITY;
                for j in 0..=high {
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        dot += q_row[d] * k_head[j * self.head_dim + d];
                    }
                    let score = dot * (self.head_dim as f32).powf(-0.5);
                    scores[j] = score;
                    if score > max {
                        max = score;
                    }
                }
                let mut sum = 0.0f32;
                for score in &mut scores[..=high] {
                    let weight = (*score - max).exp();
                    *score = weight;
                    sum += weight;
                }
                let inv = 1.0 / sum;
                let out_row = &mut out[row * d_model + head * self.head_dim
                    ..row * d_model + (head + 1) * self.head_dim];
                for j in 0..=high {
                    let weight = scores[j] * inv;
                    for d in 0..self.head_dim {
                        out_row[d] += weight * v_head[j * self.head_dim + d];
                    }
                }
            }
        }
        let attended = self.out_proj.forward(&out, time);
        for (row, chunk) in attended.chunks(d_model).enumerate() {
            for (col, value) in chunk.iter().enumerate() {
                x[row * d_model + col] += value;
            }
        }
        let mut fed = x.to_vec();
        self.norm2.apply(&mut fed, time);
        let mut hidden = self.linear1.forward(&fed, time);
        ops::gelu_erf(&mut hidden);
        let fed = self.linear2.forward(&hidden, time);
        for (row, chunk) in fed.chunks(d_model).enumerate() {
            for (col, value) in chunk.iter().enumerate() {
                x[row * d_model + col] += value;
            }
        }
    }
}

/// The streaming transformer (`StreamingTransformer`).
pub struct StreamingTransformer {
    layers: Vec<StreamingLayer>,
}

impl StreamingTransformer {
    fn load(weights: &lm::Weights, config: &TransformerConfig) -> Result<Self> {
        let dim_feedforward = config.hidden_scale * config.d_model;
        let mut layers = Vec::with_capacity(config.num_layers);
        for index in 0..config.num_layers {
            layers.push(StreamingLayer::load(
                weights,
                &format!("flow_lm.transformer.layers.{index}."),
                config.d_model,
                dim_feedforward,
                config.num_heads,
                config.max_period,
            )?);
        }
        Ok(Self { layers })
    }

    /// `[time, d_model]` with per-layer caches; the causal mask rides the
    /// cache offsets.
    fn forward(&self, x: &mut [f32], time: usize, caches: &mut [KvCache]) {
        for (index, layer) in self.layers.iter().enumerate() {
            let mut row = x.to_vec();
            layer.forward(&mut row, time, Some(&mut caches[index]));
            x.copy_from_slice(&row);
        }
    }

    fn make_cache(&self) -> Vec<KvCache> {
        vec![KvCache::new(); self.layers.len()]
    }
}

/// The reference custom `RMSNorm` (ddof=1 sample variance, multiplicative
/// alpha, eps added to the variance).
pub fn rms_norm_ddof1(x: &mut [f32], alpha: &[f32]) {
    let dim = alpha.len();
    // A zero-width norm has no rows to normalize. Returning here avoids
    // the `x.len() / dim` divide by zero.
    if dim == 0 {
        return;
    }
    for row in 0..x.len() / dim {
        let base = row * dim;
        let mean = x[base..base + dim].iter().sum::<f32>() / dim as f32;
        let var = x[base..base + dim]
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f32>()
            / (dim as f32 - 1.0)
            + 1e-5;
        let inv = 1.0 / var.sqrt();
        for (col, value) in x[base..base + dim].iter_mut().enumerate() {
            *value *= alpha[col] * inv;
        }
    }
}

/// `TimestepEmbedder`: sinusoidal features [cos | sin] through
/// Linear / SiLU / Linear / RMSNorm.
pub struct TimestepEmbedder {
    freqs: Vec<f32>,
    linear1: Linear,
    linear2: Linear,
    rms_alpha: Vec<f32>,
    frequency_embedding_size: usize,
}

impl TimestepEmbedder {
    fn load(weights: &lm::Weights, prefix: &str, model_channels: usize) -> Result<Self> {
        let frequency_embedding_size = 256;
        let half = frequency_embedding_size / 2;
        let mut freqs = Vec::with_capacity(half);
        for index in 0..half {
            freqs.push((-max_period_ln() * index as f32 / half as f32).exp());
        }
        Ok(Self {
            freqs,
            linear1: weights.linear_absolute(
                &format!("{prefix}mlp.0"),
                frequency_embedding_size,
                model_channels,
                lm::Bias::Required,
            )?,
            linear2: weights.linear_absolute(
                &format!("{prefix}mlp.2"),
                model_channels,
                model_channels,
                lm::Bias::Required,
            )?,
            rms_alpha: weights
                .tensor_absolute(&format!("{prefix}mlp.3.alpha"), &[model_channels])
                .or_else(|_| {
                    weights.tensor_absolute(&format!("{prefix}mlp.3.weight"), &[model_channels])
                })?,
            frequency_embedding_size,
        })
    }

    /// `t` scalar -> `[model_channels]`.
    pub fn forward(&self, t: f32) -> Vec<f32> {
        let half = self.frequency_embedding_size / 2;
        let mut embedding = vec![0.0f32; self.frequency_embedding_size];
        for (index, &freq) in self.freqs.iter().enumerate() {
            let args = t * freq;
            embedding[index] = args.cos();
            embedding[half + index] = args.sin();
        }
        let mut out = self.linear1.forward(&embedding, 1);
        for value in &mut out {
            *value /= 1.0 + (-*value).exp();
        }
        let out = self.linear2.forward(&out, 1);
        let mut out = out;
        rms_norm_ddof1(&mut out, &self.rms_alpha);
        out
    }
}

fn max_period_ln() -> f32 {
    10000.0f32.ln()
}

/// AdaLN res-block (`ResBlock`).
struct ResBlock {
    in_ln: LayerNorm,
    mlp1: Linear,
    mlp2: Linear,
    ada_ln: Linear,
    channels: usize,
}

impl ResBlock {
    fn load(weights: &lm::Weights, prefix: &str, channels: usize) -> Result<Self> {
        Ok(Self {
            in_ln: LayerNorm::load(
                &weights.shards()[0],
                &format!("{prefix}in_ln"),
                channels,
                1e-6,
            )?,
            mlp1: weights.linear_absolute(
                &format!("{prefix}mlp.0"),
                channels,
                channels,
                lm::Bias::Required,
            )?,
            mlp2: weights.linear_absolute(
                &format!("{prefix}mlp.2"),
                channels,
                channels,
                lm::Bias::Required,
            )?,
            ada_ln: weights.linear_absolute(
                &format!("{prefix}adaLN_modulation.1"),
                channels,
                3 * channels,
                lm::Bias::Required,
            )?,
            channels,
        })
    }

    fn forward(&self, x: &mut [f32], condition: &[f32]) {
        let channels = self.channels;
        // The reference adaLN_modulation list is [SiLU, Linear]: the
        // condition is silu-ed BEFORE the projection.
        let mut gated = condition.to_vec();
        for value in &mut gated {
            *value /= 1.0 + (-*value).exp();
        }
        let modulation = self.ada_ln.forward(&gated, 1);
        let shift = &modulation[..channels];
        let scale = &modulation[channels..2 * channels];
        let gate = &modulation[2 * channels..];
        let mut h = x.to_vec();
        self.in_ln.apply(&mut h, 1);
        for (col, value) in h.iter_mut().enumerate() {
            *value = *value * (1.0 + scale[col]) + shift[col];
        }
        let h = self.mlp1.forward(&h, 1);
        let mut h = h;
        for value in &mut h {
            *value /= 1.0 + (-*value).exp();
        }
        let h = self.mlp2.forward(&h, 1);
        for (col, value) in h.iter().enumerate() {
            x[col] += gate[col] * value;
        }
    }
}

/// `FinalLayer`: affine-free norm, modulation, output linear.
struct FinalLayer {
    norm_width: usize,
    linear: Linear,
    ada_ln: Linear,
}

impl FinalLayer {
    fn load(weights: &lm::Weights, prefix: &str, channels: usize, out: usize) -> Result<Self> {
        Ok(Self {
            norm_width: channels,
            linear: weights.linear_absolute(
                &format!("{prefix}linear"),
                channels,
                out,
                lm::Bias::Required,
            )?,
            ada_ln: weights.linear_absolute(
                &format!("{prefix}adaLN_modulation.1"),
                channels,
                2 * channels,
                lm::Bias::Required,
            )?,
        })
    }

    fn forward(&self, x: &mut [f32], condition: &[f32]) -> Vec<f32> {
        let channels = self.norm_width;
        // The reference adaLN_modulation list is [SiLU, Linear].
        let mut gated = condition.to_vec();
        for value in &mut gated {
            *value /= 1.0 + (-*value).exp();
        }
        let modulation = self.ada_ln.forward(&gated, 1);
        let shift = &modulation[..channels];
        let scale = &modulation[channels..];
        // Affine-free layer norm.
        let mean = x.iter().sum::<f32>() / channels as f32;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / channels as f32;
        let inv = 1.0 / (var + 1e-6).sqrt();
        for (col, value) in x.iter_mut().enumerate() {
            *value = (*value - mean) * inv * (1.0 + scale[col]) + shift[col];
        }
        self.linear.forward(x, 1)
    }
}

/// `SimpleMLPAdaLN`: the flow network conditioned on the transformer
/// hidden state and the two integration times.
pub struct SimpleMlpAdaln {
    time_embed: [TimestepEmbedder; 2],
    cond_embed: Linear,
    input_proj: Linear,
    res_blocks: Vec<ResBlock>,
    final_layer: FinalLayer,
    model_channels: usize,
}

impl SimpleMlpAdaln {
    fn load(weights: &lm::Weights, config: &ModelConfig) -> Result<Self> {
        let model_channels = config.flow_lm.flow.dim;
        let latent_dim = config.mimi.quantizer.dimension;
        let cond_dim = config.flow_lm.transformer.d_model;
        Ok(Self {
            time_embed: [
                TimestepEmbedder::load(weights, "flow_lm.flow_net.time_embed.0.", model_channels)?,
                TimestepEmbedder::load(weights, "flow_lm.flow_net.time_embed.1.", model_channels)?,
            ],
            cond_embed: weights.linear_absolute(
                "flow_lm.flow_net.cond_embed",
                cond_dim,
                model_channels,
                lm::Bias::Required,
            )?,
            input_proj: weights.linear_absolute(
                "flow_lm.flow_net.input_proj",
                latent_dim,
                model_channels,
                lm::Bias::Required,
            )?,
            res_blocks: (0..config.flow_lm.flow.depth)
                .map(|index| {
                    ResBlock::load(
                        weights,
                        &format!("flow_lm.flow_net.res_blocks.{index}."),
                        model_channels,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            final_layer: FinalLayer::load(
                weights,
                "flow_lm.flow_net.final_layer.",
                model_channels,
                latent_dim,
            )?,
            model_channels,
        })
    }

    /// `flow_net(c, s, t, x)`: `[ldim]` in, `[ldim]` out.
    pub fn forward(&self, condition: &[f32], s: f32, t: f32, x: &[f32]) -> Vec<f32> {
        let mut hidden = self.input_proj.forward(x, 1);
        let mut t_combined = vec![0.0f32; self.model_channels];
        for (index, value) in [s, t].iter().enumerate() {
            let embedded = self.time_embed[index].forward(*value);
            for (sum, add) in t_combined.iter_mut().zip(embedded) {
                *sum += add;
            }
        }
        for value in &mut t_combined {
            *value /= 2.0;
        }
        let embedded = self.cond_embed.forward(condition, 1);
        for (sum, add) in t_combined.iter_mut().zip(embedded) {
            *sum += add;
        }
        for block in &self.res_blocks {
            block.forward(&mut hidden, &t_combined);
        }
        self.final_layer.forward(&mut hidden, &t_combined)
    }
}

/// The flow-LM: conditioner, streaming transformer, flow net, and the
/// latent de-normalization statistics.
pub struct FlowLm {
    pub conditioner: LutConditioner,
    transformer: StreamingTransformer,
    pub flow_net: SimpleMlpAdaln,
    input_linear: Linear,
    out_norm: LayerNorm,
    out_eos: Linear,
    pub bos_emb: Vec<f32>,
    pub emb_std: Vec<f32>,
    pub emb_mean: Vec<f32>,
    pub ldim: usize,
    pub dim: usize,
}

impl FlowLm {
    fn load(weights: &lm::Weights, config: &ModelConfig) -> Result<Self> {
        let d_model = config.flow_lm.transformer.d_model;
        let ldim = config.mimi.quantizer.dimension;
        Ok(Self {
            conditioner: LutConditioner::load(weights, &config.flow_lm, d_model)?,
            transformer: StreamingTransformer::load(weights, &config.flow_lm.transformer)?,
            flow_net: SimpleMlpAdaln::load(weights, config)?,
            input_linear: weights.linear_absolute(
                "flow_lm.input_linear",
                ldim,
                d_model,
                lm::Bias::None,
            )?,
            out_norm: LayerNorm::load(&weights.shards()[0], "flow_lm.out_norm", d_model, 1e-5)?,
            out_eos: weights.linear_absolute("flow_lm.out_eos", d_model, 1, lm::Bias::Required)?,
            bos_emb: weights.tensor_absolute("flow_lm.bos_emb", &[ldim])?,
            emb_std: weights.tensor_absolute("flow_lm.emb_std", &[ldim])?,
            emb_mean: weights.tensor_absolute("flow_lm.emb_mean", &[ldim])?,
            ldim,
            dim: d_model,
        })
    }

    pub fn make_cache(&self) -> Vec<KvCache> {
        self.transformer.make_cache()
    }
}

/// One `flow_lm` step over a one-position input (`_sample_next_latent`
/// with a single-position sequence): appends the projected latent to
/// the caches and returns the normalized last hidden state `[dim]` plus
/// the EOS head value.
fn step_hidden(
    flow: &FlowLm,
    latent: &[f32],
    caches: &mut [KvCache],
    sink: Option<&mut Vec<Vec<f32>>>,
) -> Result<(Vec<f32>, f32)> {
    let mut projected = flow.input_linear.forward(latent, 1);
    flow.transformer.forward(&mut projected, 1, caches);
    if let Some(sink) = sink {
        sink.push(projected.clone());
    }
    flow.out_norm.apply(&mut projected, 1);
    let eos = flow.out_eos.forward(&projected, 1);
    Ok((projected, eos[0]))
}

/// The reference `lsd_decode`: Euler integration of the conditioned
/// flow from `x_0` over `num_steps` uniform segments.
pub fn lsd_decode(
    flow_net: &SimpleMlpAdaln,
    condition: &[f32],
    x0: &[f32],
    num_steps: usize,
) -> Result<Vec<f32>> {
    if num_steps == 0 {
        return Err(SpeechError::Input {
            why: "lsd_decode_steps must be > 0 for generation".to_string(),
        });
    }
    let mut current = x0.to_vec();
    for index in 0..num_steps {
        let s = index as f32 / num_steps as f32;
        let t = (index + 1) as f32 / num_steps as f32;
        let flow_dir = flow_net.forward(condition, s, t, &current);
        for (value, dir) in current.iter_mut().zip(flow_dir) {
            *value += dir / num_steps as f32;
        }
    }
    Ok(current)
}

/// The loaded family model.
pub struct Model {
    pub config: ModelConfig,
    pub flow_lm: FlowLm,
}

impl Model {
    /// Opens a checkpoint directory (`config.json` + safetensors). The
    /// reference splits the flow-LM and Mimi weights across two files;
    /// this port reads the flow-LM side, which carries every tensor the
    /// latent path needs.
    pub fn open(dir: &std::path::Path) -> Result<Self> {
        let config_value = quant::read_json(&dir.join("config.json"))?;
        let weights = lm::Weights::open(dir)?;
        Self::load(&weights, &config_value)
    }

    /// Loads from opened weights and the raw config dict.
    pub fn load(weights: &lm::Weights, config: &Value) -> Result<Self> {
        let config = ModelConfig::from_json(config)?;
        let flow_lm = FlowLm::load(weights, &config)?;
        Ok(Self { config, flow_lm })
    }

    /// The reference `get_state_for_audio_prompt` plus the text pass:
    /// prefills the conditioning embeddings (a loaded voice through the
    /// Mimi encoder and `speaker_proj_weight`; caller-supplied here) and
    /// then the tokenized text, leaving the caches ready for the
    /// generation steps. Returns the caches.
    pub fn start_state(
        &self,
        conditioning: &[f32],
        text_tokens: &[i32],
        caches: &mut [KvCache],
    ) -> Result<()> {
        self.start_state_capture(conditioning, text_tokens, caches, None)
    }

    /// [`Model::start_state`] with an optional transformer-output sink
    /// (one entry per prefill, the pre-norm rows).
    pub fn start_state_capture(
        &self,
        conditioning: &[f32],
        text_tokens: &[i32],
        caches: &mut [KvCache],
        mut sink: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<()> {
        if conditioning.len() % self.flow_lm.dim != 0 {
            return Err(SpeechError::Input {
                why: format!(
                    "conditioning length {} is not a multiple of d_model {}",
                    conditioning.len(),
                    self.flow_lm.dim
                ),
            });
        }
        let frames = conditioning.len() / self.flow_lm.dim;
        // The audio conditioning rides the text-embedding axis; the
        // empty text and empty latent inputs contribute no positions.
        let mut rows = conditioning.to_vec();
        self.flow_lm.transformer.forward(&mut rows, frames, caches);
        if let Some(sink) = sink.as_deref_mut() {
            sink.push(rows.clone());
        }
        if !text_tokens.is_empty() {
            let text = self.flow_lm.conditioner.forward(text_tokens)?;
            let len = text.len() / self.flow_lm.dim;
            let mut rows = text;
            self.flow_lm.transformer.forward(&mut rows, len, caches);
            if let Some(sink) = sink {
                sink.push(rows.clone());
            }
        }
        Ok(())
    }

    /// The reference single-chunk `_generate_audio_stream_short_text`
    /// down to the scaled latents. `conditioning` and `text_tokens` are
    /// caller-owned; `max_steps` caps the loop (the reference derives it
    /// from the word count and frame rate); `noise` draws one
    /// `[ldim]` standard-normal vector per step in consumption order
    /// (`temp` scales it, `noise_clamp` clips it).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_latents(
        &mut self,
        conditioning: &[f32],
        text_tokens: &[i32],
        params: &GenerateParams,
        max_steps: usize,
        keys: &mut rng::KeySequence,
    ) -> Result<Generated> {
        self.generate_latents_with_noise(
            conditioning,
            text_tokens,
            params,
            max_steps,
            &mut |width| rng::normal(keys.next(), width),
        )
    }

    /// [`Model::generate_latents`] with an injected standard-normal
    /// source: one `[ldim]` draw per step in consumption order. The
    /// fixture tests replay the recorded arrays.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_latents_with_noise(
        &mut self,
        conditioning: &[f32],
        text_tokens: &[i32],
        params: &GenerateParams,
        max_steps: usize,
        noise: &mut dyn FnMut(usize) -> Vec<f32>,
    ) -> Result<Generated> {
        self.generate_latents_inner(conditioning, text_tokens, params, max_steps, noise, false)
    }

    /// [`Model::generate_latents_with_noise`] capturing the transformer
    /// output rows for fixture parity.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_latents_with_trace(
        &mut self,
        conditioning: &[f32],
        text_tokens: &[i32],
        params: &GenerateParams,
        max_steps: usize,
        noise: &mut dyn FnMut(usize) -> Vec<f32>,
    ) -> Result<Generated> {
        self.generate_latents_inner(conditioning, text_tokens, params, max_steps, noise, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_latents_inner(
        &mut self,
        conditioning: &[f32],
        text_tokens: &[i32],
        params: &GenerateParams,
        max_steps: usize,
        noise: &mut dyn FnMut(usize) -> Vec<f32>,
        capture: bool,
    ) -> Result<Generated> {
        if params.lsd_decode_steps == 0 {
            return Err(SpeechError::Input {
                why: "lsd_decode_steps must be > 0 for generation".to_string(),
            });
        }
        // The reference guess comes from the word count via
        // prepare_text_prompt; the caller-owned token ids cannot
        // recover it, so the default falls back to the short-text
        // branch value.
        let frames_after_eos = params.frames_after_eos.unwrap_or(3 + 2);
        let mut caches = self.flow_lm.make_cache();
        let mut transformer_outputs = Vec::new();
        let mut sink = if capture {
            Some(&mut transformer_outputs)
        } else {
            None
        };
        self.start_state_capture(conditioning, text_tokens, &mut caches, sink.as_deref_mut())?;

        let ldim = self.flow_lm.ldim;
        // The hint only; the vectors still grow past it. Capping keeps a
        // huge caller `max_steps` from allocating up front.
        let capacity_hint = max_steps.min(4096);
        let mut latents: Vec<Vec<f32>> = Vec::with_capacity(capacity_hint);
        let mut decoding_input = Vec::with_capacity(capacity_hint);
        let mut eos_flags = Vec::with_capacity(capacity_hint);
        let mut eos_step = None;

        for step in 0..max_steps {
            // The first step's NaN seed is replaced by the bos embedding
            // (`sequence = mx.where(isnan(sequence), bos, sequence)`).
            let input: Vec<f32> = match latents.last() {
                Some(latent) => latent.clone(),
                None => self.flow_lm.bos_emb.clone(),
            };
            let (hidden, eos_score) =
                step_hidden(&self.flow_lm, &input, &mut caches, sink.as_deref_mut())?;
            let is_eos = eos_score > params.eos_threshold;
            if is_eos && eos_step.is_none() {
                eos_step = Some(step);
            }
            // Noise: temp-scaled standard normals, optionally clamped.
            let mut draw = noise(ldim);
            if draw.len() != ldim {
                return Err(SpeechError::Input {
                    why: format!("noise draw width {} != {ldim}", draw.len()),
                });
            }
            let std = params.temperature.max(0.0).sqrt();
            for value in &mut draw {
                *value *= std;
                if let Some(clamp) = params.noise_clamp {
                    *value = value.clamp(-clamp, clamp);
                }
            }
            let latent = lsd_decode(
                &self.flow_lm.flow_net,
                &hidden,
                &draw,
                params.lsd_decode_steps,
            )?;
            let mut scaled = latent.clone();
            for (value, (std, mean)) in scaled
                .iter_mut()
                .zip(self.flow_lm.emb_std.iter().zip(&self.flow_lm.emb_mean))
            {
                *value = *value * std + mean;
            }
            latents.push(latent);
            decoding_input.push(scaled);
            eos_flags.push(is_eos);
            if let Some(eos) = eos_step {
                if step >= eos + frames_after_eos {
                    break;
                }
            }
        }
        Ok(Generated {
            latents,
            decoding_input,
            eos_flags,
            eos_step,
            stopped_by: if eos_step.is_some() {
                StopReason::EndOfSpeech
            } else {
                StopReason::MaxSteps
            },
            transformer_outputs: if capture {
                Some(transformer_outputs)
            } else {
                None
            },
        })
    }
}

/// The reference `prepare_text_prompt`: normalization plus the
/// frames-after-eos guess. Returns the normalized text and the guess.
pub fn prepare_text_prompt(text: &str) -> Result<(String, usize)> {
    let text = text.trim();
    if text.is_empty() {
        return Err(SpeechError::Input {
            why: "Text prompt cannot be empty".to_string(),
        });
    }
    // Python str.replace runs one non-overlapping pass per call; the
    // chain mirrors the three reference calls.
    #[allow(clippy::almost_complete_range)]
    let text = text
        .replace(['\n'], " ")
        .replace(['\r'], " ")
        .replace("  ", " ");
    let word_count = text.split(' ').filter(|w| !w.is_empty()).count();
    let frames_after_eos_guess = if word_count <= 4 { 3 } else { 1 };
    let mut chars: Vec<char> = text.chars().collect();
    if let Some(first) = chars.first_mut() {
        if !first.is_uppercase() {
            let upper = first.to_uppercase().next().unwrap_or(*first);
            *first = upper;
        }
    }
    let mut text: String = chars.into_iter().collect();
    if let Some(last) = text.chars().last() {
        if last.is_alphanumeric() {
            text.push('.');
        }
    }
    if text.split(' ').filter(|w| !w.is_empty()).count() < 5 {
        text = format!("        {text}");
    }
    Ok((text, frames_after_eos_guess))
}

#[cfg(test)]
mod tests;
