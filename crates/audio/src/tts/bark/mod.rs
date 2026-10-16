//! Bark TTS family (three-stage GPT: semantic, coarse, fine).
//!
//! Port of `mlx_audio/tts/models/bark/` (bark.py, pipeline.py,
//! isftnet.py) at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/bark):
//! a semantic GPT autoregresses text-conditioned tokens, a coarse GPT
//! turns them into the first two EnCodec codebooks through sliding
//! windows, a non-causal fine GPT fills the remaining six codebooks, and
//! the eight codebooks decode through EnCodec. See [README.md](README.md)
//! for the verification evidence and open items.
//!
//! Scope of this port, matching the reference `Pipeline` with
//! `use_kv_caching=True`: batch size 1, non-streaming, no voice prompt
//! (the `.npz` history arrays are caller-owned and accepted as optional
//! slices). Text tokenization is a caller concern (the reference loads
//! a BERT tokenizer); the caller passes the raw text token ids.

use crate::codec::encodec::Encodec;
use crate::lm::cache::KvCache;
use crate::nn::{LayerNorm, Linear};
use crate::ops;
use crate::tts::sampling;
use crate::{quant, Result, SpeechError};
use serde_json::Value;

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

pub const TEXT_ENCODING_OFFSET: i32 = 10_048;
pub const SEMANTIC_PAD_TOKEN: i32 = 10_000;
pub const TEXT_PAD_TOKEN: i32 = 129_595;
pub const SEMANTIC_INFER_TOKEN: i32 = 129_599;
pub const CONTEXT_WINDOW_SIZE: usize = 1024;
pub const SEMANTIC_RATE_HZ: f32 = 49.9;
pub const SEMANTIC_VOCAB_SIZE: usize = 10_000;
pub const CODEBOOK_SIZE: usize = 1024;
pub const N_COARSE_CODEBOOKS: usize = 2;
pub const N_FINE_CODEBOOKS: usize = 8;
pub const COARSE_RATE_HZ: f32 = 75.0;
pub const COARSE_SEMANTIC_PAD_TOKEN: i32 = 12_048;
pub const COARSE_INFER_TOKEN: i32 = 12_050;
pub const SAMPLE_RATE: u32 = 24_000;

/// One GPT stack config (`SemanticConfig` / `CoarseAcousticsConfig`).
#[derive(Debug, Clone)]
pub struct GptConfig {
    pub block_size: usize,
    pub input_vocab_size: usize,
    pub output_vocab_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    /// Attention and MLP biases (`bias` field; the layernorms are always
    /// weight-only in the causal stacks).
    pub bias: bool,
}

/// `FineAcousticsConfig`.
#[derive(Debug, Clone)]
pub struct FineConfig {
    pub block_size: usize,
    pub input_vocab_size: usize,
    pub output_vocab_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub n_codes_total: usize,
    pub n_codes_given: usize,
}

/// The family `ModelConfig`.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub semantic_config: GptConfig,
    pub coarse_acoustics_config: GptConfig,
    pub fine_acoustics_config: FineConfig,
    pub sample_rate: u32,
}

impl ModelConfig {
    pub fn from_json(value: &Value) -> Result<Self> {
        let u_of = |v: &Value, k: &str, default: usize| {
            v.get(k)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(default)
        };
        let gpt = |v: &Value| -> Result<GptConfig> {
            let u = |k: &str, default: usize| u_of(v, k, default);
            let cfg = GptConfig {
                block_size: u("block_size", 1024),
                input_vocab_size: u("input_vocab_size", 0),
                output_vocab_size: u("output_vocab_size", 0),
                n_layer: u("n_layer", 12),
                n_head: u("n_head", 12),
                n_embd: u("n_embd", 768),
                bias: v.get("bias").and_then(Value::as_bool).unwrap_or(false),
            };
            // head_dim is n_embd / n_head. A zero head count panics on the
            // division, and a remainder silently drops trailing channels.
            if cfg.n_head == 0 || cfg.n_embd % cfg.n_head != 0 {
                return Err(SpeechError::BadConfig {
                    field: "n_head".to_string(),
                    why: format!(
                        "n_embd {} must be a positive multiple of n_head {}",
                        cfg.n_embd, cfg.n_head
                    ),
                });
            }
            Ok(cfg)
        };
        let fine = |v: &Value| -> Result<FineConfig> {
            let g = gpt(v)?;
            Ok(FineConfig {
                block_size: g.block_size,
                input_vocab_size: g.input_vocab_size,
                output_vocab_size: g.output_vocab_size,
                n_layer: g.n_layer,
                n_head: g.n_head,
                n_embd: g.n_embd,
                n_codes_total: u_of(v, "n_codes_total", 8),
                n_codes_given: u_of(v, "n_codes_given", 1),
            })
        };
        Ok(Self {
            semantic_config: gpt(value.get("semantic_config").unwrap_or(&Value::Null))?,
            coarse_acoustics_config: gpt(value
                .get("coarse_acoustics_config")
                .unwrap_or(&Value::Null))?,
            fine_acoustics_config: fine(
                value.get("fine_acoustics_config").unwrap_or(&Value::Null),
            )?,
            sample_rate: u_of(value, "sample_rate", 24000) as u32,
        })
    }
}

/// The causal GPT block (`Block`): weight-only layernorms, fused
/// attention projection, gelu MLP.
struct Block {
    layernorm_1: LayerNorm,
    att_proj: Linear,
    out_proj: Linear,
    layernorm_2: LayerNorm,
    mlp_in: Linear,
    mlp_out: Linear,
    n_head: usize,
    head_dim: usize,
    /// The reference causal blocks pass a float `[block_size,
    /// block_size]` additive mask tensor (`attn.bias`, a learned
    /// parameter in the checkpoints) straight into
    /// `mx.fast.scaled_dot_product_attention`, which adds it to the
    /// scores: query `offset + row` adds `mask[offset + row][j]` to key
    /// `j` over the whole buffer. `None` for the fine stack's
    /// non-causal attention.
    mask: Option<Vec<f32>>,
    mask_side: usize,
}

impl Block {
    fn forward(&self, x: &mut [f32], time: usize, cache: Option<&mut KvCache>) {
        let n_embd = self.layernorm_1.width();
        let mut attended = x.to_vec();
        self.layernorm_1.apply(&mut attended, time);
        let projected = self.att_proj.forward(&attended, time);
        let mut q = vec![0.0f32; time * n_embd];
        let mut k = vec![0.0f32; time * n_embd];
        let mut v = vec![0.0f32; time * n_embd];
        for row in 0..time {
            q[row * n_embd..(row + 1) * n_embd]
                .copy_from_slice(&projected[row * 3 * n_embd..row * 3 * n_embd + n_embd]);
            k[row * n_embd..(row + 1) * n_embd].copy_from_slice(
                &projected[row * 3 * n_embd + n_embd..row * 3 * n_embd + 2 * n_embd],
            );
            v[row * n_embd..(row + 1) * n_embd]
                .copy_from_slice(&projected[row * 3 * n_embd + 2 * n_embd..(row + 1) * 3 * n_embd]);
        }
        let qh = ops::split_heads(&q, time, self.n_head, self.head_dim);
        let kh = ops::split_heads(&k, time, self.n_head, self.head_dim);
        let vh = ops::split_heads(&v, time, self.n_head, self.head_dim);
        let offset = cache.as_ref().map_or(0, |c| c.offset);
        let (keys, values) = match cache {
            Some(cache) => {
                cache.update_and_fetch(&kh, &vh, self.n_head, time, self.head_dim);
                (cache.keys.clone(), cache.values.clone())
            }
            None => (kh.clone(), vh.clone()),
        };
        let total = offset + time;
        let mut out = vec![0.0f32; self.n_head * time * self.head_dim];
        let mut scores = vec![0.0f32; total];
        for head in 0..self.n_head {
            let q_head = &qh[head * time * self.head_dim..(head + 1) * time * self.head_dim];
            let k_head = &keys[head * total * self.head_dim..(head + 1) * total * self.head_dim];
            let v_head = &values[head * total * self.head_dim..(head + 1) * total * self.head_dim];
            for row in 0..time {
                let bound = offset + row;
                // Both stacks attend every position. The causal stacks
                // pass the learned additive 0/1 tensor as a float mask
                // (upstream quirk: it biases, it does not mask), and the
                // fine stack's NonCausalSelfAttention passes no mask at
                // all; neither path restricts the key range.
                let high = total - 1;
                let q_row = &q_head[row * self.head_dim..(row + 1) * self.head_dim];
                let mut max = f32::NEG_INFINITY;
                for j in 0..=high {
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        dot += q_row[d] * k_head[j * self.head_dim + d];
                    }
                    let mut score = dot * (self.head_dim as f32).powf(-0.5);
                    if let Some(mask) = &self.mask {
                        score += mask[bound * self.mask_side + j];
                    }
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
                let out_row = &mut out
                    [(head * time + row) * self.head_dim..(head * time + row + 1) * self.head_dim];
                for j in 0..=high {
                    let weight = scores[j] * inv;
                    for d in 0..self.head_dim {
                        out_row[d] += weight * v_head[j * self.head_dim + d];
                    }
                }
            }
        }
        // Back to [time, heads * dim].
        let joined = ops::merge_heads(&out, time, self.n_head, self.head_dim);
        let attended = self.out_proj.forward(&joined, time);
        for (row, chunk) in attended.chunks(n_embd).enumerate() {
            for (col, value) in chunk.iter().enumerate() {
                x[row * n_embd + col] += value;
            }
        }
        let mut fed = x.to_vec();
        self.layernorm_2.apply(&mut fed, time);
        let hidden = self.mlp_in.forward(&fed, time);
        let mut hidden = hidden;
        ops::gelu_erf(&mut hidden);
        let fed = self.mlp_out.forward(&hidden, time);
        for (row, chunk) in fed.chunks(n_embd).enumerate() {
            for (col, value) in chunk.iter().enumerate() {
                x[row * n_embd + col] += value;
            }
        }
    }
}

/// The causal GPT (`GPT`): token + learned position embeddings over the
/// merged-context or cached path, causal blocks, and a last-position
/// head.
pub struct Gpt {
    input_embeds: Vec<f32>,
    position_embeds: Vec<f32>,
    layers: Vec<Block>,
    layernorm_final: LayerNorm,
    lm_head: Linear,
    config: GptConfig,
}

impl Gpt {
    fn load(weights: &crate::lm::Weights, prefix: &str, config: GptConfig) -> Result<Self> {
        let n_embd = config.n_embd;
        let mut layers = Vec::with_capacity(config.n_layer);
        for index in 0..config.n_layer {
            let layer = format!("{prefix}layers.{index}.");
            let bias = if config.bias {
                crate::lm::Bias::Required
            } else {
                crate::lm::Bias::None
            };
            layers.push(Block {
                layernorm_1: LayerNorm::new(
                    weights.tensor_absolute(&format!("{layer}layernorm_1.weight"), &[n_embd])?,
                    None,
                    1e-5,
                ),
                att_proj: weights.linear_absolute(
                    &format!("{layer}attn.att_proj"),
                    n_embd,
                    3 * n_embd,
                    bias,
                )?,
                out_proj: weights.linear_absolute(
                    &format!("{layer}attn.out_proj"),
                    n_embd,
                    n_embd,
                    bias,
                )?,
                layernorm_2: LayerNorm::new(
                    weights.tensor_absolute(&format!("{layer}layernorm_2.weight"), &[n_embd])?,
                    None,
                    1e-5,
                ),
                mlp_in: weights.linear_absolute(
                    &format!("{layer}mlp.in_proj"),
                    n_embd,
                    4 * n_embd,
                    bias,
                )?,
                mlp_out: weights.linear_absolute(
                    &format!("{layer}mlp.out_proj"),
                    4 * n_embd,
                    n_embd,
                    bias,
                )?,
                n_head: config.n_head,
                head_dim: n_embd / config.n_head,
                mask: Some(weights.tensor_absolute(
                    &format!("{layer}attn.bias"),
                    &[1, 1, config.block_size, config.block_size],
                )?),
                mask_side: config.block_size,
            });
        }
        Ok(Self {
            input_embeds: weights.tensor_absolute(
                &format!("{prefix}input_embeds_layer.weight"),
                &[config.input_vocab_size, n_embd],
            )?,
            position_embeds: weights.tensor_absolute(
                &format!("{prefix}position_embeds_layer.weight"),
                &[config.block_size, n_embd],
            )?,
            layers,
            layernorm_final: LayerNorm::new(
                weights.tensor_absolute(&format!("{prefix}layernorm_final.weight"), &[n_embd])?,
                None,
                1e-5,
            ),
            lm_head: weights.linear_absolute(
                &format!("{prefix}lm_head"),
                n_embd,
                config.output_vocab_size,
                crate::lm::Bias::None,
            )?,
            config,
        })
    }

    fn embed_row(&self, token: i32) -> Result<Vec<f32>> {
        let index = usize::try_from(token).map_err(|_| SpeechError::Input {
            why: format!("negative token id {token}"),
        })?;
        let n_embd = self.config.n_embd;
        if (index + 1) * n_embd > self.input_embeds.len() {
            return Err(SpeechError::Input {
                why: format!("token id {token} outside the input vocabulary"),
            });
        }
        Ok(self.input_embeds[index * n_embd..(index + 1) * n_embd].to_vec())
    }

    /// The reference `GPT.__call__` for one row block, returning the
    /// last-position logits. `merge_context` sums the first two 256-row
    /// blocks on the cacheless prefill (the semantic prompt layout).
    fn forward(
        &self,
        tokens: &[i32],
        merge_context: bool,
        caches: &mut [KvCache],
    ) -> Result<Vec<f32>> {
        let n_embd = self.config.n_embd;
        let offset = caches.first().map_or(0, |c| c.offset);
        let mut x = if merge_context {
            if tokens.len() < 256 + 256 + 1 {
                return Err(SpeechError::Input {
                    why: "merge_context needs at least 513 tokens".to_string(),
                });
            }
            let mut x = vec![0.0f32; (tokens.len() - 256) * n_embd];
            for position in 0..256 {
                let a = self.embed_row(tokens[position])?;
                let b = self.embed_row(tokens[256 + position])?;
                for col in 0..n_embd {
                    x[position * n_embd + col] = a[col] + b[col];
                }
            }
            for (position, token) in tokens[512..].iter().enumerate() {
                let row = self.embed_row(*token)?;
                x[(256 + position) * n_embd..(256 + position + 1) * n_embd].copy_from_slice(&row);
            }
            x
        } else {
            let mut x = Vec::with_capacity(tokens.len() * n_embd);
            for token in tokens {
                x.extend_from_slice(&self.embed_row(*token)?);
            }
            x
        };
        // Learned position embeddings from the cache offset; the merged
        // prefill produces tokens.len() - 256 rows.
        let rows = if merge_context {
            tokens.len() - 256
        } else {
            tokens.len()
        };
        for row in 0..rows {
            let position = offset + row;
            if (position + 1) * n_embd > self.position_embeds.len() {
                return Err(SpeechError::Input {
                    why: format!("position {position} outside the position table"),
                });
            }
            let row_slice = &self.position_embeds[position * n_embd..(position + 1) * n_embd];
            for (col, value) in row_slice.iter().enumerate() {
                x[row * n_embd + col] += value;
            }
        }
        for (index, layer) in self.layers.iter().enumerate() {
            let mut row = x.clone();
            layer.forward(&mut row, rows, Some(&mut caches[index]));
            x.copy_from_slice(&row);
        }
        self.layernorm_final.apply(&mut x, rows);
        let last = &x[(rows - 1) * n_embd..rows * n_embd];
        Ok(self.lm_head.forward(last, 1))
    }
}

/// The non-causal fine GPT (`FineGPT`): per-codebook embeddings tied to
/// per-codebook heads, full-attention blocks with biased layernorms.
pub struct FineGpt {
    input_embeds: Vec<Vec<f32>>,
    position_embeds: Vec<f32>,
    layers: Vec<Block>,
    layernorm_final: LayerNorm,
    lm_heads: Vec<Linear>,
    config: FineConfig,
}

impl FineGpt {
    fn load(weights: &crate::lm::Weights, prefix: &str, config: FineConfig) -> Result<Self> {
        let n_embd = config.n_embd;
        let mut input_embeds = Vec::with_capacity(config.n_codes_total);
        for index in 0..config.n_codes_total {
            input_embeds.push(weights.tensor_absolute(
                &format!("{prefix}input_embeds_layers.{index}.weight"),
                &[config.input_vocab_size, n_embd],
            )?);
        }
        // The heads cover the codebooks the model predicts. A given count
        // above the total would underflow this subtraction.
        let n_heads = config
            .n_codes_total
            .checked_sub(config.n_codes_given)
            .ok_or_else(|| SpeechError::BadConfig {
                field: "n_codes_given".to_string(),
                why: format!(
                    "n_codes_given {} exceeds n_codes_total {}",
                    config.n_codes_given, config.n_codes_total
                ),
            })?;
        let mut lm_heads = Vec::with_capacity(n_heads);
        for index in 0..n_heads {
            lm_heads.push(weights.linear_absolute(
                &format!("{prefix}lm_heads.{index}"),
                n_embd,
                config.output_vocab_size,
                crate::lm::Bias::None,
            )?);
        }
        let mut layers = Vec::with_capacity(config.n_layer);
        for index in 0..config.n_layer {
            let layer = format!("{prefix}layers.{index}.");
            // FineBlock normalizes with mlx nn.LayerNorm defaults, which
            // include the shift: weight and bias, unlike the causal
            // stacks' weight-only LayerNorm.
            layers.push(Block {
                layernorm_1: LayerNorm::new(
                    weights.tensor_absolute(&format!("{layer}layernorm_1.weight"), &[n_embd])?,
                    Some(weights.tensor_absolute(&format!("{layer}layernorm_1.bias"), &[n_embd])?),
                    1e-5,
                ),
                att_proj: weights.linear_absolute(
                    &format!("{layer}attn.att_proj"),
                    n_embd,
                    3 * n_embd,
                    crate::lm::Bias::None,
                )?,
                out_proj: weights.linear_absolute(
                    &format!("{layer}attn.out_proj"),
                    n_embd,
                    n_embd,
                    crate::lm::Bias::None,
                )?,
                layernorm_2: LayerNorm::new(
                    weights.tensor_absolute(&format!("{layer}layernorm_2.weight"), &[n_embd])?,
                    Some(weights.tensor_absolute(&format!("{layer}layernorm_2.bias"), &[n_embd])?),
                    1e-5,
                ),
                mlp_in: weights.linear_absolute(
                    &format!("{layer}mlp.in_proj"),
                    n_embd,
                    4 * n_embd,
                    crate::lm::Bias::None,
                )?,
                mlp_out: weights.linear_absolute(
                    &format!("{layer}mlp.out_proj"),
                    4 * n_embd,
                    n_embd,
                    crate::lm::Bias::None,
                )?,
                n_head: config.n_head,
                head_dim: n_embd / config.n_head,
                mask: None,
                mask_side: 0,
            });
        }
        Ok(Self {
            input_embeds,
            position_embeds: weights.tensor_absolute(
                &format!("{prefix}position_embeds_layer.weight"),
                &[config.block_size, n_embd],
            )?,
            layers,
            layernorm_final: LayerNorm::new(
                weights.tensor_absolute(&format!("{prefix}layernorm_final.weight"), &[n_embd])?,
                Some(weights.tensor_absolute(&format!("{prefix}layernorm_final.bias"), &[n_embd])?),
                1e-5,
            ),
            lm_heads,
            config,
        })
    }

    /// The reference `FineGPT.__call__`: sums the codebook embeddings
    /// through `pred_idx`, runs the non-causal blocks, and projects
    /// through codebook `pred_idx`'s head. Returns `[time, vocab]`.
    fn forward(&self, pred_idx: usize, codes: &[Vec<i32>]) -> Result<Vec<f32>> {
        let config = &self.config;
        if pred_idx == 0 {
            return Err(SpeechError::Input {
                why: "cannot predict 0th codebook".to_string(),
            });
        }
        // The buffer is row-major [time][codebooks]; the reference
        // asserts the time length against the block size and the
        // per-row codebook count against n_codes_total.
        let time = codes.len();
        if time > config.block_size {
            return Err(SpeechError::Input {
                why: format!(
                    "sequence of length {time} exceeds the block size {}",
                    config.block_size
                ),
            });
        }
        if codes.iter().any(|row| row.len() != config.n_codes_total) {
            return Err(SpeechError::Input {
                why: format!("expected {} codebooks per row", config.n_codes_total),
            });
        }
        let n_embd = config.n_embd;
        let mut x = vec![0.0f32; time * n_embd];
        for (book, embeds) in self.input_embeds.iter().take(pred_idx + 1).enumerate() {
            for (row, codes_row) in codes.iter().enumerate() {
                let token = codes_row[book];
                let index = usize::try_from(token).map_err(|_| SpeechError::Input {
                    why: format!("negative code {token}"),
                })?;
                if (index + 1) * n_embd > embeds.len() {
                    return Err(SpeechError::Input {
                        why: format!("code {token} outside codebook {book} embeddings"),
                    });
                }
                let row_start = row * n_embd;
                for col in 0..n_embd {
                    x[row_start + col] += embeds[index * n_embd + col];
                }
            }
        }
        for row in 0..time {
            let row_start = row * n_embd;
            for col in 0..n_embd {
                x[row_start + col] += self.position_embeds[row * n_embd + col];
            }
        }
        for layer in &self.layers {
            let mut row = x.clone();
            layer.forward(&mut row, time, None);
            x.copy_from_slice(&row);
        }
        self.layernorm_final.apply(&mut x, time);
        Ok(self.lm_heads[pred_idx - config.n_codes_given].forward(&x, time))
    }
}

/// Sampling options with the reference defaults.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub semantic_temperature: f32,
    pub coarse_temperature: f32,
    /// `None` takes the reference argmax path in `generate_fine`.
    pub fine_temperature: Option<f32>,
    pub max_coarse_history: usize,
    pub sliding_window_len: usize,
    /// The reference `generate_coarse` `use_kv_caching` parameter. The
    /// reference function default is `False`, and the pipeline path
    /// never turns it on (see [`Model::generate`]): every coarse step
    /// re-forwards the full growing window.
    pub coarse_use_kv_caching: bool,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            semantic_temperature: 0.7,
            coarse_temperature: 0.7,
            fine_temperature: None,
            max_coarse_history: 60,
            sliding_window_len: 60,
            coarse_use_kv_caching: false,
        }
    }
}

/// Python `round()`: half-to-even in f64. The reference uses it for the
/// window count and the per-window semantic index; an exact `.5` can
/// occur for the window count (`n_steps` is even) and must not round
/// away from zero.
fn python_round(x: f64) -> usize {
    let floor = x.floor();
    let frac = x - floor;
    let round_up = if frac > 0.5 {
        true
    } else if frac < 0.5 {
        false
    } else {
        // An exact .5 rounds half to even, like Python's round().
        (floor as i64) % 2 != 0
    };
    let rounded = if round_up { floor + 1.0 } else { floor };
    rounded as usize
}

/// One generated run: the semantic tokens, the two coarse codebooks
/// (offsets removed), the eight fine codebooks, and the decoded
/// waveform.
#[derive(Debug, Clone)]
pub struct Generated {
    pub encoded_text: Vec<i32>,
    pub semantic_tokens: Vec<i32>,
    pub coarse_codes: Vec<Vec<i32>>,
    pub fine_codes: Vec<Vec<i32>>,
    pub samples: Vec<f32>,
}

/// The loaded family model: the three GPT stages.
pub struct Model {
    pub config: ModelConfig,
    semantic: Gpt,
    coarse: Gpt,
    fine: FineGpt,
}

impl Model {
    /// Opens a checkpoint directory (`config.json` + safetensors).
    pub fn open(dir: &std::path::Path) -> Result<Self> {
        let config_value = quant::read_json(&dir.join("config.json"))?;
        let weights = crate::lm::Weights::open(dir)?;
        Self::load(&weights, &config_value)
    }

    /// Loads from opened weights and the raw config dict.
    pub fn load(weights: &crate::lm::Weights, config: &Value) -> Result<Self> {
        let config = ModelConfig::from_json(config)?;
        Ok(Self {
            semantic: Gpt::load(weights, "semantic.", config.semantic_config.clone())?,
            coarse: Gpt::load(
                weights,
                "coarse_acoustics.",
                config.coarse_acoustics_config.clone(),
            )?,
            fine: FineGpt::load(
                weights,
                "fine_acoustics.",
                config.fine_acoustics_config.clone(),
            )?,
            config,
        })
    }

    /// The reference `generate_text_semantic`: the merged 256+256+1
    /// prompt, then one cached token per step over the first
    /// [`SEMANTIC_VOCAB_SIZE`] logits plus the early-stop pad logit.
    /// `draw` receives the temperature-scaled slice and returns an index
    /// into it; index [`SEMANTIC_VOCAB_SIZE`] stops the loop.
    pub fn generate_text_semantic(
        &mut self,
        text_ids: &[i32],
        semantic_history: Option<&[i32]>,
        temperature: f32,
        draw: &mut dyn FnMut(&[f32]) -> usize,
    ) -> Result<(Vec<i32>, Vec<i32>)> {
        // Caller ids are unchecked, so an id near i32::MAX must not wrap.
        let mut encoded_text: Vec<i32> = text_ids
            .iter()
            .map(|&id| {
                id.checked_add(TEXT_ENCODING_OFFSET)
                    .ok_or_else(|| SpeechError::Input {
                        why: format!("text id {id} overflows the text encoding offset"),
                    })
            })
            .collect::<Result<Vec<i32>>>()?;
        encoded_text.truncate(256);
        while encoded_text.len() < 256 {
            encoded_text.push(TEXT_PAD_TOKEN);
        }
        let mut history: Vec<i32> = match semantic_history {
            Some(rows) => {
                let mut rows = rows.to_vec();
                rows.truncate(256);
                rows
            }
            None => Vec::new(),
        };
        while history.len() < 256 {
            history.push(SEMANTIC_PAD_TOKEN);
        }
        let mut x: Vec<i32> = Vec::with_capacity(256 + 256 + 1);
        x.extend_from_slice(&encoded_text);
        x.extend_from_slice(&history);
        x.push(SEMANTIC_INFER_TOKEN);

        let mut caches = vec![KvCache::new(); self.config.semantic_config.n_layer];
        let logits = self.semantic.forward(&x, true, &mut caches)?;
        let mut generated = Vec::new();
        let total_steps = 768;
        for step in 0..total_steps {
            let row = if step == 0 {
                &logits
            } else {
                &self
                    .semantic
                    .forward(&x[x.len() - 1..], false, &mut caches)?
            };
            if row.len() <= SEMANTIC_VOCAB_SIZE {
                return Err(SpeechError::Input {
                    why: format!(
                        "semantic head width {} does not cover the pad logit",
                        row.len()
                    ),
                });
            }
            let mut relevant: Vec<f32> = row[..SEMANTIC_VOCAB_SIZE].to_vec();
            relevant.push(row[SEMANTIC_PAD_TOKEN as usize]);
            for value in &mut relevant {
                *value /= temperature.max(1e-5);
            }
            let next = draw(&relevant);
            if next == SEMANTIC_VOCAB_SIZE {
                break;
            }
            let next = next as i32;
            generated.push(next);
            x.push(next);
        }
        Ok((generated, encoded_text))
    }

    /// The reference `generate_coarse`: sliding windows over the offset
    /// coarse logits. The index math (window count, per-window semantic
    /// index, step count) runs in f64 with Python's round because the
    /// reference computes it in float64. With
    /// `coarse_use_kv_caching` (the reference's `use_kv_caching`)
    /// the first step of a window prefills and the rest decode one
    /// cached token; the reference function default, and the pipeline
    /// path, is `false`: every step re-forwards the full growing window.
    pub fn generate_coarse(
        &mut self,
        semantic_tokens: &[i32],
        semantic_history: Option<&[i32]>,
        coarse_history: Option<&[Vec<i32>]>,
        params: &GenerateParams,
        draw: &mut dyn FnMut(&[f32]) -> usize,
    ) -> Result<Vec<Vec<i32>>> {
        // The reference constants are Python floats; mirror them at f64.
        let semantic_to_coarse_ratio =
            COARSE_RATE_HZ as f64 / SEMANTIC_RATE_HZ as f64 * N_COARSE_CODEBOOKS as f64;
        let max_semantic_history =
            ((params.max_coarse_history as f64) / semantic_to_coarse_ratio).floor() as usize;
        // Voice histories are caller-owned slices; this port runs the
        // no-history path (the reference's empty arrays).
        let _ = (semantic_history, coarse_history);
        let x_semantic: Vec<i32> = semantic_tokens.to_vec();
        let n_steps = ((x_semantic.len() as f64 * semantic_to_coarse_ratio
            / N_COARSE_CODEBOOKS as f64)
            .floor() as usize)
            * N_COARSE_CODEBOOKS;
        let mut x_coarse_in: Vec<i32> = Vec::new();
        // A zero window makes the division infinite, so n_window_steps
        // saturates to usize::MAX and the window loop never makes progress.
        if params.sliding_window_len == 0 {
            return Err(SpeechError::Input {
                why: "sliding_window_len must be > 0".to_string(),
            });
        }
        let n_window_steps = python_round(n_steps as f64 / params.sliding_window_len as f64);
        let mut n_step = 0usize;
        let mut caches = vec![KvCache::new(); self.config.coarse_acoustics_config.n_layer];
        // The reference coarse slice scales by the reciprocal
        // (`relevant_logits * (1 / temperature)`), unlike the semantic
        // and fine stages, which divide.
        let coarse_scale = 1.0f32 / params.coarse_temperature.max(1e-5);
        for _window in 0..n_window_steps {
            let semantic_idx = python_round(n_step as f64 / semantic_to_coarse_ratio);
            let start = semantic_idx.saturating_sub(max_semantic_history);
            let mut x_in: Vec<i32> = x_semantic[start..].to_vec();
            x_in.truncate(256);
            while x_in.len() < 256 {
                x_in.push(COARSE_SEMANTIC_PAD_TOKEN);
            }
            x_in.push(COARSE_INFER_TOKEN);
            let history_start = x_coarse_in.len().saturating_sub(params.max_coarse_history);
            x_in.extend_from_slice(&x_coarse_in[history_start..]);
            let mut prefilled = false;
            for _ in 0..params.sliding_window_len {
                if n_step >= n_steps {
                    // The reference `continue`s; staying in the window
                    // draws nothing more, so a break is observable-only
                    // identical.
                    break;
                }
                let is_major_step = n_step % N_COARSE_CODEBOOKS == 0;
                let row = if params.coarse_use_kv_caching && prefilled {
                    self.coarse
                        .forward(&x_in[x_in.len() - 1..], false, &mut caches)?
                } else {
                    // Cached: reset per window before the prefill.
                    // Uncached: reset every step, mirroring the
                    // reference's past_kv=None with use_cache=False.
                    for cache in &mut caches {
                        cache.reset();
                    }
                    let logits = self.coarse.forward(&x_in, false, &mut caches)?;
                    prefilled = true;
                    logits
                };
                let logit_start_idx =
                    SEMANTIC_VOCAB_SIZE + (1 - is_major_step as usize) * CODEBOOK_SIZE;
                let logit_end_idx = (SEMANTIC_VOCAB_SIZE
                    + (2 - is_major_step as usize) * CODEBOOK_SIZE)
                    .min(row.len());
                if logit_start_idx >= logit_end_idx {
                    return Err(SpeechError::Input {
                        why: format!(
                            "coarse head width {} does not cover the codebook slice",
                            row.len()
                        ),
                    });
                }
                let mut relevant: Vec<f32> = row[logit_start_idx..logit_end_idx].to_vec();
                for value in &mut relevant {
                    *value *= coarse_scale;
                }
                let item_next = draw(&relevant) + logit_start_idx;
                let item_next = item_next as i32;
                x_coarse_in.push(item_next);
                x_in.push(item_next);
                n_step += 1;
            }
        }
        // Undo the offsets into the two codebooks.
        let gen_coarse: Vec<i32> = x_coarse_in[x_coarse_in
            .len()
            .saturating_sub(n_steps)
            .min(x_coarse_in.len())..]
            .to_vec();
        let frames = gen_coarse.len() / N_COARSE_CODEBOOKS;
        let mut codes = vec![vec![0i32; frames]; N_COARSE_CODEBOOKS];
        for (step, chunk) in gen_coarse.chunks_exact(N_COARSE_CODEBOOKS).enumerate() {
            for book in 0..N_COARSE_CODEBOOKS {
                codes[book][step] =
                    chunk[book] - SEMANTIC_VOCAB_SIZE as i32 - book as i32 * CODEBOOK_SIZE as i32;
            }
        }
        Ok(codes)
    }

    /// The reference `generate_fine`: iteratively fills codebooks
    /// `n_coarse..8` over 1024-row non-causal windows. `temperature`
    /// `None` takes the argmax path.
    pub fn generate_fine(
        &mut self,
        coarse_codes: &[Vec<i32>],
        temperature: Option<f32>,
        draw: &mut dyn FnMut(&[f32]) -> usize,
    ) -> Result<Vec<Vec<i32>>> {
        let block_size = self.config.fine_acoustics_config.block_size;
        let n_coarse = coarse_codes.len();
        if n_coarse == 0 {
            return Err(SpeechError::Input {
                why: "generate_fine needs at least one coarse codebook".to_string(),
            });
        }
        let frames = coarse_codes[0].len();
        let mut in_arr: Vec<Vec<i32>> = Vec::with_capacity(N_FINE_CODEBOOKS);
        for codes_row in coarse_codes {
            in_arr.push(codes_row.clone());
        }
        while in_arr.len() < N_FINE_CODEBOOKS {
            in_arr.push(vec![CODEBOOK_SIZE as i32; frames]);
        }
        let mut n_remove_from_end = 0usize;
        if frames < block_size {
            n_remove_from_end = block_size - frames;
            for book in &mut in_arr {
                book.extend(std::iter::repeat_n(CODEBOOK_SIZE as i32, n_remove_from_end));
            }
        }
        let total = in_arr[0].len();
        let n_loops = total.saturating_sub(block_size).div_ceil(512) + 1;
        // Row-major working copy [time][codebooks].
        let mut rows: Vec<Vec<i32>> = (0..total)
            .map(|frame| {
                (0..N_FINE_CODEBOOKS)
                    .map(|book| in_arr[book][frame])
                    .collect()
            })
            .collect();
        let n_history = 0usize;
        for window in 0..n_loops {
            let start_idx = (window * 512).min(total.saturating_sub(block_size));
            let start_fill_idx = (n_history + window * 512).min(total.saturating_sub(512));
            let rel_start_fill_idx = start_fill_idx - start_idx;
            let mut in_buffer: Vec<Vec<i32>> = rows[start_idx..start_idx + block_size].to_vec();
            for book in n_coarse..N_FINE_CODEBOOKS {
                let logits = self.fine.forward(book, &in_buffer)?;
                let vocab = logits.len() / block_size;
                let width = CODEBOOK_SIZE.min(vocab);
                if temperature.is_none() {
                    for row in rel_start_fill_idx..block_size {
                        let argmax = sampling::argmax(&logits[row * vocab..row * vocab + width]);
                        in_buffer[row][book] = argmax as i32;
                    }
                } else {
                    let temp = temperature.unwrap_or(1.0);
                    for row in rel_start_fill_idx..block_size {
                        let mut relevant = logits[row * vocab..row * vocab + width].to_vec();
                        for value in &mut relevant {
                            *value /= temp;
                        }
                        in_buffer[row][book] = draw(&relevant) as i32;
                    }
                }
            }
            for (offset, row) in in_buffer[rel_start_fill_idx..].iter().enumerate() {
                rows[start_fill_idx + offset] = row.clone();
            }
        }
        // Unpad to the original frame count.
        let keep = total - n_remove_from_end;
        let mut codes = vec![vec![0i32; keep]; N_FINE_CODEBOOKS];
        for book in 0..N_FINE_CODEBOOKS {
            for frame in 0..keep {
                codes[book][frame] = rows[frame][book];
            }
        }
        Ok(codes)
    }

    /// The full reference `Pipeline.__call__`: semantic, coarse, fine,
    /// then the EnCodec decode through all eight codebooks. `draw`
    /// receives each temperature-scaled logit slice; the fixture tests
    /// replay an inverse-CDF stand-in, live runs draw categorically.
    ///
    /// The reference `__call__` forwards `use_kv_caching` positionally
    /// into `generate_coarse`, where it binds to the
    /// `max_coarse_history` parameter (pipeline.py at the pinned
    /// commit: `self.generate_coarse(semantic_tokens, voice,
    /// temperature, use_kv_caching, **kwargs)`). The pipeline therefore
    /// runs the coarse stage with `max_coarse_history = True` (1), the
    /// default sliding window of 60, and no kv cache; this method
    /// reproduces that wiring instead of the caller's
    /// `max_coarse_history` / `sliding_window_len`, which stay
    /// meaningful for direct `generate_coarse` callers.
    pub fn generate(
        &mut self,
        encodec: &Encodec,
        text_ids: &[i32],
        params: &GenerateParams,
        draw: &mut dyn FnMut(&[f32]) -> usize,
    ) -> Result<Generated> {
        let (semantic_tokens, encoded_text) =
            self.generate_text_semantic(text_ids, None, params.semantic_temperature, draw)?;
        let coarse_params = GenerateParams {
            semantic_temperature: params.semantic_temperature,
            coarse_temperature: params.coarse_temperature,
            fine_temperature: params.fine_temperature,
            max_coarse_history: 1,
            sliding_window_len: 60,
            coarse_use_kv_caching: false,
        };
        let coarse_codes =
            self.generate_coarse(&semantic_tokens, None, None, &coarse_params, draw)?;
        let fine_codes = self.generate_fine(&coarse_codes, params.fine_temperature, draw)?;
        let samples = encodec.decode_codes(&fine_codes, None)?;
        Ok(Generated {
            encoded_text,
            semantic_tokens,
            coarse_codes,
            fine_codes,
            samples,
        })
    }
}

#[cfg(test)]
mod tests;

#[allow(dead_code)]
#[cfg(test)]
fn f32_rows(value: &serde_json::Value) -> Vec<f32> {
    value
        .as_array()
        .expect("rows")
        .iter()
        .map(|v| v.as_f64().expect("v") as f32)
        .collect()
}

#[cfg(test)]
fn stage_diff(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

#[cfg(test)]
impl Model {
    pub(crate) fn semantic_caches(&mut self) -> Vec<KvCache> {
        vec![KvCache::new(); self.config.semantic_config.n_layer]
    }

    pub(crate) fn semantic_prefill_logits(
        &mut self,
        tokens: &[i32],
        caches: &mut [KvCache],
    ) -> Result<Vec<f32>> {
        self.semantic.forward(tokens, true, caches)
    }

    /// Fixture-only per-stage diffs of the semantic prefill.
    #[cfg(test)]
    pub(crate) fn semantic_stage_diffs(&mut self, tokens: &[i32], stages: &serde_json::Value) {
        let n_embd = self.config.semantic_config.n_embd;
        let mut x = vec![0.0f32; (tokens.len() - 256) * n_embd];
        for position in 0..256 {
            let a = self.semantic.embed_row(tokens[position]).expect("embed");
            let b = self
                .semantic
                .embed_row(tokens[256 + position])
                .expect("embed");
            for col in 0..n_embd {
                x[position * n_embd + col] = a[col] + b[col];
            }
        }
        for (position, token) in tokens[512..].iter().enumerate() {
            let row = self.semantic.embed_row(*token).expect("embed");
            x[(256 + position) * n_embd..(256 + position + 1) * n_embd].copy_from_slice(&row);
        }
        let got = f32_rows(&stages["embeds"]);
        assert!(stage_diff(&x, &got) <= 1e-4, "embeds diff");
        let rows = tokens.len() - 256;
        for row in 0..rows {
            for col in 0..n_embd {
                x[row * n_embd + col] += self.semantic.position_embeds[row * n_embd + col];
            }
        }
        let got = f32_rows(&stages["after_embeds"]);
        assert!(stage_diff(&x, &got) <= 1e-4, "after_embeds diff");
        for index in 0..self.semantic.layers.len() {
            let block = &self.semantic.layers[index];
            let mut normed = x.clone();
            block.layernorm_1.apply(&mut normed, rows);
            let qkv = block.att_proj.forward(&normed, rows);
            if let Some(qkv_stage) = stages.get(format!("block{index}.qkv").as_str()) {
                let got = f32_rows(qkv_stage);
                assert!(stage_diff(&qkv, &got) <= 1e-4);
            }
            // The real block advances the residual stream.
            let mut row = x.clone();
            block.forward(&mut row, rows, None);
            x.copy_from_slice(&row);
            let got = f32_rows(&stages[&format!("block{index}.out")]);
            assert!(stage_diff(&x, &got) <= 1e-4, "block{index} diff");
        }
        self.semantic.layernorm_final.apply(&mut x, rows);
        let got = f32_rows(&stages["final_norm"]);
        assert!(stage_diff(&x, &got) <= 1e-4, "final_norm diff");
    }
}
