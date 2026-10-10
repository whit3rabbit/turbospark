//! The gpt2 dense backbone.
//!
//! Port of `mlx_audio/lm/models/gpt2.py` (vendored from mlx-lm) at the
//! pinned mlx-audio commit: learned token and position embeddings, fused
//! biased `c_attn` QKV, biased gelu (tanh approximation) MLP, pre-LayerNorm
//! blocks, and a tied head over the token embedding table. There is no
//! rope: attention is masked GQA over the KV cache with implicit causal
//! bounds.
//!
//! Deviations documented against the reference:
//! - The reference `ModelArgs.num_key_value_heads` field (defaulting to
//!   `n_head`) is never read by the model; every projection keeps `n_head`
//!   heads. This port parses and defaults the field the same way but the
//!   attention always runs `n_kv_heads = n_head`.
//! - HF `conv1d` checkpoints store `c_attn`/`c_fc`/`c_proj` weights
//!   transposed; the reference `Model.sanitize` transposes them into the
//!   MLX `[out, in]` layout this module loads. Family loaders own that
//!   transpose (the fixture checkpoints are already MLX layout).
//! - The indextts family reuses `TransformerBlock` with `wte` and `wpe`
//!   patched to `nn.Identity` and feeds precomputed embeddings. This port
//!   mirrors that with [`Input::Embeddings`] plus
//!   [`Gpt2Model::without_position_embeddings`].

use serde_json::Value;

use crate::nn::{self, LayerNorm, Linear};
use crate::ops;
use crate::{Result, SpeechError};

use super::cache::LayerCaches;
use super::engine::{lookup_embeddings, split_fused_linear, Attention, Input};
use super::{Bias, Weights};

/// Parses a gpt2 `config.json` (or the config dict a TTS family builds).
/// Every field except `num_key_value_heads` is required, matching the
/// reference dataclass.
pub fn from_json(value: &Value) -> Result<Gpt2Config> {
    let n_head = nn::json::usize_field(value, "n_head")?;
    Ok(Gpt2Config {
        model_type: nn::json::text_field(value, "model_type")?,
        n_ctx: nn::json::usize_field(value, "n_ctx")?,
        n_embd: nn::json::usize_field(value, "n_embd")?,
        n_head,
        n_layer: nn::json::usize_field(value, "n_layer")?,
        n_positions: nn::json::usize_field(value, "n_positions")?,
        layer_norm_epsilon: nn::json::required(value, "layer_norm_epsilon")?
            .as_f64()
            .filter(|v| *v > 0.0)
            .ok_or_else(|| SpeechError::BadConfig {
                field: "layer_norm_epsilon".into(),
                why: "must be a positive number".into(),
            })? as f32,
        vocab_size: nn::json::usize_field(value, "vocab_size")?,
        // The reference __post_init__ defaults the field to n_head; the
        // model itself never reads it.
        num_key_value_heads: value
            .get("num_key_value_heads")
            .and_then(Value::as_u64)
            .map(|n| {
                usize::try_from(n).map_err(|_| SpeechError::BadConfig {
                    field: "num_key_value_heads".into(),
                    why: "must be a positive integer fitting usize".into(),
                })
            })
            .transpose()?
            .unwrap_or(n_head),
    })
}

/// Loads a gpt2 backbone from `weights` (tensor names carry the
/// reference-forced `model.` prefix).
pub fn load(weights: &Weights, value: &Value) -> Result<Gpt2Model> {
    Gpt2Model::load(weights, value)
}

/// The parsed gpt2 configuration (`ModelArgs`).
#[derive(Debug, Clone)]
pub struct Gpt2Config {
    pub model_type: String,
    pub n_ctx: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub n_positions: usize,
    pub layer_norm_epsilon: f32,
    pub vocab_size: usize,
    /// Defaults to `n_head`; unused by the forward pass (reference parity).
    #[allow(dead_code)]
    pub num_key_value_heads: usize,
}

struct Block {
    attn: Attention,
    mlp: Gpt2Mlp,
    ln_1: LayerNorm,
    ln_2: LayerNorm,
}

/// `c_proj(gelu_approx(c_fc(x)))` with biases everywhere.
struct Gpt2Mlp {
    c_fc: Linear,
    c_proj: Linear,
}

impl Gpt2Mlp {
    fn forward(&self, x: &[f32], len: usize) -> Vec<f32> {
        let mut hidden = self.c_fc.forward(x, len);
        ops::gelu_tanh(&mut hidden);
        self.c_proj.forward(&hidden, len)
    }
}

/// A loaded gpt2 backbone ready to run.
pub struct Gpt2Model {
    pub config: Gpt2Config,
    /// `wte`: the token embedding table, reused as the tied LM head.
    /// `None` mirrors a family that patches the module to `nn.Identity`
    /// and ships checkpoints without the table (indextts); the tied-head
    /// [`Gpt2Model::forward`] is then refused.
    wte: Option<Vec<f32>>,
    /// `wpe`: the learned position table; `None` mirrors the indextts
    /// `wpe = nn.Identity()` patch.
    wpe: Option<Vec<f32>>,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
}

impl Gpt2Model {
    /// Loads every weight the configuration names. The reference sanitize
    /// forces the `model.` prefix onto every gpt2 tensor, so the names are
    /// written explicitly here (the LM head is the `wte` table itself).
    pub fn load(weights: &Weights, value: &Value) -> Result<Self> {
        Self::from_config(weights, from_json(value)?)
    }

    /// Loads from an already parsed config.
    pub fn from_config(weights: &Weights, config: Gpt2Config) -> Result<Self> {
        let dim = config.n_embd;
        let mut model = Self::load_blocks(weights, config, "model.")?;
        model.wte = Some(weights.embedding("model.wte", model.config.vocab_size, dim)?);
        model.wpe = Some(weights.embedding("model.wpe", model.config.n_positions, dim)?);
        Ok(model)
    }

    /// [`Gpt2Model::from_config`] with the `wte`/`wpe` tables left
    /// unloaded: the indextts family patches both modules to
    /// `nn.Identity`, so its converted checkpoints may not carry the
    /// tables at all. Only [`Gpt2Model::forward_hidden`] runs on such a
    /// model; the tied-head [`Gpt2Model::forward`] is refused.
    pub fn from_config_without_tables(weights: &Weights, config: Gpt2Config) -> Result<Self> {
        Self::load_blocks(weights, config, "model.")
    }

    /// [`Gpt2Model::from_config_without_tables`] for checkpoints that
    /// keep the backbone tensors under a family root other than the
    /// reference-forced `model.` (the indextts wrapper nests them under
    /// `gpt.`): `root` replaces the `model.` segment in every name.
    pub fn from_config_without_tables_at_root(
        weights: &Weights,
        config: Gpt2Config,
        root: &str,
    ) -> Result<Self> {
        Self::load_blocks(weights, config, root)
    }

    fn load_blocks(weights: &Weights, config: Gpt2Config, root: &str) -> Result<Self> {
        if config.n_embd % config.n_head != 0 {
            return Err(SpeechError::BadConfig {
                field: "n_embd".into(),
                why: format!(
                    "{} not divisible by n_head {}",
                    config.n_embd, config.n_head
                ),
            });
        }
        let dim = config.n_embd;
        let mut blocks = Vec::with_capacity(config.n_layer);
        for index in 0..config.n_layer {
            let layer = format!("{root}h.{index}.");
            let fused =
                weights.linear(&format!("{layer}attn.c_attn"), dim, 3 * dim, Bias::Required)?;
            let (q_proj, k_proj, v_proj) = split_fused_linear(&fused, dim, dim);
            let attn = Attention {
                o_proj: weights.linear(&format!("{layer}attn.c_proj"), dim, dim, Bias::Required)?,
                q_proj,
                k_proj,
                v_proj,
                q_norm: None,
                k_norm: None,
                n_heads: config.n_head,
                n_kv_heads: config.n_head,
                head_dim: dim / config.n_head,
                scale: ((dim / config.n_head) as f64).powf(-0.5) as f32,
                rope: None,
                window: None,
            };
            let mlp = Gpt2Mlp {
                c_fc: weights.linear(&format!("{layer}mlp.c_fc"), dim, 4 * dim, Bias::Required)?,
                c_proj: weights.linear(
                    &format!("{layer}mlp.c_proj"),
                    4 * dim,
                    dim,
                    Bias::Required,
                )?,
            };
            let ln_1 = LayerNorm::new(
                weights.tensor(&format!("{layer}ln_1"), &[dim])?,
                Some(weights.raw_tensor(&format!("{layer}ln_1.bias"), &[dim])?),
                config.layer_norm_epsilon,
            );
            let ln_2 = LayerNorm::new(
                weights.tensor(&format!("{layer}ln_2"), &[dim])?,
                Some(weights.raw_tensor(&format!("{layer}ln_2.bias"), &[dim])?),
                config.layer_norm_epsilon,
            );
            blocks.push(Block {
                attn,
                mlp,
                ln_1,
                ln_2,
            });
        }
        let wte = None;
        let wpe = None;
        let ln_f = LayerNorm::new(
            weights.tensor(&format!("{root}ln_f"), &[dim])?,
            Some(weights.raw_tensor(&format!("{root}ln_f.bias"), &[dim])?),
            config.layer_norm_epsilon,
        );
        Ok(Self {
            config,
            wte,
            wpe,
            blocks,
            ln_f,
        })
    }

    /// Mirrors the indextts patch `gpt.wpe = nn.Identity()`: the forward
    /// pass stops adding learned positions (the reference replaces the
    /// module; here the table is dropped).
    pub fn without_position_embeddings(mut self) -> Self {
        self.wpe = None;
        self
    }

    /// Runs the backbone and returns logits `[positions, vocab_size]`.
    /// Refused when the model was loaded without the embedding tables
    /// ([`Gpt2Model::from_config_without_tables`]).
    pub fn forward(
        &mut self,
        input: Input<'_>,
        caches: Option<&mut LayerCaches>,
    ) -> Result<Vec<f32>> {
        let projected = self.forward_hidden(input, caches)?;
        let len = projected.len() / self.config.n_embd;
        // The tied head: `wte.as_linear` in the reference.
        let wte = self.wte.as_ref().ok_or_else(|| SpeechError::Unsupported {
            why: "tied-head forward on a gpt2 backbone loaded without the \
                  wte table (the indextts patch replaces the head input with \
                  precomputed embeddings and reads the family heads instead)"
                .into(),
        })?;
        Ok(ops::linear(
            &projected,
            wte,
            None,
            len,
            self.config.n_embd,
            self.config.vocab_size,
        ))
    }

    /// Runs the backbone up to `ln_f` and returns the hidden state
    /// `[positions, n_embd]`.
    pub fn forward_hidden(
        &mut self,
        input: Input<'_>,
        mut caches: Option<&mut LayerCaches>,
    ) -> Result<Vec<f32>> {
        let dim = self.config.n_embd;
        let len = input.len(dim)?;
        if len == 0 {
            return Err(SpeechError::Input {
                why: "empty forward input".into(),
            });
        }
        let mut h = match input {
            Input::Tokens(tokens) => {
                let wte = self.wte.as_ref().ok_or_else(|| SpeechError::Unsupported {
                    why: "token-id forward on a gpt2 backbone loaded without the \
                          wte table"
                        .into(),
                })?;
                lookup_embeddings(wte, self.config.vocab_size, dim, tokens)?
            }
            Input::Embeddings(values) => values.to_vec(),
        };
        let offset = caches
            .as_deref()
            .and_then(|c| c.first())
            .map(|c| c.offset)
            .unwrap_or(0);
        if let Some(caches) = caches.as_deref() {
            if caches.len() != self.blocks.len() {
                return Err(SpeechError::Input {
                    why: format!(
                        "{} layer caches provided for {} layers",
                        caches.len(),
                        self.blocks.len()
                    ),
                });
            }
        }
        if let Some(wpe) = &self.wpe {
            // position_ids = arange(L) + cache offset; every position must
            // exist in the learned table.
            for t in 0..len {
                let position = offset + t;
                if position >= self.config.n_positions {
                    return Err(SpeechError::Input {
                        why: format!(
                            "position {position} outside n_positions {}",
                            self.config.n_positions
                        ),
                    });
                }
                let source = position * dim;
                for (c, value) in h[t * dim..(t + 1) * dim].iter_mut().enumerate() {
                    *value += wpe[source + c];
                }
            }
        }
        for (index, block) in self.blocks.iter_mut().enumerate() {
            let mut attended = h.clone();
            block.ln_1.apply(&mut attended, len);
            let attn = block.attn.forward(
                &attended,
                len,
                offset,
                caches
                    .as_deref_mut()
                    .map(|c| &mut c[index] as &mut dyn super::cache::LayerKv),
            )?;
            for (row, out) in attn.chunks_exact(dim).enumerate() {
                let base = row * dim;
                for (col, value) in out.iter().enumerate() {
                    h[base + col] += value;
                }
            }
            let mut fed = h.clone();
            block.ln_2.apply(&mut fed, len);
            let mlp = block.mlp.forward(&fed, len);
            for (row, out) in mlp.chunks_exact(dim).enumerate() {
                let base = row * dim;
                for (col, value) in out.iter().enumerate() {
                    h[base + col] += value;
                }
            }
        }
        self.ln_f.apply(&mut h, len);
        Ok(h)
    }
}
