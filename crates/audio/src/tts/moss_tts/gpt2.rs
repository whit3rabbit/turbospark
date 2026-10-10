//! The family-local GPT-2 local transformer of
//! `moss_tts_nano/gpt2.py`, which the moss_tts v1.5 local-transformer
//! variant imports verbatim (`from ..moss_tts_nano.gpt2 import
//! GPT2Model`).
//!
//! Deliberately a re-declaration of the moss_tts_nano module's loader:
//! that module's `Gpt2` constructor is private to it and the families
//! stay separate per the inventory, so this copy carries the same math
//! (rope positions, fused biased QKV split into equal thirds, tanh GELU
//! MLPs, optional token embedding) on top of the shared attention, rope,
//! and KV cache. Verified by the same fixtures as the nano port.

use crate::lm::engine::{split_fused_linear, Attention};
use crate::nn::{LayerNorm, Linear};
use crate::ops;
use crate::tts::moss_tts_nano::Gpt2Config;
use crate::{Result, SpeechError};

/// Tanh GELU (`gpt2.gelu_new`), exact GELU, or SiLU, by config name.
fn apply_activation(name: &str, values: &mut [f32]) -> Result<()> {
    match name {
        "gelu_new" => ops::gelu_tanh(values),
        "gelu" => ops::gelu_erf(values),
        "silu" => {
            for v in values.iter_mut() {
                *v *= ops::sigmoid(*v);
            }
        }
        other => {
            return Err(SpeechError::Unsupported {
                why: format!("activation_function {other}"),
            })
        }
    }
    Ok(())
}

/// The family-local `gpt2.MLP`: `fc_out(activation(fc_in(x)))`, both
/// linears biased.
struct Mlp {
    fc_in: Linear,
    fc_out: Linear,
    activation: String,
}

impl Mlp {
    fn forward(&self, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        let mut hidden = self.fc_in.forward(x, rows);
        apply_activation(&self.activation, &mut hidden)?;
        Ok(self.fc_out.forward(&hidden, rows))
    }
}

struct Block {
    ln_1: LayerNorm,
    attn: Attention,
    ln_2: LayerNorm,
    mlp: Mlp,
}

/// The family-local `gpt2.GPT2Model`: optional token embedding (the
/// local transformer is built with `use_token_embedding=False`), rope
/// positions, LayerNorm pre-norm blocks, and the final LayerNorm.
pub(crate) struct LocalGpt2 {
    pub(crate) config: Gpt2Config,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
}

impl LocalGpt2 {
    /// Loads one stack under `name`. The v1.5 local transformer keeps no
    /// token embedding (the checkpoint sanitize drops its `wte`).
    pub(crate) fn load(
        weights: &crate::lm::Weights,
        name: &str,
        config: &Gpt2Config,
    ) -> Result<Self> {
        if config.n_embd % config.n_head != 0 {
            return Err(SpeechError::BadConfig {
                field: "n_embd".into(),
                why: format!(
                    "{} not divisible by n_head {}",
                    config.n_embd, config.n_head
                ),
            });
        }
        if config.position_embedding_type != "rope" {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "position_embedding_type {}: only rope is ported",
                    config.position_embedding_type
                ),
            });
        }
        // `RotaryEmbedding(head_dim, rope_base)`: adjacent-pair rotation
        // with `angle = position * base ** (-2 d / dim)`, which is the
        // shared rope's plain traditional construction.
        let rope = crate::lm::rope::Rope::initialize(
            config.n_embd / config.n_head,
            config.rope_base,
            true,
            None,
            None,
        )?;
        let head_dim = config.n_embd / config.n_head;
        let base_scale = if config.scale_attn_weights {
            (head_dim as f64).powf(-0.5) as f32
        } else {
            1.0
        };
        let mut blocks = Vec::with_capacity(config.n_layer);
        for index in 0..config.n_layer {
            let layer = format!("{name}.h.{index}.");
            let fused = weights.linear(
                &format!("{layer}attn.c_attn"),
                config.n_embd,
                3 * config.n_embd,
                crate::lm::Bias::Required,
            )?;
            // mx.split(qkv, 3, axis=-1): equal consecutive output ranges.
            let (q_proj, k_proj, v_proj) = split_fused_linear(&fused, config.n_embd, config.n_embd);
            let layer_scale = if config.scale_attn_by_inverse_layer_idx {
                base_scale / (index + 1) as f32
            } else {
                base_scale
            };
            let attn = Attention {
                q_proj,
                k_proj,
                v_proj,
                o_proj: weights.linear(
                    &format!("{layer}attn.c_proj"),
                    config.n_embd,
                    config.n_embd,
                    crate::lm::Bias::Required,
                )?,
                q_norm: None,
                k_norm: None,
                n_heads: config.n_head,
                n_kv_heads: config.n_head,
                head_dim,
                scale: layer_scale,
                rope: Some(rope.clone()),
                window: None,
            };
            let mlp = Mlp {
                fc_in: weights.linear(
                    &format!("{layer}mlp.fc_in"),
                    config.n_embd,
                    config.n_inner,
                    crate::lm::Bias::Required,
                )?,
                fc_out: weights.linear(
                    &format!("{layer}mlp.fc_out"),
                    config.n_inner,
                    config.n_embd,
                    crate::lm::Bias::Required,
                )?,
                activation: config.activation_function.clone(),
            };
            blocks.push(Block {
                ln_1: load_layer_norm(
                    weights,
                    &format!("{layer}ln_1"),
                    config.n_embd,
                    config.layer_norm_epsilon,
                )?,
                attn,
                ln_2: load_layer_norm(
                    weights,
                    &format!("{layer}ln_2"),
                    config.n_embd,
                    config.layer_norm_epsilon,
                )?,
                mlp,
            });
        }
        Ok(Self {
            config: config.clone(),
            blocks,
            ln_f: load_layer_norm(
                weights,
                &format!("{name}.ln_f"),
                config.n_embd,
                config.layer_norm_epsilon,
            )?,
        })
    }

    /// Runs the stack over row-major embeddings `[rows, n_embd]`
    /// cacheless (the reference local transformer takes `inputs_embeds`
    /// with no cache). Returns the post `ln_f` hidden states.
    pub(crate) fn forward(&self, embeds: &[f32]) -> Result<Vec<f32>> {
        let width = self.config.n_embd;
        let rows = embeds.len() / width;
        if rows == 0 {
            return Err(SpeechError::Input {
                why: "empty local transformer input".into(),
            });
        }
        let mut h = embeds.to_vec();
        for block in &self.blocks {
            let mut attended = h.clone();
            block.ln_1.apply(&mut attended, rows);
            let attn = block.attn.forward(&attended, rows, 0, None)?;
            for (row, out) in attn.chunks_exact(width).enumerate() {
                let base = row * width;
                for (col, value) in out.iter().enumerate() {
                    h[base + col] += value;
                }
            }
            let mut fed = h.clone();
            block.ln_2.apply(&mut fed, rows);
            let mlp = block.mlp.forward(&fed, rows)?;
            for (row, out) in mlp.chunks_exact(width).enumerate() {
                let base = row * width;
                for (col, value) in out.iter().enumerate() {
                    h[base + col] += value;
                }
            }
        }
        self.ln_f.apply(&mut h, rows);
        Ok(h)
    }
}

fn load_layer_norm(
    weights: &crate::lm::Weights,
    name: &str,
    width: usize,
    epsilon: f32,
) -> Result<LayerNorm> {
    let weight = weights.tensor(name, &[width])?;
    let bias = weights.raw_tensor(&format!("{name}.bias"), &[width])?;
    Ok(LayerNorm::new(weight, Some(bias), epsilon))
}
