//! Wav2Vec2-BERT conformer encoder -> `hidden_states[17]`
//! (`w2vbert.py`): feature projection plus the conformer stack with
//! half-weight feed forwards, clamped relative-key attention, and the
//! causal GLU depthwise conv module. f32 torch-layout weights from
//! `w2vbert_mlx.safetensors`; the int8 `quantized_matmul` path of the
//! reference is refused (the port is f32 only).

use std::path::Path;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::nn::{load_tensor, Linear};
use crate::{ops, Result, SpeechError};

/// Geometry (`w2vbert.py` constants); the fixture shrinks these.
#[derive(Debug, Clone)]
pub struct W2VBertConfig {
    pub hidden: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub layers: usize,
    pub left: i64,
    pub right: i64,
}

impl Default for W2VBertConfig {
    fn default() -> Self {
        W2VBertConfig {
            hidden: 1024,
            num_heads: 16,
            head_dim: 64,
            layers: 17,
            left: 64,
            right: 8,
        }
    }
}

pub struct W2VBert {
    pub config: W2VBertConfig,
    pub(crate) feature_layer_norm: crate::nn::LayerNorm,
    pub(crate) feature_projection: Linear,
    pub(crate) layers: Vec<ConformerLayer>,
}

pub(crate) struct ConformerLayer {
    pub(crate) ffn1_norm: crate::nn::LayerNorm,
    pub(crate) ffn1: (Linear, Linear),
    pub(crate) attn_norm: crate::nn::LayerNorm,
    pub(crate) linear_q: Linear,
    pub(crate) linear_k: Linear,
    pub(crate) linear_v: Linear,
    pub(crate) linear_out: Linear,
    pub(crate) distance_embedding: Vec<f32>,
    pub(crate) conv_norm: crate::nn::LayerNorm,
    pub(crate) pointwise_conv1: Linear,
    pub(crate) depthwise_conv: Vec<f32>,
    pub(crate) depthwise_norm: crate::nn::LayerNorm,
    pub(crate) pointwise_conv2: Linear,
    pub(crate) ffn2_norm: crate::nn::LayerNorm,
    pub(crate) ffn2: (Linear, Linear),
    pub(crate) final_norm: crate::nn::LayerNorm,
}

impl W2VBert {
    /// Loads the checkpoint file (`w2vbert_mlx.safetensors`).
    pub fn open(path: &Path, config: W2VBertConfig) -> Result<Self> {
        let file = SafetensorsFile::open(path).map_err(SpeechError::from)?;
        Self::load(&file, config)
    }

    pub fn load(file: &SafetensorsFile, config: W2VBertConfig) -> Result<Self> {
        let hidden = config.hidden;
        let eps = 1e-5f32;
        let ln = |name: &str| -> Result<crate::nn::LayerNorm> {
            crate::nn::LayerNorm::load(file, name, hidden, eps)
        };
        let lin = |name: &str, input: usize, output: usize| -> Result<Linear> {
            Ok(Linear::new(
                load_tensor(file, &format!("{name}.weight"), &[output, input])?,
                Some(load_tensor(file, &format!("{name}.bias"), &[output])?),
                input,
                output,
            ))
        };
        // The input layer norm sits over the 160-d fbank features.
        let feature_layer_norm =
            crate::nn::LayerNorm::load(file, "feature_projection.layer_norm", 160, eps)?;
        let feature_projection = lin("feature_projection.projection", 160, hidden)?;
        let mut layers = Vec::with_capacity(config.layers);
        for index in 0..config.layers {
            let p = move |suffix: &str| format!("encoder.layers.{index}.{suffix}");
            let ffn1 = (
                lin(&p("ffn1.intermediate_dense"), hidden, hidden * 4)?,
                lin(&p("ffn1.output_dense"), hidden * 4, hidden)?,
            );
            let ffn2 = (
                lin(&p("ffn2.intermediate_dense"), hidden, hidden * 4)?,
                lin(&p("ffn2.output_dense"), hidden * 4, hidden)?,
            );
            layers.push(ConformerLayer {
                ffn1_norm: ln(&p("ffn1_layer_norm"))?,
                ffn1,
                attn_norm: ln(&p("self_attn_layer_norm"))?,
                linear_q: lin(&p("self_attn.linear_q"), hidden, hidden)?,
                linear_k: lin(&p("self_attn.linear_k"), hidden, hidden)?,
                linear_v: lin(&p("self_attn.linear_v"), hidden, hidden)?,
                linear_out: lin(&p("self_attn.linear_out"), hidden, hidden)?,
                distance_embedding: load_tensor(
                    file,
                    &p("self_attn.distance_embedding.weight"),
                    &[
                        config.left as usize + config.right as usize + 1,
                        config.head_dim,
                    ],
                )?,
                conv_norm: ln(&p("conv_module.layer_norm"))?,
                pointwise_conv1: {
                    // k1 conv, no bias; the torch [out, in, 1] weight
                    // flattens to the linear [out, in] pair.
                    let weight = load_tensor(
                        file,
                        &p("conv_module.pointwise_conv1.weight"),
                        &[hidden * 2, hidden, 1],
                    )?;
                    Linear::new(weight, None, hidden, hidden * 2)
                },
                depthwise_conv: load_tensor(
                    file,
                    &p("conv_module.depthwise_conv.weight"),
                    &[hidden, 1, 31],
                )?,
                depthwise_norm: ln(&p("conv_module.depthwise_layer_norm"))?,
                pointwise_conv2: {
                    let weight = load_tensor(
                        file,
                        &p("conv_module.pointwise_conv2.weight"),
                        &[hidden, hidden, 1],
                    )?;
                    Linear::new(weight, None, hidden, hidden)
                },
                ffn2_norm: ln(&p("ffn2_layer_norm"))?,
                ffn2,
                final_norm: ln(&p("final_layer_norm"))?,
            });
        }
        Ok(W2VBert {
            config,
            feature_layer_norm,
            feature_projection,
            layers,
        })
    }

    /// `hidden17`: `(1, T, 160)` fbank features -> `(T, hidden)` states
    /// of layer 17 (0-indexed: after all `layers` layers).
    pub fn hidden17(&self, features: &[f32], frames: usize) -> Result<Vec<f32>> {
        let hidden = self.config.hidden;
        let mut x = features.to_vec();
        self.feature_layer_norm.apply(&mut x, frames);
        let projected = self.feature_projection.forward(&x, frames);
        let mut state = projected;
        for layer in &self.layers {
            state = self.layer_forward(layer, &state, frames)?;
        }
        let _ = hidden;
        Ok(state)
    }

    fn layer_forward(&self, layer: &ConformerLayer, x: &[f32], frames: usize) -> Result<Vec<f32>> {
        let hidden = self.config.hidden;
        let mut out = x.to_vec();
        // ffn1 with the 0.5 residual weight.
        let mut normed = x.to_vec();
        layer.ffn1_norm.apply(&mut normed, frames);
        let mut f = layer.ffn1.0.forward(&normed, frames);
        for value in &mut f {
            *value *= sigmoid(*value);
        }
        let f = layer.ffn1.1.forward(&f, frames);
        for (o, v) in out.iter_mut().zip(&f) {
            *o += 0.5 * v;
        }
        // Relative-key attention.
        let mut normed = out.clone();
        layer.attn_norm.apply(&mut normed, frames);
        let attended = self.attention(layer, &normed, frames)?;
        for (o, v) in out.iter_mut().zip(&attended) {
            *o += v;
        }
        // Conv module (causal depthwise GLU).
        let mut normed = out.clone();
        layer.conv_norm.apply(&mut normed, frames);
        let conv = self.conv_module(layer, &normed, frames)?;
        for (o, v) in out.iter_mut().zip(&conv) {
            *o += v;
        }
        // ffn2 with the 0.5 residual weight.
        let mut normed = out.clone();
        layer.ffn2_norm.apply(&mut normed, frames);
        let mut f = layer.ffn2.0.forward(&normed, frames);
        for value in &mut f {
            *value *= sigmoid(*value);
        }
        let f = layer.ffn2.1.forward(&f, frames);
        for (o, v) in out.iter_mut().zip(&f) {
            *o += 0.5 * v;
        }
        layer.final_norm.apply(&mut out, frames);
        let _ = hidden;
        Ok(out)
    }

    pub(crate) fn attention(
        &self,
        layer: &ConformerLayer,
        x: &[f32],
        frames: usize,
    ) -> Result<Vec<f32>> {
        let heads = self.config.num_heads;
        let dim = self.config.head_dim;
        let hidden = heads * dim;
        let q = layer.linear_q.forward(x, frames);
        let k = layer.linear_k.forward(x, frames);
        let v = layer.linear_v.forward(x, frames);
        let scale = 1.0 / (dim as f32).sqrt();
        // distance = clamp(r - l, -LEFT, RIGHT) + LEFT, r = key.
        let span = (self.config.left + self.config.right) as usize;
        let mut out = vec![0.0f32; frames * hidden];
        for l in 0..frames {
            for heads_chunk in 0..heads {
                let mut weighted = vec![0.0f32; dim];
                let mut scores = vec![0.0f32; frames];
                let mut max = f32::NEG_INFINITY;
                for r in 0..frames {
                    let dist = (r as i64 - l as i64).clamp(-self.config.left, self.config.right)
                        + self.config.left;
                    let pe =
                        &layer.distance_embedding[dist as usize * dim..(dist as usize + 1) * dim];
                    let q_row =
                        &q[(l * heads + heads_chunk) * dim..(l * heads + heads_chunk + 1) * dim];
                    let k_row =
                        &k[(r * heads + heads_chunk) * dim..(r * heads + heads_chunk + 1) * dim];
                    let mut dot = 0.0f32;
                    let mut rel = 0.0f32;
                    for d in 0..dim {
                        dot += q_row[d] * k_row[d];
                        rel += q_row[d] * pe[d];
                    }
                    let score = dot * scale + rel * scale;
                    scores[r] = score;
                    if score > max {
                        max = score;
                    }
                }
                let mut sum = 0.0f32;
                for score in &mut scores {
                    let weight = (*score - max).exp();
                    *score = weight;
                    sum += weight;
                }
                let inv = 1.0 / sum;
                for r in 0..frames {
                    let weight = scores[r] * inv;
                    let v_row =
                        &v[(r * heads + heads_chunk) * dim..(r * heads + heads_chunk + 1) * dim];
                    for d in 0..dim {
                        weighted[d] += weight * v_row[d];
                    }
                }
                for d in 0..dim {
                    out[l * hidden + heads_chunk * dim + d] = weighted[d];
                }
            }
        }
        let _ = span;
        Ok(layer.linear_out.forward(&out, frames))
    }

    pub(crate) fn conv_module(
        &self,
        layer: &ConformerLayer,
        x: &[f32],
        frames: usize,
    ) -> Result<Vec<f32>> {
        let hidden = self.config.hidden;
        // pointwise_conv1 k1: 1024 -> 2048, GLU halves.
        let pw1 = layer.pointwise_conv1.forward(x, frames);
        let mut glu = vec![0.0f32; frames * hidden];
        for t in 0..frames {
            for d in 0..hidden {
                let a = pw1[t * hidden * 2 + d];
                let b = pw1[t * hidden * 2 + hidden + d];
                glu[t * hidden + d] = a / (1.0 + (-b).exp());
            }
        }
        // Causal depthwise k31: left pad 30, groups = hidden. glu is
        // time-major [t, hidden]; the conv runs channel-major.
        let kernel = 31;
        let mut padded = vec![0.0f32; hidden * (frames + kernel - 1)];
        for d in 0..hidden {
            for t in 0..frames {
                padded[d * (frames + kernel - 1) + kernel - 1 + t] = glu[t * hidden + d];
            }
        }
        let conv_out = ops::conv1d(
            &padded,
            &layer.depthwise_conv,
            None,
            hidden,
            hidden,
            kernel,
            1,
            0,
            1,
            hidden,
        );
        // The conv output is channel-major [hidden, frames]; the
        // depthwise norm runs per frame over the channels, so
        // transpose to [frames, hidden] first.
        let mut conv = vec![0.0f32; conv_out.len()];
        for t in 0..frames {
            for d in 0..hidden {
                conv[t * hidden + d] = conv_out[d * frames + t];
            }
        }
        layer.depthwise_norm.apply(&mut conv, frames);
        for value in &mut conv {
            *value *= sigmoid(*value);
        }
        Ok(layer.pointwise_conv2.forward(&conv, frames))
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
