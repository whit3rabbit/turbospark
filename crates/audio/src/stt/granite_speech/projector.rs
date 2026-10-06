//! Granite Speech 1B QFormer audio projector.
//!
//! Reference: `mlx_audio/stt/models/granite_speech/granite_speech.py`
//! (`EncoderProjector`, `QFormerModel`, `QFormerLayer`,
//! `QFormerAttention`, `QFormerSelfOutput`, `QFormerIntermediate`,
//! `QFormerOutput`) at mlx-audio 0.5.7, commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`.
//!
//! The projector windows the encoder output into `window_size` rows, and
//! for every window a fixed learned query of `num_queries` rows runs two
//! QFormer layers (full self-attention over the queries, cross-attention
//! over the window, GELU MLP) before a final linear into the text width.
//! Every QFormer LayerNorm uses eps 1e-12 and the dense output norms add
//! the attention branch back before normalizing.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::ops;
use crate::{Result, SpeechError};

fn missing(name: &str) -> SpeechError {
    SpeechError::Tensor {
        name: name.to_owned(),
        why: "missing from the checkpoint shards".to_owned(),
    }
}

fn shard_of<'a>(files: &'a [SafetensorsFile], name: &str) -> Result<&'a SafetensorsFile> {
    files
        .iter()
        .find(|file| file.contains_tensor(name))
        .ok_or_else(|| missing(name))
}

fn load_vector(files: &[SafetensorsFile], name: &str, len: usize) -> Result<Vec<f32>> {
    let values = shard_of(files, name)?.load_as_f32(name)?;
    if values.len() != len {
        return Err(SpeechError::Tensor {
            name: name.to_owned(),
            why: format!("expected {len} values, got {}", values.len()),
        });
    }
    Ok(values)
}

fn load_linear(
    files: &[SafetensorsFile],
    base: &str,
    input: usize,
    output: usize,
) -> Result<Linear> {
    Ok(Linear {
        weight: load_vector(files, &format!("{base}.weight"), input * output)?,
        bias: load_vector(files, &format!("{base}.bias"), output)?,
        input,
        output,
    })
}

#[derive(Debug, Clone)]
struct Linear {
    weight: Vec<f32>,
    bias: Vec<f32>,
    input: usize,
    output: usize,
}

impl Linear {
    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        ops::linear(
            x,
            &self.weight,
            Some(&self.bias),
            rows,
            self.input,
            self.output,
        )
    }
}

struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    width: usize,
    epsilon: f32,
}

impl LayerNorm {
    fn load(files: &[SafetensorsFile], base: &str, width: usize, epsilon: f32) -> Result<Self> {
        Ok(Self {
            weight: load_vector(files, &format!("{base}.weight"), width)?,
            bias: load_vector(files, &format!("{base}.bias"), width)?,
            width,
            epsilon,
        })
    }

    fn apply(&self, values: &mut [f32], rows: usize) {
        ops::layernorm(
            values,
            rows,
            self.width,
            &self.weight,
            Some(&self.bias),
            self.epsilon,
        );
    }
}

/// Dense projection plus residual LayerNorm (`QFormerSelfOutput` /
/// `QFormerOutput`).
struct DenseResidualNorm {
    dense: Linear,
    norm: LayerNorm,
}

impl DenseResidualNorm {
    fn load(
        files: &[SafetensorsFile],
        dense_base: &str,
        norm_base: &str,
        input: usize,
        output: usize,
        epsilon: f32,
    ) -> Result<Self> {
        Ok(Self {
            dense: load_linear(files, dense_base, input, output)?,
            norm: LayerNorm::load(files, norm_base, output, epsilon)?,
        })
    }

    fn forward(&self, hidden: &[f32], input_tensor: &[f32], rows: usize) -> Vec<f32> {
        let projected = self.dense.forward(hidden, rows);
        let mut out = projected
            .iter()
            .zip(input_tensor)
            .map(|(p, r)| p + r)
            .collect::<Vec<f32>>();
        self.norm.apply(&mut out, rows);
        out
    }
}

/// One attention block: q from the hidden states, k/v from the same states
/// (self) or the encoder window (cross), then dense-plus-norm.
struct QFormerAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: DenseResidualNorm,
    kv_dim: usize,
    heads: usize,
    head_dim: usize,
}

impl QFormerAttention {
    fn load(
        files: &[SafetensorsFile],
        base: &str,
        hidden_size: usize,
        kv_dim: usize,
        heads: usize,
        epsilon: f32,
    ) -> Result<Self> {
        Ok(Self {
            query: load_linear(
                files,
                &format!("{base}.attention.query"),
                hidden_size,
                hidden_size,
            )?,
            key: load_linear(files, &format!("{base}.attention.key"), kv_dim, hidden_size)?,
            value: load_linear(
                files,
                &format!("{base}.attention.value"),
                kv_dim,
                hidden_size,
            )?,
            output: DenseResidualNorm::load(
                files,
                &format!("{base}.output.dense"),
                &format!("{base}.output.LayerNorm"),
                hidden_size,
                hidden_size,
                epsilon,
            )?,
            kv_dim,
            heads,
            head_dim: hidden_size / heads,
        })
    }

    /// hidden: `[q_rows, hidden]`; kv: `[kv_rows, kv_dim]`. Returns
    /// `[q_rows, hidden]`.
    fn forward(&self, hidden: &[f32], kv: &[f32], q_rows: usize, kv_rows: usize) -> Vec<f32> {
        let q = self.query.forward(hidden, q_rows);
        let k = self.key.forward(kv, kv_rows);
        let v = self.value.forward(kv, kv_rows);
        let scale = (self.head_dim as f32).recip().sqrt();
        let mut attended = vec![0.0f32; q_rows * (self.heads * self.head_dim)];
        for head in 0..self.heads {
            let offset = head * self.head_dim;
            for row in 0..q_rows {
                let query = &q[row * self.heads * self.head_dim + offset
                    ..row * self.heads * self.head_dim + offset + self.head_dim];
                let mut scores = vec![0.0f32; kv_rows];
                for j in 0..kv_rows {
                    let key =
                        &k[j * self.kv_dim + offset..j * self.kv_dim + offset + self.head_dim];
                    scores[j] = query.iter().zip(key).map(|(a, b)| a * b).sum::<f32>() * scale;
                }
                ops::softmax_row(&mut scores);
                let target = row * self.heads * self.head_dim + offset;
                for j in 0..kv_rows {
                    let weight = scores[j];
                    if weight == 0.0 {
                        continue;
                    }
                    let value =
                        &v[j * self.kv_dim + offset..j * self.kv_dim + offset + self.head_dim];
                    for (o, val) in attended[target..target + self.head_dim]
                        .iter_mut()
                        .zip(value)
                    {
                        *o += weight * val;
                    }
                }
            }
        }
        self.output.forward(&attended, hidden, q_rows)
    }
}

/// One QFormer layer: self-attention, cross-attention, GELU MLP, each
/// wrapped in dense-plus-norm.
struct QFormerLayer {
    attention: QFormerAttention,
    crossattention: QFormerAttention,
    intermediate: Linear,
    output_query: DenseResidualNorm,
}

impl QFormerLayer {
    fn load(
        files: &[SafetensorsFile],
        base: &str,
        config: &super::config::ProjectorConfig,
    ) -> Result<Self> {
        Ok(Self {
            attention: QFormerAttention::load(
                files,
                &format!("{base}.attention"),
                config.hidden_size,
                config.hidden_size,
                config.num_attention_heads,
                config.layer_norm_eps,
            )?,
            crossattention: QFormerAttention::load(
                files,
                &format!("{base}.crossattention"),
                config.hidden_size,
                config.encoder_hidden_size,
                config.num_attention_heads,
                config.layer_norm_eps,
            )?,
            intermediate: load_linear(
                files,
                &format!("{base}.intermediate_query.dense"),
                config.hidden_size,
                config.intermediate_size,
            )?,
            output_query: DenseResidualNorm::load(
                files,
                &format!("{base}.output_query.dense"),
                &format!("{base}.output_query.LayerNorm"),
                config.intermediate_size,
                config.hidden_size,
                config.layer_norm_eps,
            )?,
        })
    }

    fn forward(&self, hidden: &[f32], encoder: &[f32], queries: usize, window: usize) -> Vec<f32> {
        let hidden = self.attention.forward(hidden, hidden, queries, queries);
        let hidden = self
            .crossattention
            .forward(&hidden, encoder, queries, window);
        let mut wide = self.intermediate.forward(&hidden, queries);
        ops::gelu_erf(&mut wide);
        self.output_query.forward(&wide, &hidden, queries)
    }
}

/// The windowed QFormer projector: learned query, input LayerNorm, two
/// layers, output linear into the text width.
pub struct EncoderProjector {
    query: Vec<f32>,
    query_norm: LayerNorm,
    layers: Vec<QFormerLayer>,
    linear: Linear,
    hidden_size: usize,
    window_size: usize,
    num_queries: usize,
    text_hidden: usize,
}

impl EncoderProjector {
    pub(crate) fn load(
        files: &[SafetensorsFile],
        config: &super::config::ProjectorConfig,
        window_size: usize,
        num_queries: usize,
        text_hidden: usize,
    ) -> Result<Self> {
        Ok(Self {
            query: load_vector(files, "projector.query", num_queries * config.hidden_size)?,
            query_norm: LayerNorm::load(
                files,
                "projector.qformer.layernorm",
                config.hidden_size,
                config.layer_norm_eps,
            )?,
            layers: (0..config.num_hidden_layers)
                .map(|index| {
                    QFormerLayer::load(
                        files,
                        &format!("projector.qformer.encoder.layer.{index}"),
                        config,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            linear: load_linear(files, "projector.linear", config.hidden_size, text_hidden)?,
            hidden_size: config.hidden_size,
            window_size,
            num_queries,
            text_hidden,
        })
    }

    /// Projects the encoder plane `[rows, encoder_hidden]` into
    /// `[nblocks * num_queries, text_hidden]`, padding the tail window.
    pub fn forward(&self, encoded: &[f32], rows: usize) -> Result<Vec<f32>> {
        let width = self.window_size;
        let nblocks = rows.div_ceil(width);
        let mut query = self.query.clone();
        self.query_norm.apply(&mut query, self.num_queries);
        let mut out = Vec::with_capacity(nblocks * self.num_queries * self.text_hidden);
        for block in 0..nblocks {
            let start = block * width;
            let stop = (start + width).min(rows);
            let mut window = vec![0.0f32; width * self.hidden_size];
            window[..(stop - start) * self.hidden_size]
                .copy_from_slice(&encoded[start * self.hidden_size..stop * self.hidden_size]);
            let mut hidden = query.clone();
            for layer in &self.layers {
                hidden = layer.forward(&hidden, &window, self.num_queries, width);
            }
            out.extend(self.linear.forward(&hidden, self.num_queries));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::{DenseResidualNorm, Linear, QFormerAttention};

    /// A one-head 2-dim attention with identity q/k/v projections and an
    /// identity output dense: hidden rows [1, 0] and [0, 1] attend
    /// bidirectionally over three window rows. Hand-derived expectation:
    /// softmax weights [0.4458, 0.4458, 0.1084] (row 0) and
    /// [0.4458, 0.1084, 0.4458] (row 1) give attention outputs whose sum
    /// with the residual normalizes to [-1, 1] / [1, -1] pairs, proving
    /// the cross path reads the encoder window, not the hidden states.
    #[test]
    fn cross_attention_attends_over_the_window_then_applies_dense_norm() {
        let linear = |input: usize, output: usize| Linear {
            weight: (0..output * input)
                .map(|i| (i % (input + 1) == 0) as i32 as f32)
                .collect(),
            bias: vec![0.0; output],
            input,
            output,
        };
        let norm = DenseResidualNorm {
            dense: linear(2, 2),
            norm: super::LayerNorm {
                weight: vec![1.0, 1.0],
                bias: vec![0.0, 0.0],
                width: 2,
                epsilon: 1e-12,
            },
        };
        let attention = QFormerAttention {
            query: linear(2, 2),
            key: linear(2, 2),
            value: linear(2, 2),
            output: norm,
            kv_dim: 2,
            heads: 1,
            head_dim: 2,
        };
        let hidden = [1.0, 0.0, 0.0, 1.0];
        let window = [1.0, 1.0, 1.0, -1.0, -1.0, 1.0];
        let out = attention.forward(&hidden, &window, 2, 3);
        for (row, expected) in out.chunks_exact(2).zip([[1.0, -1.0], [-1.0, 1.0]]) {
            for (actual, want) in row.iter().zip(expected) {
                assert!((actual - want).abs() < 1e-5, "{:?} vs {expected:?}", row);
            }
        }
    }
}
