//! Granite Speech 5.0 TurboCTC Conformer encoder.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::models::stt::granite_speech5_ctc::GraniteSpeech5Config;
use crate::nn::{LayerNorm, Linear};
use crate::ops;
use crate::stt::wav2vec::ctc::CtcCollapse;
use crate::{Result, SpeechError};

const LOGIT_ROW_BATCH: usize = 8;

// Local loaders keep this family's own error text (see `load_tensor` below)
// rather than the shared `nn::Linear::load` messages.
fn load_linear(
    file: &SafetensorsFile,
    prefix: &str,
    input: usize,
    output: usize,
    has_bias: bool,
) -> Result<Linear> {
    let weight = load_tensor(file, &format!("{prefix}.weight"), &[output, input])?;
    let bias = if has_bias {
        Some(load_tensor(file, &format!("{prefix}.bias"), &[output])?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias, input, output))
}

fn load_layer_norm(file: &SafetensorsFile, prefix: &str, width: usize) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        load_tensor(file, &format!("{prefix}.weight"), &[width])?,
        Some(load_tensor(file, &format!("{prefix}.bias"), &[width])?),
        1e-5,
    ))
}

struct FeedForward {
    first: Linear,
    second: Linear,
}

impl FeedForward {
    fn load(file: &SafetensorsFile, prefix: &str, hidden: usize, inner: usize) -> Result<Self> {
        Ok(Self {
            first: load_linear(file, &format!("{prefix}.linear1"), hidden, inner, true)?,
            second: load_linear(file, &format!("{prefix}.linear2"), inner, hidden, true)?,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut hidden = self.first.forward(x, rows);
        ops::silu(&mut hidden);
        self.second.forward(&hidden, rows)
    }
}

struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    relative_embedding: Vec<f32>,
    hidden: usize,
    heads: usize,
    head_dim: usize,
    context: usize,
    max_positions: usize,
}

impl Attention {
    fn load(file: &SafetensorsFile, prefix: &str, config: &GraniteSpeech5Config) -> Result<Self> {
        let hidden = config.hidden_size;
        let head_dim = config.head_dim;
        let relative_rows = 2 * config.max_position_embeddings + 1;
        Ok(Self {
            query: load_linear(file, &format!("{prefix}.q_proj"), hidden, hidden, false)?,
            key: load_linear(file, &format!("{prefix}.k_proj"), hidden, hidden, false)?,
            value: load_linear(file, &format!("{prefix}.v_proj"), hidden, hidden, false)?,
            output: load_linear(file, &format!("{prefix}.o_proj"), hidden, hidden, true)?,
            relative_embedding: load_tensor(
                file,
                &format!("{prefix}.rel_pos_emb.weight"),
                &[relative_rows, head_dim],
            )?,
            hidden,
            heads: config.num_attention_heads,
            head_dim,
            context: config.context_size,
            max_positions: config.max_position_embeddings,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let blocks = rows.div_ceil(self.context);
        let padded_rows = blocks * self.context;
        let mut padded = vec![0.0f32; padded_rows * self.hidden];
        padded[..x.len()].copy_from_slice(x);
        let q = self.query.forward(&padded, padded_rows);
        let k = self.key.forward(&padded, padded_rows);
        let v = self.value.forward(&padded, padded_rows);
        let mut attended = vec![0.0f32; padded_rows * self.hidden];
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut scores = vec![0.0f32; self.context];

        for block in 0..blocks {
            for head in 0..self.heads {
                for query_pos in 0..self.context {
                    let query_row = block * self.context + query_pos;
                    let q_start = query_row * self.hidden + head * self.head_dim;
                    for key_pos in 0..self.context {
                        let key_row = block * self.context + key_pos;
                        let k_start = key_row * self.hidden + head * self.head_dim;
                        let relative = (query_pos as isize - key_pos as isize)
                            .clamp(-(self.context as isize), self.context as isize)
                            + self.max_positions as isize;
                        let relative_start = relative as usize * self.head_dim;
                        let mut content = 0.0f32;
                        let mut position = 0.0f32;
                        for dim in 0..self.head_dim {
                            let qv = q[q_start + dim];
                            content += qv * k[k_start + dim];
                            position += qv * self.relative_embedding[relative_start + dim];
                        }
                        scores[key_pos] = content * scale + position * scale;
                    }
                    ops::softmax_row(&mut scores);
                    let out_start = query_row * self.hidden + head * self.head_dim;
                    for dim in 0..self.head_dim {
                        let mut value = 0.0f32;
                        for key_pos in 0..self.context {
                            let value_row = block * self.context + key_pos;
                            value += scores[key_pos]
                                * v[value_row * self.hidden + head * self.head_dim + dim];
                        }
                        attended[out_start + dim] = value;
                    }
                }
            }
        }
        self.output.forward(&attended, padded_rows)[..rows * self.hidden].to_vec()
    }
}

struct Convolution {
    pointwise_in: Linear,
    depthwise_weight: Vec<f32>,
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
    running_mean: Vec<f32>,
    running_var: Vec<f32>,
    pointwise_out: Linear,
    channels: usize,
    kernel: usize,
    stride: usize,
}

impl Convolution {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        config: &GraniteSpeech5Config,
        subsample: bool,
    ) -> Result<Self> {
        let hidden = config.hidden_size;
        let channels = hidden * config.conv_expansion_factor;
        let kernel = config.conv_kernel_size;
        Ok(Self {
            pointwise_in: load_linear(
                file,
                &format!("{prefix}.pointwise_lin1"),
                hidden,
                2 * channels,
                true,
            )?,
            depthwise_weight: load_tensor(
                file,
                &format!("{prefix}.depthwise_conv.weight"),
                &[channels, 1, kernel],
            )?,
            norm_weight: load_tensor(file, &format!("{prefix}.norm.weight"), &[channels])?,
            norm_bias: load_tensor(file, &format!("{prefix}.norm.bias"), &[channels])?,
            running_mean: load_tensor(file, &format!("{prefix}.norm.running_mean"), &[channels])?,
            running_var: load_tensor(file, &format!("{prefix}.norm.running_var"), &[channels])?,
            pointwise_out: load_linear(
                file,
                &format!("{prefix}.pointwise_lin2"),
                channels,
                hidden,
                true,
            )?,
            channels,
            kernel,
            stride: if subsample { 2 } else { 1 },
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> (Vec<f32>, usize) {
        let gated = self.pointwise_in.forward(x, rows);
        let mut channels_last = vec![0.0f32; rows * self.channels];
        for row in 0..rows {
            for channel in 0..self.channels {
                let value = gated[row * 2 * self.channels + channel];
                let gate = gated[row * 2 * self.channels + self.channels + channel];
                channels_last[row * self.channels + channel] = value / (1.0 + (-gate).exp());
            }
        }
        let mut channels_first = vec![0.0f32; self.channels * rows];
        for row in 0..rows {
            for channel in 0..self.channels {
                channels_first[channel * rows + row] = channels_last[row * self.channels + channel];
            }
        }
        let depthwise = ops::conv1d(
            &channels_first,
            &self.depthwise_weight,
            None,
            self.channels,
            self.channels,
            self.kernel,
            self.stride,
            (self.kernel - 1) / 2,
            1,
            self.channels,
        );
        let output_rows = (rows + self.stride - 1) / self.stride;
        let mut normalized = vec![0.0f32; output_rows * self.channels];
        for row in 0..output_rows {
            for channel in 0..self.channels {
                let value = depthwise[channel * output_rows + row];
                let inv = 1.0 / (self.running_var[channel] + 1e-5).sqrt();
                let value = (value - self.running_mean[channel]) * inv * self.norm_weight[channel]
                    + self.norm_bias[channel];
                normalized[row * self.channels + channel] = value / (1.0 + (-value).exp());
            }
        }
        (
            self.pointwise_out.forward(&normalized, output_rows),
            output_rows,
        )
    }
}

struct EncoderBlock {
    feed_forward1: FeedForward,
    norm_feed_forward1: LayerNorm,
    attention: Attention,
    norm_attention: LayerNorm,
    convolution: Convolution,
    norm_convolution: LayerNorm,
    feed_forward2: FeedForward,
    norm_feed_forward2: LayerNorm,
    norm_out: LayerNorm,
    subsample: bool,
    hidden: usize,
}

impl EncoderBlock {
    fn load(file: &SafetensorsFile, index: usize, config: &GraniteSpeech5Config) -> Result<Self> {
        let prefix = format!("encoder.layers.{index}");
        let hidden = config.hidden_size;
        let inner = config.intermediate_size;
        let subsample = config.subsample_layers.contains(&index);
        Ok(Self {
            feed_forward1: FeedForward::load(
                file,
                &format!("{prefix}.feed_forward1"),
                hidden,
                inner,
            )?,
            norm_feed_forward1: load_layer_norm(
                file,
                &format!("{prefix}.norm_feed_forward1"),
                hidden,
            )?,
            attention: Attention::load(file, &format!("{prefix}.self_attn"), config)?,
            norm_attention: load_layer_norm(file, &format!("{prefix}.norm_self_att"), hidden)?,
            convolution: Convolution::load(file, &format!("{prefix}.conv"), config, subsample)?,
            norm_convolution: load_layer_norm(file, &format!("{prefix}.norm_conv"), hidden)?,
            feed_forward2: FeedForward::load(
                file,
                &format!("{prefix}.feed_forward2"),
                hidden,
                inner,
            )?,
            norm_feed_forward2: load_layer_norm(
                file,
                &format!("{prefix}.norm_feed_forward2"),
                hidden,
            )?,
            norm_out: load_layer_norm(file, &format!("{prefix}.norm_out"), hidden)?,
            subsample,
            hidden,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> (Vec<f32>, usize) {
        let mut normalized = x.to_vec();
        self.norm_feed_forward1.apply(&mut normalized, rows);
        let feed_forward = self.feed_forward1.forward(&normalized, rows);
        let mut hidden = x.to_vec();
        for (value, update) in hidden.iter_mut().zip(feed_forward) {
            *value += 0.5 * update;
        }

        normalized.copy_from_slice(&hidden);
        self.norm_attention.apply(&mut normalized, rows);
        let attention = self.attention.forward(&normalized, rows);
        for (value, update) in hidden.iter_mut().zip(attention) {
            *value += update;
        }

        normalized.copy_from_slice(&hidden);
        self.norm_convolution.apply(&mut normalized, rows);
        let (convolution, conv_rows) = self.convolution.forward(&normalized, rows);
        if self.subsample {
            let output_rows = rows / 2;
            let mut reduced = vec![0.0f32; output_rows * self.hidden];
            for row in 0..output_rows {
                for column in 0..self.hidden {
                    let average = (hidden[(2 * row) * self.hidden + column]
                        + hidden[(2 * row + 1) * self.hidden + column])
                        * 0.5;
                    reduced[row * self.hidden + column] =
                        average + convolution[row * self.hidden + column];
                }
            }
            hidden = reduced;
        } else {
            debug_assert_eq!(conv_rows, rows);
            for (value, update) in hidden.iter_mut().zip(convolution) {
                *value += update;
            }
        }

        normalized.clone_from(&hidden);
        self.norm_feed_forward2
            .apply(&mut normalized, hidden.len() / self.hidden);
        let feed_forward = self
            .feed_forward2
            .forward(&normalized, hidden.len() / self.hidden);
        for (value, update) in hidden.iter_mut().zip(feed_forward) {
            *value += 0.5 * update;
        }
        let output_rows = hidden.len() / self.hidden;
        self.norm_out.apply(&mut hidden, output_rows);
        (hidden, output_rows)
    }
}

pub(crate) struct Encoder {
    input: Linear,
    layers: Vec<EncoderBlock>,
    output: Linear,
    output_mid: Linear,
}

impl Encoder {
    pub(crate) fn load(file: &SafetensorsFile, config: &GraniteSpeech5Config) -> Result<Self> {
        let hidden = config.hidden_size;
        Ok(Self {
            input: load_linear(
                file,
                "encoder.input_linear",
                config.num_mel_bins * 4,
                hidden,
                true,
            )?,
            layers: (0..config.num_hidden_layers)
                .map(|index| EncoderBlock::load(file, index, config))
                .collect::<Result<Vec<_>>>()?,
            output: load_linear(file, "encoder.out", hidden, config.vocab_size, true)?,
            output_mid: load_linear(file, "encoder.out_mid", config.vocab_size, hidden, true)?,
        })
    }

    pub(crate) fn forward(
        &self,
        features: &[f32],
        rows: usize,
        config: &GraniteSpeech5Config,
    ) -> Vec<usize> {
        let mut hidden = self.input.forward(features, rows);
        let mut sequence_len = rows;
        let midpoint = self.layers.len() / 2;
        for (index, layer) in self.layers.iter().enumerate() {
            (hidden, sequence_len) = layer.forward(&hidden, sequence_len);
            if index + 1 == midpoint {
                apply_midpoint_conditioning(
                    &self.output,
                    &self.output_mid,
                    &mut hidden,
                    sequence_len,
                    config.vocab_size,
                );
            }
        }
        collapse_final_logits(&self.output, &hidden, sequence_len, config.pad_token_id)
    }
}

fn apply_midpoint_conditioning(
    output: &Linear,
    output_mid: &Linear,
    hidden: &mut [f32],
    rows: usize,
    vocab_size: usize,
) {
    for first_row in (0..rows).step_by(LOGIT_ROW_BATCH) {
        let count = (rows - first_row).min(LOGIT_ROW_BATCH);
        let start = first_row * output.input;
        let end = start + count * output.input;
        let mut probabilities = output.forward(&hidden[start..end], count);
        for row in probabilities.chunks_exact_mut(vocab_size) {
            ops::softmax_row(row);
        }
        let conditioning = output_mid.forward(&probabilities, count);
        for (value, update) in hidden[start..end].iter_mut().zip(conditioning) {
            *value += update;
        }
    }
}

/// Bound final logits to a row batch before collapsing CTC token IDs.
fn collapse_final_logits(
    projection: &Linear,
    hidden: &[f32],
    rows: usize,
    blank_id: usize,
) -> Vec<usize> {
    let mut collapse = CtcCollapse::new(blank_id);
    for first_row in (0..rows).step_by(LOGIT_ROW_BATCH) {
        let count = (rows - first_row).min(LOGIT_ROW_BATCH);
        let start = first_row * projection.input;
        let end = start + count * projection.input;
        let logits = projection.forward(&hidden[start..end], count);
        for row in logits.chunks_exact(projection.output) {
            let token = row
                .iter()
                .enumerate()
                .fold(
                    (blank_id, f32::NEG_INFINITY),
                    |(best_id, best), (id, &value)| {
                        if value > best {
                            (id, value)
                        } else {
                            (best_id, best)
                        }
                    },
                )
                .0;
            collapse.push(token);
        }
    }
    collapse.finish()
}

#[cfg(test)]
mod tests {
    use super::{apply_midpoint_conditioning, collapse_final_logits};
    use crate::models::stt::granite_speech5_ctc::ctc_collapse;
    use crate::nn::Linear;
    use crate::ops;

    #[test]
    fn midpoint_conditioning_matches_full_probabilities_across_batches() {
        let output = Linear::new(
            vec![0.3, -0.1, 0.2, 0.4, -0.5, 0.1],
            Some(vec![0.1, -0.2, 0.05]),
            2,
            3,
        );
        let output_mid = Linear::new(
            vec![0.2, -0.3, 0.1, -0.2, 0.4, 0.3],
            Some(vec![0.05, -0.1]),
            3,
            2,
        );
        let hidden: Vec<f32> = (0..19 * 2)
            .map(|index| ((index * 7 + 3) % 17) as f32 * 0.09 - 0.4)
            .collect();
        let mut expected = hidden.clone();
        let mut probabilities = output.forward(&hidden, 19);
        for row in probabilities.chunks_exact_mut(3) {
            ops::softmax_row(row);
        }
        let conditioning = output_mid.forward(&probabilities, 19);
        for (value, update) in expected.iter_mut().zip(conditioning) {
            *value += update;
        }
        let mut actual = hidden;
        apply_midpoint_conditioning(&output, &output_mid, &mut actual, 19, 3);
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn final_ctc_collapse_matches_full_logits_across_batches() {
        let projection = Linear::new(
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            Some(vec![0.5, 0.0, 0.0]),
            2,
            3,
        );
        let hidden: Vec<f32> = (0..19)
            .flat_map(|row| match row {
                0 | 9 => [0.0, 0.0],
                1..=8 | 18 => [2.0, 0.0],
                _ => [0.0, 2.0],
            })
            .collect();
        let logits = projection.forward(&hidden, 19);
        let expected = ctc_collapse(&logits, 19, 3, 0);
        assert_eq!(expected, vec![1, 2, 1]);
        assert_eq!(collapse_final_logits(&projection, &hidden, 19, 0), expected);
    }
}

fn load_tensor(file: &SafetensorsFile, name: &str, expected_shape: &[usize]) -> Result<Vec<f32>> {
    let descriptor = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
        name: name.to_owned(),
        why: "missing tensor".into(),
    })?;
    if descriptor.shape != expected_shape {
        return Err(SpeechError::Tensor {
            name: name.to_owned(),
            why: format!("shape {:?}, expected {expected_shape:?}", descriptor.shape),
        });
    }
    file.load_as_f32(name).map_err(|error| SpeechError::Tensor {
        name: name.to_owned(),
        why: format!("cannot decode tensor as f32: {error}"),
    })
}
