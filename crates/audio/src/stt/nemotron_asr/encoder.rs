//! Causal FastConformer encoder for the pinned Nemotron ASR checkpoint.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::models::stt::nemotron_asr::NemotronAsrConfig;
use crate::ops;
use crate::{Result, SpeechError};

const LN_EPS: f32 = 1e-5;

fn tensor_error(name: &str, why: impl Into<String>) -> SpeechError {
    SpeechError::Tensor {
        name: name.to_string(),
        why: why.into(),
    }
}

fn load_tensor(file: &SafetensorsFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let desc = file
        .descriptor(name)
        .ok_or_else(|| tensor_error(name, "missing from safetensors"))?;
    if desc.shape != shape {
        return Err(tensor_error(
            name,
            format!("expected shape {shape:?}, found {:?}", desc.shape),
        ));
    }
    if !matches!(desc.dtype.as_str(), "F16" | "BF16" | "F32") {
        return Err(tensor_error(
            name,
            format!("expected floating point weights, found {}", desc.dtype),
        ));
    }
    file.load_as_f32(name)
        .map_err(|error| tensor_error(name, format!("load failed: {error}")))
}

fn load_conv2d(
    file: &SafetensorsFile,
    prefix: &str,
    out_channels: usize,
    in_channels: usize,
    kernel: usize,
    groups: usize,
) -> Result<Conv2d> {
    let weight_name = format!("{prefix}.weight");
    let bias_name = format!("{prefix}.bias");
    let in_per_group = in_channels / groups;
    // MLX stores Conv2d weights as [out, kernel_h, kernel_w, in_per_group].
    let source = load_tensor(
        file,
        &weight_name,
        &[out_channels, kernel, kernel, in_per_group],
    )?;
    let mut weight = vec![0.0; source.len()];
    for out in 0..out_channels {
        for input in 0..in_per_group {
            for kh in 0..kernel {
                for kw in 0..kernel {
                    let src = (((out * kernel + kh) * kernel + kw) * in_per_group) + input;
                    let dst = (((out * in_per_group + input) * kernel + kh) * kernel) + kw;
                    weight[dst] = source[src];
                }
            }
        }
    }
    Ok(Conv2d {
        weight,
        bias: load_tensor(file, &bias_name, &[out_channels])?,
        in_channels,
        out_channels,
        kernel,
        groups,
    })
}

fn load_conv1d(
    file: &SafetensorsFile,
    prefix: &str,
    out_channels: usize,
    in_channels: usize,
    kernel: usize,
    groups: usize,
) -> Result<Vec<f32>> {
    let name = format!("{prefix}.weight");
    let in_per_group = in_channels / groups;
    // MLX stores Conv1d weights as [out, kernel, in_per_group].
    let source = load_tensor(file, &name, &[out_channels, kernel, in_per_group])?;
    let mut weight = vec![0.0; source.len()];
    for out in 0..out_channels {
        for input in 0..in_per_group {
            for k in 0..kernel {
                weight[(out * in_per_group + input) * kernel + k] =
                    source[(out * kernel + k) * in_per_group + input];
            }
        }
    }
    Ok(weight)
}

fn load_linear(file: &SafetensorsFile, prefix: &str, out: usize, input: usize) -> Result<Linear> {
    let weight = load_tensor(file, &format!("{prefix}.weight"), &[out, input])?;
    let bias_name = format!("{prefix}.bias");
    let bias = if file.contains_tensor(&bias_name) {
        Some(load_tensor(file, &bias_name, &[out])?)
    } else {
        None
    };
    Ok(Linear { weight, bias })
}

fn load_bias_free_linear(
    file: &SafetensorsFile,
    prefix: &str,
    out: usize,
    input: usize,
) -> Result<Linear> {
    let linear = load_linear(file, prefix, out, input)?;
    if linear.bias.is_some() {
        return Err(tensor_error(
            &format!("{prefix}.bias"),
            "checkpoint has a bias but this Nemotron layer is configured bias-free",
        ));
    }
    Ok(linear)
}

fn require_absent(file: &SafetensorsFile, name: &str) -> Result<()> {
    if file.contains_tensor(name) {
        return Err(tensor_error(
            name,
            "unexpected bias in the configured bias-free Nemotron layer",
        ));
    }
    Ok(())
}

fn load_norm(file: &SafetensorsFile, prefix: &str, width: usize) -> Result<LayerNorm> {
    Ok(LayerNorm {
        weight: load_tensor(file, &format!("{prefix}.weight"), &[width])?,
        bias: load_tensor(file, &format!("{prefix}.bias"), &[width])?,
    })
}

struct Conv2d {
    weight: Vec<f32>,
    bias: Vec<f32>,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    groups: usize,
}

struct Linear {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

struct Subsampling {
    first: Conv2d,
    stages: Vec<(Conv2d, Conv2d)>,
    output: Linear,
    channels: usize,
    input_mels: usize,
}

struct RelPosAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    pos: Linear,
    bias_u: Vec<f32>,
    bias_v: Vec<f32>,
}

struct FeedForward {
    input: Linear,
    output: Linear,
}

struct ConformerConvolution {
    pointwise_in: Vec<f32>,
    depthwise: Vec<f32>,
    depthwise_bias: Option<Vec<f32>>,
    norm: LayerNorm,
    pointwise_out: Vec<f32>,
    channels: usize,
    kernel: usize,
}

struct ConformerLayer {
    norm_ff1: LayerNorm,
    ff1: FeedForward,
    norm_attn: LayerNorm,
    attention: RelPosAttention,
    norm_conv: LayerNorm,
    convolution: ConformerConvolution,
    norm_ff2: LayerNorm,
    ff2: FeedForward,
    norm_out: LayerNorm,
    hidden: usize,
    heads: usize,
    expansion: usize,
}

/// Offline Nemotron encoder. The attention mask preserves its trained causal
/// left context and right look-ahead, without carrying chunk cache state.
pub(crate) struct NemotronEncoder {
    subsampling: Subsampling,
    layers: Vec<ConformerLayer>,
    hidden: usize,
    default_context: [usize; 2],
}

impl Subsampling {
    fn load(file: &SafetensorsFile, config: &NemotronAsrConfig) -> Result<Self> {
        let channels = config.subsampling_channels;
        let first = load_conv2d(file, "encoder.pre_encode.conv.0", channels, 1, 3, 1)?;
        let stages = (1..config.subsampling_factor.ilog2() as usize)
            .map(|stage| {
                let depthwise_index = 2 + 3 * (stage - 1);
                let pointwise_index = depthwise_index + 1;
                Ok((
                    load_conv2d(
                        file,
                        &format!("encoder.pre_encode.conv.{depthwise_index}"),
                        channels,
                        channels,
                        3,
                        channels,
                    )?,
                    load_conv2d(
                        file,
                        &format!("encoder.pre_encode.conv.{pointwise_index}"),
                        channels,
                        channels,
                        1,
                        1,
                    )?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut mel_width = config.num_mels;
        for _ in 0..config.subsampling_factor.ilog2() {
            mel_width = mel_width / 2 + 1;
        }
        let output = load_linear(
            file,
            "encoder.pre_encode.out",
            config.encoder_hidden,
            channels * mel_width,
        )?;
        Ok(Self {
            first,
            stages,
            output,
            channels,
            input_mels: config.num_mels,
        })
    }

    fn forward(&self, features: &[f32], frames: usize) -> Result<(Vec<f32>, usize)> {
        if frames == 0 || features.len() != frames * self.input_mels {
            return Err(SpeechError::Audio(
                "Nemotron mel input must be nonempty [frame, mel]".into(),
            ));
        }
        let mut x = features.to_vec(); // [1, time, mel], channel-major
        let mut time = frames;
        let mut freq = self.input_mels;
        let (padded, padded_time, padded_freq) = pad_causal_2d(&x, 1, time, freq);
        x = self.first.apply(&padded, padded_time, padded_freq, 2);
        time = time / 2 + 1;
        freq = freq / 2 + 1;
        relu(&mut x);
        for (depthwise, pointwise) in &self.stages {
            let (padded, padded_time, padded_freq) = pad_causal_2d(&x, self.channels, time, freq);
            x = depthwise.apply(&padded, padded_time, padded_freq, 2);
            time = time / 2 + 1;
            freq = freq / 2 + 1;
            x = pointwise.apply(&x, time, freq, 1);
            relu(&mut x);
        }
        let mut flat = vec![0.0; time * self.channels * freq];
        for t in 0..time {
            for channel in 0..self.channels {
                let source = channel * time * freq + t * freq;
                let target = t * self.channels * freq + channel * freq;
                flat[target..target + freq].copy_from_slice(&x[source..source + freq]);
            }
        }
        let encoded = ops::linear(
            &flat,
            &self.output.weight,
            self.output.bias.as_deref(),
            time,
            self.channels * freq,
            self.output.weight.len() / (self.channels * freq),
        );
        Ok((encoded, time))
    }
}

impl Conv2d {
    fn apply(&self, input: &[f32], height: usize, width: usize, stride: usize) -> Vec<f32> {
        ops::conv2d(
            input,
            &self.weight,
            Some(&self.bias),
            self.in_channels,
            self.out_channels,
            height,
            width,
            self.kernel,
            self.kernel,
            stride,
            0,
            self.groups,
        )
    }
}

impl RelPosAttention {
    fn load(file: &SafetensorsFile, prefix: &str, hidden: usize, heads: usize) -> Result<Self> {
        let linear =
            |name: &str| load_bias_free_linear(file, &format!("{prefix}.{name}"), hidden, hidden);
        let head_dim = hidden / heads;
        Ok(Self {
            q: linear("linear_q")?,
            k: linear("linear_k")?,
            v: linear("linear_v")?,
            out: linear("linear_out")?,
            pos: load_bias_free_linear(file, &format!("{prefix}.linear_pos"), hidden, hidden)?,
            bias_u: load_tensor(file, &format!("{prefix}.pos_bias_u"), &[heads, head_dim])?,
            bias_v: load_tensor(file, &format!("{prefix}.pos_bias_v"), &[heads, head_dim])?,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        position: &[f32],
        time: usize,
        hidden: usize,
        heads: usize,
        context: [usize; 2],
    ) -> Vec<f32> {
        let head_dim = hidden / heads;
        let pos_len = 2 * time - 1;
        let q = ops::linear(input, &self.q.weight, None, time, hidden, hidden);
        let k = ops::linear(input, &self.k.weight, None, time, hidden, hidden);
        let v = ops::linear(input, &self.v.weight, None, time, hidden, hidden);
        let p = ops::linear(position, &self.pos.weight, None, pos_len, hidden, hidden);
        let k_heads = split_heads(&k, time, heads, head_dim);
        let v_heads = split_heads(&v, time, heads, head_dim);
        let p_heads = split_heads(&p, pos_len, heads, head_dim);
        let chunk_size = context[1] + 1;
        let left_chunks = context[0] / chunk_size;
        let mut attended = vec![0.0; heads * time * head_dim];
        for head in 0..heads {
            let k_head = &k_heads[head * time * head_dim..(head + 1) * time * head_dim];
            let v_head = &v_heads[head * time * head_dim..(head + 1) * time * head_dim];
            let p_head = &p_heads[head * pos_len * head_dim..(head + 1) * pos_len * head_dim];
            let mut scores = Vec::with_capacity(((left_chunks + 1) * chunk_size).min(time));
            for query in 0..time {
                let q_u: Vec<f32> = (0..head_dim)
                    .map(|dim| {
                        q[query * hidden + head * head_dim + dim]
                            + self.bias_u[head * head_dim + dim]
                    })
                    .collect();
                let q_v: Vec<f32> = (0..head_dim)
                    .map(|dim| {
                        q[query * hidden + head * head_dim + dim]
                            + self.bias_v[head * head_dim + dim]
                    })
                    .collect();
                let query_chunk = query / chunk_size;
                let first = query_chunk.saturating_sub(left_chunks) * chunk_size;
                let end = ((query_chunk + 1) * chunk_size).min(time);
                scores.clear();
                for key in first..end {
                    let key_start = key * head_dim;
                    let relative_start = (time - 1 + key - query) * head_dim;
                    scores.push(scaled_score(
                        dot(&q_u, &k_head[key_start..key_start + head_dim]),
                        dot(&q_v, &p_head[relative_start..relative_start + head_dim]),
                        head_dim,
                    ));
                }
                ops::softmax_row(&mut scores);
                let dst = head * time * head_dim + query * head_dim;
                for (score, key) in scores.iter().zip(first..end) {
                    for dim in 0..head_dim {
                        attended[dst + dim] += score * v_head[key * head_dim + dim];
                    }
                }
            }
        }
        let merged = merge_heads(&attended, time, heads, head_dim);
        ops::linear(&merged, &self.out.weight, None, time, hidden, hidden)
    }
}

fn scaled_score(content: f32, position: f32, head_dim: usize) -> f32 {
    (content + position) / (head_dim as f32).sqrt()
}

impl FeedForward {
    fn load(file: &SafetensorsFile, prefix: &str, hidden: usize, expanded: usize) -> Result<Self> {
        Ok(Self {
            input: load_bias_free_linear(file, &format!("{prefix}.linear1"), expanded, hidden)?,
            output: load_bias_free_linear(file, &format!("{prefix}.linear2"), hidden, expanded)?,
        })
    }

    fn forward(&self, input: &[f32], rows: usize, hidden: usize, expanded: usize) -> Vec<f32> {
        let mut value = ops::linear(input, &self.input.weight, None, rows, hidden, expanded);
        silu(&mut value);
        ops::linear(&value, &self.output.weight, None, rows, expanded, hidden)
    }
}

impl ConformerConvolution {
    fn load(file: &SafetensorsFile, prefix: &str, channels: usize, kernel: usize) -> Result<Self> {
        require_absent(file, &format!("{prefix}.pointwise_conv1.bias"))?;
        require_absent(file, &format!("{prefix}.depthwise_conv.bias"))?;
        require_absent(file, &format!("{prefix}.pointwise_conv2.bias"))?;
        Ok(Self {
            pointwise_in: load_conv1d(
                file,
                &format!("{prefix}.pointwise_conv1"),
                channels * 2,
                channels,
                1,
                1,
            )?,
            depthwise: load_conv1d(
                file,
                &format!("{prefix}.depthwise_conv"),
                channels,
                channels,
                kernel,
                channels,
            )?,
            depthwise_bias: None,
            norm: load_norm(file, &format!("{prefix}.batch_norm"), channels)?,
            pointwise_out: load_conv1d(
                file,
                &format!("{prefix}.pointwise_conv2"),
                channels,
                channels,
                1,
                1,
            )?,
            channels,
            kernel,
        })
    }

    fn forward(&self, input: &[f32], time: usize) -> Vec<f32> {
        let channels = self.channels;
        let mut channel_major = transpose_time_channels(input, time, channels);
        let pointwise = ops::conv1d(
            &channel_major,
            &self.pointwise_in,
            None,
            channels,
            channels * 2,
            1,
            1,
            0,
            1,
            1,
        );
        let mut gated = vec![0.0; channels * time];
        for channel in 0..channels {
            for frame in 0..time {
                let a = pointwise[channel * time + frame];
                let b = pointwise[(channels + channel) * time + frame];
                gated[channel * time + frame] = a / (1.0 + (-b).exp());
            }
        }
        channel_major = ops::pad_left(&gated, channels, self.kernel - 1);
        let mut depthwise = ops::conv1d(
            &channel_major,
            &self.depthwise,
            self.depthwise_bias.as_deref(),
            channels,
            channels,
            self.kernel,
            1,
            0,
            1,
            channels,
        );
        let mut norm_input = transpose_channels_time(&depthwise, time, channels);
        ops::layernorm(
            &mut norm_input,
            time,
            channels,
            &self.norm.weight,
            Some(&self.norm.bias),
            LN_EPS,
        );
        silu(&mut norm_input);
        depthwise = transpose_time_channels(&norm_input, time, channels);
        let pointwise = ops::conv1d(
            &depthwise,
            &self.pointwise_out,
            None,
            channels,
            channels,
            1,
            1,
            0,
            1,
            1,
        );
        transpose_channels_time(&pointwise, time, channels)
    }
}

impl ConformerLayer {
    fn load(
        file: &SafetensorsFile,
        layer: usize,
        hidden: usize,
        heads: usize,
        expansion: usize,
        conv_kernel: usize,
    ) -> Result<Self> {
        let prefix = format!("encoder.layers.{layer}");
        let expanded = hidden * expansion;
        Ok(Self {
            norm_ff1: load_norm(file, &format!("{prefix}.norm_feed_forward1"), hidden)?,
            ff1: FeedForward::load(file, &format!("{prefix}.feed_forward1"), hidden, expanded)?,
            norm_attn: load_norm(file, &format!("{prefix}.norm_self_att"), hidden)?,
            attention: RelPosAttention::load(file, &format!("{prefix}.self_attn"), hidden, heads)?,
            norm_conv: load_norm(file, &format!("{prefix}.norm_conv"), hidden)?,
            convolution: ConformerConvolution::load(
                file,
                &format!("{prefix}.conv"),
                hidden,
                conv_kernel,
            )?,
            norm_ff2: load_norm(file, &format!("{prefix}.norm_feed_forward2"), hidden)?,
            ff2: FeedForward::load(file, &format!("{prefix}.feed_forward2"), hidden, expanded)?,
            norm_out: load_norm(file, &format!("{prefix}.norm_out"), hidden)?,
            hidden,
            heads,
            expansion,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        position: &[f32],
        time: usize,
        context: [usize; 2],
    ) -> Vec<f32> {
        let mut residual = input.to_vec();
        let mut normalized = input.to_vec();
        apply_norm(&mut normalized, time, self.hidden, &self.norm_ff1);
        let ff1 = self
            .ff1
            .forward(&normalized, time, self.hidden, self.hidden * self.expansion);
        add_scaled(&mut residual, &ff1, 0.5);

        normalized.clone_from(&residual);
        apply_norm(&mut normalized, time, self.hidden, &self.norm_attn);
        let attention = self.attention.forward(
            &normalized,
            position,
            time,
            self.hidden,
            self.heads,
            context,
        );
        add_scaled(&mut residual, &attention, 1.0);

        normalized.clone_from(&residual);
        apply_norm(&mut normalized, time, self.hidden, &self.norm_conv);
        let convolution = self.convolution.forward(&normalized, time);
        add_scaled(&mut residual, &convolution, 1.0);

        normalized.clone_from(&residual);
        apply_norm(&mut normalized, time, self.hidden, &self.norm_ff2);
        let ff2 = self
            .ff2
            .forward(&normalized, time, self.hidden, self.hidden * self.expansion);
        add_scaled(&mut residual, &ff2, 0.5);
        apply_norm(&mut residual, time, self.hidden, &self.norm_out);
        residual
    }
}

impl NemotronEncoder {
    pub(crate) fn load(file: &SafetensorsFile, config: &NemotronAsrConfig) -> Result<Self> {
        let subsampling = Subsampling::load(file, config)?;
        let layers = (0..config.encoder_layers)
            .map(|index| {
                ConformerLayer::load(
                    file,
                    index,
                    config.encoder_hidden,
                    config.encoder_heads,
                    config.encoder_expansion,
                    config.conv_kernel,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            subsampling,
            layers,
            hidden: config.encoder_hidden,
            default_context: config.default_attention_context,
        })
    }

    pub(crate) fn encode(
        &self,
        mel: &[f32],
        frames: usize,
        mels: usize,
    ) -> Result<(Vec<f32>, usize)> {
        if mels != self.subsampling.input_mels {
            return Err(SpeechError::Audio(format!(
                "Nemotron expects {} mel bands, received {mels}",
                self.subsampling.input_mels
            )));
        }
        let (mut encoded, time) = self.subsampling.forward(mel, frames)?;
        let position = relative_position_encoding(time, self.hidden);
        for layer in &self.layers {
            encoded = layer.forward(&encoded, &position, time, self.default_context);
        }
        Ok((encoded, time))
    }
}

fn pad_causal_2d(
    input: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> (Vec<f32>, usize, usize) {
    let pad_before = 2;
    let pad_after = 1;
    let padded_height = height + pad_before + pad_after;
    let padded_width = width + pad_before + pad_after;
    let mut output = vec![0.0; channels * padded_height * padded_width];
    for channel in 0..channels {
        for row in 0..height {
            let src = channel * height * width + row * width;
            let dst = channel * padded_height * padded_width
                + (row + pad_before) * padded_width
                + pad_before;
            output[dst..dst + width].copy_from_slice(&input[src..src + width]);
        }
    }
    (output, padded_height, padded_width)
}

fn relative_position_encoding(time: usize, hidden: usize) -> Vec<f32> {
    let positions = 2 * time - 1;
    let scale = -(10_000f64.ln() / hidden as f64) as f32;
    let div: Vec<f32> = (0..hidden / 2)
        .map(|index| ((2 * index) as f32 * scale).exp())
        .collect();
    let mut output = vec![0.0; positions * hidden];
    for row in 0..positions {
        let position = (time as isize - 1 - row as isize) as f32;
        for (index, &frequency) in div.iter().enumerate() {
            let angle = position * frequency;
            output[row * hidden + index * 2] = angle.sin();
            output[row * hidden + index * 2 + 1] = angle.cos();
        }
    }
    output
}

#[cfg(test)]
fn rel_shift(input: &[f32], heads: usize, time: usize, pos_len: usize) -> Vec<f32> {
    let mut output = vec![0.0; heads * time * time];
    for head in 0..heads {
        for query in 0..time {
            for key in 0..time {
                let source_pos = time - 1 + key - query;
                output[head * time * time + query * time + key] =
                    input[head * time * pos_len + query * pos_len + source_pos];
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::{dot, rel_shift, scaled_score, split_heads, Linear, RelPosAttention};
    use crate::ops;

    #[test]
    fn relative_shift_maps_each_query_to_key_relative_position() {
        let input: Vec<f32> = (0..15).map(|value| value as f32).collect();
        assert_eq!(
            rel_shift(&input, 1, 3, 5),
            vec![2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0]
        );
    }

    #[test]
    fn relative_attention_uses_inverse_sqrt_head_scaling() {
        assert_eq!(scaled_score(8.0, 4.0, 16), 3.0);
    }

    #[test]
    fn bounded_attention_matches_dense_masked_reference() {
        let time = 6;
        let hidden = 4;
        let heads = 2;
        let head_dim = 2;
        let pos_len = 2 * time - 1;
        let identity = || Linear {
            weight: (0..hidden * hidden)
                .map(|index| {
                    if index / hidden == index % hidden {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            bias: None,
        };
        let attention = RelPosAttention {
            q: identity(),
            k: identity(),
            v: identity(),
            out: identity(),
            pos: identity(),
            bias_u: vec![0.1, -0.2, 0.3, 0.05],
            bias_v: vec![-0.15, 0.25, 0.1, -0.3],
        };
        let input: Vec<f32> = (0..time * hidden)
            .map(|index| ((index * 7 + 2) % 17) as f32 * 0.07 - 0.4)
            .collect();
        let position: Vec<f32> = (0..pos_len * hidden)
            .map(|index| ((index * 5 + 3) % 23) as f32 * 0.03 - 0.2)
            .collect();
        let k_heads = split_heads(&input, time, heads, head_dim);
        let v_heads = split_heads(&input, time, heads, head_dim);
        let p_heads = split_heads(&position, pos_len, heads, head_dim);
        for context in [[2, 1], [4, 0]] {
            let actual = attention.forward(&input, &position, time, hidden, heads, context);
            let mut ac = vec![0.0; heads * time * time];
            let mut bd = vec![0.0; heads * time * pos_len];
            for head in 0..heads {
                for query in 0..time {
                    let start = query * hidden + head * head_dim;
                    let q_u: Vec<f32> = (0..head_dim)
                        .map(|dim| input[start + dim] + attention.bias_u[head * head_dim + dim])
                        .collect();
                    let q_v: Vec<f32> = (0..head_dim)
                        .map(|dim| input[start + dim] + attention.bias_v[head * head_dim + dim])
                        .collect();
                    for key in 0..time {
                        let offset = (head * time + key) * head_dim;
                        ac[(head * time + query) * time + key] =
                            dot(&q_u, &k_heads[offset..offset + head_dim]);
                    }
                    for index in 0..pos_len {
                        let offset = (head * pos_len + index) * head_dim;
                        bd[(head * time + query) * pos_len + index] =
                            dot(&q_v, &p_heads[offset..offset + head_dim]);
                    }
                }
            }
            let bd = rel_shift(&bd, heads, time, pos_len);
            let chunk = context[1] + 1;
            let left_chunks = context[0] / chunk;
            let mut attended = vec![0.0; heads * time * head_dim];
            for head in 0..heads {
                for query in 0..time {
                    let query_chunk = query / chunk;
                    let mut scores = vec![f32::NEG_INFINITY; time];
                    for key in 0..time {
                        let key_chunk = key / chunk;
                        if query_chunk >= key_chunk && query_chunk - key_chunk <= left_chunks {
                            let index = (head * time + query) * time + key;
                            scores[key] = scaled_score(ac[index], bd[index], head_dim);
                        }
                    }
                    ops::softmax_row(&mut scores);
                    for key in 0..time {
                        for dim in 0..head_dim {
                            attended[(head * time + query) * head_dim + dim] +=
                                scores[key] * v_heads[(head * time + key) * head_dim + dim];
                        }
                    }
                }
            }
            let expected = super::merge_heads(&attended, time, heads, head_dim);
            for (index, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-5,
                    "context {context:?}, index {index}: bounded {a}, dense {b}"
                );
            }
        }
    }
}

fn split_heads(input: &[f32], time: usize, heads: usize, head_dim: usize) -> Vec<f32> {
    let hidden = heads * head_dim;
    let mut output = vec![0.0; input.len()];
    for frame in 0..time {
        for head in 0..heads {
            let src = frame * hidden + head * head_dim;
            let dst = head * time * head_dim + frame * head_dim;
            output[dst..dst + head_dim].copy_from_slice(&input[src..src + head_dim]);
        }
    }
    output
}

fn merge_heads(input: &[f32], time: usize, heads: usize, head_dim: usize) -> Vec<f32> {
    let hidden = heads * head_dim;
    let mut output = vec![0.0; input.len()];
    for frame in 0..time {
        for head in 0..heads {
            let src = head * time * head_dim + frame * head_dim;
            let dst = frame * hidden + head * head_dim;
            output[dst..dst + head_dim].copy_from_slice(&input[src..src + head_dim]);
        }
    }
    output
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn transpose_time_channels(input: &[f32], time: usize, channels: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for t in 0..time {
        for channel in 0..channels {
            output[channel * time + t] = input[t * channels + channel];
        }
    }
    output
}

fn transpose_channels_time(input: &[f32], time: usize, channels: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for channel in 0..channels {
        for t in 0..time {
            output[t * channels + channel] = input[channel * time + t];
        }
    }
    output
}

fn apply_norm(values: &mut [f32], rows: usize, width: usize, norm: &LayerNorm) {
    ops::layernorm(values, rows, width, &norm.weight, Some(&norm.bias), LN_EPS);
}

fn add_scaled(target: &mut [f32], values: &[f32], scale: f32) {
    for (target, value) in target.iter_mut().zip(values) {
        *target += value * scale;
    }
}

fn relu(values: &mut [f32]) {
    for value in values {
        *value = value.max(0.0);
    }
}

fn silu(values: &mut [f32]) {
    for value in values {
        *value *= 1.0 / (1.0 + (-*value).exp());
    }
}
