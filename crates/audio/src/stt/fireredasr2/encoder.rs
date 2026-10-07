use turbospark_model_io::safetensors::SafetensorsFile;

use crate::nn::{LayerNorm, Linear};
use crate::ops;
use crate::{Result, SpeechError};

use super::config::FireRedAsr2Config;

const SUBSAMPLE_CHANNELS: usize = 32;
const SUBSAMPLE_KERNEL: usize = 3;
const LAYER_NORM_EPS: f32 = 1e-5;

pub(super) struct Encoder {
    config: FireRedAsr2Config,
    conv1: Conv2d,
    conv2: Conv2d,
    subsample_projection: Linear,
    layers: Vec<ConformerBlock>,
}

pub(super) struct EncoderOutput {
    pub values: Vec<f32>,
    pub rows: usize,
    pub trace: Option<EncoderTrace>,
}

#[derive(Debug, Clone)]
pub(super) struct EncoderTrace {
    pub subsampled: Snapshot,
    pub first_block: Snapshot,
    pub first_block_stages: Vec<Snapshot>,
    pub final_block: Snapshot,
}

#[derive(Debug, Clone)]
pub(super) struct Snapshot {
    pub shape: Vec<usize>,
    pub indices: Vec<usize>,
    pub values: Vec<f32>,
}

pub(super) fn snapshot(values: &[f32], shape: &[usize]) -> Snapshot {
    let count = values.len().min(16);
    let indices = if count <= 1 {
        vec![0]
    } else {
        (0..count)
            .map(|index| index * (values.len() - 1) / (count - 1))
            .collect()
    };
    Snapshot {
        shape: shape.to_vec(),
        values: indices.iter().map(|&index| values[index]).collect(),
        indices,
    }
}

fn raw_tensor_name(name: &str) -> String {
    // The published safetensors file keeps the upstream PyTorch names. The
    // MLX loader sanitizes these names in memory before constructing modules.
    name.replace(
        "encoder.input_preprocessor.conv1.",
        "encoder.input_preprocessor.conv.0.",
    )
    .replace(
        "encoder.input_preprocessor.conv2.",
        "encoder.input_preprocessor.conv.2.",
    )
    .replace(".net_", ".net.")
}

fn tensor(file: &SafetensorsFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let raw_name = raw_tensor_name(name);
    let descriptor = file
        .descriptor(&raw_name)
        .ok_or_else(|| SpeechError::Tensor {
            name: raw_name.clone(),
            why: "tensor is missing".into(),
        })?;
    if descriptor.shape != shape {
        return Err(SpeechError::Tensor {
            name: raw_name,
            why: format!("expected shape {shape:?}, got {:?}", descriptor.shape),
        });
    }
    file.load_as_f32(&raw_name).map_err(Into::into)
}

// The upstream-name remapping and optional bias stay local to this family.
fn load_linear(file: &SafetensorsFile, name: &str, input: usize, output: usize) -> Result<Linear> {
    let weight = tensor(file, &format!("{name}.weight"), &[output, input])?;
    let bias_name = format!("{name}.bias");
    let raw_bias_name = raw_tensor_name(&bias_name);
    let bias = if file.contains_tensor(&raw_bias_name) {
        Some(tensor(file, &bias_name, &[output])?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias, input, output))
}

struct Conv2d {
    weight: Vec<f32>,
    bias: Vec<f32>,
    input: usize,
    output: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride: usize,
}

impl Conv2d {
    fn load(
        file: &SafetensorsFile,
        name: &str,
        input: usize,
        output: usize,
        kernel: usize,
    ) -> Result<Self> {
        // The checkpoint is PyTorch OIHW, the layout consumed by ops::conv2d.
        let weight = tensor(
            file,
            &format!("{name}.weight"),
            &[output, input, kernel, kernel],
        )?;
        Ok(Self {
            weight,
            bias: tensor(file, &format!("{name}.bias"), &[output])?,
            input,
            output,
            kernel_h: kernel,
            kernel_w: kernel,
            stride: 2,
        })
    }

    fn forward(&self, x: &[f32], height: usize, width: usize) -> (Vec<f32>, usize, usize) {
        let output_height = (height - self.kernel_h) / self.stride + 1;
        let output_width = (width - self.kernel_w) / self.stride + 1;
        let mut output = ops::conv2d(
            x,
            &self.weight,
            Some(&self.bias),
            self.input,
            self.output,
            height,
            width,
            self.kernel_h,
            self.kernel_w,
            self.stride,
            0,
            1,
        );
        for value in &mut output {
            *value = value.max(0.0);
        }
        (output, output_height, output_width)
    }
}

fn load_layer_norm(file: &SafetensorsFile, name: &str, width: usize) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        tensor(file, &format!("{name}.weight"), &[width])?,
        Some(tensor(file, &format!("{name}.bias"), &[width])?),
        LAYER_NORM_EPS,
    ))
}

struct FeedForward {
    norm: LayerNorm,
    expand: Linear,
    project: Linear,
    width: usize,
}

impl FeedForward {
    fn load(file: &SafetensorsFile, prefix: &str, width: usize) -> Result<Self> {
        Ok(Self {
            norm: load_layer_norm(file, &format!("{prefix}.net_0"), width)?,
            expand: load_linear(file, &format!("{prefix}.net_1"), width, width * 4)?,
            project: load_linear(file, &format!("{prefix}.net_4"), width * 4, width)?,
            width,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        self.forward_with_trace(x, rows, false).0
    }

    fn forward_with_trace(
        &self,
        x: &[f32],
        rows: usize,
        trace: bool,
    ) -> (Vec<f32>, Option<Vec<Snapshot>>) {
        let shape = [rows, self.width];
        let mut hidden = x.to_vec();
        self.norm.apply(&mut hidden, rows);
        let mut stages = trace.then(Vec::new);
        if let Some(stages) = &mut stages {
            stages.push(snapshot(&hidden, &shape));
        }
        let mut expanded = self.expand.forward(&hidden, rows);
        if let Some(stages) = &mut stages {
            stages.push(snapshot(&expanded, &[rows, self.width * 4]));
        }
        ops::silu(&mut expanded);
        if let Some(stages) = &mut stages {
            stages.push(snapshot(&expanded, &[rows, self.width * 4]));
        }
        let update = self.project.forward(&expanded, rows);
        if let Some(stages) = &mut stages {
            stages.push(snapshot(&update, &shape));
        }
        let output = x
            .iter()
            .zip(update)
            .map(|(&residual, update)| residual + update)
            .collect::<Vec<_>>();
        if let Some(stages) = &mut stages {
            stages.push(snapshot(&output, &shape));
        }
        (output, stages)
    }
}

struct RelativeAttention {
    q_norm: LayerNorm,
    k_norm: LayerNorm,
    v_norm: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    position: Linear,
    output: Linear,
    pos_bias_u: Vec<f32>,
    pos_bias_v: Vec<f32>,
    heads: usize,
    head_dim: usize,
    width: usize,
}

impl RelativeAttention {
    fn load(file: &SafetensorsFile, prefix: &str, width: usize, heads: usize) -> Result<Self> {
        let head_dim = width / heads;
        Ok(Self {
            q_norm: load_layer_norm(file, &format!("{prefix}.layer_norm_q"), width)?,
            k_norm: load_layer_norm(file, &format!("{prefix}.layer_norm_k"), width)?,
            v_norm: load_layer_norm(file, &format!("{prefix}.layer_norm_v"), width)?,
            q: load_linear(file, &format!("{prefix}.w_qs"), width, width)?,
            k: load_linear(file, &format!("{prefix}.w_ks"), width, width)?,
            v: load_linear(file, &format!("{prefix}.w_vs"), width, width)?,
            position: load_linear(file, &format!("{prefix}.linear_pos"), width, width)?,
            output: load_linear(file, &format!("{prefix}.fc"), width, width)?,
            pos_bias_u: tensor(file, &format!("{prefix}.pos_bias_u"), &[heads, head_dim])?,
            pos_bias_v: tensor(file, &format!("{prefix}.pos_bias_v"), &[heads, head_dim])?,
            heads,
            head_dim,
            width,
        })
    }

    fn forward(&self, x: &[f32], rows: usize, pos_emb: &[f32]) -> Vec<f32> {
        let mut q_input = x.to_vec();
        let mut k_input = x.to_vec();
        let mut v_input = x.to_vec();
        self.q_norm.apply(&mut q_input, rows);
        self.k_norm.apply(&mut k_input, rows);
        self.v_norm.apply(&mut v_input, rows);
        let q = to_head_major(
            &self.q.forward(&q_input, rows),
            rows,
            self.heads,
            self.head_dim,
        );
        let k = to_head_major(
            &self.k.forward(&k_input, rows),
            rows,
            self.heads,
            self.head_dim,
        );
        let v = to_head_major(
            &self.v.forward(&v_input, rows),
            rows,
            self.heads,
            self.head_dim,
        );
        let pos_rows = rows * 2 - 1;
        let p = to_head_major(
            &self.position.forward(pos_emb, pos_rows),
            pos_rows,
            self.heads,
            self.head_dim,
        );
        let mut context = vec![0.0; rows * self.width];
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut scores = vec![0.0f32; rows];

        for head in 0..self.heads {
            let q_base = head * rows * self.head_dim;
            let p_base = head * pos_rows * self.head_dim;
            let k_base = head * rows * self.head_dim;
            let v_base = head * rows * self.head_dim;
            for query in 0..rows {
                let q_row =
                    &q[q_base + query * self.head_dim..q_base + (query + 1) * self.head_dim];
                let bias_u = &self.pos_bias_u[head * self.head_dim..(head + 1) * self.head_dim];
                let bias_v = &self.pos_bias_v[head * self.head_dim..(head + 1) * self.head_dim];
                let q_with_u: Vec<f32> = q_row.iter().zip(bias_u).map(|(&q, &b)| q + b).collect();
                let q_with_v: Vec<f32> = q_row.iter().zip(bias_v).map(|(&q, &b)| q + b).collect();
                // The MLX _rel_shift (zero pad, reshape [2T, T], drop row 0,
                // reshape [T, 2T-1], keep the first T) selects relative
                // position T - 1 + key - query of this same query for every
                // key (see `relative_index`). Only those T of the 2T - 1
                // dots are used, so compute just them; each is the same dot.
                for key in 0..rows {
                    let k_row =
                        &k[k_base + key * self.head_dim..k_base + (key + 1) * self.head_dim];
                    let pos = rows - 1 + key - query;
                    let p_row =
                        &p[p_base + pos * self.head_dim..p_base + (pos + 1) * self.head_dim];
                    scores[key] = (dot(&q_with_u, k_row) + dot(&q_with_v, p_row)) * scale;
                }
                ops::softmax_row(&mut scores);
                for feature in 0..self.head_dim {
                    let mut sum = 0.0f32;
                    for key in 0..rows {
                        sum += scores[key] * v[v_base + key * self.head_dim + feature];
                    }
                    context[query * self.width + head * self.head_dim + feature] = sum;
                }
            }
        }
        self.output.forward(&context, rows)
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&a, &b)| a * b).sum()
}

fn to_head_major(input: &[f32], rows: usize, heads: usize, head_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        for head in 0..heads {
            let source = (row * heads + head) * head_dim;
            let target = (head * rows + row) * head_dim;
            output[target..target + head_dim].copy_from_slice(&input[source..source + head_dim]);
        }
    }
    output
}

struct ConformerConvolution {
    norm: LayerNorm,
    pointwise_in: Conv1d,
    depthwise: Conv1d,
    batch_norm: LayerNorm,
    pointwise_out: Conv1d,
    width: usize,
    kernel: usize,
}

struct Conv1d {
    weight: Vec<f32>,
    input: usize,
    output: usize,
    kernel: usize,
    groups: usize,
    padding: usize,
}

impl Conv1d {
    fn load(
        file: &SafetensorsFile,
        name: &str,
        input: usize,
        output: usize,
        kernel: usize,
        groups: usize,
        padding: usize,
    ) -> Result<Self> {
        let in_per_group = input / groups;
        // The published weights use PyTorch OIK and ops::conv1d consumes OIK.
        let weight = tensor(
            file,
            &format!("{name}.weight"),
            &[output, in_per_group, kernel],
        )?;
        Ok(Self {
            weight,
            input,
            output,
            kernel,
            groups,
            padding,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut channel_major = vec![0.0; x.len()];
        for row in 0..rows {
            for channel in 0..self.input {
                channel_major[channel * rows + row] = x[row * self.input + channel];
            }
        }
        let output = ops::conv1d(
            &channel_major,
            &self.weight,
            None,
            self.input,
            self.output,
            self.kernel,
            1,
            self.padding,
            1,
            self.groups,
        );
        let output_rows = (rows + 2 * self.padding - self.kernel) + 1;
        let mut time_major = vec![0.0; output.len()];
        for row in 0..output_rows {
            for channel in 0..self.output {
                time_major[row * self.output + channel] = output[channel * output_rows + row];
            }
        }
        time_major
    }
}

impl ConformerConvolution {
    fn load(file: &SafetensorsFile, prefix: &str, width: usize, kernel: usize) -> Result<Self> {
        Ok(Self {
            norm: load_layer_norm(file, &format!("{prefix}.pre_layer_norm"), width)?,
            pointwise_in: Conv1d::load(
                file,
                &format!("{prefix}.pointwise_conv1"),
                width,
                width * 4,
                1,
                1,
                0,
            )?,
            depthwise: Conv1d::load(
                file,
                &format!("{prefix}.depthwise_conv"),
                width * 2,
                width * 2,
                kernel,
                width * 2,
                (kernel - 1) / 2,
            )?,
            batch_norm: load_layer_norm(file, &format!("{prefix}.batch_norm"), width * 2)?,
            pointwise_out: Conv1d::load(
                file,
                &format!("{prefix}.pointwise_conv2"),
                width * 2,
                width,
                1,
                1,
                0,
            )?,
            width,
            kernel,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut hidden = x.to_vec();
        self.norm.apply(&mut hidden, rows);
        let projected = self.pointwise_in.forward(&hidden, rows);
        let mut gated = vec![0.0; rows * self.width * 2];
        for row in 0..rows {
            for channel in 0..self.width * 2 {
                let a = projected[row * self.width * 4 + channel];
                let b = projected[row * self.width * 4 + self.width * 2 + channel];
                gated[row * self.width * 2 + channel] = a / (1.0 + (-b).exp());
            }
        }
        let mut hidden = self.depthwise.forward(&gated, rows);
        self.batch_norm.apply(&mut hidden, rows);
        ops::silu(&mut hidden);
        let update = self.pointwise_out.forward(&hidden, rows);
        x.iter()
            .zip(update)
            .map(|(&residual, update)| residual + update)
            .collect()
    }
}

struct ConformerBlock {
    first_ffn: FeedForward,
    attention: RelativeAttention,
    convolution: ConformerConvolution,
    second_ffn: FeedForward,
    final_norm: LayerNorm,
}

impl ConformerBlock {
    fn load(
        file: &SafetensorsFile,
        index: usize,
        width: usize,
        heads: usize,
        kernel: usize,
    ) -> Result<Self> {
        let prefix = format!("encoder.layer_stack.{index}");
        Ok(Self {
            first_ffn: FeedForward::load(file, &format!("{prefix}.ffn1"), width)?,
            attention: RelativeAttention::load(file, &format!("{prefix}.mhsa"), width, heads)?,
            convolution: ConformerConvolution::load(
                file,
                &format!("{prefix}.conv"),
                width,
                kernel,
            )?,
            second_ffn: FeedForward::load(file, &format!("{prefix}.ffn2"), width)?,
            final_norm: load_layer_norm(file, &format!("{prefix}.layer_norm"), width)?,
        })
    }

    fn forward(
        &self,
        x: &[f32],
        rows: usize,
        pos_emb: &[f32],
        width: usize,
        trace: bool,
    ) -> (Vec<f32>, Option<Vec<Snapshot>>) {
        let (first, mut stages) = self.first_ffn.forward_with_trace(x, rows, trace);
        let mut hidden: Vec<f32> = x
            .iter()
            .zip(first)
            .map(|(&x, y)| 0.5 * x + 0.5 * y)
            .collect();
        if let Some(stages) = &mut stages {
            stages.push(snapshot(&hidden, &[rows, width]));
        }
        let update = self.attention.forward(&hidden, rows, pos_emb);
        for (x, update) in hidden.iter_mut().zip(update) {
            *x += update;
        }
        if let Some(stages) = &mut stages {
            stages.push(snapshot(&hidden, &[rows, width]));
        }
        hidden = self.convolution.forward(&hidden, rows);
        if let Some(stages) = &mut stages {
            stages.push(snapshot(&hidden, &[rows, width]));
        }
        let second = self.second_ffn.forward(&hidden, rows);
        for (x, y) in hidden.iter_mut().zip(second) {
            *x = 0.5 * *x + 0.5 * y;
        }
        if let Some(stages) = &mut stages {
            stages.push(snapshot(&hidden, &[rows, width]));
        }
        self.final_norm.apply(&mut hidden, rows);
        debug_assert_eq!(hidden.len(), rows * width);
        if let Some(stages) = &mut stages {
            stages.push(snapshot(&hidden, &[rows, width]));
        }
        (hidden, stages)
    }
}

impl Encoder {
    pub(super) fn load(file: &SafetensorsFile, config: FireRedAsr2Config) -> Result<Self> {
        let width = config.model_dim;
        let conv1 = Conv2d::load(
            file,
            "encoder.input_preprocessor.conv1",
            1,
            SUBSAMPLE_CHANNELS,
            SUBSAMPLE_KERNEL,
        )?;
        let conv2 = Conv2d::load(
            file,
            "encoder.input_preprocessor.conv2",
            SUBSAMPLE_CHANNELS,
            SUBSAMPLE_CHANNELS,
            SUBSAMPLE_KERNEL,
        )?;
        let mel_width = ((config.input_dim - 1) / 2 - 1) / 2;
        let subsample_projection = load_linear(
            file,
            "encoder.input_preprocessor.out",
            SUBSAMPLE_CHANNELS * mel_width,
            width,
        )?;
        let layers = (0..config.encoder_layers)
            .map(|index| {
                ConformerBlock::load(
                    file,
                    index,
                    width,
                    config.encoder_heads,
                    config.encoder_kernel,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            config,
            conv1,
            conv2,
            subsample_projection,
            layers,
        })
    }

    pub(super) fn forward(
        &self,
        features: &[f32],
        frames: usize,
        trace: bool,
    ) -> Result<EncoderOutput> {
        self.forward_layers(features, frames, trace, self.config.encoder_layers)
    }

    pub(super) fn forward_first_block(
        &self,
        features: &[f32],
        frames: usize,
        trace: bool,
    ) -> Result<EncoderOutput> {
        self.forward_layers(features, frames, trace, 1)
    }

    fn forward_layers(
        &self,
        features: &[f32],
        frames: usize,
        trace: bool,
        layer_limit: usize,
    ) -> Result<EncoderOutput> {
        let config = &self.config;
        if features.len() != frames * config.input_dim || frames == 0 {
            return Err(SpeechError::Input {
                why: "FireRedASR2 FBANK shape does not match the configured 80-bin input".into(),
            });
        }
        let mut padded = vec![0.0; (frames + 6) * config.input_dim];
        padded[..features.len()].copy_from_slice(features);
        let (first, first_rows, first_width) =
            self.conv1.forward(&padded, frames + 6, config.input_dim);
        let (second, second_rows, second_width) =
            self.conv2.forward(&first, first_rows, first_width);
        let rows = second_rows;
        let mut flattened = vec![0.0; rows * SUBSAMPLE_CHANNELS * second_width];
        for row in 0..rows {
            for channel in 0..SUBSAMPLE_CHANNELS {
                let source = channel * rows * second_width + row * second_width;
                let target = row * SUBSAMPLE_CHANNELS * second_width + channel * second_width;
                flattened[target..target + second_width]
                    .copy_from_slice(&second[source..source + second_width]);
            }
        }
        let mut hidden = self.subsample_projection.forward(&flattened, rows);
        let subsampled = trace.then(|| snapshot(&hidden, &[rows, config.model_dim]));
        let pos_emb = relative_positions(rows, config.model_dim);
        let mut first_block = None;
        let mut first_block_stages = None;
        for (index, layer) in self.layers.iter().take(layer_limit).enumerate() {
            let (output, stages) = layer.forward(
                &hidden,
                rows,
                &pos_emb,
                config.model_dim,
                trace && index == 0,
            );
            hidden = output;
            if trace && index == 0 {
                first_block = Some(snapshot(&hidden, &[rows, config.model_dim]));
                first_block_stages = stages;
            }
        }
        let trace = trace.then(|| EncoderTrace {
            subsampled: subsampled.expect("trace capture requested"),
            first_block: first_block.expect("configured encoder has at least one block"),
            first_block_stages: first_block_stages.expect("first block stage trace requested"),
            final_block: snapshot(&hidden, &[rows, config.model_dim]),
        });
        Ok(EncoderOutput {
            values: hidden,
            rows,
            trace,
        })
    }
}

fn relative_positions(rows: usize, width: usize) -> Vec<f32> {
    let count = rows * 2 - 1;
    let mut output = vec![0.0; count * width];
    for index in 0..count {
        let position = (rows - 1) as isize - index as isize;
        for pair in 0..width / 2 {
            let angle = position as f32 * 10_000.0f32.powf(-((2 * pair) as f32) / width as f32);
            output[index * width + 2 * pair] = angle.sin();
            output[index * width + 2 * pair + 1] = angle.cos();
        }
    }
    output
}

/// Reference for the `_rel_shift` flat index; production code uses the closed
/// form it reduces to, and a test pins the two together.
#[cfg(test)]
fn relative_index(rows: usize, query: usize, key: usize) -> Option<usize> {
    let positions = rows * 2 - 1;
    let shifted = query * positions + key + rows;
    let padded_row = shifted / (2 * rows);
    let padded_col = shifted % (2 * rows);
    (padded_col != 0).then_some(padded_row * positions + padded_col - 1)
}

#[cfg(test)]
mod tests {
    use super::{
        dot, ops, raw_tensor_name, relative_index, relative_positions, to_head_major, LayerNorm,
        Linear, RelativeAttention, LAYER_NORM_EPS,
    };

    #[test]
    fn maps_sanitized_sequential_parameter_names_back_to_checkpoint_keys() {
        assert_eq!(
            raw_tensor_name("encoder.layer_stack.0.ffn1.net_1.bias"),
            "encoder.layer_stack.0.ffn1.net.1.bias"
        );
    }

    #[test]
    fn relative_shift_matches_mlx_padding_and_reshape_order() {
        assert_eq!(
            (0..2)
                .flat_map(|query| (0..2).map(move |key| relative_index(2, query, key)))
                .collect::<Vec<_>>(),
            vec![Some(1), Some(2), Some(3), Some(4)]
        );
    }

    struct Rng(u64);

    impl Rng {
        fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n)
                .map(|_| {
                    self.0 ^= self.0 << 13;
                    self.0 ^= self.0 >> 7;
                    self.0 ^= self.0 << 17;
                    ((self.0 >> 8) % 20001) as f32 / 10000.0 * scale - scale
                })
                .collect()
        }

        fn linear(&mut self, width: usize) -> Linear {
            Linear::new(
                self.vec(width * width, 0.9 / (width as f32).sqrt()),
                Some(self.vec(width, 0.2)),
                width,
                width,
            )
        }

        fn norm(&mut self, width: usize) -> LayerNorm {
            let weight = self.vec(width, 0.3).iter().map(|w| w + 1.0).collect();
            LayerNorm::new(weight, Some(self.vec(width, 0.1)), LAYER_NORM_EPS)
        }
    }

    /// The original forward, kept verbatim: it builds the [T, T] content and
    /// [T, 2T-1] position matrices per head and picks entries through
    /// `relative_index`.
    fn reference_forward(
        attention: &RelativeAttention,
        x: &[f32],
        rows: usize,
        pos_emb: &[f32],
    ) -> Vec<f32> {
        let this = attention;
        let mut q_input = x.to_vec();
        let mut k_input = x.to_vec();
        let mut v_input = x.to_vec();
        this.q_norm.apply(&mut q_input, rows);
        this.k_norm.apply(&mut k_input, rows);
        this.v_norm.apply(&mut v_input, rows);
        let q = to_head_major(
            &this.q.forward(&q_input, rows),
            rows,
            this.heads,
            this.head_dim,
        );
        let k = to_head_major(
            &this.k.forward(&k_input, rows),
            rows,
            this.heads,
            this.head_dim,
        );
        let v = to_head_major(
            &this.v.forward(&v_input, rows),
            rows,
            this.heads,
            this.head_dim,
        );
        let pos_rows = rows * 2 - 1;
        let p = to_head_major(
            &this.position.forward(pos_emb, pos_rows),
            pos_rows,
            this.heads,
            this.head_dim,
        );
        let mut context = vec![0.0; rows * this.width];
        let scale = 1.0 / (this.head_dim as f32).sqrt();
        let mut relative = vec![0.0f32; rows * pos_rows];
        let mut matrix_ac = vec![0.0f32; rows * rows];
        let mut scores = vec![0.0f32; rows];

        for head in 0..this.heads {
            let q_base = head * rows * this.head_dim;
            let p_base = head * pos_rows * this.head_dim;
            let k_base = head * rows * this.head_dim;
            let v_base = head * rows * this.head_dim;
            for query in 0..rows {
                let q_row =
                    &q[q_base + query * this.head_dim..q_base + (query + 1) * this.head_dim];
                let bias_u = &this.pos_bias_u[head * this.head_dim..(head + 1) * this.head_dim];
                let bias_v = &this.pos_bias_v[head * this.head_dim..(head + 1) * this.head_dim];
                let q_with_u: Vec<f32> = q_row.iter().zip(bias_u).map(|(&q, &b)| q + b).collect();
                let q_with_v: Vec<f32> = q_row.iter().zip(bias_v).map(|(&q, &b)| q + b).collect();
                for key in 0..rows {
                    let k_row =
                        &k[k_base + key * this.head_dim..k_base + (key + 1) * this.head_dim];
                    matrix_ac[query * rows + key] = dot(&q_with_u, k_row);
                }
                for pos in 0..pos_rows {
                    let p_row =
                        &p[p_base + pos * this.head_dim..p_base + (pos + 1) * this.head_dim];
                    relative[query * pos_rows + pos] = dot(&q_with_v, p_row);
                }
            }
            for query in 0..rows {
                for key in 0..rows {
                    let rel = relative_index(rows, query, key).map_or(0.0, |index| relative[index]);
                    scores[key] = (matrix_ac[query * rows + key] + rel) * scale;
                }
                ops::softmax_row(&mut scores);
                for feature in 0..this.head_dim {
                    let mut sum = 0.0f32;
                    for key in 0..rows {
                        sum += scores[key] * v[v_base + key * this.head_dim + feature];
                    }
                    context[query * this.width + head * this.head_dim + feature] = sum;
                }
            }
        }
        this.output.forward(&context, rows)
    }

    #[test]
    fn relative_shift_selects_the_same_query_and_position_t_minus_1_plus_key_minus_query() {
        for rows in [1usize, 2, 3, 8, 31] {
            for query in 0..rows {
                for key in 0..rows {
                    let want = query * (2 * rows - 1) + (rows - 1 + key - query);
                    assert_eq!(relative_index(rows, query, key), Some(want));
                }
            }
        }
    }

    #[test]
    fn needed_relative_positions_only_matches_full_matrices_bitwise() {
        let (width, heads) = (24usize, 4usize);
        for (seed, rows) in [(7u64, 1usize), (11, 2), (13, 5), (17, 17), (19, 40)] {
            let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
            let attention = RelativeAttention {
                q_norm: rng.norm(width),
                k_norm: rng.norm(width),
                v_norm: rng.norm(width),
                q: rng.linear(width),
                k: rng.linear(width),
                v: rng.linear(width),
                position: rng.linear(width),
                output: rng.linear(width),
                pos_bias_u: rng.vec(width, 0.5),
                pos_bias_v: rng.vec(width, 0.5),
                heads,
                head_dim: width / heads,
                width,
            };
            let x = rng.vec(rows * width, 1.5);
            let pos_emb = relative_positions(rows, width);
            let want = reference_forward(&attention, &x, rows, &pos_emb);
            let got = attention.forward(&x, rows, &pos_emb);
            assert_eq!(got.len(), want.len());
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "rows {rows} element {i}");
            }
        }
    }
}
