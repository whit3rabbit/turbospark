//! The indextts speaker encoder (`BigVGANConditioning.speaker_encoder`).
//!
//! Port of `mlx_audio/tts/models/indextts/ecapa_tdnn/` (ecapa_tdnn.py,
//! tdnn.py, se_res2net.py, asp.py) at the pinned mlx-audio commit. This
//! is a DIFFERENT network from the `mlx_audio/codec/models/ecapa_tdnn`
//! backbone the codec family ports (`../../codec/ecapa_tdnn`): the TDNN
//! blocks reflect-pad before their grouped convs, the Res2Net chain
//! feeds each branch only from the second segment on, every SeRes2Net
//! block carries a 1x1 shortcut when its widths change, and the
//! attentive statistics pooling keeps the global-context mean/std and
//! its own BatchNorm before the 1x1 embedding conv.
//!
//! Layouts: the reference is `NLC`; this port keeps row-major `[T, C]`
//! and transposes inside the conv calls.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::conv::{load_bias, load_mlx_conv_weight, BiasMode, Conv1d};
use crate::codec::wnconv::load_f32_shaped;
use crate::ops;
use crate::{Result, SpeechError};

/// `ECPATDNNArgs`, with the reference dataclass defaults.
#[derive(Debug, Clone)]
pub struct EcapaTdnnArgs {
    pub input_size: usize,
    pub lin_neurons: usize,
    pub channels: Vec<usize>,
    pub kernel_sizes: Vec<usize>,
    pub dilations: Vec<usize>,
    pub attention_channels: usize,
    pub res2net_scale: usize,
    pub se_channels: usize,
    pub global_context: bool,
}

impl EcapaTdnnArgs {
    /// `ECPATDNNArgs(input_size, lin_neurons)` with the list defaults.
    pub fn new(input_size: usize, lin_neurons: usize) -> Self {
        EcapaTdnnArgs {
            input_size,
            lin_neurons,
            channels: vec![512, 512, 512, 512, 1536],
            kernel_sizes: vec![5, 3, 3, 3, 1],
            dilations: vec![1, 2, 3, 4, 1],
            attention_channels: 128,
            res2net_scale: 8,
            se_channels: 128,
            global_context: true,
        }
    }
}

/// Inference-mode BatchNorm (`weight`, `bias`, running stats, eps 1e-5).
struct BatchNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    running_mean: Vec<f32>,
    running_var: Vec<f32>,
}

impl BatchNorm {
    fn load(file: &SafetensorsFile, prefix: &str, ch: usize) -> Result<Self> {
        Ok(BatchNorm {
            weight: load_f32_shaped(file, &format!("{prefix}weight"), &[ch])?,
            bias: load_f32_shaped(file, &format!("{prefix}bias"), &[ch])?,
            running_mean: load_f32_shaped(file, &format!("{prefix}running_mean"), &[ch])?,
            running_var: load_f32_shaped(file, &format!("{prefix}running_var"), &[ch])?,
        })
    }

    /// Row-major `[rows, ch]`.
    fn forward(&self, x: &mut [f32], ch: usize) {
        let rows = x.len() / ch;
        for c in 0..ch {
            let scale = self.weight[c] / (self.running_var[c] + 1e-5).sqrt();
            let shift = self.bias[c] - self.running_mean[c] * scale;
            for row in 0..rows {
                x[row * ch + c] = x[row * ch + c] * scale + shift;
            }
        }
    }
}

/// Reflect-pad `pad` rows (edge sample excluded, mirrored) ahead of and
/// behind the `[T, C]` rows: the reference slices
/// `x[1:pad+1][::-1]` / `x[-(pad+1):-1][::-1]`.
fn reflect_pad(x: &[f32], len: usize, dim: usize, pad: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; (len + 2 * pad) * dim];
    for t in 0..pad {
        let source = pad - t;
        out[t * dim..(t + 1) * dim].copy_from_slice(&x[source * dim..(source + 1) * dim]);
    }
    out[pad * dim..(pad + len) * dim].copy_from_slice(x);
    for t in 0..pad {
        let source = len.checked_sub(2 + t).ok_or_else(|| SpeechError::Input {
            why: "reflect padding needs at least pad + 1 rows".to_string(),
        });
        match source {
            Ok(source) => {
                out[(len + pad + t) * dim..(len + pad + t + 1) * dim]
                    .copy_from_slice(&x[source * dim..(source + 1) * dim]);
            }
            Err(_) => {
                // Fewer rows than the pad: torch reflect would fail; the
                // conditioning windows are always long enough, so keep
                // the edge row as the least-surprising fallback.
                out[(len + pad + t) * dim..(len + pad + t + 1) * dim]
                    .copy_from_slice(&x[(len - 1) * dim..len * dim]);
            }
        }
    }
    out
}

/// The indextts `TDNN`: reflected conv + ReLU + BatchNorm over `[T, C]`.
struct Tdnn {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    dilation: usize,
    groups: usize,
    padding: usize,
    norm: BatchNorm,
}

impl Tdnn {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let padding = ((kernel - 1) * dilation) / 2;
        Ok(Tdnn {
            weight: load_mlx_conv_weight(
                file,
                &format!("{prefix}conv.weight"),
                out_ch,
                kernel,
                in_ch / groups,
            )?,
            bias: load_bias(
                file,
                &format!("{prefix}conv.bias"),
                out_ch,
                BiasMode::Required,
            )?,
            in_ch,
            out_ch,
            kernel,
            dilation,
            groups,
            padding,
            norm: BatchNorm::load(file, &format!("{prefix}norm."), out_ch)?,
        })
    }

    /// `[T, in_ch]` rows -> `[T', out_ch]` rows.
    fn forward(&self, x: &[f32], len: usize) -> Vec<f32> {
        // The reference reflect-pads (time axis, row-major) and then
        // runs the conv with padding=0.
        let padded = if self.padding > 0 {
            reflect_pad(x, len, self.in_ch, self.padding)
        } else {
            x.to_vec()
        };
        let padded_len = len + 2 * self.padding;
        // To channel-major for ops::conv1d.
        let mut channels = vec![0.0f32; padded.len()];
        for t in 0..padded_len {
            for c in 0..self.in_ch {
                channels[c * padded_len + t] = padded[t * self.in_ch + c];
            }
        }
        let conv = ops::conv1d(
            &channels,
            &self.weight,
            self.bias.as_deref(),
            self.in_ch,
            self.out_ch,
            self.kernel,
            1,
            0,
            self.dilation,
            self.groups,
        );
        let out_len = conv.len() / self.out_ch;
        // Back to row-major, through ReLU and the channel BatchNorm.
        let mut rows = vec![0.0f32; conv.len()];
        for t in 0..out_len {
            for c in 0..self.out_ch {
                rows[t * self.out_ch + c] = conv[c * out_len + t];
            }
        }
        for value in rows.iter_mut() {
            *value = value.max(0.0);
        }
        self.norm.forward(&mut rows, self.out_ch);
        rows
    }
}

/// The `Res2Net` branch chain: `scale` channel segments; segment 1 runs
/// alone, segment i >= 2 adds the previous branch output.
struct Res2Net {
    blocks: Vec<Tdnn>,
    scale: usize,
}

impl Res2Net {
    #[allow(clippy::too_many_arguments)]
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        dilation: usize,
        groups: usize,
        scale: usize,
    ) -> Result<Self> {
        let blocks = (0..scale - 1)
            .map(|i| {
                Tdnn::load(
                    file,
                    &format!("{prefix}blocks.{i}."),
                    in_ch / scale,
                    out_ch / scale,
                    kernel,
                    dilation,
                    groups,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Res2Net { blocks, scale })
    }

    /// Row-major `[T, C]` segments over the trailing channel axis.
    fn forward(&self, x: &[f32], len: usize) -> Vec<f32> {
        let dim = x.len() / len;
        let seg = dim / self.scale;
        let mut out = vec![0.0f32; x.len()];
        // Segment 0 rides through unchanged.
        for t in 0..len {
            out[t * dim..t * dim + seg].copy_from_slice(&x[t * dim..t * dim + seg]);
        }
        let mut previous: Option<Vec<f32>> = None;
        for (i, block) in self.blocks.iter().enumerate() {
            let mut input = vec![0.0f32; len * seg];
            for t in 0..len {
                let source = &x[t * dim + (i + 1) * seg..t * dim + (i + 2) * seg];
                input[t * seg..(t + 1) * seg].copy_from_slice(source);
                if let Some(previous) = &previous {
                    for s in 0..seg {
                        input[t * seg + s] += previous[t * seg + s];
                    }
                }
            }
            let branch = block.forward(&input, len);
            for t in 0..len {
                out[t * dim + (i + 1) * seg..t * dim + (i + 2) * seg]
                    .copy_from_slice(&branch[t * seg..(t + 1) * seg]);
            }
            previous = Some(branch);
        }
        out
    }
}

/// The `SE` squeeze-excite: time mean, two 1x1 convs, sigmoid gate.
struct Se {
    conv1_weight: Vec<f32>,
    conv1_bias: Option<Vec<f32>>,
    conv2_weight: Vec<f32>,
    conv2_bias: Option<Vec<f32>>,
    in_ch: usize,
    se_ch: usize,
    out_ch: usize,
}

impl Se {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_ch: usize,
        se_ch: usize,
        out_ch: usize,
    ) -> Result<Self> {
        Ok(Se {
            conv1_weight: load_mlx_conv_weight(
                file,
                &format!("{prefix}conv1.weight"),
                se_ch,
                1,
                in_ch,
            )?,
            conv1_bias: load_bias(
                file,
                &format!("{prefix}conv1.bias"),
                se_ch,
                BiasMode::Required,
            )?,
            conv2_weight: load_mlx_conv_weight(
                file,
                &format!("{prefix}conv2.weight"),
                out_ch,
                1,
                se_ch,
            )?,
            conv2_bias: load_bias(
                file,
                &format!("{prefix}conv2.bias"),
                out_ch,
                BiasMode::Required,
            )?,
            in_ch,
            se_ch,
            out_ch,
        })
    }

    /// `[T, C]` -> scaled `[T, C]` (the single-row context passes no
    /// mask, so the squeeze is the plain time mean). The 1x1 convs run
    /// on the one-frame channel-major view.
    fn forward(&self, x: &[f32], len: usize) -> Vec<f32> {
        let mut s = vec![0.0f32; self.in_ch];
        for t in 0..len {
            for (c, value) in s.iter_mut().enumerate() {
                *value += x[t * self.in_ch + c];
            }
        }
        for value in s.iter_mut() {
            *value /= len as f32;
        }
        let conv1 = ops::conv1d(
            &s,
            &self.conv1_weight,
            self.conv1_bias.as_deref(),
            self.in_ch,
            self.se_ch,
            1,
            1,
            0,
            1,
            1,
        );
        let mut activated = conv1;
        for value in activated.iter_mut() {
            *value = value.max(0.0);
        }
        let mut gate = ops::conv1d(
            &activated,
            &self.conv2_weight,
            self.conv2_bias.as_deref(),
            self.se_ch,
            self.out_ch,
            1,
            1,
            0,
            1,
            1,
        );
        for value in gate.iter_mut() {
            *value = ops::sigmoid(*value);
        }
        let mut out = x.to_vec();
        for t in 0..len {
            for (c, value) in out[t * self.out_ch..(t + 1) * self.out_ch]
                .iter_mut()
                .enumerate()
            {
                *value *= gate[c];
            }
        }
        out
    }
}

/// One `SeRes2Net` block: shortcut (when widths change), then TDNN ->
/// Res2Net -> TDNN -> SE residual add.
struct SeRes2Net {
    shortcut: Option<Conv1d>,
    tdnn1: Tdnn,
    res2net_block: Res2Net,
    tdnn2: Tdnn,
    se_block: Se,
}

impl SeRes2Net {
    #[allow(clippy::too_many_arguments)]
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        scale: usize,
        kernel: usize,
        dilation: usize,
        attention_channels: usize,
    ) -> Result<Self> {
        let shortcut = if in_ch != out_ch {
            Some(Conv1d {
                weight: load_mlx_conv_weight(
                    file,
                    &format!("{prefix}shortcut.weight"),
                    out_ch,
                    1,
                    in_ch,
                )?,
                bias: load_bias(
                    file,
                    &format!("{prefix}shortcut.bias"),
                    out_ch,
                    BiasMode::Required,
                )?,
                in_ch,
                out_ch,
                kernel: 1,
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: 1,
            })
        } else {
            None
        };
        Ok(SeRes2Net {
            shortcut,
            tdnn1: Tdnn::load(file, &format!("{prefix}tdnn1."), in_ch, out_ch, 1, 1, 1)?,
            res2net_block: Res2Net::load(
                file,
                &format!("{prefix}res2net_block."),
                out_ch,
                out_ch,
                kernel,
                dilation,
                1,
                scale,
            )?,
            tdnn2: Tdnn::load(file, &format!("{prefix}tdnn2."), out_ch, out_ch, 1, 1, 1)?,
            se_block: Se::load(
                file,
                &format!("{prefix}se_block."),
                out_ch,
                attention_channels,
                out_ch,
            )?,
        })
    }

    fn forward(&self, x: &[f32], len: usize) -> Vec<f32> {
        // The shortcut 1x1 conv runs on the channel-major view.
        let base = match &self.shortcut {
            Some(conv) => {
                let in_ch = conv.in_ch;
                let mut channels = vec![0.0f32; x.len()];
                for t in 0..len {
                    for c in 0..in_ch {
                        channels[c * len + t] = x[t * in_ch + c];
                    }
                }
                let out = conv.forward(&channels);
                let out_ch = conv.out_ch;
                let mut rows = vec![0.0f32; out.len()];
                for t in 0..len {
                    for c in 0..out_ch {
                        rows[t * out_ch + c] = out[c * len + t];
                    }
                }
                rows
            }
            None => x.to_vec(),
        };
        let inner = self.se_block.forward(
            &self.tdnn2.forward(
                &self
                    .res2net_block
                    .forward(&self.tdnn1.forward(&base, len), len),
                len,
            ),
            len,
        );
        let mut out = base;
        for (value, add) in out.iter_mut().zip(&inner) {
            *value += add;
        }
        out
    }
}

/// The attentive statistics pooling (`AttentiveStatisticsPooling`).
struct AttentiveStatisticsPooling {
    tdnn: Tdnn,
    conv_weight: Vec<f32>,
    conv_bias: Option<Vec<f32>>,
    channels: usize,
    attention_channels: usize,
    global_context: bool,
}

impl AttentiveStatisticsPooling {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        channels: usize,
        attention_channels: usize,
        global_context: bool,
    ) -> Result<Self> {
        let tdnn_in = if global_context {
            channels * 3
        } else {
            channels
        };
        Ok(AttentiveStatisticsPooling {
            tdnn: Tdnn::load(
                file,
                &format!("{prefix}tdnn."),
                tdnn_in,
                attention_channels,
                1,
                1,
                1,
            )?,
            conv_weight: load_mlx_conv_weight(
                file,
                &format!("{prefix}conv.weight"),
                channels,
                1,
                attention_channels,
            )?,
            conv_bias: load_bias(
                file,
                &format!("{prefix}conv.bias"),
                channels,
                BiasMode::Required,
            )?,
            channels,
            attention_channels,
            global_context,
        })
    }

    /// `[T, C]` -> `[2C]` (mean block then std block).
    fn forward(&self, x: &[f32], len: usize) -> Result<Vec<f32>> {
        let eps = 1e-12f32;
        let c = self.channels;
        // The global context (mask is all ones for the single-row path).
        let mut global_mean = vec![0.0f32; c];
        for t in 0..len {
            for (ci, mean) in global_mean.iter_mut().enumerate() {
                *mean += x[t * c + ci];
            }
        }
        for mean in global_mean.iter_mut() {
            *mean /= len as f32;
        }
        let mut global_std = vec![0.0f32; c];
        for t in 0..len {
            for (ci, std) in global_std.iter_mut().enumerate() {
                let diff = x[t * c + ci] - global_mean[ci];
                *std += diff * diff;
            }
        }
        for std in global_std.iter_mut() {
            *std = (*std / len as f32 + eps).sqrt();
        }
        let hidden = if self.global_context {
            let mut attn_input = Vec::with_capacity(len * c * 3);
            for t in 0..len {
                attn_input.extend_from_slice(&x[t * c..(t + 1) * c]);
                attn_input.extend_from_slice(&global_mean);
                attn_input.extend_from_slice(&global_std);
            }
            self.tdnn.forward(&attn_input, len)
        } else {
            self.tdnn.forward(x, len)
        };
        // tanh -> 1x1 conv -> softmax over the time axis per channel.
        let mut attn = hidden;
        for value in attn.iter_mut() {
            *value = value.tanh();
        }
        let mut attn = ops::conv1d(
            &attn,
            &self.conv_weight,
            self.conv_bias.as_deref(),
            self.attention_channels,
            c,
            1,
            1,
            0,
            1,
            1,
        );
        let mut max = vec![f32::NEG_INFINITY; c];
        for t in 0..len {
            for (ci, value) in attn[t * c..(t + 1) * c].iter().enumerate() {
                max[ci] = max[ci].max(*value);
            }
        }
        for t in 0..len {
            for (ci, value) in attn[t * c..(t + 1) * c].iter_mut().enumerate() {
                *value = (*value - max[ci]).exp();
            }
        }
        let mut sum = vec![0.0f32; c];
        for t in 0..len {
            for (ci, value) in attn[t * c..(t + 1) * c].iter().enumerate() {
                sum[ci] += *value;
            }
        }
        // Weighted mean, then the weighted squared deviation from it.
        let mut mean = vec![0.0f32; c];
        for t in 0..len {
            for (ci, value) in mean.iter_mut().enumerate() {
                *value += attn[t * c + ci] / sum[ci] * x[t * c + ci];
            }
        }
        let mut variance = vec![0.0f32; c];
        for t in 0..len {
            for (ci, value) in variance.iter_mut().enumerate() {
                let diff = x[t * c + ci] - mean[ci];
                *value += attn[t * c + ci] / sum[ci] * diff * diff;
            }
        }
        // concat([mean, std], axis=2): the mean block then the std block.
        let mut std = Vec::with_capacity(c);
        for var in &variance {
            std.push((var + eps).sqrt());
        }
        mean.extend_from_slice(&std);
        Ok(mean)
    }
}

#[allow(clippy::large_enum_variant)]
enum Block {
    Tdnn(Tdnn),
    SeRes2Net(SeRes2Net),
}

/// The loaded speaker encoder (`ECPATDNN`).
pub struct SpeakerEncoder {
    blocks: Vec<Block>,
    mfa: Tdnn,
    asp: AttentiveStatisticsPooling,
    asp_bn: BatchNorm,
    fc: Conv1d,
    lin_neurons: usize,
}

impl SpeakerEncoder {
    pub fn load(file: &SafetensorsFile, prefix: &str, args: &EcapaTdnnArgs) -> Result<Self> {
        if args.channels.len() != args.kernel_sizes.len()
            || args.channels.len() != args.dilations.len()
        {
            return Err(SpeechError::BadConfig {
                field: "channels".into(),
                why: "channels, kernel_sizes, and dilations must have equal lengths".into(),
            });
        }
        let mut blocks = Vec::new();
        blocks.push(Block::Tdnn(Tdnn::load(
            file,
            &format!("{prefix}blocks.0."),
            args.input_size,
            args.channels[0],
            args.kernel_sizes[0],
            args.dilations[0],
            1,
        )?));
        for i in 1..args.channels.len() - 1 {
            blocks.push(Block::SeRes2Net(SeRes2Net::load(
                file,
                &format!("{prefix}blocks.{i}."),
                args.channels[i - 1],
                args.channels[i],
                args.res2net_scale,
                args.kernel_sizes[i],
                args.dilations[i],
                args.attention_channels,
            )?));
        }
        let mfa_in = args.channels[args.channels.len() - 2] * (args.channels.len() - 2);
        let mfa = Tdnn::load(
            file,
            &format!("{prefix}mfa."),
            mfa_in,
            args.channels[args.channels.len() - 1],
            args.kernel_sizes[args.kernel_sizes.len() - 1],
            args.dilations[args.dilations.len() - 1],
            1,
        )?;
        let asp = AttentiveStatisticsPooling::load(
            file,
            &format!("{prefix}asp."),
            args.channels[args.channels.len() - 1],
            args.attention_channels,
            args.global_context,
        )?;
        let asp_channels = args.channels[args.channels.len() - 1] * 2;
        let asp_bn = BatchNorm::load(file, &format!("{prefix}asp_bn."), asp_channels)?;
        let fc = Conv1d {
            weight: load_mlx_conv_weight(
                file,
                &format!("{prefix}fc.weight"),
                args.lin_neurons,
                1,
                asp_channels,
            )?,
            bias: load_bias(
                file,
                &format!("{prefix}fc.bias"),
                args.lin_neurons,
                BiasMode::Required,
            )?,
            in_ch: asp_channels,
            out_ch: args.lin_neurons,
            kernel: 1,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
        };
        Ok(SpeakerEncoder {
            blocks,
            mfa,
            asp,
            asp_bn,
            fc,
            lin_neurons: args.lin_neurons,
        })
    }

    /// The mel rows `[T, num_mels]` to the one-row speaker embedding
    /// `[lin_neurons]`.
    pub fn forward(&self, x: &[f32], frames: usize) -> Result<Vec<f32>> {
        let mut collected: Vec<Vec<f32>> = Vec::new();
        let mut h = x.to_vec();
        for block in &self.blocks {
            match block {
                Block::Tdnn(tdnn) => h = tdnn.forward(&h, frames),
                Block::SeRes2Net(block) => {
                    h = block.forward(&h, frames);
                    collected.push(h.clone());
                }
            }
        }
        // Concat the SeRes2Net outputs along the channel axis per frame.
        let mut concat = Vec::with_capacity(h.len() * collected.len());
        for t in 0..frames {
            for part in &collected {
                let dim = part.len() / frames;
                concat.extend_from_slice(&part[t * dim..(t + 1) * dim]);
            }
        }
        let pooled = self.mfa.forward(&concat, frames);
        let pooled = self.asp.forward(&pooled, frames)?;
        let mut pooled = pooled;
        let pooled_len = pooled.len();
        self.asp_bn.forward(&mut pooled, pooled_len);
        let embedded = self.fc.forward(&pooled);
        debug_assert_eq!(embedded.len(), self.lin_neurons);
        Ok(embedded)
    }
}
