//! ECAPA-TDNN speaker-embedding backbone.
//!
//! Reference: `mlx_audio/codec/models/ecapa_tdnn/` (ecapa_tdnn.py,
//! config.py) at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/codec/models/ecapa_tdnn).
//! A TDNN stem, three SE-Res2Net blocks, multi-layer feature
//! aggregation, attentive-statistics pooling (optionally with a global
//! context branch), and a final 1x1 conv to the embedding. BatchNorm
//! layers run in inference mode against the stored running stats.
//!
//! The reference consumes channels-last `(B, T, C)` features; this
//! port keeps channel-major `[C, T]` and converts at the boundaries.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::conv::load_mlx_conv_weight;
use crate::codec::wnconv::load_f32_shaped;
use crate::ops;
use crate::{Result, SpeechError};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// Backbone geometry, one-to-one with the reference dataclass.
#[derive(Debug, Clone)]
pub struct EcapaTdnnConfig {
    pub input_size: usize,
    pub channels: usize,
    pub embed_dim: usize,
    pub kernel_sizes: Vec<usize>,
    pub dilations: Vec<usize>,
    pub attention_channels: usize,
    pub res2net_scale: usize,
    pub se_channels: usize,
    pub global_context: bool,
}

impl EcapaTdnnConfig {
    /// Parses with the dataclass defaults.
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let u =
            |field: &str| -> Result<Option<u64>> { Ok(value.get(field).and_then(|v| v.as_u64())) };
        let usize_vec = |field: &str| -> Result<Option<Vec<usize>>> {
            match value.get(field) {
                None | Some(serde_json::Value::Null) => Ok(None),
                Some(v) => {
                    let arr = v.as_array().ok_or_else(|| SpeechError::BadConfig {
                        field: field.to_string(),
                        why: "expected a list".to_string(),
                    })?;
                    Ok(Some(
                        arr.iter()
                            .map(|x| x.as_u64().map(|n| n as usize))
                            .collect::<Option<Vec<_>>>()
                            .ok_or_else(|| SpeechError::BadConfig {
                                field: field.to_string(),
                                why: "expected a list of integers".to_string(),
                            })?,
                    ))
                }
            }
        };
        Ok(EcapaTdnnConfig {
            input_size: u("input_size")?.unwrap_or(60) as usize,
            channels: u("channels")?.unwrap_or(1024) as usize,
            embed_dim: u("embed_dim")?.unwrap_or(256) as usize,
            kernel_sizes: usize_vec("kernel_sizes")?.unwrap_or_else(|| vec![5, 3, 3, 3, 1]),
            dilations: usize_vec("dilations")?.unwrap_or_else(|| vec![1, 2, 3, 4, 1]),
            attention_channels: u("attention_channels")?.unwrap_or(128) as usize,
            res2net_scale: u("res2net_scale")?.unwrap_or(8) as usize,
            se_channels: u("se_channels")?.unwrap_or(128) as usize,
            global_context: value
                .get("global_context")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        })
    }
}

/// BatchNorm inference state.
struct BatchNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    running_mean: Vec<f32>,
    running_var: Vec<f32>,
}

impl BatchNorm {
    fn load(file: &SafetensorsFile, prefix: &str, ch: usize) -> Result<Self> {
        Ok(BatchNorm {
            weight: load_f32_shaped(file, &format!("{prefix}.weight"), &[ch])?,
            bias: load_f32_shaped(file, &format!("{prefix}.bias"), &[ch])?,
            running_mean: load_f32_shaped(file, &format!("{prefix}.running_mean"), &[ch])?,
            running_var: load_f32_shaped(file, &format!("{prefix}.running_var"), &[ch])?,
        })
    }

    fn forward(&self, x: &mut [f32], ch: usize) {
        let frames = x.len() / ch;
        for c in 0..ch {
            let scale = self.weight[c] / (self.running_var[c] + 1e-5).sqrt();
            let shift = self.bias[c] - self.running_mean[c] * scale;
            for v in &mut x[c * frames..(c + 1) * frames] {
                *v = *v * scale + shift;
            }
        }
    }
}

/// Conv + ReLU + BatchNorm over `[in_ch, T]`.
struct TdnnBlock {
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    dilation: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
    norm: BatchNorm,
}

impl TdnnBlock {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        dilation: usize,
    ) -> Result<Self> {
        let _padding = (kernel - 1) * dilation / 2;
        let weight = load_conv_weight(file, &format!("{prefix}.conv"), out_ch, in_ch, kernel)?;
        let bias = load_f32_shaped(file, &format!("{prefix}.conv.bias"), &[out_ch])?;
        Ok(TdnnBlock {
            in_ch,
            out_ch,
            kernel,
            dilation,
            weight,
            bias,
            norm: BatchNorm::load(file, &format!("{prefix}.norm"), out_ch)?,
        })
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut h = ops::conv1d(
            x,
            &self.weight,
            Some(&self.bias),
            self.in_ch,
            self.out_ch,
            self.kernel,
            1,
            (self.kernel - 1) * self.dilation / 2,
            self.dilation,
            1,
        );
        for v in &mut h {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        self.norm.forward(&mut h, self.out_ch);
        h
    }
}

/// Loads a conv weight stored in the MLX `[out, K, in]` layout into
/// the PyTorch `[out, in, K]` layout the kernels consume.
fn load_conv_weight(
    file: &SafetensorsFile,
    prefix: &str,
    out_ch: usize,
    in_ch: usize,
    kernel: usize,
) -> Result<Vec<f32>> {
    load_mlx_conv_weight(file, &format!("{prefix}.weight"), out_ch, kernel, in_ch)
}

struct Res2NetBlock {
    scale: usize,
    hidden: usize,
    blocks: Vec<TdnnBlock>,
}

impl Res2NetBlock {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        channels: usize,
        kernel: usize,
        dilation: usize,
        scale: usize,
    ) -> Result<Self> {
        if channels % scale != 0 {
            return Err(SpeechError::BadConfig {
                field: "res2net_scale".to_string(),
                why: format!("channels {channels} not divisible by scale {scale}"),
            });
        }
        let hidden = channels / scale;
        let mut blocks = Vec::with_capacity(scale - 1);
        for i in 0..scale - 1 {
            blocks.push(TdnnBlock::load(
                file,
                &format!("{prefix}.blocks.{i}"),
                hidden,
                hidden,
                kernel,
                dilation,
            )?);
        }
        Ok(Res2NetBlock {
            scale,
            hidden,
            blocks,
        })
    }

    fn forward(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let ch = self.hidden * self.scale;
        // Split channels into `scale` chunks.
        let chunk = |i: usize| -> Vec<f32> {
            x[i * self.hidden * frames..(i + 1) * self.hidden * frames].to_vec()
        };
        let mut outs: Vec<Vec<f32>> = Vec::with_capacity(self.scale);
        outs.push(chunk(0));
        for (i, block) in self.blocks.iter().enumerate() {
            let c = chunk(i + 1);
            let inp: Vec<f32> = if i > 0 {
                let prev = outs.last().unwrap();
                c.iter().zip(prev.iter()).map(|(a, b)| a + b).collect()
            } else {
                c
            };
            outs.push(block.forward(&inp));
        }
        let mut out = vec![0.0f32; ch * frames];
        for (i, part) in outs.iter().enumerate() {
            out[i * self.hidden * frames..(i + 1) * self.hidden * frames].copy_from_slice(part);
        }
        out
    }
}

struct SeBlock {
    in_dim: usize,
    bottleneck: usize,
    conv1_w: Vec<f32>,
    conv1_b: Vec<f32>,
    conv2_w: Vec<f32>,
    conv2_b: Vec<f32>,
}

impl SeBlock {
    fn forward(&self, x: &[f32], frames: usize) -> Vec<f32> {
        // s = mean over time -> [in_dim]
        let mut s = vec![0.0f32; self.in_dim];
        for c in 0..self.in_dim {
            let mut acc = 0.0f32;
            for f in 0..frames {
                acc += x[c * frames + f];
            }
            s[c] = acc / frames as f32;
        }
        let mut h = vec![0.0f32; self.bottleneck];
        for o in 0..self.bottleneck {
            let mut acc = self.conv1_b[o];
            let w = &self.conv1_w[o * self.in_dim..(o + 1) * self.in_dim];
            for (c, &v) in s.iter().enumerate() {
                acc += v * w[c];
            }
            h[o] = acc.max(0.0);
        }
        let mut scale = vec![0.0f32; self.in_dim];
        for o in 0..self.in_dim {
            let mut acc = self.conv2_b[o];
            let w = &self.conv2_w[o * self.bottleneck..(o + 1) * self.bottleneck];
            for (c, &v) in h.iter().enumerate() {
                acc += v * w[c];
            }
            scale[o] = 1.0 / (1.0 + (-acc).exp());
        }
        let mut out = x.to_vec();
        for c in 0..self.in_dim {
            for f in 0..frames {
                out[c * frames + f] *= scale[c];
            }
        }
        out
    }
}

struct SeRes2NetBlock {
    tdnn1: TdnnBlock,
    res2net: Res2NetBlock,
    tdnn2: TdnnBlock,
    se: SeBlock,
}

impl SeRes2NetBlock {
    fn forward(&self, x: &[f32], frames: usize) -> Result<Vec<f32>> {
        let out = self.tdnn1.forward(x);
        let out = self.res2net.forward(&out, frames);
        let out = self.tdnn2.forward(&out);
        let out = self.se.forward(&out, frames);
        Ok(out.iter().zip(x).map(|(a, b)| a + b).collect())
    }
}

/// Attentive statistics pooling over time; returns
/// `[2 * channels]` (mean then std).
struct AttentiveStatsPool {
    channels: usize,
    attention_channels: usize,
    global_context: bool,
    tdnn: TdnnBlock,
    conv_w: Vec<f32>,
    conv_b: Vec<f32>,
}

impl AttentiveStatsPool {
    fn forward(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let ch = self.channels;
        // Attentive input: optionally append per-time global mean/std.
        let attn_ch = if self.global_context { ch * 3 } else { ch };
        let mut attn_in = vec![0.0f32; attn_ch * frames];
        if self.global_context {
            let mut mean = vec![0.0f32; ch];
            let mut var = vec![0.0f32; ch];
            for c in 0..ch {
                let mut acc = 0.0f32;
                for f in 0..frames {
                    acc += x[c * frames + f];
                }
                mean[c] = acc / frames as f32;
                let mut v = 0.0f32;
                for f in 0..frames {
                    let d = x[c * frames + f] - mean[c];
                    v += d * d;
                }
                var[c] = v / frames as f32;
            }
            for f in 0..frames {
                for c in 0..ch {
                    attn_in[c * frames + f] = x[c * frames + f];
                    attn_in[(ch + c) * frames + f] = mean[c];
                    attn_in[(2 * ch + c) * frames + f] = (var[c] + 1e-9).sqrt();
                }
            }
        } else {
            attn_in.copy_from_slice(x);
        }
        let attn = self.tdnn.forward(&attn_in);
        // Softmax over time per channel.
        let mut weights = vec![0.0f32; ch * frames];
        for c in 0..ch {
            let mut row = vec![0.0f32; frames];
            // The attention conv is kernel-1: out[c, f] = bias[c] +
            // sum_a attn[a, f] * w[c, a].
            let w = &self.conv_w[c * self.attention_channels..(c + 1) * self.attention_channels];
            for f in 0..frames {
                let mut acc = self.conv_b[c];
                for a in 0..self.attention_channels {
                    acc += attn[a * frames + f] * w[a];
                }
                row[f] = acc;
            }
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for f in 0..frames {
                row[f] = (row[f] - max).exp();
                sum += row[f];
            }
            for f in 0..frames {
                row[f] /= sum;
                weights[c * frames + f] = row[f];
            }
        }
        // Weighted mean and variance over time.
        let mut out = vec![0.0f32; 2 * ch];
        for c in 0..ch {
            let mut mean = 0.0f32;
            for f in 0..frames {
                mean += weights[c * frames + f] * x[c * frames + f];
            }
            let mut v = 0.0f32;
            for f in 0..frames {
                v += weights[c * frames + f] * x[c * frames + f] * x[c * frames + f];
            }
            v -= mean * mean;
            out[c] = mean;
            out[ch + c] = v.max(1e-9).sqrt();
        }
        out
    }
}

/// Loaded ECAPA-TDNN backbone, batch-of-one.
pub struct EcapaTdnn {
    pub config: EcapaTdnnConfig,
    block0: TdnnBlock,
    blocks: Vec<SeRes2NetBlock>,
    mfa: TdnnBlock,
    asp: AttentiveStatsPool,
    asp_bn: BatchNorm,
    fc_w: Vec<f32>,
    fc_b: Vec<f32>,
}

impl EcapaTdnn {
    /// Loads from a parsed config and a safetensors file.
    pub fn load(config: EcapaTdnnConfig, file: &SafetensorsFile) -> Result<EcapaTdnn> {
        let ch = config.channels;
        let block0 = TdnnBlock::load(
            file,
            "block0",
            config.input_size,
            ch,
            config.kernel_sizes[0],
            config.dilations[0],
        )?;
        let mut blocks = Vec::with_capacity(3);
        for i in 1..=3 {
            let prefix = format!("block{i}");
            blocks.push(SeRes2NetBlock {
                tdnn1: TdnnBlock::load(file, &format!("{prefix}.tdnn1"), ch, ch, 1, 1)?,
                res2net: Res2NetBlock::load(
                    file,
                    &format!("{prefix}.res2net_block"),
                    ch,
                    config.kernel_sizes[i],
                    config.dilations[i],
                    config.res2net_scale,
                )?,
                tdnn2: TdnnBlock::load(file, &format!("{prefix}.tdnn2"), ch, ch, 1, 1)?,
                se: SeBlock {
                    in_dim: ch,
                    bottleneck: config.se_channels,
                    conv1_w: load_conv_weight(
                        file,
                        &format!("{prefix}.se_block.conv1"),
                        config.se_channels,
                        ch,
                        1,
                    )?,
                    conv1_b: load_f32_shaped(
                        file,
                        &format!("{prefix}.se_block.conv1.bias"),
                        &[config.se_channels],
                    )?,
                    conv2_w: load_conv_weight(
                        file,
                        &format!("{prefix}.se_block.conv2"),
                        ch,
                        config.se_channels,
                        1,
                    )?,
                    conv2_b: load_f32_shaped(
                        file,
                        &format!("{prefix}.se_block.conv2.bias"),
                        &[ch],
                    )?,
                },
            });
        }
        let mfa = TdnnBlock::load(
            file,
            "mfa",
            ch * 3,
            ch * 3,
            config.kernel_sizes[4],
            config.dilations[4],
        )?;
        let asp = AttentiveStatsPool {
            channels: ch * 3,
            attention_channels: config.attention_channels,
            global_context: config.global_context,
            tdnn: TdnnBlock::load(
                file,
                "asp.tdnn",
                if config.global_context {
                    ch * 9
                } else {
                    ch * 3
                },
                config.attention_channels,
                1,
                1,
            )?,
            conv_w: load_conv_weight(file, "asp.conv", ch * 3, config.attention_channels, 1)?,
            conv_b: load_f32_shaped(file, "asp.conv.bias", &[ch * 3])?,
        };
        let asp_bn = BatchNorm::load(file, "asp_bn", ch * 6)?;
        let fc_w = load_conv_weight(file, "fc", config.embed_dim, ch * 6, 1)?;
        let fc_b = load_f32_shaped(file, "fc.bias", &[config.embed_dim])?;
        Ok(EcapaTdnn {
            config,
            block0,
            blocks,
            mfa,
            asp,
            asp_bn,
            fc_w,
            fc_b,
        })
    }

    /// Opens a checkpoint directory with `config.json` plus
    /// `model.safetensors`.
    pub fn open(dir: &std::path::Path) -> Result<EcapaTdnn> {
        let config_text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| {
            SpeechError::BadConfig {
                field: "config.json".to_string(),
                why: e.to_string(),
            }
        })?;
        let value: serde_json::Value =
            serde_json::from_str(&config_text).map_err(|e| SpeechError::BadConfig {
                field: "config.json".to_string(),
                why: e.to_string(),
            })?;
        let config = EcapaTdnnConfig::from_json(&value)?;
        let file = SafetensorsFile::open(&dir.join("model.safetensors"))?;
        EcapaTdnn::load(config, &file)
    }

    /// Embeds features `[input_size, frames]` into
    /// `[2 * 3 * channels]` pooled stats then the `embed_dim` vector.
    pub fn forward(&self, features: &[f32], _frames: usize) -> Result<Vec<f32>> {
        let out = self.block0.forward(features);
        let frames = out.len() / self.config.channels;
        let mut xl = Vec::with_capacity(3);
        let mut out = out;
        for block in &self.blocks {
            out = block.forward(&out, frames)?;
            xl.push(out.clone());
        }
        // Concatenate channel-wise.
        let mut cat = vec![0.0f32; 3 * self.config.channels * frames];
        for (i, part) in xl.iter().enumerate() {
            let offset = i * self.config.channels * frames;
            cat[offset..offset + self.config.channels * frames].copy_from_slice(part);
        }
        let out = self.mfa.forward(&cat);
        let pooled = self.asp.forward(&out, frames);
        let mut pooled_bn = pooled;
        self.asp_bn
            .forward(&mut pooled_bn, self.config.channels * 6);
        // fc: kernel-1 conv on a single frame.
        let mut embed = vec![0.0f32; self.config.embed_dim];
        for o in 0..self.config.embed_dim {
            let mut acc = self.fc_b[o];
            let w = &self.fc_w[o * self.config.channels * 6..(o + 1) * self.config.channels * 6];
            for (c, &v) in pooled_bn.iter().enumerate() {
                acc += v * w[c];
            }
            embed[o] = acc;
        }
        Ok(embed)
    }
}

#[cfg(test)]
mod tests;
