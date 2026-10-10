//! The indextts `BigVGANConditioning` vocoder.
//!
//! Port of `mlx_audio/tts/models/indextts/bigvgan.py` over the shared
//! BigVGAN core (`mlx_audio/codec/models/bigvgan/`) at the pinned
//! mlx-audio commit. The conditioning variant replaces the mel-input
//! `conv_pre` with a latent-input one (`gpt_dim` channels), adds the
//! ECPA-TDNN speaker embedding ([`super::speaker`]) through a 1x1
//! condition conv before the resblock stack and, when configured,
//! through per-stage 1x1 convs after each upsample, and keeps the
//! anti-aliased AMPBlock/resblock/Activation1d/conv_post core of the
//! codec port (the kaiser filters ride the checkpoint, so the family
//! reproduces the codec path without re-deriving numpy's window).
//!
//! `codec/bigvgan` keeps its internals private, so the AMPBlock and
//! Activation1d copies here are local to this family; both read the
//! same checkpoint tensors the codec port documents.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::conv::{load_bias, BiasMode, Conv1d};
use crate::codec::wnconv::{load_f32_shaped, WnConv1d, WnConvTranspose1d};
use crate::ops;
use crate::{Result, SpeechError};

/// `BigVGANConditioningConfig`: the base BigVGAN vocoder geometry plus
/// the conditioning widths.
#[derive(Debug, Clone)]
pub struct BigvganConditioningConfig {
    pub num_mels: usize,
    pub upsample_rates: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub upsample_initial_channel: usize,
    pub resblock: String,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub activation: String,
    pub snake_logscale: bool,
    pub use_bias_at_final: bool,
    pub use_tanh_at_final: bool,
    pub gpt_dim: usize,
    pub speaker_embedding_dim: usize,
    pub cond_d_vector_in_each_upsampling_layer: bool,
}

impl BigvganConditioningConfig {
    /// Parses with the reference dataclass defaults.
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
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
        let nested = |field: &str| -> Result<Option<Vec<Vec<usize>>>> {
            match value.get(field) {
                None | Some(serde_json::Value::Null) => Ok(None),
                Some(v) => {
                    let arr = v.as_array().ok_or_else(|| SpeechError::BadConfig {
                        field: field.to_string(),
                        why: "expected a list of lists".to_string(),
                    })?;
                    Ok(Some(
                        arr.iter()
                            .map(|row| {
                                row.as_array()
                                    .map(|r| {
                                        r.iter()
                                            .filter_map(|x| x.as_u64())
                                            .map(|n| n as usize)
                                            .collect()
                                    })
                                    .ok_or_else(|| SpeechError::BadConfig {
                                        field: field.to_string(),
                                        why: "expected a list of lists".to_string(),
                                    })
                            })
                            .collect::<Result<Vec<_>>>()?,
                    ))
                }
            }
        };
        let require = |field: &str| -> Result<u64> {
            value
                .get(field)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| SpeechError::BadConfig {
                    field: field.to_string(),
                    why: "required".to_string(),
                })
        };
        Ok(BigvganConditioningConfig {
            num_mels: require("num_mels")? as usize,
            upsample_rates: usize_vec("upsample_rates")?.ok_or_else(|| SpeechError::BadConfig {
                field: "upsample_rates".to_string(),
                why: "required".to_string(),
            })?,
            upsample_kernel_sizes: usize_vec("upsample_kernel_sizes")?.ok_or_else(|| {
                SpeechError::BadConfig {
                    field: "upsample_kernel_sizes".to_string(),
                    why: "required".to_string(),
                }
            })?,
            upsample_initial_channel: require("upsample_initial_channel")? as usize,
            resblock: value
                .get("resblock")
                .and_then(|v| v.as_str())
                .unwrap_or("1")
                .to_string(),
            resblock_kernel_sizes: usize_vec("resblock_kernel_sizes")?.unwrap_or_default(),
            resblock_dilation_sizes: nested("resblock_dilation_sizes")?.unwrap_or_default(),
            activation: value
                .get("activation")
                .and_then(|v| v.as_str())
                .unwrap_or("snakebeta")
                .to_string(),
            snake_logscale: value
                .get("snake_logscale")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            use_bias_at_final: value
                .get("use_bias_at_final")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            use_tanh_at_final: value
                .get("use_tanh_at_final")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            gpt_dim: value.get("gpt_dim").and_then(|v| v.as_u64()).unwrap_or(1) as usize,
            speaker_embedding_dim: value
                .get("speaker_embedding_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as usize,
            cond_d_vector_in_each_upsampling_layer: value
                .get("cond_d_vector_in_each_upsampling_layer")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
        })
    }
}

/// The anti-aliased `Activation1d` (2x kaiser upsample, snake-beta, 2x
/// lowpass downsample) with the checkpoint-stored filters.
struct Activation1d {
    alpha: Vec<f32>,
    beta: Vec<f32>,
    channels: usize,
    up_filter: Vec<f32>,
    down_filter: Vec<f32>,
    kernel: usize,
}

impl Activation1d {
    fn load(file: &SafetensorsFile, prefix: &str, channels: usize) -> Result<Self> {
        let alpha_raw = load_f32_shaped(file, &format!("{prefix}act.alpha"), &[channels])?;
        let beta_raw = load_f32_shaped(file, &format!("{prefix}act.beta"), &[channels])?;
        Ok(Activation1d {
            alpha: alpha_raw.iter().map(|v| v.exp()).collect(),
            beta: beta_raw.iter().map(|v| v.exp()).collect(),
            channels,
            up_filter: load_f32_shaped(file, &format!("{prefix}upsample.filter"), &[1, 12, 1])?,
            down_filter: load_f32_shaped(
                file,
                &format!("{prefix}downsample.lowpass.filter"),
                &[1, 12, 1],
            )?,
            kernel: 12,
        })
    }

    fn forward(&self, x: &mut Vec<f32>, frames: usize) {
        let ch = self.channels;
        let up = self.upsample(x, frames);
        let up_frames = up.len() / ch;
        let mut act = up;
        ops::snake_beta(&mut act, &self.alpha, &self.beta, ch, up_frames);
        let down = self.downsample(&act, up_frames);
        *x = down;
    }

    fn upsample(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let ch = self.channels;
        let ratio = 2usize;
        let k = self.kernel;
        let pad = k / ratio - 1;
        // The reference uses floor division for both pads (11 // 2 = 5,
        // not div_ceil's 6); keep the truncating operators.
        #[allow(clippy::manual_div_ceil)]
        let pad_left = pad * ratio + (k - ratio) / 2;
        #[allow(clippy::manual_div_ceil)]
        let pad_right = pad * ratio + (k - ratio + 1) / 2;
        let padded_len = frames + 2 * pad;
        let mut padded = vec![0.0f32; ch * padded_len];
        for c in 0..ch {
            for p in 0..pad {
                padded[c * padded_len + p] = x[c * frames];
                padded[c * padded_len + padded_len - 1 - p] = x[c * frames + frames - 1];
            }
            padded[c * padded_len + pad..c * padded_len + pad + frames]
                .copy_from_slice(&x[c * frames..(c + 1) * frames]);
        }
        let mut weight = vec![0.0f32; ch * k];
        for c in 0..ch {
            for kk in 0..k {
                weight[c * k + kk] = self.up_filter[kk];
            }
        }
        let mut up = ops::conv_transpose1d(&padded, &weight, None, ch, ch, k, ratio, 0, 0, ch);
        for v in &mut up {
            *v *= ratio as f32;
        }
        let out_frames = up.len() / ch;
        let kept = out_frames
            .saturating_sub(pad_right)
            .saturating_sub(pad_left);
        let mut out = vec![0.0f32; ch * kept];
        for c in 0..ch {
            out[c * kept..(c + 1) * kept]
                .copy_from_slice(&up[c * out_frames + pad_left..c * out_frames + pad_left + kept]);
        }
        out
    }

    fn downsample(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let ch = self.channels;
        let ratio = 2usize;
        let k = self.kernel;
        let even = k % 2 == 0;
        let pad_left = k / 2 - usize::from(even);
        let pad_right = k / 2;
        let padded_len = frames + pad_left + pad_right;
        let mut padded = vec![0.0f32; ch * padded_len];
        for c in 0..ch {
            for p in 0..pad_left {
                padded[c * padded_len + p] = x[c * frames];
            }
            for p in 0..pad_right {
                padded[c * padded_len + pad_left + frames + p] = x[c * frames + frames - 1];
            }
            padded[c * padded_len + pad_left..c * padded_len + pad_left + frames]
                .copy_from_slice(&x[c * frames..(c + 1) * frames]);
        }
        let mut weight = vec![0.0f32; ch * k];
        for c in 0..ch {
            for kk in 0..k {
                weight[c * k + kk] = self.down_filter[kk];
            }
        }
        ops::conv1d(&padded, &weight, None, ch, ch, k, ratio, 0, 1, ch)
    }
}

/// One AMPBlock1: three (conv1, conv2) pairs with anti-aliased
/// activations around each conv and a residual add per pair.
struct AmpBlock {
    convs1: Vec<WnConv1d>,
    convs2: Vec<WnConv1d>,
    activations: Vec<Activation1d>,
}

impl AmpBlock {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        channels: usize,
        kernel: usize,
        dilations: &[usize],
    ) -> Result<Self> {
        let mut convs1 = Vec::with_capacity(dilations.len());
        let mut convs2 = Vec::with_capacity(dilations.len());
        for (d_idx, &dilation) in dilations.iter().enumerate() {
            convs1.push(WnConv1d::load(
                file,
                &format!("{prefix}convs1.{d_idx}"),
                channels,
                channels,
                kernel,
                1,
                ((kernel - 1) * dilation) / 2,
                dilation,
                1,
            )?);
            convs2.push(WnConv1d::load(
                file,
                &format!("{prefix}convs2.{d_idx}"),
                channels,
                channels,
                kernel,
                1,
                (kernel - 1) / 2,
                1,
                1,
            )?);
        }
        let mut activations = Vec::with_capacity(dilations.len() * 2);
        for a in 0..dilations.len() * 2 {
            activations.push(Activation1d::load(
                file,
                &format!("{prefix}activations.{a}."),
                channels,
            )?);
        }
        Ok(AmpBlock {
            convs1,
            convs2,
            activations,
        })
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut h = x.to_vec();
        let ch = self.convs1[0].in_ch;
        for i in 0..self.convs1.len() {
            let frames = h.len() / ch;
            let mut a1 = h.clone();
            self.activations[i * 2].forward(&mut a1, frames);
            let c1 = self.convs1[i].forward(&a1);
            let c1_frames = c1.len() / ch;
            let mut a2 = c1;
            self.activations[i * 2 + 1].forward(&mut a2, c1_frames);
            let c2 = self.convs2[i].forward(&a2);
            h.truncate(c2.len());
            for (a, b) in h.iter_mut().zip(&c2) {
                *a += b;
            }
        }
        h
    }
}

/// The loaded conditioning vocoder, batch of one, channel-major
/// `[C, T]` internals like the codec port.
pub struct BigvganConditioning {
    pub config: BigvganConditioningConfig,
    conv_pre: WnConv1d,
    ups: Vec<WnConvTranspose1d>,
    resblocks: Vec<Vec<AmpBlock>>,
    activation_post: Activation1d,
    conv_post: WnConv1d,
    cond_layer: Conv1d,
    /// Per upsample stage, its 1x1 condition conv.
    conds: Vec<Conv1d>,
}

impl BigvganConditioning {
    /// Loads from the vocoder block of an indextts checkpoint file,
    /// where every tensor carries the `bigvgan.` prefix.
    pub fn load(
        file: &SafetensorsFile,
        config: BigvganConditioningConfig,
        prefix: &str,
    ) -> Result<Self> {
        if config.activation != "snakebeta" {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "activation {:?} is refused: the pinned reference broadcasts \
                     snake alpha over time on channels-last activations, which only \
                     type-checks when channels == frames; deployed checkpoints use \
                     snakebeta",
                    config.activation
                ),
            });
        }
        if config.resblock != "1" {
            return Err(SpeechError::Unsupported {
                why: format!("resblock {:?} not ported", config.resblock),
            });
        }
        let conv_pre = WnConv1d::load(
            file,
            &format!("{prefix}conv_pre"),
            config.gpt_dim,
            config.upsample_initial_channel,
            7,
            1,
            3,
            1,
            1,
        )?;
        let mut ups = Vec::with_capacity(config.upsample_rates.len());
        let mut resblocks = Vec::new();
        for (i, (&rate, &kernel)) in config
            .upsample_rates
            .iter()
            .zip(&config.upsample_kernel_sizes)
            .enumerate()
        {
            let in_ch = config.upsample_initial_channel >> i;
            let out_ch = config.upsample_initial_channel >> (i + 1);
            ups.push(WnConvTranspose1d::load_out_first(
                file,
                &format!("{prefix}ups.{i}.0"),
                in_ch,
                out_ch,
                kernel,
                rate,
                (kernel - rate) / 2,
            )?);
            let mut stage_blocks = Vec::new();
            for (j, (&k, d)) in config
                .resblock_kernel_sizes
                .iter()
                .zip(&config.resblock_dilation_sizes)
                .enumerate()
            {
                let block = i * config.resblock_kernel_sizes.len() + j;
                stage_blocks.push(AmpBlock::load(
                    file,
                    &format!("{prefix}resblocks.{block}."),
                    out_ch,
                    k,
                    d,
                )?);
            }
            resblocks.push(stage_blocks);
        }
        let last_ch = config.upsample_initial_channel >> config.upsample_rates.len();
        let activation_post =
            Activation1d::load(file, &format!("{prefix}activation_post."), last_ch)?;
        let conv_post = WnConv1d::load(
            file,
            &format!("{prefix}conv_post"),
            last_ch,
            1,
            7,
            1,
            3,
            1,
            1,
        )?;
        let cond_layer = Conv1d {
            weight: crate::codec::conv::load_mlx_conv_weight(
                file,
                &format!("{prefix}cond_layer.weight"),
                config.upsample_initial_channel,
                1,
                config.speaker_embedding_dim,
            )?,
            bias: load_bias(
                file,
                &format!("{prefix}cond_layer.bias"),
                config.upsample_initial_channel,
                BiasMode::Required,
            )?,
            in_ch: config.speaker_embedding_dim,
            out_ch: config.upsample_initial_channel,
            kernel: 1,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
        };
        let mut conds = Vec::new();
        if config.cond_d_vector_in_each_upsampling_layer {
            for i in 0..config.upsample_rates.len() {
                let out_ch = config.upsample_initial_channel / (1 << (i + 1));
                conds.push(Conv1d {
                    weight: crate::codec::conv::load_mlx_conv_weight(
                        file,
                        &format!("{prefix}conds.{i}.weight"),
                        out_ch,
                        1,
                        config.speaker_embedding_dim,
                    )?,
                    bias: load_bias(
                        file,
                        &format!("{prefix}conds.{i}.bias"),
                        out_ch,
                        BiasMode::Required,
                    )?,
                    in_ch: config.speaker_embedding_dim,
                    out_ch,
                    kernel: 1,
                    stride: 1,
                    padding: 0,
                    dilation: 1,
                    groups: 1,
                });
            }
        }
        Ok(BigvganConditioning {
            config,
            conv_pre,
            ups,
            resblocks,
            activation_post,
            conv_post,
            cond_layer,
            conds,
        })
    }

    /// The latent rows `[S, gpt_dim]` plus the speaker embedding
    /// `[speaker_dim]` to mono samples.
    pub fn forward(&self, latents: &[f32], steps: usize, speaker: &[f32]) -> Result<Vec<f32>> {
        if self.config.upsample_kernel_sizes.len() != self.config.upsample_rates.len() {
            return Err(SpeechError::BadConfig {
                field: "upsample_kernel_sizes".to_string(),
                why: "must match upsample_rates length".to_string(),
            });
        }
        if conds_is_mismatched(&self.conds, self.config.upsample_rates.len()) {
            return Err(SpeechError::BadConfig {
                field: "conds".to_string(),
                why: "per-stage condition conv count must match upsample_rates".to_string(),
            });
        }
        // The reference transposes NLC activations in; the port works
        // channel-major throughout, so the input arrives as [gpt_dim, S]
        // straight from the caller's transpose.
        let mut h = transpose_rows(latents, steps, self.config.gpt_dim);
        h = self.conv_pre.forward(&h);
        add_speaker(&mut h, &self.cond_layer.forward(speaker));
        let num_kernels = self.config.resblock_kernel_sizes.len();
        for (i, up) in self.ups.iter().enumerate() {
            h = up.forward(&h);
            let ch = up.out_ch;
            let frames_out = h.len() / ch;
            if let Some(cond) = self.conds.get(i) {
                add_speaker(&mut h, &cond.forward(speaker));
            }
            let mut sum = vec![0.0f32; ch * frames_out];
            for block in &self.resblocks[i] {
                let b = block.forward(&h);
                for (s, v) in sum.iter_mut().zip(&b) {
                    *s += v;
                }
            }
            let n = num_kernels as f32;
            for v in &mut sum {
                *v /= n;
            }
            h = sum;
        }
        let last_ch = self.conv_post.in_ch;
        let frames_out = h.len() / last_ch;
        self.activation_post.forward(&mut h, frames_out);
        let mut out = self.conv_post.forward(&h);
        for v in &mut out {
            if self.config.use_tanh_at_final {
                *v = v.tanh();
            } else {
                *v = v.clamp(-1.0, 1.0);
            }
        }
        Ok(out)
    }
}

fn conds_is_mismatched(conds: &[Conv1d], stages: usize) -> bool {
    !conds.is_empty() && conds.len() != stages
}

/// Adds a per-channel speaker vector to channel-major activations
/// (the reference `[B, 1, C]` broadcast over time).
fn add_speaker(h: &mut [f32], speaker: &[f32]) {
    let ch = speaker.len();
    let frames = h.len() / ch;
    for t in 0..frames {
        for (c, value) in speaker.iter().enumerate() {
            h[c * frames + t] += *value;
        }
    }
}

/// `[T, C]` rows to channel-major `[C, T]`.
fn transpose_rows(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for t in 0..rows {
        for c in 0..cols {
            out[c * rows + t] = x[t * cols + c];
        }
    }
    out
}
