//! BigVGAN: high-fidelity NN vocoder with anti-aliased snake-beta
//! activations (HiFi-GAN lineage with AMPBlocks).
//!
//! Reference: `mlx_audio/codec/models/bigvgan/` (bigvgan.py, amp.py,
//! activation.py, conv.py, resample.py) at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/codec/models/bigvgan).
//! Weight-norm convs, transposed-conv upsampling, multi-receptive-field
//! fusion (sum of resblock outputs divided by the count), and the
//! Activation1d anti-aliased activation: a 2x kaiser-sinc upsample,
//! snake-beta, and a 2x lowpass downsample. The kaiser filters are
//! stored in the checkpoint (`activation_post.upsample.filter`,
//! `activation_post.downsample.lowpass.filter`), so the loader reads
//! them rather than recomputing numpy's kaiser window.
//!
//! Deviations from the pinned reference (see the family README): the
//! `snake` activation is refused at load. The reference broadcasts its
//! alpha over the time axis of the channels-last activations, which
//! only type-checks when channels equals frames; deployed BigVGAN
//! checkpoints use `snakebeta`, whose broadcast is correct.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::wnconv::{load_f32_shaped, WnConv1d, WnConvTranspose1d};
use crate::ops;
use crate::{Result, SpeechError};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// Vocoder geometry, one-to-one with the reference dataclass.
#[derive(Debug, Clone)]
pub struct BigvganConfig {
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
}

impl BigvganConfig {
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
        Ok(BigvganConfig {
            num_mels: u("num_mels")?.ok_or_else(|| SpeechError::BadConfig {
                field: "num_mels".to_string(),
                why: "required".to_string(),
            })? as usize,
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
            upsample_initial_channel: u("upsample_initial_channel")?.ok_or_else(|| {
                SpeechError::BadConfig {
                    field: "upsample_initial_channel".to_string(),
                    why: "required".to_string(),
                }
            })? as usize,
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
        })
    }
}

/// Anti-aliased snake-beta: 2x kaiser upsample, snake-beta, 2x
/// lowpass. Ratios and kernel sizes are the reference defaults
/// (`up_ratio = down_ratio = 2`, kernel 12).
struct Activation1d {
    alpha: Vec<f32>,
    beta: Vec<f32>,
    channels: usize,
    /// `[K]` filter taps (checkpoint stores `[1, K, 1]`).
    up_filter: Vec<f32>,
    down_filter: Vec<f32>,
    kernel: usize,
}

impl Activation1d {
    fn forward(&self, x: &mut Vec<f32>, frames: usize) {
        let ch = self.channels;
        let up = self.upsample(x, frames);
        let up_frames = up.len() / ch;
        // snake-beta per channel.
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
        // Edge (replicate) pad on both sides.
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
        // Grouped transposed conv, PyTorch weight [C, 1, K].
        let mut weight = vec![0.0f32; ch * k];
        for c in 0..ch {
            for kk in 0..k {
                weight[c * k + kk] = self.up_filter[kk];
            }
        }
        let mut up = ops::conv_transpose1d(&padded, &weight, None, ch, ch, k, ratio, 0, 0, ch);
        // The reference scales by the ratio to compensate the
        // zero-inserted samples.
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
            h = h.iter().zip(c2.iter()).map(|(a, b)| a + b).collect();
        }
        h
    }
}

/// Loaded BigVGAN vocoder, batch-of-one. Input mels `[num_mels, seq]`,
/// output mono `[1, seq * prod(upsample_rates)]`.
pub struct Bigvgan {
    pub config: BigvganConfig,
    conv_pre: WnConv1d,
    ups: Vec<WnConvTranspose1d>,
    /// Per upsample stage, its resblocks (the stage output is the mean
    /// over them).
    resblocks: Vec<Vec<AmpBlock>>,
    activation_post: Activation1d,
    conv_post: WnConv1d,
}

impl Bigvgan {
    /// Opens a checkpoint directory with `config.json` plus
    /// `model.safetensors`.
    pub fn open(dir: &std::path::Path) -> Result<Bigvgan> {
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
        let config = BigvganConfig::from_json(&value)?;
        let file = SafetensorsFile::open(&dir.join("model.safetensors"))?;
        Bigvgan::load(config, &file)
    }

    /// Loads from a parsed config and a safetensors file.
    pub fn load(config: BigvganConfig, file: &SafetensorsFile) -> Result<Bigvgan> {
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
            "conv_pre",
            config.num_mels,
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
                &format!("ups.{i}.0"),
                in_ch,
                out_ch,
                kernel,
                rate,
                (kernel - rate) / 2,
            )?);
            // One AMPBlock per (kernel, dilation) pair per stage; the
            // stage output is the mean over its blocks.
            let mut stage_blocks = Vec::new();
            for (j, (&k, d)) in config
                .resblock_kernel_sizes
                .iter()
                .zip(&config.resblock_dilation_sizes)
                .enumerate()
            {
                let _ = j;
                let mut convs1 = Vec::new();
                let mut convs2 = Vec::new();
                for (d_idx, &dilation) in d.iter().enumerate() {
                    convs1.push(WnConv1d::load(
                        file,
                        &format!("resblocks.{i}.convs1.{d_idx}"),
                        out_ch,
                        out_ch,
                        k,
                        1,
                        ((k - 1) * dilation) / 2,
                        dilation,
                        1,
                    )?);
                    convs2.push(WnConv1d::load(
                        file,
                        &format!("resblocks.{i}.convs2.{d_idx}"),
                        out_ch,
                        out_ch,
                        k,
                        1,
                        (k - 1) / 2,
                        1,
                        1,
                    )?);
                }
                let alpha_raw = load_f32_shaped(
                    file,
                    &format!("resblocks.{i}.activations.0.act.alpha"),
                    &[out_ch],
                )?;
                let beta_raw = load_f32_shaped(
                    file,
                    &format!("resblocks.{i}.activations.0.act.beta"),
                    &[out_ch],
                )?;
                let mut activations = Vec::with_capacity(d.len() * 2);
                for a in 0..d.len() * 2 {
                    activations.push(Activation1d {
                        alpha: alpha_raw.iter().map(|v| v.exp()).collect(),
                        beta: beta_raw.iter().map(|v| v.exp()).collect(),
                        channels: out_ch,
                        up_filter: load_f32_shaped(
                            file,
                            &format!("resblocks.{i}.activations.{a}.upsample.filter"),
                            &[1, 12, 1],
                        )?,
                        down_filter: load_f32_shaped(
                            file,
                            &format!("resblocks.{i}.activations.{a}.downsample.lowpass.filter"),
                            &[1, 12, 1],
                        )?,
                        kernel: 12,
                    });
                }
                stage_blocks.push(AmpBlock {
                    convs1,
                    convs2,
                    activations,
                });
            }
            resblocks.push(stage_blocks);
        }
        let last_ch = config.upsample_initial_channel >> config.upsample_rates.len();
        let alpha_raw = load_f32_shaped(file, "activation_post.act.alpha", &[last_ch])?;
        let beta_raw = load_f32_shaped(file, "activation_post.act.beta", &[last_ch])?;
        let activation_post = Activation1d {
            alpha: alpha_raw.iter().map(|v| v.exp()).collect(),
            beta: beta_raw.iter().map(|v| v.exp()).collect(),
            channels: last_ch,
            up_filter: load_f32_shaped(file, "activation_post.upsample.filter", &[1, 12, 1])?,
            down_filter: load_f32_shaped(
                file,
                "activation_post.downsample.lowpass.filter",
                &[1, 12, 1],
            )?,
            kernel: 12,
        };
        let conv_post = WnConv1d::load(file, "conv_post", last_ch, 1, 7, 1, 3, 1, 1)?;
        Ok(Bigvgan {
            config,
            conv_pre,
            ups,
            resblocks,
            activation_post,
            conv_post,
        })
    }

    /// Vocodes mels `[num_mels, seq]` to mono samples.
    pub fn forward(&self, mels: &[f32], _frames: usize) -> Result<Vec<f32>> {
        if self.config.upsample_kernel_sizes.len() != self.config.upsample_rates.len() {
            return Err(SpeechError::BadConfig {
                field: "upsample_kernel_sizes".to_string(),
                why: "must match upsample_rates length".to_string(),
            });
        }
        let mut h = self.conv_pre.forward(mels);
        let num_kernels = self.config.resblock_kernel_sizes.len();
        for (i, up) in self.ups.iter().enumerate() {
            h = up.forward(&h);
            let ch = up.out_ch;
            let frames_out = h.len() / ch;
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
        let out = self.conv_post.forward(&h);
        let mut out = out;
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

#[cfg(test)]
mod tests;
