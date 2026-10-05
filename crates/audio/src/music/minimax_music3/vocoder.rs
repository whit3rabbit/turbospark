//! DAC-style stereo Flow-VAE decoder.
//!
//! Reference: `mlx_audio/music/models/minimax_music3/vocoder.py`. The
//! `in_channels` latent is treated as two channel groups (left and
//! right) that share the decoder weights; each is projected, run
//! through Snake-activated residual upsampling blocks, and tanh-bounded
//! to a mono waveform. Everything stays channel-major `[C, L]`.
//!
//! The Snake here is `x + sin(alpha * x)^2 / (alpha + 1e-9)`, which is
//! the vocoder's own constant; the shared `ops::snake` divides by the
//! bare alpha used by other families, so the kernel lives here.

use crate::Result;
use crate::SpeechError;

use super::conv::{ConvSpec, MlxConv1d, MlxConvTranspose1d};
use super::weights::WeightStore;

pub(crate) struct VocoderDims {
    pub latent_channels: usize,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub upsampling_ratios: Vec<usize>,
}

struct ResidualUnit {
    snake1_alpha: Vec<f32>,
    conv1: MlxConv1d,
    snake2_alpha: Vec<f32>,
    conv2: MlxConv1d,
}

struct VocoderBlock {
    snake1_alpha: Vec<f32>,
    conv_t1: MlxConvTranspose1d,
    res_units: Vec<ResidualUnit>,
    out_dim: usize,
}

pub(crate) struct Vocoder {
    dec_in_proj: MlxConv1d,
    conv_in: MlxConv1d,
    blocks: Vec<VocoderBlock>,
    snake_out_alpha: Vec<f32>,
    conv_out: MlxConv1d,
    dims: VocoderDims,
}

fn snake(x: &mut [f32], alpha: &[f32], channels: usize, frames: usize) {
    for c in 0..channels {
        let a = alpha[c];
        let denom = a + 1e-9;
        for f in 0..frames {
            let v = &mut x[c * frames + f];
            let s = (a * *v).sin();
            *v += s * s / denom;
        }
    }
}

impl ResidualUnit {
    fn load(
        store: &mut WeightStore,
        base: &str,
        channels: usize,
        dilation: usize,
    ) -> Result<ResidualUnit> {
        let snake1 = store.tensor(&format!("{base}.snake1.alpha"))?;
        let snake2 = store.tensor(&format!("{base}.snake2.alpha"))?;
        let padding = (7 - 1) * dilation / 2;
        let conv1 = MlxConv1d::load(
            store,
            &format!("{base}.conv1"),
            ConvSpec {
                kernel: 7,
                stride: 1,
                padding,
                dilation,
            },
        )?;
        let conv2 = MlxConv1d::load(
            store,
            &format!("{base}.conv2"),
            ConvSpec {
                kernel: 1,
                stride: 1,
                padding: 0,
                dilation: 1,
            },
        )?;
        if snake1.data.len() != channels || snake2.data.len() != channels {
            return Err(SpeechError::Tensor {
                name: format!("{base}.snake1.alpha"),
                why: format!("expected {channels} snake alpha values"),
            });
        }
        Ok(ResidualUnit {
            snake1_alpha: snake1.data,
            conv1,
            snake2_alpha: snake2.data,
            conv2,
        })
    }

    fn forward(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let channels = self.snake1_alpha.len();
        let mut activated = x.to_vec();
        snake(&mut activated, &self.snake1_alpha, channels, frames);
        let y = self.conv1.forward(&activated, frames);
        let y_frames = y.len() / channels;
        let mut activated = y;
        snake(&mut activated, &self.snake2_alpha, channels, y_frames);
        let y = self.conv2.forward(&activated, y_frames);
        let mut out = x.to_vec();
        for (o, v) in out.iter_mut().zip(y) {
            *o += v;
        }
        out
    }
}

impl VocoderBlock {
    fn load(
        store: &mut WeightStore,
        base: &str,
        in_dim: usize,
        out_dim: usize,
        stride: usize,
    ) -> Result<VocoderBlock> {
        let snake1 = store.tensor(&format!("{base}.snake1.alpha"))?;
        if snake1.data.len() != in_dim {
            return Err(SpeechError::Tensor {
                name: format!("{base}.snake1.alpha"),
                why: format!("expected {in_dim} snake alpha values"),
            });
        }
        let conv_t1 = MlxConvTranspose1d::load(
            store,
            &format!("{base}.conv_t1"),
            ConvSpec {
                kernel: 2 * stride,
                stride,
                padding: stride.div_ceil(2),
                dilation: 1,
            },
        )?;
        let mut res_units = Vec::with_capacity(3);
        for (name, dilation) in [("res_unit1", 1), ("res_unit2", 3), ("res_unit3", 9)] {
            res_units.push(ResidualUnit::load(
                store,
                &format!("{base}.{name}"),
                out_dim,
                dilation,
            )?);
        }
        Ok(VocoderBlock {
            snake1_alpha: snake1.data,
            conv_t1,
            res_units,
            out_dim,
        })
    }

    fn forward(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let in_dim = self.snake1_alpha.len();
        let mut activated = x.to_vec();
        snake(&mut activated, &self.snake1_alpha, in_dim, frames);
        let mut y = self.conv_t1.forward(&activated, frames);
        let mut frames = y.len() / self.out_dim;
        for unit in &self.res_units {
            y = unit.forward(&y, frames);
            frames = y.len() / unit.snake1_alpha.len();
        }
        y
    }
}

impl Vocoder {
    pub(crate) fn load(store: &mut WeightStore, base: &str, dims: VocoderDims) -> Result<Vocoder> {
        if dims.latent_channels % 2 != 0 {
            return Err(SpeechError::BadConfig {
                field: "dit_in_channels".to_string(),
                why: "latent channels must split into two stereo halves".to_string(),
            });
        }
        let dec_in_proj = MlxConv1d::load(
            store,
            &format!("{base}.dec_in_proj"),
            ConvSpec {
                kernel: 1,
                stride: 1,
                padding: 0,
                dilation: 1,
            },
        )?;
        let conv_in = MlxConv1d::load(
            store,
            &format!("{base}.conv_in"),
            ConvSpec {
                kernel: 7,
                stride: 1,
                padding: 3,
                dilation: 1,
            },
        )?;
        let mut blocks = Vec::with_capacity(dims.upsampling_ratios.len());
        let mut in_dim = dims.hidden_dim;
        for (index, stride) in dims.upsampling_ratios.iter().copied().enumerate() {
            let out_dim = dims.hidden_dim / (2usize << index);
            blocks.push(VocoderBlock::load(
                store,
                &format!("{base}.blocks.{index}"),
                in_dim,
                out_dim,
                stride,
            )?);
            in_dim = out_dim;
        }
        let snake_out = store.tensor(&format!("{base}.snake_out.alpha"))?;
        let conv_out = MlxConv1d::load(
            store,
            &format!("{base}.conv_out"),
            ConvSpec {
                kernel: 7,
                stride: 1,
                padding: 3,
                dilation: 1,
            },
        )?;
        Ok(Vocoder {
            dec_in_proj,
            conv_in,
            blocks,
            snake_out_alpha: snake_out.data,
            conv_out,
            dims,
        })
    }

    /// Decode latents `[1, latent_channels, T]` into a planar stereo
    /// waveform `[2, S]`.
    pub(crate) fn forward(&self, latents: &[f32], seq: usize) -> Result<Vec<f32>> {
        let half = self.dims.latent_channels / 2;
        if latents.len() != self.dims.latent_channels * seq {
            return Err(SpeechError::Input {
                why: format!(
                    "vocoder latents {} do not match {}x{seq}",
                    latents.len(),
                    self.dims.latent_channels
                ),
            });
        }
        let mut out = Vec::with_capacity(2 * seq);
        for b in 0..2 {
            // One stereo half as [half, seq], channel-major.
            let mut hidden: Vec<f32> = latents[b * half * seq..(b + 1) * half * seq].to_vec();
            let mut frames = seq;
            hidden = self.dec_in_proj.forward(&hidden, frames);
            frames = hidden.len() / self.dims.input_dim;
            hidden = self.conv_in.forward(&hidden, frames);
            frames = hidden.len() / self.dims.hidden_dim;
            for block in &self.blocks {
                hidden = block.forward(&hidden, frames);
                frames = hidden.len() / block.out_dim;
                if frames == 0 {
                    return Err(SpeechError::Input {
                        why: "vocoder collapsed to zero frames".to_string(),
                    });
                }
            }
            let channels = hidden.len() / frames;
            let mut activated = hidden;
            snake(&mut activated, &self.snake_out_alpha, channels, frames);
            let wave = self.conv_out.forward(&activated, frames);
            if wave.len() != frames {
                return Err(SpeechError::Input {
                    why: "vocoder output conv changed the frame count".to_string(),
                });
            }
            out.extend(wave.iter().map(|v| v.tanh()));
        }
        Ok(out)
    }
}
