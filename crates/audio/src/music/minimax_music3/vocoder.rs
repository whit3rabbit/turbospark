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

use super::backend::{self, ComputeBackend};
use super::conv::{ConvSpec, MlxConv1d, MlxConvTranspose1d};
use super::precision::DType;
use super::weights::{Tensor, WeightStore};
use std::rc::Rc;

pub(crate) struct VocoderDims {
    pub latent_channels: usize,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub upsampling_ratios: Vec<usize>,
}

struct ResidualUnit {
    snake1_alpha: Tensor,
    conv1: MlxConv1d,
    snake2_alpha: Tensor,
    conv2: MlxConv1d,
}

struct VocoderBlock {
    snake1_alpha: Tensor,
    conv_t1: MlxConvTranspose1d,
    res_units: Vec<ResidualUnit>,
    out_dim: usize,
}

pub(crate) struct Vocoder {
    dec_in_proj: MlxConv1d,
    conv_in: MlxConv1d,
    blocks: Vec<VocoderBlock>,
    snake_out_alpha: Tensor,
    conv_out: MlxConv1d,
    dims: VocoderDims,
    backend: Option<Rc<dyn ComputeBackend>>,
}

pub(crate) fn snake(x: &mut [f32], alpha: &[f32], channels: usize, frames: usize, dtype: DType) {
    for c in 0..channels {
        let a = dtype.round(alpha[c]);
        let denom = dtype.round(a + dtype.round(1e-9));
        for f in 0..frames {
            let v = &mut x[c * frames + f];
            let s = dtype.round(dtype.round(a * *v).sin());
            *v = dtype.round(*v + dtype.round(dtype.round(s * s) / denom));
        }
    }
}

fn activate_snake(
    x: &[f32],
    alpha: &[f32],
    channels: usize,
    frames: usize,
    dtype: DType,
    compute: &Option<Rc<dyn ComputeBackend>>,
) -> Result<Vec<f32>> {
    if dtype != DType::F32 {
        if let Some(compute) = compute {
            return compute.snake(x, alpha, channels, frames, dtype);
        }
    }
    let mut out = x.to_vec();
    snake(&mut out, alpha, channels, frames, dtype);
    Ok(out)
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
            snake1_alpha: snake1,
            conv1,
            snake2_alpha: snake2,
            conv2,
        })
    }

    fn dtype(&self, input: DType) -> DType {
        input.promote(self.conv2.dtype(self.conv1.dtype(input)))
    }
    fn forward(
        &self,
        x: &[f32],
        frames: usize,
        input_dtype: DType,
        compute: &Option<Rc<dyn ComputeBackend>>,
        stage: &str,
    ) -> Result<Vec<f32>> {
        let channels = self.snake1_alpha.len();
        let activated = activate_snake(
            x,
            &self.snake1_alpha,
            channels,
            frames,
            input_dtype,
            compute,
        )?;
        backend::trace(
            compute,
            &format!("{stage}.snake1"),
            &activated,
            input_dtype,
            &[channels, frames],
        );
        let y = self.conv1.forward_typed(&activated, frames, input_dtype)?;
        let y_frames = y.len() / channels;
        let conv1_dtype = self.conv1.dtype(input_dtype);
        backend::trace(
            compute,
            &format!("{stage}.conv1"),
            &y,
            conv1_dtype,
            &[channels, y_frames],
        );
        let activated = activate_snake(
            &y,
            &self.snake2_alpha,
            channels,
            y_frames,
            conv1_dtype,
            compute,
        )?;
        backend::trace(
            compute,
            &format!("{stage}.snake2"),
            &activated,
            conv1_dtype,
            &[channels, y_frames],
        );
        let y = self
            .conv2
            .forward_typed(&activated, y_frames, conv1_dtype)?;
        backend::trace(
            compute,
            &format!("{stage}.conv2"),
            &y,
            self.conv2.dtype(conv1_dtype),
            &[channels, y_frames],
        );
        let mut out = x.to_vec();
        for (o, v) in out.iter_mut().zip(y) {
            *o = self.dtype(input_dtype).round(*o + v);
        }
        backend::trace(
            compute,
            &format!("{stage}.output"),
            &out,
            self.dtype(input_dtype),
            &[channels, frames],
        );
        Ok(out)
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
            snake1_alpha: snake1,
            conv_t1,
            res_units,
            out_dim,
        })
    }

    fn dtype(&self, input: DType) -> DType {
        self.res_units
            .iter()
            .fold(self.conv_t1.dtype(input), |d, u| u.dtype(d))
    }
    fn forward(
        &self,
        x: &[f32],
        frames: usize,
        input_dtype: DType,
        compute: &Option<Rc<dyn ComputeBackend>>,
        stage: &str,
    ) -> Result<Vec<f32>> {
        let in_dim = self.snake1_alpha.len();
        let activated =
            activate_snake(x, &self.snake1_alpha, in_dim, frames, input_dtype, compute)?;
        backend::trace(
            compute,
            &format!("{stage}.snake"),
            &activated,
            input_dtype,
            &[in_dim, frames],
        );
        let mut y = self
            .conv_t1
            .forward_typed(&activated, frames, input_dtype)?;
        let mut frames = y.len() / self.out_dim;
        let mut dtype = self.conv_t1.dtype(input_dtype);
        backend::trace(
            compute,
            &format!("{stage}.transpose"),
            &y,
            dtype,
            &[self.out_dim, frames],
        );
        for (index, unit) in self.res_units.iter().enumerate() {
            y = unit.forward(&y, frames, dtype, compute, &format!("{stage}.unit.{index}"))?;
            dtype = unit.dtype(dtype);
            frames = y.len() / unit.snake1_alpha.len();
        }
        Ok(y)
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
            snake_out_alpha: snake_out,
            conv_out,
            dims,
            backend: store.backend(),
        })
    }

    /// Decode latents `[1, latent_channels, T]` into a planar stereo
    /// waveform `[2, S]`.
    #[cfg(test)]
    pub(crate) fn forward(&self, latents: &[f32], seq: usize) -> Result<Vec<f32>> {
        self.forward_typed(latents, seq, self.snake_out_alpha.dtype)
    }
    #[cfg(test)]
    pub(crate) fn forward_typed(
        &self,
        latents: &[f32],
        seq: usize,
        input_dtype: DType,
    ) -> Result<Vec<f32>> {
        self.forward_controlled(latents, seq, input_dtype, &mut |_, _, _| {
            super::Control::Continue
        })?
        .ok_or_else(super::never_cancelled)
    }

    pub(crate) fn forward_controlled(
        &self,
        latents: &[f32],
        seq: usize,
        input_dtype: DType,
        progress: &mut dyn FnMut(usize, usize, usize) -> super::Control,
    ) -> Result<Option<Vec<f32>>> {
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
        let stages = self.blocks.len() + 2;
        for b in 0..2 {
            if progress(b, 0, stages) == super::Control::Cancel {
                return Ok(None);
            }
            // One stereo half as [half, seq], channel-major.
            let mut hidden: Vec<f32> = latents[b * half * seq..(b + 1) * half * seq].to_vec();
            let mut frames = seq;
            let mut dtype = input_dtype;
            backend::trace(
                &self.backend,
                &format!("vocoder.{b}.input"),
                &hidden,
                dtype,
                &[half, frames],
            );
            hidden = self.dec_in_proj.forward_typed(&hidden, frames, dtype)?;
            dtype = self.dec_in_proj.dtype(dtype);
            backend::trace(
                &self.backend,
                &format!("vocoder.{b}.proj"),
                &hidden,
                dtype,
                &[self.dims.input_dim, frames],
            );
            frames = hidden.len() / self.dims.input_dim;
            hidden = self.conv_in.forward_typed(&hidden, frames, dtype)?;
            dtype = self.conv_in.dtype(dtype);
            backend::trace(
                &self.backend,
                &format!("vocoder.{b}.conv_in"),
                &hidden,
                dtype,
                &[self.dims.hidden_dim, frames],
            );
            frames = hidden.len() / self.dims.hidden_dim;
            for (index, block) in self.blocks.iter().enumerate() {
                if progress(b, index + 1, stages) == super::Control::Cancel {
                    return Ok(None);
                }
                hidden = block.forward(
                    &hidden,
                    frames,
                    dtype,
                    &self.backend,
                    &format!("vocoder.{b}.block.{index}"),
                )?;
                dtype = block.dtype(dtype);
                frames = hidden.len() / block.out_dim;
                backend::trace(
                    &self.backend,
                    &format!("vocoder.{b}.block.{index}"),
                    &hidden,
                    dtype,
                    &[block.out_dim, frames],
                );
                if frames == 0 {
                    return Err(SpeechError::Input {
                        why: "vocoder collapsed to zero frames".to_string(),
                    });
                }
            }
            let channels = hidden.len() / frames;
            if progress(b, stages - 1, stages) == super::Control::Cancel {
                return Ok(None);
            }
            let activated = activate_snake(
                &hidden,
                &self.snake_out_alpha,
                channels,
                frames,
                dtype,
                &self.backend,
            )?;
            let wave = self.conv_out.forward_typed(&activated, frames, dtype)?;
            if wave.len() != frames {
                return Err(SpeechError::Input {
                    why: "vocoder output conv changed the frame count".to_string(),
                });
            }
            dtype = self.conv_out.dtype(dtype);
            let wave: Vec<f32> = wave.iter().map(|v| dtype.round(v.tanh())).collect();
            backend::trace(
                &self.backend,
                &format!("vocoder.{b}.output"),
                &wave,
                dtype,
                &[1, frames],
            );
            out.extend(wave);
        }
        Ok(Some(out))
    }
}
