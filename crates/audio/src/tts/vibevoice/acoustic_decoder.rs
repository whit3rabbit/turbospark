//! The VibeVoice acoustic tokenizer decoder
//! (`mlx_audio/tts/models/vibevoice/acoustic_tokenizer.py`, decoder
//! only; inference ships no encoder).
//!
//! Reference: mlx-audio at commit
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/vibevoice/acoustic_tokenizer.py).
//! A `TokenizerDecoder` is a fully convolutional latent-to-waveform
//! stack: a stride-1 stem convolution (`upsample_layers.0`), then for
//! each stage that many pre-norm blocks (depthwise causal convolution
//! mixer plus an exact-GELU feed forward, each with a per-channel layer
//! scale) followed by a causal transposed-convolution upsample
//! (`upsample_layers.1..`), then a stride-1 head convolution. The
//! declared encoder ratios double as the decoder upsample strides and
//! the reversed encoder depths as the stage block counts when the
//! decoder fields are absent, which is how the deployed config ships.
//!
//! Layout contract: activations are frame-major `[frames, channels]`;
//! [`ops::conv1d`] / [`ops::conv_transpose1d`] consume channel-major
//! rows, so the wrappers transpose in and out. The causal transposed
//! convolution trims `kernel - stride` samples off the right (the
//! `trim_right_ratio = 1.0` non-streaming path; the streaming cache
//! path is not ported, matching the non-streaming scope of this
//! family). Weights load in the MLX layouts the converted saves carry
//! (`[out, kernel, in / groups]` conv, `[out, kernel, in]` conv
//! transpose) and are rotated into the PyTorch layouts `ops` takes.

use crate::nn::{Linear, RmsNorm};
use crate::ops;
use crate::{Result, SpeechError};
use turbospark_model_io::safetensors::SafetensorsFile;

/// Load a tensor from the first shard that carries it, with a shape check.
fn load_sharded(files: &[SafetensorsFile], name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let file = files
        .iter()
        .find(|file| file.contains_tensor(name))
        .ok_or_else(|| SpeechError::Tensor {
            name: name.to_owned(),
            why: "tensor is missing".into(),
        })?;
    let descriptor = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
        name: name.to_owned(),
        why: "tensor is missing".into(),
    })?;
    if descriptor.shape != shape {
        return Err(SpeechError::Tensor {
            name: name.to_owned(),
            why: format!("expected shape {shape:?}, got {:?}", descriptor.shape),
        });
    }
    Ok(file.load_as_f32(name)?)
}

fn load_linear_sharded(
    files: &[SafetensorsFile],
    name: &str,
    input: usize,
    output: usize,
    has_bias: bool,
) -> Result<Linear> {
    let weight = load_sharded(files, &format!("{name}.weight"), &[output, input])?;
    let bias = if has_bias {
        Some(load_sharded(files, &format!("{name}.bias"), &[output])?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias, input, output))
}

fn load_rms_norm_sharded(
    files: &[SafetensorsFile],
    name: &str,
    width: usize,
    epsilon: f32,
) -> Result<RmsNorm> {
    Ok(RmsNorm::new(
        load_sharded(files, &format!("{name}.weight"), &[width])?,
        epsilon,
    ))
}

/// One causal convolution: `(kernel - 1) * dilation` zeros on the left,
/// then a stride-`stride` convolution. `x [frames, in] -> [out, out_ch]`.
#[allow(clippy::too_many_arguments)]
pub(crate) struct CausalConv {
    pub(crate) weight: Vec<f32>,
    pub(crate) bias: Option<Vec<f32>>,
    pub(crate) in_channels: usize,
    pub(crate) out_channels: usize,
    pub(crate) kernel: usize,
    pub(crate) stride: usize,
    dilation: usize,
    groups: usize,
}

impl CausalConv {
    #[allow(clippy::too_many_arguments)]
    fn load(
        files: &[SafetensorsFile],
        name: &str,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        groups: usize,
        has_bias: bool,
    ) -> Result<Self> {
        // The converted checkpoints store the MLX Conv1d layout
        // `[out, kernel, in / groups]`; ops::conv1d takes the PyTorch
        // `[out, in / groups, kernel]`, so the last two axes swap.
        let mlx = load_sharded(
            files,
            &format!("{name}.weight"),
            &[out_channels, kernel, in_channels / groups],
        )?;
        let in_per_g = in_channels / groups;
        let mut weight = vec![0.0f32; mlx.len()];
        for oc in 0..out_channels {
            for k in 0..kernel {
                for ic in 0..in_per_g {
                    weight[(oc * in_per_g + ic) * kernel + k] =
                        mlx[(oc * kernel + k) * in_per_g + ic];
                }
            }
        }
        Ok(Self {
            weight,
            bias: if has_bias {
                Some(load_sharded(
                    files,
                    &format!("{name}.bias"),
                    &[out_channels],
                )?)
            } else {
                None
            },
            in_channels,
            out_channels,
            kernel,
            stride,
            dilation: 1,
            groups,
        })
    }

    pub(crate) fn forward(&self, x: &[f32], frames: usize) -> Vec<f32> {
        // Frame-major to channel-major, causal pad, conv, and back.
        let padding = (self.kernel - 1) * self.dilation;
        let mut channel_major = vec![0.0f32; self.in_channels * (frames + padding)];
        for (frame, row) in x.chunks_exact(self.in_channels).enumerate() {
            for (channel, &value) in row.iter().enumerate() {
                channel_major[channel * (frames + padding) + frame + padding] = value;
            }
        }
        let out = ops::conv1d(
            &channel_major,
            &self.weight,
            self.bias.as_deref(),
            self.in_channels,
            self.out_channels,
            self.kernel,
            self.stride,
            0,
            self.dilation,
            self.groups,
        );
        let out_frames = out.len() / self.out_channels;
        let mut frame_major = vec![0.0f32; out.len()];
        for channel in 0..self.out_channels {
            for frame in 0..out_frames {
                frame_major[frame * self.out_channels + channel] =
                    out[channel * out_frames + frame];
            }
        }
        frame_major
    }
}

/// One causal transposed convolution (the upsample): the plain
/// transposed convolution with the last `kernel - stride` samples
/// trimmed off (`padding_total = kernel - stride`, all of it right-side
/// at `trim_right_ratio = 1.0`). `x [frames, in] -> [frames * stride,
/// out]`.
#[allow(clippy::too_many_arguments)]
pub(crate) struct CausalConvTranspose {
    pub(crate) weight: Vec<f32>,
    pub(crate) bias: Option<Vec<f32>>,
    pub(crate) in_channels: usize,
    pub(crate) out_channels: usize,
    pub(crate) kernel: usize,
    pub(crate) stride: usize,
}

/// `forward` trims `kernel - stride` samples, which underflows (a panic
/// in debug, a wrapped trim in release) when the kernel is smaller than
/// the stride. A zero stride would also divide by zero in the transpose.
pub(super) fn check_conv_transpose_geometry(kernel: usize, stride: usize) -> Result<()> {
    if stride == 0 || kernel < stride {
        return Err(SpeechError::BadConfig {
            field: "acoustic_decoder upsample".to_string(),
            why: format!("kernel {kernel} must be at least stride {stride}, and stride >= 1"),
        });
    }
    Ok(())
}

impl CausalConvTranspose {
    #[allow(clippy::too_many_arguments)]
    fn load(
        files: &[SafetensorsFile],
        name: &str,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        has_bias: bool,
    ) -> Result<Self> {
        check_conv_transpose_geometry(kernel, stride)?;
        // The converted checkpoints store the MLX ConvTranspose1d
        // layout `[out, kernel, in]`; ops::conv_transpose1d takes the
        // PyTorch `[in, out, kernel]`, so the axes rotate (2, 0, 1).
        let mlx = load_sharded(
            files,
            &format!("{name}.weight"),
            &[out_channels, kernel, in_channels],
        )?;
        let mut weight = vec![0.0f32; mlx.len()];
        for oc in 0..out_channels {
            for k in 0..kernel {
                for ic in 0..in_channels {
                    weight[(ic * out_channels + oc) * kernel + k] =
                        mlx[(oc * kernel + k) * in_channels + ic];
                }
            }
        }
        Ok(Self {
            weight,
            bias: if has_bias {
                Some(load_sharded(
                    files,
                    &format!("{name}.bias"),
                    &[out_channels],
                )?)
            } else {
                None
            },
            in_channels,
            out_channels,
            kernel,
            stride,
        })
    }

    pub(crate) fn forward(&self, x: &[f32], frames: usize) -> Vec<f32> {
        // Frame-major to channel-major.
        let mut channel_major = vec![0.0f32; self.in_channels * frames];
        for (frame, row) in x.chunks_exact(self.in_channels).enumerate() {
            for (channel, &value) in row.iter().enumerate() {
                channel_major[channel * frames + frame] = value;
            }
        }
        let out = ops::conv_transpose1d(
            &channel_major,
            &self.weight,
            self.bias.as_deref(),
            self.in_channels,
            self.out_channels,
            self.kernel,
            self.stride,
            0,
            0,
            1,
        );
        let out_frames = out.len() / self.out_channels;
        let trim = self.kernel - self.stride;
        let kept = out_frames.saturating_sub(trim);
        // Channel-major back to frame-major, minus the right trim.
        let mut frame_major = vec![0.0f32; kept * self.out_channels];
        for channel in 0..self.out_channels {
            for frame in 0..kept {
                frame_major[frame * self.out_channels + channel] =
                    out[channel * out_frames + frame];
            }
        }
        frame_major
    }
}

/// Pre-norm block: depthwise causal convolution mixer and an exact-GELU
/// feed forward, each scaled by a per-channel layer scale before the
/// residual (`Block1D`).
pub(crate) struct Block {
    norm: RmsNorm,
    ffn_norm: RmsNorm,
    mixer: CausalConv,
    ffn1: Linear,
    ffn2: Linear,
    gamma: Vec<f32>,
    ffn_gamma: Vec<f32>,
    dim: usize,
}

impl Block {
    fn load(
        files: &[SafetensorsFile],
        name: &str,
        dim: usize,
        eps: f32,
        has_bias: bool,
    ) -> Result<Self> {
        Ok(Self {
            norm: load_rms_norm_sharded(files, &format!("{name}.norm"), dim, eps)?,
            ffn_norm: load_rms_norm_sharded(files, &format!("{name}.ffn_norm"), dim, eps)?,
            // Mixer.conv (DepthwiseConv) .conv (CausalConv1d) .conv (Conv1d).
            mixer: CausalConv::load(
                files,
                &format!("{name}.mixer.conv.conv.conv"),
                dim,
                dim,
                7,
                1,
                dim,
                has_bias,
            )?,
            ffn1: load_linear_sharded(
                files,
                &format!("{name}.ffn.linear1"),
                dim,
                dim * 4,
                has_bias,
            )?,
            ffn2: load_linear_sharded(
                files,
                &format!("{name}.ffn.linear2"),
                dim * 4,
                dim,
                has_bias,
            )?,
            gamma: load_sharded(files, &format!("{name}.gamma"), &[dim])?,
            ffn_gamma: load_sharded(files, &format!("{name}.ffn_gamma"), &[dim])?,
            dim,
        })
    }

    pub(crate) fn forward(&self, x: &mut [f32], frames: usize) {
        // Mixer path: x + mixer(norm(x)) * gamma.
        let mut normed = x.to_vec();
        self.norm.apply(&mut normed, frames);
        let mixed = self.mixer.forward(&normed, frames);
        for (index, value) in x.iter_mut().enumerate() {
            *value += mixed[index] * self.gamma[index % self.dim];
        }
        // FFN path: x + ffn2(gelu(ffn1(norm(x)))) * ffn_gamma.
        let mut normed = x.to_vec();
        self.ffn_norm.apply(&mut normed, frames);
        let mut hidden = self.ffn1.forward(&normed, frames);
        ops::gelu_erf(&mut hidden);
        let out = self.ffn2.forward(&hidden, frames);
        for (index, value) in x.iter_mut().enumerate() {
            *value += out[index] * self.ffn_gamma[index % self.dim];
        }
    }
}

/// The full latent-to-waveform decoder (`TokenizerDecoder`): stem conv,
/// then per stage the stage blocks followed by one upsample, then the
/// head conv. `latents [frames, vae_dim] -> samples [out_frames]`.
pub(crate) struct AcousticDecoder {
    pub(crate) stem: CausalConv,
    pub(crate) upsamples: Vec<CausalConvTranspose>,
    pub(crate) stages: Vec<Vec<Block>>,
    pub(crate) head: CausalConv,
    pub(crate) vae_dim: usize,
}

impl AcousticDecoder {
    /// Loads the decoder under `prefix` (the checkpoint's
    /// `acoustic_tokenizer.decoder.`). `ratios` are the upsample
    /// strides; `depths` the per-stage block counts (the reversed
    /// encoder depths when the decoder fields are absent).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load(
        files: &[SafetensorsFile],
        prefix: &str,
        vae_dim: usize,
        channels: usize,
        n_filters: usize,
        ratios: &[usize],
        depths: &[usize],
        eps: f32,
        has_bias: bool,
    ) -> Result<Self> {
        let n_stages = depths.len();
        let stem_out = n_filters * (1 << (n_stages - 1));
        let stem = CausalConv::load(
            files,
            &format!("{prefix}.upsample_layers.0.0.conv.conv"),
            vae_dim,
            stem_out,
            7,
            1,
            1,
            has_bias,
        )?;
        let mut upsamples = Vec::with_capacity(ratios.len());
        for (index, &ratio) in ratios.iter().enumerate() {
            let in_channels = n_filters * (1 << (n_stages - 1 - index));
            let out_channels = if index + 1 < ratios.len() {
                n_filters * (1 << (n_stages - 2 - index))
            } else {
                n_filters
            };
            upsamples.push(CausalConvTranspose::load(
                files,
                &format!("{prefix}.upsample_layers.{}.0.convtr.convtr", index + 1),
                in_channels,
                out_channels,
                ratio * 2,
                ratio,
                has_bias,
            )?);
        }
        let mut stages = Vec::with_capacity(n_stages);
        for (stage_index, &depth) in depths.iter().enumerate() {
            let dim = n_filters * (1 << (n_stages - 1 - stage_index));
            let mut blocks = Vec::with_capacity(depth);
            for block_index in 0..depth {
                blocks.push(Block::load(
                    files,
                    &format!("{prefix}.stages.{stage_index}.{block_index}"),
                    dim,
                    eps,
                    has_bias,
                )?);
            }
            stages.push(blocks);
        }
        let head = CausalConv::load(
            files,
            &format!("{prefix}.head.conv.conv"),
            n_filters,
            channels,
            7,
            1,
            1,
            has_bias,
        )?;
        Ok(Self {
            stem,
            upsamples,
            stages,
            head,
            vae_dim,
        })
    }

    /// `latents [frames, vae_dim] -> samples [frames * ratio product]`,
    /// single batch. Stage `i` blocks run before upsample `i + 1`; the
    /// stem feeds stage 0, exactly as the reference `TokenizerDecoder.
    /// __call__` walks `upsample_layers` and `stages` in lockstep.
    pub(crate) fn forward(&self, latents: &[f32], frames: usize) -> Result<Vec<f32>> {
        if latents.len() != frames * self.vae_dim {
            return Err(SpeechError::Input {
                why: format!(
                    "decoder expects [frames, {}] latents, got {} values for {frames} frames",
                    self.vae_dim,
                    latents.len()
                ),
            });
        }
        let mut x = self.stem.forward(latents, frames);
        let mut width = self.stem.out_channels;
        for (index, blocks) in self.stages.iter().enumerate() {
            for block in blocks {
                let frames = frames_of(&x, width);
                block.forward(&mut x, frames);
            }
            if index < self.upsamples.len() {
                let frames = frames_of(&x, width);
                x = self.upsamples[index].forward(&x, frames);
                width = self.upsamples[index].out_channels;
            }
        }
        let frames = frames_of(&x, width);
        let out = self.head.forward(&x, frames);
        Ok(out)
    }
}

fn frames_of(x: &[f32], channels: usize) -> usize {
    match channels {
        0 => 0,
        _ => x.len() / channels,
    }
}
