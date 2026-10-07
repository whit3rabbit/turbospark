//! VibeVoice audio tokenizer encoders (acoustic and semantic).
//!
//! Reference: `mlx_audio/stt/models/vibevoice_asr/audio_encoder.py` at
//! mlx-audio 0.5.7, commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`.
//!
//! A `TokenizerEncoder` is a fully convolutional waveform-to-latent stack:
//! a stride-1 stem convolution, then for each stage a strided downsample
//! convolution followed by that many pre-norm blocks (depthwise causal
//! convolution mixer plus a GELU feed-forward, each with a per-channel
//! layer scale), then a stride-1 head projection to the latent width.
//! The pinned config reverses the declared ratios `[8, 5, 5, 4, 2, 2]`
//! into strides `[2, 2, 4, 5, 5, 8]` (product 3200, one latent frame per
//! 3200 samples at 24 kHz).
//!
//! Layout contract: activations are frame-major `[frames, channels]` as in
//! the MLX reference; [`ops::conv1d`] consumes channel-major rows, so the
//! convolution wrapper transposes in and out. All weights load from the
//! raw PyTorch-layout safetensors (`[out_channels, in_channels, kernel]`),
//! which is exactly the layout [`ops::conv1d`] takes, so no checkpoint
//! tensor is transposed at load.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::nn::{Linear, RmsNorm};
use crate::ops;
use crate::{Result, SpeechError};

/// Load a tensor from the first shard that carries it, with a shape check.
pub(crate) fn load_sharded(
    files: &[SafetensorsFile],
    name: &str,
    shape: &[usize],
) -> Result<Vec<f32>> {
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

pub(crate) fn load_linear_sharded(
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

pub(crate) fn load_rms_norm_sharded(
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

/// One causal convolution with the reference `SConv1d` padding rules:
/// `(kernel - 1) * dilation - (stride - 1)` total padding, all of it on
/// the left for the causal models pinned here, plus the right-side extra
/// padding that aligns the output to `ceil(frames / stride)`.
pub(crate) struct SConv1d {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    padding_total: usize,
}

impl SConv1d {
    pub(crate) fn load(
        files: &[SafetensorsFile],
        name: &str,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        has_bias: bool,
    ) -> Result<Self> {
        let weight = load_sharded(
            files,
            &format!("{name}.weight"),
            &[out_channels, in_channels / groups, kernel],
        )?;
        let bias = if has_bias {
            Some(load_sharded(
                files,
                &format!("{name}.bias"),
                &[out_channels],
            )?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel,
            stride,
            dilation,
            groups,
            padding_total: (kernel - 1) * dilation - (stride - 1),
        })
    }

    /// `x [frames, in_channels] -> [out_frames, out_channels]`.
    pub(crate) fn forward(&self, x: &[f32], frames: usize) -> Vec<f32> {
        assert_eq!(x.len(), frames * self.in_channels, "SConv1d input length");
        // `SConv1d._get_extra_padding`: n_frames as an exact f64 expression,
        // then ideal length minus the input length.
        let n_frames = (frames as f64 - self.kernel as f64 + self.padding_total as f64)
            / self.stride as f64
            + 1.0;
        let ideal_length =
            (n_frames.ceil() as usize - 1) * self.stride + self.kernel - self.padding_total;
        let extra_padding = ideal_length - frames;
        // Causal: everything on the left plus the alignment tail on the right.
        let padded_frames = frames + self.padding_total + extra_padding;

        // Transpose to channel-major and pad.
        let mut channel_major = vec![0.0f32; self.in_channels * padded_frames];
        for (frame, row) in x.chunks_exact(self.in_channels).enumerate() {
            let target = frame + self.padding_total;
            for (channel, &value) in row.iter().enumerate() {
                channel_major[channel * padded_frames + target] = value;
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
        // Back to frame-major.
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

/// Pre-norm block: depthwise causal convolution mixer and a GELU feed
/// forward, each scaled by a per-channel layer scale before the residual.
struct Block1d {
    norm: RmsNorm,
    ffn_norm: RmsNorm,
    mixer: SConv1d,
    ffn1: Linear,
    ffn2: Linear,
    gamma: Vec<f32>,
    ffn_gamma: Vec<f32>,
    dim: usize,
}

impl Block1d {
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
            mixer: SConv1d::load(
                files,
                &format!("{name}.mixer.conv.conv.conv"),
                dim,
                dim,
                7,
                1,
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

    fn forward(&self, x: &mut [f32], frames: usize) {
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

/// The full waveform-to-latent encoder shared by the acoustic and semantic
/// tokenizers.
pub(crate) struct TokenizerEncoder {
    stem: SConv1d,
    downsamples: Vec<SConv1d>,
    stages: Vec<Vec<Block1d>>,
    head: SConv1d,
}

impl TokenizerEncoder {
    /// Loads one side (`model.acoustic_tokenizer.encoder` or
    /// `model.semantic_tokenizer.encoder`) from the raw checkpoint keys.
    pub(crate) fn load(
        files: &[SafetensorsFile],
        prefix: &str,
        vae_dim: usize,
        n_filters: usize,
        ratios: &[usize],
        depths: &[usize],
        eps: f32,
        has_bias: bool,
    ) -> Result<Self> {
        // Encoding walks the declared ratios in reverse.
        let strides: Vec<usize> = ratios.iter().rev().copied().collect();
        let stem = SConv1d::load(
            files,
            &format!("{prefix}.downsample_layers.0.0.conv.conv"),
            1,
            n_filters,
            7,
            1,
            1,
            1,
            has_bias,
        )?;
        let mut downsamples = Vec::with_capacity(strides.len());
        for (index, &stride) in strides.iter().enumerate() {
            let in_channels = n_filters * (1 << index);
            let out_channels = n_filters * (1 << (index + 1));
            downsamples.push(SConv1d::load(
                files,
                &format!("{prefix}.downsample_layers.{}.0.conv.conv", index + 1),
                in_channels,
                out_channels,
                stride * 2,
                stride,
                1,
                1,
                has_bias,
            )?);
        }
        let mut stages = Vec::with_capacity(depths.len());
        for (stage_index, &depth) in depths.iter().enumerate() {
            let dim = n_filters * (1 << stage_index);
            let mut blocks = Vec::with_capacity(depth);
            for block_index in 0..depth {
                blocks.push(Block1d::load(
                    files,
                    &format!("{prefix}.stages.{stage_index}.{block_index}"),
                    dim,
                    eps,
                    has_bias,
                )?);
            }
            stages.push(blocks);
        }
        let head_channels = n_filters * (1 << strides.len());
        let head = SConv1d::load(
            files,
            &format!("{prefix}.head.conv.conv"),
            head_channels,
            vae_dim,
            7,
            1,
            1,
            1,
            has_bias,
        )?;
        Ok(Self {
            stem,
            downsamples,
            stages,
            head,
        })
    }

    /// `waveform [samples] -> latents [frames, vae_dim]`, one latent frame
    /// per `hop_length` input samples (3200 for the pinned ratios).
    pub(crate) fn forward(&self, waveform: &[f32]) -> Vec<f32> {
        let mut frames = waveform.len();
        // Frame-major single channel.
        let x: Vec<f32> = waveform.to_vec();
        let mut x = self.stem.forward(&x, frames);
        frames = x.len() / self.stem.out_channels;
        for (stage, downsample) in self.stages.iter().zip(&self.downsamples) {
            x = downsample.forward(&x, frames);
            frames = x.len() / downsample.out_channels;
            for block in stage {
                block.forward(&mut x, frames);
            }
        }
        self.head.forward(&x, frames)
    }
}
