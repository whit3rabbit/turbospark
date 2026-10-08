//! Qwen2-Audio audio tower and multimodal projector.
//!
//! Reference: `mlx_audio/stt/models/qwen2_audio/qwen2_audio.py`
//! (`Qwen2AudioEncoder`, `Qwen2AudioEncoderLayer`,
//! `Qwen2AudioEncoderAttention`, `Qwen2AudioMultiModalProjector`) at
//! mlx-audio 0.5.7, commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`.
//!
//! The tower is the whisper-family frontend: two strided 3-tap convolutions
//! with GELU, additive sinusoidal positions, a stack of pre-norm
//! transformer layers with full (unmasked) self-attention, a non-causal
//! pair average (kernel 2, stride 2), and a final LayerNorm. The projector
//! is one biased linear map to the language-model width.
//!
//! Layout contract: the tower consumes band-major mel input
//! `[num_mels, frames]` and produces row-major `[pooled_frames, d_model]`;
//! [`ops::conv1d`] takes channel-major activations and PyTorch weight
//! layout `[out, in, kernel]`. The pinned checkpoint stores the conv
//! kernels in the MLX layout `[out, kernel, in]` (the upstream sanitize
//! skips the PyTorch transpose whenever any tensor carries "scales", and
//! this conversion quantizes the language model), so the loader transposes
//! them once at load.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::nn::{LayerNorm, Linear};
use crate::ops;
use crate::{Result, SpeechError};

/// Load a plain tensor from the first shard that carries it, with a shape
/// check.
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

fn load_layer_norm_sharded(
    files: &[SafetensorsFile],
    name: &str,
    width: usize,
) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        load_sharded(files, &format!("{name}.weight"), &[width])?,
        Some(load_sharded(files, &format!("{name}.bias"), &[width])?),
        1.0e-5,
    ))
}

/// One tower convolution loaded from the MLX kernel layout.
struct TowerConv {
    weight: Vec<f32>,
    bias: Vec<f32>,
    in_channels: usize,
    out_channels: usize,
}

impl TowerConv {
    fn load(
        files: &[SafetensorsFile],
        name: &str,
        in_channels: usize,
        out_channels: usize,
    ) -> Result<Self> {
        // Checkpoint layout [out, kernel, in]; transpose to the PyTorch
        // [out, in, kernel] layout ops::conv1d consumes.
        let stored = load_sharded(
            files,
            &format!("{name}.weight"),
            &[out_channels, 3, in_channels],
        )?;
        let mut weight = vec![0.0f32; out_channels * in_channels * 3];
        for out in 0..out_channels {
            for kernel in 0..3 {
                for input in 0..in_channels {
                    weight[(out * in_channels + input) * 3 + kernel] =
                        stored[(out * 3 + kernel) * in_channels + input];
                }
            }
        }
        Ok(Self {
            weight,
            bias: load_sharded(files, &format!("{name}.bias"), &[out_channels])?,
            in_channels,
            out_channels,
        })
    }

    fn forward(&self, x: &[f32], stride: usize) -> Vec<f32> {
        ops::conv1d(
            x,
            &self.weight,
            Some(&self.bias),
            self.in_channels,
            self.out_channels,
            3,
            stride,
            1,
            1,
            1,
        )
    }
}

struct EncoderAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    heads: usize,
    head_dim: usize,
    scale: f32,
}

impl EncoderAttention {
    fn load(files: &[SafetensorsFile], prefix: &str, d_model: usize, heads: usize) -> Result<Self> {
        let head_dim = d_model / heads;
        Ok(Self {
            q_proj: load_linear_sharded(
                files,
                &format!("{prefix}.q_proj"),
                d_model,
                d_model,
                true,
            )?,
            k_proj: load_linear_sharded(
                files,
                &format!("{prefix}.k_proj"),
                d_model,
                d_model,
                false,
            )?,
            v_proj: load_linear_sharded(
                files,
                &format!("{prefix}.v_proj"),
                d_model,
                d_model,
                true,
            )?,
            out_proj: load_linear_sharded(
                files,
                &format!("{prefix}.out_proj"),
                d_model,
                d_model,
                true,
            )?,
            heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    fn forward(&self, x: &[f32], seq: usize) -> Vec<f32> {
        let query = self.q_proj.forward(x, seq);
        let key = self.k_proj.forward(x, seq);
        let value = self.v_proj.forward(x, seq);
        let query = ops::split_heads(&query, seq, self.heads, self.head_dim);
        let key = ops::split_heads(&key, seq, self.heads, self.head_dim);
        let value = ops::split_heads(&value, seq, self.heads, self.head_dim);
        // Full bidirectional attention, no mask.
        let attended = ops::mha(
            &query,
            &key,
            &value,
            None,
            self.heads,
            seq,
            seq,
            self.head_dim,
            self.scale,
        );
        let merged = ops::merge_heads(&attended, seq, self.heads, self.head_dim);
        self.out_proj.forward(&merged, seq)
    }
}

struct EncoderLayer {
    attention: EncoderAttention,
    self_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
}

impl EncoderLayer {
    fn load(
        files: &[SafetensorsFile],
        prefix: &str,
        d_model: usize,
        ffn: usize,
        heads: usize,
    ) -> Result<Self> {
        Ok(Self {
            attention: EncoderAttention::load(
                files,
                &format!("{prefix}.self_attn"),
                d_model,
                heads,
            )?,
            self_attn_layer_norm: load_layer_norm_sharded(
                files,
                &format!("{prefix}.self_attn_layer_norm"),
                d_model,
            )?,
            fc1: load_linear_sharded(files, &format!("{prefix}.fc1"), d_model, ffn, true)?,
            fc2: load_linear_sharded(files, &format!("{prefix}.fc2"), ffn, d_model, true)?,
            final_layer_norm: load_layer_norm_sharded(
                files,
                &format!("{prefix}.final_layer_norm"),
                d_model,
            )?,
        })
    }

    fn forward(&self, x: &[f32], seq: usize) -> Vec<f32> {
        let mut normed = x.to_vec();
        self.self_attn_layer_norm.apply(&mut normed, seq);
        let attended = self.attention.forward(&normed, seq);
        let mut residual = x.to_vec();
        for (value, add) in residual.iter_mut().zip(attended) {
            *value += add;
        }
        let mut normed = residual.clone();
        self.final_layer_norm.apply(&mut normed, seq);
        let mut hidden = self.fc1.forward(&normed, seq);
        ops::gelu_erf(&mut hidden);
        let projected = self.fc2.forward(&hidden, seq);
        for (value, add) in residual.iter_mut().zip(projected) {
            *value += add;
        }
        residual
    }
}

/// Whisper-style sinusoidal position table `[length, channels]`, the
/// upstream `sinusoids` helper: `max_timescale` 10000, `sin` features in
/// the first half and `cos` in the second.
fn sinusoids(length: usize, channels: usize) -> Vec<f32> {
    let half = channels / 2;
    let log_timescale = 10_000f32.ln() / (half - 1) as f32;
    let inv_timescales: Vec<f32> = (0..half)
        .map(|index| (-log_timescale * index as f32).exp())
        .collect();
    let mut out = vec![0.0f32; length * channels];
    for time in 0..length {
        for (index, &inv) in inv_timescales.iter().enumerate() {
            let angle = time as f32 * inv;
            out[time * channels + index] = angle.sin();
            out[time * channels + half + index] = angle.cos();
        }
    }
    out
}

/// The loaded audio tower plus the linear projector to the LM width.
pub(crate) struct AudioTower {
    conv1: TowerConv,
    conv2: TowerConv,
    layers: Vec<EncoderLayer>,
    layer_norm: LayerNorm,
    embed_positions: Vec<f32>,
    d_model: usize,
}

impl AudioTower {
    /// Loads the tower from the checkpoint shards under the raw
    /// `audio_tower.` prefix.
    pub(crate) fn load(
        files: &[SafetensorsFile],
        d_model: usize,
        encoder_layers: usize,
        encoder_heads: usize,
        encoder_ffn_dim: usize,
        num_mel_bins: usize,
        max_source_positions: usize,
    ) -> Result<Self> {
        let layers = (0..encoder_layers)
            .map(|index| {
                EncoderLayer::load(
                    files,
                    &format!("audio_tower.layers.{index}"),
                    d_model,
                    encoder_ffn_dim,
                    encoder_heads,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv1: TowerConv::load(files, "audio_tower.conv1", num_mel_bins, d_model)?,
            conv2: TowerConv::load(files, "audio_tower.conv2", d_model, d_model)?,
            layers,
            layer_norm: load_layer_norm_sharded(files, "audio_tower.layer_norm", d_model)?,
            embed_positions: sinusoids(max_source_positions + 1, d_model),
            d_model,
        })
    }

    /// Encodes band-major mel input `[num_mels, frames]` into the pooled,
    /// normalized tower output `[pooled_frames, d_model]`. This is the
    /// projector input, not the projected features.
    pub(crate) fn forward(&self, input_features: &[f32], frames: usize) -> Vec<f32> {
        let mut x = self.conv1.forward(input_features, 1);
        ops::gelu_erf(&mut x);
        let conv_frames = (frames + 2 - 3) / 2 + 1;
        let mut x = self.conv2.forward(&x, 2);
        ops::gelu_erf(&mut x);

        // To time-major rows and add the position table.
        let mut rows = vec![0.0f32; conv_frames * self.d_model];
        for time in 0..conv_frames {
            for channel in 0..self.d_model {
                rows[time * self.d_model + channel] = x[channel * conv_frames + time]
                    + self.embed_positions[time * self.d_model + channel];
            }
        }

        for layer in &self.layers {
            rows = layer.forward(&rows, conv_frames);
        }

        // Non-causal pair average (kernel 2, stride 2): mean of rows
        // 2t and 2t + 1. The conv output length is even by construction.
        let pooled = conv_frames / 2;
        let mut pooled_rows = vec![0.0f32; pooled * self.d_model];
        for time in 0..pooled {
            for channel in 0..self.d_model {
                let a = rows[(2 * time) * self.d_model + channel];
                let b = rows[(2 * time + 1) * self.d_model + channel];
                pooled_rows[time * self.d_model + channel] = (a + b) / 2.0;
            }
        }
        self.layer_norm.apply(&mut pooled_rows, pooled);
        pooled_rows
    }
}

/// `Linear -> LM width` multimodal projector (biased).
pub(crate) struct Projector {
    linear: Linear,
}

impl Projector {
    pub(crate) fn load(
        files: &[SafetensorsFile],
        d_model: usize,
        hidden_size: usize,
    ) -> Result<Self> {
        Ok(Self {
            linear: load_linear_sharded(
                files,
                "multi_modal_projector.linear",
                d_model,
                hidden_size,
                true,
            )?,
        })
    }

    pub(crate) fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        self.linear.forward(x, rows)
    }
}

#[cfg(test)]
mod tests {
    use super::sinusoids;

    /// The position table reproduces the closed form at spot values, and
    /// matches the upstream shape contract (max positions + 1 rows).
    #[test]
    fn sinusoids_match_the_closed_form() {
        let table = sinusoids(9, 8);
        assert_eq!(table.len(), 9 * 8);
        // t = 0: sin 0, ..., cos 0.
        assert_eq!(table[0], 0.0);
        assert_eq!(table[4], 1.0);
        // t = 1, feature 1: inv = 10000^(-2 / 6); sin leads, cos follows
        // in the second half.
        let angle = 10_000f32.powf(-2.0 / 6.0);
        assert!((table[8 + 1] - angle.sin()).abs() < 1.0e-6);
        assert!((table[8 + 4 + 1] - angle.cos()).abs() < 1.0e-6);
    }
}
