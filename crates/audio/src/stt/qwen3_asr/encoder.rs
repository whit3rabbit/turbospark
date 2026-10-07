//! Qwen3-ASR chunked 128-band audio encoder.
//!
//! Transcribed from `mlx_audio/stt/models/qwen3_asr/qwen3_asr.py` at the
//! pinned mlx-audio 0.5.7 commit. Conv2d checkpoint weights are converted
//! from MLX `[out, kh, kw, in]` to the shared CPU kernel's PyTorch layout.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::models::stt::qwen3_asr::config::AudioEncoderConfig;
use crate::nn::{LayerNorm, Linear};
use crate::ops;
use crate::{Result, SpeechError};

use super::frontend::{AudioFeatures, MEL_BINS};

const LAYER_NORM_EPS: f32 = 1.0e-5;
const CONV_KERNEL: usize = 3;
const CONV_STRIDE: usize = 2;
const CONV_PADDING: usize = 1;

fn tensor_name(file: &SafetensorsFile, name: &str) -> String {
    if file.contains_tensor(name) {
        name.to_owned()
    } else {
        let prefixed = format!("thinker.{name}");
        if file.contains_tensor(&prefixed) {
            prefixed
        } else {
            name.to_owned()
        }
    }
}

fn load_tensor(file: &SafetensorsFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let name = tensor_name(file, name);
    let desc = file.descriptor(&name).ok_or_else(|| SpeechError::Tensor {
        name: name.clone(),
        why: "required tensor is missing".into(),
    })?;
    if desc.shape != shape {
        return Err(SpeechError::Tensor {
            name,
            why: format!("expected shape {shape:?}, got {:?}", desc.shape),
        });
    }
    file.load_as_f32(&name).map_err(Into::into)
}

// The checkpoint's tensor-name remapping and error text stay local, so the
// shared layer types are built here from this family's own `load_tensor`.
fn load_linear(
    file: &SafetensorsFile,
    prefix: &str,
    input: usize,
    output: usize,
    has_bias: bool,
) -> Result<Linear> {
    let weight = load_tensor(file, &format!("{prefix}.weight"), &[output, input])?;
    let bias = if has_bias {
        Some(load_tensor(file, &format!("{prefix}.bias"), &[output])?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias, input, output))
}

fn load_layer_norm(file: &SafetensorsFile, prefix: &str, width: usize) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        load_tensor(file, &format!("{prefix}.weight"), &[width])?,
        Some(load_tensor(file, &format!("{prefix}.bias"), &[width])?),
        LAYER_NORM_EPS,
    ))
}

struct Conv2d {
    weight: Vec<f32>,
    bias: Vec<f32>,
    input_channels: usize,
    output_channels: usize,
}

impl Conv2d {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
    ) -> Result<Self> {
        let name = tensor_name(file, &format!("{prefix}.weight"));
        let desc = file.descriptor(&name).ok_or_else(|| SpeechError::Tensor {
            name: name.clone(),
            why: "required convolution weight is missing".into(),
        })?;
        let raw = file.load_as_f32(&name)?;
        let mlx_shape = [output_channels, CONV_KERNEL, CONV_KERNEL, input_channels];
        let pytorch_shape = [output_channels, input_channels, CONV_KERNEL, CONV_KERNEL];
        let weight = if desc.shape == mlx_shape {
            let mut converted = vec![0.0f32; raw.len()];
            for output in 0..output_channels {
                for input in 0..input_channels {
                    for ky in 0..CONV_KERNEL {
                        for kx in 0..CONV_KERNEL {
                            let mlx_index = ((output * CONV_KERNEL + ky) * CONV_KERNEL + kx)
                                * input_channels
                                + input;
                            let pytorch_index = ((output * input_channels + input) * CONV_KERNEL
                                + ky)
                                * CONV_KERNEL
                                + kx;
                            converted[pytorch_index] = raw[mlx_index];
                        }
                    }
                }
            }
            converted
        } else if desc.shape == pytorch_shape {
            raw
        } else {
            return Err(SpeechError::Tensor {
                name,
                why: format!(
                    "expected MLX shape {mlx_shape:?} or PyTorch shape {pytorch_shape:?}, got {:?}",
                    desc.shape
                ),
            });
        };
        let bias = load_tensor(file, &format!("{prefix}.bias"), &[output_channels])?;
        Ok(Self {
            weight,
            bias,
            input_channels,
            output_channels,
        })
    }

    fn forward(&self, x: &[f32], height: usize, width: usize) -> (Vec<f32>, usize, usize) {
        let out_height = (height + 2 * CONV_PADDING - CONV_KERNEL) / CONV_STRIDE + 1;
        let out_width = (width + 2 * CONV_PADDING - CONV_KERNEL) / CONV_STRIDE + 1;
        let output = ops::conv2d(
            x,
            &self.weight,
            Some(&self.bias),
            self.input_channels,
            self.output_channels,
            height,
            width,
            CONV_KERNEL,
            CONV_KERNEL,
            CONV_STRIDE,
            CONV_PADDING,
            1,
        );
        (output, out_height, out_width)
    }
}

struct AudioLayer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    attention_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_norm: LayerNorm,
    hidden: usize,
    heads: usize,
    head_dim: usize,
}

impl AudioLayer {
    fn load(file: &SafetensorsFile, prefix: &str, config: &AudioEncoderConfig) -> Result<Self> {
        let hidden = config.d_model;
        let attention = format!("{prefix}.self_attn");
        Ok(Self {
            q_proj: load_linear(file, &format!("{attention}.q_proj"), hidden, hidden, true)?,
            k_proj: load_linear(file, &format!("{attention}.k_proj"), hidden, hidden, true)?,
            v_proj: load_linear(file, &format!("{attention}.v_proj"), hidden, hidden, true)?,
            out_proj: load_linear(file, &format!("{attention}.out_proj"), hidden, hidden, true)?,
            attention_norm: load_layer_norm(
                file,
                &format!("{prefix}.self_attn_layer_norm"),
                hidden,
            )?,
            fc1: load_linear(
                file,
                &format!("{prefix}.fc1"),
                hidden,
                config.encoder_ffn_dim,
                true,
            )?,
            fc2: load_linear(
                file,
                &format!("{prefix}.fc2"),
                config.encoder_ffn_dim,
                hidden,
                true,
            )?,
            final_norm: load_layer_norm(file, &format!("{prefix}.final_layer_norm"), hidden)?,
            hidden,
            heads: config.encoder_attention_heads,
            head_dim: hidden / config.encoder_attention_heads,
        })
    }

    fn forward(&self, x: &[f32], rows: usize, block_size: usize) -> Vec<f32> {
        let mut normalized = x.to_vec();
        self.attention_norm.apply(&mut normalized, rows);
        let mut queries = self.q_proj.forward(&normalized, rows);
        let keys = self.k_proj.forward(&normalized, rows);
        let values = self.v_proj.forward(&normalized, rows);
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        for query in &mut queries {
            *query *= scale;
        }

        let mut attended = vec![0.0f32; rows * self.hidden];
        for block_start in (0..rows).step_by(block_size) {
            let block_end = (block_start + block_size).min(rows);
            let block_rows = block_end - block_start;
            for head in 0..self.heads {
                let head_offset = head * self.head_dim;
                let mut key_head = Vec::with_capacity(block_rows * self.head_dim);
                let mut value_head = Vec::with_capacity(block_rows * self.head_dim);
                for row in block_start..block_end {
                    let offset = row * self.hidden + head_offset;
                    key_head.extend_from_slice(&keys[offset..offset + self.head_dim]);
                    value_head.extend_from_slice(&values[offset..offset + self.head_dim]);
                }
                for query_row in block_start..block_end {
                    let offset = query_row * self.hidden + head_offset;
                    let query = &queries[offset..offset + self.head_dim];
                    let result = ops::sdpa(
                        query,
                        &key_head,
                        &value_head,
                        None,
                        1,
                        block_rows,
                        self.head_dim,
                        self.head_dim,
                        1.0,
                    );
                    attended[offset..offset + self.head_dim].copy_from_slice(&result);
                }
            }
        }
        let projected = self.out_proj.forward(&attended, rows);
        let mut residual = x.to_vec();
        for (value, add) in residual.iter_mut().zip(projected) {
            *value += add;
        }

        let mut normalized = residual.clone();
        self.final_norm.apply(&mut normalized, rows);
        let mut feed_forward = self.fc1.forward(&normalized, rows);
        ops::gelu_erf(&mut feed_forward);
        let feed_forward = self.fc2.forward(&feed_forward, rows);
        for (value, add) in residual.iter_mut().zip(feed_forward) {
            *value += add;
        }
        residual
    }
}

pub struct AudioEncoder {
    convs: [Conv2d; 3],
    conv_out: Linear,
    layers: Vec<AudioLayer>,
    post_norm: LayerNorm,
    proj1: Linear,
    proj2: Linear,
    config: AudioEncoderConfig,
}

impl AudioEncoder {
    pub fn load(file: &SafetensorsFile, config: &AudioEncoderConfig) -> Result<Self> {
        let width = config.downsample_hidden_size;
        let convs = [
            Conv2d::load(file, "audio_tower.conv2d1", 1, width)?,
            Conv2d::load(file, "audio_tower.conv2d2", width, width)?,
            Conv2d::load(file, "audio_tower.conv2d3", width, width)?,
        ];
        let frequency = conv_output_length(conv_output_length(conv_output_length(MEL_BINS)));
        let conv_out = load_linear(
            file,
            "audio_tower.conv_out",
            width * frequency,
            config.d_model,
            false,
        )?;
        let layers = (0..config.encoder_layers)
            .map(|index| AudioLayer::load(file, &format!("audio_tower.layers.{index}"), config))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            convs,
            conv_out,
            layers,
            post_norm: load_layer_norm(file, "audio_tower.ln_post", config.d_model)?,
            proj1: load_linear(
                file,
                "audio_tower.proj1",
                config.d_model,
                config.d_model,
                true,
            )?,
            proj2: load_linear(
                file,
                "audio_tower.proj2",
                config.d_model,
                config.output_dim,
                true,
            )?,
            config: config.clone(),
        })
    }

    /// Conv stack and time-major reshape of one chunk: the `[width, channels *
    /// frequency]` rows fed to `conv_out`, and `width`.
    fn conv_stage(
        &self,
        features: &AudioFeatures,
        index: usize,
        chunk_size: usize,
        valid_input_frames: usize,
        max_chunk_len: usize,
        frequency: usize,
    ) -> Result<(Vec<f32>, usize)> {
        let start_frame = index * chunk_size;
        let mut chunk = vec![0.0f32; MEL_BINS * max_chunk_len];
        for mel in 0..MEL_BINS {
            let source = mel * features.frames + start_frame;
            let target = mel * max_chunk_len;
            chunk[target..target + valid_input_frames]
                .copy_from_slice(&features.values[source..source + valid_input_frames]);
        }

        let mut values = chunk;
        let mut height = MEL_BINS;
        let mut width = max_chunk_len;
        for conv in &self.convs {
            let (mut output, out_height, out_width) = conv.forward(&values, height, width);
            ops::gelu_erf(&mut output);
            values = output;
            height = out_height;
            width = out_width;
        }
        if height != frequency {
            return Err(SpeechError::Tensor {
                name: "audio_tower.conv2d".into(),
                why: format!("frequency width {height} != expected {frequency}"),
            });
        }

        let mut rows = vec![0.0f32; width * self.config.downsample_hidden_size * frequency];
        for time in 0..width {
            for channel in 0..self.config.downsample_hidden_size {
                for mel in 0..frequency {
                    let source = (channel * frequency + mel) * width + time;
                    let target = time * (self.config.downsample_hidden_size * frequency)
                        + channel * frequency
                        + mel;
                    rows[target] = values[source];
                }
            }
        }
        Ok((rows, width))
    }

    pub fn forward(&self, features: &AudioFeatures) -> Result<Vec<f32>> {
        if features.frames == 0 || features.values.len() != MEL_BINS * features.frames {
            return Err(SpeechError::Input {
                why: "Qwen3 audio features must have shape [128, frames]".into(),
            });
        }
        let chunk_size = 2 * self.config.n_window;
        let chunks = features.frames.div_ceil(chunk_size);
        let chunk_lengths = (0..chunks)
            .map(|index| (features.frames - index * chunk_size).min(chunk_size))
            .collect::<Vec<_>>();
        let max_chunk_len = *chunk_lengths.iter().max().unwrap_or(&0);
        let max_conv_frames =
            conv_output_length(conv_output_length(conv_output_length(max_chunk_len)));
        if max_conv_frames > self.config.max_source_positions {
            return Err(SpeechError::Input {
                why: "Qwen3 audio chunk exceeds positional embedding capacity".into(),
            });
        }

        let frequency = conv_output_length(conv_output_length(conv_output_length(MEL_BINS)));
        let mut hidden_states = Vec::new();
        let mut valid_lengths = Vec::with_capacity(chunks);
        let positions = sinusoidal_positions(max_conv_frames, self.config.d_model);
        let embedding_scale = if self.config.scale_embedding {
            (self.config.d_model as f32).sqrt()
        } else {
            1.0
        };

        // The conv stack dominates this stage and `ops::conv2d` is single
        // threaded, so chunks (independent of each other) run on separate
        // threads in batches of one per core. Each chunk is computed by the
        // same code and consumed in chunk order below, so the values and the
        // order in which errors surface are unchanged. The linear that
        // follows threads itself, so it stays on this thread.
        let batch = ops::thread_count().max(1);
        for batch_start in (0..chunks).step_by(batch) {
            let batch_len = batch.min(chunks - batch_start);
            let staged = ops::par_map(batch_len, |offset| {
                self.conv_stage(
                    features,
                    batch_start + offset,
                    chunk_size,
                    chunk_lengths[batch_start + offset],
                    max_chunk_len,
                    frequency,
                )
            });
            for (offset, staged) in staged.into_iter().enumerate() {
                let valid_input_frames = chunk_lengths[batch_start + offset];
                let (rows, width) = staged?;
                let mut projected = self.conv_out.forward(&rows, width);
                for (row, embedding) in projected.chunks_exact_mut(self.config.d_model).enumerate()
                {
                    for (dim, value) in embedding.iter_mut().enumerate() {
                        *value =
                            *value * embedding_scale + positions[row * self.config.d_model + dim];
                    }
                }

                let valid_output_frames = subsampled_length(valid_input_frames);
                if valid_output_frames > width {
                    return Err(SpeechError::Tensor {
                        name: "audio_tower.conv2d".into(),
                        why: format!(
                            "subsampled length {valid_output_frames} exceeds padded width {width}"
                        ),
                    });
                }
                hidden_states
                    .extend_from_slice(&projected[..valid_output_frames * self.config.d_model]);
                valid_lengths.push(valid_output_frames);
            }
        }

        let rows = valid_lengths.iter().sum::<usize>();
        let max_valid = *valid_lengths.iter().max().unwrap_or(&0);
        let attention_window = max_valid * (self.config.n_window_infer / chunk_size);
        if rows == 0 || attention_window == 0 {
            return Err(SpeechError::Input {
                why: "Qwen3 audio encoder produced no valid frames".into(),
            });
        }
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, rows, attention_window);
        }
        self.post_norm.apply(&mut hidden_states, rows);
        let mut output = self.proj1.forward(&hidden_states, rows);
        ops::gelu_erf(&mut output);
        Ok(self.proj2.forward(&output, rows))
    }
}

/// Three stride-2, same-padded convolutions, which each return `ceil(n / 2)`.
fn conv_output_length(length: usize) -> usize {
    length.div_ceil(2)
}

/// Reference `_get_feat_extract_output_lengths`, which treats every full
/// 100-frame chunk as exactly 13 encoder rows.
fn subsampled_length(length: usize) -> usize {
    let remainder = length % 100;
    let tail = if remainder == 0 {
        0
    } else {
        conv_output_length(conv_output_length(conv_output_length(remainder)))
    };
    (length / 100) * 13 + tail
}

fn sinusoidal_positions(rows: usize, width: usize) -> Vec<f32> {
    let half = width / 2;
    let increment = 10_000.0f32.ln() / (half - 1) as f32;
    let inverse = (0..half)
        .map(|index| (-increment * index as f32).exp())
        .collect::<Vec<_>>();
    let mut output = vec![0.0f32; rows * width];
    for row in 0..rows {
        for index in 0..half {
            let angle = row as f32 * inverse[index];
            output[row * width + index] = angle.sin();
            output[row * width + half + index] = angle.cos();
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::{conv_output_length, sinusoidal_positions, subsampled_length};
    use super::{
        AudioEncoder, AudioEncoderConfig, AudioFeatures, AudioLayer, Conv2d, LayerNorm, Linear,
        Result, SpeechError, CONV_KERNEL, LAYER_NORM_EPS, MEL_BINS,
    };
    use crate::ops;

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

        fn linear(&mut self, input: usize, output: usize) -> Linear {
            Linear::new(
                self.vec(input * output, 1.2 / (input as f32).sqrt()),
                Some(self.vec(output, 0.2)),
                input,
                output,
            )
        }

        fn norm(&mut self, width: usize) -> LayerNorm {
            let weight = self.vec(width, 0.3).iter().map(|w| w + 1.0).collect();
            LayerNorm::new(weight, Some(self.vec(width, 0.1)), LAYER_NORM_EPS)
        }

        fn conv(&mut self, input: usize, output: usize) -> Conv2d {
            Conv2d {
                weight: self.vec(output * input * CONV_KERNEL * CONV_KERNEL, 0.3),
                bias: self.vec(output, 0.1),
                input_channels: input,
                output_channels: output,
            }
        }
    }

    /// Tiny geometry with the real structure: 8-frame chunks (n_window 4)
    /// and a short last chunk.
    fn encoder(rng: &mut Rng) -> AudioEncoder {
        let (width, d_model, heads, ffn) = (3usize, 8usize, 2usize, 16usize);
        let config = AudioEncoderConfig {
            num_mel_bins: MEL_BINS,
            encoder_layers: 2,
            encoder_attention_heads: heads,
            encoder_ffn_dim: ffn,
            d_model,
            max_source_positions: 64,
            n_window: 4,
            output_dim: 6,
            n_window_infer: 16,
            downsample_hidden_size: width,
            scale_embedding: true,
        };
        let frequency = conv_output_length(conv_output_length(conv_output_length(MEL_BINS)));
        AudioEncoder {
            convs: [
                rng.conv(1, width),
                rng.conv(width, width),
                rng.conv(width, width),
            ],
            conv_out: Linear::new(
                rng.vec(width * frequency * d_model, 0.2),
                None,
                width * frequency,
                d_model,
            ),
            layers: (0..2)
                .map(|_| AudioLayer {
                    q_proj: rng.linear(d_model, d_model),
                    k_proj: rng.linear(d_model, d_model),
                    v_proj: rng.linear(d_model, d_model),
                    out_proj: rng.linear(d_model, d_model),
                    attention_norm: rng.norm(d_model),
                    fc1: rng.linear(d_model, ffn),
                    fc2: rng.linear(ffn, d_model),
                    final_norm: rng.norm(d_model),
                    hidden: d_model,
                    heads,
                    head_dim: d_model / heads,
                })
                .collect(),
            post_norm: rng.norm(d_model),
            proj1: rng.linear(d_model, d_model),
            proj2: rng.linear(d_model, 6),
            config,
        }
    }

    /// The previous serial chunk loop, kept verbatim.
    fn reference_forward(self_: &AudioEncoder, features: &AudioFeatures) -> Result<Vec<f32>> {
        if features.frames == 0 || features.values.len() != MEL_BINS * features.frames {
            return Err(SpeechError::Input {
                why: "Qwen3 audio features must have shape [128, frames]".into(),
            });
        }
        let chunk_size = 2 * self_.config.n_window;
        let chunks = features.frames.div_ceil(chunk_size);
        let chunk_lengths = (0..chunks)
            .map(|index| (features.frames - index * chunk_size).min(chunk_size))
            .collect::<Vec<_>>();
        let max_chunk_len = *chunk_lengths.iter().max().unwrap_or(&0);
        let max_conv_frames =
            conv_output_length(conv_output_length(conv_output_length(max_chunk_len)));
        if max_conv_frames > self_.config.max_source_positions {
            return Err(SpeechError::Input {
                why: "Qwen3 audio chunk exceeds positional embedding capacity".into(),
            });
        }

        let frequency = conv_output_length(conv_output_length(conv_output_length(MEL_BINS)));
        let mut hidden_states = Vec::new();
        let mut valid_lengths = Vec::with_capacity(chunks);
        let positions = sinusoidal_positions(max_conv_frames, self_.config.d_model);
        let embedding_scale = if self_.config.scale_embedding {
            (self_.config.d_model as f32).sqrt()
        } else {
            1.0
        };

        for (index, &valid_input_frames) in chunk_lengths.iter().enumerate() {
            let start_frame = index * chunk_size;
            let mut chunk = vec![0.0f32; MEL_BINS * max_chunk_len];
            for mel in 0..MEL_BINS {
                let source = mel * features.frames + start_frame;
                let target = mel * max_chunk_len;
                chunk[target..target + valid_input_frames]
                    .copy_from_slice(&features.values[source..source + valid_input_frames]);
            }

            let mut values = chunk;
            let mut height = MEL_BINS;
            let mut width = max_chunk_len;
            for conv in &self_.convs {
                let (mut output, out_height, out_width) = conv.forward(&values, height, width);
                ops::gelu_erf(&mut output);
                values = output;
                height = out_height;
                width = out_width;
            }
            if height != frequency {
                return Err(SpeechError::Tensor {
                    name: "audio_tower.conv2d".into(),
                    why: format!("frequency width {height} != expected {frequency}"),
                });
            }

            let mut rows = vec![0.0f32; width * self_.config.downsample_hidden_size * frequency];
            for time in 0..width {
                for channel in 0..self_.config.downsample_hidden_size {
                    for mel in 0..frequency {
                        let source = (channel * frequency + mel) * width + time;
                        let target = time * (self_.config.downsample_hidden_size * frequency)
                            + channel * frequency
                            + mel;
                        rows[target] = values[source];
                    }
                }
            }
            let mut projected = self_.conv_out.forward(&rows, width);
            for (row, embedding) in projected.chunks_exact_mut(self_.config.d_model).enumerate() {
                for (dim, value) in embedding.iter_mut().enumerate() {
                    *value = *value * embedding_scale + positions[row * self_.config.d_model + dim];
                }
            }

            let valid_output_frames = subsampled_length(valid_input_frames);
            if valid_output_frames > width {
                return Err(SpeechError::Tensor {
                    name: "audio_tower.conv2d".into(),
                    why: format!(
                        "subsampled length {valid_output_frames} exceeds padded width {width}"
                    ),
                });
            }
            hidden_states
                .extend_from_slice(&projected[..valid_output_frames * self_.config.d_model]);
            valid_lengths.push(valid_output_frames);
        }

        let rows = valid_lengths.iter().sum::<usize>();
        let max_valid = *valid_lengths.iter().max().unwrap_or(&0);
        let attention_window = max_valid * (self_.config.n_window_infer / chunk_size);
        if rows == 0 || attention_window == 0 {
            return Err(SpeechError::Input {
                why: "Qwen3 audio encoder produced no valid frames".into(),
            });
        }
        for layer in &self_.layers {
            hidden_states = layer.forward(&hidden_states, rows, attention_window);
        }
        self_.post_norm.apply(&mut hidden_states, rows);
        let mut output = self_.proj1.forward(&hidden_states, rows);
        ops::gelu_erf(&mut output);
        Ok(self_.proj2.forward(&output, rows))
    }

    #[test]
    fn parallel_chunk_stage_matches_the_serial_loop_bitwise() {
        // 38 chunks (the last one short) span several per-core batches.
        for (seed, frames) in [(5u64, 301usize), (9, 8), (13, 61)] {
            let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
            let encoder = encoder(&mut rng);
            let features = AudioFeatures {
                values: rng.vec(MEL_BINS * frames, 2.0),
                frames,
            };
            let want = reference_forward(&encoder, &features).unwrap();
            let got = encoder.forward(&features).unwrap();
            assert_eq!(got.len(), want.len());
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "frames {frames} element {i}");
            }
        }
    }

    #[test]
    fn chunk_output_lengths_match_qwen3_reference_formula() {
        for (frames, expected) in [(1, 1), (79, 10), (100, 13), (179, 23), (200, 26), (279, 36)] {
            assert_eq!(subsampled_length(frames), expected, "frames={frames}");
        }
        assert_eq!(conv_output_length(128), 64);
    }

    #[test]
    fn sinusoidal_positions_are_sin_then_cos_and_start_at_zero() {
        let pos = sinusoidal_positions(2, 4);
        assert_eq!(&pos[..4], &[0.0, 0.0, 1.0, 1.0]);
        assert!((pos[4] - 1.0f32.sin()).abs() < 1e-6);
        assert!((pos[6] - 1.0f32.cos()).abs() < 1e-6);
    }

    #[test]
    #[ignore = "requires the pinned Qwen3-ASR checkpoint in TURBOSPARK_QWEN3_ASR_DIR"]
    fn pinned_audio_tower_matches_mlx_fixture() {
        use serde::Deserialize;
        use serde_json::Value;
        use std::path::Path;
        use turbospark_model_io::safetensors::SafetensorsFile;

        use crate::models::stt::qwen3_asr::config::Qwen3Config;
        use crate::models::stt::qwen3_asr::frontend::{compute_features, SAMPLE_RATE};

        #[derive(Deserialize)]
        struct Fixture {
            output_shape: [usize; 2],
            selected_rows: Vec<usize>,
            selected_values: Vec<Vec<f32>>,
        }

        let model_dir = std::env::var_os("TURBOSPARK_QWEN3_ASR_DIR")
            .expect("set TURBOSPARK_QWEN3_ASR_DIR to the pinned checkpoint directory");
        let model_dir = Path::new(&model_dir);
        let root: Value =
            serde_json::from_slice(&std::fs::read(model_dir.join("config.json")).unwrap()).unwrap();
        let config = Qwen3Config::from_json(&root).unwrap();
        let weights = SafetensorsFile::open(&model_dir.join("model.safetensors")).unwrap();
        let encoder = super::AudioEncoder::load(&weights, &config.audio).unwrap();

        let tau = std::f64::consts::TAU;
        let samples = (0..44_720)
            .map(|index| {
                let seconds = index as f64 / SAMPLE_RATE as f64;
                (0.1 * (tau * 440.0 * seconds).sin() + 0.03 * (tau * 997.0 * seconds).sin()) as f32
            })
            .collect::<Vec<_>>();
        let features = compute_features(&samples).unwrap();
        let output = encoder.forward(&features).unwrap();
        let fixture: Fixture =
            serde_json::from_str(include_str!("../../../testdata/qwen3_asr_encoder.json")).unwrap();
        assert_eq!(
            fixture.output_shape,
            [
                output.len() / config.audio.output_dim,
                config.audio.output_dim
            ]
        );
        assert_eq!(fixture.selected_rows.len(), fixture.selected_values.len());

        let mut max_error = 0.0f32;
        let mut total_error = 0.0f64;
        let mut compared = 0usize;
        for (row, expected) in fixture.selected_rows.iter().zip(&fixture.selected_values) {
            assert_eq!(expected.len(), config.audio.output_dim);
            for (column, &want) in expected.iter().enumerate() {
                let got = output[row * config.audio.output_dim + column];
                let error = (got - want).abs();
                max_error = max_error.max(error);
                total_error += error as f64;
                compared += 1;
            }
        }
        eprintln!(
            "Qwen3 audio tower parity: max_abs_error={max_error:.6}, mean_abs_error={:.6}",
            total_error / compared as f64
        );
        assert!(
            max_error <= 0.05,
            "max error {max_error} exceeds fixture tolerance"
        );
    }
}
