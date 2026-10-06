//! MOSS whisper audio encoder.
//!
//! Reference: `MossWhisperEncoder` and glmasr's `WhisperEncoderLayer` in
//! `mlx_audio/stt/models/moss_transcribe_diarize/moss_transcribe_diarize.py`
//! and `mlx_audio/stt/models/glmasr/glmasr.py` at mlx-audio 0.5.7, commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`. The layout is the HF whisper
//! encoder: two stride-1/stride-2 GELU convolutions, learned position
//! embeddings added on the time axis, pre-norm attention/FFN blocks with
//! bias-carrying projections (no RoPE), and a final layer norm.
//!
//! The pinned checkpoint stores this half unquantized (bfloat16), so every
//! tensor here refuses a quantized layout instead of guessing a scheme.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::ops;
use crate::quant::{is_quantized, QuantScheme};
use crate::stt::qwen3_asr::decoder::Linear;
use crate::{Result, SpeechError};

/// Placeholder scheme for [`Linear::load`]; the plain loader below only ever
/// hits the unquantized path where the scheme is unused.
const UNUSED_SCHEME: QuantScheme = QuantScheme {
    bits: 4,
    group_size: 64,
};

fn tensor_error(name: &str, why: impl Into<String>) -> SpeechError {
    SpeechError::Tensor {
        name: name.to_owned(),
        why: why.into(),
    }
}

/// Loads one unquantized linear in the HF `[out, in]` layout. A `.scales`
/// tensor beside the weight means the checkpoint quantized the encoder,
/// which this port has never verified, and is refused.
fn plain_linear(file: &SafetensorsFile, base: &str, input: usize, output: usize) -> Result<Linear> {
    if is_quantized(file, base) {
        return Err(SpeechError::Unsupported {
            why: format!(
                "{base} carries quantization scales; the verified MOSS profile keeps the \
                 whisper encoder and VQ adaptor unquantized"
            ),
        });
    }
    Linear::load(file, base, input, output, UNUSED_SCHEME)
}

/// Loads one plain (F32/F16/BF16) tensor and checks its shape.
fn plain_tensor(file: &SafetensorsFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let descriptor = file
        .descriptor(name)
        .ok_or_else(|| tensor_error(name, "required tensor is missing"))?;
    if descriptor.shape != shape {
        return Err(tensor_error(
            name,
            format!("expected shape {shape:?}, got {:?}", descriptor.shape),
        ));
    }
    if !matches!(descriptor.dtype.as_str(), "F32" | "F16" | "BF16") {
        return Err(SpeechError::Unsupported {
            why: format!(
                "{name} has dtype {}; the verified MOSS profile keeps the encoder plain",
                descriptor.dtype
            ),
        });
    }
    Ok(file.load_as_f32(name)?)
}

/// Conv1d over time-major `[rows, in]` input. MLX checkpoints store the
/// weight as `[out, kernel, in]`; unquantized freshly converted checkpoints
/// may still carry the PyTorch `[out, in, kernel]` layout, which upstream's
/// sanitize transposes at load. Both are accepted here.
struct Conv1d {
    weight: Vec<f32>,
    bias: Vec<f32>,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
}

impl Conv1d {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
    ) -> Result<Self> {
        let weight_name = format!("{prefix}.weight");
        let descriptor = file
            .descriptor(&weight_name)
            .ok_or_else(|| tensor_error(&weight_name, "required tensor is missing"))?;
        if descriptor.shape.len() != 3 {
            return Err(tensor_error(
                &weight_name,
                format!("expected a 3-D weight, got {:?}", descriptor.shape),
            ));
        }
        let raw = file.load_as_f32(&weight_name)?;
        let weight = match descriptor.shape.as_slice() {
            // MLX [out, kernel, in] -> PyTorch [out, in, kernel].
            [o, k, i] if *o == output && *k == kernel && *i == input => {
                let mut weight = vec![0.0; raw.len()];
                for out in 0..output {
                    for channel in 0..input {
                        for tap in 0..kernel {
                            weight[(out * input + channel) * kernel + tap] =
                                raw[(out * kernel + tap) * input + channel];
                        }
                    }
                }
                weight
            }
            // Already PyTorch [out, in, kernel].
            [o, i, k] if *o == output && *i == input && *k == kernel => raw,
            other => {
                return Err(tensor_error(
                    &weight_name,
                    format!("unexpected conv weight shape {other:?}"),
                ))
            }
        };
        let bias = plain_tensor(file, &format!("{prefix}.bias"), &[output])?;
        Ok(Self {
            weight,
            bias,
            input,
            output,
            kernel,
            stride,
            padding: 1,
        })
    }

    fn forward(&self, input: &[f32], rows: usize) -> (Vec<f32>, usize) {
        debug_assert_eq!(input.len(), rows * self.input);
        let mut channels_first = vec![0.0; input.len()];
        for row in 0..rows {
            for channel in 0..self.input {
                channels_first[channel * rows + row] = input[row * self.input + channel];
            }
        }
        let output_rows = (rows + 2 * self.padding - self.kernel) / self.stride + 1;
        let output = ops::conv1d(
            &channels_first,
            &self.weight,
            Some(&self.bias),
            self.input,
            self.output,
            self.kernel,
            self.stride,
            self.padding,
            1,
            1,
        );
        let mut time_major = vec![0.0; output.len()];
        for row in 0..output_rows {
            for channel in 0..self.output {
                time_major[row * self.output + channel] = output[channel * output_rows + row];
            }
        }
        (time_major, output_rows)
    }
}

struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    width: usize,
    epsilon: f32,
}

impl LayerNorm {
    fn load(file: &SafetensorsFile, prefix: &str, width: usize, epsilon: f32) -> Result<Self> {
        Ok(Self {
            weight: plain_tensor(file, &format!("{prefix}.weight"), &[width])?,
            bias: plain_tensor(file, &format!("{prefix}.bias"), &[width])?,
            width,
            epsilon,
        })
    }

    fn apply(&self, values: &mut [f32], rows: usize) {
        ops::layernorm(
            values,
            rows,
            self.width,
            &self.weight,
            Some(&self.bias),
            self.epsilon,
        );
    }
}

fn transpose_heads(input: &[f32], rows: usize, heads: usize, dim: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        for head in 0..heads {
            let source = (row * heads + head) * dim;
            let target = (head * rows + row) * dim;
            output[target..target + dim].copy_from_slice(&input[source..source + dim]);
        }
    }
    output
}

struct WhisperAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    heads: usize,
    head_dim: usize,
}

impl WhisperAttention {
    fn load(file: &SafetensorsFile, prefix: &str, embed_dim: usize, heads: usize) -> Result<Self> {
        let head_dim = embed_dim / heads;
        Ok(Self {
            q_proj: plain_linear(file, &format!("{prefix}.q_proj"), embed_dim, embed_dim)?,
            // The whisper attention k_proj has no bias, q/v/out do.
            k_proj: plain_linear(file, &format!("{prefix}.k_proj"), embed_dim, embed_dim)?,
            v_proj: plain_linear(file, &format!("{prefix}.v_proj"), embed_dim, embed_dim)?,
            out_proj: plain_linear(file, &format!("{prefix}.out_proj"), embed_dim, embed_dim)?,
            heads,
            head_dim,
        })
    }

    fn forward(&self, input: &[f32], rows: usize) -> Vec<f32> {
        let q = self.q_proj.forward(input, rows);
        let k = self.k_proj.forward(input, rows);
        let v = self.v_proj.forward(input, rows);
        let q = transpose_heads(&q, rows, self.heads, self.head_dim);
        let k = transpose_heads(&k, rows, self.heads, self.head_dim);
        let v = transpose_heads(&v, rows, self.heads, self.head_dim);
        let mut attended = vec![0.0; rows * self.heads * self.head_dim];
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        for head in 0..self.heads {
            let head_start = head * rows * self.head_dim;
            let query_head = &q[head_start..head_start + rows * self.head_dim];
            let key_head = &k[head_start..head_start + rows * self.head_dim];
            let value_head = &v[head_start..head_start + rows * self.head_dim];
            for position in 0..rows {
                let query = &query_head[position * self.head_dim..(position + 1) * self.head_dim];
                let result = ops::sdpa(
                    query,
                    key_head,
                    value_head,
                    None,
                    1,
                    rows,
                    self.head_dim,
                    self.head_dim,
                    scale,
                );
                let target = position * self.heads * self.head_dim + head * self.head_dim;
                attended[target..target + self.head_dim].copy_from_slice(&result);
            }
        }
        self.out_proj.forward(&attended, rows)
    }
}

struct WhisperEncoderLayer {
    attention: WhisperAttention,
    attention_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_norm: LayerNorm,
    embed_dim: usize,
}

impl WhisperEncoderLayer {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        embed_dim: usize,
        heads: usize,
        ffn_dim: usize,
    ) -> Result<Self> {
        Ok(Self {
            attention: WhisperAttention::load(
                file,
                &format!("{prefix}.self_attn"),
                embed_dim,
                heads,
            )?,
            attention_norm: LayerNorm::load(
                file,
                &format!("{prefix}.self_attn_layer_norm"),
                embed_dim,
                1e-5,
            )?,
            fc1: plain_linear(file, &format!("{prefix}.fc1"), embed_dim, ffn_dim)?,
            fc2: plain_linear(file, &format!("{prefix}.fc2"), ffn_dim, embed_dim)?,
            final_norm: LayerNorm::load(
                file,
                &format!("{prefix}.final_layer_norm"),
                embed_dim,
                1e-5,
            )?,
            embed_dim,
        })
    }

    fn forward(&self, input: &[f32], rows: usize) -> Vec<f32> {
        let mut normalized = input.to_vec();
        self.attention_norm.apply(&mut normalized, rows);
        let attention = self.attention.forward(&normalized, rows);
        let mut hidden: Vec<f32> = input.iter().zip(attention).map(|(&x, a)| x + a).collect();
        let mut normalized = hidden.clone();
        self.final_norm.apply(&mut normalized, rows);
        let mut feed_forward = self.fc1.forward(&normalized, rows);
        ops::gelu_erf(&mut feed_forward);
        let feed_forward = self.fc2.forward(&feed_forward, rows);
        for (value, add) in hidden.iter_mut().zip(feed_forward) {
            *value += add;
        }
        debug_assert_eq!(hidden.len(), rows * self.embed_dim);
        hidden
    }
}

/// The MOSS whisper encoder: conv1/conv2 with GELU, learned position
/// embeddings, `encoder_layers` pre-norm blocks, and a final layer norm.
pub struct MossWhisperEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    embed_positions: Vec<f32>,
    layers: Vec<WhisperEncoderLayer>,
    layer_norm: LayerNorm,
    d_model: usize,
    max_source_positions: usize,
}

impl MossWhisperEncoder {
    pub fn load(
        file: &SafetensorsFile,
        config: &super::config::WhisperAudioConfig,
    ) -> Result<Self> {
        let embed_dim = config.d_model;
        let prefix = "model.whisper_encoder";
        let conv1 = Conv1d::load(
            file,
            &format!("{prefix}.conv1"),
            config.num_mel_bins,
            embed_dim,
            3,
            1,
        )?;
        let conv2 = Conv1d::load(file, &format!("{prefix}.conv2"), embed_dim, embed_dim, 3, 2)?;
        let embed_positions = plain_tensor(
            file,
            &format!("{prefix}.embed_positions.weight"),
            &[config.max_source_positions, embed_dim],
        )?;
        let layers = (0..config.encoder_layers)
            .map(|index| {
                WhisperEncoderLayer::load(
                    file,
                    &format!("{prefix}.layers.{index}"),
                    embed_dim,
                    config.encoder_attention_heads,
                    config.encoder_ffn_dim,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let layer_norm = LayerNorm::load(file, &format!("{prefix}.layer_norm"), embed_dim, 1e-5)?;
        Ok(Self {
            conv1,
            conv2,
            embed_positions,
            layers,
            layer_norm,
            d_model: embed_dim,
            max_source_positions: config.max_source_positions,
        })
    }

    /// Encodes band-major-per-frame time-major mel input `[frames, 80]` and
    /// returns the final normalized hidden states `[out_rows, d_model]`.
    pub fn forward(&self, mel: &[f32], mel_rows: usize) -> Result<(Vec<f32>, usize)> {
        if mel.len() != mel_rows * self.conv1.input {
            return Err(SpeechError::Tensor {
                name: "MOSS mel features".into(),
                why: format!(
                    "mel input {} does not match {} frames of {} bands",
                    mel.len(),
                    mel_rows,
                    self.conv1.input
                ),
            });
        }
        let (mut hidden, conv1_rows) = self.conv1.forward(mel, mel_rows);
        ops::gelu_erf(&mut hidden);
        let (mut hidden, rows) = self.conv2.forward(&hidden, conv1_rows);
        ops::gelu_erf(&mut hidden);
        if rows > self.max_source_positions {
            return Err(SpeechError::Input {
                why: format!(
                    "MOSS encoder produced {rows} frames, above the {} position limit",
                    self.max_source_positions
                ),
            });
        }
        for row in 0..rows {
            let base = row * self.d_model;
            for (value, position) in hidden[base..base + self.d_model]
                .iter_mut()
                .zip(&self.embed_positions[base..base + self.d_model])
            {
                *value += position;
            }
        }
        for layer in &self.layers {
            hidden = layer.forward(&hidden, rows);
        }
        self.layer_norm.apply(&mut hidden, rows);
        Ok((hidden, rows))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_transpose_round_trips_known_values() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(
            transpose_heads(&x, 2, 2, 2),
            vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
        );
    }

    #[test]
    fn conv_output_length_matches_the_whisper_geometry() {
        // Stride 1 padding 1 kernel 3 keeps 3000 frames; stride 2 halves it.
        let stride = 1;
        let rows_after_conv1 = (3000 + 2 - 3) / stride + 1;
        let rows_after_conv2 = (rows_after_conv1 + 2 - 3) / 2 + 1;
        assert_eq!(rows_after_conv1, 3000);
        assert_eq!(rows_after_conv2, 1500);
    }
}
