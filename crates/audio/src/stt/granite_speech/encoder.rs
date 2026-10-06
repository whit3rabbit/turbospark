//! Granite Speech 1B Conformer encoder.
//!
//! Reference: `mlx_audio/stt/models/granite_speech/granite_speech.py`
//! (`CTCEncoder`, `ConformerBlock`, `ConformerAttention`,
//! `ConformerFeedForward`, `ConformerConvModule`, `DepthWiseConv1d`,
//! `BatchNorm1d`) at mlx-audio 0.5.7, commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`.
//!
//! Layout conventions: linear weights stay in the checkpoint's `[out, in]`
//! layout and conv1d weights in the PyTorch `[out, in / groups, kernel]`
//! layout (this pin stores torch layout; the reference sanitize transposes
//! it to MLX's `[out, kernel, in]`, which is the transpose of what this
//! port's kernels already consume). The attention, feed-forward, and norm
//! bodies run row-major `[time, channel]`; the convolution module transposes
//! to channel-major `[channel, time]` and back.
//!
//! The attention is context-blocked multi-query attention: the zero-padded
//! sequence splits into `context_size` blocks, every block shares one
//! relative-position embedding over `2 * max_pos_emb + 1` entries indexed
//! by clipped in-block distance, and only the trailing block's positional
//! logits are masked to the float minimum for rows or columns past the
//! unpadded remainder.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::ops;
use crate::{Result, SpeechError};

/// MLX `nn.LayerNorm` / `BatchNorm1d` default epsilon in the encoder.
const NORM_EPS: f32 = 1e-5;

fn missing(name: &str) -> SpeechError {
    SpeechError::Tensor {
        name: name.to_owned(),
        why: "missing from the checkpoint shards".to_owned(),
    }
}

fn shard_of<'a>(files: &'a [SafetensorsFile], name: &str) -> Result<&'a SafetensorsFile> {
    files
        .iter()
        .find(|file| file.contains_tensor(name))
        .ok_or_else(|| missing(name))
}

fn load_vector(files: &[SafetensorsFile], name: &str, len: usize) -> Result<Vec<f32>> {
    let values = shard_of(files, name)?.load_as_f32(name)?;
    if values.len() != len {
        return Err(SpeechError::Tensor {
            name: name.to_owned(),
            why: format!("expected {len} values, got {}", values.len()),
        });
    }
    Ok(values)
}

/// Loads one linear weight in `[out, in]` layout plus its bias.
fn load_linear(
    files: &[SafetensorsFile],
    base: &str,
    input: usize,
    output: usize,
) -> Result<Linear> {
    Ok(Linear {
        weight: load_vector(files, &format!("{base}.weight"), input * output)?,
        bias: Some(load_vector(files, &format!("{base}.bias"), output)?),
        input,
        output,
    })
}

/// Loads one bias-free linear weight in `[out, in]` layout, refusing a
/// checkpoint that carries a bias the reference projection does not have.
fn load_bias_free_linear(
    files: &[SafetensorsFile],
    base: &str,
    input: usize,
    output: usize,
) -> Result<Linear> {
    let bias_name = format!("{base}.bias");
    if files.iter().any(|file| file.contains_tensor(&bias_name)) {
        return Err(SpeechError::Unsupported {
            why: format!("{base} must stay bias-free like the reference projection"),
        });
    }
    Ok(Linear {
        weight: load_vector(files, &format!("{base}.weight"), input * output)?,
        bias: None,
        input,
        output,
    })
}

#[derive(Debug, Clone)]
struct Linear {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    input: usize,
    output: usize,
}

impl Linear {
    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        ops::linear(
            x,
            &self.weight,
            self.bias.as_deref(),
            rows,
            self.input,
            self.output,
        )
    }

    /// Applies the linear over channel-major input `[input, frames]`,
    /// returning channel-major `[output, frames]`.
    fn forward_channels(&self, x: &[f32], frames: usize) -> Vec<f32> {
        debug_assert_eq!(x.len() % self.input, 0);
        let rows_layout = transpose_to_rows(x, frames, self.input);
        let out = self.forward(&rows_layout, frames);
        transpose_to_channels(&out, frames, self.output)
    }
}

struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    width: usize,
}

impl LayerNorm {
    fn load(files: &[SafetensorsFile], base: &str, width: usize) -> Result<Self> {
        Ok(Self {
            weight: load_vector(files, &format!("{base}.weight"), width)?,
            bias: load_vector(files, &format!("{base}.bias"), width)?,
            width,
        })
    }

    fn apply(&self, values: &mut [f32], rows: usize) {
        ops::layernorm(
            values,
            rows,
            self.width,
            &self.weight,
            Some(&self.bias),
            NORM_EPS,
        );
    }
}

/// BatchNorm1d in inference form, precomputed to per-channel scale and
/// shift: `(x - mean) / sqrt(var + eps) * weight + bias`.
struct BatchNorm {
    scale: Vec<f32>,
    shift: Vec<f32>,
    channels: usize,
}

impl BatchNorm {
    fn load(files: &[SafetensorsFile], base: &str, channels: usize) -> Result<Self> {
        let weight = load_vector(files, &format!("{base}.weight"), channels)?;
        let bias = load_vector(files, &format!("{base}.bias"), channels)?;
        let mean = load_vector(files, &format!("{base}.running_mean"), channels)?;
        let var = load_vector(files, &format!("{base}.running_var"), channels)?;
        let scale: Vec<f32> = weight
            .iter()
            .zip(var.iter())
            .map(|(w, v)| w / (v + NORM_EPS).sqrt())
            .collect();
        let shift: Vec<f32> = bias
            .iter()
            .zip(mean.iter().zip(scale.iter()))
            .map(|(b, (m, s))| b - m * s)
            .collect();
        Ok(Self {
            scale,
            shift,
            channels,
        })
    }

    /// Applies over channel-major `[channel, time]`.
    fn apply(&self, values: &mut [f32], frames: usize) {
        for channel in 0..self.channels {
            let scale = self.scale[channel];
            let shift = self.shift[channel];
            for value in &mut values[channel * frames..(channel + 1) * frames] {
                *value = *value * scale + shift;
            }
        }
    }
}

struct ConformerAttention {
    pre_norm: LayerNorm,
    to_q: Linear,
    to_kv: Linear,
    to_out: Linear,
    rel_pos_emb: Vec<f32>,
    heads: usize,
    dim_head: usize,
    dim: usize,
    context_size: usize,
    attention_dists: Vec<i32>,
}

impl ConformerAttention {
    fn load(
        files: &[SafetensorsFile],
        base: &str,
        config: &super::config::EncoderConfig,
    ) -> Result<Self> {
        let inner = config.num_heads * config.dim_head;
        Ok(Self {
            pre_norm: LayerNorm::load(files, &format!("{base}.pre_norm"), config.hidden_dim)?,
            to_q: load_bias_free_linear(files, &format!("{base}.to_q"), config.hidden_dim, inner)?,
            to_kv: load_bias_free_linear(
                files,
                &format!("{base}.to_kv"),
                config.hidden_dim,
                inner * 2,
            )?,
            to_out: load_linear(files, &format!("{base}.to_out"), inner, config.hidden_dim)?,
            rel_pos_emb: load_vector(
                files,
                &format!("{base}.rel_pos_emb.weight"),
                (2 * config.max_pos_emb + 1) * config.dim_head,
            )?,
            heads: config.num_heads,
            dim_head: config.dim_head,
            dim: config.hidden_dim,
            context_size: config.context_size,
            attention_dists: reference_attention_dists(config.context_size, config.max_pos_emb),
        })
    }

    /// Context-blocked attention over `x [rows, dim]`; returns
    /// `[rows, dim]` with the padding rows sliced away.
    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut normalized = x.to_vec();
        self.pre_norm.apply(&mut normalized, rows);
        let blocks = rows.div_ceil(self.context_size);
        let padded = blocks * self.context_size;
        let mut input = vec![0.0f32; padded * self.dim];
        input[..rows * self.dim].copy_from_slice(&normalized);
        let q = self.to_q.forward(&input, padded);
        let kv = self.to_kv.forward(&input, padded);
        let (k, v) = split_kv_halves(&kv, padded, self.dim);
        let c = self.context_size;
        let scale = (self.dim_head as f32).recip().sqrt();
        let remainder = rows % c;
        let mask_last_block = remainder != 0;
        // attended is [padded, heads, dim_head] flattened in block, head,
        // row order; regrouped below into row-major [padded, dim].
        let mut attended = Vec::with_capacity(padded * self.dim);
        for block in 0..blocks {
            let row_base = block * c;
            for head in 0..self.heads {
                let head_offset = head * self.dim_head;
                for i in 0..c {
                    let query_start = (row_base + i) * self.dim + head_offset;
                    let query = &q[query_start..query_start + self.dim_head];
                    let mut scores = vec![0.0f32; c];
                    for (j, score) in scores.iter_mut().enumerate() {
                        let key_start = (row_base + j) * self.dim + head_offset;
                        let key = &k[key_start..key_start + self.dim_head];
                        let content: f32 = query.iter().zip(key).map(|(a, b)| a * b).sum();
                        let dist = self.attention_dists[i * c + j] as usize;
                        let rel_start = dist * self.dim_head;
                        let rel = &self.rel_pos_emb[rel_start..rel_start + self.dim_head];
                        let positional: f32 =
                            query.iter().zip(rel).map(|(a, b)| a * b).sum::<f32>() * scale;
                        *score = content * scale + positional;
                    }
                    if mask_last_block && block + 1 == blocks {
                        // The reference masks the trailing block's positional
                        // logits to the dtype minimum for rows or columns
                        // past the unpadded remainder.
                        for (j, score) in scores.iter_mut().enumerate() {
                            if i >= remainder || j >= remainder {
                                *score = f32::MIN;
                            }
                        }
                    }
                    ops::softmax_row(&mut scores);
                    let mut out = vec![0.0f32; self.dim_head];
                    for (j, &weight) in scores.iter().enumerate() {
                        if weight == 0.0 {
                            continue;
                        }
                        let value_start = (row_base + j) * self.dim + head_offset;
                        let value = &v[value_start..value_start + self.dim_head];
                        for (o, val) in out.iter_mut().zip(value) {
                            *o += weight * val;
                        }
                    }
                    attended.extend_from_slice(&out);
                }
            }
        }
        let mut regrouped = vec![0.0f32; padded * self.dim];
        let mut cursor = 0usize;
        for block in 0..blocks {
            for head in 0..self.heads {
                for i in 0..c {
                    let row = block * c + i;
                    let target = row * self.dim + head * self.dim_head;
                    regrouped[target..target + self.dim_head]
                        .copy_from_slice(&attended[cursor..cursor + self.dim_head]);
                    cursor += self.dim_head;
                }
            }
        }
        regrouped.truncate(rows * self.dim);
        self.to_out.forward(&regrouped, rows)
    }
}

/// Splits a `[rows, 2 * dim]` fused projection into its column halves,
/// the reference `k, v = mx.split(kv, 2, axis=-1)`.
fn split_kv_halves(kv: &[f32], rows: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    let mut k = vec![0.0f32; rows * dim];
    let mut v = vec![0.0f32; rows * dim];
    for row in 0..rows {
        let source = row * 2 * dim;
        k[row * dim..(row + 1) * dim].copy_from_slice(&kv[source..source + dim]);
        v[row * dim..(row + 1) * dim].copy_from_slice(&kv[source + dim..source + 2 * dim]);
    }
    (k, v)
}

/// The shared per-block distance table: `clip(i - j, -context, context)
/// + max_pos_emb` over `context x context` block-local positions.
fn reference_attention_dists(context_size: usize, max_pos_emb: usize) -> Vec<i32> {
    let mut dists = vec![0i32; context_size * context_size];
    for i in 0..context_size {
        for j in 0..context_size {
            let distance = i as i64 - j as i64;
            let clipped = distance.clamp(-(context_size as i64), context_size as i64);
            dists[i * context_size + j] = (clipped + max_pos_emb as i64) as i32;
        }
    }
    dists
}

struct ConformerFeedForward {
    pre_norm: LayerNorm,
    up: Linear,
    down: Linear,
}

impl ConformerFeedForward {
    fn load(
        files: &[SafetensorsFile],
        base: &str,
        config: &super::config::EncoderConfig,
    ) -> Result<Self> {
        let wide = config.hidden_dim * config.feedforward_mult;
        Ok(Self {
            pre_norm: LayerNorm::load(files, &format!("{base}.pre_norm"), config.hidden_dim)?,
            up: load_linear(files, &format!("{base}.up_proj"), config.hidden_dim, wide)?,
            down: load_linear(files, &format!("{base}.down_proj"), wide, config.hidden_dim)?,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut normalized = x.to_vec();
        self.pre_norm.apply(&mut normalized, rows);
        let mut wide = self.up.forward(&normalized, rows);
        ops::silu(&mut wide);
        self.down.forward(&wide, rows)
    }
}

struct ConformerConvModule {
    norm: LayerNorm,
    up_conv: Linear,
    depth_conv_weight: Vec<f32>,
    batch_norm: BatchNorm,
    down_conv: Linear,
    channels: usize,
    kernel: usize,
    hidden: usize,
}

impl ConformerConvModule {
    fn load(
        files: &[SafetensorsFile],
        base: &str,
        config: &super::config::EncoderConfig,
    ) -> Result<Self> {
        let inner = config.hidden_dim * config.conv_expansion_factor;
        Ok(Self {
            norm: LayerNorm::load(files, &format!("{base}.norm"), config.hidden_dim)?,
            up_conv: load_linear(
                files,
                &format!("{base}.up_conv"),
                config.hidden_dim,
                inner * 2,
            )?,
            depth_conv_weight: load_vector(
                files,
                &format!("{base}.depth_conv.conv.weight"),
                inner * config.conv_kernel_size,
            )?,
            batch_norm: BatchNorm::load(files, &format!("{base}.batch_norm"), inner)?,
            down_conv: load_linear(
                files,
                &format!("{base}.down_conv"),
                inner,
                config.hidden_dim,
            )?,
            channels: inner,
            kernel: config.conv_kernel_size,
            hidden: config.hidden_dim,
        })
    }

    /// GLU-conv branch over `x [rows, hidden]`, returning `[rows, hidden]`.
    fn forward(&self, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        let mut normalized = x.to_vec();
        self.norm.apply(&mut normalized, rows);
        // Channel-major [hidden, rows] for the pointwise convolutions.
        let channels = transpose_to_channels(&normalized, rows, self.hidden);
        let wide = self.up_conv.forward_channels(&channels, rows);
        // GLU: the first half gates the second half channelwise.
        let mut gated = vec![0.0f32; self.channels * rows];
        for channel in 0..self.channels {
            let a = &wide[channel * rows..(channel + 1) * rows];
            let b = &wide[(self.channels + channel) * rows..(self.channels + channel + 1) * rows];
            for (out, (va, vb)) in gated[channel * rows..(channel + 1) * rows]
                .iter_mut()
                .zip(a.iter().zip(b))
            {
                *out = va / (1.0 + (-vb).exp());
            }
        }
        // Depthwise conv with the reference's symmetric kernel/2 padding.
        let mut deep = ops::conv1d(
            &gated,
            &self.depth_conv_weight,
            None,
            self.channels,
            self.channels,
            self.kernel,
            1,
            self.kernel / 2,
            1,
            self.channels,
        );
        self.batch_norm.apply(&mut deep, rows);
        ops::silu(&mut deep);
        let projected = self.down_conv.forward_channels(&deep, rows);
        Ok(transpose_to_rows(&projected, rows, self.hidden))
    }
}

fn transpose_to_channels(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            out[col * rows + row] = x[row * cols + col];
        }
    }
    out
}

fn transpose_to_rows(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for col in 0..cols {
        for row in 0..rows {
            out[row * cols + col] = x[col * rows + row];
        }
    }
    out
}

struct ConformerBlock {
    ff1: ConformerFeedForward,
    attn: ConformerAttention,
    conv: ConformerConvModule,
    ff2: ConformerFeedForward,
    post_norm: LayerNorm,
}

impl ConformerBlock {
    fn load(
        files: &[SafetensorsFile],
        base: &str,
        config: &super::config::EncoderConfig,
    ) -> Result<Self> {
        Ok(Self {
            ff1: ConformerFeedForward::load(files, &format!("{base}.ff1"), config)?,
            attn: ConformerAttention::load(files, &format!("{base}.attn"), config)?,
            conv: ConformerConvModule::load(files, &format!("{base}.conv"), config)?,
            ff2: ConformerFeedForward::load(files, &format!("{base}.ff2"), config)?,
            post_norm: LayerNorm::load(files, &format!("{base}.post_norm"), config.hidden_dim)?,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        // x = 0.5 * ff1(x) + x
        let ff1 = self.ff1.forward(x, rows);
        let mut x: Vec<f32> = x.iter().zip(ff1).map(|(r, b)| r + 0.5 * b).collect();
        // x = attn(x) + x
        let attended = self.attn.forward(&x, rows);
        for (value, add) in x.iter_mut().zip(attended) {
            *value += add;
        }
        // x = conv(x) + x
        let conv = self.conv.forward(&x, rows)?;
        for (value, add) in x.iter_mut().zip(conv) {
            *value += add;
        }
        // x = 0.5 * ff2(x) + x
        let ff2 = self.ff2.forward(&x, rows);
        for (value, add) in x.iter_mut().zip(ff2) {
            *value += 0.5 * add;
        }
        self.post_norm.apply(&mut x, rows);
        Ok(x)
    }
}

/// The CTC-style encoder tower: input projection, `num_layers` Conformer
/// blocks with midpoint output self-conditioning, returned in the hidden
/// width (the CTC head itself stays unused by the ASR path).
pub struct GraniteSpeechEncoder {
    input_linear: Linear,
    layers: Vec<ConformerBlock>,
    out: Linear,
    out_mid: Linear,
    config: super::config::EncoderConfig,
}

impl GraniteSpeechEncoder {
    pub(crate) fn load(
        files: &[SafetensorsFile],
        config: &super::config::EncoderConfig,
    ) -> Result<Self> {
        let layers = (0..config.num_layers)
            .map(|index| ConformerBlock::load(files, &format!("encoder.layers.{index}"), config))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            input_linear: load_linear(
                files,
                "encoder.input_linear",
                config.input_dim,
                config.hidden_dim,
            )?,
            layers,
            out: load_linear(files, "encoder.out", config.hidden_dim, config.output_dim)?,
            out_mid: load_linear(
                files,
                "encoder.out_mid",
                config.output_dim,
                config.hidden_dim,
            )?,
            config: config.clone(),
        })
    }

    /// Encodes the paired mel plane `[rows, input_dim]` row-major into
    /// `[rows, hidden_dim]`.
    pub fn forward(&self, features: &[f32], rows: usize) -> Result<Vec<f32>> {
        if features.len() != rows * self.config.input_dim {
            return Err(SpeechError::Input {
                why: format!(
                    "expected {rows} x {} features, got {}",
                    self.config.input_dim,
                    features.len()
                ),
            });
        }
        let mut x = self.input_linear.forward(features, rows);
        for (index, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, rows)?;
            // Reference midpoint self-conditioning after layer
            // num_layers / 2 (1-based enumerate).
            if index + 1 == self.config.num_layers / 2 {
                let mid = self.out.forward(&x, rows);
                let mut soft = mid;
                for row in 0..rows {
                    ops::softmax_row(
                        &mut soft[row * self.config.output_dim..(row + 1) * self.config.output_dim],
                    );
                }
                let back = self.out_mid.forward(&soft, rows);
                for (value, add) in x.iter_mut().zip(back) {
                    *value += add;
                }
            }
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::reference_attention_dists;

    #[test]
    fn attention_dists_clip_and_offset_block_local_distance() {
        // Row i is the query, column j the key: the reference builds
        // `seq[:, None] - seq[None, :]`, so entry (i, j) is clip(i - j),
        // offset by 6: rows read [6, 5, 4, 3], [7, 6, 5, 4], and so on.
        let dists = reference_attention_dists(4, 6);
        assert_eq!(dists, vec![6, 5, 4, 3, 7, 6, 5, 4, 8, 7, 6, 5, 9, 8, 7, 6]);
        let big = reference_attention_dists(2, 1);
        assert_eq!(big, vec![1, 0, 2, 1]);
    }

    #[test]
    fn kv_split_takes_column_halves_not_flat_prefixes() {
        // Row-major [2 rows, 4 cols]; k = cols 0..2 of each row, v = cols
        // 2..4 of each row. A flat split_at would interleave both halves.
        let kv: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (k, v) = super::split_kv_halves(&kv, 2, 2);
        assert_eq!(k, vec![1.0, 2.0, 5.0, 6.0]);
        assert_eq!(v, vec![3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn channel_transposes_round_trip() {
        let x: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let channels = super::transpose_to_channels(&x, 2, 3);
        assert_eq!(channels, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        assert_eq!(super::transpose_to_rows(&channels, 2, 3), x);
    }
}
