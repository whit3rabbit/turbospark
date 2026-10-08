//! Causal audio encoder of Voxtral Realtime: conv stem, 32-layer causal
//! transformer with interleaved RoPE and a 750-token sliding window, and
//! the 4x downsample + MLP adapter to the decoder width.
//!
//! Reference: `mlx_audio/stt/models/voxtral_realtime/encoder.py` at
//! mlx-audio 0.5.7, commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`.
//!
//! Biases are selective and hardcoded like the reference: `wq`, `wv`, and
//! `wo` carry plain bias tensors, `wk` does not; the SwiGLU FFN biases only
//! `w2`. The transformer linears are MLX affine-quantized 4-bit in the
//! pinned checkpoint; the conv stem and the adapter projection are plain
//! F16 and load through the plain path of the shared quant loader.

use turbospark_model_io::safetensors::SafetensorsFile;

use super::RingCache;
use crate::nn::{bad_config, load_tensor, Linear, RmsNorm};
use crate::ops;
use crate::quant::{load_quantized, QuantScheme};
use crate::{Result, SpeechError};

/// Pinned encoder geometry (upstream `EncoderConfig` plus the audio
/// frontend fields the conv stem consumes).
#[derive(Debug, Clone, PartialEq)]
pub struct EncoderGeometry {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub norm_eps: f32,
    pub rope_theta: f32,
    pub sliding_window: usize,
    pub downsample_factor: usize,
}

/// Causal conv1d: left-only zero padding of `kernel - stride`.
struct CausalConv {
    weight: Vec<f32>, // [out_ch, in_ch, kernel]
    bias: Vec<f32>,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
}

impl CausalConv {
    /// Loads the conv from the checkpoint's MLX layout `[out, kernel, in]`
    /// and transposes to the PyTorch-major `[out, in, kernel]` the shared
    /// conv kernel consumes.
    fn load(
        files: &[SafetensorsFile],
        base: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
    ) -> Result<Self> {
        let file = files
            .iter()
            .find(|file| file.contains_tensor(&format!("{base}.weight")))
            .ok_or_else(|| SpeechError::Tensor {
                name: format!("{base}.weight"),
                why: "tensor is missing".into(),
            })?;
        let descriptor =
            file.descriptor(&format!("{base}.weight"))
                .ok_or_else(|| SpeechError::Tensor {
                    name: format!("{base}.weight"),
                    why: "descriptor missing".into(),
                })?;
        if descriptor.shape != [out_ch, kernel, in_ch] {
            return Err(SpeechError::Tensor {
                name: format!("{base}.weight"),
                why: format!(
                    "expected shape [{out_ch}, {kernel}, {in_ch}], got {:?}",
                    descriptor.shape
                ),
            });
        }
        let mlx = file.load_as_f32(&format!("{base}.weight"))?;
        let mut weight = vec![0.0f32; mlx.len()];
        for out in 0..out_ch {
            for k in 0..kernel {
                for input in 0..in_ch {
                    weight[out * in_ch * kernel + input * kernel + k] =
                        mlx[out * kernel * in_ch + k * in_ch + input];
                }
            }
        }
        let bias = load_tensor(file, &format!("{base}.bias"), &[out_ch])?;
        Ok(Self {
            weight,
            bias,
            in_ch,
            out_ch,
            kernel,
            stride,
        })
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let padded = ops::pad_left(x, self.in_ch, self.kernel - self.stride);
        ops::conv1d(
            &padded,
            &self.weight,
            Some(&self.bias),
            self.in_ch,
            self.out_ch,
            self.kernel,
            self.stride,
            0,
            1,
            1,
        )
    }
}

/// One encoder transformer layer.
struct EncoderLayer {
    attention_norm: RmsNorm,
    attention: EncoderAttention,
    ffn_norm: RmsNorm,
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl EncoderLayer {
    fn forward(
        &self,
        x: &[f32],
        rows: usize,
        rope: &super::RopeTables,
        cache: Option<&mut RingCache>,
    ) -> Result<Vec<f32>> {
        let mut normed = x.to_vec();
        self.attention_norm.apply(&mut normed, rows);
        let attended = match cache {
            Some(mut cache) => self
                .attention
                .forward_windowed(&normed, rows, rope, &mut cache)?,
            None => self.attention.forward_causal(&normed, rows, rope)?,
        };
        let mut residual = x.to_vec();
        for (value, add) in residual.iter_mut().zip(attended) {
            *value += add;
        }
        let mut normed = residual.clone();
        self.ffn_norm.apply(&mut normed, rows);
        let mut gate = self.gate.forward(&normed, rows);
        ops::silu(&mut gate);
        let up = self.up.forward(&normed, rows);
        for (gate, up) in gate.iter_mut().zip(up) {
            *gate *= up;
        }
        let down = self.down.forward(&gate, rows);
        for (value, add) in residual.iter_mut().zip(down) {
            *value += add;
        }
        Ok(residual)
    }
}

struct EncoderAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    heads: usize,
    head_dim: usize,
}

impl EncoderAttention {
    fn load(
        files: &[SafetensorsFile],
        prefix: &str,
        dim: usize,
        heads: usize,
        head_dim: usize,
        scheme: QuantScheme,
    ) -> Result<Self> {
        let width = heads * head_dim;
        Ok(Self {
            wq: load_linear(files, &format!("{prefix}.wq"), dim, width, true, scheme)?,
            wk: load_linear(files, &format!("{prefix}.wk"), dim, width, false, scheme)?,
            wv: load_linear(files, &format!("{prefix}.wv"), dim, width, true, scheme)?,
            wo: load_linear(files, &format!("{prefix}.wo"), width, dim, true, scheme)?,
            heads,
            head_dim,
        })
    }

    /// In-window causal attention: query row `i` sees keys `0..=i`.
    fn forward_causal(&self, x: &[f32], rows: usize, rope: &super::RopeTables) -> Result<Vec<f32>> {
        let width = self.heads * self.head_dim;
        let mut query =
            ops::split_heads(&self.wq.forward(x, rows), rows, self.heads, self.head_dim);
        let mut keys = ops::split_heads(&self.wk.forward(x, rows), rows, self.heads, self.head_dim);
        let values = ops::split_heads(&self.wv.forward(x, rows), rows, self.heads, self.head_dim);
        ops::rope_interleaved(
            &mut query,
            self.heads,
            rows,
            self.head_dim,
            &rope.cos,
            &rope.sin,
        );
        // Upstream rotates keys with the same positions as the query
        // (encoder.py rotates both q and k with offset=rope_offset).
        ops::rope_interleaved(
            &mut keys,
            self.heads,
            rows,
            self.head_dim,
            &rope.cos,
            &rope.sin,
        );
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut attended = vec![0.0f32; width * rows];
        for head in 0..self.heads {
            let head_offset = head * rows * self.head_dim;
            for position in 0..rows {
                let query_row = &query[head_offset + position * self.head_dim
                    ..head_offset + (position + 1) * self.head_dim];
                let output = ops::sdpa(
                    query_row,
                    &keys[head_offset..head_offset + (position + 1) * self.head_dim],
                    &values[head_offset..head_offset + (position + 1) * self.head_dim],
                    None,
                    1,
                    position + 1,
                    self.head_dim,
                    self.head_dim,
                    scale,
                );
                let target = position * width + head * self.head_dim;
                attended[target..target + self.head_dim].copy_from_slice(&output);
            }
        }
        Ok(self.wo.forward(&attended, rows))
    }

    /// Sliding-window attention against a ring cache: query row at absolute
    /// position `p` sees the retained cache plus its own causal prefix.
    /// With the ring trimmed to the window size this is exactly the set
    /// `(p - window, p]` the upstream rotating cache mask selects.
    fn forward_windowed(
        &self,
        x: &[f32],
        rows: usize,
        rope: &super::RopeTables,
        cache: &mut RingCache,
    ) -> Result<Vec<f32>> {
        let width = self.heads * self.head_dim;
        let mut query =
            ops::split_heads(&self.wq.forward(x, rows), rows, self.heads, self.head_dim);
        let keys = ops::split_heads(&self.wk.forward(x, rows), rows, self.heads, self.head_dim);
        let values = ops::split_heads(&self.wv.forward(x, rows), rows, self.heads, self.head_dim);
        ops::rope_interleaved(
            &mut query,
            self.heads,
            rows,
            self.head_dim,
            &rope.cos,
            &rope.sin,
        );
        // The cache stores plain (un-rotated) keys upstream; rotation
        // happens per step before attention. Match that: rotate the chunk
        // keys with the chunk positions, then append.
        let mut rotated_keys = keys.clone();
        ops::rope_interleaved(
            &mut rotated_keys,
            self.heads,
            rows,
            self.head_dim,
            &rope.cos,
            &rope.sin,
        );
        cache.append(&rotated_keys, &values, self.heads, rows, self.head_dim);
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        // The ring stays untrimmed while this chunk's queries run (upstream
        // answers query p before any later key exists); the trim happens
        // after every query row of the chunk has attended.
        let mut attended = vec![0.0f32; width * rows];
        for head in 0..self.heads {
            let head_offset = head * rows * self.head_dim;
            for position in 0..rows {
                let query_row = &query[head_offset + position * self.head_dim
                    ..head_offset + (position + 1) * self.head_dim];
                let query_pos = rope.start + position;
                let (lo, hi) = cache.visible_range(query_pos);
                let output = ops::sdpa_strided(
                    query_row,
                    &cache.keys[head],
                    lo * self.head_dim,
                    self.head_dim,
                    &cache.values[head],
                    lo * self.head_dim,
                    self.head_dim,
                    None,
                    1,
                    hi - lo,
                    self.head_dim,
                    self.head_dim,
                    scale,
                );
                let target = position * width + head * self.head_dim;
                attended[target..target + self.head_dim].copy_from_slice(&output);
            }
        }
        cache.trim_to_capacity(self.heads, self.head_dim);
        Ok(self.wo.forward(&attended, rows))
    }
}

/// Load a linear (plain or affine-quantized) from the shard carrying it,
/// with the `[output, input]` shape check.
pub(crate) fn load_linear(
    files: &[SafetensorsFile],
    base: &str,
    input: usize,
    output: usize,
    has_bias: bool,
    scheme: QuantScheme,
) -> Result<Linear> {
    let file = files
        .iter()
        .find(|file| file.contains_tensor(&format!("{base}.weight")))
        .ok_or_else(|| SpeechError::Tensor {
            name: format!("{base}.weight"),
            why: "tensor is missing".into(),
        })?;
    let (weight, bias) = load_quantized(file, base, scheme)?;
    if weight.len() != input * output {
        return Err(SpeechError::Tensor {
            name: format!("{base}.weight"),
            why: format!("expected {} values, got {}", input * output, weight.len()),
        });
    }
    if bias.is_some() != has_bias {
        return Err(SpeechError::Tensor {
            name: format!("{base}.bias"),
            why: format!("expected bias presence {has_bias}"),
        });
    }
    Ok(Linear::new(weight, bias, input, output))
}

/// The full causal audio encoder.
pub(crate) struct AudioEncoder {
    conv0: CausalConv,
    conv1: CausalConv,
    layers: Vec<EncoderLayer>,
    final_norm: RmsNorm,
    proj0: Linear,
    proj2: Linear,
    geometry: EncoderGeometry,
    #[allow(dead_code)]
    decoder_dim: usize,
}

impl AudioEncoder {
    pub(crate) fn load(
        files: &[SafetensorsFile],
        geometry: &EncoderGeometry,
        decoder_dim: usize,
        scheme: QuantScheme,
    ) -> Result<Self> {
        let dim = geometry.dim;
        let layers = (0..geometry.n_layers)
            .map(|index| {
                let prefix = format!("encoder.transformer_layers.{index}");
                Ok(EncoderLayer {
                    attention_norm: super::load_norm(
                        files,
                        &format!("{prefix}.attention_norm"),
                        dim,
                        geometry.norm_eps,
                    )?,
                    attention: EncoderAttention::load(
                        files,
                        &format!("{prefix}.attention"),
                        dim,
                        geometry.n_heads,
                        geometry.head_dim,
                        scheme,
                    )?,
                    ffn_norm: super::load_norm(
                        files,
                        &format!("{prefix}.ffn_norm"),
                        dim,
                        geometry.norm_eps,
                    )?,
                    gate: load_linear(
                        files,
                        &format!("{prefix}.feed_forward_w1"),
                        dim,
                        geometry.hidden_dim,
                        false,
                        scheme,
                    )?,
                    up: load_linear(
                        files,
                        &format!("{prefix}.feed_forward_w3"),
                        dim,
                        geometry.hidden_dim,
                        false,
                        scheme,
                    )?,
                    down: load_linear(
                        files,
                        &format!("{prefix}.feed_forward_w2"),
                        geometry.hidden_dim,
                        dim,
                        true,
                        scheme,
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv0: CausalConv::load(
                files,
                "encoder.conv_layers_0_conv.conv",
                super::N_MELS,
                dim,
                3,
                1,
            )?,
            conv1: CausalConv::load(files, "encoder.conv_layers_1_conv.conv", dim, dim, 3, 2)?,
            layers,
            final_norm: super::load_norm(
                files,
                "encoder.transformer_norm",
                dim,
                geometry.norm_eps,
            )?,
            proj0: super::load_plain_linear_sharded(
                files,
                "encoder.audio_language_projection_0",
                dim * geometry.downsample_factor,
                decoder_dim,
                false,
            )?,
            proj2: super::load_plain_linear_sharded(
                files,
                "encoder.audio_language_projection_2",
                decoder_dim,
                decoder_dim,
                false,
            )?,
            geometry: geometry.clone(),
            decoder_dim,
        })
    }

    /// Runs the two causal conv layers over band-major mel input
    /// `[num_mels, frames]`, applies GELU after each, drops the leading
    /// rows so the length divides the downsample factor, and returns the
    /// row-major conv output `[seq, dim]`.
    pub(crate) fn conv_stem(&self, mel: &[f32]) -> Vec<f32> {
        let _frames = mel.len() / super::N_MELS;
        let mut x = self.conv0.forward(mel); // [dim, frames]
        ops::gelu_erf(&mut x);
        x = self.conv1.forward(&x); // [dim, conv_len]
        ops::gelu_erf(&mut x);
        let conv_len = x.len() / self.geometry.dim;
        let trunc = conv_len % self.geometry.downsample_factor;
        let seq = conv_len - trunc;
        let mut out = vec![0.0f32; seq * self.geometry.dim];
        for channel in 0..self.geometry.dim {
            for (position, slot) in out
                .iter_mut()
                .skip(channel)
                .step_by(self.geometry.dim)
                .enumerate()
            {
                *slot = x[channel * conv_len + trunc + position];
            }
        }
        out
    }

    /// Applies the 32 transformer layers causally to `conv_out
    /// [seq, dim]` and RMS-normalizes. Only valid for `seq <=
    /// sliding_window` (the upstream non-chunked path constraint).
    pub(crate) fn encode_full(&self, conv_out: &[f32], seq: usize) -> Result<Vec<f32>> {
        if seq > self.geometry.sliding_window {
            return Err(bad_config(
                "sliding_window",
                "encode_full requires the conv output to fit the sliding window",
            ));
        }
        let rope = super::RopeTables::new(0, seq, self.geometry.head_dim, self.geometry.rope_theta);
        let mut hidden = conv_out.to_vec();
        for layer in &self.layers {
            hidden = layer.forward(&hidden, seq, &rope, None)?;
        }
        self.final_norm.apply(&mut hidden, seq);
        Ok(hidden)
    }

    /// Chunked encoding with the per-layer 750-entry rotating caches,
    /// mirroring `encode_chunks` + `transformer_norm`. Chunks are exactly
    /// `sliding_window` wide except the trailing one.
    pub(crate) fn encode_chunked(&self, conv_out: &[f32], seq: usize) -> Result<Vec<f32>> {
        let window = self.geometry.sliding_window;
        let mut hidden = Vec::with_capacity(seq * self.geometry.dim);
        let mut caches: Vec<RingCache> = (0..self.geometry.n_layers)
            .map(|_| RingCache::new(self.geometry.n_heads, window))
            .collect();
        for chunk_start in (0..seq).step_by(window) {
            let chunk_len = window.min(seq - chunk_start);
            let chunk = &conv_out
                [chunk_start * self.geometry.dim..(chunk_start + chunk_len) * self.geometry.dim];
            let rope = super::RopeTables::new(
                chunk_start,
                chunk_len,
                self.geometry.head_dim,
                self.geometry.rope_theta,
            );
            let mut current = chunk.to_vec();
            for (index, layer) in self.layers.iter().enumerate() {
                let cache = caches.get_mut(index).expect("one ring cache per layer");
                current = layer.forward(&current, chunk_len, &rope, Some(cache))?;
            }
            let mut normalized = current;
            self.final_norm.apply(&mut normalized, chunk_len);
            hidden.extend_from_slice(&normalized);
        }
        Ok(hidden)
    }

    /// 4x temporal downsample (concatenate consecutive rows) and the
    /// two-layer GELU MLP projection to the decoder width. Returns the
    /// row-major adapter output and its row count.
    pub(crate) fn downsample_and_project(&self, encoded: &[f32], seq: usize) -> (Vec<f32>, usize) {
        let ds = self.geometry.downsample_factor;
        let ds_len = seq / ds;
        if ds_len == 0 {
            return (Vec::new(), 0);
        }
        // Row-major [ds_len, dim * ds] is just the leading rows kept.
        let projected0 = self
            .proj0
            .forward(&encoded[..ds_len * ds * self.geometry.dim], ds_len);
        let mut gated = projected0;
        ops::gelu_erf(&mut gated);
        let projected2 = self.proj2.forward(&gated, ds_len);
        (projected2, ds_len)
    }

    pub(crate) fn geometry(&self) -> &EncoderGeometry {
        &self.geometry
    }

    #[allow(dead_code)]
    pub(crate) fn decoder_dim(&self) -> usize {
        self.decoder_dim
    }
}

#[cfg(test)]
mod tests {
    use super::{AudioEncoder, EncoderGeometry, RingCache};
    use crate::quant::QuantScheme;
    use crate::stt::voxtral_realtime::tests_support::write_safetensors;
    use std::path::Path;

    fn tiny_geometry() -> EncoderGeometry {
        EncoderGeometry {
            dim: 8,
            n_layers: 2,
            n_heads: 2,
            head_dim: 4,
            hidden_dim: 12,
            norm_eps: 1e-5,
            rope_theta: 10_000.0,
            sliding_window: 6,
            downsample_factor: 2,
        }
    }

    fn deterministic(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) as f32 * 0.05
            })
            .collect()
    }

    /// Writes a tiny fake encoder checkpoint with the pinned tensor names;
    /// transformer linears are 4-bit groups of 4 so the shared quant
    /// loader accepts them.
    fn write_fake_encoder(path: &Path) {
        let geometry = tiny_geometry();
        let mut tensors: Vec<(&str, &str, Vec<usize>, Vec<u8>)> = Vec::new();
        let f32le = |values: &[f32]| {
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>()
        };
        let quantized = |tensors: &mut Vec<(&str, &str, Vec<usize>, Vec<u8>)>,
                         base: String,
                         input: usize,
                         output: usize,
                         bias: bool| {
            let groups = input / 4;
            let words_per_row = (input * 4).div_ceil(32);
            let words = vec![0x2144_1214u32; output * words_per_row];
            tensors.push((
                Box::leak(format!("{base}.weight").into_boxed_str()),
                "U32",
                vec![output, words_per_row],
                words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            ));
            tensors.push((
                Box::leak(format!("{base}.scales").into_boxed_str()),
                "F32",
                vec![output, groups],
                f32le(&vec![0.01; output * groups]),
            ));
            tensors.push((
                Box::leak(format!("{base}.biases").into_boxed_str()),
                "F32",
                vec![output, groups],
                f32le(&vec![0.0; output * groups]),
            ));
            if bias {
                tensors.push((
                    Box::leak(format!("{base}.bias").into_boxed_str()),
                    "F32",
                    vec![output],
                    f32le(&vec![0.01; output]),
                ));
            }
        };
        tensors.push((
            "encoder.conv_layers_0_conv.conv.weight",
            "F32",
            // Stored in the MLX layout [out, kernel, in].
            vec![geometry.dim, 3, super::super::N_MELS],
            f32le(&deterministic(geometry.dim * 3 * super::super::N_MELS, 11)),
        ));
        tensors.push((
            "encoder.conv_layers_0_conv.conv.bias",
            "F32",
            vec![geometry.dim],
            f32le(&vec![0.0; geometry.dim]),
        ));
        tensors.push((
            "encoder.conv_layers_1_conv.conv.weight",
            "F32",
            vec![geometry.dim, 3, geometry.dim],
            f32le(&deterministic(geometry.dim * 3 * geometry.dim, 12)),
        ));
        tensors.push((
            "encoder.conv_layers_1_conv.conv.bias",
            "F32",
            vec![geometry.dim],
            f32le(&vec![0.0; geometry.dim]),
        ));
        for index in 0..geometry.n_layers {
            let prefix = format!("encoder.transformer_layers.{index}");
            quantized(
                &mut tensors,
                format!("{prefix}.attention.wq"),
                geometry.dim,
                geometry.n_heads * geometry.head_dim,
                true,
            );
            quantized(
                &mut tensors,
                format!("{prefix}.attention.wk"),
                geometry.dim,
                geometry.n_heads * geometry.head_dim,
                false,
            );
            quantized(
                &mut tensors,
                format!("{prefix}.attention.wv"),
                geometry.dim,
                geometry.n_heads * geometry.head_dim,
                true,
            );
            quantized(
                &mut tensors,
                format!("{prefix}.attention.wo"),
                geometry.n_heads * geometry.head_dim,
                geometry.dim,
                true,
            );
            quantized(
                &mut tensors,
                format!("{prefix}.feed_forward_w1"),
                geometry.dim,
                geometry.hidden_dim,
                false,
            );
            quantized(
                &mut tensors,
                format!("{prefix}.feed_forward_w3"),
                geometry.dim,
                geometry.hidden_dim,
                false,
            );
            quantized(
                &mut tensors,
                format!("{prefix}.feed_forward_w2"),
                geometry.hidden_dim,
                geometry.dim,
                true,
            );
            tensors.push((
                Box::leak(format!("{prefix}.attention_norm.weight").into_boxed_str()),
                "F32",
                vec![geometry.dim],
                f32le(&vec![1.0; geometry.dim]),
            ));
            tensors.push((
                Box::leak(format!("{prefix}.ffn_norm.weight").into_boxed_str()),
                "F32",
                vec![geometry.dim],
                f32le(&vec![1.0; geometry.dim]),
            ));
        }
        tensors.push((
            "encoder.transformer_norm.weight",
            "F32",
            vec![geometry.dim],
            f32le(&vec![1.0; geometry.dim]),
        ));
        tensors.push((
            "encoder.audio_language_projection_0.weight",
            "F32",
            vec![8, geometry.dim * geometry.downsample_factor],
            f32le(&deterministic(
                8 * geometry.dim * geometry.downsample_factor,
                21,
            )),
        ));
        tensors.push((
            "encoder.audio_language_projection_2.weight",
            "F32",
            vec![8, 8],
            f32le(&deterministic(64, 22)),
        ));
        write_safetensors(path, &tensors);
    }

    /// For in-window sequences the chunked ring path must reproduce the
    /// non-chunked causal path bit for bit, and chunking across the window
    /// boundary must track the windowed causal reference.
    #[test]
    fn chunked_encoding_matches_the_causal_reference() {
        let dir = std::env::temp_dir().join(format!("voxtral-encoder-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fake.safetensors");
        write_fake_encoder(&path);
        let file = turbospark_model_io::safetensors::SafetensorsFile::open(&path).unwrap();
        let geometry = tiny_geometry();
        let encoder = AudioEncoder::load(
            &[file],
            &geometry,
            8,
            QuantScheme {
                bits: 4,
                group_size: 4,
            },
        )
        .expect("encoder loads");

        // In-window: 6 conv rows, one chunk both ways.
        let seq = 6usize;
        let conv_out = deterministic(seq * geometry.dim, 42);
        let full = encoder.encode_full(&conv_out, seq).expect("full");
        let chunked = encoder.encode_chunked(&conv_out, seq).expect("chunked");
        if std::env::var_os("VOXTRAL_DEBUG_ATTN").is_some() {
            let layer = &encoder.layers[0];
            let mut normed = conv_out.clone();
            layer.attention_norm.apply(&mut normed, seq);
            let mut query = crate::ops::split_heads(
                &layer.attention.wq.forward(&normed, seq),
                seq,
                geometry.n_heads,
                geometry.head_dim,
            );
            let mut keys = crate::ops::split_heads(
                &layer.attention.wk.forward(&normed, seq),
                seq,
                geometry.n_heads,
                geometry.head_dim,
            );
            let values = crate::ops::split_heads(
                &layer.attention.wv.forward(&normed, seq),
                seq,
                geometry.n_heads,
                geometry.head_dim,
            );
            let rope =
                super::super::RopeTables::new(0, seq, geometry.head_dim, geometry.rope_theta);
            crate::ops::rope_interleaved(
                &mut query,
                geometry.n_heads,
                seq,
                geometry.head_dim,
                &rope.cos,
                &rope.sin,
            );
            crate::ops::rope_interleaved(
                &mut keys,
                geometry.n_heads,
                seq,
                geometry.head_dim,
                &rope.cos,
                &rope.sin,
            );
            let mut ring = RingCache::new(geometry.n_heads, geometry.sliding_window);
            ring.append(&keys, &values, geometry.n_heads, seq, geometry.head_dim);
            let scale = 1.0f32 / (geometry.head_dim as f32).sqrt();
            for position in 0..seq {
                let q_row =
                    &query[position * geometry.head_dim..(position + 1) * geometry.head_dim];
                let via_causal = crate::ops::sdpa(
                    q_row,
                    &keys[..(position + 1) * geometry.head_dim],
                    &values[..(position + 1) * geometry.head_dim],
                    None,
                    1,
                    position + 1,
                    geometry.head_dim,
                    geometry.head_dim,
                    scale,
                );
                let (lo, hi) = ring.visible_range(position);
                let via_ring = crate::ops::sdpa_strided(
                    q_row,
                    &ring.keys[0],
                    lo * geometry.head_dim,
                    geometry.head_dim,
                    &ring.values[0],
                    lo * geometry.head_dim,
                    geometry.head_dim,
                    None,
                    1,
                    hi - lo,
                    geometry.head_dim,
                    geometry.head_dim,
                    scale,
                );
                let same = via_causal
                    .iter()
                    .zip(&via_ring)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                eprintln!(
                    "enc pos {position}: causal {:?} ring {:?} same {same}",
                    via_causal.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    via_ring.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
        if std::env::var_os("VOXTRAL_DEBUG_ATTN").is_some() {
            let layer = &encoder.layers[0];
            let rope =
                super::super::RopeTables::new(0, seq, geometry.head_dim, geometry.rope_theta);
            let out_causal = layer.forward(&conv_out, seq, &rope, None).unwrap();
            let mut ring = RingCache::new(geometry.n_heads, geometry.sliding_window);
            let out_ring = layer
                .forward(&conv_out, seq, &rope, Some(&mut ring))
                .unwrap();
            for (index, (a, b)) in out_causal.iter().zip(&out_ring).enumerate() {
                if a.to_bits() != b.to_bits() {
                    eprintln!(
                        "layer0 diff at {index}: causal {:x} ring {:x}",
                        a.to_bits(),
                        b.to_bits()
                    );
                }
            }
            // Also compare the attention modules directly.
            let mut normed = conv_out.clone();
            layer.attention_norm.apply(&mut normed, seq);
            let att_causal = layer.attention.forward_causal(&normed, seq, &rope).unwrap();
            let att_ring = layer
                .attention
                .forward_windowed(&normed, seq, &rope, &mut ring)
                .unwrap();
            for (index, (a, b)) in att_causal.iter().zip(&att_ring).enumerate() {
                if a.to_bits() != b.to_bits() {
                    eprintln!(
                        "att diff at {index}: causal {:x} ring {:x}",
                        a.to_bits(),
                        b.to_bits()
                    );
                }
            }
        }
        for (index, (a, b)) in full.iter().zip(&chunked).enumerate() {
            if a.to_bits() != b.to_bits() && std::env::var_os("VOXTRAL_DEBUG_ATTN").is_some() {
                eprintln!(
                    "diff at {index}: full {:x} chunked {:x}",
                    a.to_bits(),
                    b.to_bits()
                );
            }
        }
        for (a, b) in full.iter().zip(&chunked) {
            assert_eq!(a.to_bits(), b.to_bits(), "in-window chunked vs full");
        }

        // Across the window boundary: 10 rows, window 6. The chunked path
        // must track the sliding-window causal semantics; compare against
        // an explicit windowed reference over the same rows.
        let seq = 10usize;
        let conv_out = deterministic(seq * geometry.dim, 43);
        let chunked = encoder.encode_chunked(&conv_out, seq).expect("chunked");
        let reference = windowed_reference(&encoder, &conv_out, seq);
        for (a, b) in chunked.iter().zip(&reference) {
            assert_eq!(a.to_bits(), b.to_bits(), "windowed chunked vs reference");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Cache-free windowed causal reference over the conv output.
    fn windowed_reference(encoder: &AudioEncoder, conv_out: &[f32], seq: usize) -> Vec<f32> {
        let geometry = encoder.geometry();
        let mut hidden = conv_out.to_vec();
        for layer in &encoder.layers {
            let mut normed = hidden.clone();
            layer.attention_norm.apply(&mut normed, seq);
            let mut query = crate::ops::split_heads(
                &layer.attention.wq.forward(&normed, seq),
                seq,
                geometry.n_heads,
                geometry.head_dim,
            );
            let mut keys = crate::ops::split_heads(
                &layer.attention.wk.forward(&normed, seq),
                seq,
                geometry.n_heads,
                geometry.head_dim,
            );
            let values = crate::ops::split_heads(
                &layer.attention.wv.forward(&normed, seq),
                seq,
                geometry.n_heads,
                geometry.head_dim,
            );
            let rope =
                super::super::RopeTables::new(0, seq, geometry.head_dim, geometry.rope_theta);
            crate::ops::rope_interleaved(
                &mut query,
                geometry.n_heads,
                seq,
                geometry.head_dim,
                &rope.cos,
                &rope.sin,
            );
            crate::ops::rope_interleaved(
                &mut keys,
                geometry.n_heads,
                seq,
                geometry.head_dim,
                &rope.cos,
                &rope.sin,
            );
            let scale = 1.0 / (geometry.head_dim as f32).sqrt();
            let width = geometry.n_heads * geometry.head_dim;
            let mut attended = vec![0.0f32; width * seq];
            for head in 0..geometry.n_heads {
                for position in 0..seq {
                    let lo = (position + 1).saturating_sub(geometry.sliding_window);
                    let visible = position + 1 - lo;
                    let q_row = &query[(head * seq + position) * geometry.head_dim
                        ..(head * seq + position + 1) * geometry.head_dim];
                    let output = crate::ops::sdpa(
                        q_row,
                        &keys[(head * seq + lo) * geometry.head_dim
                            ..(head * seq + position + 1) * geometry.head_dim],
                        &values[(head * seq + lo) * geometry.head_dim
                            ..(head * seq + position + 1) * geometry.head_dim],
                        None,
                        1,
                        visible,
                        geometry.head_dim,
                        geometry.head_dim,
                        scale,
                    );
                    let target = position * width + head * geometry.head_dim;
                    attended[target..target + geometry.head_dim].copy_from_slice(&output);
                }
            }
            let attended = layer.attention.wo.forward(&attended, seq);
            let mut residual = hidden.clone();
            for (value, add) in residual.iter_mut().zip(attended) {
                *value += add;
            }
            let mut normed = residual.clone();
            layer.ffn_norm.apply(&mut normed, seq);
            let mut gate = layer.gate.forward(&normed, seq);
            crate::ops::silu(&mut gate);
            let up = layer.up.forward(&normed, seq);
            for (gate, up) in gate.iter_mut().zip(up) {
                *gate *= up;
            }
            let down = layer.down.forward(&gate, seq);
            for (value, add) in residual.iter_mut().zip(down) {
                *value += add;
            }
            hidden = residual;
        }
        encoder.final_norm.apply(&mut hidden, seq);
        hidden
    }
}
