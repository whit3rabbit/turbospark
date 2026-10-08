//! Mistral-style LLM decoder of Voxtral Realtime: 26 layers of
//! grouped-query attention (32 query heads, 8 KV heads, head_dim 128) with
//! interleaved RoPE, an 8192-token sliding window, time-conditioned
//! adaptive RMSNorm, SwiGLU FFN, and tied token embeddings as the logits
//! head.
//!
//! Reference: `mlx_audio/stt/models/voxtral_realtime/decoder.py` at
//! mlx-audio 0.5.7, commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`.
//!
//! No decoder linear carries a bias. The big linears are MLX
//! affine-quantized 4-bit in the pinned checkpoint; the token embedding
//! matrix is plain F16 there and doubles as the LM head through
//! `logits = h @ E^T` (upstream `Embedding.as_linear`). The adaptive RMS
//! scales are computed once per delay value from the sinusoidal time
//! embedding (`compute_time_embedding`), matching the upstream
//! `precompute_ada_scales` post-load hook.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::nn::{load_tensor, Linear, RmsNorm};
use crate::ops;
use crate::quant::QuantScheme;
use crate::{Result, SpeechError};

use super::encoder::load_linear;
use super::{RingCache, RopeTables};

/// Pinned decoder geometry (upstream `DecoderConfig`).
#[derive(Debug, Clone, PartialEq)]
pub struct DecoderGeometry {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub vocab_size: usize,
    pub norm_eps: f32,
    pub rope_theta: f32,
    pub sliding_window: usize,
    pub ada_dim: usize,
}

/// Sinusoidal time embedding for the adaptive RMSNorm conditioning:
/// `concat(cos(t * inv_freq), sin(t * inv_freq))` with
/// `inv_freq = theta^(-d / half)`. `t_value` is the delay token count.
pub(crate) fn compute_time_embedding(t_value: f32, dim: usize, theta: f32) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0.0f32; dim];
    for d in 0..half {
        let freq = (-theta.ln() * d as f32 / half as f32).exp();
        let angle = t_value * freq;
        out[d] = angle.cos();
        out[half + d] = angle.sin();
    }
    out
}

/// Per-layer `Linear(dim -> bottleneck) -> GELU -> Linear(bottleneck ->
/// dim)` producing the adaptive scale applied as `h * (1 + scale)`.
struct AdaRmsNorm {
    down: Linear,
    up: Linear,
}

impl AdaRmsNorm {
    fn load(
        files: &[SafetensorsFile],
        prefix: &str,
        dim: usize,
        bottleneck: usize,
    ) -> Result<Self> {
        Ok(Self {
            down: super::load_plain_linear_sharded(
                files,
                &format!("{prefix}.ada_down"),
                dim,
                bottleneck,
                false,
            )?,
            up: super::load_plain_linear_sharded(
                files,
                &format!("{prefix}.ada_up"),
                bottleneck,
                dim,
                false,
            )?,
        })
    }

    fn compute_scale(&self, t_cond: &[f32]) -> Vec<f32> {
        let mut hidden = self.down.forward(t_cond, 1);
        ops::gelu_erf(&mut hidden);
        self.up.forward(&hidden, 1)
    }
}

struct DecoderAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    #[allow(dead_code)]
    window: usize,
}

impl DecoderAttention {
    fn load(
        files: &[SafetensorsFile],
        prefix: &str,
        geometry: &DecoderGeometry,
        scheme: QuantScheme,
    ) -> Result<Self> {
        let width = geometry.n_heads * geometry.head_dim;
        let kv_width = geometry.n_kv_heads * geometry.head_dim;
        Ok(Self {
            wq: load_linear(
                files,
                &format!("{prefix}.wq"),
                geometry.dim,
                width,
                false,
                scheme,
            )?,
            wk: load_linear(
                files,
                &format!("{prefix}.wk"),
                geometry.dim,
                kv_width,
                false,
                scheme,
            )?,
            wv: load_linear(
                files,
                &format!("{prefix}.wv"),
                geometry.dim,
                kv_width,
                false,
                scheme,
            )?,
            wo: load_linear(
                files,
                &format!("{prefix}.wo"),
                width,
                geometry.dim,
                false,
                scheme,
            )?,
            heads: geometry.n_heads,
            kv_heads: geometry.n_kv_heads,
            head_dim: geometry.head_dim,
            window: geometry.sliding_window,
        })
    }

    /// Causal (sliding-window-bounded) attention over `[rows, dim]`,
    /// appending rotated keys and values to `cache`.
    fn forward(
        &self,
        x: &[f32],
        rows: usize,
        start_pos: usize,
        rope: &RopeTables,
        cache: &mut RingCache,
    ) -> Vec<f32> {
        let width = self.heads * self.head_dim;
        let mut query =
            ops::split_heads(&self.wq.forward(x, rows), rows, self.heads, self.head_dim);
        let mut keys = ops::split_heads(
            &self.wk.forward(x, rows),
            rows,
            self.kv_heads,
            self.head_dim,
        );
        let values = ops::split_heads(
            &self.wv.forward(x, rows),
            rows,
            self.kv_heads,
            self.head_dim,
        );
        ops::rope_interleaved(
            &mut query,
            self.heads,
            rows,
            self.head_dim,
            &rope.cos,
            &rope.sin,
        );
        ops::rope_interleaved(
            &mut keys,
            self.kv_heads,
            rows,
            self.head_dim,
            &rope.cos,
            &rope.sin,
        );
        cache.append(&keys, &values, self.kv_heads, rows, self.head_dim);
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        debug_assert_eq!(self.heads % self.kv_heads, 0);
        let group = self.heads / self.kv_heads;
        let mut attended = vec![0.0f32; width * rows];
        for head in 0..self.heads {
            let kv_head = head / group;
            let head_offset = head * rows * self.head_dim;
            for position in 0..rows {
                let query_row = &query[head_offset + position * self.head_dim
                    ..head_offset + (position + 1) * self.head_dim];
                let (lo, hi) = cache.visible_range(start_pos + position);
                let output = ops::sdpa_strided(
                    query_row,
                    &cache.keys[kv_head],
                    lo * self.head_dim,
                    self.head_dim,
                    &cache.values[kv_head],
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
        cache.trim_to_capacity(self.kv_heads, self.head_dim);
        self.wo.forward(&attended, rows)
    }
}

struct DecoderLayer {
    attention_norm: RmsNorm,
    attention: DecoderAttention,
    ffn_norm: RmsNorm,
    ada: AdaRmsNorm,
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &[f32],
        rows: usize,
        start_pos: usize,
        rope: &RopeTables,
        ada_scale: &[f32],
        cache: &mut RingCache,
    ) -> Vec<f32> {
        let mut normed = x.to_vec();
        self.attention_norm.apply(&mut normed, rows);
        let attended = self
            .attention
            .forward(&normed, rows, start_pos, rope, cache);
        let mut residual = x.to_vec();
        for (value, add) in residual.iter_mut().zip(attended) {
            *value += add;
        }
        let mut normed = residual.clone();
        self.ffn_norm.apply(&mut normed, rows);
        // The adaptive scale is a [dim] vector applied to every row.
        for row in normed.chunks_exact_mut(ada_scale.len()) {
            for (value, scale) in row.iter_mut().zip(ada_scale) {
                *value *= 1.0 + scale;
            }
        }
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
        residual
    }
}

/// Cached decode state shared by the greedy loop.
pub(crate) struct DecodeCache {
    layers: Vec<RingCache>,
}

/// The loaded decoder with its tied embedding head.
pub(crate) struct Decoder {
    embedding: Vec<f32>, // [vocab_size, dim]
    layers: Vec<DecoderLayer>,
    final_norm: RmsNorm,
    ada_scales: Vec<Vec<f32>>,
    geometry: DecoderGeometry,
    /// The delay token count the cached `ada_scales` were computed for.
    ada_delay: i32,
}

impl Decoder {
    pub(crate) fn load(
        files: &[SafetensorsFile],
        geometry: &DecoderGeometry,
        scheme: QuantScheme,
    ) -> Result<Self> {
        let layers = (0..geometry.n_layers)
            .map(|index| {
                let prefix = format!("decoder.layers.{index}");
                Ok(DecoderLayer {
                    attention_norm: super::load_norm(
                        files,
                        &format!("{prefix}.attention_norm"),
                        geometry.dim,
                        geometry.norm_eps,
                    )?,
                    attention: DecoderAttention::load(
                        files,
                        &format!("{prefix}.attention"),
                        geometry,
                        scheme,
                    )?,
                    ffn_norm: super::load_norm(
                        files,
                        &format!("{prefix}.ffn_norm"),
                        geometry.dim,
                        geometry.norm_eps,
                    )?,
                    ada: AdaRmsNorm::load(
                        files,
                        &format!("{prefix}.ada_rms_norm_t_cond"),
                        geometry.dim,
                        geometry.ada_dim,
                    )?,
                    gate: load_linear(
                        files,
                        &format!("{prefix}.feed_forward_w1"),
                        geometry.dim,
                        geometry.hidden_dim,
                        false,
                        scheme,
                    )?,
                    up: load_linear(
                        files,
                        &format!("{prefix}.feed_forward_w3"),
                        geometry.dim,
                        geometry.hidden_dim,
                        false,
                        scheme,
                    )?,
                    down: load_linear(
                        files,
                        &format!("{prefix}.feed_forward_w2"),
                        geometry.hidden_dim,
                        geometry.dim,
                        false,
                        scheme,
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embedding: load_tensor(
                files
                    .iter()
                    .find(|file| file.contains_tensor("decoder.tok_embeddings.weight"))
                    .ok_or_else(|| SpeechError::Tensor {
                        name: "decoder.tok_embeddings.weight".into(),
                        why: "tensor is missing".into(),
                    })?,
                "decoder.tok_embeddings.weight",
                &[geometry.vocab_size, geometry.dim],
            )?,
            layers,
            final_norm: super::load_norm(files, "decoder.norm", geometry.dim, geometry.norm_eps)?,
            ada_scales: Vec::new(),
            geometry: geometry.clone(),
            ada_delay: -1,
        })
    }

    pub(crate) fn geometry(&self) -> &DecoderGeometry {
        &self.geometry
    }

    /// Precomputes the per-layer adaptive scales for a delay token count,
    /// mirroring the upstream post-load hook. Recomputes when the delay
    /// changes.
    pub(crate) fn precompute_ada_scales(&mut self, n_delay_tokens: i32) {
        if self.ada_delay == n_delay_tokens && !self.ada_scales.is_empty() {
            return;
        }
        let t_cond = compute_time_embedding(n_delay_tokens as f32, self.geometry.dim, 10_000.0);
        self.ada_scales = self
            .layers
            .iter()
            .map(|layer| layer.ada.compute_scale(&t_cond))
            .collect();
        self.ada_delay = n_delay_tokens;
    }

    /// Token embedding rows `[ids.len(), dim]`; ids are signed 32-bit per
    /// the crate boundary contract.
    pub(crate) fn embed(&self, ids: &[i32]) -> Result<Vec<f32>> {
        if ids
            .iter()
            .any(|&id| id < 0 || id as usize >= self.geometry.vocab_size)
        {
            return Err(SpeechError::Input {
                why: "token id outside the checkpoint vocabulary".into(),
            });
        }
        Ok(ops::embedding(&self.embedding, self.geometry.dim, ids))
    }

    /// Logits for one post-norm hidden row through the tied embedding
    /// head (`h @ E^T`, the upstream `Embedding.as_linear`).
    pub(crate) fn tied_logits(&self, hidden: &[f32]) -> Vec<f32> {
        ops::linear(
            hidden,
            &self.embedding,
            None,
            1,
            self.geometry.dim,
            self.geometry.vocab_size,
        )
    }

    /// Evaluates the prompt embeddings and returns the final post-norm row
    /// plus the populated cache for the greedy loop.
    pub(crate) fn prefill(&self, embeds: &[f32], rows: usize) -> (Vec<f32>, DecodeCache) {
        let window = self.geometry.sliding_window;
        let mut caches = (0..self.geometry.n_layers)
            .map(|_| RingCache::new(self.geometry.n_kv_heads, window))
            .collect::<Vec<_>>();
        let mut hidden = embeds.to_vec();
        for (index, layer) in self.layers.iter().enumerate() {
            let rope = RopeTables::new(0, rows, self.geometry.head_dim, self.geometry.rope_theta);
            let ada = &self.ada_scales[index];
            let cache = &mut caches[index];
            hidden = layer.forward(&hidden, rows, 0, &rope, ada, cache);
        }
        self.final_norm.apply(&mut hidden, rows);
        let last = hidden[(rows - 1) * self.geometry.dim..].to_vec();
        (last, DecodeCache { layers: caches })
    }

    /// One decode step at absolute `position` against the cache; returns
    /// the post-norm row.
    pub(crate) fn step(&self, embed: &[f32], position: usize, cache: &mut DecodeCache) -> Vec<f32> {
        let rope = RopeTables::new(
            position,
            1,
            self.geometry.head_dim,
            self.geometry.rope_theta,
        );
        let mut hidden = embed.to_vec();
        for (index, layer) in self.layers.iter().enumerate() {
            let ada = &self.ada_scales[index];
            let layer_cache = &mut cache.layers[index];
            hidden = layer.forward(&hidden, 1, position, &rope, ada, layer_cache);
        }
        self.final_norm.apply(&mut hidden, 1);
        hidden
    }
}

#[cfg(test)]
mod tests {
    use super::{compute_time_embedding, Decoder, DecoderGeometry};
    use crate::quant::QuantScheme;
    use crate::stt::voxtral_realtime::tests_support::write_safetensors;
    use crate::stt::voxtral_realtime::RopeTables;
    use std::path::Path;

    fn tiny_geometry() -> DecoderGeometry {
        DecoderGeometry {
            dim: 8,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 2,
            hidden_dim: 12,
            vocab_size: 31,
            norm_eps: 1e-5,
            rope_theta: 10_000.0,
            sliding_window: 6,
            ada_dim: 4,
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

    /// Writes a tiny fake decoder checkpoint with the pinned tensor names.
    /// Weights are 4-bit groups of 4 so the shared quant loader accepts
    /// them under the pinned scheme.
    fn write_fake_decoder(path: &Path) {
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
                         output: usize| {
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
        };
        tensors.push((
            "decoder.tok_embeddings.weight",
            "F32",
            vec![geometry.vocab_size, geometry.dim],
            f32le(&deterministic(geometry.vocab_size * geometry.dim, 7)),
        ));
        tensors.push((
            "decoder.norm.weight",
            "F32",
            vec![geometry.dim],
            f32le(&vec![1.0; geometry.dim]),
        ));
        for index in 0..geometry.n_layers {
            let prefix = format!("decoder.layers.{index}");
            quantized(
                &mut tensors,
                format!("{prefix}.attention.wq"),
                geometry.dim,
                geometry.n_heads * geometry.head_dim,
            );
            quantized(
                &mut tensors,
                format!("{prefix}.attention.wk"),
                geometry.dim,
                geometry.n_kv_heads * geometry.head_dim,
            );
            quantized(
                &mut tensors,
                format!("{prefix}.attention.wv"),
                geometry.dim,
                geometry.n_kv_heads * geometry.head_dim,
            );
            quantized(
                &mut tensors,
                format!("{prefix}.attention.wo"),
                geometry.n_heads * geometry.head_dim,
                geometry.dim,
            );
            quantized(
                &mut tensors,
                format!("{prefix}.feed_forward_w1"),
                geometry.dim,
                geometry.hidden_dim,
            );
            quantized(
                &mut tensors,
                format!("{prefix}.feed_forward_w3"),
                geometry.dim,
                geometry.hidden_dim,
            );
            quantized(
                &mut tensors,
                format!("{prefix}.feed_forward_w2"),
                geometry.hidden_dim,
                geometry.dim,
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
            tensors.push((
                Box::leak(format!("{prefix}.ada_rms_norm_t_cond.ada_down.weight").into_boxed_str()),
                "F32",
                vec![geometry.ada_dim, geometry.dim],
                f32le(&deterministic(
                    geometry.ada_dim * geometry.dim,
                    100 + index as u64,
                )),
            ));
            tensors.push((
                Box::leak(format!("{prefix}.ada_rms_norm_t_cond.ada_up.weight").into_boxed_str()),
                "F32",
                vec![geometry.dim, geometry.ada_dim],
                f32le(&deterministic(
                    geometry.dim * geometry.ada_dim,
                    200 + index as u64,
                )),
            ));
        }
        write_safetensors(path, &tensors);
    }

    /// Cache-free full forward over `embeds` at absolute `start_pos`,
    /// written out longhand so the production cached path has an
    /// independent reference.
    fn full_forward(decoder: &Decoder, embeds: &[f32], start_pos: usize) -> Vec<f32> {
        let geometry = decoder.geometry();
        let rows = embeds.len() / geometry.dim;
        let mut hidden = embeds.to_vec();
        for (index, layer) in decoder.layers.iter().enumerate() {
            let mut normed = hidden.clone();
            layer.attention_norm.apply(&mut normed, rows);
            let mut query = crate::ops::split_heads(
                &layer.attention.wq.forward(&normed, rows),
                rows,
                geometry.n_heads,
                geometry.head_dim,
            );
            let mut keys = crate::ops::split_heads(
                &layer.attention.wk.forward(&normed, rows),
                rows,
                geometry.n_kv_heads,
                geometry.head_dim,
            );
            let values = crate::ops::split_heads(
                &layer.attention.wv.forward(&normed, rows),
                rows,
                geometry.n_kv_heads,
                geometry.head_dim,
            );
            let rope = RopeTables::new(start_pos, rows, geometry.head_dim, geometry.rope_theta);
            crate::ops::rope_interleaved(
                &mut query,
                geometry.n_heads,
                rows,
                geometry.head_dim,
                &rope.cos,
                &rope.sin,
            );
            crate::ops::rope_interleaved(
                &mut keys,
                geometry.n_kv_heads,
                rows,
                geometry.head_dim,
                &rope.cos,
                &rope.sin,
            );
            let scale = 1.0 / (geometry.head_dim as f32).sqrt();
            let group = geometry.n_heads / geometry.n_kv_heads;
            let width = geometry.n_heads * geometry.head_dim;
            let mut attended = vec![0.0f32; width * rows];
            for head in 0..geometry.n_heads {
                let kv_head = head / group;
                for position in 0..rows {
                    let q_row = &query[(head * rows + position) * geometry.head_dim
                        ..(head * rows + position + 1) * geometry.head_dim];
                    let visible = start_pos + position + 1;
                    let output = crate::ops::sdpa(
                        q_row,
                        &keys[(kv_head * rows) * geometry.head_dim
                            ..(kv_head * rows + visible) * geometry.head_dim],
                        &values[(kv_head * rows) * geometry.head_dim
                            ..(kv_head * rows + visible) * geometry.head_dim],
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
            let attended = layer.attention.wo.forward(&attended, rows);
            let mut residual = hidden.clone();
            for (value, add) in residual.iter_mut().zip(attended) {
                *value += add;
            }
            let mut normed = residual.clone();
            layer.ffn_norm.apply(&mut normed, rows);
            for row in normed.chunks_exact_mut(decoder.ada_scales[index].len()) {
                for (value, scale) in row.iter_mut().zip(&decoder.ada_scales[index]) {
                    *value *= 1.0 + scale;
                }
            }
            let mut gate = layer.gate.forward(&normed, rows);
            crate::ops::silu(&mut gate);
            let up = layer.up.forward(&normed, rows);
            for (gate, up) in gate.iter_mut().zip(up) {
                *gate *= up;
            }
            let down = layer.down.forward(&gate, rows);
            if std::env::var_os("VOXTRAL_DEBUG_ATTN").is_some() {
                let normed_slice = &normed[normed.len() - geometry.dim..];
                let gate_slice = &gate[gate.len() - geometry.hidden_dim..];
                let down_slice = &down[down.len() - geometry.dim..];
                eprintln!(
                    "post-ffn rows {rows} start {start_pos}: normed0 {:x} gate0 {:x} down0 {:x}",
                    normed_slice[0].to_bits(),
                    gate_slice[0].to_bits(),
                    down_slice[0].to_bits(),
                );
            }
            for (value, add) in residual.iter_mut().zip(down) {
                *value += add;
            }
            hidden = residual;
        }
        decoder.final_norm.apply(&mut hidden, rows);
        hidden
    }

    /// Cache-free forward where every query attends its causal prefix cut
    /// by the sliding window: the ideal each ring query must reproduce.
    fn windowed_full_forward(decoder: &Decoder, embeds: &[f32]) -> Vec<f32> {
        let geometry = decoder.geometry();
        let rows = embeds.len() / geometry.dim;
        let mut hidden = embeds.to_vec();
        for (index, layer) in decoder.layers.iter().enumerate() {
            let mut normed = hidden.clone();
            layer.attention_norm.apply(&mut normed, rows);
            let mut query = crate::ops::split_heads(
                &layer.attention.wq.forward(&normed, rows),
                rows,
                geometry.n_heads,
                geometry.head_dim,
            );
            let mut keys = crate::ops::split_heads(
                &layer.attention.wk.forward(&normed, rows),
                rows,
                geometry.n_kv_heads,
                geometry.head_dim,
            );
            let values = crate::ops::split_heads(
                &layer.attention.wv.forward(&normed, rows),
                rows,
                geometry.n_kv_heads,
                geometry.head_dim,
            );
            let rope = RopeTables::new(0, rows, geometry.head_dim, geometry.rope_theta);
            crate::ops::rope_interleaved(
                &mut query,
                geometry.n_heads,
                rows,
                geometry.head_dim,
                &rope.cos,
                &rope.sin,
            );
            crate::ops::rope_interleaved(
                &mut keys,
                geometry.n_kv_heads,
                rows,
                geometry.head_dim,
                &rope.cos,
                &rope.sin,
            );
            let scale = 1.0 / (geometry.head_dim as f32).sqrt();
            let group = geometry.n_heads / geometry.n_kv_heads;
            let width = geometry.n_heads * geometry.head_dim;
            let mut attended = vec![0.0f32; width * rows];
            for head in 0..geometry.n_heads {
                let kv_head = head / group;
                for position in 0..rows {
                    let lo = (position + 1).saturating_sub(geometry.sliding_window);
                    let visible = position + 1 - lo;
                    let q_row = &query[(head * rows + position) * geometry.head_dim
                        ..(head * rows + position + 1) * geometry.head_dim];
                    let output = crate::ops::sdpa(
                        q_row,
                        &keys[(kv_head * rows + lo) * geometry.head_dim
                            ..(kv_head * rows + position + 1) * geometry.head_dim],
                        &values[(kv_head * rows + lo) * geometry.head_dim
                            ..(kv_head * rows + position + 1) * geometry.head_dim],
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
            let attended = layer.attention.wo.forward(&attended, rows);
            let mut residual = hidden.clone();
            for (value, add) in residual.iter_mut().zip(attended) {
                *value += add;
            }
            let mut normed = residual.clone();
            layer.ffn_norm.apply(&mut normed, rows);
            for row in normed.chunks_exact_mut(decoder.ada_scales[index].len()) {
                for (value, scale) in row.iter_mut().zip(&decoder.ada_scales[index]) {
                    *value *= 1.0 + scale;
                }
            }
            let mut gate = layer.gate.forward(&normed, rows);
            crate::ops::silu(&mut gate);
            let up = layer.up.forward(&normed, rows);
            for (gate, up) in gate.iter_mut().zip(up) {
                *gate *= up;
            }
            let down = layer.down.forward(&gate, rows);
            for (value, add) in residual.iter_mut().zip(down) {
                *value += add;
            }
            hidden = residual;
        }
        decoder.final_norm.apply(&mut hidden, rows);
        hidden
    }

    /// The cached prefill plus decode steps must reproduce the cache-free
    /// full forward below the sliding window, match the windowed forward
    /// once the ring starts dropping, and the tied head must equal a
    /// direct matmul against the embedding rows.
    #[test]
    fn cached_prefill_and_steps_match_the_references() {
        let dir = std::env::temp_dir().join(format!("voxtral-decoder-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fake.safetensors");
        write_fake_decoder(&path);
        let file = turbospark_model_io::safetensors::SafetensorsFile::open(&path).unwrap();
        let geometry = tiny_geometry();
        let mut decoder = Decoder::load(
            &[file],
            &geometry,
            QuantScheme {
                bits: 4,
                group_size: 4,
            },
        )
        .unwrap();
        decoder.precompute_ada_scales(2);
        assert_eq!(decoder.ada_scales.len(), geometry.n_layers);
        // Same delay keeps the scales; a new delay recomputes them.
        let before = decoder.ada_scales.clone();
        decoder.precompute_ada_scales(2);
        assert_eq!(decoder.ada_scales, before);
        decoder.precompute_ada_scales(3);
        assert_ne!(decoder.ada_scales, before);
        decoder.precompute_ada_scales(2);

        let rows = 4usize;
        let mut full_input = deterministic(rows * geometry.dim, 42);

        // Below the window: cached == full forward, bit for bit.
        let reference = full_forward(&decoder, &full_input, 0);
        let (last, mut cache) = decoder.prefill(&full_input, rows);
        for (got, want) in last.iter().zip(&reference[(rows - 1) * geometry.dim..]) {
            assert_eq!(got.to_bits(), want.to_bits(), "prefill last row");
        }

        // Tied logits equal the dot product of the hidden row against the
        // embedding rows, checked exactly for a few vocabulary entries.
        let logits = decoder.tied_logits(&last);
        for vocab_index in [0usize, 1, 15, geometry.vocab_size - 1] {
            let embed_row =
                &decoder.embedding[vocab_index * geometry.dim..(vocab_index + 1) * geometry.dim];
            let dot: f32 = last.iter().zip(embed_row).map(|(a, b)| a * b).sum();
            assert_eq!(
                logits[vocab_index].to_bits(),
                dot.to_bits(),
                "tied logits at vocab {vocab_index}"
            );
        }

        // Steps 0 and 1 stay within the window (4 + 2 = 6): still bit-equal
        // to the full forward.
        for step in 0..2usize {
            let token_embed = deterministic(geometry.dim, 1_000 + step as u64);
            full_input.extend_from_slice(&token_embed);
            let total_rows = rows + step + 1;
            let reference = full_forward(&decoder, &full_input, 0);
            let stepped = decoder.step(&token_embed, rows + step, &mut cache);
            for (got, want) in stepped
                .iter()
                .zip(&reference[(total_rows - 1) * geometry.dim..])
            {
                assert_eq!(got.to_bits(), want.to_bits(), "in-window step {step}");
            }
        }
        assert_eq!(cache.layers[0].len(), geometry.sliding_window);
        assert_eq!(cache.layers[0].start(), 0);

        // Step 2 overflows the window: the ring drops the oldest entries,
        // and each stepped row must equal the windowed full forward, whose
        // every query sees exactly its causal prefix cut by the window.
        for step in 2..5usize {
            let token_embed = deterministic(geometry.dim, 1_000 + step as u64);
            full_input.extend_from_slice(&token_embed);
            let total_rows = rows + step + 1;
            let reference = windowed_full_forward(&decoder, &full_input);
            let stepped = decoder.step(&token_embed, rows + step, &mut cache);
            for (got, want) in stepped
                .iter()
                .zip(&reference[reference.len() - geometry.dim..])
            {
                assert_eq!(got.to_bits(), want.to_bits(), "windowed step {step}");
            }
            assert_eq!(cache.layers[0].len(), geometry.sliding_window);
            assert_eq!(
                cache.layers[0].start(),
                total_rows - geometry.sliding_window
            );
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The time embedding places cos first, sin second, and follows the
    /// documented frequency ladder; zero delay is the identity on cos.
    #[test]
    fn time_embedding_matches_the_upstream_formula() {
        let dim = 8usize;
        let embedding = compute_time_embedding(2.0, dim, 10_000.0);
        for d in 0..dim / 2 {
            let freq = (-10_000f32.ln() * d as f32 / (dim / 2) as f32).exp();
            let angle = 2.0 * freq;
            assert!((embedding[d] - angle.cos()).abs() < 1e-6, "cos {d}");
            assert!(
                (embedding[dim / 2 + d] - angle.sin()).abs() < 1e-6,
                "sin {d}"
            );
        }
        let zero = compute_time_embedding(0.0, dim, 10_000.0);
        assert!(zero[..dim / 2].iter().all(|v| *v == 1.0));
        assert!(zero[dim / 2..].iter().all(|v| v.abs() < 1e-6));
    }
}
