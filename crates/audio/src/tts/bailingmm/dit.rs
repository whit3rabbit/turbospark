//! The DiT flow-matching stack the bailingmm generate loop drives:
//! timestep/condition embedding, the DiT transformer, the latent
//! aggregator, and the CFM solver inside `FlowLoss`.
//!
//! Port of the `DiTBlock` / `DiT` / `Aggregator` / `TimestepEmbedder` /
//! `SinusPositionEmbedding` / `Solver` / `CFM` / `FlowLoss` /
//! `get_epss_timesteps` classes in
//! `mlx_audio/tts/models/bailingmm/bailingmm.py` at the pinned
//! mlx-audio commit. Notes carried from the reference:
//!
//! - The local `RotaryEmbedding` pairs `(x[2i], x[2i + 1])` with
//!   frequencies `base ** (-2i / dim)` (the GPT-J interleaved
//!   convention, always at offset 0 and full bidirectional attention
//!   with no mask), which is the shared lm rope at
//!   `traditional = true`.
//! - `FeedForward` uses MLX `nn.gelu_approx` (the tanh approximation).
//! - The solver integrates an Euler step plus
//!   `sigma * sqrt(temperature) * sqrt(|dt|) * noise`; at the generate
//!   default `temperature 0.0` the shift is exactly zero (the reference
//!   still draws the normal, and the noise source here is consulted in
//!   the same order so recorded draws stay aligned).
//! - `CFM.sample` transposes the `[batch, channels, patch]` noise into
//!   the integration layout `[batch, patch, channels]`, applies the
//!   EPSS timestep table (or a linspace) plus the sway adjustment
//!   `t + coef * (cos(pi/2 t) - 1 + t)`, and the guide path
//!   `pred + (pred - null) * cfg` doubles the batch with a zero
//!   condition. The sub-epsilon `cfg` branch returns the FULL sequence
//!   output (not cropped to the patch) exactly like the reference.

use crate::lm::{Bias, Weights};
use crate::nn::{Linear, RmsNorm};
use crate::ops;
use crate::{Result, SpeechError};

/// The noise source for the flow sampler. Recorded replays (fixtures)
/// and seeded draws both implement this, in reference draw order:
/// the initial flow noise per generate step, then one solver draw per
/// integration step.
pub trait FlowNoise {
    /// The initial latent noise, shape `[batch, channels, patch]`.
    fn flow_noise(&mut self, shape: [usize; 3]) -> Vec<f32>;
    /// The solver noise, shape `[batch, patch, channels]`.
    fn solver_noise(&mut self, shape: [usize; 3]) -> Vec<f32>;
}

/// Seeded threefry draws in row-major order (the crate RNG; the draw
/// sequence is crate-local by design, see `tts::sampling`).
pub struct SeededNoise {
    keys: crate::music::minimax_music3::rng::KeySequence,
}

impl SeededNoise {
    pub fn new(seed: u64) -> Self {
        Self {
            keys: crate::music::minimax_music3::rng::KeySequence::new(seed),
        }
    }
}

impl FlowNoise for SeededNoise {
    fn flow_noise(&mut self, shape: [usize; 3]) -> Vec<f32> {
        let count = shape[0] * shape[1] * shape[2];
        crate::music::minimax_music3::rng::normal(self.keys.next(), count)
    }

    fn solver_noise(&mut self, shape: [usize; 3]) -> Vec<f32> {
        self.flow_noise(shape)
    }
}

/// `get_epss_timesteps`: the pinned EPSS tables for the supported step
/// counts, a linspace otherwise.
pub fn get_epss_timesteps(steps: usize) -> Vec<f32> {
    let table: &[u32] = match steps {
        5 => &[0, 2, 4, 8, 16, 32],
        6 => &[0, 2, 4, 6, 8, 16, 32],
        7 => &[0, 2, 4, 6, 8, 16, 24, 32],
        10 => &[0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12 => &[0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16 => &[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        _ => {
            return (0..=steps).map(|i| i as f32 / steps as f32).collect();
        }
    };
    table.iter().map(|&t| t as f32 / 32.0).collect()
}

/// The sway adjustment `t + coef * (cos(pi/2 t) - 1 + t)`.
pub fn sway_sampling(t: &[f32], coef: f32) -> Vec<f32> {
    t.iter()
        .map(|&v| {
            let v = v as f64;
            (v + coef as f64 * ((std::f64::consts::PI / 2.0 * v).cos() - 1.0 + v)) as f32
        })
        .collect()
}

struct FeedForward {
    ff_in: Linear,
    ff_out: Linear,
}

impl FeedForward {
    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut hidden = self.ff_in.forward(x, rows);
        ops::gelu_tanh(&mut hidden);
        self.ff_out.forward(&hidden, rows)
    }
}

/// The reference local `RotaryEmbedding`: interleaved pairs
/// `(x[2i], x[2i+1])` with angles `position * inv_freq[i]`, always at
/// offset 0. The `inv_freq` vector is a saved module attribute on the
/// MLX side (real checkpoints carry it), so it loads from the
/// checkpoint and only falls back to the classic
/// `base ** (-2i / dim)` table when a save dropped it.
#[derive(Debug, Clone)]
pub struct MingRotary {
    inv_freq: Vec<f32>,
}

impl MingRotary {
    pub fn load_absolute(weights: &Weights, name: &str, dims: usize) -> Result<Self> {
        let count = dims / 2;
        let inv_freq = if weights.contains_absolute(name) {
            weights.tensor_absolute(name, &[count])?
        } else {
            (0..count)
                .map(|i| 10000.0f32.powf(-2.0 * i as f32 / dims as f32))
                .collect()
        };
        Ok(Self { inv_freq })
    }

    /// Rotates `[heads, seq, dims]` rows in place (interleaved pairs,
    /// position = the in-sequence offset).
    pub fn apply(&self, x: &mut [f32], heads: usize, seq: usize) {
        let pairs = self.inv_freq.len();
        let dim = pairs * 2;
        for h in 0..heads {
            for pos in 0..seq {
                let base = (h * seq + pos) * dim;
                for (pair, &inv) in self.inv_freq.iter().enumerate() {
                    let angle = pos as f32 * inv;
                    let (cos, sin) = (angle.cos(), angle.sin());
                    let even = x[base + 2 * pair];
                    let odd = x[base + 2 * pair + 1];
                    x[base + 2 * pair] = even * cos - odd * sin;
                    x[base + 2 * pair + 1] = even * sin + odd * cos;
                }
            }
        }
    }
}

/// Bidirectional attention over `[batch, seq, heads * head_dim]` rows
/// with the interleaved rope at offset 0. `batch` must be 1 or 2 (the
/// CFG guide doubles it).
struct DitAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    heads: usize,
    head_dim: usize,
    rope: MingRotary,
}

impl DitAttention {
    fn forward(&self, x: &[f32], batch: usize, seq: usize) -> Vec<f32> {
        let dim = self.heads * self.head_dim;
        let q = self.to_q.forward(x, batch * seq);
        let k = self.to_k.forward(x, batch * seq);
        let v = self.to_v.forward(x, batch * seq);
        let scale = (self.head_dim as f64).powf(-0.5) as f32;
        let mut out = vec![0.0f32; batch * seq * dim];
        // Per batch item: [heads, seq, head_dim] views, rope, full
        // softmax attention.
        for b in 0..batch {
            let mut qh = head_major_batch(&q, b, seq, self.heads, self.head_dim);
            let mut kh = head_major_batch(&k, b, seq, self.heads, self.head_dim);
            let vh = head_major_batch(&v, b, seq, self.heads, self.head_dim);
            self.rope.apply(&mut qh, self.heads, seq);
            self.rope.apply(&mut kh, self.heads, seq);
            for h in 0..self.heads {
                let q_head = &qh[h * seq * self.head_dim..(h + 1) * seq * self.head_dim];
                let k_head = &kh[h * seq * self.head_dim..(h + 1) * seq * self.head_dim];
                let v_head = &vh[h * seq * self.head_dim..(h + 1) * seq * self.head_dim];
                let mut scores = vec![0.0f32; seq];
                for i in 0..seq {
                    let q_row = &q_head[i * self.head_dim..(i + 1) * self.head_dim];
                    let mut max = f32::NEG_INFINITY;
                    for j in 0..seq {
                        let mut dot = 0.0f32;
                        for d in 0..self.head_dim {
                            dot += q_row[d] * k_head[j * self.head_dim + d];
                        }
                        let score = dot * scale;
                        scores[j] = score;
                        if score > max {
                            max = score;
                        }
                    }
                    let mut sum = 0.0f32;
                    for score in &mut scores {
                        let weight = (*score - max).exp();
                        *score = weight;
                        sum += weight;
                    }
                    let inv = 1.0 / sum;
                    let out_row = &mut out[(b * seq + i) * dim + h * self.head_dim
                        ..(b * seq + i) * dim + (h + 1) * self.head_dim];
                    for j in 0..seq {
                        let weight = scores[j] * inv;
                        for d in 0..self.head_dim {
                            out_row[d] += weight * v_head[j * self.head_dim + d];
                        }
                    }
                }
            }
        }
        self.to_out.forward(&out, batch * seq)
    }
}

struct DiTBlock {
    norm1: RmsNorm,
    attn: DitAttention,
    norm2: RmsNorm,
    mlp: FeedForward,
}

impl DiTBlock {
    fn forward(&self, x: &mut [f32], batch: usize, seq: usize) {
        let dim = self.norm1.weight.len();
        let mut attended = x.to_vec();
        self.norm1.apply(&mut attended, batch * seq);
        let attn = self.attn.forward(&attended, batch, seq);
        for (value, add) in x.iter_mut().zip(&attn) {
            *value += add;
        }
        let mut fed = x.to_vec();
        self.norm2.apply(&mut fed, batch * seq);
        let mlp = self.mlp.forward(&fed, batch * seq);
        let _ = dim;
        for (value, add) in x.iter_mut().zip(&mlp) {
            *value += add;
        }
    }
}

/// `SinusPositionEmbedding` + `TimestepEmbedder`: the 256-wide
/// `concat([sin, cos])` table scaled by 1000, then a silu MLP.
struct TimestepEmbedder {
    mlp_in: Linear,
    mlp_out: Linear,
    dim: usize,
}

impl TimestepEmbedder {
    fn forward(&self, t: f32, batch: usize) -> Vec<f32> {
        let half = 256 / 2;
        let step = 10000.0f64.ln() / (half as f64 - 1.0);
        let mut hidden = vec![0.0f32; batch * 256];
        for b in 0..batch {
            for i in 0..half {
                let angle = 1000.0 * (t as f64) * (-(i as f64) * step).exp();
                hidden[b * 256 + i] = (angle as f32).sin();
                hidden[b * 256 + half + i] = (angle as f32).cos();
            }
        }
        let projected = self.mlp_in.forward(&hidden, batch);
        let mut activated = projected;
        for value in &mut activated {
            *value /= 1.0 + (-*value).exp();
        }
        let _ = self.dim;
        self.mlp_out.forward(&activated, batch)
    }
}

struct FinalLayer {
    norm: RmsNorm,
    linear: Linear,
}

impl FinalLayer {
    fn forward(&self, x: &mut [f32], rows: usize) -> Vec<f32> {
        self.norm.apply(x, rows);
        self.linear.forward(x, rows)
    }
}

/// The `DiT` denoiser: `[y, embedded history, embedded now]` through the
/// blocks, then the final layer.
pub struct DiT {
    pub in_channels: usize,
    pub out_channels: usize,
    t_embedder: TimestepEmbedder,
    x_embedder: Linear,
    c_embedder: Linear,
    blocks: Vec<DiTBlock>,
    final_layer: FinalLayer,
}

#[allow(clippy::too_many_arguments)]
impl DiT {
    pub fn load_absolute(
        weights: &Weights,
        prefix: &str,
        depth: usize,
        num_heads: usize,
        mlp_ratio: f32,
        in_channels: usize,
        llm_cond_dim: usize,
        hidden_dim: usize,
    ) -> Result<Self> {
        if hidden_dim % num_heads != 0 {
            return Err(SpeechError::BadConfig {
                field: "hidden_size".into(),
                why: format!("{hidden_dim} not divisible by num_heads {num_heads}"),
            });
        }
        let head_dim = hidden_dim / num_heads;
        let inner = (hidden_dim as f64 * mlp_ratio as f64) as usize;
        // `name` carries the full tensor path (the call sites below
        // build it from `prefix` themselves).
        let linear = |name: &str, input: usize, output: usize| -> Result<Linear> {
            weights.linear_absolute(name, input, output, Bias::Required)
        };
        let block = |index: usize| -> Result<DiTBlock> {
            let base = format!("{prefix}blocks.{index}.");
            Ok(DiTBlock {
                norm1: RmsNorm::new(
                    weights.tensor_absolute(&format!("{base}norm1.weight"), &[hidden_dim])?,
                    1e-6,
                ),
                attn: DitAttention {
                    to_q: linear(&format!("{base}attn.to_q"), hidden_dim, hidden_dim)?,
                    to_k: linear(&format!("{base}attn.to_k"), hidden_dim, hidden_dim)?,
                    to_v: linear(&format!("{base}attn.to_v"), hidden_dim, hidden_dim)?,
                    to_out: linear(&format!("{base}attn.to_out.0"), hidden_dim, hidden_dim)?,
                    heads: num_heads,
                    head_dim,
                    rope: MingRotary::load_absolute(
                        weights,
                        &format!("{prefix}rotary_embed.inv_freq"),
                        head_dim,
                    )?,
                },
                norm2: RmsNorm::new(
                    weights.tensor_absolute(&format!("{base}norm2.weight"), &[hidden_dim])?,
                    1e-6,
                ),
                mlp: FeedForward {
                    ff_in: linear(&format!("{base}mlp.ff.0.0"), hidden_dim, inner)?,
                    ff_out: linear(&format!("{base}mlp.ff.2"), inner, hidden_dim)?,
                },
            })
        };
        let t_embedder = TimestepEmbedder {
            mlp_in: linear(&format!("{prefix}t_embedder.time_mlp.0"), 256, hidden_dim)?,
            mlp_out: linear(
                &format!("{prefix}t_embedder.time_mlp.2"),
                hidden_dim,
                hidden_dim,
            )?,
            dim: hidden_dim,
        };
        Ok(Self {
            in_channels,
            out_channels: in_channels,
            t_embedder,
            x_embedder: linear(&format!("{prefix}x_embedder"), in_channels, hidden_dim)?,
            c_embedder: linear(
                &format!("{prefix}c_embedder.cond_embedder"),
                llm_cond_dim,
                hidden_dim,
            )?,
            blocks: (0..depth).map(block).collect::<Result<Vec<_>>>()?,
            final_layer: FinalLayer {
                norm: RmsNorm::new(
                    weights.tensor_absolute(
                        &format!("{prefix}final_layer.norm_final.weight"),
                        &[hidden_dim],
                    )?,
                    1e-6,
                ),
                linear: linear(
                    &format!("{prefix}final_layer.linear"),
                    hidden_dim,
                    in_channels,
                )?,
            },
        })
    }

    /// The full-sequence output `[batch, 1 + history + patch, channels]`.
    /// `t` is one scalar per run (the reference broadcasts it over the
    /// CFG-doubled batch); `c` is `[batch, 1, llm_cond_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &[f32],
        t: f32,
        c: &[f32],
        latent_history: &[f32],
        batch: usize,
        history_len: usize,
        patch: usize,
    ) -> Vec<f32> {
        let t_embed = self.t_embedder.forward(t, batch);
        let x_now = self.x_embedder.forward(x, batch * patch);
        let x_history = self.x_embedder.forward(latent_history, batch * history_len);
        let c_embed = self.c_embedder.forward(c, batch);
        let hidden_dim = t_embed.len() / batch;
        // Sequence: [y (1), history, now].
        let seq = 1 + history_len + patch;
        let mut full = vec![0.0f32; batch * seq * hidden_dim];
        for b in 0..batch {
            // y = t_embed + c_embed
            for i in 0..hidden_dim {
                full[b * seq * hidden_dim + i] =
                    t_embed[b * hidden_dim + i] + c_embed[b * hidden_dim + i];
            }
            // history rows
            let hist_base = (b * history_len) * hidden_dim;
            let dst_base = (b * seq + 1) * hidden_dim;
            full[dst_base..dst_base + history_len * hidden_dim]
                .copy_from_slice(&x_history[hist_base..hist_base + history_len * hidden_dim]);
            // now rows
            let now_base = (b * patch) * hidden_dim;
            let dst_base = (b * seq + 1 + history_len) * hidden_dim;
            full[dst_base..dst_base + patch * hidden_dim]
                .copy_from_slice(&x_now[now_base..now_base + patch * hidden_dim]);
        }
        for block in &self.blocks {
            block.forward(&mut full, batch, seq);
        }
        self.final_layer.forward(&mut full, batch * seq)
    }

    /// `forward_with_cfg` over ALREADY doubled inputs: `x`,
    /// `latent_history`, and `c` carry `batch` items where the second
    /// half is the null guide (zero condition) and `batch` is even. The
    /// reference concatenates the halves inside; here the caller (the
    /// solver velocity) builds them, and this entry crops the last
    /// `patch` positions of each half and applies
    /// `pred + (pred - null) * cfg`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_cfg(
        &self,
        x: &[f32],
        t: f32,
        c: &[f32],
        cfg_scale: f32,
        latent_history: &[f32],
        batch: usize,
        history_len: usize,
        patch: usize,
    ) -> Vec<f32> {
        let half = batch / 2;
        let seq = 1 + history_len + patch;
        let channels = self.out_channels;
        let per = seq * channels;
        let out = self.forward(x, t, c, latent_history, batch, history_len, patch);
        let pred = &out[..half * per];
        let null = &out[half * per..];
        let rows = patch;
        let mut guided = vec![0.0f32; half * rows * channels];
        for b in 0..half {
            for r in 0..rows {
                for ch in 0..channels {
                    let src = b * per + (seq - rows + r) * channels + ch;
                    let idx = (b * rows + r) * channels + ch;
                    guided[idx] = pred[src] + (pred[src] - null[src]) * cfg_scale;
                }
            }
        }
        guided
    }
}

fn head_major_batch(x: &[f32], batch: usize, seq: usize, heads: usize, dim: usize) -> Vec<f32> {
    let total = seq * heads * dim;
    let base = batch * total;
    let mut out = vec![0.0f32; heads * seq * dim];
    for t in 0..seq {
        for h in 0..heads {
            let src = base + (t * heads + h) * dim;
            let dst = (h * seq + t) * dim;
            out[dst..dst + dim].copy_from_slice(&x[src..src + dim]);
        }
    }
    out
}

/// The `Aggregator` (`linear_proj_audio`): a learned CLS token
/// prepended to the embedded latent patch, DiT blocks, and the first
/// output position.
pub struct Aggregator {
    cls: Vec<f32>,
    x_embedder: Linear,
    blocks: Vec<DiTBlock>,
    final_layer: FinalLayer,
    out_dim: usize,
}

impl Aggregator {
    #[allow(clippy::too_many_arguments)]
    pub fn load_absolute(
        weights: &Weights,
        prefix: &str,
        depth: usize,
        num_heads: usize,
        mlp_ratio: f32,
        in_channels: usize,
        llm_input_dim: usize,
        hidden_dim: usize,
    ) -> Result<Self> {
        if hidden_dim % num_heads != 0 {
            return Err(SpeechError::BadConfig {
                field: "hidden_size".into(),
                why: format!("{hidden_dim} not divisible by num_heads {num_heads}"),
            });
        }
        let head_dim = hidden_dim / num_heads;
        let inner = (hidden_dim as f64 * mlp_ratio as f64) as usize;
        // `name` carries the full tensor path (the call sites below
        // build it from `prefix` themselves).
        let linear = |name: &str, input: usize, output: usize| -> Result<Linear> {
            weights.linear_absolute(name, input, output, Bias::Required)
        };
        let block = |index: usize| -> Result<DiTBlock> {
            let base = format!("{prefix}blocks.{index}.");
            Ok(DiTBlock {
                norm1: RmsNorm::new(
                    weights.tensor_absolute(&format!("{base}norm1.weight"), &[hidden_dim])?,
                    1e-6,
                ),
                attn: DitAttention {
                    to_q: linear(&format!("{base}attn.to_q"), hidden_dim, hidden_dim)?,
                    to_k: linear(&format!("{base}attn.to_k"), hidden_dim, hidden_dim)?,
                    to_v: linear(&format!("{base}attn.to_v"), hidden_dim, hidden_dim)?,
                    to_out: linear(&format!("{base}attn.to_out.0"), hidden_dim, hidden_dim)?,
                    heads: num_heads,
                    head_dim,
                    rope: MingRotary::load_absolute(
                        weights,
                        &format!("{prefix}rotary_embed.inv_freq"),
                        head_dim,
                    )?,
                },
                norm2: RmsNorm::new(
                    weights.tensor_absolute(&format!("{base}norm2.weight"), &[hidden_dim])?,
                    1e-6,
                ),
                mlp: FeedForward {
                    ff_in: linear(&format!("{base}mlp.ff.0.0"), hidden_dim, inner)?,
                    ff_out: linear(&format!("{base}mlp.ff.2"), inner, hidden_dim)?,
                },
            })
        };
        Ok(Self {
            // nn.Embedding(1, hidden) stores [1, hidden]; the flat row is
            // the CLS token.
            cls: weights
                .tensor_absolute(&format!("{prefix}word_embedder.weight"), &[1, hidden_dim])?,
            x_embedder: linear(&format!("{prefix}x_embedder"), in_channels, hidden_dim)?,
            blocks: (0..depth).map(block).collect::<Result<Vec<_>>>()?,
            final_layer: FinalLayer {
                norm: RmsNorm::new(
                    weights.tensor_absolute(
                        &format!("{prefix}final_layer.norm_final.weight"),
                        &[hidden_dim],
                    )?,
                    1e-6,
                ),
                linear: linear(
                    &format!("{prefix}final_layer.linear"),
                    hidden_dim,
                    llm_input_dim,
                )?,
            },
            out_dim: llm_input_dim,
        })
    }

    /// `[batch, patches, in_channels]` -> `[batch, 1, llm_input_dim]`.
    pub fn forward(&self, x: &[f32], batch: usize, patches: usize) -> Vec<f32> {
        let channels = x.len() / (batch * patches);
        let embedded = self.x_embedder.forward(x, batch * patches);
        let hidden_dim = self.cls.len();
        let seq = 1 + patches;
        let mut full = vec![0.0f32; batch * seq * hidden_dim];
        for b in 0..batch {
            full[b * seq * hidden_dim..b * seq * hidden_dim + hidden_dim]
                .copy_from_slice(&self.cls);
            let src = (b * patches) * hidden_dim;
            let dst = (b * seq + 1) * hidden_dim;
            full[dst..dst + patches * hidden_dim]
                .copy_from_slice(&embedded[src..src + patches * hidden_dim]);
        }
        for block in &self.blocks {
            block.forward(&mut full, batch, seq);
        }
        let mut out = self.final_layer.forward(&mut full, batch * seq);
        out.truncate(batch * seq * self.out_dim);
        // Take position 0 of each batch item.
        let mut first = vec![0.0f32; batch * self.out_dim];
        for b in 0..batch {
            first[b * self.out_dim..(b + 1) * self.out_dim]
                .copy_from_slice(&out[b * seq * self.out_dim..(b * seq + 1) * self.out_dim]);
        }
        let _ = channels;
        first
    }
}

/// `FlowLoss`: the DiT denoiser wrapped in the CFM sampler.
pub struct FlowLoss {
    pub z_channels: usize,
    model: DiT,
}

/// The flow sampling options (`CFM.sample` / `FlowLoss.sample`).
#[derive(Debug, Clone)]
pub struct FlowParams {
    pub cfg: f32,
    pub patch: usize,
    pub sigma: f32,
    pub temperature: f32,
    pub steps: usize,
    /// `use_epss` selects the pinned timestep tables; `false` is a
    /// linspace.
    pub use_epss: bool,
    pub sway_sampling_coef: f32,
}

impl FlowLoss {
    /// `sample`: draws the initial noise through `noise`, integrates the
    /// flow, and returns the final `[batch, patch, z_channels]` latent.
    pub fn sample(
        &self,
        z: &[f32],
        latent_history: &[f32],
        params: &FlowParams,
        noise: &mut dyn FlowNoise,
    ) -> Vec<f32> {
        let channels = self.z_channels;
        let patch = params.patch;
        // noise [1, channels, patch] -> y0 [1, patch, channels].
        let drawn = noise.flow_noise([1, channels, patch]);
        let mut y = vec![0.0f32; patch * channels];
        for p in 0..patch {
            for c in 0..channels {
                y[p * channels + c] = drawn[c * patch + p];
            }
        }
        let history_len = latent_history.len() / channels;
        let mut t = if params.use_epss {
            get_epss_timesteps(params.steps)
        } else {
            (0..=params.steps)
                .map(|i| i as f32 / params.steps as f32)
                .collect()
        };
        t = sway_sampling(&t, params.sway_sampling_coef);

        // The Euler-plus-noise solver (`Solver.integrate`).
        for step in 1..t.len() {
            let t0 = t[step - 1];
            let t1 = t[step];
            let dt = t1 - t0;
            let f0 = self.velocity(
                t0,
                &y,
                z,
                latent_history,
                params,
                history_len,
                patch,
                channels,
            );
            for (value, derivative) in y.iter_mut().zip(&f0) {
                *value += dt * derivative;
            }
            // The reference draws the solver noise unconditionally and
            // multiplies by sigma * sqrt(temperature) * sqrt(|dt|);
            // temperature 0.0 collapses the shift to exact zeros.
            let draw = noise.solver_noise([1, patch, channels]);
            if params.temperature != 0.0 {
                let shift = params.sigma * params.temperature.sqrt() * dt.abs().sqrt();
                for (value, sample) in y.iter_mut().zip(&draw) {
                    *value += shift * sample;
                }
            }
        }
        y
    }

    /// The solver velocity: the guided (or plain) DiT prediction.
    #[allow(clippy::too_many_arguments)]
    fn velocity(
        &self,
        t: f32,
        y: &[f32],
        z: &[f32],
        latent_history: &[f32],
        params: &FlowParams,
        history_len: usize,
        patch: usize,
        channels: usize,
    ) -> Vec<f32> {
        if params.cfg < 1e-5 {
            // The reference fn returns the FULL sequence output on this
            // branch (not cropped to the patch).
            return self
                .model
                .forward(y, t, z, latent_history, 1, history_len, patch);
        }
        // forward_with_cfg doubles the batch: x and the history are
        // concatenated along batch with a zero condition on the null
        // half.
        let cond = z.len();
        let mut c2 = vec![0.0f32; 2 * cond];
        c2[..cond].copy_from_slice(z);
        let mut x2 = y.to_vec();
        x2.extend_from_slice(y);
        let mut history2 = latent_history.to_vec();
        history2.extend_from_slice(latent_history);
        self.model
            .forward_with_cfg(&x2, t, &c2, params.cfg, &history2, 2, history_len, patch)
            .into_iter()
            .take(patch * channels)
            .collect()
    }
}

impl FlowLoss {
    /// Stage probe for the tests: the full-sequence DiT call over the
    /// doubled inputs.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn flowloss_full_probe(
        &self,
        x: &[f32],
        t: f32,
        c: &[f32],
        history: &[f32],
        batch: usize,
        history_len: usize,
        patch: usize,
    ) -> Vec<f32> {
        self.model
            .forward(x, t, c, history, batch, history_len, patch)
    }

    /// Stage probe for the tests: the per-block outputs of the
    /// full-sequence DiT call over the doubled inputs.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn block_probe(
        &self,
        x: &[f32],
        t: f32,
        c: &[f32],
        history: &[f32],
        batch: usize,
        history_len: usize,
        patch: usize,
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        let t_embed = self.model.t_embedder.forward(t, batch);
        let x_now = self.model.x_embedder.forward(x, batch * patch);
        let x_history = self.model.x_embedder.forward(history, batch * history_len);
        let c_embed = self.model.c_embedder.forward(c, batch);
        let hidden_dim = t_embed.len() / batch;
        let seq = 1 + history_len + patch;
        let mut full = vec![0.0f32; batch * seq * hidden_dim];
        for b in 0..batch {
            for i in 0..hidden_dim {
                full[b * seq * hidden_dim + i] =
                    t_embed[b * hidden_dim + i] + c_embed[b * hidden_dim + i];
            }
            let hist_base = (b * history_len) * hidden_dim;
            let dst_base = (b * seq + 1) * hidden_dim;
            full[dst_base..dst_base + history_len * hidden_dim]
                .copy_from_slice(&x_history[hist_base..hist_base + history_len * hidden_dim]);
            let now_base = (b * patch) * hidden_dim;
            let dst_base = (b * seq + 1 + history_len) * hidden_dim;
            full[dst_base..dst_base + patch * hidden_dim]
                .copy_from_slice(&x_now[now_base..now_base + patch * hidden_dim]);
        }
        let mut traces = Vec::new();
        let mut attn_outs = Vec::new();
        let pre_block = full.clone();
        for block in &self.model.blocks {
            let dim = block.norm1.weight.len();
            let mut attended = full.clone();
            block.norm1.apply(&mut attended, batch * seq);
            let attn = block.attn.forward(&attended, batch, seq);
            attn_outs.push(attn.clone());
            for (value, add) in full.iter_mut().zip(&attn) {
                *value += add;
            }
            let mut fed = full.clone();
            block.norm2.apply(&mut fed, batch * seq);
            let mlp = block.mlp.forward(&fed, batch * seq);
            for (value, add) in full.iter_mut().zip(&mlp) {
                *value += add;
            }
            traces.push(full.clone());
            let _ = dim;
        }
        (pre_block, traces)
    }

    /// Stage probe for the tests: the raw guided DiT call.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn model_forward_with_cfg(
        &self,
        x: &[f32],
        t: f32,
        c: &[f32],
        cfg: f32,
        history: &[f32],
        batch: usize,
        history_len: usize,
        patch: usize,
    ) -> Vec<f32> {
        self.model
            .forward_with_cfg(x, t, c, cfg, history, batch, history_len, patch)
    }
}

/// Loads the `FlowLoss` DiT (`flowloss.cfm.model.*` tensor prefix).
#[allow(clippy::too_many_arguments)]
pub fn load_flow_loss(
    weights: &Weights,
    prefix: &str,
    depth: usize,
    num_heads: usize,
    mlp_ratio: f32,
    z_channels: usize,
    llm_cond_dim: usize,
    hidden_dim: usize,
) -> Result<FlowLoss> {
    let model = DiT::load_absolute(
        weights,
        &format!("{prefix}cfm.model."),
        depth,
        num_heads,
        mlp_ratio,
        z_channels,
        llm_cond_dim,
        hidden_dim,
    )?;
    Ok(FlowLoss { z_channels, model })
}
