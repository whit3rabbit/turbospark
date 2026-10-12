//! The DramaBox audio DiT: layers, rope, timestep embeddings, and the
//! audio-only LTX transformer
//! (`mlx_audio/tts/models/dramabox/{layers,rope,timestep,transformer,scheduler}.py`).
//!
//! Reference: mlx-audio at commit
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/dramabox).
//! Pre-norm AdaLN-zero blocks: self attention over the patch stream
//! with split-interleaved LTX rope and per-head q/k RMS norms, gated
//! attention (2 * sigmoid(gate logits) per head), cross attention to
//! the text context with the prompt-AdaLN modulation of keys/values,
//! a tanh-approx GELU feed forward, and the scale/shift output head
//! over an affine-free LayerNorm. Single batch, f32.

use crate::nn::{Linear, RmsNorm};
use crate::{ops, Result, SpeechError};
use turbospark_model_io::safetensors::SafetensorsFile;

use super::sharded::{load_linear_sharded, load_sharded};

// ---------------------------------------------------------------------------
// Rope
// ---------------------------------------------------------------------------

/// `generate_freq_grid_np` (the double-precision path the deployed
/// config ships, `frequencies_precision = "float64"`): the exponent
/// linspace runs in f64 and the result casts to f32.
fn freq_grid(theta: f64, max_pos_count: usize, inner_dim: usize) -> Vec<f32> {
    let num_elements = 2 * max_pos_count;
    let count = inner_dim / num_elements;
    // np.linspace(log(1)/log(theta), log(theta)/log(theta), count)
    let step = if count > 1 {
        1.0 / (count - 1) as f64
    } else {
        0.0
    };
    (0..count)
        .map(|i| {
            let unit = i as f64 * step;
            theta.powf(unit) * (std::f64::consts::PI / 2.0)
        })
        .map(|v| v as f32)
        .collect()
}

/// One modality's rope tables (`precompute_freqs_cis` for the split
/// type). `positions` holds per-token bounds `[token, dim, 2]` (the
/// connector passes the index as both bounds); the middle-index grid
/// averages them, each dimension divides by its max_pos and flips to
/// `(frac * 2 - 1)`, and the per-token frequency row is the dimension
/// concatenation of the grid products, padded on the LEFT with
/// cos(1)/sin(0) up to `inner_dim / 2`. The grid computes in f64
/// (`frequencies_precision = "float64"`) and the trig runs in f32.
/// Returns `(cos, sin)` shaped `[heads, tokens, head_dim / 2]` with
/// `head_dim = inner_dim / heads`; the `[tokens, inner_dim / 2]`
/// frequency row is cut across the heads by the reference reshape.
#[allow(clippy::too_many_arguments)]
pub(crate) fn split_freqs_cis(
    positions: &[f32],
    tokens: usize,
    dims: usize,
    max_pos: &[f32],
    theta: f64,
    inner_dim: usize,
    num_heads: usize,
    double_precision: bool,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if dims != max_pos.len() {
        return Err(SpeechError::BadConfig {
            field: "positional_embedding_max_pos".into(),
            why: format!("{dims} position dims for {} max_pos entries", max_pos.len()),
        });
    }
    let num_elements = 2 * dims;
    if inner_dim % num_elements != 0 {
        return Err(SpeechError::BadConfig {
            field: "inner_dim".into(),
            why: format!("{inner_dim} does not divide into {num_elements} rope elements"),
        });
    }
    let grid = freq_grid(theta, dims, inner_dim);
    let per_dim = inner_dim / num_elements;
    let half = inner_dim / 2;
    let width = per_dim * dims;
    if width > half {
        return Err(SpeechError::BadConfig {
            field: "inner_dim".into(),
            why: format!("frequency rows {width} exceed the half width {half}"),
        });
    }
    // The frequency row is [tokens, half]; the split reshape cuts it
    // across the heads: head h takes columns h * head_half ..
    // (h + 1) * head_half (the reference reshapes
    // `[T, half] -> [T, heads, head_half]`).
    let head_half = half / num_heads;
    let mut cos_row = vec![0.0f32; tokens * half];
    let mut sin_row = vec![0.0f32; tokens * half];
    for token in 0..tokens {
        for d in 0..dims {
            let middle = (positions[token * 2 * dims + d * 2]
                + positions[token * 2 * dims + d * 2 + 1])
                / 2.0;
            let phase = (middle / max_pos[d]) * 2.0 - 1.0;
            for (f, &index) in grid.iter().enumerate() {
                let angle = index * phase;
                let column = d * per_dim + f;
                let (c, s) = if double_precision {
                    let a = angle as f64;
                    (a.cos() as f32, a.sin() as f32)
                } else {
                    (angle.cos(), angle.sin())
                };
                cos_row[token * half + column] = c;
                sin_row[token * half + column] = s;
            }
        }
        // Left padding: cos 1, sin 0.
        for column in width..half {
            cos_row[token * half + column] = 1.0;
            sin_row[token * half + column] = 0.0;
        }
    }
    let mut cos = vec![0.0f32; num_heads * tokens * head_half];
    let mut sin = vec![0.0f32; num_heads * tokens * head_half];
    for head in 0..num_heads {
        for token in 0..tokens {
            for column in 0..head_half {
                cos[(head * tokens + token) * head_half + column] =
                    cos_row[token * half + head * head_half + column];
                sin[(head * tokens + token) * head_half + column] =
                    sin_row[token * half + head * head_half + column];
            }
        }
    }
    Ok((cos, sin))
}

/// `apply_split_rotary_emb` over head-major `[heads, tokens, dim]`
/// input with `[heads, tokens, dim / 2]` tables: the first and second
/// halves of each head rotate as a complex pair.
pub(crate) fn apply_split_rotary_emb(
    x: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    heads: usize,
    tokens: usize,
    dim: usize,
) {
    let half = dim / 2;
    for head in 0..heads {
        for token in 0..tokens {
            let base = (head * tokens + token) * dim;
            let table = (head * tokens + token) * half;
            for f in 0..half {
                let first = x[base + f];
                let second = x[base + half + f];
                let (c, s) = (cos[table + f], sin[table + f]);
                x[base + f] = first * c - second * s;
                x[base + half + f] = second * c + first * s;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Layers
// ---------------------------------------------------------------------------

/// `rms_norm` from `layers.py` (no weight at these call sites).
pub(crate) fn rms_norm(x: &mut [f32], rows: usize, eps: f32) {
    let width = x.len().checked_div(rows.max(1)).unwrap_or(0);
    for row in 0..rows {
        let base = row * width;
        let mut sum = 0.0f32;
        for value in &x[base..base + width] {
            sum += value * value;
        }
        let inv = 1.0 / (sum / width as f32 + eps).sqrt();
        for value in &mut x[base..base + width] {
            *value *= inv;
        }
    }
}

/// `gelu_approx`: the tanh formulation.
pub(crate) fn gelu_approx(x: &mut [f32]) {
    let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
    for value in x.iter_mut() {
        let inner = sqrt_2_over_pi * (*value + 0.044715 * value.powi(3));
        *value = 0.5 * *value * (1.0 + inner.tanh());
    }
}

/// `FeedForward`: tanh-GELU up projection, identity, biased down.
pub(crate) struct FeedForward {
    proj_in: Linear,
    proj_out: Linear,
}

impl FeedForward {
    pub(crate) fn load(files: &[SafetensorsFile], name: &str, dim: usize) -> Result<Self> {
        let inner = dim * 4;
        Ok(Self {
            proj_in: load_linear_sharded(files, &format!("{name}.net.0.proj"), dim, inner)?,
            proj_out: load_linear_sharded(files, &format!("{name}.net.2"), inner, dim)?,
        })
    }

    pub(crate) fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut hidden = self.proj_in.forward(x, rows);
        gelu_approx(&mut hidden);
        self.proj_out.forward(&hidden, rows)
    }
}

/// The dramabox `Attention`: biased q/k/v/out, inner-width q/k RMS
/// norms, optional LTX rope, optional additive mask, and the gated
/// attention head scales.
pub(crate) struct Attention {
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_gate_logits: Option<Linear>,
    to_out: Linear,
    heads: usize,
    dim_head: usize,
    scale: f32,
    rope_split: bool,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load(
        files: &[SafetensorsFile],
        name: &str,
        query_dim: usize,
        context_dim: usize,
        heads: usize,
        dim_head: usize,
        norm_eps: f32,
        apply_gated_attention: bool,
        rope_split: bool,
    ) -> Result<Self> {
        let inner = heads * dim_head;
        Ok(Self {
            q_norm: RmsNorm::new(
                load_sharded(files, &format!("{name}.q_norm.weight"), &[inner])?,
                norm_eps,
            ),
            k_norm: RmsNorm::new(
                load_sharded(files, &format!("{name}.k_norm.weight"), &[inner])?,
                norm_eps,
            ),
            to_q: load_linear_sharded(files, &format!("{name}.to_q"), query_dim, inner)?,
            to_k: load_linear_sharded(files, &format!("{name}.to_k"), context_dim, inner)?,
            to_v: load_linear_sharded(files, &format!("{name}.to_v"), context_dim, inner)?,
            to_gate_logits: if apply_gated_attention {
                Some(load_linear_sharded(
                    files,
                    &format!("{name}.to_gate_logits"),
                    query_dim,
                    heads,
                )?)
            } else {
                None
            },
            to_out: load_linear_sharded(files, &format!("{name}.to_out.0"), inner, query_dim)?,
            heads,
            dim_head,
            scale: (dim_head as f64).powf(-0.5) as f32,
            rope_split,
        })
    }

    /// `x [tokens, query_dim]`, `context [ctx_tokens, context_dim]`,
    /// optional rope tables `[heads, tokens, dim_head / 2]`, optional
    /// additive mask `[tokens, ctx_tokens]`. `all_perturbed` (the STG
    /// branch) replaces the attention output with the raw values; the
    /// gate and out projection still apply. Returns
    /// `[tokens, query_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        x: &[f32],
        tokens: usize,
        context: &[f32],
        ctx_tokens: usize,
        mask: Option<&[f32]>,
        cos: Option<&[f32]>,
        sin: Option<&[f32]>,
        all_perturbed: bool,
    ) -> Vec<f32> {
        let heads = self.heads;
        let dim_head = self.dim_head;
        let inner = heads * dim_head;

        let value = self.to_v.forward(context, ctx_tokens);

        if all_perturbed {
            // The reference `if all_perturbed: out = value` short
            // circuit: the q/k/rope/attention path is skipped, the gate
            // and out projection are not.
            let mut out = value;
            if let Some(gate_proj) = &self.to_gate_logits {
                let gates = gate_proj.forward(x, tokens);
                for t in 0..tokens {
                    for h in 0..heads {
                        let gate = 2.0 / (1.0 + (-gates[t * heads + h]).exp());
                        for d in 0..dim_head {
                            out[t * inner + h * dim_head + d] *= gate;
                        }
                    }
                }
            }
            return self.to_out.forward(&out, tokens);
        }

        let mut q = self.to_q.forward(x, tokens);
        self.q_norm.apply(&mut q, tokens);
        let mut k = self.to_k.forward(context, ctx_tokens);
        self.k_norm.apply(&mut k, ctx_tokens);

        // Head-major [heads, tokens, dim_head], rope on q and k.
        let mut qh = head_major(&q, tokens, heads, dim_head);
        let mut kh = head_major(&k, ctx_tokens, heads, dim_head);
        let vh = head_major(&value, ctx_tokens, heads, dim_head);
        if let (Some(cos), Some(sin)) = (cos, sin) {
            if self.rope_split {
                apply_split_rotary_emb(&mut qh, cos, sin, heads, tokens, dim_head);
                apply_split_rotary_emb(&mut kh, cos, sin, heads, ctx_tokens, dim_head);
            }
        }

        // Masked softmax attention per head.
        let mut out = vec![0.0f32; heads * tokens * dim_head];
        let mut scores = vec![0.0f32; ctx_tokens];
        for head in 0..heads {
            let q_head = &qh[head * tokens * dim_head..(head + 1) * tokens * dim_head];
            let k_head = &kh[head * ctx_tokens * dim_head..(head + 1) * ctx_tokens * dim_head];
            let v_head = &vh[head * ctx_tokens * dim_head..(head + 1) * ctx_tokens * dim_head];
            for i in 0..tokens {
                let q_row = &q_head[i * dim_head..(i + 1) * dim_head];
                let mut max = f32::NEG_INFINITY;
                for (j, score) in scores.iter_mut().enumerate() {
                    let mut dot = 0.0f32;
                    for d in 0..dim_head {
                        dot += q_row[d] * k_head[j * dim_head + d];
                    }
                    let mut value = dot * self.scale;
                    if let Some(mask) = mask {
                        value += mask[i * ctx_tokens + j];
                    }
                    *score = value;
                    if value > max {
                        max = value;
                    }
                }
                let mut sum = 0.0f32;
                for score in &mut scores {
                    let weight = (*score - max).exp();
                    *score = weight;
                    sum += weight;
                }
                let inv = 1.0 / sum;
                let out_row =
                    &mut out[(head * tokens + i) * dim_head..(head * tokens + i + 1) * dim_head];
                for (j, score) in scores.iter().enumerate() {
                    let weight = score * inv;
                    for (d, slot) in out_row.iter_mut().enumerate() {
                        *slot += weight * v_head[j * dim_head + d];
                    }
                }
            }
        }

        // Back to [tokens, heads * dim_head].
        let mut joined = vec![0.0f32; tokens * inner];
        for t in 0..tokens {
            for h in 0..heads {
                let src = (h * tokens + t) * dim_head;
                let dst = t * inner + h * dim_head;
                joined[dst..dst + dim_head].copy_from_slice(&out[src..src + dim_head]);
            }
        }

        // Gated attention: 2 * sigmoid(gate logits) per head.
        if let Some(gate_proj) = &self.to_gate_logits {
            let gates = gate_proj.forward(x, tokens);
            for t in 0..tokens {
                for h in 0..heads {
                    let gate = 2.0 / (1.0 + (-gates[t * heads + h]).exp());
                    for d in 0..dim_head {
                        joined[t * inner + h * dim_head + d] *= gate;
                    }
                }
            }
        }
        self.to_out.forward(&joined, tokens)
    }
}

/// Reinterprets `[tokens, heads * dim]` as `[heads, tokens, dim]`.
fn head_major(x: &[f32], tokens: usize, heads: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; heads * tokens * dim];
    for t in 0..tokens {
        for h in 0..heads {
            let src = t * heads * dim + h * dim;
            let dst = (h * tokens + t) * dim;
            out[dst..dst + dim].copy_from_slice(&x[src..src + dim]);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Timestep embeddings
// ---------------------------------------------------------------------------

/// `get_timestep_embedding` with `flip_sin_to_cos = True` and
/// `downscale_freq_shift = 0` (the PixArt config): the concatenated
/// `[sin, cos]` halves swap, so COS comes first, with frequencies
/// `exp(-ln(max_period) * d / half)`.
fn timestep_embedding(t: f32, dim: usize, max_period: f32) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0.0f32; dim];
    for (d, slot) in out[..half].iter_mut().enumerate() {
        let freq = (-max_period.ln() * d as f32 / half as f32).exp();
        *slot = (t * freq).cos();
    }
    for (d, slot) in out[half..].iter_mut().enumerate() {
        let freq = (-max_period.ln() * d as f32 / half as f32).exp();
        *slot = (t * freq).sin();
    }
    out
}

/// `AdaLayerNormSingle`: the PixArt timestep projector and the
/// coefficient linear over the silu'd embedding.
pub(crate) struct AdaLayerNormSingle {
    time_proj_in: Linear,
    time_proj_out: Linear,
    linear: Linear,
}

impl AdaLayerNormSingle {
    pub(crate) fn load(
        files: &[SafetensorsFile],
        name: &str,
        embedding_dim: usize,
        coefficient: usize,
    ) -> Result<Self> {
        Ok(Self {
            time_proj_in: load_linear_sharded(
                files,
                &format!("{name}.emb.timestep_embedder.linear_1"),
                256,
                embedding_dim,
            )?,
            time_proj_out: load_linear_sharded(
                files,
                &format!("{name}.emb.timestep_embedder.linear_2"),
                embedding_dim,
                embedding_dim,
            )?,
            linear: load_linear_sharded(
                files,
                &format!("{name}.linear"),
                embedding_dim,
                coefficient * embedding_dim,
            )?,
        })
    }

    /// `timestep [tokens] -> (timestep_out [tokens, coefficient * dim],
    /// embedded [tokens, dim])`.
    pub(crate) fn forward(&self, timesteps: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let tokens = timesteps.len();
        let mut freqs = Vec::with_capacity(tokens * 256);
        for &t in timesteps {
            freqs.extend(timestep_embedding(t, 256, 10000.0));
        }
        let mut embedded = self.time_proj_in.forward(&freqs, tokens);
        ops::silu(&mut embedded);
        let embedded = self.time_proj_out.forward(&embedded, tokens);
        let mut silu_embedded = embedded.clone();
        ops::silu(&mut silu_embedded);
        Ok((self.linear.forward(&silu_embedded, tokens), embedded))
    }
}

// ---------------------------------------------------------------------------
// The transformer
// ---------------------------------------------------------------------------

/// `adaln_embedding_coefficient`: 6 + 3 with cross-attention AdaLN.
pub(crate) fn adaln_embedding_coefficient(cross_attention_adaln: bool) -> usize {
    6 + 3 * usize::from(cross_attention_adaln)
}

/// One `BasicAudioTransformerBlock`.
pub(crate) struct TransformerBlock {
    pub(crate) attn1: Attention,
    pub(crate) attn2: Attention,
    pub(crate) ff: FeedForward,
    scale_shift_table: Vec<f32>,
    pub(crate) prompt_scale_shift_table: Vec<f32>,
    pub(crate) norm_eps: f32,
    dim: usize,
}

impl TransformerBlock {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load(
        files: &[SafetensorsFile],
        name: &str,
        dim: usize,
        heads: usize,
        dim_head: usize,
        context_dim: usize,
        norm_eps: f32,
        cross_attention_adaln: bool,
        apply_gated_attention: bool,
        rope_split: bool,
    ) -> Result<Self> {
        let coefficient = adaln_embedding_coefficient(cross_attention_adaln);
        Ok(Self {
            attn1: Attention::load(
                files,
                &format!("{name}.audio_attn1"),
                dim,
                dim,
                heads,
                dim_head,
                norm_eps,
                apply_gated_attention,
                rope_split,
            )?,
            attn2: Attention::load(
                files,
                &format!("{name}.audio_attn2"),
                dim,
                context_dim,
                heads,
                dim_head,
                norm_eps,
                apply_gated_attention,
                false,
            )?,
            ff: FeedForward::load(files, &format!("{name}.audio_ff"), dim)?,
            scale_shift_table: load_sharded(
                files,
                &format!("{name}.audio_scale_shift_table"),
                &[coefficient, dim],
            )?,
            prompt_scale_shift_table: if cross_attention_adaln {
                load_sharded(
                    files,
                    &format!("{name}.audio_prompt_scale_shift_table"),
                    &[2, dim],
                )?
            } else {
                Vec::new()
            },
            norm_eps,
            dim,
        })
    }

    /// `scale_shift_table[start:stop][None] + timestep[...]` reshaped
    /// per token; returns `stop - start` rows of `[tokens, dim]`.
    pub(crate) fn ada_values(
        &self,
        timestep_out: &[f32],
        tokens: usize,
        start: usize,
        stop: usize,
    ) -> Vec<Vec<f32>> {
        let dim = self.dim;
        let coefficient = self.scale_shift_table.len() / dim;
        let mut values = vec![0.0f32; tokens * (stop - start) * dim];
        for row in 0..(stop - start) {
            for token in 0..tokens {
                for col in 0..dim {
                    values[(row * tokens + token) * dim + col] = self.scale_shift_table
                        [(start + row) * dim + col]
                        + timestep_out[token * coefficient * dim + (start + row) * dim + col];
                }
            }
        }
        (0..(stop - start))
            .map(|row| values[row * tokens * dim..(row + 1) * tokens * dim].to_vec())
            .collect()
    }

    /// Runs the block over `x [tokens, dim]`. `skip_audio_self_attn`
    /// (the STG perturbed pass) replaces the self-attention output with
    /// the raw values.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        x: &mut [f32],
        tokens: usize,
        context: &[f32],
        ctx_tokens: usize,
        timestep_out: &[f32],
        prompt_timestep: &[f32],
        cos: &[f32],
        sin: &[f32],
        skip_audio_self_attn: bool,
    ) -> Result<()> {
        let dim = self.dim;
        let (shift_msa, scale_msa, gate_msa) = {
            let values = self.ada_values(timestep_out, tokens, 0, 3);
            (values[0].clone(), values[1].clone(), values[2].clone())
        };
        let mut norm_x = x.to_vec();
        rms_norm(&mut norm_x, tokens, self.norm_eps);
        for row in 0..tokens {
            for col in 0..dim {
                norm_x[row * dim + col] = norm_x[row * dim + col]
                    * (1.0 + scale_msa[row * dim + col])
                    + shift_msa[row * dim + col];
            }
        }
        let attn = self.attn1.forward(
            &norm_x,
            tokens,
            &norm_x,
            tokens,
            None,
            Some(cos),
            Some(sin),
            skip_audio_self_attn,
        );
        for row in 0..tokens {
            for col in 0..dim {
                x[row * dim + col] += attn[row * dim + col] * gate_msa[row * dim + col];
            }
        }

        // Cross attention with the prompt-AdaLN key/value modulation.
        // The context width equals the block dim (the real config has
        // audio_cross_attention_dim == inner_dim).
        let (shift_q, scale_q, gate) = {
            let values = self.ada_values(timestep_out, tokens, 6, 9);
            (values[0].clone(), values[1].clone(), values[2].clone())
        };
        // prompt_timestep carries ONE row (the scalar sigma passes
        // through the prompt AdaLN as a single token), broadcast over
        // every context token.
        let mut prompt_values = vec![0.0f32; 2 * dim];
        for row in 0..2 {
            for col in 0..dim {
                prompt_values[row * dim + col] = self.prompt_scale_shift_table[row * dim + col]
                    + prompt_timestep[row * dim + col];
            }
        }
        let shift_kv = &prompt_values[..dim];
        let scale_kv = &prompt_values[dim..];

        let mut attn_input = x.to_vec();
        rms_norm(&mut attn_input, tokens, self.norm_eps);
        for row in 0..tokens {
            for col in 0..dim {
                attn_input[row * dim + col] = attn_input[row * dim + col]
                    * (1.0 + scale_q[row * dim + col])
                    + shift_q[row * dim + col];
            }
        }
        let mut modulated_context = context.to_vec();
        let ctx_dim = context.len() / ctx_tokens.max(1);
        for row in 0..ctx_tokens {
            for col in 0..ctx_dim {
                modulated_context[row * ctx_dim + col] =
                    modulated_context[row * ctx_dim + col] * (1.0 + scale_kv[col]) + shift_kv[col];
            }
        }
        let cross = self.attn2.forward(
            &attn_input,
            tokens,
            &modulated_context,
            ctx_tokens,
            None,
            None,
            None,
            false,
        );
        for row in 0..tokens {
            for col in 0..dim {
                x[row * dim + col] += cross[row * dim + col] * gate[row * dim + col];
            }
        }

        let (shift_mlp, scale_mlp, gate_mlp) = {
            let values = self.ada_values(timestep_out, tokens, 3, 6);
            (values[0].clone(), values[1].clone(), values[2].clone())
        };
        let mut fed = x.to_vec();
        rms_norm(&mut fed, tokens, self.norm_eps);
        for row in 0..tokens {
            for col in 0..dim {
                fed[row * dim + col] = fed[row * dim + col] * (1.0 + scale_mlp[row * dim + col])
                    + shift_mlp[row * dim + col];
            }
        }
        let ff = self.ff.forward(&fed, tokens);
        for row in 0..tokens {
            for col in 0..dim {
                x[row * dim + col] += ff[row * dim + col] * gate_mlp[row * dim + col];
            }
        }
        Ok(())
    }
}
