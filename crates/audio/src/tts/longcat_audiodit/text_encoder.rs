//! UMT5 text encoder for AudioDiT (`text_encoder.py`): the T5 v1.1
//! encoder structure with a relative-attention-bias table in every
//! block (the UMT5 signature), RMS-style layer norms, gated
//! exact-GELU feed forwards, and no score scaling.

use crate::nn::{Linear, RmsNorm};
use crate::{ops, Result, SpeechError};
use turbospark_model_io::safetensors::SafetensorsFile;

fn tensor(file: &SafetensorsFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    // Only the element count is fixed here; the file keeps the MLX
    // tree shapes.
    let descriptor = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
        name: name.to_owned(),
        why: "tensor is missing".into(),
    })?;
    if descriptor.shape.iter().product::<usize>() != shape.iter().product::<usize>() {
        return Err(SpeechError::Tensor {
            name: name.to_owned(),
            why: format!("expected {shape:?} elements, got {:?}", descriptor.shape),
        });
    }
    crate::nn::load_tensor(file, name, &descriptor.shape)
}

fn linear(
    file: &SafetensorsFile,
    name: &str,
    input: usize,
    output: usize,
    bias: bool,
) -> Result<Linear> {
    let weight = tensor(file, &format!("{name}.weight"), &[input * output])?;
    let bias_value = if bias {
        Some(tensor(file, &format!("{name}.bias"), &[output])?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias_value, input, output))
}

/// `T5Attention._relative_position_bucket` (bidirectional): the sign
/// selects the upper half, the magnitude is log-bucketed between
/// `max_exact` and the half width.
pub(crate) fn relative_position_bucket(
    relative: &[i64],
    num_buckets: usize,
    max_distance: usize,
) -> Vec<usize> {
    let half = num_buckets / 2;
    let max_exact = half / 2;
    relative
        .iter()
        .map(|&r| {
            let mut bucket = if r > 0 { half } else { 0 };
            let magnitude = r.unsigned_abs() as usize;
            let converted = if magnitude < max_exact {
                magnitude
            } else {
                let large = (magnitude as f64 / max_exact as f64).ln()
                    / (max_distance as f64 / max_exact as f64).ln()
                    * (half - max_exact) as f64
                    + max_exact as f64;
                (large as usize).min(half - 1)
            };
            bucket += converted;
            bucket
        })
        .collect()
}

/// One UMT5 encoder block: pre-norm attention with its own relative
/// bias table, then a pre-norm (optionally gated) feed forward.
struct T5Block {
    attention: T5Attention,
    layer_norm_sa: RmsNorm,
    gated: bool,
    wi_0: Linear,
    wi_1: Linear,
    wo: Linear,
    layer_norm_ff: RmsNorm,
}

impl T5Block {
    #[allow(
        clippy::too_many_arguments,
        reason = "block construction carries the config geometry explicitly"
    )]
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        d_model: usize,
        d_ff: usize,
        n_heads: usize,
        d_kv: usize,
        num_buckets: usize,
        max_distance: usize,
        gated: bool,
        eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            attention: T5Attention::load(
                file,
                &format!("{prefix}.SelfAttention"),
                d_model,
                n_heads,
                d_kv,
                num_buckets,
                max_distance,
            )?,
            layer_norm_sa: RmsNorm::new(
                tensor(file, &format!("{prefix}.layer_norm_sa.weight"), &[d_model])?,
                eps,
            ),
            gated,
            wi_0: linear(
                file,
                &format!("{prefix}.DenseReluDense.wi_0"),
                d_model,
                d_ff,
                false,
            )?,
            wi_1: linear(
                file,
                &format!("{prefix}.DenseReluDense.wi_1"),
                d_model,
                d_ff,
                false,
            )?,
            wo: linear(
                file,
                &format!("{prefix}.DenseReluDense.wo"),
                d_ff,
                d_model,
                false,
            )?,
            layer_norm_ff: RmsNorm::new(
                tensor(file, &format!("{prefix}.layer_norm_ff.weight"), &[d_model])?,
                eps,
            ),
        })
    }

    /// `hidden [seq, d_model]` updated in place; the additive key mask
    /// (0 kept, -1e9 denied) applies to every query row.
    fn forward(&self, hidden: &mut [f32], seq: usize, key_mask: &[f32]) -> Result<()> {
        let mut normed = hidden.to_vec();
        self.layer_norm_sa.apply(&mut normed, seq);
        let attended = self.attention.forward(&normed, seq, key_mask)?;
        for (dst, src) in hidden.iter_mut().zip(attended) {
            *dst += src;
        }
        let mut normed = hidden.to_vec();
        self.layer_norm_ff.apply(&mut normed, seq);
        let mut hidden_gates = self.wi_0.forward(&normed, seq);
        ops::gelu_erf(&mut hidden_gates);
        let mut feed = hidden_gates;
        if self.gated {
            let linear_part = self.wi_1.forward(&normed, seq);
            for (g, l) in feed.iter_mut().zip(linear_part) {
                *g *= l;
            }
        }
        let out = self.wo.forward(&feed, seq);
        for (dst, src) in hidden.iter_mut().zip(out) {
            *dst += src;
        }
        Ok(())
    }
}

/// T5 attention with a per-block learned relative-position bias and no
/// score scaling.
struct T5Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    relative_attention_bias: Vec<f32>,
    num_buckets: usize,
    max_distance: usize,
    n_heads: usize,
    d_kv: usize,
}

impl T5Attention {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        d_model: usize,
        n_heads: usize,
        d_kv: usize,
        num_buckets: usize,
        max_distance: usize,
    ) -> Result<Self> {
        Ok(Self {
            q: linear(file, &format!("{prefix}.q"), d_model, n_heads * d_kv, false)?,
            k: linear(file, &format!("{prefix}.k"), d_model, n_heads * d_kv, false)?,
            v: linear(file, &format!("{prefix}.v"), d_model, n_heads * d_kv, false)?,
            o: linear(file, &format!("{prefix}.o"), n_heads * d_kv, d_model, false)?,
            relative_attention_bias: tensor(
                file,
                &format!("{prefix}.relative_attention_bias.weight"),
                &[num_buckets * n_heads],
            )?,
            num_buckets,
            max_distance,
            n_heads,
            d_kv,
        })
    }

    /// `compute_bias` plus the key mask: the additive `[n_heads, seq,
    /// seq]` bias.
    fn bias(&self, seq: usize, key_mask: &[f32]) -> Vec<f32> {
        let mut relative = Vec::with_capacity(seq * seq);
        for i in 0..seq {
            for j in 0..seq {
                relative.push(j as i64 - i as i64);
            }
        }
        let buckets = relative_position_bucket(&relative, self.num_buckets, self.max_distance);
        let mut out = vec![0.0f32; self.n_heads * seq * seq];
        for (p, &bucket) in buckets.iter().enumerate() {
            let denied = (1.0 - key_mask[p % seq]) * -1.0e9;
            for h in 0..self.n_heads {
                out[h * seq * seq + p] =
                    self.relative_attention_bias[bucket * self.n_heads + h] + denied;
            }
        }
        out
    }

    /// `x [seq, inner] -> [seq, inner]`.
    fn forward(&self, x: &[f32], seq: usize, key_mask: &[f32]) -> Result<Vec<f32>> {
        let inner = self.n_heads * self.d_kv;
        let projected_q = self.q.forward(x, seq);
        let projected_k = self.k.forward(x, seq);
        let projected_v = self.v.forward(x, seq);
        let q = ops::split_heads(&projected_q, seq, self.n_heads, self.d_kv);
        let k = ops::split_heads(&projected_k, seq, self.n_heads, self.d_kv);
        let v = ops::split_heads(&projected_v, seq, self.n_heads, self.d_kv);

        let bias = self.bias(seq, key_mask);
        let mut out = vec![0.0f32; seq * inner];
        for h in 0..self.n_heads {
            let q_head = &q[h * seq * self.d_kv..(h + 1) * seq * self.d_kv];
            let k_head = &k[h * seq * self.d_kv..(h + 1) * seq * self.d_kv];
            let v_head = &v[h * seq * self.d_kv..(h + 1) * seq * self.d_kv];
            let bias_head = &bias[h * seq * seq..(h + 1) * seq * seq];
            let attended = ops::sdpa(
                q_head,
                k_head,
                v_head,
                Some(bias_head),
                seq,
                seq,
                self.d_kv,
                self.d_kv,
                1.0,
            );
            out[h * seq * self.d_kv..(h + 1) * seq * self.d_kv].copy_from_slice(&attended);
        }
        let merged = ops::merge_heads(&out, seq, self.n_heads, self.d_kv);
        Ok(self.o.forward(&merged, seq))
    }
}

/// The mask is read per key position, so it must cover exactly the
/// tokens. A short mask would index past its end in the attention bias.
pub(crate) fn check_mask_len(input_ids: &[i32], mask: &[f32]) -> Result<()> {
    if mask.len() != input_ids.len() {
        return Err(SpeechError::Input {
            why: format!(
                "attention mask length {} does not match {} input ids",
                mask.len(),
                input_ids.len()
            ),
        });
    }
    Ok(())
}

/// The `UMT5Encoder`: shared embedding, per-block relative-bias
/// blocks, final norm. Returns `(last_hidden_state,
/// initial_embedding)`.
pub(crate) struct Umt5Encoder {
    shared: Vec<f32>,
    blocks: Vec<T5Block>,
    final_layer_norm: RmsNorm,
    d_model: usize,
}

impl Umt5Encoder {
    pub(crate) fn load(
        file: &SafetensorsFile,
        prefix: &str,
        config: &super::TextEncoderConfig,
    ) -> Result<Self> {
        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            blocks.push(T5Block::load(
                file,
                &format!("{prefix}.block.{i}"),
                config.d_model,
                config.d_ff,
                config.num_heads,
                config.d_kv,
                config.relative_attention_num_buckets,
                config.relative_attention_max_distance,
                config.is_gated_act,
                config.layer_norm_epsilon as f32,
            )?);
        }
        Ok(Self {
            shared: tensor(
                file,
                &format!("{prefix}.shared.weight"),
                &[config.vocab_size * config.d_model],
            )?,
            blocks,
            final_layer_norm: RmsNorm::new(
                tensor(
                    file,
                    &format!("{prefix}.final_layer_norm.weight"),
                    &[config.d_model],
                )?,
                config.layer_norm_epsilon as f32,
            ),
            d_model: config.d_model,
        })
    }

    /// `input_ids [seq]` with the attention mask (1.0 kept). Returns
    /// `(last_hidden [seq, d_model], initial_embedding [seq,
    /// d_model])`.
    pub(crate) fn forward(&self, input_ids: &[i32], mask: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        check_mask_len(input_ids, mask)?;
        let seq = input_ids.len();
        let d_model = self.d_model;
        let mut initial = vec![0.0f32; seq * d_model];
        for (t, &id) in input_ids.iter().enumerate() {
            if id < 0 {
                return Err(SpeechError::Input {
                    why: format!("negative text token id {id}"),
                });
            }
            let r = id as usize;
            if (r + 1) * d_model > self.shared.len() {
                return Err(SpeechError::Input {
                    why: format!("text token id {id} outside the shared table"),
                });
            }
            initial[t * d_model..(t + 1) * d_model]
                .copy_from_slice(&self.shared[r * d_model..(r + 1) * d_model]);
        }
        let mut hidden = initial.clone();
        let key_mask: Vec<f32> = mask.to_vec();
        for block in &self.blocks {
            block.forward(&mut hidden, seq, &key_mask)?;
        }
        self.final_layer_norm.apply(&mut hidden, seq);
        Ok((hidden, initial))
    }
}
