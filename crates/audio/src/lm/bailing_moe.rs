//! The bailing_moe backbone: dense attention plus mixture-of-experts MLPs.
//!
//! Port of `mlx_audio/lm/models/bailing_moe.py` (vendored from mlx-lm) at
//! the pinned mlx-audio commit. Structure: llama-shaped sequential blocks
//! with RMSNorm pre-norm; a fused biased-or-unbiased `query_key_value`
//! projection split into q/k/v, optional per-head q/k RMSNorm (named
//! `query_layernorm`/`key_layernorm`), and a `dense` output projection;
//! non-traditional rope on `rotary_dim` features; the first
//! `first_k_dense_replace` layers carry a dense SwiGLU MLP and the rest a
//! sparse MoE block (softmax or sigmoid router over stacked SwitchGLU
//! experts, optional shared-expert SwiGLU added on top), with an untied or
//! tied head over `word_embeddings`.
//!
//! Router port notes, pinned by the fixtures:
//! - Scores are f32 softmax (or sigmoid) over the gate projection. The
//!   optional `expert_bias` joins the selection scores only; the routing
//!   weights come from the unmodified scores.
//! - The reference selects the top-k with `mx.argpartition`, whose order
//!   inside the selected set is unspecified; the aggregation sum over the
//!   slot axis is exact under permutation for top-2 (IEEE addition is
//!   commutative) and within tolerance above. This port emits slots in
//!   descending score order, ties broken by lower expert index, matching
//!   the workspace invariant that slot order is router ranking order.
//! - `norm_topk_prob` divides by the selected weight sum plus 1e-20;
//!   `routed_scaling_factor` multiplies afterwards unconditionally (the
//!   reference stores `moe_router_enable_routed_scaling` but never reads
//!   it on this path; `norm_softmax` is likewise stored and unused).
//! - Group routing (`n_group > 1`) zeroes the `n_group - topk_group`
//!   groups with the smallest top-2 group scores before the top-k; the
//!   reference's argpartition leaves the dropped-group choice unspecified
//!   on ties, this port takes the lowest group indices first.

use serde_json::Value;

use crate::nn::{self, Linear, RmsNorm};
use crate::{ops, Result, SpeechError};

use super::cache::LayerCaches;
use super::engine::{lookup_embeddings, split_fused_linear, Attention, Input, Mlp};
use super::rope::Rope;
use super::switch_layers::SwitchGlu;
use super::{Bias, Weights};

/// Parses a bailing_moe `config.json`. The first block of fields matches
/// the required reference dataclass fields, the rest carry its defaults.
pub fn from_json(value: &Value) -> Result<BailingMoeConfig> {
    let bool_or =
        |key: &str, default: bool| value.get(key).and_then(Value::as_bool).unwrap_or(default);
    let num_attention_heads = nn::json::usize_field(value, "num_attention_heads")?;
    Ok(BailingMoeConfig {
        model_type: nn::json::text_field(value, "model_type")?,
        hidden_size: nn::json::usize_field(value, "hidden_size")?,
        intermediate_size: nn::json::usize_field(value, "intermediate_size")?,
        max_position_embeddings: nn::json::usize_field(value, "max_position_embeddings")?,
        moe_intermediate_size: nn::json::usize_field(value, "moe_intermediate_size")?,
        num_experts: nn::json::usize_field(value, "num_experts")?,
        num_shared_experts: nn::json::usize_field(value, "num_shared_experts")?,
        norm_topk_prob: nn::json::bool_field(value, "norm_topk_prob")?,
        num_attention_heads,
        num_experts_per_tok: nn::json::usize_field(value, "num_experts_per_tok")?,
        num_hidden_layers: nn::json::usize_field(value, "num_hidden_layers")?,
        num_key_value_heads: nn::json::usize_field(value, "num_key_value_heads")?,
        rms_norm_eps: nn::json::required(value, "rms_norm_eps")?
            .as_f64()
            .filter(|v| *v > 0.0)
            .ok_or_else(|| SpeechError::BadConfig {
                field: "rms_norm_eps".into(),
                why: "must be a positive number".into(),
            })? as f32,
        rope_theta: nn::json::required(value, "rope_theta")?
            .as_f64()
            .filter(|v| *v > 0.0)
            .ok_or_else(|| SpeechError::BadConfig {
                field: "rope_theta".into(),
                why: "must be a positive number".into(),
            })? as f32,
        vocab_size: nn::json::usize_field(value, "vocab_size")?,
        first_k_dense_replace: nn::json::usize_field(value, "first_k_dense_replace")?,
        rope_scaling: value.get("rope_scaling").cloned().filter(|v| !v.is_null()),
        use_bias: bool_or("use_bias", false),
        use_qkv_bias: bool_or("use_qkv_bias", false),
        norm_head: bool_or("norm_head", false),
        // Stored for parity; the reference never reads it on this path.
        norm_softmax: bool_or("norm_softmax", false),
        use_qk_norm: bool_or("use_qk_norm", false),
        tie_word_embeddings: bool_or("tie_word_embeddings", false),
        partial_rotary_factor: value
            .get("partial_rotary_factor")
            .and_then(Value::as_f64)
            .unwrap_or(1.0) as f32,
        rotary_dim: value
            .get("rotary_dim")
            .and_then(Value::as_u64)
            .map(|n| n as usize),
        moe_router_enable_expert_bias: bool_or("moe_router_enable_expert_bias", false),
        // Stored for parity; the reference always applies
        // routed_scaling_factor on this path.
        moe_router_enable_routed_scaling: bool_or("moe_router_enable_routed_scaling", true),
        routed_scaling_factor: value
            .get("routed_scaling_factor")
            .and_then(Value::as_f64)
            .unwrap_or(1.0) as f32,
        score_function: value
            .get("score_function")
            .and_then(Value::as_str)
            .unwrap_or("softmax")
            .to_owned(),
        n_group: value
            .get("n_group")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(1),
        topk_group: value
            .get("topk_group")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(4),
        moe_shared_expert_intermediate_size: value
            .get("moe_shared_expert_intermediate_size")
            .and_then(Value::as_u64)
            .map(|n| n as usize),
        moe_router_enable_shared_expert: bool_or("moe_router_enable_shared_expert", true),
    })
}

/// Loads a bailing_moe backbone from `weights`. Bailing checkpoints use
/// `model.word_embeddings` rather than `model.embed_tokens`, so the prefix
/// autodetect does not fire and the tensor names inside the loader carry
/// the `model.` prefix explicitly.
pub fn load(weights: &Weights, value: &Value) -> Result<BailingMoeModel> {
    BailingMoeModel::load(weights, value)
}

/// The parsed bailing_moe configuration (`ModelArgs`).
#[derive(Debug, Clone)]
pub struct BailingMoeConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub moe_intermediate_size: usize,
    pub num_experts: usize,
    pub num_shared_experts: usize,
    pub norm_topk_prob: bool,
    pub num_attention_heads: usize,
    pub num_experts_per_tok: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: usize,
    pub first_k_dense_replace: usize,
    pub rope_scaling: Option<Value>,
    pub use_bias: bool,
    pub use_qkv_bias: bool,
    pub norm_head: bool,
    /// Stored for parity; unused by the reference forward.
    #[allow(dead_code)]
    pub norm_softmax: bool,
    pub use_qk_norm: bool,
    pub tie_word_embeddings: bool,
    pub partial_rotary_factor: f32,
    pub rotary_dim: Option<usize>,
    pub moe_router_enable_expert_bias: bool,
    /// Stored for parity; the reference multiplies unconditionally.
    #[allow(dead_code)]
    pub moe_router_enable_routed_scaling: bool,
    pub routed_scaling_factor: f32,
    pub score_function: String,
    pub n_group: usize,
    pub topk_group: usize,
    pub moe_shared_expert_intermediate_size: Option<usize>,
    pub moe_router_enable_shared_expert: bool,
}

/// The router (`BailingMoeGate`): a linear over the experts plus optional
/// correction bias, then `group_expert_select`.
struct MoeGate {
    gate_proj: Linear,
    expert_bias: Option<Vec<f32>>,
    top_k: usize,
    n_group: usize,
    topk_group: usize,
    routed_scaling_factor: f32,
    norm_topk_prob: bool,
    score_function: String,
}

impl MoeGate {
    /// Port of `group_expert_select` for one batch. Returns the selected
    /// expert ids and weights, both `[len, top_k]`, slots in descending
    /// score order.
    fn select(&self, x: &[f32], len: usize) -> Result<(Vec<i32>, Vec<f32>)> {
        match self.score_function.as_str() {
            "softmax" | "sigmoid" => {}
            other => {
                return Err(SpeechError::Unsupported {
                    why: format!("moe score_function {other}"),
                })
            }
        }
        if self.top_k == 0 || self.top_k > self.gate_proj.output {
            return Err(SpeechError::BadConfig {
                field: "num_experts_per_tok".into(),
                why: format!(
                    "must select between 1 and {} experts",
                    self.gate_proj.output
                ),
            });
        }
        let gates = self.gate_proj.forward(x, len);
        let experts = self.gate_proj.output;
        let mut indices = vec![0i32; len * self.top_k];
        let mut weights = vec![0.0f32; len * self.top_k];
        let mut scores = vec![0.0f32; experts];
        let mut group_scores = vec![0.0f32; self.n_group];
        for token in 0..len {
            let row = &gates[token * experts..(token + 1) * experts];
            // Router scores in f32 regardless of the storage dtype.
            scores.copy_from_slice(row);
            match self.score_function.as_str() {
                "sigmoid" => {
                    for score in &mut scores {
                        *score = ops::sigmoid(*score);
                    }
                }
                _ => ops::softmax_row(&mut scores),
            }
            // Weights come from the original scores; selection may use the
            // correction bias.
            let mut selection = scores.clone();
            if let Some(bias) = &self.expert_bias {
                for (score, bias) in selection.iter_mut().zip(bias) {
                    *score += bias;
                }
            }
            if self.n_group > 1 {
                if experts % self.n_group != 0 {
                    return Err(SpeechError::BadConfig {
                        field: "n_group".into(),
                        why: format!("{experts} experts not divisible by {}", self.n_group),
                    });
                }
                let group_size = experts / self.n_group;
                // Sum of the top-2 scores per group.
                for group in 0..self.n_group {
                    let window = &selection[group * group_size..(group + 1) * group_size];
                    let mut first = f32::NEG_INFINITY;
                    let mut second = f32::NEG_INFINITY;
                    for &score in window {
                        if score > first {
                            second = first;
                            first = score;
                        } else if score > second {
                            second = score;
                        }
                    }
                    group_scores[group] = first + second;
                }
                // Drop the n_group - topk_group weakest groups (lowest
                // indices first on ties; the reference argpartition leaves
                // tie order unspecified).
                let mut order: Vec<usize> = (0..self.n_group).collect();
                order.sort_by(|&a, &b| {
                    group_scores[a]
                        .partial_cmp(&group_scores[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.cmp(&b))
                });
                let drop = self.n_group.saturating_sub(self.topk_group);
                for &group in &order[..drop.min(order.len())] {
                    for score in &mut selection[group * group_size..(group + 1) * group_size] {
                        *score = 0.0;
                    }
                }
            }
            // Top-k by selection score, descending, ties to the lower
            // expert index.
            let mut order: Vec<usize> = (0..experts).collect();
            order.sort_by(|&a, &b| {
                selection[b]
                    .partial_cmp(&selection[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            let mut total = 0.0f32;
            for (slot, &expert) in order.iter().take(self.top_k).enumerate() {
                let weight = scores[expert];
                indices[token * self.top_k + slot] = expert as i32;
                weights[token * self.top_k + slot] = weight;
                total += weight;
            }
            if self.top_k > 1 && self.norm_topk_prob {
                // The reference guards with + 1e-20.
                let denominator = total + 1e-20;
                for weight in &mut weights[token * self.top_k..(token + 1) * self.top_k] {
                    *weight /= denominator;
                }
            }
            for weight in &mut weights[token * self.top_k..(token + 1) * self.top_k] {
                *weight *= self.routed_scaling_factor;
            }
        }
        Ok((indices, weights))
    }
}

/// The sparse MoE block (`BailingMoeSparseMoeBlock`).
struct SparseMoe {
    switch_mlp: SwitchGlu,
    gate: MoeGate,
    shared_experts: Option<Mlp>,
    top_k: usize,
}

impl SparseMoe {
    /// `sum_s expert_s(x) * weight_s + shared_experts(x)`.
    fn forward(&self, x: &[f32], len: usize) -> Result<Vec<f32>> {
        let (indices, weights) = self.gate.select(x, len)?;
        let expert_outputs = self.switch_mlp.forward(x, len, &indices)?;
        let mut out = vec![0.0f32; len * self.switch_mlp.output_dims()];
        for token in 0..len {
            let base = token * self.top_k;
            for slot in 0..self.top_k {
                let weight = weights[base + slot];
                let source = (base + slot) * self.switch_mlp.output_dims();
                for (col, out_value) in out[token * self.switch_mlp.output_dims()
                    ..(token + 1) * self.switch_mlp.output_dims()]
                    .iter_mut()
                    .enumerate()
                {
                    *out_value += weight * expert_outputs[source + col];
                }
            }
        }
        if let Some(shared) = &self.shared_experts {
            let shared_out = shared.forward(x, len);
            for (out_value, shared_value) in out.iter_mut().zip(&shared_out) {
                *out_value += shared_value;
            }
        }
        Ok(out)
    }
}

enum MlpBlock {
    Dense(Mlp),
    Sparse(Box<SparseMoe>),
}

struct Block {
    attention: Attention,
    mlp: MlpBlock,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

/// A loaded bailing_moe backbone ready to run.
pub struct BailingMoeModel {
    pub config: BailingMoeConfig,
    word_embeddings: Vec<f32>,
    layers: Vec<Block>,
    norm: RmsNorm,
    /// Untied head; a tied head reuses `word_embeddings`.
    lm_head: Option<Linear>,
}

impl BailingMoeModel {
    /// Loads every weight the configuration names. Bailing checkpoints use
    /// `model.word_embeddings` rather than `model.embed_tokens`, so the
    /// prefix autodetect does not fire: open the weights with an explicit
    /// `model.` prefix (the LM head stays prefix-free).
    pub fn load(weights: &Weights, value: &Value) -> Result<Self> {
        let config = from_json(value)?;
        Self::from_config(weights, config)
    }

    /// Loads from an already parsed config.
    pub fn from_config(weights: &Weights, config: BailingMoeConfig) -> Result<Self> {
        let dim = config.hidden_size;
        if dim % config.num_attention_heads != 0 {
            return Err(SpeechError::BadConfig {
                field: "hidden_size".into(),
                why: format!(
                    "{} not divisible by num_attention_heads {}",
                    dim, config.num_attention_heads
                ),
            });
        }
        let head_dim = dim / config.num_attention_heads;
        let q_width = config.num_attention_heads * head_dim;
        let kv_width = config.num_key_value_heads * head_dim;
        let mlp_bias = if config.use_bias {
            Bias::Required
        } else {
            Bias::None
        };
        let qkv_bias = if config.use_qkv_bias {
            Bias::Required
        } else {
            Bias::None
        };
        let rotary_dim = match config.rotary_dim {
            Some(dim) => dim,
            None => (head_dim as f64 * config.partial_rotary_factor as f64) as usize,
        };
        let rope = Rope::initialize(
            rotary_dim,
            config.rope_theta,
            false,
            config.rope_scaling.as_ref(),
            Some(config.max_position_embeddings),
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let layer = format!("model.layers.{index}.");
            let fused = weights.linear(
                &format!("{layer}attention.query_key_value"),
                dim,
                q_width + 2 * kv_width,
                qkv_bias,
            )?;
            let (q_proj, k_proj, v_proj) = split_fused_linear(&fused, q_width, kv_width);
            let (q_norm, k_norm) = if config.use_qk_norm {
                (
                    Some(RmsNorm::new(
                        weights
                            .tensor(&format!("{layer}attention.query_layernorm"), &[head_dim])?,
                        config.rms_norm_eps,
                    )),
                    Some(RmsNorm::new(
                        weights.tensor(&format!("{layer}attention.key_layernorm"), &[head_dim])?,
                        config.rms_norm_eps,
                    )),
                )
            } else {
                (None, None)
            };
            let mlp = if index >= config.first_k_dense_replace {
                MlpBlock::Sparse(Box::new(Self::load_sparse_moe(weights, &config, &layer)?))
            } else {
                MlpBlock::Dense(Self::load_dense_mlp(
                    weights,
                    &config,
                    &format!("{layer}mlp."),
                    config.intermediate_size,
                )?)
            };
            layers.push(Block {
                attention: Attention {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj: weights.linear(
                        &format!("{layer}attention.dense"),
                        q_width,
                        dim,
                        mlp_bias,
                    )?,
                    q_norm,
                    k_norm,
                    n_heads: config.num_attention_heads,
                    n_kv_heads: config.num_key_value_heads,
                    head_dim,
                    scale: (head_dim as f64).powf(-0.5) as f32,
                    rope: Some(rope.clone()),
                    window: None,
                },
                mlp,
                input_layernorm: RmsNorm::new(
                    weights.tensor(&format!("{layer}input_layernorm"), &[dim])?,
                    config.rms_norm_eps,
                ),
                post_attention_layernorm: RmsNorm::new(
                    weights.tensor(&format!("{layer}post_attention_layernorm"), &[dim])?,
                    config.rms_norm_eps,
                ),
            });
        }
        let word_embeddings = weights.embedding("model.word_embeddings", config.vocab_size, dim)?;
        if config.norm_head && config.tie_word_embeddings {
            return Err(SpeechError::BadConfig {
                field: "norm_head".into(),
                why: "requires an untied lm_head; the reference sanitize would KeyError on the missing tensor".into(),
            });
        }
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            // mlx-lm saves keep the head at the root; the bailingmm
            // family nests it one `model.` deeper
            // (`model.lm_head.weight`), so both absolute names load.
            let mut head =
                match weights.linear_absolute("lm_head", dim, config.vocab_size, Bias::None) {
                    Ok(head) => head,
                    Err(_) => weights.linear_absolute(
                        "model.lm_head",
                        dim,
                        config.vocab_size,
                        Bias::None,
                    )?,
                };
            if config.norm_head {
                // Reference sanitize: every column (fixed hidden index,
                // varying vocab row) is normalized with a 1e-7 guard.
                let cols = config.vocab_size;
                let rows = dim;
                let mut column = vec![0.0f32; rows];
                for col in 0..cols {
                    for (r, value) in column.iter_mut().enumerate() {
                        *value = head.weight[r * cols + col];
                    }
                    let norm = column.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-7;
                    for (r, value) in column.iter().enumerate() {
                        head.weight[r * cols + col] = value / norm;
                    }
                }
            }
            Some(head)
        };
        let norm = RmsNorm::new(weights.tensor("model.norm", &[dim])?, config.rms_norm_eps);
        Ok(Self {
            config,
            word_embeddings,
            layers,
            norm,
            lm_head,
        })
    }

    /// `prefix` is the full tensor prefix up to the projection names:
    /// `model.layers.N.mlp.` for dense blocks or
    /// `model.layers.N.mlp.shared_experts.` inside the MoE blocks.
    fn load_dense_mlp(
        weights: &Weights,
        config: &BailingMoeConfig,
        prefix: &str,
        intermediate_size: usize,
    ) -> Result<Mlp> {
        let dim = config.hidden_size;
        let bias = if config.use_bias {
            Bias::Required
        } else {
            Bias::None
        };
        Ok(Mlp {
            gate_proj: weights.linear(
                &format!("{prefix}gate_proj"),
                dim,
                intermediate_size,
                bias,
            )?,
            up_proj: weights.linear(&format!("{prefix}up_proj"), dim, intermediate_size, bias)?,
            down_proj: weights.linear(
                &format!("{prefix}down_proj"),
                intermediate_size,
                dim,
                bias,
            )?,
        })
    }

    fn load_sparse_moe(
        weights: &Weights,
        config: &BailingMoeConfig,
        layer: &str,
    ) -> Result<SparseMoe> {
        let dim = config.hidden_size;
        let hidden = config.moe_intermediate_size;
        let experts = config.num_experts;
        // SwitchLinear stores an [experts, out] bias when the checkpoint
        // trains use_bias; the reference SwitchGLU default is bias-free.
        let switch_bias = |name: String, out: usize| -> Result<Option<Vec<f32>>> {
            if config.use_bias {
                Ok(Some(weights.tensor(&name, &[experts, out])?))
            } else {
                Ok(None)
            }
        };
        let switch_mlp = SwitchGlu::new(
            weights.tensor(
                &format!("{layer}mlp.switch_mlp.gate_proj"),
                &[experts, hidden, dim],
            )?,
            weights.tensor(
                &format!("{layer}mlp.switch_mlp.up_proj"),
                &[experts, hidden, dim],
            )?,
            weights.tensor(
                &format!("{layer}mlp.switch_mlp.down_proj"),
                &[experts, dim, hidden],
            )?,
            switch_bias(format!("{layer}mlp.switch_mlp.gate_proj.bias"), hidden)?,
            switch_bias(format!("{layer}mlp.switch_mlp.up_proj.bias"), hidden)?,
            switch_bias(format!("{layer}mlp.switch_mlp.down_proj.bias"), dim)?,
            experts,
            dim,
            hidden,
        );
        let gate_proj = weights.tensor(&format!("{layer}mlp.gate.gate_proj"), &[experts, dim])?;
        let expert_bias = if config.moe_router_enable_expert_bias {
            Some(weights.raw_tensor(&format!("{layer}mlp.gate.expert_bias"), &[experts])?)
        } else {
            None
        };
        let shared_experts =
            if config.num_shared_experts > 0 && config.moe_router_enable_shared_expert {
                // shared_dim * num_shared_experts, defaulting shared_dim to the
                // MoE intermediate size.
                let shared_dim = config.moe_shared_expert_intermediate_size.unwrap_or(hidden);
                Some(Self::load_dense_mlp(
                    weights,
                    config,
                    &format!("{layer}mlp.shared_experts."),
                    shared_dim * config.num_shared_experts,
                )?)
            } else {
                None
            };
        Ok(SparseMoe {
            switch_mlp,
            gate: MoeGate {
                gate_proj: Linear::new(gate_proj, None, dim, experts),
                expert_bias,
                top_k: config.num_experts_per_tok,
                n_group: config.n_group,
                topk_group: config.topk_group,
                routed_scaling_factor: config.routed_scaling_factor,
                norm_topk_prob: config.norm_topk_prob,
                score_function: config.score_function.clone(),
            },
            shared_experts,
            top_k: config.num_experts_per_tok,
        })
    }

    /// The word embedding table (`model.word_embeddings`), exposed for
    /// the bailingmm family: its prompt assembly embeds pre-tokenized
    /// ids itself and feeds `Input::Embeddings` from the result.
    pub fn word_embeddings(&self) -> &[f32] {
        &self.word_embeddings
    }

    /// Runs the backbone and returns logits `[positions, vocab_size]`.
    pub fn forward(
        &mut self,
        input: Input<'_>,
        caches: Option<&mut LayerCaches>,
    ) -> Result<Vec<f32>> {
        let projected = self.forward_hidden(input, caches)?;
        let len = projected.len() / self.config.hidden_size;
        Ok(match &self.lm_head {
            Some(head) => head.forward(&projected, len),
            None => ops::linear(
                &projected,
                &self.word_embeddings,
                None,
                len,
                self.config.hidden_size,
                self.config.vocab_size,
            ),
        })
    }

    /// Runs the backbone up to the final norm and returns the hidden state
    /// `[positions, hidden_size]`.
    pub fn forward_hidden(
        &mut self,
        input: Input<'_>,
        mut caches: Option<&mut LayerCaches>,
    ) -> Result<Vec<f32>> {
        let dim = self.config.hidden_size;
        let len = input.len(dim)?;
        if len == 0 {
            return Err(SpeechError::Input {
                why: "empty forward input".into(),
            });
        }
        let mut h = match input {
            Input::Tokens(tokens) => {
                lookup_embeddings(&self.word_embeddings, self.config.vocab_size, dim, tokens)?
            }
            Input::Embeddings(values) => values.to_vec(),
        };
        let offset = caches
            .as_deref()
            .and_then(|c| c.first())
            .map(|c| c.offset)
            .unwrap_or(0);
        if let Some(caches) = caches.as_deref() {
            if caches.len() != self.layers.len() {
                return Err(SpeechError::Input {
                    why: format!(
                        "{} layer caches provided for {} layers",
                        caches.len(),
                        self.layers.len()
                    ),
                });
            }
        }
        for (index, block) in self.layers.iter_mut().enumerate() {
            let mut attended = h.clone();
            block.input_layernorm.apply(&mut attended, len);
            let attn = block.attention.forward(
                &attended,
                len,
                offset,
                caches
                    .as_deref_mut()
                    .map(|c| &mut c[index] as &mut dyn super::cache::LayerKv),
            )?;
            for (row, out) in attn.chunks_exact(dim).enumerate() {
                let base = row * dim;
                for (col, value) in out.iter().enumerate() {
                    h[base + col] += value;
                }
            }
            let mut fed = h.clone();
            block.post_attention_layernorm.apply(&mut fed, len);
            let mlp = match &block.mlp {
                MlpBlock::Dense(mlp) => mlp.forward(&fed, len),
                MlpBlock::Sparse(moe) => moe.forward(&fed, len)?,
            };
            for (row, out) in mlp.chunks_exact(dim).enumerate() {
                let base = row * dim;
                for (col, value) in out.iter().enumerate() {
                    h[base + col] += value;
                }
            }
        }
        self.norm.apply(&mut h, len);
        Ok(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Router behavior the decoder fixture cannot pin: the top-2 slot
    /// aggregation is exact under permutation, so slot ORDER (the
    /// workspace reduction-order invariant), the bias-only-selection rule,
    /// norm_topk_prob, and the unconditional scaling factor all get
    /// dedicated checks here.
    #[test]
    fn gate_ranks_by_score_and_applies_norm_and_scaling() {
        // Four experts; gate weights picked so the softmax scores rank
        // experts 2 > 0 > 3 > 1.
        let mut weight = vec![0.0f32; 4];
        weight[2] = 3.0;
        weight[0] = 2.0;
        weight[3] = 1.0;
        weight[1] = -1.0;
        let gate = MoeGate {
            gate_proj: Linear::new(weight, None, 1, 4),
            expert_bias: None,
            top_k: 2,
            n_group: 1,
            topk_group: 4,
            routed_scaling_factor: 2.0,
            norm_topk_prob: true,
            score_function: "softmax".to_owned(),
        };
        let (indices, weights) = gate.select(&[1.0], 1).unwrap();
        assert_eq!(indices, vec![2, 0]);
        // softmax([3, -1, 2, 1]) top-2 = (e2, e0); normalized then doubled.
        let e = |v: f32| v.exp();
        let denom = e(3.0) + e(-1.0) + e(2.0) + e(1.0);
        let w2 = e(3.0) / denom;
        let w0 = e(2.0) / denom;
        let norm = w2 + w0 + 1e-20;
        let expected = [w2 / norm * 2.0, w0 / norm * 2.0];
        for (got, want) in weights.iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
    }

    #[test]
    fn gate_bias_affects_selection_but_not_weights() {
        // Expert 1 has the highest raw score; a large positive bias on
        // expert 3 must move the SELECTION to include 3 while the weights
        // stay the unbiased scores.
        let mut weight = vec![0.0f32; 4];
        weight[1] = 3.0;
        weight[0] = 2.0;
        weight[3] = 1.0;
        let gate = MoeGate {
            gate_proj: Linear::new(weight, None, 1, 4),
            expert_bias: Some(vec![0.0, 0.0, 0.0, 100.0]),
            top_k: 2,
            n_group: 1,
            topk_group: 4,
            routed_scaling_factor: 1.0,
            norm_topk_prob: false,
            score_function: "softmax".to_owned(),
        };
        let (indices, weights) = gate.select(&[1.0], 1).unwrap();
        // Selection: biased scores pick 3 and 1.
        assert_eq!(indices, vec![3, 1]);
        // Weights: the unbiased softmax scores of 3 and 1.
        let denom = f32::exp(3.0) + f32::exp(2.0) + f32::exp(1.0) + 1.0;
        let want3 = f32::exp(1.0) / denom;
        let want1 = f32::exp(3.0) / denom;
        assert!((weights[0] - want3).abs() < 1e-6);
        assert!((weights[1] - want1).abs() < 1e-6);
    }

    #[test]
    fn gate_ties_break_toward_the_lower_expert_index() {
        let weight = vec![1.0f32; 4];
        let gate = MoeGate {
            gate_proj: Linear::new(weight, None, 1, 4),
            expert_bias: None,
            top_k: 2,
            n_group: 1,
            topk_group: 4,
            routed_scaling_factor: 1.0,
            norm_topk_prob: false,
            score_function: "softmax".to_owned(),
        };
        let (indices, _) = gate.select(&[1.0], 1).unwrap();
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn unsupported_score_function_is_named() {
        let gate = MoeGate {
            gate_proj: Linear::new(vec![0.0; 4], None, 1, 4),
            expert_bias: None,
            top_k: 2,
            n_group: 1,
            topk_group: 4,
            routed_scaling_factor: 1.0,
            norm_topk_prob: false,
            score_function: "tanh".to_owned(),
        };
        let error = gate.select(&[0.0], 1).unwrap_err();
        assert!(error.to_string().contains("score_function tanh"), "{error}");
    }
}
