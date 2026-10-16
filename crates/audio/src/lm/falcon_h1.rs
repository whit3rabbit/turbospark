//! The falcon_h1 hybrid backbone: every decoder layer carries a Mamba2
//! mixer, GQA attention, and an MLP on two shared pre-norms.
//!
//! Port of `mlx_audio/lm/models/falcon_h1.py` (vendored from mlx-lm
//! 0.31.3) at the pinned mlx-audio commit. Layer math:
//! `h = h + mamba(ln(h)) + attn(ln(h))` then `h = h + mlp(ln(h))`, a
//! final RMSNorm, and a tied head scaled by
//! `lm_head_multiplier / embedding_multiplier` (the sanitize fold puts
//! `embedding_multiplier` into the table, so the runtime ratio undoes it
//! before applying the head multiplier) or an untied `lm_head`.
//!
//! `Model.sanitize` is ported with its own shape gate: the layer-0 conv1d
//! weight arrives either as the torch layout `[out, 1, kernel]` or the
//! mlx layout `[out, kernel, 1]`, and `shape[-1] > shape[1]` (the torch
//! case, since the kernel is 4) both decides the flat tap order question
//! and marks the checkpoint as unsanitized, folding the muP multipliers
//! (`embedding`, `lm_head`, `attention_in`, `attention_out`, `mlp`
//! multipliers, `ssm_in` times the [`compute_mup_vector`] row vector,
//! `ssm_out`) into the named projections exactly once. An mlx-layout save
//! is left untouched, so pre-converted checkpoints and the fixture saves
//! load through the same path. With depthwise groups the two layouts
//! flatten to the same `(channel, tap)` order, so only the fold gate
//! reads the shape. The reference `key_proj` fold branch names no
//! falcon_h1 tensor and is not carried.
//!
//! [`FalconH1Model::forward_stack`] is the arktts seam: it runs the layer
//! stack plus the final norm on caller-computed embeddings (the reference
//! `FalconH1Model` has no `inputs_embeds` path, and arktts must apply its
//! own embedding multiplier to the text-plus-codebook composite before
//! entering the stack).

use serde_json::Value;

use crate::nn::{Linear, RmsNorm};
use crate::ops;
use crate::{Result, SpeechError};

use super::cache::KvCache;
use super::engine::{lookup_embeddings, Attention, Input, Mlp};
use super::rope::Rope;
use super::ssm;
use super::{Bias, Weights};

/// The parsed falcon_h1 configuration (`ModelArgs`); every field carries
/// the reference dataclass default. `initializer_range`,
/// `mlp_expansion_factor`, `mamba_chunk_size`, `mamba_expand`, and
/// `mamba_use_mlp` are accepted from the config but never read by the
/// vendored model math, so they are not modeled here.
#[derive(Debug, Clone)]
pub struct FalconConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub attention_bias: bool,
    pub mlp_bias: bool,
    pub projectors_bias: bool,
    pub mamba_proj_bias: bool,
    pub mamba_conv_bias: bool,
    pub mamba_d_conv: usize,
    pub mamba_d_head: usize,
    pub mamba_d_ssm: usize,
    pub mamba_d_state: usize,
    pub mamba_n_groups: usize,
    pub mamba_n_heads: usize,
    pub mamba_norm_before_gate: bool,
    pub mamba_rms_norm: bool,
    pub mlp_multipliers: [f64; 2],
    pub ssm_multipliers: Vec<f64>,
    pub attention_in_multiplier: f64,
    pub attention_out_multiplier: f64,
    pub key_multiplier: f64,
    pub embedding_multiplier: f64,
    pub lm_head_multiplier: f64,
    pub ssm_in_multiplier: f64,
    pub ssm_out_multiplier: f64,
    pub rope_theta: f32,
    pub rope_traditional: bool,
    pub rope_scaling: Option<Value>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
}

impl FalconConfig {
    /// Width derived from the reference constructor:
    /// `intermediate_size + 2 * n_groups * d_state`.
    pub fn conv_dim(&self) -> usize {
        self.mamba_d_ssm + 2 * self.mamba_n_groups * self.mamba_d_state
    }

    /// The `in_proj` output width (also the muP row vector width).
    pub fn projection_size(&self) -> usize {
        self.mamba_d_ssm + self.conv_dim() + self.mamba_n_heads
    }

    /// The arktts `FalconSlowStack` construction gate: every multiplier
    /// the reference folds in `sanitize` must sit at unity, because that
    /// family cannot fold them (its embedding is a composite and a fold
    /// would scale only the text half). `embedding_multiplier` is
    /// deliberately absent: arktts applies it at runtime.
    pub fn ensure_unity_multipliers(&self) -> Result<()> {
        let mut offenders: Vec<String> = Vec::new();
        let scalars = [
            ("attention_in_multiplier", self.attention_in_multiplier),
            ("attention_out_multiplier", self.attention_out_multiplier),
            ("key_multiplier", self.key_multiplier),
            ("ssm_in_multiplier", self.ssm_in_multiplier),
            ("ssm_out_multiplier", self.ssm_out_multiplier),
        ];
        for (name, value) in scalars {
            if value != 1.0 {
                offenders.push(name.to_owned());
            }
        }
        if self.mlp_multipliers.iter().any(|&m| m != 1.0) {
            offenders.push("mlp_multipliers".to_owned());
        }
        if self.ssm_multipliers.iter().any(|&m| m != 1.0) {
            offenders.push("ssm_multipliers".to_owned());
        }
        if offenders.is_empty() {
            Ok(())
        } else {
            Err(SpeechError::Unsupported {
                why: format!(
                    "falcon_h1 consumers that apply their own embedding multiplier require \
                     unity muP multipliers, but {} {} away from 1.0",
                    offenders.join(", "),
                    if offenders.len() == 1 { "is" } else { "are" }
                ),
            })
        }
    }
}

/// Parses a falcon_h1 `config.json`; unknown keys (a checkpoint carries
/// the full HF dict) are ignored and every modeled field keeps the
/// reference dataclass default.
pub fn from_json(value: &Value) -> Result<FalconConfig> {
    let u = |key: &str, default: u64| -> usize {
        value.get(key).and_then(Value::as_u64).unwrap_or(default) as usize
    };
    let f = |key: &str, default: f64| -> f64 {
        value.get(key).and_then(Value::as_f64).unwrap_or(default)
    };
    let b = |key: &str, default: bool| -> bool {
        value.get(key).and_then(Value::as_bool).unwrap_or(default)
    };
    let vector = |key: &str, default: &[f64]| -> Vec<f64> {
        match value.get(key).and_then(Value::as_array) {
            Some(list) if !list.is_empty() => list
                .iter()
                .map(|entry| entry.as_f64().unwrap_or(1.0))
                .collect(),
            _ => default.to_vec(),
        }
    };
    let mlp_default = vector("mlp_multipliers", &[0.8838834764831844, 0.5859375]);
    Ok(FalconConfig {
        model_type: value
            .get("model_type")
            .and_then(Value::as_str)
            .unwrap_or("falcon_h1")
            .to_owned(),
        hidden_size: u("hidden_size", 1024),
        num_hidden_layers: u("num_hidden_layers", 36),
        intermediate_size: u("intermediate_size", 2048),
        num_attention_heads: u("num_attention_heads", 8),
        num_key_value_heads: u("num_key_value_heads", 2),
        head_dim: u("head_dim", 64),
        vocab_size: u("vocab_size", 32784),
        rms_norm_eps: f("rms_norm_eps", 1e-5) as f32,
        attention_bias: b("attention_bias", false),
        mlp_bias: b("mlp_bias", false),
        projectors_bias: b("projectors_bias", false),
        mamba_proj_bias: b("mamba_proj_bias", false),
        mamba_conv_bias: b("mamba_conv_bias", true),
        mamba_d_conv: u("mamba_d_conv", 4),
        mamba_d_head: u("mamba_d_head", 64),
        mamba_d_ssm: u("mamba_d_ssm", 1536),
        mamba_d_state: u("mamba_d_state", 128),
        mamba_n_groups: u("mamba_n_groups", 1),
        mamba_n_heads: u("mamba_n_heads", 24),
        mamba_norm_before_gate: b("mamba_norm_before_gate", false),
        mamba_rms_norm: b("mamba_rms_norm", false),
        mlp_multipliers: [mlp_default[0], mlp_default[1]],
        ssm_multipliers: vector(
            "ssm_multipliers",
            &[
                0.3535533905932738,
                0.25,
                0.3535533905932738,
                0.5,
                0.3535533905932738,
            ],
        ),
        attention_in_multiplier: f("attention_in_multiplier", 1.0),
        attention_out_multiplier: f("attention_out_multiplier", 0.9375),
        key_multiplier: f("key_multiplier", 0.390625),
        embedding_multiplier: f("embedding_multiplier", 5.656854249492381),
        lm_head_multiplier: f("lm_head_multiplier", 0.0390625),
        ssm_in_multiplier: f("ssm_in_multiplier", 1.25),
        ssm_out_multiplier: f("ssm_out_multiplier", 0.23570226039551587),
        rope_theta: f("rope_theta", 100000000000.0) as f32,
        rope_traditional: b("rope_traditional", false),
        rope_scaling: value.get("rope_scaling").cloned().filter(|v| !v.is_null()),
        max_position_embeddings: u("max_position_embeddings", 131072),
        tie_word_embeddings: b("tie_word_embeddings", true),
    })
}

/// Loads a falcon_h1 backbone from `weights` plus the raw config dict
/// (the free-function form the sibling backbones expose).
pub fn load(weights: &Weights, value: &Value) -> Result<FalconH1Model> {
    FalconH1Model::load(weights, value)
}

/// Port of `compute_mup_vector`: the `ssm_multipliers` broadcast over the
/// `in_proj` output segments
/// `[d_ssm, d_ssm, groups * d_state, groups * d_state, heads]`. The
/// arktts family consumes this through the shared config.
pub fn compute_mup_vector(config: &FalconConfig) -> Vec<f32> {
    let groups_time_state_size = config.mamba_n_groups * config.mamba_d_state;
    let sizes = [
        config.mamba_d_ssm,
        config.mamba_d_ssm,
        groups_time_state_size,
        groups_time_state_size,
        config.mamba_n_heads,
    ];
    let mut out = Vec::with_capacity(config.projection_size());
    for (&size, &multiplier) in sizes.iter().zip(&config.ssm_multipliers) {
        out.extend(std::iter::repeat_n(multiplier as f32, size));
    }
    out
}

/// The layer cache: the Mamba mixer's conv state
/// `[kernel - 1, conv_dim]` and selective-scan state
/// `[heads, head_dim, d_state]` (both `None` until the first call), plus
/// the attention half's KV cache.
#[derive(Debug, Clone, Default)]
pub struct FalconLayerCache {
    pub conv: Option<Vec<f32>>,
    pub ssm: Option<Vec<f32>>,
    pub kv: KvCache,
}

impl FalconLayerCache {
    /// Clears every buffer.
    pub fn reset(&mut self) {
        self.conv = None;
        self.ssm = None;
        self.kv.reset();
    }
}

/// `FalconH1RMSNormGated`: RMSNorm around the SwiGLU gate, in either
/// order (`norm_before_gate`). The `n_groups` constructor argument takes
/// no part in the math.
struct GatedNorm {
    weight: Vec<f32>,
    eps: f32,
    norm_before_gate: bool,
}

impl GatedNorm {
    fn forward(&self, y: &mut Vec<f32>, gate: &[f32], len: usize) {
        if !self.norm_before_gate {
            swiglu_in_place(y, gate);
        }
        let mut normalized = std::mem::take(y);
        RmsNorm::new(self.weight.clone(), self.eps).apply(&mut normalized, len);
        *y = normalized;
        if self.norm_before_gate {
            swiglu_in_place(y, gate);
        }
    }
}

/// `swiglu(gate, x)`: `silu(gate) * x` in place over `x`.
fn swiglu_in_place(x: &mut [f32], gate: &[f32]) {
    for (value, &g) in x.iter_mut().zip(gate) {
        *value *= g * (1.0 / (1.0 + (-g).exp()));
    }
}

/// One `mx.split` over consecutive axis -1 ranges: `bounds` are column
/// cut points, and each returned segment keeps the row-major
/// `[len, segment_width]` layout.
fn split_rows(buffer: &[f32], len: usize, bounds: &[usize]) -> Vec<Vec<f32>> {
    let width = buffer.len() / len;
    let mut starts = Vec::with_capacity(bounds.len() + 2);
    starts.push(0);
    starts.extend_from_slice(bounds);
    starts.push(width);
    (0..starts.len() - 1)
        .map(|index| {
            let (start, end) = (starts[index], starts[index + 1]);
            let mut out = Vec::with_capacity(len * (end - start));
            for row in buffer.chunks_exact(width) {
                out.extend_from_slice(&row[start..end]);
            }
            out
        })
        .collect()
}

struct Mixer {
    in_proj: Linear,
    /// Depthwise conv taps `[conv_dim, kernel]`.
    conv_weight: Vec<f32>,
    conv_bias: Option<Vec<f32>>,
    out_proj: Linear,
    dt_bias: Vec<f32>,
    a_log: Vec<f32>,
    d: Vec<f32>,
    norm: Option<GatedNorm>,
    d_ssm: usize,
    d_state: usize,
    n_groups: usize,
    head_dim: usize,
    kernel: usize,
    conv_dim: usize,
    time_step_limit: (f32, f32),
}

impl Mixer {
    /// Port of `FalconH1Mixer.__call__` minus the cache advance (a no-op
    /// in batch 1: `ArraysCache.advance` decrements `lengths`, which is
    /// `None` here).
    fn forward(
        &mut self,
        x: &[f32],
        len: usize,
        mut cache: Option<&mut FalconLayerCache>,
    ) -> Vec<f32> {
        let projected = self.in_proj.forward(x, len);
        let outer_bounds = [self.d_ssm, self.d_ssm + self.conv_dim];
        let mut outer = split_rows(&projected, len, &outer_bounds);
        let dt = outer.pop().expect("dt segment");
        let conv_input = outer.pop().expect("conv segment");
        let gate = outer.pop().expect("gate segment");

        // _conv: left-pad with the running state (zeros on the first
        // call) or with zeros when cacheless, keep the last kernel - 1
        // rows as the new state, run the depthwise kernel, and silu.
        let kernel = self.kernel;
        let conv_dim = self.conv_dim;
        let mut padded = vec![0.0f32; (len + kernel - 1) * conv_dim];
        if let Some(cache) = cache.as_deref_mut() {
            let state = cache
                .conv
                .get_or_insert_with(|| vec![0.0; (kernel - 1) * conv_dim]);
            let kept = state.len();
            padded[..kept].copy_from_slice(state);
            padded[kept..].copy_from_slice(&conv_input);
            state.copy_from_slice(&padded[len * conv_dim..]);
        } else {
            padded[(kernel - 1) * conv_dim..].copy_from_slice(&conv_input);
        }
        let mut convolved = vec![0.0f32; len * conv_dim];
        for channel in 0..conv_dim {
            let taps = &self.conv_weight[channel * kernel..(channel + 1) * kernel];
            for t in 0..len {
                let mut acc = self.conv_bias.as_ref().map_or(0.0, |b| b[channel]);
                for (j, &tap) in taps.iter().enumerate() {
                    acc += tap * padded[(t + j) * conv_dim + channel];
                }
                convolved[t * conv_dim + channel] = acc;
            }
        }
        ops::silu(&mut convolved);

        let inner_bounds = [self.d_ssm, self.d_ssm + self.n_groups * self.d_state];
        let mut inner = split_rows(&convolved, len, &inner_bounds);
        let c_rows = inner.pop().expect("C segment");
        let b_rows = inner.pop().expect("B segment");
        let hidden_states = inner.pop().expect("x segment");
        let (mut y, state) = ssm::ssm_update(
            &hidden_states,
            &self.a_log,
            &b_rows,
            &c_rows,
            &self.d,
            &dt,
            &self.dt_bias,
            self.head_dim,
            self.d_state,
            cache.as_ref().and_then(|cache| cache.ssm.as_deref()),
            self.time_step_limit,
        );
        if let Some(cache) = cache {
            cache.ssm = Some(state);
        }

        match &self.norm {
            Some(norm) => norm.forward(&mut y, &gate, len),
            None => swiglu_in_place(&mut y, &gate),
        }
        self.out_proj.forward(&y, len)
    }
}

struct Layer {
    mamba: Mixer,
    attention: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    pre_ff_layernorm: RmsNorm,
}

impl Layer {
    fn forward(
        &mut self,
        h: &mut Vec<f32>,
        len: usize,
        offset: usize,
        cache: Option<&mut FalconLayerCache>,
    ) -> Result<()> {
        let mut normed = h.clone();
        self.input_layernorm.apply(&mut normed, len);
        let (mamba_h, attn_h) = match cache {
            Some(cache) => {
                let kv_offset = cache.kv.offset;
                let mamba_h = self.mamba.forward(&normed, len, Some(&mut *cache));
                let attn_h = self.attention.forward(
                    &normed,
                    len,
                    kv_offset,
                    Some(&mut cache.kv as &mut dyn super::cache::LayerKv),
                )?;
                (mamba_h, attn_h)
            }
            None => {
                let mamba_h = self.mamba.forward(&normed, len, None);
                let attn_h = self.attention.forward(&normed, len, offset, None)?;
                (mamba_h, attn_h)
            }
        };
        // residual + mamba_h + attn_h, in the reference addition order.
        for (index, value) in h.iter_mut().enumerate() {
            *value += mamba_h[index] + attn_h[index];
        }
        let mut fed = h.clone();
        self.pre_ff_layernorm.apply(&mut fed, len);
        let fed_forward = self.mlp.forward(&fed, len);
        for (index, value) in fed_forward.into_iter().enumerate() {
            h[index] += value;
        }
        Ok(())
    }
}

/// A loaded falcon_h1 backbone ready to run.
pub struct FalconH1Model {
    pub config: FalconConfig,
    embed_tokens: Vec<f32>,
    layers: Vec<Layer>,
    final_layernorm: RmsNorm,
    /// Untied head; a tied head reuses `embed_tokens` under the
    /// `lm_head_multiplier / embedding_multiplier` ratio.
    lm_head: Option<Linear>,
    head_scale: f32,
}

/// The row scale a folded linear applies to its `[out, in]` weight. A
/// `Scalar(1.0)` is the no-fold case.
enum RowScale<'a> {
    Scalar(f64),
    /// `scale * vector[row]`, the `in_proj` muP fold.
    PerRow {
        scale: f64,
        vector: &'a [f32],
    },
}

fn scaled_linear(
    weights: &Weights,
    base: &str,
    input: usize,
    output: usize,
    bias: Bias,
    scale: RowScale<'_>,
) -> Result<Linear> {
    let mut linear = weights.linear_absolute(base, input, output, bias)?;
    match scale {
        RowScale::Scalar(factor) => {
            if factor != 1.0 {
                for value in &mut linear.weight {
                    *value *= factor as f32;
                }
            }
        }
        RowScale::PerRow { scale, vector } => {
            for (row, &multiplier) in vector.iter().enumerate() {
                let factor = (scale * multiplier as f64) as f32;
                for value in &mut linear.weight[row * input..(row + 1) * input] {
                    *value *= factor;
                }
            }
        }
    }
    Ok(linear)
}

/// The reference `Model.sanitize` gate: the checkpoint is unsanitized
/// (torch conv layout, muP multipliers unfolded) exactly when the
/// layer-0 conv1d weight satisfies `shape[-1] > shape[1]`.
fn needs_sanitization(weights: &Weights, prefix: &str) -> Result<bool> {
    let name = format!("{prefix}layers.0.mamba.conv1d.weight");
    let shape = weights
        .shards()
        .iter()
        .find_map(|file| {
            file.descriptor(&name)
                .map(|descriptor| descriptor.shape.clone())
        })
        .ok_or_else(|| SpeechError::Tensor {
            name: name.clone(),
            why: "missing; falcon_h1 checkpoints always carry the mamba conv".into(),
        })?;
    if shape.len() != 3 {
        return Err(SpeechError::BadConfig {
            field: "mamba.conv1d.weight".into(),
            why: format!("expected a 3-d tensor, got shape {shape:?}"),
        });
    }
    Ok(shape[2] > shape[1])
}

/// Loads the depthwise conv weight as `[conv_dim, kernel]` taps. Both the
/// mlx `[conv_dim, kernel, 1]` and torch `[conv_dim, 1, kernel]` saves
/// flatten to the same `(channel, tap)` order.
fn load_conv_taps(
    weights: &Weights,
    base: &str,
    conv_dim: usize,
    kernel: usize,
) -> Result<Vec<f32>> {
    let name = format!("{base}.weight");
    let shape = weights
        .shards()
        .iter()
        .find_map(|file| {
            file.descriptor(&name)
                .map(|descriptor| descriptor.shape.clone())
        })
        .ok_or_else(|| SpeechError::Tensor {
            name: name.clone(),
            why: "tensor is missing".into(),
        })?;
    match shape.as_slice() {
        [out, middle, last]
            if *out == conv_dim
                && (*middle == kernel && *last == 1 || *middle == 1 && *last == kernel) =>
        {
            weights.tensor_absolute(&name, shape.as_slice())
        }
        other => Err(SpeechError::Tensor {
            name: name.clone(),
            why: format!(
                "expected [{conv_dim}, {kernel}, 1] or [{conv_dim}, 1, {kernel}], got {other:?}"
            ),
        }),
    }
}

impl FalconH1Model {
    /// Loads a falcon_h1 backbone from `weights` plus the raw config. The
    /// tensor prefix is autodetected (`model.` for a full-model save,
    /// empty otherwise).
    pub fn load(weights: &Weights, value: &Value) -> Result<Self> {
        Self::from_config(weights, from_json(value)?)
    }

    /// Loads from an already parsed config.
    pub fn from_config(weights: &Weights, config: FalconConfig) -> Result<Self> {
        let prefix =
            if weights.contains_absolute(&format!("model.layers.0.self_attn.q_proj.weight")) {
                "model."
            } else {
                ""
            };
        Self::load_impl(weights, config, prefix, Some("embed_tokens"))
    }

    /// Loads the layers and final norm with no embedding table, for the
    /// arktts slow stack: that family owns the text table (the composite
    /// embedding sums codebook rows into it) and drives the stack through
    /// [`FalconH1Model::forward_stack`]. `prefix` covers the layer
    /// tensors (`model.slow.` for arktts).
    pub fn load_without_embed_table(
        weights: &Weights,
        config: FalconConfig,
        prefix: &str,
    ) -> Result<Self> {
        Self::load_impl(weights, config, prefix, None)
    }

    fn load_impl(
        weights: &Weights,
        config: FalconConfig,
        prefix: &str,
        embed_table: Option<&str>,
    ) -> Result<Self> {
        if config.ssm_multipliers.len() != 5 {
            return Err(SpeechError::BadConfig {
                field: "ssm_multipliers".into(),
                why: format!("must have 5 entries, got {}", config.ssm_multipliers.len()),
            });
        }
        if config.mamba_d_ssm != config.mamba_n_heads * config.mamba_d_head {
            return Err(SpeechError::BadConfig {
                field: "mamba_d_ssm".into(),
                why: format!(
                    "must equal mamba_n_heads * mamba_d_head ({} != {} * {})",
                    config.mamba_d_ssm, config.mamba_n_heads, config.mamba_d_head
                ),
            });
        }
        if config.mamba_n_groups == 0 || config.mamba_n_heads % config.mamba_n_groups != 0 {
            return Err(SpeechError::BadConfig {
                field: "mamba_n_groups".into(),
                why: format!(
                    "must be non-zero and divide mamba_n_heads {} (got {})",
                    config.mamba_n_heads, config.mamba_n_groups
                ),
            });
        }
        if config.num_key_value_heads == 0
            || config.num_attention_heads % config.num_key_value_heads != 0
        {
            return Err(SpeechError::BadConfig {
                field: "num_key_value_heads".into(),
                why: format!(
                    "must be non-zero and divide num_attention_heads {} (got {})",
                    config.num_attention_heads, config.num_key_value_heads
                ),
            });
        }
        let sanitize = needs_sanitization(weights, prefix)?;
        let mup = compute_mup_vector(&config);
        if mup.len() != config.projection_size() {
            return Err(SpeechError::BadConfig {
                field: "ssm_multipliers".into(),
                why: format!(
                    "broadcasts to {} entries but in_proj needs {}",
                    mup.len(),
                    config.projection_size()
                ),
            });
        }
        // The sanitize suffix table; an unsanitized checkpoint folds,
        // a converted one already carries the folded values.
        let fold = |suffix: &str| -> f64 {
            if !sanitize {
                return 1.0;
            }
            if suffix.ends_with("embed_tokens.weight") {
                config.embedding_multiplier
            } else if suffix.ends_with("lm_head.weight") {
                config.lm_head_multiplier
            } else if suffix.ends_with("q_proj.weight") || suffix.ends_with("k_proj.weight") {
                config.attention_in_multiplier
            } else if suffix.ends_with("o_proj.weight") {
                config.attention_out_multiplier
            } else if suffix.ends_with("out_proj.weight") {
                config.ssm_out_multiplier
            } else if suffix.ends_with("gate_proj.weight") {
                config.mlp_multipliers[0]
            } else if suffix.ends_with("down_proj.weight") {
                config.mlp_multipliers[1]
            } else {
                1.0
            }
        };
        let hidden = config.hidden_size;
        let q_width = config.num_attention_heads * config.head_dim;
        let kv_width = config.num_key_value_heads * config.head_dim;
        let attention_bias = if config.attention_bias {
            Bias::Required
        } else {
            Bias::None
        };
        let mlp_bias = if config.mlp_bias {
            Bias::Required
        } else {
            Bias::None
        };
        let rope = Rope::initialize(
            config.head_dim,
            config.rope_theta,
            config.rope_traditional,
            config.rope_scaling.as_ref(),
            Some(config.max_position_embeddings),
        )?;
        let conv_dim = config.conv_dim();
        let kernel = config.mamba_d_conv;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let layer = format!("{prefix}layers.{index}.");
            let attention = Attention {
                q_proj: scaled_linear(
                    weights,
                    &format!("{layer}self_attn.q_proj"),
                    hidden,
                    q_width,
                    attention_bias,
                    RowScale::Scalar(fold("q_proj.weight")),
                )?,
                k_proj: scaled_linear(
                    weights,
                    &format!("{layer}self_attn.k_proj"),
                    hidden,
                    kv_width,
                    attention_bias,
                    RowScale::Scalar(fold("k_proj.weight")),
                )?,
                v_proj: weights.linear_absolute(
                    &format!("{layer}self_attn.v_proj"),
                    hidden,
                    kv_width,
                    attention_bias,
                )?,
                o_proj: scaled_linear(
                    weights,
                    &format!("{layer}self_attn.o_proj"),
                    q_width,
                    hidden,
                    attention_bias,
                    RowScale::Scalar(fold("o_proj.weight")),
                )?,
                q_norm: None,
                k_norm: None,
                n_heads: config.num_attention_heads,
                n_kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                scale: (config.head_dim as f64).powf(-0.5) as f32,
                rope: Some(rope.clone()),
                window: None,
                bidirectional: false,
                exec: None,
            };
            let mlp = Mlp {
                gate_proj: scaled_linear(
                    weights,
                    &format!("{layer}feed_forward.gate_proj"),
                    hidden,
                    config.intermediate_size,
                    mlp_bias,
                    RowScale::Scalar(fold("gate_proj.weight")),
                )?,
                up_proj: weights.linear_absolute(
                    &format!("{layer}feed_forward.up_proj"),
                    hidden,
                    config.intermediate_size,
                    mlp_bias,
                )?,
                down_proj: scaled_linear(
                    weights,
                    &format!("{layer}feed_forward.down_proj"),
                    config.intermediate_size,
                    hidden,
                    mlp_bias,
                    RowScale::Scalar(fold("down_proj.weight")),
                )?,
                exec: None,
            };
            let in_proj = scaled_linear(
                weights,
                &format!("{layer}mamba.in_proj"),
                hidden,
                config.projection_size(),
                if config.mamba_proj_bias {
                    Bias::Required
                } else {
                    Bias::None
                },
                RowScale::PerRow {
                    scale: config.ssm_in_multiplier,
                    vector: &mup,
                },
            )?;
            let norm = if config.mamba_rms_norm {
                Some(GatedNorm {
                    weight: weights.tensor_absolute(
                        &format!("{layer}mamba.norm.weight"),
                        &[config.mamba_d_ssm],
                    )?,
                    eps: config.rms_norm_eps,
                    norm_before_gate: config.mamba_norm_before_gate,
                })
            } else {
                None
            };
            layers.push(Layer {
                mamba: Mixer {
                    dt_bias: weights.tensor_absolute(
                        &format!("{layer}mamba.dt_bias"),
                        &[config.mamba_n_heads],
                    )?,
                    a_log: weights
                        .tensor_absolute(&format!("{layer}mamba.A_log"), &[config.mamba_n_heads])?,
                    d: weights
                        .tensor_absolute(&format!("{layer}mamba.D"), &[config.mamba_n_heads])?,
                    conv_weight: load_conv_taps(
                        weights,
                        &format!("{layer}mamba.conv1d"),
                        conv_dim,
                        kernel,
                    )?,
                    conv_bias: if config.mamba_conv_bias {
                        Some(
                            weights.tensor_absolute(
                                &format!("{layer}mamba.conv1d.bias"),
                                &[conv_dim],
                            )?,
                        )
                    } else {
                        None
                    },
                    out_proj: scaled_linear(
                        weights,
                        &format!("{layer}mamba.out_proj"),
                        config.mamba_d_ssm,
                        hidden,
                        if config.projectors_bias {
                            Bias::Required
                        } else {
                            Bias::None
                        },
                        RowScale::Scalar(fold("out_proj.weight")),
                    )?,
                    norm,
                    in_proj,
                    d_ssm: config.mamba_d_ssm,
                    d_state: config.mamba_d_state,
                    n_groups: config.mamba_n_groups,
                    head_dim: config.mamba_d_head,
                    kernel,
                    conv_dim,
                    time_step_limit: (0.0, f32::INFINITY),
                },
                attention,
                mlp,
                input_layernorm: RmsNorm::new(
                    weights
                        .tensor_absolute(&format!("{layer}input_layernorm.weight"), &[hidden])?,
                    config.rms_norm_eps,
                ),
                pre_ff_layernorm: RmsNorm::new(
                    weights
                        .tensor_absolute(&format!("{layer}pre_ff_layernorm.weight"), &[hidden])?,
                    config.rms_norm_eps,
                ),
            });
        }
        let embed_tokens = match embed_table {
            Some(table) => {
                let mut values = weights.tensor_absolute(
                    &format!("{prefix}{table}.weight"),
                    &[config.vocab_size, hidden],
                )?;
                let factor = fold("embed_tokens.weight");
                if factor != 1.0 {
                    for value in &mut values {
                        *value *= factor as f32;
                    }
                }
                values
            }
            None => Vec::new(),
        };
        let final_layernorm = RmsNorm::new(
            weights.tensor_absolute(&format!("{prefix}final_layernorm.weight"), &[hidden])?,
            config.rms_norm_eps,
        );
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            // The head sits outside `model.` in full-model saves.
            let base = if weights.contains_absolute(&format!("{prefix}lm_head.weight")) {
                format!("{prefix}lm_head")
            } else {
                "lm_head".to_owned()
            };
            Some(scaled_linear(
                weights,
                &base,
                hidden,
                config.vocab_size,
                Bias::None,
                RowScale::Scalar(fold("lm_head.weight")),
            )?)
        };
        Ok(Self {
            head_scale: (config.lm_head_multiplier / config.embedding_multiplier) as f32,
            config,
            embed_tokens,
            layers,
            final_layernorm,
            lm_head,
        })
    }

    /// `make_cache`: one `[conv, ssm] + KV` pair per layer.
    pub fn make_cache(&self) -> Vec<FalconLayerCache> {
        (0..self.config.num_hidden_layers)
            .map(|_| FalconLayerCache::default())
            .collect()
    }

    /// The embedding lookup for [`Input::Tokens`] (no multiplier applied;
    /// the reference folds it at load) or the caller's embeddings.
    pub fn embed(&self, input: Input<'_>) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        match input {
            Input::Tokens(tokens) => {
                lookup_embeddings(&self.embed_tokens, self.config.vocab_size, hidden, tokens)
            }
            Input::Embeddings(values) => {
                Input::Embeddings(values).len(hidden)?;
                Ok(values.to_vec())
            }
        }
    }

    /// Runs the layer stack plus the final norm over caller-computed
    /// embeddings `[positions, hidden_size]`. The arktts slow stack
    /// multiplies its composite embedding by `embedding_multiplier`
    /// before calling this; [`FalconH1Model::forward_hidden`] passes the
    /// plain table lookup instead. `caches` may be `None` for a cacheless
    /// call; when present it must hold one cache per layer and advances
    /// with every call.
    pub fn forward_stack(
        &mut self,
        hidden: &[f32],
        mut caches: Option<&mut Vec<FalconLayerCache>>,
    ) -> Result<Vec<f32>> {
        let len = hidden.len() / self.config.hidden_size;
        if len == 0 {
            return Err(SpeechError::Input {
                why: "empty forward input".into(),
            });
        }
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
        let mut h = hidden.to_vec();
        for (index, layer) in self.layers.iter_mut().enumerate() {
            let offset = caches
                .as_deref()
                .and_then(|caches| caches.get(index))
                .map(|cache| cache.kv.offset)
                .unwrap_or(0);
            layer.forward(
                &mut h,
                len,
                offset,
                caches
                    .as_deref_mut()
                    .and_then(|caches| caches.get_mut(index)),
            )?;
        }
        self.final_layernorm.apply(&mut h, len);
        Ok(h)
    }

    /// Runs the backbone up to the final norm and returns the hidden
    /// state `[positions, hidden_size]`.
    pub fn forward_hidden(
        &mut self,
        input: Input<'_>,
        caches: Option<&mut Vec<FalconLayerCache>>,
    ) -> Result<Vec<f32>> {
        let embedded = self.embed(input)?;
        self.forward_stack(&embedded, caches)
    }

    /// The tied or untied head over post-norm hidden states.
    fn apply_head(&self, projected: &[f32], len: usize) -> Vec<f32> {
        let hidden = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        match &self.lm_head {
            Some(head) => head.forward(projected, len),
            None => {
                let mut logits =
                    ops::linear(projected, &self.embed_tokens, None, len, hidden, vocab);
                for value in &mut logits {
                    *value *= self.head_scale;
                }
                logits
            }
        }
    }

    /// Runs the backbone and returns logits `[positions, vocab_size]`.
    pub fn forward(
        &mut self,
        input: Input<'_>,
        caches: Option<&mut Vec<FalconLayerCache>>,
    ) -> Result<Vec<f32>> {
        let hidden = self.forward_hidden(input, caches)?;
        let len = hidden.len() / self.config.hidden_size;
        Ok(self.apply_head(&hidden, len))
    }

    /// Runs the backbone once and returns both the post-norm hidden
    /// states and the logits taken from them (the audio families read
    /// hidden states and logits of the same pass).
    pub fn forward_hidden_and_logits(
        &mut self,
        input: Input<'_>,
        caches: Option<&mut Vec<FalconLayerCache>>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let hidden = self.forward_hidden(input, caches)?;
        let len = hidden.len() / self.config.hidden_size;
        let logits = self.apply_head(&hidden, len);
        Ok((hidden, logits))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `compute_mup_vector` broadcasts each multiplier over its segment.
    #[test]
    fn mup_vector_segments_match_the_reference_sizes() {
        let config = FalconConfig {
            mamba_d_ssm: 16,
            mamba_d_state: 8,
            mamba_n_groups: 2,
            mamba_n_heads: 4,
            ssm_multipliers: vec![0.5, 1.0, 2.0, 4.0, 8.0],
            ..from_json(&serde_json::json!({})).unwrap()
        };
        let mup = compute_mup_vector(&config);
        assert_eq!(mup.len(), 16 + 16 + 16 + 16 + 4);
        assert!(mup[..16].iter().all(|&v| v == 0.5));
        assert!(mup[16..32].iter().all(|&v| v == 1.0));
        assert!(mup[32..48].iter().all(|&v| v == 2.0));
        assert!(mup[48..64].iter().all(|&v| v == 4.0));
        assert!(mup[64..].iter().all(|&v| v == 8.0));
    }

    /// The arktts unity gate names the offending fields and ignores
    /// `embedding_multiplier` (applied at runtime there).
    #[test]
    fn unity_gate_rejects_folded_multipliers() {
        let mut config = from_json(&serde_json::json!({})).unwrap();
        config.ssm_in_multiplier = 1.25;
        config.mlp_multipliers = [1.0, 2.0];
        let error = config.ensure_unity_multipliers().unwrap_err();
        let message = error.to_string();
        assert!(message.contains("ssm_in_multiplier"), "{message}");
        assert!(message.contains("mlp_multipliers"), "{message}");
        config.ssm_in_multiplier = 1.0;
        config.mlp_multipliers = [1.0, 1.0];
        config.attention_out_multiplier = 1.0;
        config.key_multiplier = 1.0;
        config.ssm_out_multiplier = 1.0;
        config.ssm_multipliers = vec![1.0; 5];
        config.embedding_multiplier = 5.0;
        config.ensure_unity_multipliers().unwrap();
    }

    /// The split helper cuts axis -1 columns while keeping rows.
    #[test]
    fn split_rows_cuts_contiguous_ranges() {
        // 3 rows of width 4.
        let buffer: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let parts = split_rows(&buffer, 3, &[1, 3]);
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0], &[0.0, 4.0, 8.0]);
        assert_eq!(parts[1], &[1.0, 2.0, 5.0, 6.0, 9.0, 10.0]);
        assert_eq!(parts[2], &[3.0, 7.0, 11.0]);
    }

    /// The mixer's group width follows the `in_proj` segments:
    /// `[d_ssm, conv_dim, heads]` with the conv split into
    /// `[d_ssm, groups * d_state]`.
    #[test]
    fn mixer_widths_match_the_reference_segments() {
        let config = from_json(&serde_json::json!({
            "hidden_size": 32,
            "mamba_d_ssm": 16,
            "mamba_d_state": 8,
            "mamba_n_groups": 2,
            "mamba_n_heads": 4,
            "mamba_d_head": 4,
            "mamba_d_conv": 4,
        }))
        .unwrap();
        assert_eq!(config.conv_dim(), 16 + 2 * 2 * 8);
        assert_eq!(config.projection_size(), 16 + 48 + 4);
    }
}
