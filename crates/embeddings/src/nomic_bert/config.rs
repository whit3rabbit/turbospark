//! `config.json` for the Nomic BERT encoder (`model_type = "nomic_bert"`).
//!
//! `NomicBertConfig` extends `GPT2Config`
//! (configuration_hf_nomic_bert.py:4-56), so the widths are `n_embd`,
//! `n_head`, `n_inner`, `n_layer`, `n_positions`. A field that is absent
//! takes the pinned class default; a value this port does not implement
//! is an error. Flags the pinned modeling code stores but never reads
//! (`use_flash_attn`, `fused_bias_fc`, `fused_dropout_add_ln`,
//! `dense_seq_output`, `use_xentropy`) are accepted with any value.
//! Flags it ignores but that name a different architecture (`causal`,
//! `parallel_block`, `use_rms_norm`) are refused when set, because a
//! checkpoint trained with them is not what this code computes.

use serde_json::Value;

use crate::error::{bad_config, EmbeddingError};

/// The gated MLP rounds its hidden width up to this multiple
/// (`NomciBertGatedMLP`, modeling_hf_nomic_bert.py:455, 463-464).
pub const MLP_MULTIPLE_OF: usize = 256;

/// The encoder hyperparameters this port consumes.
#[derive(Debug, Clone, PartialEq)]
pub struct NomicBertConfig {
    /// Vocabulary as written; the embedding table has
    /// [`padded_vocab_size`](Self::padded_vocab_size) rows.
    pub vocab_size: usize,
    pub pad_vocab_size_multiple: usize,
    pub n_embd: usize,
    pub n_head: usize,
    /// `n_inner` as written; see [`mlp_hidden`](Self::mlp_hidden).
    pub n_inner: usize,
    pub n_layer: usize,
    /// Position budget (`max_position_embeddings`). Rotary models have no
    /// position table; this caps the accepted length.
    pub n_positions: usize,
    pub type_vocab_size: usize,
    pub layer_norm_epsilon: f32,
    pub rotary_emb_base: f64,
    /// Dynamic NTK factor; `None` when the config disables scaling
    /// (`rotary_scaling_factor` null or 0).
    pub rotary_scaling_factor: Option<f64>,
    pub max_trained_positions: usize,
}

fn usize_field(v: &Value, key: &str, default: Option<usize>) -> Result<usize, EmbeddingError> {
    match v.get(key) {
        None => default.ok_or_else(|| bad_config(key, "missing")),
        Some(x) => {
            let n = x
                .as_u64()
                .ok_or_else(|| bad_config(key, "not a non-negative integer"))?;
            usize::try_from(n).map_err(|_| bad_config(key, "does not fit usize"))
        }
    }
}

fn f64_field(v: &Value, key: &str, default: f64) -> Result<f64, EmbeddingError> {
    let x = match v.get(key) {
        None => default,
        Some(x) => x.as_f64().ok_or_else(|| bad_config(key, "not a number"))?,
    };
    if !x.is_finite() {
        return Err(bad_config(key, "not finite"));
    }
    Ok(x)
}

fn bool_field(v: &Value, key: &str, default: bool) -> Result<bool, EmbeddingError> {
    match v.get(key) {
        None => Ok(default),
        Some(x) => x.as_bool().ok_or_else(|| bad_config(key, "not a boolean")),
    }
}

fn unsupported(why: String) -> EmbeddingError {
    EmbeddingError::Unsupported { why }
}

impl NomicBertConfig {
    pub fn from_json(v: &Value) -> Result<Self, EmbeddingError> {
        match v.get("model_type").and_then(Value::as_str) {
            Some("nomic_bert") => {}
            other => {
                return Err(bad_config(
                    "model_type",
                    format!("expected nomic_bert, got {other:?}"),
                ))
            }
        }
        // Only the SwiGLU block is implemented: fc11 * silu(fc12)
        // (modeling:818-831, 472-478).
        match v.get("activation_function").and_then(Value::as_str) {
            Some("swiglu") => {}
            other => {
                return Err(unsupported(format!(
                    "activation_function {other:?}; only swiglu is implemented"
                )))
            }
        }
        // rotary_emb_fraction 0 means absolute position embeddings
        // (modeling:385-391); a fraction below 1 rotates only a prefix.
        let fraction = f64_field(v, "rotary_emb_fraction", 0.0)?;
        if fraction != 1.0 {
            return Err(unsupported(format!(
                "rotary_emb_fraction {fraction}; only 1.0 is implemented"
            )));
        }
        if bool_field(v, "rotary_emb_interleaved", false)? {
            return Err(unsupported(
                "rotary_emb_interleaved = true (GPT-J pairs) is not implemented".into(),
            ));
        }
        // xPos scaling (modeling:544-548, 682-691).
        if !v.get("rotary_emb_scale_base").is_none_or(Value::is_null) {
            return Err(unsupported(
                "rotary_emb_scale_base is set; only null is implemented".into(),
            ));
        }
        // Pre-LN takes the other branch of NomicBertBlock.forward
        // (modeling:870-887).
        if bool_field(v, "prenorm", false)? {
            return Err(unsupported("prenorm = true is not implemented".into()));
        }
        for key in ["qkv_proj_bias", "mlp_fc1_bias", "mlp_fc2_bias"] {
            // Class defaults are true; the pinned checkpoint has none.
            if bool_field(v, key, true)? {
                return Err(unsupported(format!("{key} = true is not implemented")));
            }
        }
        for key in [
            "causal",
            "parallel_block",
            "parallel_block_tied_norm",
            "use_rms_norm",
        ] {
            if bool_field(v, key, false)? {
                return Err(unsupported(format!("{key} = true is not implemented")));
            }
        }
        if v.get("num_heads_kv").is_some_and(|x| !x.is_null()) {
            return Err(unsupported(
                "num_heads_kv (MQA/GQA) is not implemented".into(),
            ));
        }

        // `if config.rotary_scaling_factor:` picks the NTK class
        // (modeling:727); null and 0 select the plain rotary.
        let rotary_scaling_factor = match v.get("rotary_scaling_factor") {
            None => Some(1.0),
            Some(Value::Null) => None,
            Some(x) => {
                let f = x
                    .as_f64()
                    .ok_or_else(|| bad_config("rotary_scaling_factor", "not a number"))?;
                if !f.is_finite() || f < 0.0 {
                    return Err(bad_config("rotary_scaling_factor", "must be >= 0"));
                }
                (f != 0.0).then_some(f)
            }
        };
        let config = Self {
            vocab_size: usize_field(v, "vocab_size", None)?,
            pad_vocab_size_multiple: usize_field(v, "pad_vocab_size_multiple", Some(1))?,
            n_embd: usize_field(v, "n_embd", None)?,
            n_head: usize_field(v, "n_head", None)?,
            n_inner: usize_field(v, "n_inner", None)?,
            n_layer: usize_field(v, "n_layer", None)?,
            n_positions: usize_field(v, "n_positions", None)?,
            type_vocab_size: usize_field(v, "type_vocab_size", Some(2))?,
            layer_norm_epsilon: f64_field(v, "layer_norm_epsilon", 1e-5)? as f32,
            rotary_emb_base: f64_field(v, "rotary_emb_base", 1000.0)?,
            rotary_scaling_factor,
            max_trained_positions: usize_field(v, "max_trained_positions", Some(2048))?,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), EmbeddingError> {
        for (k, n) in [
            ("vocab_size", self.vocab_size),
            ("pad_vocab_size_multiple", self.pad_vocab_size_multiple),
            ("n_embd", self.n_embd),
            ("n_head", self.n_head),
            ("n_inner", self.n_inner),
            ("n_layer", self.n_layer),
            ("n_positions", self.n_positions),
            ("max_trained_positions", self.max_trained_positions),
        ] {
            if n == 0 {
                return Err(bad_config(k, "must be positive"));
            }
        }
        if self.type_vocab_size == 0 {
            return Err(unsupported(
                "type_vocab_size = 0 (no token-type embedding) is not implemented".into(),
            ));
        }
        if self.n_embd % self.n_head != 0 {
            return Err(bad_config(
                "n_head",
                format!(
                    "n_embd {} is not a multiple of {} heads",
                    self.n_embd, self.n_head
                ),
            ));
        }
        // The NTK exponent dim / (dim - 2) needs dim > 2.
        if self.head_dim() % 2 != 0 || self.head_dim() < 4 {
            return Err(bad_config("n_head", "head_dim must be even and >= 4"));
        }
        if self.layer_norm_epsilon <= 0.0 || self.rotary_emb_base <= 0.0 {
            return Err(bad_config(
                "layer_norm_epsilon / rotary_emb_base",
                "must be positive",
            ));
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    /// `NomicBertModel.__init__` pads `vocab_size` up to a multiple of
    /// `pad_vocab_size_multiple` before building the table
    /// (modeling:1033-1035).
    pub fn padded_vocab_size(&self) -> usize {
        self.vocab_size.div_ceil(self.pad_vocab_size_multiple) * self.pad_vocab_size_multiple
    }

    /// Gated MLP width: `n_inner` rounded up to a multiple of 256.
    pub fn mlp_hidden(&self) -> usize {
        self.n_inner.div_ceil(MLP_MULTIPLE_OF) * MLP_MULTIPLE_OF
    }

    /// `1 / norm_factor` with `norm_factor = sqrt(head_dim)` in f32
    /// (modeling:719-723). The reference divides; for head_dim 16 and 64
    /// the reciprocal is exact, so multiplying gives the same bits.
    pub fn attn_scale(&self) -> f32 {
        1.0 / (self.head_dim() as f32).sqrt()
    }
}
