//! LiquidAI `lfm3` `config.json` parsing (`LFM3-1.7B`, base and post-trained).
//!
//! Every key read off `LiquidAI/LFM3-1.7B` @ 40ebe0aa and its MLX conversion
//! (`docs/LFM3_PHASE0.md`), whose configs agree. The behavioural keys the
//! flow hard-codes (`use_sandwich_norm`, `conv_bias`, the rope types, the
//! sliding window on sliding layers only) are REQUIRED to hold the value the
//! flow implements, and refused otherwise: a checkpoint that turned one of
//! them off would decode fluently and wrongly on this port's flow.
//!
//! The partial rotary factor is read from the FULL-attention class only
//! (`rope_parameters.full_attention.partial_rotary_factor`); the sliding
//! class must declare none (or 1.0), because the flow rotates the whole head
//! there. The rotation is HF's zero-tail form, i.e. `rope_proportional_neox`
//! with a head_dim divisor, which is why `rope_neox_subdim` is false.

use model_io::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig, MlaConfig,
    ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};

use crate::gemma4_checkpoint::Gemma4Error;

const MAX_MODEL_LAYERS: i64 = 4096;

pub fn parse_lfm3_config(json: &str) -> Result<ArchConfig, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    crate::arch_registry::refuse_foreign_config(&root, ModelFamily::Lfm3)
        .map_err(Gemma4Error::Config)?;
    let tc = root.get("text_config").unwrap_or(&root);
    let err = |m: String| Gemma4Error::Config(m);

    let i = |key: &str| -> Result<i64, Gemma4Error> {
        tc.get(key)
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| err(format!("missing {key}")))
    };
    let f = |key: &str| -> Result<f64, Gemma4Error> {
        tc.get(key)
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| err(format!("missing {key}")))
    };
    let b = |key: &str| -> Result<bool, Gemma4Error> {
        tc.get(key)
            .and_then(serde_json::Value::as_bool)
            .ok_or_else(|| err(format!("missing {key}")))
    };

    let hidden_size = i("hidden_size")?;
    let num_heads = i("num_attention_heads")?;
    let num_kv_heads = i("num_key_value_heads")?;
    let num_layers = i("num_hidden_layers")?;
    let intermediate_size = i("intermediate_size")?;
    let vocab_size = i("vocab_size")?;
    let head_dim = i("head_dim")?;
    let sliding_window = i("sliding_window")?;
    let taps = i("conv_L_cache")?;
    if [hidden_size, num_heads, num_kv_heads, num_layers, head_dim]
        .iter()
        .any(|&v| v <= 0)
    {
        return Err(err(
            "hidden size, attention heads, KV heads, layer count, and head_dim must be positive"
                .into(),
        ));
    }
    if num_layers > MAX_MODEL_LAYERS {
        return Err(err(format!(
            "num_hidden_layers {num_layers} exceeds the maximum supported layer count of \
             {MAX_MODEL_LAYERS}"
        )));
    }
    if num_heads % num_kv_heads != 0 {
        return Err(err(format!(
            "num_attention_heads {num_heads} is not divisible by num_key_value_heads {num_kv_heads}"
        )));
    }
    if intermediate_size <= 0 || vocab_size <= 0 || sliding_window <= 0 {
        return Err(err(
            "intermediate_size, vocab_size, and sliding_window must be positive".into(),
        ));
    }
    if !(2..=16).contains(&taps) {
        return Err(err(format!(
            "conv_L_cache {taps} short-conv taps is outside 2..=16"
        )));
    }

    let norm_eps = f("norm_eps")?;
    if (norm_eps - 1e-5).abs() > 1e-12 {
        return Err(err(format!("lfm3 requires norm_eps 1e-5, got {norm_eps}")));
    }
    if !b("use_sandwich_norm")? {
        return Err(err(
            "use_sandwich_norm false is a different layer graph (no post norms) this flow \
             does not implement"
                .into(),
        ));
    }
    if b("conv_bias")? {
        return Err(err(
            "conv_bias true adds three bias vectors the short-conv flow does not read".into(),
        ));
    }
    // Absent means tied for this architecture (`Lfm3ForCausalLM` ties
    // `lm_head` to `embed_tokens`), and the published config says true.
    let tie_word_embeddings = tc
        .get("tie_word_embeddings")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true);

    let mask = layer_mask(tc, num_layers)?;
    let (full_theta, partial) = rope_class(tc, "full_attention")?;
    let (swa_theta, swa_partial) = rope_class(tc, "sliding_attention")?;
    if swa_partial != 1.0 {
        return Err(err(format!(
            "sliding_attention partial_rotary_factor {swa_partial}: the flow rotates the whole \
             head on sliding layers"
        )));
    }
    let rotary = partial * head_dim as f64;
    if rotary <= 0.0 || rotary.fract() != 0.0 || (rotary as i64) % 2 != 0 {
        return Err(err(format!(
            "partial_rotary_factor {partial} x head_dim {head_dim} must be a positive even \
             integer"
        )));
    }

    Ok(ArchConfig {
        hidden_size,
        intermediate_size,
        moe_intermediate_size: 0,
        num_heads,
        num_kv_heads,
        num_full_kv_heads: num_kv_heads,
        head_dim,
        full_head_dim: head_dim,
        vocab_size,
        sliding_window,
        final_logit_softcap: 0.0,
        rope_theta: swa_theta,
        full_rope_theta: full_theta,
        partial_rotary_factor: partial,
        num_layers,
        num_dense_leading_layers: 0,
        dense_lead_intermediate_size: 0,
        num_experts: 0,
        top_k_experts: 0,
        tie_word_embeddings,
        attention_k_eq_v: false,
        full_attention_layer_mask: mask,
        hidden_activation: "silu".to_string(),
        family: ModelFamily::Lfm3,
        attn_output_gate: false,
        attention_scale: (head_dim as f64).powf(-0.5),
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: true,
        shared_expert_gated: false,
        rope_neox_subdim: false,
        linear_attention: LinearAttentionConfig::NONE,
        mla: MlaConfig::NONE,
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: RopeScalingConfig::NONE,
        vision: VisionConfig::NONE,
        short_conv_kernel_size: taps,
        ple: PleConfig::NONE,
    })
}

/// `layer_types` -> mask codes: `conv` 6, `full_attention` 1,
/// `sliding_attention` 0. Any other string refuses: a fourth mixer kind is
/// a flow this port does not have.
fn layer_mask(tc: &serde_json::Value, num_layers: i64) -> Result<Vec<u8>, Gemma4Error> {
    let types = tc
        .get("layer_types")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Gemma4Error::Config("missing layer_types".into()))?;
    if types.len() as i64 != num_layers {
        return Err(Gemma4Error::Config(format!(
            "layer_types has {} entries for {num_layers} layers",
            types.len()
        )));
    }
    types
        .iter()
        .map(|t| match t.as_str() {
            Some("conv") => Ok(6u8),
            Some("full_attention") => Ok(1u8),
            Some("sliding_attention") => Ok(0u8),
            other => Err(Gemma4Error::Config(format!(
                "layer_types entry {other:?} is not conv / full_attention / sliding_attention"
            ))),
        })
        .collect()
}

/// One `rope_parameters` class: `(theta, partial_rotary_factor)`, with the
/// factor's absence meaning 1.0 (the reference's own `.get(..., 1.0)`).
/// Only `rope_type: "default"` is accepted; a scaled type would need the
/// truncated-scaled frequencies `compute_rope_parameters` builds.
fn rope_class(tc: &serde_json::Value, class: &str) -> Result<(f64, f64), Gemma4Error> {
    let p = tc
        .get("rope_parameters")
        .and_then(|r| r.get(class))
        .ok_or_else(|| Gemma4Error::Config(format!("missing rope_parameters.{class}")))?;
    let rope_type = p.get("rope_type").and_then(serde_json::Value::as_str);
    if rope_type != Some("default") {
        return Err(Gemma4Error::Config(format!(
            "rope_parameters.{class}.rope_type {rope_type:?}: only \"default\" is implemented"
        )));
    }
    let theta = p
        .get("rope_theta")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| {
            Gemma4Error::Config(format!("missing rope_parameters.{class}.rope_theta"))
        })?;
    if !theta.is_finite() || theta <= 0.0 {
        return Err(Gemma4Error::Config(format!(
            "rope_parameters.{class}.rope_theta must be finite and positive, got {theta}"
        )));
    }
    let partial = p
        .get("partial_rotary_factor")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(1.0);
    Ok((theta, partial))
}
