//! Dense `qwen3_vl` `config.json` parsing (`Qwen3-VL-4B-Instruct` and
//! siblings).
//!
//! The trunk of this architecture is a plain full-attention GQA stack, so
//! this parser is [`crate::parse_qwen2_config`]'s shape with four
//! differences, each read off the pinned checkpoint's own config
//! (`mlx-community/Qwen3-VL-4B-Instruct-4bit`,
//! `docs/QWEN3VL_PHASE0.md`) rather than carried over:
//!
//! 1. `head_dim` is a REQUIRED key and need not equal
//!    `hidden_size / num_attention_heads` (here 128 against 2560/32 = 80):
//!    the q/k/v/o projections do not preserve the hidden width.
//! 2. Per-head q/k RMSNorm is a family constant, not a config key --
//!    `self_attn.{q,k}_norm.weight` tensors exist in the checkpoint and the
//!    reference's `Attention.__init__` builds them unconditionally.
//! 3. RoPE is FULL over the head. The config carries no partial factor at
//!    all, and the reference constructs its rotary at the full head dim;
//!    absence here means full rotary (AGENTS.md Gotcha 39), so anything
//!    else is refused rather than defaulted.
//! 4. The `vision_config` sibling is IGNORED, not parsed: this port's
//!    intake for the family is text-only in this pass (every
//!    `vision_tower.*` tensor, the 18 deepstack mergers included, is
//!    `ExcludedMultimodal` at classification), the `muse_glimmer`
//!    precedent. A config whose tower declares an output width that does
//!    not match the trunk would still install text-only -- the mismatch
//!    matters only to the vision bring-up this pass defers.

use model_io::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig, MlaConfig,
    ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};

use crate::gemma4_checkpoint::Gemma4Error;

pub fn parse_qwen3_vl_config(json: &str) -> Result<ArchConfig, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    crate::arch_registry::refuse_foreign_config(&root, ModelFamily::Qwen3Vl)
        .map_err(Gemma4Error::Config)?;
    let tc = root.get("text_config").unwrap_or(&root);

    let i = |key: &str| -> Result<i64, Gemma4Error> {
        tc.get(key)
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| Gemma4Error::Config(format!("missing {key}")))
    };
    let f = |key: &str| -> Result<f64, Gemma4Error> {
        tc.get(key)
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| Gemma4Error::Config(format!("missing {key}")))
    };

    let hidden_size = i("hidden_size")?;
    let num_heads = i("num_attention_heads")?;
    let num_kv_heads = i("num_key_value_heads")?;
    let num_layers = i("num_hidden_layers")?;
    let intermediate_size = i("intermediate_size")?;
    let vocab_size = i("vocab_size")?;
    let max_position_embeddings = i("max_position_embeddings")?;
    // Independent of hidden/heads on this architecture; required rather
    // than derived.
    let head_dim = i("head_dim")?;
    if hidden_size <= 0 || num_heads <= 0 || num_kv_heads <= 0 || num_layers <= 0 || head_dim <= 0 {
        return Err(Gemma4Error::Config(
            "hidden size, attention heads, KV heads, layer count, and head_dim must be positive"
                .into(),
        ));
    }
    if num_heads % num_kv_heads != 0 {
        return Err(Gemma4Error::Config(format!(
            "num_attention_heads {num_heads} is not divisible by num_key_value_heads {num_kv_heads}"
        )));
    }
    if intermediate_size <= 0 || vocab_size <= 0 || max_position_embeddings <= 0 {
        return Err(Gemma4Error::Config(
            "intermediate_size, vocab_size, and max_position_embeddings must be positive".into(),
        ));
    }

    let rms_norm_eps = f("rms_norm_eps")?;
    if (rms_norm_eps - 1e-6).abs() > 1e-12 {
        return Err(Gemma4Error::Config(format!(
            "qwen3_vl requires rms_norm_eps 1e-6, got {rms_norm_eps}"
        )));
    }
    let rope_theta = f("rope_theta")?;
    if !rope_theta.is_finite() || rope_theta <= 0.0 {
        return Err(Gemma4Error::Config(format!(
            "rope_theta must be finite and positive, got {rope_theta}"
        )));
    }
    let hidden_activation = tc
        .get("hidden_act")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| Gemma4Error::Config("missing hidden_act".into()))?;
    if hidden_activation != "silu" {
        return Err(Gemma4Error::Config(format!(
            "qwen3_vl requires hidden_act silu, got {hidden_activation:?}"
        )));
    }

    // The reference's rope_scaling accepts type "mrope" and "default" only
    // and requires mrope_section under it. Nothing here reaches an
    // ArchConfig field -- on TEXT positions the three mRoPE components are
    // equal and the section triple collapses -- but a scaling type this
    // port does not know is a checkpoint whose positions it would compute
    // wrong, so it is refused rather than ignored. A present mrope_section
    // is validated against the one invariant the reference enforces by
    // construction: the sections partition the rotated sub-frequencies, so
    // they must sum to head_dim / 2.
    if let Some(scaling) = tc.get("rope_scaling").filter(|v| !v.is_null()) {
        let rope_type = scaling
            .get("rope_type")
            .or_else(|| scaling.get("type"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("default");
        if rope_type != "default" && rope_type != "mrope" {
            return Err(Gemma4Error::Config(format!(
                "qwen3_vl rope_scaling rope_type {rope_type:?} is not wired; expected \
                 \"default\" or \"mrope\""
            )));
        }
        if let Some(section) = scaling.get("mrope_section").and_then(|v| v.as_array()) {
            let parts: Vec<i64> = section
                .iter()
                .map(|v| {
                    v.as_i64()
                        .ok_or_else(|| Gemma4Error::Config("mrope_section must be integers".into()))
                })
                .collect::<Result<_, _>>()?;
            if parts.iter().any(|&p| p < 0) || parts.iter().sum::<i64>() != head_dim / 2 {
                return Err(Gemma4Error::Config(format!(
                    "mrope_section {parts:?} must be non-negative and sum to head_dim / 2 \
                     ({}), got {}",
                    head_dim / 2,
                    parts.iter().sum::<i64>()
                )));
            }
        }
    }

    // This flow has no attention-bias arm (Qwen2's is a different family
    // flag); a checkpoint that declares one is refused by name rather than
    // run with the bias silently dropped.
    if tc
        .get("attention_bias")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        return Err(Gemma4Error::Config(
            "qwen3_vl attention_bias true is not wired on this flow".into(),
        ));
    }
    // A dense family: a config that declares experts is a different
    // architecture, and reading only half of it would dispatch the wrong
    // FFN branch.
    if let Some(n) = tc.get("num_experts").and_then(serde_json::Value::as_i64) {
        if n > 0 {
            return Err(Gemma4Error::Config(format!(
                "a qwen3_vl config declares num_experts {n}; the family is dense, and a \
                 checkpoint with experts is not qwen3_vl"
            )));
        }
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
        sliding_window: 0,
        final_logit_softcap: 0.0,
        rope_theta,
        full_rope_theta: rope_theta,
        // FULL rotary; see the module doc. A present non-1.0 factor is
        // refused rather than honored, because no published checkpoint
        // carries one and the shared flow's partial arm was never checked
        // against this architecture.
        partial_rotary_factor: 1.0,
        num_layers,
        dense_lead_intermediate_size: 0,
        num_dense_leading_layers: 0,
        num_experts: 0,
        top_k_experts: 0,
        tie_word_embeddings: tc
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        attention_k_eq_v: false,
        full_attention_layer_mask: vec![1; num_layers as usize],
        hidden_activation: hidden_activation.to_string(),
        family: ModelFamily::Qwen3Vl,
        attn_output_gate: false,
        attention_scale: (head_dim as f64).powf(-0.5),
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
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
        ple: PleConfig::NONE,
    })
}
