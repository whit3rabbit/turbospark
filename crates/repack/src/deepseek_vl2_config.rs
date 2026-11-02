//! DeepSeek-VL2 `config.json` parsing (`docs/DEEPSEEK_VL2_VISION_PHASE0.md`).
//!
//! The wrapper's TEXT model is `language_config`: a DeepSeekMoE trunk whose
//! attention is plain multi-head (`use_mla: false`). It maps onto the
//! `deepseek2` family with an INACTIVE `mla` block, which is what the flow
//! switches on; no family was added for it.
//!
//! Where the config is silent the REFERENCE's defaults are the format's
//! answer (AGENTS.md Gotcha 39), not a sibling's: `rope_theta` 10000,
//! `rms_norm_eps` 1e-6, `scoring_func` softmax, `topk_method` greedy. The
//! flow hardcodes the epsilon and the router rule, so a config that states
//! anything else is refused rather than run with the wrong function.

use model_io::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig, MlaConfig,
    ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};

use crate::gemma4_checkpoint::Gemma4Error;

/// The reference `TextConfig` default for an absent `rope_theta`.
const DEFAULT_ROPE_THETA: f64 = 10_000.0;

/// Parse a DeepSeek-VL2 configuration into the canonical architecture shape.
///
/// Accepts the wrapper form (`language_config` beside `vision_config`) and a
/// bare language config. The vision tower is never enabled here: like every
/// other family, a tower is declared by the bytes (`enable_*_vision`), not by
/// this text parser.
pub fn parse_deepseek_vl2_config(json: &str) -> Result<ArchConfig, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    crate::arch_registry::refuse_foreign_config(&root, ModelFamily::Deepseek2)
        .map_err(Gemma4Error::Config)?;
    let lc = root.get("language_config").unwrap_or(&root);

    let i = |key: &str| -> Result<i64, Gemma4Error> {
        lc.get(key)
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| Gemma4Error::Config(format!("missing language_config.{key}")))
    };
    let opt_f = |key: &str| lc.get(key).and_then(serde_json::Value::as_f64);

    if lc.get("use_mla").and_then(serde_json::Value::as_bool) != Some(false) {
        return Err(Gemma4Error::Config(
            "this parser covers the non-MLA DeepSeek-VL2 trunks (use_mla: false); an MLA trunk \
             is the V2/V3 shape the GGUF intake owns"
                .into(),
        ));
    }
    // The reference selects its attention class on `qk_nope_head_dim == 0`;
    // an MLA-shaped config under `use_mla: false` would be contradictory.
    if lc
        .get("qk_nope_head_dim")
        .and_then(serde_json::Value::as_i64)
        .unwrap_or(0)
        != 0
    {
        return Err(Gemma4Error::Config(
            "use_mla is false but qk_nope_head_dim is nonzero".into(),
        ));
    }

    let hidden_size = i("hidden_size")?;
    let num_heads = i("num_attention_heads")?;
    let num_kv_heads = i("num_key_value_heads")?;
    let num_layers = i("num_hidden_layers")?;
    let dense_lead_intermediate_size = i("intermediate_size")?;
    let moe_intermediate_size = i("moe_intermediate_size")?;
    let num_experts = i("n_routed_experts")?;
    let top_k_experts = i("num_experts_per_tok")?;
    let n_shared = i("n_shared_experts")?;
    let lead = i("first_k_dense_replace")?;
    let vocab_size = i("vocab_size")?;
    if hidden_size <= 0
        || num_heads <= 0
        || num_kv_heads != num_heads
        || num_layers <= 0
        || num_experts <= 0
        || top_k_experts <= 0
        || n_shared <= 0
        || moe_intermediate_size <= 0
        || dense_lead_intermediate_size <= 0
        || vocab_size <= 0
    {
        return Err(Gemma4Error::Config(
            "DeepSeek-VL2 needs positive dimensions and as many KV heads as query heads (the \
             flow's plain attention is multi-head)"
                .into(),
        ));
    }
    if hidden_size % num_heads != 0 {
        return Err(Gemma4Error::Config(format!(
            "hidden_size {hidden_size} is not divisible by num_attention_heads {num_heads}"
        )));
    }
    if lead < 1 || lead >= num_layers {
        return Err(Gemma4Error::Config(format!(
            "first_k_dense_replace {lead} must leave a dense lead and at least one MoE layer \
             (the flow maps checkpoint layer L to routed blob L - lead)"
        )));
    }
    if i("moe_layer_freq").unwrap_or(1) != 1 {
        return Err(Gemma4Error::Config(
            "moe_layer_freq other than 1 is not wired".into(),
        ));
    }
    if lc
        .get("topk_method")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("greedy")
        != "greedy"
        || lc
            .get("n_group")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(1)
            != 1
        || lc
            .get("topk_group")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(1)
            != 1
    {
        return Err(Gemma4Error::Config(
            "only greedy top-k with one routing group is wired (softmax over all experts, top-k, \
             no renormalization)"
                .into(),
        ));
    }
    if lc
        .get("scoring_func")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("softmax")
        != "softmax"
    {
        return Err(Gemma4Error::Config(
            "only a softmax router is wired for DeepSeek-VL2".into(),
        ));
    }
    if lc
        .get("norm_topk_prob")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        return Err(Gemma4Error::Config(
            "norm_topk_prob is not wired: the reference gate returns the raw softmax values".into(),
        ));
    }
    let routed_scaling_factor = opt_f("routed_scaling_factor").unwrap_or(1.0);
    if (routed_scaling_factor - 1.0).abs() > 1e-12 {
        return Err(Gemma4Error::Config(format!(
            "routed_scaling_factor {routed_scaling_factor} is not wired (the flow multiplies by 1)"
        )));
    }
    let rms_norm_eps = opt_f("rms_norm_eps").unwrap_or(1e-6);
    if (rms_norm_eps - 1e-6).abs() > 1e-12 {
        return Err(Gemma4Error::Config(format!(
            "DeepSeek-VL2 requires rms_norm_eps 1e-6, got {rms_norm_eps}"
        )));
    }
    let rope_theta = opt_f("rope_theta").unwrap_or(DEFAULT_ROPE_THETA);
    if !rope_theta.is_finite() || rope_theta <= 0.0 {
        return Err(Gemma4Error::Config(format!(
            "rope_theta must be finite and positive, got {rope_theta}"
        )));
    }
    if lc.get("rope_scaling").is_some_and(|v| !v.is_null()) {
        return Err(Gemma4Error::Config(
            "rope_scaling is not wired for the DeepSeek-VL2 trunk".into(),
        ));
    }
    if lc
        .get("rope_traditional")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        return Err(Gemma4Error::Config(
            "traditional (interleaved) rope is not wired; the flow rotates NeoX half-split".into(),
        ));
    }
    if lc
        .get("attention_bias")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        return Err(Gemma4Error::Config(
            "attention biases are not wired for the DeepSeek-VL2 trunk".into(),
        ));
    }
    let hidden_act = lc
        .get("hidden_act")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("silu");
    if hidden_act != "silu" {
        return Err(Gemma4Error::Config(format!(
            "DeepSeek-VL2 requires hidden_act silu, got {hidden_act:?}"
        )));
    }
    let tie = lc
        .get("tie_word_embeddings")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    if tie {
        return Err(Gemma4Error::Config(
            "the DeepSeek-VL2 trunk ships an untied lm_head".into(),
        ));
    }

    let head_dim = hidden_size / num_heads;
    Ok(ArchConfig {
        hidden_size,
        // The fused shared expert: `n_shared_experts` of `moe_intermediate`
        // run as ONE SwiGLU (the checkpoint stores them already fused).
        intermediate_size: moe_intermediate_size * n_shared,
        moe_intermediate_size,
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
        partial_rotary_factor: 1.0,
        num_layers,
        num_experts,
        top_k_experts,
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        full_attention_layer_mask: vec![1; num_layers as usize],
        hidden_activation: hidden_act.to_string(),
        family: ModelFamily::Deepseek2,
        attn_output_gate: false,
        attention_scale: (head_dim as f64).powf(-0.5),
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        shared_expert_gated: false,
        rope_neox_subdim: true,
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
        short_conv_kernel_size: 0,
        ple: PleConfig::NONE,
        dense_lead_intermediate_size,
        num_dense_leading_layers: lead,
    })
}
