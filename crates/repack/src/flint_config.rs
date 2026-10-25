//! Recursive Flint `config.json` parsing (`model_type: "flint"`).
//!
//! The checkpoint ships no modeling code, so every refusal below is a
//! behaviour the decode flow does not implement, stated at the door rather
//! than discovered as fluent wrong output. Facts: `docs/FLINT_PHASE0.md`.

use model_io::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig, MlaConfig,
    ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};

use crate::gemma4_checkpoint::{Gemma4Error, Gemma4Quant};

/// The canonical tensor path the embedding table lives under, which is where
/// `Gemma4Quant` overrides are keyed once `canonicalize_flint_header` has
/// renamed the header.
const EMBED_BASE: &str = "language_model.model.embed_tokens";

/// Parse a Flint configuration into the canonical architecture shape.
pub fn parse_flint_config(json: &str) -> Result<ArchConfig, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    crate::arch_registry::refuse_foreign_config(&root, ModelFamily::Flint)
        .map_err(Gemma4Error::Config)?;
    // Flat config; a `text_config` wrapper is honored if one ever appears.
    let tc = root.get("text_config").unwrap_or(&root);

    let get = |key: &str| tc.get(key);
    let i = |key: &str| -> Result<i64, Gemma4Error> {
        get(key)
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| Gemma4Error::Config(format!("missing {key}")))
    };
    let f = |key: &str| -> Result<f64, Gemma4Error> {
        get(key)
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| Gemma4Error::Config(format!("missing {key}")))
    };
    let s = |key: &str| -> Result<&str, Gemma4Error> {
        get(key)
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| Gemma4Error::Config(format!("missing {key}")))
    };
    let b = |key: &str| -> Option<bool> { get(key).and_then(serde_json::Value::as_bool) };
    let int_list = |key: &str| -> Result<Vec<i64>, Gemma4Error> {
        get(key)
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| Gemma4Error::Config(format!("missing {key}")))?
            .iter()
            .map(|v| {
                v.as_i64()
                    .ok_or_else(|| Gemma4Error::Config(format!("{key} holds a non-integer")))
            })
            .collect()
    };

    let hidden_size = i("hidden_size")?;
    let num_heads = i("num_attention_heads")?;
    let num_kv_heads = i("num_key_value_heads")?;
    let head_dim = i("head_dim")?;
    let num_layers = i("num_hidden_layers")?;
    let dense_width = i("intermediate_size")?;
    let moe_width = i("moe_intermediate_size")?;
    let shared_width = i("shared_expert_intermediate_size")?;
    let num_experts = i("num_experts")?;
    let top_k = i("num_experts_per_tok")?;
    let vocab_size = i("vocab_size")?;
    let sliding_window = i("sliding_window")?;
    for (name, v) in [
        ("hidden_size", hidden_size),
        ("num_attention_heads", num_heads),
        ("num_key_value_heads", num_kv_heads),
        ("head_dim", head_dim),
        ("num_hidden_layers", num_layers),
        ("intermediate_size", dense_width),
        ("moe_intermediate_size", moe_width),
        ("shared_expert_intermediate_size", shared_width),
        ("num_experts", num_experts),
        ("num_experts_per_tok", top_k),
        ("vocab_size", vocab_size),
        ("sliding_window", sliding_window),
    ] {
        if v <= 0 {
            return Err(Gemma4Error::Config(format!(
                "{name} must be positive, got {v}"
            )));
        }
    }
    if num_heads % num_kv_heads != 0 {
        return Err(Gemma4Error::Config(format!(
            "num_attention_heads {num_heads} is not divisible by num_key_value_heads {num_kv_heads}"
        )));
    }
    if top_k > num_experts {
        return Err(Gemma4Error::Config(format!(
            "num_experts_per_tok {top_k} exceeds num_experts {num_experts}"
        )));
    }

    // Behaviours the flow does not implement. Each is a refusal rather than
    // a default, because any of them would still produce fluent text.
    let rms_norm_eps = f("rms_norm_eps")?;
    if (rms_norm_eps - 1e-6).abs() > 1e-12 {
        return Err(Gemma4Error::Config(format!(
            "Flint requires rms_norm_eps 1e-6, got {rms_norm_eps}"
        )));
    }
    if s("hidden_act")? != "silu" {
        return Err(Gemma4Error::Config(format!(
            "Flint requires hidden_act silu, got {:?}",
            s("hidden_act")?
        )));
    }
    if s("scoring_func")? != "sigmoid" || b("norm_topk_prob") != Some(true) {
        return Err(Gemma4Error::Config(
            "Flint's router is sigmoid scoring renormalised over the selected top-k \
             (scoring_func sigmoid, norm_topk_prob true); other routers are not wired"
                .into(),
        ));
    }
    if b("router_expert_bias") != Some(false) {
        return Err(Gemma4Error::Config(
            "Flint router_expert_bias must be false; a biased router is not wired".into(),
        ));
    }
    if b("attention_sink") != Some(true) || b("attention_bias") != Some(false) {
        return Err(Gemma4Error::Config(
            "Flint requires attention_sink true and attention_bias false".into(),
        ));
    }
    if b("use_sliding_window") != Some(true) {
        return Err(Gemma4Error::Config(
            "Flint requires use_sliding_window true".into(),
        ));
    }

    // RoPE: a plain default-type theta, read from `rope_parameters` and
    // cross-checked against the flat key the file also carries.
    let rope_params = get("rope_parameters")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| Gemma4Error::Config("missing rope_parameters".into()))?;
    if rope_params.get("rope_type").and_then(|v| v.as_str()) != Some("default") {
        return Err(Gemma4Error::Config(
            "Flint requires rope_type default; scaled RoPE is not wired".into(),
        ));
    }
    let rope_theta = rope_params
        .get("rope_theta")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| Gemma4Error::Config("missing rope_parameters.rope_theta".into()))?;
    if !rope_theta.is_finite() || rope_theta <= 0.0 || (f("rope_theta")? - rope_theta).abs() > 0.0 {
        return Err(Gemma4Error::Config(format!(
            "rope_theta {rope_theta} must be positive and agree with the flat rope_theta"
        )));
    }

    // Layer kinds. The mask is derived from `layer_types` and then held to
    // `no_rope_layer_indices`, because the flow ties NoPE to mask 1 and a
    // checkpoint that separated them would decode wrongly without an error.
    let layer_types = get("layer_types")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Gemma4Error::Config("missing layer_types".into()))?;
    if layer_types.len() as i64 != num_layers {
        return Err(Gemma4Error::Config(format!(
            "layer_types has {} entries for {num_layers} layers",
            layer_types.len()
        )));
    }
    let mask: Vec<u8> = layer_types
        .iter()
        .map(|v| match v.as_str() {
            Some("full_attention") => Ok(1u8),
            Some("sliding_attention") => Ok(0u8),
            other => Err(Gemma4Error::Config(format!("unknown layer type {other:?}"))),
        })
        .collect::<Result<_, _>>()?;
    let full_layers: Vec<i64> = mask
        .iter()
        .enumerate()
        .filter(|(_, &m)| m == 1)
        .map(|(idx, _)| idx as i64)
        .collect();
    let mut no_rope = int_list("no_rope_layer_indices")?;
    no_rope.sort_unstable();
    if no_rope != full_layers {
        return Err(Gemma4Error::Config(format!(
            "no_rope_layer_indices {no_rope:?} differ from the full-attention layers \
             {full_layers:?}; the flow ties NoPE to the full-attention mask"
        )));
    }

    // Dense lead: `mlp_only_layers` must be the prefix 0..k, because the
    // routed-expert blobs are indexed from the first MoE layer.
    let mut dense = int_list("mlp_only_layers")?;
    dense.sort_unstable();
    if dense != (0..dense.len() as i64).collect::<Vec<_>>() {
        return Err(Gemma4Error::Config(format!(
            "mlp_only_layers {dense:?} is not a leading prefix; Flint's dense layers must \
             precede the MoE layers"
        )));
    }
    if b("tie_word_embeddings") != Some(false) {
        return Err(Gemma4Error::Config(
            "Flint ships an untied lm_head; tie_word_embeddings must be false".into(),
        ));
    }

    Ok(ArchConfig {
        hidden_size,
        intermediate_size: shared_width,
        moe_intermediate_size: moe_width,
        num_heads,
        num_kv_heads,
        num_full_kv_heads: num_kv_heads,
        head_dim,
        full_head_dim: head_dim,
        vocab_size,
        sliding_window,
        final_logit_softcap: 0.0,
        rope_theta,
        // NoPE on the full-attention layers: zero, as `muse_glimmer`.
        full_rope_theta: 0.0,
        partial_rotary_factor: 1.0,
        num_layers,
        num_dense_leading_layers: dense.len() as i64,
        dense_lead_intermediate_size: dense_width,
        num_experts,
        top_k_experts: top_k,
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        full_attention_layer_mask: mask,
        hidden_activation: "silu".to_string(),
        family: ModelFamily::Flint,
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
        router_scoring_func: "sigmoid".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: RopeScalingConfig::NONE,
        vision: VisionConfig::NONE,
        short_conv_kernel_size: 0,
        ple: PleConfig::NONE,
    })
}

/// `config.json -> quantization`, plus the embedding width.
///
/// **The embedding is INT8 under an INT4 default and `RecursiveSI/flint-int4`
/// does not say so in `quantization`**: it declares only `{bits: 4,
/// group_size: 64}`, and the 8-bit table is recorded in the producer's own
/// `mlx_export` block (`embedding_bits`). Reading the generic quantization
/// alone would size the embedding as 4-bit and reject its (correctly shaped)
/// packed rows, or worse accept a mis-sized read. The override is keyed by
/// the canonical path `canonicalize_flint_header` produces.
///
/// `mlx-community/flint-mixed-3-4bit` instead writes stock MLX per-tensor
/// entries (`model.embed_tokens` at 8, every `switch_mlp` projection at 3)
/// beside the same `mlx_export` block. Those keys are in the SOURCE
/// namespace, so they are renamed onto the canonical one here, the same rename
/// the header gets; left as they were, `bits_for` would never find them and
/// every 3-bit expert would be read as INT4.
///
/// What is admitted is what the kernels cover: an embedding at 4 or 8, and
/// routed experts at 3 or 4, all experts at one width that agrees with
/// `mlx_export.routed_expert_bits`. 3-bit is admitted for `switch_mlp`
/// tensors only (the `moe_*_int3` routed pair); anything else per-tensor is
/// refused.
///
/// `lm_head` needs no override: it is 4-bit, the default.
pub fn parse_flint_quantization(json: &str) -> Result<Gemma4Quant, Gemma4Error> {
    let routed = |name: &str| name.contains(ROUTED_MARKER);
    let mut quant =
        crate::gemma4_checkpoint::parse_quantization_admitting(json, |name, bits, gs| {
            routed(name) && bits == 3 && gs == crate::gemma4_checkpoint::AFFINE_GROUP_SIZE
        })?;
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    let export = root
        .get("mlx_export")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| Gemma4Error::Config("missing mlx_export block".into()))?;
    let embedding_bits = export
        .get("embedding_bits")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| Gemma4Error::Config("missing mlx_export.embedding_bits".into()))?
        as u32;
    if !matches!(embedding_bits, 4 | 8) {
        return Err(Gemma4Error::Config(format!(
            "mlx_export.embedding_bits {embedding_bits} is not 4 or 8"
        )));
    }
    let routed_bits = export
        .get("routed_expert_bits")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| Gemma4Error::Config("missing mlx_export.routed_expert_bits".into()))?
        as u32;
    if !matches!(routed_bits, 3 | 4) {
        return Err(Gemma4Error::Config(format!(
            "mlx_export.routed_expert_bits {routed_bits} is not 3 or 4"
        )));
    }

    let mut overrides = std::collections::HashMap::new();
    for (key, bits) in std::mem::take(&mut quant.bits_overrides) {
        let canonical = SOURCE_TO_CANONICAL
            .iter()
            .find_map(|(source, target)| key.strip_prefix(source).map(|t| format!("{target}{t}")))
            .unwrap_or(key);
        if canonical == EMBED_BASE {
            if bits != embedding_bits {
                return Err(Gemma4Error::Config(format!(
                    "quantization declares the embedding at {bits} bits and \
                     mlx_export.embedding_bits at {embedding_bits}"
                )));
            }
        } else if routed(&canonical) {
            if bits != routed_bits {
                return Err(Gemma4Error::Config(format!(
                    "quantization declares {canonical} at {bits} bits and \
                     mlx_export.routed_expert_bits at {routed_bits}; every routed expert \
                     must be one width"
                )));
            }
        } else if bits != quant.default_bits {
            return Err(Gemma4Error::Config(format!(
                "quantization declares {canonical} at {bits} bits; per-tensor widths beyond \
                 the embedding and the routed experts are not modelled"
            )));
        }
        overrides.insert(canonical, bits);
    }
    quant.bits_overrides = overrides;
    // A width `mlx_export` states but no per-tensor entry carries (the INT4
    // build's INT8 embedding) is recorded here. Routed experts are never
    // inferred this way: a routed width off the default with no per-tensor
    // entries is a config the walk cannot size, so it is refused.
    if embedding_bits != quant.default_bits {
        quant
            .bits_overrides
            .entry(EMBED_BASE.to_string())
            .or_insert(embedding_bits);
    }
    let routed_declared = quant.bits_overrides.keys().any(|k| routed(k));
    if routed_bits != quant.default_bits && !routed_declared {
        return Err(Gemma4Error::Config(format!(
            "mlx_export.routed_expert_bits is {routed_bits} under a {}-bit default, but \
             quantization declares no routed expert at that width",
            quant.default_bits
        )));
    }
    Ok(quant)
}

/// Where a Flint checkpoint's routed experts live, which is the only place
/// 3-bit is admitted.
const ROUTED_MARKER: &str = ".mlp.switch_mlp.";

/// `canonicalize_flint_header`'s rename, for quantization keys (which name a
/// tensor without its `.weight` suffix, so `lm_head` has no trailing dot).
const SOURCE_TO_CANONICAL: [(&str, &str); 4] = [
    ("model.embed_tokens", "language_model.model.embed_tokens"),
    ("model.layers.", "language_model.model.layers."),
    ("model.norm", "language_model.model.norm"),
    ("lm_head", "language_model.lm_head"),
];
