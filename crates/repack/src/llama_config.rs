//! Dense Llama / Mistral `config.json` parsing (HF transformers and MLX).
//!
//! Witnesses, read off the published files:
//! - `unsloth/Llama-3.2-1B-Instruct` @ 5a8abab4a5d6f164389b1079fb721cfab8d7126c
//!   (`model_type: llama`, `rope_scaling.rope_type: llama3`, tied embeddings,
//!   explicit `head_dim`);
//! - `mlx-community/Llama-3.2-1B-Instruct-4bit` @
//!   08231374eeacb049a0eade7922910865b8fce912 (the same config plus MLX's
//!   `quantization` block);
//! - `HuggingFaceTB/SmolLM2-135M-Instruct` @
//!   12fd25f77366fa6b3b4b768ec3050bf629380bac (`rope_scaling: null`, tied);
//! - `mistralai/Mistral-7B-Instruct-v0.2` @
//!   63a8b081895390a26e140280378bc85ec8bce07a (`model_type: mistral`,
//!   `sliding_window: null`).
//!
//! The llama flow is what runs these, so the parser refuses every config
//! field that flow cannot honour rather than dropping it: a non-1e-5 RMS
//! epsilon (the flow hardcodes it for this family), a live sliding window
//! (Mistral v0.1), attention or MLP biases, routed experts, a VLM wrapper,
//! and any rope scaling other than none or `llama3`.

use model_io::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig, MlaConfig,
    ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};

use crate::gemma4_checkpoint::Gemma4Error;

/// The epsilon the llama flow applies for `ModelFamily::Llama`
/// (`families/llama/state.rs`). It is not an `ArchConfig` field, so a
/// config declaring another value must be refused, not installed.
const LLAMA_RMS_EPS: f64 = 1e-5;

/// HF `LlamaConfig`'s default `rope_theta`, used when the key is absent.
const DEFAULT_ROPE_THETA: f64 = 10_000.0;

/// Parse a dense Llama or Mistral configuration into the canonical
/// architecture shape.
pub fn parse_llama_config(json: &str) -> Result<ArchConfig, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    crate::arch_registry::refuse_foreign_config(&root, ModelFamily::Llama)
        .map_err(Gemma4Error::Config)?;
    let config_err = |msg: String| Gemma4Error::Config(msg);

    // ROOT ONLY, with one wrapper. `text_config.model_type: llama` is how
    // LLaVA-style wrappers describe their trunk, and their tower and weight
    // layout were not read. A config with no model_type at all (trimmed
    // fixtures) is accepted, which is `refuse_foreign_config`'s rule too. The
    // Idefics3 / SmolVLM wrappers are the exception: their trunk fields sit in
    // `text_config`, their tensors were read off published indexes
    // (`docs/IDEFICS3_VISION_PHASE0.md`) and their tower has its own parser.
    let idefics3_wrapper = matches!(
        root.get("model_type").and_then(serde_json::Value::as_str),
        Some(crate::gemma4_checkpoint::IDEFICS3_MODEL_TYPE)
            | Some(crate::gemma4_checkpoint::SMOLVLM_MODEL_TYPE)
    );
    let trunk = if idefics3_wrapper {
        let trunk = root
            .get("text_config")
            .filter(|v| v.is_object())
            .ok_or_else(|| {
                config_err("an idefics3 or smolvlm config carries its trunk in text_config".into())
            })?;
        match trunk.get("model_type").and_then(serde_json::Value::as_str) {
            None | Some("llama") => {}
            Some(other) => {
                return Err(config_err(format!(
                    "text_config.model_type {other:?}; only a llama trunk installs under this \
                     wrapper"
                )))
            }
        }
        trunk
    } else {
        match root.get("model_type").and_then(serde_json::Value::as_str) {
            None | Some("llama") | Some("mistral") => {}
            Some(other) => {
                return Err(config_err(format!(
                    "model_type {other:?} wraps a llama trunk; only plain llama and mistral \
                     configs install through this intake"
                )))
            }
        }
        if root.get("vision_config").is_some() || root.get("text_config").is_some() {
            return Err(config_err(
                "a llama config with vision_config or text_config is a multimodal wrapper, \
                 which this intake does not install"
                    .into(),
            ));
        }
        &root
    };

    let i = |key: &str| -> Result<i64, Gemma4Error> {
        trunk
            .get(key)
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| Gemma4Error::Config(format!("missing {key}")))
    };
    let hidden_size = i("hidden_size")?;
    let num_heads = i("num_attention_heads")?;
    // HF `LlamaConfig` defaults this to `num_attention_heads` (plain MHA).
    let num_kv_heads = match trunk.get("num_key_value_heads") {
        None | Some(serde_json::Value::Null) => num_heads,
        Some(v) => v
            .as_i64()
            .ok_or_else(|| config_err("num_key_value_heads must be an integer".into()))?,
    };
    let num_layers = i("num_hidden_layers")?;
    let intermediate_size = i("intermediate_size")?;
    let vocab_size = i("vocab_size")?;
    let max_position_embeddings = i("max_position_embeddings")?;
    if hidden_size <= 0 || num_heads <= 0 || num_kv_heads <= 0 || num_layers <= 0 {
        return Err(config_err(
            "hidden size, attention heads, and layer count must be positive".into(),
        ));
    }
    if num_heads % num_kv_heads != 0 {
        return Err(config_err(format!(
            "num_attention_heads {num_heads} is not divisible by num_key_value_heads {num_kv_heads}"
        )));
    }
    if intermediate_size <= 0 || vocab_size <= 0 || max_position_embeddings <= 0 {
        return Err(config_err(
            "intermediate_size, vocab_size, and max_position_embeddings must be positive".into(),
        ));
    }
    // An explicit `head_dim` may differ from hidden / heads (Mistral Nemo);
    // `null` or absent means the quotient, which must then be exact.
    let head_dim = match trunk.get("head_dim") {
        None | Some(serde_json::Value::Null) => {
            if hidden_size % num_heads != 0 {
                return Err(config_err(format!(
                    "hidden_size {hidden_size} is not divisible by num_attention_heads \
                     {num_heads} and no head_dim is given"
                )));
            }
            hidden_size / num_heads
        }
        Some(v) => v
            .as_i64()
            .filter(|d| *d > 0 && d % 2 == 0)
            .ok_or_else(|| config_err(format!("head_dim {v} must be a positive even integer")))?,
    };

    let rms_norm_eps = trunk
        .get("rms_norm_eps")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| config_err("missing rms_norm_eps".into()))?;
    if (rms_norm_eps - LLAMA_RMS_EPS).abs() > 1e-12 {
        return Err(config_err(format!(
            "the llama flow applies rms_norm_eps {LLAMA_RMS_EPS:e}; this config declares \
             {rms_norm_eps:e}, which it cannot honour"
        )));
    }
    let hidden_activation = trunk
        .get("hidden_act")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("silu");
    if hidden_activation != "silu" {
        return Err(config_err(format!(
            "the llama flow is SwiGLU; hidden_act {hidden_activation:?} is not wired"
        )));
    }
    for key in ["attention_bias", "mlp_bias"] {
        if trunk.get(key).and_then(serde_json::Value::as_bool) == Some(true) {
            return Err(config_err(format!(
                "{key} is true; the llama flow carries no projection biases for this family"
            )));
        }
    }
    for key in ["num_local_experts", "num_experts"] {
        if trunk
            .get(key)
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(0)
            > 0
        {
            return Err(config_err(format!(
                "{key} > 0: routed experts have no safetensors writer on the llama flow \
                 (install a Mixtral from GGUF)"
            )));
        }
    }
    match trunk.get("sliding_window") {
        None | Some(serde_json::Value::Null) => {}
        Some(v) => {
            return Err(config_err(format!(
                "sliding_window {v} is set; the llama flow runs full attention only \
                 (Mistral v0.1's window is not wired)"
            )))
        }
    }
    if trunk
        .get("partial_rotary_factor")
        .and_then(serde_json::Value::as_f64)
        .is_some_and(|p| p != 1.0)
    {
        return Err(config_err(
            "partial_rotary_factor other than 1.0 is not wired for llama".into(),
        ));
    }

    // transformers v5 moves the rope keys into `rope_parameters`; older
    // configs keep `rope_theta` at the root and `rope_scaling` beside it.
    let rope_parameters = trunk.get("rope_parameters").filter(|v| !v.is_null());
    let rope_theta = rope_parameters
        .and_then(|p| p.get("rope_theta"))
        .or_else(|| trunk.get("rope_theta"))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(DEFAULT_ROPE_THETA);
    if !rope_theta.is_finite() || rope_theta <= 1.0 {
        return Err(config_err(format!(
            "rope_theta must be finite and above 1, got {rope_theta}"
        )));
    }
    let scaling = rope_parameters
        .or_else(|| trunk.get("rope_scaling"))
        .filter(|v| !v.is_null());
    let rope_freq_factors = match scaling {
        None => Vec::new(),
        Some(s) => llama3_factors(s, head_dim, rope_theta)?,
    };

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
        partial_rotary_factor: 1.0,
        num_layers,
        dense_lead_intermediate_size: 0,
        num_dense_leading_layers: 0,
        num_experts: 0,
        top_k_experts: 0,
        tie_word_embeddings: trunk
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        attention_k_eq_v: false,
        full_attention_layer_mask: vec![1; num_layers as usize],
        hidden_activation: "silu".to_string(),
        family: ModelFamily::Llama,
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
        short_conv_kernel_size: 0,
        rope_freq_factors,
        ple: PleConfig::NONE,
    })
}

/// A `rope_scaling` (or `rope_parameters`) object to per-pair divisors.
/// `default` means none; `llama3` builds the vector GGUF would carry;
/// anything else (linear, dynamic, yarn, longrope) is refused because the
/// llama flow does not apply it.
fn llama3_factors(
    scaling: &serde_json::Value,
    head_dim: i64,
    rope_theta: f64,
) -> Result<Vec<f32>, Gemma4Error> {
    let kind = scaling
        .get("rope_type")
        .or_else(|| scaling.get("type"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("default");
    match kind {
        "default" => Ok(Vec::new()),
        "llama3" => {
            let num = |key: &str| {
                scaling
                    .get(key)
                    .and_then(serde_json::Value::as_f64)
                    .ok_or_else(|| {
                        Gemma4Error::Config(format!("llama3 rope_scaling missing {key}"))
                    })
            };
            compute::llama3_rope_factors(
                head_dim as usize,
                rope_theta,
                num("factor")?,
                num("low_freq_factor")?,
                num("high_freq_factor")?,
                num("original_max_position_embeddings")?,
            )
            .ok_or_else(|| {
                Gemma4Error::Config(format!(
                    "llama3 rope_scaling has unusable values: {scaling}"
                ))
            })
        }
        other => Err(Gemma4Error::Config(format!(
            "rope scaling type {other:?} is not applied by the llama flow"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `unsloth/Llama-3.2-1B-Instruct`'s config, trimmed to what the parser
    /// reads.
    const LLAMA32_1B: &str = r#"{
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "hidden_size": 2048, "intermediate_size": 8192, "num_attention_heads": 32,
        "num_hidden_layers": 16, "num_key_value_heads": 8, "head_dim": 64,
        "vocab_size": 128256, "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-05, "hidden_act": "silu", "rope_theta": 500000.0,
        "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                         "original_max_position_embeddings": 8192, "rope_type": "llama3"},
        "tie_word_embeddings": true, "attention_bias": false, "mlp_bias": false
    }"#;

    const MISTRAL_V02: &str = r#"{
        "architectures": ["MistralForCausalLM"], "model_type": "mistral",
        "hidden_size": 4096, "intermediate_size": 14336, "num_attention_heads": 32,
        "num_hidden_layers": 32, "num_key_value_heads": 8, "vocab_size": 32000,
        "max_position_embeddings": 32768, "rms_norm_eps": 1e-05, "hidden_act": "silu",
        "rope_theta": 1000000.0, "sliding_window": null, "tie_word_embeddings": false
    }"#;

    fn with(base: &str, edit: impl FnOnce(&mut serde_json::Value)) -> String {
        let mut v: serde_json::Value = serde_json::from_str(base).unwrap();
        edit(&mut v);
        v.to_string()
    }

    #[test]
    fn llama32_parses_with_llama3_divisors() {
        let arch = parse_llama_config(LLAMA32_1B).unwrap();
        assert_eq!(arch.family, ModelFamily::Llama);
        assert_eq!(arch.num_experts, 0);
        assert_eq!(arch.head_dim, 64);
        assert!(arch.tie_word_embeddings);
        assert_eq!(arch.full_rope_theta, 500000.0);
        let want = compute::llama3_rope_factors(64, 500000.0, 32.0, 1.0, 4.0, 8192.0).unwrap();
        assert_eq!(arch.rope_freq_factors, want);
        assert_eq!(arch.rope_freq_factors.len(), 32);
        assert_eq!(arch.attention_scale, 0.125);
    }

    #[test]
    fn mistral_v02_parses_with_no_scaling() {
        let arch = parse_llama_config(MISTRAL_V02).unwrap();
        assert_eq!(arch.family, ModelFamily::Llama);
        assert_eq!(arch.head_dim, 128);
        assert!(arch.rope_freq_factors.is_empty());
        assert!(!arch.tie_word_embeddings);
    }

    #[test]
    fn transformers_v5_rope_parameters_are_read() {
        let cfg = with(LLAMA32_1B, |v| {
            let s = v["rope_scaling"].take();
            v.as_object_mut().unwrap().remove("rope_scaling");
            v.as_object_mut().unwrap().remove("rope_theta");
            let mut p = s;
            p["rope_theta"] = serde_json::json!(500000.0);
            v["rope_parameters"] = p;
        });
        let arch = parse_llama_config(&cfg).unwrap();
        assert_eq!(arch.full_rope_theta, 500000.0);
        assert_eq!(arch.rope_freq_factors.len(), 32);
    }

    #[test]
    fn what_the_flow_cannot_honour_is_refused() {
        let refusals: Vec<(&str, String)> = vec![
            (
                "eps",
                with(MISTRAL_V02, |v| v["rms_norm_eps"] = serde_json::json!(1e-6)),
            ),
            (
                "window",
                with(MISTRAL_V02, |v| {
                    v["sliding_window"] = serde_json::json!(4096)
                }),
            ),
            (
                "bias",
                with(LLAMA32_1B, |v| {
                    v["attention_bias"] = serde_json::json!(true)
                }),
            ),
            (
                "mlp bias",
                with(LLAMA32_1B, |v| v["mlp_bias"] = serde_json::json!(true)),
            ),
            (
                "act",
                with(LLAMA32_1B, |v| v["hidden_act"] = serde_json::json!("gelu")),
            ),
            (
                "moe",
                with(MISTRAL_V02, |v| {
                    v["num_local_experts"] = serde_json::json!(8)
                }),
            ),
            (
                "yarn",
                with(LLAMA32_1B, |v| {
                    v["rope_scaling"] = serde_json::json!({"rope_type": "yarn", "factor": 4.0})
                }),
            ),
            (
                "linear",
                with(LLAMA32_1B, |v| {
                    v["rope_scaling"] = serde_json::json!({"type": "linear", "factor": 2.0})
                }),
            ),
            (
                "llama3 missing key",
                with(LLAMA32_1B, |v| {
                    v["rope_scaling"]
                        .as_object_mut()
                        .unwrap()
                        .remove("low_freq_factor");
                }),
            ),
            (
                "wrapper",
                with(LLAMA32_1B, |v| v["model_type"] = serde_json::json!("llava")),
            ),
            (
                "vision",
                with(LLAMA32_1B, |v| {
                    v["vision_config"] = serde_json::json!({"depth": 2})
                }),
            ),
            (
                "qwen2",
                with(LLAMA32_1B, |v| v["model_type"] = serde_json::json!("qwen2")),
            ),
        ];
        for (what, cfg) in refusals {
            assert!(parse_llama_config(&cfg).is_err(), "{what} must be refused");
        }
    }

    #[test]
    fn absent_optional_keys_take_hf_defaults() {
        let cfg = with(MISTRAL_V02, |v| {
            let o = v.as_object_mut().unwrap();
            o.remove("num_key_value_heads");
            o.remove("rope_theta");
            o.remove("sliding_window");
            o.remove("hidden_act");
        });
        let arch = parse_llama_config(&cfg).unwrap();
        assert_eq!(arch.num_kv_heads, arch.num_heads);
        assert_eq!(arch.full_rope_theta, DEFAULT_ROPE_THETA);
    }

    /// The SmolVLM-256M wrapper shape: the trunk is read out of `text_config`,
    /// and any other wrapper with a llama `text_config` is still refused.
    #[test]
    fn only_the_idefics3_wrappers_read_their_trunk_from_text_config() {
        let trunk: serde_json::Value = serde_json::from_str(LLAMA32_1B).unwrap();
        let wrap = |model_type: &str| {
            serde_json::json!({
                "model_type": model_type,
                "text_config": trunk,
                "vision_config": {},
            })
            .to_string()
        };
        let flat = parse_llama_config(LLAMA32_1B).unwrap();
        for wrapper in ["idefics3", "smolvlm"] {
            assert_eq!(parse_llama_config(&wrap(wrapper)).unwrap(), flat);
        }
        let e = parse_llama_config(&wrap("llava")).unwrap_err().to_string();
        assert!(e.contains("llava"), "{e}");
    }
}
