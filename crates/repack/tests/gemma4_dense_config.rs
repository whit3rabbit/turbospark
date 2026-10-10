//! `parse_gemma4_config` against the DENSE production field values.
//!
//! Same rule as `tests/qwen36_config.rs` states for the MoE half: the
//! fixture carries the real checkpoint's `text_config` verbatim so the whole
//! test reduces to one assertion -- the parse equals
//! `model_io::gemma4_12b_dense()` field for field. The real carrier is
//! `cowWhySo/humanizer-12b-mlx-8bit` @ baea046b (a Gemma 4 12B trunk
//! fine-tune), whose MoE keys are JSON `null` under `enable_moe_block:
//! false`; a shape-only fixture cannot catch a key mapped to the wrong
//! field, and neither can one whose MoE keys are absent rather than null,
//! which is the exact form this parser had to learn.

use turbospark_repack::{parse_gemma4_config, Gemma4Error};

/// The real `text_config`, keys verbatim off the pinned config.json. The
/// `layer_types` array is generated: it is 48 entries of five
/// `sliding_attention` then one `full_attention`, and writing them out
/// verbatim buys nothing over the arithmetic the source itself documents
/// (`full_attention` iff `(i + 1) % 6 == 0`, per the publisher's own
/// `check_source_config`).
fn text_config() -> serde_json::Value {
    serde_json::json!({
        "attention_bias": false,
        "attention_dropout": 0.0,
        "attention_k_eq_v": true,
        "bos_token_id": 2,
        "dtype": "bfloat16",
        "enable_moe_block": false,
        "eos_token_id": 1,
        "final_logit_softcapping": 30.0,
        "global_head_dim": 512,
        "head_dim": 256,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 3840,
        "hidden_size_per_layer_input": 0,
        "initializer_range": 0.02,
        "intermediate_size": 15360,
        "layer_types": (0..48)
            .map(|i| if (i + 1) % 6 == 0 { "full_attention" } else { "sliding_attention" })
            .collect::<Vec<_>>(),
        "max_position_embeddings": 262_144,
        "model_type": "gemma4_unified_text",
        "moe_intermediate_size": null,
        "num_attention_heads": 16,
        "num_experts": null,
        "num_global_key_value_heads": 1,
        "num_hidden_layers": 48,
        "num_key_value_heads": 8,
        "num_kv_shared_layers": 0,
        "pad_token_id": 0,
        "rms_norm_eps": 1.0e-6,
        "rope_parameters": {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
                "rope_type": "proportional"
            },
            "sliding_attention": {
                "rope_theta": 10_000.0,
                "rope_type": "default"
            }
        },
        "sliding_window": 1024,
        "tie_word_embeddings": true,
        "top_k_experts": null,
        "use_bidirectional_attention": "vision",
        "use_cache": true,
        "use_double_wide_mlp": false,
        "vocab_size": 262_144,
        "vocab_size_per_layer_input": 262_144
    })
}

fn config_json() -> String {
    serde_json::json!({
        "architectures": ["Gemma4UnifiedForConditionalGeneration"],
        "model_type": "gemma4_unified",
        "text_config": text_config(),
    })
    .to_string()
}

#[test]
fn dense_humanizer_config_equals_the_dense_baseline() {
    let arch = parse_gemma4_config(&config_json()).expect("config parses");
    let expected = model_io::gemma4_12b_dense();
    assert_eq!(arch.hidden_size, expected.hidden_size);
    assert_eq!(arch.intermediate_size, expected.intermediate_size);
    assert_eq!(arch.moe_intermediate_size, expected.moe_intermediate_size);
    assert_eq!(arch.num_heads, expected.num_heads);
    assert_eq!(arch.num_kv_heads, expected.num_kv_heads);
    assert_eq!(arch.num_full_kv_heads, expected.num_full_kv_heads);
    assert_eq!(arch.head_dim, expected.head_dim);
    assert_eq!(arch.full_head_dim, expected.full_head_dim);
    assert_eq!(arch.vocab_size, expected.vocab_size);
    assert_eq!(arch.sliding_window, expected.sliding_window);
    assert_eq!(arch.final_logit_softcap, expected.final_logit_softcap);
    assert_eq!(arch.rope_theta, expected.rope_theta);
    assert_eq!(arch.full_rope_theta, expected.full_rope_theta);
    assert_eq!(arch.partial_rotary_factor, expected.partial_rotary_factor);
    assert_eq!(arch.num_layers, expected.num_layers);
    assert_eq!(arch.num_experts, expected.num_experts);
    assert_eq!(arch.top_k_experts, expected.top_k_experts);
    assert_eq!(
        arch.full_attention_layer_mask,
        expected.full_attention_layer_mask
    );
    assert_eq!(arch.attention_k_eq_v, expected.attention_k_eq_v);
    assert_eq!(arch.tie_word_embeddings, expected.tie_word_embeddings);
    assert_eq!(arch.hidden_activation, expected.hidden_activation);
    assert_eq!(arch.family, expected.family);
    assert_eq!(arch.attention_scale, expected.attention_scale);
    assert_eq!(
        arch.embedding_scaled_by_sqrt_hidden,
        expected.embedding_scaled_by_sqrt_hidden
    );
    assert_eq!(arch.ffn_sandwich_norms, expected.ffn_sandwich_norms);
}

#[test]
fn enable_moe_block_true_with_no_experts_is_refused() {
    let mut root: serde_json::Value = serde_json::from_str(&config_json()).unwrap();
    root["text_config"]["enable_moe_block"] = serde_json::json!(true);
    let err = parse_gemma4_config(&root.to_string()).expect_err("contradiction refuses");
    assert!(
        matches!(err, Gemma4Error::Config(ref d) if d.contains("enable_moe_block")),
        "unexpected error: {err:?}"
    );
}

#[test]
fn enable_moe_block_false_with_experts_is_refused() {
    let mut root: serde_json::Value = serde_json::from_str(&config_json()).unwrap();
    root["text_config"]["enable_moe_block"] = serde_json::json!(false);
    root["text_config"]["num_experts"] = serde_json::json!(128);
    root["text_config"]["top_k_experts"] = serde_json::json!(8);
    root["text_config"]["moe_intermediate_size"] = serde_json::json!(704);
    let err = parse_gemma4_config(&root.to_string()).expect_err("contradiction refuses");
    assert!(
        matches!(err, Gemma4Error::Config(ref d) if d.contains("enable_moe_block")),
        "unexpected error: {err:?}"
    );
}

#[test]
fn the_unified_model_type_resolves_to_gemma4() {
    // The registry rows the probe and stream walk resolve through; without
    // them every `gemma4_unified` checkpoint refuses as unregistered before
    // the parser above is ever reached.
    let root: serde_json::Value = serde_json::from_str(&config_json()).unwrap();
    assert_eq!(
        turbospark_repack::config_json_family(&root),
        Some(model_io::ModelFamily::Gemma4)
    );
}
