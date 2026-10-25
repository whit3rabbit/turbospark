//! `parse_flint_config` against the real `RecursiveSI/flint-int4` config
//! (revision 3b93f9053b530488405ae9fca51d4900d6ce21f0), with the production
//! values inline so a key wired to the wrong field fails without a network.

use model_io::flint_40b_a2_5b;
use turbospark_repack::{parse_flint_config, parse_flint_quantization};

/// The published `config.json`, `layer_types` expanded from its pattern.
fn real_config() -> serde_json::Value {
    let full = [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48];
    let layer_types: Vec<&str> = (0..49)
        .map(|i| {
            if full.contains(&i) {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect();
    serde_json::json!({
        "architectures": ["FlintForCausalLM"],
        "attention_bias": false,
        "attention_sink": true,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "layer_types": layer_types,
        "max_position_embeddings": 262144,
        "mlp_only_layers": [0, 1],
        "mlx_export": {
            "bits": 4,
            "embedding_bits": 8,
            "routed_expert_bits": 4,
            "scale_rule": "mse"
        },
        "model_type": "flint",
        "moe_intermediate_size": 512,
        "no_rope_layer_indices": full,
        "norm_topk_prob": true,
        "num_attention_heads": 32,
        "num_experts": 256,
        "num_experts_per_tok": 6,
        "num_hidden_layers": 49,
        "num_key_value_heads": 4,
        "quantization": {"bits": 4, "group_size": 64},
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "rope_theta": 10000.0,
        "router_expert_bias": false,
        "scoring_func": "sigmoid",
        "shared_expert_intermediate_size": 1024,
        "sliding_window": 513,
        "tie_word_embeddings": false,
        "use_sliding_window": true,
        "vocab_size": 150144
    })
}

fn parse(v: &serde_json::Value) -> Result<model_io::ArchConfig, String> {
    parse_flint_config(&v.to_string()).map_err(|e| e.to_string())
}

#[test]
fn real_config_equals_the_baseline() {
    assert_eq!(parse(&real_config()).unwrap(), flint_40b_a2_5b());
}

#[test]
fn embedding_is_declared_int8_and_everything_else_int4() {
    let q = parse_flint_quantization(&real_config().to_string()).unwrap();
    assert_eq!(q.default_bits, 4);
    assert_eq!(q.bits_for("language_model.model.embed_tokens"), 8);
    assert_eq!(q.bits_for("language_model.lm_head"), 4);
    assert_eq!(q.bits_for("language_model.model.layers.2.mlp.gate_proj"), 4);
}

#[test]
fn refuses_what_the_flow_does_not_implement() {
    let mutate = |f: &dyn Fn(&mut serde_json::Value)| {
        let mut v = real_config();
        f(&mut v);
        parse(&v).expect_err("mutation must be refused")
    };
    assert!(mutate(&|v| v["scoring_func"] = "softmax".into()).contains("sigmoid"));
    assert!(mutate(&|v| v["norm_topk_prob"] = false.into()).contains("sigmoid"));
    assert!(mutate(&|v| v["router_expert_bias"] = true.into()).contains("router_expert_bias"));
    assert!(mutate(&|v| v["mlp_only_layers"] = serde_json::json!([0, 2])).contains("prefix"));
    // NoPE must equal the full-attention set.
    assert!(
        mutate(&|v| v["no_rope_layer_indices"] = serde_json::json!([0, 1, 5])).contains("NoPE")
    );
    assert!(mutate(&|v| v["tie_word_embeddings"] = true.into()).contains("untied"));
    assert!(mutate(&|v| v["rms_norm_eps"] = 1e-5.into()).contains("rms_norm_eps"));
    // A sibling family's config is refused by name.
    assert!(mutate(&|v| v["model_type"] = "qwen2".into()).contains("family"));
}
