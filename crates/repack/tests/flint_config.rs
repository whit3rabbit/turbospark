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

/// `mlx-community/flint-mixed-3-4bit` (revision b1c30585): stock MLX
/// per-tensor entries in the SOURCE namespace, the embedding at 8 and every
/// routed projection at 3, beside an `mlx_export` that says the same.
fn mixed_config() -> serde_json::Value {
    let mut v = real_config();
    v["mlx_export"]["routed_expert_bits"] = 3.into();
    let mut q = serde_json::json!({
        "group_size": 64,
        "bits": 4,
        "model.embed_tokens": {"group_size": 64, "bits": 8},
    });
    for layer in 2..49 {
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            q[format!("model.layers.{layer}.mlp.switch_mlp.{proj}")] =
                serde_json::json!({"group_size": 64, "bits": 3});
        }
    }
    v["quantization"] = q;
    v
}

#[test]
fn mixed_build_routes_three_bit_experts_under_canonical_names() {
    let q = parse_flint_quantization(&mixed_config().to_string()).unwrap();
    assert_eq!(q.bits_for("language_model.model.embed_tokens"), 8);
    assert_eq!(q.bits_for("language_model.lm_head"), 4);
    assert_eq!(
        q.bits_for("language_model.model.layers.2.mlp.shared_expert.gate_proj"),
        4
    );
    for proj in ["gate_proj", "up_proj", "down_proj"] {
        assert_eq!(
            q.bits_for(&format!(
                "language_model.model.layers.48.mlp.switch_mlp.{proj}"
            )),
            3
        );
    }
    // The source-namespace keys are gone, not kept beside their renames.
    assert!(q
        .bits_overrides
        .keys()
        .all(|k| k.starts_with("language_model.")));
}

#[test]
fn mixed_build_refuses_widths_without_kernels() {
    let mutate = |f: &dyn Fn(&mut serde_json::Value)| {
        let mut v = mixed_config();
        f(&mut v);
        parse_flint_quantization(&v.to_string())
            .expect_err("mutation must be refused")
            .to_string()
    };
    // 3-bit is a routed width only.
    let attn = "model.layers.2.self_attn.q_proj";
    assert!(
        mutate(&|v| v["quantization"][attn] = serde_json::json!({"group_size": 64, "bits": 3}))
            .contains("3-bit"),
    );
    let shared = "model.layers.2.mlp.shared_expert.gate_proj";
    assert!(mutate(
        &|v| v["quantization"][shared] = serde_json::json!({"group_size": 64, "bits": 3})
    )
    .contains("3-bit"),);
    // Experts at two widths, or disagreeing with `mlx_export`.
    let one = "model.layers.7.mlp.switch_mlp.up_proj";
    assert!(
        mutate(&|v| v["quantization"][one] = serde_json::json!({"group_size": 64, "bits": 4}))
            .contains("one width"),
    );
    assert!(mutate(&|v| v["mlx_export"]["routed_expert_bits"] = 4.into()).contains("one width"));
    // A routed width the per-tensor entries never state cannot be sized.
    assert!(
        mutate(&|v| v["quantization"] = serde_json::json!({"group_size": 64, "bits": 4}))
            .contains("declares no routed expert")
    );
    // 2-bit experts have no Flint kernel.
    assert!(mutate(&|v| v["mlx_export"]["routed_expert_bits"] = 2.into()).contains("not 3 or 4"));
}

#[test]
fn the_published_mixed_config_parses() {
    // A local copy only; there is no network fetch in this suite.
    let Some(home) = std::env::var_os("HOME") else {
        return;
    };
    let path = std::path::Path::new(&home).join("models/flint-mixed-3-4bit/config.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!("skipping: {} not present", path.display());
        return;
    };
    assert_eq!(parse_flint_config(&text).unwrap(), flint_40b_a2_5b());
    let q = parse_flint_quantization(&text).unwrap();
    assert_eq!(
        q.bits_for("language_model.model.layers.2.mlp.switch_mlp.down_proj"),
        3
    );
    assert_eq!(q.bits_for("language_model.model.embed_tokens"), 8);
}
