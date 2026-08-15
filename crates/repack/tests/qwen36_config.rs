//! `parse_qwen_gdn_moe_config` against the PRODUCTION field values.
//!
//! Unlike the Gemma config tests, this fixture is not a tiny synthetic
//! shape: it carries the real `mlx-community/Qwen3.6-35B-A3B-4bit`
//! `text_config` values verbatim, so the whole test reduces to one
//! assertion -- the parse equals `model_io::qwen_gdn_moe_35b_a3b()` field for
//! field. That is the only assertion that can catch a key mapped to the
//! wrong field, which a shape-only fixture cannot (every dimension would
//! be some other made-up number either way).
//!
//! `tests/qwen36_checkpoint_network.rs` runs the same assertion against the
//! config fetched over the network; this one runs in the default suite.

use turbospark_repack::parse_qwen_gdn_moe_config;

/// 40 layers, gated-DeltaNet everywhere except every 4th
/// (`full_attention_interval = 4`).
fn layer_types() -> Vec<&'static str> {
    (0..40)
        .map(|i| {
            if (i + 1) % 4 == 0 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect()
}

/// The production `text_config`, trimmed to the keys the parser reads plus
/// the mrope fields that must be ignored.
fn text_config() -> serde_json::Value {
    serde_json::json!({
        "attn_output_gate": true,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "layer_types": layer_types(),
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262_144,
        "moe_intermediate_size": 512,
        "num_attention_heads": 16,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 40,
        "num_key_value_heads": 2,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1.0e-6,
        "rope_parameters": {
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000_000,
            "rope_type": "default"
        },
        "shared_expert_intermediate_size": 512,
        "tie_word_embeddings": false,
        "vocab_size": 248_320
    })
}

fn config_json() -> String {
    serde_json::json!({
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": text_config(),
        "vision_config": {"depth": 27, "hidden_size": 1152},
        "quantization": {
            "group_size": 64,
            "bits": 4,
            "language_model.model.layers.0.mlp.gate": {"group_size": 64, "bits": 8},
            "language_model.model.layers.0.mlp.shared_expert_gate": {"group_size": 64, "bits": 8}
        }
    })
    .to_string()
}

#[test]
fn parses_the_production_config_into_the_pinned_baseline() {
    let arch = parse_qwen_gdn_moe_config(&config_json()).expect("config parses");
    assert_eq!(
        arch,
        model_io::qwen_gdn_moe_35b_a3b(),
        "parsed config does not match the pinned Qwen3.6-35B-A3B baseline"
    );
}

#[test]
fn attention_scale_is_the_reference_head_dim_power() {
    let arch = parse_qwen_gdn_moe_config(&config_json()).expect("config parses");
    // mlx-lm `Qwen3NextAttention.__init__`: `self.scale = head_dim**-0.5`.
    // Exactly 1/16 at head_dim 256, so `==` is safe here (AGENTS.md
    // Gotcha 24 is about scales that are NOT binary fractions).
    assert_eq!(arch.attention_scale, 0.0625);
    assert_eq!(arch.attention_scale, 1.0 / 16.0);
}

#[test]
fn linear_layers_are_2_and_full_layers_are_1() {
    let arch = parse_qwen_gdn_moe_config(&config_json()).expect("config parses");
    let mask = &arch.full_attention_layer_mask;
    assert_eq!(mask.len(), 40);
    assert_eq!(mask.iter().filter(|&&m| m == 1).count(), 10);
    assert_eq!(mask.iter().filter(|&&m| m == 2).count(), 30);
    assert_eq!(
        mask[0], 2,
        "layer 0 is linear, which is why manifest_quant probes in_proj_qkv there"
    );
    assert_eq!(mask[3], 1);
}

#[test]
fn accepts_a_text_only_config_without_the_wrapper() {
    let unwrapped = text_config().to_string();
    assert_eq!(
        parse_qwen_gdn_moe_config(&unwrapped).expect("unwrapped config parses"),
        model_io::qwen_gdn_moe_35b_a3b()
    );
}

#[test]
fn rejects_a_missing_layer_types() {
    let mut tc = text_config();
    tc.as_object_mut().unwrap().remove("layer_types");
    let err = parse_qwen_gdn_moe_config(&tc.to_string()).expect_err("must reject");
    assert!(err.to_string().contains("layer_types"), "{err}");
}

#[test]
fn rejects_a_layer_types_length_mismatch() {
    let mut tc = text_config();
    tc["layer_types"] = serde_json::json!(["linear_attention", "full_attention"]);
    let err = parse_qwen_gdn_moe_config(&tc.to_string()).expect_err("must reject");
    assert!(err.to_string().contains("num_hidden_layers is 40"), "{err}");
}

#[test]
fn rejects_an_unknown_layer_type() {
    let mut tc = text_config();
    let mut types = layer_types();
    types[0] = "sliding_attention";
    tc["layer_types"] = serde_json::json!(types);
    let err = parse_qwen_gdn_moe_config(&tc.to_string()).expect_err("must reject");
    assert!(err.to_string().contains("sliding_attention"), "{err}");
}

#[test]
fn rejects_an_odd_rotary_dim() {
    // head_dim 254 x 0.25 is not an integer; head_dim 8 x 0.25 = 2 is fine,
    // so this is specifically the fractional/odd guard, not a size guard.
    let mut tc = text_config();
    tc["head_dim"] = serde_json::json!(254);
    let err = parse_qwen_gdn_moe_config(&tc.to_string()).expect_err("must reject");
    assert!(err.to_string().contains("even integer"), "{err}");
}

#[test]
fn quantization_reads_the_router_and_shared_gate_as_int8() {
    let quant =
        turbospark_repack::parse_gemma4_quantization(&config_json()).expect("quantization parses");
    let manifest = turbospark_repack::manifest_quant(&quant, model_io::ModelFamily::QwenGdnMoe);
    // `validate_quant` accepts router 8 only; routedExpert 2 or 4.
    assert_eq!(manifest["router"]["weightBits"], 8);
    assert_eq!(manifest["routedExpert"]["weightBits"], 4);
    assert_eq!(manifest["sharedExpert"]["weightBits"], 4);
    assert_eq!(manifest["attention"]["weightBits"], 4);
    assert_eq!(manifest["embedding"]["weightBits"], 4);
}
