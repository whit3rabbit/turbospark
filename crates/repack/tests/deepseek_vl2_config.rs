//! DeepSeek-VL2 trunk intake, offline (`docs/DEEPSEEK_VL2_VISION_PHASE0.md`).
//!
//! The fixture is `mlx-community/deepseek-vl2-tiny-4bit`'s own `config.json`
//! language section, so a parser that agrees with this text agrees with a
//! published file rather than with a guess.

use model_io::{deepseek_vl2_tiny, ModelFamily};
use turbospark_repack::{config_json_family, parse_deepseek_vl2_config};

const CONFIG: &str = r#"{
  "model_type": "deepseek_vl_v2",
  "quantization": {"group_size": 64, "bits": 4},
  "language_config": {
    "architectures": ["DeepseekV2ForCausalLM"],
    "bos_token_id": 0,
    "eos_token_id": 1,
    "first_k_dense_replace": 1,
    "hidden_size": 1280,
    "intermediate_size": 6848,
    "kv_lora_rank": null,
    "lm_head": true,
    "max_position_embeddings": 4096,
    "model_type": "deepseek_v2",
    "moe_intermediate_size": 896,
    "n_group": 1,
    "n_routed_experts": 64,
    "n_shared_experts": 2,
    "num_attention_heads": 10,
    "num_experts_per_tok": 6,
    "num_hidden_layers": 12,
    "num_key_value_heads": 10,
    "q_lora_rank": null,
    "qk_nope_head_dim": 0,
    "qk_rope_head_dim": 0,
    "rm_head": false,
    "topk_group": 1,
    "topk_method": "greedy",
    "use_mla": false,
    "v_head_dim": 0,
    "vocab_size": 129280
  }
}"#;

fn mutated(edit: impl FnOnce(&mut serde_json::Value)) -> String {
    let mut v: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    edit(&mut v);
    v.to_string()
}

#[test]
fn the_published_config_parses_to_the_registered_baseline() {
    let parsed = parse_deepseek_vl2_config(CONFIG).expect("parses");
    let baseline = deepseek_vl2_tiny();
    // The whole struct, so a field the parser forgets to derive is caught.
    assert_eq!(parsed, baseline);
    assert_eq!(parsed.family, ModelFamily::Deepseek2);
    assert!(!parsed.mla.is_active(), "plain attention is the point");
    assert_eq!(parsed.intermediate_size, 2 * parsed.moe_intermediate_size);
    assert_eq!(parsed.full_attention_layer_mask, vec![1u8; 12]);
}

#[test]
fn the_wrapper_resolves_to_the_deepseek2_family() {
    let root: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    assert_eq!(config_json_family(&root), Some(ModelFamily::Deepseek2));
}

#[test]
fn an_mla_trunk_is_refused() {
    let text = mutated(|v| v["language_config"]["use_mla"] = true.into());
    let err = parse_deepseek_vl2_config(&text).unwrap_err().to_string();
    assert!(err.contains("use_mla"), "{err}");
    let text = mutated(|v| v["language_config"]["qk_nope_head_dim"] = 128.into());
    assert!(parse_deepseek_vl2_config(&text).is_err());
}

#[test]
fn router_and_rope_settings_the_flow_does_not_implement_are_refused() {
    for (key, value) in [
        ("scoring_func", serde_json::json!("sigmoid")),
        ("norm_topk_prob", serde_json::json!(true)),
        ("routed_scaling_factor", serde_json::json!(2.5)),
        ("topk_method", serde_json::json!("noaux_tc")),
        ("n_group", serde_json::json!(8)),
        ("rms_norm_eps", serde_json::json!(1e-5)),
        ("rope_traditional", serde_json::json!(true)),
        ("attention_bias", serde_json::json!(true)),
        ("tie_word_embeddings", serde_json::json!(true)),
        ("hidden_act", serde_json::json!("gelu")),
        ("moe_layer_freq", serde_json::json!(2)),
        (
            "rope_scaling",
            serde_json::json!({"type": "linear", "factor": 2.0}),
        ),
    ] {
        let text = mutated(|v| v["language_config"][key] = value.clone());
        assert!(
            parse_deepseek_vl2_config(&text).is_err(),
            "{key} = {value} must be refused"
        );
    }
}

#[test]
fn a_config_without_a_dense_lead_is_refused() {
    let text = mutated(|v| v["language_config"]["first_k_dense_replace"] = 0.into());
    let err = parse_deepseek_vl2_config(&text).unwrap_err().to_string();
    assert!(err.contains("first_k_dense_replace"), "{err}");
}

#[test]
fn rope_theta_and_eps_default_to_the_references_values_when_the_config_is_silent() {
    let parsed = parse_deepseek_vl2_config(CONFIG).unwrap();
    assert_eq!(parsed.full_rope_theta, 10_000.0);
    // And an explicit theta is honoured.
    let text = mutated(|v| v["language_config"]["rope_theta"] = 500_000.0.into());
    assert_eq!(
        parse_deepseek_vl2_config(&text).unwrap().full_rope_theta,
        500_000.0
    );
}
